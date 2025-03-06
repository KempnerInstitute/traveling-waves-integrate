import torch
import torch.nn as nn
import numpy as np
import torch.nn as nn


"""
coRNN model with mlp decoder.
"""

class Model(nn.Module):
    def __init__(self, N, c_in, c_mid, c_out, num_classes, min_iters, max_iters, img_size, dt, T, readout, fc_readout):
        super(Model, self).__init__()

        self.N = N
        self.c_in = c_in
        self.c_out = c_out
        self.img_size = img_size
        self.K = int((max_iters - min_iters) / 2) - 1
        self.num_classes = num_classes
        self.readout = readout

        self.classifier = CoRNNSegmentationModel(N,
                                                 c_in,
                                                 c_mid, 
                                                 c_out,
                                                 n_classes=num_classes,
                                                 dt=dt,
                                                 T=T,
                                                 image_size=img_size,
                                                 readout=readout,
                                                 fc_readout=fc_readout)
        
    def forward(self, x):
        return self.classifier(x, return_fft=False)
            

class CoRNNSegmentationModel(nn.Module):
    """
    A drop-in segmentation model using the coRNN approach:
      - Input:  (B,1,H,W)
      - Output: (B,n_classes,H,W)
    """
    def __init__(self, 
                 N,
                 c_in,
                 c_mid,
                 c_out,
                 n_classes=7,  # e.g. background + 6 polygon types
                 dt=0.5,       # ODE time step
                 T=50,         # Number of RNN unroll steps
                 image_size=32,
                 readout="fft",
                 fc_readout=None):
        super().__init__()

        self.N = N
        self.c_in = c_in
        self.c_out = c_out
        self.n_classes = n_classes
        self.dt = dt
        self.T = T
        self.spatial = image_size
        self.n_hid = image_size * image_size  # flatten H,W => 1 channel, H*W pixels
        self.K = self.T//2 + 1
        self.readout = readout
        self.fc_readout = fc_readout

        # ====== 1) Encoders for omega, alpha, and init hy  ======
        #  B) omega_encoder (3x3 conv)
        self.omega_encoder = nn.Sequential(
            nn.Conv2d(c_in, c_mid, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(c_mid, c_mid, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(c_mid, c_out, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
        )

        #  C) alpha_encoder (3x3 conv)
        self.alpha_encoder = nn.Sequential(
            nn.Conv2d(c_in, c_mid, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(c_mid, c_mid, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(c_mid, c_out, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
        )

        #  D) hy_encoder (for initial hidden y)
        self.hy_encoder = nn.Sequential(
            nn.Conv2d(c_in, c_mid, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(c_mid, c_mid, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(c_mid, c_mid, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(c_mid, c_out, kernel_size=3, padding=1, stride=1),
            nn.Tanh()
        )

        # ====== 2) The coRNNCell  ======
        self.cell = coRNNCell(channels=c_out, spatial=N, dt=dt)

        # -------- 3) FFT-based readout  --------
        # We'll do: (T//2 + 1) -> 64 -> n_classes
        # Then reshape to (B, n_classes, H, W)

        self.fc_time = nn.Linear(self.T, self.K, bias=False)
        

    def forward(self, x, return_fft=False):
        """
        x: (B,1,H,W)  single frame input
        returns: (B,n_classes,H,W)  per-pixel logits
        """
        B, C, H, W = x.shape

        # 1) Precompute omega, alpha from single input x
        omega = self.omega_encoder(x)     # shape (B,1,H,W)
        alpha = self.alpha_encoder(x)     # shape (B,1,H,W)

        # Flatten to (B, H*W)
        omega = omega.view(B, self.c_out, -1)
        alpha = alpha.view(B, self.c_out, -1)

        # 2) Initialize hidden states (hy, hz)
        hy_init = self.hy_encoder(x)  # shape (B,1,H,W)
        hy = hy_init.view(B, self.c_out, -1)      # (B,H*W)
        hz = torch.zeros_like(hy)     # start velocity at 0

        # 3) Run dynamics
        y_seq = []
        for t in range(self.T):
            hy, hz = self.cell(
                x_t=x,  # optional usage
                hy=hy, 
                hz=hz, 
                omega=omega, 
                alpha=alpha
            )
            y_seq.append(hy)  # each is (B,H*W)
        y_seq = torch.stack(y_seq, dim=1)  # (B, T, C, H*W)
        y_seq = y_seq.reshape(B, self.T, self.c_out, H, W)

        # Readout
        logits = self.readout(self, y_seq, B, H, W)
        return logits, y_seq
        
    def _init_encoders(self,):
         with torch.no_grad():
            # Initialize omega encoder conv layers
            for i, layer in enumerate(self.omega_encoder):
                if isinstance(layer, nn.Conv2d):
                    # nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                    nn.init.dirac_(layer.weight)
                    nn.init.zeros_(layer.bias)

            # Initialize alpha encoder conv layers
            for i, layer in enumerate(self.alpha_encoder):
                if isinstance(layer, nn.Conv2d):
                    nn.init.constant_(layer.weight, 0.00)
                    nn.init.constant_(layer.bias, 0.00)

            # Initialize hy encoder conv layers
            for i, layer in enumerate(self.hy_encoder):
                if isinstance(layer, nn.Conv2d):
                    if i < len(self.hy_encoder) - 1:
                        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                    else:
                        nn.init.kaiming_normal_(layer.weight, nonlinearity='tanh')
                    nn.init.zeros_(layer.bias)


class coRNNCell(nn.Module):
    def __init__(self, channels, spatial, dt):
        """
        n_inp: number of input channels (1 in your case).
        spatial: width=height of the square grid (e.g. 32).
        dt: ODE timestep.
        """
        super(coRNNCell, self).__init__()
        self.channels = channels
        self.dt = dt
        self.spatial = spatial

        # Local (2D) coupling: learnable Laplacian kernel
        self.Wy = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            padding_mode='circular',
            bias=False,
            groups=1# num_channels  # Each channel gets its own independent kernel
        )

        laplacian_kernel = torch.tensor(
            [[ 0.,  1.,  0.],
             [ 1., -4.0,  1.],
             [ 0.,  1.,  0.]]
        ).reshape(1, 1, 3, 3)

        # Initialize each channel's kernel with the Laplacian
        with torch.no_grad():
            nn.init.constant_(self.Wy.weight, 0)
            for i in range(channels):
                self.Wy.weight[i:i+1, i:i+1].copy_(laplacian_kernel)
        


    def forward(self, x_t, hy, hz, omega, alpha):
        """
        x_t: [batch_size, 1, H, W] for time step t (not used directly here, 
             but you can incorporate it if you want input-dependent forcing).
        hy:  [batch_size, H*W] hidden 'position'
        hz:  [batch_size, H*W] hidden 'velocity'
        mask, omega, alpha: each [batch_size, H*W], precomputed from the initial frame
        """
        B = x_t.shape[0]

        # Reshape hy -> (B,1,H,W) so we can apply the local conv
        hy_2d = hy.view(B, self.channels, self.spatial, self.spatial)
        spring_force_2d = self.Wy(hy_2d)  # shape (B,1,H,W)
        spring_force = torch.tanh(spring_force_2d).view(B, self.channels, -1)  # flatten back to (B,H*W)
        
        # ODE: 
        #   hz_{t+1} = hz_t + dt*( spring_force - omega*hy - alpha*hz )
        #   hy_{t+1} = hy_t + dt*hz_{t+1}
        new_hz = hz + self.dt * (spring_force - omega * hy - alpha * hz)
        new_hy = hy + self.dt * new_hz

        return new_hy, new_hz
    

    