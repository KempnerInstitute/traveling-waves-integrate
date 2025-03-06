import torch
import torch.nn as nn

# GPT Helped

"""
Conv RNN baseline (fft fed to readout).
"""

class Model(nn.Module):
    def __init__(self, N, c_in, c_mid, c_out, num_classes, min_iters, max_iters, img_size, dt, T, cell_type='GRU', readout="fft", fc_readout=None):
        super(Model, self).__init__()

        self.N = N
        self.c_in = c_in
        self.c_out = c_out
        self.img_size = img_size
        self.K = int((max_iters - min_iters) / 2) - 1
        self.num_classes = num_classes
        self.readout = readout

        self.classifier = RNNSegmentationModel(
            N, c_in, c_mid, c_out, num_classes, dt, T, img_size, cell_type, readout, fc_readout)
        
    def forward(self, x):
        return self.classifier(x, return_fft=False)


class RNNSegmentationModel(nn.Module):
    def __init__(self, N, c_in, c_mid, c_out, n_classes=7, dt=0.5, T=50, image_size=32, cell_type='GRU', readout="fft", fc_readout=None):
        super().__init__()
        
        self.N = N
        self.c_in = c_in
        self.c_out = c_out
        self.n_classes = n_classes
        self.dt = dt
        self.T = T
        self.spatial = image_size
        self.n_hid = image_size * image_size
        self.K = self.T // 2 + 1
        self.readout = readout
        self.fc_readout = fc_readout

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

        self.hz_encoder = nn.Sequential(
            nn.Conv2d(c_in, c_mid, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(c_mid, c_mid, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(c_mid, c_out, kernel_size=3, padding=1, stride=1),
        )

        self.hc_encoder = nn.Sequential(
            nn.Conv2d(c_in, c_mid, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(c_mid, c_mid, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(c_mid, c_out, kernel_size=3, padding=1, stride=1),
        )
        
        self.recurrent_cell = FlexibleRecurrentCell(c_out, c_out, cell_type)

        self.fc_time = nn.Linear(self.T, self.K, bias=False)


    def forward(self, x, return_fft=False):
        hy = self.hy_encoder(x) # B x c_out x N x N
        hz = self.hz_encoder(x) # B x c_out x N x N
        hc = self.hc_encoder(x) # B x c_out x N x N
        B, C, H, W = hy.shape

        y_seq = []
        for t in range(self.T):
            hy, hz, hc = self.recurrent_cell(hy, hz, hc)
            y_seq.append(hy)
        y_seq = torch.stack(y_seq, dim=1) # (B, T, c_out, H, W)

        logits = self.readout(self, y_seq, B, H, W)
        return logits, y_seq
        

    
class FlexibleRecurrentCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, cell_type='GRU', kernel_size=3):
        super(FlexibleRecurrentCell, self).__init__()

        self.cell_type = cell_type.lower()
        self.hidden_channels = hidden_channels

        #if self.cell_type == 'gru':
        #    self.cell = nn.GRUCell(input_channels, hidden_channels)
        #    self.update_state = self._gru_update
        #elif self.cell_type == 'rnn':
        #    self.cell = nn.RNNCell(input_channels, hidden_channels)
        #    self.update_state = self._rnn_update
        #elif self.cell_type == 'lstm':
        #    self.cell = nn.LSTMCell(input_channels, hidden_channels)
        #    self.update_state = self._lstm_update
        if self.cell_type == 'rnn':
            self.cell = ConvRNNCell(input_channels=input_channels, hidden_channels=hidden_channels, kernel_size=kernel_size)
            self.update_state = self._rnn_update
        elif self.cell_type == 'gru':
            self.cell = ConvGRUCell(input_channels=input_channels, hidden_channels=hidden_channels, kernel_size=kernel_size)
            self.update_state = self._gru_update
        elif self.cell_type == 'lstm':
            self.cell = ConvLSTMCell(input_channels=input_channels, hidden_channels=hidden_channels, kernel_size=kernel_size)
            self.update_state = self._lstm_update
        else:
            raise ValueError("Invalid cell_type. Choose from 'GRU', 'RNN', 'LSTM', 'ConvRNN', 'ConvGRU', 'ConvLSTM'.")

        self.transform_hidden = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.ReLU(),
            nn.Linear(hidden_channels * 2, hidden_channels)
        )

    def _rnn_update(self, hy, hz, hc=None):
        hz = self.cell(hy, hz)
        return hz, None

    def _gru_update(self, hy, hz, hc=None):
        hz = self.cell(hy, hz)
        return hz, None

    def _lstm_update(self, hy, hz, hc):
        hz, hc = self.cell(hy, hz, hc)
        return hz, hc

    def forward(self, hy, hz, hc=None):
        hz, hc = self.update_state(hy, hz, hc)
        hy = self.transform_hidden(torch.transpose(hz, 1, 3))
        hy = torch.transpose(hy, 1, 3)
        return hy, hz, hc


class ConvRNNCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3, stride=1, padding=1):
        super(ConvRNNCell, self).__init__()
        
        self.conv_in = nn.Conv2d(input_channels, hidden_channels, kernel_size, stride, padding, padding_mode='circular',)
        self.conv_hidden = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, stride, padding, padding_mode='circular',)
        self.activation = nn.ReLU()
        
        # Local (2D) coupling: learnable Laplacian kernel
        #_init_conv(self.conv_in, hidden_channels)
        #_init_conv(self.conv_hidden, hidden_channels)

    def forward(self, input, hidden):
        combined = self.conv_in(input) + self.conv_hidden(hidden)
        hidden = self.activation(combined)
        return hidden


class ConvGRUCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3, stride=1, padding=1):
        super(ConvGRUCell, self).__init__()

        self.conv_z = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, stride, padding, padding_mode='circular',)
        self.conv_r = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, stride, padding, padding_mode='circular',)
        self.conv_h = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, stride, padding, padding_mode='circular',)
        self.activation = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        # Local (2D) coupling: learnable Laplacian kernel
        #_init_conv(self.conv_z, hidden_channels)
        #_init_conv(self.conv_r, hidden_channels)
        #_init_conv(self.conv_h, hidden_channels)

    def forward(self, input, hidden):
        combined = torch.cat([input, hidden], dim=1)
        
        z = self.sigmoid(self.conv_z(combined))
        r = self.sigmoid(self.conv_r(combined))
        
        combined_reset = torch.cat([input, r * hidden], dim=1)
        h_tilde = self.activation(self.conv_h(combined_reset))
        
        hidden = ((1 - z) * hidden) + (z * h_tilde)
        
        return hidden


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3, stride=1, padding=1):
        super(ConvLSTMCell, self).__init__()

        self.conv = nn.Conv2d(input_channels + hidden_channels, 4 * hidden_channels, kernel_size, stride, padding, padding_mode='circular',)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        # Local (2D) coupling: learnable Laplacian kernel
        #_init_conv(self.conv, hidden_channels)

    def forward(self, input, hidden, cell):
        combined = torch.cat([input, hidden], dim=1)
        gates = self.conv(combined)
        input_gate, forget_gate, cell_gate, output_gate = torch.split(gates, hidden.size(1), dim=1)
        
        input_gate = self.sigmoid(input_gate)
        forget_gate = self.sigmoid(forget_gate)
        output_gate = self.sigmoid(output_gate)
        cell_gate = self.tanh(cell_gate)
        
        cell = (forget_gate * cell) + (input_gate * cell_gate)
        hidden = output_gate * self.tanh(cell)
        
        return hidden, cell
    

def _init_conv(conv, channels):
    laplacian_kernel = torch.tensor(
        [[ 0.,  1.,  0.],
            [ 1., -4.0,  1.],
            [ 0.,  1.,  0.]]
    ).reshape(1, 1, 3, 3)
    # Initialize each channel's kernel with the Laplacian
    with torch.no_grad():
        nn.init.constant_(conv.weight, 0)
        for i in range(channels):
            conv.weight[i:i+1, i:i+1].copy_(laplacian_kernel)