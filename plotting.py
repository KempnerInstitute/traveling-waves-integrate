import matplotlib.pyplot as plt
import wandb
import numpy as np
import torch
import matplotlib.animation as animation
import seaborn as sns
import wandb

def plot_hidden_state_video(y_seq, wandb_name, sample_idx=0, channel=0, interval=200, fpath="curr"):
    """
    Given y_seq of shape (B,T,C,H,W), animate the hidden state for the sample
    `sample_idx` across timesteps T.
    
    - `interval` controls the animation speed (milliseconds between frames).
    - returns: HTML object that, when displayed in Jupyter, shows the animation.
    """
    sns.set_theme()

    B, T, C, H, W = y_seq.shape
    y_seq = y_seq[:,:,channel]
    y_seq = torch.transpose(y_seq, 0, 1).detach()
    assert 0 <= sample_idx < B, f"sample_idx must be in [0..{B-1}]"
    
    # Subsample to 100 frames if sequence is too long
    if T > 100:
        indices = np.linspace(0, T-1, 100, dtype=int)
        y_seq = y_seq[indices]
        T = 100
    
    # We'll animate frames across t=0..T-1
    y_seq_np = y_seq[:, sample_idx].cpu().numpy()  # -> (T,H,W)
    
    # We can pick vmin/vmax across the entire timeseries for a stable color scale
    vmin = y_seq_np.min()
    vmax = y_seq_np.max()
    
    fig, ax = plt.subplots()
    im = ax.imshow(y_seq_np[0], cmap='bwr', vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax)
    
    def animate(t):
        im.set_array(y_seq_np[t])
        ax.set_xlabel(f"t = {t}")
        return [im]
    
    ani = animation.FuncAnimation(
        fig, animate, 
        frames=T, 
        interval=interval, 
        blit=True
    )

    fpath = f"{fpath}.gif"
    ani.save(fpath, writer='pillow', fps=5)
    plt.close(fig)
    wandb.log({f"{wandb_name}_s-{sample_idx}_c-{channel}": wandb.Video(fpath)})


def plot_results(x2, pred, S, idx, x2_cmap):
    if x2 is not None:
        num_ft_maps = x2.shape[1]
        fig, axes = plt.subplots(1, num_ft_maps, figsize=(num_ft_maps * 2, 2))
        for i in range(len(axes)):
            axes[i].imshow(x2[idx][i], cmap=x2_cmap)
        plt.savefig("fig1")
        plt.show()

    if pred is not None:
        num_preds = pred.shape[1]
        fig, axes = plt.subplots(1, num_preds, figsize=(num_preds * 2, 2))
        for i in range(len(axes)):
            axes[i].imshow(pred[idx][i], cmap='gray')
        plt.savefig("fig2")
        plt.show()

    if S is not None:
        num_freq = S.shape[1]
        fig, axes = plt.subplots(1, num_freq, figsize=(num_freq * 2, 2))
        for i in range(num_freq):
            axes[i].imshow(S[idx][i], cmap='gist_rainbow')
        plt.savefig("fig3")
        plt.show()

def plot_phases(phases):
    w = 20
    h = int(len(phases) / w)
    fig, axes = plt.subplots(h, w, figsize=(w, h))
    count = 0
    for i in range(h):
        for j in range(w):
            if h != 1:
                axes[i][j].imshow(phases[count].reshape(32, 32),
                                  cmap='gist_rainbow')
                axes[i][j].set_xticks([])
                axes[i][j].set_yticks([])
            else:
                axes[j].imshow(phases[count].reshape(32, 32), 
                               cmap='gist_rainbow')
                axes[j].set_xticks([])
                axes[j].set_yticks([])
            count += 1
    plt.show()


def plot_eval(x, x_target, x_pred, pred_diff, pred_mask, epoch):
    b, c, n, n = x.shape
    if c == 1:
        x = np.squeeze(x, axis=1)
        x_pred = np.squeeze(x_pred, axis=1)
        pred_diff = np.squeeze(pred_diff, axis=1)
        cmap = 'gray'
    elif c == 3:
        x = np.transpose(x, (0, 2, 3, 1))
        x_pred = np.transpose(x_pred, (0, 2, 3, 1))
        pred_diff = np.transpose(pred_diff, (0, 2, 3, 1))
        cmap = None
    else:
        print(f"ERROR: c={c} has to be 1 or 3")
        exit()

    fig, axes = plt.subplots(b, 5, figsize=(10, 8))
    for i in range(b):
        axes[i][0].imshow(x[i], cmap=cmap)
        axes[i][1].imshow(x_target[i], cmap=cmap)
        axes[i][2].imshow(x_pred[i], cmap=cmap)
        axes[i][3].imshow(pred_mask[i], cmap=cmap)
        axes[i][4].imshow(pred_diff[i], cmap=cmap)
        axes[i][0].axis("off")
        axes[i][1].axis("off")
        axes[i][2].axis("off")
        axes[i][3].axis("off")
        axes[i][4].axis("off")
    axes[0][0].set_title("GT Image")
    axes[0][1].set_title("GT Mask")
    axes[0][2].set_title("Pred Image")
    axes[0][3].set_title("Pred Mask")
    axes[0][4].set_title("Image Diff")
    wandb.log({"test_imgs_v1": plt}, step=epoch)
    plt.close()


def plot_slots(slots, epoch, num_samples=5, plot_name=''):
    B, S, C, N, M = slots.shape
    if C == 1:
        slots = np.squeeze(slots, 2)
        cmap = 'gray'
    else:
        slots = np.transpose(slots, (0, 1, 3, 4, 2))
        cmap = None
    fig, axes = plt.subplots(num_samples, S, figsize=(S, 8)) 
    for i in range(num_samples):
        for j in range(S):
            axes[i][j].imshow(slots[i, j], cmap=cmap)
            axes[i][j].axis("off")
    wandb.log({f"Slots_{plot_name}": plt}, step=epoch)
    plt.close()

def plot_clusters(x, x_target, pred_mask, slots, epoch, num_samples=5, plot_name=''):
    B, S, C, N, M = slots.shape
    if C == 1:
        slots = np.squeeze(slots, 2)
        cmap = 'gray'
    else:
        slots = np.transpose(slots, (0, 1, 3, 4, 2))
        cmap = None
    fig, axes = plt.subplots(num_samples, S + 3, figsize=(S + 3, 8)) 
    for i in range(num_samples):
        for j in range(S + 3):
            if j == 0:
                axes[i][j].imshow(x[i])
            elif j == 1:
                axes[i][j].imshow(x_target[i])
            elif j == 2:
                axes[i][j].imshow(pred_mask[i])
            else:
                axes[i][j].imshow(slots[i, j - 3], cmap=cmap)
            axes[i][j].axis("off")
    wandb.log({f"Clusters_{plot_name}": plt}, step=epoch)
    plt.close()


"""
Parameters
x: (batch_size x num_fourier x N x N) array
epoch: the int epoch to log this plot in wandb
global_scale: if True, scales colormap using np.min(x[i]), np.max(x[i]). Otherwise, local to that image.
plot_name: the name to log the plot in wandb. Default is None (doesn't log to wandb)

Returns
None
"""
def plot_fourier(x, epoch, global_scale=True, plot_name=None, max_plots=40):
    batch_size, num_fourier = x.shape[0], x.shape[1]

    step = max(num_fourier // max_plots, 1)
    num_f_plot = num_fourier // step

    fig, axes = plt.subplots(batch_size, num_f_plot, figsize=(num_f_plot, batch_size))
    for i in range(batch_size):
        if global_scale:
            vmin, vmax = np.min(x[i]), np.max(x[i])
        else:
            vmin, vmax = None, None
        for j in range(num_f_plot):
            axes[i][j].imshow(x[i][j * step], cmap='gray', vmin=vmin, vmax=vmax)
            axes[i][j].set_xticks([])
            axes[i][j].set_yticks([])
    if plot_name is not None:
        wandb.log({plot_name: plt}, step=epoch)
    plt.close()


"""
Parameters
phases: (batch_size x timesteps x N x N) array
epoch: the int epoch to log this plot in wandb
global_scale: if True, scales colormap using np.min(x[i]), np.max(x[i]). Otherwise, local to that image.
plot_name: the name to log the plot in wandb. Default is None (doesn't log to wandb)

Returns
None
"""
def plot_phases2(phases, epoch, global_scale=True, plot_name=None, max_plots=40):
    batch_size, timesteps = phases.shape[0], phases.shape[1]
    
    step = max(timesteps // max_plots, 1)
    num_plots = timesteps // step
    
    fig, axes = plt.subplots(batch_size, num_plots, figsize=(num_plots, batch_size))
    for i in range(batch_size):
        if global_scale:
            vmin, vmax = np.min(phases[i]), np.max(phases[i])
        else:
            vmin, vmax = None, None
        for j in range(num_plots):
            axes[i][j].imshow(phases[i][j * step], cmap='gist_rainbow', vmin=vmin, vmax=vmax)
            axes[i][j].set_xticks([])
            axes[i][j].set_yticks([])
    if plot_name is not None:
        wandb.log({plot_name: plt}, step=epoch)
    plt.close()


def plot_masks(masks, epoch, plot_name=None,):
    batch_size, C = masks.shape[0], masks.shape[1] 
    
    fig, axes = plt.subplots(batch_size, C, figsize=(C, batch_size))
    for i in range(batch_size):
        vmin, vmax = None, None
        for j in range(C):
            axes[i][j].imshow(masks[i][j], vmin=vmin, vmax=vmax)
            axes[i][j].set_xticks([])
            axes[i][j].set_yticks([])
    if plot_name is not None:
        wandb.log({plot_name: plt}, step=epoch)
    plt.close()


"""
mask: batch x n x n (filled with index values)
num_slots: integer
"""
def build_color_mask(mask, mask_colors, num_slots):
    img_size = mask.shape[-1]
    #mask_colors = np.random.randint(0, 256, size=(num_slots, 3))
    colored_mask = np.zeros((mask.shape[0], img_size, img_size, 3), dtype=np.uint8)
    # Assign colors to each pixel based on class
    for i in range(num_slots):
        colored_mask[mask == i] = mask_colors[i]
    return colored_mask

def plot_clusters2(x, x_target, x_target2, x_classifier, y_kmeans, y_agglomerative, epoch, num_samples=5, plot_name=''):
    fig, axes = plt.subplots(num_samples, 6, figsize=(6, num_samples + 3)) 
    for i in range(num_samples):
        axes[i][0].imshow(x[i])
        axes[i][1].imshow(x_target[i])
        axes[i][2].imshow(x_target2[i])
        axes[i][3].imshow(x_classifier[i])
        axes[i][4].imshow(y_kmeans[i])
        axes[i][5].imshow(y_agglomerative[i])
        axes[i][0].set_title("X")
        axes[i][1].set_title("GT_C")
        axes[i][2].set_title("GT_I")
        axes[i][3].set_title("XC")
        axes[i][4].set_title("KM")
        axes[i][5].set_title("AGG")
        axes[i][0].axis("off")
        axes[i][1].axis("off")
        axes[i][2].axis("off")
        axes[i][3].axis("off")
        axes[i][4].axis("off")
        axes[i][5].axis("off")
    wandb.log({f"Clusters_{plot_name}": plt}, step=epoch)
    plt.close()

def plot_clusters3(x, x_target, x_classifier, epoch, plot_name='', num_classes=21):
    num_samples = len(x)
    width = 3
    fig, axes = plt.subplots(num_samples, width, figsize=(width, num_samples)) 
    for i in range(num_samples):
        axes[i][0].imshow(x[i])
        #axes[i][1].imshow(x_target[i], vmin=0, vmax=num_classes - 1)
        #axes[i][2].imshow(x_classifier[i], vmin=0, vmax=num_classes - 1)
        axes[i][1].imshow(x_target[i])
        axes[i][2].imshow(x_classifier[i])
        axes[i][0].set_title("X")
        axes[i][1].set_title("GT_C")
        axes[i][2].set_title("XC")
        axes[i][0].axis("off")
        axes[i][1].axis("off")
        axes[i][2].axis("off")
    wandb.log({f"Masks_{plot_name}": plt}, step=epoch)
    plt.close()