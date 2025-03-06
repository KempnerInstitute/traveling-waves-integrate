import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np

# ChatGPT helped make these functions


"""
Calculate the average pixel-wise accuracy for each image in the batch, ignoring a class.

Args:
    predictions (torch.Tensor): Predicted segmentation masks, shape (B, H, W)
    ground_truth (torch.Tensor): Ground truth segmentation masks, shape (B, H, W)
    ignore_class (int, optional): Class to ignore in accuracy calculation. Default is 0.

Returns:
    torch.Tensor: Average pixel-wise accuracy for each image, shape (B,)
"""
def calc_acc(predictions, ground_truth, ignore_class=0):
    # Ensure the tensors are of the same shape
    if predictions.shape != ground_truth.shape:
        raise ValueError("Shape of predictions and ground_truth must match.")

    # Create a mask for pixels that are NOT the ignore_class
    mask = ground_truth != ignore_class  # Shape: (B, H, W)

    # Calculate correct predictions where mask is True
    correct = (predictions == ground_truth) & mask  # Shape: (B, H, W)

    # Sum correct predictions per image
    correct_per_image = correct.view(correct.size(0), -1).sum(dim=1).float()  # Shape: (B,)

    # Sum valid (non-ignored) pixels per image
    total_per_image = mask.view(mask.size(0), -1).sum(dim=1).float()  # Shape: (B,)

    # Calculate accuracy per image
    accuracy = correct_per_image / total_per_image  # Shape: (B,)
    accuracy = accuracy.cpu().numpy()
    return np.sum(accuracy)

"""
Calculate the mean Intersection over Union (IoU) for each image in the batch, ignoring a class.

Args:
    predictions (torch.Tensor): Predicted segmentation masks, shape (B, H, W)
    ground_truth (torch.Tensor): Ground truth segmentation masks, shape (B, H, W)
    ignore_class (int, optional): Class to ignore in IoU calculation. Default is 0.
    num_classes (int, optional): Total number of classes. If None, inferred from data.

Returns:
    torch.Tensor: Mean IoU for each image, shape (B,)
"""
def calc_iou(predictions, ground_truth, ignore_class=0, num_classes=None):
    
    if predictions.shape != ground_truth.shape:
        raise ValueError("Shape of predictions and ground_truth must match.")

    if num_classes is None:
        num_classes = int(max(predictions.max(), ground_truth.max()) + 1)

    batch_size = predictions.size(0)

    # Create a mask to ignore the specified class
    mask = ground_truth != ignore_class  # Shape: (B, H, W)
    mask = mask.unsqueeze(1)

    # Expand dimensions to (B, C, H, W) for one-hot encoding
    predictions_one_hot = torch.nn.functional.one_hot(predictions, num_classes=num_classes).permute(0, 3, 1, 2)  # (B, C, H, W)
    ground_truth_one_hot = torch.nn.functional.one_hot(ground_truth, num_classes=num_classes).permute(0, 3, 1, 2)  # (B, C, H, W)

    # APPLY MASK
    predictions_one_hot = predictions_one_hot * mask
    ground_truth_one_hot = ground_truth_one_hot * mask
    intersection = (predictions_one_hot & ground_truth_one_hot) # B x C x H x W
    union = (predictions_one_hot | ground_truth_one_hot) # B x C x H x W

    intersection = intersection.sum((1, 2, 3))
    union = union.sum((1, 2, 3))
    iou = intersection / union
    iou = iou.cpu().numpy()
    return np.sum(iou)

def compute_pixelwise_accuracy(pred_mask, true_mask):
    """
    Compute pixel-wise accuracy for each element in the batch without explicit looping.
    
    Args:
    pred_mask, true_mask: (B, H, W) with integer labels in [0..n_classes-1]
    
    Returns:
    A tensor of pixel-wise accuracies for each item in the batch.
    """
    # Ensure that predictions and ground truth have the same shape
    assert pred_mask.shape == true_mask.shape, "Shape mismatch between prediction and ground truth"

    # Calculate element-wise correctness
    correct_pixels = (pred_mask == true_mask).float()  # (B, H, W)

    # Calculate total pixels per batch element
    total_pixels_per_batch = true_mask.size(1) * true_mask.size(2)  # H * W

    # Sum correct predictions over each (H, W) for each batch
    correct_per_batch = correct_pixels.view(pred_mask.size(0), -1).sum(dim=1)  # (B,)

    # Compute pixel-wise accuracy per batch element
    pixel_wise_accuracies = correct_per_batch / total_pixels_per_batch  # (B,)

    return pixel_wise_accuracies.sum().item()


def compute_iou(masks1, masks2):
    """
    Calculate IoU for batches of segmentation masks, ignoring specific values.
    
    Parameters:
        masks1 (torch.Tensor): A batch of ground truth masks with shape (batch_size, height, width).
        masks2 (torch.Tensor): A batch of predicted masks with shape (batch_size, height, width).
        ignore_value (int): Specifies the value in masks1 to be ignored.
    
    Returns:
        torch.Tensor: A tensor containing IoU scores for each item in the batch.
    """
    batch_size = masks1.shape[0]
    
    # Flatten the height and width dimensions for vectorized operations
    masks1 = masks1.view(batch_size, -1)
    masks2 = masks2.view(batch_size, -1)
    
    # Compute intersection and union where the valid mask is True
    intersection = (masks1 & masks2).sum(dim=1).float()
    union = (masks1 | masks2).sum(dim=1).float()

    # Compute IoU, ignoring invalid areas
    iou = torch.where(union > 0, intersection / union, torch.tensor(float('nan')))
    
    return iou.sum().item()