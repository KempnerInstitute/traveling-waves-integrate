import os
import random
import numpy as np
from PIL import Image
import torchvision
import torchvision.transforms as transforms

# Chatgpt helped

def paste_digit_non_overlapping(canvas_img, canvas_mask, digit_img, digit_label):
    """
    Attempts to paste digit_img onto canvas_img at a location where it doesn't overlap any previously placed digits.
    The segmentation mask is updated accordingly.
    
    Args:
        canvas_img (PIL.Image): Grayscale canvas image.
        canvas_mask (PIL.Image): Segmentation mask (mode 'L').
        digit_img (PIL.Image): MNIST digit image (28x28).
        digit_label (int): MNIST digit (0-9). The mask label will be digit_label + 1.
        
    Returns:
        bool: True if the digit was placed successfully, False otherwise.
    """
    canvas_w, canvas_h = canvas_img.size
    digit_w, digit_h = digit_img.size
    new_label = digit_label + 1

    # Convert the canvas mask to a numpy array for checking.
    mask_array = np.array(canvas_mask)
    digit_array = np.array(digit_img)
    # Create a binary mask for the digit (digit exists where pixel > 0).
    digit_binary = (digit_array > 0).astype(np.uint8)

    # Try a number of attempts to place the digit without overlapping.
    for attempt in range(max_attempts):
        max_x = canvas_w - digit_w
        max_y = canvas_h - digit_h
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        # Check if the region in canvas_mask is completely empty (all zeros).
        region = mask_array[y:y+digit_h, x:x+digit_w]
        # Only consider the region where the digit will be placed (where digit_binary==1)
        if np.all(region[digit_binary == 1] == 0):
            # Valid placement found, paste the digit image.
            canvas_img.paste(digit_img, (x, y), digit_img)
            # Update the region in the mask: set new_label where digit exists.
            region[digit_binary == 1] = new_label
            mask_array[y:y+digit_h, x:x+digit_w] = region
            # Update the canvas_mask image.
            canvas_mask.paste(Image.fromarray(mask_array))
            return True
    # If no valid placement is found after max_attempts, skip this digit.
    return False

def eval_sim(x, y):
    for i in range(len(x)):
        for j in range(len(y)):
            assert not np.all(x[i] == y[j])

seed = 50
random.seed(seed)
np.random.seed(seed)

# --- Parameters ---
canvas_size = (128, 128)  # (width, height) of composite image
dataset_splits = {
    'train': 10000,
    'val': 1000,
    'test': 1000,
}
min_digits = 1
max_digits = 4
max_attempts = 200  # Maximum attempts to place a digit without overlapping

base_dir = 'data/datasets/multi_mnist/'
os.makedirs(base_dir, exist_ok=True)

# --- Load MNIST ---
# Increase size to 1.5x
transform = transforms.Compose([transforms.Resize((42, 42), interpolation=transforms.InterpolationMode.BILINEAR),
                                transforms.ToTensor()])
mnist_dataset = torchvision.datasets.MNIST(root='data/datasets/mnist/', 
                                           train=True, download=False, transform=transform)

# --- Create and save dataset splits ---
all_images = []
all_masks = []
for split, num_samples in dataset_splits.items():
    print(f"Creating {num_samples} samples for the {split} split...")
    images_list = []
    masks_list = []
    metadata_list = []  # To store the count of digits per imag
    
    for idx in range(num_samples):
        # Create blank canvases for image and mask (grayscale, mode 'L').
        canvas_img = Image.new('L', canvas_size, color=0)   # Black background for the image.
        canvas_mask = Image.new('L', canvas_size, color=0)    # Zero label for background.
        
        # Decide on the number of digits for this sample.
        num_digits = random.randint(min_digits, max_digits)
        placed_digits = 0
        
        for _ in range(num_digits):
            # Randomly select a digit from MNIST.
            rand_idx = random.randint(0, len(mnist_dataset) - 1)
            digit_img, digit_label = mnist_dataset[rand_idx]
            # Convert digit tensor to PIL image.
            digit_img = transforms.ToPILImage()(digit_img)
            # Optionally threshold to get a clean binary image.
            digit_img = digit_img.point(lambda p: 255 if p > 50 else 0)
            
            # Attempt to place the digit without overlapping.
            if paste_digit_non_overlapping(canvas_img, canvas_mask, digit_img, digit_label):
                placed_digits += 1
            else:
                # Could not place the digit without overlap after max_attempts.
                pass
        
        # If no digit was placed (all placements failed), we keep the blank canvas.
        # Convert the image and mask to NumPy arrays.
        image_np = np.array(canvas_img)
        mask_np = np.array(canvas_mask)
        images_list.append(image_np)
        masks_list.append(mask_np)
        metadata_list.append(placed_digits)
    
    # Stack the list into a NumPy array.
    images_np = np.stack(images_list)
    masks_np = np.stack(masks_list)
    metadata_np = np.array(metadata_list) # Shape: (num_samples,)

    all_images.append(images_np)
    all_masks.append(masks_np)
    
    # Save the arrays as .npy files.
    images_path = os.path.join(base_dir, f"{split}_images.npy")
    masks_path  = os.path.join(base_dir, f"{split}_masks.npy")
    metadata_path = os.path.join(base_dir, f"{split}_metadata.npy")
    np.save(images_path, images_np)
    np.save(masks_path, masks_np)
    np.save(metadata_path, metadata_np)
    print(f"Saved {split} images to {images_path}, masks to {masks_path}, and metadata to {metadata_path}.")

print("Evaluating similarity - making sure train, val, and test splits are all distinct.")
eval_sim(all_images[0], all_images[1])
eval_sim(all_images[0], all_images[2])
eval_sim(all_images[1], all_images[2])

print("All splits created and saved successfully!")