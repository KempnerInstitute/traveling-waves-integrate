import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# Define tetromino shapes with a priority order for overlap resolution
tetromino_shapes = {
    'I': (np.array([[1, 1, 1, 1]], dtype=int), 1),
    'O': (np.array([[1, 1], [1, 1]], dtype=int), 2),
    'T': (np.array([[0, 1, 0], [1, 1, 1]], dtype=int), 3),
    'S/Z': (np.array([[0, 1, 1], [1, 1, 0]], dtype=int), 4),
    'J/L': (np.array([[1, 0, 0], [1, 1, 1]], dtype=int), 5)
}

# Function to generate a random color
def generate_random_color():
    return np.random.randint(0, 256, size=3, dtype=np.uint8)

# Function to scale up tetromino shapes
def scale_up(shape, scale=7):
    """Scale up tetromino by the specified scale factor."""
    return np.kron(shape, np.ones((scale, scale), dtype=int))

# Function to add texture to each square
def add_square_texture(image, scaled_shape, start_row, start_col, color, scale=7):
    texture_color = color // 2  # Slightly darker color for texture
    for row in range(0, scaled_shape.shape[0], scale):
        for col in range(0, scaled_shape.shape[1], scale):
            if np.any(scaled_shape[row:row + scale, col:col + scale]):
                texture_mask = np.zeros((scale, scale), dtype=bool)
                texture_mask[0, :] = 1
                texture_mask[-1, :] = 1
                texture_mask[:, 0] = 1
                texture_mask[:, -1] = 1
                
                for i in range(3):
                    image[start_row + row:start_row + row + scale, start_col + col:start_col + col + scale, i] = \
                        np.clip(image[start_row + row:start_row + row + scale, start_col + col:start_col + col + scale, i] + texture_mask * texture_color[i], 0, 255)

# Randomly rotate or flip tetrominos
def rotate_or_flip_tetromino(shape, is_mirrored):
    shape = np.rot90(shape, np.random.choice([0, 1, 2, 3]))
    if is_mirrored and np.random.rand() < 0.5:
        shape = np.fliplr(shape)
    return shape

# OVERLAP
def place_tetromino2(image, mask, shape, label, priority, is_mirrored=False):
    shape = rotate_or_flip_tetromino(shape, is_mirrored)
    scaled_shape = scale_up(shape)
    color = generate_random_color()

    rows, cols = scaled_shape.shape
    grid_height, grid_width = image.shape[0], image.shape[1]

    start_row, start_col = np.random.randint(0, grid_height - rows), np.random.randint(0, grid_width - cols)
    
    for i in range(3):  # For each color channel
        image[start_row:start_row + rows, start_col:start_col + cols, i] = np.where(
            (scaled_shape == 1) & ((mask[start_row:start_row + rows, start_col:start_col + cols] == 0) |
                                   (mask[start_row:start_row + rows, start_col:start_col + cols] < priority)),
            (scaled_shape * color[i]),
            image[start_row:start_row + rows, start_col:start_col + cols, i]
        )

    add_square_texture(image, scaled_shape, start_row, start_col, color)
    
    mask[start_row:start_row + rows, start_col:start_col + cols] = np.where(
        (scaled_shape == 1) & ((mask[start_row:start_row + rows, start_col:start_col + cols] == 0) |
                               (mask[start_row:start_row + rows, start_col:start_col + cols] < priority)),
        priority,
        mask[start_row:start_row + rows, start_col:start_col + cols]
    )
    
    return image, mask

# NO OVERLAP
def place_tetromino(image, mask, shape, label, priority, is_mirrored=False):
    shape = rotate_or_flip_tetromino(shape, is_mirrored)
    scaled_shape = scale_up(shape)
    color = generate_random_color()

    rows, cols = scaled_shape.shape
    grid_height, grid_width = image.shape[0], image.shape[1]

    valid_placement = False

    while not valid_placement:
        start_row, start_col = np.random.randint(0, grid_height - rows), np.random.randint(0, grid_width - cols)
        # Check for overlap: the mask in this region should be all zeros
        if np.all(mask[start_row:start_row + rows, start_col:start_col + cols] == 0):
            valid_placement = True

    for i in range(3):  # For each color channel
        image[start_row:start_row + rows, start_col:start_col + cols, i] = np.where(
            scaled_shape == 1,
            scaled_shape * color[i],
            image[start_row:start_row + rows, start_col:start_col + cols, i]
        )

    add_square_texture(image, scaled_shape, start_row, start_col, color)
    
    mask[start_row:start_row + rows, start_col:start_col + cols] = np.where(
        scaled_shape == 1,
        priority,
        mask[start_row:start_row + rows, start_col:start_col + cols]
    )
    
    return image, mask

def generate_tetromino_image_and_mask(grid_size=64, max_tetrominoes=5):
    image = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
    mask = np.zeros((grid_size, grid_size), dtype=int)
    
    num_tetrominoes = np.random.randint(1, max_tetrominoes + 1)
    available_shapes = list(tetromino_shapes.items()) * (num_tetrominoes // len(tetromino_shapes) + 1)
    np.random.shuffle(available_shapes)
    
    for i in range(num_tetrominoes):
        name, (shape, priority) = available_shapes[i]
        label = list(tetromino_shapes.keys()).index(name) + 1
        is_mirrored = name in ['S/Z', 'J/L']
        image, mask = place_tetromino(image, mask, shape, label, priority, is_mirrored)
    
    return image, mask

def generate_and_stack_dataset(num_samples):
    images = []
    masks = []
    
    for _ in tqdm(range(num_samples), desc="Generating data"):
        image, mask = generate_tetromino_image_and_mask()
        images.append(image)
        masks.append(mask)
    
    images_array = np.stack(images, axis=0)
    masks_array = np.stack(masks, axis=0)
    
    return images_array, masks_array

def save_arrays(base_dir, images_array, masks_array, set_name):
    np.save(os.path.join(base_dir, f'{set_name}_images.npy'), images_array)
    np.save(os.path.join(base_dir, f'{set_name}_masks.npy'), masks_array)

def eval_sim(x, y):
    for i in range(len(x)):
        for j in range(len(y)):
            assert not np.all(x[i] == y[j])

def create_tetromino_dataset(base_dir='tetromino_dataset', train_size=800, val_size=100, test_size=100):    
    np.random.seed(42)  # Set a consistent random seed for reproducibility

    train_images, train_masks = generate_and_stack_dataset(train_size)
    save_arrays(base_dir, train_images, train_masks, 'train')
    
    val_images, val_masks = generate_and_stack_dataset(val_size)
    save_arrays(base_dir, val_images, val_masks, 'val')
    
    test_images, test_masks = generate_and_stack_dataset(test_size)
    save_arrays(base_dir, test_images, test_masks, 'test')

    eval_sim(train_images, test_images)
    eval_sim(val_images, test_images)


np.random.seed(0)
create_tetromino_dataset(base_dir='data/datasets/new_tetrominoes/', train_size=10000, val_size=1000, test_size=1000)