DATASET_CONFIG = {
    'new_tetronimoes': {
        'x_train_path': "/data/datasets/new_tetrominoes/train_images.npy", # WRITE ABSOLUTE PATHS
        'y_train_path': "/data/datasets/new_tetrominoes/train_masks.npy",
        'x_val_path': "/data/datasets/new_tetrominoes/val_images.npy",
        'y_val_path': "/data/datasets/new_tetrominoes/val_masks.npy",
        'x_test_path': "/data/datasets/new_tetrominoes/test_images.npy",
        'y_test_path': "/data/datasets/new_tetrominoes/test_masks.npy",
        'img_size': 64,
        'channels': 3,
    },
    'multi_mnist': {
        'x_train_path': "/data/datasets/multi_mnist/train_images.npy",
        'y_train_path': "/data/datasets/multi_mnist/train_masks.npy",
        'x_val_path': "/data/datasets/multi_mnist/val_images.npy",
        'y_val_path': "/data/datasets/multi_mnist/val_masks.npy",
        'x_test_path': "/data/datasets/multi_mnist/test_images.npy",
        'y_test_path': "/data/datasets/multi_mnist/test_masks.npy",
        'img_size': 128,
        'channels': 1,
    },
    'mnist': {
        'train_path': '/data/datasets/mnist/',
        'test_path': '/data/datasets/mnist/',
        'img_size': 56,  # Image size for resizing
        'channels': 1, # originally 1, but color jitter makes it 3
    },
}