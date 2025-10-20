"""Configuration settings for the project."""

# Segmentation configuration
SEGMENTATION_CONFIG = {
    "model_type": "unet",  # or "maskrcnn", "deeplabv3", etc.
    "input_size": (512, 512),
    "batch_size": 8,
    "epochs": 50,
    "learning_rate": 0.001,
}

# Matching configuration
MATCHING_CONFIG = {
    "feature_type": "combined",  # shape, texture, or combined
    "similarity_threshold": 0.7,
    "use_pca": True,
    "pca_components": 64,
}

# Assembly configuration
ASSEMBLY_CONFIG = {
    "canvas_size": (4000, 3000),  # Adjust based on puzzle size
    "transformation_type": "projective",  # affine or projective
}

# Paths
PATHS = {
    "train_images": "./data/train/images",
    "train_masks": "./data/train/masks",
    "test_images": "./data/test/images",
    "output_masks": "./data/masks",
    "model_weights": "./models/weights",
}
