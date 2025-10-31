from pathlib import Path
from datetime import datetime

class Config:
    """Training configuration"""
    # Model
    MODEL_NAME = "unet"
    BACKBONE = "efficientnet-b4"
    IN_CHANNELS = 3
    NUM_CLASSES = 1  # Binary segmentation (piece vs background)
    
    # Data
    IMG_SIZE = (512, 512)  # Resize images to this size
    BATCH_SIZE = 8
    NUM_WORKERS = 8
    
    # Training
    EPOCHS = 200
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # Paths
    TRAIN_IMG_DIR = "./data/train/images"
    TRAIN_MASK_DIR = "./data/train/masks"
    VAL_IMG_DIR = "./data/val/images"
    VAL_MASK_DIR = "./data/val/masks"
    TEST_IMG_DIR = "./data/test/images"
    TEST_MASK_DIR = "./data/test/masks"
    
    # Timestamp
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Output paths with timestamp
    MODEL_DIR = f"./output/training/{MODEL_NAME}_{TIMESTAMP}"
    OUTPUT_DIR = f"./output/training/{MODEL_NAME}_{TIMESTAMP}"
    
    # Create directories
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
