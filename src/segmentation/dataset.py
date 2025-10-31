from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class PuzzleDataset(Dataset):
    """Dataset for puzzle piece segmentation"""

    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        
        # Get all image files
        self.image_files = sorted(list(self.img_dir.glob("*.jpg")))
        
        print(f"Found {len(self.image_files)} images in {img_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        try:
            # Load image
            img_path = self.image_files[idx]
            image = np.array(Image.open(img_path).convert("RGB"))
            
            # Load mask
            mask_name = img_path.stem + "_mask.png"
            mask_path = self.mask_dir / mask_name
            mask = np.array(Image.open(mask_path).convert("L"))
            
            # Normalize mask to 0 and 1
            mask = (mask > 0).astype(np.float32)
            
            # Apply transformations
            if self.transform:
                transformed = self.transform(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]
            
            return image, mask.unsqueeze(0)  # Add channel dimension to mask
        except Exception as e:
            print(f"Error loading {self.image_files[idx]}: {e}")
            raise

def get_train_transform(img_size):
    """Training data augmentation for puzzle piece segmentation"""
    return A.Compose([
        # Resize
        A.Resize(height=img_size[0], width=img_size[1]),

        # Geometric
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=180,
            border_mode=0,
            p=0.7
        ),
        
        # Lighting
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.5
        ),
        A.RandomGamma(gamma_limit=(70, 130), p=0.3),
        
        # Color
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.3
        ),
        
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_val_transform(img_size):
    """Validation data normalization"""
    return A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
