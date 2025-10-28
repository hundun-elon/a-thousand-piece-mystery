import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import segmentation_models_pytorch as smp
import json
from datetime import datetime
import wandb

class Config:
    """Training configuration"""
    # Model
    MODEL_NAME = "unet"
    BACKBONE = "efficientnet-b4"
    IN_CHANNELS = 3
    NUM_CLASSES = 1  # Binary segmentation (piece vs background)
    
    # Data
    IMG_SIZE = (512, 512)  # Resize images to this size
    BATCH_SIZE = 16
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

class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation"""
    
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice

class CombinedLoss(nn.Module):
    """Combined BCE and Dice Loss"""
    
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    
    def forward(self, pred, target):
        return self.bce(pred, target) + self.dice(pred, target)

def get_train_transform(img_size):
    """Balanced training data augmentation for puzzle piece segmentation"""
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
    """Validation data augmentation (no augmentation, just normalization)"""
    return A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def calculate_iou(pred, target, threshold=0.5):
    """Calculate Intersection over Union (IoU)"""
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    target = target.float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + 1e-7) / (union + 1e-7)
    return iou.item()

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    
    pbar = tqdm(loader, desc="Training")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        iou = calculate_iou(outputs, masks)
        
        running_loss += loss.item()
        running_iou += iou
        
        pbar.set_postfix({'loss': loss.item(), 'iou': iou})
    
    epoch_loss = running_loss / len(loader)
    epoch_iou = running_iou / len(loader)
    
    return epoch_loss, epoch_iou

def validate_epoch(model, loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Calculate metrics
            iou = calculate_iou(outputs, masks)
            
            running_loss += loss.item()
            running_iou += iou
            
            pbar.set_postfix({'loss': loss.item(), 'iou': iou})
    
    epoch_loss = running_loss / len(loader)
    epoch_iou = running_iou / len(loader)
    
    return epoch_loss, epoch_iou

def main():
    ### W&B SETUP ###
    config = Config()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    wandb.init(
        project="puzzle-segmentation",
        name=f"{config.MODEL_NAME}_{config.BACKBONE}_{timestamp}",
        config={k: v for k, v in config.__dict__.items() if not k.startswith("__")}
    )

    ### SETUP ###
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()

    ### CONFIG ###

    print("\n" + "="*70)
    print("CONFIGURATION")
    print("="*70)
    print(f"Model: {config.MODEL_NAME} with {config.BACKBONE} backbone")
    print(f"Image size: {config.IMG_SIZE}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Learning rate: {config.LEARNING_RATE}")

    ### DATALOADERS ###
    print("\n" + "="*70)
    print("CREATING DATALOADERS")
    print("="*70)

    # Create datasets
    train_dataset = PuzzleDataset(
        config.TRAIN_IMG_DIR,
        config.TRAIN_MASK_DIR,
        transform=get_train_transform(config.IMG_SIZE)
    )

    val_dataset = PuzzleDataset(
        config.VAL_IMG_DIR,
        config.VAL_MASK_DIR,
        transform=get_val_transform(config.IMG_SIZE)
    )

    test_dataset = PuzzleDataset(
        config.TEST_IMG_DIR,
        config.TEST_MASK_DIR,
        transform=get_val_transform(config.IMG_SIZE)
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    print(f"\n✓ Training batches: {len(train_loader)}")
    print(f"✓ Validation batches: {len(val_loader)}")
    print(f"✓ Test batches: {len(test_loader)}")

    ### MODEL INIT ###
    print("\n" + "="*70)
    print("INITIALIZING MODEL")
    print("="*70)

    # Create model
    model = smp.Unet(
        encoder_name=config.BACKBONE,
        encoder_weights="imagenet",
        in_channels=config.IN_CHANNELS,
        classes=config.NUM_CLASSES,
    ).to(device)

    print(f"Model created: UNet with {config.BACKBONE} backbone")
    print(f"Encoder initialized with ImageNet weights")

    # Initialize loss
    criterion = CombinedLoss()
    print("\nLoss function: Combined BCE + Dice Loss")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=7
    )

    print("Optimizer: AdamW")
    print("Scheduler: ReduceLROnPlateau (monitors validation IoU)")

    ### TRAINING ###
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)

    # Training history
    history = {
        'train_loss': [],
        'train_iou': [],
        'val_loss': [],
        'val_iou': [],
        'lr': []
    }

    best_val_iou = 0.0
    patience_counter = 0
    early_stop_patience = 100

    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
        print("-" * 70)
        
        # Train
        train_loss, train_iou = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_iou = validate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_iou)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        history['lr'].append(current_lr)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")

        # Log to W&B
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_iou": train_iou,
            "val_loss": val_loss,
            "val_iou": val_iou,
            "learning_rate": current_lr
        })

        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            patience_counter = 0
            
            model_save_path = Path(config.MODEL_DIR) / f"{config.MODEL_NAME}_{config.BACKBONE}_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'config': config.__dict__
            }, model_save_path)
            
            print(f"✓ Best model saved! Val IoU: {val_iou:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best Validation IoU: {best_val_iou:.4f}")

    print("\n" + "="*70)

    ### PLOT TRAIN HISTORY ###
    print("PLOTTING TRAINING HISTORY")
    print("="*70)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # IoU
    axes[1].plot(history['train_iou'], label='Train IoU', linewidth=2)
    axes[1].plot(history['val_iou'], label='Val IoU', linewidth=2)
    axes[1].axhline(y=best_val_iou, color='r', linestyle='--', label=f'Best Val IoU: {best_val_iou:.4f}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('IoU')
    axes[1].set_title('Training and Validation IoU')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Learning Rate
    axes[2].plot(history['lr'], linewidth=2, color='green')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{config.OUTPUT_DIR}/{config.MODEL_NAME}_{config.BACKBONE}_training_history.png", dpi=150, bbox_inches='tight')
    # plt.show()

    print("✓ Training history plots saved")

    ### EVALUATE ###
    print("\n" + "="*70)
    print("EVALUATING ON TEST SET")
    print("="*70)

    # Load best model
    checkpoint = torch.load(Path(config.MODEL_DIR) / f"{config.MODEL_NAME}_{config.BACKBONE}_best.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Loaded best model weights")

    # Evaluate
    test_loss, test_iou = validate_epoch(model, test_loader, criterion, device)

    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  IoU: {test_iou:.4f}")

    ### SAVE RESULTS ###
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    results = {
        'model': config.MODEL_NAME,
        'backbone': config.BACKBONE,
        'image_size': config.IMG_SIZE,
        'batch_size': config.BATCH_SIZE,
        'epochs_trained': len(history['train_loss']),
        'best_val_iou': float(best_val_iou),
        'test_iou': float(test_iou),
        'test_loss': float(test_loss),
        'final_lr': float(history['lr'][-1]),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'history': {k: [float(v) for v in vals] for k, vals in history.items()}
    }

    results_path = Path(config.OUTPUT_DIR) / f"{config.MODEL_NAME}_{config.BACKBONE}results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to {results_path}")

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"\nFinal Results:")
    print(f"  Best Validation IoU: {best_val_iou:.4f}")
    print(f"  Test IoU: {test_iou:.4f}")

    wandb.finish()

if __name__ == "__main__":
    main()