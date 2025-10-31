import os
from pathlib import Path
import torch
import cv2
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

def predict_masks(model_path, image_dir, provided_masks_dir, predicted_masks_dir, input_size=(512, 512)):
    """
    Predict binary masks for unlabeled images using the trained model.
    """

    # Initialize model
    model = smp.Unet(
        encoder_name="efficientnet-b4",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,   
    )

    # Load trained weights
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Validation transform (resize + normalize)
    val_transform = A.Compose([
        A.Resize(height=input_size[0], width=input_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # Ensure output directory exists
    os.makedirs(predicted_masks_dir, exist_ok=True)

    image_dir = Path(image_dir)
    provided_masks_dir = Path(provided_masks_dir)
    predicted_masks_dir = Path(predicted_masks_dir)

    # Identify unlabeled images (no existing mask in ../data/masks/)
    all_images = sorted(list(image_dir.glob("*.jpg")))
    unlabeled_images = [p for p in all_images if not (provided_masks_dir / f"{p.stem}_mask.png").exists()]

    print(f"Found {len(unlabeled_images)} unlabeled images.")

    # Inference loop
    for i, img_path in enumerate(unlabeled_images, start=1):
        # Read image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        # Apply validation transform
        transformed = val_transform(image=img)
        x = transformed["image"].unsqueeze(0).to(device)

        # Predict mask
        with torch.no_grad():
            pred = model(x)
            pred = torch.sigmoid(pred).squeeze().cpu().numpy()

        # Resize mask back to original size with bilinear interpolation and binarize
        mask_resized = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        mask_binary = (mask_resized > 0.5).astype(np.uint8)

        # Save as PNG (0/255)
        out_path = predicted_masks_dir / f"{img_path.stem}_mask.png"
        cv2.imwrite(str(out_path), mask_binary * 255)

        print(f"[{i}/{len(unlabeled_images)}] Saved: {out_path.name}")

    print("Mask prediction complete!")