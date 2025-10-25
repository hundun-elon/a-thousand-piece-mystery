"""
run_experiment.py
---------------------------------------------------
End-to-end inference pipeline for:
1. Predicting segmentation masks
2. Extracting piece features
3. Building the adjacency graph
4. Assembling the final puzzle
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict
import torch
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np
from pathlib import Path
import cv2

def predict_masks(model_path: str, image_dir: str, output_dir: str, input_size=(256, 256)):
    """
    Predict binary masks for unlabeled images using the trained model.

    Args:
        model_path (str): Path to the trained .pth model weights.
        image_dir (str): Directory containing all puzzle piece images.
        output_dir (str): Directory where predicted masks will be saved.
        input_size (tuple): Image size (H, W) used during training.
    """

    print("Loading trained model...")

    # Initialize model
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
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
    os.makedirs(output_dir, exist_ok=True)

    image_dir = Path(image_dir)
    output_dir = Path(output_dir)

    # Identify unlabeled images (no existing mask in ../data/masks/)
    all_images = sorted(list(image_dir.glob("*.jpg")))
    masks_dir = image_dir.parent / "masks"
    unlabeled_images = [p for p in all_images if not (masks_dir / f"{p.stem}_mask.png").exists()]

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
        out_path = output_dir / f"{img_path.stem}_mask.png"
        cv2.imwrite(str(out_path), mask_binary * 255)

        print(f"[{i}/{len(unlabeled_images)}] Saved: {out_path.name}")

    print("Mask prediction complete!")

def extract_features(image_dir: str, mask_dir: str) -> Dict[str, dict]:
    """Extract geometric and texture features for each puzzle piece."""
    print("Extracting features from puzzle pieces...")
    # TODO: Compute descriptors, corners, edge embeddings, etc.
    features = {}
    # Example:
    # for mask_path in Path(mask_dir).glob("*_mask.png"):
    #     features[mask_path.stem.replace('_mask','')] = compute_piece_features(mask_path)
    print("Feature extraction complete.")
    return features

def build_adjacency_graph(features: Dict[str, dict], output_path: str):
    """Construct adjacency graph from extracted features."""
    print("Building adjacency graph...")
    # TODO: Compare features between pieces, compute similarity, threshold
    graph = {}
    # Example:
    # graph = compute_graph(features)
    with open(output_path, 'w') as f:
        json.dump(graph, f, indent=4)
    print(f"Adjacency graph saved to {output_path}")

def assemble_puzzle(graph_path: str, features: Dict[str, dict], output_path: str):
    """Assemble puzzle based on graph and geometric relationships."""
    print("Assembling puzzle from adjacency graph...")
    # TODO: Use transformations (affine/projective) to align pieces
    # Example:
    # final_image = place_pieces(features, graph)
    # cv2.imwrite(output_path, final_image)
    print(f"Puzzle assembly complete. Saved to {output_path}")

def main(args):
    image_dir = Path(args.images)
    mask_dir = Path(args.output_masks)
    graph_path = Path(args.output_graph)
    assembled_path = Path(args.output_final)

    print("[STEP 1] Predicting segmentation masks...")
    predict_masks(args.model, image_dir, mask_dir)

    # print("[STEP 2] Extracting features...")
    # features = extract_features(image_dir, mask_dir)

    # print("[STEP 3] Building adjacency graph...")
    # build_adjacency_graph(features, graph_path)

    # print("[STEP 4] Assembling final puzzle...")
    # assemble_puzzle(graph_path, features, assembled_path)

    # print("[DONE] All steps completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run end-to-end puzzle reconstruction experiment.")
    
    parser.add_argument("--model", type=str, default="./output/training/deeplabv3plus_resnet50_best.pth",
                        help="Path to the trained segmentation model.")
    
    parser.add_argument("--images", type=str, default="./data/images/",
                        help="Directory containing puzzle piece images.")
    
    parser.add_argument("--output_masks", type=str, default="./output/predicted_masks",
                        help="Output directory for predicted masks.")
    
    parser.add_argument("--output_graph", type=str, default="./output/graph.json",
                        help="Output path for adjacency graph JSON.")
    
    parser.add_argument("--output_final", type=str, default="./output/final_puzzle.png",
                        help="Output path for final assembled puzzle image.")

    args = parser.parse_args()
    main(args)
