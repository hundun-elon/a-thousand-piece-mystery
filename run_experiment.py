"""
run_experiment.py
---------------------------------------------------
End-to-end inference pipeline for:
1. Predicting segmentation masks
2. Extracting piece features
3. Building the adjacency graph
4. Assembling the final puzzle
"""

import argparse
from pathlib import Path

from src.segmentation.inference import predict_masks

def main(args):
    image_dir = Path(args.images_dir)
    provided_masks_dir = Path(args.provided_masks_dir)
    predicted_masks_dir = Path(args.predicted_masks_dir)
    output_graph_dir = Path(args.output_graph_dir)
    assembled_path = Path(args.output_final_path)

    print("[STEP 1] Predicting segmentation masks...")
    predict_masks(args.model_path, image_dir, provided_masks_dir, predicted_masks_dir)

    # print("[STEP 2] Extracting features...")

    # print("[STEP 3] Matching sides...")

    # print("[STEP 3] Building adjacency graph...")

    # print("[STEP 4] Assembling final puzzle...")

    # print("[DONE] All steps completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run end-to-end puzzle reconstruction experiment.")
    
    parser.add_argument("--model_path", type=str, default="./output/training/unet_20251028_192417/unet_efficientnet-b4_best.pth",
                        help="Path to the trained segmentation model.")
    
    parser.add_argument("--images_dir", type=str, default="./data/images/",
                        help="Directory containing puzzle piece images.")
    
    parser.add_argument("--predicted_masks_dir", type=str, default="./output/predicted_masks/",
                        help="Output directory for predicted masks.")
    
    parser.add_argument("--provided_masks_dir", type=str, default="./data/masks/",
                        help="Directory containing provided masks")
    
    parser.add_argument("--output_graph_dir", type=str, default="./output/",
                        help="Output directory for adjacency graph.")
    
    parser.add_argument("--output_final_path", type=str, default="./output/final_puzzle.png",
                        help="Output path for final assembled puzzle image.")

    args = parser.parse_args()
    main(args)
