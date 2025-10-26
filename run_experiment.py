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
from typing import Dict, List
import torch
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import networkx as nx

def predict_masks(model_path: str, image_dir: str, provided_masks_dir: str, predicted_masks_dir: str, input_size=(256, 256)):
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

def extract_features(image_dir: str, provided_masks_dir: str, predicted_masks_dir: str) -> Dict[str, dict]:
    """
    Extract geometric and texture features for each puzzle piece.
    Returns dict: piece_id -> features
    """
    print("Extracting features from puzzle pieces...")
    features = {}

    image_dir = Path(image_dir)
    provided_masks_dir = Path(provided_masks_dir)
    predicted_masks_dir = Path(predicted_masks_dir)

    all_images = sorted(list(image_dir.glob("*.jpg")))

    for img_path in all_images:
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Select mask
        mask_path = provided_masks_dir / f"{img_path.stem}_mask.png"
        if not mask_path.exists():
            mask_path = predicted_masks_dir / f"{img_path.stem}_mask.png"

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask_bin = (mask > 127).astype(np.uint8)

        # Contour extraction
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue
        contour = max(contours, key=cv2.contourArea)

        # Polygon approximation
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        approx = approx.reshape(-1, 2)

        # Rotated rectangle for edge sides
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect).astype(int)

        # Count flat edges
        flat_edges = 0
        for i in range(4):
            pt1 = box[i]
            pt2 = box[(i+1)%4]
            dx = pt2[0] - pt1[0]
            dy = pt2[1] - pt1[1]
            angle = np.degrees(np.arctan2(dy, dx))
            if abs(angle) < 5 or abs(abs(angle)-180) < 5:
                flat_edges += 1

        # Color histograms along edges
        edge_histograms = []
        for i in range(4):
            pt1 = box[i]
            pt2 = box[(i+1)%4]
            line_x = np.linspace(pt1[0], pt2[0], num=10).astype(int)
            line_y = np.linspace(pt1[1], pt2[1], num=10).astype(int)
            colors = img_rgb[line_y, line_x, :]
            hist_r = np.histogram(colors[:,0], bins=8, range=(0,255))[0]
            hist_g = np.histogram(colors[:,1], bins=8, range=(0,255))[0]
            hist_b = np.histogram(colors[:,2], bins=8, range=(0,255))[0]
            hist = np.concatenate([hist_r, hist_g, hist_b])
            hist = hist / (hist.sum() + 1e-6)
            edge_histograms.append(hist)

        features[img_path.stem] = {
            "contour": contour,
            "approx_poly": approx,
            "box": box,
            "flat_edges": flat_edges,
            "edge_histograms": edge_histograms,
            "mask": mask_bin,
        }

    print(f"✓ Extracted features for {len(features)} pieces.")
    return features

def build_graph(features: Dict[str, dict], output_graph_dir: str, top_k=3, mutual_only=True) -> Dict[str, List[str]]:
    """
    Build adjacency graph based on edge histogram similarity.
    Optionally keep only mutual edges.
    Returns: graph dict, list of edge pieces
    """
    print("Building adjacency graph...")
    graph = {}
    piece_ids = list(features.keys())

    for pid in piece_ids:
        current = features[pid]
        scores = []
        for other_pid in piece_ids:
            if other_pid == pid:
                continue
            other = features[other_pid]

            # min distance between any pair of edges
            min_dist = np.inf
            for h1 in current['edge_histograms']:
                for h2 in other['edge_histograms']:
                    dist = np.linalg.norm(h1 - h2)
                    if dist < min_dist:
                        min_dist = dist
            scores.append((min_dist, other_pid))

        # keep top_k neighbors
        scores.sort(key=lambda x: x[0])
        graph[pid] = [pid2 for _, pid2 in scores[:top_k]]

    # Mutual filtering
    if mutual_only:
        mutual_graph = {}
        for pid, neighbors in graph.items():
            mutual_neighbors = [n for n in neighbors if pid in graph[n]]
            mutual_graph[pid] = mutual_neighbors
        graph = mutual_graph

    # Edge pieces (pieces with at least one flat edge)
    edge_pieces = [pid for pid, f in features.items() if f['flat_edges'] >= 1]

    print(f"✓ Graph built with {len(graph)} nodes, {len(edge_pieces)} edge pieces identified.")

    # Plot graph
    G = nx.Graph()
    for node, neighbors in graph.items():
        for n in neighbors:
            G.add_edge(node, n)

    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, seed=42)
    node_colors = ['red' if n in edge_pieces else 'skyblue' for n in G.nodes()]
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=300, font_size=8)
    plt.title("Puzzle Adjacency Graph (Red = Edge Pieces)")
    plt.savefig(output_graph_dir / "graph.png", dpi=150, bbox_inches='tight')
    plt.show()

    # Save adjacency graph as JSON
    with open(output_graph_dir / "graph.json", "w") as f:
        json.dump(graph, f, indent=2)

    print(f"✓ Graph visualization saved as 'graph.png' and adjacency saved as 'graph.json'.")

def assemble_puzzle(graph_path: str, features: Dict[str, dict], output_path: str):
    """Assemble puzzle based on graph and geometric relationships."""
    print("Assembling puzzle from adjacency graph...")
    # TODO: Use transformations (affine/projective) to align pieces
    # Example:
    # final_image = place_pieces(features, graph)
    # cv2.imwrite(output_path, final_image)
    print(f"Puzzle assembly complete. Saved to {output_path}")

def main(args):
    image_dir = Path(args.images_dir)
    provided_masks_dir = Path(args.provided_masks_dir)
    predicted_masks_dir = Path(args.predicted_masks_dir)
    output_graph_dir = Path(args.output_graph_dir)
    assembled_path = Path(args.output_final_path)

    print("[STEP 1] Predicting segmentation masks...")
    predict_masks(args.model_path, image_dir, provided_masks_dir, predicted_masks_dir)

    print("[STEP 2] Extracting features...")
    features = extract_features(image_dir, provided_masks_dir, predicted_masks_dir)

    print("[STEP 3] Building adjacency graph...")
    build_graph(features, output_graph_dir, top_k=3, mutual_only=True)

    # print("[STEP 4] Assembling final puzzle...")
    # assemble_puzzle(graph_path, features, assembled_path)

    # print("[DONE] All steps completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run end-to-end puzzle reconstruction experiment.")
    
    parser.add_argument("--model_path", type=str, default="./output/training/deeplabv3plus_resnet50_best.pth",
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
