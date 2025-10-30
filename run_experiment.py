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

def predict_masks(model_path: str, image_dir: str, provided_masks_dir: str, predicted_masks_dir: str, input_size=(512, 512)):
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

def extract_features(image_dir: str, provided_masks_dir: str, predicted_masks_dir: str) -> Dict[str, dict]:
    """
    Extract features from puzzle pieces including contours, corners, sides, and piece types.
    Uses color information along sides for better matching.
    
    Returns:
        Dict mapping piece_id to features including:
        - contours, corners, sides, piece_type, side_types, feature_vectors
    """
    from scipy.interpolate import interp1d
    from scipy.spatial import distance
    
    image_dir = Path(image_dir)
    provided_masks_dir = Path(provided_masks_dir)
    predicted_masks_dir = Path(predicted_masks_dir)
    
    features = {}
    all_images = sorted(list(image_dir.glob("*.jpg")))
    
    print(f"Extracting features from {len(all_images)} pieces...")
    
    for img_path in all_images:
        piece_id = img_path.stem
        
        # Load image
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load mask from provided or predicted
        mask_path = provided_masks_dir / f"{piece_id}_mask.png"
        if not mask_path.exists():
            mask_path = predicted_masks_dir / f"{piece_id}_mask.png"
        
        if not mask_path.exists():
            print(f"Warning: No mask found for {piece_id}, skipping")
            continue
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # 1. Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            continue
        contour = max(contours, key=cv2.contourArea)
        contour = contour.squeeze()
        
        if len(contour.shape) != 2 or contour.shape[0] < 4:
            continue
        
        # Get bounding box to find actual edges
        x, y, w, h = cv2.boundingRect(contour)
        img_h, img_w = binary.shape
        
        # 2. Find corners - use minimum area rectangle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int8(box)
        
        # Refine corners by finding nearest contour points
        corners = []
        for corner_approx in box:
            dists = np.linalg.norm(contour - corner_approx, axis=1)
            idx = np.argmin(dists)
            corners.append(contour[idx])
        corners = np.array(corners)
        
        # Sort corners clockwise from top-left
        center = corners.mean(axis=0)
        angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        corners = corners[sorted_indices]
        
        # Start from top-left (min x+y)
        tl_idx = np.argmin(corners[:, 0] + corners[:, 1])
        corners = np.roll(corners, -tl_idx, axis=0)
        
        # 3. Extract sides using contours and corners
        sides = []
        side_colors = []
        
        for i in range(4):
            c1 = corners[i]
            c2 = corners[(i + 1) % 4]
            
            # Find contour points between corners
            dists1 = np.linalg.norm(contour - c1, axis=1)
            dists2 = np.linalg.norm(contour - c2, axis=1)
            idx1 = np.argmin(dists1)
            idx2 = np.argmin(dists2)
            
            if idx1 < idx2:
                side_points = contour[idx1:idx2+1]
            else:
                side_points = np.vstack([contour[idx1:], contour[:idx2+1]])
            
            sides.append(side_points)
            
            # Extract colors along the side
            colors = []
            for pt in side_points:
                x_coord, y_coord = int(pt[0]), int(pt[1])
                if 0 <= y_coord < img_rgb.shape[0] and 0 <= x_coord < img_rgb.shape[1]:
                    colors.append(img_rgb[y_coord, x_coord])
                else:
                    colors.append([0, 0, 0])
            side_colors.append(np.array(colors))
        
        # 4. Normalize sides
        normalized_sides = []
        normalized_colors = []
        num_points = 100
        
        for side, colors in zip(sides, side_colors):
            if len(side) < 2:
                normalized_sides.append(np.zeros((num_points, 2)))
                normalized_colors.append(np.zeros((num_points, 3)))
                continue
            
            # Translate to origin
            centroid = side.mean(axis=0)
            side_centered = side - centroid
            
            # Scale to unit length
            total_length = np.sum(np.linalg.norm(np.diff(side, axis=0), axis=1))
            if total_length > 0:
                side_scaled = side_centered / total_length
            else:
                side_scaled = side_centered
            
            # Rotate so first-last vector is horizontal
            first_last = side_scaled[-1] - side_scaled[0]
            angle = np.arctan2(first_last[1], first_last[0])
            cos_a, sin_a = np.cos(-angle), np.sin(-angle)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            side_rotated = side_scaled @ rotation_matrix.T
            
            # Interpolate geometry to fixed number of points
            t_original = np.linspace(0, 1, len(side_rotated))
            t_new = np.linspace(0, 1, num_points)
            
            interp_x = interp1d(t_original, side_rotated[:, 0], kind='linear')
            interp_y = interp1d(t_original, side_rotated[:, 1], kind='linear')
            
            normalized_side = np.column_stack([interp_x(t_new), interp_y(t_new)])
            normalized_sides.append(normalized_side)
            
            # Interpolate colors
            if len(colors) > 1:
                interp_r = interp1d(t_original, colors[:, 0], kind='linear')
                interp_g = interp1d(t_original, colors[:, 1], kind='linear')
                interp_b = interp1d(t_original, colors[:, 2], kind='linear')
                normalized_color = np.column_stack([interp_r(t_new), interp_g(t_new), interp_b(t_new)])
            else:
                normalized_color = np.zeros((num_points, 3))
            normalized_colors.append(normalized_color)
        
        # 5. Detect piece type and side types using boundary detection
        side_types = []
        
        # Check if piece is at image boundary (indicates flat edge)
        margin = 10  # pixels from edge to consider as boundary
        
        for i, (side, normalized_side) in enumerate(zip(sides, normalized_sides)):
            # Calculate straightness
            start = normalized_side[0]
            end = normalized_side[-1]
            line_points = np.linspace(start, end, len(normalized_side))
            deviations = np.linalg.norm(normalized_side - line_points, axis=1)
            max_deviation = np.max(deviations)
            mean_deviation = np.mean(deviations)
            std_deviation = np.std(deviations)
            
            # Check if side is at image boundary
            side_x_coords = side[:, 0]
            side_y_coords = side[:, 1]
            
            at_left = np.sum(side_x_coords < x + margin) > len(side) * 0.8
            at_right = np.sum(side_x_coords > x + w - margin) > len(side) * 0.8
            at_top = np.sum(side_y_coords < y + margin) > len(side) * 0.8
            at_bottom = np.sum(side_y_coords > y + h - margin) > len(side) * 0.8
            
            at_boundary = at_left or at_right or at_top or at_bottom
            
            # A side is flat if it's straight OR at the boundary
            is_straight = (max_deviation < 0.03 and std_deviation < 0.015)
            
            if at_boundary or is_straight:
                side_types.append('flat')
            else:
                # Determine if protruding or sunken based on signed area
                signed_area = 0
                for j in range(len(normalized_side) - 1):
                    x1, y1 = normalized_side[j]
                    x2, y2 = normalized_side[j + 1]
                    signed_area += (x1 * y2 - x2 * y1)
                
                # Use perpendicular distance as well
                mid_idx = len(normalized_side) // 2
                mid_point = normalized_side[mid_idx]
                line_mid = (start + end) / 2
                
                to_point = mid_point - line_mid
                along_line = end - start
                perp = np.array([-along_line[1], along_line[0]])
                side_sign = np.dot(to_point, perp)
                
                if side_sign > 0:
                    side_types.append('protruding')
                else:
                    side_types.append('sunken')
        
        # Determine piece type
        num_flat = side_types.count('flat')
        if num_flat == 0:
            piece_type = 'interior'
        elif num_flat == 1:
            piece_type = 'edge'
        elif num_flat == 2:
            flat_indices = [i for i, st in enumerate(side_types) if st == 'flat']
            diff = abs(flat_indices[0] - flat_indices[1])
            if diff == 1 or diff == 3:
                piece_type = 'corner'
            else:
                piece_type = 'edge'
        elif num_flat >= 3:
            piece_type = 'corner'
        else:
            piece_type = 'interior'
        
        # Flatten normalized sides and colors into feature vectors
        feature_vectors = []
        for side_geom, side_color in zip(normalized_sides, normalized_colors):
            side_color_norm = side_color / 255.0
            combined = np.column_stack([side_geom, side_color_norm]).flatten()
            feature_vectors.append(combined)
        
        features[piece_id] = {
            'contour': contour,
            'corners': corners,
            'sides': sides,
            'normalized_sides': normalized_sides,
            'normalized_colors': normalized_colors,
            'piece_type': piece_type,
            'side_types': side_types,
            'feature_vectors': feature_vectors,
            'bbox': (x, y, w, h)
        }
    
    # Print statistics
    type_counts = {}
    for f in features.values():
        ptype = f['piece_type']
        type_counts[ptype] = type_counts.get(ptype, 0) + 1
    
    print(f"Extracted features from {len(features)} pieces:")
    print(f"  Corner pieces: {type_counts.get('corner', 0)}")
    print(f"  Edge pieces: {type_counts.get('edge', 0)}")
    print(f"  Interior pieces: {type_counts.get('interior', 0)}")
    
    return features

def build_graph(features: Dict[str, dict], output_graph_dir: str):
    """
    Build adjacency graph by matching piece sides with constraints.
    Visualize using networkx.
    """
    from scipy.spatial import distance
    
    os.makedirs(output_graph_dir, exist_ok=True)
    
    print("Building adjacency graph...")
    
    G = nx.Graph()
    piece_ids = list(features.keys())
    
    # Add nodes
    for piece_id in piece_ids:
        G.add_node(piece_id, piece_type=features[piece_id]['piece_type'])
    
    # 6. Match pieces with simplified constraints
    matches = []
    
    print("Computing pairwise matches...")
    for i, piece_id1 in enumerate(piece_ids):
        if i % 100 == 0:
            print(f"  Processing piece {i}/{len(piece_ids)}...")
        
        f1 = features[piece_id1]
        
        for side_idx1 in range(4):
            side_type1 = f1['side_types'][side_idx1]
            
            # Skip flat sides
            if side_type1 == 'flat':
                continue
            
            for j, piece_id2 in enumerate(piece_ids):
                if i >= j:  # Avoid self-matching and duplicates
                    continue
                
                f2 = features[piece_id2]
                
                for side_idx2 in range(4):
                    side_type2 = f2['side_types'][side_idx2]
                    
                    # Skip flat sides
                    if side_type2 == 'flat':
                        continue
                    
                    # Sunken can only match protruding and vice versa
                    if not ((side_type1 == 'sunken' and side_type2 == 'protruding') or
                            (side_type1 == 'protruding' and side_type2 == 'sunken')):
                        continue
                    
                    # Relaxed edge constraints - only enforce for opposite sides of edges
                    flat_indices1 = [idx for idx, st in enumerate(f1['side_types']) if st == 'flat']
                    flat_indices2 = [idx for idx, st in enumerate(f2['side_types']) if st == 'flat']
                    
                    valid_match = True
                    
                    # Only strict constraint: opposite side of edge can only match interior
                    if len(flat_indices1) == 1 and f1['piece_type'] == 'edge':
                        flat_idx1 = flat_indices1[0]
                        opposite_idx = (flat_idx1 + 2) % 4
                        
                        if side_idx1 == opposite_idx:
                            if f2['piece_type'] != 'interior':
                                valid_match = False
                    
                    if len(flat_indices2) == 1 and f2['piece_type'] == 'edge':
                        flat_idx2 = flat_indices2[0]
                        opposite_idx = (flat_idx2 + 2) % 4
                        
                        if side_idx2 == opposite_idx:
                            if f1['piece_type'] != 'interior':
                                valid_match = False
                    
                    if not valid_match:
                        continue
                    
                    # Calculate similarity score using both geometry and color
                    fv1 = f1['feature_vectors'][side_idx1]
                    fv2 = f2['feature_vectors'][side_idx2]
                    
                    # Flip one vector for proper alignment (matching tab to slot)
                    fv2_reshaped = fv2.reshape(-1, 5)  # [x, y, r, g, b] per point
                    fv2_flipped = fv2_reshaped[::-1].flatten()
                    
                    # Split into geometry and color
                    geom1 = fv1.reshape(-1, 5)[:, :2].flatten()
                    geom2 = fv2_flipped.reshape(-1, 5)[:, :2].flatten()
                    color1 = fv1.reshape(-1, 5)[:, 2:].flatten()
                    color2 = fv2_flipped.reshape(-1, 5)[:, 2:].flatten()
                    
                    geom_dist = distance.euclidean(geom1, geom2)
                    color_dist = distance.euclidean(color1, color2)
                    
                    # Weighted combination - color is more important
                    combined_dist = 0.4 * geom_dist + 0.6 * color_dist
                    similarity = 1.0 / (1.0 + combined_dist)
                    
                    matches.append({
                        'piece1': piece_id1,
                        'side1': side_idx1,
                        'piece2': piece_id2,
                        'side2': side_idx2,
                        'similarity': similarity,
                        'distance': combined_dist,
                        'geom_dist': geom_dist,
                        'color_dist': color_dist
                    })
    
    print(f"Found {len(matches)} potential matches")
    
    # Sort matches by similarity
    matches.sort(key=lambda x: x['similarity'], reverse=True)
    
    # More aggressive edge addition - take top N matches per piece
    added_edges = 0
    matches_per_piece = 4  # Allow up to 4 connections per piece
    piece_match_counts = {pid: 0 for pid in piece_ids}
    
    for match in matches:
        p1, p2 = match['piece1'], match['piece2']
        
        # Check if either piece has reached its limit
        if piece_match_counts[p1] >= matches_per_piece or piece_match_counts[p2] >= matches_per_piece:
            continue
        
        # Add the edge
        G.add_edge(
            p1, p2,
            weight=match['similarity'],
            side1=match['side1'],
            side2=match['side2'],
            geom_dist=match['geom_dist'],
            color_dist=match['color_dist']
        )
        
        piece_match_counts[p1] += 1
        piece_match_counts[p2] += 1
        added_edges += 1
        
        # Stop when we have enough edges
        if added_edges >= len(piece_ids) * 2:
            break
    
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Print some matching statistics
    if len(matches) > 0:
        top_match = matches[0]
        print(f"Best match similarity: {top_match['similarity']:.4f}")
        print(f"  Geometry distance: {top_match['geom_dist']:.4f}")
        print(f"  Color distance: {top_match['color_dist']:.4f}")
        
        # Show distribution of connections
        connection_dist = {}
        for pid in piece_ids:
            conn_count = len([n for n in G.neighbors(pid)])
            connection_dist[conn_count] = connection_dist.get(conn_count, 0) + 1
        print("Connection distribution:")
        for count in sorted(connection_dist.keys()):
            print(f"  {count} connections: {connection_dist[count]} pieces")
    
    # 7. Plot with networkx
    plt.figure(figsize=(24, 20))
    
    # Use better layout algorithm
    if G.number_of_edges() > 0:
        pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42, weight='weight')
    else:
        pos = nx.circular_layout(G)
    
    # Color nodes by piece type
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        ptype = G.nodes[node]['piece_type']
        if ptype == 'corner':
            node_colors.append('#FF3B3B')  # Bright red
            node_sizes.append(1200)
        elif ptype == 'edge':
            node_colors.append('#FFA500')  # Orange
            node_sizes.append(1000)
        elif ptype == 'interior':
            node_colors.append('#4A90E2')  # Blue
            node_sizes.append(800)
        else:
            node_colors.append('#95A5A6')
            node_sizes.append(800)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                          alpha=0.9, edgecolors='black', linewidths=2)
    
    # Draw edges if any
    if G.number_of_edges() > 0:
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        
        # Color edges by weight
        edge_colors = plt.cm.YlGn([w for w in weights])
        
        nx.draw_networkx_edges(G, pos, width=[w * 5 for w in weights], 
                              alpha=0.6, edge_color=edge_colors)
    
    # Draw labels for sample nodes
    if len(G.nodes()) <= 50:
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', font_color='white')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF3B3B', edgecolor='black', label=f'Corner ({sum(1 for n in G.nodes() if G.nodes[n]["piece_type"] == "corner")})'),
        Patch(facecolor='#FFA500', edgecolor='black', label=f'Edge ({sum(1 for n in G.nodes() if G.nodes[n]["piece_type"] == "edge")})'),
        Patch(facecolor='#4A90E2', edgecolor='black', label=f'Interior ({sum(1 for n in G.nodes() if G.nodes[n]["piece_type"] == "interior")})')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=14)
    
    plt.title(f"Puzzle Piece Adjacency Graph\n{G.number_of_nodes()} pieces, {G.number_of_edges()} connections", 
             fontsize=18, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    # Save graph
    graph_path = Path(output_graph_dir) / "adjacency_graph.png"
    plt.savefig(graph_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"Saved graph to {graph_path}")
    
    # Save graph data as JSON
    graph_data = {
        'nodes': [{'id': n, 'piece_type': G.nodes[n]['piece_type']} for n in G.nodes()],
        'edges': [{'source': u, 'target': v, 'weight': float(G[u][v]['weight']), 
                   'side1': int(G[u][v]['side1']), 'side2': int(G[u][v]['side2']),
                   'geom_dist': float(G[u][v]['geom_dist']), 
                   'color_dist': float(G[u][v]['color_dist'])} 
                  for u, v in G.edges()]
    }
    
    json_path = Path(output_graph_dir) / "adjacency_graph.json"
    with open(json_path, 'w') as f:
        json.dump(graph_data, f, indent=2)
    print(f"Saved graph data to {json_path}")
    
    plt.close()

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

    # print("[STEP 2] Extracting features...")
    # features = extract_features(image_dir, provided_masks_dir, predicted_masks_dir)

    # print("[STEP 3] Building adjacency graph...")
    # build_graph(features, output_graph_dir)

    # print("[STEP 4] Assembling final puzzle...")
    # assemble_puzzle(graph_path, features, assembled_path)

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
