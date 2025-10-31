import numpy as np
import cv2
import scipy
from scipy.interpolate import interp1d
from scipy import ndimage
from pathlib import Path

def _load_piece(piece_id, img_dir="../data/images", provided_masks_dir="../data/masks", predicted_masks_dir="../output/predicted_masks", scale=0.25):
    img_path = Path(img_dir) / f"{piece_id}.jpg"
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    mask_path = Path(provided_masks_dir) / f"{piece_id}_mask.png"
    if not mask_path.exists():
        mask_path = Path(predicted_masks_dir) / f"{piece_id}_mask.png"
    
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # ---- Downscale both image and mask ----
    if scale != 1.0:
        new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
        img_rgb = cv2.resize(img_rgb, new_size, interpolation=cv2.INTER_AREA)
        binary = cv2.resize(binary, new_size, interpolation=cv2.INTER_NEAREST)
    
    return img_rgb, binary

def _harris_corners_from_mask(mask, block_size=20, ksize=7, k=0.08, neighborhood_size=70, score_threshold=0.3, minmax_percentile=40, min_corner_spacing=50):
    mask_float = np.float32(mask)
    harris = cv2.cornerHarris(mask_float, blockSize=block_size, ksize=ksize, k=k)
    harris = harris * (mask_float > 0) # weight by mask and zero any responses outside the mask

    # Normalize Harris response to 0..1
    data = harris.copy()
    data -= data.min()
    if data.max() > 0:
        data /= data.max()

    # Threshold by score
    data[data < score_threshold] = 0.

    # Local maxima
    data_max = ndimage.maximum_filter(data, neighborhood_size)
    maxima = (data == data_max)

    # Local min & percentile-based contrast threshold
    data_min = ndimage.minimum_filter(data, neighborhood_size)
    diff = data_max - data_min
    threshold_value = np.percentile(diff[diff > 0], minmax_percentile)
    maxima[diff < threshold_value] = 0

    # Connected components -> center of mass
    labeled, num_objects = ndimage.label(maxima)
    if num_objects == 0:
        return np.empty((0, 2), dtype=int)
    yx = np.array(ndimage.center_of_mass(data, labeled, range(1, num_objects + 1)))
    
    corners = yx[:, ::-1].astype(int)  # (x, y)

    # Assign Harris strength to each detected corner
    strengths = np.array([harris[y, x] for x, y in corners])
    order = np.argsort(-strengths)  # sort descending by strength
    corners = corners[order]
    strengths = strengths[order]

    # Non-max suppression by distance, keeping strongest
    keep = []
    for i, pt in enumerate(corners):
        if len(keep) == 0:
            keep.append(i)
            continue
        dists = np.linalg.norm(corners[keep] - pt, axis=1)
        if np.all(dists >= min_corner_spacing):
            keep.append(i)

    filtered_corners = corners[keep]

    return filtered_corners

def _extract_largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contours found")
    
    # Get largest contour
    contour = max(contours, key=cv2.contourArea).squeeze()
    
    oriented_area = cv2.contourArea(contour, oriented=True)
    if oriented_area < 0:
        return contour[::-1].copy()  # Reverse contour if counter-clockwise
    
    return contour.copy()

def _get_best_fitting_rect_coords(xy, d_threshold=30, perp_angle_thresh=5, min_width=150, min_height=150):
    N = len(xy)
    if N < 4:
        return None

    distances = scipy.spatial.distance.cdist(xy, xy)
    
    # Precompute angles between all points
    angles = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            dx = xy[j, 0] - xy[i, 0]
            dy = xy[j, 1] - xy[i, 1]
            angle = np.degrees(np.arctan2(dy, dx)) if dx != 0 else 90
            angles[i, j] = angles[j, i] = angle

    def angle_diff(a1, a2):
        # Compute smallest angle difference accounting for wraparound
        diff = abs(a1 - a2) % 180
        return min(diff, 180 - diff)
    
    def is_perpendicular(angle1, angle2, thresh=perp_angle_thresh):
        # Check if two angles are perpendicular
        return angle_diff(angle1, angle2 - 90) <= thresh

    possible_rectangles = []

    # Try all combinations of 4 points
    from itertools import combinations
    for combo in combinations(range(N), 4):
        pts = xy[list(combo)]
        
        # Check if distances are sufficient
        if any(distances[combo[i], combo[j]] < d_threshold for i in range(4) for j in range(i+1, 4)):
            continue
        
        # Try to order these 4 points into a valid rectangle
        # We need to find an ordering where consecutive edges are perpendicular
        from itertools import permutations
        for perm in permutations(combo):
            # Check if this ordering forms a rectangle
            edge_angles = [angles[perm[i], perm[(i+1)%4]] for i in range(4)]
            
            # Check consecutive edges are perpendicular
            valid = True
            for i in range(4):
                if not is_perpendicular(edge_angles[i], edge_angles[(i+1)%4]):
                    valid = False
                    break
            
            if valid:
                # Found a valid rectangle with this ordering
                if not any(set(perm) == set(r) for r in possible_rectangles):
                    possible_rectangles.append(list(perm))
                    break  # Found valid ordering for this combo, try next combo

    if not possible_rectangles:
        return None

    # Polygon area
    def poly_area(pts):
        x, y = pts[:, 0], pts[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    # Filter rectangles by minimum width/height using edges
    filtered_rects = []
    edge_info = []
    for rect in possible_rectangles:
        pts = xy[rect]
        # compute consecutive edge lengths
        edges = [np.linalg.norm(pts[i] - pts[(i+1)%4]) for i in range(4)]
        width = max(edges)
        height = min(edges)
        if width >= min_width and height >= min_height:
            filtered_rects.append(rect)
            edge_info.append((width, height, edges))

    if not filtered_rects:
        return None

    # Score rectangles
    scores = []
    for idx, rect in enumerate(filtered_rects):
        pts = xy[rect]
        area = poly_area(pts)
        mse = 0
        for i in range(4):
            # Get the two edge vectors
            v1 = pts[(i+1)%4] - pts[i]
            v2 = pts[(i+2)%4] - pts[(i+1)%4]
            
            # Compute angle between vectors using dot product
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1, 1)  # Numerical stability
            angle_deg = np.degrees(np.arccos(cos_angle))
            
            # We want 90°, so measure deviation
            deviation = abs(angle_deg - 90)
            mse += deviation ** 2

        mse = mse / 4  # Average squared error

        width, height, edges = edge_info[idx]
        squareness = height / width
        score = area * squareness * scipy.stats.norm(0, 150).pdf(mse)
        scores.append(score)

    best_rect_idx = filtered_rects[np.argmax(scores)]

    return xy[best_rect_idx]

def _extract_corners(mask, contour, epsilon_factor=0.02):
    corners = _harris_corners_from_mask(mask)

    # Try to find rectangle from corners
    if corners.shape[0] >= 4:
        try:
            best_rect_corners = _get_best_fitting_rect_coords(corners)
        except Exception:
            best_rect_corners = None
    else:
        best_rect_corners = None

    # Fallback to contour polygon approximation
    if best_rect_corners is None or len(best_rect_corners) != 4:
        peri = cv2.arcLength(contour.astype(np.float32), True)
        approx = cv2.approxPolyDP(contour.astype(np.float32), epsilon_factor * peri, True)
        approx = approx.reshape(-1, 2)

        # If not exactly 4 points, choose 4 most distant points
        if len(approx) != 4:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(approx)
            pts = approx[hull.vertices]
            if len(pts) > 4:
                # Pick 4 corners spread across the hull
                idxs = np.linspace(0, len(pts) - 1, 4, dtype=int)
                best_rect_corners = pts[idxs]
            else:
                best_rect_corners = pts
        else:
            best_rect_corners = approx

    return best_rect_corners

def _extract_sides(contour, corners):
    # Find nearest contour points to corners
    corner_indices = []
    for corner in corners:
        dists = np.linalg.norm(contour - corner, axis=-1)
        corner_indices.append(np.argmin(dists))
    
    corner_indices = sorted(corner_indices)
    
    # Extract sides between consecutive corners
    sides = []
    for a in range(4):
        i, j = corner_indices[a], corner_indices[(a+1) % 4]
        rolled_contour = np.roll(contour, -i, axis=0)
        segment_length = (j - i + 1) % len(contour)
        if segment_length == 0:
            segment_length = len(contour)
        side_contour = rolled_contour[:segment_length]
        sides.append(side_contour)
    
    return sides

def _extract_side_colors(side, img_rgb, mask, offset_inward=15):
    side = side.copy().astype(np.float64)
    
    # Compute inward normals
    tangents = np.gradient(side, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-8
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])

    # Ensure normals point inward using mask 
    if mask is not None:
        mid_idx = len(side) // 2
        test_pt = side[mid_idx] + normals[mid_idx] * offset_inward
        x, y = int(test_pt[0]), int(test_pt[1])
        if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x] > 0):
            normals = -normals

    inner_side = side + offset_inward * normals

    # Sample colors 
    colors = []
    for pt in inner_side:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= y < img_rgb.shape[0] and 0 <= x < img_rgb.shape[1]:
            colors.append(img_rgb[y, x])
        else:
            colors.append([0, 0, 0])
    colors = np.array(colors, dtype=np.float32)
    
    return colors

def _extract_geom_features(side, num_points=100):
    side = side.copy().astype(np.float64)

    # Center and scale
    first_point, last_point = side[0], side[-1]
    baseline_center = (first_point + last_point) / 2
    side_translated = side - baseline_center

    baseline_vector = last_point - first_point
    baseline_length = np.linalg.norm(baseline_vector)
    side_scaled = side_translated * (2.0 / baseline_length)

    # Rotate so baseline aligns with x-axis
    theta = -np.arctan2(baseline_vector[1], baseline_vector[0])
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    side_rotated = side_scaled @ rotation_matrix.T

    # Ensure consistent direction
    should_reverse = side_rotated[0, 0] > side_rotated[-1, 0]
    if should_reverse:
        side_rotated = side_rotated[::-1]

    # Interpolate to uniform number of points
    t_original = np.linspace(0, 1, len(side_rotated))
    t_new = np.linspace(0, 1, num_points)
    interp_x = interp1d(t_original, side_rotated[:, 0], kind='linear')
    interp_y = interp1d(t_original, side_rotated[:, 1], kind='linear')
    normalized_side = np.column_stack([interp_x(t_new), interp_y(t_new)])

    return normalized_side, should_reverse

def _extract_color_features(colors, num_points=100):
    colors = colors.copy().astype(np.float32)

    # Z-score normalized RGB
    mean = np.mean(colors, axis=0, keepdims=True)
    std = np.std(colors, axis=0, keepdims=True) + 1e-6
    colors_zscore = (colors - mean) / std

    # Normalized RGB [0, 1]
    colors_rgb_norm = colors / 255.0

    # Convert to HSV
    colors_hsv = cv2.cvtColor(colors.astype(np.uint8).reshape(1, -1, 3), cv2.COLOR_RGB2HSV)
    colors_hsv = colors_hsv.reshape(-1, 3).astype(np.float32)
    colors_hsv[:, 0] /= 179.0
    colors_hsv[:, 1:] /= 255.0

    # Interpolate
    t_original = np.linspace(0, 1, len(colors))
    t_new = np.linspace(0, 1, num_points)

    def interp_channel(data):
        return np.column_stack([interp1d(t_original, data[:, i], kind='linear')(t_new)
                                for i in range(data.shape[1])])

    colors_zscore = interp_channel(colors_zscore)
    colors_rgb_norm = interp_channel(colors_rgb_norm)
    colors_hsv = interp_channel(colors_hsv)

    # Combine
    color_features = np.hstack([colors_zscore, colors_rgb_norm, colors_hsv])
    return color_features

def _classify_side(normalized_side):    
    # Straightness in normalized coordinates
    start = normalized_side[0]
    end = normalized_side[-1]
    line_points = np.linspace(start, end, len(normalized_side))
    deviations = normalized_side - line_points
    norm_devs = np.linalg.norm(deviations, axis=1)
    max_dev = np.max(norm_devs)
    is_straight = max_dev < 0.1
        
    if is_straight:
        return 'flat'
    
    # Use y-coordinates of deviations (since normalized always has sunken up)
    signed_devs = normalized_side[:, 1] - line_points[:, 1]
    avg_dev = np.mean(signed_devs)
        
    if avg_dev > 0:
        return 'sunken'  # deviation is upwards -> sunken
    else:
        return 'protruding'  # deviation is downwards -> protruding

def extract_features(all_images):
    geom_features = []
    color_features = []
    side_indices = []  # (piece_idx, side_idx) tuples
    
    num_points = 100  # Number of points for normalized sides
    
    print(f"Extracting features from {len(all_images)} puzzle pieces...")
    
    for piece_idx, img_path in enumerate(all_images):
        # Load image and mask
        img_rgb, mask = _load_piece(img_path.stem)
        
        # Extract contour, corners, and sides
        contour = _extract_largest_contour(mask)
        corners = _extract_corners(mask, contour)
        sides = _extract_sides(contour, corners)
        
        # Process each side
        for side_idx, side in enumerate(sides):
            # --- 1. Extract colors first ---
            colors = _extract_side_colors(side, img_rgb, mask, offset_inward=15)

            # --- 2. Extract geometric features ---
            geom_feat, should_reverse = _extract_geom_features(side, num_points=num_points)

            # --- 3. Reverse colors if geometry was reversed ---
            if should_reverse:
                colors = colors[::-1]

            # --- 4. Extract color features ---
            color_feat = _extract_color_features(colors, num_points)

            # --- 5. Classify and store ---
            side_type = _classify_side(geom_feat)
            if side_type == 'flat':
                continue
            geom_features.append(geom_feat)
            color_features.append(color_feat)
            side_indices.append((piece_idx, side_idx))
        
        if (piece_idx + 1) % 100 == 0:
            print(f"Processed {piece_idx + 1}/{len(all_images)} pieces...")
    
    # Convert to numpy arrays
    geom_features = np.array(geom_features)
    color_features = np.array(color_features)
    side_indices = np.array(side_indices, dtype=int)
    
    print(f"\nExtracted {len(geom_features)} non-flat sides from {len(all_images)} puzzle pieces")
    
    return geom_features, color_features, side_indices