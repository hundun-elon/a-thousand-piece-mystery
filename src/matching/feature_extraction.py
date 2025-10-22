import cv2
import numpy as np

def extract_edge_features(piece_image, mask, num_interp_points=64):
    """
    Extract geometric features from puzzle piece edges and classify flat sides.
    
    Parameters
    ----------
    piece_image : np.ndarray
        RGB image of the puzzle piece.
    mask : np.ndarray
        Binary mask (same size as piece_image).
    num_interp_points : int
        Number of interpolated points per edge for geometric features.
    
    Returns
    -------
    piece_data : dict
        Dictionary mapping 'edges' and 'is_flat' for this piece:
        {
            'edges': [np.ndarray of flattened geometric features per side],
            'is_flat': [bool per side]
        }
    """

    piece_data = {'edges': [], 'is_flat': []}

    # --- 1. Get main contour (largest area)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)

    # --- 2. Ensure clockwise orientation
    if cv2.contourArea(contour, oriented=True) < 0:
        contour = contour[::-1]

    # --- 3. Detect corners automatically using approxPolyDP
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    corners = approx.reshape(-1, 2)

    if len(corners) != 4:
        # fallback: choose four points farthest from the center
        center = np.mean(contour, axis=0)
        dists = np.linalg.norm(contour - center, axis=1)
        corners = contour[np.argsort(dists)[-4:]]
    # sort corners clockwise
    center = np.mean(corners, axis=0)
    angles = np.arctan2(corners[:,1]-center[1], corners[:,0]-center[0])
    corners = corners[np.argsort(angles)]

    # --- 4. Extract sides from corners
    corner_indices = []
    for corner in corners:
        dists = np.linalg.norm(contour - corner, axis=1)
        corner_indices.append(np.argmin(dists))
    corner_indices = np.sort(corner_indices)

    sides = []
    for i in range(4):
        i1, i2 = corner_indices[i], corner_indices[(i+1)%4]
        if i1 < i2:
            side = contour[i1:i2+1]
        else:
            side = np.concatenate([contour[i1:], contour[:i2+1]], axis=0)
        sides.append(side)

    # --- 5. Compute flattened geometric features and classify flat sides
    for side in sides:
        # Translate so first point is origin
        side_norm = side - side[0]
        # Normalize length
        length = np.linalg.norm(side_norm[-1])
        if length > 0:
            side_norm = side_norm / length

        # Interpolate to fixed number of points
        seg_lengths = np.linalg.norm(np.diff(side_norm, axis=0), axis=1)
        cumlen = np.insert(np.cumsum(seg_lengths), 0, 0)
        if cumlen[-1] == 0:
            interp_x = np.zeros(num_interp_points)
            interp_y = np.zeros(num_interp_points)
        else:
            Anew = np.linspace(0, cumlen[-1], num_interp_points)
            interp_x = np.interp(Anew, cumlen, side_norm[:,0])
            interp_y = np.interp(Anew, cumlen, side_norm[:,1])

        # Flattened geometric feature
        geom_feat = np.concatenate([interp_x, interp_y])
        piece_data['edges'].append(geom_feat)

        # --- 6. Classify flat side
        dist = np.linalg.norm(side[0] - side[-1])
        cont_len = np.sum(seg_lengths)
        is_flat = (cont_len > 0) and (dist / cont_len >= 0.9)
        piece_data['is_flat'].append(is_flat)

    return piece_data
