import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def match_sides_with_knn(geom_features, color_features, n_neighbors=1, scale=True):
    # --- Step 1. Create rotated/reversed sides ---
    rotated_sides = (geom_features * (-1, -1))[:, ::-1, :]

    # --- Step 2. Flatten geometry and color features ---
    geom_features = geom_features.reshape(geom_features.shape[0], -1)
    rotated_geom = rotated_sides.reshape(rotated_sides.shape[0], -1)
    color_flat = color_features.reshape(color_features.shape[0], -1)

    # Combine geometry + color features
    combined_features = np.concatenate([geom_features, color_flat], axis=1)
    rotated_combined = np.concatenate([rotated_geom, color_flat], axis=1)

    # --- Step 3. Feature scaling ---
    if scale:
        scaler = StandardScaler()
        combined_features = scaler.fit_transform(combined_features)
        rotated_combined = scaler.transform(rotated_combined)

    # --- Step 4. Fit KNN model ---
    knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="brute")
    knn.fit(rotated_combined)

    # --- Step 5. Compute matches ---
    distances, indices = knn.kneighbors(combined_features)

    return distances, indices

def get_valid_matches(distances, indices, side_indices, max_distance=0.5):
    valid_matches = []
    
    for query_idx in range(len(distances)):
        query_piece, query_side = side_indices[query_idx]
        match_idx = indices[query_idx][0]
        match_distance = distances[query_idx][0]
        match_piece, match_side = side_indices[match_idx]
        
        # No self-matching and distance filter
        if query_piece != match_piece and match_distance <= max_distance:
            valid_matches.append({
                'query_piece': query_piece, 'query_side': query_side,
                'match_piece': match_piece, 'match_side': match_side,
                'distance': match_distance
            })
    
    return valid_matches

def run_side_matching(normalized_sides, color_features, side_indices, n_neighbors=1, max_distance=0.5, scale=True):
    print(f"Matching {len(normalized_sides)} sides using KNN...")

    # Step 1. Perform matching
    distances, indices = match_sides_with_knn(
        normalized_sides, color_features, n_neighbors=n_neighbors, scale=scale
    )

    # Step 2. Filter valid matches
    valid_matches = get_valid_matches(distances, indices, side_indices, max_distance=max_distance)

    print(f"Found {len(valid_matches)} valid matches.")

    return valid_matches