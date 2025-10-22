import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def build_adjacency_graph(piece_dict, threshold=0.8, max_neighbours=4):
    """
    Build adjacency graph of puzzle pieces based on edge features,
    ignoring flat sides (Rule a).
    
    Parameters
    ----------
    piece_dict : dict
        Dictionary mapping piece names to data from extract_edge_features():
        {
            'edges': list of flattened geometric features per side,
            'is_flat': list of bool per side
        }
    threshold : float
        Cosine similarity threshold for matching.
    max_neighbours : int
        Maximum neighbours per piece.

    Returns
    -------
    adjacency : dict
        Mapping {piece_name: [neighbour_piece_names, ...]}.
    """
    pieces = list(piece_dict.keys())
    adjacency = {name: [] for name in pieces}

    # --- 1. Flatten all non-flat edge features
    edge_index = []  # (piece_name, side_idx)
    all_features = []

    for name in pieces:
        edges = piece_dict[name]['edges']
        is_flat = piece_dict[name]['is_flat']
        for i, (feat, flat) in enumerate(zip(edges, is_flat)):
            if flat:
                continue  # skip flat sides
            edge_index.append((name, i))
            all_features.append(feat)

    if len(all_features) == 0:
        return adjacency

    all_features = np.stack(all_features)
    all_features = all_features / np.linalg.norm(all_features, axis=1, keepdims=True)

    # --- 2. Compute pairwise cosine similarity
    sim = cosine_similarity(all_features)

    # --- 3. Collect candidate matches for each piece
    piece_matches = {name: [] for name in pieces}

    for idx1 in range(len(edge_index)):
        name1, side1 = edge_index[idx1]
        for idx2 in range(idx1 + 1, len(edge_index)):
            name2, side2 = edge_index[idx2]
            if name1 == name2:
                continue  # skip same piece
            score = sim[idx1, idx2]
            if score >= threshold:
                piece_matches[name1].append((name2, score))
                piece_matches[name2].append((name1, score))

    # --- 4. Keep top-k neighbours per piece
    for name in pieces:
        top_k = sorted(piece_matches[name], key=lambda x: x[1], reverse=True)[:max_neighbours]
        adjacency[name] = [n for n, _ in top_k]

    return adjacency
