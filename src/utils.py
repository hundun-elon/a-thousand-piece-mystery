import cv2
from pathlib import Path

def load_piece(piece_id, scale=0.25):
    """Load image and mask for a single piece"""
    img_path = Path("../data/images") / f"{piece_id}.jpg"
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    mask_path = Path(PROVIDED_MASKS_DIR) / f"{piece_id}_mask.png"
    if not mask_path.exists():
        mask_path = Path(PREDICTED_MASKS_DIR) / f"{piece_id}_mask.png"
    
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # ---- Downscale both image and mask ----
    if scale != 1.0:
        new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
        img_rgb = cv2.resize(img_rgb, new_size, interpolation=cv2.INTER_AREA)
        binary = cv2.resize(binary, new_size, interpolation=cv2.INTER_NEAREST)
    
    return img_rgb, binary
