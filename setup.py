#!/usr/bin/env python3
"""
COMS4036A/COMS7050A: Computer Vision - Puzzle Project Setup
Creates the complete project directory structure with placeholder files.
"""

import os
from pathlib import Path


def create_directory_structure():
    """Create all necessary directories for the project."""
    
    directories = [
        # Data directories
        "data/images",
        "data/masks",
        "data/train/images",
        "data/train/masks",
        "data/test/images",
        
        # Model directories
        "models/segmentation",
        "models/matching",
        "models/weights",
        
        # Source code directories
        "src/segmentation",
        "src/matching",
        "src/assembly",
        "src/utils",
        
        # Notebooks directory
        "notebooks",
        
        # Output directories
        "output/masks",
        "output/visualizations",
        "output/experiments",
        
        # Report directory
        "report/figures",
        "report/latex",
        
        # Tests directory
        "tests",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {directory}/")
    
    return directories


def create_init_files():
    """Create __init__.py files to make directories Python packages."""
    
    init_dirs = [
        "src",
        "src/segmentation",
        "src/matching",
        "src/assembly",
        "src/utils",
        "models",
        "models/segmentation",
        "models/matching",
        "tests",
    ]
    
    for directory in init_dirs:
        init_file = Path(directory) / "__init__.py"
        init_file.touch()
        print(f"✓ Created: {init_file}")


def create_source_files():
    """Create placeholder Python source files."""
    
    files = {
        # Segmentation module
        "src/segmentation/train.py": '''"""Train segmentation model."""

def train_segmentation_model(train_images, train_masks, config):
    """Train the segmentation model.
    
    Args:
        train_images: Path to training images
        train_masks: Path to training masks
        config: Training configuration dictionary
    """
    pass


if __name__ == "__main__":
    # Training script entry point
    pass
''',
        
        "src/segmentation/predict.py": '''"""Predict masks for unlabelled images."""

def predict_masks(model, image_dir, output_dir):
    """Generate masks for test images.
    
    Args:
        model: Trained segmentation model
        image_dir: Directory containing images
        output_dir: Directory to save predicted masks
    """
    pass


if __name__ == "__main__":
    # Prediction script entry point
    pass
''',
        
        "src/segmentation/model.py": '''"""Segmentation model architecture."""

class SegmentationModel:
    """Segmentation model for puzzle piece extraction."""
    
    def __init__(self, config):
        """Initialize the model."""
        pass
    
    def train(self, train_data):
        """Train the model."""
        pass
    
    def predict(self, image):
        """Predict mask for a single image."""
        pass
''',
        
        # Matching module
        "src/matching/feature_extraction.py": '''"""Extract features from puzzle piece edges."""

def extract_edge_features(piece_image, mask):
    """Extract geometric and texture features from piece edges.
    
    Args:
        piece_image: RGB image of puzzle piece
        mask: Binary mask of the piece
        
    Returns:
        Dictionary containing edge features
    """
    pass


def extract_shape_descriptor(contour):
    """Extract shape descriptor from piece contour."""
    pass


def extract_texture_descriptor(image, edge_points):
    """Extract texture features along edge points."""
    pass
''',
        
        "src/matching/graph_builder.py": '''"""Build adjacency graph from piece features."""

def build_adjacency_graph(pieces, features, threshold=0.8):
    """Build graph of piece adjacencies.
    
    Args:
        pieces: List of piece identifiers
        features: Dictionary of extracted features per piece
        threshold: Similarity threshold for matching
        
    Returns:
        Adjacency dictionary
    """
    pass


def match_pieces(piece1_features, piece2_features):
    """Compute similarity score between two pieces."""
    pass
''',
        
        # Assembly module
        "src/assembly/assembler.py": '''"""Assemble puzzle pieces into final image."""

def assemble_puzzle(pieces, adjacency_graph):
    """Assemble puzzle using adjacency information.
    
    Args:
        pieces: Dictionary of piece images and metadata
        adjacency_graph: Graph of piece connections
        
    Returns:
        Assembled puzzle image
    """
    pass


def apply_transformations(piece, transform_matrix):
    """Apply affine/projective transformation to piece."""
    pass


def place_piece_on_canvas(canvas, piece, position):
    """Place a piece on the final canvas."""
    pass
''',
        
        # Utilities
        "src/utils/metrics.py": '''"""Evaluation metrics for segmentation and matching."""

def compute_iou(pred_mask, true_mask):
    """Compute Intersection over Union.
    
    Args:
        pred_mask: Predicted binary mask
        true_mask: Ground truth binary mask
        
    Returns:
        IoU score (float)
    """
    pass


def compute_graph_metrics(pred_graph, true_graph):
    """Compute precision and recall for adjacency graph.
    
    Returns:
        Dictionary with precision, recall, and F1 score
    """
    pass
''',
        
        "src/utils/visualization.py": '''"""Visualization utilities."""

def visualize_segmentation(image, mask, output_path=None):
    """Visualize image with predicted mask overlay."""
    pass


def visualize_graph(pieces, adjacency_graph, output_path=None):
    """Visualize puzzle piece adjacency graph."""
    pass


def visualize_assembly(assembled_image, output_path=None):
    """Save assembled puzzle visualization."""
    pass
''',
        
        "src/utils/data_loader.py": '''"""Data loading and preprocessing utilities."""

def load_images_and_masks(image_dir, mask_dir=None):
    """Load images and optional masks.
    
    Args:
        image_dir: Directory containing images
        mask_dir: Directory containing masks (optional)
        
    Returns:
        Lists of images and masks
    """
    pass


def preprocess_image(image):
    """Preprocess image for model input."""
    pass
''',
        
        # Configuration
        "src/config.py": '''"""Configuration settings for the project."""

# Segmentation configuration
SEGMENTATION_CONFIG = {
    "model_type": "unet",  # or "maskrcnn", "deeplabv3", etc.
    "input_size": (512, 512),
    "batch_size": 8,
    "epochs": 50,
    "learning_rate": 0.001,
}

# Matching configuration
MATCHING_CONFIG = {
    "feature_type": "combined",  # shape, texture, or combined
    "similarity_threshold": 0.7,
    "use_pca": True,
    "pca_components": 64,
}

# Assembly configuration
ASSEMBLY_CONFIG = {
    "canvas_size": (4000, 3000),  # Adjust based on puzzle size
    "transformation_type": "projective",  # affine or projective
}

# Paths
PATHS = {
    "train_images": "./data/train/images",
    "train_masks": "./data/train/masks",
    "test_images": "./data/test/images",
    "output_masks": "./data/masks",
    "model_weights": "./models/weights",
}
''',
    }
    
    for filepath, content in files.items():
        file_path = Path(filepath)
        file_path.write_text(content)
        print(f"✓ Created: {filepath}")


def create_notebooks():
    """Create Jupyter notebook templates."""
    
    notebooks = {
        "notebooks/01_data_exploration.ipynb": "Data Exploration and Analysis",
        "notebooks/02_segmentation_training.ipynb": "Segmentation Model Training",
        "notebooks/03_feature_extraction.ipynb": "Feature Extraction and Analysis",
        "notebooks/04_graph_construction.ipynb": "Adjacency Graph Construction",
        "notebooks/05_puzzle_assembly.ipynb": "Puzzle Assembly and Visualization",
        "notebooks/06_evaluation.ipynb": "Performance Evaluation",
    }
    
    for notebook_path, title in notebooks.items():
        # Create minimal notebook structure
        notebook_content = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [f"# {title}\n\n"]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": ["# Imports\nimport numpy as np\nimport matplotlib.pyplot as plt\n"]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        import json
        Path(notebook_path).write_text(json.dumps(notebook_content, indent=2))
        print(f"✓ Created: {notebook_path}")


def create_requirements_file():
    """Create requirements.txt with common dependencies."""
    
    requirements = """# Core dependencies
numpy>=1.21.0
opencv-python>=4.5.0
pillow>=9.0.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
scikit-image>=0.19.0
scipy>=1.7.0

# Deep learning (choose based on your approach)
torch>=2.0.0
torchvision>=0.15.0
# tensorflow>=2.12.0

# Segmentation models
segmentation-models-pytorch>=0.3.0
# OR
# keras-applications>=1.0.8

# Graph and network analysis
networkx>=2.8.0

# Image processing
albumentations>=1.3.0

# Data handling
pandas>=1.3.0
tqdm>=4.62.0

# Visualization
seaborn>=0.11.0

# Utilities
pyyaml>=6.0
"""
    
    Path("requirements.txt").write_text(requirements)
    print("✓ Created: requirements.txt")


def create_run_script():
    """Create the main run.sh script."""
    
    script = """#!/bin/bash
# Main execution script for puzzle project
# This script runs inference only (training must be done beforehand)

set -e  # Exit on error

echo "========================================="
echo "COMS4036A Puzzle Project - Inference"
echo "========================================="

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Step 1: Generate segmentation masks
echo ""
echo "Step 1: Generating segmentation masks..."
python -m src.segmentation.predict

# Step 2: Build adjacency graph
echo ""
echo "Step 2: Building adjacency graph..."
python -m src.matching.graph_builder

# Step 3: Assemble puzzle
echo ""
echo "Step 3: Assembling puzzle..."
python -m src.assembly.assembler

echo ""
echo "========================================="
echo "Pipeline complete!"
echo "- Masks saved to: ./data/masks/"
echo "- Graph saved to: ./graph.json"
echo "- Final image saved to: ./final.png"
echo "========================================="
"""
    
    run_script_path = Path("run.sh")
    run_script_path.write_text(script)
    run_script_path.chmod(0o755)  # Make executable
    print("✓ Created: run.sh (executable)")


def create_readme():
    """Create comprehensive README.md."""
    
    readme = """# COMS4036A Computer Vision - 1000-Piece Puzzle Project

## Project Overview
This project implements an end-to-end computer vision pipeline to solve a 1000-piece puzzle using segmentation, feature matching, and geometric assembly techniques.

## Team Members
- [Member 1 Name] - [Student Number]
- [Member 2 Name] - [Student Number]
- [Member 3 Name] - [Student Number]

## Project Structure

```
.
├── data/                      # Data directory (not in git)
│   ├── images/               # All 1000 puzzle piece images
│   ├── masks/                # 500 training masks + 500 predicted
│   ├── train/                # Training data subset
│   │   ├── images/
│   │   └── masks/
│   └── test/                 # Test data subset
│       └── images/
│
├── src/                       # Source code
│   ├── segmentation/         # Segmentation module
│   │   ├── model.py         # Model architecture
│   │   ├── train.py         # Training script
│   │   └── predict.py       # Inference script
│   ├── matching/             # Feature matching module
│   │   ├── feature_extraction.py
│   │   └── graph_builder.py
│   ├── assembly/             # Puzzle assembly module
│   │   └── assembler.py
│   ├── utils/                # Utility functions
│   │   ├── data_loader.py
│   │   ├── metrics.py
│   │   └── visualization.py
│   └── config.py             # Configuration settings
│
├── models/                    # Model definitions and weights
│   ├── segmentation/
│   ├── matching/
│   └── weights/              # Trained model weights
│
├── notebooks/                 # Jupyter notebooks for experiments
│   ├── 01_data_exploration.ipynb
│   ├── 02_segmentation_training.ipynb
│   ├── 03_feature_extraction.ipynb
│   ├── 04_graph_construction.ipynb
│   ├── 05_puzzle_assembly.ipynb
│   └── 06_evaluation.ipynb
│
├── output/                    # Output directory
│   ├── masks/                # Predicted masks
│   ├── visualizations/       # Plots and figures
│   └── experiments/          # Experiment results
│
├── report/                    # Report files
│   ├── figures/              # Report figures
│   └── latex/                # LaTeX source files
│
├── tests/                     # Unit tests
│
├── requirements.txt           # Python dependencies
├── run.sh                     # Main execution script
├── setup.py                   # Project setup script
├── graph.json                 # Output: adjacency graph
├── final.png                  # Output: assembled puzzle
└── README.md                  # This file
```

## Setup Instructions

### 1. Clone and Setup
```bash
# Run the setup script to create project structure
python setup.py

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Organization
Place the provided data in the following structure:
```
data/
  images/          # All 1000 images
  masks/           # 500 labeled masks
```

The training scripts will automatically split data into train/test sets.

## Pipeline Components

### 1. Segmentation (25%)
**Goal:** Extract puzzle pieces from images

**Approach:**
- Model: [U-Net / Mask R-CNN / DeepLabV3 / Custom]
- Input: RGB images of puzzle pieces
- Output: Binary masks (0=background, 1=piece)
- Metric: Mean IoU

**Files:**
- `src/segmentation/model.py`: Model architecture
- `src/segmentation/train.py`: Training pipeline
- `src/segmentation/predict.py`: Inference on test set

### 2. Graph Construction (25%)
**Goal:** Determine which pieces are adjacent

**Approach:**
- Extract edge features (shape + texture)
- Compute similarity between piece edges
- Build adjacency graph with neighbors
- Metric: Precision & Recall of edges

**Files:**
- `src/matching/feature_extraction.py`: Feature extraction
- `src/matching/graph_builder.py`: Graph construction

### 3. Puzzle Assembly (15%)
**Goal:** Reconstruct the complete puzzle

**Approach:**
- Use adjacency graph to position pieces
- Apply affine/projective transformations
- Assemble pieces on canvas
- Metric: Visual quality and connected regions

**Files:**
- `src/assembly/assembler.py`: Assembly algorithm

## Running the Project

### Training Phase
```bash
# Train segmentation model
python -m src.segmentation.train

# Extract features and build initial graph
python -m src.matching.feature_extraction
```

### Inference Phase (Submission)
```bash
# Run complete pipeline
./run.sh

# This generates:
# - ./data/masks/*.png (500 predicted masks)
# - ./graph.json (adjacency graph)
# - ./final.png (assembled puzzle)
```

## Deliverables

### 1. Segmentation Masks (Moodle)
- ZIP file containing 500 predicted masks
- Format: `IMG_XXXXXXX_mask.png` (binary PNG, values 0 or 1)

### 2. Adjacency Graph (Moodle)
- JSON file: `graph.json`
- Format: `{"piece_id": ["neighbor1", "neighbor2", ...]}`

### 3. Final Submission Package (Moodle)
```
submission.zip
  ├── src/                    # All source code
  ├── models/weights/         # Trained model weights
  ├── notebooks/              # Jupyter notebooks
  ├── requirements.txt        # Dependencies
  ├── run.sh                  # Inference script
  ├── report.pdf              # IEEE format report (4-6 pages)
  └── README.md               # This file
```

**Note:** Do NOT include data/images or data/masks in submission.

## Report Structure (35%)

IEEE two-column format, 4-6 pages:

1. **Pipeline Overview**
   - High-level architecture
   - Design rationale

2. **Methods**
   - Segmentation approach
   - Feature extraction and matching
   - Assembly algorithm

3. **Metrics & Results**
   - Segmentation: IoU scores
   - Graph: Precision, Recall, F1
   - Assembly: Visual quality metrics

4. **Reflection**
   - What worked and why
   - What failed and why
   - Limitations and future work

5. **Final Reconstruction**
   - Full-page assembled puzzle (last page, landscape)

## Mark Allocation

| Component                | Weight |
|--------------------------|--------|
| Segmentation Performance | 25%    |
| Graph Connectivity       | 25%    |
| Final Image Quality      | 15%    |
| Report                   | 35%    |
| **Total**                | **100%** |

## Development Workflow

### Branch Strategy (Suggested)
- `main`: Stable code only
- `dev`: Integration branch
- `feature/segmentation`: Segmentation work
- `feature/matching`: Matching work
- `feature/assembly`: Assembly work

### Commit Guidelines
- Use clear, descriptive commit messages
- Commit working code frequently
- Don't commit large data files

## Useful Resources

- [OpenCV Documentation](https://docs.opencv.org/)
- [PyTorch Segmentation Models](https://github.com/qubvel/segmentation_models.pytorch)
- [NetworkX Documentation](https://networkx.org/documentation/stable/)
- IEEE Paper Template: [Overleaf](https://www.overleaf.com/latex/templates/ieee-conference-template/grfzhhncsfqn)

## Tips and Best Practices

1. **Start Simple**: Get a basic pipeline working end-to-end first
2. **Iterate**: Improve each component incrementally
3. **Visualize**: Create visualizations at each stage
4. **Metrics**: Track metrics throughout development
5. **Document**: Keep notes on what you try and results
6. **Version Control**: Commit frequently with meaningful messages

## Troubleshooting

### Common Issues
- **Out of memory**: Reduce batch size or image resolution
- **Poor segmentation**: Try data augmentation or different model
- **Slow training**: Use GPU if available, reduce model size
- **Poor matching**: Augment features with color/texture information

## Due Date
**14:00 on 31 October 2025**

## Contact
For questions, contact Prof. Richard Klein or use the course forum.

---
*Good luck with the puzzle!* 🧩
"""
    
    Path("README.md").write_text(readme)
    print("✓ Created: README.md")


def create_gitignore():
    """Create .gitignore file."""
    
    gitignore = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.venv/
venv/
ENV/
env/

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb_checkpoints/

# Data (large files)
data/images/
data/masks/
*.jpg
*.jpeg
*.png
*.tif
*.tiff

# Model weights (large files)
models/weights/*.pth
models/weights/*.h5
models/weights/*.ckpt
*.pkl

# Outputs
output/
*.csv
graph.json
final.png

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Report builds
report/latex/*.aux
report/latex/*.log
report/latex/*.out
report/latex/*.synctex.gz
report/latex/*.bbl
report/latex/*.blg

# Temporary files
*.tmp
*.bak
"""
    
    Path(".gitignore").write_text(gitignore)
    print("✓ Created: .gitignore")


def create_placeholder_files():
    """Create placeholder output files."""
    
    # Create placeholder graph.json
    import json
    placeholder_graph = {
        "README": "This file will be generated by run.sh",
        "format": {
            "piece_id": ["neighbor1", "neighbor2"]
        }
    }
    Path("graph.json").write_text(json.dumps(placeholder_graph, indent=2))
    print("✓ Created: graph.json (placeholder)")


def main():
    """Main setup function."""
    print("\n" + "="*60)
    print("COMS4036A Computer Vision - Puzzle Project Setup")
    print("="*60 + "\n")
    
    print("Creating directory structure...")
    create_directory_structure()
    
    print("\nCreating __init__.py files...")
    create_init_files()
    
    print("\nCreating source files...")
    create_source_files()
    
    print("\nCreating notebooks...")
    create_notebooks()
    
    print("\nCreating configuration files...")
    create_requirements_file()
    create_run_script()
    create_readme()
    create_gitignore()
    
    print("\nCreating placeholder files...")
    create_placeholder_files()
    
    print("\n" + "="*60)
    print("Setup complete! ✓")
    print("="*60)
    print("\nNext steps:")
    print("1. python -m venv .venv")
    print("2. source .venv/bin/activate")
    print("3. pip install -r requirements.txt")
    print("4. Place your data in data/images/ and data/masks/")
    print("5. Read README.md for detailed instructions")
    print("\nHappy coding! 🧩\n")


if __name__ == "__main__":
    main()
