# COMS4036A Computer Vision - 1000-Piece Puzzle Project

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
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

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
