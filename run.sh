#!/bin/bash
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
