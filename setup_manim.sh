#!/bin/bash

# Setup script for Manim Causal Inference Animation

echo "Setting up Manim environment for Causal Inference Animation..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install Python3 first."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv manim_env

# Activate virtual environment
echo "Activating virtual environment..."
source manim_env/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install manim
pip install scipy numpy matplotlib

echo "Setup complete!"
echo ""
echo "To run the animation:"
echo "1. Activate the environment: source manim_env/bin/activate"
echo "2. Run the animation: manim -pql causal_inference_simple.py CausalInferenceDemo"
echo ""
echo "Available scenes:"
echo "- CausalInferenceDemo: Main demonstration"
echo "- MathFormulationOnly: Just the mathematical formulation"
echo "- ParameterEffectsDemo: Shows parameter effects"
echo ""
echo "Output will be saved in media/videos/causal_inference_simple/480p15/"
