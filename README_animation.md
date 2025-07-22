# Causal Inference Animation for Manim

This project creates animated visualizations of causal inference in auditory-visual duration perception using Manim.

## Files

- `causal_inference_no_latex.py` - Main animation file with multiple scenes
- `causal_inference_simple.py` - Original version with LaTeX (may have issues)
- `setup_manim.sh` - Setup script for dependencies
- `requirements.txt` - Python dependencies

## Available Animations

### 1. `SimpleCausalInferenceDemo`
Basic conceptual explanation of causal inference
```bash
manim -pql causal_inference_no_latex.py SimpleCausalInferenceDemo
```

### 2. `CausalInferenceSimulation` 
**Main simulation with actual data plots** - Shows histograms and probability distributions
```bash
manim -pql causal_inference_no_latex.py CausalInferenceSimulation
```

### 3. `VisualDemo`
Simple visualization of different conflict scenarios
```bash
manim -pql causal_inference_no_latex.py VisualDemo
```

### 4. `ConceptualDemo`
Decision tree visualization
```bash
manim -pql causal_inference_no_latex.py ConceptualDemo
```

### 5. `InteractiveStyleDemo`
Shows parameter effects similar to your interactive plots
```bash
manim -pql causal_inference_no_latex.py InteractiveStyleDemo
```

## Why LaTeX Doesn't Work

LaTeX issues in Manim are common and can be caused by:

1. **Missing LaTeX installation**: Manim requires a full LaTeX distribution
2. **LaTeX engine compatibility**: Different systems use different LaTeX engines
3. **Package conflicts**: Some LaTeX packages may conflict with Manim
4. **Font issues**: LaTeX font rendering can be problematic

### Solutions:

#### Option 1: Install LaTeX properly
```bash
# macOS
brew install --cask mactex

# Ubuntu/Debian
sudo apt-get install texlive-full

# Then try LaTeX version
manim -pql causal_inference_simple.py CausalInferenceDemo
```

#### Option 2: Use Text instead of MathTex (Current approach)
The `causal_inference_no_latex.py` file uses `Text()` instead of `MathTex()` to avoid LaTeX issues.

#### Option 3: Configure Manim for LaTeX
Add to your `manim.cfg`:
```ini
[CLI]
tex_template = TexTemplateLibrary.ctex
```

## Quick Start

1. **Install dependencies:**
```bash
pip install manim scipy numpy matplotlib
```

2. **Run the main simulation (with plots):**
```bash
manim -pql causal_inference_no_latex.py CausalInferenceSimulation
```

3. **Run all scenes:**
```bash
manim -pql causal_inference_no_latex.py SimpleCausalInferenceDemo
manim -pql causal_inference_no_latex.py CausalInferenceSimulation
manim -pql causal_inference_no_latex.py VisualDemo
manim -pql causal_inference_no_latex.py ConceptualDemo
manim -pql causal_inference_no_latex.py InteractiveStyleDemo
```

## Output

Videos will be saved in: `media/videos/causal_inference_no_latex/480p15/`

## Key Features

- **Actual simulation data**: Uses the same causal inference computation as your notebook
- **Multiple scenarios**: Shows different conflict levels and their effects
- **Probability distributions**: Displays auditory, visual, and final estimate distributions
- **Parameter effects**: Demonstrates how conflict, noise, and priors affect fusion
- **No LaTeX dependency**: Uses text-based equations to avoid LaTeX issues

## Troubleshooting

1. **ImportError**: Make sure Manim is installed: `pip install manim`
2. **LaTeX errors**: Use the `causal_inference_no_latex.py` version
3. **No output**: Check that output directory exists: `mkdir -p media/videos/`
4. **Slow rendering**: Use `-ql` flag for faster low-quality preview

## Customization

To modify parameters, edit the scenario dictionaries in each scene:
```python
scenarios = [
    {"name": "No Conflict", "S_a": 0.8, "conflict": 0.0, "p_c": 0.9, "color": GREEN},
    {"name": "Small Conflict", "S_a": 0.8, "conflict": 0.4, "p_c": 0.7, "color": YELLOW},
    {"name": "Large Conflict", "S_a": 0.8, "conflict": 0.8, "p_c": 0.3, "color": RED},
]
```
