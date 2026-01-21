#!/bin/bash
# setup_codebleu.sh - Script to set up CodeBLEU from CodeXGLUE in a conda environment

# Set up error handling
set -e

echo "Setting up CodeBLEU from CodeXGLUE repository..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please make sure miniconda is installed."
    exit 1
fi

# Clone the CodeXGLUE repository if it doesn't exist
if [ ! -d "CodeXGLUE" ]; then
    echo "Cloning CodeXGLUE repository..."
    git clone https://github.com/microsoft/CodeXGLUE.git
else
    echo "CodeXGLUE repository already exists."
fi

# Navigate to the evaluator directory
cd CodeXGLUE/Code-Code/code-to-code-trans/evaluator

# Install dependencies (even if requirements.txt might not exist)
echo "Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "No requirements.txt found, installing dependencies manually..."
    pip install nltk tree-sitter
fi

# Download NLTK tokenizers
echo "Downloading NLTK tokenizers..."
python -c "import nltk; nltk.download('punkt')"

# Create a simple script to make CodeBLEU easy to import
cd ../../../../
echo "Creating a Python module for easier importing..."

mkdir -p codebleu_module
touch codebleu_module/__init__.py

cat > codebleu_module/calc_codebleu.py << 'EOL'
import sys
import os
import argparse

# Add the path to the CodeXGLUE evaluator
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../CodeXGLUE/Code-Code/code-to-code-trans/evaluator'))

# Import the original calc_codebleu
from calc_codebleu import compute_codebleu, bleu_callback, weighted_ngram_match, syntax_match, dataflow_match

# Function to calculate CodeBLEU
def calculate_codebleu(references, candidates, lang="python", weights=None):
    """
    Calculate CodeBLEU score for given references and candidates.
    
    Args:
        references: List of reference code strings
        candidates: List of candidate code strings
        lang: Programming language (default: python)
        weights: Dictionary of weights for different components
        
    Returns:
        Dictionary containing CodeBLEU scores
    """
    # Save references and candidates to temporary files
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as ref_file:
        ref_file.write('\n'.join(references))
        ref_path = ref_file.name
        
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as cand_file:
        cand_file.write('\n'.join(candidates))
        cand_path = cand_file.name
    
    # Set default weights if not provided
    if weights is None:
        weights = {
            'ngram_match_weight': 0.25,
            'weighted_ngram_match_weight': 0.25,
            'syntax_match_weight': 0.25,
            'dataflow_match_weight': 0.25
        }
    
    # Calculate CodeBLEU
    result = compute_codebleu(
        ref_file=ref_path,
        hyp_file=cand_path,
        lang=lang,
        params=weights
    )
    
    # Clean up temporary files
    os.unlink(ref_path)
    os.unlink(cand_path)
    
    return result
EOL

# Install the module
pip install -e codebleu_module

# Go back to the starting directory
cd ..

echo "CodeBLEU setup complete! You can now import it in your Python scripts with: from codebleu_module.calc_codebleu import calculate_codebleu"