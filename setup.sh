#!/bin/bash

# OCT Image Classification - Quick Setup Script
# For WSL2 + Intel Xeon + AMD GPU systems

echo "=============================================="
echo "OCT Image Classification Setup"
echo "=============================================="
echo ""

# Detect system info
echo "System Information:"
echo "- CPU: $(lscpu | grep 'Model name' | cut -d':' -f2 | xargs)"
echo "- CPU Cores: $(nproc)"
echo "- Total RAM: $(free -h | awk '/^Mem:/ {print $2}')"
echo "- OS: $(lsb_release -d | cut -d':' -f2 | xargs)"
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

if [ $? -ne 0 ]; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

echo "✓ Python found"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

if [ $? -ne 0 ]; then
    echo "Installing python3-venv..."
    sudo apt-get update
    sudo apt-get install -y python3-venv
    python3 -m venv venv
fi

echo "✓ Virtual environment created"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install dependencies
echo ""
echo "=============================================="
echo "Installing Dependencies"
echo "=============================================="
echo ""
echo "Choose TensorFlow version:"
echo "1) intel-tensorflow (RECOMMENDED for Intel Xeon - Fastest)"
echo "2) tensorflow-cpu (Standard CPU support)"
echo "3) Skip (I'll install manually)"
echo ""
read -p "Enter choice [1-3]: " tf_choice

case $tf_choice in
    1)
        echo "Installing Intel-optimized TensorFlow..."
        pip install intel-tensorflow
        ;;
    2)
        echo "Installing standard TensorFlow-CPU..."
        pip install tensorflow-cpu
        ;;
    3)
        echo "Skipping TensorFlow installation..."
        ;;
    *)
        echo "Invalid choice. Installing tensorflow-cpu by default..."
        pip install tensorflow-cpu
        ;;
esac

# Install other requirements
echo ""
echo "Installing other dependencies..."
pip install matplotlib seaborn scikit-learn pillow numpy pandas jupyter ipykernel

echo ""
echo "✓ Dependencies installed"
echo ""

# Create directories
echo "Creating project directories..."
mkdir -p data models notebooks docs

echo "✓ Directories created"
echo ""

# Check if dataset exists
echo "=============================================="
echo "Dataset Setup"
echo "=============================================="
echo ""

if [ ! -d "data/kermany2018" ]; then
    echo "⚠️  Dataset not found in data/kermany2018/"
    echo ""
    echo "To download the dataset:"
    echo "1. Manual: Visit https://www.kaggle.com/datasets/paultimothymooney/kermany2018/data"
    echo "2. Kaggle API: Run the following commands:"
    echo "   pip install kaggle"
    echo "   kaggle datasets download -d paultimothymooney/kermany2018"
    echo "   unzip kermany2018.zip -d data/kermany2018/"
    echo ""
else
    echo "✓ Dataset found!"
fi

# Configure WSL resources
echo "=============================================="
echo "WSL Configuration Check"
echo "=============================================="
echo ""

if [ -f "/mnt/c/Users/$USER/.wslconfig" ]; then
    echo "✓ WSL config found at /mnt/c/Users/$USER/.wslconfig"
else
    echo "⚠️  No WSL config found. Consider creating one for better performance."
    echo ""
    echo "Create C:\\Users\\YourName\\.wslconfig with:"
    echo ""
    echo "[wsl2]"
    echo "memory=48GB"
    echo "processors=28"
    echo "swap=16GB"
    echo ""
fi

# Print next steps
echo ""
echo "=============================================="
echo "Setup Complete! 🎉"
echo "=============================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Download dataset (if not done):"
echo "   See instructions above"
echo ""
echo "3. Launch Jupyter:"
echo "   jupyter notebook notebooks/oct_classification.ipynb"
echo ""
echo "4. Run cells sequentially"
echo "   - Make sure to run Cell 1.5 for CPU optimization!"
echo "   - Adjust BATCH_SIZE to 64 for better performance"
echo ""
echo "Expected training time: 2-4 hours on Xeon E5-2680 v4"
echo "Expected accuracy: 92-96%"
echo ""
echo "Documentation:"
echo "- Quick start: docs/PROJECT_SETUP.md"
echo "- AMD/CPU setup: docs/AMD_GPU_SETUP.md"
echo ""
echo "Happy training! 🚀"
echo ""
