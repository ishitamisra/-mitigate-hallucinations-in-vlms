# Installation Guide

This guide will help you set up the LLaVA VCD+ASD project for mitigating hallucinations in vision-language models.

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for optimal performance)
- At least 16GB of GPU memory for the 13B model
- 32GB+ of system RAM (recommended)

## Installation Methods

### Method 1: Direct Installation (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd mitigate-hallucinations-in-vlms

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package in development mode
pip install -e .
```

### Method 2: Requirements-based Installation

```bash
# Clone the repository
git clone <repository-url>
cd mitigate-hallucinations-in-vlms

# Install dependencies directly
pip install -r requirements.txt
```

### Method 3: Conda Environment

```bash
# Create conda environment
conda create -n llava-vcd-asd python=3.10
conda activate llava-vcd-asd

# Clone and install
git clone <repository-url>
cd mitigate-hallucinations-in-vlms
pip install -e .
```

## Verify Installation

Test that everything is working correctly:

```bash
# Run the demo (will download the model on first run)
python examples/demo.py
```

## Optional Dependencies

### For Development

```bash
pip install -e ".[dev]"
```

This includes:
- pytest for testing
- black for code formatting
- flake8 for linting
- isort for import sorting

### For Jupyter Notebooks

```bash
pip install -e ".[notebook]"
```

This includes:
- jupyter for notebook support
- matplotlib and seaborn for visualization

## GPU Setup

### CUDA Installation

Make sure you have CUDA installed. Check your CUDA version:

```bash
nvidia-smi
nvcc --version
```

### PyTorch GPU Support

The requirements include CUDA-enabled PyTorch. If you need a specific CUDA version:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Model Download

The LLaVA model will be automatically downloaded from HuggingFace on first use. This may take several minutes depending on your internet connection.

To pre-download the model:

```python
from transformers import AutoProcessor, AutoModelForImageTextToText

# This will download and cache the model
processor = AutoProcessor.from_pretrained("liuhaotian/llava-v1.5-13b")
model = AutoModelForImageTextToText.from_pretrained("liuhaotian/llava-v1.5-13b")
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Use a smaller model variant if available
   - Reduce batch size or sequence length
   - Enable gradient checkpointing

2. **CUDA Errors**
   - Ensure CUDA version compatibility
   - Check GPU memory availability with `nvidia-smi`

3. **Import Errors**
   - Make sure you're in the correct virtual environment
   - Verify all dependencies are installed: `pip list`

4. **Model Download Issues**
   - Check internet connection
   - Clear HuggingFace cache: `rm -rf ~/.cache/huggingface/`

### Performance Tips

1. **Use Mixed Precision**: The model automatically uses `float16` for better memory efficiency

2. **Optimize for Your Hardware**:
   ```python
   # For better performance on modern GPUs
   import torch
   torch.backends.cudnn.benchmark = True
   ```

3. **Monitor GPU Usage**:
   ```bash
   watch -n 1 nvidia-smi
   ```

## Next Steps

After installation:

1. **Run the demo**: `python examples/demo.py`
2. **Train steering vectors**: `python scripts/train_steering_vectors.py`
3. **Evaluate the model**: `python scripts/evaluate_model.py --compare_all`

## Getting Help

If you encounter issues:

1. Check the [troubleshooting section](#troubleshooting) above
2. Review the [README.md](README.md) for usage examples
3. Open an issue on the project repository

## System Requirements Summary

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.8 | 3.10+ |
| GPU Memory | 8GB | 16GB+ |
| System RAM | 16GB | 32GB+ |
| Storage | 50GB | 100GB+ |
| CUDA | 11.8 | 12.1+ |