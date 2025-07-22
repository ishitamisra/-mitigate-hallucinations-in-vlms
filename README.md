# LLaVA Fine-tuning with Visual Contrastive Decoding and Activation Steering

This repository implements a comprehensive approach to mitigating object hallucinations in Large Vision-Language Models (LVLMs) by combining **Visual Contrastive Decoding (VCD)** with **Activation Steering Decoding (ASD)** techniques, applied to the LLaVA 1.5 model.

## 🎯 Overview

Object hallucinations in vision-language models occur when models generate descriptions of objects that are not present in the input image. This project addresses this critical issue by:

1. **Visual Contrastive Decoding (VCD)**: Using contrastive decoding between original and blurred images to reduce hallucinations
2. **Activation Steering Decoding (ASD)**: Applying bidirectional steering vectors to guide model activations away from hallucination patterns
3. **COCO Dataset Evaluation**: Comprehensive evaluation on MS-COCO dataset for robust performance assessment

## 📚 References

- **[VCD Paper]**: [Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding](https://arxiv.org/abs/2311.16922)
- **[ASD Paper]**: [Activation Steering Decoding: Mitigating Hallucination in Large Vision-Language Models through Bidirectional Hidden State Intervention](https://openreview.net/pdf?id=XfvmkVvnCq)
- **[LLaVA Model]**: [liuhaotian/llava-v1.5-13b](https://huggingface.co/liuhaotian/llava-v1.5-13b)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd mitigate-hallucinations-in-vlms

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.models.llava_vcd_asd import LLaVAVCDASD
from PIL import Image

# Initialize model
model = LLaVAVCDASD(model_name="liuhaotian/llava-v1.5-13b")

# Load an image
image = Image.open("path/to/your/image.jpg")

# Generate with baseline LLaVA
baseline_text = model.generate_baseline(image, "Describe this image.")

# Generate with VCD + ASD (requires pre-computed steering vectors)
enhanced_text = model.generate_with_vcd_asd(image, "Describe this image.")
```

### Run Demo

```bash
python examples/demo.py
```

## 🔧 Training Steering Vectors

Before using the full VCD+ASD functionality, you need to compute steering vectors:

```bash
# Basic training
python scripts/train_steering_vectors.py

# With hyperparameter optimization and evaluation
python scripts/train_steering_vectors.py \
    --num_samples 1000 \
    --optimize_hyperparams \
    --evaluate \
    --output_dir outputs/
```

## 📊 Evaluation

### Comprehensive Evaluation

```bash
# Compare all methods (Baseline, VCD, VCD+ASD)
python scripts/evaluate_model.py \
    --compare_all \
    --num_samples 500 \
    --steering_vectors_path outputs/steering_vectors.pt
```

### Individual Method Evaluation

```bash
# Evaluate baseline only
python scripts/evaluate_model.py --evaluate_baseline

# Evaluate VCD only
python scripts/evaluate_model.py --evaluate_vcd_only

# Evaluate VCD+ASD
python scripts/evaluate_model.py --evaluate_vcd_asd --steering_vectors_path outputs/steering_vectors.pt
```

## 📁 Project Structure

```
├── src/                          # Source code
│   ├── models/                   # Model implementations
│   │   └── llava_vcd_asd.py     # Main LLaVA VCD+ASD model
│   ├── training/                 # Training utilities
│   │   └── steering_trainer.py  # Steering vector computation
│   └── evaluation/               # Evaluation utilities
│       └── hallucination_detector.py  # Hallucination detection
├── scripts/                      # Executable scripts
│   ├── train_steering_vectors.py # Train steering vectors
│   └── evaluate_model.py        # Model evaluation
├── examples/                     # Example usage
│   └── demo.py                  # Interactive demo
├── configs/                      # Configuration files
│   └── model_config.yaml        # Model and training config
├── requirements.txt              # Dependencies
└── README.md                    # This file
```

## 🧪 Methodology

### Visual Contrastive Decoding (VCD)

VCD reduces hallucinations by contrasting predictions on original images with those on degraded (blurred) versions:

```
L' = L + α(L - L_blur)
```

Where:
- `L` is the original logits
- `L_blur` is logits from blurred image
- `α` is the contrast weight

### Activation Steering Decoding (ASD)

ASD applies bidirectional steering to model activations:

1. **Steering Vector Computation**: Compute direction vectors from truth vs. hallucination examples
2. **Bidirectional Steering**: Apply both positive (toward truth) and negative (away from hallucination) steering
3. **Contrast Decoding**: Enhance final predictions using steering contrast

## 📈 Results

The combined VCD+ASD approach typically achieves:

- **10-25% reduction** in hallucination rates compared to baseline LLaVA
- **Improved precision** in object detection and description
- **Maintained fluency** and coherence in generated text

*Note: Exact results depend on dataset, model size, and hyperparameter settings.*

## 🛠️ Configuration

Model parameters can be configured via `configs/model_config.yaml`:

```yaml
# VCD parameters
vcd:
  alpha: 0.1
  blur_intensity: 0.5

# ASD parameters  
asd:
  lambda_positive: 0.2
  lambda_negative: 0.4
  target_layer: 16
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Original VCD and ASD paper authors
- LLaVA model developers
- HuggingFace transformers library
- MS-COCO dataset

---

**Full technical report**: https://docs.google.com/document/d/1CtdF3Ng1q5PONy0UtQdlqVN2QSvhrn68BcJbfRp9bh4/edit?usp=sharing 
