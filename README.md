# Medical Image Generation with GANs

A production-ready implementation of Generative Adversarial Networks (GANs) for medical image generation, designed for research and educational purposes.

## ⚠️ IMPORTANT DISCLAIMER

**THIS IS A RESEARCH DEMONSTRATION TOOL FOR EDUCATIONAL PURPOSES ONLY**

- **NOT FOR CLINICAL USE** - Do not use for medical diagnosis or treatment decisions
- **NOT MEDICAL ADVICE** - Always consult qualified healthcare professionals
- **RESEARCH ONLY** - Intended for academic research and educational demonstrations
- **NO CLINICAL VALIDATION** - Generated images are not validated for clinical applications

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Medical-Image-Generation-with-GANs.git
cd Medical-Image-Generation-with-GANs
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the interactive demo:
```bash
streamlit run demo/app.py
```

### Training a Model

1. Train with default configuration:
```bash
python scripts/train.py
```

2. Train with custom configuration:
```bash
python scripts/train.py --config configs/train_config.yaml
```

3. Resume training from checkpoint:
```bash
python scripts/train.py --resume checkpoints/checkpoint_epoch_0200.pth
```

### Evaluation

1. Evaluate trained model:
```bash
python scripts/evaluate.py
```

2. Evaluate with custom checkpoint:
```bash
python scripts/evaluate.py --checkpoint checkpoints/final_model.pth
```

## 📁 Project Structure

```
medical-image-generation/
├── src/                    # Source code
│   ├── models/            # GAN architectures
│   ├── data/              # Data loading and preprocessing
│   ├── train/             # Training utilities
│   ├── eval/              # Evaluation metrics
│   └── utils/             # Utility functions
├── configs/               # Configuration files
├── scripts/               # Training and evaluation scripts
├── demo/                  # Streamlit demo application
├── tests/                 # Unit tests
├── notebooks/             # Jupyter notebooks
├── assets/                # Generated samples and visualizations
├── logs/                  # Training logs and tensorboard
├── checkpoints/           # Model checkpoints
└── requirements.txt       # Python dependencies
```

## Architecture

### Models

- **DCGAN Generator**: Deep Convolutional GAN with transposed convolutions
- **DCGAN Discriminator**: Convolutional discriminator with batch normalization
- **StyleGAN Generator**: Progressive growing generator (experimental)
- **PatchGAN Discriminator**: Patch-based discriminator for high-resolution images

### Key Features

- **Device Fallback**: Automatic CUDA → MPS → CPU device selection
- **Deterministic Training**: Reproducible results with proper seeding
- **Medical Image Preprocessing**: Specialized transforms for medical imaging
- **Comprehensive Evaluation**: FID, IS, LPIPS, SSIM, and clinical metrics
- **Interactive Demo**: Streamlit-based web interface
- **Synthetic Data**: Built-in synthetic medical image generation

## Evaluation Metrics

### Generation Quality
- **FID (Fréchet Inception Distance)**: Measures distribution similarity
- **IS (Inception Score)**: Evaluates image quality and diversity
- **LPIPS**: Learned perceptual similarity
- **SSIM**: Structural similarity index
- **MSE/PSNR**: Pixel-level reconstruction metrics

### Discriminator Performance
- **AUC**: Area under ROC curve
- **AP**: Average precision
- **Accuracy**: Classification accuracy
- **Real/Fake Accuracy**: Separate accuracy for each class

## 🔧 Configuration

### Training Configuration (`configs/train_config.yaml`)

```yaml
model:
  generator_type: "dcgan"
  latent_dim: 100
  img_size: 256
  hidden_dim: 64

training:
  batch_size: 32
  num_epochs: 100
  lr_g: 0.0002
  lr_d: 0.0002

data:
  synthetic: true
  num_synthetic_samples: 1000
```

### Evaluation Configuration (`configs/eval_config.yaml`)

```yaml
evaluation:
  num_samples: 1000
  metrics: ["fid", "is", "lpips", "ssim"]

output:
  save_dir: "evaluation_results"
  save_plots: true
```

## Interactive Demo

The Streamlit demo provides:

- **Real-time Generation**: Generate images with custom parameters
- **Visualization**: Side-by-side comparison of generated images
- **Analysis**: Statistical analysis of generated images
- **Download**: Export generated images for further analysis
- **Model Comparison**: Compare different model configurations

### Running the Demo

```bash
streamlit run demo/app.py
```

Access the demo at `http://localhost:8501`

## Training Process

### Data Pipeline

1. **Synthetic Data Generation**: Creates realistic medical images with anatomical structures
2. **Preprocessing**: Normalization, resizing, and augmentation
3. **Data Loading**: Efficient batching with multiple workers

### Training Loop

1. **Discriminator Training**: Distinguish real from fake images
2. **Generator Training**: Fool the discriminator
3. **Evaluation**: Regular assessment of generation quality
4. **Checkpointing**: Save model states and generated samples

### Monitoring

- **TensorBoard**: Real-time training metrics and visualizations
- **Sample Generation**: Regular generation of sample images
- **Model Checkpoints**: Automatic saving of best models

## Testing

Run the test suite:

```bash
pytest tests/
```

Run specific tests:

```bash
pytest tests/test_models.py
pytest tests/test_data.py
```

## Usage Examples

### Basic Training

```python
from src.models import DCGANGenerator, DCGANDiscriminator
from src.data import MedicalImageDataModule
from src.train import train_gan

# Create models
generator = DCGANGenerator(latent_dim=100, img_size=256)
discriminator = DCGANDiscriminator(img_size=256)

# Setup data
data_module = MedicalImageDataModule(synthetic=True)
data_module.setup()

# Train
trainer = train_gan(
    generator=generator,
    discriminator=discriminator,
    train_loader=data_module.train_dataloader(),
    config={'num_epochs': 100}
)
```

### Evaluation

```python
from src.eval import evaluate_model

# Evaluate trained model
report = evaluate_model(
    generator=generator,
    discriminator=discriminator,
    dataloader=val_loader,
    num_samples=1000
)

print(f"FID: {report['generation_quality']['fid']:.2f}")
print(f"IS: {report['generation_quality']['is_mean']:.2f}")
```

### Custom Data Loading

```python
from src.data import MedicalImageDataset, get_medical_transforms

# Load real medical images
dataset = MedicalImageDataset(
    data_dir="path/to/medical/images",
    img_size=256,
    transform=get_medical_transforms(is_training=True)
)
```

## Research Applications

### Potential Use Cases

1. **Data Augmentation**: Generate additional training samples
2. **Privacy Preservation**: Create synthetic datasets without patient data
3. **Educational Tools**: Train medical students with synthetic cases
4. **Research Validation**: Test algorithms on controlled synthetic data
5. **Rare Disease Simulation**: Generate examples of rare conditions

### Limitations

- **Clinical Validation**: Not validated for clinical use
- **Limited Diversity**: May not capture full range of medical variations
- **Quality Dependency**: Results depend on training data and parameters
- **Ethical Considerations**: Must be used responsibly and transparently

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Development Setup

```bash
pip install -e ".[dev]"
pre-commit install
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **MONAI**: Medical imaging deep learning framework
- **PyTorch**: Deep learning framework
- **Streamlit**: Web application framework
- **Medical imaging community**: For datasets and research

## Support

For questions and support:

- **Issues**: Use GitHub issues for bug reports and feature requests
- **Discussions**: Use GitHub discussions for general questions
- **Documentation**: Check the code documentation and examples

## Version History

- **v0.1.0**: Initial release with DCGAN implementation
- **v0.2.0**: Added StyleGAN and comprehensive evaluation
- **v0.3.0**: Interactive demo and improved documentation

---

**Remember**: This tool is for research and educational purposes only. Always consult qualified healthcare professionals for medical advice and never use generated images for clinical diagnosis or treatment decisions.
# Medical-Image-Generation-with-GANs
