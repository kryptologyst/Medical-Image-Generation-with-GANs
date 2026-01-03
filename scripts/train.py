#!/usr/bin/env python3
"""Training script for medical image generation with GANs."""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import yaml
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models import DCGANGenerator, DCGANDiscriminator
from data import MedicalImageDataModule
from train import train_gan
from utils import get_device, set_seed


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_models(config: Dict[str, Any]) -> tuple:
    """Create generator and discriminator models.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Tuple of (generator, discriminator).
    """
    model_config = config['model']
    
    if model_config['generator_type'] == 'dcgan':
        generator = DCGANGenerator(
            latent_dim=model_config['latent_dim'],
            img_channels=model_config['img_channels'],
            img_size=model_config['img_size'],
            hidden_dim=model_config['hidden_dim'],
        )
    else:
        raise ValueError(f"Unknown generator type: {model_config['generator_type']}")
    
    if model_config['discriminator_type'] == 'dcgan':
        discriminator = DCGANDiscriminator(
            img_channels=model_config['img_channels'],
            img_size=model_config['img_size'],
            hidden_dim=model_config['hidden_dim'],
        )
    else:
        raise ValueError(f"Unknown discriminator type: {model_config['discriminator_type']}")
    
    return generator, discriminator


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train medical image generation GAN')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (auto, cuda, mps, cpu)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override device if specified
    if args.device is not None:
        config['device']['device'] = args.device
    
    # Set device
    if config['device']['device'] == 'auto':
        device = get_device()
    else:
        device = torch.device(config['device']['device'])
    
    # Set seed
    set_seed(config['device']['seed'])
    
    print(f"Using device: {device}")
    print(f"Configuration: {config}")
    
    # Create models
    generator, discriminator = create_models(config)
    
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Create data module
    data_config = config['data']
    data_module = MedicalImageDataModule(
        data_dir=data_config.get('data_dir'),
        batch_size=config['training']['batch_size'],
        img_size=config['model']['img_size'],
        img_channels=config['model']['img_channels'],
        num_workers=data_config['num_workers'],
        synthetic=data_config['synthetic'],
        num_synthetic_samples=data_config['num_synthetic_samples'],
    )
    
    data_module.setup()
    
    # Get data loaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Train model
    trainer = train_gan(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )
    
    print("Training completed successfully!")


if __name__ == '__main__':
    main()
