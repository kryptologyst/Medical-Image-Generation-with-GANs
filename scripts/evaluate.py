#!/usr/bin/env python3
"""Evaluation script for medical image generation with GANs."""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models import DCGANGenerator, DCGANDiscriminator
from data import MedicalImageDataModule
from eval import evaluate_model
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


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    """Load model checkpoint.
    
    Args:
        model: Model to load checkpoint into.
        checkpoint_path: Path to checkpoint file.
        device: Device to load checkpoint on.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate medical image generation GAN')
    parser.add_argument('--config', type=str, default='configs/eval_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint file (overrides config)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (auto, cuda, mps, cpu)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.checkpoint is not None:
        config['model']['checkpoint_path'] = args.checkpoint
    if args.output is not None:
        config['output']['save_dir'] = args.output
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
    
    # Load checkpoint
    checkpoint_path = config['model']['checkpoint_path']
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    load_checkpoint(generator, checkpoint_path, device)
    print(f"Loaded generator from: {checkpoint_path}")
    
    # Create data module
    data_config = config['data']
    data_module = MedicalImageDataModule(
        data_dir=data_config.get('data_dir'),
        batch_size=config['evaluation']['batch_size'],
        img_size=config['model']['img_size'],
        img_channels=config['model']['img_channels'],
        num_workers=data_config['num_workers'],
        synthetic=data_config['synthetic'],
        num_synthetic_samples=data_config['num_synthetic_samples'],
    )
    
    data_module.setup()
    
    # Get data loader
    dataloader = data_module.val_dataloader()
    
    print(f"Evaluation samples: {len(dataloader.dataset)}")
    
    # Evaluate model
    report = evaluate_model(
        generator=generator,
        discriminator=discriminator,
        dataloader=dataloader,
        device=device,
        num_samples=config['evaluation']['num_samples'],
        save_dir=config['output']['save_dir'],
    )
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    print("\nGeneration Quality Metrics:")
    for metric, value in report['generation_quality'].items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nDiscriminator Performance:")
    for metric, value in report['discriminator_performance'].items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nResults saved to: {config['output']['save_dir']}")
    print("Evaluation completed successfully!")


if __name__ == '__main__':
    main()
