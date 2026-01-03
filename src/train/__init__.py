"""Training module for medical image generation."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from ..models import DCGANGenerator, DCGANDiscriminator, WassersteinLoss
from ..utils import get_device, set_seed, save_checkpoint, load_checkpoint, calculate_image_metrics


class GANTrainer:
    """Trainer for GAN models on medical images.
    
    Args:
        generator: Generator model.
        discriminator: Discriminator model.
        device: Device to train on.
        config: Training configuration.
    """
    
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        device: torch.device,
        config: Dict[str, Any],
    ):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        self.config = config
        
        # Loss functions
        self.criterion = nn.BCELoss()
        self.wasserstein_loss = WassersteinLoss()
        
        # Optimizers
        self.optimizer_G = optim.Adam(
            self.generator.parameters(),
            lr=config.get('lr_g', 0.0002),
            betas=(0.5, 0.999),
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=config.get('lr_d', 0.0002),
            betas=(0.5, 0.999),
        )
        
        # Training parameters
        self.latent_dim = config.get('latent_dim', 100)
        self.batch_size = config.get('batch_size', 32)
        self.num_epochs = config.get('num_epochs', 100)
        self.d_steps = config.get('d_steps', 1)  # Discriminator steps per generator step
        self.g_steps = config.get('g_steps', 1)  # Generator steps per discriminator step
        
        # Logging
        self.log_dir = Path(config.get('log_dir', 'logs'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        
        # Checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.best_fid = float('inf')
        
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            dataloader: Training data loader.
            
        Returns:
            Dictionary of training metrics.
        """
        self.generator.train()
        self.discriminator.train()
        
        total_d_loss = 0.0
        total_g_loss = 0.0
        num_batches = len(dataloader)
        
        pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, real_images in enumerate(pbar):
            real_images = real_images.to(self.device)
            batch_size = real_images.size(0)
            
            # Create labels
            real_labels = torch.ones(batch_size, 1, device=self.device)
            fake_labels = torch.zeros(batch_size, 1, device=self.device)
            
            # Train Discriminator
            d_loss = 0.0
            for _ in range(self.d_steps):
                self.optimizer_D.zero_grad()
                
                # Real images
                real_output = self.discriminator(real_images)
                d_loss_real = self.criterion(real_output, real_labels)
                
                # Fake images
                z = torch.randn(batch_size, self.latent_dim, device=self.device)
                fake_images = self.generator(z)
                fake_output = self.discriminator(fake_images.detach())
                d_loss_fake = self.criterion(fake_output, fake_labels)
                
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.optimizer_D.step()
            
            # Train Generator
            g_loss = 0.0
            for _ in range(self.g_steps):
                self.optimizer_G.zero_grad()
                
                z = torch.randn(batch_size, self.latent_dim, device=self.device)
                fake_images = self.generator(z)
                fake_output = self.discriminator(fake_images)
                g_loss = self.criterion(fake_output, real_labels)
                
                g_loss.backward()
                self.optimizer_G.step()
            
            # Update metrics
            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'D Loss': f'{d_loss.item():.4f}',
                'G Loss': f'{g_loss.item():.4f}',
            })
            
            # Log to tensorboard
            global_step = self.current_epoch * num_batches + batch_idx
            self.writer.add_scalar('Train/D_Loss', d_loss.item(), global_step)
            self.writer.add_scalar('Train/G_Loss', g_loss.item(), global_step)
        
        return {
            'd_loss': total_d_loss / num_batches,
            'g_loss': total_g_loss / num_batches,
        }
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model.
        
        Args:
            dataloader: Validation data loader.
            
        Returns:
            Dictionary of validation metrics.
        """
        self.generator.eval()
        self.discriminator.eval()
        
        total_d_loss = 0.0
        total_g_loss = 0.0
        total_metrics = {'mse': 0.0, 'psnr': 0.0, 'ssim': 0.0}
        num_batches = len(dataloader)
        
        with torch.no_grad():
            for real_images in tqdm(dataloader, desc="Validation"):
                real_images = real_images.to(self.device)
                batch_size = real_images.size(0)
                
                # Create labels
                real_labels = torch.ones(batch_size, 1, device=self.device)
                fake_labels = torch.zeros(batch_size, 1, device=self.device)
                
                # Discriminator loss
                real_output = self.discriminator(real_images)
                d_loss_real = self.criterion(real_output, real_labels)
                
                z = torch.randn(batch_size, self.latent_dim, device=self.device)
                fake_images = self.generator(z)
                fake_output = self.discriminator(fake_images)
                d_loss_fake = self.criterion(fake_output, fake_labels)
                
                d_loss = d_loss_real + d_loss_fake
                g_loss = self.criterion(fake_output, real_labels)
                
                total_d_loss += d_loss.item()
                total_g_loss += g_loss.item()
                
                # Calculate image metrics
                metrics = calculate_image_metrics(real_images, fake_images)
                for key, value in metrics.items():
                    total_metrics[key] += value
        
        # Average metrics
        avg_metrics = {
            'd_loss': total_d_loss / num_batches,
            'g_loss': total_g_loss / num_batches,
        }
        for key, value in total_metrics.items():
            avg_metrics[key] = value / num_batches
        
        return avg_metrics
    
    def generate_samples(self, num_samples: int = 16) -> torch.Tensor:
        """Generate sample images.
        
        Args:
            num_samples: Number of samples to generate.
            
        Returns:
            Generated images tensor.
        """
        self.generator.eval()
        
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=self.device)
            fake_images = self.generator(z)
        
        return fake_images
    
    def save_samples(self, epoch: int, num_samples: int = 16) -> None:
        """Save generated samples.
        
        Args:
            epoch: Current epoch.
            num_samples: Number of samples to generate and save.
        """
        fake_images = self.generate_samples(num_samples)
        
        # Save as numpy array
        samples_dir = self.log_dir / 'samples'
        samples_dir.mkdir(exist_ok=True)
        
        samples_path = samples_dir / f'epoch_{epoch:04d}.npy'
        np.save(samples_path, fake_images.cpu().numpy())
        
        # Log to tensorboard
        self.writer.add_images('Generated_Samples', fake_images, epoch)
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> None:
        """Train the GAN model.
        
        Args:
            train_loader: Training data loader.
            val_loader: Optional validation data loader.
        """
        print(f"Starting training for {self.num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                
                # Log validation metrics
                for key, value in val_metrics.items():
                    self.writer.add_scalar(f'Val/{key}', value, epoch)
            
            # Save samples
            if epoch % 10 == 0:
                self.save_samples(epoch)
            
            # Save checkpoint
            if epoch % 20 == 0:
                checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pth'
                save_checkpoint(
                    self.generator,
                    self.optimizer_G,
                    epoch,
                    train_metrics['g_loss'],
                    checkpoint_path,
                    discriminator_state=self.discriminator.state_dict(),
                    discriminator_optimizer_state=self.optimizer_D.state_dict(),
                    config=self.config,
                )
            
            # Print epoch summary
            print(f"Epoch {epoch:4d}/{self.num_epochs}: "
                  f"D Loss: {train_metrics['d_loss']:.4f}, "
                  f"G Loss: {train_metrics['g_loss']:.4f}")
            
            if val_loader is not None:
                print(f"  Val - D Loss: {val_metrics['d_loss']:.4f}, "
                      f"G Loss: {val_metrics['g_loss']:.4f}")
        
        print("Training completed!")
        
        # Save final model
        final_path = self.checkpoint_dir / 'final_model.pth'
        save_checkpoint(
            self.generator,
            self.optimizer_G,
            self.num_epochs - 1,
            train_metrics['g_loss'],
            final_path,
            discriminator_state=self.discriminator.state_dict(),
            discriminator_optimizer_state=self.optimizer_D.state_dict(),
            config=self.config,
        )
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint = load_checkpoint(
            checkpoint_path,
            self.generator,
            self.optimizer_G,
            self.device,
        )
        
        self.discriminator.load_state_dict(checkpoint['discriminator_state'])
        self.optimizer_D.load_state_dict(checkpoint['discriminator_optimizer_state'])
        self.current_epoch = checkpoint['epoch']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")


def train_gan(
    generator: nn.Module,
    discriminator: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    config: Optional[Dict[str, Any]] = None,
) -> GANTrainer:
    """Train a GAN model.
    
    Args:
        generator: Generator model.
        discriminator: Discriminator model.
        train_loader: Training data loader.
        val_loader: Optional validation data loader.
        config: Training configuration.
        
    Returns:
        Trained GANTrainer instance.
    """
    if config is None:
        config = {}
    
    # Set default config
    default_config = {
        'latent_dim': 100,
        'batch_size': 32,
        'num_epochs': 100,
        'lr_g': 0.0002,
        'lr_d': 0.0002,
        'd_steps': 1,
        'g_steps': 1,
        'log_dir': 'logs',
        'checkpoint_dir': 'checkpoints',
    }
    config = {**default_config, **config}
    
    # Setup device and seeding
    device = get_device()
    set_seed(config.get('seed', 42))
    
    # Create trainer
    trainer = GANTrainer(generator, discriminator, device, config)
    
    # Train
    trainer.train(train_loader, val_loader)
    
    return trainer
