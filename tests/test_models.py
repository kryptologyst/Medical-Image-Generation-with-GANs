"""Tests for medical image generation models."""

import pytest
import torch
import numpy as np

from src.models import DCGANGenerator, DCGANDiscriminator, WassersteinLoss
from src.utils import get_device, set_seed, normalize_image, create_synthetic_medical_image


class TestDCGANGenerator:
    """Test DCGAN Generator."""
    
    def test_generator_creation(self):
        """Test generator creation with default parameters."""
        generator = DCGANGenerator()
        assert isinstance(generator, DCGANGenerator)
        assert generator.latent_dim == 100
        assert generator.img_channels == 1
        assert generator.img_size == 256
    
    def test_generator_forward(self):
        """Test generator forward pass."""
        generator = DCGANGenerator(latent_dim=50, img_size=128)
        batch_size = 4
        z = torch.randn(batch_size, 50)
        
        output = generator(z)
        
        assert output.shape == (batch_size, 1, 128, 128)
        assert torch.all(output >= -1) and torch.all(output <= 1)  # Tanh output
    
    def test_generator_parameters(self):
        """Test generator has trainable parameters."""
        generator = DCGANGenerator()
        params = list(generator.parameters())
        
        assert len(params) > 0
        assert all(p.requires_grad for p in params)


class TestDCGANDiscriminator:
    """Test DCGAN Discriminator."""
    
    def test_discriminator_creation(self):
        """Test discriminator creation with default parameters."""
        discriminator = DCGANDiscriminator()
        assert isinstance(discriminator, DCGANDiscriminator)
        assert discriminator.img_channels == 1
        assert discriminator.img_size == 256
    
    def test_discriminator_forward(self):
        """Test discriminator forward pass."""
        discriminator = DCGANDiscriminator(img_size=128)
        batch_size = 4
        x = torch.randn(batch_size, 1, 128, 128)
        
        output = discriminator(x)
        
        assert output.shape == (batch_size, 1)
        assert torch.all(output >= 0) and torch.all(output <= 1)  # Sigmoid output
    
    def test_discriminator_parameters(self):
        """Test discriminator has trainable parameters."""
        discriminator = DCGANDiscriminator()
        params = list(discriminator.parameters())
        
        assert len(params) > 0
        assert all(p.requires_grad for p in params)


class TestWassersteinLoss:
    """Test Wasserstein Loss."""
    
    def test_wasserstein_loss_creation(self):
        """Test Wasserstein loss creation."""
        loss_fn = WassersteinLoss()
        assert isinstance(loss_fn, WassersteinLoss)
    
    def test_wasserstein_loss_forward(self):
        """Test Wasserstein loss forward pass."""
        loss_fn = WassersteinLoss()
        batch_size = 4
        
        real_output = torch.randn(batch_size, 1)
        fake_output = torch.randn(batch_size, 1)
        
        # Mock discriminator for gradient penalty
        class MockDiscriminator:
            def __call__(self, x):
                return torch.randn_like(x)
        
        discriminator = MockDiscriminator()
        
        d_loss, g_loss = loss_fn(real_output, fake_output, discriminator)
        
        assert isinstance(d_loss, torch.Tensor)
        assert isinstance(g_loss, torch.Tensor)
        assert d_loss.requires_grad
        assert g_loss.requires_grad


class TestUtils:
    """Test utility functions."""
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ['cuda', 'mps', 'cpu']
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Test that seed affects random generation
        torch.manual_seed(42)
        a = torch.randn(10)
        
        set_seed(42)
        torch.manual_seed(42)
        b = torch.randn(10)
        
        assert torch.allclose(a, b)
    
    def test_normalize_image(self):
        """Test image normalization."""
        # Create test image
        img = torch.randn(1, 3, 32, 32)
        
        normalized = normalize_image(img, 0.0, 1.0)
        
        assert torch.all(normalized >= 0.0)
        assert torch.all(normalized <= 1.0)
        assert normalized.shape == img.shape
    
    def test_create_synthetic_medical_image(self):
        """Test synthetic medical image creation."""
        img = create_synthetic_medical_image(height=64, width=64)
        
        assert isinstance(img, np.ndarray)
        assert img.shape == (64, 64)
        assert img.dtype == np.float32
        assert np.all(img >= 0) and np.all(img <= 1)


class TestIntegration:
    """Integration tests."""
    
    def test_generator_discriminator_integration(self):
        """Test generator and discriminator work together."""
        generator = DCGANGenerator(latent_dim=50, img_size=64)
        discriminator = DCGANDiscriminator(img_size=64)
        
        batch_size = 2
        z = torch.randn(batch_size, 50)
        
        # Generate fake images
        fake_images = generator(z)
        
        # Discriminate fake images
        fake_output = discriminator(fake_images)
        
        assert fake_images.shape == (batch_size, 1, 64, 64)
        assert fake_output.shape == (batch_size, 1)
    
    def test_training_step_simulation(self):
        """Test a simulated training step."""
        generator = DCGANGenerator(latent_dim=50, img_size=64)
        discriminator = DCGANDiscriminator(img_size=64)
        
        batch_size = 2
        real_images = torch.randn(batch_size, 1, 64, 64)
        z = torch.randn(batch_size, 50)
        
        # Forward passes
        fake_images = generator(z)
        real_output = discriminator(real_images)
        fake_output = discriminator(fake_images.detach())
        
        # Loss calculation
        criterion = torch.nn.BCELoss()
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        
        d_loss_real = criterion(real_output, real_labels)
        d_loss_fake = criterion(fake_output, fake_labels)
        d_loss = d_loss_real + d_loss_fake
        
        fake_output_gen = discriminator(fake_images)
        g_loss = criterion(fake_output_gen, real_labels)
        
        assert d_loss.item() > 0
        assert g_loss.item() > 0


if __name__ == '__main__':
    pytest.main([__file__])
