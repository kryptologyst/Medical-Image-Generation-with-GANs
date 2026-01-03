#!/usr/bin/env python3
"""Setup script for medical image generation project."""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("🏥 Medical Image Generation Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    commands = [
        ("pip install --upgrade pip", "Upgrading pip"),
        ("pip install -r requirements.txt", "Installing dependencies"),
        ("pip install -e .", "Installing package in development mode"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            print(f"❌ Setup failed at: {description}")
            sys.exit(1)
    
    # Create necessary directories
    directories = [
        "logs",
        "checkpoints", 
        "assets",
        "evaluation_results",
        "notebook_logs",
        "notebook_checkpoints",
        "notebook_evaluation"
    ]
    
    print("📁 Creating directories...")
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Run tests
    print("\n🧪 Running tests...")
    if run_command("pytest tests/ -v", "Running unit tests"):
        print("✅ All tests passed!")
    else:
        print("⚠️ Some tests failed, but setup continues...")
    
    # Test model creation
    print("\n🔧 Testing model creation...")
    test_script = """
import sys
sys.path.append('src')
from models import DCGANGenerator, DCGANDiscriminator
from utils import get_device

device = get_device()
generator = DCGANGenerator()
discriminator = DCGANDiscriminator()

print(f'Generator parameters: {sum(p.numel() for p in generator.parameters()):,}')
print(f'Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}')
print('Models created successfully!')
"""
    
    if run_command(f'python -c "{test_script}"', "Testing model creation"):
        print("✅ Model creation test passed!")
    
    # Test data pipeline
    print("\n📊 Testing data pipeline...")
    data_test_script = """
import sys
sys.path.append('src')
from data import MedicalImageDataModule

data_module = MedicalImageDataModule(synthetic=True, num_synthetic_samples=10)
data_module.setup()

train_loader = data_module.train_dataloader()
batch = next(iter(train_loader))

print(f'Batch shape: {batch.shape}')
print('Data pipeline working correctly!')
"""
    
    if run_command(f'python -c "{data_test_script}"', "Testing data pipeline"):
        print("✅ Data pipeline test passed!")
    
    # Final instructions
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run the interactive demo:")
    print("   streamlit run demo/app.py")
    print("\n2. Train a model:")
    print("   python scripts/train.py")
    print("\n3. Evaluate a model:")
    print("   python scripts/evaluate.py")
    print("\n4. Run Jupyter notebook:")
    print("   jupyter notebook notebooks/medical_image_generation_demo.ipynb")
    print("\n⚠️ Remember: This is for research and educational purposes only!")
    print("   NOT FOR CLINICAL USE")


if __name__ == "__main__":
    main()
