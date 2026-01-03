"""Streamlit demo for medical image generation."""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models import DCGANGenerator, DCGANDiscriminator
from utils import get_device, set_seed


# Page configuration
st.set_page_config(
    page_title="Medical Image Generation Demo",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Disclaimer banner
st.error("""
⚠️ **RESEARCH DEMO ONLY** - This is a research demonstration tool for educational purposes. 
**NOT FOR CLINICAL USE** - Do not use for medical diagnosis or treatment decisions. 
Always consult qualified healthcare professionals for medical advice.
""")

# Title and description
st.title("🏥 Medical Image Generation Demo")
st.markdown("""
This demo showcases generative adversarial networks (GANs) for creating synthetic medical images.
The generated images are for research and educational purposes only.
""")

# Sidebar configuration
st.sidebar.header("Configuration")

# Model parameters
st.sidebar.subheader("Model Parameters")
latent_dim = st.sidebar.slider("Latent Dimension", 50, 512, 100)
img_size = st.sidebar.selectbox("Image Size", [128, 256, 512], index=1)
hidden_dim = st.sidebar.slider("Hidden Dimension", 32, 128, 64)

# Generation parameters
st.sidebar.subheader("Generation Parameters")
num_samples = st.sidebar.slider("Number of Samples", 1, 16, 4)
seed = st.sidebar.number_input("Random Seed", 0, 10000, 42)

# Model loading
@st.cache_resource
def load_model(latent_dim, img_size, hidden_dim):
    """Load the generator model."""
    try:
        # Try to load from checkpoint
        checkpoint_path = Path("checkpoints/final_model.pth")
        if checkpoint_path.exists():
            device = get_device()
            generator = DCGANGenerator(
                latent_dim=latent_dim,
                img_channels=1,
                img_size=img_size,
                hidden_dim=hidden_dim,
            )
            
            checkpoint = torch.load(checkpoint_path, map_location=device)
            generator.load_state_dict(checkpoint['model_state_dict'])
            generator.to(device)
            generator.eval()
            
            return generator, device
        else:
            st.warning("No trained model found. Using untrained model for demonstration.")
            device = get_device()
            generator = DCGANGenerator(
                latent_dim=latent_dim,
                img_channels=1,
                img_size=img_size,
                hidden_dim=hidden_dim,
            )
            generator.to(device)
            generator.eval()
            return generator, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Load model
generator, device = load_model(latent_dim, img_size, hidden_dim)

if generator is not None:
    # Generation section
    st.header("🎨 Generate Medical Images")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("Generate New Images", type="primary"):
            with st.spinner("Generating images..."):
                # Set seed for reproducibility
                set_seed(seed)
                
                # Generate images
                with torch.no_grad():
                    z = torch.randn(num_samples, latent_dim, device=device)
                    generated_images = generator(z)
                
                # Convert to numpy
                images = generated_images.cpu().numpy()
                
                # Store in session state
                st.session_state.generated_images = images
                st.session_state.generation_seed = seed
    
    with col2:
        st.info(f"""
        **Model Info:**
        - Parameters: {sum(p.numel() for p in generator.parameters()):,}
        - Device: {device}
        - Image Size: {img_size}x{img_size}
        """)
    
    # Display generated images
    if 'generated_images' in st.session_state:
        st.header("📊 Generated Images")
        
        images = st.session_state.generated_images
        seed_used = st.session_state.generation_seed
        
        st.write(f"Generated {len(images)} images (seed: {seed_used})")
        
        # Create grid of images
        cols = st.columns(4)
        for i, img in enumerate(images):
            with cols[i % 4]:
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(img[0], cmap='gray')
                ax.set_title(f"Sample {i+1}")
                ax.axis('off')
                st.pyplot(fig)
        
        # Download option
        st.subheader("💾 Download Generated Images")
        
        # Save as numpy array
        np_bytes = np.save(None, images)
        st.download_button(
            label="Download as .npy file",
            data=np_bytes,
            file_name=f"generated_medical_images_seed_{seed_used}.npy",
            mime="application/octet-stream"
        )
    
    # Analysis section
    if 'generated_images' in st.session_state:
        st.header("📈 Image Analysis")
        
        images = st.session_state.generated_images
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            mean_val = np.mean(images)
            st.metric("Mean Pixel Value", f"{mean_val:.3f}")
        
        with col2:
            std_val = np.std(images)
            st.metric("Std Pixel Value", f"{std_val:.3f}")
        
        with col3:
            min_val = np.min(images)
            st.metric("Min Pixel Value", f"{min_val:.3f}")
        
        with col4:
            max_val = np.max(images)
            st.metric("Max Pixel Value", f"{max_val:.3f}")
        
        # Histogram
        st.subheader("Pixel Value Distribution")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(images.flatten(), bins=50, alpha=0.7, density=True)
        ax.set_xlabel("Pixel Value")
        ax.set_ylabel("Density")
        ax.set_title("Distribution of Generated Image Pixel Values")
        st.pyplot(fig)
    
    # Model comparison section
    st.header("🔬 Model Comparison")
    
    st.markdown("""
    Compare different model configurations:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Current Model")
        st.write(f"- Latent Dim: {latent_dim}")
        st.write(f"- Image Size: {img_size}x{img_size}")
        st.write(f"- Hidden Dim: {hidden_dim}")
    
    with col2:
        st.subheader("Alternative Configurations")
        st.write("- **Small Model**: 50 latent dim, 64 hidden dim")
        st.write("- **Large Model**: 512 latent dim, 128 hidden dim")
        st.write("- **High Res**: 512x512 image size")
    
    # Technical details
    with st.expander("🔧 Technical Details"):
        st.markdown("""
        **Architecture:**
        - Generator: DCGAN with transposed convolutions
        - Discriminator: DCGAN with convolutional layers
        - Loss: Binary Cross Entropy
        
        **Training:**
        - Optimizer: Adam (β₁=0.5, β₂=0.999)
        - Learning Rate: 0.0002
        - Batch Size: 32
        
        **Data:**
        - Synthetic medical images with anatomical structures
        - Normalized to [0, 1] range
        - Grayscale images
        """)
    
    # Limitations and warnings
    with st.expander("⚠️ Limitations and Warnings"):
        st.markdown("""
        **Important Limitations:**
        
        1. **Research Only**: This is a research demonstration, not a clinical tool
        2. **Synthetic Data**: Trained on synthetic data, not real medical images
        3. **No Clinical Validation**: Generated images are not validated for clinical use
        4. **Limited Diversity**: May not capture full range of medical image variations
        5. **Quality Varies**: Generated image quality depends on training and model parameters
        
        **Ethical Considerations:**
        - Generated images should not be used to mislead or deceive
        - Always disclose that images are synthetic when used in research
        - Respect patient privacy and data protection regulations
        - Use responsibly in educational and research contexts only
        """)

else:
    st.error("Failed to load model. Please check if the model checkpoint exists.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Medical Image Generation Demo | Research and Educational Use Only</p>
    <p><small>Built with PyTorch, Streamlit, and MONAI</small></p>
</div>
""", unsafe_allow_html=True)
