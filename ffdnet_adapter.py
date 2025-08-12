"""
FFDNet adapter for spatial noise maps
------------------------------------
This module provides a simple interface to the FFDNet model 
with a focus on using spatial noise maps derived from DIP residuals.
"""
import os
import numpy as np
import torch
from torch.autograd import Variable
from pathlib import Path
import sys

# Add paths for imports
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

class FFDNetAdapter:
    """
    A wrapper class for FFDNet that makes it easy to use with 
    spatial noise maps derived from DIP residuals
    """
    def __init__(self, model_path=None, use_cuda=True):
        """
        Initialize the FFDNet adapter
        
        Args:
            model_path: Path to the pre-trained FFDNet model (.pth)
            use_cuda: Whether to use CUDA (if available)
        """
        from models.network_ffdnet import FFDNet
        
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
        
        # Default model path
        if model_path is None:
            model_path = os.path.join(current_dir, 'model_zoo', 'ffdnet_gray.pth')
        
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"FFDNet model not found at {model_path}")
        
        # Load model
        self.model = FFDNet(in_nc=1)
        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"FFDNet model loaded from {model_path}")

    def create_spatial_map_from_residual(self, denoised, noisy, window_size=32):
        """
        Create a spatial noise map from a residual image using sliding window
        
        Args:
            residual: Residual image (difference between original and denoised)
            window_size: Size of the sliding window
            
        Returns:
            Spatial noise map
        """
        # Ensure both are in range [0, 1]
        if denoised.max() > 1.0:
            denoised = denoised / 255.0
        if noisy.max() > 1.0:
            noisy = noisy / 255.0
        
        # Calculate absolute residual
        residual = np.abs(denoised - noisy).squeeze(0)
        
        # For multi-channel, take mean across channels
        if residual.ndim > 2 and residual.shape[0] > 1:
            residual = np.mean(residual, axis=0, keepdims=True)
        
        # # Scale to noise map range [0, 75]
        # # This scaling is based on empirical observations of typical noise levels
        # noise_map = np.clip(residual * 75.0, 0, 75)

        # Get image dimensions
        h, w = residual.shape
        
        # Pad image to handle borders
        pad_size = window_size // 2
        padded = np.pad(residual, pad_size, mode='reflect')
        
        # Create output noise map
        noise_map = np.zeros_like(residual)
        
        # Use sliding window to estimate local noise levels
        stride = window_size // 2  # 50% overlap
        for i in range(0, h, stride):
            for j in range(0, w, stride):
                # Extract window from padded image
                window = padded[i:i+window_size, j:j+window_size]
                
                # Calculate local noise estimate (MAD estimator)
                local_noise = np.sqrt(np.mean(window**2))
                
                # Update noise map
                noise_map[i:min(i+stride, h), j:min(j+stride, w)] = local_noise
        
        # Multiply by 255 to get noise level in pixel range
        return noise_map * 255
    
    def denoise_with_map(self, noisy_image, noise_map):
        """
        Denoise an image using a spatial noise map
        
        Args:
            noisy_image: Noisy image as numpy array [C,H,W] in range [0,1]
            noise_map: Spatial noise map as numpy array [H,W] in range [0,75]
                       (represents estimated noise standard deviation per pixel)
        
        Returns:
            tuple: (denoised image as numpy array, average sigma value used)
        """
        # Ensure inputs are in the right format
        assert noisy_image.ndim == 3, "Noisy image must be [C,H,W]"
        
        if noise_map.ndim == 2:
            # Expand to [1,H,W]
            noise_map = noise_map[np.newaxis, :, :]
        
        # Ensure values are in correct range
        if noisy_image.max() > 1.0:
            noisy_image = noisy_image / 255.0
            
        # Convert to torch tensors
        noisy_tensor = torch.from_numpy(noisy_image).float().to(self.device)
        
        # Current FFDNet implementation expects a single noise level parameter (sigma)
        # Calculate mean noise level from the spatial map
        noise_level = float(np.mean(noise_map))
        print(f"Using average noise level: {noise_level:.2f}")
        
        # Create the sigma tensor as expected by FFDNet
        sigma = torch.full((1, 1, 1, 1), noise_level / 255.0).to(self.device)
        # sigma = torch.full((1, 1, 1, 1), noise_level).to(self.device)
        
        # Add batch dimension if needed
        if noisy_tensor.dim() == 3:
            noisy_tensor = noisy_tensor.unsqueeze(0)
        
        # Apply FFDNet denoising
        with torch.no_grad():
            denoised_tensor = self.model(noisy_tensor, sigma)
        
        # Convert back to numpy
        denoised_np = denoised_tensor.detach().cpu().numpy().squeeze()
        
        return denoised_np, noise_level
    
    # def create_spatial_map_from_residual(self, denoised, noisy):
    #     """
    #     Create a spatial noise map from the residual between denoised and noisy images
        
    #     Args:
    #         denoised: Denoised image as numpy array
    #         noisy: Original noisy image as numpy array
            
    #     Returns:
    #         Spatial noise map suitable for FFDNet
    #     """
    #     # Ensure both are in range [0, 1]
    #     if denoised.max() > 1.0:
    #         denoised = denoised / 255.0
    #     if noisy.max() > 1.0:
    #         noisy = noisy / 255.0
        
    #     # Calculate absolute residual
    #     residual = np.abs(denoised - noisy)
        
    #     # For multi-channel, take mean across channels
    #     if residual.ndim > 2 and residual.shape[0] > 1:
    #         residual = np.mean(residual, axis=0, keepdims=True)
        
    #     # Scale to noise map range [0, 75]
    #     # This scaling is based on empirical observations of typical noise levels
    #     noise_map = np.clip(residual * 75.0, 0, 75)
        
    #     return noise_map
