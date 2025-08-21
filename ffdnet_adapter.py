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
import torch.nn.functional as F

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
    
    def denoise_with_map(self, noisy_image, noise_map, actual_sigma=False):
        """
        Denoise an image using a spatial noise map
        
        Args:
            noisy_image: Noisy image as numpy array [C,H,W] in range [0,1]
            noise_map: Spatial noise map as numpy array [H,W] in range [0,75]
                       (represents estimated noise standard deviation per pixel)
            actual_sigma: If True, use the true sigma
        
        Returns:
            tuple: (denoised image as numpy array, average sigma value used)
        """
        # Ensure inputs are in the right format
        assert noisy_image.ndim == 3, "Noisy image must be [C,H,W]"
        
        if not actual_sigma:
            if noise_map.ndim == 2:
                # Expand to [1,H,W]
                noise_map = noise_map[np.newaxis, :, :]
        
        # Ensure values are in correct range
        if noisy_image.max() > 1.0:
            noisy_image = noisy_image / 255.0
            
        # Convert to torch tensors
        noisy_tensor = torch.from_numpy(noisy_image).float().to(self.device)
        
        ## Current FFDNet implementation expects a single noise level parameter (sigma)
        ## Calculate mean noise level from the spatial map
        if actual_sigma:
            noise_level = noise_map * 255.0
            noise_map = torch.full((1, 1, 1, 1), noise_map).to(self.device)
        else:
            noise_level = float(torch.mean(noise_map)) * 255.0

        # print(f"Using average noise level: {noise_level:.2f}")
        
        # # Create the sigma tensor as expected by FFDNet
        # noise_map = torch.full((1, 1, 1, 1), noise_level / 255.0).to(self.device)
        # # sigma = torch.full((1, 1, 1, 1), noise_level).to(self.device)
        
        # Add batch dimension if needed
        if noisy_tensor.dim() == 3:
            noisy_tensor = noisy_tensor.unsqueeze(0)
        
        # Apply FFDNet denoising
        with torch.no_grad():
            denoised_tensor = self.model(noisy_tensor, noise_map.to(noisy_tensor.device))
        
        # Convert back to numpy
        denoised_np = denoised_tensor.detach().cpu().numpy().squeeze()
        
        return denoised_np, noise_level
    
    def estimate_noise_map(
        self,
        noisy,
        denoised,
        method="var",
        win=11,
        gauss_sigma=None,      # if None, uses win/6
        edge_ref=None,  # e.g., denoised image for edge-aware weighting
        edge_alpha=2.0,                  # larger => stronger suppression near edges
        out="ffdnet255",  # unit / ffdnet255
        clip_min=0.0,
        clip_max=None,         # if None, no upper clip (before 'out' scaling)
        per_channel=False,                # if True, compute σ per channel; else on luma
        eps=1e-8,
    ):
        """
        Returns noise-map with shape (N,1,H,W) by default (or (N,C,H,W) if per_channel=True).

        Inputs should be float tensors on the same scale as your images:
        - If images are in [0,1], σ will be in [0,1] (out="unit").
        - If out="ffdnet255", returns σ in 8-bit units (≈ [0,75] typical for FFDNet).
        """
        residual = noisy - denoised

        x = residual[None, :]
        x = torch.from_numpy(x)
        if not per_channel:
            x = _rgb_to_luma(x)  # (N,1,H,W)

        N, C, H, W = x.shape
        device, dtype = x.device, x.dtype

        ks = int(win // 2 * 2 + 1)  # ensure odd
        if gauss_sigma is None:
            gauss_sigma = ks / 6.0   # gentle default

        k2d = _gaussian_kernel2d(ks, gauss_sigma, device=device).to(dtype)

        # Optional edge-aware weighting: w = 1 / (1 + alpha * |∇(edge_ref)|)
        if edge_ref is not None:
            ref = edge_ref
            if not per_channel:
                ref = _rgb_to_luma(ref)
            # Sobel gradients
            gx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], device=device, dtype=dtype).view(1,1,3,3)
            gy = gx.transpose(2,3)
            def sobel(z):
                z = F.pad(z, (1,1,1,1), mode="reflect")
                sx = F.conv2d(z, gx)
                sy = F.conv2d(z, gy)
                return torch.sqrt(sx*sx + sy*sy + eps)
            grad = sobel(ref)
            w = 1.0 / (1.0 + edge_alpha * grad / (grad.mean() + eps))
            # Weighted local moments
            mw  = _conv2d_same(w, k2d) + eps
            m1  = _conv2d_same(w * x, k2d) / mw
            m2  = _conv2d_same(w * (x * x), k2d) / mw
            var = (m2 - m1 * m1).clamp_min(0.0)
            sigma = torch.sqrt(var + eps)
        else:
            if method == "var":
                m1 = _conv2d_same(x, k2d)
                m2 = _conv2d_same(x * x, k2d)
                var = (m2 - m1 * m1).clamp_min(0.0)
                sigma = torch.sqrt(var + eps)
            elif method == "mad":
                # Robust local MAD via unfold (slower, but sturdy)
                pad = ks // 2
                xu = F.pad(x, (pad,pad,pad,pad), mode="reflect")
                patches = F.unfold(xu, kernel_size=ks, stride=1)  # (N*C*ks*ks, H*W)
                patches = patches.view(N, C, ks*ks, H*W)
                med = patches.median(dim=2).values  # (N,C,L)
                abs_dev = (patches - med.unsqueeze(2)).abs()
                mad = abs_dev.median(dim=2).values.view(N, C, H, W)
                sigma = 1.4826 * mad
            else:
                raise ValueError("method must be 'var' or 'mad'.")

        # Optional pre-clip (on native scale)
        if clip_max is not None:
            sigma = sigma.clamp(min=clip_min, max=clip_max)
        else:
            sigma = sigma.clamp_min(clip_min)

        # Output scaling
        if out == "unit":
            return sigma  # same scale as inputs (e.g., [0,1])
        elif out == "ffdnet255":
            # Convert unit-scale σ (for images in [0,1]) to 8-bit units.
            sigma_255 = 255.0 * sigma
            # It’s common to clip to [0,75] for FFDNet models trained on AWGN up to 75.
            return sigma_255.clamp_min(0.0)
        else:
            raise ValueError("out must be 'unit' or 'ffdnet255'")
    
    
def _gaussian_kernel2d(ks: int, sigma: float, device):
    ax = torch.arange(ks, device=device) - (ks - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    k = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    k = k / k.sum()
    return k

def _conv2d_same(x, k2d):
    # depthwise conv on 1-channel inputs; pad reflect to avoid edge bias
    pad = k2d.shape[-1] // 2
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    k = k2d.view(1, 1, *k2d.shape)
    return F.conv2d(x, k)

def _rgb_to_luma(x):
    # x: (N,C,H,W), returns (N,1,H,W)
    if x.shape[1] == 1:
        return x
    w = torch.tensor([0.2989, 0.5870, 0.1140], device=x.device, dtype=x.dtype).view(1,3,1,1)
    return (x * w).sum(dim=1, keepdim=True)
