"""
FFDNet Denoising Pipeline with Multiple Noise Estimation Methods
----------------------------------------------------------------
This script implements a comprehensive denoising framework that combines:
1. Pre-denoising methods: Deep Image Prior (DIP) and Noise2Self
2. Noise estimation techniques: local variance, PCA, and patch similarity methods
3. FFDNet denoising with estimated noise maps vs. oracle performance
4. Support for both uniform and spatially-varying noise scenarios

The pipeline evaluates different approaches for noise map estimation and compares
their effectiveness when used with FFDNet for final denoising.
"""
import os
import sys
import glob
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import time
from skimage.util import view_as_windows
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

# Global random state - set once at module level
rng = None

def set_noise_estimation_seed(seed=42):
    global rng
    rng = np.random.default_rng(seed)
    np.random.seed(seed)


# Initialize with default seed
set_noise_estimation_seed(42)

# Add current directory to path
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

# Import the FFDNet adapter
from ffdnet_adapter import FFDNetAdapter

# Import DIP utilities
from deep_image_prior.utils.denoising_utils import (
    get_image, get_noisy_image, crop_image, pil_to_np, np_to_torch, torch_to_np
)

def estimate_noise_local_variance(image, patch_size=7, percentile=10):
    """Estimate noise variance using local patch statistics"""
    # Extract all patches at once
    patches = view_as_windows(image, (patch_size, patch_size))

    # Calculate variance for all patches
    local_vars = np.var(patches, axis=(2, 3)).flatten()

    # Select smoothest regions (lowest variance)
    threshold = np.percentile(local_vars, percentile)
    smooth_regions_var = local_vars[local_vars <= threshold]

    # Estimate noise as median of variances in smooth regions
    noise_variance = np.median(smooth_regions_var)
    return np.sqrt(noise_variance)

def estimate_noise_pca(image, patch_size=7, num_patches=1000, num_similar=50):
    """Estimate noise using PCA on similar patches"""
    # Extract all possible patches
    patches = view_as_windows(image, (patch_size, patch_size))
    patches = patches.reshape(-1, patch_size * patch_size)

    # Randomly choose reference patches
    num_patches = min(num_patches, patches.shape[0])
    ref_indices = rng.choice(patches.shape[0], size=num_patches, replace=False)

    noise_estimates = []

    for idx in ref_indices:
        ref_patch = patches[idx]

        # Find similar patches (L2 distance)
        dists = np.sum((patches - ref_patch) ** 2, axis=1)
        similar_idx = np.argpartition(dists, num_similar)[:num_similar]
        group = patches[similar_idx]

        if group.shape[0] > 1:
            # Remove DC component from each patch
            group = group - np.mean(group, axis=1, keepdims=True)
            # Center across patches
            group -= np.mean(group, axis=0, keepdims=True)

            # PCA via SVD
            U, s, Vt = np.linalg.svd(group, full_matrices=False)
            eigs = (s ** 2) / (group.shape[0] - 1)

            # Estimate variance from the smallest few eigenvalues
            num_noise_components = max(1, patch_size * patch_size // 4)
            noise_var = np.median(eigs[:num_noise_components])

            if noise_var > 0:
                noise_estimates.append(np.sqrt(noise_var))

    return float(np.median(noise_estimates)) if noise_estimates else 0.0

def estimate_noise_patch_similarity(image, patch_size=7, stride=4):
    """Estimate noise by comparing similar patches"""
    h, w = image.shape

    # Calculate output dimensions for strided patches
    out_h = (h - patch_size) // stride + 1
    out_w = (w - patch_size) // stride + 1

    # Extract all patches at once
    patches = view_as_windows(image, (patch_size, patch_size), step=stride)
    patches = patches.reshape(out_h * out_w, patch_size * patch_size)

    # For each patch, find its most similar patch and compute difference
    noise_estimates = []
    num_samples = min(1000, len(patches))

    for i in range(num_samples):
        ref_patch = patches[i]
        # Compute distances to all other patches
        distances = np.sum((patches - ref_patch) ** 2, axis=1)
        distances[i] = np.inf  # Exclude self

        # Find most similar patch
        min_idx = np.argmin(distances)
        similar_patch = patches[min_idx]

        # The difference between similar patches is mostly noise
        diff = ref_patch - similar_patch
        # Estimate noise from difference (divide by sqrt(2) for two noisy signals)
        noise_std = np.std(diff) / np.sqrt(2)
        noise_estimates.append(noise_std)

    # Return robust estimate
    return np.median(noise_estimates)


def estimate_noise_local_variance_non_uniform(image, block_size=32, overlap=16, percentile=10):
    """Estimate spatially varying noise using overlapping blocks"""
    h, w = image.shape
    stride = block_size - overlap
    noise_map = np.zeros(image.shape)
    weight_map = np.zeros(image.shape)

    # Create smooth weight (Gaussian window)
    weight = _gaussian_window(block_size)

    # Process each block
    for i in range(0, h - block_size + 1, stride):
        for j in range(0, w - block_size + 1, stride):
            # Extract block
            block = image[i:i + block_size, j:j + block_size]

            # Estimate noise in this block using estimate_noise_local_variance
            noise_std = estimate_noise_local_variance(image=block)

            # Add to noise map with weights
            noise_map[i:i + block_size, j:j + block_size] += noise_std * weight
            weight_map[i:i + block_size, j:j + block_size] += weight

    # Normalize by weights
    noise_map = np.divide(noise_map, weight_map,
                          out=np.zeros_like(noise_map),
                          where=weight_map > 0)

    # Handle edges that weren't covered
    if np.any(weight_map == 0):
        # Fill uncovered areas with nearest values
        mask = weight_map == 0
        noise_map = _fill_missing_values(noise_map, mask)

    return noise_map

def estimate_noise_pca_non_uniform(image, patch_size=4, num_patches=20, num_similar=10,
                                   grid_step=8, local_window=32):
    """Estimate spatially varying noise using PCA on local regions"""

    h, w = image.shape
    half_window = local_window // 2

    # Initialize noise map and weight map for accumulation
    noise_map = np.zeros((h, w))
    weight_map = np.zeros((h, w))

    # Create Gaussian weight for smooth blending (same as method1)
    weight = _gaussian_window(local_window)

    # Sample on grid with reduced margin to get closer to edges
    margin = max(half_window, patch_size * 2)
    y_coords = np.arange(margin, h - margin, grid_step)
    x_coords = np.arange(margin, w - margin, grid_step)

    # Process each grid location
    for i, y in enumerate(y_coords):
        for j, x in enumerate(x_coords):
            # Extract local window around this point
            y_start = max(0, y - half_window)
            y_end = min(h, y + half_window)
            x_start = max(0, x - half_window)
            x_end = min(w, x + half_window)

            local_region = image[y_start:y_end, x_start:x_end]

            # Estimate noise in this local window
            local_noise = estimate_noise_pca(
                image=local_region,
                patch_size=patch_size,
                num_patches=num_patches,
                num_similar=num_similar
            )

            if local_noise > 0:
                # Add to noise map with gaussian weight
                noise_map[y_start:y_end, x_start:x_end] += local_noise * weight[:y_end - y_start, :x_end - x_start]
                weight_map[y_start:y_end, x_start:x_end] += weight[:y_end - y_start, :x_end - x_start]

    # Normalize by weights
    noise_map = np.divide(noise_map, weight_map,
                          out=np.zeros_like(noise_map),
                          where=weight_map > 0)

    # Handle edges that weren't covered
    if np.any(weight_map == 0):
        mask = weight_map == 0
        noise_map = _fill_missing_values(noise_map, mask)

    return noise_map

# Helper functions
def _gaussian_window(size):
    """Create 2D Gaussian window for smooth blending"""
    x = np.arange(size) - size // 2
    y = np.arange(size) - size // 2
    X, Y = np.meshgrid(x, y)
    sigma = size / 4
    window = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
    return window / window.max()


def _fill_missing_values(array, mask):
    """Fill missing values using nearest neighbor interpolation"""
    from scipy.ndimage import distance_transform_edt

    ind = distance_transform_edt(mask, return_distances=False, return_indices=True)
    return array[tuple(ind)]

def run_noise2self(img_noisy_np, img_np):
    """
    Run Noise2Self denoising algorithm
    Returns both the denoised image and PSNR value
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if len(img_noisy_np.shape) == 2:
        img_noisy_np = img_noisy_np[None, None, :, :]
    elif len(img_noisy_np.shape) == 3:
        img_noisy_np = img_noisy_np[None, :, :, :]
    
    noisy_tensor = torch.from_numpy(img_noisy_np).float().to(device)
    
    class Noise2SelfNet(nn.Module):
        def __init__(self):
            super(Noise2SelfNet, self).__init__()
            self.conv1 = nn.Conv2d(1, 128, 3, padding=1)
            self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
            self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
            self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
            self.conv5 = nn.Conv2d(128, 128, 3, padding=1)
            self.conv6 = nn.Conv2d(128, 1, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(128)
            self.bn2 = nn.BatchNorm2d(128)
            self.bn3 = nn.BatchNorm2d(128)
            self.bn4 = nn.BatchNorm2d(128)
            self.bn5 = nn.BatchNorm2d(128)
            
        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn4(self.conv4(x)))
            x = F.relu(self.bn5(self.conv5(x)))
            x = self.conv6(x)
            return x
    
    net = Noise2SelfNet().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.0003, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)
    criterion = nn.MSELoss()
    
    net.train()
    for epoch in range(300):
        h, w = noisy_tensor.shape[2], noisy_tensor.shape[3]
        mask_h = torch.randint(0, h, (h//4,))
        mask_w = torch.randint(0, w, (w//4,))
        
        input_tensor = noisy_tensor.clone()
        target_tensor = noisy_tensor.clone()
        
        for i in mask_h:
            for j in mask_w:
                neighbors = []
                if i > 0: neighbors.append(noisy_tensor[0, 0, i-1, j])
                if i < h-1: neighbors.append(noisy_tensor[0, 0, i+1, j])
                if j > 0: neighbors.append(noisy_tensor[0, 0, i, j-1])
                if j < w-1: neighbors.append(noisy_tensor[0, 0, i, j+1])
                
                if neighbors:
                    input_tensor[0, 0, i, j] = torch.stack(neighbors).mean()
        
        optimizer.zero_grad()
        output = net(input_tensor)
        
        mask = torch.zeros_like(noisy_tensor, dtype=torch.bool)
        for i in mask_h:
            for j in mask_w:
                mask[0, 0, i, j] = True
        
        loss = criterion(output[mask], target_tensor[mask])
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    net.eval()
    with torch.no_grad():
        denoised = net(noisy_tensor)
        denoised_np = denoised.cpu().numpy().squeeze()
        
        if len(img_np.shape) == 2:
            img_np_for_psnr = img_np
        else:
            img_np_for_psnr = img_np.squeeze()
            
        psnr = compare_psnr(img_np_for_psnr, denoised_np)
    
    return denoised_np, psnr

def run_dip(img_path, loss_mode="sure"):
    """
    Run DIP denoising using the original implementation
    Returns the denoised image and PSNR value
    """
    # Use the original script via subprocess to ensure we don't modify it
    import subprocess
    import tempfile
    import json
    
    # Create temporary output file
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        output_file = tmp.name
    
    # Run DIP denoising using the original implementation
    dip_script = current_dir / "runme_dip_denoising.py"
    cmd = [
        "python", str(dip_script), 
        "--loss_mode", loss_mode, 
        "--img_path", str(img_path),
        "--output_file", output_file
    ]
    
    print(f"Running DIP with {loss_mode} loss...")
    subprocess.run(cmd, check=True)
    
    # Read results from the temporary file
    try:
        with open(output_file, 'r') as f:
            results = json.load(f)
        
        denoised_path = results["denoised_path"]
        psnr = results["psnr"]
        
        # Load the denoised image
        denoised = np.array(Image.open(denoised_path)) / 255.0
        
        # Remove temporary file
        os.unlink(output_file)
        
        return denoised, psnr
    
    except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
        print(f"Error reading DIP results: {e}")
        
        # Fallback to using our own DIP implementation if the original fails
        print("Falling back to internal DIP implementation...")
        from dip_ffdnet_pipeline import run_dip_denoising
        
        denoised, noisy, clean, psnr = run_dip_denoising(img_path, loss_mode)
        return denoised, psnr

def process_image(img_path, ffdnet_adapter, denoising_method="dip"):
    """
    Process a single image through the pipeline
    
    Args:
        img_path: Path to the image
        ffdnet_adapter: FFDNet adapter instance
        denoising_method: Method to use ("dip", "noise2self", "local", etc.)
    """
    print(f"\nProcessing {Path(img_path).name}...")
    
    # Load the image
    img_pil = crop_image(get_image(img_path, -1)[0], d=32)
    img_np = pil_to_np(img_pil)
    
    # Add synthetic noise
    sigma = 25
    sigma_ = sigma / 255.0

    if denoising_method == "local_non_uniform" or denoising_method == "pca_non_uniform":
        # Define non-uniform noise limits
        sigma_min = 0.08  # low noise in center
        sigma_max = 0.12  # high noise at edges

        _, H, W = img_np.shape
        Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

        difficulty = "hard"  # "easy", "hard"

        if difficulty == "easy":
            sigma_map = np.full((H, W), sigma_min)
            sigma_map[:, W//2:] = sigma_max  # Left half low noise, right half high noise

        else:
            # normalize distance from center
            cy, cx = H // 2, W // 2
            dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            dist_norm = dist / dist.max()

            sigma_map = sigma_min + (sigma_max - sigma_min) * dist_norm

        # Create non-uniform noise
        noise = np.random.normal(0, sigma_map, size=img_np.shape)

        sigma_map = torch.from_numpy(sigma_map).to(torch.float32)

        # Noisy image
        img_noisy_np = np.clip(img_np + noise, 0, 1).astype(np.float32)

        results_dir = current_dir / "results"
        results_dir.mkdir(exist_ok=True)
        
        output_path = results_dir / f"true_noise_map_{sigma_min}_{sigma_max}_{difficulty}.pdf"

        plt.figure()
        plt.imshow(sigma_map, cmap='jet')
        plt.colorbar()
        plt.title(f'True Noise Map {denoising_method}')
        plt.savefig(output_path, dpi=40)
        plt.close()

        print(f"noise level range: {sigma_min}, {sigma_max}")

    else:

        _, img_noisy_np = get_noisy_image(img_np, sigma_)
    
    # 1. Run denoising method or noise estimation
    t_start = time.time()
    if denoising_method == "dip":
        print("Running Deep Image Prior (DIP) denoising...")
        try:
            denoised_first, first_psnr = run_dip(img_path, loss_mode="sure")
        except Exception as e:
            print(f"Error running DIP: {e}")
            print("Using the internal implementation instead...")
            from dip_ffdnet_pipeline import run_dip_denoising
            denoised_first, img_noisy_np, img_np, first_psnr = run_dip_denoising(img_path, loss_mode="sure")
        method_name = "DIP"
        scalar_sigma = False
    elif denoising_method == "noise2self":
        print("Running Noise2Self denoising...")
        denoised_first, first_psnr = run_noise2self(img_noisy_np, img_np)
        method_name = "Noise2Self"
        scalar_sigma = False
    elif denoising_method == "local":
        print("Running local noise estimation...")
        noise_map = estimate_noise_local_variance(image=img_noisy_np.squeeze(0))
        method_name = "LocalVar"
        scalar_sigma = True
    elif denoising_method == "pca":
        print("Running pca noise estimation...")
        noise_map = estimate_noise_pca(image=img_noisy_np.squeeze(0))
        method_name = "PCA"
        scalar_sigma = True
    elif denoising_method == "patch":
        print("Running patch similarity noise estimation...")
        noise_map = estimate_noise_patch_similarity(image=img_noisy_np.squeeze(0))
        method_name = "PatchSim"
        scalar_sigma = True
    elif denoising_method == "local_non_uniform":
        print("Running local non-uniform noise estimation...")
        noise_map = estimate_noise_local_variance_non_uniform(image=img_noisy_np.squeeze(0))
        noise_map = torch.from_numpy(noise_map).to(torch.float32)

        output_path = results_dir / f"estimated_noise_map_{denoising_method}_{sigma_min}_{sigma_max}_{difficulty}.pdf"

        plt.figure()
        plt.imshow(sigma_map, cmap='jet')
        plt.colorbar()
        plt.title(f'Estimated Noise Map {denoising_method}')
        plt.savefig(output_path, dpi=40)
        plt.close()

        method_name = "LocalVar_NU"
        scalar_sigma = False
    elif denoising_method == "pca_non_uniform":
        print("Running pca non-uniform noise estimation...")
        noise_map = estimate_noise_pca_non_uniform(image=img_noisy_np.squeeze(0))
        noise_map = torch.from_numpy(noise_map).to(torch.float32)

        output_path = results_dir / f"estimated_noise_map_{denoising_method}_{sigma_min}_{sigma_max}_{difficulty}.pdf"

        plt.figure()
        plt.imshow(noise_map, cmap='jet')
        plt.colorbar()
        plt.title(f'Estimated Noise Map {denoising_method}')
        plt.savefig(output_path, dpi=40)
        plt.close()

        method_name = "PCA_NU"
        scalar_sigma = False
    else:
        raise ValueError(f"Unknown denoising method: {denoising_method}")
    
    t_first = time.time() - t_start
    
    if "local" not in denoising_method and "pca" not in denoising_method and "patch" not in denoising_method: 
        print(f"{method_name} completed in {t_first:.2f}s with PSNR: {first_psnr:.2f} dB")
        denoised_first = denoised_first.astype(np.float32)

        # 2. Create noise map from residual
        t_start = time.time()
        noise_map = ffdnet_adapter.estimate_noise_map(denoised_first, img_noisy_np, method='mad', out='unit')
        t_map = time.time() - t_start
        print(f"Noise map created in {t_map:.2f}s")


    # 3. Run FFDNet with the estimated noise map
    t_start = time.time()
    denoised_ffdnet, avg_sigma = ffdnet_adapter.denoise_with_map(img_noisy_np, noise_map, scalar_sigma=scalar_sigma)
    t_ffdnet = time.time() - t_start

    # 4. Run FFDNet with the true noise map (oracle)
    t_start = time.time()
    if denoising_method == "local_non_uniform" or denoising_method == "pca_non_uniform":
        # Use the true spatial noise map
        denoised_ffdnet_oracle, _ = ffdnet_adapter.denoise_with_map(img_noisy_np, sigma_map, scalar_sigma=False)
    else:
        # Use the true scalar noise level
        denoised_ffdnet_oracle, _ = ffdnet_adapter.denoise_with_map(img_noisy_np, sigma_, scalar_sigma=True)
    t_ffdnet_oracle = time.time() - t_start
    
    # 5. Calculate PSNR
    ffdnet_psnr = compare_psnr(img_np.squeeze(), denoised_ffdnet.squeeze())
    print(f"FFDNet after {method_name} completed in {t_ffdnet:.2f}s with PSNR: {ffdnet_psnr:.2f} dB")
    # Calculate oracle PSNR
    ffdnet_oracle_psnr = compare_psnr(img_np.squeeze(), denoised_ffdnet_oracle.squeeze())
    print(f"Oracle FFDNet completed in {t_ffdnet_oracle:.2f}s with PSNR: {ffdnet_oracle_psnr:.2f} dB")

    if "local" in denoising_method or "pca" in denoising_method or "patch" in denoising_method:
        first_psnr = ffdnet_psnr
    
    # 6. Visualize and save results
    results_dir = current_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    output_path = results_dir / f"{Path(img_path).stem}_{method_name}_results.pdf"
    
    plt.figure(figsize=(20, 10))
    
    plt.subplot(231)
    plt.imshow(img_np.squeeze(), cmap='gray')
    plt.title("Clean")
    plt.axis('off')
    
    plt.subplot(232)
    plt.imshow(img_noisy_np.squeeze(), cmap='gray')
    plt.title(f"Noisy (σ={sigma})")
    plt.axis('off')
    
    if "local" not in denoising_method and "pca" not in denoising_method and "patch" not in denoising_method: 
        plt.subplot(233)
        plt.imshow(denoised_first.squeeze(), cmap='gray')
        plt.title(f"{method_name} (PSNR: {first_psnr:.2f} dB)")
        plt.axis('off')
    
        plt.subplot(234)
        plt.imshow(noise_map.squeeze(), cmap='jet')
        plt.title(f"Noise Map from {method_name} (avg σ={avg_sigma:.1f})")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
    
    plt.subplot(235)
    plt.imshow(denoised_ffdnet.squeeze(), cmap='gray')
    plt.title(f"FFDNet (σ={avg_sigma:.1f}, PSNR: {ffdnet_psnr:.2f} dB)")
    plt.axis('off')

    plt.subplot(236)
    plt.imshow(denoised_ffdnet_oracle.squeeze(), cmap='gray')
    if denoising_method == "local_non_uniform" or denoising_method == "pca_non_uniform":
        plt.title(f"Oracle FFDNet (σ=[{sigma_min:.2f}:{sigma_max:.1f}], PSNR: {ffdnet_oracle_psnr:.2f} dB)")
    else:
        plt.title(f"Oracle FFDNet (σ={sigma:.1f}, PSNR: {ffdnet_oracle_psnr:.2f} dB)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=40)
    plt.close()
    
    print(f"Results saved to {output_path}")
    
    # Return results for summary
    return {
        'image': Path(img_path).name,
        'method': method_name,
        'first_psnr': first_psnr,
        'ffdnet_psnr': ffdnet_psnr,
        'ffdnet_oracle_psnr': ffdnet_oracle_psnr,
        'avg_sigma': avg_sigma,
        'first_time': t_first,
        'ffdnet_time': t_ffdnet,
        'ffdnet_oracle_time': t_ffdnet_oracle
    }

def main(denoising_method="dip"):
    """
    Main function to run the pipeline on all test images
    
    Args:
        denoising_method: Method to use for denoising or noise estimation
    """
    print(f"Starting {denoising_method.upper()}-FFDNet Pipeline")
    print("=" * 40)
    
    # Find test images
    test_dir = current_dir / "test_set"
    if not test_dir.exists():
        print(f"Test directory not found at {test_dir}. Using testsets/FFDNet_gray instead.")
        test_dir = current_dir.parent / "testsets" / "FFDNet_gray"
    
    # Create list of test images
    test_images = list(test_dir.glob("*.png"))
    if not test_images:
        print(f"No test images found in {test_dir}")
        return
    
    print(f"Found {len(test_images)} test images")
    
    # Initialize FFDNet adapter
    ffdnet_adapter = FFDNetAdapter()
    
    # Process each image
    results = []
    for img_path in test_images:
        result = process_image(img_path, ffdnet_adapter, denoising_method)
        results.append(result)
    
    # Print summary
    method_display = results[0]['method'] if results else denoising_method.upper()
    print("\nSummary of Results:")
    print("=" * 120)
    print(f"{'Image':<20} {f'{method_display} PSNR':>12} {'FFDNet PSNR':>15} {'FFDNet Oracle':>15} {'Avg Sigma':>12}")
    print("-" * 120)
    
    avg_first_psnr = 0
    avg_ffdnet_psnr = 0
    avg_ffdnet_oracle_psnr = 0
    avg_sigma_all = 0
    
    for r in results:
        print(f"{r['image']:<20} {r['first_psnr']:>12.2f} {r['ffdnet_psnr']:>15.2f} {r['ffdnet_oracle_psnr']:>15.2f} {r['avg_sigma']:>12.1f}")
        
        avg_first_psnr += r['first_psnr']
        avg_ffdnet_psnr += r['ffdnet_psnr']
        avg_ffdnet_oracle_psnr += r['ffdnet_oracle_psnr']
        avg_sigma_all += r['avg_sigma']
    
    # Calculate averages
    if results:
        avg_first_psnr /= len(results)
        avg_ffdnet_psnr /= len(results)
        avg_ffdnet_oracle_psnr /= len(results)
        avg_sigma_all /= len(results)
        
        print("-" * 120)
        print(f"{'Average':<20} {avg_first_psnr:>12.2f} {avg_ffdnet_psnr:>15.2f} {avg_ffdnet_oracle_psnr:>15.2f} {avg_sigma_all:>12.1f}")
    
    print("=" * 120)
    
    # Save summary to file
    results_dir = current_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / f"summary_{denoising_method}.md", "w") as f:
        f.write(f"# {method_display}-FFDNet Pipeline Comparison Results\n\n")
        f.write(f"This file contains the results of running the {method_display}-FFDNet pipeline on test images.\n\n")
        f.write("## Denoising Methods Compared\n\n")
        f.write(f"1. **{method_display}**: {method_display} denoising method\n")
        f.write(f"2. **FFDNet**: FFDNet with noise maps from {method_display}\n")
        f.write("3. **FFDNet Oracle**: FFDNet with true noise level (σ=25)\n\n")
        f.write("## Results\n\n")
        f.write(f"| {'Image':<20} | {f'{method_display} PSNR':>12} | {'FFDNet PSNR':>15} | {'FFDNet Oracle':>15} | {'Avg Sigma':>12} |\n")
        f.write("|" + "-" * 20 + "|" + "-" * 14 + "|" + "-" * 17 + "|" + "-" * 17 + "|" + "-" * 14 + "|\n")
        
        for r in results:
            f.write(f"| {r['image']:<20} | {r['first_psnr']:>12.2f} | {r['ffdnet_psnr']:>15.2f} | {r['ffdnet_oracle_psnr']:>15.2f} | {r['avg_sigma']:>12.1f} |\n")
        
        f.write("|" + "-" * 20 + "|" + "-" * 14 + "|" + "-" * 17 + "|" + "-" * 17 + "|" + "-" * 14 + "|\n")
        f.write(f"| {'Average':<20} | {avg_first_psnr:>12.2f} | {avg_ffdnet_psnr:>15.2f} | {avg_ffdnet_oracle_psnr:>15.2f} | {avg_sigma_all:>12.1f} |\n\n")
        
        # Add analysis section
        f.write("## Analysis\n\n")
        if avg_ffdnet_oracle_psnr > avg_ffdnet_psnr > avg_first_psnr:
            f.write(f"FFDNet Oracle (true noise level) > FFDNet (estimated noise) > {method_display}.\n")
            f.write("This shows the importance of accurate noise estimation for FFDNet performance.\n")
        elif avg_ffdnet_psnr > avg_first_psnr:
            f.write(f"FFDNet with spatial noise maps from {method_display} provides better results than {method_display} alone.\n")
        else:
            f.write(f"{method_display} provides better results than FFDNet with spatial noise maps.\n")


if __name__ == "__main__":
    import sys
    
    # Allow command line argument to specify denoising method
    denoising_method = "dip"  # default
    if len(sys.argv) > 1:
        method_arg = sys.argv[1].lower()
        if method_arg in ["dip", "noise2self", "local", "pca", "patch", "local_non_uniform", "pca_non_uniform"]:
            denoising_method = method_arg
        else:
            print(f"Unknown method '{method_arg}'. Using default 'dip'.")
    
    print(f"Running pipeline with {denoising_method} method...")
    main(denoising_method)
