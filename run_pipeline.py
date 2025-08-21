"""
Simplified DIP-FFDNet Pipeline
------------------------------
This script provides a simplified version of the pipeline:
1. Run DIP denoising (original implementation)
2. Extract residuals to create noise maps
3. Use FFDNet with these noise maps
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
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

# Add current directory to path
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

# Import the FFDNet adapter
from ffdnet_adapter import FFDNetAdapter

# Import DIP utilities
from deep_image_prior.utils.denoising_utils import (
    get_image, get_noisy_image, crop_image, pil_to_np, np_to_torch, torch_to_np
)

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
    Returns both the denoised image and PSNR value
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
        denoising_method: "dip" or "noise2self"
    """
    print(f"\nProcessing {Path(img_path).name}...")
    
    # Load the image
    img_pil = crop_image(get_image(img_path, -1)[0], d=32)
    img_np = pil_to_np(img_pil)
    
    # Add synthetic noise (for consistency with DIP)
    sigma = 25
    sigma_ = sigma / 255.0
    img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)
    
    # 1. Run denoising (DIP or Noise2Self) with SURE loss or self-supervised
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
    elif denoising_method == "noise2self":
        print("Running Noise2Self denoising...")
        denoised_first, first_psnr = run_noise2self(img_noisy_np, img_np)
        method_name = "Noise2Self"
    else:
        raise ValueError(f"Unknown denoising method: {denoising_method}")
    
    t_first = time.time() - t_start
    print(f"{method_name} completed in {t_first:.2f}s with PSNR: {first_psnr:.2f} dB")
    
    denoised_first = denoised_first.astype(np.float32)
    
    # 2. Create noise map from residual
    t_start = time.time()
    noise_map = ffdnet_adapter.estimate_noise_map(denoised_first, img_noisy_np, method='mad', out='unit')
    t_map = time.time() - t_start
    print(f"Noise map created in {t_map:.2f}s")
    
    # 3. Run FFDNet with the spatial noise map
    t_start = time.time()
    denoised_ffdnet, avg_sigma = ffdnet_adapter.denoise_with_map(img_noisy_np, noise_map)
    t_ffdnet = time.time() - t_start

    # 3. Run FFDNet with the actual noise map
    t_start = time.time()
    denoised_ffdnet_oracle, _ = ffdnet_adapter.denoise_with_map(img_noisy_np, sigma_, actual_sigma=True)
    t_ffdnet_oracle = time.time() - t_start
    
    # 4. Calculate PSNR
    ffdnet_psnr = compare_psnr(img_np.squeeze(), denoised_ffdnet.squeeze())
    print(f"FFDNet after {method_name} completed in {t_ffdnet:.2f}s with PSNR: {ffdnet_psnr:.2f} dB")
    # Calculate FFDNet PSNR - oracle
    ffdnet_oracle_psnr = compare_psnr(img_np.squeeze(), denoised_ffdnet_oracle.squeeze())
    print(f"Oracle FFDNet completed in {t_ffdnet_oracle:.2f}s with PSNR: {ffdnet_oracle_psnr:.2f} dB")
    
    # 5. Visualize and save results
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
    plt.title(f"Noisy (σ=25)")
    plt.axis('off')
    
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
    plt.title(f"FFDNet (σ={sigma:.1f}, PSNR: {ffdnet_oracle_psnr:.2f} dB)")
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
        denoising_method: "dip" or "noise2self"
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
        if method_arg in ["dip", "noise2self"]:
            denoising_method = method_arg
        else:
            print(f"Unknown method '{method_arg}'. Using default 'dip'.")
    
    print(f"Running pipeline with {denoising_method} method...")
    main(denoising_method)
