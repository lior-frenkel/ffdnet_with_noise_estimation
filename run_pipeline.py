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

def process_image(img_path, ffdnet_adapter):
    """
    Process a single image through the pipeline
    """
    print(f"\nProcessing {Path(img_path).name}...")
    
    # Load the image
    img_pil = crop_image(get_image(img_path, -1)[0], d=32)
    img_np = pil_to_np(img_pil)
    
    # Add synthetic noise (for consistency with DIP)
    sigma = 25
    sigma_ = sigma / 255.0
    img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)
    
    # 1. Run DIP denoising with SURE loss
    t_start = time.time()
    try:
        denoised_dip, dip_psnr = run_dip(img_path, loss_mode="sure")
    except Exception as e:
        print(f"Error running DIP: {e}")
        print("Using the internal implementation instead...")
        from dip_ffdnet_pipeline import run_dip_denoising
        denoised_dip, img_noisy_np, img_np, dip_psnr = run_dip_denoising(img_path, loss_mode="sure")
    
    t_dip = time.time() - t_start
    print(f"DIP completed in {t_dip:.2f}s with PSNR: {dip_psnr:.2f} dB")
    
    # 2. Create noise map from residual
    t_start = time.time()
    noise_map = ffdnet_adapter.create_spatial_map_from_residual(denoised_dip, img_noisy_np)
    t_map = time.time() - t_start
    print(f"Noise map created in {t_map:.2f}s")
    
    # 3. Run FFDNet with the spatial noise map
    t_start = time.time()
    denoised_ffdnet = ffdnet_adapter.denoise_with_map(img_noisy_np, noise_map)
    t_ffdnet = time.time() - t_start
    
    # Calculate FFDNet PSNR
    ffdnet_psnr = compare_psnr(img_np.squeeze(), denoised_ffdnet.squeeze())
    print(f"FFDNet completed in {t_ffdnet:.2f}s with PSNR: {ffdnet_psnr:.2f} dB")
    
        # 5. Visualize and save results
    results_dir = current_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    output_path = results_dir / f"{Path(img_path).stem}_results.pdf"
    
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
    plt.imshow(denoised_dip.squeeze(), cmap='gray')
    plt.title(f"DIP (PSNR: {dip_psnr:.2f} dB)")
    plt.axis('off')
    
    plt.subplot(234)
    plt.imshow(noise_map.squeeze(), cmap='jet')
    plt.title("Noise Map from DIP")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.subplot(235)
    plt.imshow(denoised_ffdnet.squeeze(), cmap='gray')
    plt.title(f"FFDNet (PSNR: {ffdnet_psnr:.2f} dB)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=40)
    plt.close()
    
    print(f"Results saved to {output_path}")
    
    # Return results for summary
    return {
        'image': Path(img_path).name,
        'dip_psnr': dip_psnr,
        'ffdnet_psnr': ffdnet_psnr,
        'dip_time': t_dip,
        'ffdnet_time': t_ffdnet
    }

def main():
    """
    Main function to run the pipeline on all test images
    """
    print("Starting DIP-FFDNet Pipeline")
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
        result = process_image(img_path, ffdnet_adapter)
        results.append(result)
    
    # Print summary
    print("\nSummary of Results:")
    print("=" * 80)
    print(f"{'Image':<20} {'DIP PSNR':>10} {'FFDNet PSNR':>15}")
    print("-" * 80)
    
    avg_dip_psnr = 0
    avg_ffdnet_psnr = 0
    
    for r in results:
        print(f"{r['image']:<20} {r['dip_psnr']:>10.2f} {r['ffdnet_psnr']:>15.2f}")
        
        avg_dip_psnr += r['dip_psnr']
        avg_ffdnet_psnr += r['ffdnet_psnr']
    
    # Calculate averages
    if results:
        avg_dip_psnr /= len(results)
        avg_ffdnet_psnr /= len(results)
        
        print("-" * 80)
        print(f"{'Average':<20} {avg_dip_psnr:>10.2f} {avg_ffdnet_psnr:>15.2f}")
    
    print("=" * 80)
    
    # Save summary to file
    results_dir = current_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / "summary.md", "w") as f:
        f.write("# DIP-FFDNet Pipeline Comparison Results\n\n")
        f.write("This file contains the results of running the DIP-FFDNet pipeline on test images.\n\n")
        f.write("## Denoising Methods Compared\n\n")
        f.write("1. **DIP**: Deep Image Prior with SURE loss (default DIP method)\n")
        f.write("2. **FFDNet**: FFDNet with noise maps from DIP\n\n")
        f.write("## Results\n\n")
        f.write(f"| {'Image':<20} | {'DIP PSNR':>10} | {'FFDNet PSNR':>15} |\n")
        f.write("|" + "-" * 20 + "|" + "-" * 12 + "|" + "-" * 17 + "|\n")
        
        for r in results:
            f.write(f"| {r['image']:<20} | {r['dip_psnr']:>10.2f} | {r['ffdnet_psnr']:>15.2f} |\n")
        
        f.write("|" + "-" * 20 + "|" + "-" * 12 + "|" + "-" * 17 + "|\n")
        f.write(f"| {'Average':<20} | {avg_dip_psnr:>10.2f} | {avg_ffdnet_psnr:>15.2f} |\n\n")
        
        # Add analysis section
        f.write("## Analysis\n\n")
        if avg_ffdnet_psnr > avg_dip_psnr:
            f.write("FFDNet with spatial noise maps from DIP provides better results than DIP alone.\n")
        else:
            f.write("DIP with SURE loss provides better results than FFDNet with spatial noise maps.\n")


if __name__ == "__main__":
    main()
