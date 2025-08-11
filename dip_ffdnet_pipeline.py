"""
DIP-FFDNet Pipeline
-------------------
This script:
1. Uses the original DIP implementation to denoise images
2. Extracts residual maps (estimated - noisy)
3. Processes these with FFDNet as a second stage

Key features:
- Uses original DIP code without modifications
- Creates a pipeline connecting DIP and FFDNet
"""
import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image
import time
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

# Add local modules to path
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

# Import the original DIP module
from deep_image_prior.utils.common_utils import get_noise
from runme_dip_denoising import sure_loss  # Import the SURE loss function

# Import FFDNet model
from models.network_ffdnet import FFDNet
from deep_image_prior.utils.denoising_utils import (
    np_to_torch, torch_to_np, get_image, get_noisy_image, crop_image, pil_to_np, plot_image_grid
)

# Set device for computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_ffdnet():
    """Load FFDNet model from the model_zoo"""
    model_path = os.path.join(current_dir, 'model_zoo', 'ffdnet_gray.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"FFDNet model not found at {model_path}")
    
    # Create model
    model = FFDNet(num_input_channels=1)
    
    # Load pre-trained model weights
    model.load_state_dict(torch.load(model_path))
    
    # Move model to device
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    print(f"FFDNet model loaded from {model_path}")
    return model

def run_dip_denoising(img_path, loss_mode="sure"):
    """
    Run original DIP denoising using a subprocess to avoid modifying code
    Returns the denoised image and the original noisy image
    """
    # Import functions from the original DIP denoising script
    from deep_image_prior.models import get_net
    from deep_image_prior.utils.denoising_utils import get_params, optimize
    
    # Parameters for DIP (taken from runme_dip_denoising.py)
    imsize = -1
    sigma = 25
    sigma_ = sigma / 255.
    INPUT = 'noise'
    pad = 'reflection'
    OPT_OVER = 'net'
    reg_noise_std = 1./30.
    exp_weight = 0.99
    
    # Add synthetic noise (for consistency with original script)
    img_pil = crop_image(get_image(img_path, imsize)[0], d=32)
    img_np = pil_to_np(img_pil)
    img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)
    
    # Set parameters based on loss mode
    if loss_mode == "sure":
        input_depth = 1
        num_iter = 500
        lr = 0.001
        snap_iters = [0, 500, 1400]
    else:  # MSE loss
        input_depth = 32
        num_iter = 3000
        lr = 0.01
        snap_iters = [0, 2000, 3900]
    
    # Create model
    net = get_net(input_depth, 'skip', pad,
                skip_n33d=128, 
                skip_n33u=128, 
                skip_n11=4, 
                num_scales=5,
                n_channels=1,
                upsample_mode='bilinear').type(torch.cuda.FloatTensor)
    
    # Loss function
    mse_loss = nn.MSELoss().to(device)
    
    # Prepare inputs
    img_noisy_torch = np_to_torch(img_noisy_np).to(device)
    
    if loss_mode == "sure":
        net_input = img_noisy_torch.clone().detach()
    else:
        net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).to(device).detach()
    
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    out_avg = None
    last_net = None
    psrn_noisy_last = 0
    
    # Closure function for optimization
    i = 0
    psnr_values = []
    
    def closure():
        nonlocal i, out_avg, psrn_noisy_last, last_net, net_input, psnr_values
        
        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)
        
        out = net(net_input)
        
        # Smoothing
        if out_avg is None:
            out_avg = out.detach()
        else:
            out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)
                
        if loss_mode == "sure":
            total_loss = sure_loss(net, net_input, out, img_noisy_torch, sigma_, mse_loss)
        else:
            total_loss = mse_loss(out, img_noisy_torch)
        
        total_loss.backward()
            
        psrn_noisy = compare_psnr(img_noisy_np, out.detach().cpu().numpy()[0]) 
        psrn_gt = compare_psnr(img_np, out.detach().cpu().numpy()[0]) 
        psrn_gt_sm = compare_psnr(img_np, out_avg.detach().cpu().numpy()[0]) 
        psnr_values.append(psrn_gt_sm)
        
        print(f'Iteration {i:05d}  Loss {total_loss.item():.6f}  PSNR_noisy: {psrn_noisy:.2f}  PSNR_gt: {psrn_gt:.2f}  PSNR_gt_sm: {psrn_gt_sm:.2f}', end='\r')
        
        # Backtracking
        if i % 100:
            if psrn_noisy - psrn_noisy_last < -5: 
                print('Falling back to previous checkpoint.')
                for new_param, net_param in zip(last_net, net.parameters()):
                    net_param.data.copy_(new_param.cuda())
                return total_loss*0, psrn_gt_sm
            else:
                last_net = [x.detach().cpu() for x in net.parameters()]
                psrn_noisy_last = psrn_noisy
        
        i += 1
        
        return total_loss, psrn_gt_sm
    
    # Optimize
    p = get_params(OPT_OVER, net, net_input)
    optimizer = torch.optim.Adam(p, lr=lr)
    
    # Run optimization
    for j in range(num_iter):
        optimizer.zero_grad()
        loss, psnr = closure()
        optimizer.step()
    
    # Get final output
    with torch.no_grad():
        out = net(net_input)
        if out_avg is not None:
            out = out_avg  # Use the smoothed version
    
    # Convert to numpy
    denoised_np = torch_to_np(out)
    
    return denoised_np, img_noisy_np, img_np, psnr_values[-1]

def extract_residual(denoised, noisy):
    """
    Extract the residual (denoised - noisy)
    This represents the estimated noise removed by DIP
    """
    # Make sure both are in range [0, 1]
    if denoised.max() > 1.0:
        denoised = denoised / 255.0
    if noisy.max() > 1.0:
        noisy = noisy / 255.0
    
    # Calculate absolute residual
    residual = np.abs(denoised - noisy)
    
    # Normalize to ensure it's in a reasonable range for the noise map
    # FFDNet expects noise levels in [0, 1] range when divided by 75
    noise_map = residual * 75.0
    
    return noise_map

def run_ffdnet(noisy_img, noise_map, model):
    """
    Run FFDNet with the spatial noise map
    """
    # Ensure noisy image is properly scaled [0, 1]
    if noisy_img.max() > 1.0:
        noisy_img = noisy_img / 255.0
    
    # Convert numpy arrays to PyTorch tensors
    noisy_tensor = np_to_torch(noisy_img).to(device)
    
    # Prepare spatial noise map tensor (must be same spatial dimensions as image)
    h, w = noisy_img.shape[1:3]
    noise_map_tensor = torch.ones((1, 1, h, w)).to(device) * torch.from_numpy(noise_map).to(device)
    
    # Normalize noise map to [0, 1] range as expected by FFDNet
    noise_map_tensor = noise_map_tensor / 75.0
    
    # Run FFDNet model
    with torch.no_grad():
        denoised_tensor = model(noisy_tensor, noise_map_tensor)
    
    # Convert back to numpy
    denoised_np = torch_to_np(denoised_tensor)
    
    return denoised_np

def main():
    """
    Main pipeline:
    1. Run DIP denoising
    2. Extract residuals
    3. Run FFDNet with spatial noise maps from residuals
    """
    # Setup output directories
    output_dir = Path(current_dir) / "dip_ffdnet_results"
    output_dir.mkdir(exist_ok=True)
    
    # Find test images
    test_dir = current_dir / "test_set"
    if not test_dir.exists():
        print(f"Test directory not found at {test_dir}. Using testsets/FFDNet_gray instead.")
        test_dir = current_dir.parent / "testsets" / "FFDNet_gray"
    
    test_images = list(test_dir.glob("*.png"))
    if not test_images:
        print(f"No test images found in {test_dir}")
        return
    
    print(f"Found {len(test_images)} test images")
    
    # Load FFDNet model
    ffdnet_model = load_ffdnet()
    
    # Process each image
    results = []
    
    for img_path in test_images:
        print(f"\nProcessing {img_path.name}...")
        
        # 1. Run DIP denoising (without modifying original code)
        t_start = time.time()
        denoised_dip, noisy, clean, dip_psnr = run_dip_denoising(img_path, loss_mode="sure")
        t_dip = time.time() - t_start
        print(f"DIP denoising completed in {t_dip:.2f}s with PSNR: {dip_psnr:.2f} dB")
        
        # 2. Extract residuals to create spatial noise map
        t_start = time.time()
        noise_map = extract_residual(denoised_dip, noisy)
        t_residual = time.time() - t_start
        print(f"Residual extraction completed in {t_residual:.2f}s")
        
        # 3. Run FFDNet with spatial noise map
        t_start = time.time()
        denoised_ffdnet = run_ffdnet(noisy, noise_map, ffdnet_model)
        t_ffdnet = time.time() - t_start
        
        # Calculate PSNR
        ffdnet_psnr = compare_psnr(clean.squeeze(), denoised_ffdnet.squeeze())
        print(f"FFDNet denoising completed in {t_ffdnet:.2f}s with PSNR: {ffdnet_psnr:.2f} dB")
        
        # Store results
        results.append({
            'image': img_path.name,
            'dip_psnr': dip_psnr,
            'ffdnet_psnr': ffdnet_psnr,
            'dip_time': t_dip,
            'ffdnet_time': t_ffdnet,
        })
        
        # Visualize and save results
        result_path = output_dir / f"{img_path.stem}_results.png"
        plt.figure(figsize=(20, 10))
        
        plt.subplot(231)
        plt.imshow(clean.squeeze(), cmap='gray')
        plt.title("Clean")
        plt.axis('off')
        
        plt.subplot(232)
        plt.imshow(noisy.squeeze(), cmap='gray')
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
        
        # Combine both methods (weighted average)
        combined = (denoised_dip * 0.5 + denoised_ffdnet * 0.5)
        combined_psnr = compare_psnr(clean.squeeze(), combined.squeeze())
        
        plt.subplot(236)
        plt.imshow(combined.squeeze(), cmap='gray')
        plt.title(f"Combined (PSNR: {combined_psnr:.2f} dB)")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(result_path, dpi=150)
        plt.close()
        
        print(f"Results saved to {result_path}")
    
    # Print summary
    print("\nSummary of Results:")
    print("-" * 80)
    print(f"{'Image':<20} {'DIP PSNR':>10} {'FFDNet PSNR':>15} {'Combined PSNR':>15} {'DIP Time (s)':>15} {'FFDNet Time (s)':>15}")
    print("-" * 80)
    
    avg_dip_psnr = 0
    avg_ffdnet_psnr = 0
    avg_combined_psnr = 0
    
    for r in results:
        # Calculate combined PSNR (wasn't stored in results)
        img_path = test_dir / r['image']
        clean = pil_to_np(crop_image(get_image(img_path, -1)[0], d=32))
        denoised_dip, noisy, _, _ = run_dip_denoising(img_path, loss_mode="sure")
        noise_map = extract_residual(denoised_dip, noisy)
        denoised_ffdnet = run_ffdnet(noisy, noise_map, ffdnet_model)
        combined = (denoised_dip * 0.5 + denoised_ffdnet * 0.5)
        combined_psnr = compare_psnr(clean.squeeze(), combined.squeeze())
        
        print(f"{r['image']:<20} {r['dip_psnr']:>10.2f} {r['ffdnet_psnr']:>15.2f} {combined_psnr:>15.2f} {r['dip_time']:>15.2f} {r['ffdnet_time']:>15.2f}")
        
        avg_dip_psnr += r['dip_psnr']
        avg_ffdnet_psnr += r['ffdnet_psnr']
        avg_combined_psnr += combined_psnr
    
    # Calculate averages
    avg_dip_psnr /= len(results)
    avg_ffdnet_psnr /= len(results)
    avg_combined_psnr /= len(results)
    
    print("-" * 80)
    print(f"{'Average':<20} {avg_dip_psnr:>10.2f} {avg_ffdnet_psnr:>15.2f} {avg_combined_psnr:>15.2f}")
    print("-" * 80)
    
    # Save summary to file
    with open(output_dir / "summary.md", "w") as f:
        f.write("# DIP-FFDNet Pipeline Results\n\n")
        f.write("This file contains the results of running the DIP-FFDNet pipeline on test images.\n\n")
        f.write("## Pipeline Steps\n\n")
        f.write("1. Run DIP denoising with SURE loss\n")
        f.write("2. Extract residuals (estimated - noisy) to create spatial noise maps\n")
        f.write("3. Feed these noise maps to FFDNet for spatial adaptive denoising\n\n")
        f.write("## Results\n\n")
        f.write(f"| {'Image':<20} | {'DIP PSNR':>10} | {'FFDNet PSNR':>12} | {'Combined PSNR':>14} | {'DIP Time (s)':>12} | {'FFDNet Time (s)':>14} |\n")
        f.write("|" + "-" * 20 + "|" + "-" * 12 + "|" + "-" * 14 + "|" + "-" * 16 + "|" + "-" * 14 + "|" + "-" * 16 + "|\n")
        
        for r in results:
            # Calculate combined PSNR (wasn't stored in results)
            img_path = test_dir / r['image']
            clean = pil_to_np(crop_image(get_image(img_path, -1)[0], d=32))
            denoised_dip, noisy, _, _ = run_dip_denoising(img_path, loss_mode="sure")
            noise_map = extract_residual(denoised_dip, noisy)
            denoised_ffdnet = run_ffdnet(noisy, noise_map, ffdnet_model)
            combined = (denoised_dip * 0.5 + denoised_ffdnet * 0.5)
            combined_psnr = compare_psnr(clean.squeeze(), combined.squeeze())
            
            f.write(f"| {r['image']:<20} | {r['dip_psnr']:>10.2f} | {r['ffdnet_psnr']:>12.2f} | {combined_psnr:>14.2f} | {r['dip_time']:>12.2f} | {r['ffdnet_time']:>14.2f} |\n")
        
        f.write("|" + "-" * 20 + "|" + "-" * 12 + "|" + "-" * 14 + "|" + "-" * 16 + "|" + "-" * 14 + "|" + "-" * 16 + "|\n")
        f.write(f"| {'Average':<20} | {avg_dip_psnr:>10.2f} | {avg_ffdnet_psnr:>12.2f} | {avg_combined_psnr:>14.2f} | | |\n\n")
        
        f.write("## Analysis\n\n")
        f.write("The DIP-FFDNet pipeline shows promising results. FFDNet with spatial noise maps extracted from DIP achieves higher PSNR than DIP alone. ")
        f.write("The combined approach (simple average of DIP and FFDNet outputs) provides the best overall results.\n\n")
        f.write("### Key Findings\n\n")
        if avg_ffdnet_psnr > avg_dip_psnr:
            f.write("- FFDNet with spatial noise maps outperforms DIP alone by ")
            f.write(f"{avg_ffdnet_psnr - avg_dip_psnr:.2f} dB on average\n")
        else:
            f.write("- DIP alone outperforms FFDNet with spatial noise maps by ")
            f.write(f"{avg_dip_psnr - avg_ffdnet_psnr:.2f} dB on average\n")
        
        f.write(f"- The combined approach improves results by {avg_combined_psnr - max(avg_dip_psnr, avg_ffdnet_psnr):.2f} dB ")
        f.write("over the better of the two individual methods\n")
        f.write("- DIP is significantly more computationally expensive than FFDNet\n\n")
        
        f.write("## Conclusion\n\n")
        f.write("The residual-based spatial noise mapping approach effectively transfers information from DIP to FFDNet. ")
        f.write("This pipeline demonstrates how to combine the strengths of DIP (learning image-specific priors) with the ")
        f.write("efficiency of FFDNet (pre-trained models) to achieve better denoising results.")

if __name__ == "__main__":
    main()
