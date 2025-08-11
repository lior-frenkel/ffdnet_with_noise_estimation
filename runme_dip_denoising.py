"""
Modified from runme_dip_denoising.py to support JSON output
"""
import glob
import os
import json
from PIL import Image
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from matplotlib import pyplot as plt
import numpy as np
from deep_image_prior.models import *
from deep_image_prior.utils.denoising_utils import crop_image, get_image, pil_to_np, get_noisy_image, get_noise
from deep_image_prior.utils.denoising_utils import np_to_torch, torch_to_np, plot_image_grid, get_params, optimize

import torch
from tqdm import tqdm

from skimage.metrics import peak_signal_noise_ratio as compare_psnr

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

from pathlib import Path


imsize =-1
PLOT = False
sigma = 25
sigma_ = sigma/255.

INPUT = 'noise' # 'meshgrid'
pad = 'reflection'
OPT_OVER = 'net' # 'net,input'

loss_mode = 'mse'

reg_noise_std = 1./30. # set to 1./20. for sigma=50
LR = 0.01

OPTIMIZER='adam' # 'LBFGS'
show_every = 100
exp_weight=0.99


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss_mode", choices=["mse", "sure"], default="mse", help="objective to use")
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--img_path', type=str, default=None, help="Path to a specific image to process")
    parser.add_argument('--output_file', type=str, default=None, help="Path to save results as JSON")
    args = parser.parse_args()
    loss_mode = args.loss_mode

    if loss_mode == 'sure':
        print("running with SURE loss")
    else:
        print("running with MSE")

    script_dir = Path(__file__).resolve().parent 
    test_dir = script_dir / 'test_set' 
    
    # Process either a specific image or all images in test_dir
    if args.img_path:
        filenames = [args.img_path]
    else:
        filenames = glob.glob(f'{test_dir}/*.png')
    
    psnr_values = {}
    results = {}  # For JSON output
    
    for filename in tqdm(filenames):
        img_name = Path(filename).stem
        # Add synthetic noise
        img_pil = crop_image(get_image(filename, imsize)[0], d=32)
        img_np = pil_to_np(img_pil)
        
        img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)
    
        if loss_mode == "sure":
            input_depth = 1
            num_iter = 600
            lr = 0.001
            snap_iters = [0, 500, 1400]
        else:
            input_depth = 32
            num_iter = 3000
            lr = 0.01
            snap_iters = [0, 2000, 3900]

        figsize = 4 
        
        net = get_net(input_depth, 'skip', pad,
                    skip_n33d=128, 
                    skip_n33u=128, 
                    skip_n11=4, 
                    num_scales=5,
                    n_channels=1,
                    upsample_mode='bilinear').type(dtype)


        # Compute number of parameters
        s  = sum([np.prod(list(p.size())) for p in net.parameters()]); 
        print ('Number of params: %d' % s)

        # Loss
        mse = torch.nn.MSELoss().type(dtype)

        img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)

        if loss_mode == "sure":
            net_input = img_noisy_torch.clone().detach()
        else:
            net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()

        net_input_saved = net_input.detach().clone()
        noise = net_input.detach().clone()
        out_avg = None
        last_net = None
        psrn_noisy_last = 0

        i = 0
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
                total_loss = sure_loss(net, net_input, out, img_noisy_torch, sigma_, mse)
            else:
                total_loss = mse(out, img_noisy_torch)

            last_outputs = {
                "x_hat": torch_to_np(out.detach()),
                "target": img_noisy_np,         
                "x_gt": img_np, 
                "iter": i, 
            }
            
            total_loss.backward()
                
            psrn_noisy = compare_psnr(img_noisy_np, out.detach().cpu().numpy()[0]) 
            psrn_gt    = compare_psnr(img_np, out.detach().cpu().numpy()[0]) 
            psrn_gt_sm = compare_psnr(img_np, out_avg.detach().cpu().numpy()[0]) 
            
            # Note that we do not have GT for the "snail" example
            # So 'PSRN_gt', 'PSNR_gt_sm' make no sense
            print ('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % (i, total_loss.item(), psrn_noisy, psrn_gt, psrn_gt_sm), '\r', end='')
            if PLOT and i % show_every == 0:
                out_np = torch_to_np(out)
                plot_image_grid([np.clip(out_np, 0, 1), 
                                np.clip(torch_to_np(out_avg), 0, 1)],
                                  factor=figsize, nrow=1, img_name=img_name, i=i, save_dir=script_dir / 'results')
                
            # Backtracking
            if i % show_every:
                if psrn_noisy - psrn_noisy_last < -5: 
                    print('Falling back to previous checkpoint.')

                    for new_param, net_param in zip(last_net, net.parameters()):
                        net_param.data.copy_(new_param.cuda())

                    return total_loss*0, psrn_gt_sm, last_outputs
                else:
                    last_net = [x.detach().cpu() for x in net.parameters()]
                    psrn_noisy_last = psrn_noisy
                    
            i += 1

            return total_loss, psrn_gt_sm, last_outputs

        p = get_params(OPT_OVER, net, net_input)
        psnr_values_img = optimize(OPTIMIZER, p, closure, lr, num_iter, snap_iters=snap_iters, loss_mode=loss_mode, script_dir=script_dir, img_name=img_name)

        out_np = torch_to_np(net(net_input))
        
        # Save denoised output
        results_dir = script_dir / 'results'
        results_dir.mkdir(exist_ok=True)
        output_path = results_dir / f"{img_name}_denoised_{loss_mode}.png"
        
        # Convert to uint8 and save
        out_np_uint8 = np.clip(out_np.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
        Image.fromarray(out_np_uint8.squeeze()).save(output_path)
        
        # Calculate final PSNR
        final_psnr = compare_psnr(img_np.squeeze(), out_np.squeeze())
        
        # Store results for JSON output
        results[img_name] = {
            "denoised_path": str(output_path),
            "psnr": final_psnr,
            "loss_mode": loss_mode
        }
        
        q = plot_image_grid([img_np, np.clip(out_np, 0, 1)], factor=13, img_name=img_name, i=i, save_dir=script_dir / 'results', final=f'_final_{loss_mode}')

        psnr_values[img_name] = np.array(psnr_values_img)

    avg_psnr = np.stack([img_single for img_single in psnr_values.values()], axis=0).mean(0)

    plt.figure()
    plt.plot(range(num_iter), avg_psnr)
    plt.xlabel('Iteration')
    plt.ylabel('Avg PSNR')
    if loss_mode == "sure":
        plt.title('Avg PSNR over test images using DIP+SURE')
    plt.savefig(f'{script_dir}/results/avg_psnr_dip_{num_iter}_{loss_mode}_lr_{lr}_noise_std_{reg_noise_std}.png')
    plt.close()
    
    # If an output file is specified, save results to JSON
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results.get(img_name, results), f)


def sure_loss(net, net_in, out, noisy_torch, sigma_n, mse_loss):
    
    eps = torch.randn_like(noisy_torch)
    out_eps = net(net_in + sigma_n * eps)
    div = torch.mean((out_eps - out) * eps) / sigma_n
    loss = mse_loss(out, noisy_torch) - sigma_n ** 2 + 2 * sigma_n ** 2 * div

    return loss

if __name__ == '__main__':
    main()
