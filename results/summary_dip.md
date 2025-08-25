# DIP-FFDNet Pipeline Comparison Results

This file contains the results of running the DIP-FFDNet pipeline on test images.

## Denoising Methods Compared

1. **DIP**: DIP denoising method
2. **FFDNet**: FFDNet with noise maps from DIP
3. **FFDNet Oracle**: FFDNet with true noise level (σ=25)

## Results

| Image                |     DIP PSNR |     FFDNet PSNR |   FFDNet Oracle |    Avg Sigma |
|--------------------|--------------|-----------------|-----------------|--------------|
| 1_Cameraman256.png   |        26.64 |           29.13 |           29.58 |         25.8 |
| 2_house.png          |        28.69 |           33.27 |           33.37 |         26.0 |
| 3_peppers256.png     |        26.81 |           30.43 |           30.78 |         26.8 |
| 4_Lena512.png        |        27.80 |           32.43 |           32.64 |         26.5 |
| 5_barbara.png        |        26.52 |           29.67 |           29.99 |         27.3 |
| 6_boat.png           |        27.18 |           29.83 |           30.15 |         26.8 |
| 7_hill.png           |        27.60 |           29.72 |           30.04 |         27.0 |
| 8_couple.png         |        26.48 |           29.73 |           30.06 |         27.2 |
|--------------------|--------------|-----------------|-----------------|--------------|
| Average              |        27.22 |           30.53 |           30.83 |         26.7 |

## Analysis

FFDNet Oracle (true noise level) > FFDNet (estimated noise) > DIP.
This shows the importance of accurate noise estimation for FFDNet performance.
