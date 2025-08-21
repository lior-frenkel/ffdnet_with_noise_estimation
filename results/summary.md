# DIP-FFDNet Pipeline Comparison Results

This file contains the results of running the DIP-FFDNet pipeline on test images.

## Denoising Methods Compared

1. **DIP**: Deep Image Prior with SURE loss (default DIP method)
2. **FFDNet**: FFDNet with noise maps from DIP
3. **FFDNet Oracle**: FFDNet with true noise level (σ=25)

## Results

| Image                |   DIP PSNR |     FFDNet PSNR |   FFDNet Oracle |    Avg Sigma |
|--------------------|------------|-----------------|-----------------|--------------|
| 1_Cameraman256.png   |      26.61 |           29.10 |           29.65 |         25.6 |
| 2_house.png          |      29.63 |           33.23 |           33.32 |         26.1 |
| 3_peppers256.png     |      26.22 |           30.23 |           30.60 |         26.9 |
| 4_Lena512.png        |      29.89 |           32.38 |           32.57 |         26.0 |
| 5_barbara.png        |      25.79 |           29.45 |           29.90 |         27.8 |
| 6_boat.png           |      27.83 |           29.88 |           30.12 |         26.6 |
| 7_hill.png           |      26.84 |           29.65 |           30.01 |         27.0 |
| 8_couple.png         |      26.30 |           29.77 |           30.09 |         26.9 |
|--------------------|------------|-----------------|-----------------|--------------|
| Average              |      27.39 |           30.46 |           30.78 |         26.6 |

## Analysis

FFDNet Oracle (true noise level) > FFDNet (estimated noise) > DIP with SURE loss.
This shows the importance of accurate noise estimation for FFDNet performance.
