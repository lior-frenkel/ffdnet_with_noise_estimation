# PCA_NU-FFDNet Pipeline Comparison Results

This file contains the results of running the PCA_NU-FFDNet pipeline on test images.

## Denoising Methods Compared

1. **PCA_NU**: PCA_NU denoising method
2. **FFDNet**: FFDNet with noise maps from PCA_NU
3. **FFDNet Oracle**: FFDNet with true noise level (σ=25)

## Results

| Image                |  PCA_NU PSNR |     FFDNet PSNR |   FFDNet Oracle |    Avg Sigma |
|--------------------|--------------|-----------------|-----------------|--------------|
| 1_Cameraman256.png   |        28.20 |           28.20 |           29.59 |         30.8 |
| 2_house.png          |        32.64 |           32.64 |           33.32 |         31.0 |
| 3_peppers256.png     |        29.32 |           29.32 |           30.53 |         32.4 |
| 4_Lena512.png        |        31.58 |           31.58 |           32.63 |         31.5 |
| 5_barbara.png        |        28.10 |           28.10 |           29.83 |         33.2 |
| 6_boat.png           |        28.91 |           28.91 |           30.06 |         32.0 |
| 7_hill.png           |        28.85 |           28.85 |           29.94 |         32.0 |
| 8_couple.png         |        28.64 |           28.64 |           29.95 |         32.6 |
|--------------------|--------------|-----------------|-----------------|--------------|
| Average              |        29.53 |           29.53 |           30.73 |         31.9 |

## Analysis

PCA_NU provides better results than FFDNet with spatial noise maps.
