# PCA-FFDNet Pipeline Comparison Results

This file contains the results of running the PCA-FFDNet pipeline on test images.

## Denoising Methods Compared

1. **PCA**: PCA denoising method
2. **FFDNet**: FFDNet with noise maps from PCA
3. **FFDNet Oracle**: FFDNet with true noise level (σ=25)

## Results

| Image                |     PCA PSNR |     FFDNet PSNR |   FFDNet Oracle |    Avg Sigma |
|--------------------|--------------|-----------------|-----------------|--------------|
| 1_Cameraman256.png   |        28.52 |           28.52 |           29.58 |         32.1 |
| 2_house.png          |        32.64 |           32.64 |           33.37 |         32.4 |
| 3_peppers256.png     |        29.41 |           29.41 |           30.78 |         33.4 |
| 4_Lena512.png        |        31.77 |           31.77 |           32.64 |         31.7 |
| 5_barbara.png        |        28.66 |           28.66 |           29.99 |         33.1 |
| 6_boat.png           |        29.07 |           29.07 |           30.15 |         32.1 |
| 7_hill.png           |        28.87 |           28.87 |           30.04 |         32.8 |
| 8_couple.png         |        28.82 |           28.82 |           30.06 |         32.8 |
|--------------------|--------------|-----------------|-----------------|--------------|
| Average              |        29.72 |           29.72 |           30.83 |         32.6 |

## Analysis

PCA provides better results than FFDNet with spatial noise maps.
