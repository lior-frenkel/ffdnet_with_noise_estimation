# DIP-FFDNet Pipeline Comparison Results

This file contains the results of running the DIP-FFDNet pipeline on test images.

## Denoising Methods Compared

1. **DIP**: Deep Image Prior with SURE loss (default DIP method)
2. **FFDNet**: FFDNet with noise maps from DIP

## Results

| Image                |   DIP PSNR |     FFDNet PSNR |
|--------------------|------------|-----------------|
| 1_Cameraman256.png   |      26.20 |           28.16 |
| 2_house.png          |      24.17 |           32.89 |
| 3_peppers256.png     |      25.94 |           29.51 |
| 4_Lena512.png        |      29.55 |           29.41 |
| 5_barbara.png        |      26.57 |           28.95 |
| 6_boat.png           |      26.93 |           28.83 |
| 7_hill.png           |      27.67 |           28.60 |
| 8_couple.png         |      26.84 |           28.78 |
|--------------------|------------|-----------------|
| Average              |      26.73 |           29.39 |

## Analysis

FFDNet with spatial noise maps from DIP provides better results than DIP alone.
