# DIP-FFDNet Pipeline Comparison Results

This file contains the results of running the DIP-FFDNet pipeline on test images.

## Denoising Methods Compared

1. **DIP**: Deep Image Prior with SURE loss (default DIP method)
2. **FFDNet**: FFDNet with noise maps from DIP

## Results

| Image                |   DIP PSNR |     FFDNet PSNR |    Avg Sigma |
|--------------------|------------|-----------------|--------------|
| 1_Cameraman256.png   |      26.83 |           29.56 |         26.0 |
| 2_house.png          |      28.87 |           33.22 |         26.5 |
| 3_peppers256.png     |      26.64 |           30.36 |         27.2 |
| 4_Lena512.png        |      30.11 |           32.57 |         26.1 |
| 5_barbara.png        |      26.86 |           29.81 |         27.2 |
| 6_boat.png           |      27.59 |           29.99 |         26.8 |
| 7_hill.png           |      27.92 |           29.81 |         26.6 |
| 8_couple.png         |      26.70 |           29.81 |         27.3 |
|--------------------|------------|-----------------|--------------|
| Average              |      27.69 |           30.64 |         26.7 |

## Analysis

FFDNet with spatial noise maps from DIP provides better results than DIP alone.
