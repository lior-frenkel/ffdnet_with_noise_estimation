# DIP-FFDNet Pipeline Comparison Results

This file contains the results of running the DIP-FFDNet pipeline on test images.

## Denoising Methods Compared

1. **DIP**: Deep Image Prior with SURE loss (default DIP method)
2. **FFDNet**: FFDNet with noise maps from DIP

## Results

| Image                |   DIP PSNR |     FFDNet PSNR |    Avg Sigma |
|--------------------|------------|-----------------|--------------|
| 1_Cameraman256.png   |      27.13 |           27.91 |          6.1 |
| 2_house.png          |      26.29 |           31.46 |          6.5 |
| 3_peppers256.png     |      24.69 |           30.22 |          6.7 |
| 4_Lena512.png        |      29.89 |           29.31 |          6.1 |
| 5_barbara.png        |      26.80 |           28.83 |          6.4 |
| 6_boat.png           |      26.91 |           28.78 |          6.4 |
| 7_hill.png           |      27.48 |           28.63 |          6.3 |
| 8_couple.png         |      26.52 |           28.91 |          6.4 |
|--------------------|------------|-----------------|--------------|
| Average              |      26.96 |           29.26 |          6.4 |

## Analysis

FFDNet with spatial noise maps from DIP provides better results than DIP alone.
