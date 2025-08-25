# PatchSim-FFDNet Pipeline Comparison Results

This file contains the results of running the PatchSim-FFDNet pipeline on test images.

## Denoising Methods Compared

1. **PatchSim**: PatchSim denoising method
2. **FFDNet**: FFDNet with noise maps from PatchSim
3. **FFDNet Oracle**: FFDNet with true noise level (σ=25)

## Results

| Image                | PatchSim PSNR |     FFDNet PSNR |   FFDNet Oracle |    Avg Sigma |
|--------------------|--------------|-----------------|-----------------|--------------|
| 1_Cameraman256.png   |        26.12 |           26.12 |           29.58 |         18.2 |
| 2_house.png          |        25.97 |           25.97 |           33.37 |         17.9 |
| 3_peppers256.png     |        27.65 |           27.65 |           30.78 |         20.0 |
| 4_Lena512.png        |        25.83 |           25.83 |           32.64 |         17.8 |
| 5_barbara.png        |        26.85 |           26.85 |           29.99 |         19.4 |
| 6_boat.png           |        25.54 |           25.54 |           30.15 |         17.9 |
| 7_hill.png           |        24.32 |           24.32 |           30.04 |         16.2 |
| 8_couple.png         |        25.57 |           25.57 |           30.06 |         18.1 |
|--------------------|--------------|-----------------|-----------------|--------------|
| Average              |        25.98 |           25.98 |           30.83 |         18.2 |

## Analysis

PatchSim provides better results than FFDNet with spatial noise maps.
