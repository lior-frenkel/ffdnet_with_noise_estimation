# Noise2Self-FFDNet Pipeline Comparison Results

This file contains the results of running the Noise2Self-FFDNet pipeline on test images.

## Denoising Methods Compared

1. **Noise2Self**: Noise2Self denoising method
2. **FFDNet**: FFDNet with noise maps from Noise2Self
3. **FFDNet Oracle**: FFDNet with true noise level (σ=25)

## Results

| Image                | Noise2Self PSNR |     FFDNet PSNR |   FFDNet Oracle |    Avg Sigma |
|--------------------|--------------|-----------------|-----------------|--------------|
| 1_Cameraman256.png   |        23.78 |           28.62 |           29.55 |         32.4 |
| 2_house.png          |        24.81 |           32.59 |           33.27 |         34.6 |
| 3_peppers256.png     |        23.71 |           29.11 |           30.63 |         34.9 |
| 4_Lena512.png        |        23.67 |           30.92 |           32.59 |         36.9 |
| 5_barbara.png        |        24.08 |           28.22 |           29.93 |         33.8 |
| 6_boat.png           |        23.71 |           28.62 |           30.16 |         35.3 |
| 7_hill.png           |        23.83 |           28.35 |           30.05 |         35.6 |
| 8_couple.png         |        23.62 |           28.38 |           30.03 |         35.2 |
|--------------------|--------------|-----------------|-----------------|--------------|
| Average              |        23.90 |           29.35 |           30.78 |         34.8 |

## Analysis

FFDNet Oracle (true noise level) > FFDNet (estimated noise) > Noise2Self.
This shows the importance of accurate noise estimation for FFDNet performance.
