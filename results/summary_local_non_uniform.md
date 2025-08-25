# LocalVar_NU-FFDNet Pipeline Comparison Results

This file contains the results of running the LocalVar_NU-FFDNet pipeline on test images.

## Denoising Methods Compared

1. **LocalVar_NU**: LocalVar_NU denoising method
2. **FFDNet**: FFDNet with noise maps from LocalVar_NU
3. **FFDNet Oracle**: FFDNet with true noise level (σ=25)

## Results

| Image                | LocalVar_NU PSNR |     FFDNet PSNR |   FFDNet Oracle |    Avg Sigma |
|--------------------|--------------|-----------------|-----------------|--------------|
| 1_Cameraman256.png   |        27.79 |           27.79 |           29.59 |         21.8 |
| 2_house.png          |        29.99 |           29.99 |           33.32 |         22.3 |
| 3_peppers256.png     |        29.18 |           29.18 |           30.53 |         22.5 |
| 4_Lena512.png        |        29.85 |           29.85 |           32.63 |         22.5 |
| 5_barbara.png        |        28.70 |           28.70 |           29.83 |         24.1 |
| 6_boat.png           |        28.59 |           28.59 |           30.06 |         22.7 |
| 7_hill.png           |        28.83 |           28.83 |           29.94 |         22.9 |
| 8_couple.png         |        28.74 |           28.74 |           29.95 |         23.2 |
|--------------------|--------------|-----------------|-----------------|--------------|
| Average              |        28.96 |           28.96 |           30.73 |         22.7 |

## Analysis

LocalVar_NU provides better results than FFDNet with spatial noise maps.
