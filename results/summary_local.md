# LocalVar-FFDNet Pipeline Comparison Results

This file contains the results of running the LocalVar-FFDNet pipeline on test images.

## Denoising Methods Compared

1. **LocalVar**: LocalVar denoising method
2. **FFDNet**: FFDNet with noise maps from LocalVar
3. **FFDNet Oracle**: FFDNet with true noise level (σ=25)

## Results

| Image                | LocalVar PSNR |     FFDNet PSNR |   FFDNet Oracle |    Avg Sigma |
|--------------------|--------------|-----------------|-----------------|--------------|
| 1_Cameraman256.png   |        25.78 |           25.78 |           29.58 |         17.7 |
| 2_house.png          |        30.07 |           30.07 |           33.37 |         21.3 |
| 3_peppers256.png     |        29.23 |           29.23 |           30.78 |         21.5 |
| 4_Lena512.png        |        29.88 |           29.88 |           32.64 |         21.3 |
| 5_barbara.png        |        28.58 |           28.58 |           29.99 |         21.4 |
| 6_boat.png           |        28.35 |           28.35 |           30.15 |         21.2 |
| 7_hill.png           |        28.35 |           28.35 |           30.04 |         21.3 |
| 8_couple.png         |        28.78 |           28.78 |           30.06 |         21.7 |
|--------------------|--------------|-----------------|-----------------|--------------|
| Average              |        28.63 |           28.63 |           30.83 |         20.9 |

## Analysis

LocalVar provides better results than FFDNet with spatial noise maps.
