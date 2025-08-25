# FFDNet Denoising Pipeline with Multiple Noise Estimation Methods

This project implements a comprehensive denoising framework that evaluates different noise estimation techniques with FFDNet. The pipeline supports multiple approaches:

1. **Pre-denoising methods**: Deep Image Prior (DIP) and Noise2Self
2. **Direct noise estimation**: Local variance, PCA, and patch similarity methods  
3. **Spatially-varying noise**: Support for non-uniform noise scenarios
4. **Performance comparison**: Oracle vs. estimated noise maps

## Directory Structure

- `run_pipeline.py`: Main pipeline supporting all denoising methods
- `ffdnet_adapter.py`: Interface to FFDNet model with spatial noise map support
- `runme_dip_denoising.py`: Original DIP implementation
- `deep_image_prior/`: DIP utilities and models
- `models/`: FFDNet network architecture
- `model_zoo/`: Pre-trained FFDNet weights
- `test_set/`: Test images
- `results/`: Output directory for results and summaries

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run with different methods:
   ```bash
   # Pre-denoising approaches
   python run_pipeline.py dip          # Deep Image Prior
   python run_pipeline.py noise2self   # Noise2Self algorithm
   
   # Direct noise estimation
   python run_pipeline.py local        # Local variance method
   python run_pipeline.py pca          # PCA-based estimation
   python run_pipeline.py patch        # Patch similarity method
   
   # Non-uniform noise scenarios
   python run_pipeline.py local_non_uniform  # Spatially-varying local estimation
   python run_pipeline.py pca_non_uniform    # Spatially-varying PCA estimation
   ```

## Methods Overview

### Pre-denoising Approaches
- **DIP**: Uses SURE loss optimization for self-supervised denoising
- **Noise2Self**: Masked training approach for noise removal

### Direct Noise Estimation  
- **Local Variance**: Estimates noise from patch statistics in smooth regions
- **PCA**: Uses principal component analysis on similar patch groups
- **Patch Similarity**: Compares similar patches to isolate noise components

### Spatially-Varying Methods
- **Non-uniform scenarios**: Handles varying noise levels across the image
- **Block-based estimation**: Overlapping windows with Gaussian weighting
- **Grid-based estimation**: Sparse sampling with interpolation

## Output

The pipeline generates:
- Comparison figures showing clean, noisy, denoised, and noise map visualizations
- Performance summaries with PSNR values for all methods
- Oracle comparisons using true noise parameters
- Method-specific result files and analysis

## References

- FFDNet: [Toward a Fast and Flexible Solution for CNN based Image Denoising](https://arxiv.org/abs/1710.04026)
- Deep Image Prior: [Deep Image Prior](https://arxiv.org/abs/1711.10925)
- Noise2Self: [Noise2Self: Blind Denoising by Self-Supervision](https://arxiv.org/abs/1901.11365)
- Deep Image Prior: [Deep Image Prior](https://arxiv.org/abs/1711.10925)
