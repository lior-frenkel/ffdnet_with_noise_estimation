# FFDNet Noise Estimation with DIP Residuals

This project combines Deep Image Prior (DIP) denoising with FFDNet to achieve better denoising results. The key idea is to:

1. Run the original DIP denoising
2. Extract residuals (estimated - noisy) to create spatial noise maps
3. Use these maps with FFDNet to perform spatially-adaptive denoising


## Directory Structure

- `runme_dip_denoising.py`: The original DIP implementation
- `run_pipeline.py`: The main pipeline code
- `ffdnet_adapter.py`: Clean interface to the FFDNet model
- `deep_image_prior/`: Original DIP code
- `models/`: FFDNet model implementation
- `model_zoo/`: Pre-trained FFDNet model
- `test_set/`: Test images
- `results/`: Output directory for results

## Usage

1. Make sure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the pipeline:
   ```bash
   # Run the pipeline
   python run_pipeline.py
   ```

## How It Works

1. **DIP Denoising**:
   - Uses the original implementation with / without SURE loss
   - Produces a denoised image and estimates the noise

2. **Residual Extraction**:
   - Computes the absolute difference between DIP output and noisy input
   - Scales appropriately for use as a noise map

3. **FFDNet with Spatial Noise Maps**:
   - Uses the pre-trained FFDNet model
   - Feeds the noise map to guide denoising strength across the image

## References

- FFDNet: [Toward a Fast and Flexible Solution for CNN based Image Denoising](https://arxiv.org/abs/1710.04026)
- Deep Image Prior: [Deep Image Prior](https://arxiv.org/abs/1711.10925)
