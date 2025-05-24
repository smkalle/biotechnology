# Voltage Imaging Processing

A Python application (`imaging_processing.py`) for analyzing voltage-sensitive dye imaging data.

## Features

- **Load** multi-page TIFF stacks
- **Noise reduction** via Gaussian smoothing
- **Segmentation** using adaptive thresholding and connected-component labeling
- **Intensity measurement** of the largest ROI per frame
- **Export** results to CSV
- **Plot** intensity vs. time and save as PNG

## Installation

```bash
pip install numpy scikit-image matplotlib tifffile
