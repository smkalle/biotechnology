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

Usage
bash
Copy
Edit
python imaging_processing.py \
  --input path/to/stack.tif \
  --sigma 1.0 \
  --threshold-z 2.0 \
  --output intensities.csv \
  --plot intensity_plot.png
Command-Line Arguments
Flag	Description	Default
--input, -i	Path to input TIFF stack	(required)
--sigma, -s	Gaussian smoothing σ	1.0
--threshold-z, -z	Threshold = mean + z * std for segmentation	2.0
--output, -o	Path to output CSV file	intensities.csv
--plot, -p	Path to save intensity vs. time plot (PNG)	(no plot saved)

Workflow
Load
Read TIFF stack into a (t, y, x) NumPy array.

Preprocess
Apply Gaussian filter frame-by-frame to reduce noise.

Segment
Threshold each frame at mean + z_thresh * std and label connected regions.

Measure
For each time point, compute the mean intensity of the largest labeled region (fallback to global mean if none).

Export & Plot
Save time-intensity data to CSV and generate a publication-ready plot.

Example
bash
Copy
Edit
python imaging_processing.py \
  --input sample_data/voltage_stack.tif \
  --sigma 1.5 \
  --threshold-z 3.0 \
  --output results/voltage_intensity.csv \
  --plot results/voltage_intensity.png
Extending the Pipeline
Replace simple thresholding with watershed or active contours for robust ROI segmentation.

Batch-process multiple stacks by looping over a directory of input files.

Integrate with Jupyter notebooks for interactive exploration.

License
Released under the MIT License.
Ensure compliance with any dataset licenses before redistribution.
