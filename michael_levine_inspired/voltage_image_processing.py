#!/usr/bin/env python3
"""
imaging_processing.py

A complete workflow for:
  1. Loading an image sequence (TIFF stack)
  2. Noise reduction (Gaussian filtering)
  3. ROI segmentation (threshold + labeling)
  4. Intensity measurement over time
  5. Plotting and CSV export of intensity vs. time

Usage:
  python imaging_processing.py \
    --input images.tif \
    --sigma 1.0 \
    --threshold-z 2.0 \
    --output intensities.csv
"""

import argparse
import numpy as np
import tifffile
from skimage import filters, measure
import matplotlib.pyplot as plt
import csv
import os

def load_images(path: str) -> np.ndarray:
    """Load a multi-page TIFF as a (t, y, x) numpy array."""
    stack = tifffile.imread(path)
    if stack.ndim == 3:
        return stack
    # handle if stored as (y, x, t)
    return np.moveaxis(stack, -1, 0)

def preprocess(images: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian smoothing to each frame."""
    return np.stack([filters.gaussian(img, sigma=sigma) for img in images])

def segment(images: np.ndarray, z_thresh: float) -> list[np.ndarray]:
    """
    Segment each frame by thresholding at mean+z_thresh*std.
    Returns list of labeled masks.
    """
    labels = []
    for img in images:
        mu, sd = img.mean(), img.std()
        mask = img > (mu + z_thresh * sd)
        lbl = measure.label(mask)
        labels.append(lbl)
    return labels

def measure_intensities(orig: np.ndarray,
                        labels: list[np.ndarray]) -> list[float]:
    """
    For each timepoint, measure the mean intensity of the largest region;
    if no region, fallback to global mean.
    """
    intensities = []
    for img, lbl in zip(orig, labels):
        if lbl.max() == 0:
            intensities.append(img.mean())
        else:
            props = measure.regionprops(lbl, intensity_image=img)
            # pick region with largest area
            largest = max(props, key=lambda p: p.area)
            intensities.append(largest.mean_intensity)
    return intensities

def plot_intensity(times: list[int], intensities: list[float],
                   out_png: str = None):
    """Plot intensity vs. time and optionally save figure."""
    plt.figure(figsize=(6, 4))
    plt.plot(times, intensities, marker='o')
    plt.xlabel("Time Point")
    plt.ylabel("Mean Intensity (ROI)")
    plt.title("Voltage-Sensitive Dye Signal Over Time")
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=150)
        print(f"Saved plot to {out_png}")
    plt.show()

def save_csv(times: list[int], intensities: list[float], filepath: str):
    """Export time vs. intensity to CSV."""
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["time_point", "mean_intensity"])
        writer.writerows(zip(times, intensities))
    print(f"Results saved to {filepath}")

def main():
    p = argparse.ArgumentParser(description="Imaging Data Processing Pipeline")
    p.add_argument("--input",  "-i", required=True,
                   help="Path to TIFF stack")
    p.add_argument("--sigma",  "-s", type=float, default=1.0,
                   help="Gaussian smoothing sigma")
    p.add_argument("--threshold-z", "-z", type=float, default=2.0,
                   help="Threshold = mean + z*std")
    p.add_argument("--output", "-o", default="intensities.csv",
                   help="CSV file for intensity results")
    p.add_argument("--plot",   "-p", default=None,
                   help="Path to save intensity plot (PNG)")
    args = p.parse_args()

    # 1. Load
    imgs = load_images(args.input)
    print(f"Loaded {imgs.shape[0]} frames of size {imgs.shape[1]}×{imgs.shape[2]}")

    # 2. Preprocess
    filt = preprocess(imgs, sigma=args.sigma)
    print(f"Applied Gaussian filter (σ={args.sigma})")

    # 3. Segment
    labs = segment(filt, z_thresh=args.threshold_z)
    print(f"Segmented each frame using z-threshold = {args.threshold_z}")

    # 4. Measure
    times = list(range(len(imgs)))
    ints = measure_intensities(imgs, labs)
    print("Measured mean intensity for each time point")

    # 5. Save & Plot
    save_csv(times, ints, args.output)
    plot_intensity(times, ints, out_png=args.plot)

if __name__ == "__main__":
    main()
