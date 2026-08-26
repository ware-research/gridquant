# GridQuant Help

## Overview

GridQuant is an image processing tool for detecting electrodes/circles in microscopy images and quantifying fluorescence intensity.

Supported formats:
- CZI microscopy files

---

# Workflow

## 1. Select Images

Choose either:
- A single image file
- A folder containing multiple images

GridQuant will automatically detect the CZIs and available channels.

---

## 2. Select a Preset

Presets contain the parameters required for different chip designs and imaging conditions.

Examples:
- d1_kinetix
- b3_axio
- c5_kinetix

A preset controls:
- Circle size limits
- ROI size
- Background size
- Fluorescence processing settings

Preset Structure:
- Name
- "is_arrray" An array of electrodes or a single electrode
- "min_diameter" The minimum diameter of the electrode for detection
- "max_diameter" The maximum diameter of the elecrode for detection
- "roi_inner" The inner circle of the region of interest for analysis
- "roi_outer" The outer circle of the region of interest for analysis
- "bckg_inner" The inner circle of the background for analysis
- "bckg_outer" The outer circle of the background for analysis
- "moving_avg_n" 
- "p1" cv2 edge detection parameter (Less Strict ---> More Strict)
- "p2" cv2 edge voting parameter (Less Strict ---> More Strict)
- "dp" Resolution of the image used for the cv2 circle detection (1 = full resolution, 2 = half the resolution)

---

# Processing Options

## Save Results

Exports fluorescence measurements and metadata after processing.

The output contains:
- Fluorescence measurements
- GridQuant version
- Preset used
- Processing parameters

---

## Debug

Enables additional diagnostic output.

Useful for:
- Troubleshooting circle detection
- Checking image dimensions
- Inspecting intermediate calculations

---

## Show Final Registration

Displays detected electrode locations over the brightfield image.

Used to verify:
- Correct circle detection
- Correct registration
- Missing/incorrect electrodes

*** Use this to verify that the correct electrodes were identified and the ROI and Background are correct. 

---

## Show IF Images

Displays fluorescence images with:
- ROI boundaries
- Background subtraction regions

Useful for verifying fluorescence quantification.

---

## Show Plots

Displays analysis plots including:
- Fluorescence intensity plots
- Single electrode summaries
- Outlier detection plots

---

# Preset Parameters

## Circle Detection

### Minimum Diameter

Smallest circle diameter accepted during detection.

### Maximum Diameter

Largest circle diameter accepted during detection.

---

## Fluorescence ROI

### ROI Inner

Controls the inner boundary of fluorescence measurement.

Negative values shrink the measurement area inside the detected circle.

### ROI Outer

Controls the outer boundary of fluorescence measurement.

---

## Background Region

### Background Inner

Defines where background measurement begins outside the electrode.

### Background Outer

Defines where background measurement ends.

Background intensity is subtracted from the measured fluorescence.

---

# Array Mode

Array mode is used when multiple electrodes are present.

Processing steps:

1. Detect circles
2. Remove outliers
3. Fill missing electrodes
4. Organize grid
5. Snake electrode indexing
6. Quantify fluorescence

---

# Single Electrode Mode

Single electrode mode is used when only one electrode is present.

Processing steps:

1. Detect one circle
2. Generate ROI
3. Measure fluorescence
4. Subtract background

---

# Troubleshooting

## No circles detected

Check:
- Correct preset selected
- Diameter limits
- Image quality
- Brightfield channel

---

## Fluorescence values appear incorrect

Check:
- ROI settings
- Background settings
- Correct fluorescence channel selected

---

## Missing electrodes

Try:
- Adjusting detection parameters
- Checking final registration plot
- Using a different preset
