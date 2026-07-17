'''
MIT License

Copyright (c) 2024, Oregon Health & Science University

Contributor(s): Jason Ware (warej@ohsu.edu)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE OF ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Please consider citing!

'''

print("This is a sketchy unnofficial, vibe-coded, temporary release to enable TIFF prossessing while I build it into the main code. -J :)")
print("export and select folders exactly as you would with DFQ")
import cv2
import numpy as np
import tifffile
from tkinter import Tk
from tkinter.filedialog import askdirectory
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
import csv
import os
import re
from datetime import datetime

# ***********************************************************************************************************
# ********************************** User defined settings: *************************************************
# ***********************************************************************************************************
settings = "b3_kinetix_tiff"  # options: c3_kinetix, c2_kinetix, c5_axio, b3_kinetix

debug = True
show_final_registration = False
show_if_images = True
show_plots = False

# Don't edit this line:
global is_array, bitdepth, min_diameter, max_diameter, roi_inner, roi_outer, bckg_inner, bckg_outer, moving_avg_n, p1, p2, dp

# User preset profiles:
match settings.lower():
    case "c5_kinetix":
        print("loading preset for c5_kinetix")
        is_array = True
        min_diameter = 16
        max_diameter = 20
        roi_inner = -15
        roi_outer = -5
        bckg_inner = 15
        bckg_outer = 30
        moving_avg_n = 10
        p1 = 50
        p2 = 20
        dp = 1.0

    case "c2_kinetix":
        print("loading preset for c2_kinetix")
        is_array = True
        min_diameter = 20
        max_diameter = 30
        roi_inner = -20
        roi_outer = -7
        bckg_inner = 30
        bckg_outer = 50
        moving_avg_n = 10
        p1 = 50
        p2 = 20
        dp = 1.0

    case "c5_axio":
        print("loading preset for c5_axio")
        is_array = True
        min_diameter = 16
        max_diameter = 20
        roi_inner = -15
        roi_outer = -5
        bckg_inner = 25
        bckg_outer = 45
        moving_avg_n = 2
        p1 = 50
        p2 = 20
        dp = 1.0

    case "b3_kinetix":
        print("loading preset for b3_kinetix")
        is_array = False
        min_diameter = 30
        max_diameter = 40
        roi_inner = -8
        roi_outer = 0
        bckg_inner = 55
        bckg_outer = 100
        moving_avg_n = 2
        p1 = 50
        p2 = 20
        dp = 1.0

    case "b3_kinetix_tiff":
        print("loading preset for b3_kinetix_tiff")
        is_array = False
        min_diameter = 35
        max_diameter = 45
        roi_inner = -5
        roi_outer = 5
        bckg_inner = 80
        bckg_outer = 120
        moving_avg_n = 2
        p1 = 30
        p2 = 15
        dp = 1.0

# ***********************************************************************************************************
# ************** Do not edit below this line unless you know what you're doing ******************************
# ***********************************************************************************************************

bitdepth = 16384

# ---------------------------------------------------------------------------
# Channel detection helpers
# ---------------------------------------------------------------------------

# Ordered list of fluorescence channels (brightfield handled separately)
FLUOR_CHANNEL_ORDER = ["DAPI", "EGFP", "DsRed", "Cy5"]
BRIGHTFIELD_KEY = "RL_Brightfield"

# Regex patterns used to identify each channel from a filename (case-insensitive).
# DsRed accepts common variant spellings: dsred, ds_red, DS_Red, DsRed, dsRed …
CHANNEL_PATTERNS = {
    "RL_Brightfield": re.compile(r"RL[_\s]?Brightfield",       re.IGNORECASE),
    "DAPI":           re.compile(r"(?<![A-Za-z])DAPI(?![A-Za-z])",                   re.IGNORECASE),
    "EGFP":           re.compile(r"(?<![A-Za-z])EGFP(?![A-Za-z])",                   re.IGNORECASE),
    "DsRed":          re.compile(r"(?<![A-Za-z])Ds[_\s]?Red(?![A-Za-z])",            re.IGNORECASE),
    "Cy5":            re.compile(r"(?<![A-Za-z])Cy5(?![A-Za-z])",                    re.IGNORECASE),
}


def detect_channel_from_filename(filename):
    """
    Return the channel name embedded in *filename* (basename only, so that
    parent-directory names cannot produce false matches), or None if
    unrecognised.
    """
    basename = os.path.basename(filename)
    for channel_name, pattern in CHANNEL_PATTERNS.items():
        if pattern.search(basename):
            return channel_name
    return None


def load_tiff_as_gray_uint8(filepath):
    """Load a TIFF, collapse to 2-D grayscale, normalise to uint8."""
    img = tifffile.imread(filepath)
    # Collapse extra dimensions (Z-stacks, singleton axes, …)
    while img.ndim > 2:
        img = img[0]
    img = img.astype(np.float32)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img.astype(np.uint8)


def load_tiff_raw(filepath):
    """Load a TIFF and return the raw 2-D array (preserving bit-depth)."""
    img = tifffile.imread(filepath)
    while img.ndim > 2:
        img = img[0]
    return img


def load_collection(collection_dir):
    """
    Scan *collection_dir* for TIFF files and return:
        brightfield_path  – path to the RL Brightfield image (str or None)
        fluor_channels    – ordered list of (channel_name, filepath) for
                            fluorescence channels present in the folder,
                            in FLUOR_CHANNEL_ORDER sequence.

    Every file inspected is printed so that filename-matching failures are
    immediately visible.
    """
    tiff_extensions = {".tif", ".tiff"}
    channel_files = {}  # channel_name -> filepath

    all_files = sorted(os.listdir(collection_dir))
    tiff_files = [f for f in all_files
                  if os.path.splitext(f)[1].lower() in tiff_extensions]

    if not tiff_files:
        print(f"  [WARN] No TIFF files found in {collection_dir}.")
        return None, []

    print(f"  Found {len(tiff_files)} TIFF file(s) – scanning for channel names:")

    for fname in tiff_files:
        channel = detect_channel_from_filename(fname)
        fpath   = os.path.join(collection_dir, fname)

        if channel is None:
            print(f"    ✗  {fname}  →  unrecognised (skipping)")
            continue

        if channel in channel_files:
            print(f"    ⚠  {fname}  →  {channel}  (DUPLICATE – keeping "
                  f"'{os.path.basename(channel_files[channel])}')")
        else:
            channel_files[channel] = fpath
            print(f"    ✓  {fname}  →  {channel}")

    brightfield_path = channel_files.get(BRIGHTFIELD_KEY)

    fluor_channels = [
        (ch, channel_files[ch])
        for ch in FLUOR_CHANNEL_ORDER
        if ch in channel_files
    ]

    if brightfield_path is None:
        print(f"  [WARN] No RL Brightfield image found in {collection_dir}.")

    if not fluor_channels:
        print(f"  [WARN] No fluorescence channels found in {collection_dir}.")
    else:
        print(f"  Fluorescence channels queued (in order): "
              f"{[ch for ch, _ in fluor_channels]}")

    return brightfield_path, fluor_channels


# ---------------------------------------------------------------------------
# Circle detection / organisation (unchanged from V1.2)
# ---------------------------------------------------------------------------

def detect_circles(image, min_diameter, max_diameter):
    blurred = cv2.medianBlur(image, 5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=2 * max_diameter,
        param1=p1,
        param2=p2,
        minRadius=min_diameter,
        maxRadius=max_diameter,
    )
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        return circles
    return []


def organize_circles(circles, nd_mean):
    X, Y, R = circles
    indexed_circles = [X, Y, R,
                       np.repeat(np.nan, len(X)),
                       np.repeat(np.nan, len(X))]  # X,Y,R,col,row

    maxX = max(indexed_circles[0])
    maxY = max(indexed_circles[1])

    while np.count_nonzero(np.isnan(indexed_circles[4])) > 0:
        for i in range(len(indexed_circles[0])):
            dY = maxY - indexed_circles[1][i]
            dX = maxX - indexed_circles[0][i]
            iX = int(round(dX / nd_mean, 0))
            iY = int(round(dY / nd_mean, 0))
            indexed_circles[3][i] = iX
            indexed_circles[4][i] = iY
            if debug:
                print(f"sorting indices. Remaining:"
                      f"{np.count_nonzero(np.isnan(indexed_circles[3]))}\r")
    return indexed_circles


def process_brightfield_image(brightfield_path, min_diameter, max_diameter):
    """Load the brightfield TIFF and detect circles."""
    img_uint8 = load_tiff_as_gray_uint8(brightfield_path)
    if debug:
        print(f"  Brightfield image shape: {img_uint8.shape}")
    circles = detect_circles(img_uint8, min_diameter, max_diameter)
    return img_uint8, circles


def draw_circles_and_display(image, circles):
    if not debug:
        return

    max_dimension = 800
    h, w = image.shape[:2]
    scale = min(max_dimension / h, max_dimension / w)
    resized_image = cv2.resize(image, (int(w * scale), int(h * scale)))

    fig, ax = plt.subplots()
    ax.imshow(resized_image, cmap="gray")

    for x, y, r in circles:
        x = int(x * scale)
        y = int(y * scale)
        r = int(r * scale)
        circle_patch = patches.Circle((x, y), r, edgecolor="red",
                                      facecolor="none", linewidth=2)
        ax.add_patch(circle_patch)
        ax.plot(x, y, "go", markersize=5)
        ax.text(x + 11, y - r, f"({x},{y},{r})", color="white",
                fontsize=8, ha="right")

    ax.set_aspect("equal")
    ax.grid(True)
    plt.title("Detected Circles")
    plt.show()


def remove_outliers(circles):
    X, Y, R = [], [], []
    for x, y, r in circles:
        X.append(x); Y.append(y); R.append(r)

    meanR = np.mean(R)
    X = np.array(X)
    Y = np.array(Y)

    nearest_distances = []
    for i in range(len(X)):
        min_distance = float("inf")
        for j in range(len(X)):
            if i != j:
                distance = np.sqrt((X[i] - X[j]) ** 2 + (Y[i] - Y[j]) ** 2)
                if distance < min_distance:
                    min_distance = distance
        nearest_distances.append(min_distance)

    nearest_distances = np.array(nearest_distances)
    nd_mean = nearest_distances.mean()
    nd_stdev = nearest_distances.std()
    if debug:
        print(f"electrode spacing: {nd_mean} +/- {nd_stdev} px")

    to_remove = [i for i, dist in enumerate(nearest_distances)
                 if dist < (nd_mean - nd_mean * 0.1)
                 or dist > (nd_mean + nd_mean * 0.1)]

    X_final = np.delete(X, to_remove)
    Y_final = np.delete(Y, to_remove)

    if debug:
        plt.figure(figsize=(10, 8))
        plt.plot(X_final, Y_final, "go", label="Valid Points")
        if to_remove:
            plt.plot(X[to_remove], Y[to_remove], "ro", label="Removed Points")
        for i in range(len(X)):
            plt.text(X[i], Y[i], f"{round(nearest_distances[i])}",
                     fontsize=9, ha="right", color="blue")
        plt.xlabel("X Coordinate"); plt.ylabel("Y Coordinate")
        plt.title("Circle Centers with Nearest Neighbor Distances")
        plt.legend(); plt.grid(True); plt.show()

    return (
        (X_final, Y_final, np.repeat(meanR, len(X_final))),
        (X[to_remove], Y[to_remove], np.repeat(meanR, len(X[to_remove]))),
        nd_mean,
    )


def fill_missing_points(circles, nd_mean):
    X, Y, R = circles
    new_circles = []
    spacing_X = spacing_Y = nd_mean
    tolerance = 0.5
    tol_X = spacing_X * tolerance
    tol_Y = spacing_Y * tolerance
    concensus = False

    while not concensus:
        points_added = 0
        for i in range(len(X)):
            for dx in [-spacing_X, 0, spacing_X]:
                for dy in [-spacing_Y, 0, spacing_Y]:
                    if dx == 0 and dy == 0:
                        continue
                    cx = X[i] + dx
                    cy = Y[i] + dy
                    if not any(abs(cx - X[j]) < tol_X and abs(cy - Y[j]) < tol_Y
                               for j in range(len(X))):
                        if (min(X) - spacing_X * 0.5 < cx < max(X) + spacing_X * 0.5 and
                                min(Y) - spacing_Y * 0.5 < cy < max(Y) + spacing_Y * 0.5):
                            points_added += 1
                            new_circles.append((cx, cy, R[0]))
                            X = np.append(X, cx)
                            Y = np.append(Y, cy)
        if points_added == 0:
            concensus = True
        elif debug:
            print(f"\rDistance to consensus: {points_added}    ")

    combined_R = np.concatenate((R, np.array([c[2] for c in new_circles])))

    if debug:
        plt.figure(figsize=(10, 8))
        plt.plot(X, Y, "go", label="Original Circles")
        if new_circles:
            plt.plot([c[0] for c in new_circles], [c[1] for c in new_circles],
                     "bo", label="Added Circles")
        plt.xlabel("X Coordinate"); plt.ylabel("Y Coordinate")
        plt.title("Detected and Added Circles")
        plt.legend(); plt.grid(True); plt.show()

    return (X, Y, np.repeat(R[0], len(X))), new_circles


def plot_final_register(brightfield_image, circles, added, removed):
    if not show_final_registration:
        return

    max_dimension = 800
    h, w = brightfield_image.shape[:2]
    scale = min(max_dimension / h, max_dimension / w)
    resized_image = cv2.resize(brightfield_image, (int(w * scale), int(h * scale)))

    fig, ax = plt.subplots()
    ax.imshow(resized_image, cmap="gray")

    for i in range(len(circles[0])):
        x = int(circles[0][i] * scale)
        y = int(circles[1][i] * scale)
        r = int(circles[2][i] * scale)
        index = circles[5][i]
        circle_patch = patches.Circle((x, y), r, edgecolor="green",
                                      facecolor="none", linewidth=2)
        ax.add_patch(circle_patch)
        ax.text(x, y - r - 5, f"{index}", color="black",
                fontsize=8, ha="center")

    for x, y, r in zip(removed[0], removed[1], removed[2]):
        x, y, r = int(x * scale), int(y * scale), int(r * scale)
        ax.add_patch(patches.Circle((x, y), r, edgecolor="red",
                                    facecolor="none", linewidth=2, alpha=0.5))

    for x, y, r in added:
        x, y, r = int(x * scale), int(y * scale), int(r * scale)
        ax.add_patch(patches.Circle((x, y), r, edgecolor="blue",
                                    facecolor="none", linewidth=2))

    ax.set_aspect("equal"); ax.grid(True)
    plt.title("Final Circle Registration"); plt.show()


def snake_circles(organized_circles):
    X, Y, R, col_indices, row_indices = organized_circles
    max_row = int(max(row_indices))
    max_col = int(max(col_indices))
    snake_indices = np.full_like(col_indices, -1)
    index = 0

    for row in range(max_row + 1):
        col_range = range(max_col + 1) if row % 2 != 0 else reversed(range(max_col + 1))
        for col in col_range:
            match = np.where((col_indices == col) & (row_indices == row))
            if len(match[0]) > 0:
                snake_indices[match] = index
                index += 1

    return np.vstack((X, Y, R, col_indices, row_indices, snake_indices))


# ---------------------------------------------------------------------------
# Fluorescence quantification (TIFF version – array mode)
# ---------------------------------------------------------------------------

def quantify_fluorescence(fluor_channels, circles):
    """
    fluor_channels : list of (channel_name, filepath)
    circles        : snaked_circles array [X, Y, R, col, row, snake_idx]
    Returns        : list of (channel_name, [(snake_idx, corrected_fluor), …])
    """
    fluorescence_averages = []

    for channel_name, fpath in fluor_channels:
        fluorescence_image = load_tiff_raw(fpath)
        if debug:
            print(f"  [{channel_name}] image shape: {fluorescence_image.shape}")
        else:
            print(f"  Processing {channel_name}…")

        channel_averages = []

        for x, y, r, idx in zip(circles[0], circles[1], circles[2], circles[5]):
            global roi_inner
            _roi_inner = roi_inner
            if _roi_inner <= -r:
                _roi_inner = -r + 1

            outer_radius            = r + roi_outer
            inner_radius            = r + _roi_inner
            background_outer_radius = r + bckg_outer
            background_inner_radius = r + bckg_inner

            roi_mask = np.zeros_like(fluorescence_image, dtype=np.uint16)
            cv2.circle(roi_mask, (int(x), int(y)), int(round(outer_radius)), bitdepth, thickness=-1)
            cv2.circle(roi_mask, (int(x), int(y)), int(round(inner_radius)), 0,        thickness=-1)

            background_mask = np.zeros_like(fluorescence_image, dtype=np.uint16)
            cv2.circle(background_mask, (int(x), int(y)), int(round(background_outer_radius)), bitdepth, thickness=-1)
            cv2.circle(background_mask, (int(x), int(y)), int(round(background_inner_radius)), 0,        thickness=-1)

            roi_values        = fluorescence_image[roi_mask        == bitdepth]
            background_values = fluorescence_image[background_mask == bitdepth]

            avg_fluor  = np.mean(roi_values)        if roi_values.size        > 0 else np.nan
            avg_bkg    = np.mean(background_values) if background_values.size > 0 else np.nan
            corrected  = avg_fluor - avg_bkg
            channel_averages.append((idx, corrected))

            # Annotate image for display
            cv2.circle(fluorescence_image, (int(x), int(y)), int(round(outer_radius)),            (0, bitdepth, 0), 1)
            cv2.circle(fluorescence_image, (int(x), int(y)), int(round(inner_radius)),            (0, bitdepth, 0), 1)
            cv2.circle(fluorescence_image, (int(x), int(y)), int(round(background_outer_radius)), (bitdepth, 0, 0), 1, 2)
            cv2.circle(fluorescence_image, (int(x), int(y)), int(round(background_inner_radius)), (bitdepth, 0, 0), 1, 2)

        fluorescence_averages.append((channel_name, channel_averages))

        if show_if_images:
            plt.figure(figsize=(10, 8))
            plt.imshow(fluorescence_image, cmap="gray")
            plt.title(f"{channel_name} with ROIs and Background")
            plt.show()

    if debug:
        plt.figure(figsize=(12, 10))
        for i, (channel_name, averages) in enumerate(fluorescence_averages):
            indices, values = zip(*averages)
            plt.subplot(len(fluorescence_averages), 1, i + 1)
            plt.plot(indices, values, "go", label=channel_name)
            plt.xlabel("Snake Index")
            plt.ylabel("Corrected Average Fluorescence")
            plt.title(f"Corrected Average Fluorescence vs. Snake Index – {channel_name}")
            plt.legend()
        plt.tight_layout(); plt.show()

    return fluorescence_averages


# ---------------------------------------------------------------------------
# Fluorescence quantification – single electrode mode
# ---------------------------------------------------------------------------

def quantify_fluorescence_single(fluor_channels, circle):
    """
    fluor_channels : list of (channel_name, filepath)
    circle         : (x, y, r)
    Returns        : list of (channel_name, corrected_fluorescence)
                     — one entry per channel, all channels processed.
    """
    x, y, r = circle
    channel_results = []

    for channel_name, fpath in fluor_channels:
        fluorescence_image = load_tiff_raw(fpath)
        if debug:
            print(f"  [{channel_name}] image shape: {fluorescence_image.shape}")
        else:
            print(f"  Processing {channel_name}…")

        _roi_inner = roi_inner
        if _roi_inner <= -r:
            _roi_inner = -r + 1

        outer_radius            = r + roi_outer
        inner_radius            = r + _roi_inner
        background_outer_radius = r + bckg_outer
        background_inner_radius = r + bckg_inner

        roi_mask = np.zeros_like(fluorescence_image, dtype=np.uint16)
        cv2.circle(roi_mask, (int(x), int(y)), int(round(outer_radius)), bitdepth, thickness=-1)
        cv2.circle(roi_mask, (int(x), int(y)), int(round(inner_radius)), 0,        thickness=-1)

        background_mask = np.zeros_like(fluorescence_image, dtype=np.uint16)
        cv2.circle(background_mask, (int(x), int(y)), int(round(background_outer_radius)), bitdepth, thickness=-1)
        cv2.circle(background_mask, (int(x), int(y)), int(round(background_inner_radius)), 0,        thickness=-1)

        roi_values        = fluorescence_image[roi_mask        == bitdepth]
        background_values = fluorescence_image[background_mask == bitdepth]

        avg_fluor = np.mean(roi_values)        if roi_values.size        > 0 else np.nan
        avg_bkg   = np.mean(background_values) if background_values.size > 0 else np.nan
        corrected = avg_fluor - avg_bkg
        channel_results.append((channel_name, corrected))

        if show_if_images:
            display = cv2.normalize(fluorescence_image.astype(np.float32),
                                    None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
            cv2.circle(display, (int(x), int(y)), int(round(outer_radius)),            (0, 255, 0), 1)
            cv2.circle(display, (int(x), int(y)), int(round(inner_radius)),            (0, 255, 0), 1)
            cv2.circle(display, (int(x), int(y)), int(round(background_outer_radius)), (255, 0, 0), 2)
            cv2.circle(display, (int(x), int(y)), int(round(background_inner_radius)), (255, 0, 0), 2)
            plt.figure(figsize=(6, 6))
            plt.imshow(display)
            plt.title(f"Single Electrode – {channel_name}\n"
                      f"ROI (green), Background (red)\n"
                      f"Corrected fluorescence: {corrected:.2f}")
            plt.axis("off"); plt.tight_layout(); plt.show()

    return channel_results


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def fancy_plot(fluorescence_averages):
    """
    fluorescence_averages : list of (channel_name, [(snake_idx, value), …])
    Returns               : list of (channel_name, avg_value)
    """
    avg_values = []

    def _remove_outliers(data, window_size=moving_avg_n):
        cleaned = []
        flags   = []
        half    = window_size // 2
        for i in range(len(data)):
            window = data[max(0, i - half):min(len(data), i + half)]
            q1, q3 = np.percentile(window, 25), np.percentile(window, 75)
            iqr    = q3 - q1
            if (q1 - 1.5 * iqr) <= data[i] <= (q3 + 1.5 * iqr):
                cleaned.append(data[i]); flags.append(False)
            else:
                cleaned.append(np.nan); flags.append(True)
        cleaned = np.array(cleaned)
        not_nan = ~np.isnan(cleaned)
        idx     = np.arange(len(cleaned))
        cleaned = np.interp(idx, idx[not_nan], cleaned[not_nan])
        return cleaned, flags

    plt.figure(figsize=(12, 10))

    for i, (channel_name, averages) in enumerate(fluorescence_averages):
        averages_sorted  = sorted(averages, key=lambda x: x[0])
        indices, values  = zip(*averages_sorted)
        cleaned_values, outlier_flags = _remove_outliers(list(values))
        smoothed_values  = uniform_filter1d(cleaned_values, size=moving_avg_n)
        peaks, _         = find_peaks(
            smoothed_values,
            height=np.mean(smoothed_values) + np.std(smoothed_values),
            threshold=0.5,
            distance=15,
        )

        plt.subplot(len(fluorescence_averages), 1, i + 1)
        for j, (index, value) in enumerate(zip(indices, values)):
            color = "yo" if outlier_flags[j] else "go"
            plt.plot(index, value, color, alpha=0.5)

        plt.plot(indices, smoothed_values, "b-", label=f"Smoothed {channel_name}")
        plt.plot(np.array(indices)[peaks], smoothed_values[peaks], "ro",
                 label="Peaks", alpha=0.5)

        for peak in peaks:
            plt.text(indices[peak], smoothed_values[peak],
                     f"{indices[peak]:.1f}\n{smoothed_values[peak]:.2f}",
                     fontsize=8, ha="right", color="red")

        avg_value = np.mean(cleaned_values)
        avg_values.append((channel_name, avg_value))
        plt.axhline(y=avg_value, color="black", linestyle="--", linewidth=1, label="Average")
        plt.text(indices[-1], avg_value, f"Avg: {avg_value:.2f}",
                 fontsize=8, color="black", ha="right", va="bottom")
        plt.xlabel("Snake Index"); plt.ylabel("Corrected Average Fluorescence")
        plt.title(f"Corrected Average Fluorescence with Peaks – {channel_name}")
        plt.legend()

    plt.tight_layout()
    if show_plots:
        plt.show()

    print("\nAverage Fluorescence Values:")
    for channel_name, avg_value in avg_values:
        print(f"  {channel_name}: {avg_value:.2f}")
    print("\n-------------------------------------------------\n")
    return avg_values


def plot_single_electrode(channel_results):
    """
    channel_results : list of (channel_name, corrected_fluorescence)
                      — covers ALL channels found in the collection folder.
    Returns         : same list (matches fancy_plot signature for CSV saving)
    """
    channels = [c for c, _ in channel_results]
    values   = [v for _, v in channel_results]

    if show_plots:
        plt.figure(figsize=(max(4, len(channels) * 1.5), 5))
        bars = plt.bar(range(len(channels)), values, color="steelblue", edgecolor="black")
        plt.xticks(range(len(channels)), channels)
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:.2f}", ha="center", va="bottom", fontsize=10)
        plt.xlabel("Channel"); plt.ylabel("Corrected Average Fluorescence")
        plt.title("Single Electrode – Corrected Fluorescence per Channel")
        plt.tight_layout(); plt.show()

    print("\nSingle Electrode Fluorescence Values:")
    for ch, val in channel_results:
        print(f"  {ch}: {val:.2f}")
    print("\n-------------------------------------------------\n")
    return channel_results


# ---------------------------------------------------------------------------
# Experiment folder selection
# ---------------------------------------------------------------------------

def select_experiment_folder():
    """
    Open a folder-picker dialog and return the selected experiment directory.
    The experiment directory is expected to contain one or more collection
    sub-folders, each holding per-channel TIFF files.
    """
    Tk().withdraw()
    folder = askdirectory(title="Select Experiment Folder "
                                "(contains collection sub-folders)")
    return folder if folder else None


def find_collections(experiment_dir):
    """
    Return a sorted list of sub-folder paths inside *experiment_dir* that
    contain at least one TIFF file.
    """
    tiff_extensions = {".tif", ".tiff"}
    collections = []
    for entry in sorted(os.scandir(experiment_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        has_tiff = any(
            os.path.splitext(f)[1].lower() in tiff_extensions
            for f in os.listdir(entry.path)
        )
        if has_tiff:
            collections.append(entry.path)
    return collections


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    experiment_dir = select_experiment_folder()
    if not experiment_dir:
        print("No experiment folder selected.")
        raise SystemExit

    collections = find_collections(experiment_dir)
    if not collections:
        print(f"No collection sub-folders with TIFF files found in:\n  {experiment_dir}")
        raise SystemExit

    print(f"\nFound {len(collections)} collection(s) in: {experiment_dir}\n")

    all_avg_values = []   # [(collection_dir, [(channel_name, avg_value), …]), …]
    total = len(collections)

    for i, collection_dir in enumerate(collections, start=1):
        print(f"--- Collection {i}/{total}: {os.path.basename(collection_dir)} ---")

        brightfield_path, fluor_channels = load_collection(collection_dir)

        if brightfield_path is None:
            print("  Skipping (no RL Brightfield image).\n")
            continue

        if not fluor_channels:
            print("  Skipping (no recognised fluorescence channels).\n")
            continue

        print(f"  Brightfield : {os.path.basename(brightfield_path)}")
        for ch_name, ch_path in fluor_channels:
            print(f"  {ch_name:<14}: {os.path.basename(ch_path)}")

        # Detect circles from brightfield
        brightfield_image, circles = process_brightfield_image(
            brightfield_path, min_diameter, max_diameter
        )
        draw_circles_and_display(brightfield_image, circles)

        if is_array:
            # --- Array mode ---
            if len(circles) == 0:
                print("  WARNING: No circles detected. Skipping.\n")
                continue

            circles, removed, nd_mean = remove_outliers(circles)
            circles, added            = fill_missing_points(circles, nd_mean)
            organized_circles         = organize_circles(circles, nd_mean)
            snaked_circles            = snake_circles(organized_circles)
            plot_final_register(brightfield_image, snaked_circles, added, removed)

            fluorescence_averages     = quantify_fluorescence(fluor_channels, snaked_circles)
            avg_values                = fancy_plot(fluorescence_averages)

        else:
            # --- Single electrode mode ---
            if len(circles) == 0:
                print("  WARNING: No electrode detected. Skipping.\n")
                continue
            if len(circles) > 1:
                print(f"  WARNING: {len(circles)} circles detected; using the first one.")

            single_circle  = (int(circles[0][0]), int(circles[0][1]), int(circles[0][2]))
            if debug:
                print(f"  Using circle: x={single_circle[0]}, "
                      f"y={single_circle[1]}, r={single_circle[2]}")

            channel_results = quantify_fluorescence_single(fluor_channels, single_circle)
            avg_values      = plot_single_electrode(channel_results)

        all_avg_values.append((collection_dir, avg_values))

    # Summary
    print("\n========== Summary of Average Fluorescence Values ==========")
    for collection_dir, avg_values in all_avg_values:
        print(f"Collection: {os.path.basename(collection_dir)}")
        for channel_name, avg_value in avg_values:
            print(f"  {channel_name}: {avg_value:.2f}")

    # CSV export
    save_csv = input("\nSave summary as CSV file? (y/n): ").strip().lower()
    if save_csv == "y":
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(experiment_dir, f"summary_{timestamp}.csv")
        with open(output_file, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Collection", "Channel", "Average Fluorescence"])
            for collection_dir, avg_values in all_avg_values:
                for channel_name, avg_value in avg_values:
                    writer.writerow([
                        os.path.basename(collection_dir),
                        channel_name,
                        avg_value,
                    ])
        print(f"Summary saved to {output_file}")