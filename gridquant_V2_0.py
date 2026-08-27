'''
MIT License

Copyright (c) 2024, Oregon Health & Science Univeristy

Contributor(s): Christian Ross, Jason Ware (warej@ohsu.edu)

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
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Please consider citing!
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from scipy.stats import t
import csv 
import os
import json
from datetime import datetime  # Import datetime for timestamp
import threading 
from dataclasses import dataclass
import czifile
import xml.etree.ElementTree as ET

@dataclass
class ProcessingResult:
    file_path: str
    channel_names: list
    avg_values: list
    fluorescence_averages: object | None = None
    brightfield_image: object | None = None
    circles: object | None = None
    snaked_circles: object | None = None
    single_circle: object | None = None
    added: object | None = None
    removed: object | None = None
    outlier_plot_data: dict | None = None
    missing_plot_data: dict | None = None
    fancy_plot_data: dict | None = None
    roi_plot_data: dict | None = None


#***********************************************************************************************************
#**********************************User defined settings:***************************************************
#***********************************************************************************************************
    
# Default runtime settings
debug = False
show_final_registration = False
show_if_images = False
show_plots = False

is_array = False
min_diameter = 20
max_diameter = 30
roi_inner = -9
roi_outer = 2
bckg_inner = 55
bckg_outer = 100
moving_avg_n = 2
p1 = 50
p2 = 10
dp = 1.0

bitdepth = 16384


#***********************************************************************************************************
#**************Do not edit below this line unless you know what you're doing********************************
#***********************************************************************************************************

def detect_circles(image, min_diameter, max_diameter):
    blurred = cv2.medianBlur(image, 5)  # Blur the grayscale image to reduce noise
    
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=dp, 
        minDist=2*max_diameter, 
        param1=p1, 
        param2=p2, 
        minRadius=min_diameter, 
        maxRadius=max_diameter
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        return circles
    return []

def organize_circles(circles, nd_mean):
    X, Y, R = circles

    indexed_circles = [X,Y,R,np.repeat(np.nan,len(X)),np.repeat(np.nan,len(X))] #X,Y,R,col,row

    maxX = max(indexed_circles[0])
    maxY = max(indexed_circles[1])

    while np.count_nonzero(np.isnan(indexed_circles[4])) > 0:
        for i in range(0,len(indexed_circles[0])):
            dY = maxY - indexed_circles[1][i]
            dX = maxX - indexed_circles[0][i]
            iX = int(round(dX/nd_mean,0))
            iY = int(round(dY/nd_mean,0))
            indexed_circles[3][i] = iX
            indexed_circles[4][i] = iY
            if debug:
                print(f"sorting indicies. Remaining:{np.count_nonzero(np.isnan(indexed_circles[3]))}\r")

    return indexed_circles

def process_czi_image(file_path, min_diameter, max_diameter):
    with czifile.CziFile(file_path) as czi:
        image = czi.asarray()
        if debug:
            print(f"Image shape: {image.shape}")  # Print the shape for debugging

        # If the image has fewer dimensions, adjust the slicing accordingly
        if len(image.shape) == 4:
            # Assuming the shape is (Z, Y, X, C) or (C, Z, Y, X)
            brightfield_image = image[0, :, :, 0]  # Adjust this based on your image structure
        elif len(image.shape) == 3:
            # Assuming the shape is (Y, X, C)
            brightfield_image = image[:, :, 0]
        elif len(image.shape) == 5:
            # Assuming the shape is (Z, C, Y, X, N)
            brightfield_image = image[0, 0, :, :, 0]
        else:
            raise ValueError("Unexpected image dimensions.")

        brightfield_image = cv2.normalize(brightfield_image, None, 0, 255, cv2.NORM_MINMAX)
        brightfield_image = brightfield_image.astype(np.uint8)

        circles = detect_circles(brightfield_image, min_diameter, max_diameter)
        #organized_circles = organize_circles(circles)

        return brightfield_image, circles
    
def get_czi_channel_names(file_path):

    channels = []

    try:
        with czifile.CziFile(file_path) as czi:
            xml_string = czi.metadata()
        root = ET.fromstring(xml_string)

        # Get actual acquired image channels
        channel_nodes = root.findall(".//{*}Information/{*}Image/{*}Dimensions/{*}Channels/{*}Channel")

        for ch in channel_nodes:

            name = ch.attrib.get("Name")

            if name:
                channels.append(name)

    except Exception as e:
        print("Could not read channel names:", e)

    return channels

def remove_outliers(circles):
    # Extract x, y, and r lists
    X = []
    Y = []
    R = []
    
    for x, y, r in circles:
        X.append(x)
        Y.append(y)
        R.append(r)

    meanR = np.mean(R)
    
    X = np.array(X)
    Y = np.array(Y)
    
    # Initialize lists to store nearest neighbor distances
    nearest_distances = []

    for i in range(len(X)):
        x_i = X[i]
        y_i = Y[i]
        min_distance = float('inf')
        
        for j in range(len(X)):
            if i != j:
                x_j = X[j]
                y_j = Y[j]
                distance = np.sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2)
                if distance < min_distance:
                    min_distance = distance
        
        nearest_distances.append(min_distance)
    
    nearest_distances = np.array(nearest_distances)

    # Calculate mode and standard deviation
    nd_mean = nearest_distances.mean()
    nd_stdev = nearest_distances.std()
    if debug:
        print(f"electrode spacing: {nd_mean} +/- {nd_stdev} px")

    # Identify points to remove
    to_remove = [i for i, dist in enumerate(nearest_distances) if dist < (nd_mean - (nd_mean*0.1)) or dist > (nd_mean + (nd_mean*0.1))]
    
    X_final = np.delete(X, to_remove)
    Y_final = np.delete(Y, to_remove)

    plot_data = {
        "X": X,
        "Y": Y,
        "X_final": X_final,
        "Y_final": Y_final,
        "to_remove": to_remove,
        "nearest_distances": nearest_distances
    }

    return (X_final,Y_final,np.repeat(meanR,len(X_final))), ((X[to_remove], Y[to_remove], np.repeat(meanR,len(X[to_remove])))), nd_mean, plot_data

def fill_missing_points(circles, nd_mean):
    X, Y, R = circles

    new_circles = []

    spacing_X = nd_mean
    spacing_Y = nd_mean
    
    tolerance = 0.5  # 50% tolerance
    tol_X = spacing_X * tolerance
    tol_Y = spacing_Y * tolerance

    concensus = False

    while concensus == False: # Loop through the existing circles and check for missing points until all are found
        points_added = 0
        for i in range(len(X)):
            for dx in [-spacing_X, 0, spacing_X]:
                for dy in [-spacing_Y, 0, spacing_Y]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    candidate_x = X[i] + dx
                    candidate_y = Y[i] + dy
                    
                    # Check if a circle exists near the candidate location
                    if not any(np.abs(candidate_x - X[j]) < tol_X and np.abs(candidate_y - Y[j]) < tol_Y for j in range(len(X))):
                        if candidate_x > min(X) - spacing_X * 0.5 and candidate_x < max(X) + spacing_X * 0.5: # check that point is not outside of the ROI
                            if candidate_y > min(Y) - spacing_Y * 0.5 and candidate_y < max(Y) + spacing_Y * 0.5:
                                points_added += 1
                                new_circles.append((candidate_x, candidate_y, R[0]))
                                X = np.append(X,candidate_x)
                                Y = np.append(Y,candidate_y)

        if points_added == 0:
            concensus = True
        else:
            if debug:
                print("\r", f"Distance to concensus: {points_added}    ")
            

    # Combine the original and new circles

    combined_R = np.concatenate((R, np.array([c[2] for c in new_circles])))

    miss_plot_data = {
            "X": X,
            "Y": Y,
            "new_circles": new_circles,
        }

    return (X,Y,np.repeat(R[0],len(X))), (new_circles), miss_plot_data

def plot_final_register(brightfield_image, circles, added, removed):
    if not show_final_registration:
        return  # Skip plotting if show_final_registration is False

    # Determine the scaling factor
    max_dimension = 800
    h, w = brightfield_image.shape[:2]
    scale = min(max_dimension / h, max_dimension / w)

    # Resize the image to fit within the 800x800 box
    resized_image = cv2.resize(brightfield_image, (int(w * scale), int(h * scale)))

    # Create a figure and axis with matplotlib
    fig, ax = plt.subplots()
    ax.imshow(resized_image, cmap='gray')

    # Draw all circles in green
    for i in range(0,len(circles[0])):
        x = circles[0][i]
        y = circles[1][i]
        r = circles[2][i]
        col = circles[3][i]
        row = circles[4][i]
        index = circles[5][i]


        # Scale the circle positions and radius
        x = int(x * scale)
        y = int(y * scale)
        r = int(r * scale)

        # Draw the circle as a patch
        circle_patch = patches.Circle((x, y), r, edgecolor='green', facecolor='none', linewidth=2)
        ax.add_patch(circle_patch)

        # Annotate with row and column indices
        ax.text(x, y - r - 5, f"{index}", color='black', fontsize=8, ha='center')

    # Draw the removed circles in red with 50% transparency
    for x, y, r in zip(removed[0], removed[1], removed[2]):
        x = int(x * scale)
        y = int(y * scale)
        r = int(r * scale)

        circle_patch = patches.Circle((x, y), r, edgecolor='red', facecolor='none', linewidth=2, alpha=0.5)
        ax.add_patch(circle_patch)

    # Draw the added circles in blue
    for x, y, r in added:
        x = int(x * scale)
        y = int(y * scale)
        r = int(r * scale)

        circle_patch = patches.Circle((x, y), r, edgecolor='blue', facecolor='none', linewidth=2)
        ax.add_patch(circle_patch)

    # Set the aspect of the plot to be equal
    ax.set_aspect('equal')

    # Add grid lines
    ax.grid(True)

    # Display the result with interactive zoom
    plt.title("Final Circle Registration")
    plt.show()

def show_histogram_slider(image, title="Fluorescence Image"):
    """Display a fluorescence image with an interactive histogram range slider."""
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        image = image.squeeze()

    vmin = float(np.nanmin(image))
    vmax = float(np.nanmax(image))
    init_min = float(np.percentile(image, 1))
    init_max = float(np.percentile(image, 99))

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 0.05, 0.05, 1], hspace=0.35)
    ax_image = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[3, 0])
    ax_slider_min = fig.add_subplot(gs[1, 0])
    ax_slider_max = fig.add_subplot(gs[2, 0])

    clipped = np.clip(image, init_min, init_max)
    im_obj = ax_image.imshow(clipped, cmap='gray', vmin=init_min, vmax=init_max)
    ax_image.set_title(title)
    ax_image.axis('off')

    hist_values, hist_bins, _ = ax_hist.hist(image.ravel(), bins=256, color='gray')
    ax_hist.set_title('Intensity histogram')
    ax_hist.set_xlabel('Pixel value')
    ax_hist.set_ylabel('Count')
    ax_hist.set_xlim(vmin, vmax)

    annotation_text = ax_hist.text(
        0.5, 0.85,
        f'Lower = {init_min:.1f}    Upper = {init_max:.1f}',
        transform=ax_hist.transAxes,
        ha='center', va='center', color='black', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
    )

    slider_min = Slider(ax_slider_min, 'Min', vmin, vmax, valinit=init_min, valstep=max(1.0, (vmax - vmin) / 255.0))
    slider_max = Slider(ax_slider_max, 'Max', vmin, vmax, valinit=init_max, valstep=max(1.0, (vmax - vmin) / 255.0))

    def update(val):
        new_min = slider_min.val
        new_max = slider_max.val
        if new_min >= new_max:
            new_max = new_min + 1
            slider_max.set_val(new_max)

        im_obj.set_data(np.clip(image, new_min, new_max))
        im_obj.set_clim(new_min, new_max)
        annotation_text.set_text(f'Lower = {new_min:.1f}    Upper = {new_max:.1f}')
        fig.canvas.draw_idle()

    slider_min.on_changed(update)
    slider_max.on_changed(update)

    plt.tight_layout()
    #plt.show()


def snake_circles(organized_circles):
    # Extract the circle data
    X, Y, R, col_indices, row_indices = organized_circles
    
    # Find the maximum row and column indices
    max_row = int(max(row_indices))
    max_col = int(max(col_indices))
    
    # Create a new list to store the snaking indices
    snake_indices = np.full_like(col_indices, -1)  # Initialize with -1 or some invalid value
    
    # Initialize the index counter
    index = 0
    
    for row in range(max_row + 1):
        if row % 2 != 0:
            # For even rows, index from left to right
            for col in range(max_col + 1):
                match = np.where((col_indices == col) & (row_indices == row))
                if len(match[0]) > 0:
                    snake_indices[match] = index
                    index += 1
        else:
            # For odd rows, index from right to left
            for col in reversed(range(max_col + 1)):
                match = np.where((col_indices == col) & (row_indices == row))
                if len(match[0]) > 0:
                    snake_indices[match] = index
                    index += 1
    
    # Append the snaking indices to the organized circles array
    organized_circles_with_snake = np.vstack((X, Y, R, col_indices, row_indices, snake_indices))
    
    return organized_circles_with_snake

def quantify_fluorescence(file_path, circles, log_callback=None):
    with czifile.CziFile(file_path) as czi:
        image = czi.asarray()
        roi_plot_data = []

        if debug:
            print(f"Image shape: {image.shape}")  # Print the shape for debugging
        
        # Determine the number of dimensions in the image
        num_dimensions = len(image.shape)
        
        if num_dimensions == 4:  # CYXZ (Single scene and time point)
            num_channels = image.shape[0]  # Number of fluorescence channels
            y_dim = image.shape[1]  # Y dimension
            x_dim = image.shape[2]  # X dimension
            num_z_stacks = image.shape[3]  # Number of z-stacks (depth slices)
        elif num_dimensions == 5:  # ZCYXN (Single scene and time point)
            num_channels = image.shape[1]  # Number of fluorescence channels
            y_dim = image.shape[2]  # Y dimension
            x_dim = image.shape[3]  # X dimension
            num_z_stacks = image.shape[0]  # Number of z-stacks (depth slices)
        elif num_dimensions == 3:  # YXC (Single-channel image)
            num_channels = 1
            y_dim, x_dim, _ = image.shape
            num_z_stacks = 1
        else:
            raise ValueError("Unexpected image dimensions.")
        
        # Prepare a list to store the average fluorescence for each circle in each channel
        fluorescence_averages = []
        
        for channel in range(num_channels):
            # Extract the fluorescence image for the current channel and first Z-stack
            if num_channels > 1:
                fluorescence_image = image[channel, :, :, 0] if num_dimensions == 4 else image[0, channel, :, :, 0]
            else:
                fluorescence_image = image[:, :, 0] if num_dimensions == 3 else image[0, 0, :, :, 0]

            plot_image = fluorescence_image.copy()

            # Check and print the shape of the fluorescence image to ensure it matches expectations
            if debug:
                print(f"Fluorescence Image Shape (Channel {channel + 1}): {fluorescence_image.shape}")
            else:
                print(f"Processing Channel {channel + 1}...")

            if log_callback:
                log_callback(f"Processing Channel {channel + 1}/{num_channels}")

            # Prepare to store ROI and background averages
            channel_averages = []
            channel_roi_data = []
            for x, y, r, idx in zip(circles[0], circles[1], circles[2], circles[5]):
                # Define the ROI and background boundaries
                local_roi_inner = roi_inner
                if local_roi_inner <= -r:
                    local_roi_inner = -r
                outer_radius = r + roi_outer
                inner_radius = r + local_roi_inner
                background_outer_radius = r + bckg_outer
                background_inner_radius = r + bckg_inner

                # Create masks for ROI and background
                roi_mask = np.zeros_like(fluorescence_image, dtype=np.uint16)
                cv2.circle(roi_mask, (int(x), int(y)), int(round(outer_radius, 0)), bitdepth, thickness=-1)
                cv2.circle(roi_mask, (int(x), int(y)), int(round(inner_radius, 0)), 0, thickness=-1)

                background_mask = np.zeros_like(fluorescence_image, dtype=np.uint16)
                cv2.circle(background_mask, (int(x), int(y)), int(round(background_outer_radius, 0)), bitdepth, thickness=-1)
                cv2.circle(background_mask, (int(x), int(y)), int(round(background_inner_radius, 0)), 0, thickness=-1)

                # Apply the masks to the fluorescence image
                roi_values = fluorescence_image[roi_mask == bitdepth]
                background_values = fluorescence_image[background_mask == bitdepth]

                # Drop empty slices before calculating means
                if roi_values.size > 0:
                    average_fluorescence = np.mean(roi_values)
                else:
                    average_fluorescence = np.nan  # Assign NaN if empty

                if background_values.size > 0:
                    average_background = np.mean(background_values)
                else:
                    average_background = np.nan  # Assign NaN if empty

                corrected_fluorescence = average_fluorescence - average_background
                channel_averages.append((idx, corrected_fluorescence))

                channel_roi_data.append({
                    "x": x,
                    "y": y,
                    "roi_outer": outer_radius,
                    "roi_inner": inner_radius,
                    "background_outer": background_outer_radius,
                    "background_inner": background_inner_radius,
                    "index": idx
                })
                
            # Store the averages for this channel
            fluorescence_averages.append(channel_averages)

            roi_plot_data.append({
                "channel": channel + 1,
                "image": plot_image,
                "circles": channel_roi_data
            })

        return np.array(fluorescence_averages), roi_plot_data

def quantify_fluorescence_single(file_path, circle, log_callback=None):
    """
    Quantify fluorescence for a single electrode (non-array mode).
    circle: (x, y, r) tuple for the single detected electrode.
    Returns list of (channel, corrected_fluorescence) tuples.
    """
    x, y, r = circle
    with czifile.CziFile(file_path) as czi:
        image = czi.asarray()

        if debug:
            print(f"Image shape: {image.shape}")

        num_dimensions = len(image.shape)

        if num_dimensions == 4:
            num_channels = image.shape[0]
        elif num_dimensions == 5:
            num_channels = image.shape[1]
        elif num_dimensions == 3:
            num_channels = 1
        else:
            raise ValueError("Unexpected image dimensions.")

        roi_plot_data = []
        channel_results = []  # list of (channel_num, corrected_fluorescence)

        for channel in range(num_channels):
            if num_channels > 1:
                fluorescence_image = image[channel, :, :, 0] if num_dimensions == 4 else image[0, channel, :, :, 0]
            else:
                fluorescence_image = image[:, :, 0] if num_dimensions == 3 else image[0, 0, :, :, 0]

            if debug:
                print(f"Fluorescence Image Shape (Channel {channel + 1}): {fluorescence_image.shape}")
            else:
                print(f"Processing Channel {channel + 1}...")


            if log_callback:
                log_callback(f"Processing Channel {channel + 1}/{num_channels}")

            global roi_inner
            _roi_inner = roi_inner
            if _roi_inner <= -r:
                _roi_inner = -r + 1

            outer_radius = r + roi_outer
            inner_radius = r + _roi_inner
            background_outer_radius = r + bckg_outer
            background_inner_radius = r + bckg_inner

            # Create masks
            roi_mask = np.zeros_like(fluorescence_image, dtype=np.uint16)
            cv2.circle(roi_mask, (int(x), int(y)), int(round(outer_radius, 0)), bitdepth, thickness=-1)
            cv2.circle(roi_mask, (int(x), int(y)), int(round(inner_radius, 0)), 0, thickness=-1)

            background_mask = np.zeros_like(fluorescence_image, dtype=np.uint16)
            cv2.circle(background_mask, (int(x), int(y)), int(round(background_outer_radius, 0)), bitdepth, thickness=-1)
            cv2.circle(background_mask, (int(x), int(y)), int(round(background_inner_radius, 0)), 0, thickness=-1)

            roi_values = fluorescence_image[roi_mask == bitdepth]
            background_values = fluorescence_image[background_mask == bitdepth]

            average_fluorescence = np.mean(roi_values) if roi_values.size > 0 else np.nan
            average_background = np.mean(background_values) if background_values.size > 0 else np.nan
            corrected_fluorescence = average_fluorescence - average_background

            roi_plot_data.append({
                "channel": channel + 1,
                "image": fluorescence_image.copy(),
                "circles": [{
                    "x": x,
                    "y": y,
                    "r": r,
                    "roi_outer": outer_radius,
                    "roi_inner": inner_radius,
                    "background_outer": background_outer_radius,
                    "background_inner": background_inner_radius,
                    "index": 0
                }]
            })

            channel_results.append((channel + 1, corrected_fluorescence))

        return channel_results, roi_plot_data


def plot_single_electrode(channel_results):
    """
    Returns fluorescence data only.
    Plotting is handled by GUI.
    """

    print("\nSingle Electrode Fluorescence Values:")

    for ch, val in channel_results:
        print(f"  Channel {ch}: {val:.2f}")

    print("\n-------------------------------------------------\n")

    return [(ch, val) for ch, val in channel_results]

def get_channel_name(channel, channel_names):
    if 1 <= channel <= len(channel_names):
        return channel_names[channel - 1]
    return f"Channel {channel}"

def fancy_plot(fluorescence_averages):
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks

    avg_values = []  # List to store average values for each channel
    plot_data = []

    def remove_outliers(data, window_size = moving_avg_n):
        cleaned_data = []
        outlier_flags = []
        half_window = window_size // 2

        for i in range(len(data)):
            # Define the window around the current point
            start = max(0, i - half_window)
            end = min(len(data), i + half_window)
            window = data[start:end]

            # Calculate Q1, Q3, and IQR
            q1 = np.percentile(window, 25)
            q3 = np.percentile(window, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # Check if the current point is an outlier
            if lower_bound <= data[i] <= upper_bound:
                cleaned_data.append(data[i])
                outlier_flags.append(False)
            else:
                cleaned_data.append(np.nan)
                outlier_flags.append(True)

        # Optionally, interpolate to fill NaNs
        cleaned_data = np.array(cleaned_data)
        not_nan = ~np.isnan(cleaned_data)
        indices = np.arange(len(cleaned_data))
        cleaned_data = np.interp(indices, indices[not_nan], cleaned_data[not_nan])

        return cleaned_data, outlier_flags

    for channel, averages in enumerate(fluorescence_averages):
        # Sort the averages by snake index
        averages_sorted = sorted(averages, key=lambda x: x[0])
        indices, values = zip(*averages_sorted)

        # Remove outliers and get outlier flags
        cleaned_values, outlier_flags = remove_outliers(values)

        # Apply a moving average with a window of 10 points
        smoothed_values = uniform_filter1d(cleaned_values, size=moving_avg_n)

        # Identify significant peaks
        peaks, properties = find_peaks(
            smoothed_values,
            height=np.mean(smoothed_values) + 1 * np.std(smoothed_values),
            threshold=0.5,
            distance=15
        )

        # Calculate and plot the average value
        avg_value = np.mean(cleaned_values)
        avg_values.append((channel + 1, avg_value))  # Store channel and avg_value

        plot_data.append({
            "channel": channel + 1,
            "indices": np.array(indices),
            "values": np.array(values),
            "cleaned_values": cleaned_values,
            "outlier_flags": outlier_flags,
            "smoothed_values": smoothed_values,
            "peaks": peaks,
            "avg_value": avg_value,
        })

    # Print all average values to the command line
    print("\nAverage Fluorescence Values:")
    for channel, avg_value in avg_values:
        print(f"Channel {channel}: {avg_value:.2f}")

    print("\n-------------------------------------------------\n")

    return avg_values, plot_data # Return the average values for further use

def calculate_average_values(fluorescence_averages, moving_avg_n):

    avg_values=[]

    for channel, averages in enumerate(fluorescence_averages):

        values=[x[1] for x in averages]

        avg=np.mean(values)

        avg_values.append(
            (channel+1, avg)
        )

    return avg_values

def run_pipeline(config, progress_callback=None, log_callback=None, cancel_event=None):

        global debug
        global show_final_registration
        global show_if_images
        global show_plots

        global is_array
        global min_diameter
        global max_diameter
        global roi_inner
        global roi_outer
        global bckg_inner
        global bckg_outer
        global moving_avg_n
        global p1
        global p2
        global dp


        # For GUI: Logs which chip configuration is selected

        print("\nSelected configuration:")
        for key, value in config.items():
            print(f"{key}: {value}")
        print()

        if log_callback:

            log_callback("")
            log_callback("=" * 50)
            log_callback("CONFIGURATION")
            log_callback("=" * 50)

            for key, value in config.items():

                log_callback(
                    f"{key}: {value}"
                )

            log_callback("")

        # Load GUI settings
        show_plots = config.get("show_plots", False)
        debug = config.get("debug", False)
        show_final_registration = config.get("show_final_registration", False)
        show_if_images = config.get("show_if_images", False)
        
        # Load preset values
        is_array = config.get("is_array", False)
        min_diameter = int(config.get("min_diameter", 20))
        max_diameter = int(config.get("max_diameter", 30))
        roi_inner = int(config.get("roi_inner", -9))
        roi_outer = int(config.get("roi_outer", 2))
        bckg_inner = int(config.get("bckg_inner", 55))
        bckg_outer = int(config.get("bckg_outer", 100))
        moving_avg_n = int(config.get("moving_avg_n", 2))
        p1 = float(config.get("p1", 50))
        p2 = float(config.get("p2", 10))
        dp = float(config.get("dp", 1))
        path = config["path"]

        if path.endswith(".czi"):
            file_paths = [path]

        else:
            file_paths = [
                os.path.join(path, f)
                for f in os.listdir(path)
                if f.endswith(".czi")
            ]

        results = []

        total_files = len(file_paths)

        for i, file_path in enumerate(file_paths, start=1):

            added = None
            removed = None
            snaked_circles = None
            outlier_plot_data = None
            missing_plot_data = None
            single_circle = None
            figures = []

            channel_names = get_czi_channel_names(file_path)
            print("Detected Channels:")
            for name in channel_names:
                print(name)

            if log_callback:
                log_callback("")
                log_callback("=" * 50)
                log_callback(
                    f"PROCESSING FILE {i}/{total_files}"
                )
                log_callback("=" * 50)

                log_callback(
                    os.path.basename(file_path)
                )

            if cancel_event and cancel_event.is_set():
                print("Cancelled Processing")
                break

            print(f"Processing file {i}/{total_files}: {file_path}")


            brightfield_image, circles = process_czi_image(file_path, min_diameter, max_diameter)

            if len(circles) == 0:
                raise ValueError(f"No circles were detected in: \n {os.path.basename(file_path)}")

            if is_array:

                circles, removed, nd_mean, outlier_plot_data = remove_outliers(circles)

                circles, added, missing_plot_data = fill_missing_points(circles, nd_mean)

                organized_circles = organize_circles(circles, nd_mean)

                snaked_circles = snake_circles(organized_circles)

                fluorescence_averages, roi_plot_data = quantify_fluorescence(file_path, snaked_circles, log_callback)

                avg_values, fancy_plot_data = fancy_plot(fluorescence_averages)

                named_avg_values = []

                for channel, value in avg_values:

                    name = get_channel_name(channel, channel_names)
                    named_avg_values.append((name, value))

            else:

                if len(circles) == 0:
                    print(f"No electrode detected in {file_path}")
                    continue

                single_circle = (circles[0][0], circles[0][1], circles[0][2])

                channel_results, roi_plot_data = quantify_fluorescence_single(file_path, single_circle, log_callback)

                avg_values = plot_single_electrode(channel_results)
                
                named_avg_values = []

                for channel, value in avg_values:

                    name = get_channel_name(channel, channel_names)

                    named_avg_values.append((name, value))

            results.append(
                ProcessingResult(
                    file_path=file_path,
                    channel_names=channel_names,
                    avg_values=named_avg_values,
                    fluorescence_averages=fluorescence_averages if is_array else None,
                    brightfield_image=brightfield_image,
                    circles=circles,
                    snaked_circles=snaked_circles if is_array else None,
                    single_circle=single_circle if not is_array else None,
                    added=added if is_array else None,
                    removed=removed if is_array else None,
                    outlier_plot_data=outlier_plot_data if is_array else None,
                    missing_plot_data=missing_plot_data if is_array else None,
                    fancy_plot_data=fancy_plot_data if is_array else None,
                    roi_plot_data=roi_plot_data, 
                )

            )

            # send progress to GUI
            if progress_callback:
                progress_callback(i, total_files, file_path)

            if log_callback:
                log_callback(f"Completed file {i}/{total_files}: {os.path.basename(file_path)}")

        if debug: 
            print("Number of results:", len(results))
            print(results)

        return results


