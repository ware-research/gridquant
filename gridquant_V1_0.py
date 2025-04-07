'''
MIT License

Copyright (c) 2024, Oregon Health & Science Univeristy

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
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Please consider citing!
'''

import czifile
import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from scipy.stats import t
import csv 
import os
from datetime import datetime  # Import datetime for timestamp

#***********************************************************************************************************
#**********************************User defined settings:***************************************************
#***********************************************************************************************************
settings = "C5_axio" #options: c3_kinetix, c2_kinetix, c5_axio

debug = True
show_final_registration = False
show_if_imges = False
show_plots = False

#dont edit this line:
global bitdepth, min_diameter, max_diameter, roi_inner, roi_outer, bckg_inner, bckg_outer, moving_avg_n, p1, p2, dp

#user preset profiles:
match settings.lower():
    case "c5_kinetix":
        print("loading preset for c3_kinetix")
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
    
#***********************************************************************************************************
#**************Do not edit below this line unless you know what you're doing********************************
#***********************************************************************************************************

bitdepth = 16384

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

def select_path():
    Tk().withdraw()  # We don't want a full GUI, so keep the root window from appearing
    choice = input("Select (d)irectory or (f)ile: ").strip().lower()
    if choice == 'd':
        from tkinter.filedialog import askdirectory
        return askdirectory(title="Select Directory")
    elif choice == 'f':
        from tkinter.filedialog import askopenfilename
        return askopenfilename(
            title="Select CZI Image",
            filetypes=[("CZI files", "*.czi")]
        )
    else:
        print("Invalid choice. Please enter 'd' or 'f'.")
        return None

def draw_circles_and_display(image, circles):
    if not debug:
        return  # Skip plotting if debug is False

    # Determine the scaling factor
    max_dimension = 800
    h, w = image.shape[:2]
    scale = min(max_dimension / h, max_dimension / w)

    # Resize the image to fit within the 800x800 box
    resized_image = cv2.resize(image, (int(w * scale), int(h * scale)))

    # Create a figure and axis with matplotlib
    fig, ax = plt.subplots()
    ax.imshow(resized_image, cmap='gray')

    # Draw the circles and labels
    for x, y, r in circles:
        # Scale the circle positions and radius
        x = int(x * scale)
        y = int(y * scale)
        r = int(r * scale)

        # Draw the circle as a patch
        circle_patch = patches.Circle((x, y), r, edgecolor='red', facecolor='none', linewidth=2)
        ax.add_patch(circle_patch)

        # Draw the centroid
        ax.plot(x, y, 'go', markersize=5)  # Green dot for centroid
        # Label the centroid with its coordinates
        label = f"({x},{y},{r})"
        ax.text(x + 11, y - r, label, color='white', fontsize=8, ha='right')

    # Set the aspect of the plot to be equal
    ax.set_aspect('equal')

    # Add grid lines
    ax.grid(True)

    # Display the result with interactive zoom
    plt.title("Detected Circles")
    plt.show()

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
    
    if debug:
        # Plot points and annotate distances
        plt.figure(figsize=(10, 8))
        plt.plot(X_final, Y_final, 'go', label='Valid Points')
        if to_remove:
            plt.plot(X[to_remove], Y[to_remove], 'ro', label='Removed Points')
        
        for i in range(len(X)):
            plt.text(X[i], Y[i], f'{round(nearest_distances[i])}', fontsize=9, ha='right', color='blue')
        
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Circle Centers with Nearest Neighbor Distances')
        plt.legend()
        plt.grid(True)
        plt.show()

    return (X_final,Y_final,np.repeat(meanR,len(X_final))), ((X[to_remove], Y[to_remove], np.repeat(meanR,len(X[to_remove])))), nd_mean

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

    if debug:
        # Plot the result
        plt.figure(figsize=(10, 8))
        plt.plot(X, Y, 'go', label='Original Circles')
        if new_circles:
            plt.plot([c[0] for c in new_circles], [c[1] for c in new_circles], 'bo', label='Added Circles')
        
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Detected and Added Circles')
        plt.legend()
        plt.grid(True)
        plt.show()

    return (X,Y,np.repeat(R[0],len(X))), (new_circles)

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

def quantify_fluorescence(file_path, circles):
    with czifile.CziFile(file_path) as czi:
        image = czi.asarray()
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

            # Check and print the shape of the fluorescence image to ensure it matches expectations
            if debug:
                print(f"Fluorescence Image Shape (Channel {channel + 1}): {fluorescence_image.shape}")
            else:
                print(f"Processing Channel {channel + 1}...")

            # Prepare to store ROI and background averages
            channel_averages = []
            
            for x, y, r, idx in zip(circles[0], circles[1], circles[2], circles[5]):
                # Define the ROI and background boundaries
                outer_radius = r + roi_outer
                inner_radius = r + roi_inner
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
                
                # Draw the ROI and background boundaries on the fluorescence image for visualization
                cv2.circle(fluorescence_image, (int(x), int(y)), int(round(outer_radius, 0)), (0, bitdepth, 0), 1)
                cv2.circle(fluorescence_image, (int(x), int(y)), int(round(inner_radius, 0)), (0, bitdepth, 0), 1)
                cv2.circle(fluorescence_image, (int(x), int(y)), int(round(background_outer_radius, 0)), (bitdepth, 0, 0), 1, 2)
                cv2.circle(fluorescence_image, (int(x), int(y)), int(round(background_inner_radius, 0)), (bitdepth, 0, 0), 1, 2)
            
            # Store the averages for this channel
            fluorescence_averages.append(channel_averages)
            
            if show_if_imges:
                # Display the fluorescence image with ROIs and background
                plt.figure(figsize=(10, 8))
                plt.imshow(fluorescence_image, cmap='gray')
                plt.title(f'Fluorescence Channel {channel + 1} with ROIs and Background')
                plt.show()
        if debug:
            # Plot average fluorescence against snake_index
            plt.figure(figsize=(12, 10))
            for channel, averages in enumerate(fluorescence_averages):
                indices, values = zip(*averages)  # Unzip indices and values
                plt.subplot(num_channels, 1, channel + 1)
                plt.plot(indices, values, 'go', label=f'Channel {channel + 1}')
                plt.xlabel('Snake Index')
                plt.ylabel('Corrected Average Fluorescence')
                plt.title(f'Corrected Average Fluorescence vs. Snake Index for Channel {channel + 1}')
                plt.legend()

            plt.tight_layout()
            plt.show()

        return np.array(fluorescence_averages)

def fancy_plot(fluorescence_averages):
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks

    avg_values = []  # List to store average values for each channel

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

    plt.figure(figsize=(12, 10))

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

        # Plot the cleaned data and outliers
        plt.subplot(len(fluorescence_averages), 1, channel + 1)
        for i, (index, value) in enumerate(zip(indices, values)):
            if outlier_flags[i]:
                plt.plot(index, value, 'yo', alpha=0.5)  # Outliers in yellow
            else:
                plt.plot(index, value, 'go', alpha=0.5)  # Cleaned data in green

        # Plot the smoothed data
        plt.plot(indices, smoothed_values, 'b-', label=f'Smoothed Channel {channel + 1}')
        
        # Mark the peaks
        plt.plot(np.array(indices)[peaks], smoothed_values[peaks], "ro", label='Peaks', alpha=0.5)

        # Annotate each peak with its index and height
        for peak in peaks:
            plt.text(
                indices[peak], smoothed_values[peak],
                f'{indices[peak]:.1f}\n{smoothed_values[peak]:.2f}',
                fontsize=8,
                ha='right',
                color='red'
            )

        # Calculate and plot the average value
        avg_value = np.mean(cleaned_values)
        avg_values.append((channel + 1, avg_value))  # Store channel and avg_value
        plt.axhline(y=avg_value, color='black', linestyle='--', linewidth=1, label='Average')
        plt.text(
            indices[-1], avg_value,
            f'Avg: {avg_value:.2f}',
            fontsize=8,
            color='black',
            ha='right',
            va='bottom'
        )
        
        plt.xlabel('Snake Index')
        plt.ylabel('Corrected Average Fluorescence')
        plt.title(f'Corrected Average Fluorescence with Peaks for Channel {channel + 1}')
        plt.legend()

    plt.tight_layout()
    if show_plots:
        plt.show()

    # Print all average values to the command line
    print("\nAverage Fluorescence Values:")
    for channel, avg_value in avg_values:
        print(f"Channel {channel}: {avg_value:.2f}")

    print("\n-------------------------------------------------\n")

    return avg_values  # Return the average values for further use

if __name__ == "__main__":
    path = select_path()
    if path:
        if path.endswith(".czi"):  # Single file selected
            file_paths = [path]
        else:  # Directory selected
            file_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".czi")]

        all_avg_values = []  # Collect all average fluorescence values
        total_files = len(file_paths)  # Total number of files to process

        for i, file_path in enumerate(file_paths, start=1):
            print(f"Processing file {i}/{total_files}: {file_path}")
            brightfield_image, circles = process_czi_image(file_path, min_diameter, max_diameter)
            draw_circles_and_display(brightfield_image, circles)
            circles, removed, nd_mean = remove_outliers(circles)
            circles, added = fill_missing_points(circles, nd_mean)
            organized_circles = organize_circles(circles, nd_mean)
            snaked_circles = snake_circles(organized_circles)  # Rename variable to avoid conflict
            plot_final_register(brightfield_image, snaked_circles, added, removed)
            fluorescence_averages = quantify_fluorescence(file_path, snaked_circles)
            avg_values = fancy_plot(fluorescence_averages)  # Capture and print average values
            all_avg_values.append((file_path, avg_values))  # Store file and its average values

        # Display all average fluorescence values together
        print("\nSummary of Average Fluorescence Values:")
        for file_path, avg_values in all_avg_values:
            print(f"File: {file_path}")
            for channel, avg_value in avg_values:
                print(f"  Channel {channel}: {avg_value:.2f}")

        # Ask user if they want to save the summary as a CSV
        save_csv = input("\nSave summary as CSV file? (y/n): ").strip().lower()
        if save_csv == 'y':
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Generate timestamp
            output_dir = path if not path.endswith(".czi") else os.path.dirname(path)
            output_file = os.path.join(output_dir, f"summary_{timestamp}.csv")  # Add timestamp to filename
            with open(output_file, mode='w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(["File", "Channel", "Average Fluorescence"])
                for file_path, avg_values in all_avg_values:
                    for channel, avg_value in avg_values:
                        csv_writer.writerow([file_path, channel, avg_value])
            print(f"Summary saved to {output_file}")
    else:
        print("No valid path selected.")
