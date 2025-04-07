# gridquant (Immunofluorescence Quantification of Dielectrophoresis)

## Requirements:
most current version was tested in python v 3.13.1
-should work in all python versions 3.10 and later.
included in requirements.txt (can be installed using "py -m pip install -r requirements.txt"
- czifile==2019.7.2.1
- opencv-python==4.11.0.86
- numpy==2.2.3
- matplotlib==3.10.0
- scipy==1.15.2
- DateTime==5.5

Other dependencies (generally included in standard python installations):
- tkinter
- os
- csv

## Usage:
This software is used to quantify images with fluorescence signals in a regular quare grid. Presets are included for imaging C2-C5 DEP chips manufactured at Oregon Health & Science University by the laboratory of Dr. Stuart Ibsen. Presets also include the use of Zeiss microscopes with a 5x obsective imaged using an Zeiss Axiocam 506 or Kinetix22 camera. Additional presets can be configurated in the user defined settings.
##General Usage:
Save images as CZI files.
run the .py script
input selection into command prompt window and press enter
- inputting "f" allows the user to select a single file
- inputting "d" allows the user to select a directory. Any .czi files in any subdirectory will be opened and analyzed. Progress is shown when each file is loaded.

The software will find all circles in the specified range, and find the most common grid shape. any unaligned circles are removed. Then any missing circles are added in. The result is a forced grid that covers every electrode. Outliers (bright or dark electrodes) are removed by user-defined statistics upon analysis.
The results can be saved when prompted at the end of the program. A .csv file will be saved to the directory opened containing the final summary plot.

Any positive fluroescence values can be considered significant. Brightfield values will typically be negative as the electrodes tend to darken durring collection. This is normal.

## User Defined Settings:
### Outputs:
- debug: 
  - type:bool
  - if True:
    shows fine detail of plotting, circle registration (removal and addition), image size output, distance to grid concensus, raw data plot and other busy information.
- show_final_registration:
  - type:bool
  - if True:
    shows final circle registration plot which displays found circles in green, removed circles in red, and added circles in blue. Also displays snake index identifiers on each circle.
- show_if_images:
  - type:bool
  - if True:
    displays immunofluorescence images for each channel including the ROI for quantification and background
- show_plots:
  - type:bool
  - if True:
    displays the fancy_plot function output which includes peaks, a moving average trendline, and the mean IF intensity.

### Preset settings:
- min_diameter:
  - type: int
  - the minnimum diameter (in pixels) that the algorithm will search for. The initial circle van be viewed in the first plot if debug mode is turned on. This circle should be aligned with the outer edge of the inner electrode. 
- max_diameter:
  - type: int
  - the maximum diameter (in pixels) that the algorithm will search for.
- roi_inner
  - type: int
  - the distance from the original circle that the signal ROI inner bounds will be placed. Typically negative and must be less than roi_outer.
- roi_outer
  - type: int
  - the distance from the original circle that the signal ROI outer bounds will be placed. Typically negative and must be greater than roi_inner.
- bckg_inner
  - type: int
  - the distance from the original circle that the background ROI inner bounds will be placed. Typically positive and must be less than bckg_outer. The background ROI should typically cover most of the interior of the auxilliary electrodes
- roi_outer
  - type: int
  - the distance from the original circle that the background ROI outer bounds will be placed. Typically negative and must be greater than bckg_inner. The background ROI should typically cover most of the interior of the auxilliary electrodes
- moving_avg_n
  - type: int
  - the range of moving averages used to define the line drawn on the fancy_plot output and to remove outliers. This does impact output values even if show_plots is False as outliers are removed.
- p1, p2, dp:
  - type: float
  - variables used to control circle selection. More info can be found here: https://docs.opencv.org/3.4/d4/d70/tutorial_hough_circle.html

## Notes:
A sample directory is included taken on a zeiss imager D2 microscope with an axiocam 506. Use the preset "C5_axio" to analyze. The two included files are identical and should yield the same values.

For more information, bug reports, and other advice, contact Jason Ware (warej@ohsu.edu)
Contact ibsen@ohsu.edu or warej@ohsu.edu for research collaboration access to dielectrophoresis devices. 


