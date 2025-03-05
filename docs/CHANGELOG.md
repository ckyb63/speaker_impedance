# Change log

## Latest

## [0.1.9] 5 Mar, 2025

### Added

- Added an export ML Dataset button to the GUI which will combine all runs to one files for easier ML model training.

### Changed

- Ensured the environemntal data is recorded in the folder and each file name.
- Removed Pressure from arduino data collection.
- Readme files updated.

### Notes

- New GUI completed first stress test.

## [0.1.8] 4 Mar, 2025

### Changed

- Updated Readme files across the repository.
- Cleaned up information in some files.

## [0.1.7] 4 Mar, 2025

### Added

- Added a new GUI for the Auto Impedance Collection script.
- Added a temporary MATLAB code to plot data for 99% Humidity against ~45% Humidity.

### Fixed

- Fixed and Updated Dependencies in requirements.txt due to a vulnerability warning.
- Successfully fixed and colleted data using the new GUI.

## [0.1.6] 3 Mar, 2025

### Added

- Added a new GUI for the Auto Impedance Collection script.
- Added a new Arduino code for reporting specifically the Temperature, Humidity, and Pressure sensor.

### Changed

- There is now a direct pyqt6 converted version of the Auto Impedance Collection script.

### Fixed

- Fixed the issue where the matplotlib backend was not working in the Auto Impedance Collection script.
- Fixed the issue where the GUI_Predict.py was not working by removing the prediction functions from the GUI for now.
- Fixed GUI application crash when taking measurements.

## [0.1.5] 1 Mar, 2025

### Added

- Improved dark theme UI with consistent styling across applications
- Horizontal layout for advanced settings in both GUIs
- Enhanced speaker icon design with better visual representation
- Compact layout optimizations for laptop screens
- Real-time status indicators with improved visibility

### Changed

- Reorganized advanced settings into horizontal pairs for better space utilization
- Reduced vertical spacing in both applications
- Improved environmental readings display layout
- Updated documentation to reflect new UI changes

### Removed

- Project folder as it is using Impedance-main written by Keisuke.
- Datacollection.docx, it is not supposed to be here.

## [0.1.3] 27 Feb, 2025

- Entire New Impedance model training.
- Modified the impedance model written by Keisuke.

## [0.1.2] 26 Feb, 2025

### Added

- Added Temperature, Humidity, and Pressure to the data collection process as well as the GUI.
- Added new Arduino code for reporting specifically the Temperature, Humidity, and Pressure sensor.

### Changed

- Measurement unit details in the GUI.
- Arduino Code files organized in a folder.

## [0.1.1] 25 Feb, 2025

- Added a PyQt6 GUI to the Auto Impedance Collection script

## [0.1.0] 20 Feb, 2025

- Initial release and update of the Speaker Impedance project after previous scattered updates.
