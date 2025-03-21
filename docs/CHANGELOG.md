# Change log

## Latest

## [0.10.1] 15 Mar, 2025

### Updated

- Updated and Corrected some comments in both the Arduino and GUI code.
- Updated readme screenshot formatting.

## [0.10.0] 14 Mar, 2025

### Added

- Arduino code that utilizes the sensor suit on the Arduino Nano 33 BLE Sense.
  - Now measures Temperature, Humidity, Pressure, and Audio Levels in Raw RMS values and dBA.
- Several features and functions in the GUI.

### Updated

- [The GUI](/Analog%20Discovery/Auto_Impedance_New_GUI_PyQt6.py)
  - Environmental Data
    - Revamped the Environmental Section in the GUI.
    - Added COM Port Selection for the Arduino in the GUI.
      - It now automatically connects to the COM device if there is only one detected as avaliable.
  - Added scrollbar to the control panel if the screen is short on space.
  - Added a warning dialog if the Arduino is not connected.
  - Allowed measurements to still continue without environmental data, which reverts to the previous GUI behavior.
  - Reorganized the GUI for better readability.
  - Added tabs to the GUI settings for better organization.
    - Main and Advanced Settings.
  - Improved the UI/UX of the GUI, styling.
  - Added overall tooltips to the GUI for better user guidance.
  - Added better status indicators and messages in the GUI status.
  - Added Ctrl+S to start the measurement process.
  - Added auto arduino connection if there is only one detected.

### Fixed

- Bad Qt imports.
- Removed and fixed unecessary sys.exit() calls.

### Changed

- Updated the Readme.md
  - Added Picture of the GUI.

### Notes

- The audio levels are reported in raw RMS values, and attempted calibrated dBA values, it is not accurate but it will be accurate relative to the other values.

## [0.9.5] 13 Mar, 2025

### Added

- New Arduino code for reporting Temperature, Humidity, Pressure, and Decibel levels using the Arduino Nano 33 BLE Sense.

### Changed

- Reverted Arudino_Pa_Tp_Hu.ino to the previous version.

### Notes

- The Audio Levels are reported in raw RMS values, and attempted calibrated dBA values, it is not accurate but it will be accurate relative to the other values.

## [0.9.3] 12 Mar, 2025

### Fixed

- requirements.txt security update.

## [0.9.2] 8 Mar, 2025

### Changed

- Updated Readme file, and Requirements.txt file.

## [0.9.1] 7 Mar, 2025

### Changed

- Updated the MATLAB scripts to be more flexible and easier to use, also saves figures in the figures folder.
- Updated the version numbers to follow more proper versioning scheme. (MAJOR.Additions.Changes|fixes|removed|notes)
- Changelog updated to follow: Added, Changed, Fixed, Removed, Notes.

### Notes

- Temporary measurements made to test and confirm if impedanc is affected by the following factors:
  - Humidity
  - External Noise

## [0.9.0] 6 Mar, 2025 (0.1.11)

### Added

- Added a new MATLAB script to plot the impedance data for the A - 8 Humidity  difference dataset as a test to see if the impedance is affected by humidity.

## [0.8.1] 6 Mar, 2025 (0.1.10)

### Added

- Favicon for the GUI, found in the assets folder.

### Changed

- Moved both the tkinter and pyqt6 older GUI into a folder called "Older"
- Updated the Readme files across the repository.

### Fixed

- ML All in one file name now properly includes temperature and humidity.

## [0.8.0] 5 Mar, 2025 (0.1.9)

### Added

- Added an export ML Dataset button to the GUI which will combine all runs to one files for easier ML model training.

### Changed

- Ensured the environemntal data is recorded in the folder and each file name.
- Removed Pressure from arduino data collection.
- Readme files updated.

### Notes

- New GUI completed first stress test of 100 measurment runs.

## [0.7.1] 4 Mar, 2025 (0.1.8)

### Changed

- Updated Readme files across the repository.
- Cleaned up information in some files.

## [0.7.0] 4 Mar, 2025 (0.1.7)

### Added

- Added a new GUI for the Auto Impedance Collection script.
- Added a temporary MATLAB code to plot data for 99% Humidity against ~45% Humidity.

### Fixed

- Fixed and Updated Dependencies in requirements.txt due to a vulnerability warning.
- Successfully fixed and colleted data using the new GUI.

## [0.6.0] 3 Mar, 2025 (0.1.6)

### Added

- Added a new GUI for the Auto Impedance Collection script.
- Added a new Arduino code for reporting specifically the Temperature, Humidity, and Pressure sensor.

### Changed

- There is now a direct pyqt6 converted version of the Auto Impedance Collection script.

### Fixed

- Fixed the issue where the matplotlib backend was not working in the Auto Impedance Collection script.
- Fixed the issue where the GUI_Predict.py was not working by removing the prediction functions from the GUI for now.
- Fixed GUI application crash when taking measurements.

## [0.5.0] 1 Mar, 2025 (0.1.5)

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

## [0.3.1] 27 Feb, 2025 (0.1.3)

### Added

- Entire New Impedance model training.

### Changed

- Modified the impedance model written by Keisuke.

## [0.3.0] 26 Feb, 2025 (0.1.2)

### Added

- Added Temperature, Humidity, and Pressure to the data collection process as well as the GUI.
- Added new Arduino code for reporting specifically the Temperature, Humidity, and Pressure sensor.

### Changed

- Measurement unit details in the GUI.
- Arduino Code files organized in a folder.

## [0.2.0] 25 Feb, 2025 (0.1.1)

- Added a PyQt6 GUI to the Auto Impedance Collection script

## [0.1.0] 20 Feb, 2025

- Initial release and update of the Speaker Impedance project after previous scattered updates.

## [0.1.0-pre] 14 Oct ~ 18 Nov, 2024

- Initial development of the Speaker Impedance project.
  - TKinter GUI
  - Tympan Module
  - MATLAB Visualization Code
