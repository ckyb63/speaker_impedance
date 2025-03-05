% Script to average and plot impedance data from A5 and A8 runs
% This script loads CSV files from the Analog Discovery/Collected_Data folder,
% averages data across 100 runs for each condition, and plots the results.

clear;
clc;
close all;

% Define paths to data folders - using filesep for cross-platform compatibility
% Try multiple possible paths to handle different working directory scenarios
fprintf('Attempting to locate data folders...\n');

% Option 1: Direct path from current directory
baseFolderOption1 = fullfile('Analog Discovery', 'Collected_Data');

% Option 2: Path from one directory up
baseFolderOption2 = fullfile('..', 'Analog Discovery', 'Collected_Data');

% Option 3: Path as originally specified but with correct separator
baseFolderOption3 = ['Analog Discovery', filesep, 'Collected_Data'];

% Option 4: Absolute path to the Collected_Data folder
% You can customize this if you know the absolute path
% baseFolderOption4 = 'C:\Users\maxch\Documents\Purdue Files\Audio Research\Github\speaker_impedance\Analog Discovery\Collected_Data';

% Try each path option until we find one that works
if exist(baseFolderOption1, 'dir')
    baseFolder = baseFolderOption1;
    fprintf('Found data using path option 1: %s\n', baseFolder);
elseif exist(baseFolderOption2, 'dir')
    baseFolder = baseFolderOption2;
    fprintf('Found data using path option 2: %s\n', baseFolder);
elseif exist(baseFolderOption3, 'dir')
    baseFolder = baseFolderOption3;
    fprintf('Found data using path option 3: %s\n', baseFolder);
% elseif exist(baseFolderOption4, 'dir')
%     baseFolder = baseFolderOption4;
%     fprintf('Found data using path option 4: %s\n', baseFolder);
else
    % If none of the options work, let user specify the path interactively
    fprintf('Could not automatically locate the data folder.\n');
    fprintf('Please select the ''Collected_Data'' folder manually.\n');
    baseFolder = uigetdir('.', 'Select the Collected_Data folder');
    
    if baseFolder == 0
        error('Folder selection canceled. Cannot proceed without data folder.');
    end
    fprintf('Using user-selected path: %s\n', baseFolder);
end

% Define subfolder paths
A5_folder = fullfile(baseFolder, 'A_5');
A8_folder = fullfile(baseFolder, 'A_8');

% Check if the paths exist
fprintf('Checking paths...\n');
if ~exist(baseFolder, 'dir')
    error('Base folder does not exist: %s\nCheck if the path is correct.', baseFolder);
end

if ~exist(A5_folder, 'dir')
    error('A5 folder does not exist: %s\nCheck if the path is correct.', A5_folder);
end

if ~exist(A8_folder, 'dir')
    error('A8 folder does not exist: %s\nCheck if the path is correct.', A8_folder);
end

% Print folder contents to help diagnose issues
fprintf('Contents of base folder: %s\n', baseFolder);
dirContents = dir(baseFolder);
for i = 1:length(dirContents)
    fprintf('  %s\n', dirContents(i).name);
end

fprintf('Contents of A5 folder: %s\n', A5_folder);
dirContentsA5 = dir(A5_folder);
for i = 1:length(dirContentsA5)
    fprintf('  %s\n', dirContentsA5(i).name);
end

fprintf('Contents of A8 folder: %s\n', A8_folder);
dirContentsA8 = dir(A8_folder);
for i = 1:length(dirContentsA8)
    fprintf('  %s\n', dirContentsA8(i).name);
end

% Initialize arrays to store averaged data
% We'll initialize them after reading the first file to get dimensions

% Function to read and process data from a folder
function [avgFreq, avgMag, avgPhase, avgRs, avgXs] = processFolder(folderPath)
    % Get list of all CSV files in the folder
    fileList = dir(fullfile(folderPath, '*.csv'));
    
    % Check if we found the expected number of files
    numFiles = length(fileList);
    if numFiles == 0
        error('No CSV files found in folder: %s\nCheck if the path is correct or if the folder contains CSV files.', folderPath);
    end
    
    if numFiles ~= 100
        warning('Expected 100 files, but found %d in folder %s', numFiles, folderPath);
    end
    
    fprintf('Processing %d files from folder: %s\n', numFiles, folderPath);
    
    % Initialize arrays for the first file to determine dimensions
    firstFilePath = fullfile(folderPath, fileList(1).name);
    fprintf('Reading first file: %s\n', firstFilePath);
    
    % Check if the file exists
    if ~exist(firstFilePath, 'file')
        error('File does not exist: %s\nCheck if the path is correct.', firstFilePath);
    end
    
    % Read the first file using a more compatible method
    try
        % First get the full content to determine metadata
        fileContent = fileread(firstFilePath);
        lines = strsplit(fileContent, '\n');
        
        % Skip the first two lines (metadata and headers)
        % Use csvread for the data (skipping the first 2 lines)
        firstData = csvread(firstFilePath, 2, 0);
    catch e
        error('Error reading file %s: %s', firstFilePath, e.message);
    end
    
    % Get dimensions
    numPoints = size(firstData, 1);
    
    % Initialize arrays for accumulating data
    freqData = zeros(numPoints, numFiles);
    magData = zeros(numPoints, numFiles);
    phaseData = zeros(numPoints, numFiles);
    rsData = zeros(numPoints, numFiles);
    xsData = zeros(numPoints, numFiles);
    
    % Store first file data
    freqData(:, 1) = firstData(:, 1);  % Frequency (Hz)
    phaseData(:, 1) = firstData(:, 2); % Phase (degrees)
    magData(:, 1) = firstData(:, 3);   % Magnitude (Ohm)
    rsData(:, 1) = firstData(:, 4);    % Real part (Ohm)
    xsData(:, 1) = firstData(:, 5);    % Imaginary part (Ohm)
    
    % Read all remaining files
    for i = 2:numFiles
        filePath = fullfile(folderPath, fileList(i).name);
        try
            % Skip the first two lines (metadata and headers)
            data = csvread(filePath, 2, 0);
            
            % Store data into arrays
            freqData(:, i) = data(:, 1);  % Should be the same for all files
            phaseData(:, i) = data(:, 2);
            magData(:, i) = data(:, 3);
            rsData(:, i) = data(:, 4);
            xsData(:, i) = data(:, 5);
        catch e
            warning('Error reading file %s: %s', filePath, e.message);
        end
    end
    
    % Calculate averages across all files
    avgFreq = mean(freqData, 2);
    avgPhase = mean(phaseData, 2);
    avgMag = mean(magData, 2);
    avgRs = mean(rsData, 2);
    avgXs = mean(xsData, 2);
    
    fprintf('Finished processing folder: %s\n', folderPath);
end

% Process data from both folders
fprintf('Processing A5 data...\n');
[A5_freq, A5_mag, A5_phase, A5_rs, A5_xs] = processFolder(A5_folder);

fprintf('Processing A8 data...\n');
[A8_freq, A8_mag, A8_phase, A8_rs, A8_xs] = processFolder(A8_folder);

% Create plots
figure('Position', [100, 100, 1200, 800], 'Color', 'white');

% Plot 1: Magnitude
subplot(2, 2, 1);
semilogx(A5_freq, A5_mag, 'b-', 'LineWidth', 1.5);
hold on;
semilogx(A8_freq, A8_mag, 'r-', 'LineWidth', 1.5);
hold off;
grid on;
xlabel('Frequency (Hz)');
ylabel('|Z| (Ohm)');
title('Impedance Magnitude');
legend('A5', 'A8', 'Location', 'best');

% Plot 2: Phase
subplot(2, 2, 2);
semilogx(A5_freq, A5_phase, 'b-', 'LineWidth', 1.5);
hold on;
semilogx(A8_freq, A8_phase, 'r-', 'LineWidth', 1.5);
hold off;
grid on;
xlabel('Frequency (Hz)');
ylabel('Phase (deg)');
title('Impedance Phase');
legend('A5', 'A8', 'Location', 'best');

% Plot 3: Real Part
subplot(2, 2, 3);
semilogx(A5_freq, A5_rs, 'b-', 'LineWidth', 1.5);
hold on;
semilogx(A8_freq, A8_rs, 'r-', 'LineWidth', 1.5);
hold off;
grid on;
xlabel('Frequency (Hz)');
ylabel('Rs (Ohm)');
title('Resistance (Real Part)');
legend('A5', 'A8', 'Location', 'best');

% Plot 4: Imaginary Part
subplot(2, 2, 4);
semilogx(A5_freq, A5_xs, 'b-', 'LineWidth', 1.5);
hold on;
semilogx(A8_freq, A8_xs, 'r-', 'LineWidth', 1.5);
hold off;
grid on;
xlabel('Frequency (Hz)');
ylabel('Xs (Ohm)');
title('Reactance (Imaginary Part)');
legend('A5', 'A8', 'Location', 'best');

% Add a super title
sgtitle('Comparison of Averaged Impedance Data: A5 vs A8', 'FontSize', 14, 'FontWeight', 'bold');

% Save the figure
saveas(gcf, 'A5_vs_A8_impedance_comparison.fig');
saveas(gcf, 'A5_vs_A8_impedance_comparison.png');

fprintf('Analysis complete. Figure saved as A5_vs_A8_impedance_comparison.fig and .png\n'); 