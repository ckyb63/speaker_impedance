% Script to plot impedance data for a given speaker and length measurements
% This script loads CSV files from the Analog Discovery/Collected_Data folder,
% averages data across 100 runs for each condition, and plots the results.

%NOTE: This script plots data without environmental measurements.

% Author: Max Chen, v25.03

% Clear workspace and command window
clear;
clc;
close all;

% Variables to make the script more flexible
speaker = 'B';
lengths = '20';

legend1 = 'Ambient';
legend2 = 'Noisy';

folder_1_name = 'B_20';
folder_2_name = 'B_23';

% Define paths to data folders - using filesep for cross-platform compatibility, 2 methods to look for the data folder
fprintf('Attempting to locate data folders...\n');

% Option 1: Direct path from current directory
baseFolderOption1 = fullfile('Analog Discovery', 'Collected_Data');

% Option 2: Path from one directory up
baseFolderOption2 = fullfile('..', 'Analog Discovery', 'Collected_Data');

% Try each path option
if exist(baseFolderOption1, 'dir')
    baseFolder = baseFolderOption1;
    fprintf('Found data using path option 1: %s\n', baseFolder);
elseif exist(baseFolderOption2, 'dir')
    baseFolder = baseFolderOption2;
    fprintf('Found data using path option 2: %s\n', baseFolder);
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
folder_1 = fullfile(baseFolder, folder_1_name);
folder_2 = fullfile(baseFolder, folder_2_name);

% Check if the paths exist
fprintf('Checking paths...\n');
if ~exist(baseFolder, 'dir')
    error('Base folder does not exist: %s\nCheck if the path is correct.', baseFolder);
end

if ~exist(folder_1, 'dir')
    error(folder_1_name + ' folder does not exist: %s\nCheck if the path is correct.', folder_1);
end

if ~exist(folder_2, 'dir')
    error(folder_2_name + ' folder does not exist: %s\nCheck if the path is correct.', folder_2);
end

% Print folder contents to help diagnose issues
fprintf('Contents of base folder: %s\n', baseFolder);
dirContents = dir(baseFolder);
for i = 1:length(dirContents)
    fprintf('  %s\n', dirContents(i).name);
end

fprintf('Contents of %s folder: %s\n', folder_1_name, folder_1);
dirContentsA5 = dir(folder_1);
for i = 1:length(dirContentsA5)
    fprintf('  %s\n', dirContentsA5(i).name);
end

fprintf('Contents of %s folder: %s\n', folder_2_name, folder_2);
dirContentsA8 = dir(folder_2);
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
fprintf('Processing %s data...\n', folder_1_name);
[freq_1, mag_1, phase_1, rs_1, xs_1] = processFolder(folder_1);

fprintf('Processing %s data...\n', folder_2_name);
[freq_2, mag_2, phase_2, rs_2, xs_2] = processFolder(folder_2);

% Create plots
figure('Position', [100, 100, 1200, 800], 'Color', 'white');

% Plot 1: Impedance Magnitude
subplot(2, 2, 1);
semilogx(freq_1, mag_1, 'b-', 'LineWidth', 1.5);
hold on;
semilogx(freq_2, mag_2, 'r-', 'LineWidth', 1.5);
hold off;
grid on;
xlabel('Frequency (Hz)');
ylabel('|Z| (Ohm)');
title('Impedance Magnitude');
legend(legend1, legend2, 'Location', 'best');

% Plot 2: Impedance Phase
subplot(2, 2, 2);
semilogx(freq_1, phase_1, 'b-', 'LineWidth', 1.5);
hold on;
semilogx(freq_2, phase_2, 'r-', 'LineWidth', 1.5);
hold off;
grid on;
xlabel('Frequency (Hz)');
ylabel('Phase (deg)');
title('Impedance Phase');
legend(legend1, legend2, 'Location', 'best');

% Plot 3: Real Part
subplot(2, 2, 3);
semilogx(freq_1, rs_1, 'b-', 'LineWidth', 1.5);
hold on;
semilogx(freq_2, rs_2, 'r-', 'LineWidth', 1.5);
hold off;
grid on;
xlabel('Frequency (Hz)');
ylabel('Rs (Ohm)');
title('Resistance (Real Part)');
legend(legend1, legend2, 'Location', 'best');

% Plot 4: Imaginary Part
subplot(2, 2, 4);
semilogx(freq_1, xs_1, 'b-', 'LineWidth', 1.5);
hold on;
semilogx(freq_2, xs_2, 'r-', 'LineWidth', 1.5);
hold off;
grid on;
xlabel('Frequency (Hz)');
ylabel('Xs (Ohm)');
title('Reactance (Imaginary Part)');
legend(legend1, legend2, 'Location', 'best');

% Add a super for the four plots title
sgtitle(sprintf('Comparison of Averaged Impedance Data: %s vs %s at %s  %s mm', legend1, legend2, speaker, lengths), 'FontSize', 14, 'FontWeight', 'bold');

saveas(gcf, sprintf('figures/%s_vs_%s_impedance_comparison_%s_%s.fig', legend1, legend2, speaker, lengths));

% Save the figure as a PNG, it includes the variable to be compared and the speaker, length used to collect the data.
saveas(gcf, sprintf('figures/%s_vs_%s_impedance_comparison_%s_%s.png', legend1, legend2, speaker, lengths));
fprintf('Analysis complete. Figure saved as %s_vs_%s_impedance_comparison_%s_%s.png\n', legend1, legend2, speaker, lengths); 