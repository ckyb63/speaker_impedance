% Script to average and plot impedance data from two combined CSV files
% This script loads two combined CSV files from the ML_Datasets folder,
% averages data across the 100 runs for each frequency, and plots the results.

clear;
clc;
close all;

% Define the path to the ML_Datasets folder
ml_folder = fullfile('..', 'Analog Discovery', 'ML_Datasets');

% Specify the names of the combined CSV files
combined_file1 = fullfile(ml_folder, 'A_8_T21.8_H80.7_All.csv'); % Update with your actual combined file name
combined_file2 = fullfile(ml_folder, 'A_8_T22.0_H22.0_All.csv'); % Update with your actual combined file name

% Check if the combined files exist
if ~exist(combined_file1, 'file')
    error('Combined file 1 does not exist: %s\nCheck if the path is correct.', combined_file1);
end

if ~exist(combined_file2, 'file')
    error('Combined file 2 does not exist: %s\nCheck if the path is correct.', combined_file2);
end

% Read the combined CSV files
data1 = readtable(combined_file1);
data2 = readtable(combined_file2);

% Extract relevant columns from both files
frequencies1 = data1.Frequency_Hz_; % Assuming the column is named 'Frequency (Hz)'
magnitudes1 = data1.Trace_Z__Ohm_;     % Assuming the column is named 'Trace |Z| (Ohm)'
phases1 = data1.Trace__deg_;      % Assuming the column is named 'Trace θ (deg)'
real_parts1 = data1.TraceRs_Ohm_;     % Assuming the column is named 'Trace Rs (Ohm)'
imaginary_parts1 = data1.TraceXs_Ohm_; % Assuming the column is named 'Trace Xs (Ohm)'
humidities1 = data1.Humidity___;  % Assuming the column is named 'Humidity (%)'

frequencies2 = data2.Frequency_Hz_; % Assuming the column is named 'Frequency (Hz)'
magnitudes2 = data2.Trace_Z__Ohm_;     % Assuming the column is named 'Trace |Z| (Ohm)'
phases2 = data2.Trace__deg_;      % Assuming the column is named 'Trace θ (deg)'
real_parts2 = data2.TraceRs_Ohm_;     % Assuming the column is named 'Trace Rs (Ohm)'
imaginary_parts2 = data2.TraceXs_Ohm_; % Assuming the column is named 'Trace Xs (Ohm)'
humidities2 = data2.Humidity___;  % Assuming the column is named 'Humidity (%)'

% Average the data across the 100 runs for the first dataset
[unique_freqs1, ~, freq_idx1] = unique(frequencies1);
avg_magnitudes1 = accumarray(freq_idx1, magnitudes1) ./ accumarray(freq_idx1, 1);
avg_phases1 = accumarray(freq_idx1, phases1) ./ accumarray(freq_idx1, 1);
avg_real_parts1 = accumarray(freq_idx1, real_parts1) ./ accumarray(freq_idx1, 1);
avg_imaginary_parts1 = accumarray(freq_idx1, imaginary_parts1) ./ accumarray(freq_idx1, 1);
avg_humidities1 = mean(humidities1); % Average humidity for the first dataset

% Average the data across the 100 runs for the second dataset
[unique_freqs2, ~, freq_idx2] = unique(frequencies2);
avg_magnitudes2 = accumarray(freq_idx2, magnitudes2) ./ accumarray(freq_idx2, 1);
avg_phases2 = accumarray(freq_idx2, phases2) ./ accumarray(freq_idx2, 1);
avg_real_parts2 = accumarray(freq_idx2, real_parts2) ./ accumarray(freq_idx2, 1);
avg_imaginary_parts2 = accumarray(freq_idx2, imaginary_parts2) ./ accumarray(freq_idx2, 1);
avg_humidities2 = mean(humidities2); % Average humidity for the second dataset

% Create plots
figure('Position', [100, 100, 1200, 800], 'Color', 'white');

% Plot 1: Magnitude
subplot(2, 2, 1);
semilogx(unique_freqs1, avg_magnitudes1, 'b-', 'LineWidth', 1.5);
hold on;
semilogx(unique_freqs2, avg_magnitudes2, 'r-', 'LineWidth', 1.5);
hold off;
grid on;
xlabel('Frequency (Hz)');
ylabel('|Z| (Ohm)');
title(sprintf('Impedance Magnitude (Avg Humidity: %.2f%% vs %.2f%%)', avg_humidities1, avg_humidities2));
legend(sprintf('%.2f%%', avg_humidities1), sprintf('%.2f%%', avg_humidities2), 'Location', 'best');

% Plot 2: Phase
subplot(2, 2, 2);
semilogx(unique_freqs1, avg_phases1, 'b-', 'LineWidth', 1.5);
hold on;
semilogx(unique_freqs2, avg_phases2, 'r-', 'LineWidth', 1.5);
hold off;
grid on;
xlabel('Frequency (Hz)');
ylabel('Phase (deg)');
title(sprintf('Impedance Phase (Avg Humidity: %.2f%% vs %.2f%%)', avg_humidities1, avg_humidities2));
legend(sprintf('%.2f%%', avg_humidities1), sprintf('%.2f%%', avg_humidities2), 'Location', 'best');

% Plot 3: Real Part
subplot(2, 2, 3);
semilogx(unique_freqs1, avg_real_parts1, 'b-', 'LineWidth', 1.5);
hold on;
semilogx(unique_freqs2, avg_real_parts2, 'r-', 'LineWidth', 1.5);
hold off;
grid on;
xlabel('Frequency (Hz)');
ylabel('Rs (Ohm)');
title(sprintf('Resistance (Real Part) (Avg Humidity: %.2f%% vs %.2f%%)', avg_humidities1, avg_humidities2));
legend(sprintf('%.2f%%', avg_humidities1), sprintf('%.2f%%', avg_humidities2), 'Location', 'best');

% Plot 4: Imaginary Part
subplot(2, 2, 4);
semilogx(unique_freqs1, avg_imaginary_parts1, 'b-', 'LineWidth', 1.5);
hold on;
semilogx(unique_freqs2, avg_imaginary_parts2, 'r-', 'LineWidth', 1.5);
hold off;
grid on;
xlabel('Frequency (Hz)');
ylabel('Xs (Ohm)');
title(sprintf('Reactance (Imaginary Part) (Avg Humidity: %.2f%% vs %.2f%%)', avg_humidities1, avg_humidities2));
legend(sprintf('%.2f%%', avg_humidities1), sprintf('%.2f%%', avg_humidities2), 'Location', 'best');

% Add a super title
sgtitle('Comparison of Averaged Impedance Data from A - 8', 'FontSize', 14, 'FontWeight', 'bold');

% Save the figure
saveas(gcf, 'Averaged_impedance_data_comparison_A8.png');

fprintf('Analysis complete. Figure saved as Averaged_impedance_data_comparison.png\n');