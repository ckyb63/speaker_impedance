% Script to plot impedance data for different lengths at different sound levels
% This script loads CSV files from the ML_Dataset_326 folder,
% averages data across runs for each frequency, and plots the results.

% Clear workspace and command window
clear;
clc;
close all;

% Define the path to the ML_Dataset_326 folder
ml_folder = fullfile('..', 'ML_Dataset_327');

% Define the files to compare for 40dB
files_40dB = {
    'D_Open_T21.5_H35.2_dBA37.6_All.csv',    ... Open configuration at 40dB
    'D_Blocked_T21.6_H35.5_dBA38.7_All.csv', ... Blocked configuration at 40dB
    'D_17_T21.6_H35.7_dBA39.3_All.csv'       ... 17mm length at 40dB
};

% Define the files to compare for 80dB
files_80dB = {
    'D_Open_T21.6_H36.2_dBA79.8_All.csv',    ... Open configuration at 80dB
    'D_Blocked_T21.5_H36.1_dBA79.6_All.csv', ... Blocked configuration at 80dB
    'D_17_T21.5_H36.0_dBA79.4_All.csv'       ... 17mm length at 80dB
};

% Function to process files and return averaged data
function [unique_freqs, avg_magnitudes, avg_phases, avg_real_parts, avg_imaginary_parts, avg_humidities, avg_dB, normalized_magnitudes] = process_files(files, ml_folder)
    % Initialize cell arrays to store processed data
    unique_freqs = cell(1, length(files));
    avg_magnitudes = cell(1, length(files));
    avg_phases = cell(1, length(files));
    avg_real_parts = cell(1, length(files));
    avg_imaginary_parts = cell(1, length(files));
    avg_humidities = zeros(1, length(files));
    avg_dB = zeros(1, length(files));
    normalized_magnitudes = cell(1, length(files));
    
    % Process each file
    for i = 1:length(files)
        % Read the CSV file
        data = readtable(fullfile(ml_folder, files{i}));
        
        % Extract relevant columns
        frequencies = data.Frequency_Hz_;
        magnitudes = data.Trace_Z__Ohm_;
        phases = data.Trace__deg_;
        real_parts = data.TraceRs_Ohm_;
        imaginary_parts = data.TraceXs_Ohm_;
        humidities = data.Humidity___;
        dB_levels = data.SmoothedDBA;  % Extract dB levels
        
        % Average the data across runs
        [unique_freqs{i}, ~, freq_idx] = unique(frequencies);
        avg_magnitudes{i} = accumarray(freq_idx, magnitudes) ./ accumarray(freq_idx, 1);
        avg_phases{i} = accumarray(freq_idx, phases) ./ accumarray(freq_idx, 1);
        avg_real_parts{i} = accumarray(freq_idx, real_parts) ./ accumarray(freq_idx, 1);
        avg_imaginary_parts{i} = accumarray(freq_idx, imaginary_parts) ./ accumarray(freq_idx, 1);
        avg_humidities(i) = mean(humidities);
        avg_dB(i) = mean(dB_levels);  % Calculate average dB level
        
        % Find index corresponding to 50 Hz for normalization
        [~, idx50Hz] = min(abs(unique_freqs{i} - 50));
        nominalImpedance = avg_magnitudes{i}(idx50Hz);
        normalized_magnitudes{i} = avg_magnitudes{i} / nominalImpedance;
    end
end

% Check if all files exist
for i = 1:length(files_40dB)
    file_path = fullfile(ml_folder, files_40dB{i});
    if ~exist(file_path, 'file')
        error('40dB file does not exist: %s\nCheck if the path is correct.', file_path);
    end
end

for i = 1:length(files_80dB)
    file_path = fullfile(ml_folder, files_80dB{i});
    if ~exist(file_path, 'file')
        error('80dB file does not exist: %s\nCheck if the path is correct.', file_path);
    end
end

% Process both sets of files
[unique_freqs_40, avg_magnitudes_40, avg_phases_40, avg_real_parts_40, avg_imaginary_parts_40, avg_humidities_40, avg_dB_40, normalized_magnitudes_40] = process_files(files_40dB, ml_folder);
[unique_freqs_80, avg_magnitudes_80, avg_phases_80, avg_real_parts_80, avg_imaginary_parts_80, avg_humidities_80, avg_dB_80, normalized_magnitudes_80] = process_files(files_80dB, ml_folder);

% Create labels with actual dB levels
labels_40dB = cell(1, length(files_40dB));
labels_80dB = cell(1, length(files_80dB));
for i = 1:length(files_40dB)
    labels_40dB{i} = sprintf('Open (%.1f dB)', avg_dB_40(1));
    labels_80dB{i} = sprintf('Open (%.1f dB)', avg_dB_80(1));
end
for i = 2:length(files_40dB)
    labels_40dB{i} = sprintf('Blocked (%.1f dB)', avg_dB_40(2));
    labels_80dB{i} = sprintf('Blocked (%.1f dB)', avg_dB_80(2));
end
for i = 3:length(files_40dB)
    labels_40dB{i} = sprintf('17mm (%.1f dB)', avg_dB_40(3));
    labels_80dB{i} = sprintf('17mm (%.1f dB)', avg_dB_80(3));
end

% Define colors for the plots
colors_40dB = {'b-', 'r-', 'g-'};
colors_80dB = {'b--', 'r--', 'g--'};

% Create plots for 40dB measurements
figure('Position', [100, 100, 1200, 800], 'Color', 'white');

% Plot 1: Magnitude
subplot(2, 2, 1);
hold on;
for i = 1:length(files_40dB)
    semilogx(unique_freqs_40{i}, avg_magnitudes_40{i}, colors_40dB{i}, 'LineWidth', 1.5);
end
hold off;
grid on;
xlabel('Frequency (Hz)');
ylabel('|Z| (Ohm)');
title('Impedance Magnitude Comparison');
set(gca, 'XScale', 'log');

% Plot 2: Normalized Magnitude
subplot(2, 2, 2);
hold on;
for i = 1:length(files_40dB)
    semilogx(unique_freqs_40{i}, normalized_magnitudes_40{i}, colors_40dB{i}, 'LineWidth', 1.5);
end
hold off;
grid on;
xlabel('Frequency (Hz)');
ylabel('|Z| / |Z|_{50Hz}');
title('Normalized Impedance Magnitude Comparison');
set(gca, 'XScale', 'log');

% Plot 3: Phase
subplot(2, 2, 3);
hold on;
for i = 1:length(files_40dB)
    semilogx(unique_freqs_40{i}, avg_phases_40{i}, colors_40dB{i}, 'LineWidth', 1.5);
end
hold off;
grid on;
xlabel('Frequency (Hz)');
ylabel('Phase (deg)');
title('Impedance Phase Comparison');
set(gca, 'XScale', 'log');

% Plot 4: Rs vs Xs
subplot(2, 2, 4);
hold on;
for i = 1:length(files_40dB)
    plot(avg_real_parts_40{i}, avg_imaginary_parts_40{i}, colors_40dB{i}, 'LineWidth', 1.5);
end
hold off;
grid on;
xlabel('Rs (Ohm)');
ylabel('Xs (Ohm)');
title('Rs vs Xs Comparison');

% Add a super title
sgtitle('Comparison of Impedance Data', 'FontSize', 14, 'FontWeight', 'bold');

% Add single legend for the figure
legend(labels_40dB, 'Location', 'eastoutside', 'Orientation', 'vertical', 'FontSize', 12, 'Position', [0.9 0.15 0.1 0.7]);

% Save the 40dB figure
saveas(gcf, 'figures/Impedance_data_comparison_40dB.fig');
saveas(gcf, 'figures/Impedance_data_comparison_40dB.png');

% Create plots for 80dB measurements
figure('Position', [100, 100, 1200, 800], 'Color', 'white');

% Plot 1: Magnitude
subplot(2, 2, 1);
hold on;
for i = 1:length(files_80dB)
    semilogx(unique_freqs_80{i}, avg_magnitudes_80{i}, colors_80dB{i}, 'LineWidth', 1.5);
end
hold off;
grid on;
xlabel('Frequency (Hz)');
ylabel('|Z| (Ohm)');
title('Impedance Magnitude Comparison');
set(gca, 'XScale', 'log');

% Plot 2: Normalized Magnitude
subplot(2, 2, 2);
hold on;
for i = 1:length(files_80dB)
    semilogx(unique_freqs_80{i}, normalized_magnitudes_80{i}, colors_80dB{i}, 'LineWidth', 1.5);
end
hold off;
grid on;
xlabel('Frequency (Hz)');
ylabel('|Z| / |Z|_{50Hz}');
title('Normalized Impedance Magnitude Comparison');
set(gca, 'XScale', 'log');

% Plot 3: Phase
subplot(2, 2, 3);
hold on;
for i = 1:length(files_80dB)
    semilogx(unique_freqs_80{i}, avg_phases_80{i}, colors_80dB{i}, 'LineWidth', 1.5);
end
hold off;
grid on;
xlabel('Frequency (Hz)');
ylabel('Phase (deg)');
title('Impedance Phase Comparison');
set(gca, 'XScale', 'log');

% Plot 4: Rs vs Xs
subplot(2, 2, 4);
hold on;
for i = 1:length(files_80dB)
    plot(avg_real_parts_80{i}, avg_imaginary_parts_80{i}, colors_80dB{i}, 'LineWidth', 1.5);
end
hold off;
grid on;
xlabel('Rs (Ohm)');
ylabel('Xs (Ohm)');
title('Rs vs Xs Comparison');

% Add a super title
sgtitle('Comparison of Impedance Data', 'FontSize', 14, 'FontWeight', 'bold');

% Add single legend for the figure
legend(labels_80dB, 'Location', 'eastoutside', 'Orientation', 'vertical', 'FontSize', 12, 'Position', [0.9 0.15 0.1 0.7]);

% Save the 80dB figure
saveas(gcf, 'figures/Impedance_data_comparison_80dB.fig');
saveas(gcf, 'figures/Impedance_data_comparison_80dB.png');

% Create combined plot with all measurements
figure('Position', [100, 100, 1200, 800], 'Color', 'white');

% Plot 1: Magnitude
subplot(2, 2, 1);
hold on;
for i = 1:length(files_40dB)
    semilogx(unique_freqs_40{i}, avg_magnitudes_40{i}, colors_40dB{i}, 'LineWidth', 1.5);
end
for i = 1:length(files_80dB)
    semilogx(unique_freqs_80{i}, avg_magnitudes_80{i}, colors_80dB{i}, 'LineWidth', 1.5);
end
hold off;
grid on;
xlabel('Frequency (Hz)');
ylabel('|Z| (Ohm)');
title('Impedance Magnitude Comparison');
set(gca, 'XScale', 'log');

% Plot 2: Normalized Magnitude
subplot(2, 2, 2);
hold on;
for i = 1:length(files_40dB)
    semilogx(unique_freqs_40{i}, normalized_magnitudes_40{i}, colors_40dB{i}, 'LineWidth', 1.5);
end
for i = 1:length(files_80dB)
    semilogx(unique_freqs_80{i}, normalized_magnitudes_80{i}, colors_80dB{i}, 'LineWidth', 1.5);
end
hold off;
grid on;
xlabel('Frequency (Hz)');
ylabel('|Z| / |Z|_{50Hz}');
title('Normalized Impedance Magnitude Comparison');
set(gca, 'XScale', 'log');

% Plot 3: Phase
subplot(2, 2, 3);
hold on;
for i = 1:length(files_40dB)
    semilogx(unique_freqs_40{i}, avg_phases_40{i}, colors_40dB{i}, 'LineWidth', 1.5);
end
for i = 1:length(files_80dB)
    semilogx(unique_freqs_80{i}, avg_phases_80{i}, colors_80dB{i}, 'LineWidth', 1.5);
end
hold off;
grid on;
xlabel('Frequency (Hz)');
ylabel('Phase (deg)');
title('Impedance Phase Comparison');
set(gca, 'XScale', 'log');

% Plot 4: Rs vs Xs
subplot(2, 2, 4);
hold on;
for i = 1:length(files_40dB)
    plot(avg_real_parts_40{i}, avg_imaginary_parts_40{i}, colors_40dB{i}, 'LineWidth', 1.5);
end
for i = 1:length(files_80dB)
    plot(avg_real_parts_80{i}, avg_imaginary_parts_80{i}, colors_80dB{i}, 'LineWidth', 1.5);
end
hold off;
grid on;
xlabel('Rs (Ohm)');
ylabel('Xs (Ohm)');
title('Rs vs Xs Comparison');

% Add a super title
sgtitle('Combined Comparison of Impedance Data', 'FontSize', 14, 'FontWeight', 'bold');

% Add single legend for the figure
legend([labels_40dB, labels_80dB], 'Location', 'eastoutside', 'Orientation', 'vertical', 'FontSize', 12, 'Position', [0.9 0.15 0.1 0.7]);

% Save the combined figure
saveas(gcf, 'figures/Impedance_data_comparison_combined.fig');
saveas(gcf, 'figures/Impedance_data_comparison_combined.png');

% Print the actual dB levels
fprintf('\nActual dB levels measured:\n');
fprintf('40dB measurements:\n');
for i = 1:length(files_40dB)
    fprintf('- %s: %.1f dB\n', files_40dB{i}, avg_dB_40(i));
end
fprintf('\n80dB measurements:\n');
for i = 1:length(files_80dB)
    fprintf('- %s: %.1f dB\n', files_80dB{i}, avg_dB_80(i));
end

fprintf('\nAnalysis complete. Figures saved as:\n');
fprintf('- Impedance_data_comparison_%ddB.png\n', avg_dB_40(1));
fprintf('- Impedance_data_comparison_%ddB.png\n', avg_dB_80(1));
fprintf('- Impedance_data_comparison_combined.png\n'); 