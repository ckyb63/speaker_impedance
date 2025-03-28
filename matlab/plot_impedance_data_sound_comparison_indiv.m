% Script to plot impedance data for different lengths at different sound levels
% This script loads CSV files from the collected_data_mar26 folder,
% averages data across runs for each frequency, and plots the results.

% Clear workspace and command window
clear;
clc;
close all;

% Define the path to the collected_data_mar26 folder
data_folder = fullfile('..', 'collected_data_mar26');

% Define the folders to compare for 40dB
folders_40dB = {
    'A_Open_T21.6C_H26.5pct',    ... Open configuration at 40dB
    'A_Blocked_T21.5C_H26.9pct', ... Blocked configuration at 40dB
    'A_17_T21.7C_H26.3pct'       ... 17mm length at 40dB
};

% Define the folders to compare for 80dB
folders_80dB = {
    'A_Open_T21.3C_H26.6pct',    ... Open configuration at 80dB
    'A_Blocked_T21.3C_H26.5pct', ... Blocked configuration at 80dB
    'A_17_T21.2C_H26.3pct'       ... 17mm length at 80dB
};

% Function to process folders and return averaged data
function [unique_freqs, avg_magnitudes, avg_phases, avg_real_parts, avg_imaginary_parts, avg_humidities, avg_dB, normalized_magnitudes] = process_folders(folders, data_folder)
    % Initialize cell arrays to store processed data
    unique_freqs = cell(1, length(folders));
    avg_magnitudes = cell(1, length(folders));
    avg_phases = cell(1, length(folders));
    avg_real_parts = cell(1, length(folders));
    avg_imaginary_parts = cell(1, length(folders));
    avg_humidities = zeros(1, length(folders));
    avg_dB = zeros(1, length(folders));
    normalized_magnitudes = cell(1, length(folders));
    
    % Process each folder
    for i = 1:length(folders)
        % Get all CSV files in the folder
        folder_path = fullfile(data_folder, folders{i});
        csv_files = dir(fullfile(folder_path, '*.csv'));
        
        % Initialize arrays to store all data
        all_frequencies = [];
        all_magnitudes = [];
        all_phases = [];
        all_real_parts = [];
        all_imaginary_parts = [];
        all_humidities = [];
        all_dB_levels = [];
        
        % Process each CSV file in the folder
        for j = 1:length(csv_files)
            % Read the CSV file
            data = readtable(fullfile(folder_path, csv_files(j).name));
            
            % Extract relevant columns
            all_frequencies = [all_frequencies; data.Frequency_Hz_];
            all_magnitudes = [all_magnitudes; data.Trace_Z__Ohm_];
            all_phases = [all_phases; data.Trace__deg_];
            all_real_parts = [all_real_parts; data.TraceRs_Ohm_];
            all_imaginary_parts = [all_imaginary_parts; data.TraceXs_Ohm_];
            all_humidities = [all_humidities; data.Humidity___];
            all_dB_levels = [all_dB_levels; data.SmoothedDBA];
        end
        
        % Average the data across runs
        [unique_freqs{i}, ~, freq_idx] = unique(all_frequencies);
        avg_magnitudes{i} = accumarray(freq_idx, all_magnitudes) ./ accumarray(freq_idx, 1);
        avg_phases{i} = accumarray(freq_idx, all_phases) ./ accumarray(freq_idx, 1);
        avg_real_parts{i} = accumarray(freq_idx, all_real_parts) ./ accumarray(freq_idx, 1);
        avg_imaginary_parts{i} = accumarray(freq_idx, all_imaginary_parts) ./ accumarray(freq_idx, 1);
        avg_humidities(i) = mean(all_humidities);
        avg_dB(i) = mean(all_dB_levels);
        
        % Find index corresponding to 50 Hz for normalization
        [~, idx50Hz] = min(abs(unique_freqs{i} - 50));
        nominalImpedance = avg_magnitudes{i}(idx50Hz);
        normalized_magnitudes{i} = avg_magnitudes{i} / nominalImpedance;
    end
end

% Check if all folders exist
for i = 1:length(folders_40dB)
    folder_path = fullfile(data_folder, folders_40dB{i});
    if ~exist(folder_path, 'dir')
        error('40dB folder does not exist: %s\nCheck if the path is correct.', folder_path);
    end
end

for i = 1:length(folders_80dB)
    folder_path = fullfile(data_folder, folders_80dB{i});
    if ~exist(folder_path, 'dir')
        error('80dB folder does not exist: %s\nCheck if the path is correct.', folder_path);
    end
end

% Process both sets of folders
[unique_freqs_40, avg_magnitudes_40, avg_phases_40, avg_real_parts_40, avg_imaginary_parts_40, avg_humidities_40, avg_dB_40, normalized_magnitudes_40] = process_folders(folders_40dB, data_folder);
[unique_freqs_80, avg_magnitudes_80, avg_phases_80, avg_real_parts_80, avg_imaginary_parts_80, avg_humidities_80, avg_dB_80, normalized_magnitudes_80] = process_folders(folders_80dB, data_folder);

% Create labels with actual dB levels
labels_40dB = cell(1, length(folders_40dB));
labels_80dB = cell(1, length(folders_80dB));
for i = 1:length(folders_40dB)
    labels_40dB{i} = sprintf('Open (%.1f dB)', avg_dB_40(1));
    labels_80dB{i} = sprintf('Open (%.1f dB)', avg_dB_80(1));
end
for i = 2:length(folders_40dB)
    labels_40dB{i} = sprintf('Blocked (%.1f dB)', avg_dB_40(2));
    labels_80dB{i} = sprintf('Blocked (%.1f dB)', avg_dB_80(2));
end
for i = 3:length(folders_40dB)
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
for i = 1:length(folders_40dB)
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
for i = 1:length(folders_40dB)
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
for i = 1:length(folders_40dB)
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
for i = 1:length(folders_40dB)
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
saveas(gcf, 'figures/Impedance_data_comparison_mar26_40dB.fig');
saveas(gcf, 'figures/Impedance_data_comparison_mar26_40dB.png');

% Create plots for 80dB measurements
figure('Position', [100, 100, 1200, 800], 'Color', 'white');

% Plot 1: Magnitude
subplot(2, 2, 1);
hold on;
for i = 1:length(folders_80dB)
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
for i = 1:length(folders_80dB)
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
for i = 1:length(folders_80dB)
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
for i = 1:length(folders_80dB)
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
saveas(gcf, 'figures/Impedance_data_comparison_mar26_80dB.fig');
saveas(gcf, 'figures/Impedance_data_comparison_mar26_80dB.png');

% Create combined plot with all measurements
figure('Position', [100, 100, 1200, 800], 'Color', 'white');

% Plot 1: Magnitude
subplot(2, 2, 1);
hold on;
for i = 1:length(folders_40dB)
    semilogx(unique_freqs_40{i}, avg_magnitudes_40{i}, colors_40dB{i}, 'LineWidth', 1.5);
end
for i = 1:length(folders_80dB)
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
for i = 1:length(folders_40dB)
    semilogx(unique_freqs_40{i}, normalized_magnitudes_40{i}, colors_40dB{i}, 'LineWidth', 1.5);
end
for i = 1:length(folders_80dB)
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
for i = 1:length(folders_40dB)
    semilogx(unique_freqs_40{i}, avg_phases_40{i}, colors_40dB{i}, 'LineWidth', 1.5);
end
for i = 1:length(folders_80dB)
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
for i = 1:length(folders_40dB)
    plot(avg_real_parts_40{i}, avg_imaginary_parts_40{i}, colors_40dB{i}, 'LineWidth', 1.5);
end
for i = 1:length(folders_80dB)
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
saveas(gcf, 'figures/Impedance_data_comparison_mar26_combined.fig');
saveas(gcf, 'figures/Impedance_data_comparison_mar26_combined.png');

% Print the actual dB levels
fprintf('\nActual dB levels measured:\n');
fprintf('40dB measurements:\n');
for i = 1:length(folders_40dB)
    fprintf('- %s: %.1f dB\n', folders_40dB{i}, avg_dB_40(i));
end
fprintf('\n80dB measurements:\n');
for i = 1:length(folders_80dB)
    fprintf('- %s: %.1f dB\n', folders_80dB{i}, avg_dB_80(i));
end

fprintf('\nAnalysis complete. Figures saved as:\n');
fprintf('- Impedance_data_comparison_mar26_%ddB.png\n', avg_dB_40(1));
fprintf('- Impedance_data_comparison_mar26_%ddB.png\n', avg_dB_80(1));
fprintf('- Impedance_data_comparison_mar26_combined.png\n'); 