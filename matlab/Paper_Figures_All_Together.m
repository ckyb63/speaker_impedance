close all
clc
clear

% Define the main folder path and group identifiers 
% ****** Change the folder path to fit the folder location.
mainFolderPath = 'C:\Users\maxch\Documents\Purdue Files\Audio Research\Collected_Data_Sep16';
groups = {'A', 'B', 'C', 'D'};  % Group identifiers for different sets

% Structure to store data for each group
allGroupData = struct();

% Loop over each group (A, B, C, D)
for g = 1:length(groups)
    group = groups{g};
    
    % Define subfolders for the current group
    subfolders = { ...
        ['\' group '\' group '_Blocked'], ...
        ['\' group '\' group '_5'], ...
        ['\' group '\' group '_8'], ...
        ['\' group '\' group '_9'], ...
        ['\' group '\' group '_11'], ...
        ['\' group '\' group '_14'], ...
        ['\' group '\' group '_17'], ...
        ['\' group '\' group '_20'], ...
        ['\' group '\' group '_23'], ...
        ['\' group '\' group '_24'], ...
        ['\' group '\' group '_26'], ...
        ['\' group '\' group '_29'], ...
        ['\' group '\' group '_39'], ...
        ['\' group '\' group '_Open'] ...
    };

    % Initialize structure to store averaged data for the current group
    avgData = struct();
    
    % Track processed files for progress bar
    fileCounter = 0;
    totalFiles = sum(cellfun(@(folder) length(dir(fullfile(mainFolderPath, folder, '*.csv'))), subfolders));
    hWaitbar = waitbar(0, sprintf('Loading files for group %s...', group), 'Name', sprintf('Loading Set %s', group), 'WindowStyle','alwaysontop');
    
    % Read frequency data once from the first file in the first subfolder
    firstSubfolderPath = fullfile(mainFolderPath, subfolders{1});
    firstFileList = dir(fullfile(firstSubfolderPath, '*.csv'));
    if isempty(firstFileList)
        error('No CSV files found in the first subfolder.');
    end
    firstFile = readtable(fullfile(firstSubfolderPath, firstFileList(1).name));
    frequency = firstFile{:, 1};  % Use this frequency for all subfolders
    
    % Find index corresponding to 50 Hz
    [~, idx50Hz] = min(abs(frequency - 50));

    % Loop through each subfolder to accumulate and average data
    for s = 1:length(subfolders)
        subfolderPath = fullfile(mainFolderPath, subfolders{s});
        fileList = dir(fullfile(subfolderPath, '*.csv'));

        % Extract the subfolder name and prepare the label
        [~, folderName] = fileparts(subfolders{s});
        if ~isempty(regexp(folderName, '\d+', 'once'))
            label = [folderName(3:end), ' mm'];  % Append "mm" to numeric parts
        else
            label = strrep(folderName, '_', ' ');  % Replace "_" for non-numeric labels
        end

        % Initialize accumulators
        totalPhase = zeros(size(frequency));
        totalImpedanceMagnitude = zeros(size(frequency));
        totalRs = zeros(size(frequency));
        totalXs = zeros(size(frequency));

        % Loop through each file in the subfolder
        for k = 1:length(fileList)
            fileCounter = fileCounter + 1;
            waitbar(fileCounter / totalFiles, hWaitbar, sprintf('Loading file %d of %d...', fileCounter, totalFiles));

            % Read CSV file
            data = readtable(fullfile(subfolderPath, fileList(k).name));
            
            % Ensure frequency alignment
            if any(data{:, 1} ~= frequency)
                error('Frequency mismatch found in file: %s', fileList(k).name);
            end

            % Accumulate each column by frequency point
            totalPhase = totalPhase + data{:, 2};
            totalImpedanceMagnitude = totalImpedanceMagnitude + data{:, 3};
            totalRs = totalRs + data{:, 4};
            totalXs = totalXs + data{:, 5};
        end

        % Calculate averages
        numFiles = length(fileList);
        avgData(s).frequency = frequency;
        avgData(s).phase = totalPhase / numFiles;
        avgData(s).impedanceMagnitude = totalImpedanceMagnitude / numFiles;
        avgData(s).Rs = totalRs / numFiles;
        avgData(s).Xs = totalXs / numFiles;
        avgData(s).label = label;

        % Normalize impedance magnitude to 50 Hz
        nominalImpedance = avgData(s).impedanceMagnitude(idx50Hz);
        avgData(s).impedanceMagnitude = avgData(s).impedanceMagnitude / nominalImpedance;
    end

    % Close progress bar
    close(hWaitbar);

    % Save each group’s data into the main structure
    allGroupData.(sprintf('data_%s', group)) = avgData;
end

% Save each group's data into separate variables in the workspace
data_A = allGroupData.data_A;
data_B = allGroupData.data_B;
data_C = allGroupData.data_C;
data_D = allGroupData.data_D;

%% Now you can plot all groups together or individually using these variables
close all
clc

% Define the main color map as a deeper gradient of blue for each unique length
% numLengths = length(allGroupData.data_A); % Assuming all groups have the same lengths
% colorMap = [linspace(0.3, 0.5, numLengths)', ones(numLengths, 1) * 0.7, ones(numLengths, 1) * 0.9]; % Deeper blue gradient
% 
% % Convert HSV to RGB for the blue gradient
% colorMap = hsv2rgb(colorMap);

% Generate color palette for the plots
numLengths = length(allGroupData.data_A);
hueRange = linspace(210/360, 270/360, numLengths);
saturationRange = linspace(0.6, 1, numLengths);
valueRange = linspace(0.5, 1, numLengths);
hsvColors = [hueRange', saturationRange', valueRange'];
colorMap = hsv2rgb(hsvColors);

% Define a cell array to store labels for the legend only once
uniqueLabels = cell(numLengths, 1);

% ---- Plot Frequency vs Normalized |Z| with shared legend ----
figure("Name","Frequency vs |Z|",'WindowState', 'maximized');
for g = 1:length(groups)
    % Get the data for the current group
    data = allGroupData.(sprintf('data_%s', groups{g}));
    
    subplot(2, 2, g);
    hold on;
    for s = 1:numLengths
        % Strip "A" from Blocked and Open labels for the legend
        label = data(s).label;
        label = strrep(label, 'A', '');  % Remove "A" from labels
        
        semilogx(data(s).frequency, data(s).impedanceMagnitude, '-', ...
                 'DisplayName', label, ...
                 'Color', colorMap(s, :), 'LineWidth', 1);
        
        % Store labels only once for creating a single legend
        if g == 1
            uniqueLabels{s} = label;
        end
    end
    xlabel('Frequency (Hz)');
    set(gca, 'XScale', 'log');
    xlim([20, 20000]);
    ylabel('|Z| / |Z|_{50Hz}');
    title(sprintf('Group %s - Frequency vs |Z|', groups{g}));
    hold off;
end
sgtitle('Frequency vs Nominal |Z| at 50 Hz - All Groups');

% Create shared legend
legend(uniqueLabels, 'Location', 'eastoutside', 'Orientation', 'vertical', 'FontSize', 15, 'Position', [0.9 0.15 0.1 0.7]);

% Save the figure as a PNG
saveas(gcf, 'Frequency_vs_Normalized_Z.png');

% ---- Plot Frequency vs Phase with shared legend ----
figure("Name","Frequency vs Phase",'WindowState', 'maximized');
for g = 1:length(groups)
    % Get the data for the current group
    data = allGroupData.(sprintf('data_%s', groups{g}));
    
    subplot(2, 2, g);
    hold on;
    for s = 1:numLengths
        % Strip "A" from Blocked and Open labels for the legend
        label = data(s).label;
        label = strrep(label, 'A', '');  % Remove "A" from labels
        
        semilogx(data(s).frequency, data(s).phase, '-', ...
                 'DisplayName', label, ...
                 'Color', colorMap(s, :), 'LineWidth', 1);
    end
    xlabel('Frequency (Hz)');
    set(gca, 'XScale', 'log');
    xlim([20, 20000]);
    ylabel('Phase (Degrees)');
    title(sprintf('Group %s - Frequency vs Phase', groups{g}));
    hold off;
end
sgtitle('Frequency vs Phase - All Groups');
legend(uniqueLabels, 'Location', 'eastoutside', 'Orientation', 'vertical', 'FontSize', 15, 'Position', [0.9 0.15 0.1 0.7]);
% Save the figure as a PNG
saveas(gcf, 'Frequency_vs_Phase.png');

% ---- Plot Rs vs Xs with shared legend ----
figure("Name","Rs vs Xs",'WindowState', 'maximized');
for g = 1:length(groups)
    % Get the data for the current group
    data = allGroupData.(sprintf('data_%s', groups{g}));
    
    subplot(2, 2, g);
    hold on;
    for s = 1:numLengths
        % Strip "A" from Blocked and Open labels for the legend
        label = data(s).label;
        label = strrep(label, 'A', '');  % Remove "A" from labels
        
        plot(data(s).Rs, data(s).Xs, '-', ...
             'DisplayName', label, ...
             'Color', colorMap(s, :), 'LineWidth', 1);
    end
    xlabel('Rs (Ohm)');
    ylabel('Xs (Ohm)');
    title(sprintf('Group %s - Rs vs Xs', groups{g}));
    hold off;
end
sgtitle('Rs vs Xs - All Groups');
legend(uniqueLabels, 'Location', 'eastoutside', 'Orientation', 'vertical', 'FontSize', 15, 'Position', [0.9 0.15 0.1 0.7]);
% Save the figure as a PNG
saveas(gcf, 'Rs_vs_Xs.png');