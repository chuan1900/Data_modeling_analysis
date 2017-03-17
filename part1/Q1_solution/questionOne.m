%% Initialization 
clear; close all; clc;

load('DataA.mat');
% m is the number of samples
% n is the number of features
[m,n] = size(fea);

ifTest = true;

%% Dealing with missing values 
% First to count NaN elements for every feature of every sample and check 
% their positions

% Delete the sample with >5 feature NaN and delete them
sample_nan = zeros(m,1);

for i = 1:m,
	sample_nan(i,1) = sum(isnan(fea(i,:)),2);
end

% if ifTest == true,
% 	fprintf(' %d\n', sample_nan);
% 	fprintf('m = %d\n', m);
% end

array_sample_nan = find(sample_nan > 5);
fea(array_sample_nan,:) = [];
sample_nan(array_sample_nan,:) = [];

% Renew m & n
m = size(fea,1);
n = size(fea,2);

% if ifTest == true,
% 	fprintf(' %d\n', sample_nan);
% 	fprintf('m = %d\n', m);
% end

% Delete the feature with >10000 samples NaN and delete them
feature_nan = zeros(n,1);

for i = 1:n,
	feature_nan(i,1) = sum(isnan(fea(:,i)),1);
end

% if ifTest == true,
% 	fprintf(' %d\n', feature_nan);
% 	fprintf('m = %d\n', m);
% end

array_feature_nan = find(feature_nan > 10000);
fea(:, array_feature_nan) = [];
feature_nan(array_feature_nan,:) = [];

% Renew m & n
m = size(fea,1);
n = size(fea,2);

% if ifTest == true,
% 	fprintf(' %d\n', feature_nan);
% 	fprintf('m = %d\n', m);
% end

% Fill the remaining missing data with mean value
mean_value = zeros(n,1);
for i = 1:n,
	without_nan = fea(:,i);
	arr_nan = find(isnan(without_nan));
	without_nan(arr_nan,:) = [];
	mean_value(i,1) = mean(without_nan);
end

% if ifTest == true,
% 	fprintf('mean = ');
% 	fprintf(' %d', mean_value);
% end

for i = 1:n,
	arr_nan = find(isnan(fea(:,i)));
	fea(arr_nan,i) = mean_value(i);
end

% Check if there is any NaN elements left
anyNaN = sum(sum(isnan(fea)));


%% Dealing with outliers 

num_outlr = ones(n,1);
fea_mean = mean(fea);
fea_std = std(fea);

for i=1:n,
    outlr = find((fea(:,i) < fea_mean(:,i)-3*fea_std(:,i)) | ...
        (fea(:,i)>(fea_mean(:,i)+3*fea_std(:,i))));
    fea_temp = fea(:,i);
    fea_temp(outlr) = [];
    fea(outlr, i) = mean(fea_temp);
    num_outlr(i,1) = size(outlr,1);
end

num_outliers = sum(num_outlr);
    
% fprintf('number of outliers is %d.\n', num_outliers);


%% Normalization and Plot 
%Min-Max normalization
norm_mm_fea = zeros(m,n);
max_min = 1 ./ (max(fea) - min(fea));
norm_mm_fea = (fea - ones(m,1) * min(fea)) .* (ones(m,1) * max_min);

% z_score normalization
norm_z_fea = zeros(m,n);
mu_fea = mean(fea);
sigma_fea = std(fea);
norm_z_fea = (fea - ones(m,1) * mu_fea) .* (ones(m,1) * (1 ./ sigma_fea));

% Plot histograms of feature 9 and 24
xbins = 200;
width=10; height=18;
figure(1);
subplot(3,2,1)
hist(fea(:,9),xbins)
title('Feature 9 without normalization');
ylabel('# of samples');

subplot(3,2,2)
hist(fea(:,24), xbins)
ylabel('# of samples');
title('Feature 24 without normalization');

subplot(3,2,3)
hist(norm_mm_fea(:,9), xbins)
ylabel('# of samples');
title('Feature 9 with min-max normalization');

subplot(3,2,4)
hist(norm_mm_fea(:,24), xbins)
ylabel('# of samples');
title('Feature 24 with min-max normalization');

subplot(3,2,5)
hist(norm_z_fea(:,9), xbins)
ylabel('# of samples');
title('Feature 9 with z-score normalization');

subplot(3,2,6)
hist(norm_z_fea(:,24), xbins)
ylabel('# of samples');
title('Feature 24 with z-score normalization');

set(gcf, 'Units', 'Inches', 'Position', [0, 0, width, height],...
    'PaperUnits', 'Inches', 'PaperSize', [width, height])
saveas(gcf, 'Feature9&24.png');

%% Compare and comment on the differences before and after normalization
width = 10;
height = 18;
num_lag = 40;
figure(2);
subplot(3,2,1)
autocorr(fea(:,9),num_lag);
title('Feature 9 without normalization');

subplot(3,2,2)
autocorr(fea(:,24),num_lag);
title('Feature 24 without normalization');

subplot(3,2,3)
autocorr(norm_mm_fea(:,9),num_lag);
title('Feature 9 with min-max normalization');

subplot(3,2,4)
autocorr(norm_mm_fea(:,24),num_lag);
title('Feature 24 with min-max normalization');

subplot(3,2,5)
autocorr(norm_z_fea(:,9),num_lag);
title('Feature 9 with z-score normalization');

subplot(3,2,6)
autocorr(norm_z_fea(:,24),num_lag);
title('Feature 24 with z-score normalization');

set(gcf, 'Units', 'Inches', 'Position', [0, 0, width, height],...
    'PaperUnits', 'Inches', 'PaperSize', [width, height])
saveas(gcf, 'autocorrelation.png');