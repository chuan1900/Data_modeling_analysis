%% Initialization 
clear; close all; clc;
load('DataB.mat');

%% Compute the eigenvectors and eigenvalues 
[m, n] = size(fea);
fea = fea - ones(m,1) * mean(fea);
Sigma = 1/m .* fea' * fea;
[eigVector, eigValue] = eig(Sigma);

%% Plot two dimensional representation of data 
figure(1);
[coeff,score,latent,tsquared,explained,mu] = pca(fea);
fea_pca12 = fea * coeff(:, 1:2);
scatter(fea_pca12(:,1), fea_pca12(:,2), 20, gnd, 'filled');
xlabel('1st component');
ylabel('2nd component');
title('Representation of data points based on 1st/2nd PC');

width = 8; height = 6;
set(gcf, 'Units', 'Inches', 'Position', [0, 0, width, height],...
    'PaperUnits', 'Inches', 'PaperSize', [width, height])
saveas(gcf, 'pca12.png');

figure(2);

fea_pca56 = fea * coeff(:, 5:6);
scatter(fea_pca56(:,1), fea_pca56(:,2), 20, gnd, 'filled');
title('Representation of data points based on 5th/6th PC');
xlabel('5th component');
ylabel('6th component');
set(gcf, 'Units', 'Inches', 'Position', [0, 0, width, height],...
    'PaperUnits', 'Inches', 'PaperSize', [width, height])
saveas(gcf, 'pca56.png');

%% Naive Bayes Classfier
NBCset = [2, 4, 10, 30, 60, 200, 500, 784];
l = length(NBCset);
class_error = ones(1,l);
retain_variance = ones(1,l);

for i =1 : l,
    fea_pca = fea * coeff(:, 1:NBCset(i));
%     rp = randperm(m);
%     fea_pca = fea_pca(rp,:);
%     gnd = gnd(rp,:);
%     fea_pca_train = fea_pca(1:0.9*m,:);
%     fea_pca_test = fea_pca(0.9*m+1:m,:);
%     gnd_train = gnd(1:0.9*m,:);  % training set 90% 
%     gnd_test = gnd(0.9*m+1:m,:); % test set 10%
%     prediction = classify(fea_pca_test,...
%         fea_pca_train,gnd_train,'diaglinear');
    [prediction,class_error(i)] = classify(...
        fea_pca,fea_pca, gnd);
%    class_error(i,1) = sum(prediction ~= gnd_test)/length(gnd);    
%     retain_variance(i,1) = sum(latent(1:NBCset(i)))/sum(latent);
    retain_variance(1,i) = ...
        sum(sum(eigValue(1:NBCset(i),1:NBCset(i))))/...
        sum(sum(eigValue));    
end

figure(3);
x = 1:numel(NBCset);
y = [retain_variance;class_error]';
bar(x,y)
set(gca,'XTick', x);
set(gca,'XTickLabel', NBCset);
xlabel('number of components');
ylabel('Class Error & Retained Variance');
title('RetainVariance vs ClassError');

set(gcf, 'Units', 'Inches', 'Position', [0, 0, width, height],...
    'PaperUnits', 'Inches', 'PaperSize', [width, height])
saveas(gcf, 'VarianceVsClass.png');

%% Linear Discriminant Analysis 
addpath(genpath('drtoolbox'));
[feaLDA, mapping] = compute_mapping([gnd fea], 'LDA', 2);
% [feaLDA, mapping] = lda(fea, gnd, 2)

figure(4);
scatter(feaLDA(:,1), feaLDA(:,2), 20, gnd, 'filled');
xlabel('1st component');
ylabel('2nd component');
title('LDA Reduction');
rmpath(genpath('drtoolbox'));

set(gcf, 'Units', 'Inches', 'Position', [0, 0, width, height],...
    'PaperUnits', 'Inches', 'PaperSize', [width, height])
saveas(gcf, 'LDAReduction.png');
