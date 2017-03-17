
% initialization 
load('DataB.mat');
Label3=(gnd==3);
Fea3=fea(Label3,:);
neighbourNum=5;
projDimension=4;
[r,c] = size(fea);

% get digitImage, each row of Fea3(1* 784) corresponds to a 28 * 28 image
digitImage=reshape(Fea3',28,28,size(Fea3,1));
scale=0.05;
skip=1;

% solution to quetion 1 using LLE 
% refer to http://www.cs.nyu.edu/~roweis/lle/code.html
LLE_Matrix3 = lle(Fea3',neighbourNum,projDimension)';
figure(1);
plotImages(digitImage, LLE_Matrix3(:,1:2),scale,skip);

% solution to quetion 2 using ISOMAP
% refer to http://isomap.stanford.edu/
addpath 'isomap';
%addpath(genpath('isomap'));

D = L2_distance(fea', fea', 1);
options.dims = 1:4;
[Y, R, E] = Isomap(D, 'k', 5, options);
fea3 = Y.coords{4,1}';
threes3 = fea3(gnd==3, 3:4);
figure;
plotImages(digitImage, threes3, 0.05, 1);



%% Solution to question3: Naive Bayes Classification using LLE projected matrix
LLE_Fea = lle(fea',neighbourNum,projDimension)';
for i=1:50
    p = randperm(r);
    split = floor(r*0.7); 
    PredictClass1 = classify(LLE_Fea(p(split+1:end),:),LLE_Fea(p(1:split),:),gnd(p(1:split),:),'diaglinear');
    error1(i) = sum(PredictClass1 ~= gnd(p(split+1:end),:));
    properError1(i) = (error1(i) / r)*100;
end

avgProperError1 = mean(properError1);
fprintf('Error Rate of Naive Bayes Classifier on LLE Reduction: %f\n',avgProperError1);



%% Solution to question 3: Naive Bayes Classification using ISOMAP projected matrix

Iso_Fea = Y.coords{4,1}';

for i=1:50
    p = randperm(r);
    split = floor(r*0.7); 
    PredictClass2 = classify(Iso_Fea(p(split+1:end),:),Iso_Fea(p(1:split),:),gnd(p(1:split),:),'diaglinear');
    error2(i) = sum(PredictClass2 ~= gnd(p(split+1:end),:));
    properError2(i) = (error2(i) / r)*100;
end

avgProperError2 = mean(properError2);
fprintf('Error Rate of Naive Bayes Classifier on ISOMAP Reduction: %f\n',avgProperError2);



%% Solution to question 3: Naive Bayes Classification using PCA projected matrix

% [pc,score,latent] = princomp(fea);
addpath(genpath('drtoolbox'));
[PCA_Fea, mapping] = compute_mapping([gnd fea], 'PCA', 4);
%PCA_Fea=fea*pc(:,1:4);
for i=1:50
    p = randperm(r);
    split = floor(r*0.7); 
    PredictClass3 = classify(PCA_Fea(p(split+1:end),:),PCA_Fea(p(1:split),:),gnd(p(1:split),:),'diaglinear');
    error3(i) = sum(PredictClass3 ~= gnd(p(split+1:end),:));
    properError3(i) = (error3(i) / r)*100;
end

avgProperError3 = mean(properError3);
fprintf('Error Rate of Naive Bayes Classifier on PCA Reduction: %f\n',avgProperError3);



%% Solution to question 3: Naive Bayes Classification using LDA projected matrix

LDA_Fea = compute_mapping([gnd,fea],'LDA',4);
for i=1:50
    p = randperm(r);
    split = floor(r*0.7); 
    PredictClass4 = classify(LDA_Fea(p(split+1:end),:),LDA_Fea(p(1:split),:),gnd(p(1:split),:),'diaglinear');
    error4(i) = sum(PredictClass4 ~= gnd(p(split+1:end),:));
    properError4(i) = (error4(i) / r)*100;
end

avgProperError4 = mean(properError4);
fprintf('Error Rate of Naive Bayes Classifier on LDA Reduction: %f\n',avgProperError4);
rmpath(genpath('drtoolbox'));


