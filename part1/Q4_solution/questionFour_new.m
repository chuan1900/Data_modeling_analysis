%% Initialization 
clear; close all; clc;
load('DataC.mat');

% m is the number of samples
% n is the number of features
[m,n] = size(fea);

%% Min-Max Normalization and Seperate the training and test set
max_min = 1 ./ (max(fea) - min(fea));
fea = (fea - ones(m,1) * min(fea)) .* (ones(m,1) * max_min);
rp = randperm(m);
fea = fea(rp,:);
gnd = gnd(rp,:);
fea_train = fea(1:0.7*m,:);
fea_test = fea(0.7*m+1:m,:);
gnd_train = gnd(1:0.7*m,:);
gnd_test = gnd(0.7*m+1:m,:);

initFeas = (1:n);
num_select = 12;
%% SFS and sum of squared Euclidean distances
tic;
c = cvpartition(gnd,'k',10);
opts = statset('display','off');
fun1 = @(fea_train,gnd_train, fea_test,gnd_test)...
    ((-1)*sqrEuclidDist([fea_train;fea_test],[gnd_train;gnd_test]));
[inmodel1,history1] = sequentialfs(fun1,fea,gnd,...
    'cv',c,'direction','forward','nfeatures',num_select,'options',opts)
selFeas1 = initFeas(inmodel1==1);
toc;
FilterSFS = selFeas1;
FilterSFS_time = toc;

%% SFS and Naive Bayes classifier
tic;
c = cvpartition(gnd,'k',10);
opts = statset('display','off');
fun2 = @(fea_train,gnd_train, fea_test,gnd_test)...
    (sum(gnd_test ~= classify(fea_test, fea_train,...
    gnd_train,'diaglinear')));  
[inmodel2,history2] = sequentialfs(fun2,fea,gnd,...
    'cv',c,'direction','forward','nfeatures',num_select,'options',opts)
selFeas2 = initFeas(inmodel2==1);
WrapSFS_time = toc;
WrapSFS = selFeas2;

%% SBS and Naive Bayes classifier
tic;
c = cvpartition(gnd,'k',10);
opts = statset('display','off');
fun3 = @(fea_train,gnd_train, fea_test,gnd_test)...
    (sum(gnd_test ~= classify(fea_test, fea_train,...
    gnd_train,'diaglinear')));  
[inmodel3,history3] = sequentialfs(fun3,fea,gnd,...
    'cv',c,'direction','backward','nfeatures',num_select,'options',opts)
selFeas3 = initFeas(inmodel3==1);
toc;
WrapSBS_time = toc;
WrapSBS = selFeas3;


%% accuracy and running time 
% Average accuracy and running time of FilterSFS
FilterSFS_Fea = fea(:,FilterSFS);
tic;
corr_FilterSFS = Bayes_fun(FilterSFS_Fea,gnd, 5);
toc;
corr_FilterSFS_time = toc;

% Average accuracy and running time of WrapSFS
WrapSFS_Fea = fea(:,WrapSFS);
tic;
corr_WrapSFS = Bayes_fun(WrapSFS_Fea,gnd, 5);
toc;
corr_WrapSFS_time = toc;

% Average accuracy and running time of WrapSBS
WrapSBS_Fea = fea(:,WrapSBS);
tic;
corr_WrapSBS = Bayes_fun(WrapSBS_Fea,gnd, 5);
toc;
corr_WrapSBS_time = toc;

% Average accuracy and running time of All 21 features
tic;
corr_ALL = Bayes_fun(fea,gnd, 5);
toc;
corr_ALL_time = toc;