clear; close all; clc;
load('DataC.mat');
% r is the number of samples/rows
r = size(fea,1);
% c is the number of features/columns
c = size(fea,2);

%% Min-Max Normalization
max_min = 1 ./ (max(fea) - min(fea));
fea = (fea - ones(r,1) * min(fea)) .* (ones(r,1) * max_min);

%% Sequential Forward Selection (SFS) with sum of squared Euclidean distances

tic;
initFeas = (1:c);
% the features to be selected
selFeas = [];
totalSel = 8;
EuDistance = ones(totalSel,c);

fea_label1 = fea(find(gnd==1),:);
fea_label2 = fea(find(gnd==2),:);
fea_label3 = fea(find(gnd==3),:);
[m1,~] = size(fea_label1);
[m2,~] = size(fea_label2);
[m3,~] = size(fea_label3);

for k=1:totalSel,
    for j=1:c,
        l1=0; l2=0; l3=0;
        if(ismember(j, selFeas) == 0),
            uslCols = cat(2, selFeas, initFeas(j));
            for i=1:m1,
                l1 = l1+ sum(sum((ones(m2,1)*fea_label1(i,uslCols)...
                    -fea_label2(:,uslCols)).^2));
            end
            for i=1:m2,
                l2 = l2+ sum(sum((ones(m3,1)*fea_label2(i,uslCols)...
                    -fea_label3(:,uslCols)).^2));
            end
            for i=1:m1,
                l3 = l3+ sum(sum((ones(m3,1)*fea_label1(i,uslCols)...
                    -fea_label3(:,uslCols)).^2));
            end
            EuDistance(k,j) = l1/m1/m2 + l2/m2/m3 + l3/m1/m3;
        end
    end
    [maxValue, index1] = max(EuDistance(k,:));
    selFeas(end+1) = index1;
end

toc;
FilterSFS = selFeas;
FilterSFS_time = toc;


%% SFS with the Naive Bayes classifier
tic;
selFeasBayes = [];
initFeas = 1:c;
totalSel = 10;

for i = 1:totalSel
    naiveBayes = zeros(1,c);
    for j = 1:c
        if(ismember(j, selFeasBayes) == 0)
            % use Bayes_function to implementing Naive Bayes Classifier
            PreClass = Bayes_fun(fea(:,cat(2, selFeasBayes, initFeas(j))),gnd, 10);
            naiveBayes(1,j) = PreClass;
       end
    end

    [maxValue, index] = max(naiveBayes);
    selFeasBayes(end+1) = index;
end
toc;
WrapSFS_time = toc;
WrapSFS = selFeasBayes;

%% zhang's SFS with Naive Bayes Classifier
% iter = 20;
% tic;
% sel2 = ones(iter,8);
% for k = 1:iter
% 
%     selFeasBayes = [];
% 
%     for i = 1:totalSel
%         naiveBayes = zeros(1,c);
%         for j = 1:c
%             if(ismember(j, selFeasBayes) == 0)
%                 % use Bayes_function to implementing Naive Bayes Classifier
%                 PreClass = Bayes_fun(fea(:,cat(2, selFeasBayes, initFeas(j))),gnd, 10);
%                 naiveBayes(1,j) = PreClass;
%             end
%         end
%         [maxValue, index] = max(naiveBayes);
%         selFeasBayes(end+1) = index;
%     end
%     
%     sel2(k,:) = selFeasBayes;
% 
% end
% 
% num_fea_2 = ones(1,c);
% for i=1:c,
%     num_fea_2(1,i) = sum(sum(sel2==i));
% end
% 
% [~, index2] = sort(num_fea_2,'descend');
% selFeas2 = index2(1,1:8);
% 
% WrapSFS_time1 = toc/20;
% WrapSFS1 = selFeas2;

%% SBS with the Naive Bayes classifier
tic;
% the features to be selected and in this section the features are the worst one
selFeasBayes = [];
initFeas = 1:c;
totalSel = 8;

for i = 1: (c - totalSel)
    naiveBayes = zeros(1,c);
    for j = 1:c
        if(ismember(j, selFeasBayes) == 0)
            useFeas = initFeas;
            useFeas(j) = [];
            for elm = selFeasBayes
                useFeas = useFeas(useFeas ~= elm);
            end
            PreClass = Bayes_fun(fea(:,useFeas),gnd, 10);
            naiveBayes(1,j) = PreClass;
       end
    end

    [maxValue, index] = max(naiveBayes);
    selFeasBayes(end+1) = index;
end
remainFeasBayes = initFeas;

% to get the good features
for elm = selFeasBayes
    remainFeasBayes = remainFeasBayes(remainFeasBayes ~= elm);
end
toc;
WrapSBS_time = toc;
WrapSBS = remainFeasBayes;

%% Average Accuracy and Run Time

%%%%
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

% Average accuracy and running time of WrapSFS
% WrapSFS_Fea1 = fea(:,WrapSFS1);
% tic;
% corr_WrapSFS1 = Bayes_fun(WrapSFS_Fea1,gnd, 5);
% toc;
% corr_WrapSFS1_time = toc;

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



% %first half training, last half testing
% p = randperm(r);
% split = r/2; 
% 
% featSubset = fea(:,WrapSFS);
% PredictClass = classify(featSubset(p(split+1:end),:),featSubset(p(1:split),:),gnd(p(1:split),:),'diaglinear');
% error = sum(PredictClass ~= gnd(p(split+1:end),:));
% properError = (error / r)*100;
% properAccuracy1 = 100 - properError;

% featSubset = fea(:,WrapSFS1);
% PredictClass = classify(featSubset(p(split+1:end),:),featSubset(p(1:split),:),gnd(p(1:split),:),'diaglinear');
% error = sum(PredictClass ~= gnd(p(split+1:end),:));
% properError = (error / r)*100;
% properAccuracy2 = 100 - properError;
