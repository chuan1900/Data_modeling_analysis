load('DataD.mat');
for i = 1 : 57
    z_score(:,i) = zscore(fea(:,i));
end;
fea = z_score;
gnd(gnd==-1) = 2; %#transform label "-1" to "2" for classification, this will not affect the result

fea_train = z_score(1:1100, :);  %# first half as training data
fea_test = z_score(1101:2200, :);  %# second half as training data
gnd_train = gnd(1:1100, :);
gnd_test = gnd(1101:2200, :);

%% Dertermine besk k for kNN classification

kNNOpt = kNN(fea_train, gnd_train);
[~,index] = max(kNNOpt(:,2));
best_k = kNNOpt(index,1);
fprintf('Optimal parameter for kNN is:\n');
fprintf(' k = %d\n', best_k);
fprintf(' Average accuracy on training set = %f\n', kNNOpt(index,2));
%# Use best k for kNN train on fea_train and predict on fea_test
kNN_mdl = fitcknn(fea_train, gnd_train, 'NumNeighbors',best_k);
[kNN_label,scorekNN,cost] = predict(kNN_mdl, fea_test);
confusionMatrix = confusionmat(gnd_test,kNN_label);
TP = confusionMatrix(1,1);
TN = confusionMatrix(2,2);
N = sum(sum(confusionMatrix));
accuracy = (TP+TN)/N;
fprintf(' Accuracy on test set = %f\n', accuracy);
%% Determine term c and gamma value for SVM
SVMOpt = SVM(fea_train,fea_test, gnd_train,gnd_test);
[~,index] = max(SVMOpt(:,3));
best_c =  SVMOpt(index,1);
best_gamma = SVMOpt(index,2);
fprintf('Optimal parameters for SVM are:\n');
fprintf(' c = %f, gamma = %f\n', best_c, best_gamma);
fprintf(' Average accuracy on training set = %f\n', SVMOpt(index,3));
%# Use best pair of c-gamma for SVM to train on fea_train and predict on fea_test, plot ROC curve
SVM_mdl= fitcsvm(fea_train,gnd_train,...
                             'BoxConstraint',10 , 'KernelFunction',...
                         'rbf', 'KernelScale',sqrt(1/(2*best_gamma)));
SVM_mdl = fitPosterior(SVM_mdl);
[SVM_label,score_SVM] = predict(SVM_mdl, fea_test);
%# Plot ROC
targets = ind2vec(gnd_test');
targets = full(targets);
plotroc(targets, score_SVM');
saveas(gcf,'ROC.png');
%# Calculate accuracy
confusionMatrix = confusionmat(gnd_test,SVM_label);
TP = confusionMatrix(1,1);
TN = confusionMatrix(2,2);
N = sum(sum(confusionMatrix));
accuracy = (TP+TN)/N;
fprintf(' Accuracy on test set = %f\n', accuracy);


%% Compare performances of five classifiers
resultkNN = [];
resultSVM = [];
resultNB = [];
resultTree = [];
resultNN = [];

for i = 1:20
%# Divide data set into two parts randomly
    seq = randperm(2200);
    fea_train = fea(seq(1:1100),:);
    fea_test = fea(seq(1101:2200),:);
    gnd_train = gnd(seq(1:1100),:);
    gnd_test = gnd(seq(1101:2200),:);

    %% kNN
    ts = clock;
    kNN_mdl = fitcknn(fea_train, gnd_train, 'NumNeighbors',best_k);
    te = clock;
    traintime_kNN(i) = etime(te,ts);
    ts = clock;
    [kNN_label,scorekNN,cost] = predict(kNN_mdl, fea_test);
    te = clock;
    classifytime_kNN(i) = etime(te,ts);
    resultkNN = [resultkNN; EvalResult(gnd_test,kNN_label)];

    %% SVM
    ts = clock;
    SVM_mdl= fitcsvm(fea_train,gnd_train,...
                             'BoxConstraint',best_c , 'KernelFunction',...
                         'rbf', 'KernelScale',sqrt(1/(2*best_gamma)));
    te = clock;
    traintime_SVM(i) = etime(te,ts);
    ts = clock;
    [SVM_label,scoreSVM] = predict(SVM_mdl, fea_test);
    te = clock;
    classifytime_SVM(i) = etime(te,ts);
    resultSVM = [resultSVM; EvalResult(gnd_test,SVM_label)];

    %% Naive Bayes
    ts = clock;
    NB_mdl = fitcnb(fea_train,gnd_train);
    te = clock;
    traintime_NB(i) = etime(te,ts);
    ts = clock;
    NB_label = predict(NB_mdl,fea_test);
    te = clock;
    classifytime_NB(i) = etime(te,ts);
    resultNB = [resultNB; EvalResult(gnd_test,NB_label)];

    %% Decision Tree
    ts = clock;
    tree = fitctree(fea_train,gnd_train);
    te = clock;
    traintime_tree(i) = etime(te,ts);
    ts = clock;
    %view(tree,'Mode','Graph');
    [tree_label,score,node,cnum] = predict(tree, fea_test);
    te = clock;
    classifytime_tree(i) = etime(te,ts);
    resultTree = [resultTree; EvalResult(gnd_test,tree_label)];

    %% Neural Network
    x = fea_train';
    t = ind2vec(gnd_train');
    t = full(t);
    %# Create a Pattern Recognition Network
    hiddenLayerSize = 10;
    net = patternnet(hiddenLayerSize);
    %# Setup Division of Data for Training, Validation, Testing
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 15/100;
    %# Train the Network
    ts = clock;
    [net,tr] = train(net,x,t);
    te = clock;
    traintime_NN(i) = etime(te,ts);
    %# Test the Network
    %testX = x(:,tr.testInd);
    %testT = t(:,tr.testInd);
    testX = fea_test';
    testT = gnd_test;
    ts = clock;
    testY = net(testX);
    te = clock;
    classifytime_NN(i) = etime(te,ts);
    testIndices = vec2ind(testY);
    %errors = gsubtract(t,outputs);
    resultNN = [resultNN; EvalResult(testT,testIndices')];
    %performance = perform(net,testT,testY);
    % View the Network
    %view(net);
    % Plots
    % Uncomment these lines to enable various plots.
    %figure, plotperform(tr)
    %figure, plottrainstate(tr)
    %figure, plotconfusion(t,y)
    %figure, plotroc(t,y)
    %figure, ploterrhist(e)
end;
%# Calculte average and std
kNN_average = mean(resultkNN);
kNN_std = std(resultkNN);
kNN_traintime = mean(traintime_kNN);
kNN_classifytime = mean(classifytime_kNN);

SVM_average = mean(resultSVM);
SVM_std = std(resultSVM);
SVM_traintime = mean(traintime_SVM);
SVM_classifytime = mean(classifytime_SVM);

NB_average = mean(resultNB);
NB_std = std(resultNB);
NB_traintime = mean(traintime_NB);
NB_classifytime = mean(classifytime_NB);

tree_average = mean(resultTree);
tree_std = std(resultTree);
tree_traintime = mean(traintime_tree);
tree_classifytime = mean(classifytime_tree);

NN_average = mean(resultNN);
NN_std = std(resultNN);
NN_traintime = mean(traintime_NN);
NN_classifytime = mean(classifytime_NN);
%
