function [ corr ] = Bayes_fun( fea, gnd, foldNum)
%foldNum = 10;
Indices = crossvalind('Kfold',gnd, foldNum);
cp = classperf(gnd);

for i = 1:foldNum
    testSet = (Indices == i);
    trainSet = (Indices ~= i);   % The rest part as training set
    preClass = classify(fea(testSet,:),fea(trainSet,:),gnd(trainSet,:),'diaglinear');
    classperf(cp,preClass,testSet);
end;
    
    corr = cp.CorrectRate;