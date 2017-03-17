function SVMOpt = SVM(fea_train,fea_test, gnd_train,gnd_test)

SVMOpt = [];
gnd_train(gnd_train==2) = 0;
Fold_Number=5;
c = [0.1, 0.5, 1, 2, 5, 10, 20, 50];
gamma = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10];
sigma = sqrt(1./(2*gamma));
figure('visible','off')
for i = 1:8
    for k = 1:8


        indices = crossvalind('Kfold',gnd_train, Fold_Number);   %# get indices of 5-fold CV
        cp = classperf(gnd_train);
        for j = 1:Fold_Number                                  %# for each fold
            test = (indices == j);
            train = (indices ~= j);
            %# train an SVM model over training instances
            svm = svmtrain(fea_train(train,:),gnd_train(train,:),...
                         'BoxConstraint',c(k) , 'Kernel_Function','rbf', 'RBF_Sigma',sigma(i));
            %# test using test instances
            pred = svmclassify(svm, fea_train(test,:));
            %# evaluate and update performance object
            cp = classperf(cp, pred, test);

            %# plot ROC curves
            Xnew = fea_train(test,:);
            shift = svm.ScaleData.shift;
            scale = svm.ScaleData.scaleFactor;
            Xnew = bsxfun(@plus,Xnew,shift);
            Xnew = bsxfun(@times,Xnew,scale);
            sv = svm.SupportVectors;
            alphaHat = svm.Alpha;
            bias = svm.Bias;
            kfun = svm.KernelFunction;
            kfunargs = svm.KernelFunctionArgs;
            f = kfun(sv,Xnew,kfunargs{:})'*alphaHat(:) + bias;
            f = -f;
            [X,Y,T,AUC] = perfcurve(gnd_train(test,:),f,1);
            plot(X,Y);
            hold on;
        end
        %# get accuracy
        SVMOpt = [SVMOpt; [c(k),gamma(i),cp.CorrectRate]];
    end;
end;
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC curve of SVM');
hold off;
saveas(gcf,'SVM.png');
