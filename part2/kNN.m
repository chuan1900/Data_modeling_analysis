function kNNOpt = kNN(fea_train,gnd_train)
kNNOpt = [];
kVal = 1;
for i = 1:16

    mdl = fitcknn(fea_train, gnd_train, 'NumNeighbors',kVal,'kFold',5);

    cvError = kfoldLoss(mdl);
    cvCorr = 1 - cvError;
    kNNOpt = [kNNOpt; [kVal cvCorr]];

    kVal = kVal + 2;

end;
%#plot accuracy curves;
plot(kNNOpt(:,1),kNNOpt(:,2));
set(gca,'XTick',1:2:33);
title('kNN Classification');
xlabel('k value');
ylabel('classification accuracy');
saveas(gcf,'kNN.png');
