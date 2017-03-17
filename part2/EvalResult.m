function  Eval = EvalResult( gnd_test, label )
confusionMatrix = confusionmat(gnd_test,label);
TP = confusionMatrix(1,1);
FN = confusionMatrix(1,2);
FP = confusionMatrix(2,1);
TN = confusionMatrix(2,2);
N = sum(sum(confusionMatrix));

accuracy = (TP+TN)/N;
sensitivity = TP/(TP+FN);
%specificity = TN/(TN+FP);
precision = TP/(TP+FP);
recall = sensitivity;
f_measure = 2*((precision*recall)/(precision + recall));

Eval = [accuracy precision recall f_measure]; 
