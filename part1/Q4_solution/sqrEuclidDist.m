function EuDistance = sqrEuclidDist(fea, gnd);

fea_label1 = fea(find(gnd==1),:);
fea_label2 = fea(find(gnd==2),:);
fea_label3 = fea(find(gnd==3),:);
[m1,~] = size(fea_label1);
[m2,~] = size(fea_label2);
[m3,~] = size(fea_label3);
l1=0; l2=0; l3=0;

for i=1:m1,
    l1 = l1+ sum(sum((ones(m2,1)*fea_label1(i,:)...
        -fea_label2(:,:)).^2));
end

for i=1:m2,
    l2 = l2+ sum(sum((ones(m3,1)*fea_label2(i,:)...
        -fea_label3(:,:)).^2));
end
for i=1:m1,
    l3 = l3+ sum(sum((ones(m3,1)*fea_label1(i,:)...
        -fea_label3(:,:)).^2));
end

EuDistance = l1/m1/m2 + l2/m2/m3 + l3/m1/m3;

% l1 = Euc_fun(fea_label1,fea_label2).^2;
% l2 = Euc_fun(fea_label2,fea_label3).^2;
% l3 = Euc_fun(fea_label1,fea_label3).^2;
% EuDistance = l1/m1/m2 + l2/m2/m3 + l3/m1/m3;