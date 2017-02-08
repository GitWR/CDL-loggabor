clear;
close all;
clc;
t_star=cputime;%计时用
Train_lables=zeros(1,40);
Test_lables=zeros(1,40); 
%step1 加载所用数据
load ImgData_HE_itera
%step2:下面的循环是为训练样本添加标签
l=5;
k=l;
a=linspace(1,8,8);%共有40类
i1=1;
while (k<=40)
    while (i1<=8)
    for i=1:8
        i_train=l*(i-1)+1;
        Train_lables(i_train:k)=a(i1);
        k=k+5;
        i1=i1+1;
    end
    end
end
%step4:下面的循环是为测试样本添加标签
l1=5;
k1=l1;
a1=linspace(1,8,8);%共有40类
i2=1;
while (k1<=40)
    while (i2<=8)
    for i=1:8
        i_test=l1*(i-1)+1;
        Test_lables(i_test:k1)=a1(i2);
        k1=k1+5;
        i2=i2+1;
    end
    end
end
%step5:计算训练样本的协方差和log映射
param.d=2000;%样本维度
d=param.d;
basis=eye(d);%单位矩阵
cov_train=cell(1,40);%因为协方差后样本的维度很大，可能在后面影响运算，因此将其存在胞中，方便计算
cov_train_disturb=cell(1,40);
log_cov_train=cell(1,40);%用于存储log-Euclidean
cov_test=cell(1,40);%因为协方差后样本的维度很大，可能在后面影响运算，因此将其存在胞中，方便计算
cov_test_disturb=cell(1,40);
log_cov_test=cell(1,40);%用于存储log-Euclidean
ImgData_HE_train=cell(1,40);
ImgData_HE_test=cell(1,40);
accuracy_matrix=zeros(1,10);%存储最终的平均测试精度
%for iteration=1:10
    ImgData_HE_train=All_ImgData_HE_train{4};
    ImgData_HE_test=All_ImgData_HE_test{4};
for i=1:40
    single_sample=double(ImgData_HE_train{i})/255;
    N=size(single_sample,2);
    tmp=basis'*single_sample;
    single_sample=tmp;
    train_mean=mean(single_sample,2);%求均值，并进行中心化操作
    center_single_sample=single_sample-repmat(train_mean,1,N);%中心化操作
    cov_train{i}=center_single_sample*center_single_sample';%样本的协方差
    tol=1e-3;
    if (N<=d||d>=100)
        cov_train_disturb{i}=cov_train{i}+tol*basis*trace(cov_train{i});%如果样本个数远远小于样本维度或者样本维度大于一个定值，那么加上一个干扰项从而避免奇异
    end
    log_cov_train{i}=logm(cov_train_disturb{i});
end
%clear center_matrix_train;
%step6:计算测试样本的协方差和log映射
for i=1:40
    single_sample1=double(ImgData_HE_test{i})/255;
    N1=size(single_sample1,2);
    tmp1=basis'*single_sample1;
    single_sample1=tmp1;
    test_mean=mean(single_sample1,2);%求均值，并进行中心化操作
    center_single_sample1=single_sample1-repmat(test_mean,1,N1);%中心化操作
    cov_test{i}=center_single_sample1*center_single_sample1';%单个样本的协方差
    tol=1e-3;
    if (N1<=d||d>=100)
        cov_test_disturb{i}=cov_test{i}+tol*basis*trace(cov_test{i});%如果样本个数远远小于样本维度或者样本维度大于一个定值，那么加上一个干扰项从而避免奇异
    end
    log_cov_test{i}=logm(cov_test_disturb{i});
end
%step7:构造训练和测试用的核矩阵
kmatrix_train=zeros(size(log_cov_train,2),size(log_cov_train,2));%定义训练用的核矩阵
kmatrix_test=zeros(size(log_cov_train,2),size(log_cov_test,2));%定义测试用的核矩阵
%构造训练样本的核矩阵
for i=1:size(log_cov_train,2)
    for j=1:size(log_cov_train,2)
        cov_i_Train=log_cov_train{i};% cov_i_Train is actually the log-mapped cov
        cov_j_Train=log_cov_train{j};% cov_i_Train is actually the log-mapped cov
        cov_i_Train_reshape=reshape(cov_i_Train,size(cov_i_Train,1)*size(cov_i_Train,2),1);%拉成一个高纬的列向量
        cov_j_Train_reshape=reshape(cov_j_Train,size(cov_j_Train,1)*size(cov_j_Train,2),1);%拉成一个高纬的列向量
        kmatrix_train(i,j)=cov_i_Train_reshape'*cov_j_Train_reshape;%240*240
        kmatrix_train(j,i)=kmatrix_train(i,j);
    end
end

%构造测试样本的核矩阵
%transfer_matrix=ones(40000,14400);%定义一个过渡矩阵
for i=1:size(log_cov_train,2)
    for j=1:size(log_cov_test,2)
        cov_i_Train=log_cov_train{i};% cov_i_Train is actually the log-mapped cov
        cov_j_Test=log_cov_test{j};% cov_i_Train is actually the log-mapped cov
        cov_i_Train_reshape=reshape(cov_i_Train,size(cov_i_Train,1)*size(cov_i_Train,2),1);%拉成一个高纬的列向量
        cov_j_Test_reshape=reshape(cov_j_Test,size(cov_j_Test,1)*size(cov_j_Test,2),1);%拉成一个高纬的列向量
        kmatrix_test(i,j)=cov_i_Train_reshape'*cov_j_Test_reshape;%240*160
    end
end
%step8:KDA的训练过程
gnd=Train_lables(1,:); %Colunm vector of the label information for each data point                   
options.Regu=1;
[eigvector, eigvalue] = fun_CDL_Train(options, gnd, kmatrix_train);%该函数主要做特征分解用的
%options.ReguAlpha=0.01; 
nClasses=unique(gnd);
Dim = length(nClasses) - 1; %% the low dimensionality of LDA is c(#classes)-1,19
kk = Dim;
alpha_matrix = eigvector(:,1:kk); %150*19
data_DR_train = alpha_matrix'*kmatrix_train; % 类似于训练样本降维后的新样本，19*20
%clear kmatrix_train;
%F_test = alpha_matrix'*KMatrix_Test; % 测试样本降维后的新样本,19*39
% K_train=kmatrix_train;
% clear kmatrix_train
% K_train=max(K_train,K_train');
% %下面是初始化过程
% row_train=size(K_train,1);
% Train_lables_unique=unique(Train_lables);%统计共有多少类，同时消除从复的数字
% aim_dim=length(Train_lables_unique)-1;%KDA最终降维到c-1，其中c为类别数
% %下面要调整K
% sum_K_train=sum(K_train,2);%按照行求和
% H_train=repmat(sum_K_train./row_train,1,row_train);
% K_train_adjust=K_train-H_train-H_train'+sum(sum_K_train)/(row_train^2);
% K_train_adjust=max(K_train_adjust,K_train_adjust');
% clear H_train;
% %下面进行特征分解获得特征值和特征向量
% [u_vectors,d_values]=eig(K_train_adjust);
% [x,index]=sort(diag(d_values));
% d_values_sort=flipud(x);%是一个列向量
% index_sort=flipud(index);%从大到小的顺序改为从小到大的顺序
% u_vectors_sort=u_vectors(:,index_sort);%按照特征值得顺序调整特征向量的顺序
% aim_eigenvectors=u_vectors_sort(:,1:aim_dim);
% W=zeros(cov_size,cov_size);%权重矩阵
% start_count=1;
% end_count=0;
% num__train_class=zeros(1,size(Train_Matrix,3));%用于存储每一类的样本个数
% for i=1:size(Train_Matrix,3)
%     num__train_class(:,i)=size(Train_Matrix(:,:,i),2);
% end
% %W是一个分块矩阵，每一块是一个6*6的矩阵，且每块中的每个元素值都为1/ num__train_class(:,i)
% for i=1:size(Train_Matrix,3)
%     end_count=end_count+ num__train_class(:,i);
%     for j=start_count:end_count
%         for k=start_count:end_count
%             W(j,k)=1/num__train_class(:,i);
%         end
%     end
%     start_count=start_count+num__train_class(:,i);
% end
% %计算GDA中关键的alpha的值
% GDA_matrix=aim_eigenvectors'*W*aim_eigenvectors;%根据论文中的公式（17）而得,目的是为了求解beta
% [beta_vectors,beta_values]=eig(GDA_matrix);
% d_values_sort_aim=diag(d_values_sort(1:aim_dim,:));%论文中的tao
% alpha=aim_eigenvectors*inv(d_values_sort_aim)*beta_vectors;%alpha的值
% [m,n]=size(alpha);
% alpha_norm=zeros(m,n);%存储单位化后的alpha值
% for i=1:n
%     alpha_norm(:,i)=alpha(:,i)/sqrt(alpha(:,i)'*K_train_adjust*alpha(:,i));
% end
% data_DR_train=alpha_norm'*K_train_adjust;%原始样本在核映射空间降维后的样本（maybe like this ，haha）
%step9:KDA的测试过程
data_DR_test=alpha_matrix'*kmatrix_test;
%clear kmatrix_test;
% for i=1:size(kmatrix_test,2)
%     dist_onegroup=zeros(1,size(kmatrix_train,2));%用于存储每一组距离
%     for j=1:size(kmatrix_train,2)
%         dist_onegroup(:,j)=norm(kmatrix_test(:,i)-kmatrix_train(:,j));
%     end
%     [dist_sort,index2]=sort(dist_onegroup);%按升序排列
%     class1=floor((index2(1)-1)/useto_train)+1;
%     class2=floor((index2(2)-1)/useto_train)+1;
%     class3=floor((index2(3)-1)/useto_train)+1;
%     if class1~=class2 && class2~=class3 
%                class=class1; 
%     elseif class1==class2 
%                class=class1; 
%      elseif class2==class3 
%                class=class2; 
%      end 
%       if class==Test_lables(i) 
%               accuracy_number=accuracy_number+1;
%       end
% end
 Class = knnclassify(data_DR_test', data_DR_train', Train_lables');
 [a_test,b_test]=size(data_DR_test);
% clear center_matrix_test;
 accuracy_number=sum(Class'==Test_lables);%统计最终正确识别的样本个数
 accuracy=accuracy_number/b_test;
 %accuracy_matrix(:,iteration)=accuracy*100;
% TestRate= fun_DCOVTest(data_DR_train(1:kk,:),data_DR_test(1:kk,:),Train_lables,Test_lables);
t_2 = cputime - t_star;%机器运行时间
disp(['运行时间为:  ' num2str(t_2) ' seconds']);
fprintf(1,'第%d次迭代识别准确的样本个数为：%d %d\n',1,accuracy_number);
fprintf(1,'第%d次迭代的测试精确度为:%d %d\n',1,accuracy*100);
%end
% mean_accuracy=sum(accuracy_matrix)/iteration;
% fprintf(1,'平均测试精确度为: %d\n',mean_accuracy);

