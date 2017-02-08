clear;
close all;
clc;
t_star=cputime;%��ʱ��
Train_lables=zeros(1,40);
Test_lables=zeros(1,40); 
%step1 ������������
load ImgData_HE_itera
%step2:�����ѭ����Ϊѵ��������ӱ�ǩ
l=5;
k=l;
a=linspace(1,8,8);%����40��
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
%step4:�����ѭ����Ϊ����������ӱ�ǩ
l1=5;
k1=l1;
a1=linspace(1,8,8);%����40��
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
%step5:����ѵ��������Э�����logӳ��
param.d=2000;%����ά��
d=param.d;
basis=eye(d);%��λ����
cov_train=cell(1,40);%��ΪЭ�����������ά�Ⱥܴ󣬿����ں���Ӱ�����㣬��˽�����ڰ��У��������
cov_train_disturb=cell(1,40);
log_cov_train=cell(1,40);%���ڴ洢log-Euclidean
cov_test=cell(1,40);%��ΪЭ�����������ά�Ⱥܴ󣬿����ں���Ӱ�����㣬��˽�����ڰ��У��������
cov_test_disturb=cell(1,40);
log_cov_test=cell(1,40);%���ڴ洢log-Euclidean
ImgData_HE_train=cell(1,40);
ImgData_HE_test=cell(1,40);
accuracy_matrix=zeros(1,10);%�洢���յ�ƽ�����Ծ���
%for iteration=1:10
    ImgData_HE_train=All_ImgData_HE_train{4};
    ImgData_HE_test=All_ImgData_HE_test{4};
for i=1:40
    single_sample=double(ImgData_HE_train{i})/255;
    N=size(single_sample,2);
    tmp=basis'*single_sample;
    single_sample=tmp;
    train_mean=mean(single_sample,2);%���ֵ�����������Ļ�����
    center_single_sample=single_sample-repmat(train_mean,1,N);%���Ļ�����
    cov_train{i}=center_single_sample*center_single_sample';%������Э����
    tol=1e-3;
    if (N<=d||d>=100)
        cov_train_disturb{i}=cov_train{i}+tol*basis*trace(cov_train{i});%�����������ԶԶС������ά�Ȼ�������ά�ȴ���һ����ֵ����ô����һ��������Ӷ���������
    end
    log_cov_train{i}=logm(cov_train_disturb{i});
end
%clear center_matrix_train;
%step6:�������������Э�����logӳ��
for i=1:40
    single_sample1=double(ImgData_HE_test{i})/255;
    N1=size(single_sample1,2);
    tmp1=basis'*single_sample1;
    single_sample1=tmp1;
    test_mean=mean(single_sample1,2);%���ֵ�����������Ļ�����
    center_single_sample1=single_sample1-repmat(test_mean,1,N1);%���Ļ�����
    cov_test{i}=center_single_sample1*center_single_sample1';%����������Э����
    tol=1e-3;
    if (N1<=d||d>=100)
        cov_test_disturb{i}=cov_test{i}+tol*basis*trace(cov_test{i});%�����������ԶԶС������ά�Ȼ�������ά�ȴ���һ����ֵ����ô����һ��������Ӷ���������
    end
    log_cov_test{i}=logm(cov_test_disturb{i});
end
%step7:����ѵ���Ͳ����õĺ˾���
kmatrix_train=zeros(size(log_cov_train,2),size(log_cov_train,2));%����ѵ���õĺ˾���
kmatrix_test=zeros(size(log_cov_train,2),size(log_cov_test,2));%��������õĺ˾���
%����ѵ�������ĺ˾���
for i=1:size(log_cov_train,2)
    for j=1:size(log_cov_train,2)
        cov_i_Train=log_cov_train{i};% cov_i_Train is actually the log-mapped cov
        cov_j_Train=log_cov_train{j};% cov_i_Train is actually the log-mapped cov
        cov_i_Train_reshape=reshape(cov_i_Train,size(cov_i_Train,1)*size(cov_i_Train,2),1);%����һ����γ��������
        cov_j_Train_reshape=reshape(cov_j_Train,size(cov_j_Train,1)*size(cov_j_Train,2),1);%����һ����γ��������
        kmatrix_train(i,j)=cov_i_Train_reshape'*cov_j_Train_reshape;%240*240
        kmatrix_train(j,i)=kmatrix_train(i,j);
    end
end

%������������ĺ˾���
%transfer_matrix=ones(40000,14400);%����һ�����ɾ���
for i=1:size(log_cov_train,2)
    for j=1:size(log_cov_test,2)
        cov_i_Train=log_cov_train{i};% cov_i_Train is actually the log-mapped cov
        cov_j_Test=log_cov_test{j};% cov_i_Train is actually the log-mapped cov
        cov_i_Train_reshape=reshape(cov_i_Train,size(cov_i_Train,1)*size(cov_i_Train,2),1);%����һ����γ��������
        cov_j_Test_reshape=reshape(cov_j_Test,size(cov_j_Test,1)*size(cov_j_Test,2),1);%����һ����γ��������
        kmatrix_test(i,j)=cov_i_Train_reshape'*cov_j_Test_reshape;%240*160
    end
end
%step8:KDA��ѵ������
gnd=Train_lables(1,:); %Colunm vector of the label information for each data point                   
options.Regu=1;
[eigvector, eigvalue] = fun_CDL_Train(options, gnd, kmatrix_train);%�ú�����Ҫ�������ֽ��õ�
%options.ReguAlpha=0.01; 
nClasses=unique(gnd);
Dim = length(nClasses) - 1; %% the low dimensionality of LDA is c(#classes)-1,19
kk = Dim;
alpha_matrix = eigvector(:,1:kk); %150*19
data_DR_train = alpha_matrix'*kmatrix_train; % ������ѵ��������ά�����������19*20
%clear kmatrix_train;
%F_test = alpha_matrix'*KMatrix_Test; % ����������ά���������,19*39
% K_train=kmatrix_train;
% clear kmatrix_train
% K_train=max(K_train,K_train');
% %�����ǳ�ʼ������
% row_train=size(K_train,1);
% Train_lables_unique=unique(Train_lables);%ͳ�ƹ��ж����࣬ͬʱ�����Ӹ�������
% aim_dim=length(Train_lables_unique)-1;%KDA���ս�ά��c-1������cΪ�����
% %����Ҫ����K
% sum_K_train=sum(K_train,2);%���������
% H_train=repmat(sum_K_train./row_train,1,row_train);
% K_train_adjust=K_train-H_train-H_train'+sum(sum_K_train)/(row_train^2);
% K_train_adjust=max(K_train_adjust,K_train_adjust');
% clear H_train;
% %������������ֽ�������ֵ����������
% [u_vectors,d_values]=eig(K_train_adjust);
% [x,index]=sort(diag(d_values));
% d_values_sort=flipud(x);%��һ��������
% index_sort=flipud(index);%�Ӵ�С��˳���Ϊ��С�����˳��
% u_vectors_sort=u_vectors(:,index_sort);%��������ֵ��˳���������������˳��
% aim_eigenvectors=u_vectors_sort(:,1:aim_dim);
% W=zeros(cov_size,cov_size);%Ȩ�ؾ���
% start_count=1;
% end_count=0;
% num__train_class=zeros(1,size(Train_Matrix,3));%���ڴ洢ÿһ�����������
% for i=1:size(Train_Matrix,3)
%     num__train_class(:,i)=size(Train_Matrix(:,:,i),2);
% end
% %W��һ���ֿ����ÿһ����һ��6*6�ľ�����ÿ���е�ÿ��Ԫ��ֵ��Ϊ1/ num__train_class(:,i)
% for i=1:size(Train_Matrix,3)
%     end_count=end_count+ num__train_class(:,i);
%     for j=start_count:end_count
%         for k=start_count:end_count
%             W(j,k)=1/num__train_class(:,i);
%         end
%     end
%     start_count=start_count+num__train_class(:,i);
% end
% %����GDA�йؼ���alpha��ֵ
% GDA_matrix=aim_eigenvectors'*W*aim_eigenvectors;%���������еĹ�ʽ��17������,Ŀ����Ϊ�����beta
% [beta_vectors,beta_values]=eig(GDA_matrix);
% d_values_sort_aim=diag(d_values_sort(1:aim_dim,:));%�����е�tao
% alpha=aim_eigenvectors*inv(d_values_sort_aim)*beta_vectors;%alpha��ֵ
% [m,n]=size(alpha);
% alpha_norm=zeros(m,n);%�洢��λ�����alphaֵ
% for i=1:n
%     alpha_norm(:,i)=alpha(:,i)/sqrt(alpha(:,i)'*K_train_adjust*alpha(:,i));
% end
% data_DR_train=alpha_norm'*K_train_adjust;%ԭʼ�����ں�ӳ��ռ併ά���������maybe like this ��haha��
%step9:KDA�Ĳ��Թ���
data_DR_test=alpha_matrix'*kmatrix_test;
%clear kmatrix_test;
% for i=1:size(kmatrix_test,2)
%     dist_onegroup=zeros(1,size(kmatrix_train,2));%���ڴ洢ÿһ�����
%     for j=1:size(kmatrix_train,2)
%         dist_onegroup(:,j)=norm(kmatrix_test(:,i)-kmatrix_train(:,j));
%     end
%     [dist_sort,index2]=sort(dist_onegroup);%����������
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
 accuracy_number=sum(Class'==Test_lables);%ͳ��������ȷʶ�����������
 accuracy=accuracy_number/b_test;
 %accuracy_matrix(:,iteration)=accuracy*100;
% TestRate= fun_DCOVTest(data_DR_train(1:kk,:),data_DR_test(1:kk,:),Train_lables,Test_lables);
t_2 = cputime - t_star;%��������ʱ��
disp(['����ʱ��Ϊ:  ' num2str(t_2) ' seconds']);
fprintf(1,'��%d�ε���ʶ��׼ȷ����������Ϊ��%d %d\n',1,accuracy_number);
fprintf(1,'��%d�ε����Ĳ��Ծ�ȷ��Ϊ:%d %d\n',1,accuracy*100);
%end
% mean_accuracy=sum(accuracy_matrix)/iteration;
% fprintf(1,'ƽ�����Ծ�ȷ��Ϊ: %d\n',mean_accuracy);

