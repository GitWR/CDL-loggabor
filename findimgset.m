clear;
close all;
clc;
train_each_subjects=5;%ÿ������10����,ǰ�������ѵ����������ڲ���
test_each_subjects=5;%ÿ������10����,ǰ�������ѵ����������ڲ���
eachNum=41;%ÿ��������41��ͼƬ
ImgData_HE_train=cell(1,40);%�洢ѵ����
ImgData_HE_test=cell(1,40);%�洢���Լ�
%ImgData_HE_train=cell(1,train_each_subjects);
% useto_train=41;
% useto_test=11;
% Train_Matrix=zeros(20*20,useto_train,8);%��40�࣬ÿһ���ǰ6����ѵ����4��������
train_matrix_singlegraphic=zeros(2000,eachNum);
test_matrix_singlegraphic=zeros(2000,eachNum);
% Test_Matrix=zeros(20*20,useto_test,8);
%  cov_matrix_train=zeros(10304*10304,240)%���ڴ洢ѵ��������Э����s
%  cov_matrix_test=zeros(10304*10304,160)%���ڴ洢����������Э����
 Train_lables=zeros(1,240);
 Test_lables=zeros(1,88); 
 %step1:�����ѭ����ѵ�����ֵ�����
%  index_train=1;
%  while (index_train<=40)
%      index_train=index_train+1;
All_ImgData_HE_train=cell(1,10);%����ѵ�����洢10�β�ͬ�Ľ��
All_ImgData_HE_test=cell(1,10);%���ڲ��Լ��洢10�β�ͬ�Ľ��
nscale= 5;%�߶�
norient=8;%����
minWaveLength=3;
mult=2;
sigmaOnf=0.65;
dThetaOnSigma=1.5;
feedback=0;
for l=1:10
   a=randperm(10);
   count_train=0;
   count_test=0;
 for j=1:8
 for i=1:train_each_subjects
     count_train=count_train+1;
 for k=1:eachNum
     imageName=strcat(num2str(k),'.png');
     %A=imread(['F:\matlab-code\ORLface\ORL_train\',j,imageName,'\']); 
     %path=['F:\matlab-code\ORLface\ORL_train\',sprintf('%d',j),'\'];%ORL
     %path=['F:\matlab-code\CroppedYale\',sprintf('%d',j),'\'];%yale
     path=['F:\matlab-code\ETH80\',sprintf('%d',j),'\',sprintf('%d',a(1,i)),'\'];%ETH80
     A=imread([path,imageName]);
     A=imresize(A,[20,20]);
     A=rgb2gray(A);
     transition=[];%����һ�����ɾ��󣬴洢ÿһ�����������ͼ����õ��ľ���
     A_filtered=Log_gabor(A,nscale, norient, minWaveLength, mult,sigmaOnf,dThetaOnSigma,feedback);
        for i_filter=2:2:40
         tmp=A_filtered{i_filter};
         transition=[transition tmp];
        end
     transition=abs(transition);
     transition=downsample(transition,4);
     [col,row]=size(transition);
     train_matrix_singlegraphic(:,k)= reshape(transition,col*row,1);
     ImgData_HE_train{count_train}=train_matrix_singlegraphic;
     %transition_reshape=reshape(transition,row*col,1);%�Ѿ���gabor�˲����������8�����ͼ�����õľ�������һ����������ά����1144*1
     %all_sample_train(:,count_train)=transition_reshape;%��ÿһ����������Ϊall_sample��һ�и�ֵ��all_sample
     %train_matrix_singlegraphic(:,i)= reshape(A,112*92,1);
 end
    %ImgData_HE{index_train}=
     %Train_Matrix(:,:,j)=train_matrix_singlegraphic;
 end
 end
    All_ImgData_HE_train{l}=ImgData_HE_train;
%  end
%save ETH80 ImgData_HE
% Train_Matrix_reshape=all_sample_train;%reshape(Train_Matrix,112*92,525);
% train_mean=mean(Train_Matrix_reshape,2);
% train_mean_matrix=repmat(train_mean,1,size(Train_Matrix_reshape,2));
% center_matrix_train=Train_Matrix_reshape-train_mean_matrix;
 for j=1:8
 for i=train_each_subjects+1:train_each_subjects+test_each_subjects
     count_test=count_test+1;
 for k=1:eachNum
     imageName=strcat(num2str(k),'.png');
     %A=imread(['F:\matlab-code\ORLface\ORL_train\',j,imageName,'\']); 
     %path=['F:\matlab-code\ORLface\ORL_train\',sprintf('%d',j),'\'];%ORL
     %path=['F:\matlab-code\CroppedYale\',sprintf('%d',j),'\'];%yale
     path=['F:\matlab-code\ETH80\',sprintf('%d',j),'\',sprintf('%d',a(1,i)),'\'];%ETH80
     A=imread([path,imageName]);
     A=imresize(A,[20,20]);
     A=rgb2gray(A);
     transition=[];%����һ�����ɾ��󣬴洢ÿһ�����������ͼ����õ��ľ���
     A_filtered_test=Log_gabor(A,nscale, norient, minWaveLength, mult,sigmaOnf,dThetaOnSigma,feedback);
     for i_filter_test=2:2:40
         tmp=A_filtered_test{i_filter_test};
         transition=[transition tmp];
     end
     transition=abs(transition);
     transition=downsample(transition,4);
     [col,row]=size(transition);
     test_matrix_singlegraphic(:,k)= reshape(transition,col*row,1);
     ImgData_HE_test{count_test}=test_matrix_singlegraphic;
     %transition_reshape=reshape(transition,row*col,1);%�Ѿ���gabor�˲����������8�����ͼ�����õľ�������һ����������ά����1144*1
     %all_sample_train(:,count_train)=transition_reshape;%��ÿһ����������Ϊall_sample��һ�и�ֵ��all_sample
     %train_matrix_singlegraphic(:,i)= reshape(A,112*92,1);
 end
    %ImgData_HE{index_train}=
     %Train_Matrix(:,:,j)=train_matrix_singlegraphic;
 end
 end
   All_ImgData_HE_test{l}=ImgData_HE_test;
end
save ImgData_HE_itera All_ImgData_HE_train All_ImgData_HE_test 
