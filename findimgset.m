clear;
close all;
clc;
train_each_subjects=5;%每个类有10个集,前五个用于训练后五个用于测试
test_each_subjects=5;%每个类有10个集,前五个用于训练后五个用于测试
eachNum=41;%每个集都有41张图片
ImgData_HE_train=cell(1,40);%存储训练集
ImgData_HE_test=cell(1,40);%存储测试集
%ImgData_HE_train=cell(1,train_each_subjects);
% useto_train=41;
% useto_test=11;
% Train_Matrix=zeros(20*20,useto_train,8);%用40类，每一类的前6个做训练后4个做测试
train_matrix_singlegraphic=zeros(2000,eachNum);
test_matrix_singlegraphic=zeros(2000,eachNum);
% Test_Matrix=zeros(20*20,useto_test,8);
%  cov_matrix_train=zeros(10304*10304,240)%用于存储训练样本的协方差s
%  cov_matrix_test=zeros(10304*10304,160)%用于存储测试样本的协方差
 Train_lables=zeros(1,240);
 Test_lables=zeros(1,88); 
 %step1:下面的循环是训练部分的数据
%  index_train=1;
%  while (index_train<=40)
%      index_train=index_train+1;
All_ImgData_HE_train=cell(1,10);%对于训练集存储10次不同的结果
All_ImgData_HE_test=cell(1,10);%对于测试集存储10次不同的结果
nscale= 5;%尺度
norient=8;%方向
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
     transition=[];%定义一个过渡矩阵，存储每一个经过处理的图像而得到的矩阵
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
     %transition_reshape=reshape(transition,row*col,1);%把经过gabor滤波器处理过的8方向的图像而获得的矩阵拉成一个列向量，维度是1144*1
     %all_sample_train(:,count_train)=transition_reshape;%把每一个列向量作为all_sample的一列赋值给all_sample
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
     transition=[];%定义一个过渡矩阵，存储每一个经过处理的图像而得到的矩阵
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
     %transition_reshape=reshape(transition,row*col,1);%把经过gabor滤波器处理过的8方向的图像而获得的矩阵拉成一个列向量，维度是1144*1
     %all_sample_train(:,count_train)=transition_reshape;%把每一个列向量作为all_sample的一列赋值给all_sample
     %train_matrix_singlegraphic(:,i)= reshape(A,112*92,1);
 end
    %ImgData_HE{index_train}=
     %Train_Matrix(:,:,j)=train_matrix_singlegraphic;
 end
 end
   All_ImgData_HE_test{l}=ImgData_HE_test;
end
save ImgData_HE_itera All_ImgData_HE_train All_ImgData_HE_test 
