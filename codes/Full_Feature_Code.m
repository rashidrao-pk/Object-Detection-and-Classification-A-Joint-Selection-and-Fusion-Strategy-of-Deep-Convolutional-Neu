clc
clear all
close all
workspace;
datasetname='F:\Datasets\Skin\Classification_DS\ISBI2016';
rootFolder = fullfile(datasetname);
allfoldernames= struct2table(dir(rootFolder));
for i=3:height(allfoldernames) 
    new(i-2)=allfoldernames.name(i);
end

clear i
class=new;

trainingDataSizePercent = 0.8;
%% Load Image dataset
imgSets = [];
for i = 1:length(class)
    imgSets = [ imgSets, imageSet(fullfile(datasetname, class {i})) ];
end

%% Class Balancing Using Min Count
minClassCount = min([imgSets.Count]);
imgSets = partition(imgSets, minClassCount, 'sequential'); % Or 'randomize'

net1 = alexnet;
net2 = vgg19;
featureLayer = 'fc7';
featureLayer = 'fc7';

%% Prepare Training and Validation Image Sets
[trainingSet, validationSet] = partition(imgSets, trainingDataSizePercent, 'sequential'); % Or 'randomize'
ytrain=(repelem({trainingSet.Description}', ...
[trainingSet.Count], 1));
imagecount=1;
% s = size(imagespath,4);
tic
disp('Feature Extraction Started');
for i=1 : size(trainingSet,2)
    m=size(trainingSet(i).ImageLocation,2);
    temp=trainingSet(i).ImageLocation;
     for j=1 :  m
        v{imagecount,1}=temp{j};
        if(~isempty(strfind(temp{j},new(1,i))))
                v{imagecount,2}=new(1,i);    
        else
            v{imagecount,2}='None';
        end     
            img=imread(v{imagecount,1});
            %% Preprocessing
            img = imadjust(img,stretchlim(img),[]);
            inr1 = net1.Layers(1,1).InputSize(1);
            inc1 = net1.Layers(1,1).InputSize(2);
            inch1 = net1.Layers(1,1).InputSize(3);
            
            inr2 = net2.Layers(1,1).InputSize(1);
            inc2 = net2.Layers(1,1).InputSize(2);
            
            img_dcnn1=imresize(img,[inr1,inc1]);
            img_dcnn2=imresize(img,[inr2,inc2]);
            
            if(size(img, 3) == 3)
            img_gray=rgb2gray(img);
            end
            img=imresize(img,[256,256]);
            % subplot(221),imshow(img),title('Input');
           img = imadjust(img,stretchlim(img),[]);
            %% Feature Extraction 
           % Filtering Features
           feature_SIFT{imagecount,1} =   fun_Module_SIFT(img_gray);
           % 1st Neural Network Features
           feature_dcnn1{imagecount,1}   =   activations(net1, img_dcnn1, featureLayer1, ...
                                        'MiniBatchSize', 64, 'OutputAs', 'rows');
           % 2nd Neural Network Features
           feature_dcnn2{imagecount,1}   =   activations(net2, img_dcnn2, featureLayer2, ...
                                        'MiniBatchSize', 64, 'OutputAs', 'rows');
            clear featureVector img img_gray hogVisualization
        imagecount=imagecount+1;
        disp([ num2str(i) '-' num2str(j)]);
        
     end 
end
disp(['Feature Extraction took ',num2str(toc),' seconds']);
clear img img_gray image_count i imgSets j m new rootFolder temp trainingDataSizePercentage 

for i=1:length(feature_SIFT)
    FV1_SIFT(i,:)=double(feature_SIFT{i});
end

for i=1:length(feature_dcnn1)
    ftemp=double(feature_dcnn1{i});
    FV2_DCNN1(i,:)=ftemp;
end

for i=1:length(feature_dcnn2)
    ftemp=double(feature_dcnn2{i});
    FV3_DCNN2(i,:)=ftemp;
end

X=v(:,2);
%% Feature Reduction
% tic
% [r, c]=size(FV5_DCNN);
% FV5_hog_Reduced = Find_Entropy(FV5_hog,c); 
% FV5_hog_Reduced = FV5_hog_Reduced(:,1:2000);
% disp(['Feature Reduction took ',num2str(toc),' seconds']);
%% SerialBasesFusion    
    tic
fused_all=horzcat(FV1_SIFT,FV2_DCNN1,FV3_DCNN2);  %% All
  
  FV1=horzcat(FV1_SIFT,FV2_DCNN1);
  FV2=horzcat(FV2_DCNN1,FV3_DCNN2);
  FV3=horzcat(FV1_SIFT,FV3_DCNN2);
  FV4=FV1_SIFT; % FV1_filter
  FV5=FV2_DCNN1;   % FV2_stats
  FV6=FV3_DCNN2;  % FV3_color
     
  disp(['Serial Based Fusion took ',num2str(toc),' seconds']);

%% All Exp
FV_1=num2cell(FV1);
Final_FV_1=horzcat(X,FV_1);
Final_FV_1=cell2table(Final_FV_1);

FV_2=num2cell(FV2);    
Final_FV_2=horzcat(X,FV_2);
Final_FV_2=cell2table(Final_FV_2);

FV_3=num2cell(FV3);
Final_FV_3=horzcat(X,FV_3);
Final_FV_3=cell2table(Final_FV_3);

FV_4=num2cell(FV4);
Final_FV_4=horzcat(X,FV_4);
Final_FV_4=cell2table(Final_FV_4);

FV_5=num2cell(FV5);
Final_FV_5=horzcat(X,FV_5);
Final_FV_5=cell2table(Final_FV_5);

FV_6=num2cell(FV6);
Final_FV_6=horzcat(X,FV_6);
Final_FV_6=cell2table(Final_FV_6);
  
% % % % % % % % % % % % %   
  disp(['Feature Vector Finalization took ',num2str(toc),' seonds']);
  clear fused1 fused2 fused3 fused4 fused5 fused6 fused_ent allfoldernames c class 
  clear fused feature_color feature_filter feature_lbp feature_stats i r imagecount
  clear FV2_STATS FV5_DCNN FV4_LBP FV3_COLOR FV1_FILTER
  clear FV1 FV2 FV3 FV4 FV5 FV6 FV7 FV8 FV9
  clear FV_1 FV_2 FV_3 FV_4 FV_5 FV_6 FV_7 FV_8 FV_9
  
