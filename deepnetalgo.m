% %https://www.mathworks.com/help/vision/examples/image-category-classification-using-deep-learning.html
clear all
clc
close all
%% GPU UTILIZATION
p = gcp
if size(p),1==1
    p = gcp('nocreate');    % If no pool, do not create new one.
else
p = gcp
end
%% Dataset Paths
outputFolder='G:\';
rootFolder = fullfile(outputFolder, 'Documents');
allfoldernames= struct2table(dir(rootFolder));
for (i=3:height(allfoldernames))
    new(i-2)=allfoldernames.name(i);
end
clear i
categories=new;
%categories1 = {'airplanes', 'ferry', 'laptop'};
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource','foldernames');
tbl = countEachLabel(imds);
minSetCount = min(tbl{:,2}); % determine the smallest amount of images in a category
% Use splitEachLabel method to trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');
% Notice that each set now has exactly the same number of images.
countEachLabel(imds)


%%

% Find the first instance of an image for each category
%% Pretrained Net AlexNet
net = inceptionv3();
net.Layers(1)
net.Layers(end)

imr=net.Layers(1, 1).InputSize(:,1);
imc=net.Layers(1, 1).InputSize(:,2);

imds.ReadFcn = @(filename)readAndPreprocessImage(filename,imr,imc);
[trainingSet, testSet] = splitEachLabel(imds, 0.7, 'random');
% Get the network weights for the second convolutional layer
w1 = net.Layers(2).Weights;
%%   Resize weigts for vgg only
w1 = imresize(w1,[imr imc]);
%%
featureLayer = 'fc7';
%featureLayer = 'pool5-drop_7x7_s1';
%%
trainingFeatures = activations(net, trainingSet, featureLayer, ...
 'MiniBatchSize', 64, 'OutputAs', 'columns');


%%
% Get training labels from the trainingSet
trainingLabels = cellstr(trainingSet.Labels);

% Train multiclass SVM classifier using a fast linear solver, and set
% 'ObservationsIn' to 'columns' to match the arrangement used for training
% features.
classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');
%%
%%
% Extract test features using the CNN
testFeatures = activations(net, testSet, featureLayer, ...
 'MiniBatchSize', 64, 'OutputAs', 'columns');
%%
% Pass CNN image features to trained classifier

predictedLabels = predict(classifier, testFeatures, 'ObservationsIn', 'columns');


% Get the known labels
testLabels = testSet.Labels;

% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels, predictedLabels);

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2))

% Display the mean accuracy
mean(diag(confMat))
% Accuracy 
accuracy = mean(predictedLabels == testSet.Labels)
%%
 trainingFeatures =trainingFeatures';
 testFeatures =testFeatures';


%%
x1=trainingFeatures;
y1=trainingLabels;
xy1=array2table(x1);
xy1.type=y1;
%%
x2=testFeatures;
y2=testLabels;
xy2=array2table(x2);
xy2.type=y2;

cd F:\Study\MS(CS)\Papers\5_object\mat\
%%save model

save('xy_vgg_train_4096_DR','xy1','net','classifier');
save('xy_vgg_test_4096_DR','xy2','classifier','net');


%% Testing 
idx = randperm(numel(testSet.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
end
