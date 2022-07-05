%  Testing Visualization
load('svmmodel.mat');

[filename, pathname] = uigetfile({'*.*';'*.bmp';'*.jpg';'*.gif'}, 'Pick a Leaf Image File');
imgname=horzcat(pathname,filename)
[class,dimg]=tellmeClass( imgname ,svmmodel);

%% Inserting Class Label
[r c j]=size(dimg);
position = [(r/2),2];
value = char(class(1));
RGB = insertText(dimg,position,value);
figure,imshow(RGB),title('Numeric values');
