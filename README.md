# Object-Detection-and-Classification-A-Joint-Selection-and-Fusion-Strategy-of-Deep-Convolutional-Neu Object Detection and Classification: A Joint Selection and Fusion Strategy of Deep Convolutional Neural Network and SIFT Point Features

Object detection is an important task in the domain of computer vision and it gains much attention from last 2 decades based on their emerging application such as video surveillance and pedestrian detection. In this research, we deal with complex object detection and classification using three famous datasets such as Caltech101, PASCAL 3D, and 3D dataset. These datasets contain hundreds of object classes and thousands of images. To inspire with these datasets challenges, we proposed a new method for object detection and classification based on DCNN features extraction along with SIFT points. The proposed method consists of two major steps, which are executed in parallel. In the first step, SIFT point features are extracted from mapped RGB segmented object. Then, in the second step, DCNN features are extracted from pre-trained deep models such as AlexNet and VGG. The both SIFT point and DCNN features are combined in one matrix by a fusion method, which is later utilized for classification. The detailed description of each step is given below. Also, flow diagram of proposed method is shown in the Figure 1
![main](https://user-images.githubusercontent.com/25412736/165833103-2b8d62dc-6367-4464-84f9-0d9699981592.jpg)

# Segmentation using Improved Saliancy Method

![frrr](https://user-images.githubusercontent.com/25412736/165833373-0f131af6-fe4b-4285-8a0d-037d9115a4bd.jpg)

# SIFT Features
The SIFT features are computed in four steps. In the first step, local key points are determined that are important and stable for given images. The features
are extracted from each key point that explains the local image region samples, which are related to its scale space coordinate image. Then in the second step, weak features are removed by a specific threshold value. In the third step, orientations are assigned to each key point based on local image gradient directions. Finally, 1x128 dimensional feature vector is obtained and perform bilinear interpolation to improve the robustness of features

# Deep CNN Features
In this article, we employed two pre-trained deep CNN models such as VGG19 and AlexNet, which are used for features extraction. These models incorporate convolution layer, pooling layer, normalization layer, ReLu layer, and FC layer. As discussed above that convolution layer extract local features from an
image

# Features Extraction and Fusion

![fused_final_model](https://user-images.githubusercontent.com/25412736/165833120-00538d95-86b3-455f-87f8-2a590b0199ce.png)

# Max Pooling
![max pooling](https://user-images.githubusercontent.com/25412736/165833825-f3a55557-8005-45d5-bdbd-61d30ac069c2.png)


𝚷(𝑭𝒖𝒔𝒆𝒅) = (𝑁 × 1000) + (𝑁 × 1000) + (𝑁 × 100)
𝚷(𝑭𝒖𝒔𝒆𝒅) = 𝑁 × 𝑓𝑖

The size of final feature vector is 1 × 2100, which feed to ensemble classifier for classification. The ensemble classifier is a supervised learning method, which need to training data for prediction. Ensemble method combines several classifiers data to produce a better system


# Results

![Labeled_Caltech](https://user-images.githubusercontent.com/25412736/165834152-3239e7f0-e4be-4ab3-bf08-02a1448b91fe.png)
<br><center>Caltech-101 Dataset</center>

![Uploading Labeled_Pascal.png…]()
<br><center>Pascal3D Dataset</center>

![Uploading Labeled_3Ddataset.png…]()
<br><center>3D Dataset</center>

![time](https://user-images.githubusercontent.com/25412736/165834086-b5ff6d1a-cdbb-4a97-9c51-f3b7df6af1e7.jpg)

![confusion](https://user-images.githubusercontent.com/25412736/165834228-60473efa-4334-4c62-8289-f1c7190a88a6.jpg)

![caltech101](https://user-images.githubusercontent.com/25412736/165834261-d3c59dc2-54eb-48f0-acb1-70fb08e50b1d.jpg)
<br><center>Caltech-101 Dataset</center>

![PASCAl3D](https://user-images.githubusercontent.com/25412736/165834267-96182f58-ceeb-4c00-b366-48b654193b1f.jpg)
<br><center>Pascal3D Dataset</center>

![Barkely](https://user-images.githubusercontent.com/25412736/165834279-b80ccbdd-200a-4077-a1f5-1299923216f3.jpg)
<br><center>Barkely Dataset </center>

Muhammad Rashid, Muhammad Attique Khan, Muhammad Sharif, Mudassar Raza, Muhammad Masood, Farhat Afza (Multimedia Tools and Applications: IF 7.42 | 08-12-2018) 


