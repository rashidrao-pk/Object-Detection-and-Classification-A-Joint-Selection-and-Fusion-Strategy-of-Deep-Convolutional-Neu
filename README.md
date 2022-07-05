# Object-Detection-and-Classification-A-Joint-Selection-and-Fusion-Strategy-of-Deep-Convolutional-Neu Object Detection and Classification: A Joint Selection and Fusion Strategy of Deep Convolutional Neural Network and SIFT Point Features

## Proposed Method
Object detection is an important task in the domain of computer vision and it gains much attention from last 2 decades based on their emerging application such as video surveillance and pedestrian detection. In this research, we deal with complex object detection and classification using three famous datasets such as Caltech101, PASCAL 3D, and 3D dataset. These datasets contain hundreds of object classes and thousands of images. To inspire with these datasets challenges, we proposed a new method for object detection and classification based on DCNN features extraction along with SIFT points. The proposed method consists of two major steps, which are executed in parallel. In the first step, SIFT point features are extracted from mapped RGB segmented object. Then, in the second step, DCNN features are extracted from pre-trained deep models such as AlexNet and VGG. The both SIFT point and DCNN features are combined in one matrix by a fusion method, which is later utilized for classification. The detailed description of each step is given below. Also, flow diagram of proposed method is shown in the Figure 1
![main](https://user-images.githubusercontent.com/25412736/165833103-2b8d62dc-6367-4464-84f9-0d9699981592.jpg)

# Segmentation using Improved Saliancy Method
In this step, we extract a single object from an image by an existing saliency method namely HDCT saliency estimation. The idea behind the improvement of saliency method is to implement the color spaces before gives the input image to saliency method. The LAB color transformation is utilized for this purpose. The LAB color transformation perceives color in 3 dimensions consisting of L for lightness, A and B are used for color components green-red and blue-yellow. The device independent is a major advantage of this transformation. The three components L is brighter white at 100 and darker black at 0, whereas ‘A’ and ‘B’ channels show the natural values for the RGB image. This transformation is defined as follows:
Let U(i,j) denotes an input RGB image having length N×M, then for RGB to LAB conversion, first RGB to XYZ conversion is performed as

![image](https://user-images.githubusercontent.com/25412736/177391991-186c5dcb-0fc0-46e7-89b8-28fce4fb4433.png)

Where, φ(X), φ(Y), and φ(Z) denotes the X, Y, and Z channels, which are extracted from red (φ^r), green (φ^g), and blue channel (φ^b). The φ^r, φ^g, and φ^b channels are defined as: 
![image](https://user-images.githubusercontent.com/25412736/177392075-8db330c1-020c-4285-8a55-1846f986fd17.png)

Then LAB conversion is defined as:
![image](https://user-images.githubusercontent.com/25412736/177392153-57614dc4-8578-4964-b52c-3a4afd61ef7a.png)

Where, f_x, f_y, and f_z are linear functions which are computed as:
![image](https://user-images.githubusercontent.com/25412736/177392224-f65ebf86-befc-4a41-853f-7a7a1e33ff06.png)

Thereafter, we employ a saliency approach for salient object detection. Salient region detection technique detects the salient region from an image by utilizing high dimensional color transform. In this work, the superpixel saliency features are used to detect the initial salient regions of the dermoscopic images. The superpixels of the LAB image are given as:

![image](https://user-images.githubusercontent.com/25412736/177392258-1dc16e1c-52b2-4ca4-8df5-df7a7235e6b7.png)

For low computational cost and better performance, we utilized the SLIC superpixel [36] with a total number of superpixels N=400. The color features are computed from LAB color space. The parameters which are used for color features extraction are mean, variance, standard deviation, and skewness. These color features are concatenated with the histogram features because the histogram features are effective for saliency approach. The euclidean distance is calculated between extracted color features as:

![image](https://user-images.githubusercontent.com/25412736/177392318-48f11fd5-2525-4335-bd61-08066feb313b.png)

Where, l_i and l_j denotes the ith and jth features in the given matrix A. In this work, the global contrast/color statistics of objects are used to define the saliency values of the pixels by using a histogram-based method. The saliency value of pixel defined as:
![image](https://user-images.githubusercontent.com/25412736/177392406-4d9ddd44-9323-4c0a-ac75-665a904a8504.png)

Where D ⃗(A) is the color distance between the features 〖l_i〗_i and the l_j in the LAB color space. By rearranging the above equation, we get the saliency value for each color as:

![image](https://user-images.githubusercontent.com/25412736/177392504-c0224c06-fc2b-45b7-89b3-32ac70932dde.png)

Where n, c_j, f_l  denotes the total number of different pixel color, the color value of pixel φ_k, and the frequency of the pixel color respectively. For shape and texture features, we utilized the HOG and the SFTA texture features. After the calculation of feature vector for each superpixel, the random forest regression is used to estimate the salient degree of each region. Further to identify the very salient pixels calculated from initial saliency map the Trimap is constructed by using adaptive thresholding. First, the input images divided into 2×2, 3×3, and 4×4 patches and then apply the Otsu thresholding on each patch individually. Finally, the Trimap is obtained by using global thresholding as:
![image](https://user-images.githubusercontent.com/25412736/177392565-83a1f95b-bb27-49aa-8d74-22bc762363c3.png)

Where τ denotes the global threshold value. After getting the optimal coefficient α (estimate for the saliency map) construct the saliency map as:

![image](https://user-images.githubusercontent.com/25412736/177392631-b3d382f3-4c8a-4b79-a70f-820796cd5556.png)
Where K denotes the high dimensional vector to present the color of the input image. The final map is obtained by adding the spatial and color-based saliency map as:
![image](https://user-images.githubusercontent.com/25412736/177392688-1f47b126-6807-4744-ba9c-1fadef7e2176.png)

The spatial saliency map is defined as:
![image](https://user-images.githubusercontent.com/25412736/177392735-0c9d0cb1-4c27-4d79-a0fc-93e8df7ab23c.png)

Where the K= 0.5, and 〖min〗_j∈β(d(P_i,P_j )) and 〖min〗_j∈f(d(P_i,P_j )) are the Euclidian distance from the ith pixel to definite background pixel and to definite foreground pixel respectively. The improved saliency method effects are shown in the Figure 2. In the Figure 2, the ist row shows input images, second rows present LAB transformation, third row defines improved saliency image in binary form, and last row depicts the mapped RGB image

![frrr](https://user-images.githubusercontent.com/25412736/165833373-0f131af6-fe4b-4285-8a0d-037d9115a4bd.jpg)


# SIFT Features
The SIFT features are computed in four steps. In the first step, local key points are determined that are important and stable for given images. The features
are extracted from each key point that explains the local image region samples, which are related to its scale space coordinate image. Then in the second step, weak features are removed by a specific threshold value. In the third step, orientations are assigned to each key point based on local image gradient directions. Finally, 1x128 dimensional feature vector is obtained and perform bilinear interpolation to improve the robustness of features

![image](https://user-images.githubusercontent.com/25412736/177392856-f1b3f96a-5988-44e9-9e84-97fc503e0d4e.png)

Where, ξ(u,v,σ) is scale space of an image, ψ_G (u,v,kσ) denotes the variable-scale Gaussian, k is multiplicative factor, and D(u,v,σ) denotes the difference of Gaussian convolved with segmented image

# Deep CNN Features
In this article, we employed two pre-trained deep CNN models such as VGG19 and AlexNet, which are used for features extraction. These models incorporate convolution layer, pooling layer, normalization layer, ReLu layer, and FC layer. As discussed above that convolution layer extract local features from an
image, which is formaulated as:

![image](https://user-images.githubusercontent.com/25412736/177392952-0657254b-b28d-4200-b75c-4f50cdb59a67.png)

Where, 〖g_i〗^((L)) denotes the output layer L, 〖b_i〗^((L)) is baise value, 〖ψ_(i,j)〗^((L)) denotes the filter connecting the jth feature map, and h_j denotes the L-1 output layer. Then, pooling layer is defined which extract maximum responses from the lower convolutional layer with an objective of reducing irrelavent features. The max pooling also resolve the problem of overfitting and mostly 2×2 polling is performed on extracted matrix. Matahmatically, max pooling is define as:
![image](https://user-images.githubusercontent.com/25412736/177393041-645d62d2-a12e-42e3-b4ec-1329e676fee7.png)

Where, S^L denots the stride, 〖m_1〗^((L)), 〖m_2〗^((L)), and 〖m_3〗^((L)) are defined filters for feature map such as 2×2, 3×3. The other layers such as ReLu and fully connected (FC) are defined as:

![image](https://user-images.githubusercontent.com/25412736/177393123-6954514b-2646-4a10-b601-416106c360aa.png)

Where, 〖Re〗_i^((l) ) denotes the ReLu layer, 〖Fc〗_i^((l)) denotes the FC layer. The FC layer follows the convolution and pooling layers. The FC layer is similar to convolution layer and most of the researchers perform activation on FC layer for deep feature extraction

# Features Extraction and Fusion

![fused_final_model](https://user-images.githubusercontent.com/25412736/165833120-00538d95-86b3-455f-87f8-2a590b0199ce.png)

# Max Pooling
![max pooling](https://user-images.githubusercontent.com/25412736/165833825-f3a55557-8005-45d5-bdbd-61d30ac069c2.png)


𝚷(𝑭𝒖𝒔𝒆𝒅) = (𝑁 × 1000) + (𝑁 × 1000) + (𝑁 × 100)
𝚷(𝑭𝒖𝒔𝒆𝒅) = 𝑁 × 𝑓𝑖

The size of final feature vector is 1 × 2100, which feed to ensemble classifier for classification. The ensemble classifier is a supervised learning method, which need to training data for prediction. Ensemble method combines several classifiers data to produce a better system. The formulation of ensemble method is given below. 
Let we have extracted features and their corresponding labels ((f_1,y_1 ),(f_2,y_2 ),…,(f_n,y_n )), where f_i denotes the extracted features which are typically vectors of form (f_(i+1),f_(i+2),…,f_(i+n)), then the unknown function is defined as y=f(x). An ensemble classifier is a set of classifiers whose individual decisions are combined in on classifier by tyical weights and voting. Hence the ensemble classifier is formulated as
![image](https://user-images.githubusercontent.com/25412736/177394287-25b40b4f-5cf9-4b60-a9e2-294f94092bfb.png)

Where, h_k (x)=h_1 (x),h_2 (x),…,h_k (x) and w ̂_k=w ̂_1,w ̂_2,… w ̂_k. The proposed method is tested on three datasets such as Caltech101, PASCAL 3D+ dataset, and 3D dataset. The sample labeled results are shown in the Figure 5 and 6. 


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


