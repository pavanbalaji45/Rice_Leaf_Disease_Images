# Rice_Leaf_Disease_Images

Dataset Link : https://www.kaggle.com/datasets/nirmalsankalana/rice-leaf-disease-image?select=Bacterialblight

Rice Leaf Disease Identification Using Deep Learning
This project uses deep learning techniques to identify and classify rice leaf diseases from images. It focuses on a dataset containing 5932 images of rice leaves, each labeled with one of four common rice leaf diseases: Bacterial Blight, Blast, Brown Spot, and Tungro.

The dataset was curated by Prabira Kumar Sethy, published in 2020, and is used here to train and evaluate machine learning models. The objective is to develop an image classification system capable of accurately identifying these diseases.

****Table of Contents**
Project Overview
Dataset
Technologies Used
Installation
Usage
Model Details
Results
Visualizations
License

Project Overview
In this project, we aim to build a classification model to identify rice leaf diseases using image data. The project employs Convolutional Neural Networks (CNN) and transfer learning using pre-trained models like VGG19 to classify images into one of four disease categories.

The model is trained and tested using a subset of the dataset, and its performance is evaluated using accuracy metrics.

Dataset
The Rice Leaf Disease Image Samples dataset includes 5932 images of rice leaves, each labeled as one of the following diseases:

Bacterial Blight
Blast
Brown Spot
Tungro
Each image serves as a valuable sample for understanding and identifying these diseases, which can help researchers and practitioners in the agricultural field.

Technologies Used
This project uses the following technologies:

Python: Programming language used for data processing, modeling, and evaluation.
TensorFlow & Keras: Frameworks used for building and training the deep learning models.
Scikit-learn: Used for splitting the dataset and preprocessing.
OpenCV & Pillow: Libraries for image manipulation and preprocessing.
Matplotlib & Seaborn: Used for visualizations and plotting the results.

Model Details
1. Basic CNN Model
A custom CNN model is built with the following layers:

Conv2D: Convolution layers to extract features.
MaxPooling2D: Pooling layers to reduce dimensionality.
Flatten: Flatten the output for fully connected layers.
Dense: Fully connected layers for classification.
Output Layer: Softmax activation for 4 output classes.

2. VGG19 Transfer Learning
We also explore transfer learning using the VGG19 model, which is pre-trained on the ImageNet dataset. The last fully connected layers are replaced with custom layers for this specific task.

Layers:

Pre-trained VGG19 Base: The convolutional layers are frozen (not trained).
Fully connected layers: Added custom layers after the base model to classify the rice diseases.
