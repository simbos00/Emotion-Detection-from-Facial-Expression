# Emotion Detection from Facial Expression

Emotion detection from facial expression with Convolutional Neural Networks.  It works with images, videos, and real-time webcome input.



# Introduction

The aim of this project is to develop a deep learning model that can accurately classify facial expressions into one of seven emotional categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral. The project specifically focuses on implementing and evaluating Convolutional Neural Networks (CNN) for this task, excluding alternative model architectures or additional datasets. Our approach involves training a CNN model from scratch on the FER-2013 dataset (Sambare 2025) to enable the model to learn and distinguish between the specified facial expression categories.

# Description of the Repo

In this repo you can find different python scripts and different files. Here's a brief description:

* AI\_Project.pdf - A complete report of the project.
* train\_model.py - The script we used to train our model from scratch, including the downloading of the dataset.
* realtime\_emotion.py - The script that uses our CNN model to detect emotions from the webcam input.
* confmat\_debug\_imgtest.ipynb - The script that uses our CNN model to detect emotions from pictures.
* VideoRecognition\_TBBT.py - The script that uses our CNN model to detect emotions from a video.
* Clip\_SheldonSmile.mp4 - The video we use as an example.
* SHAP\_analysis.ipynb - The script we used for the SHAP analysis.
* im1. im2, im3 - Pictures showing some benchmark outputs of our model.

# Methodology

To achieve our objective we used a Convolutional Neural Network (CNN). A CNN is a deep learning architecture specifically designed for processing grid-like data, such as images, by preserving spatial information.

The core of the network consists of four distinct convolutional blocks. Each block follows a strict, repetitive design pattern: Convolution → Batch Normalization → Max Pooling → Dropout. 
Once the feature extraction is complete, the multi-dimensional feature maps are collapsed into a 1D vector via a Flatten layer. This vector is fed into a dense classification head consisting of two fully connected hidden layers, the first containing 256 units and the second expands to 512.
Both dense layers utilize ELU activation, consistently with the convolutional part. They are also regularized using Batch Normalization and Dropout.

The model ends with an Output Layer of 7 neurons (matching the 7 emotion classes in FER-2013) activated by a Softmax function. This produces a probability distribution summing to 1.0, allowing the model to make a categorical prediction.

Considering the whole model, the number of parameters is 3,002,887 (corresponding to 11.46 MB).


# How to run

To run the .py scripts we used a computer running Linux Fedora. You can use Windows too but the latest version of TensorFlow (2.20)

does not support the training on the GPU on Windows, so the training time is significantly longer.



The packages needed are: tensorflow kagglehub matplotlib opencv-python.

To train it using an Nvidia GPU cudatoolkit=11.8 cudnn=8.9 and tensorflow\[and-cuda] are needed. (More info in the report).



The ipynb files are made to run on Google Colab. The "emotion\_model\_weights.keras" file and the folder `shap` must be uploaded to Google Drive.

# Results and Conclusion:

In this study, we explored the use of Convolutional Neural Networks (CNNs) for facial emotion recognition using real-world static images, recorded videos, and live camera feeds.
Through the test dataset we achieved an accuracy of 67%. 
While the results are promising, several factors must be addressed to enhance model performance: increasing image resolution to capture finer details, increasing the training set, reducing the feature imbalance, accounting for the non-universality of facial expressions through additional covariates, and integrating cultural variations in how emotions are expressed and perceived. 
Despite the model’s inherent complexity and low explainability, SHAP analysis highlighted the importance of the ”communication triangle” or ”T-shape” (comprising the eyes and mouth) in emotion recognition, aligning with eye-tracking studies revealing that human observers rely on the same facial regions to interpret emotions. 
Furthermore, deploying such models in real-world scenarios necessitates a rigorous focus
on ethical standards, privacy protections, and regulatory compliance (e.g., EU AI Act).

