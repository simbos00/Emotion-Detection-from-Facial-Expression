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

# How to run





The ipynb files are made to run on Google Colab. The "emotion\_model\_weights.keras" file and the folder `shap` must be uploaded to Google Drive.

