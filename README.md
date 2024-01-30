## Overview
Distracted driving poses significant risks, including accidents, injuries, and fatalities. Identifying and mitigating instances of distraction while driving is crucial to reducing road accidents.

The ballooning of car insurance claims led Directline Insurance, Kenya, to engage us in this project, with a vision to lower the rising claims from their customers.
## BUSINESS OBJECTIVES
Main Objective:
Create a robust system for real-time detection of distracted drivers using computer vision.
### Specific Objectives:
i.) Dataset Acquisition: Source a credible dataset comprising images of drivers exhibiting various forms of distraction (e.g., phone usage, eating, grooming). Ensure the data acquired is correctly labeled and put into appropriate categories. This will allow the model to have a strong basis on which to train on.<br>
ii.) Model Development: Utilize computer vision techniques to build a deep learning model capable of accurately identifying distracted driver behaviors from visual cues in the data, providing more detailed insights for targeted interventions.<br>
iii.) Real-Time Implementation: Deploy a feedback system that alerts the driver when they’re being distracted by integrating auditory alerts or haptic feedback, enhancing the immediacy and effectiveness of distraction warnings.

### Success/Performance Criteria:
i.) Accuracy: Measure the model's ability to correctly identify instances of driver distraction against a labeled dataset. e’re aiming for a minimum test accuracy of 0.9.<br>
ii.) Real-Time Performance: Assess the system's efficiency in processing and detecting distractions within an acceptable time frame (e.g., milliseconds).<br>
iii.) Successful haptic or audio feedback on detecting distraction.<br>
iv.) Robustness: Test the model's performance across diverse environmental conditions, varying lighting, and different types of distractions to ensure consistent and reliable detection.<br>
## IMAGE DATA UNDERSTANDING
We sourced our data from the creators at the American University in Cairo. It consists of 14,478 images of different people, whose actions indicate either safe driving or distracted driving.<br>
The 10 classes to be predicted are:<br>
    c0: safe driving<br>
    c1: texting - right<br>
    c2: talking on the phone - right<br>
    c3: texting - left<br>
    c4: talking on the phone - left<br>
    c5: operating the radio<br>
    c6: drinking<br>
    c7: reaching behind<br>
    c8: hair and makeup<br>
    c9: talking to passenger<br>

We notice that the images are in two sets: Camera 1 and Camera 2. Camera 1 images were taken from the right of the object while Camera 2 images were taken from the left.
## PROJECT STRUCTURE

### 1. Exploratory Data Analysis (EDA)
Data Understanding: Load and inspect the dataset, exploring key metrics and labels, and distribution
Data Cleaning and Analysis: clean, transform, and analyse business understanding data
Image Augmentation:augment images to improve the performance of deep neural networks for computer vision tasks such as classification, segmentation, and object detection.
Blur Detection and Advanced Augmentation: we introduced a blur detection algorithm and an augmentation algorithm that involves deblurring, contrast restoration, flipping, rotation, and brightness adjustment.

### 2. Model Development
#### CNN Models
We built CNN models on both Camera 1 images alone the combined data for both cameras
We also built CNN models on the deblurred images
#### VGG16 and VGG19 Models
We built VGG16 and VGG19 models on both Camera 1 images alone the combined data for both cameras
#### Transfer Learning and Image Deblurring
To improve computational efficiency and improve performance, we used pretrained models on deblurred images 
These models are:
VGG16
VGG19
EfficientNetB0
EfficientNetB3
#### GoogLeNet Inception Network
We trained both GoogLeNet Inception with both two and three inception modules on the deblurred data.

### 3. Model Evaluation
Assess the performance of each model using appropriate metrics.
Compare the accuracy of the diffrent models.

### 4. Model Deployment
We deployed model on Streamlit, in such a way that it can detect a distracted driver and play an audio alerting him or her to focus on the road.
Streamlit app- https://distracteddriverdetection.streamlit.app/

## RESULTS
GoogLeNet Inception with 3 inception modules was the best performing with an accuracy of 35%.

## Contributors
Leonard Gachimu<br>
Rowlandson Kariuki<br>
Onesphoro Kibunja<br>
Francis Njenga<br>
Mourine Mwangi<br>
Victor Mawira<br>
Khadija Omar


