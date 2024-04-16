# Brain Tumor MRI Classification Using CNN Models

## What is a brain tumor?

A brain tumor refers to an accumulation of abnormal cells within the brain. The skull, which houses the brain, is extremely rigid, leaving little room for any internal growth. Such growth can lead to complications, whether the tumor is benign (noncancerous) or malignant (cancerous). As these tumors enlarge, they can elevate the pressure inside the skull, potentially leading to brain damage and posing serious health risks.

## The significance of the topic

Early detection and classification of brain tumors are crucial areas of study in medical imaging. This research significantly aids in choosing the most appropriate treatment methods to save patients' lives.

## Basic Requirements

| **Package Name**      | **Version** |
| --------------------- | ----------- |
| `python`              |  3.11.5     |
| `tensorflow`          |  2.16.1     |
| `keras`               |  3.1.1      |
| `matplotlib`          |  3.7.2      |
| `scikit-learn`        |  1.3.0      |

## Dataset

The dataset was taken from [here](https://www.kaggle.com/masoudnickparvar/brain-tumor-mri-dataset).

### Dataset Details

This dataset contains **7022** images of human brain MRI images which are classified into 4 classes:

- glioma
- meningioma
- no tumor
- pituitary

About 22% of the images are intended for model testing and the rest for model training.
Pay attention that The size of the images in this dataset is different. You can resize images to the desired size after pre-processing and removing the extra margins.

### Data Pre-processing

Crop the part of the image that contains only the brain (which is the most important part of the image): The cropping technique is used to find the extreme top, bottom, left and right points of the brain using OpenCV.

## Pre-trained Model

A pre-trained model is a model that was trained on a large benchmark dataset to solve a problem similar to the one that we want to solve. Accordingly, due to the computational cost of training such models, it is common practice to import and use models from published literature (e.g. VGG, Inception, ResNet). For this project, I decided to use **ResNet152V2** model to perform image classification for brain tumor MRI images.[ResNet152V2](https://keras.io/api/applications/resnet/)

## Note
You can see more details about training steps and testing results inside [brain_tumor_cnn_classification.ipynb](https://github.com/btlambodh/brain-tumor-classification/blob/main/brain_tumor_cnn_classification.ipynb)
