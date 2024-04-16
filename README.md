# Brain Tumor MRI Classification Using CNN Models
We have developed a system that utilizes Convolutional Neural Networks to improve medical image classification for identifying brain tumors in patients. Our project focuses on designing a system that utilizes deep learning techniques, particularly Convolutional Neural Networks (CNNs), to address a specific problem in the domain of Computer Vision (CV), such as medical image analysis for disease detection. We applied various data augmentations to improve performance and explored how different augmentation techniques affected the accuracy and robustness of CNN model in the medical image classification tasks of identifying various types of brain tumors.

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

The training dataset underwent splitting into training and validation sets, allocating 20% of the data specifically for validation purposes. 
The delineation is as follows:

•	No Tumor: 
Training Data: 1595 
Testing Data: 405 

•	Glioma:
Training Data: 1321 
Testing Data: 300 

•	Meningioma:
Training Data: 1339 
Testing Data: 306 

•	Pituitary Tumor:
Training Data: 1457 
Testing Data: 300 


### Data Pre-processing

Preprocessing steps are fundamental in refining input data quality and elevating CNN model performance. In our experimentation with various data augmentation techniques, we observed a discernible impact on the performance metrics of our model. Employing a trial-and-error approach, we noted that an increase in the number of augmentation methods corresponded with a decline in the model's performance and accuracy in predicting tumor characteristics. For instance, upon employing the entirety of the listed augmentation techniques, the accuracy score plateaued at 70%. Conversely, when restricting augmentation solely to normalization, specifically rescaling pixel values to the range [0,1], a notable improvement was evident, with the accuracy score surging to 94.38%. The subsequent preprocessing methodologies that we experimented with encompassed:

Data Augmentation:
This step involved the application of diverse techniques to augment the dataset, thereby enhancing its variability and bolstering the CNN model's ability to generalize. Techniques such as rotation, width and height shifting, shearing, zooming, flipping, brightness adjustment, and channel shifting were employed to achieve this augmentation.

Normalization:
In this process, pixel values within the images were rescaled to fit within the range of [0, 1]. This standardization ensured uniformity across the dataset and facilitated smoother convergence during the model training phase.

Standardization:
Both feature-wise and sample-wise standardization were implemented to regulate the distribution of features within the dataset. Feature-wise standardization ensured that each feature maintained a mean of 0 and a standard deviation of 1 across the dataset, while sample-wise standardization maintained consistency within individual samples.

ZCA Whitening:
This technique was utilized to reduce redundancy within image features by decorrelating them. By employing ZCA whitening, the convergence speed of the CNN model was improved, as redundant information was minimized, leading to more efficient training processes.


## Pre-trained Model

A pre-trained model is a model that was trained on a large benchmark dataset to solve a problem similar to the one that we want to solve. Accordingly, due to the computational cost of training such models, it is common practice to import and use models from published literature (e.g. VGG, Inception, ResNet). For this project, I decided to use **ResNet152V2** model to perform image classification for brain tumor MRI images.[ResNet152V2](https://keras.io/api/applications/resnet/)

## Note
You can see more details about training steps and testing results inside [brain_tumor_cnn_classification.ipynb](https://github.com/btlambodh/brain-tumor-classification/blob/main/brain_tumor_cnn_classification.ipynb)
