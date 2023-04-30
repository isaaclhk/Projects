# Chest X-ray Classification (Pneumonia)

## Background
Classification of chest X-rays using artificial intelligence is a crucial and beneficial application of machine learning 
in the field of medical diagnosis. Pneumonia is a respiratory infection that is caused by bacteria, viruses, or fungi and is a leading cause of mortality worldwide, 
especially in young children (WHO, 2022). 
Early and accurate detection of pneumonia is crucial to initiate timely and effective treatment, which can save lives and prevent further complications.

Traditionally, pneumonia is diagnosed through chest X-rays that are interpreted by trained radiologists or physicians. However, this process can be time-consuming and expensive, which can lead to delayed treatment. 
Moreover, in developing countries, where access to trained radiologists and medical equipment are limited, there is a significant shortage of healthcare professionals who can accurately diagnose pneumonia (Frija et al., 2021).

Artificial intelligence can be used to overcome these challenges and improve the accuracy and efficiency of pneumonia diagnosis. 
By training machine learning models on large datasets of chest X-rays, these models can learn to recognize patterns and features that are indicative of pneumonia. 
When presented with a new X-ray image, the model can classify it as either healthy or pneumonia with a high degree of accuracy and speed, thus reducing the need for human intervention and improving the diagnosis process.
The benefits of using artificial intelligence in pneumonia diagnosis extend beyond accuracy and efficiency. 
It can also help reduce the workload of healthcare professionals, especially in resource-limited settings, allowing them to focus on more complex cases and providing better quality care to patients. 
Moreover, it can help bridge the gap in healthcare access, making pneumonia diagnosis more widely available to people who live in underserved areas or have limited access to healthcare.

In this project, we will create a convolutional neural network to classify chest x-ray images, predicting whether or not the the image
belongs to a patient with pneumonia. 

### About the Dataset
[This dataset](https://data.mendeley.com/datasets/rscbjbr9sj/2) comprises of anterior-posterior chest X-ray images taken from pediatric patients aged one to five years old who received medical care at Guangzhou Women and Children's Medical Center in Guangzhou. These images were acquired during the patients' routine clinical care. To ensure quality control, all chest radiographs were initially reviewed and low-quality or unreadable scans were removed. The images were then graded by two expert physicians, and a third expert checked the evaluation set to account for any grading errors (Kermany et al., 2018).

Originally, the dataset was divided into three folders (train, test, val) with subfolders for each image category (Pneumonia/Normal). However, I've combined all three folders to create two main folders (Pneumonia/Healthy) in order to manually shuffle and split the images into training, validation, and test sets of custom proportions. The dataset comprises 5,863 X-ray images (JPEG) categorized into two categories (Pneumonia/Normal).

## Code

```
#importing packages
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import tensorflow as tf
import pickle

##Setting directories
DIR = r"C:\Users\isaac\OneDrive\Documents\Projects\datasets\CXR"
os.listdir(DIR)

PNEUMONIA_DIR = os.path.join(DIR, 'PNEUMONIA')
HEALTHY_DIR = os.path.join(DIR, 'HEALTHY')

##Checking number of samples
print(f'number of pneumonia xray images:\
\n{len(os.listdir(PNEUMONIA_DIR))}')
print(f'number of healthy xray images:\
\n{len(os.listdir(HEALTHY_DIR))}')
```
We start by importing the relevant packages and and defining the working directory. </br>
Counting the length of pneumonia and healthy directories, the output shows that we have an imbalanced dataset. 

output:
```
number of pneumonia xray images:
4273
number of healthy xray images:
1583
```
Next, we begin to form our dataset by converting these xray images into numpy arrays. Here, I've defined a function that helps us do that. We'll also visualize the images to check if they are what we'd expected.
```
#create dataset
categories = ['healthy', 'pneumonia']

def create_dataset():
    images = []
    classes = []
    for category in categories:
        path = os.path.join(DIR, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img))
            new_array = cv2.resize(img_array, (299, 299))
            images.append(new_array)
            classes.append(class_num)
    
    return images, classes
            
images, classes = create_dataset()

#shuffle images
import random
random.seed(123)
temp = list(zip(images, classes))
random.shuffle(temp)
images, classes = zip(*temp)
images, classes = list(images), list(classes)

#visualize images
plt.figure(figsize = (20, 10))
for i in range(8):
    ax = plt.subplot(2,4, i+1)
    plt.imshow(images[i])
    plt.title(categories[classes[i]])
    plt.axis('off')
```



## References
1. https://www.who.int/news-room/fact-sheets/detail/pneumonia</br>
2. Frija, G., Blažić, I., Frush, D. P., Hierath, M., Kawooya, M., Donoso-Bach, L., & Brkljačić, B. (2021). 
3. How to improve access to medical imaging in low-and middle-income countries?. EClinicalMedicine, 38, 101034. </br>
Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), “Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification”, Mendeley Data, V2, doi: 10.17632/rscbjbr9sj.2
