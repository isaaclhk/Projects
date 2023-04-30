# Chest X-ray Classification (Pneumonia)

## Background
Classification of chest X-rays (CXR) using artificial intelligence is a crucial and beneficial application of machine learning 
in the field of medical diagnosis. Pneumonia is a respiratory infection that is caused by bacteria, viruses, or fungi and is a leading cause of mortality worldwide, 
especially in young children (WHO, 2022). 
Early and accurate detection of pneumonia is crucial to initiate timely and effective treatment, which can save lives and prevent further complications.

Traditionally, pneumonia is diagnosed through CXRs that are interpreted by trained radiologists or physicians. However, this process can be time-consuming and expensive, which can lead to delayed treatment. 
Moreover, in developing countries, where access to trained radiologists and medical equipment are limited, there is a significant shortage of healthcare professionals who can accurately diagnose pneumonia (Frija et al., 2021).

Artificial intelligence can be used to overcome these challenges and improve the accuracy and efficiency of pneumonia diagnosis. 
By training machine learning models on large datasets of chest X-rays, these models can learn to recognize patterns and features that are indicative of pneumonia. 
When presented with a new X-ray image, the model can classify it as either healthy or pneumonia with a high degree of accuracy and speed, thus reducing the need for human intervention and improving the diagnosis process.
The benefits of using artificial intelligence in pneumonia diagnosis extend beyond accuracy and efficiency. 
It can also help reduce the workload of healthcare professionals, especially in resource-limited settings, allowing them to focus on more complex cases and providing better quality care to patients. 
Moreover, it can help bridge the gap in healthcare access, making pneumonia diagnosis more widely available to people who live in underserved areas or have limited access to healthcare.

In this project, we will create a convolutional neural network to classify CXRs images, predicting whether or not the the image
belongs to a patient with pneumonia. 

### About the Dataset
[This dataset](https://data.mendeley.com/datasets/rscbjbr9sj/2) comprises of anterior-posterior CXR images taken from pediatric patients aged one to five years old who received medical care at Guangzhou Women and Children's Medical Center in Guangzhou. These images were acquired during the patients' routine clinical care. To ensure quality control, all chest radiographs were initially reviewed and low-quality or unreadable scans were removed. The images were then graded by two expert physicians, and a third expert checked the evaluation set to account for any grading errors (Kermany et al., 2018).

Originally, the dataset was divided into three folders (train, test, val) with subfolders for each image category (Pneumonia/Normal). However, I've combined all three folders to create two main folders (Pneumonia/Healthy) in order to manually shuffle and split the images into training, validation, and test sets of custom proportions. The dataset comprises 5,863 X-ray images (JPEG) categorized into two categories (Pneumonia/Normal).

## Code
### Preparation
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
Having an imbalanced dataset is potentially problematic as it could lead the machine learning model we are training to be biased towards the majority class. In this dataset, pneumonia xray images are overrepresented. If a model is trained on this dataset, it will see comparatively fewer healthy xray images. Consequently, it may not perform as well in classifying healthy CXRs. This might cause the model to over specialize in recognizing pneumonia CXRs, resulting in more false positives. One method to overcome the problems that come with imbalanced datasets is data augmentation, which will be demonstrated later in this project.

Next, we begin to form our dataset by converting these xray images into numpy arrays. Here, I've defined a function that helps us do that. We'll also visualize the images to check that they're are what we'd expected.
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
output:
![visualize_cxr](https://user-images.githubusercontent.com/71438259/235347171-604bad5f-0bed-4519-9415-d77e09f8b038.png)

In general, healthy chest x-rays should appear clear and black, indicating that air is passing through the lung spaces freely without much resistance. On the contrary, pneumonia chest x-rays might show areas of patchy or diffused opacity, indicating inflammation or fluid build-up. Interpreting chest x-rays can be challenging. Physicians undergo years of training and rely on their extensive clinical experience to read these images. In addition, it is not always reliable to form a diagnosis based solely on xray images. clinicians rely on additional information such as laboratory tests, the patient's history and symptoms to form accurate diagnoses.
</br>

We split the dataset into training, validation, and testing sets in an 8-1-1 ratio. The split is stratified in a way that ensures that there are approximately equal proportions of healthy and pneumonia images in each set.

```
#split data
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(
    images, classes, 
    test_size = 0.2, 
    random_state = 123, 
    shuffle = True,
    stratify = classes)

x_validation, x_test, y_validation, y_test = train_test_split(
    x_val, y_val,
    test_size = 0.5,
    random_state = 123,
    shuffle = True,
    stratify = y_val)

#check if stratified
print(f'proportion of healthy cxr images in the training set = {y_train.count(0)/ len(y_train):.4f}')
print(f'proportion of healthy cxr images in the validation set = {y_validation.count(0)/ len(y_validation):.4f}')
print(f'proportion of healthy cxr images in the testing set = {y_test.count(0)/ len(y_test):.4f}')
```

output:
```
proportion of healthy cxr images in the training set = 0.2703
proportion of healthy cxr images in the validation set = 0.2713
proportion of healthy cxr images in the testing set = 0.2696
```

### Data augmentation
As explained earlier, imbalanced datasets can cause machine learning models to be biased towards the majority class, resulting in an inferior, less accurate model. To overcome this problem in this project, we will create augmented versions of healthy CXR images in the training set and add them to the original training set. We will then train our model on this combined training set with equal number of pneumonia and healthy CXR images. No augmented images will be added to the validation and testing sets as we want a reliable evaluation of the model's performance on authentic CXR images.

```
#calculate number of batches of augmented data to generate
num_needed = len(train_pneumonia) - len(train_healthy) #total number of augmented images required
num_needed

#generate augmented images
def generate_augmented_data(train_healthy, num_batches, batch_size):
    os.mkdir(os.path.join(DIR, 'AUGMENTED')) #create new directory
    AUGMENTED_DIR = os.path.join(DIR, 'AUGMENTED')
    i = 0
    for batch in datagen.flow(train_healthy, batch_size = batch_size,
                              save_to_dir = AUGMENTED_DIR,
                              save_format = 'jpeg'):
        i+= 1
        if i >= num_batches:
            break
        
    return AUGMENTED_DIR # returns path where augmented images are saved

AUGMENTED_DIR = generate_augmented_data(train_healthy, num_batches = num_needed, batch_size = 1)
print('number of augmented images created: {}'.format(len(os.listdir(AUGMENTED_DIR))))
```

In the above code, I've defined a function that creates an empty folder 'AUGMENTED', then generates and saves the specified number of augmented images into that folder.
We've created enough images to correct the imbalance in the original training set.
</br>

output:
```
number of augmented images created: 2152
```

Now that we have the augmented images, we'll add them to the original training set.

```
#combine augmented and authentic healthy cxr for balanced training set
def combined_training_set(AUGMENTED_DIR, num_augmented, x_train, y_train):
    augmented_imgs = []
    for img in os.listdir(AUGMENTED_DIR):
        img_array = cv2.imread(os.path.join(AUGMENTED_DIR, img))
        augmented_imgs.append(img_array)
    
    combined_x_train = np.concatenate((np.array(augmented_imgs), x_train), axis = 0)
    combined_y_train = np.array([0]*num_needed + y_train)
    
    return combined_x_train, combined_y_train
    
combined_x_train, combined_y_train = combined_training_set(
    AUGMENTED_DIR, 
    num_needed, 
    x_train, 
    y_train)

#verify shape
combined_x_train.shape #(6836, 299, 299, 3)
combined_y_train.shape #(6836,)
```

Next, we shuffle the combined training set before saving it. It is crucial to save our data at this point so that we need not reproduce the augemented images. More importantly, saving allows us to work on a consistent set of splits everytime we resume the work on this model.

```
#shuffle training set
from sklearn.utils import shuffle
combined_x_train, combined_y_train = shuffle(combined_x_train, combined_y_train, random_state = 123)

DIR = r"C:\Users\isaac\OneDrive\Documents\Projects\datasets\CXR"
save_DIR = os.path.join(DIR, 'saved') #directory for saved files

'''
#save
with open(os.path.join(save_DIR, 'cxr_dataset.pkl'), 'wb') as f:
    pickle.dump([combined_x_train, combined_y_train, x_validation, y_validation, x_test, y_test], f)
    
with open(os.path.join(save_DIR, 'cxr_dataset.pkl'), 'rb') as f:
    combined_x_train, combined_y_train, x_validation, y_validation, x_test, y_test = pickle.load(f)
'''

#check if dataset balanced
print('number of healthy chest xrays: {}'.format(list(combined_y_train).count(0)))
print('number of pneumonia chest xrays: {}'.format(list(combined_y_train).count(1)))
```

output:
```
number of healthy chest xrays: 3418
number of pneumonia chest xrays: 3418
```

## References
1. https://www.who.int/news-room/fact-sheets/detail/pneumonia</br>
2. Frija, G., Blažić, I., Frush, D. P., Hierath, M., Kawooya, M., Donoso-Bach, L., & Brkljačić, B. (2021). 
3. How to improve access to medical imaging in low-and middle-income countries?. EClinicalMedicine, 38, 101034. </br>
Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), “Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification”, Mendeley Data, V2, doi: 10.17632/rscbjbr9sj.2
