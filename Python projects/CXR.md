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

## Data Preprocessing
### Loading data
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

![visualize_cxr](https://user-images.githubusercontent.com/71438259/235392071-6fed25b7-fd98-481e-be3d-2bfbfaf46dde.png)

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
As explained earlier, imbalanced datasets can cause machine learning models to be biased towards the majority class, resulting in an inferior, less accurate model. To overcome this problem in this project, we will create augmented versions of healthy CXR images in the training set and add them to the original training set. We will then train our model on this combined training set with equal number of pneumonia and healthy CXR images. No augmented images will be added to the validation and testing sets as we want a reliable evaluation of the model's performance on authentic CXR images. </br>

To generate augmented images, we must first specify how we'd like to adjust the original image. Some ways in which we can augment images are listed in [this tensorflow documentation](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator).

```
#prepare for data augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range = 0.3,
    horizontal_flip = True,
    brightness_range = [0.9, 1.1],
    width_shift_range = 0.05,
    height_shift_range = 0.05)
    
#visualize augmented images
aug_iter = datagen.flow(np.array(images), batch_size = 1)
fig, ax = plt.subplots(1,5, figsize=(20,10))
# generate batch of images
for i in range(5):
	# convert to unsigned integers
	image = next(aug_iter)[0].astype('uint8')
	# plot image
	ax[i].imshow(image)
	ax[i].axis('off')
```    

![augmented_cxr](https://user-images.githubusercontent.com/71438259/235392089-e9e65ea4-7c36-437f-a5d3-29048aaec481.png)

A visual inspection of the augmented images show that our images are reasonably realistic, and look like what we might find in another set of samples.
we observe that some images have been randomly flipped horizontally. In reality, dextrocardia, a congenital condition where one's heart is situated on the right side of his/her chest instead of the left, is rare. However, we've simulated this by flipping some of our CXR images.

 
 ```
#separate healthy and pneumonia images in training set
def sep_images_by_class(x_train, y_train):
    train_healthy = []
    train_pneumonia = []
    temp = zip(x_train, y_train)
    for x, y in temp:
        if y == 0:
            train_healthy.append(x)
        else:
            train_pneumonia.append(x)

    train_healthy = np.array(train_healthy)
    train_pneumonia = np.array(train_pneumonia)
    
    return train_healthy, train_pneumonia

train_healthy, train_pneumonia = sep_images_by_class(x_train, y_train)

```
Since we intend to add augmented images of only healthy CXRs, we must first separate the images by class as I've done above.


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

Great! Now we have a balanced training set consisting of augmented and authentic CXR images.</br>
Before training the model, we can optimize it's performance using cache and prefetch.

### Configure the dataset for performance

```
# prefetch and cache for optimization
def performance_optimizer(x,y,batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((x,y))
    dataset = dataset.batch(batch_size = batch_size).prefetch(buffer_size = tf.data.AUTOTUNE)
    dataset = dataset.cache()
    return dataset

training_tf = performance_optimizer(combined_x_train, combined_y_train, 32)
validation_tf = performance_optimizer(x_validation, y_validation, 32)
test_tf = performance_optimizer(x_test, y_test, 32)

```
Caching and prefetching are important techniques in deep learning that can improve the training time and overall performance of a model. Caching involves storing the data in memory to avoid the overhead of reading from disk repeatedly, while prefetching loads data in advance to minimize the waiting time during training. By using these techniques, the model can access the data more quickly and efficiently, which can lead to faster and more accurate training.

## Transfer learning

In this project, we will use transfer learning to classify our CXRs. Transfer learning is a technique in machine learning where a pre-trained model is used as a starting point for a new task, rather than training a new model from scratch. The pre-trained model has already learned to recognize general features and patterns from a large dataset, and this knowledge can be transferred to the new task with some fine-tuning. Transfer learning can save time and resources, as well as improve the performance of the new model.</br>

### Inception-v3
For this project, we will use the inception-v3 model (Figure 1). The Inception-v3 model is a deep neural network architecture used for image classification tasks. It is an improved version of the original Inception model and was developed by Google researchers in 2015. The original inception model was designed to improve the efficiency of image classification tasks by reducing the number of parameters required in the network. The model achieves this by using a combination of convolutional layers with different kernel sizes, which allows it to capture features at different scales. The architecture also includes a module called "Inception module," which uses multiple filters of different sizes in parallel to capture features at different levels of abstraction (Szegedy et al., 2015). Inception-v3 has made several improvements to the original inception model. These changes include the use of depthwise-separable convolutions to improve efficiency and minimize computations, batch normalization, better regularization techniques, and changes in it's auxillary classifiers to improve training and reduce overfitting (Szegedy et al., 2016).

![inceptionv3onc--oview](https://user-images.githubusercontent.com/71438259/235567839-f8ead8d4-c35e-4547-be64-f87396bc06af.png)
*Fig.1. A High level diagram of the inception-v3 model*

```
#initial training
base_model = tf.keras.applications.inception_v3.InceptionV3(
    include_top=False,
    weights='imagenet',
    input_shape=(299, 299, 3))

base_model.trainable = False
```
First, we import the Inception-v3 model from keras and exclude the fully connected layer of the pre-trained model as this will be replaced by our own dense layer. We also specify that the pre-trained weights of the Inception-v3 model, trained on the ImageNet dataset, should be used as initial values for the model parameters. The input_shape is specified as (299, 299, 3). The first two numbers (299, 299) refer to the number of pixels on the height and width of the image, respectively. The third dimension, '3', corresponds to the three primary colors (red, green, and blue) that make up each pixel of the image. It is recommended to use an input size of (299, 299, 3) because the model was pre-trained on input images preprocessed to that resolution. This allows the pre-trained weights of the model to be applied correctly to new input data, which can result in better performance compared to using a different input size or format.

```
preprocess_input = tf.keras.applications.inception_v3.preprocess_input
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(1)
inputs = tf.keras.Input(shape=(299, 299, 3))

x = preprocess_input(inputs)
x = base_model(x, training = False) 
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.5)(x) # add dropout layer
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)
```



## References
1. https://www.who.int/news-room/fact-sheets/detail/pneumonia</br>
2. Frija, G., Blažić, I., Frush, D. P., Hierath, M., Kawooya, M., Donoso-Bach, L., & Brkljačić, B. (2021). 
3. How to improve access to medical imaging in low-and middle-income countries?. EClinicalMedicine, 38, 101034. </br>
Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), “Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification”, Mendeley Data, V2, doi: 10.17632/rscbjbr9sj.2
4. https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
5. Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the inception architecture for computer vision. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2818-2826).
6. Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., ... & Rabinovich, A. (2015). Going deeper with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1-9).
