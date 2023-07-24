# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 13:58:23 2023

@author: isaac
"""
#importing packages
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import tensorflow as tf
import pickle


#Setting directories
DIR = r"C:\Users\isaac\OneDrive\Documents\Projects\datasets\CXR"
os.listdir(DIR)

PNEUMONIA_DIR = os.path.join(DIR, 'PNEUMONIA')
HEALTHY_DIR = os.path.join(DIR, 'HEALTHY')

##Checking number of samples
print(f'number of pneumonia xray images:\
\n{len(os.listdir(PNEUMONIA_DIR))}')
print(f'number of healthy xray images:\
\n{len(os.listdir(HEALTHY_DIR))}')

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

combined_x_train.shape
combined_y_train.shape

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

# prefetch and cache for optimization
def performance_optimizer(x,y,batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((x,y))
    dataset = dataset.batch(batch_size = batch_size).prefetch(buffer_size = tf.data.AUTOTUNE)
    dataset = dataset.cache()
    return dataset

training_tf = performance_optimizer(combined_x_train, combined_y_train, 32)
validation_tf = performance_optimizer(x_validation, y_validation, 32)
test_tf = performance_optimizer(x_test, y_test, 32)


#initial training
base_model = tf.keras.applications.inception_v3.InceptionV3(
    include_top=False,
    weights='imagenet',
    input_shape=(299, 299, 3))

base_model.trainable = False

base_model.summary() # examine model architecture

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

#define custom metric: f1_score
import tensorflow.keras.backend as K

class f1_score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = K.round(K.clip(y_pred, 0, 1))
        y_true = tf.cast(y_true, tf.float32) #float to compute F1 score (continuous, 0 to 1)
        tp = K.sum(y_true * y_pred)
        fp = K.sum(K.clip(y_pred - y_true, 0, 1))
        fn = K.sum(K.clip(y_true - y_pred, 0, 1))
        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        precision = self.tp / (self.tp + self.fp + K.epsilon())
        recall = self.tp / (self.tp + self.fn + K.epsilon())
        f1_score = 2 * precision * recall / (precision + recall + K.epsilon())
        return f1_score

    def reset_state(self):
        self.tp.assign(0)
        self.fp.assign(0)
        self.fn.assign(0)

#compile model
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
model.compile(optimizer=optimizer,
                        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                        metrics=['AUC', f1_score(), 'accuracy'])

model.summary() #examine model architecture

history = model.fit(training_tf,
          validation_data=validation_tf,
          epochs = 15)


'''
#save and load model
model.save_weights(os.path.join(save_DIR, 'initial_weights'))
model.load_weights(os.path.join(save_DIR, 'initial_weights'))
model.evaluate(validation_tf)

with open(os.path.join(save_DIR, 'initial_history'), 'wb') as f: #save
    pickle.dump(history.history, f)
with open(os.path.join(save_DIR, 'initial_history'), 'rb') as f: #load
    history = pickle.load(f)

'''

#plot initial learning curves
plt.figure(figsize = (10, 16))
plt.style.use('ggplot')

plt.subplot(4, 1, 1)
plt.plot(history['loss'], label = 'Training Loss')
plt.plot(history['val_loss'], label = 'Validation Loss')
plt.ylabel('Cross Entropy')
plt.xlabel('Epoch')
plt.legend(loc = 'upper right')
plt.title('Learning Curves', pad = 20, fontsize = 15, fontweight = 'bold')

plt.subplot(4,1,2)
plt.plot(history['auc'], label = 'AUC')
plt.plot(history['val_auc'], label = 'Validation AUC')
plt.ylabel('AUC')
plt.xlabel('Epoch')
plt.legend(loc = 'lower right')

plt.subplot(4,1,3)
plt.plot(history['f1_score'], label = 'F1 Score')
plt.plot(history['val_f1_score'], label = 'Validation F1 Score')
plt.ylabel('F1 Score')
plt.xlabel('Epoch')
plt.legend(loc = 'lower right')

plt.subplot(4,1,4)
plt.plot(history['accuracy'], label = 'Accuracy')
plt.plot(history['val_accuracy'], label = 'Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc = 'lower right')

#fine tuning
base_model.trainable = True
print(f'number of layers in the base model: {len(base_model.layers)}')

# Freeze bottom layersHaving a learning rate that is too high during fine-tuning can potentially destroy the pre-trained weights, particularly if the new dataset is small and dissimilar from the original dataset used for pre-training.
for layer in base_model.layers[:250]:
    layer.trainable = False

#compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5) #lower lr
model.compile(optimizer=optimizer,
                        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                        metrics=['AUC', f1_score(), 'accuracy'])
model.summary()

len(model.trainable_variables)

history_fine = model.fit(training_tf,
                         validation_data=validation_tf,
                         initial_epoch = 15,
                         epochs= 20)

'''
#save
model.save_weights('weights_fine')
model.load_weights('weights_fine')
model.evaluate(validation_tf)
'''

#update metrics
history_fine = history_fine.history 

history['loss'] += history_fine['loss']
history['auc'] += history_fine['auc']
history['f1_score'] += history_fine['f1_score']
history['accuracy'] += history_fine['accuracy']

history['val_loss'] += history_fine['val_loss']
history['val_auc'] += history_fine['val_auc']
history['val_f1_score'] += history_fine['val_f1_score']
history['val_accuracy'] += history_fine['val_accuracy']

'''
#save history
with open('history_fine.pkl', 'wb') as f:
    pickle.dump(history, f)

with open('history_fine.pkl', 'rb') as f:
    history = pickle.load(f)
'''

#plot learning curves after fine tuning
plt.figure(figsize = (10, 16))
plt.style.use('ggplot')

plt.subplot(4, 1, 1)
plt.plot(history['loss'], label = 'Training Loss')
plt.plot(history['val_loss'], label = 'Validation Loss')
plt.ylabel('Cross Entropy')
plt.xlabel('Epoch')
plt.axvline(14, label = 'start fine tuning', color = 'm')
plt.legend(loc = 'upper right')
plt.title('Learning Curves', pad = 20, fontsize = 15, fontweight = 'bold')

plt.subplot(4,1,2)
plt.plot(history['auc'], label = 'AUC')
plt.plot(history['val_auc'], label = 'Validation AUC')
plt.ylabel('AUC')
plt.xlabel('Epoch')
plt.axvline(14, label = 'start fine tuning', color = 'm')
plt.legend(loc = 'lower right')

plt.subplot(4,1,3)
plt.plot(history['f1_score'], label = 'F1 Score')
plt.plot(history['val_f1_score'], label = 'Validation F1 Score')
plt.ylabel('F1 Score')
plt.xlabel('Epoch')
plt.axvline(14, label = 'start fine tuning', color = 'm')
plt.legend(loc = 'lower right')

plt.subplot(4,1,4)
plt.plot(history['accuracy'], label = 'Accuracy')
plt.plot(history['val_accuracy'], label = 'Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.axvline(14, label = 'start fine tuning', color = 'm')
plt.legend(loc = 'lower right')


#evaluation
loss, auc, f1_score, accuracy = model.evaluate(test_tf)

print(f'loss = {loss}\n\
auc = {auc}\n\
f1_score = {f1_score}\n\
accuracy = {accuracy}')

#confusion matrix
predictions = model.predict(test_tf)
predictions = tf.nn.sigmoid(predictions)
predictions = tf.where(predictions <0.5, 0, 1)


from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize = (8, 6))
sns.heatmap(cm, annot = True, linewidths = 2, 
            linecolor = 'black',
            xticklabels = ['healthy', 'pneumonia'], 
            yticklabels = ['healthy', 'pneumonia'],
            fmt = 'g',
            cmap = 'Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted label')
plt.ylabel('True label')


#plot images and predicted labels
categories = ['healthy', 'pneumonia']
plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(x_test[i+10].astype("uint8"))
  plt.title(categories[int(predictions[i+10])])
  plt.axis("off")

