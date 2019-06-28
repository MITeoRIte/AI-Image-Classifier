#!/usr/bin/env python
# coding: utf-8

# # Exercise 3 - Putting it all together - Model Training
# 
# In this notebook we will tie everything together to and use what we have learnt to try our hand in the competition.
# 
# The steps are similar to Exercise 2:
# 1. Explore the competition dataset
# 2. Build a convnet from scratch that performs reasonably well
# 3. Evaluate training and validation accuracy
# 4. Score the model against test set and submit result
# 
# Let's get started!

# ## Download and Explore the Dataset

# Let's start by downloading our dataset, a .zip of 1,226 PNG pictures of different poses, and extracting it locally.

# The contents of the `.zip` are extracted to the base directory, which contains `train` and `val` subdirectories for you to do training and validation. The folders have the following structure:
# 
# ```
# ---------------
# train
# |- ChairPose
# |- ChildPose
# |- Dabbing
# |- HandGun
# |- HandShake
# |- HulkSmash
# |- KoreanHeart
# |- KungfuCrane
# |- KungfuSalute
# |- Salute
# |- WarriorPose
# 
# val
# |- ChairPose
# |- ChildPose
# |- Dabbing
# |- HandGun
# |- HandShake
# |- HulkSmash
# |- KoreanHeart
# |- KungfuCrane
# |- KungfuSalute
# |- Salute
# |- WarriorPose
# ```

# In[1]:


get_ipython().system('nvidia-smi')


# In[1]:


# Creating two directories - "data" and "data/trainset_11classes_0_00000" 
get_ipython().system('mkdir final_data2 && mkdir final_data2/trainset_11classes_0_00000')
# Unzip the data into the folder "data/trainset_11classes_0_00000"
get_ipython().system('unzip -qq -n data2.zip -d final_data2/trainset_11classes_0_00000')
get_ipython().system('unzip -qq -n trainset_4classes_2_20406.zip -d final_data2/trainset_11classes_0_00000')
# Switch directory to "data/trainset_11classes_0_00000" and show its content
get_ipython().system('cd final_data2/trainset_11classes_0_00000 && ls')


# In[3]:


import os

base_dir = 'final_data2/trainset_11classes_0_00000/data/trainset_11classes_0_000'

# Directory to our training data
train_folder = os.path.join(base_dir, 'train')

# Directory to our validation data
val_folder = os.path.join(base_dir, 'val')


# In[4]:


# List folders and number of files
print("Directory, Number of files")
for root, subdirs, files in os.walk(base_dir):
    print(root, len(files))


# In[5]:


import keras
from keras.preprocessing.image import ImageDataGenerator

# Batch size
bs = 32

# All images will be resized to this value
image_size = (224, 224)

# All images will be rescaled by 1./255. We apply data augmentation here.
train_datagen = ImageDataGenerator(rescale=1./255,
                                   brightness_range= [0.5,1.5],
                                   horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 32 using train_datagen generator
print("Preparing generator for train dataset")
train_generator = train_datagen.flow_from_directory(
    directory= train_folder, # This is the source directory for training images 
    target_size=image_size, # All images will be resized to value set in image_size
    batch_size=bs,
    class_mode='categorical')

# Flow validation images in batches of 32 using val_datagen generator
print("Preparing generator for validation dataset")
val_generator = val_datagen.flow_from_directory(
    directory= val_folder, 
    target_size=image_size,
    batch_size=bs,
    class_mode='categorical')


# In[2]:


from keras.models import Model, Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Dense, Dropout, MaxPooling2D, Flatten

keras.layers.ZeroPadding2D(padding=(1, 1), data_format=None)

# Here we specify the input shape of our data 
# This should match the size of images ('image_size') along with the number of channels (3)
input_shape = (224, 224, 3)

# Define the number of classes
num_classes = 15

# Defining a baseline model. Here we use the [keras functional api](https://keras.io/getting-started/functional-api-guide) to build the model. 
# TODO: explore different architectures and training schemes
base_model =VGG16(include_top=False, weights='imagenet', input_shape=(224,224,3))
for layer in base_model.layers:
    layer.trainable = False
model= Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(1024,activation='relu'))
model.add(Dropout(1.5))
model.add(Dense(1024,activation='relu'))
model.add(Dropout(1.5))
model.add(Dense(15,activation='softmax'))


# In[14]:


model.summary()


# In[15]:


from keras import optimizers

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(),
              metrics=['accuracy'])


# In[16]:


from keras.callbacks import ModelCheckpoint

bestValidationCheckpointer = ModelCheckpoint('train_model.hdf5', monitor='val_acc', save_best_only=True, verbose=1)


# In[18]:


history = model.fit_generator(
        train_generator, # train generator has 973 train images
        steps_per_epoch=train_generator.samples // bs + 1,
        epochs=10,
        validation_data=val_generator, # validation generator has 253 validation images
        validation_steps=val_generator.samples // bs + 1,
        callbacks=[bestValidationCheckpointer]
)


# In[11]:


from keras.models import load_model

model_path = 'saved_model.hdf5'
model = load_model( model_path )


# Then, we validate accuracy of the loaded model on our good old validation set.

# In[12]:


val_generator.reset()

scores = model.evaluate_generator(val_generator, steps=val_generator.samples // val_generator.batch_size + 1, verbose=1)
print('Val loss:', scores[0])
print('Val accuracy:', scores[1])


# In[ ]:




