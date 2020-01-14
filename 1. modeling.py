#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras import optimizers
from keras.callbacks import ModelCheckpoint 
from keras.callbacks import EarlyStopping


# ### augmentation

# In[2]:


train_img_gen = ImageDataGenerator(rescale=1./255,
                                  rotation_range=90,
                                  width_shift_range=0.1,
                                  height_shift_range=0.1,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True,
                                  vertical_flip=True)

test_img_gen = ImageDataGenerator(rescale=1./255)


# ### 이진분류 1 : bad+dried vs good

# In[3]:


train_generator = train_img_gen.flow_from_directory(
        'images/reclassification/train_1/',
        target_size = (224, 224),
        batch_size = 32,
        class_mode = 'binary')

test_generator = test_img_gen.flow_from_directory(
        'images/reclassification/test_1/',
        target_size = (224, 224),
        batch_size = 32,
        class_mode = 'binary')


# In[4]:


train_generator.class_indices, test_generator.class_indices3


# In[9]:


model = Sequential()

""" ========== VGG16모델 가중치 사용하기 ========== """
model.add(Conv2D(filters=64,kernel_size = (3,3),activation = 'relu',padding = 'same',trainable = False,input_shape = (224, 224, 3)))
model.add(Conv2D(filters=64,kernel_size = (3,3),activation = 'relu',padding = 'same',trainable = False))
model.add(MaxPooling2D(pool_size = (2,2),strides = (2,2)))

model.add(Conv2D(filters=128,kernel_size = (3,3),activation = 'relu',padding = 'same',trainable = False))
model.add(Conv2D(filters=128,kernel_size = (3,3),activation = 'relu',padding = 'same',trainable = False))
model.add(MaxPooling2D(pool_size = (2,2),strides = (2,2)))

model.add(Conv2D(filters=256,kernel_size = (3,3),activation = 'relu',padding = 'same',trainable = False))
model.add(Conv2D(filters=256,kernel_size = (3,3),activation = 'relu',padding = 'same',trainable = False))
model.add(Conv2D(filters=256,kernel_size = (3,3),activation = 'relu',padding = 'same',trainable = False))
model.add(MaxPooling2D(pool_size = (2,2),strides = (2,2)))

model.add(Conv2D(filters=512,kernel_size = (3,3),activation = 'relu',padding = 'same',trainable = False))
model.add(Conv2D(filters=512,kernel_size = (3,3),activation = 'relu',padding = 'same',trainable = False))
model.add(Conv2D(filters=512,kernel_size = (3,3),activation = 'relu',padding = 'same',trainable = False))
model.add(MaxPooling2D(pool_size = (2,2),strides = (2,2)))

model.add(Conv2D(filters=512,kernel_size = (3,3),activation = 'relu',padding = 'same',trainable = False))
model.add(Conv2D(filters=512,kernel_size = (3,3),activation = 'relu',padding = 'same',trainable = False))
model.add(Conv2D(filters=512,kernel_size = (3,3),activation = 'relu',padding = 'same',trainable = False))
model.add(MaxPooling2D(pool_size = (2,2),strides = (2,2)))

model.load_weights("vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")


""" ========== FC 레이어 ========== """
model.add(Flatten())

model.add(Dense(32, activation = 'relu', kernel_initializer = 'he_normal'))
model.add(Dropout(0.5))
model.add(Dense(32, activation = 'relu', kernel_initializer = 'he_normal'))
model.add(Dropout(0.5))
model.add(Dense(1, activation = 'sigmoid'))


""" ========== compile ========== """
model.compile(optimizer = optimizers.Adam(learning_rate=0.0001),
              loss = 'binary_crossentropy',
               metrics = ['accuracy'])


""" ========== stop/save option ========== """
early = EarlyStopping(monitor="val_loss", patience=10)
checkpoint = ModelCheckpoint(filepath="model/bad_dried_good_01_{epoch:04d}_{loss:.4f}_{accuracy:.4f}_{val_loss:.4f}_{val_accuracy:.4f}.hdf5", 
                             monitor="val_accuracy",
                             mode="max",
                             verbose=1,
                             save_best_only=True)

history1 = model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator.filenames)//32 * 10,
    epochs=100,
    callbacks=[checkpoint],
    validation_data = valid_generator)


# In[10]:


plt.figure(figsize=(10,8))
plt.plot(history1.history["accuracy"])
plt.plot(history1.history["val_accuracy"])
plt.plot(history1.history["loss"])
plt.plot(history1.history["val_loss"])
plt.legend(["acc","val_acc","loss","val_loss"], loc = "best")


# ### 이진분류 2 : bad vs dried

# In[3]:


train_generator = train_img_gen.flow_from_directory(
        'images/reclassification/train_2/',
        target_size = (224, 224),
        batch_size = 32,
        class_mode = 'binary')

test_generator = test_img_gen.flow_from_directory(
        'images/reclassification/test_2/',
        target_size = (224, 224),
        batch_size = 32,
        class_mode = 'binary')


# In[4]:


train_generator.class_indices, test_generator.class_indices3


# In[9]:


model = Sequential()

""" ========== VGG16모델 가중치 사용하기 ========== """
model.add(Conv2D(filters=64,kernel_size = (3,3),activation = 'relu',padding = 'same',trainable = False,input_shape = (224, 224, 3)))
model.add(Conv2D(filters=64,kernel_size = (3,3),activation = 'relu',padding = 'same',trainable = False))
model.add(MaxPooling2D(pool_size = (2,2),strides = (2,2)))

model.add(Conv2D(filters=128,kernel_size = (3,3),activation = 'relu',padding = 'same',trainable = False))
model.add(Conv2D(filters=128,kernel_size = (3,3),activation = 'relu',padding = 'same',trainable = False))
model.add(MaxPooling2D(pool_size = (2,2),strides = (2,2)))

model.add(Conv2D(filters=256,kernel_size = (3,3),activation = 'relu',padding = 'same',trainable = False))
model.add(Conv2D(filters=256,kernel_size = (3,3),activation = 'relu',padding = 'same',trainable = False))
model.add(Conv2D(filters=256,kernel_size = (3,3),activation = 'relu',padding = 'same',trainable = False))
model.add(MaxPooling2D(pool_size = (2,2),strides = (2,2)))

model.add(Conv2D(filters=512,kernel_size = (3,3),activation = 'relu',padding = 'same',trainable = False))
model.add(Conv2D(filters=512,kernel_size = (3,3),activation = 'relu',padding = 'same',trainable = False))
model.add(Conv2D(filters=512,kernel_size = (3,3),activation = 'relu',padding = 'same',trainable = False))
model.add(MaxPooling2D(pool_size = (2,2),strides = (2,2)))

model.add(Conv2D(filters=512,kernel_size = (3,3),activation = 'relu',padding = 'same',trainable = False))
model.add(Conv2D(filters=512,kernel_size = (3,3),activation = 'relu',padding = 'same',trainable = False))
model.add(Conv2D(filters=512,kernel_size = (3,3),activation = 'relu',padding = 'same',trainable = False))
model.add(MaxPooling2D(pool_size = (2,2),strides = (2,2)))

model.load_weights("vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")


""" ========== FC 레이어 ========== """
model.add(Flatten())

model.add(Dense(32, activation = 'relu', kernel_initializer = 'he_normal'))
model.add(Dropout(0.5))
model.add(Dense(32, activation = 'relu', kernel_initializer = 'he_normal'))
model.add(Dropout(0.5))
model.add(Dense(1, activation = 'sigmoid'))


""" ========== compile ========== """
model.compile(optimizer = optimizers.Adam(learning_rate=0.0001),
              loss = 'binary_crossentropy',
               metrics = ['accuracy'])


""" ========== stop/save option ========== """
early = EarlyStopping(monitor="val_loss", patience=10)
checkpoint = ModelCheckpoint(filepath="model/bad_dried_01_{epoch:04d}_{loss:.4f}_{accuracy:.4f}_{val_loss:.4f}_{val_accuracy:.4f}.hdf5", 
                             monitor="val_accuracy",
                             mode="max",
                             verbose=1,
                             save_best_only=True)

history1 = model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator.filenames)//32 * 10,
    epochs=100,
    callbacks=[checkpoint],
    validation_data = valid_generator)


# In[10]:


plt.figure(figsize=(10,8))
plt.plot(history1.history["accuracy"])
plt.plot(history1.history["val_accuracy"])
plt.plot(history1.history["loss"])
plt.plot(history1.history["val_loss"])
plt.legend(["acc","val_acc","loss","val_loss"], loc = "best")

