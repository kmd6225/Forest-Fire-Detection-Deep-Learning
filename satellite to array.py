#!/usr/bin/env python
# coding: utf-8

# In[2]:


from PIL import Image
import numpy as np
import pandas as pd

#Function to convert image to array or list
def loadImage (inFileName, outType ) :
    img = Image.open( inFileName )
    new_img = img.resize((20,20))
    new_img.load()
    data = np.asarray(new_img, dtype="int32" )
    if outType == "anArray":
        return data
    if outType == "aList":
        return list(data)
    


# In[36]:


def loadImage2 (inFileName) :
    img = Image.open( inFileName )
    new_img = img.resize((20,20))
    return(new_img)


# In[3]:


import os
path_of_the_directory= 'wildfire computer vision/Forest Fire Satellite View'

fire_list = []

for filename in os.listdir(path_of_the_directory):
    f = os.path.join(path_of_the_directory,filename)
    if os.path.isfile(f):
        fire_list.append(loadImage(f, "anArray"))


# In[4]:


import os
path_of_the_directory= 'wildfire computer vision/Forest Satellite View'

forest_list = []

for filename in os.listdir(path_of_the_directory):
    f = os.path.join(path_of_the_directory,filename)
    if os.path.isfile(f):
        forest_list.append(loadImage(f, "anArray"))


# In[6]:


fire_train_list = fire_list[20:len(fire_list)]
fire_test_list = fire_list[1:20]

fire_train_label = np.repeat(1,54).tolist()
fire_test_label = np.repeat(1,20).tolist()


# In[7]:


forest_train_list = forest_list[20:len(forest_list)]
forest_test_list = forest_list[1:20]

forest_train_label = np.repeat(0,54).tolist()
forest_test_label = np.repeat(0,20).tolist()


# In[8]:


list_train = fire_train_list + forest_train_list

list_test = fire_test_list + forest_test_list

train_label = fire_train_label + forest_train_label
test_label = fire_test_label + forest_test_label


# In[9]:


X_train = np.array([list_train[0]])
y_train = [train_label[0]]
for i in range(1,len(list_train)):
    anArray = np.array([list_train[i]])
    if anArray.shape == (1,20,20,3):
        X_train = np.append(X_train,  anArray, axis = 0)
        y_train.append(train_label[i])
    print(i)


# In[10]:


X_test = np.array([list_test[0]])
y_test = [test_label[0]]

for i in range(1,len(list_test)):
    anArray = np.array([list_test[i]])
    if anArray.shape == (1,20,20,3):
        X_test = np.append(X_test,  anArray, axis = 0)
        y_test.append(test_label[i])
    print(i)


# In[11]:


from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout

import matplotlib.pyplot as plt


# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[18]:


y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
y_test = np.asarray(y_test).astype('float32').reshape((-1,1))


# In[ ]:


y_test


# In[ ]:


model=Sequential()

model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu",input_shape=(20,20,3)))

model.add(MaxPooling2D(pool_size=3))

model.add(Conv2D(filters=8,kernel_size=1,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=3))




model.add(Flatten())

model.add(Dense(700,activation="relu"))



model.add(Dense(1,activation="sigmoid"))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X_train, y_train, epochs = 100)


# In[19]:


model.evaluate(X_test, y_test)

