#load packages 

import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image


from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential


#load neural network model for fish identification

smokey = tf.keras.models.load_model("smokey_V1")

#function for converting image to an array 

def loadImage (picture, outType ) :
    img = Image.open(picture)
    new_img = img.resize((20,20))
    new_img.load()
    data = np.asarray(new_img, dtype="int32" )
    if outType == "anArray":
        return (data)
    if outType == "aList":
        return list(data)


#Intro headers and text    
    

#File uploader 

sat_image = st.file_uploader(label = 'Upload a satellite picture in .jpg format of a forest to test the model', type = 'jpg')

#call the load image function to resize the uploaded picture and convert it to a 1 dimensional array and then a list

img_list = loadImage(fish_image, 'anArray')
img_array = img_list.astype('float32')/255
img_array = np.expand_dims(img_array,0)

#display image for user

st.image(Image.open(sat_image))
st.write('')

#convert list containing image information to a dataframe

#fish_df = pd.DataFrame(img_list)

#have fishnet make a prediction 
prediction = smokey.predict(img_array)


#Make predictions discrete

if prediction >= 0.5:
    prediction2 = 'a wildfire'
    
else: 
    prediction2 = 'no wildfire'


#Output prediction to the screen

st.write('Smokey believes that there is {}!'.format(prediction2))
st.write('')

st.header('How the Model Works')
st.write('')
st.write('When submitting a picture to Fishnet, the image is converted to a row of numbers with 1,200 columns. Each cell represents the rgb value \n of the image. Fishnet uses these values to classify the fish as either a trout or a largemouth bass.')
st.write('Fishnet is a feed-forward neural network with 1,200 neurons in the initial layer with a single hidden layer consisting of 800 neurons. The input \n and hidden layers are connected by a relu activation function. The hidden layer is connected to the output layer by the sigmoid activation function.')
st.write('') 
st.header('Information on Data Collection and Model Training')
st.write('A python notebook in juypter was used to automatically search bing for pictures of trout and largemouth bass. The python notebook \n downloaded about 200 images of trout and 200 of bass. These images were used to train fishnet')
st.write('The images were split into training and validation sets using an 80-20 split. Fishnet had 30 epochs with a batch size of 80 observations. \n Fishnet was 74% accurate on the validation set.')
st.write('')
st.header('Confusion Matrix')
st.image('fishConfusionMatrix.png')
st.write('The bottom right corner represents observations in the validation set that fishnet correctly classified a fish as a largemouth bass. \n The top left corner represents observations where fish net correctly classified a fish as a trout.\n The bottom left and top right are missclassfied observations in the validation set.')