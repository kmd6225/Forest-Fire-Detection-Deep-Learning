#load packages 

import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import image
import numpy as np
import tensorflow as tf
from PIL import Image


from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential


#load neural network model for fish identification

smokey = tf.keras.models.load_model("smokey")

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

    
st.title('Smokey Forest Fire Detection Powered By Deep Learning')
st.write('')

st.image(image.imread('Uncle_Sam_style_Smokey_Bear_Only_You.jpg'))
st.write('')

st.write('This app takes satellite images as input and predicts whether or not a forest fire is occuring.')
st.write('')
st.header('Upload a satellite image of a forest to determine if a wildfire is occuring. Note: the error on the screen will be removed once you upload an image.')
st.write('')

#File uploader 

sat_image = st.file_uploader(label = 'Upload a satellite picture in .jpg format of a forest to test the model', type = 'jpg')

#call the load image function to resize the uploaded picture and convert it to a 1 dimensional array and then a list

img_list = loadImage(sat_image, 'anArray')
img_array = np.array([img_list])
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

st.write('Smokey believes that there is {}. The probabilty of a wildfire is {}%!'.format(prediction2,round(prediction[0][0] * 100),2))


st.header('How the Model Works')
st.write('')
st.write('When submitting a picture to Smokey, the image is converted to a row of numbers with 1,200 columns. Each cell represents the rgb value \n of the image. Smokey uses these values to classify the forest as either being on fire or not being on fire.')
st.write('Smokey is a convolutional network')
st.write('') 
st.header('Information on Data Collection and Model Training')
st.write('A python notebook in juypter was used to automatically search bing for pictures of wildfires and forests. The python notebook \n downloaded about 80 images of wildfirese and 80 images of forests. These images were used to train Smokey.')
st.write('The images were split into training and validation sets using an 80-20 split.')
