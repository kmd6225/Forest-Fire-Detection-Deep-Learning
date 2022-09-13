# Forest Fire Detection "Smokey Web App"

## Introduction

After scraping the images from the web and training the convolutional neural network using python, I deployed the model in a streamlit app hosted on my local machine. Below are some screenshots of the app in action. 

## Detecting Fires

![](https://github.com/kmd6225/Forest-Fire-Detection-Deep-Learning/blob/main/demo1.png?raw=true)

The model was able to determine that a fire was occurring based off of this satellite image. 
![](https://github.com/kmd6225/Forest-Fire-Detection-Deep-Learning/blob/main/demo2.png?raw=true)

The model was also able to identify this fire near Boulder Colorado, which is a bit more obscured than the image above. 
![](https://github.com/kmd6225/Forest-Fire-Detection-Deep-Learning/blob/main/demo7.png?raw=true)

The model identified this wildfire near Salamanca Mexico, assigning a wildfire probability of 96%.
![](https://github.com/kmd6225/Forest-Fire-Detection-Deep-Learning/blob/main/demo3.png?raw=true)

The model identified that no wildfire is occurring here, only assigning a fire probability of 2%. 
![](https://github.com/kmd6225/Forest-Fire-Detection-Deep-Learning/blob/main/demo4.png?raw=true)

The model was not fooled by this trickier picture. Despite clouds resembling smoke, the model assigned a fire probability of 14% and correctly identified that no fire is occurring.  
![](https://github.com/kmd6225/Forest-Fire-Detection-Deep-Learning/blob/main/demo5.png?raw=true)

The interface of the app also includes information about data collection and model training. 
