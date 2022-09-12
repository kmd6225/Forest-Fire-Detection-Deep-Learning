#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from bing_image_downloader import downloader


# In[4]:


def get_images(query):
    """Download images and place them in a directory.
    
    :param query: Query to search for
    :return: Images matching query
    
    """
    print(query)
    
    downloader.download(query, 
                    limit=80, 
                    output_dir='desktop/wildfire computer vision', 
                    adult_filter_off=False, 
                    force_replace=False, 
                    timeout=60)


# In[3]:


satellite = ['Forest Fire Satellite View', 'Forest Satellite View']


# In[5]:


for s in satellite:
    print('Fetching images of', s)
    get_images(s)

