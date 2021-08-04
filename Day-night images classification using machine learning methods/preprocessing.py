# coding: utf-8


"""  https://github.com/Shivam1795  """


## Import all the required libraries !!
import os
import cv2
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mplimg



def dataloder(path):
    
    """

        Function loads in images and their labels and places them in a list.
        
        Input:
            
            path: Path to the train or test folder.
    
    """
    
    ## Initialize an empty list to store all the images and their associated labels  !!
    data = []
    
    ## Define classes !!
    image_classes = ['day', 'night']
    
    ## Iterate over each class folder !!
    for img_class in image_classes:
        
        ## Iterate over all the images to read them !!
        for img_path in glob.glob(os.path.join(path, img_class, '*')):
            
            ## Reading image !!
            image = mplimg.imread(img_path)
            
            ## Check if the image exists !!
            if image is not None:
                ## Store all the images and their associated labels inside a list in the form of tuples !!
                data.append((image, img_class))
                
            
    return data





def data_standardize(data_list, img_size):
    
    """

        To transform all the images in the standard input size and encode labels.
        
        Input:
            
            data_list: List of images along with labels.
            
            img_size: A tuple consists of standard dimensions for images (wxh)
    
    """
    
    ## Initialize an empty list to store all the standardized data !!
    standardize_data = []
    
    ## Iterate over all the data stored in the list to standardize it !!
    for data in data_list:
        
        ## Reading images and labels from input list !!
        image = data[0]
        label = data[1]
        
        ## Resizing the image !!
        standard_img = cv2.resize(image, img_size)
        
        ## Label encoding (day = 1 & night = 0) !!
        label_encoder = 0
        if label == 'day':
            label_encoder = 1
        
        ## Append resized images and encoded labels in a list !!
        standardize_data.append((standard_img, label_encoder))
        
    return standardize_data





def get_features(data, is_list=False):
    
    """

        For a list of images and labels:
            To calculate average brightness for all the images in the given list and return a list
            of average brightness and the corresponding label for each image.
        
        For a single image:
            The function returns average brightness.
        
        ** Using average brightness as a feature for day and night image classification.

        Input:
        
            data: List of images along with labels.
            
            is_list: True, if passing a list of images and labels in the first argument 
                     and False if passing a single image.
    
    """
    
    if is_list == True:
        
        ## Initialize an empty list to store average values and corresponding labels !!
        featuredData = []

        ## Iterate over all the items in the list !!
        for item in data:

            ## Load image and label !!
            image = item[0]
            label = item[1]

            ## Convert a RGB image to HSV image !!
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

            ## Calculate the sum of pixel values in the 'Value' channel of the hsv image !!
            brightness = np.sum(hsv[:,:,2])

            ## Compute average brightness !!
            avg_brightness = brightness/(image.shape[0]*image.shape[1])

            ## Append average brightness and corresponding label to a list !!
            featuredData.append((avg_brightness, label))

        return featuredData
    
    else:
        
        image = data
        
        ## Convert a RGB image to HSV image !!
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        ## Calculate the sum of pixel values in the 'Value' channel of the hsv image !!
        brightness = np.sum(hsv[:,:,2])

        ## Compute average brightness !!
        avg_brightness = brightness/(image.shape[0]*image.shape[1])
        
        return avg_brightness
        






def Data_Visualization(training_data, test_data):
    
    """

        The function displays two random images, one from the training dataset and
        another from the test dataset, along with their corresponding labels.

        Input:
        
            training_data: List of images and labels in training_data.

            test_data: List of images and labels in test_data.
        
    """
    
    ## Initialize subplots !!
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,20))
    
    ## Select a random index (integer) for training data !!
    random_index = random.randint(0, len(training_data)-1)

    ## Load random image and label from training data !!
    image = training_data[random_index][0]
    label = training_data[random_index][1]

    ##Display image !!
    ax1.imshow(image)
    ax1.set_title('Random image from training data (Label: {}, Dimension: {})'.format(label, image.shape))
    ax1.axis(False)

    ## Select a random index (integer) for test data !!
    random_index = random.randint(0, len(test_data)-1)

    ## Load random image and label from test data !!
    image = test_data[random_index][0]
    label = test_data[random_index][1]

    ## Display image !!
    ax2.imshow(image)
    ax2.axis(False)
    ax2.set_title('Random image from test data (Label: {}, Dimension: {})'.format(label, image.shape))