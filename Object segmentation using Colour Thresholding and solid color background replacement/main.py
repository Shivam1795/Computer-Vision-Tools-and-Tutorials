"""Object segmentation using Colour Thresholding and 
    solid color background replacement"""

"""  https://github.com/Shivam1795  """

## Import all the required libraries !!
import cv2
import numpy as np
import matplotlib.pyplot as plt


## Read image !!
image = cv2.imread('images/spaceship_blueScreen.jpg')

## Display the data type and dimensions of the image array !!
print('Data type and Dimensions of the image:', type(image), image.shape)

## convert a copy of the original BGR image to RGB  !!
img_copy = cv2.cvtColor(np.copy(image), cv2.COLOR_BGR2RGB)


## Define boundaries to select blue background in RGB image (lower and upper limits) !!
## Just play with these values to find correct upper and lower threshold values !!
lower_lim = np.array([0, 50, 100])
upper_lim = np.array([80, 100, 255])

## Uncomment these lines for hp.jpg image !!
#lower_lim = np.array([0, 0, 85])
#upper_lim = np.array([100, 100, 255])

## Create a mask !!
mask = cv2.inRange(img_copy, lower_lim, upper_lim)

## Generate a masked image !!
masked_img = np.copy(img_copy)

## In a copy of the original image replace the pixel values with vector [0,0,0] at those locations, wherein the corresponding masked image pixels are non-zero !!
masked_img[mask != 0] = [0, 0, 0]

## Read background image and convert to RGB from BGR !!
background = cv2.cvtColor(cv2.imread('images/space.jpg'), cv2.COLOR_BGR2RGB)

## Check the shape of the background !!
print('Background image original shape :',background.shape)

## Resize background image to our original image size !!
resized_background = cv2.resize(background, (image.shape[1], image.shape[0]), interpolation = cv2.INTER_AREA)

## Check the shape of the background !!
print('Background image new shape :',resized_background.shape)

## Crop background to fit our masked image !!
resized_background[mask == 0] = [0, 0, 0]

## Add two images to generate the final image !!
final_img = resized_background + masked_img

## Display final image !!
cv2.imshow('Final_image',cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)


## Thanks !!

