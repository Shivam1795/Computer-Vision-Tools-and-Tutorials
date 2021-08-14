"""
@ author ' https://github.com/Shivam1795 '

"""

## Import all the libraries  !!
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

## To ignore Numpy warnings  !! 
np.seterr(divide = 'ignore') 

## To reset all the sliders  !!
def resetSliders(event):
    lpf.reset()
    hpf.reset()
    Xtrans.reset()
    Ytrans.reset()

## To update sift in X coordinate of filter mask, using Xshift slider value  !!
def x_shift_update(val):
    global Xshift
    Xshift = val
    UPDATE()

## To update sift in Y coordinate of filter mask, using Yshift slider value  !!
def y_shift_update(val):
    global Yshift
    Yshift = val
    UPDATE()

## To update LPF threshold using lpfSlider values  !!
def lpfThreshold_update(val):
    global lpf_threshold
    lpf_threshold = val
    UPDATE()

## To update HPF threshold using hpfSlider values  !!
def hpfThreshold_update(val):
    global hpf_threshold
    hpf_threshold = val
    UPDATE()

## To display updated images  !!
def UPDATE():
    filtered_img, filtered_spectrum = applyFilter()
    plt2.set_array(filtered_spectrum)
    plt3.set_array(filtered_img)

## To apply filters on an image and reconstruct filtered image  !!
def applyFilter():

    ## Compute DFT of image  !!
    dft = cv2.dft(np.float32(gray_image), flags=cv2.DFT_COMPLEX_OUTPUT)

    ## Shift center of frequency spectrum from top-left corner to the center of the image  !!
    dft_shift = np.fft.fftshift(dft)
    
    ## Fix thickness of circle to -1 to create a disk  !!
    thickness = -1
    ## Define center coordinates for the circular masks  !!
    center = (int(Xshift + dft_shift.shape[1]/2), int(Yshift + dft_shift.shape[0]/2))

    ## Use lpf_threshold as the radius for LPF circular mask  !!
    lpf_radius = int(lpf_threshold)
    ## Inside LPF circular mask, keep pixel intensity value 1 to preserve low frequency components  !!
    lpf_mask_color = (1, 1)
    ## Outside LPF circular mask, keep pixel intensity value 0  to discart high frequency components  !!
    lpf_mask_img = np.zeros_like(dft_shift)

    ## Generate LPF mask  !!
    lpf_mask = cv2.circle(lpf_mask_img, center, lpf_radius, lpf_mask_color, thickness)

    ## Use hpf_threshold as the radius for HPF circular mask !!
    hpf_radius = int(hpf_threshold)
    ## Inside HPF circular mask, keep pixel intensity value 0 to discart low frequency components  !!
    hpf_mask_color = (0, 0)
    ## Outside HPF circular mask, keep pixel intensity value 1 to preserve high frequency components  !!
    hpf_mask_img = np.ones_like(dft_shift)

    ## Generate HPF mask  !!
    hpf_mask = cv2.circle(hpf_mask_img, center, hpf_radius, hpf_mask_color, thickness)

    ## Apply filters on frequency spectrum  !!
    filtered_dft_shift = dft_shift * lpf_mask * hpf_mask

    ## Convert resulting spectrum to log scale to display  !!
    filtered_spectrum = 20 * np.log(cv2.magnitude(filtered_dft_shift[:,:,0], filtered_dft_shift[:,:,1]))

    ## Reverse shifting of center of frequency spectrum from center of image to top-left corner  !!
    filtered_idft_shift = np.fft.ifftshift(filtered_dft_shift)

    ## Computing Inverse DFT  !!
    filtered_image_back = cv2.idft(filtered_idft_shift)

    ## Reconstructing filtered image  !!
    filtered_image = cv2.magnitude(filtered_image_back[:,:,0], filtered_image_back[:,:,1])

    ## Return reconstructed image and filtered frequency spectrum  !!
    return (filtered_image, filtered_spectrum)


## Read images  !!
image = cv2.imread('images/flower1.jpg')

## Convert to RGB image  !!
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

## Create a Grayscale copy of input image  !!
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

## Translation in X  !!
Xshift = 0

## Translation in Y  !!
Yshift = 0

## Passing all the frequency  !!
hpf_threshold = 0

## Passing all the frequency  !!
lpf_threshold = math.sqrt(gray_image.shape[0]**2 + gray_image.shape[1]**2)//2 + 1

## Call applyFilter function for initial display  !!
filtered_img, filtered_spectrum = applyFilter()

## Display images  !!
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
plt.subplots_adjust(left=0.05, bottom=0.45, right=0.95, top=0.95)

plt1 = ax1.imshow(gray_image, cmap='gray');ax1.axis(False); ax1.title.set_text('Original Grayscale image')
plt2 = ax2.imshow(filtered_spectrum, cmap='gray');ax2.axis(False); ax2.title.set_text('Application of filters on image frequency spectrum')
plt3 = ax3.imshow(filtered_img, cmap='gray');ax3.axis(False); ax3.title.set_text('Reconstructed Grayscale image')

## Initialize "LPF freq cutoff" slider  !!
lpfSlider = plt.axes([0.15, 0.1, 0.75, 0.03])
lpf = Slider(lpfSlider, 'LPF freq cutoff:', valmin = 0, valmax = int(math.sqrt(gray_image.shape[0]**2 + gray_image.shape[1]**2)/2), valinit = int(math.sqrt(gray_image.shape[0]**2 + gray_image.shape[1]**2)/2), valfmt='%1.2f', closedmax=True, color='red')
## Set callback function for "LPF freq cutoff" slider  !!
lpf.on_changed(lpfThreshold_update)

## Initialize "HPF freq cutoff" slider  !!
hpfSlider = plt.axes([0.15, 0.16, 0.75, 0.03])
hpf = Slider(hpfSlider, 'HPF freq cutoff:', valmin = 0, valmax = int(math.sqrt(gray_image.shape[0]**2 + gray_image.shape[1]**2)/2), valinit = 0, valfmt='%1.2f', slidermax= lpf, closedmax=True, color='green')
## Set callback function for "HPF freq cutoff" slider  !!
hpf.on_changed(hpfThreshold_update)

## Initialize "Y Shift" slider  !!
YSlider = plt.axes([0.15, 0.22, 0.75, 0.03])
Ytrans = Slider(YSlider, 'Y Shift:', valmin = -int(gray_image.shape[0]/2), valmax = int(gray_image.shape[0]/2), valinit = 0, valfmt='%1.2f', closedmax=True)
## Set callback function for "Y Shift" slider  !!
Ytrans.on_changed(y_shift_update)

## Initialize "X Shift" slider  !!
XSlider = plt.axes([0.15, 0.28, 0.75, 0.03])
Xtrans = Slider(XSlider, 'X Shift:', valmin = -int(gray_image.shape[1]/2), valmax = int(gray_image.shape[1]/2), valinit = 0, valfmt='%1.2f', closedmax=True, color='yellow')
## Set callback function for "X Shift" slider  !!
Xtrans.on_changed(x_shift_update)

## Initialize Reset button  !!
rsetButton = plt.axes([0.5, 0.01, 0.08, 0.07])
rset = Button(rsetButton, 'Reset')
## Set callback function for Reset button  !!
rset.on_clicked(resetSliders)

## Display all the plots  !!
plt.show()