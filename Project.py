import numpy as np
import skimage
from skimage.io import imread, imshow 
import matplotlib.pyplot as plt
from scipy.signal import gaussian
from numpy.fft import fft2, ifft2
import cv2
from sklearn.decomposition import PCA
import matplotlib.image as img
from skimage import segmentation
from matplotlib import pyplot as plt
import os
from scipy.signal import gaussian, convolve2d
from PIL import Image


def wiener_filter(img, kernel, K):
	kernel /= np.sum(kernel)
	dummy = np.copy(img)
	dummy = fft2(dummy)
	kernel = fft2(kernel, s = img.shape)
	kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
	dummy = dummy * kernel
	dummy = np.abs(ifft2(dummy))
	return dummy

def gaussian_kernel(kernel_size = 3):
	h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
	h = np.dot(h, h.transpose())
	h /= np.sum(h)
	return h



img_rgb = cv2.imread('Y3.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)


# Apply Wiener Filter
kernel = gaussian_kernel(5)
filtered_img = wiener_filter(img_gray, kernel, K = 20)

filtered_img2 = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)


#//////////////////////////////////////////////////////////////////////////////////////////


blue,green,red = cv2.split(filtered_img2) 
# #it will split the original image into Blue, Green and Red arrays.

# #initialize PCA with first 20 principal components
pca = PCA(130)
 
# #Applying to red channel and then applying inverse transform to transformed array.
red_transformed = pca.fit_transform(red)
red_inverted = pca.inverse_transform(red_transformed)
 
# #Applying to Green channel and then applying inverse transform to transformed array.
green_transformed = pca.fit_transform(green)
green_inverted = pca.inverse_transform(green_transformed)
 
# #Applying to Blue channel and then applying inverse transform to transformed array.
blue_transformed = pca.fit_transform(blue)
blue_inverted = pca.inverse_transform(blue_transformed)

img_compressed = (np.dstack((red_inverted, red_inverted, red_inverted))).astype(np.uint8)


#//////////////////////////////////////////////////////////////////////////////////////////


def mean_image(image,label):
    img_reshape = image.reshape((image.shape[0]*image.shape[1],image.shape[2]))
    sli_1d = np.reshape(label,-1)    
    uni = np.unique(sli_1d)
    uu = np.zeros(img_reshape.shape)
    for i in uni:
        loc = np.where(sli_1d == i)[0]
        #print(loc)
        mm = np.mean(img_reshape[loc,:],axis = 0)
        uu[loc,:]=mm
    Superpixeled_Image = np.reshape(uu,[image.shape[0],image.shape[1],image.shape[2]]).astype('uint8')
    print(Superpixeled_Image.shape)
    # plt.imshow(Superpixeled_Image)
    # plt.title('Superpixeled_Image')
    return Superpixeled_Image
    
       
converted_image = cv2.cvtColor(img_compressed, cv2.COLOR_BGR2RGB)
label=segmentation.slic(img_compressed,compactness=20, n_segments=800)

Segmented_Img= mean_image(img_compressed,label) # displays the output image.



####### Display results ###########
display = [img_rgb, filtered_img, img_compressed, Segmented_Img]
label = ['Original Image', 'Wiener Filter applied', 'Compressed Image', 'Segmented Image']



fig = plt.figure(figsize=(12, 10))
for i in range(len(display)):
 	fig.add_subplot(2, 2, i+1)
 	plt.imshow(display[i], cmap = 'gray')
 	plt.title(label[i])

plt.show()


#UPDATED
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import gaussian
from numpy.fft import fft2, ifft2
import cv2
from sklearn.decomposition import PCA
from skimage import segmentation
from imutils import paths


def wiener_filter(img, kernel, K):
	kernel /= np.sum(kernel)
	dummy = np.copy(img)
	dummy = fft2(dummy)
	kernel = fft2(kernel, s = img.shape)
	kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
	dummy = dummy * kernel
	dummy = np.abs(ifft2(dummy))
	return dummy

def gaussian_kernel(kernel_size = 3):
	h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
	h = np.dot(h, h.transpose())
	h /= np.sum(h)
	return h

#Load image dataset
imagepaths = list(paths.list_images("D:\KPZZNJR\B.Tec\s8\Project\image"))
for imagepath in imagepaths:
    img_rgb = cv2.imread(imagepath)
    
for i in img_rgb:
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)


# Apply Wiener Filter
kernel = gaussian_kernel(5)
for i in range(10):
    filtered_img = wiener_filter(img_gray, kernel, K = 20)
    
for i in range(10):
    filtered_img2 = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)


#//////////////////////////////////////////////////////////////////////////////////////////

for i in range(10):
    blue,green,red = cv2.split(filtered_img2) 
# #it will split the original image into Blue, Green and Red arrays.

# #initialize PCA with first 20 principal components
pca = PCA(130)
 
# #Applying to red channel and then applying inverse transform to transformed array.
for i in range(10):
    red_transformed = pca.fit_transform(red)
    red_inverted = pca.inverse_transform(red_transformed)
 
# #Applying to Green channel and then applying inverse transform to transformed array.
for i in range(10):
    green_transformed = pca.fit_transform(green)
    green_inverted = pca.inverse_transform(green_transformed)
 
# #Applying to Blue channel and then applying inverse transform to transformed array.
for i in range(10):
    blue_transformed = pca.fit_transform(blue)
    blue_inverted = pca.inverse_transform(blue_transformed)
    
for i in range(10):
    img_compressed = (np.dstack((red_inverted, red_inverted, red_inverted))).astype(np.uint8)


#//////////////////////////////////////////////////////////////////////////////////////////


def mean_image(image,label):
    img_reshape = image.reshape((image.shape[0]*image.shape[1],image.shape[2]))
    sli_1d = np.reshape(label,-1)    
    uni = np.unique(sli_1d)
    uu = np.zeros(img_reshape.shape)
    for i in uni:
        loc = np.where(sli_1d == i)[0]
        #print(loc)
        mm = np.mean(img_reshape[loc,:],axis = 0)
        uu[loc,:]=mm
    Superpixeled_Image = np.reshape(uu,[image.shape[0],image.shape[1],image.shape[2]]).astype('uint8')
    print(Superpixeled_Image.shape)
    return Superpixeled_Image
    
for i in range(10):       
    converted_image = cv2.cvtColor(img_compressed, cv2.COLOR_BGR2RGB)
    label=segmentation.slic(img_compressed,compactness=20, n_segments=800)
    Segmented_Img= mean_image(img_compressed,label) # displays the output image.



####### Display results ###########

fig = plt.figure(figsize=(15, 10))
for i in range(10):
 	fig.add_subplot(2, 5, i+1)
 	plt.imshow(img_rgb, cmap = 'gray')
 	# plt.title(label[i])
     
fig = plt.figure(figsize=(15, 10))    
for i in range(10):
 	fig.add_subplot(2, 5, i+1)
 	plt.imshow(filtered_img, cmap = 'gray')

fig = plt.figure(figsize=(15, 10))  
for i in range(10):
 	fig.add_subplot(2, 5, i+1)
 	plt.imshow(img_compressed, cmap = 'gray')

fig = plt.figure(figsize=(15, 10))  
for i in range(10):
 	fig.add_subplot(2, 5, i+1)
 	plt.imshow(Segmented_Img, cmap = 'gray')     

plt.show()

