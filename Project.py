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


#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from sklearn.metrics import classification_report,confusion_matrix
import io
from PIL import Image
from warnings import filterwarnings
for dirname, _, filenames in os.walk('D:/brain_tumor2/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


colors_dark = ["#1F1F1F", "#313131", '#636363', '#AEAEAE', '#DADADA']
colors_red = ["#331313", "#582626", '#9E1717', '#D35151', '#E9B4B4']
colors_green = ['#01411C','#4B6F44','#4F7942','#74C365','#D0F0C0']

sns.palplot(colors_dark)
sns.palplot(colors_green)
sns.palplot(colors_red)

labels = ['glioma_tumor','no_tumor','meningioma_tumor','pituitary_tumor']
X_train = []
y_train = []
image_size = 150
for i in labels:
    folderPath = os.path.join('D:/brain_tumor2/input/brain-tumor-classification-mri/','Training',i)
    for j in tqdm(os.listdir(folderPath)):
        img = cv2.imread(os.path.join(folderPath,j))
        img = cv2.resize(img,(image_size, image_size))
        X_train.append(img)
        y_train.append(i)
        
for i in labels:
    folderPath = os.path.join('D:/brain_tumor2/input/brain-tumor-classification-mri/','Testing',i)
    for j in tqdm(os.listdir(folderPath)):
        img = cv2.imread(os.path.join(folderPath,j))
        img = cv2.resize(img,(image_size,image_size))
        X_train.append(img)
        y_train.append(i)
        
X_train = np.array(X_train)
y_train = np.array(y_train)


k=0
fig, ax = plt.subplots(1,4,figsize=(20,20))
fig.text(s='Sample Image From Each Label',size=18,fontweight='bold',
             fontname='monospace',color=colors_dark[1],y=0.62,x=0.4,alpha=0.8)
for i in labels:
    j=0
    while True :
        if y_train[j]==i:
            ax[k].imshow(X_train[j])
            ax[k].set_title(y_train[j])
            ax[k].axis('off')
            k+=1
            break
        j+=1
        

X_train, y_train = shuffle(X_train,y_train, random_state=101)
X_train.shape

X_train,X_test,y_train,y_test = train_test_split(X_train,y_train, test_size=0.1,random_state=101)
y_train_new = []
for i in y_train:
    y_train_new.append(labels.index(i))
y_train = y_train_new
y_train = tf.keras.utils.to_categorical(y_train)


y_test_new = []
for i in y_test:
    y_test_new.append(labels.index(i))
y_test = y_test_new
y_test = tf.keras.utils.to_categorical(y_test)


effnet = EfficientNetB0(weights='imagenet',include_top=False,input_shape=(image_size,image_size,3))

model = effnet.output
model = tf.keras.layers.GlobalAveragePooling2D()(model)
model = tf.keras.layers.Dropout(rate=0.5)(model)
model = tf.keras.layers.Dense(4,activation='softmax')(model)
model = tf.keras.models.Model(inputs=effnet.input, outputs = model)
model.summary()


model.compile(loss='categorical_crossentropy',optimizer = 'Adam', metrics= ['accuracy'])
tensorboard = TensorBoard(log_dir = 'logs')
checkpoint = ModelCheckpoint("effnet.h5",monitor="val_accuracy",save_best_only=True,mode="auto",verbose=1)
reduce_lr = ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.3, patience = 2, min_delta = 0.001, mode='auto',verbose=1)


history = model.fit(X_train,y_train,validation_split=0.1, epochs =12, verbose=1, batch_size=32, callbacks=[tensorboard,checkpoint,reduce_lr])


filterwarnings('ignore')

epochs = [i for i in range(12)]
fig, ax = plt.subplots(1,2,figsize=(14,7))
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

fig.text(s='Epochs vs. Training and Validation Accuracy/Loss',size=18,fontweight='bold',
             fontname='monospace',color=colors_dark[1],y=1,x=0.28,alpha=0.8)

sns.despine()
ax[0].plot(epochs, train_acc, marker='o',markerfacecolor=colors_green[2],color=colors_green[3],
           label = 'Training Accuracy')
ax[0].plot(epochs, val_acc, marker='o',markerfacecolor=colors_red[2],color=colors_red[3],
           label = 'Validation Accuracy')
ax[0].legend(frameon=False)
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Accuracy')

sns.despine()
ax[1].plot(epochs, train_loss, marker='o',markerfacecolor=colors_green[2],color=colors_green[3],
           label ='Training Loss')
ax[1].plot(epochs, val_loss, marker='o',markerfacecolor=colors_red[2],color=colors_red[3],
           label = 'Validation Loss')
ax[1].legend(frameon=False)
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Training & Validation Loss')

fig.show()


pred = model.predict(X_test)
pred = np.argmax(pred,axis=1)
y_test_new = np.argmax(y_test,axis=1)
model.save('modelv1.h5')
new_model = tf.keras.models.load_model('D:/KPZZNJR/B.Tec/s8/Project/modelv1.h5')
new_model.summary()


print(classification_report(y_test_new,pred))


fig,ax=plt.subplots(1,1,figsize=(14,7))
sns.heatmap(confusion_matrix(y_test_new,pred),ax=ax,xticklabels=labels,yticklabels=labels,annot=True,
           cmap=colors_green[::-1],alpha=0.7,linewidths=2,linecolor=colors_dark[3])
fig.text(s='Heatmap of the Confusion Matrix',size=18,fontweight='bold',
             fontname='monospace',color=colors_dark[1],y=0.92,x=0.28,alpha=0.8)

plt.show()

