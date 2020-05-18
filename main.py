# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import zipfile
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import numpy as np
from cv2 import cvtColor,imshow,imread,imwrite,COLOR_BGR2YCR_CB,COLOR_YCR_CB2BGR,equalizeHist,createCLAHE,waitKey,calcHist
import glob
from math import sqrt
#import png
import os

debug=0

def autocontr(img):
    imageMin=np.min(img)
    imageMax=np.max(img)
    size=img.shape
    scale=(imageMax-imageMin)
    imin=np.multiply(np.ones(size).astype(np.float),imageMin/scale)
    scaledimg=np.divide(img.astype(np.float),scale)
    scaledimg = (255*(np.subtract(scaledimg,imin)))
    scaledimg = scaledimg.astype(np.uint8)
    return scaledimg

def gamma_corr(img,gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(img, table)


#this function will resize the image to keep aspect ratio.   
def img_resize(image, width =None,height=None, interpol=None):
    dim=None
    (ht,wt)=image.shape[:2]
    ratio=-1
    hratio=0
    hratio=0
    if width is None and height is None :
        return image
    # maintain aspect ratio
    elif width is None:
        ratio = height/float(ht)
        dim = (int(wt*ratio),height)
    elif height is None:
        ratio = width/float(wt)
        dim = (width,int(ht*ratio))
    # otherwise make CV resize as default which loses on aspect ratio
    else:
        dim=(width,height)
        wratio= width/float(wt)
        hratio= height/float(ht)
    #upscaling = cubic , downscaling = area interpolation 
    if ratio==-1:
        if interpol==None:
            if hratio<1 and wratio<1:
                   resized_img=cv2.resize(image,dim,interpolation=cv2.INTER_AREA)
            elif hratio<1 and wratio>1:
                   resized_img=cv2.resize(image,(wt,height),interpolation=cv2.INTER_AREA)
                   resized_img=cv2.resize(resized_img,(width,height),interpolation=cv2.INTER_CUBIC)
            elif hratio>1 and wratio<1:       
                   resized_img=cv2.resize(image,(width,ht),interpolation=cv2.INTER_AREA)
                   resized_img=cv2.resize(resized_img,(width,height),interpolation=cv2.INTER_CUBIC)
            else:
                   resized_img=cv2.resize(image,dim,interpolation=cv2.INTER_CUBIC)
        else:
            resized_img=cv2.resize(image,dim,interpolation=interpol)
    elif ratio>0 and ratio<=1:
        if interpol==None:
            resized_img=cv2.resize(image,dim,interpolation=cv2.INTER_AREA)
        else:
            resized_img=cv2.resize(image,dim,interpolation=interpol)
    else:
        if interpol==None:
            resized_img=cv2.resize(image,dim,interpolation=cv2.INTER_CUBIC)
        else:
            resized_img=cv2.resize(image,dim,interpolation=interpol)
    return resized_img

# this function will do a ROI on the Img
def crop_image_coords(img):
    #autocontrast the image
    equ = cv2.equalizeHist(img)
    gam = gamma_corr(img,2.2)
    ret,thresh = cv2.threshold(img,20,255,cv2.THRESH_BINARY)#,cv2.THRESH_OTSU)

    if debug==1:
        cv2.imshow('gamma',gam)
        cv2.imshow('equalized',equ)
        cv2.imshow('thresh',thresh)
    #contour the image
    contours,hierarchy = cv2.findContours(thresh, 1, 2)
    max_ar=-1
    best=[]
    #find best contour with the max area
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_ar:
            max_ar=area
            best=cnt    
    '''
        #this will draw a rect outside the circle. we look into it if CNN struggles. we try this as a desperate last effort!
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.imshow('circle',cv2.drawContours(img,[box],0,(0,0,255),2))
    '''
    #find the circle that fits the picture in the bg!
    (x,y),radius = cv2.minEnclosingCircle(best)
    center = (int(x),int(y))
    radius = int(radius)
    #calculate the max rectangle that can fit a circle which is a square! with radius*1.414 as the side! 
    halfside=radius/sqrt(2)
    # points on the image used to crop
    #left top corner
    x1=int(max(x-halfside,0))
    y1=int(max(y-halfside,0))
    #left bottom corner
    x2=int(min(x+halfside,len(img[0])))
    y2=int(min(y+halfside,len(img)))
    #numpy slicing 
    if debug==1:
        img_temp=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        cv2.imshow('circle',cv2.circle(img_temp,center,radius,(0,255,0),2))
    return [y1,y2,x1,x2]


def generate(src_path,dst_path):
    imagelist=glob.glob(src_path+"*.png")
    if debug==1:
        imagelist=['4f0866b90c27.png','59ee65760535.png','239f2c348ea4.png','2c77bf969079.png','d25b8a8ad3c4.png','299086c6d1b5.png','663a923d5398.png','4158c340fa49.png','8846b09384a4.png','523b3f0fc646.png']
    for i in imagelist:
        if debug==1:
            img=cv2.imread(src_path+i)
        else:
            img=cv2.imread(i)
    #looks like 480px as ht isnt that bad after all.
        img_res=img_resize(img,height=480)
        if debug==0:
            filename=str.split(i,"images\\")[1]
            save_path=dst_path+filename
    #lets do some corrections on the image, i found many images which are of poor brightness/contrast/gamma. I am having trouble to classify so will the CNN 
    #But before that we need to ROI the image because the black regions are of no interest and it simply will spoil our efforts in applying the contrast 
    #one method i think is to contour the image and discard the rest. simple and effective with minimal info loss(!).
        img_gray= cv2.cvtColor(img_res,cv2.COLOR_BGR2GRAY)
        #numpy slicing...
        coords = crop_image_coords(img_gray)
        img_crop=img_res[coords[0]:coords[1],coords[2]:coords[3]]
        if debug==1:
            cv2.imshow('cropped',img_crop)
            cv2.waitKey(0) 
        img_crop=img_resize(img_crop,480,480)
    #resize the image to 480x480 
        if debug==0:
            cv2.imwrite(save_path, img_crop)
    return 0


#MODEL BUILDING STARTS HERE




# %% [code]
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,Dropout, BatchNormalization
import numpy as np
from sklearn.model_selection import train_test_split
from cv2 import imread, imshow, imwrite, equalizeHist
from keras.utils import to_categorical

def clahe_equalize(img,clipLimit=2.0,tileGridsize=(8,8)):
   imgYCRCB = cvtColor(img,COLOR_BGR2YCR_CB)
   #Equalize the luma channel
   img_BGR = cvtColor(imgYCRCB,COLOR_YCR_CB2BGR)
   clahe = createCLAHE(clipLimit, tileGridsize)
   imgYCRCB[:,:,0] = clahe.apply(imgYCRCB[:,:,0])
   img_BGR = cvtColor(imgYCRCB,COLOR_YCR_CB2BGR)
return img_BGR

def preprocess(img):

    #its for histogram normalization/Equalization! ref to why i am doing this : https://www.sciencedirect.com/topics/computer-science/normalized-histogram
    return clahe_equalize(img)


filename = "/kaggle/input/aptos2019-blindness-detection/train.csv"
data=pd.read_csv(filename)
data_x=data[[data.columns[0]]]
data_y=data[[data.columns[1]]]

train_data_x, test_data_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.30)

#one-hot encode target column, free unused variables
data_x=None
data_y=None
train_y = to_categorical(train_y)
test_y = to_categorical(test_y)


train_x=[]
test_x=[]
#load all training set
for i in train_data_x["id_code"]:
    #preprocess(i)
    train_x.append(imread("/tmp/train_resize/"+str(i)+".png"))
train_x_np=np.empty((len(train_x),480,480,3),dtype=np.float64)
for i in range(len(train_x)):
    train_x_np[i]=preprocess(train_x[i])
train_x=None
#train_x_np=train_x_np.reshape(len(train_x_np),480,480,1)
train_x_np=train_x_np/255.0

# %% [code]
#load the test set
for i in test_data_x["id_code"]:
    #preprocesss(i)
    test_x.append(imread("/tmp/train_resize/"+str(i)+".png"))
test_x_np =np.empty((len(test_x),480,480,3),dtype=np.float64)
for i in range(len(test_x)):
    test_x_np[i]=preprocess(test_x[i])
test_x=None
#test_x_np=test_x_np.reshape(len(test_x_np),480,480,1)

test_x_np=test_x_np/255.0

# %% [markdown]
# **Build the Model architecture here**

# %% [code]
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,Dropout
#model
model = Sequential()

#K.I.S.S model
#add model layers - 
#VGGNet inspired.

model.add(Conv2D(32, kernel_size=5,strides=1, activation='relu', input_shape=(480,480,3)))
model.add(Conv2D(32, kernel_size=5,strides=1, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))
model.add(GaussianNoise(0.1))

model.add(Conv2D(64, kernel_size=3,strides=1, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=3,strides=1, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))
model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=3,strides=1, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=3,strides=1, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))
model.add(GaussianNoise(0.01))

model.add(Conv2D(256, kernel_size=3,strides=1, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))
model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size=3,strides=1, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Flatten())
#NoFC
model.add(Dense(5, activation='softmax'))


# %% [markdown]
# **Compile and Fit**

# %% [code]
#compile model using accuracy to measure model performance
#Kaggle is compiling this part before committing. so please remove comments and add them when committing (trial)
#model.compile(optimizer='adam', loss='kullback_leibler_divergence', metrics=['accuracy'])
#train the model
#model.fit(train_x_np, train_y, validation_data=(test_x_np, test_y), epochs=20)

# %% [code]
