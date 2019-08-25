import numpy as np
import cv2
import glob
from math import sqrt
#import png
import os

#change the destination to store test images and training images accordingly
dst="D:\\trainown\\"

#os.mkdir("D:\\trainown\\sample")

#change the source to read test images and training images accordingly
imagelist=glob.glob("D:\\aptos2019\\train_images\\*.png")

debug=0
#implemented an autocontrast alg. later found it would be useless.
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

#implemented a gamma correction method. 
def gamma_corr(img,gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(img, table)


#this function will resize the image to keep aspect ratio(or not if user needs) with the requisite interpolation methods.   
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
    ret,thresh = cv2.threshold(img,20,255,cv2.THRESH_BINARY)#,cv2.THRESH_OTSU) # Both Otsu and Triangle fail to threshold some images well. 

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
        #this will draw a rect outside the circle. we look into it if CNN struggles to pick features. 
        #we try this as a desperate last effort!
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

if debug==1:
    imagelist=['4f0866b90c27.png','59ee65760535.png','239f2c348ea4.png','2c77bf969079.png','d25b8a8ad3c4.png','299086c6d1b5.png','663a923d5398.png','4158c340fa49.png','8846b09384a4.png','523b3f0fc646.png']


for i in imagelist:
    if debug==1:
        img=cv2.imread("D:\\aptos2019\\train_images\\"+i)
    else:
        img=cv2.imread(i)
    #looks like 480px as ht isnt that bad after all.
    img_res=img_resize(img,height=480)
    if debug==0:
        filename=str.split(i,"images\\")[1]
        save1=dst+filename
    #lets do some corrections on the image, i found many images which are of poor contrast and gamma.
    #But before that we need to ROI the image because the black regions are of no interest and it simply will spoil our efforts in applying the contrast 
    #one method i think is to contour the image and discard the rest. simple and effective with minimal info loss(!).
    img_gray= cv2.cvtColor(img_res,cv2.COLOR_BGR2GRAY)

    #numpy slicing...
    coords = crop_image_coords(img_gray)
    img_crop=img_res[coords[0]:coords[1],coords[2]:coords[3]]
    #
    if debug==1:
        cv2.imshow('cropped',img_crop)
        cv2.waitKey(0) 
    img_crop=img_resize(img_crop,480,480)
    #resize the image to 480x480 
    if debug==0:
        cv2.imwrite(save1, img_crop)
