
#Step1: image resizing! all images should have 
import cv2
import numpy as np
import glob
from math import sqrt


imagelist=glob.glob("D:\\aptos2019\\train_images\\*.png")
debug=1

#this function will resize the image to keep aspect ratio.   
def img_resize(image, width =None,height=None, interpol=None):
    dim=None
    (ht,wt)=image.shape[:2]
    ratio=1
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
    #upscaling = cubic , downscaling = area interpolation 
    if ratio<=1:
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
    #threshold the image
    ret,thresh = cv2.threshold(img,127,255,cv2.THRESH_OTSU)
    #contour the image
    contours,hierarchy = cv2.findContours(thresh, 1, 2)
    cnt = contours[0]

    '''
        #this will draw a rect outside the circle. we look into it if CNN struggles. we try this as a desperate last effort!
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.imshow('circle',cv2.drawContours(img,[box],0,(0,0,255),2))
    '''
    #find the circle that fits the picture in the bg!
    (x,y),radius = cv2.minEnclosingCircle(cnt)
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
    #numpy slicing to the rescue!
    if debug==1:
        img_temp=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        cv2.imshow('circle',cv2.circle(img_temp,center,radius,(0,255,0),2))
    return [y1,y2,x1,x2]



img=cv2.imread(imagelist[0])
#looks like 480px as ht isnt that bad after all.
img_res=img_resize(img,height=480)

#lets do some corrections on the image, i found many images which are of poor brightness/contrast/gamma. I am having trouble to classify so will the CNN 
#But before that we need to ROI the image because the black regions are of no interest and it simply will spoil our efforts in applying the contrast 
#one method i think is to contour the image and discard the rest. simple and effective with minimal info loss(!).
img_gray= cv2.cvtColor(img_res,cv2.COLOR_BGR2GRAY)

#numpy slicing...
coords = crop_image_coords(img_gray)
img_crop=img_res[coords[0]:coords[1],coords[2]:coords[3]]

#
cv2.imshow('cropped',img_crop)
cv2.waitKey(0)

#next step. if you would like to.... find a way to automatically adjust gamma and contrast of the image!
