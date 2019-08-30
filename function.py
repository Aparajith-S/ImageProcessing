from cv2 import cvtColor,imshow,imread,imwrite,COLOR_BGR2YCR_CB,COLOR_YCR_CB2BGR,equalizeHist,createCLAHE,waitKey,calcHist
import numpy as np
from matplotlib import pyplot as plt

def displayhist(img):
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = calcHist([img],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.show()

def equalize(img):
  imgYCRCB = cvtColor(img,COLOR_BGR2YCR_CB)
  #Equalize the luma channel
  imgYCRCB[:,:,0] = equalizeHist(imgYCRCB[:,:,0])
  img_BGR = cvtColor(imgYCRCB,COLOR_YCR_CB2BGR)
  return img_BGR
  
#its better to use Clahe - Contrast ltd. adapt. hist. eq. as it equalizes histograms not with a global contrast.  
def clahe_equalize(img,clipLimit=2.0,tileGridsize=(8,8)):
   imgYCRCB = cvtColor(img,COLOR_BGR2YCR_CB)
   #Equalize the luma channel
   img_BGR = cvtColor(imgYCRCB,COLOR_YCR_CB2BGR)
   clahe = createCLAHE(clipLimit, tileGridsize)
   imgYCRCB[:,:,0] = clahe.apply(imgYCRCB[:,:,0])
   img_BGR = cvtColor(imgYCRCB,COLOR_YCR_CB2BGR)
   return img_BGR

img=imread("C:\\aptos2019\\train_resize\\000c1434d8d7.png")

img_eq=equalize(img)
img_clahe=clahe_equalize(img,3.0)
imshow('original',img)
imshow('equalizehist',img_eq)
imshow('claheHist',img_clahe)

displayhist(img)
displayhist(img_eq)
displayhist(img_clahe)

waitKey(0)


# please compare both the methods and see if you can view changes in the image. 
# One idea is to put it in a loop and view a couple of images that are really horrible.
# i used equalize. There seems to be a better way- using adaptive histogram equalizaton - 
# this looks promising but, we need to check this on few images which are of really good and really horrible quality. 
