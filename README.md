# Image Preprocessing and initial classifier for Aptos 2019 

This is a starting point code created for an image classifier to detect diabetic retinopathy.
link to problem statement: https://www.kaggle.com/c/aptos2019-blindness-detection 

## Desired output 

Images have five possible ratings, 

    {0 , 1 , 2 , 3 , 4 }

Each image is characterized by a tuple (e,e), which corresponds to its scores by Rater A (human) and Rater B (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix O is constructed, such that O corresponds to the number of images that received a rating i by A and a rating j by B. An N-by-N matrix of weights, w, is calculated based on the difference between raters' scores:

An N-by-N histogram matrix of expected ratings, E, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that E and O have the same sum.

## Code

`function.py` is used to test the images to do various corrections to check which one yields a better result.

`main.py` contains everything, from preprocessing till the model training and validation.

## Data
This repo does not contain data owing to its large size, however, The data is available here : https://www.kaggle.com/c/aptos2019-blindness-detection/data
