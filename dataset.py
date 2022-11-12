import os
import cv2
import numpy as np
from PIL import Image

def loadImages(dataPath):
    """
    load all Images in the folder and transfer a list of tuples. The first 
    element is the numpy array of shape (m, n) representing the image. 
    The second element is its classification (1 or 0)
      Parameters:
        dataPath: The folder path.
      Returns:
        dataset: The list of tuples.
    """
    # Begin your code (Part 1)
    """----------------Explanation-----------------
    Given the dataPath, which contains two folders: "face" and "non-face", 
    I use os.path.join() to merge the path with two folders' names respectively.
    
    If the given subfolder is named "face", then the dataset within it should be 
    labeled as 1, otherwise, it'll be labeled as 0.
    
    The for-loop is used to read every image 
    in the current directory. 
    Seperate two folders' implementations just to avoid the process read the 
    "non-face" folder first, in that case, the main function would fail, since it 
    simply labels the first image opened as a face image.
    
    Using Image.open() from PIL to open the image files 
    and converting them into numpy arrays, and put the arrays as well as the 
    images' label into the dataset. 
    """
    dataset = []
    T = os.path.join(dataPath,"face")
    F = os.path.join(dataPath,"non-face")
    for image_file in os.listdir(T):
      im = Image.open(os.path.join(T,image_file),"r")
      dataset.append((np.array(im),1))
    for image_file in os.listdir(F):
      im = Image.open(os.path.join(F,image_file),"r")
      dataset.append((np.array(im),0))
    # End your code (Part 1)
    return dataset
