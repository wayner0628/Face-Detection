import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
import numpy as np

def detect(dataPath, clf):
    """
    Please read detectData.txt to understand the format. Load the image and get
    the face images. Transfer the face images to 19 x 19 and grayscale images.
    Use clf.classify() function to detect faces. Show face detection results.
    If the result is True, draw the green box on the image. Otherwise, draw
    the red box on the image.
      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
    """------------------Explanation-------------------
    At first, open the given path, which will be a txt-file containing all informaintion
    needed.

    Then, read the txt-file and save its contents in a list.
    
    When reading a line, I maintain a variable called "num" representing whether
    the former picture is done. That is, if a new picture is read, we would have 
    its face amount, so we update the variable to this value, as a classificaion is
    done for a face region, the variable is reduced by 1, then when it goes down to 
    0, we know that the all of the face region is scanned, the result can be output,
    and the next line in txt-file would be the information of the next image.

    After acquiring the current image's file-name, merging it with the dataPath's 
    directory path, then we can open the image file.

    The next step is using some function in pillow(PIL) to get the face region and
    change the formet, which involves crop(), resize() and convert().

    After all, let the trained classifier classify the processed image, if it's a
    face use rectangle() to paint it green on the original image, otherwise, paint 
    it red.
    """
    fin = open(dataPath,'r')
    path = os.path.dirname(dataPath)
    text = []
    num = 0
    fig, ax = plt.subplots(nrows = 2,ncols = 1,figsize = (30,30))
    ax[0].axis('off')
    ax[1].axis('off')
    row = 0
    for line in fin.readlines():
      text.append(line)
    for t in text:
      if(num == 0):
        s = t.split(' ')
        file = os.path.join(path,s[0])
        im = Image.open(file)
        num = int(s[1])
        draw = ImageDraw.ImageDraw(im)
      else:  
        num -= 1
        s = t.split(' ')
        box = im.crop( (int(s[0]), int(s[1]), int(s[0])+int(s[2]), int(s[1])+int(s[3])) )
        face = box.resize( (19,19) )
        gray = face.convert('L')
        is_face = clf.classify( np.array(gray) )
        if(is_face):
          draw.rectangle( ((int(s[0]),int(s[1])),(int(s[0])+int(s[2]), int(s[1])+int(s[3]))), fill = None, outline = 'green', width = 3 )
        else:
          draw.rectangle( ((int(s[0]),int(s[1])), (int(s[0])+int(s[2]), int(s[1])+int(s[3]))), fill = None, outline = 'red', width = 3 )
        if(num == 0):
          ax[row].imshow(im)
          row += 1

    plt.show()
    fin.close
    # End your code (Part 4)
