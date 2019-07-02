
# coding: utf-8

# In[11]:


import cv2
import tensorflow
import keras
from keras.models import load_model
from scipy.spatial import distance as dist
import numpy as np
import matplotlib as plt
import os
import glob
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
import glob


# In[12]:


def segment_img(img,stride,npw,nph):
#     img = cv2.imread(img)
    height,width, color = img.shape
    test = 0
    # print(height,width,color)
    orig = img
    num_strides_height = height/stride
    num_strides_width = width/stride
    rows = int(num_strides_height)
    columns = int(num_strides_width)
    # print(rows,columns)
    for r in range(rows):
        for c in range(columns):
            x1 = (c)*stride
            y1 = (r)*stride
            x2 = x1+npw
            y2 = y1+nph
            if test == 0:
                in_img = img[y1:y2,x1:x2]
                if(len(in_img.shape)<3):
                    in_img = in_img.reshape(1,227,227,1)
                    in_img = in_img/255
                else:
                    in_img = in_img.astype('uint8')
                    in_img = cv2.cvtColor(in_img,cv2.COLOR_RGB2GRAY)
                    in_img = in_img.reshape(1,227,227,1)
                    in_img = in_img/255
#The below line is used to predict
                result = model.predict(in_img, batch_size=None, verbose=0, steps=None)
                re = result[0]
                if re[0]>0.8:
                    print('cracked')
                    print('confidence: ',re[0]*100,'%')
                    orig = cv2.rectangle(orig,(x1,y1),(x2,y2),(0,0,255),2)
                else:
                    jkl=0
                    # print('no crack')
            if ((width-x2 >= npw) or (width==npw)):
                test = 0
            else:
                break
        if ((height-y2) >= nph or (height==nph)):
            test=0
        else:
            break
    return(orig)


# In[14]:


model = load_model('D://Crack_Detection/model.h5')
os.chdir('D://Crack_Detection')
stride = 50 #pixels to stride frwd or down
#Shape of neural net dectectable image width x height
neural_pixels_width = 227
neural_pixels_height = 227
npw = neural_pixels_width
nph = neural_pixels_height
cap = cv2.VideoCapture(1)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not(ret) :
        print("Connection Lost")
        cap = cv2.VideoCapture(1)
    else:
        # Our operations on the frame come here
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = segment_img(frame,stride,npw,nph)
        cv2.imwrite("test.png",frame)
        # Display the resulting frame
        cv2.imshow('frame',results)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

