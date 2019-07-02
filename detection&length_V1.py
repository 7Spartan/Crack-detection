
# coding: utf-8

# In[1]:


import cv2
# import tensorflow
# import keras
# from keras.models import load_model
# from scipy.spatial import distance as dist
import numpy as np
import matplotlib as plt
import os


# In[25]:


def find_length(img,pix_width):
    os.chdir('C:/Users/deepa/Downloads')
    if (len(img)<3):
        img = img
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Original",img)
    cv2.imwrite('Original_1.png',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    gray = cv2.GaussianBlur(img, (7, 7), 0)
    img = cv2.Canny(gray, 50, 100)
    cv2.imshow("Blur & Canny",img)
    cv2.imwrite('Blur & Canny_2.png',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    img = cv2.dilate(img, None, iterations=1)
    cv2.imshow("dialate",img)
    cv2.imwrite('dialate_3.png',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=4)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    # minimum size of particles we want to keep (number of pixels)
    #can be changed to mean of blob sizes etc,..
    min_size = 800
    #Create a template of same size to put the required pixels on it
    img2 = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
    #Save the image
    cv2.imwrite('test.png',img2)
    cv2.imwrite('template_4.png',img2)
    cv2.imshow("tempalte",img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    #Erode to null the dilute effect
    img = cv2.erode(img2, None, iterations=1)
    #read the image with only crack
    img2 = cv2.imread('C:/Users/deepa/Downloads/test.png')
    # orig = cv2.imread('/home/spartan/Desktop/Cracks/16.jpg')
    #Bitwise and on the original image to give only crack part
    img3 = cv2.bitwise_and(img2,orig)
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    img3 = cv2.dilate(img3, None, iterations=3)
    cv2.imshow("Original",img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    # cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL,
    #                         cv2.CHAIN_APPROX_SIMPLE)
    ret,thresh = cv2.threshold(img3,127,255,0)
    _,contours,hierarchy = cv2.findContours(thresh, 1, 2)
    n=len(contours)-1
    cnt = contours[n]
    print(len(contours))
    # print(cnt)
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    center = (int(x),int(y))
    radius = int(radius)
    cv2.circle(orig,center,radius,(255,255,255),2)
    crack_length = radius*2*pix_width
    cv2.imwrite('circle_5.png',orig)
    print('Length of crack is ',crack_length,' centimeters')
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(orig,str(crack_length)+" pxls",(150,150),font,0.5,(0,255,255),2,cv2.LINE_AA)
    cv2.imshow("length",orig)
    cv2.imwrite('Length_6.png',orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    return orig,crack_length


# In[26]:


img = 'C:/Users/deepa/Downloads/test_1.jpg'
img = cv2.imread(img)
orig = img
pix_wid = 1
img = img[:227,:227]
img,crack_length = find_length(orig, pix_wid)
print('length of crack is ',crack_length,' cm')
cv2.imshow("Imageop",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)


# In[31]:


img = '/home/spartan/Desktop/Cracks/Train_img/9.jpg'
img = cv2.imread(img)
orig = img
pix_wid = 1
img=img[:227,:227]
# print(img.shape)
model = load_model('/home/spartan/Desktop/Cracks/model.h5')
# img=img_preprocess(img)
# img=img.reshape(277,277,1)

if(len(img.shape)<3):
#         img = cv2.resize(img, (128,128)) #To match the NVIDIA model architecture
    img = img.reshape(1,227,227,1)
    img = img/255
else:
    img = img.astype('uint8')
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
#     cv2.imshow("Edge",img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     cv2.waitKey(1)
#         img = cv2.GaussianBlur(img,(3,3),0) #o deviation of kernal, kernal size (3,3)
#         img = cv2.resize(img, (128,128)) #To match the NVIDIA model architecture
    img = img.reshape(1,227,227,1)
    img = img/255

result = model.predict(img, batch_size=None, verbose=1, steps=None)
re = result[0]
if re[0]>re[1]:
    print('cracked')
    img,crack_length = find_length(orig, pix_wid)
    print('length of crack is ',crack_length,' cm')
    cv2.imshow("Imageop",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    
else:
    print('no crack')


# In[20]:




