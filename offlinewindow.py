# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\Neuron Labs\Prototype\June 20\Layouts\offlinewindow.ui'
#
# Created by: PyQt5 UI code generator 5.12.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QThread, QTimer
from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton, QVBoxLayout, QApplication, QSlider
# from pyqtgraph import ImageView
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer
import cv2
import tensorflow
import keras
import scipy as sp
from keras.models import load_model
from keras.models import Model
from keras.applications import vgg16
from keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator
from keras.models import *
from keras.layers import *
from keras import optimizers
from keras.utils import np_utils
from scipy.spatial import distance as dist
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
import requests
from io import BytesIO
import os
import pickle
# import tqdm
import itertools

import os
path_s = r'D:\Crack_Detection'
if not os.path.exists(path_s):
    os.makedirs(path_s)
os.chdir(path_s)
print(os.getcwd())

model = load_model('D:/SFU/Capstone/CD_V5.h5')
weights = model.layers[-1].get_weights()[0]
class_weights = weights[:, 1]
intermediate = Model(inputs=model.input, outputs=model.get_layer("block5_conv3").output)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(1127, 824)
        MainWindow.setMinimumSize(QtCore.QSize(1027, 655))
        MainWindow.showMaximized()
        # MainWindow.setMaximumSize(QtCore.QSize(1980, 1080))
        MainWindow.setMouseTracking(False)
        MainWindow.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.graphicsView = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.graphicsView.sizePolicy().hasHeightForWidth())
        self.graphicsView.setSizePolicy(sizePolicy)
        self.graphicsView.setObjectName("graphicsView")
        self.gridLayout.addWidget(self.graphicsView, 3, 2, 6, 7)
        self.repdirselect = QtWidgets.QToolButton(self.centralwidget)
        self.repdirselect.setObjectName("repdirselect")
        self.gridLayout.addWidget(self.repdirselect, 9, 6, 1, 1)
        self.title = QtWidgets.QLabel(self.centralwidget)
        self.title.setMinimumSize(QtCore.QSize(350, 100))
        self.title.setMaximumSize(QtCore.QSize(350, 100))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.title.setFont(font)
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        self.title.setObjectName("title")
        self.gridLayout.addWidget(self.title, 1, 5, 1, 1)
        self.NLlogo = QtWidgets.QLabel(self.centralwidget)
        self.NLlogo.setMinimumSize(QtCore.QSize(135, 60))
        self.NLlogo.setMaximumSize(QtCore.QSize(250, 100))
        self.NLlogo.setText("")
        self.NLlogo.setPixmap(QtGui.QPixmap(":/newPrefix/Neuron labs_3-Copy1.png"))
        self.NLlogo.setScaledContents(True)
        self.NLlogo.setObjectName("NLlogo")
        self.gridLayout.addWidget(self.NLlogo, 1, 0, 1, 1)
        self.repdir = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.repdir.sizePolicy().hasHeightForWidth())
        self.repdir.setSizePolicy(sizePolicy)
        self.repdir.setObjectName("repdir")
        self.gridLayout.addWidget(self.repdir, 9, 3, 1, 3)
        self.RepGen = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.RepGen.sizePolicy().hasHeightForWidth())
        self.RepGen.setSizePolicy(sizePolicy)
        self.RepGen.setMaximumSize(QtCore.QSize(150, 30))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.RepGen.setFont(font)
        self.RepGen.setObjectName("RepGen")
        self.gridLayout.addWidget(self.RepGen, 9, 7, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 1, 6, 1, 1)
        self.uavlogo = QtWidgets.QLabel(self.centralwidget)
        self.uavlogo.setMinimumSize(QtCore.QSize(200, 50))
        self.uavlogo.setMaximumSize(QtCore.QSize(400, 100))
        self.uavlogo.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.uavlogo.setText("")
        self.uavlogo.setPixmap(QtGui.QPixmap(":/newPrefix/uav.png"))
        self.uavlogo.setScaledContents(True)
        self.uavlogo.setObjectName("uavlogo")
        self.gridLayout.addWidget(self.uavlogo, 1, 7, 1, 2)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 9, 8, 1, 1)
        self.vfileBtn = QtWidgets.QToolButton(self.centralwidget)
        self.vfileBtn.setObjectName("vfileBtn")
        self.gridLayout.addWidget(self.vfileBtn, 4, 1, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(20, 30, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout.addItem(spacerItem2, 5, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setMaximumSize(QtCore.QSize(300, 30))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 3, 0, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem3, 7, 0, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout.addItem(spacerItem4, 2, 0, 1, 1)
        self.vfileName = QtWidgets.QLineEdit(self.centralwidget)
        self.vfileName.setObjectName("vfileName")
        self.gridLayout.addWidget(self.vfileName, 4, 0, 1, 1)
        # self.runBtn = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        # self.runBtn.setFont(font)
        # self.runBtn.setObjectName("runBtn")
        # self.gridLayout.addWidget(self.runBtn, 6, 0, 1, 1)
        self.trainChk = QtWidgets.QCheckBox(self.centralwidget)
        self.trainChk.setMaximumSize(QtCore.QSize(200, 30))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.trainChk.setFont(font)
        self.trainChk.setObjectName("trainChk")
        self.gridLayout.addWidget(self.trainChk, 8, 0, 1, 1)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem5, 1, 1, 1, 4)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1127, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.vfileBtn.clicked.connect(self.selectFile)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.repdirselect.setText(_translate("MainWindow", "..."))
        self.title.setText(_translate("MainWindow", "Real time Crack Detection "))
        self.RepGen.setText(_translate("MainWindow", "Generate Report"))
        self.vfileBtn.setText(_translate("MainWindow", "..."))
        self.label_5.setText(_translate("MainWindow", "Select video file to process"))
        # self.runBtn.setText(_translate("MainWindow", "Run"))
        self.trainChk.setText(_translate("MainWindow", "Training Mode"))
        # if fileName:
            #******** ADD code to run here **#

    def selectFile(self):

        video_path, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select file", "","Image Files (*.mkv *.avi *mp4)")
        _translate = QtCore.QCoreApplication.translate
        self.vfileName.setText(_translate("MainWindow",video_path))
        capture = cv2.VideoCapture(video_path)
        batch_size = 1
        frames = []
        frame_count = 0
        stride = 120 #pixels to stride frwd or down
        #Shape of neural net dectectable image width x height
        neural_pixels_width = 224
        neural_pixels_height = 224
        npw = neural_pixels_width
        nph = neural_pixels_height
        while True:
            ret, frame = capture.read()
            # Bail out when the video file ends
            if not ret:
                print('not ret')
                break
            # Save each frame of the video to a list
            frame_count += 1
            frames.append(frame)

        print(len(frames))
        if not os.path.exists(path_s+"\\Processed"):
            os.makedirs(path_s+"\\Processed")
        j=0
        for i in range(len(frames)):
            j += 1
            progressBar(j,len(frames))
            results = segment_img(frames[i],stride,npw,nph)
        #         cv2.imshow("Bounding box",results)
        #     name = '{0}.jpg'.format(i)
        #     name = os.path.join(path_s,'\\Processed', name)
            os.chdir(path_s + "\\Processed")
            cv2.imwrite(str(i) + ".jpg", results)
            height, width, channel = results.shape
            step = channel * width
            qImg = QImage(results.data, width, height, step, QImage.Format_RGB888)
            pmap = QtGui.QPixmap.fromImage(qImg)
            self.graphicsView.setAlignment(Qt.AlignCenter)
            # self.videoView.setScaledContents(True)  # to display in full area
            self.graphicsView.setPixmap(pmap)
        #     print('writing to file:{0}'.format(name))
        capture.release()
        images=(glob.glob("D:\\Crack_Detection\\Processed\\*.jpg"))
        # Sort the images by integer index
        images = sorted(images, key=lambda x: float(os.path.split(x)[1][:-3]))

        img = cv2.imread(images[5])
        size = img.shape[1], img.shape[0]
        print(size)
        out = cv2.VideoWriter('Detection.avi',cv2.VideoWriter_fourcc(*'MJPG'), 15,size)
        for i in range(len(images)):
            img = cv2.imread(images[i])
            out.write(img)
            progressBar(i,len(images))
        # os.chdir("C:\\Crack_Detection")
        out.release()
        print("\n Detection completed!")
        os.chdir("D:\\Crack_Detection")


def segment_img(img,stride,npw,nph):
#     img = cv2.imread(img)
    height,width, color = img.shape
    test = 0
    #print(height,width,color)
    orig = img
    num_strides_height = height/stride
    num_strides_width = width/stride
    rows = int(num_strides_height)
    columns = int(num_strides_width)
    #print(rows,columns)
    for r in range(rows):
        for c in range(columns):
            x1 = (c)*stride
            y1 = (r)*stride
            x2 = x1+npw
            y2 = y1+nph
            if test == 0:
                in_img = img[y1:y2,x1:x2] #This is the input image to the detection algorithm
                if(len(in_img.shape)<3):
                    # in_img = in_img.reshape(1,227,227,1)
                    in_img = in_img/255
                else:
                    in_img = in_img.astype('uint8')
                    # in_img = cv2.cvtColor(in_img,cv2.COLOR_RGB2GRAY)
                    # in_img = in_img.reshape(1,227,227,1)
                    in_img = in_img/255
                # result = model.predict(in_img, batch_size=None, verbose=0, steps=None)
                # re = result[0]
                pred = model.predict(in_img[np.newaxis,:,:,:])
                pred_class = np.argmax(pred)
                if pred_class == 1:
                    #print('cracked')
                    # orig = cv2.rectangle(orig,(x1,y1),(x2,y2),(0,0,255),2)
                    out = plot_activation(in_img)
                    out = out*255
                    out = plt.imsave("crack.png",out)
                    out = cv2.imread('crack.png')
                    out = cv2.applyColorMap(out,cv2.COLORMAP_JET)
                    # out = plt.imwrite(out, cmap='jet', alpha=0.35)
                    alpha=0.25
                    cv2.addWeighted(out, alpha, orig[y1:y2,x1:x2], 1 - alpha,0, orig[y1:y2,x1:x2])
    #                 img,crack_length = find_length(orig, pix_wid)
    #                 print('length of crack is ',crack_length,' cm')
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

def progressBar(value, endvalue, bar_length=50):
        percent = float(value) / endvalue
        arrow = '-' * int(round(percent * bar_length)-1) + '>'
        spaces = ' ' * (bar_length - len(arrow))

        sys.stdout.write("\rProgress: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
        sys.stdout.flush()

def plot_activation(img):
    # pred = model.predict(img[np.newaxis,:,:,:])
    # pred_class = np.argmax(pred)

    # weights = model.layers[-1].get_weights()[0]
    # class_weights = weights[:, pred_class]
    # intermediate = Model(inputs=model.input, outputs=model.get_layer("block5_conv3").output)
    conv_output = intermediate.predict(img[np.newaxis,:,:,:])
    conv_output = np.squeeze(conv_output)

    h = int(img.shape[0]/conv_output.shape[0])
    w = int(img.shape[1]/conv_output.shape[1])

    activation_maps = sp.ndimage.zoom(conv_output, (h, w, 1), order=1)
    out = np.dot(activation_maps.reshape((img.shape[0]*img.shape[1], 512)), class_weights).reshape(img.shape[0],img.shape[1])

#     plt.imshow(img.astype('float32').reshape(img.shape[0],img.shape[1],3))
#     plt.title('Crack' if pred_class == 1 else 'No Crack')
    return(out)


import images

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())
