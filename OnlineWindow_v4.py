# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\Neuron Labs\Prototype\June 20\Scripts\ver.2\Onlinewindow.ui'
#
# Created by: PyQt5 UI code generator 5.12.2
#
# WARNING! All changes made in this file will be lost!

import cv2
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
import itertools
import images_file

model = load_model('D:/SFU/Capstone/CD_V5.h5')
weights = model.layers[-1].get_weights()[0]
class_weights = weights[:, 1]
intermediate = Model(inputs=model.input, outputs=model.get_layer("block5_conv3").output)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(1127, 823)
        MainWindow.setMinimumSize(QtCore.QSize(1027, 655))
        # MainWindow.setMaximumSize(QtCore.QSize(1980, 1080))
        MainWindow.showMaximized()
        MainWindow.setMouseTracking(False)
        MainWindow.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.title = QtWidgets.QLabel(self.centralwidget)
        self.title.setMinimumSize(QtCore.QSize(175, 60))
        self.title.setMaximumSize(QtCore.QSize(350, 100))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.title.setFont(font)
        self.title.setObjectName("title")
        self.gridLayout.addWidget(self.title, 1, 4, 1, 1)
        self.Startbtn = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.Startbtn.setFont(font)
        self.Startbtn.setObjectName("Startbtn")
        self.gridLayout.addWidget(self.Startbtn, 5, 1, 1, 1)
        self.repdirselect = QtWidgets.QToolButton(self.centralwidget)
        self.repdirselect.setObjectName("repdirselect")
        self.gridLayout.addWidget(self.repdirselect, 4, 5, 1, 1)
        self.videoView = QtWidgets.QLabel(self.centralwidget)
        self.videoView.setFrameShape(QtWidgets.QFrame.Box)
        self.videoView.setText("")
        self.videoView.setObjectName("videoView")
        self.gridLayout.addWidget(self.videoView, 2, 1, 1, 7)
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
        self.gridLayout.addWidget(self.RepGen, 4, 6, 1, 1)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setMaximumSize(QtCore.QSize(160, 30))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 4, 1, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 1, 5, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 4, 7, 1, 1)
        self.uavlogo = QtWidgets.QLabel(self.centralwidget)
        self.uavlogo.setMinimumSize(QtCore.QSize(200, 50))
        self.uavlogo.setMaximumSize(QtCore.QSize(400, 100))
        self.uavlogo.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.uavlogo.setText("")
        self.uavlogo.setPixmap(QtGui.QPixmap(":/newPrefix/uav.png"))
        self.uavlogo.setScaledContents(True)
        self.uavlogo.setObjectName("uavlogo")
        self.gridLayout.addWidget(self.uavlogo, 1, 6, 1, 2)
        self.indicator = QtWidgets.QLabel(self.centralwidget)
        self.indicator.setMinimumSize(QtCore.QSize(30, 30))
        self.indicator.setMaximumSize(QtCore.QSize(30, 30))
        self.indicator.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.indicator.setText("")
        self.indicator.setPixmap(QtGui.QPixmap(":/icon/D:/Neuron Labs/Prototype/June 20/Bin/Images/Greenlight.png"))
        self.indicator.setScaledContents(True)
        self.indicator.setObjectName("indicator")
        self.gridLayout.addWidget(self.indicator, 4, 0, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem2, 1, 3, 1, 1)
        self.NLlogo = QtWidgets.QLabel(self.centralwidget)
        self.NLlogo.setMinimumSize(QtCore.QSize(135, 60))
        self.NLlogo.setMaximumSize(QtCore.QSize(250, 100))
        self.NLlogo.setText("")
        self.NLlogo.setPixmap(QtGui.QPixmap(":/newPrefix/Neuron labs_3-Copy1.png"))
        self.NLlogo.setScaledContents(True)
        self.NLlogo.setObjectName("NLlogo")
        self.gridLayout.addWidget(self.NLlogo, 1, 1, 1, 1)
        self.repdir = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.repdir.sizePolicy().hasHeightForWidth())
        self.repdir.setSizePolicy(sizePolicy)
        self.repdir.setObjectName("repdir")
        self.gridLayout.addWidget(self.repdir, 4, 2, 1, 3)
        self.calBtn = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.calBtn.setFont(font)
        self.calBtn.setObjectName("calBtn")
        self.gridLayout.addWidget(self.calBtn, 5, 7, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1127, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.timer = QTimer()
        self.timer_indicator = QTimer()
        self.timer_indicator.start(20)
        self.timer_indicator.timeout.connect(self.refresh_indicator)
        self.timer.timeout.connect(self.viewCam)
        self.Startbtn.clicked.connect(self.conTimer)
        self.repdirselect.clicked.connect(self.selectFile)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def refresh_indicator(self):
        self.cap = cv2.VideoCapture(1)
        ret, frame = self.cap.read()
        if ret == True:
            self.indicator.setPixmap(QtGui.QPixmap(":/newPrefix/Greenlight.png"))
        else:
            self.indicator.setPixmap(QtGui.QPixmap(":/newPrefix/Redlight.png"))
            self.videoView.setText("Video Capture Device Not Connected")
            self.videoView.setAlignment(Qt.AlignCenter)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.title.setText(_translate("MainWindow", "Real time Crack Detection "))
        self.Startbtn.setText(_translate("MainWindow", "START"))
        self.repdirselect.setText(_translate("MainWindow", "..."))
        self.RepGen.setText(_translate("MainWindow", "Generate Report"))
        self.label.setText(_translate("MainWindow", "Connection status"))
        self.calBtn.setText(_translate("MainWindow", "Calibrate"))

    def selectFile(self):

        video_path, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select file", "","Image Files (*.mkv *.avi *mp4)")
        _translate = QtCore.QCoreApplication.translate
        self.repdir.setText(_translate("MainWindow",video_path))

    def viewCam(self):
        # self.timer_indicator.stop()
        # ret, image = self.cap.read()   # read image data
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        stride = 120 #pixels to stride frwd or down
        #Shape of neural net dectectable image width x height
        neural_pixels_width = 224
        neural_pixels_height = 224
        npw = neural_pixels_width
        nph = neural_pixels_height
        ret, frame = self.cap.read()
        if ret == True:
            self.timer_indicator.stop()
            self.indicator.setPixmap(QtGui.QPixmap(":/newPrefix/Greenlight.png"))
            self.Startbtn.setText(QtCore.QCoreApplication.translate("MainWindow", "STOP"))
            image = segment_img(frame,stride,npw,nph)
            height, width, channel = image.shape
            step = channel * width
            qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
            pmap = QtGui.QPixmap.fromImage(qImg)
            self.videoView.setAlignment(Qt.AlignCenter)
            # self.videoView.setScaledContents(True)  # to display in full area
            self.videoView.setPixmap(pmap)
        else:
            self.indicator.setPixmap(QtGui.QPixmap(":/newPrefix/Redlight.png"))
            self.Startbtn.setText(QtCore.QCoreApplication.translate("MainWindow", "START"))
            self.videoView.setText("Video Capture Device Not Connected")
            self.videoView.setAlignment(Qt.AlignCenter)

    def conTimer(self):
        # if timer is stopped
        video_path = self.repdir.text()
        if not self.timer.isActive():
            # create video capture

            # self.cap = cv2.VideoCapture(video_path)
            self.cap = cv2.VideoCapture(1)
            # self.Startbtn.setText(QtCore.QCoreApplication.translate("MainWindow", "STOP"))

            # start timer
            self.timer.start(20)
        # if timer is runnin
        else:
            # stop timer
            self.timer.stop()
            # release video capture
            self.cap.release()
            self.Startbtn.setText(QtCore.QCoreApplication.translate("MainWindow", "START"))

def segment_img(img,stride,npw,nph):
#     img = cv2.imread(img)
    height,width, color = img.shape
    test = 0
    # print(height,width,color)
#     orig = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    orig=img
    num_strides_height = (height/stride)
    num_strides_width = (width/stride)
    rows = int(num_strides_height)+1
    columns = int(num_strides_width)+1
    blank_image = np.zeros((height,width), np.uint8)
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
#                     in_img = in_img.reshape(1,227,227,1)
                    in_img = in_img/255
                else:
                    in_img = in_img.astype('uint8')
#                     in_img = cv2.cvtColor(in_img,cv2.COLOR_RGB2GRAY)
#                     in_img = in_img.reshape(1,227,227,1)
                    in_img = in_img/255
#The below line is used to predict
                pred = model.predict(in_img[np.newaxis,:,:,:])
                pred_class = np.argmax(pred)
#                 result = model.predict(in_img, batch_size=None, verbose=0, steps=None)
#                 re = result[0]

                if pred_class == 1:
                    alpha = 0.25
                    # orig = cv2.rectangle(orig,(x1,y1),(x2,y2),(0,0,255),2)
                    out = plot_activation(in_img,pred_class)
                    out = out*255
                    out = plt.imsave("crack.png",out)
                    out = cv2.imread('crack.png')
                    out = cv2.applyColorMap(out,cv2.COLORMAP_JET)
                    # out = plt.imwrite(out, cmap='jet', alpha=0.35)
                    cv2.addWeighted(out, alpha, orig[y1:y2,x1:x2], 1 - alpha,0, orig[y1:y2,x1:x2])
#                     cv2.applyColorMap(out,orig[y1:y2,x1:x2],COLORMAP_JET)
#                     blank_image[y1:y2,x1:x2] = out
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
#     orig = blank_image
    return(orig)

def plot_activation(img,pred_class):
    # pred = model.predict(img[np.newaxis,:,:,:])
    # pred_class = np.argmax(pred)
    conv_output = intermediate.predict(img[np.newaxis,:,:,:])
    conv_output = np.squeeze(conv_output)
    h = int(img.shape[0]/conv_output.shape[0])
    w = int(img.shape[1]/conv_output.shape[1])
    activation_maps = sp.ndimage.zoom(conv_output, (h, w, 1), order=1)
    out = np.dot(activation_maps.reshape((img.shape[0]*img.shape[1], 512)), class_weights).reshape(img.shape[0],img.shape[1])
#     plt.imshow(img.astype('float32').reshape(img.shape[0],img.shape[1],3))
#     plt.title('Crack' if pred_class == 1 else 'No Crack')
    return(out)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
