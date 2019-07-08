# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\Neuron Labs\Prototype\June 20\Layouts\Onlinewindow.ui'
#
# Created by: PyQt5 UI code generator 5.12.2
#
# WARNING! All changes made in this file will be lost!
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QThread, QTimer
from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton, QVBoxLayout, QApplication, QSlider
from pyqtgraph import ImageView
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer





class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(1127, 823)
        MainWindow.setMinimumSize(QtCore.QSize(1027, 655))
        MainWindow.setMaximumSize(QtCore.QSize(1980, 1080))
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
        self.repdirselect = QtWidgets.QToolButton(self.centralwidget)
        self.repdirselect.setObjectName("repdirselect")
        self.gridLayout.addWidget(self.repdirselect, 3, 5, 1, 1)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout.addWidget(self.pushButton, 4, 1, 1, 1)
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
        self.gridLayout.addWidget(self.repdir, 3, 2, 1, 3)
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
        self.gridLayout.addWidget(self.RepGen, 3, 6, 1, 1)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setMaximumSize(QtCore.QSize(160, 30))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 3, 1, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 3, 7, 1, 1)
        self.uavlogo = QtWidgets.QLabel(self.centralwidget)
        self.uavlogo.setMinimumSize(QtCore.QSize(200, 50))
        self.uavlogo.setMaximumSize(QtCore.QSize(400, 100))
        self.uavlogo.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.uavlogo.setText("")
        self.uavlogo.setPixmap(QtGui.QPixmap(":/newPrefix/uav.png"))
        self.uavlogo.setScaledContents(True)
        self.uavlogo.setObjectName("uavlogo")
        self.gridLayout.addWidget(self.uavlogo, 1, 6, 1, 2)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 1, 5, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem2, 1, 3, 1, 1)
        self.indicator = QtWidgets.QLabel(self.centralwidget)
        self.indicator.setMinimumSize(QtCore.QSize(30, 30))
        self.indicator.setMaximumSize(QtCore.QSize(30, 30))
        self.indicator.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.indicator.setText("")
        self.indicator.setPixmap(QtGui.QPixmap(":/icon/D:/Neuron Labs/Prototype/June 20/Bin/Images/Greenlight.png"))
        self.indicator.setScaledContents(True)
        self.indicator.setObjectName("indicator")
        self.gridLayout.addWidget(self.indicator, 3, 0, 1, 1)
        self.videoView = QtWidgets.QLabel(self.centralwidget)
        self.videoView.setFrameShape(QtWidgets.QFrame.Box)
        self.videoView.setText("")
        self.videoView.setObjectName("videoView")
        self.gridLayout.addWidget(self.videoView, 2, 1, 1, 7)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1127, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.timer = QTimer()
        self.timer.start(20)

        self.timer.timeout.connect(self.viewcam)
        # self.RepGen.clicked.connect(self.Contimer)



        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.title.setText(_translate("MainWindow", "Real time Crack Detection "))
        self.repdirselect.setText(_translate("MainWindow", "..."))
        self.pushButton.setText(_translate("MainWindow", "Calibrate"))
        self.RepGen.setText(_translate("MainWindow", "Generate Report"))
        self.label.setText(_translate("MainWindow", "Connection status"))


# class camera(QWidget):
#     #class constructor
#     def __init__(self):
#         # call QWidget constructor
#         super().__init__()
#
    def viewcam(self):
        # read image from camera
        self.cap = cv2.VideoCapture(0)
        # start timer

        ret, image = self.cap.read()

        # convert image to RGB

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # get image infos
        height, width, channel = image.shape
        step = channel * width
        # create QImage from image
        qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
        pixmap = QtGui.QPixmap(qImg)
        # pixmap = pixmap.scaled(self.videoView.width(),self.videoView.height(), QtCore.Qt.KeepAspectRatio)
        # pixmap = pixmap.scaled(QtCore.Qt.AlignCenter)
        # show image in img_label
        self.videoView.setPixmap(pixmap)
        # self.videoView.setPixmap(QPixmap.fromImage(qImg))
        # QPixmap.scaled(self.videoView.width(),self.videoView.height(), QtCore.Qt.KeepAspectRatio)

    #start stop timer

    # def Contimer(self):
    #      # if timer is stopped
    #     if not self.timer.isActive():
    #         # create video capture
    #         self.cap = cv2.VideoCapture(0)
    #         # start timer
    #         self.timer.start(20)
    #     #if timer is runnin
    #     else:
    #         #stop timer
    #         self.timer.stop()
    #         #release video capture
    #         self.cap.release()





import images


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
