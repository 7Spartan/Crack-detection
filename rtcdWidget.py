# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\Neuron Labs\Prototype\June 20\Scripts\ver.2\rtcdWidget.ui'
#
# Created by: PyQt5 UI code generator 5.12.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import os
import sys

class Ui_QWidget(object):
    def setupUi(self, QWidget):
        QWidget.setObjectName("QWidget")
        QWidget.resize(668, 478)
        QWidget.setToolTipDuration(-1)
        QWidget.setAutoFillBackground(False)
        self.label = QtWidgets.QLabel(QWidget)
        self.label.setGeometry(QtCore.QRect(10, 10, 201, 101))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap(":/newPrefix/Neuron labs_3-Copy1.png"))
        self.label.setScaledContents(True)
        self.label.setTextInteractionFlags(QtCore.Qt.NoTextInteraction)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(QWidget)
        self.label_2.setGeometry(QtCore.QRect(410, 30, 241, 51))
        self.label_2.setText("")
        self.label_2.setPixmap(QtGui.QPixmap(":/newPrefix/uav.png"))
        self.label_2.setScaledContents(True)
        self.label_2.setTextInteractionFlags(QtCore.Qt.NoTextInteraction)
        self.label_2.setObjectName("label_2")
        self.verticalLayoutWidget = QtWidgets.QWidget(QWidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(190, 150, 305, 231))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.onlineMbtn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Bauhaus 93")
        font.setPointSize(15)
        self.onlineMbtn.setFont(font)
        self.onlineMbtn.setMouseTracking(True)
        self.onlineMbtn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.onlineMbtn.setWhatsThis("")
        self.onlineMbtn.setObjectName("onlineMbtn")
        self.verticalLayout.addWidget(self.onlineMbtn)
        self.onlineMbtn_2 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Bauhaus 93")
        font.setPointSize(15)
        self.onlineMbtn_2.setFont(font)
        self.onlineMbtn_2.setMouseTracking(True)
        self.onlineMbtn_2.setFocusPolicy(QtCore.Qt.NoFocus)
        self.onlineMbtn_2.setWhatsThis("")
        self.onlineMbtn_2.setObjectName("onlineMbtn_2")
        self.verticalLayout.addWidget(self.onlineMbtn_2)
        self.offlineMbtn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Bauhaus 93")
        font.setPointSize(15)
        self.offlineMbtn.setFont(font)
        self.offlineMbtn.setMouseTracking(True)
        self.offlineMbtn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.offlineMbtn.setWhatsThis("")
        self.offlineMbtn.setObjectName("offlineMbtn")
        self.verticalLayout.addWidget(self.offlineMbtn)
        self.onlineMbtn.clicked.connect(self.realtime_lite)
        self.onlineMbtn_2.clicked.connect(self.realtime)
        self.offlineMbtn.clicked.connect(self.offline)

        self.retranslateUi(QWidget)
        QtCore.QMetaObject.connectSlotsByName(QWidget)

    def retranslateUi(self, QWidget):
        _translate = QtCore.QCoreApplication.translate
        QWidget.setWindowTitle(_translate("QWidget", "Real Time Crack Detection"))
        self.onlineMbtn.setText(_translate("QWidget", "RealTime Mode (Lite)"))
        self.onlineMbtn_2.setText(_translate("QWidget", "RealTime Mode"))
        self.offlineMbtn.setText(_translate("QWidget", "Offline Mode"))

    def realtime_lite(self):
        os.system("python OnlineWindow_v3.py")

    def realtime(self):
        os.system("python OnlineWindow_v4.py")

    def offline(self):
        os.system("python offlinewindow.py")


import images


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    QWidget = QtWidgets.QWidget()
    ui = Ui_QWidget()
    ui.setupUi(QWidget)
    QWidget.show()
    sys.exit(app.exec_())
