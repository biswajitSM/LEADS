# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\romanbarth\Workspace\PhD\Software\Python\LEADS_croppingExtension\leads\gui\crop_images_ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(946, 120)
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.shiftImageCheckBox = QtWidgets.QCheckBox(Form)
        self.shiftImageCheckBox.setObjectName("shiftImageCheckBox")
        self.gridLayout_2.addWidget(self.shiftImageCheckBox, 0, 0, 1, 2)
        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setObjectName("label_3")
        self.gridLayout_2.addWidget(self.label_3, 1, 0, 1, 1)
        self.angleSpinBox = QtWidgets.QDoubleSpinBox(Form)
        self.angleSpinBox.setDecimals(4)
        self.angleSpinBox.setMinimum(-360)
        self.angleSpinBox.setMaximum(360)
        self.angleSpinBox.setObjectName("angleSpinBox")
        self.gridLayout_2.addWidget(self.angleSpinBox, 1, 1, 1, 1)
        self.gridLayout_3.addLayout(self.gridLayout_2, 0, 0, 1, 1)
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_4 = QtWidgets.QLabel(Form)
        self.label_4.setObjectName("label_4")
        self.gridLayout_4.addWidget(self.label_4, 0, 0, 1, 1)
        self.xShiftSpinBox = QtWidgets.QDoubleSpinBox(Form)
        self.xShiftSpinBox.setDecimals(4)
        self.xShiftSpinBox.setMinimum(-9999999)
        self.xShiftSpinBox.setMaximum(9999999)
        self.xShiftSpinBox.setObjectName("xShiftSpinBox")
        self.gridLayout_4.addWidget(self.xShiftSpinBox, 0, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(Form)
        self.label_5.setObjectName("label_5")
        self.gridLayout_4.addWidget(self.label_5, 1, 0, 1, 1)
        self.yShiftSpinBox = QtWidgets.QDoubleSpinBox(Form)
        self.yShiftSpinBox.setDecimals(4)
        self.yShiftSpinBox.setMinimum(-9999999)
        self.yShiftSpinBox.setMaximum(9999999)
        self.yShiftSpinBox.setObjectName("yShiftSpinBox")
        self.gridLayout_4.addWidget(self.yShiftSpinBox, 1, 1, 1, 1)
        self.gridLayout_3.addLayout(self.gridLayout_4, 0, 1, 1, 1)
        self.gridLayout.addLayout(self.gridLayout_3, 0, 4, 1, 1)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.loadImageBtn = QtWidgets.QPushButton(Form)
        self.loadImageBtn.setObjectName("loadImageBtn")
        self.verticalLayout_3.addWidget(self.loadImageBtn)
        self.loadImageJroisBtn = QtWidgets.QPushButton(Form)
        self.loadImageJroisBtn.setObjectName("loadImageJroisBtn")
        self.verticalLayout_3.addWidget(self.loadImageJroisBtn)
        self.gridLayout.addLayout(self.verticalLayout_3, 0, 0, 1, 1)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.cropSelectedBtn = QtWidgets.QPushButton(Form)
        self.cropSelectedBtn.setObjectName("cropSelectedBtn")
        self.verticalLayout_4.addWidget(self.cropSelectedBtn)
        self.saveImageJroisBtn = QtWidgets.QPushButton(Form)
        self.saveImageJroisBtn.setObjectName("saveImageJroisBtn")
        self.verticalLayout_4.addWidget(self.saveImageJroisBtn)
        self.gridLayout.addLayout(self.verticalLayout_4, 0, 1, 1, 1)
        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.defaultFramesBtn = QtWidgets.QPushButton(Form)
        self.defaultFramesBtn.setObjectName("defaultFramesBtn")
        self.gridLayout_5.addWidget(self.defaultFramesBtn, 0, 0, 1, 1)
        self.bkgSubstractionCheckBox = QtWidgets.QCheckBox(Form)
        self.bkgSubstractionCheckBox.setObjectName("bkgSubstractionCheckBox")
        self.gridLayout_5.addWidget(self.bkgSubstractionCheckBox, 0, 1, 1, 1)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.NumColorsLabel = QtWidgets.QLabel(Form)
        self.NumColorsLabel.setObjectName("NumColorsLabel")
        self.horizontalLayout_7.addWidget(self.NumColorsLabel)
        self.numColorsCbox = QtWidgets.QComboBox(Form)
        self.numColorsCbox.setObjectName("numColorsCbox")
        self.numColorsCbox.addItem("")
        self.numColorsCbox.addItem("")
        self.numColorsCbox.addItem("")
        self.horizontalLayout_7.addWidget(self.numColorsCbox)
        self.gridLayout_5.addLayout(self.horizontalLayout_7, 1, 0, 1, 2)
        self.gridLayout.addLayout(self.gridLayout_5, 0, 3, 1, 1)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(Form)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.frameStartSbox = QtWidgets.QSpinBox(Form)
        self.frameStartSbox.setMaximum(100000)
        self.frameStartSbox.setObjectName("frameStartSbox")
        self.horizontalLayout.addWidget(self.frameStartSbox)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.frameEndBox = QtWidgets.QSpinBox(Form)
        self.frameEndBox.setMinimum(-1)
        self.frameEndBox.setMaximum(1000000)
        self.frameEndBox.setProperty("value", -1)
        self.frameEndBox.setObjectName("frameEndBox")
        self.horizontalLayout_2.addWidget(self.frameEndBox)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.gridLayout.addLayout(self.verticalLayout, 0, 2, 1, 1)

        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")

        self.offsetLabel = QtWidgets.QLabel(Form)
        self.offsetLabel.setObjectName("DescriptionLabel")
        self.horizontalLayout_3.addWidget(self.offsetLabel)

        self.MultiImageSeriesPushButton = QtWidgets.QPushButton(Form)
        self.MultiImageSeriesPushButton.setObjectName("MultiImageSeriesPushButton")
        self.horizontalLayout_3.addWidget(self.MultiImageSeriesPushButton)

        self.ShowAllLayersBtn = QtWidgets.QPushButton(Form)
        self.ShowAllLayersBtn.setObjectName("ShowAllLayersBtn")
        self.horizontalLayout_3.addWidget(self.ShowAllLayersBtn)

        self.HideAllLayersBtn = QtWidgets.QPushButton(Form)
        self.HideAllLayersBtn.setObjectName("HideAllLayersBtn")
        self.horizontalLayout_3.addWidget(self.HideAllLayersBtn)

        self.ToggleChannelVisbilityLabel = QtWidgets.QLabel(Form)
        self.ToggleChannelVisbilityLabel.setObjectName("ToggleChannelVisbilityLabel")
        self.horizontalLayout_3.addWidget(self.ToggleChannelVisbilityLabel)

        self.ToggleChannelVisbility = QtWidgets.QSpinBox(Form)
        self.ToggleChannelVisbility.setMinimum(1)
        self.ToggleChannelVisbility.setMaximum(10)
        self.ToggleChannelVisbility.setProperty("value", 1)
        self.ToggleChannelVisbility.setObjectName("ToggleChannelVisbility")
        self.horizontalLayout_3.addWidget(self.ToggleChannelVisbility)

        self.UpdateTextBtn = QtWidgets.QPushButton(Form)
        self.UpdateTextBtn.setObjectName("UpdateTextBtn")
        self.horizontalLayout_3.addWidget(self.UpdateTextBtn)

        self.FOVLabel = QtWidgets.QLabel(Form)
        self.FOVLabel.setObjectName("FOVLabel")
        self.horizontalLayout_3.addWidget(self.FOVLabel)

        self.FOVLineEdit = QtWidgets.QLineEdit(Form)
        self.FOVLineEdit.setObjectName("FOVLineEdit")
        self.horizontalLayout_3.addWidget(self.FOVLineEdit)

        self.FOVSpinBox = QtWidgets.QSpinBox(Form)
        self.FOVSpinBox.setMinimum(0)
        self.FOVSpinBox.setMaximum(0)
        self.FOVSpinBox.setProperty("value", 0)
        self.FOVSpinBox.setObjectName("FOVSpinBox")
        self.horizontalLayout_3.addWidget(self.FOVSpinBox)

        self.FOVCountLabel = QtWidgets.QLabel(Form)
        self.FOVCountLabel.setObjectName("FOVCountLabel")
        self.horizontalLayout_3.addWidget(self.FOVCountLabel)

        self.BatchProcessBtn = QtWidgets.QPushButton(Form)
        self.BatchProcessBtn.setObjectName("BatchProcessBtn")
        self.horizontalLayout_3.addWidget(self.BatchProcessBtn)

        self.ShiftEstimateBtn = QtWidgets.QPushButton(Form)
        self.ShiftEstimateBtn.setObjectName("ShiftEstimateBtn")
        self.horizontalLayout_3.addWidget(self.ShiftEstimateBtn)

        self.gridLayout.addLayout(self.horizontalLayout_3, 1, 0, 1, 4)

        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")

        self.StatusBarLabel = QtWidgets.QLabel(Form)
        self.StatusBarLabel.setObjectName("StatusBarLabel")
        self.horizontalLayout_4.addWidget(self.StatusBarLabel)

        self.CopyPathBtn = QtWidgets.QPushButton(Form)
        self.CopyPathBtn.setObjectName("CopyPathBtn")
        self.horizontalLayout_4.addWidget(self.CopyPathBtn)

        self.FileDirectoryLabel = QtWidgets.QLabel(Form)
        self.FileDirectoryLabel.setObjectName("FileDirectoryLabel")
        self.horizontalLayout_4.addWidget(self.FileDirectoryLabel)

        self.gridLayout.addLayout(self.horizontalLayout_4, 2, 0, 1, 3)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.shiftImageCheckBox.setText(_translate("Form", "Shift Image         "))
        self.label_3.setText(_translate("Form", "angle(°)"))
        self.label_4.setText(_translate("Form", "x-shift"))
        self.label_5.setText(_translate("Form", "y-shift"))
        self.loadImageBtn.setText(_translate("Form", "Load image sequences"))
        self.loadImageJroisBtn.setText(_translate("Form", "Load imageJ ROIs"))
        self.cropSelectedBtn.setText(_translate("Form", "crop selected ROIs"))
        self.saveImageJroisBtn.setText(_translate("Form", "Save as imageJ ROIs"))
        self.defaultFramesBtn.setText(_translate("Form", "default frames"))
        self.bkgSubstractionCheckBox.setText(_translate("Form", "bkgSubstaction"))
        self.NumColorsLabel.setText(_translate("Form", "#Colors"))
        self.numColorsCbox.setCurrentText(_translate("Form", "1"))
        self.numColorsCbox.setItemText(0, _translate("Form", "1"))
        self.numColorsCbox.setItemText(1, _translate("Form", "2"))
        self.numColorsCbox.setItemText(2, _translate("Form", "3"))
        self.MultiImageSeriesPushButton.setText(_translate("Form", "Open multiple image series"))
        self.label.setText(_translate("Form", "Frame start"))
        self.label_2.setText(_translate("Form", "Frame end"))
        self.ShowAllLayersBtn.setText(_translate("Form", "Show all layers"))
        self.HideAllLayersBtn.setText(_translate("Form", "Hide all layers"))
        self.ToggleChannelVisbilityLabel.setText(_translate("Form", "Show n'th color: n="))
        self.UpdateTextBtn.setText(_translate("Form", "Update text"))
        self.FOVLabel.setText(_translate("Form", "FOV tag:"))
        self.FOVCountLabel.setText(_translate("Form", "/0"))
        self.BatchProcessBtn.setText(_translate("Form", "Batch process"))
        self.ShiftEstimateBtn.setText(_translate("Form", "Estimate shift"))
        self.StatusBarLabel.setText(_translate("Form", "Idle"))
        self.FileDirectoryLabel.setText(_translate("Form", "Current directory:"))        
        self.CopyPathBtn.setText(_translate("Form", "Open path"))        

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
