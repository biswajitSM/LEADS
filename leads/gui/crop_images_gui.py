import napari
import numpy as np
import pyqtgraph as pg
import dask.array as da
import matplotlib.path as mpltPath
import os, sys, glob, itertools, yaml, re, shutil, pims

from . import crop_images_ui
from .. import io
from .. import crop_images 

from PIL import Image
from tqdm import trange
from PyQt5 import QtCore, QtGui, QtWidgets
from skimage import measure
from pathlib import Path
from roifile import ImagejRoi
from tifffile import imwrite
from pyqtgraph import PlotWidget, plot

from vispy.color import Colormap, ColorArray
from scipy.interpolate import interp1d
from napari.layers.utils.text import TextManager
from napari.layers.utils._text_utils import format_text_properties

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
class LineProfileWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        # super(MainWindow, self).__init__(*args, **kwargs)
        
        self.setWindowTitle("Line profiles")
        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)

        self.data_line = []
        for n in range(10): # take max 10 lines
            self.data_line.append(self.graphWidget.plot([0], [0]))

# ---------------------------------------------------------------------
    def update_plot_data(self, profiles, colors):

        x = list(range(len(profiles[0])))
        numLines = len(self.data_line)
        for p in range(len(profiles)):
            if p >= numLines:
                print('Can show max. '+str(int(numLines))+' lines. Disable some layers.')
                continue
            color = colors[p] * 255
            self.data_line[p].setData(x, profiles[p], pen=pg.mkPen(color=color))
        for p in range(len(profiles), len(self.data_line)):
            self.data_line[p].setData([0], [0])

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
class BatchProcessingDialog(QtWidgets.QDialog):
    """ The dialog allowing the user to configure paths for batch processing """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.window = parent
        self.setWindowTitle("Batch Processing")
        # self.resize(300, 0)
        self.resize(481, 436)

        # load settings
        self.settings = io.load_user_settings()

        # initialize list of QtItems
        numItems = 2
        self.horizontalLayoutWidget = ['']*numItems
        self.horizontalLayout       = ['']*numItems
        self.Label                  = ['']*numItems
        self.PathLineEdit           = ['']*numItems
        self.BrowseButton           = ['']*numItems

        # OK and Cancel buttons
        self.OKCancelButtonBox = QtWidgets.QDialogButtonBox(self)
        self.OKCancelButtonBox.setGeometry(QtCore.QRect(40, 390, 341, 32))
        self.OKCancelButtonBox.setOrientation(QtCore.Qt.Horizontal)
        self.OKCancelButtonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.OKCancelButtonBox.setObjectName("OKCancelButtonBox")

        # group box to contain several h layouts within a v layout
        # add up to 10 horizontalLayouts like these
        self.groupBox = QtWidgets.QGroupBox(self)
        self.groupBox.setGeometry(QtCore.QRect(10, 10, 463, 380))
        self.groupBox.setObjectName("groupBox")

        # directory to process
        for nPath in range(2):
            self.horizontalLayoutWidget[nPath] = QtWidgets.QWidget(self.groupBox)
            self.horizontalLayoutWidget[nPath].setGeometry(QtCore.QRect(10, 30+nPath*40, 441, 30))
            self.horizontalLayoutWidget[nPath].setObjectName("horizontalLayoutWidget_"+str(nPath))

            self.horizontalLayout[nPath] = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget[nPath])
            self.horizontalLayout[nPath].setContentsMargins(0, 0, 0, 0)
            self.horizontalLayout[nPath].setObjectName("horizontalLayout_"+str(nPath))
            
            self.Label[nPath] = QtWidgets.QLabel(self.horizontalLayoutWidget[nPath])
            self.Label[nPath].setObjectName("Label_"+str(nPath))
            self.horizontalLayout[nPath].addWidget(self.Label[nPath])
            
            self.PathLineEdit[nPath] = QtWidgets.QLineEdit(self.horizontalLayoutWidget[nPath])
            self.PathLineEdit[nPath].setObjectName("PathLineEdit_"+str(nPath))
            self.horizontalLayout[nPath].addWidget(self.PathLineEdit[nPath])   

            try:
                if nPath == 0:
                    self.PathLineEdit[nPath].setText(self.settings["Batchcrop"]["PWD BATCH"])
                    self.BatchCropPath = self.settings["Batchcrop"]["PWD BATCH"]
                elif nPath == 1:
                    self.PathLineEdit[nPath].setText(self.settings["Batchcrop"]["PWD SAVE"])
                    self.BatchSavePath = self.settings["Batchcrop"]["PWD SAVE"]
            except:
                pass
            
            self.BrowseButton[nPath] = QtWidgets.QPushButton(self.horizontalLayoutWidget[nPath])
            self.BrowseButton[nPath].setObjectName("BrowseButton_"+str(nPath))
            self.horizontalLayout[nPath].addWidget(self.BrowseButton[nPath]) 

        self.retranslateUi()
        self.connect_signals_init()
        self.OKCancelButtonBox.accepted.connect(self.accept)
        self.OKCancelButtonBox.rejected.connect(self.reject)
        QtCore.QMetaObject.connectSlotsByName(self)

# ---------------------------------------------------------------------
    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("Dialog", "Dialog"))
        self.Label[0].setText(_translate("Dialog", "Folder to process:"))
        self.Label[1].setText(_translate("Dialog", "Folder to save all crops collectively to:"))
        for nPath in range(2):            
            self.BrowseButton[nPath].setText(_translate("Dialog", "Browse"))

# ---------------------------------------------------------------------
    def connect_signals_init(self):
        for nPath in range(2):
            self.BrowseButton[nPath].clicked.connect(lambda _, nPath=nPath: self.BrowseDirectory(nPath=nPath))
            self.PathLineEdit[nPath].textChanged.connect(lambda _, nPath=nPath: self.updatePath(nPath=nPath))

# ---------------------------------------------------------------------
    def BrowseDirectory(self, nPath):      
        settings = io.load_user_settings()
        try:
            if nPath == 0:
                directory = settings["Batchcrop"]["PWD BATCH"]
            elif nPath == 1:
                directory = settings["Batchcrop"]["PWD SAVE"]
        except:
            directory = None
            pass  
        try:
            filepath = io.FileDialog(directory, "Select a folder",).openFolderNameDialog()  
        except:
            filepath = io.FileDialog(None, "Select a folder",).openFolderNameDialog() 
        

        if os.path.isfile(filepath):
            folderpath = os.path.dirname(filepath)
        elif os.path.isdir(filepath):
            folderpath = filepath
        else:
            return
        if len(folderpath) > 0:
            if nPath == 0:
                settings["Batchcrop"]["PWD BATCH"] = folderpath
            elif nPath == 1:
                settings["Batchcrop"]["PWD SAVE"] = folderpath
            io.save_user_settings(settings)

        if nPath == 0:
            self.BatchCropPath = folderpath
        elif nPath == 1:
            self.BatchSavePath = folderpath

        # update self.PathLineEdit's to display the path
        self.PathLineEdit[nPath].setText(folderpath)

# ---------------------------------------------------------------------
    def updatePath(self, nPath):
        filepath = self.PathLineEdit[nPath].text()
        if filepath is None:
            # self.BatchCropPath = ''
            return

        # checks if path is a file
        if os.path.isfile(filepath):
            folderpath = os.path.dirname(filepath)
        elif os.path.isdir(filepath):
            folderpath = filepath
        else: # do nothing, this will execute while a path is being written in the lineEdit
            # self.BatchCropPath = ''
            return

        if len(folderpath) > 0:
            settings = io.load_user_settings()
            if nPath == 0:
                settings["Batchcrop"]["PWD BATCH"] = folderpath
            elif nPath == 1:
                settings["Batchcrop"]["PWD SAVE"] = folderpath
            io.save_user_settings(settings)

        if nPath == 0:
            self.BatchCropPath = folderpath
        elif nPath == 1:
            self.BatchSavePath = folderpath
        
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
class MultiImageSeriesDialog(QtWidgets.QDialog):
    """ The dialog allowing the user to add multiple image series """
# ---------------------------------------------------------------------
    def __init__(self, parent=None):
        super().__init__(parent)
        self.window = parent
        self.setWindowTitle("Choose image series to show together")
        # self.resize(300, 0)
        self.resize(481, 436)

        # how many series do we allow? take 7 for now
        self.numSeries = 7        

        # initialize variables
        self.MultiImageSeries_folderpath  = ['']*self.numSeries
        self.MultiImageSeries_description = ['']*self.numSeries

        # initialize list of QtItems
        self.horizontalLayoutWidget = ['']*self.numSeries
        self.horizontalLayout       = ['']*self.numSeries
        self.NumberLabel            = ['']*self.numSeries
        self.PathLineEdit           = ['']*self.numSeries
        self.BrowseButton           = ['']*self.numSeries
        self.DescriptionLabel       = ['']*self.numSeries
        self.DescriptionLineEdit    = ['']*self.numSeries

        # load settings
        self.settings = io.load_user_settings()

        # OK and Cancel buttons
        self.OKCancelButtonBox = QtWidgets.QDialogButtonBox(self)
        self.OKCancelButtonBox.setGeometry(QtCore.QRect(40, 390, 341, 32))
        self.OKCancelButtonBox.setOrientation(QtCore.Qt.Horizontal)
        self.OKCancelButtonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.OKCancelButtonBox.setObjectName("OKCancelButtonBox")

        # group box to contain several h layouts within a v layout
        # add up to 10 horizontalLayouts like these
        self.groupBox = QtWidgets.QGroupBox(self)
        self.groupBox.setGeometry(QtCore.QRect(10, 10, 463, 380))
        self.groupBox.setObjectName("groupBox")

        # iterate through numSeries
        for nSeries in range(self.numSeries):
            
            self.horizontalLayoutWidget[nSeries] = QtWidgets.QWidget(self.groupBox)
            self.horizontalLayoutWidget[nSeries].setGeometry(QtCore.QRect(10, 30+nSeries*40, 441, 30))
            self.horizontalLayoutWidget[nSeries].setObjectName("horizontalLayoutWidget_"+str(nSeries))

            self.horizontalLayout[nSeries] = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget[nSeries])
            self.horizontalLayout[nSeries].setContentsMargins(0, 0, 0, 0)
            self.horizontalLayout[nSeries].setObjectName("horizontalLayout_"+str(nSeries))
            
            self.NumberLabel[nSeries] = QtWidgets.QLabel(self.horizontalLayoutWidget[nSeries])
            self.NumberLabel[nSeries].setObjectName("NumberLabel_"+str(nSeries))
            self.horizontalLayout[nSeries].addWidget(self.NumberLabel[nSeries])
            
            self.PathLineEdit[nSeries] = QtWidgets.QLineEdit(self.horizontalLayoutWidget[nSeries])
            self.PathLineEdit[nSeries].setObjectName("PathLineEdit_"+str(nSeries))
            self.horizontalLayout[nSeries].addWidget(self.PathLineEdit[nSeries])
            try:
                self.PathLineEdit[nSeries].setText(self.settings["crop"]["PWD MultiImageSeries"]["path"+str(nSeries)])
                self.MultiImageSeries_folderpath[nSeries] = self.settings["crop"]["PWD MultiImageSeries"]["path"+str(nSeries)]
            except Exception:
                pass
            pass    
            
            self.BrowseButton[nSeries] = QtWidgets.QPushButton(self.horizontalLayoutWidget[nSeries])
            self.BrowseButton[nSeries].setObjectName("BrowseButton_"+str(nSeries))
            self.horizontalLayout[nSeries].addWidget(self.BrowseButton[nSeries])  
            
            self.DescriptionLabel[nSeries] = QtWidgets.QLabel(self.horizontalLayoutWidget[nSeries])
            self.DescriptionLabel[nSeries].setObjectName("DescriptionLabel_"+str(nSeries))
            self.horizontalLayout[nSeries].addWidget(self.DescriptionLabel[nSeries])
            
            self.DescriptionLineEdit[nSeries] = QtWidgets.QLineEdit(self.horizontalLayoutWidget[nSeries])
            self.DescriptionLineEdit[nSeries].setObjectName("DescriptionLineEdit_"+str(nSeries))
            self.horizontalLayout[nSeries].addWidget(self.DescriptionLineEdit[nSeries])
            try:
                self.DescriptionLineEdit[nSeries].setText(self.settings["crop"]["Description MultiImageSeries"]["descr"+str(nSeries)])
                self.MultiImageSeries_description[nSeries] = self.settings["crop"]["Description MultiImageSeries"]["descr"+str(nSeries)]
            except Exception:
                pass
            pass

        self.retranslateUi()
        self.connect_signals_init()
        self.OKCancelButtonBox.accepted.connect(self.getValues)
        self.OKCancelButtonBox.rejected.connect(self.reject)
        QtCore.QMetaObject.connectSlotsByName(self)

# ---------------------------------------------------------------------
    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("Dialog", "Dialog"))
        self.groupBox.setTitle(_translate("Dialog", "Choose image series to show together"))
        for nSeries in range(self.numSeries):
            self.NumberLabel[nSeries].setText(_translate("Dialog", str(nSeries)+":"))
            self.BrowseButton[nSeries].setText(_translate("Dialog", "Browse"))
            self.DescriptionLabel[nSeries].setText(_translate("Dialog", "label:"))

# ---------------------------------------------------------------------
    def connect_signals_init(self):
        for nSeries in range(self.numSeries):
            self.BrowseButton[nSeries].clicked.connect(lambda _, nSeries=nSeries: self.BrowseDirectory(nSeries=nSeries))
            self.PathLineEdit[nSeries].textChanged.connect(lambda _, nSeries=nSeries: self.updatePath(nSeries=nSeries))
            self.DescriptionLineEdit[nSeries].textChanged.connect(lambda _, nSeries=nSeries: self.updateLabel(nSeries=nSeries))

# ---------------------------------------------------------------------
    def updatePath(self, nSeries):
        filepath = self.PathLineEdit[nSeries].text()
        if filepath is None:
            self.MultiImageSeries_folderpath[nSeries] = ''
            return

        # checks if path is a file
        if os.path.isfile(filepath):
            folderpath = os.path.dirname(filepath)
        elif os.path.isdir(filepath):
            folderpath = filepath
        else: # do nothing, this will execute while a path is being written in the lineEdit
            self.MultiImageSeries_folderpath[nSeries] = ''
            return

        # save current folderpath for the next time when the dialog is opened
        settings = io.load_user_settings()
        if folderpath is not None:
            settings["crop"]["PWD"] = folderpath
        io.save_user_settings(settings)
        self.MultiImageSeries_folderpath[nSeries] = folderpath

# ---------------------------------------------------------------------
    def updateLabel(self, nSeries):
        self.MultiImageSeries_description[nSeries] = self.DescriptionLineEdit[nSeries].text()

# ---------------------------------------------------------------------
    def BrowseDirectory(self, nSeries):
        settings = io.load_user_settings()
        try:
            folderpath = os.path.dirname(settings["crop"]["PWD"])
            # filepath = io.FileDialog(folderpath, "open a tif file stack",
            #                      "Tif File (*.tif *.tiff)").openFileNameDialog()
            filepath = io.FileDialog(folderpath, "Select a folder containing .tif files",).openFolderNameDialog()
        except Exception as e:
            print(e)
            # filepath = io.FileDialog(None, "open a tif file stack",
            #                      "Tif File (*.tif *.tiff)").openFileNameDialog()
            filepath = io.FileDialog(None, "Select a folder containing .tif files",).openFolderNameDialog()
            pass
        if os.path.isfile(filepath):
            folderpath = os.path.dirname(filepath)
        else:
            folderpath = filepath
        # save current folderpath for the next time when the dialog is opened
        settings = io.load_user_settings()
        if folderpath is not None:
            settings["crop"]["PWD"] = folderpath
        io.save_user_settings(settings)
        self.MultiImageSeries_folderpath[nSeries] = folderpath

        # update self.PathLineEdit's to display the path
        self.PathLineEdit[nSeries].setText(self.MultiImageSeries_folderpath[nSeries])
        
# ---------------------------------------------------------------------
    def getValues(self):
        # If we accept the inputs as they are, remove all empty paths and their labels (if there are any)    
        filter_list = [element!='' for element in self.MultiImageSeries_folderpath]
        self.MultiImageSeries_description = list(itertools.compress(
            self.MultiImageSeries_description, filter_list))
        self.MultiImageSeries_folderpath = list(itertools.compress(
            self.MultiImageSeries_folderpath, filter_list))
        settings = io.load_user_settings()
        if self.MultiImageSeries_folderpath is not None:
            settings["crop"]["PWD MultiImageSeries"] = {}
            settings["crop"]["Description MultiImageSeries"] = {}
            for nSeries in range(len(self.MultiImageSeries_folderpath)):
                settings["crop"]["PWD MultiImageSeries"]["path"+str(nSeries)] = self.MultiImageSeries_folderpath[nSeries]
                settings["crop"]["Description MultiImageSeries"]["descr"+str(nSeries)] = self.MultiImageSeries_description[nSeries]
        io.save_user_settings(settings)
        self.accept()
        
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# when clicking a spinbox button, pyqt thinks that the duration of a click is
# long enough to execute the pressed button more than once. Result: The counter jumps
# by 2 instead of the desired 1. Workaround: The class CustomStyle sets a very long
# auto-repeat to avoid this. Source: https://stackoverflow.com/questions/40746350/why-qspinbox-jumps-twice-the-step-value
class CustomStyle(QtWidgets.QProxyStyle):
    def styleHint(self, hint, option=None, widget=None, returnData=None):
        if hint == QtWidgets.QStyle.SH_SpinBox_KeyPressAutoRepeatRate:
            return 10**10
        elif hint == QtWidgets.QStyle.SH_SpinBox_ClickAutoRepeatRate:
            return 10**10
        elif hint == QtWidgets.QStyle.SH_SpinBox_ClickAutoRepeatThreshold:
            # You can use only this condition to avoid the auto-repeat,
            # but better safe than sorry ;-)
            return 10**10
        else:
            return super().styleHint(hint, option, widget, returnData)

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
class NapariTabs(QtWidgets.QWidget):
# ---------------------------------------------------------------------
    def __init__(self, viewer):
        super().__init__()
        self.ui = crop_images_ui.Ui_Form()
        self.ui.setupUi(self)
        self.viewer = viewer

        # set defaults
        self.use_current_image_path = False
        self.current_image_path     = None
        self.MaxNumColors           = 4
        self.defaultShape           = np.array([[10, 0], [80, 0], [80, 50], [10, 50]])
        self.defaultLine            = np.array([[10, 120], [50, 150]])
        self.color_list = np.array([[0, 158, 115], [204, 121, 167], [86, 180, 233], [230, 159, 0], [240, 228, 66], 
            [0, 114, 178], [213, 94, 0], [255, 255, 255]]) / 255 # color palette taken from https://www.nature.com/articles/nmeth.1618        

        # number of colors
        self.ui.numColorsCbox.setCurrentText("2")
        self.ui.numColorsCbox.currentIndexChanged.connect(self.change_num_colors)
        # frame number
        self.frame_start = self.ui.frameStartSbox.value()
        self.frame_end = self.ui.frameEndBox.value()        
        if self.frame_end == -1:
            self.frame_end = None
        # image rotating and shifting
        self.ui.angleSpinBox.valueChanged.connect(self.ShiftRotateImage)
        self.ui.xShiftSpinBox.valueChanged.connect(self.ShiftRotateImage)
        self.ui.yShiftSpinBox.valueChanged.connect(self.ShiftRotateImage)
        self.ui.angleSpinBox.setStyle(CustomStyle())
        self.ui.xShiftSpinBox.setStyle(CustomStyle())
        self.ui.yShiftSpinBox.setStyle(CustomStyle())
        self.ui.angleSpinBox.setKeyboardTracking(False) # only update when editing the field is done
        self.ui.xShiftSpinBox.setKeyboardTracking(False)
        self.ui.yShiftSpinBox.setKeyboardTracking(False)
        # image preprocessing
        self.ui.bkgSubstractionCheckBox.stateChanged.connect(self.toggle_bkg_subtraction)
        # processing
        self.ui.loadImageBtn.clicked.connect(self.call_load_img_seq)
        self.ui.defaultFramesBtn.clicked.connect(self.set_default_frame_num)
        self.ui.cropSelectedBtn.clicked.connect(self.crop)
        self.ui.loadImageJroisBtn.clicked.connect(self.load_imageJ_rois)
        self.ui.saveImageJroisBtn.clicked.connect(self.save_imageJ_rois)
        # load multiple image series
        self.MultiImageSeriesFlag = False
        self.ui.ShowAllLayersBtn.clicked.connect(lambda: self.ToggleAllLayers('show'))
        self.ui.HideAllLayersBtn.clicked.connect(lambda: self.ToggleAllLayers('hide'))
        self.ui.MultiImageSeriesPushButton.clicked.connect(self.call_MultiImageSeries)
        self.ChannelVisbility = self.ui.ToggleChannelVisbility.value()
        self.ui.ToggleChannelVisbility.valueChanged.connect(self.toggleChannelVisibility)
        # initialize and update text next to ROIs
        self.ObtainLayerParameters()
        self.ui.UpdateTextBtn.clicked.connect(self.UpdateText)
        # looping through FOVs
        self.ui.FOVLineEdit.textChanged.connect(self.SearchFOVs)
        self.ui.FOVSpinBox.setStyle(CustomStyle())
        self.ui.FOVSpinBox.valueChanged.connect(self.SwitchFOV)
        self.ui.FOVSpinBox.setKeyboardTracking(False)
        # button to toggle batch processing
        self.ui.BatchProcessBtn.clicked.connect(self.call_BatchProcessing)

        # finally, get the window for a line profile
        self.LPwin = LineProfileWindow()
        self.viewer.dims.events.current_step.connect(self.updateSlider)
        # self.LPwin.show() # dont show it yet, only if we actually have something to plot


        # upon clicking somewhere in the viewer, decide if we clicked into
        # an image or in a shapes layer. Depending on that, make the layer we clicked
        # into the active one
        @viewer.mouse_drag_callbacks.append
        def SwitchLayerUponMouseClick(self, event):
            # at startup, we dont have any layers yet. in this case, do nothing
            # !!! here, self is already self.viewer
            if (not self.layers) or (not self._active_layer):
                return
            selected = self._active_layer.name
            # if the selected layer, is the profile layer, we dont do this
            if 'profile' in selected.lower():
                return
            # if we're currently drawing a shape, also dont trigger it
            if hasattr(self._active_layer, 'nshapes'): # only works for shapes layers
                if self._active_layer.mode != 'select':
                    return
            # get the mouse click coordinates. We have to get that
            # from the currently active layer
            mouseCoordinates = [self._active_layer.coordinates]
            if len(mouseCoordinates[0])>2:  # omit z component on position 0 if present
                mouseCoordinates[0] = mouseCoordinates[0][1:]
            # loop through shape layers. If the mouse click was
            # in any of the shapes, select this layer. 
            # Otherwise, select the uppermost image layer
            for nLayer in range(len(self.layers)):
                if not hasattr(self.layers[nLayer], 'nshapes'):
                    continue
                if len(self.layers[nLayer].data) == 0:
                    continue
                if self.layers[nLayer].data[0].shape[0] != 4:
                    continue # is not a rectangle
                
                # now see if we find any shape which contains mouseCoordinates
                for nShape in range(len(self.layers[nLayer].data)):
                    polygon = self.layers[nLayer].data[nShape]
                    path = mpltPath.Path(polygon)
                    if path.contains_points(mouseCoordinates):
                        # deselect what was selected before
                        self.layers[selected].events.deselect()
                        self.layers[selected].selected = False
                        # select where we clicked
                        self.layers[nLayer].events.select()
                        self.layers[nLayer].selected = True
                        self._active_layer = self.layers[nLayer]
                        return
            # if the function didnt return so far, we didnt find 
            # any shape which has the mouse coord inside
            # in thise case, check if we had an image layer selected before
            # if yes, just stay there -> do nothing
            # if no, change to the lowest layer
            if hasattr(self.layers[selected], 'nshapes'): # if it was a shape layer
                self.layers[selected].events.deselect()
                self.layers[selected].selected = False
                self.layers[0].events.select()
                self.layers[0].selected = True
                self._active_layer = self.layers[0]


        # when changing layers, update the x-y and angle boxes
        @viewer.events.active_layer.connect
        def UpdateXYShiftRotationAngleUponSwitchingLayer(event):
            # at startup, we dont have any layers yet. in this case, do nothing
            if (not self.viewer.layers) or (not hasattr(self, 'numLayers') or (not self.viewer._active_layer)):
                return
            # check if we have currently as many layers as written in self.numLayers
            # if not, we are likely just building up the viewer after changing the FOV
            if hasattr(self, 'numLayers'):
                numLayers = 0
                for nLayer in range(len(self.viewer.layers)):
                    if not hasattr( self.viewer.layers[nLayer], 'nshapes'):
                        numLayers += 1
                if numLayers != self.numLayers:
                    return
            else:
                return
            self.GetRelatedSelectedLayers(display=False)
            if self.series2treat is not None:
                ShiftColorsIndividually = False
                if self.numLayers == self.numColors:
                    ShiftColorsIndividually = True # if there is only one image series, move the layers independently
                if ShiftColorsIndividually:
                    currentLayer = self.viewer.active_layer.name
                    layerNames = [''] * self.numLayers
                    for nLayer in range(self.numLayers):
                        layerNames[nLayer] = self.viewer.layers[nLayer].name
                    currentLayer = layerNames.index(currentLayer)
                    index = currentLayer
                else:
                    # it doesnt matter which color we take here since both colors have the same
                    # x-shift, y-shift and angle value. So let's just take the first one
                    index = 0
                # now we update what is shown in the spin boxes. First, however, block signals 
                # in order to prevent triggering self.ShiftRotateImage 
                self.ui.xShiftSpinBox.blockSignals(True)
                self.ui.yShiftSpinBox.blockSignals(True)
                self.ui.angleSpinBox.blockSignals(True)
                if not hasattr(self, 'xShift'):
                    a=1
                self.ui.xShiftSpinBox.setValue(self.xShift[self.series2treat][index])
                self.ui.yShiftSpinBox.setValue(self.yShift[self.series2treat][index])
                self.ui.angleSpinBox.setValue(self.RotationAngle[self.series2treat][index]) # when updating the angle, we can cal self.ShiftRotateImage
                self.ui.xShiftSpinBox.blockSignals(False) # unblock the signals
                self.ui.yShiftSpinBox.blockSignals(False)
                self.ui.angleSpinBox.blockSignals(False)


            # also update the help to show the folder path
            if not self.viewer._active_layer:
                return
            activeLayerName = self.viewer._active_layer.name
            for nLayer in range(len(self.viewer.layers)):
                if activeLayerName == self.viewer.layers[nLayer].name:
                    break
            if nLayer > self.numLayers-1:
                nLayer = self.numLayers-1
            nSeries = int( np.floor( nLayer/self.numColors ) )
            self.ui.FileDirectoryLabel.setText(
                os.path.dirname('Current directory: '+self.image_meta[nSeries]['filenames'][0])
            )

# ---------------------------------------------------------------------
    def updateSlider(self, dummy):
        # if line profiles are shown, also update those
        if self.LPwin.isVisible():
            self.profile_line()

# ---------------------------------------------------------------------
    def SwitchFOV(self):
        nFOV = self.ui.FOVSpinBox.value()
        if nFOV < 0:
            nFOV = 0
        tag = self.ui.FOVLineEdit.text()
        tag = tag.lower() # make it case-insensitive
        # we are quite liberal in matching the folder with the search str
        # as long as it contains the tag and it contains nFOV, we're good
        # from the available paths, see which one fits best
        numPaths = len(self.FOVpaths)
        paths = [''] * numPaths
        for nPath in range(numPaths):
            if self.FOVpaths[nPath] is None:
                continue
            if len(self.FOVpaths[nPath]) == 0:
                continue
            subdirs = []
            nums = []#np.zeros(len(self.FOVpaths[nPath]))
            for subpath in range(len(self.FOVpaths[nPath])):
                basename = os.path.basename(self.FOVpaths[nPath][subpath])
                basename = basename.lower()
                if '_analysis' in basename:
                    continue
                if tag in basename:
                    subdirs.append( basename )
                    try:
                        nums.append( int( re.findall(r'\d+', subdirs[-1].lower() )[0] ) )
                    except:
                        continue
            # check if we have subdirs at all. If not, the path doesnt contain subfolders with keyword 'tag'
            if len(subdirs) == 0:
                continue
            # check if we have duplicates. If so, print this to notify the user but dont do anything else
            sort_index = np.argsort(nums)
            nums = np.sort(nums, axis=None)
            duplicate = nums[:-1][nums[1:] == nums[:-1]]
            if len(duplicate) > 0:
                print('Duplicate FOV found in '+os.path.dirname(self.FOVpaths[nPath][0])+'. Returning.')
                return         
            # check if we have less subdirs then expected. If so, return an empty path here
            if nFOV >= len(sort_index):
                continue
            paths[nPath] = os.path.join(
                os.path.dirname(self.FOVpaths[nPath][0]), 
                subdirs[sort_index[nFOV]] )
        # give user feedback which paths are gonna be loaded
        if all( [len(path)==0 for path in paths] ):
            return

        # write those paths into the settings.yaml file to have at the next startup
        settings = io.load_user_settings()
        for nPath in range(numPaths):
            if len(paths[nPath]) > 0:
                settings["crop"]["PWD MultiImageSeries"]["path"+str(nPath)] = paths[nPath]
        io.save_user_settings(settings)

        print('Loading the following set of FOVs:')
        for nPath in range(numPaths):
            print('['+str(int(nPath+1))+'] '+paths[nPath])                        

        # now call load_img_seq with this set of paths
        self.load_img_seq(path_in=paths)
        
# ---------------------------------------------------------------------
    def SearchFOVs(self):
        # check if we already have one or several paths given. If not, do nothing
        if not hasattr(self, 'numSeries'):
            return
        tag = self.ui.FOVLineEdit.text()
        tag = tag.lower() # make it case-insensitive
        # get current path(s)
        if self.MultiImageSeriesFlag: # if we load multiple image series
            paths = self.MultiImageSeries_folderpath
        else:            
            paths = [self.image_meta[0].get("folderpath")]
        # loop through each path. look one level higher if we find folders which contain
        # the tag given by the user
        numPaths = len(paths)
        self.numFOVs = [0] * numPaths
        self.FOVpaths = [''] * numPaths
        for nPath in range(numPaths):
            parent = os.path.dirname(paths[nPath])
            list_subfolders = [f.path for f in os.scandir(parent) if f.is_dir()]
            self.FOVpaths[nPath] = [s for s in list_subfolders if tag in s.lower()]
            self.numFOVs[nPath] = len(self.FOVpaths[nPath])
        if all(not s for s in self.numFOVs):
            self.MaxNumFOV = 0 # if we didnt find any, set the maximum spinbox value to 1 and that's it
        else:
            nonzeroIndex = [i for i, j in enumerate(self.numFOVs) if j > 1]
            if len(nonzeroIndex) > 0:
                self.MaxNumFOV = min([self.numFOVs[i] for i in nonzeroIndex])
            else:
                self.MaxNumFOV = 0
        # set the maximum value of the FOVQSpinBox to self.MaxNumFOV
        self.ui.FOVCountLabel.setText("/"+str(int(self.MaxNumFOV-1)))
        self.ui.FOVSpinBox.setMaximum(self.MaxNumFOV-1)         

# ---------------------------------------------------------------------
    def UpdateText(self):
        # loop through all shape layers, give each shape in the layer the same label
        isShapeLayer = self.FindAllShapeLayers()
        numShapeLayers = len(isShapeLayer)
        # get all current shape layers, then remove them and add them with updated properties
        shapeLayers = []
        for nShapeLayer in range(numShapeLayers):
            shapeLayers.append(self.viewer.layers[isShapeLayer[nShapeLayer]])
        for nShapeLayer in reversed(range(numShapeLayers)):
            self.viewer.layers.remove(isShapeLayer[nShapeLayer]) # removes the layer at index isShapeLayer[nShapeLayer]
        
        # now add them back with updated properties
        currentTime  = self.viewer.dims.point[0]
        for nShapeLayer in range(numShapeLayers):
            data       = shapeLayers[nShapeLayer].data
            if len(data)==0:
                continue
            # remove column with all 0 entries
            data       = [data[i][:,(data[i]!=0).any(axis=0)] for i in range(len(data))]
            # alternatively, the first column holds the current time step, remove this as well
            data       = [data[i][:,(data[i]!=currentTime).any(axis=0)] for i in range(len(data))]
            shape_type = shapeLayers[nShapeLayer].shape_type
            name       = shapeLayers[nShapeLayer].name
            edge_color = shapeLayers[nShapeLayer].edge_color
            edge_width = self.edge_width
            opacity    = self.opacity
            text       = self.text_kwargs
            properties = {}
            properties['label'] = np.array([name] * len(data))
            
            self.viewer.add_shapes(data, shape_type=shape_type, name=name,
                edge_color=edge_color, edge_width=edge_width, opacity=opacity, 
                text=text, properties=properties)
            self.viewer.layers[name].mode = 'select'


# ---------------------------------------------------------------------
    def ObtainLayerParameters(self):
    # specify the display parameters for the text
        self.text_kwargs = {
        'text': '{label}',
        'size': 8,
        'color': 'white',
        'anchor': 'upper_left',
        'translation': [0,0]
        }
        self.edge_width = 5
        self.opacity = 0.2

# ---------------------------------------------------------------------
    def ShiftRotateImage(self):
        # use the built-in translate/rotate method of napari viewer        
        self.GetRelatedSelectedLayers()
        if self.series2treat is not None:
            ShiftColorsIndividually = False
            if self.numLayers == self.numColors:
                ShiftColorsIndividually = True # if there is only one image series, move the layers independently
            if ShiftColorsIndividually:
                currentLayer = self.viewer.active_layer.name
                layerNames = [''] * self.numLayers
                for nLayer in range(self.numLayers):
                    layerNames[nLayer] = self.viewer.layers[nLayer].name
                currentLayer = layerNames.index(currentLayer)
                self.xShift[self.series2treat][currentLayer] = self.ui.xShiftSpinBox.value()
                self.yShift[self.series2treat][currentLayer] = self.ui.yShiftSpinBox.value()
                self.RotationAngle[self.series2treat][currentLayer] = self.ui.angleSpinBox.value()
            else:
                if not hasattr(self, 'xShift'):
                    self.RotationAngle = [0] * self.numSeries
                    self.xShift        = [0] * self.numSeries
                    self.yShift        = [0] * self.numSeries
                    for nSeries in range(self.numSeries):
                        self.RotationAngle[nSeries] = [0] * self.MaxNumColors
                        self.xShift[nSeries]        = [0] * self.MaxNumColors
                        self.yShift[nSeries]        = [0] * self.MaxNumColors
                for nColor in range(self.numColors):
                    self.xShift[self.series2treat][nColor] = self.ui.xShiftSpinBox.value()
                    self.yShift[self.series2treat][nColor] = self.ui.yShiftSpinBox.value()
                    self.RotationAngle[self.series2treat][nColor] = self.ui.angleSpinBox.value()
            
            # save the current x-y shift into a yaml file
            folderpath = self.image_meta[self.series2treat]['folderpath']
            self.yamlFileName = os.path.join(folderpath, 'shift.yaml')                           
            self.LoadShiftYamlFile()

            if not hasattr(self.shift_yaml, 'angle'):
                self.shift_yaml["angle"] = {}
            for nColor in range(self.numColors):
                self.shift_yaml["x"]["col"+str(int(nColor))] = self.xShift[self.series2treat][nColor]
                self.shift_yaml["y"]["col"+str(int(nColor))] = self.yShift[self.series2treat][nColor]
                self.shift_yaml["angle"]["col"+str(int(nColor))] = self.RotationAngle[self.series2treat][nColor]
        
            # save the yaml file
            self.shift_yaml = io.to_dict_walk(self.shift_yaml)
            os.makedirs(os.path.dirname(self.yamlFileName), exist_ok=True)
            with open(self.yamlFileName, "w") as shift_file:
                yaml.dump(dict(self.shift_yaml), shift_file, default_flow_style=False)

            # save current settings: selected layer, all contrast ranges and 
            # contrast min/max, as well as the current frame
            # we then call self.load_img_seq(), which will read the updated shifts + angles
            # from the yaml file we just saved such that the image is correctly rotated
            # see if we already have a value for angle_old. If not, this is the first rotation
            # we apply. Do it also when angle=0. Otherwise, we can check if the 
            # new angle value is different from the old one. Only execute the rotation
            # function if this is true.
            if not hasattr(self, 'angle_old'):
                self.angle_old = [[n for n in m] for m in self.RotationAngle]
                angleDiff = np.array((1,1))
            else:
                angleDiff = np.asarray(self.RotationAngle[self.series2treat]) - np.asarray(self.angle_old[self.series2treat])
            self.angle_old = [[n for n in m] for m in self.RotationAngle]
            if sum(abs(angleDiff))!=0:                
                self.GetCurrentDisplaySettings()
                self.use_current_image_path = True
                save_series2treat = self.series2treat
                self.load_img_seq(omitROIlayer=True)
                self.series2treat = save_series2treat
                # apply settings as before
                self.ApplyDisplaySettings()  

            # shift
            for nColor in range(self.numColors):
                # [z, y, x]
                translationVector = np.array( [0, self.yShift[self.series2treat][nColor], self.xShift[self.series2treat][nColor]] )
                self.viewer.layers[self.series2treat*self.numColors+nColor].translate = translationVector

            # if line profiles are shown, also update those
            if self.LPwin.isVisible():
                self.profile_line()

# ---------------------------------------------------------------------
    def LoadShiftYamlFile(self):
        if os.path.isfile(self.yamlFileName): # if it exists, load it. assume there is only one such file
            try:
                yaml_file = open(self.yamlFileName, "r")
            except FileNotFoundError:
                return self.MakeShiftYamlFile()
            try:
                self.shift_yaml = yaml.load(yaml_file, Loader=yaml.FullLoader)
                yaml_file.close()
                # self.shift_yaml = io.AutoDict(self.shift_yaml)
            except:
                self.MakeShiftYamlFile()
        else: # if is doesnt exist, we create all structures from scratch                
            self.MakeShiftYamlFile() 

# ---------------------------------------------------------------------
    def MakeShiftYamlFile(self):
        self.shift_yaml = {}
        self.shift_yaml["x"] = {}
        self.shift_yaml["y"] = {}
        self.shift_yaml["angle"] = {}
        index = self.series2treat
        if not index:
            index = 0
        for nColor in range(self.numColors):
            self.shift_yaml["x"]["col"+str(int(nColor))] = self.xShift[index][nColor]
            self.shift_yaml["y"]["col"+str(int(nColor))] = self.yShift[index][nColor]
            self.shift_yaml["angle"]["col"+str(int(nColor))] = self.RotationAngle[index][nColor]        

# ---------------------------------------------------------------------
    def GetCurrentDisplaySettings(self):
        self.currentTime  = self.viewer.dims.point[0]
        self.currentLayer = self.viewer.active_layer.name
        layerNames = [''] * self.numLayers
        self.visibility = [False] * self.numLayers
        for nLayer in range(self.numLayers): # loops only through non-shape layers
            layerNames[nLayer] = self.viewer.layers[nLayer].name
            self.visibility[nLayer] = self.viewer.layers[nLayer].visible
 
        # now loop through shape layers
        isShapeLayer = self.FindAllShapeLayers()
        numShapeLayers = len(isShapeLayer)
        shapeLayers = []
        for nShapeLayer in range(numShapeLayers):
            shapeLayers.append(self.viewer.layers[isShapeLayer[nShapeLayer]])        
        self.shapeLayerData = []
        for nShapeLayer in range(numShapeLayers):
            layer = {}
            data       = shapeLayers[nShapeLayer].data
            if len(data)==0:
                continue
            # remove column with all 0 entries
            data       = [data[i][:,(data[i]!=0).any(axis=0)] for i in range(len(data))]
            # alternatively, the first column holds the current time step, remove this as well
            data       = [data[i][:,(data[i]!=self.currentTime).any(axis=0)] for i in range(len(data))]
            shape_type = shapeLayers[nShapeLayer].shape_type
            name       = shapeLayers[nShapeLayer].name
            edge_color = shapeLayers[nShapeLayer].edge_color
            edge_width = self.edge_width
            opacity    = self.opacity
            text       = self.text_kwargs
            properties = {}
            properties['label'] = np.array([name] * len(data))
            layer['data']       = data
            layer['shape_type'] = shape_type
            layer['name']       = name
            layer['edge_color'] = edge_color
            layer['edge_width'] = edge_width
            layer['opacity']    = opacity
            layer['text']       = text
            layer['properties'] = properties
            self.shapeLayerData.append(layer)

        self.currentLayer = layerNames.index(self.currentLayer)
        self.currentContrastRanges = [0] * self.numLayers
        self.currentContrast       = [0] * self.numLayers
        for nLayer in range(self.numLayers): # for image layers
            self.currentContrastRanges[nLayer] = self.viewer.layers[nLayer].contrast_limits_range
            self.currentContrast[nLayer]       = self.viewer.layers[nLayer].contrast_limits

# ---------------------------------------------------------------------
    def ApplyDisplaySettings(self):
        # first add back all shape layers
        for nLayer in range(len(self.shapeLayerData)):
            data       = self.shapeLayerData[nLayer]['data'] 
            shape_type = self.shapeLayerData[nLayer]['shape_type'] 
            name       = self.shapeLayerData[nLayer]['name']  
            edge_color = self.shapeLayerData[nLayer]['edge_color'] 
            edge_width = self.shapeLayerData[nLayer]['edge_width'] 
            opacity    = self.shapeLayerData[nLayer]['opacity']
            text       = self.shapeLayerData[nLayer]['text']  
            properties = self.shapeLayerData[nLayer]['properties']
            self.viewer.add_shapes(data, shape_type=shape_type, name=name,
                edge_color=edge_color, edge_width=edge_width, opacity=opacity, 
                text=text, properties=properties)
            self.viewer.layers[name].mode = 'select'
        del self.shapeLayerData # free up some memory
        
        self.viewer.dims.set_point(0, self.currentTime)
        # deselect what is currently selected
        selected = self.viewer.active_layer.name
        self.viewer.layers[selected].events.deselect()
        self.viewer.layers[selected].selected = False  
        # select what was selected before      
        self.viewer.layers[self.currentLayer].events.select()
        self.viewer.layers[self.currentLayer].selected = True 
        self.viewer._active_layer = self.viewer.layers[self.currentLayer]
        for nLayer in range(self.numLayers): # the last entry in layers is the ROI
            self.viewer.layers[nLayer].contrast_limits_range = self.currentContrastRanges[nLayer]
            self.viewer.layers[nLayer].contrast_limits       = self.currentContrast[nLayer]
            self.viewer.layers[nLayer].visible               = self.visibility[nLayer]
        del self.visibility

# ---------------------------------------------------------------------
    def GetRelatedSelectedLayers(self, display=True):
        layers2treat = [0] * self.numLayers
        for nLayer in range(self.numLayers): # the last entry in layers is the ROI
            layers2treat[nLayer] = self.viewer.layers[nLayer].selected
        layers2treat = [i for i, x in enumerate(layers2treat) if x]        
        if len(layers2treat)==0:
            self.series2treat = None
            if display:
                print('Select a layer different from the ROI layer.')
            return
        self.series2treat = int( np.floor(layers2treat[0] / self.numColors) )

# ---------------------------------------------------------------------
    def removeprefix(self, text, prefix):
        if text.startswith(prefix):
            return text[len(prefix):]
        return text

# ---------------------------------------------------------------------
    def ToggleAllLayers(self, mode):
        if mode == 'hide':
            SetVisibility = False
        elif mode == 'show':
            SetVisibility = True
        else:
            return    
        for nLayer in range(self.numLayers): # the last entry in layers is the ROI
            self.viewer.layers[nLayer].visible = SetVisibility
        if SetVisibility:
            self.FixContrast()
        
# ---------------------------------------------------------------------
    def toggleChannelVisibility(self):
        self.ChannelVisbility = self.ui.ToggleChannelVisbility.value()
        if self.ChannelVisbility > self.numColors:
            self.ChannelVisbility = self.numColors
            self.ui.ToggleChannelVisbility.setValue(self.ChannelVisbility)
        counter = 0
        # first set all visibilities to False
        for nLayer in range(self.numLayers): # the last entry in layers is the ROI
            self.viewer.layers[nLayer].visible = False
        start = self.ChannelVisbility-1
        stop  = self.numLayers
        step  = self.numColors
        for nLayer in range(start, stop, step): # the last entry in layers is the ROI
            self.viewer.layers[nLayer].visible = True
        self.FixContrast()

# ---------------------------------------------------------------------
    def call_BatchProcessing(self):
        d = BatchProcessingDialog()
        if d.exec_(): # will return true only if dialog is accepted (pressed 'ok')
            if not hasattr(d, 'BatchCropPath'):
                print('Select a folder to process. Returning.')
                return
            self.BatchCropPath = d.BatchCropPath
            if hasattr(d, 'BatchSavePath'):
                self.BatchSavePath = d.BatchSavePath
            self.BatchCrop_from_directory()

# ---------------------------------------------------------------------
    def call_MultiImageSeries(self):
        dialog = MultiImageSeriesDialog()
        self.MultiImageSeriesFlag = False # default
        if hasattr(self, 'RotationAngle'):
            delattr(self, 'RotationAngle')
            delattr(self, 'xShift')
            delattr(self, 'yShift')
        if dialog.exec_(): # will return true only if dialog is accepted (pressed 'ok')
            self.MultiImageSeriesFlag = True
            self.MultiImageSeries_folderpath  = dialog.MultiImageSeries_folderpath
            self.MultiImageSeries_description = dialog.MultiImageSeries_description
            self.CreateColormaps()            
            self.load_img_seq()
            if self.numLayers == self.numColors:
                self.ToggleAllLayers('show') # if there is only one series

# ---------------------------------------------------------------------
    def call_load_img_seq(self):
        self.MultiImageSeriesFlag = False
        if hasattr(self, 'RotationAngle'):
            delattr(self, 'RotationAngle')
            delattr(self, 'xShift')
            delattr(self, 'yShift')
        self.load_img_seq()

# ---------------------------------------------------------------------
    def FixContrast(self):
        for nLayer in range(self.numLayers): # the last entry in layers is the ROI
            self.viewer.layers[nLayer].contrast_limits_range = self.viewer.layers[nLayer].contrast_limits_range
            self.viewer.layers[nLayer].contrast_limits       = self.viewer.layers[nLayer].contrast_limits
                    
# ---------------------------------------------------------------------
    def CreateColormaps(self):        
        # there are only a specific number of colors given. In case we'd need more
        # just replicate the array. The colors will repeat!
        # replicate = np.ceil(self.numSeries*self.numColors/(len(self.color_list)/2))
        # self.color_list = np.tile(self.color_list,(replicate,1))
        # alternative: interpolate the list
        if hasattr(self, 'MultiImageSeries_folderpath'):
            self.numSeries = len(self.MultiImageSeries_folderpath)
        else: 
            self.numSeries = 1
        self.numColors = int( self.ui.numColorsCbox.currentText() )
        required_numColors = self.numSeries*self.numColors
        if len(self.color_list) < required_numColors:
            x = np.arange(0, len(self.color_list))
            r = interp1d(x, self.color_list[:,0])
            g = interp1d(x, self.color_list[:,1])
            b = interp1d(x, self.color_list[:,2])
            xint = np.linspace(0, len(self.color_list)-1, required_numColors, endpoint=True)
            xint = np.expand_dims(xint, axis=1)
            self.color_list = np.concatenate((r(xint), g(xint), b(xint)), axis=1)
        numColors = len(self.color_list)
        self.cmap = [0] * numColors
        controls = np.linspace(0., 1., 100)
        numInterpolationPoints = 100
        for nCmap in range(numColors):
            carray = [np.linspace(0, self.color_list[nCmap][0], numInterpolationPoints), 
                np.linspace(0, self.color_list[nCmap][1], numInterpolationPoints), 
                np.linspace(0, self.color_list[nCmap][2], numInterpolationPoints)]
            carray = np.transpose(carray)
            self.cmap[nCmap] = Colormap(carray, controls)

# ---------------------------------------------------------------------
    def load_img_seq(self, path_in="", omitROIlayer=False):
        # first remove all layers
        for l in reversed(self.viewer.layers[:]):
            self.viewer.layers.remove(l)          
        self.numColors = int( self.ui.numColorsCbox.currentText() )

        if self.MultiImageSeriesFlag: # if we load multiple image series
            if not path_in:
                paths = self.MultiImageSeries_folderpath
            else:
                paths = path_in
            self.numSeries = len(paths)

            # if a dictionary for rotation angles, x- and y-shift doesnt yet exist, create it
            if not hasattr(self, 'RotationAngle'):
                self.RotationAngle = [0] * self.numSeries
                self.xShift        = [0] * self.numSeries
                self.yShift        = [0] * self.numSeries
                for nSeries in range(self.numSeries):
                    self.RotationAngle[nSeries] = [0] * self.MaxNumColors
                    self.xShift[nSeries]        = [0] * self.MaxNumColors
                    self.yShift[nSeries]        = [0] * self.MaxNumColors

            # if self already has cmaps, delete them
            if hasattr(self.viewer, 'cmap'):
                delattr(self.viewer, 'cmap')

            self.image_meta = [''] * self.numSeries
            cmap_numer = 0
            for nSeries in range(self.numSeries):
                # see if there's a yaml file found in this folder. If yes, load it                
                self.yamlFileName = os.path.join(paths[nSeries], 'shift.yaml')                           
                self.series2treat = nSeries
                self.LoadShiftYamlFile()
                if not hasattr(self.shift_yaml, 'angle'):
                    self.shift_yaml["angle"] = {}
                for nColor in range(self.MaxNumColors):  
                    try:
                        self.xShift[nSeries][nColor] = self.shift_yaml["x"]["col"+str(int(nColor))]
                    except:
                        self.xShift[nSeries][nColor] = 0
                    try:
                        self.yShift[nSeries][nColor] = self.shift_yaml["y"]["col"+str(int(nColor))]
                    except:
                        self.yShift[nSeries][nColor] = 0
                    try:
                        self.RotationAngle[nSeries][nColor] = self.shift_yaml["angle"]["col"+str(int(nColor))]
                    except:
                        self.RotationAngle[nSeries][nColor] = 0
                                        

                # read image
                if paths[nSeries]: # if we actually have a path
                    self.image_meta[nSeries] = crop_images.daskread_img_seq(num_colors=int(self.ui.numColorsCbox.currentText()), 
                        bkg_subtraction=bool(self.ui.bkgSubstractionCheckBox.checkState()), path=paths[nSeries],
                        RotationAngle=self.RotationAngle[nSeries])
                else: # in case we get an empty string, for instance from self.SwitchFOVs()
                    self.image_meta[nSeries] = {}
                    self.image_meta[nSeries]['folderpath'] = ''
                    self.image_meta[nSeries]['num_colors'] = int(self.ui.numColorsCbox.currentText())
                    ZeroDaskArray = da.zeros((10, 10, 10), chunks=(10, 10, 10))
                    for color_numer in range(self.image_meta[nSeries]['num_colors']):
                        self.image_meta[nSeries]['stack_color_'+str(color_numer)] = ZeroDaskArray
                        self.image_meta[nSeries]['min_int_color_'+str(color_numer)] = 0
                        self.image_meta[nSeries]['max_int_color_'+str(color_numer)] = 1
                        self.image_meta[nSeries]['min_contrast_color_'+str(color_numer)] = 0
                        self.image_meta[nSeries]['max_contrast_color_'+str(color_numer)] = 1
                    
                if not self.image_meta[nSeries]: # if we got an empty dict return
                    continue

                # add images to the various layers
                color_numer = 0
                while color_numer <= self.image_meta[nSeries]['num_colors']-1 and color_numer <= 5:
                    image_name = self.MultiImageSeries_description[nSeries]+'_color_'+str(color_numer)
                    image_name = self.removeprefix(image_name, '_')
                    self.viewer.add_image(self.image_meta[nSeries]['stack_color_'+str(color_numer)], colormap=self.cmap[cmap_numer],
                            contrast_limits=[self.image_meta[nSeries]['min_int_color_'+str(color_numer)],self.image_meta[nSeries]['max_int_color_'+str(color_numer)]],
                            blending='additive', multiscale=False, name=image_name)
                    self.viewer.layers[image_name].contrast_limits_range = [self.image_meta[nSeries]['min_contrast_color_'+str(color_numer)],self.image_meta[nSeries]['max_contrast_color_'+str(color_numer)]*2]
                    self.viewer.layers[image_name].contrast_limits = [self.image_meta[nSeries]['min_contrast_color_'+str(color_numer)],self.image_meta[nSeries]['max_contrast_color_'+str(color_numer)]]                    
                    color_numer += 1
                    cmap_numer += 1

            # if we only had empty directories, return
            if all([not elem for elem in self.image_meta]):
                return
                        
            # add the line layer            
            lineLayer = self.viewer.add_shapes(self.defaultLine, shape_type='line', name='Profile',
            edge_width=self.edge_width, edge_color='coral', face_color='royalblue')
            self.viewer.layers['Profile'].mode = 'select'

            if not omitROIlayer:
                # add the roi layer
                self.viewer.add_shapes(self.defaultShape, shape_type='rectangle', name='ROI_00',
                                edge_color='yellow', edge_width=self.edge_width, opacity=self.opacity, 
                                text=self.text_kwargs)
                self.viewer.layers['ROI_00'].mode = 'select'

            self.numLayers = len(self.viewer.layers)- 2*(not omitROIlayer) # subtract the obligatory ROI and profile layer            
            self.toggleChannelVisibility() # changes visibility of layers

            # apply shifts loaded from yaml file
            self.applyShiftsFromYaml()

        else: # if we load a single image series
            self.numSeries = 1
            nSeries = 0

            if path_in:
                self.use_current_image_path = True
            if not self.use_current_image_path:
                self.image_meta = [''] * self.numSeries            
            else:
                if path_in:
                    self.image_meta[nSeries]['folderpath'] = path_in[0]                

            # if a dictionary for rotation angles, x- and y-shift doesnt yet exist, create it
            if not hasattr(self, 'RotationAngle'):
                self.RotationAngle = [0] * self.numSeries
                self.xShift        = [0] * self.numSeries
                self.yShift        = [0] * self.numSeries
                # numColors          = int( self.ui.numColorsCbox.currentText() )
                for nSeries in range(self.numSeries):
                    self.RotationAngle[nSeries] = [0] * self.MaxNumColors
                    self.xShift[nSeries]        = [0] * self.MaxNumColors
                    self.yShift[nSeries]        = [0] * self.MaxNumColors

            # see if there's a yaml file found in this folder. If yes, load it
            if self.use_current_image_path:
                folderpath = self.image_meta[nSeries]['folderpath']
                if len(folderpath)==0:
                    self.yamlFileName = ''
                else:
                    self.yamlFileName = os.path.join(folderpath, 'shift.yaml')
            else:
                self.yamlFileName = ''
            self.series2treat = 0
            self.LoadShiftYamlFile()
            for nColor in range(self.numColors):
                try:
                    self.xShift[0][nColor] = self.shift_yaml["x"]["col"+str(int(nColor))]
                except:
                    self.xShift[0][nColor] = 0
                try:
                    self.yShift[0][nColor] = self.shift_yaml["y"]["col"+str(int(nColor))]
                except:
                    self.yShift[0][nColor] = 0
                try:
                    self.RotationAngle[0][nColor] = self.shift_yaml["angle"]["col"+str(int(nColor))]
                except:
                    self.RotationAngle[0][nColor] = 0


            # load the image
            self.CreateColormaps()
            if self.use_current_image_path:
                current_image_path = self.image_meta[nSeries]['folderpath']
                self.image_meta[nSeries] = crop_images.daskread_img_seq(num_colors=int(self.ui.numColorsCbox.currentText()), 
                bkg_subtraction=bool(self.ui.bkgSubstractionCheckBox.checkState()), path=current_image_path, 
                RotationAngle=self.RotationAngle[0])
            else:
                self.image_meta[nSeries] = crop_images.daskread_img_seq(num_colors=int(self.ui.numColorsCbox.currentText()), 
                bkg_subtraction=bool(self.ui.bkgSubstractionCheckBox.checkState()), 
                RotationAngle=self.RotationAngle[0])
            if not self.image_meta[nSeries]: # if we got an empty dict return
                return
            self.use_current_image_path = False
            self.current_image_path = self.image_meta[nSeries]["folderpath"]
            color_numer = 0
            while color_numer <= self.image_meta[nSeries]['num_colors']-1 and color_numer <= 5:
                self.viewer.add_image(self.image_meta[nSeries]['stack_color_'+str(color_numer)], colormap=self.cmap[color_numer],
                        contrast_limits=[self.image_meta[nSeries]['min_int_color_'+str(color_numer)],self.image_meta[nSeries]['max_int_color_'+str(color_numer)]],
                        blending='additive', multiscale=False, name='color_'+str(color_numer))
                self.viewer.layers['color_'+str(color_numer)].contrast_limits = [self.image_meta[nSeries]['min_contrast_color_'+str(color_numer)],self.image_meta[nSeries]['max_contrast_color_'+str(color_numer)]]
                color_numer += 1
            
            # add the line layer          
            lineLayer = self.viewer.add_shapes(self.defaultLine, shape_type='line', name='Profile',
            edge_width=self.edge_width, edge_color='coral', face_color='royalblue')
            self.viewer.layers['Profile'].mode = 'select'

            if not omitROIlayer:
                # add the roi layer
                self.viewer.add_shapes(self.defaultShape, shape_type='rectangle', name='ROI_00',
                                edge_color='yellow', edge_width=self.edge_width, opacity=self.opacity, text=self.text_kwargs)
                self.viewer.layers['ROI_00'].mode = 'select'
            
            self.numLayers = len(self.viewer.layers) - 1 - (not omitROIlayer) # minus ROI and profile layer
            # apply shifts loaded from yaml file
            self.applyShiftsFromYaml()

        self.FixContrast()
        self.UpdateText()

        # save the currently chosen number of colors in a yaml file
        for nSeries in range(self.numSeries):
            folderpath = self.image_meta[nSeries]['folderpath']
            if len(folderpath)==0:
                continue
            self.yamlFileName = os.path.join(folderpath, 'shift.yaml')                           
            self.LoadShiftYamlFile()
            self.shift_yaml["numColors"] = self.numColors
            self.shift_yaml = io.to_dict_walk(self.shift_yaml)
            os.makedirs(os.path.dirname(self.yamlFileName), exist_ok=True)
            with open(self.yamlFileName, "w") as shift_file:
                yaml.dump(dict(self.shift_yaml), shift_file, default_flow_style=False)

        # make the line draggable and trigger a callback when it's being dragged
        @lineLayer.mouse_drag_callbacks.append
        def profile_line_drag(layer, event):
            self.profile_line()
            yield
            while event.type == 'mouse_move':
                self.profile_line()
                yield

# ---------------------------------------------------------------------
    def profile_line(self):
        # first get the line
        # if 'Profile' in self.viewer.layers
        try:
            line = self.viewer.layers['Profile'].data[0]
        except:
            return # there was no line layer
        # get the current time
        currentTime  = self.viewer.dims.point[0]
        profile_data = []
        color_list = []
        geometric_transform = True        
        if not hasattr(self, 'xShift'):
            geometric_transform = False
        for nSeries in range(self.numSeries):
            filenames = self.image_meta[nSeries]['filenames']
            for col in range(self.numColors):
                nLayer = nSeries*self.numColors+col
                if not self.viewer.layers[nLayer].visible:
                    continue # we only show profiles of visible layers
                index = int( currentTime*self.numColors + col )
                if index > len(filenames)-1:
                    continue # this happens when one image series is longer that the others
                image = np.array(pims.ImageSequence(filenames[index]), dtype=np.uint16)
                image = image[0]
                if geometric_transform:
                    image = crop_images.geometric_shift(image, angle=self.RotationAngle[nSeries][col],
                                    shift_x=self.xShift[nSeries][col], shift_y=self.yShift[nSeries][col])
                profile_data.append(
                    measure.profile_line(image, line[0], line[1], 
                    mode='nearest', linewidth=3, reduce_func=np.mean)
                    )
                color_list.append(self.color_list[nLayer])

        self.LPwin.update_plot_data(profile_data, color_list)
        if not self.LPwin.isVisible():
            self.LPwin.show()

# ---------------------------------------------------------------------
    def applyShiftsFromYaml(self):
        for nSeries in range(self.numSeries):
            for nColor in range(self.numColors):
                if (self.xShift[nSeries][nColor]!=0) or (self.yShift[nSeries][nColor]!=0):
                    nLayer = nSeries*self.numColors + nColor                        
                    translationVector = np.array( [0, self.yShift[nSeries][nColor], self.xShift[nSeries][nColor]] ) # [z, y, x]
                    self.viewer.layers[nLayer].translate = translationVector

# ---------------------------------------------------------------------
    def crop(self):
        # for each of the image series, get the frame_start and frame_end
        # in case the image series has less frames than frame_end set in the 
        # spin box, take that
        self.frame_start = self.ui.frameStartSbox.value()
        self.frame_end = self.ui.frameEndBox.value()
        if self.frame_end == -1:
            self.frame_end = None

        # Find all layers which contain shapes and extract labels from those
        isShapeLayer = self.FindAllShapeLayers()
        numShapeLayers = len(isShapeLayer)
        shape_layers = [0] * numShapeLayers
        ROIlabels    = [0] * numShapeLayers
        for nShapeLayer in range(numShapeLayers):
            shape_layers[nShapeLayer] = self.viewer.layers[isShapeLayer[nShapeLayer]]
            ROIlabels[nShapeLayer] = self.viewer.layers[isShapeLayer[nShapeLayer]].name.lower()

        if type(self.image_meta) is not list:
            self.image_meta = [self.image_meta]

        for nSeries in range(self.numSeries):
            print('Cropping image series '+str(int(nSeries+1))+' / '+str(int(self.numSeries))+' ...')
            doShift = False
            if hasattr(self, 'xShift'):
                if np.any(self.xShift[nSeries]) or np.any(self.yShift[nSeries]) or np.any(self.RotationAngle[nSeries]):
                    doShift       = True
                xShift        = self.xShift[nSeries]
                yShift        = self.yShift[nSeries]
                RotationAngle = self.RotationAngle[nSeries]
            else:
                doShift = False
                xShift        = 0
                yShift        = 0
                RotationAngle = 0
            if not hasattr(self, 'numColors'):
                self.numColors = int( self.ui.numColorsCbox.currentText() )
            if len(self.image_meta[nSeries]['folderpath'])==0:
                print('Dummy layer. Skipping.')
                continue
            crop_images.crop_rect_shapes(self.image_meta[nSeries], shape_layers,
                            frame_start=self.frame_start, 
                            frame_end=self.frame_end,
                            geometric_transform=doShift,
                            shift_x=xShift, 
                            shift_y=yShift, 
                            angle=RotationAngle,
                            label=ROIlabels, 
                            defaultShape=self.defaultShape,
                            numColors=self.numColors)
        print("Cropping finished")

# ---------------------------------------------------------------------
    def GetROIDescription(self, roi_file):
        # the format is something like xxxxxxxx_descrip_ti_on.roi
        # find from the underscore to the dot to get the description and convert to small letters    
        split_list = os.path.basename( roi_file ).rsplit("_") # we only want the filename
        merged_list = ""
        # see if one of the list entries contain the keyword 'shifted'
        shifted_index = [i for i, j in enumerate(split_list) if j == 'shifted']
        if len(shifted_index) > 0:
            startIndex = shifted_index[-1]+3 # after 'shifted' there is one field with 'dx' and one with 'dy'
        else:
            startIndex = 1
        for l in range(startIndex, len(split_list)):
            merged_list += split_list[l]
            if l < len(split_list)-1:
                merged_list += '_'
        if len(merged_list)==0:
            label = ''
        else:
            label = merged_list.split(".")[-2].lower()
        return label

# ---------------------------------------------------------------------
    def countLayerNames(self, label):
        if len(label)==0:
            label = 'ROI'
        ROIlayercount = 0
        for nLayer in range(len(self.viewer.layers)):
            if re.match((label+'_\d\d').lower(),self.viewer.layers[nLayer].name.lower()):
                ROIlayercount += 1
        return ROIlayercount

# ---------------------------------------------------------------------
    def set_default_frame_num(self):
        self.ui.frameStartSbox.setProperty("value", 0)
        self.ui.frameEndBox.setProperty("value", -1)

# ---------------------------------------------------------------------
    def change_num_colors(self):
        self.numColors = int( self.ui.numColorsCbox.currentText() )
        if hasattr(self, 'MultiImageSeriesFlag'):
            self.use_current_image_path = True
            self.load_img_seq()

# ---------------------------------------------------------------------
    def load_imageJ_rois(self):
        if self.current_image_path is None:            
            self.current_image_path = self.image_meta[0]['folderpath']
        if len(self.current_image_path)==0:
            self.current_image_path = self.image_meta[0]['folderpath']
        try:
            roi_file_list = io.FileDialog(self.current_image_path+'_analysis', "Select ROI files",
                            "ROI files *.roi").openFileNamesDialog()
        except:
            try:
                roi_file_list = io.FileDialog(self.current_image_path, "Select ROI files",
                            "ROI files *.roi").openFileNamesDialog()
            except:
                roi_file_list = io.FileDialog(None, "Select ROI files",
                            "ROI files *.roi").openFileNamesDialog()
        # for every ROI, add a new layer. we do this in order to enable giving
        # each layer a name which can be used as description
        self.numROIs = len(roi_file_list)
        ROIlabel = [''] * self.numROIs
        for nROI in range(self.numROIs):
            ROIlabel[nROI] = self.GetROIDescription(roi_file_list[nROI])
        sort_ind = sorted((name, index) for index, name in enumerate(ROIlabel))
        ROIlabel = [ROIlabel[sort_ind[x][1]] for x in range(self.numROIs)]
        roi_file_list = [roi_file_list[sort_ind[x][1]] for x in range(self.numROIs)]
        # change empty strings to 'ROI'
        emptyROIlabelIndices = [i for i, j in enumerate(ROIlabel) if j == '']
        for k in emptyROIlabelIndices:
            ROIlabel[k] = 'ROI'
        # find out how many layers we currently have and how many of them are called the same
        # LayerCounter = [0] * self.numROIs
        # # for each unique name
        # for uniqueROIlabel in set(ROIlabel):
        #     # get how many layers with that name exist
        #     ROIlayercount = self.countLayerNames(uniqueROIlabel)
        #     # get the indices with the same current ROIlabel
        #     uniqueROIindices = [i for i, j in enumerate(ROIlabel) if j == uniqueROIlabel]
        #     for k in range(len(uniqueROIindices)):
        #         LayerCounter[uniqueROIindices[k]] = ROIlayercount+1+k
        
        uniqueROIlabels = list(set(ROIlabel))
        # shapes_to_add = [''] * len( uniqueROIlabels )
        for uniqueROI in range(len( uniqueROIlabels )):
            shapes_to_add = []
            uniqueROIindices = [i for i, j in enumerate(ROIlabel) if j == uniqueROIlabels[uniqueROI]]
            for k in range(len(uniqueROIindices)):
                shapes_to_add.append(crop_images.roi_to_rect_shape(roi_file_list[uniqueROIindices[k]]))
                # LayerCounter[uniqueROIindices[k]] = ROIlayercount+1+k
            ROIlayercount = self.countLayerNames(uniqueROIlabels[uniqueROI])
            layerName     = uniqueROIlabels[uniqueROI] + '_' + format(ROIlayercount, '02')
            self.viewer.add_shapes(shapes_to_add, shape_type='rectangle', name=layerName,
                        edge_color='yellow', edge_width=self.edge_width, 
                        opacity=self.opacity, text=self.text_kwargs)

        self.UpdateText()

# ---------------------------------------------------------------------
    def FindAllShapeLayers(self):
        numLayers = len(self.viewer.layers)
        isShapeLayer = []
        for nLayer in range(numLayers): # only shape layers have the attribute 'nshapes'
            if hasattr(self.viewer.layers[nLayer], 'nshapes'):
                if not any([shapetype=='line' for shapetype in self.viewer.layers[nLayer].shape_type]):
                    isShapeLayer.append(nLayer)
        return isShapeLayer

# ---------------------------------------------------------------------
    def save_imageJ_rois(self):
        # get all shape layers
        isShapeLayer = self.FindAllShapeLayers()

        # for each layer, see if the ROI is at the default position. If yes,
        # don't save it
        for nShapeLayer in isShapeLayer:
            shape_layer = self.viewer.layers[nShapeLayer]
            ROIlabel = self.viewer.layers[nShapeLayer].name.lower()
            # save it to each image series we have open
            if not self.MultiImageSeriesFlag:
                # savepath = os.path.dirname(self.image_meta[0]['folderpath'])
                savepath = self.image_meta[0]['folderpath'] + '_analysis'
                ROInames = crop_images.save_rectshape_as_imageJroi(shape_layer, 
                    folder_to_save=savepath, label=ROIlabel, defaultShape=self.defaultShape)
                if not ROInames:
                    continue
                folderpath = self.image_meta[0]['folderpath']
                self.yamlFileName = os.path.join(folderpath, 'shift.yaml')                           
                self.LoadShiftYamlFile()
                self.shift_yaml["numColors"] = self.numColors
                if not hasattr(self.shift_yaml, 'angle'):
                    self.shift_yaml["angle"] = {}
                for nColor in range(self.numColors):
                    if hasattr(self, 'xShift'):
                        self.shift_yaml["x"]["col"+str(int(nColor))] = self.xShift[0][nColor]
                        self.shift_yaml["y"]["col"+str(int(nColor))] = self.yShift[0][nColor]                        
                        self.shift_yaml["angle"]["col"+str(int(nColor))] = self.RotationAngle[0][nColor]
                    else: 
                        self.shift_yaml["x"]["col"+str(int(nColor))] = 0
                        self.shift_yaml["y"]["col"+str(int(nColor))] = 0
                        self.shift_yaml["angle"]["col"+str(int(nColor))] = 0

                # save the yaml file
                self.shift_yaml = io.to_dict_walk(self.shift_yaml)
                for nROI in range(len(ROInames)):
                    yamlFileName = os.path.join(savepath, ROInames[nROI]+'.yaml')
                    with open(yamlFileName, "w") as shift_file:
                        yaml.dump(dict(self.shift_yaml), shift_file, default_flow_style=False)
            else:
                for nSeries in range(self.numSeries):
                    folderpath = self.image_meta[nSeries]['folderpath']
                    if len(folderpath)==0:
                        continue
                    savepath = folderpath + '_analysis'
                    ROInames = crop_images.save_rectshape_as_imageJroi(shape_layer, 
                        folder_to_save=savepath, label=ROIlabel, defaultShape=self.defaultShape)
                    
                    self.yamlFileName = os.path.join(folderpath, 'shift.yaml')                           
                    self.LoadShiftYamlFile()
                    self.shift_yaml["numColors"] = self.numColors
                    if not hasattr(self.shift_yaml, 'angle'):
                        self.shift_yaml["angle"] = {}
                    for nColor in range(self.numColors):
                        self.shift_yaml["x"]["col"+str(int(nColor))] = self.xShift[nSeries][nColor]
                        self.shift_yaml["y"]["col"+str(int(nColor))] = self.yShift[nSeries][nColor]
                        self.shift_yaml["angle"]["col"+str(int(nColor))] = self.RotationAngle[nSeries][nColor]
                    # save the yaml file
                    self.shift_yaml = io.to_dict_walk(self.shift_yaml)
                    for nROI in range(len(ROInames)):
                        yamlFileName = os.path.join(savepath, ROInames[nROI]+'.yaml')
                        with open(yamlFileName, "w") as shift_file:
                            yaml.dump(dict(self.shift_yaml), shift_file, default_flow_style=False)
        print('ROIs saved.')

# ---------------------------------------------------------------------
    def toggle_bkg_subtraction(self):
        self.use_current_image_path = True
        self.load_img_seq()

# ---------------------------------------------------------------------
    def closeEvent(self, event):
        settings = io.load_user_settings()
        if self.folderpath is not None:
            settings["crop"]["PWD"] = self.current_image_path
        io.save_user_settings(settings)
        QtWidgets.qApp.closeAllWindows()

# ---------------------------------------------------------------------
    def load_user_settings(self):
        settings = io.load_user_settings()
        try:
            self.current_image_path = settings["crop"]["PWD"]
        except Exception as e:
            print(e)
            pass

# ---------------------------------------------------------------------
    def TIFisMultipage(self, tif_file_list):
        if type(tif_file_list) is not list:
            tif_file_list = [tif_file_list]
        isMultipage = []
        for tif_file in tif_file_list:
            img = Image.open(tif_file)
            i = 0
            while True:
                try:   
                    img.seek(i)
                except EOFError:
                    break       
                i += 1
                if i > 1:
                    isMultipage.append(True)
                    break
            if i == 1:
                isMultipage.append(False)
        return isMultipage

# ---------------------------------------------------------------------
    def BatchCrop_from_directory(self):
        '''
        Crops from all the subfolders with images and corresponding ROIs
        FOVs can be sorted by their description into another folder while keeping their name and source path
        '''

        directory = self.BatchCropPath
        saveCollectively = False
        if hasattr(self, 'BatchSavePath'):
            save_directory = self.BatchSavePath
            if os.path.isdir(save_directory):
                saveCollectively = True

        # Get relevant subdirectories
        # sub_dirs, tags = BatchIdentifyRelevantDirectories(directory, num_colors)
        sub_dirs = self.BatchIdentifyRelevantDirectories(directory)

        # go through each sub-directory and get all .roi files
        print('Starting cropping routine...')
        for i in range(len(sub_dirs)):
            roi_file_list = glob.glob(sub_dirs[i] + "/*.roi", recursive = False)
            ROIdescriptions = self.BatchGetROIDescription(roi_file_list)
            roi_coord_list, roi_original_names = self.BatchLoadROIs(roi_file_list)

            # for the dir where the ROI is located, we might look for .tif files to which the roi is applied
            # - in the same folder
            # - in all subfolders
            # - if the current folder (in which we found the roi) contains '_analysis', look for a folder without that tag and concurrent subfolders
            subfolders = [] # initialize list of subfolders
            # look in the same folder and its sub folders. Omit multipage-tifs since those are the cropped ones
            for iteration in range(2):
                if iteration == 0:
                    tmpdir = sub_dirs[i]
                else:
                    if sub_dirs[i].endswith('_analysis'):
                        tmpdir = sub_dirs[i][0:-len('_analysis')]
                    else: 
                        continue
                tif_file_list = glob.glob(tmpdir + "/**/*.tif", recursive=True)
                isMultipage = self.TIFisMultipage(tif_file_list)
                tif_file_list = [tif_file_list[i] for i in range(len(tif_file_list)) if not isMultipage[i]]
                subfolders.extend( list(set([os.path.dirname(file) for file in tif_file_list])) )
            subfolders = list(set(subfolders))

            # give user feedback
            print('[', str(i+1), '/', str(len(sub_dirs)), '] ', 
            'Applying ROIs in "', sub_dirs[i], '" to the following folder(s):')
            [print('- ', subfolder) for subfolder in subfolders]

            # # there might be several subfolders to which this ROI is to be applied. 
            # # Get all of those subfolders 
            # # (but only the ones which are exactly one level lower in the hierarchy and
            # # which contain .tif images)
            # subfolders = [os.path.dirname(file) for file in roi_file_list]
            # subfolders_temp = [1]
            # while len(subfolders_temp) > 0:
            #     subfolders_temp = [f.path for f in os.scandir(sub_dirs[i]) if f.is_dir()]            
            #     for subfolder in subfolders_temp:
            #         # files         = glob.glob(subfolder + '/*.tif', recursive=False) # look for tif files
            #         roi_files = glob.glob(subfolder + "/*.roi", recursive = False) # look for roi files in this subfolder. These are likely to be the prcessed ones already
            #         if len(roi_files)>0:# and (not any(roi_subfolder_item.count('-f')>1 for roi_subfolder_item in roi_subfolder)):
            #             if not subfolders: # if subfolders is still empty (=first iteration)
            #                 print('[', str(i+1), '/', str(len(sub_dirs)), '] ', 
            #                 'Applying ROIs in "', sub_dirs[i], '" to the following folders:')
            #             subfolders.append(subfolder)
            #             print('- ', subfolder)
            #         else: 
            #             print('[', str(i+1), '/', str(len(sub_dirs)), '] ', 
            #                 'No ROIs or no suitable images found to process.')

            for sf in range(len(subfolders)):
                # look if there's a yaml file which contains the number of colors
                self.yamlFileName = os.path.join(subfolders[sf], 'shift.yaml')
                try:                     
                    self.LoadShiftYamlFile()
                    numColors = self.shift_yaml["numColors"]
                except:
                    numColors = int( self.ui.numColorsCbox.currentText() )
                self.Batchcrop(sub_dirs[i], subfolders[sf], roi_coord_list, roi_file_list, 
                roi_original_names, ROIdescriptions, save_directory, numColors, 
                saveCollectively, sf, len(subfolders))
            print()
            
        print('Batch cropping finished.')

# ---------------------------------------------------------------
    def Batchcrop(self, dir, sub_dir, roi_coord_list, roi_file_list, roi_original_names, 
        ROIdescriptions, sort_directory, num_colors, sort_FOVs, nDir, numDirs):
        # dir is the directory where ROIs are stored
        # sub_dir is where the crops go

        folderpath = sub_dir
        
        # default. can be changed in the future (20200918)
        frame_start = 0
        imgseq = pims.ImageSequence(folderpath+os.path.sep+'*.tif')
        frame_end = len(imgseq)
        num_frames = frame_end-frame_start
        num_frames_update = num_colors * ((frame_end - frame_start) // num_colors)
        frame_end = frame_start + num_frames_update
        
        dir_to_save = os.path.join(folderpath+'_analysis')
        if not os.path.isdir(dir_to_save):
            os.makedirs(dir_to_save)

        names_roi_tosave = []
        names_roi_tosave_no_frames = []
        img_array_all = {}
        for i in range(len(roi_coord_list)):  
            rect = roi_coord_list[i]
            rect_params = crop_images.get_rect_params(rect) 
            key = 'arr' + str(i)
            img_array_all[key] = np.zeros((round(num_frames_update/num_colors), num_colors,
                                        rect_params['width'], rect_params['length']),
                                        dtype=np.uint16)        
            rect_0 = rect[0].astype(int)
            nam = 'x' + str(rect_0[0]) + '-y' + str(rect_0[1]) +\
                '-l' + str(rect_params['length']) + '-w' + str(rect_params['width']) +\
                '-a' + str(rect_params['angle']) + '-f' + str(frame_start) + '-f' +\
                str(frame_end) + '_' + ROIdescriptions[i]
            names_roi_tosave.append(nam)
            nam_no_frames = 'x' + str(rect_0[0]) + '-y' + str(rect_0[1]) +\
                '-l' + str(rect_params['length']) + '-w' + str(rect_params['width']) +\
                '-a' + str(rect_params['angle']) + '_' + ROIdescriptions[i]
            names_roi_tosave_no_frames.append(nam_no_frames)
        rect_keys = list(img_array_all.keys())

        # if we sort, figure out the name of each file in the sorted directory:
        # we first need the file's superior dir, the current dir, and sub_dir
        name_sorted = []
        CropPath = os.path.normpath(self.BatchCropPath)
        if sort_FOVs:
            sorted_dir = []
            for i in range(len(roi_coord_list)):
                # find out how many levels are in between the folder we're cropping from
                # and the folder in which we found the ROI
                foundParent = False
                children = []
                currentPath = os.path.dirname(roi_file_list[i])
                while foundParent == False:
                    if os.path.normpath(currentPath) == CropPath:
                        foundParent = True
                    children.append(os.path.basename(currentPath))
                    currentPath = os.path.dirname(currentPath)
                children.reverse()
                children = '_'.join(children)

                # create a dir to save the FOV to if it doesnt exist yet
                sorted_dir.append( os.path.join(sort_directory, ROIdescriptions[i]) )
                if not os.path.isdir(sorted_dir[-1]):
                    os.makedirs(sorted_dir[-1])
                name_sorted.append( 
                    os.path.join(sorted_dir[-1],\
                    children + '_' +\
                    names_roi_tosave_no_frames[i]) )

        # go through each roi name and see if it already exists. If yes, skip it
        skipROI  = []
        skipIMG  = []
        skipSORT = []
        for i in range(len(roi_coord_list)):
            ROIpath = os.path.join(dir_to_save, names_roi_tosave[i]+'.roi')
            IMGpath = os.path.join(dir_to_save, names_roi_tosave_no_frames[i]+'.tif')
            if os.path.isfile(ROIpath): # check if ROIs exist
                skipROI.append(True)
            else:
                skipROI.append(False)
            if os.path.isfile(IMGpath): # check if images exist
                skipIMG.append(True)
            else:
                skipIMG.append(False)
            if sort_FOVs:
                if os.path.isfile(name_sorted[i]): # check if images exist
                    skipSORT.append(True)
                else:
                    skipSORT.append(False)
            else:
                skipSORT.append(True)

        if all(skipIMG):
            print('All crops already exist in subfolder '+str(nDir+1)+'/'+str(numDirs)+'.')
            for i in range(len(roi_coord_list)):
                if (not skipROI[i]):
                    roi_ij = ImagejRoi.frompoints(crop_images.rect_shape_to_roi(roi_coord_list[i]))
                    roi_ij.tofile(os.path.join(dir_to_save, names_roi_tosave_no_frames[i]+'.roi')) # saving the ROI in the subfolder
                    roi_ij.tofile(os.path.join(dir, names_roi_tosave_no_frames[i]+'.roi')) # saving the ROI in the folder where the ROI was found but with new name
                    # os.remove(os.path.join(dir, roi_original_names[i])) # remove the original file
                if (not skipSORT[i]):
                    roi_ij = ImagejRoi.frompoints(crop_images.rect_shape_to_roi(roi_coord_list[i]))
                    roi_ij.tofile(name_sorted[i]+'.roi') # saving the ROI in the sorted dir
                    # copying the images from the dir to the sorted one
                    shutil.copy2(os.path.join(dir_to_save, names_roi_tosave[i]+'.tif'),\
                        name_sorted[i]+'.tif') # copy the file to the sorted dir
            return

        numROIs = len(roi_coord_list)
        # see if we can find the yaml file which is associated to each ROI
        angle   = [0] * numROIs
        shift_x = [0] * numROIs
        shift_y = [0] * numROIs
        for j in range(numROIs):
            angle[j], shift_x[j], shift_y[j] = crop_images.readROIassociatedYamlFile(roi_file_list[j], num_colors)
        # loop through colors, then time, finally ROIs and crop
        for col in range(num_colors):
            # select the correct frames
            imgseq_col = imgseq[col:num_frames_update:num_colors]
            for i in trange(len(imgseq_col), desc='Batch cropping color '+str(col+1)+'/'+str(num_colors)+' in subfolder '+str(nDir+1)+'/'+str(numDirs)+'...'):
                # frame_index = i*num_colors+col
                # if num_colors>1:
                #     if col==0:
                #         frame_index += 1
                #     elif col==1:
                #         frame_index -= 1
                if i==0: # decide if the whole image has to be shifted or only the crop
                    minWidth  = float('inf')
                    minLength = float('inf')
                    for nRect in range(len(roi_coord_list)):
                        rect_params = crop_images.get_rect_params(roi_coord_list[nRect]) 
                        minWidth = np.min([minWidth, rect_params['width']])
                        minLength = np.min([minLength, rect_params['length']])
                    percentage = 0.1 # 10% of the smallest crop
                    shift_wholeImage = False
                    j = 0
                    while not shift_wholeImage and j<len(shift_x): # as soon as we find a ROI which requires to shift to complete image, we can stop
                        if (shift_x[j][col]/minLength>percentage) or (shift_y[j][col]/minWidth>percentage):
                            shift_wholeImage = True
                        else:
                            shift_wholeImage = False
                        j += 1
                # load image
                img = np.array(imgseq_col[i], dtype=np.uint16)                
                for j in range(numROIs):
                    if (not skipIMG[j]):
                        # shift the whole image if necessary
                        if shift_wholeImage and ((angle[j][col]!=0) or (shift_x[j][col]!=0) or (shift_y[j][col]!=0)):
                            img_shift = crop_images.geometric_shift(img, angle=angle[j][col],
                                                shift_x=shift_x[j][col], shift_y=shift_y[j][col])
                        else:
                            img_shift = img
                        # ... or just shift the crop
                        img_cropped = crop_images.crop_rect(img_shift, roi_coord_list[j])
                        if (not shift_wholeImage) and ((angle[j][col]!=0) or (shift_x[j][col]!=0) or (shift_y[j][col]!=0)):
                            img_cropped = crop_images.geometric_shift(img_cropped, angle=angle[j][col],
                                            shift_x=shift_x[j][col], shift_y=shift_y[j][col])
                        img_array_all[rect_keys[j]][i, col, :, :] = img_cropped            
        
        for i in range(len(roi_coord_list)):
            try:
                if (not skipROI[i]):
                    roi_ij = ImagejRoi.frompoints(crop_images.rect_shape_to_roi(roi_coord_list[i]))
                    roi_ij.tofile(os.path.join(dir_to_save, names_roi_tosave_no_frames[i]+'.roi')) # saving the ROI in the subfolder
                    roi_ij.tofile(os.path.join(dir, names_roi_tosave_no_frames[i]+'.roi')) # saving the ROI in the folder where the ROI was found but with new name
                    # os.remove(os.path.join(dir,roi_original_names[i])) # remove the original file        
                if (not skipIMG[i]):
                    imwrite(os.path.join(dir_to_save, names_roi_tosave[i]+'.tif'),
                            img_array_all[rect_keys[i]], imagej=True,
                            metadata={'axis': 'TCYX', 'channels': num_colors,
                            'mode': 'composite',})
                if (not skipSORT[i]):
                        roi_ij.tofile(name_sorted[i]+'.roi') # saving the ROI in the sorted dir
                        shutil.copy2(os.path.join(dir_to_save, names_roi_tosave[i]+'.tif'),\
                            name_sorted[i]+'.tif') # copy the file to the sorted dir
            except:
                print(names_roi_tosave[i]+" not found anymore. Skipping.")
                
# ---------------------------------------------------------------
    def BatchLoadROIs(self, roi_file_list):
        roi_file_list_updated = []
        roi_original_names_updated = []
        for j in range(len(roi_file_list)):    
            if (roi_file_list[j].count('-f')<2):
                roi_file_list_updated.append(roi_file_list[j])
                roi_original_names_updated.append(roi_file_list[j])
        roi_file_list = roi_file_list_updated
        roi_original_names = roi_original_names_updated
        
        roi_coord_list = []
        for roi_file in roi_file_list:
            roi = ImagejRoi.fromfile(roi_file)
            roi_coord = np.flip(roi.coordinates())
            roi_coord = np.array([roi_coord[2], roi_coord[1], roi_coord[0], roi_coord[3]], dtype=np.float32)
            roi_coord[roi_coord<0] = 1
            roi_coord_list.append(roi_coord)
        return roi_coord_list, roi_original_names

# ---------------------------------------------------------------
    def BatchGetROIDescription(self, roi_file_list, descriptions=[]):
        # the format is something like xxxxxxxx_description.roi
        # find from the LAST underscore to the dot to get the description and convert to small letters
        for k in range(len(roi_file_list)):      
            split_list = os.path.basename( roi_file_list[k] ).rsplit("_") # we only want the filename
            # split_list = split_list[-1]
            # merged_list = ""
            # for l in range(1, len(split_list)):
            #     merged_list += split_list[l]
            #     if l < len(split_list)-1:
            #         merged_list += '_'
            descriptions.append( split_list[-1].split(".")[-2].lower() )
        return descriptions

# ---------------------------------------------------------------
    def BatchIdentifyRelevantDirectories(self, directory):
        
        # scan directory for folders
        sub_dirs = self.BatchFindAllSubdirectories(directory)
        sub_dirs.append(directory) 
        print()

        # check if there are any ambiguities we have to ask the user about
        # basename_list = []
        sub_dirs_updated = []
        for i in trange(len(sub_dirs), desc='Check for .roi files'):
            roi_file_list = glob.glob(sub_dirs[i] + '/*.roi', recursive = False)

            # only consider if there are any roi files and if these roi files do 
            # not contain two "-f" in their name (those are the processed ones)
            if roi_file_list:
                if (not any(roi_file_list_item.count('-f')>1 for roi_file_list_item in roi_file_list)):
                    sub_dirs_updated.append(sub_dirs[i])
        sub_dirs = sub_dirs_updated

        # if there is a choice to be made, we dont currently ask the user. We simply take the complete
        # folder. ALready processed ROIs/tifs are skipped anyway
        # if len(sub_dirs)>1:
        #     print("Found more than one potential folder to Batchcrop images from. Which one to take?")
        #     for i in range(len(sub_dirs)):
        #         print("[", i, "] ", sub_dirs[i])
        #     choice = [int(item) for item in input("Enter the list items (separated by blank, type -1 for all): ").split()]
        #     if choice[0]==-1:
        #         choice = range(0, len(sub_dirs))
        #     sub_dirs = [sub_dirs[i] for i in choice]
        if not sub_dirs:
            print('No directories found. Try again.')
            self.call_BatchProcessing()
            return
            
        # outdated: remove all subdirectories which contain a roi file which has 2x "-f" in the name since those are files which are saved together with the crops    
        # sub_dirs_updated = []
        # for i in trange(len(sub_dirs), desc='Discard already processed ROIs'):
        #     subfolders = [f.path for f in os.scandir(sub_dirs[i]) if f.is_dir()]
        #     roi_file_list = glob.glob(sub_dirs[i] + "/**/*.roi", recursive = True)
        #     # for j in range(len(subfolders)):
        #     #     roi_file_list = glob.glob(sub_dirs[i] + "/**/*.roi", recursive = True)
        #         for k in range(len(roi_file_list)):
        #             if (roi_file_list[k].count('-f')<2) and (sub_dirs[i] not in sub_dirs_updated):
        #                 sub_dirs_updated.append(sub_dirs[i])
        # sub_dirs = sub_dirs_updated
        
        print()
        print("Cropping from the following folders: ")
        for j in range(len(sub_dirs)):
            print("- ", sub_dirs[j])
        print()
            
        return sub_dirs

# ---------------------------------------------------------------
    def BatchFindAllSubdirectories(self, dirname, count=0):
        subfolders = [f.path for f in os.scandir(dirname) if f.is_dir()]
        for dirname in list(subfolders):
            count += 1
            subfolders.extend(self.BatchFindAllSubdirectories(dirname, count))
        return subfolders

# ---------------------------------------------------------------------
def main():
    with napari.gui_qt():
        viewer = napari.Viewer(title="Crop or make Kymograph")
        ui = NapariTabs(viewer)
        viewer.window.add_dock_widget(ui)        

if __name__ == "__main__":
    main()