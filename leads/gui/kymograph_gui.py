import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import qdarkstyle
import pyqtgraph.Qt as pg_qt #import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.dockarea as pg_da
import pyqtgraph.exporters
from tifffile import imwrite
from ..kymograph import (read_img_seq, read_img_stack,
                median_bkg_substration, peakfinder_savgol,
                analyze_maxpeak, loop_sm_dist)
from .kymograph_ui import Ui_Form
import PySimpleGUI as sg
import os, sys, glob, time, subprocess
import yaml

DEFAULTS = {
    "ColorMap" : 'plasma',
    }
DEFAULT_PARAMETERS = {
    "Acquisition Time" : 100, # in millisecond
    "Pixel Size" : 115, # in nanometer
    "ROI Width" : 11,
    }

pg.setConfigOption('background', 'k') # 'w' for white and 'k' for black background
pg.setConfigOption('imageAxisOrder', 'col-major') # the row and cols are reversed
class kymographGui(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.layout = pg_qt.QtGui.QGridLayout()
        self.dockarea = pg_da.DockArea()
        self.ui.centralWidget.setLayout(self.layout)
        self.layout.addWidget(self.dockarea)

        # initiate some parameters
        self.numColors = self.ui.numColorsComboBox_3.currentText()
        self.ui.numColorsComboBox_3.currentIndexChanged.connect(self.change_num_colors)
        self.LineROIwidth = self.ui.roiWidthSpinBox.value()
        self.pixelSize = 1e-3 * self.ui.pixelSizeSpinBox.value() # converted to um from nm
        self.acquisitionTime = 1e-3 * self.ui.AcquisitionTimeSpinBox_2.value() # converted to sec from ms
        if self.numColors == "2": self.acquisitionTime = 2 * self.acquisitionTime
        self.scalebar_img = None
        # add plotting widgets
        self.add_col1_imvs()
        if self.numColors == "2":
            self.add_col2_imvs()
        # connect the signals
        self.connect_signals()
        self.ui.loadImageStackButton.clicked.connect(self.load_img_stack)
        self.ui.loadImageSeqButton.clicked.connect(self.restore_default_dockstate)
        self.defaultDockState = self.dockarea.saveState()
        self.ui.saveParamsBtn.clicked.connect(self.save_yaml_params)
        self.ui.pixelSizeSpinBox.valueChanged.connect(self.set_scalebar)
        self.ui.AcquisitionTimeSpinBox_2.valueChanged.connect(self.set_scalebar)
        self.ui.processImageCheckBox.stateChanged.connect(self.processed_image_check)
        self.ui.mergeColorsCheckBox.stateChanged.connect(self.merge_colors)
        self.ui.swapColorsCheckBox.stateChanged.connect(self.swap_colors)
        self.ui.detectLoopsBtn.clicked.connect(self.detect_loops)
        self.ui.saveSectionBtn.clicked.connect(self.save_section)
        self.ui.frameStartSpinBox.valueChanged.connect(self.frames_changed) #keyboardTracking=False, so works when entered
        self.ui.frameEndSpinBox.valueChanged.connect(self.frames_changed)
        self.ui.RealTimeKymoCheckBox.stateChanged.connect(self.realtime_kymo)
        self.ui.updateKymoBtn.clicked.connect(self.update_kymo)

    def connect_signals(self):
        self.ui.roiWidthSpinBox.valueChanged.connect(self.set_roi_width)
        self.roirect_left.sigRegionChanged.connect(self.roi_changed)
        self.infline_left.sigPositionChanged.connect(self.infiline_left_update)
        if self.numColors == "2":
            self.imv00.sigTimeChanged.connect(self.sync_videos)
            self.infline_right.sigPositionChanged.connect(self.infiline_right_update)
        # self.infline_left.sigDragged
    def add_col1_imvs(self):
        self.imv00 = pg.ImageView(name='color 1')
        self.imv00.setPredefinedGradient(DEFAULTS["ColorMap"])
        self.hide_imgv_roi_norm(self.imv00)
        self.imv00.fps = 7
        self.roirect_left = pg.LineROI([20, 20], [40, 20], width=11, pen=(3, 9))
        # self.roirect_left = pg.MultiLineROI([[10, 30], [30, 50], [50, 30]], width=5, pen=(2,9))
        self.imv00.addItem(self.roirect_left)
        self.plot3 = pg.PlotItem(name='green_kymo')#, clickable=True
        # self.plot3.setAspectLocked(False)

        self.imv10 = pg.ImageView(view=self.plot3)
        self.imv10.setPredefinedGradient(DEFAULTS["ColorMap"])
        self.hide_imgv_cmap(self.imv10); self.plot3.hideAxis('left')
        self.region3_noLoop = pg.LinearRegionItem((10, 80))
        self.imv10.addItem(self.region3_noLoop, ignoreBounds=True)
        label = pg.InfLineLabel(self.region3_noLoop.lines[1], "Non loop region",
                                position=0.75, rotateAxis=(1,0), anchor=(1, 1))
        self.region3_Loop = pg.LinearRegionItem((120, 200), brush=pg_qt.QtGui.QBrush(pg_qt.QtGui.QColor(255, 0, 0, 50)))
        self.imv10.addItem(self.region3_Loop, ignoreBounds=True)
        label = pg.InfLineLabel(self.region3_Loop.lines[1], "Loop region",
                                position=0.75, rotateAxis=(1,0), anchor=(1, 1))        
        self.infline_left = pg.InfiniteLine(movable=True, angle=90, pen=(3, 9), label='x={value:0.0f}',
            labelOpts={'position':0.75, 'color': (200,200,100), 'fill': (200,200,200,25), 'movable': True})
        self.infline_left.addMarker(marker='v', position=0.7, size=10)
        self.infline_left.addMarker(marker='^', position=0.3, size=10)
        self.imv10.addItem(self.infline_left)

        self.imv20 = pg.ImageView()
        self.imv20.setPredefinedGradient(DEFAULTS["ColorMap"])
        self.hide_imgv_cmap(self.imv20)
        self.plotLoopPos = pg.PlotItem(name='Loop Position')
        self.plotLoopPos.hideAxis('left')
        self.imv21 = pg.ImageView(view=self.plotLoopPos)
        self.imv21.setPredefinedGradient(DEFAULTS["ColorMap"])
        self.hide_imgv_roi_norm(self.imv21)
        

        self.d0_left = pg_da.Dock("d0left - DNA")
        self.d0_left.addWidget(self.imv00)
        self.d1_left = pg_da.Dock("d1left-Kymograph DNA")
        self.d1_left.addWidget(self.imv10)
        self.d2_left = pg_da.Dock("d2left-Kymograph noLoop and Loop")
        self.d2_left.addWidget(self.imv20, 0, 0, 1, 1)
        self.d2_left.addWidget(self.imv21, 0, 1, 1, 5)

        self.dockarea.addDock(self.d0_left, 'left')
        self.dockarea.addDock(self.d1_left, 'bottom', self.d0_left)
        self.dockarea.addDock(self.d2_left, 'bottom', self.d1_left)        
        self.region3_noLoop.sigRegionChanged.connect(self.region_noLoop_changed)
        self.region3_Loop.sigRegionChanged.connect(self.region_Loop_changed)
        # initiation of otherplots
        self.plot_loop_errbar = None

    def add_col2_imvs(self):
        self.imv01 = pg.ImageView(name='color 2')
        self.imv01.setPredefinedGradient(DEFAULTS["ColorMap"])
        self.hide_imgv_roi_norm(self.imv01)
        self.imv01.playRate = 7
        self.roirect_right = pg.LineROI([20, 20], [40, 20], width=11, pen=(3, 9))        
        self.imv01.addItem(self.roirect_right)

        self.plot4 = pg.PlotItem(name='red_kymo')
        self.plot4.hideAxis('left')#; self.plot4.hideAxis('bottom')
        self.plot4.setXLink(self.plot3); self.plot4.setYLink(self.plot3)
        self.imv11 = pg.ImageView(view=self.plot4)
        self.imv11.setPredefinedGradient(DEFAULTS["ColorMap"])
        self.hide_imgv_cmap(self.imv11)
        self.infline_right = pg.InfiniteLine(movable=True, angle=90, pen=(3, 9))
        self.infline_right.addMarker(marker='v', position=0.7, size=10)
        self.infline_right.addMarker(marker='^', position=0.3, size=10)
        self.imv11.addItem(self.infline_right)
        
        self.imv22 = pg.ImageView()
        self.imv22.setPredefinedGradient(DEFAULTS["ColorMap"])
        self.hide_imgv_cmap(self.imv22)
        self.imv23 = pg.ImageView()
        self.imv23.setPredefinedGradient(DEFAULTS["ColorMap"])
        self.hide_imgv_roi_norm(self.imv23)

        self.d0_right = pg_da.Dock("d0right-single molecule")
        self.d0_right.addWidget(self.imv01)
        self.d1_right = pg_da.Dock("d1right-Kymograph single molecule")
        self.d1_right.addWidget(self.imv11)
        self.d2_right = pg_da.Dock("d2right-single molecule on NoLoop and Loop")
        self.d2_right.addWidget(self.imv22, 0, 0, 1, 1)
        self.d2_right.addWidget(self.imv23, 0, 1, 1, 3)

        self.dockarea.addDock(self.d0_right, 'right')
        self.dockarea.addDock(self.d1_right, 'bottom', self.d0_right)
        self.dockarea.addDock(self.d2_right, 'bottom', self.d1_right)

    def remove_all_widgets(self):
        try:
            self.d0_left.close()
            self.d1_left.close()
            self.d2_left.close()
            self.d0_right.close()
            self.d1_right.close()
            self.d2_right.close()
            if self.plot_loop_errbar is not None:
                self.d3_left.close()
                self.d3_right.close()
        except: print('already removed')

    def restore_default_dockstate(self):
        self.dockarea.restoreState(self.defaultDockState)
        self.imv10.showMaximized()
        if self.numColors == "2":
            self.imv11.showMaximized()

    def set_scalebar(self):
        self.pixelSize = 1e-3 * self.ui.pixelSizeSpinBox.value()
        self.acquisitionTime = 1e-3 * self.ui.AcquisitionTimeSpinBox_2.value() # converted to sec from ms
        if self.scalebar_img is not None:
            self.scalebar_img.size = 2/self.pixelSize
            self.scalebar_img.updateBar()
            if self.numColors == "2":
                self.scalebar_img2.size = 2/self.pixelSize
                self.scalebar_img2.updateBar()
                self.scalebar_kymoloop_right.size = 10/self.acquisitionTime
                self.scalebar_kymoloop_right.updateBar()
        else:
            self.scalebar_img = pg.ScaleBar(size=2/self.pixelSize, suffix='um') #self.pixelSize in um
            self.scalebar_img.text.setText('2 \u03BCm')
            self.scalebar_img.setParentItem(self.imv00.view)
            self.scalebar_img.anchor((1, 1), (1, 1), offset=(-40, -40))
            ## kymo (proper scale can't be added in a plot)
            # scalebar_kymo_left = pg.ScaleBar(20/self.acquisitionTime, suffix='s')
            # scalebar_kymo_left.text.setText('20 sec')
            # scalebar_kymo_left.setParentItem(self.imv10.)
            # scalebar_kymo_left.anchor((1, 1), (1, 1), offset=(-100, -100))
            # kymo loop 
            # scalebar_kymoloop_left = pg.ScaleBar(10/self.acquisitionTime, suffix='s')#self.acquisitionTime in sec
            # scalebar_kymoloop_left.text.setText('10 sec')
            # scalebar_kymoloop_left.setParentItem(self.imv21.view)
            # scalebar_kymoloop_left.anchor((1, 1), (1, 1), offset=(-40, -40))
            if self.numColors == "2":
                self.scalebar_img2 = pg.ScaleBar(size=2/self.pixelSize, suffix='um') #2 um
                self.scalebar_img2.text.setText('2 um')
                self.scalebar_img2.setParentItem(self.imv01.view)
                self.scalebar_img2.anchor((1, 1), (1, 1), offset=(-40, -40))            
                # kymo
                # scalebar_kymo_right = pg.ScaleBar(20/self.acquisitionTime, suffix='s')
                # scalebar_kymo_right.text.setText('20 sec')
                # scalebar_kymo_right.setParentItem(self.plot4)
                # scalebar_kymo_right.anchor((1, 1), (1, 1), offset=(-100, -100))
                # kymo loop right side
                self.scalebar_kymoloop_right = pg.ScaleBar(10/self.acquisitionTime, suffix='s')
                self.scalebar_kymoloop_right.text.setText('10 sec')
                self.scalebar_kymoloop_right.setParentItem(self.imv23.view)
                self.scalebar_kymoloop_right.anchor((1, 1), (1, 1), offset=(-40, -40))

    def load_img_stack(self):
        filepath = sg.tkinter.filedialog.askopenfilename(title = "Select tif file/s",
                                                        filetypes = (("tif files","*.tif"),("all files","*.*")))
        self.filepath = filepath
        self.image_meta = read_img_stack(self.filepath)
        self.frame_start = 0
        self.frame_end = -1
        self.set_img_stack()
        self.load_yaml_params()
        self.set_yaml_params()
        if self.ui.processImageCheckBox.isChecked():
            self.image_meta = self.get_processed_image()
            self.set_img_stack()

    def set_img_stack(self):
        print("Loading and processing the image ...")
        start_time = time.time()
        if self.numColors == "2":
            self.imgarr_left = self.image_meta['img_arr_color_0'][self.frame_start:self.frame_end, ...]
            self.imgarr_right = self.image_meta['img_arr_color_1'][self.frame_start:self.frame_end, ...]
        elif self.numColors == "1":
            self.imgarr_left = self.image_meta['img_arr_color_0'][self.frame_start:self.frame_end, ...]
        self.imv00.setImage(self.imgarr_left)
        self.imv00.showMaximized()
        if self.numColors == "2":
            if self.ui.mergeColorsCheckBox.isChecked():
                arr_combined = np.concatenate((self.imgarr_right[:, :, :, np.newaxis],
                                            self.imgarr_left[:, :, :, np.newaxis],
                                            np.zeros_like(self.imgarr_right[:, :, :, np.newaxis])),
                                            axis=3)
                self.imv01.setImage(arr_combined, levelMode='rgba')
            else:
                self.imv01.setImage(self.imgarr_right)
            self.imv01.showMaximized()
        self.roi_changed()
        self.region_Loop_changed()
        self.region_noLoop_changed()
        self.set_scalebar()
        print("took %s seconds!" % (time.time() - start_time))
        print("the loading is DONE!")

    def load_img_seq(self):
        return

    def load_yaml_params(self):
        self.folderpath = os.path.dirname(self.filepath)
        filename = os.path.basename(self.filepath)
        (self.filename_base, ext) = os.path.splitext(filename)
        self.filepath_yaml = os.path.join(self.folderpath, self.filename_base + '_params.yaml')
        if os.path.isfile(self.filepath_yaml):
            with open(self.filepath_yaml) as f:
                self.params_yaml = yaml.load(f)
        else:
            with open(self.filepath_yaml, 'w') as f:
                self.params_yaml = {
                    'filepath' : self.filepath,
                    'folderpath' : self.folderpath,
                    'Pixel Size' : self.ui.pixelSizeSpinBox.value(),
                    'ROI width' : None,
                    'Region Errbar': [10, 30],
                }
                yaml.dump(self.params_yaml, f)
        return self.params_yaml
    
    def save_yaml_params(self):
        self.params_yaml = self.load_yaml_params()
        self.params_yaml['Pixel Size'] = self.ui.pixelSizeSpinBox.value()
        self.params_yaml['ROI width'] = self.ui.currROIWidthLabel.text()
        self.params_yaml['roi1 state'] = {}
        self.params_yaml['roi1 state']['position'] = list(self.roirect_left.pos()) #[self.roirect_left.pos()[0], self.roirect_left.pos()[1]]
        self.params_yaml['roi1 state']['size'] = list(self.roirect_left.size())
        self.params_yaml['roi1 state']['angle'] = float(self.roirect_left.angle())
        self.params_yaml['region3_noLoop'] = list(self.region3_noLoop.getRegion())
        self.params_yaml['region3_Loop'] = list(self.region3_Loop.getRegion())
        if self.plot_loop_errbar is not None:
            self.params_yaml['Region Errbar'] = list(self.region_errbar.getRegion())

        with open(self.filepath_yaml, 'w') as f:
            yaml.dump(self.params_yaml, f)

    def set_yaml_params(self):
        if self.params_yaml['ROI width'] is not None:
            self.roirect_left.setPos(self.params_yaml['roi1 state']['position'])
            self.roirect_left.setSize(self.params_yaml['roi1 state']['size'])
            self.roirect_left.setAngle(self.params_yaml['roi1 state']['angle'])

            self.region3_Loop.setRegion(self.params_yaml['region3_Loop'])
            self.region3_Loop.setRegion(self.params_yaml['region3_Loop'])

    def processed_image_check(self):
        if self.ui.processImageCheckBox.isChecked():
            self.image_meta = self.get_processed_image()
            self.set_img_stack()
        else:
            self.image_meta = read_img_stack(self.filepath)
            self.set_img_stack()

    def get_processed_image(self):
        fpath_processed = os.path.join(self.folderpath, self.filename_base + '_processed.tif')
        if os.path.isfile(fpath_processed):
            self.image_meta = read_img_stack(fpath_processed)
        else:
            if self.numColors == "2":
                self.imgarr_left = median_bkg_substration(self.imgarr_left)
                self.imgarr_right = median_bkg_substration(self.imgarr_right)
                comb_arr = np.concatenate((self.imgarr_left[:,np.newaxis,:,:],
                                           self.imgarr_right[:,np.newaxis,:,:]),
                                           axis=1)
                imwrite(fpath_processed, comb_arr, imagej=True,
                        metadata={'axis': 'TCYX', 'channels': self.numColors,
                        'mode': 'composite',})
            elif self.numColors == "1":
                self.imgarr_left = median_bkg_substration(self.imgarr_left)
                imwrite(fpath_processed, self.imgarr_left, imagej=True,
                        metadata={'axis': 'TCYX', 'channels': self.numColors,
                        'mode': 'composite',})
            self.image_meta = read_img_stack(fpath_processed)
        return self.image_meta

    def update_kymo(self):
        ROIwidth = self.roirect_left.getState()['size'][1]
        self.ui.currROIWidthLabel.setText(str(round(ROIwidth, 2)))
        roi1_data = self.roirect_left.getArrayRegion(self.imgarr_left,
                                            self.imv00.imageItem, axes=(1, 2))
        self.kymo_left = np.sum(roi1_data, axis=2)
        self.kymo_left = self.kymo_left / np.max(self.kymo_left)
        self.imv10.setImage(self.kymo_left)
        if self.numColors == "2":
            self.roirect_right.setState(self.roirect_left.getState())
            roi2_data = self.roirect_right.getArrayRegion(self.imgarr_right,
                                                self.imv01.imageItem, axes=(1, 2))                                                       
            self.kymo_right = np.sum(roi2_data, axis=2)
            self.kymo_right = self.kymo_right / np.max(self.kymo_right)
            if self.ui.mergeColorsCheckBox.isChecked():
                kymo_comb = np.concatenate((self.kymo_right[:, :, np.newaxis],
                                        self.kymo_left[:, :, np.newaxis],
                                        np.zeros_like(self.kymo_right[:, :, np.newaxis])),
                                        axis=2)
                self.imv11.setImage(kymo_comb, levelMode='rgba')
                self.imv11.ui.histogram.show()
            else:
                self.imv11.setImage(self.kymo_right)

    def sync_videos(self):
        frame_imv00 = self.imv00.currentIndex
        self.imv01.setCurrentIndex(frame_imv00)

    def roi_changed(self):
        # ROIwidth = self.roirect_left.getState()['size'][1]
        # self.ui.roiWidthSpinBox.setValue(ROIwidth)
        if self.ui.RealTimeKymoCheckBox.isChecked():
            self.update_kymo()
            self.region_noLoop_changed()
            self.region_Loop_changed()
        # also clear the loop position data point
        if self.plot_loop_errbar is not None:
            self.plotLoopPosData.clear()

    def roi_changed_disconnect(self):
        pass

    def realtime_kymo(self):
        if self.ui.RealTimeKymoCheckBox.isChecked():
            self.roi_changed()
        else:
            self.roirect_left.sigRegionChanged.connect(self.roi_changed_disconnect)

    def set_roi_width(self):
        roi1_state_update = self.roirect_left.getState()
        roi1_state_update['size'][1] = self.ui.roiWidthSpinBox.value()
        self.roirect_left.setState(roi1_state_update)

    def infiline_left_update(self):
        frame_numer = int(self.infline_left.value())
        if frame_numer >= 0:
            self.imv00.setCurrentIndex(frame_numer)
            if self.numColors == "2":
                self.imv01.setCurrentIndex(frame_numer)
                pos = self.infline_left.getPos()
                self.infline_right.setPos(pos)

    def infiline_right_update(self):
        frame_numer = int(self.infline_right.value())
        if frame_numer >= 0:
            self.imv00.setCurrentIndex(frame_numer)
            self.imv01.setCurrentIndex(frame_numer)
            pos = self.infline_right.getPos()
            self.infline_left.setPos(pos)

    def hide_imgv_cmap(self, imgv):
        imgv.ui.roiBtn.hide()
        imgv.ui.menuBtn.hide()
        imgv.ui.histogram.hide()

    def hide_imgv_roi_norm(self, imgv):
        # hides the roi and norm tabs from the cmap widget
        # and makes it more compact
        imgv.ui.roiBtn.hide()
        imgv.ui.menuBtn.hide()
        # self.imv00.ui.histogram.

    def change_num_colors(self):
        self.numColors = self.ui.numColorsComboBox_3.currentText()
        if self.numColors == "1":
            self.remove_all_widgets()
            self.add_col1_imvs()
            self.connect_signals()
        elif self.numColors == "2":
            # self.remove_all_widgets()
            # self.add_col1_imvs()
            self.add_col2_imvs()
            self.connect_signals()
        self.defaultDockState = self.dockarea.saveState()
        
    def region_noLoop_changed(self):
        minX, maxX = self.region3_noLoop.getRegion()
        self.kymo_left_noLoop = self.kymo_left[int(minX):int(maxX), :]
        self.imv20.setImage(self.kymo_left_noLoop)
        if self.numColors == "2":
            self.kymo_right_noLoop = self.kymo_right[int(minX):int(maxX), :]
            if self.ui.mergeColorsCheckBox.isChecked():
                kymo_noLoop_comb = np.concatenate((self.kymo_right_noLoop[:, :, np.newaxis],
                                        self.kymo_left_noLoop[:, :, np.newaxis],
                                        np.zeros_like(self.kymo_right_noLoop[:, :, np.newaxis])),
                                        axis=2)
                self.imv22.setImage(kymo_noLoop_comb, levelMode='rgba')
            else:
                self.imv22.setImage(self.kymo_right_noLoop)

    def region_Loop_changed(self):
        minX, maxX = self.region3_Loop.getRegion()
        self.kymo_left_loop = self.kymo_left[int(minX):int(maxX), :]
        self.imv21.setImage(self.kymo_left_loop)
        if self.numColors == "2":
            self.kymo_right_loop = self.kymo_right[int(minX):int(maxX), :]
            if self.ui.mergeColorsCheckBox.isChecked():
                kymo_loop_comb = np.concatenate((self.kymo_right_loop[:, :, np.newaxis],
                                        self.kymo_left_loop[:, :, np.newaxis],
                                        np.zeros_like(self.kymo_right_loop[:, :, np.newaxis])),
                                        axis=2)
                self.imv23.setImage(kymo_loop_comb, levelMode='rgba')
            else:
                self.imv23.setImage(self.kymo_right_loop)
        # also clear the loop position data point
        if self.plot_loop_errbar is not None:
            self.plotLoopPosData.clear()

    def merge_colors(self):
        # rgba images need to have 3 colors (arr[t, x, y, c]). c must be 3 
        if self.ui.mergeColorsCheckBox.isChecked():
            arr_combined = np.concatenate((self.imgarr_right[:, :, :, np.newaxis],
                                        self.imgarr_left[:, :, :, np.newaxis],
                                        np.zeros_like(self.imgarr_right[:, :, :, np.newaxis])),
                                        axis=3)
            self.imv01.setImage(arr_combined, levelMode='rgba')
            self.imv01.showMaximized()
            kymo_comb = np.concatenate((self.kymo_right[:, :, np.newaxis],
                                    self.kymo_left[:, :, np.newaxis],
                                    np.zeros_like(self.kymo_right[:, :, np.newaxis])),
                                    axis=2)
            self.imv11.setImage(kymo_comb, levelMode='rgba')
            self.imv11.ui.histogram.show()
            kymo_loop_comb = np.concatenate((self.kymo_right_loop[:, :, np.newaxis],
                                    self.kymo_left_loop[:, :, np.newaxis],
                                    np.zeros_like(self.kymo_right_loop[:, :, np.newaxis])),
                                    axis=2)
            kymo_noLoop_comb = np.concatenate((self.kymo_right_noLoop[:, :, np.newaxis],
                                    self.kymo_left_noLoop[:, :, np.newaxis],
                                    np.zeros_like(self.kymo_right_noLoop[:, :, np.newaxis])),
                                    axis=2)
            self.imv22.setImage(kymo_noLoop_comb, levelMode='rgba')
            self.imv23.setImage(kymo_loop_comb, levelMode='rgba')
        else:
            # set back the imagedata
            self.imv01.setImage(self.imgarr_right, levelMode='mono')
            self.imv01.showMaximized()
            self.imv11.setImage(self.kymo_right, levelMode='mono')
            self.imv22.setImage(self.kymo_right_noLoop, levelMode='mono')
            self.imv23.setImage(self.kymo_right_loop, levelMode='mono')
        return

    def swap_colors(self):
        if self.numColors == "2" and self.ui.swapColorsCheckBox.isChecked():
            temp_arr = self.image_meta['img_arr_color_0']
            self.image_meta['img_arr_color_0'] = self.image_meta['img_arr_color_1']
            self.image_meta['img_arr_color_1'] = temp_arr
            self.set_img_stack()
        else:
            self.set_img_stack()

    def set_loop_detection_widgets(self):
        # set the docking positions to default
        self.dockarea.restoreState(self.defaultDockState)
        
        # setting the docking and plot positions for errorbar and loop kinetics
        self.plot_loop_errbar = pg.PlotWidget()
        self.plot_loop_errbar.plotItem.setLabels(left='std(Intensity)', bottom='pixels')
        x = self.plot_loop_errbar.getPlotItem()
        self.plot_loop_kinetics = pg.PlotWidget()
        self.d3_left = pg_da.Dock("Loop detections")
        self.d3_left.addWidget(self.plot_loop_errbar, 0, 0)
        self.d3_left.addWidget(self.plot_loop_kinetics, 0, 1)
        self.dockarea.addDock(self.d3_left, 'bottom', self.d2_left)
        if self.numColors == "2":
            self.d3_right = pg_da.Dock("Single Molecule detections")
            self.dockarea.addDock(self.d3_right, 'bottom', self.d2_right)
            self.plot_loop_vs_sm = pg.PlotWidget()
            self.d3_right.addWidget(self.plot_loop_vs_sm)
            self.plot_loop_vs_sm_smdata = self.plot_loop_vs_sm.plot(title='SM',
                symbol='o', symbolSize=4, pen=pg.mkPen(None),
                symbolPen=pg.mkPen(None), symbolBrush='r')
            self.plot_loop_vs_sm_loopdata = self.plot_loop_vs_sm.plot(title='loop',
                symbol='o', symbolSize=4, pen=pg.mkPen(None),
                symbolPen=pg.mkPen(None), symbolBrush='g')
            self.plot_loop_vs_sm_linetop = pg.InfiniteLine(
                                    movable=False, angle=0, pen=(3, 9))
            self.plot_loop_vs_sm_linebottom = pg.InfiniteLine(
                                    movable=False, angle=0, pen=(3, 9))
            self.plot_loop_vs_sm_smdata.getViewBox().invertY(True)
        # adding errorbar plot items for data updating later
        self.errbar_loop = pg.ErrorBarItem(beam=0.5, pen=pg.mkPen('r'))
        self.plot_loop_errbar.addItem(self.errbar_loop)
        self.errdata_loop = self.plot_loop_errbar.plot(symbol='o', symbolSize=2, pen=pg.mkPen('r'))
        self.errbar_noLoop = pg.ErrorBarItem(beam=0.5, pen=pg.mkPen('b'))
        self.plot_loop_errbar.addItem(self.errbar_noLoop)
        self.errdata_noLoop = self.plot_loop_errbar.plot(symbol='o', symbolSize=2, pen=pg.mkPen('b'))
        legend = pg.LegendItem((80,60), offset=(70,20))
        legend.setParentItem(self.plot_loop_errbar.getPlotItem())
        legend.addItem(self.errdata_loop, 'dna with loop')
        legend.addItem(self.errdata_noLoop, 'dna with No loop')

        self.region_errbar = pg.LinearRegionItem(self.params_yaml['Region Errbar'])
        self.plot_loop_errbar.addItem(self.region_errbar, ignoreBounds=True)
        # adding plot items for loop kinetics
        self.plot_data_loop = self.plot_loop_kinetics.plot(title='Loop',
                symbol='o', symbolSize=4, pen=pg.mkPen(None),
                symbolPen=pg.mkPen(None), symbolBrush='g')
        self.plot_data_loopUp = self.plot_loop_kinetics.plot(title='Loop Up',
                symbol='o', symbolSize=4, pen=pg.mkPen(None),
                symbolPen=pg.mkPen(None), symbolBrush='r')
        self.plot_data_loopDown = self.plot_loop_kinetics.plot(title='Loop Down',
                symbol='o', symbolSize=4, pen=pg.mkPen(None),
                symbolPen=pg.mkPen(None), symbolBrush='b')
        self.plot_sm_dist = self.plot_loop_kinetics.plot(title='Loop Smoothed', pen=[255,0,255, 255])
        self.plot_data_loop_filt = self.plot_loop_kinetics.plot(title='Loop Smoothed', pen=pg.mkPen('g', width=2))
        self.plot_data_loopUp_filt = self.plot_loop_kinetics.plot(title='Loop Up Smoothed', pen='r')
        self.plot_data_loopDown_filt = self.plot_loop_kinetics.plot(title='Loop Down Smoothed', pen='b')
        self.plot_loop_kinetics.plotItem.setLabels(left='DNA/kb', bottom='Frame Number')
        legend = pg.LegendItem((80,60), offset=(150,20))
        legend.setParentItem(self.plot_loop_kinetics.plotItem)
        legend.addItem(self.plot_data_loop, 'Loop')
        legend.addItem(self.plot_data_loopUp, 'Up')
        legend.addItem(self.plot_data_loopDown, 'Down')
        # loop position in imv21
        self.plotLoopPosData = self.plotLoopPos.scatterPlot(
            symbol='o', symbolSize=5, pen=pg.mkPen('r'))#symbolBrush=pg.mkPen('r')
        # change the default docking positions to the new one
        self.defaultDockState = self.dockarea.saveState()
        self.dockarea.restoreState(self.defaultDockState)

    def detect_loops(self):
        if self.plot_loop_errbar is None:
            self.set_loop_detection_widgets()
        self.d3_left.setStretch(200, 200)
        if self.numColors == "2":
            self.d3_right.setStretch(200, 200)
        # set errorbars for loop positions
        loop_std = np.std(self.kymo_left_loop, axis=0)
        loop_avg = np.average(self.kymo_left_loop, axis=0)
        self.errbar_loop.setData(x=np.arange(len(loop_avg)),
                                y=loop_avg, height=loop_std)
        self.errdata_loop.setData(loop_avg)#, pen={'color': 0.8, 'width': 2}

        noLoop_std = np.std(self.kymo_left_noLoop, axis=0)
        noLoop_avg = np.average(self.kymo_left_noLoop, axis=0)
        self.errbar_noLoop.setData(x=np.arange(len(noLoop_avg)),
                                   y=noLoop_avg, height=noLoop_std)
        self.errdata_noLoop.setData(noLoop_avg)
        # detect loop position
        loop_region_left = int(self.region_errbar.getRegion()[0])
        loop_region_right = int(self.region_errbar.getRegion()[1])
        pix_width = 11
        peak_dict = peakfinder_savgol(self.kymo_left_loop.T,
                loop_region_left, -loop_region_right,
                prominence_min=1/3, pix_width=pix_width, plotting=False,
                kymo_noLoop=self.kymo_left_noLoop.T, #use this carefully, safe way is to put it none or just comment this line
                )
        peak_dict = analyze_maxpeak(peak_dict['Max Peak'], smooth_length=7,
                frame_width = loop_region_right-loop_region_left,
                dna_length=48, pix_width=pix_width,
                )
        frame_no = peak_dict["Max Peak"]["FrameNumber"]
        peak_pos = peak_dict["Max Peak"]["PeakPosition"]
        int_peak = peak_dict["Max Peak"]["PeakIntensity"]
        int_peakup = peak_dict["Max Peak"]["PeakUpIntensity"]
        int_peakdown = peak_dict["Max Peak"]["PeakDownIntensity"]
        self.plotLoopPosData.setData(frame_no, peak_pos)
        self.plot_data_loop.setData(frame_no, int_peak)
        self.plot_data_loopUp.setData(frame_no, int_peakup)
        self.plot_data_loopDown.setData(frame_no, int_peakdown)
        self.plot_data_loop_filt.setData(frame_no,
                        peak_dict["Max Peak"]["PeakIntFiltered"])
        self.plot_data_loopUp_filt.setData(frame_no,
                        peak_dict["Max Peak"]["PeakIntUpFiltered"])
        self.plot_data_loopDown_filt.setData(frame_no,
                        peak_dict["Max Peak"]["PeakIntDownFiltered"])
        if self.numColors == "2":
            smpeak_dict = peakfinder_savgol(self.kymo_right_loop.T,
                loop_region_left, -loop_region_right,
                prominence_min=1/2, pix_width=pix_width, plotting=True,
                # kymo_noLoop=self.kymo_left_noLoop.T, #use this carefully, safe way is to put it none or just comment this line
                )
            loop_sm_dict = loop_sm_dist(peak_dict, smpeak_dict, smooth_length=21)
            self.plot_sm_dist.setData(loop_sm_dict['FrameNumver'],
                                      loop_sm_dict['PeakDiffFiltered'])
            # update data in smVsLoop plot
            self.plot_loop_vs_sm_smdata.setData(
                    smpeak_dict['Max Peak']['FrameNumber'],
                    smpeak_dict['Max Peak']['PeakPosition'])
            self.plot_loop_vs_sm_loopdata.setData(
                    peak_dict['Max Peak']['FrameNumber'],
                    peak_dict['Max Peak']['PeakPosition'])
            self.plot_loop_vs_sm.setYRange(loop_region_left-5, loop_region_right+5)
            self.plot_loop_vs_sm_linetop.setValue(loop_region_left)
            self.plot_loop_vs_sm_linebottom.setValue(loop_region_right)

    def save_section(self):
        temp_folder = os.path.abspath(os.path.join(self.folderpath, 'temp'))
        print(temp_folder)
        if not os.path.isdir(temp_folder):
            os.mkdir(temp_folder)
        if self.ui.saveSectionComboBox.currentText() == "d0left":
            roi_state = self.roirect_left.getState()
            self.roirect_left.setPos((-100, -100)) # move away from the imageItem
            print("Converting to video ...")
            i = 0
            while i < self.imgarr_left.shape[0]:
                self.imv00.setCurrentIndex(i)
                exporter = pyqtgraph.exporters.ImageExporter(self.imv00.imageItem)
                exporter.export(os.path.join(temp_folder, 'temp_'+str(i)+'.png'))
                # self.imv00.jumpFrames(1)
                i += 1
            self.roirect_left.setState(roi_state) #set back to its previous state
            filelist = glob.glob(temp_folder+'/temp_*.png')
            filename = self.folderpath+'/'+self.filename_base + 'left.avi'
            os.chdir(temp_folder)
            subprocess.call(['ffmpeg', '-y', '-r', '10', '-i', 'temp_%d0.png', filename])
            for file in filelist:
                os.remove(file)
            print("Video conversion FINISHED")
        elif self.ui.saveSectionComboBox.currentText() == "d0right":
            roi_state = self.roirect_right.getState()
            self.roirect_right.setPos((-100, -100)) # move away from the imageItem
            print("Converting to video ...")
            i = 0
            while i < self.imgarr_left.shape[0]:
                self.imv01.setCurrentIndex(i)
                exporter = pyqtgraph.exporters.ImageExporter(self.imv01.imageItem)
                exporter.export(os.path.join(temp_folder, 'temp_'+str(i)+'.png'))
                # self.imv00.jumpFrames(1)
                i += 1
            self.roirect_right.setState(roi_state) #set back to its previous state
            filelist = glob.glob(temp_folder+'/temp_*.png')
            filename = self.folderpath+'/'+self.filename_base + 'left.avi'
            os.chdir(temp_folder)
            subprocess.call(['ffmpeg', '-y', '-r', '10', '-i', 'temp_%d0.png', filename])
            for file in filelist:
                os.remove(file)
            print("Video conversion FINISHED")


    def frames_changed(self):
        print("Changing the frames and resetting plts...")
        start_time = time.time()
        self.frame_start = self.ui.frameStartSpinBox.value()
        self.frame_end = self.ui.frameEndSpinBox.value()
        self.set_img_stack()
        print("took %s seconds to reset!" % (time.time() - start_time))
        print("DONE:Changing the frames.")

class ParametersDialog(QtWidgets.QDialog):
    """ The dialog showing analysis parameters """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.window = parent
        self.setWindowTitle("Parameters")
        self.resize(300, 0)
        self.setModal(False)

        vbox = QtWidgets.QVBoxLayout(self)
        general_groupbox = QtWidgets.QGroupBox("General")
        vbox.addWidget(general_groupbox)
        general_grid = QtWidgets.QGridLayout(general_groupbox)

        # Pixel Size
        general_grid.addWidget(QtWidgets.QLabel("Pixel Size:"), 0, 0)
        self.pix_spinbox = QtWidgets.QSpinBox()
        self.pix_spinbox.setRange(0, 1e3)
        self.pix_spinbox.setSuffix(" nm")
        self.pix_spinbox.setValue(DEFAULT_PARAMETERS["Pixel Size"])
        self.pix_spinbox.setKeyboardTracking(False)
        # self.pix_spinbox.valueChanged.connect()
        general_grid.addWidget(self.pix_spinbox, 0, 1)

        # ROI Width
        general_grid.addWidget(QtWidgets.QLabel("ROI Size:"), 1, 0)
        self.roi_spinbox = QtWidgets.QSpinBox()
        self.roi_spinbox.setRange(0, 1e3)
        self.roi_spinbox.setSuffix(" pixels")
        self.roi_spinbox.setValue(DEFAULT_PARAMETERS["ROI Width"])
        self.roi_spinbox.setKeyboardTracking(False)
        # self.roi_spinbox.valueChanged.connect()
        general_grid.addWidget(self.roi_spinbox, 1, 1)

        # Acquisition Time
        general_grid.addWidget(QtWidgets.QLabel("Acquisition time:"), 2, 0)
        self.aqt_spinbox = QtWidgets.QSpinBox()
        self.aqt_spinbox.setRange(0, 1e5)
        self.aqt_spinbox.setSuffix(" ms")
        self.aqt_spinbox.setValue(DEFAULT_PARAMETERS["Acquisition Time"])
        # self.aqt_spinbox.valueChanged.connect()
        general_grid.addWidget(self.aqt_spinbox, 2, 1)


class Window(QtWidgets.QMainWindow):
    """ The main window """

    def __init__(self):
        super().__init__()
        # Init GUI
        self.setWindowTitle("LEADS : Kymograph Analysis")
        this_directory = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(this_directory, "assets", "kymograph_window_bar.png")
        icon = QtGui.QIcon(icon_path)
        self.setWindowIcon(icon)
        self.view = kymographGui()
        self.setCentralWidget(self.view)
        self.parameters_dialog = ParametersDialog(self)
        self.init_menu_bar()

    def init_menu_bar(self):
        menu_bar = self.menuBar()

        """ File """
        file_menu = menu_bar.addMenu("File")
        open_action = file_menu.addAction("Open image stack")
        open_action.setShortcut("Ctrl+O")
        """ View """
        view_menu = menu_bar.addMenu("View")
        default_view_state = view_menu.addAction("Default View State")

        """ Analyze """
        analyze_menu = menu_bar.addMenu("Analyze")
        parameters_action = analyze_menu.addAction("Parameters")
        parameters_action.setShortcut("Ctrl+P")
        parameters_action.triggered.connect(self.parameters_dialog.show)

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = Window()
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    win.show()
    sys.exit(app.exec_())
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtWidgets.QApplication.instance().exec_()

if __name__ == '__main__':
    main()
    # app = pg_qt.QtGui.QApplication([])
    # ui = kymographGui()
    # ui.show()
    # if (sys.flags.interactive != 1) or not hasattr(pg_qt.QtCore, 'PYQT_VERSION'):
    #     pg_qt.QtGui.QApplication.instance().exec_()