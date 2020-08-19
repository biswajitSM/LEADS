import napari
import numpy as np
from .crop_images_ui import Ui_Form
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import PySimpleGUI as sg

from ..crop_images import (
    daskread_img_seq, crop_rect_shapes,
    addroi_to_shapelayer, save_rectshape_as_imageJroi)

class NapariTabs(QtWidgets.QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.viewer = viewer

        # number of colors
        self.ui.numColorsCbox.setCurrentText("2")
        self.num_colors = self.ui.numColorsCbox.currentText()
        self.ui.numColorsCbox.currentIndexChanged.connect(self.change_num_colors)
        # frame number
        self.frame_start = self.ui.frameStartSbox.value()
        self.frame_end = self.ui.frameEndBox.value()        
        if self.frame_end == -1:
            self.frame_end = None
        # processing
        self.ui.loadImageBtn.clicked.connect(self.load_img_seq)
        self.ui.defaultFramesBtn.clicked.connect(self.set_default_frame_num)
        self.ui.cropSelectedBtn.clicked.connect(self.crop)
        self.ui.loadImageJroisBtn.clicked.connect(self.load_imageJ_rois)
        self.ui.saveImageJroisBtn.clicked.connect(self.save_imageJ_rois)

    def load_img_seq(self):
        self.image_meta = daskread_img_seq(num_colors=int(self.ui.numColorsCbox.currentText()))
        for l in reversed(self.viewer.layers[:]):
            self.viewer.layers.remove(l)
        color_list = ['green', 'red', 'blue']
        color_numer = 0
        while color_numer <= self.image_meta['num_colors']-1 and color_numer <= 5:
            self.viewer.add_image(self.image_meta['stack_color_'+str(color_numer)], colormap=color_list[color_numer],
                    contrast_limits=[self.image_meta['min_int_color_'+str(color_numer)],self.image_meta['max_int_color_'+str(color_numer)]],
                    blending='additive', multiscale=False, name='color_'+str(color_numer))
            color_numer += 1
        shp_to_add = np.array([[10, 0], [80, 0], [80, 50], [10, 50]])
        self.viewer.add_shapes(shp_to_add, shape_type='rectangle', name='rect_roi',
                        edge_color='yellow', edge_width=5, opacity=0.2)
        self.viewer.layers['rect_roi'].mode = 'select'

    def crop(self):
        try:
            self.image_meta
            self.frame_start = self.ui.frameStartSbox.value()
            self.frame_end = self.ui.frameEndBox.value()
            if self.frame_end == -1:
                self.frame_end = None
            shape_layer = self.viewer.layers['rect_roi']
            crop_rect_shapes(self.image_meta, shape_layer,
                            frame_start=self.frame_start, frame_end=self.frame_end,
                            )
        except:
            self.load_img_seq()

    def set_default_frame_num(self):
        self.ui.frameStartSbox.setProperty("value", 0)
        self.ui.frameEndBox.setProperty("value", -1)

    def change_num_colors(self):
        self.num_colors = self.ui.numColorsCbox.currentText()

    def load_imageJ_rois(self):
        roi_file_list = sg.tkinter.filedialog.askopenfilenames(
            title = "Select ROI files", 
            filetypes = (("ROI files","*.roi"),("all files","*.*")))
        _ = addroi_to_shapelayer(viewer.layers['rect_roi'], roi_file_list)

    def save_imageJ_rois(self):
        shape_layer = self.viewer.layers['rect_roi']
        save_rectshape_as_imageJroi(shape_layer)

if __name__ == "__main__":
    with napari.gui_qt():
        viewer = napari.Viewer(title="Crop or make Kymograph")
        ui = NapariTabs(viewer)
        viewer.window.add_dock_widget(ui)