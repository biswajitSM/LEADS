import napari
import numpy as np
from . import crop_images_ui
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import PySimpleGUI as sg

from .. import crop_images 

class NapariTabs(QtWidgets.QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.ui = crop_images_ui.Ui_Form()
        self.ui.setupUi(self)
        self.viewer = viewer

        # set defaults
        self.use_current_image_path = False

        # number of colors
        self.ui.numColorsCbox.setCurrentText("2")
        self.num_colors = self.ui.numColorsCbox.currentText()
        self.ui.numColorsCbox.currentIndexChanged.connect(self.change_num_colors)
        # frame number
        self.frame_start = self.ui.frameStartSbox.value()
        self.frame_end = self.ui.frameEndBox.value()        
        if self.frame_end == -1:
            self.frame_end = None
        # image preprocessing
        #self.subtractBackground = self.ui.bkgSubstractionCheckBox.checkState()        
        self.ui.bkgSubstractionCheckBox.stateChanged.connect(self.toggle_bkg_subtraction)
        # processing
        self.ui.loadImageBtn.clicked.connect(self.load_img_seq)
        self.ui.defaultFramesBtn.clicked.connect(self.set_default_frame_num)
        self.ui.cropSelectedBtn.clicked.connect(self.crop)
        self.ui.loadImageJroisBtn.clicked.connect(self.load_imageJ_rois)
        self.ui.saveImageJroisBtn.clicked.connect(self.save_imageJ_rois)

    def load_img_seq(self, path_in=""):
        if self.use_current_image_path:
            current_image_path = self.image_meta.get("folderpath")
            self.image_meta = crop_images.daskread_img_seq(num_colors=int(self.ui.numColorsCbox.currentText()), 
            bkg_subtraction=bool(self.ui.bkgSubstractionCheckBox.checkState()), path=current_image_path)
        else:
            self.image_meta = crop_images.daskread_img_seq(num_colors=int(self.ui.numColorsCbox.currentText()), 
            bkg_subtraction=bool(self.ui.bkgSubstractionCheckBox.checkState()))
        self.use_current_image_path = False
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
            self.shift_bool = self.ui.shiftImageCheckBox.isChecked()
            self.shift_x = self.ui.xShiftSpinBox.value()
            self.shift_y = self.ui.yShiftSpinBox.value()
            self.angle = self.ui.angleSpinBox.value()
            if self.frame_end == -1:
                self.frame_end = None
            shape_layer = self.viewer.layers['rect_roi']
            crop_images.crop_rect_shapes(self.image_meta, shape_layer,
                            frame_start=self.frame_start, frame_end=self.frame_end,
                            geometric_transform=self.shift_bool,
                            shift_x=self.shift_x, shift_y=self.shift_y, angle=self.angle
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
        _ = crop_images.addroi_to_shapelayer(self.viewer.layers['rect_roi'], roi_file_list)

    def save_imageJ_rois(self):
        shape_layer = self.viewer.layers['rect_roi']
        crop_images.save_rectshape_as_imageJroi(shape_layer)

    def toggle_bkg_subtraction(self):
        #current_image_path = self.image_meta.get("folderpath")#self.image_meta["folderpath"]
        self.use_current_image_path = True
        self.load_img_seq()      

def main():
    with napari.gui_qt():
        viewer = napari.Viewer(title="Crop or make Kymograph")
        ui = NapariTabs(viewer)
        viewer.window.add_dock_widget(ui)

if __name__ == "__main__":
    main()
    