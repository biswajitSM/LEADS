import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import qdarkstyle
import pyqtgraph as pg
import pyqtgraph.dockarea as pg_da
import pyqtgraph.exporters
from tifffile import imwrite
from ..kymograph import (read_img_stack,
                median_bkg_substration, applySparseSIM, peakfinder_savgol,
                analyze_maxpeak, loop_sm_dist)
from .. import kymograph
from .. import io
from ..utils import hdf5dict, makevideo, figure_params, step_detect
import os, sys, glob, time, subprocess, webbrowser
import yaml
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
plt.rcParams.update(figure_params.params_dict)
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import xlsxwriter
from skimage.io.collection import alphanumeric_key
import h5py
import tqdm
from . import crop_images_gui
import pandas as pd
import re
from copy import deepcopy, copy

if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)


DEFAULTS = {
    "Number of colors" : "2",
    "ColorMap" : 'plasma',
    }
DEFAULT_PARAMETERS = {
    "Acquisition Time" : 100, # in millisecond
    "Pixel Size" : 119, # in nanometer
    "ROI Width" : 9,
    "Peak Prominence" : 0.25,
    "DNA Length" : 48.5, # in kilo bases
    "DNA Contour Length" : 17.66, # in micrometer
    "SxO concentration": 100, # in nM
    "DNA Persistence Length": 35.1, # in nanometer, for [SxO]=100 nM
    "DNA Puncta Diameter" : 5,
    "Search Range" : 10,
    "Memory" : 5,
    "Filter Length": 10,
    }

pg.setConfigOption('background', pg.mkColor((0, 0, 0, 0))) # 'w' for white and 'k' for black background
pg.setConfigOption('imageAxisOrder', 'col-major') # the row and cols are reversed
grads = pyqtgraph.graphicsItems.GradientEditorItem.Gradients
grads['parula'] = {'ticks': [(0.0, (53, 42, 134, 255)), (0.25, (19, 128, 213, 255)), (0.5, (37, 180, 169, 255)), (0.75, (191, 188, 96, 255)), (1.0, (248, 250, 13, 255))], 'mode': 'rgb'}
grads['jet'] = {'ticks': [[0.0, [0, 0, 127]], [0.11, [0, 0, 255]], [0.125, [0, 0, 255]], [0.34, [0, 219, 255]], [0.35, [0, 229, 246]], [0.375, [20, 255, 226]], [0.64, [238, 255, 8]], [0.65, [246, 245, 0]], [0.66, [255, 236, 0.0]], [0.89, [255, 18, 0]], [0.91, [231, 0, 0]], [1, [127, 0, 0]]], 'mode': 'rgb'}
grads['seismic'] = {'ticks': [[0.0, [0, 0, 76]], [0.25, [0, 0, 255]], [0.5, [255, 255, 255]], [0.75, [255, 0, 0]], [1.0, [127, 0, 0]]], 'mode': 'rgb'}
grads['gnuplot'] = {'ticks': [(0.0, (0, 0, 0)), (0.25, (127, 4, 255)), (0.5, (180, 32, 0)), (0.75, (221, 107, 0)), (1.0, (255, 255, 0))], 'mode': 'rgb'}

class FileDialog(QtWidgets.QDialog):
    """ The dialog letting the user choose ROIs """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.window = parent
        self.setWindowTitle("Open folder and filter ROIs")
        self.resize(650, 0)
        self.setModal(False)

        vbox = QtWidgets.QVBoxLayout(self)
        general_groupbox = QtWidgets.QGroupBox("Open a folder with .tif/.tiff files and filter")
        vbox.addWidget(general_groupbox)
        general_grid = QtWidgets.QGridLayout(general_groupbox)

        # qlineEdit for browsing the right directory
        self.PathQLabel = QtWidgets.QLabel()
        self.PathQLabel.setObjectName("PathQLabel")
        general_grid.addWidget(self.PathQLabel, 0, 1, 1, 2)  #row,col,rowspan,colspan

        self.BrowseButton = QtWidgets.QPushButton()
        self.BrowseButton.setObjectName("BrowseButton")
        self.BrowseButton.setText("Browse")
        self.BrowseButton.setAutoDefault(False) # prevent the button from executing upon pressing enter
        self.BrowseButton.clicked.connect(self.window.load_folder)
        general_grid.addWidget(self.BrowseButton, 0, 0)          

        # key
        general_grid.addWidget(QtWidgets.QLabel("key:"), 1, 0)
        self.ROI_key_lineedit = QtWidgets.QLineEdit()
        self.ROI_key_lineedit.returnPressed.connect(self.window.searchROIs)
        general_grid.addWidget(self.ROI_key_lineedit, 1, 1, 1, 2)  #row,col,rowspan,colspan

        # let user input a ROI number
        self.fileNumberSpinBox = QtWidgets.QDoubleSpinBox()
        self.fileNumberSpinBox.setDecimals(0)
        self.fileNumberSpinBox.setMinimum(1)
        self.fileNumberSpinBox.setMaximum(99)
        self.fileNumberSpinBox.valueChanged.connect(self.window.any_file)
        general_grid.addWidget(self.fileNumberSpinBox, 2, 0)
        
        self.fileNumberLabel = QtWidgets.QLabel()
        self.fileNumberLabel.setText("/?")
        general_grid.addWidget(self.fileNumberLabel, 2, 1)

        # print filenames and filtered filenames
        self.PrintAllFilenamesButton = QtWidgets.QPushButton()
        self.PrintAllFilenamesButton.setObjectName("PrintAllFilenamesButton")
        self.PrintAllFilenamesButton.setText("Print all filenames")
        self.PrintAllFilenamesButton.setAutoDefault(False)
        self.PrintAllFilenamesButton.clicked.connect(self.window.print_all_filenames)
        general_grid.addWidget(self.PrintAllFilenamesButton, 3, 0)          

        self.PrintFilteredFilenamesButton = QtWidgets.QPushButton()
        self.PrintFilteredFilenamesButton.setObjectName("PrintFilteredFilenamesButton")
        self.PrintFilteredFilenamesButton.setText("Print filtered filenames")
        self.PrintFilteredFilenamesButton.setAutoDefault(False)
        self.PrintFilteredFilenamesButton.clicked.connect(self.window.print_filtered_filenames)
        general_grid.addWidget(self.PrintFilteredFilenamesButton, 3, 1)
        
        self.GenerateRoadblockExcelButton = QtWidgets.QPushButton()
        self.GenerateRoadblockExcelButton.setObjectName("GenerateRoadblockExcelButton")
        self.GenerateRoadblockExcelButton.setText("Generate excel overview")
        self.GenerateRoadblockExcelButton.setAutoDefault(False)
        self.GenerateRoadblockExcelButton.clicked.connect(self.window.generate_overview_excel)
        general_grid.addWidget(self.GenerateRoadblockExcelButton, 3, 2)   
        
class ParametersDialog(QtWidgets.QDialog):
    """ The dialog showing parameters """

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
        self.pix_spinbox.setRange(0, int(1e3))
        self.pix_spinbox.setSuffix(" nm")
        self.pix_spinbox.setValue(DEFAULT_PARAMETERS["Pixel Size"])
        self.pix_spinbox.setKeyboardTracking(False)
        self.pix_spinbox.valueChanged.connect(self.on_paramter_change)
        general_grid.addWidget(self.pix_spinbox, 0, 1)

        # ROI Width
        general_grid.addWidget(QtWidgets.QLabel("ROI Size:"), 1, 0)
        self.roi_spinbox = QtWidgets.QSpinBox()
        self.roi_spinbox.setRange(0, int(1e3))
        self.roi_spinbox.setSuffix(" pixels")
        self.roi_spinbox.setValue(DEFAULT_PARAMETERS["ROI Width"])
        self.roi_spinbox.setKeyboardTracking(False)
        self.roi_spinbox.valueChanged.connect(self.on_roi_change)
        general_grid.addWidget(self.roi_spinbox, 1, 1)

        # Acquisition Time
        general_grid.addWidget(QtWidgets.QLabel("Acquisition time:"), 2, 0)
        self.aqt_spinbox = QtWidgets.QSpinBox()
        self.aqt_spinbox.setRange(0, int(1e5))
        self.aqt_spinbox.setSuffix(" ms")
        self.aqt_spinbox.setValue(DEFAULT_PARAMETERS["Acquisition Time"])
        self.aqt_spinbox.valueChanged.connect(self.on_paramter_change)
        general_grid.addWidget(self.aqt_spinbox, 2, 1)

        # Default DNA end finding checkbox
        general_grid.addWidget(QtWidgets.QLabel("By default find DNA ends?"), 3, 0)
        self.DefaultFindDNAends_checkbox = QtWidgets.QCheckBox()
        self.DefaultFindDNAends_checkbox.setChecked(True)
        general_grid.addWidget(self.DefaultFindDNAends_checkbox, 3, 1)

        # Save new analysis file or overwrite?
        general_grid.addWidget(QtWidgets.QLabel("Save additional .hdf5 analysis files?"), 4, 0)
        self.SaveNewAnalysisFile_checkbox = QtWidgets.QCheckBox()
        self.SaveNewAnalysisFile_checkbox.setChecked(False)
        general_grid.addWidget(self.SaveNewAnalysisFile_checkbox, 4, 1)

        # give name to additional file
        general_grid.addWidget(QtWidgets.QLabel("Additional .hdf5 suffix:"), 5, 0)
        self.SaveNewAnalysisFile_lineedit = QtWidgets.QLineEdit()
        self.SaveNewAnalysisFile_lineedit.setText('')
        general_grid.addWidget(self.SaveNewAnalysisFile_lineedit, 5, 1)


    def on_paramter_change(self):
        self.window.set_scalebar()

    def on_roi_change(self):
        self.window.set_roi_width()

class SparseSIMDialog(QtWidgets.QDialog):
    """ The dialog showing parameters for Sparse SIM """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.window = parent
        self.setWindowTitle("Parameters")
        self.resize(300, 0)
        self.setModal(False)
        self.backup_saved = False


        vbox = QtWidgets.QVBoxLayout(self)
        general_groupbox = QtWidgets.QGroupBox("General")
        vbox.addWidget(general_groupbox)
        general_grid = QtWidgets.QGridLayout(general_groupbox)

        # parameters for sparseSIM:
        # sigma, sparse_iter = 100, fidelity = 150, sparsity = 10, tcontinuity = 0.5,
                         # background = 1, deconv_iter = 7, deconv_type = 1

        # PSF width sigma
        general_grid.addWidget(QtWidgets.QLabel("PSF width:"), 0, 0)
        self.sigma_spinbox = QtWidgets.QSpinBox()
        self.sigma_spinbox.setRange(0, int(1e3))
        self.sigma_spinbox.setSuffix(" pixel(s)")
        self.sigma_spinbox.setValue(3)
        self.sigma_spinbox.setKeyboardTracking(False)
        self.sigma_spinbox.valueChanged.connect(self.on_paramter_change)
        general_grid.addWidget(self.sigma_spinbox, 0, 1)

        # number of iterations
        general_grid.addWidget(QtWidgets.QLabel("# iterations:"), 1, 0)
        self.numIter_spinbox = QtWidgets.QSpinBox()
        self.numIter_spinbox.setRange(0, int(1e3))
        self.numIter_spinbox.setSuffix("")
        self.numIter_spinbox.setValue(100)
        self.numIter_spinbox.setKeyboardTracking(False)
        self.numIter_spinbox.valueChanged.connect(self.on_paramter_change)
        general_grid.addWidget(self.numIter_spinbox, 1, 1)

        # fidelity parameter
        general_grid.addWidget(QtWidgets.QLabel("Fidelity:"), 2, 0)
        self.fidelity_spinbox = QtWidgets.QSpinBox()
        self.fidelity_spinbox.setRange(0, int(1e5))
        self.fidelity_spinbox.setSuffix("")
        self.fidelity_spinbox.setValue(150)
        self.fidelity_spinbox.setKeyboardTracking(False)
        self.fidelity_spinbox.valueChanged.connect(self.on_paramter_change)
        general_grid.addWidget(self.fidelity_spinbox, 2, 1)

        # sparsity parameter
        general_grid.addWidget(QtWidgets.QLabel("Sparsity:"), 3, 0)
        self.sparsity_spinbox = QtWidgets.QSpinBox()
        self.sparsity_spinbox.setRange(0, int(1e5))
        self.sparsity_spinbox.setSuffix("")
        self.sparsity_spinbox.setValue(10)
        self.sparsity_spinbox.setKeyboardTracking(False)
        self.sparsity_spinbox.valueChanged.connect(self.on_paramter_change)
        general_grid.addWidget(self.sparsity_spinbox, 3, 1)

        # tcontinuity parameter
        general_grid.addWidget(QtWidgets.QLabel("Time continuity:"), 4, 0)
        self.tcontinuity_spinbox = QtWidgets.QDoubleSpinBox()
        self.tcontinuity_spinbox.setRange(0, int(1e5))
        self.tcontinuity_spinbox.setSuffix("")
        self.tcontinuity_spinbox.setSingleStep(0.1)
        self.tcontinuity_spinbox.setValue(0.5)
        self.tcontinuity_spinbox.setKeyboardTracking(False)
        self.tcontinuity_spinbox.valueChanged.connect(self.on_paramter_change)
        general_grid.addWidget(self.tcontinuity_spinbox, 4, 1)

        # background parameter
        general_grid.addWidget(QtWidgets.QLabel("Fidelity:"), 5, 0)
        self.background_combobox = QtWidgets.QComboBox()        
        self.background_combobox.addItems(["No background"])
        self.background_combobox.addItems(["Weak background (HI)"])
        self.background_combobox.addItems(["Strong background (HI)"])
        self.background_combobox.addItems(["Weak background (LI)"])
        self.background_combobox.addItems(["With background (LI)"])
        self.background_combobox.addItems(["Strong background (LI)"])
        self.background_combobox.currentIndexChanged.connect(self.on_paramter_change)
        general_grid.addWidget(self.background_combobox, 5, 1)

        # deconv_iter parameter
        general_grid.addWidget(QtWidgets.QLabel("Deconvolution iterations:"), 6, 0)
        self.deconv_iter_spinbox = QtWidgets.QSpinBox()
        self.deconv_iter_spinbox.setRange(0, int(1e5))
        self.deconv_iter_spinbox.setSuffix("")
        self.deconv_iter_spinbox.setValue(7)
        self.deconv_iter_spinbox.setKeyboardTracking(False)
        self.deconv_iter_spinbox.valueChanged.connect(self.on_paramter_change)
        general_grid.addWidget(self.deconv_iter_spinbox, 6, 1)

        # deconv_type parameter
        general_grid.addWidget(QtWidgets.QLabel("Deconvolution type:"), 7, 0)
        self.deconv_type_combobox = QtWidgets.QComboBox()        
        self.deconv_type_combobox.addItems(["LandWeber deconvolution"])
        self.deconv_type_combobox.addItems(["Richardson-Lucy deconvolution"])
        self.deconv_type_combobox.addItems(["No deconvolution"])
        self.deconv_type_combobox.currentIndexChanged.connect(self.on_paramter_change)
        general_grid.addWidget(self.deconv_type_combobox, 7, 1)

    def on_paramter_change(self):
        if not self.backup_saved:
            # before playing with the img_arrays, let us save a backup copy. otherwise we keep changing the changed arrays
            if self.window.numColors == "3":
                np.save(os.path.join(self.window.folderpath, 'imgarr_left.npy'), self.window.imgarr_left)
                np.save(os.path.join(self.window.folderpath, 'imgarr_right.npy'), self.window.imgarr_right)
                np.save(os.path.join(self.window.folderpath, 'imgarr_col3.npy'), self.window.imgarr_col3)
            elif self.window.numColors == "2":
                np.save(os.path.join(self.window.folderpath, 'imgarr_left.npy'), self.window.imgarr_left)
                np.save(os.path.join(self.window.folderpath, 'imgarr_right.npy'), self.window.imgarr_right)
            elif self.window.numColors == "1":
                np.save(os.path.join(self.window.folderpath, 'imgarr_left.npy'), self.window.imgarr_left)
            self.backup_saved = True
        self.window.get_processed_image(preview=True)
        self.window.set_img_stack()

class MultiPeakDialog(QtWidgets.QDialog):
    """ The dialog showing Multipeak analysis """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.window = parent
        self.setWindowTitle("Peak Analysis")
        self.resize(300, 0)
        self.setModal(False)

        vbox = QtWidgets.QVBoxLayout(self)
        general_groupbox = QtWidgets.QGroupBox("Peak Finding Parameters")
        vbox.addWidget(general_groupbox)
        general_grid = QtWidgets.QGridLayout(general_groupbox)
        ## prominence ##
        general_grid.addWidget(QtWidgets.QLabel("Peak Prominence:"), 0, 0)
        self.prominence_spinbox = QtWidgets.QDoubleSpinBox()
        self.prominence_spinbox.setRange(0, 1)
        self.prominence_spinbox.setSingleStep(0.01)
        self.prominence_spinbox.setValue(DEFAULT_PARAMETERS["Peak Prominence"])
        self.prominence_spinbox.setKeyboardTracking(False)
        general_grid.addWidget(self.prominence_spinbox, 0, 1)
        # normalize/correction with no-loop data
        self.loopcorrection_checkbox = QtWidgets.QCheckBox("Correction with Non-loop")
        self.loopcorrection_checkbox.setChecked(True)
        general_grid.addWidget(self.loopcorrection_checkbox, 0, 2, 1, 2)
        # Slider
        self.prominence_slider = DoubleSlider()
        self.prominence_slider.setOrientation(QtCore.Qt.Horizontal)
        self.prominence_slider.setValue(DEFAULT_PARAMETERS["Peak Prominence"])
        self.prominence_slider.setSingleStep(0.01)
        general_grid.addWidget(self.prominence_slider, 1, 0, 1, 3)
        # Preview
        self.preview_checkbox = QtWidgets.QCheckBox("Preview")
        self.preview_checkbox.setChecked(True)
        general_grid.addWidget(self.preview_checkbox, 1, 3)
        # DNA length
        general_grid.addWidget(QtWidgets.QLabel("DNA length:"), 3, 0)
        self.DNAlength_spinbox = QtWidgets.QDoubleSpinBox()
        self.DNAlength_spinbox.setRange(1, 1e3)
        self.DNAlength_spinbox.setValue(DEFAULT_PARAMETERS["DNA Length"])
        self.DNAlength_spinbox.setSuffix(" kb")
        self.DNAlength_spinbox.setKeyboardTracking(False)
        general_grid.addWidget(self.DNAlength_spinbox, 3, 1)
        # Puncta size
        general_grid.addWidget(QtWidgets.QLabel("Puncta diameter:"), 3, 2)
        self.DNApuncta_spinbox = QtWidgets.QSpinBox()
        self.DNApuncta_spinbox.setRange(1, int(1e3))
        self.DNApuncta_spinbox.setValue(DEFAULT_PARAMETERS["DNA Puncta Diameter"])
        self.DNApuncta_spinbox.setSuffix(" pixels")
        self.DNApuncta_spinbox.setKeyboardTracking(False)
        general_grid.addWidget(self.DNApuncta_spinbox, 3, 3)
        # DNA Contour length
        general_grid.addWidget(QtWidgets.QLabel("DNA Contour length:"), 4, 0)
        self.DNAcontourlength_spinbox = QtWidgets.QDoubleSpinBox()
        self.DNAcontourlength_spinbox.setRange(1, 1e3)
        self.DNAcontourlength_spinbox.setValue(DEFAULT_PARAMETERS["DNA Contour Length"])
        self.DNAcontourlength_spinbox.setSuffix(" \u03BCm")
        self.DNAcontourlength_spinbox.setKeyboardTracking(False)
        general_grid.addWidget(self.DNAcontourlength_spinbox, 4, 1)
        # DNA Persistence Length
        general_grid.addWidget(QtWidgets.QLabel("DNA Persistence length:"), 4, 2)
        self.DNApersistencelength_spinbox = QtWidgets.QDoubleSpinBox()
        self.DNApersistencelength_spinbox.setRange(1, 1e3)
        self.DNApersistencelength_spinbox.setValue(DEFAULT_PARAMETERS["DNA Persistence Length"])
        self.DNApersistencelength_spinbox.setSuffix(" nm")
        self.DNApersistencelength_spinbox.setKeyboardTracking(False)
        general_grid.addWidget(self.DNApersistencelength_spinbox, 4, 3)
        # Threshold All Peaks : compare peaks on all lines and apply threshold
        self.peakthreshold_checkbox = QtWidgets.QCheckBox("Threshold All Peaks")
        self.peakthreshold_checkbox.setChecked(True)
        general_grid.addWidget(self.peakthreshold_checkbox, 5, 0)
        self.peakthreshold_spinbox = QtWidgets.QSpinBox()
        self.peakthreshold_spinbox.setRange(1, 99)
        self.peakthreshold_spinbox.setValue(5)
        self.peakthreshold_spinbox.setSuffix(" %Max")
        self.peakthreshold_spinbox.setKeyboardTracking(False)
        general_grid.addWidget(self.peakthreshold_spinbox, 5, 1)
        # Smoothing length
        general_grid.addWidget(QtWidgets.QLabel("Smoothing Length:"), 5, 2)
        self.smoothlength_spinbox = QtWidgets.QSpinBox()
        self.smoothlength_spinbox.setRange(1, int(1e3))
        self.smoothlength_spinbox.setValue(7)
        self.smoothlength_spinbox.setKeyboardTracking(False)
        general_grid.addWidget(self.smoothlength_spinbox, 5, 3)
        # Peak widths: Min and Max
        self.minwidth_spinbox = QtWidgets.QSpinBox()
        self.minwidth_spinbox.setRange(1, int(1e2))
        self.minwidth_spinbox.setValue(1)
        self.minwidth_spinbox.setSuffix(" pixels")
        self.minwidth_spinbox.setKeyboardTracking(False)
        general_grid.addWidget(QtWidgets.QLabel("Min Peak Width:"), 6, 0)
        general_grid.addWidget(self.minwidth_spinbox, 6, 1)
        self.maxwidth_spinbox = QtWidgets.QSpinBox()
        self.maxwidth_spinbox.setRange(1, int(1e3))
        self.maxwidth_spinbox.setValue(20)
        self.maxwidth_spinbox.setSuffix(" pixels")
        self.maxwidth_spinbox.setKeyboardTracking(False)
        general_grid.addWidget(QtWidgets.QLabel("Max Peak Width:"), 6, 2)
        general_grid.addWidget(self.maxwidth_spinbox, 6, 3)
        # SxO concentration
        self.SxOconc_spinbox = QtWidgets.QSpinBox()
        self.SxOconc_spinbox.setRange(0, int(1e3))
        self.SxOconc_spinbox.setValue(100)
        self.SxOconc_spinbox.setSuffix(" nM")
        self.SxOconc_spinbox.setKeyboardTracking(False)
        general_grid.addWidget(QtWidgets.QLabel("SxO conc.:"), 7, 0)
        general_grid.addWidget(self.SxOconc_spinbox, 7, 1)
        ## Single molecule peak detection parameters
        singlemolecule_groupbox = QtWidgets.QGroupBox("Single Molecule Parameters")
        vbox.addWidget(singlemolecule_groupbox)
        smol_grid = QtWidgets.QGridLayout(singlemolecule_groupbox)
        # smol Prominence
        smol_grid.addWidget(QtWidgets.QLabel("Peak Prominence:"), 0, 0)
        self.smol_prominence_spinbox = QtWidgets.QDoubleSpinBox()
        self.smol_prominence_spinbox.setRange(0, 1)
        self.smol_prominence_spinbox.setSingleStep(0.01)
        self.smol_prominence_spinbox.setValue(DEFAULT_PARAMETERS["Peak Prominence"])
        self.smol_prominence_spinbox.setKeyboardTracking(False)
        smol_grid.addWidget(self.smol_prominence_spinbox, 0, 1)
        # smol Slider
        self.smol_prominence_slider = DoubleSlider()
        self.smol_prominence_slider.setOrientation(QtCore.Qt.Horizontal)
        self.smol_prominence_slider.setValue(DEFAULT_PARAMETERS["Peak Prominence"])
        self.smol_prominence_slider.setSingleStep(0.01)
        smol_grid.addWidget(self.smol_prominence_slider, 1, 0)
        # smol Preview
        self.smol_preview_checkbox = QtWidgets.QCheckBox("Preview")
        self.smol_preview_checkbox.setChecked(True)
        smol_grid.addWidget(self.smol_preview_checkbox, 1, 1)

        ## linking paramters ##
        linking_groupbox = QtWidgets.QGroupBox("Multi Peaks Linking Parameters")
        vbox.addWidget(linking_groupbox)
        linking_grid = QtWidgets.QGridLayout(linking_groupbox)
        # search range
        linking_grid.addWidget(QtWidgets.QLabel("Search Range:"), 0, 0)
        self.searchrange_spinbox = QtWidgets.QSpinBox()
        self.searchrange_spinbox.setRange(1, int(1e3))
        self.searchrange_spinbox.setValue(DEFAULT_PARAMETERS["Search Range"])
        self.searchrange_spinbox.setKeyboardTracking(False)
        linking_grid.addWidget(self.searchrange_spinbox, 0, 1)
        # Memory
        linking_grid.addWidget(QtWidgets.QLabel("Memory:"), 0, 2)
        self.memory_spinbox = QtWidgets.QSpinBox()
        self.memory_spinbox.setRange(1, int(1e3))
        self.memory_spinbox.setValue(DEFAULT_PARAMETERS["Memory"])
        self.memory_spinbox.setKeyboardTracking(False)
        linking_grid.addWidget(self.memory_spinbox, 0, 3)
        # Filter length
        linking_grid.addWidget(QtWidgets.QLabel("Filter Length:"), 1, 0)
        self.filterlen_spinbox = QtWidgets.QSpinBox()
        self.filterlen_spinbox.setRange(1, int(1e3))
        self.filterlen_spinbox.setValue(DEFAULT_PARAMETERS["Filter Length"])
        self.filterlen_spinbox.setKeyboardTracking(False)
        linking_grid.addWidget(self.filterlen_spinbox, 1, 1)
        # plot all peaks
        self.linkplot_pushbutton = QtWidgets.QPushButton("Link and Plot All Peaks")
        linking_grid.addWidget(self.linkplot_pushbutton, 1, 2, 1, 2)
        # link col1 and col2
        self.link_col1col2_checkbox = QtWidgets.QCheckBox("LinkTwoColors")
        self.max_frame_diff_spinbox = QtWidgets.QSpinBox()
        self.max_frame_diff_spinbox.setRange(1, int(1e3))
        self.max_frame_diff_spinbox.setValue(10)
        self.max_frame_diff_spinbox.setPrefix("\u0394frames ")
        self.max_frame_diff_spinbox.setKeyboardTracking(False)
        self.max_pix_diff_spinbox = QtWidgets.QSpinBox()
        self.max_pix_diff_spinbox.setRange(1, int(1e3))
        self.max_pix_diff_spinbox.setValue(10)
        self.max_pix_diff_spinbox.setPrefix("\u0394pixs ")
        self.max_pix_diff_spinbox.setKeyboardTracking(False)
        self.min_coloc_diff_spinbox = QtWidgets.QSpinBox()
        self.min_coloc_diff_spinbox.setRange(1, int(1e3))
        self.min_coloc_diff_spinbox.setValue(10)
        self.min_coloc_diff_spinbox.setPrefix("\u0394coloc ")
        self.min_coloc_diff_spinbox.setKeyboardTracking(False)
        linking_grid.addWidget(self.link_col1col2_checkbox, 2, 0)
        linking_grid.addWidget(self.max_frame_diff_spinbox, 2, 1)
        linking_grid.addWidget(self.max_pix_diff_spinbox, 2, 2)
        linking_grid.addWidget(self.min_coloc_diff_spinbox, 2, 3)

        ## Track merging ##
        merging_groupbox = QtWidgets.QGroupBox("Merge tracks")
        vbox.addWidget(merging_groupbox)
        merging_grid = QtWidgets.QGridLayout(merging_groupbox)
        # left tracks
        merging_grid.addWidget(QtWidgets.QLabel("Left tracks:"), 0, 0)
        self.merge_left_lineedit = QtWidgets.QLineEdit()
        self.merge_left_lineedit.setText('1,2,3')
        self.searchrange_spinbox.setKeyboardTracking(False)
        merging_grid.addWidget(self.merge_left_lineedit, 0, 1)
        # right tracks
        merging_grid.addWidget(QtWidgets.QLabel("Right tracks:"), 0, 2)
        self.merge_right_lineedit = QtWidgets.QLineEdit()
        self.merge_right_lineedit.setText('1,2,3')
        self.searchrange_spinbox.setKeyboardTracking(False)
        merging_grid.addWidget(self.merge_right_lineedit, 0, 3)
        # merge button
        self.merge_tracks_button = QtWidgets.QPushButton("Merge tracks")
        merging_grid.addWidget(self.merge_tracks_button, 0, 4)

        ## Kinetics ##
        plot_groupbox = QtWidgets.QGroupBox("Plotting Kinetics")
        vbox.addWidget(plot_groupbox)
        plot_grid = QtWidgets.QGridLayout(plot_groupbox)
        # left section fro peak-number
        plot_grid.addWidget(QtWidgets.QLabel("Left Peak No:"), 0, 0)
        self.leftpeak_num_combobox = QtWidgets.QComboBox()
        self.leftpeak_num_combobox.addItems(["1"])
        plot_grid.addWidget(self.leftpeak_num_combobox, 0, 1)
        # right section fro peak-number
        plot_grid.addWidget(QtWidgets.QLabel("Right Peak No:"), 0, 2)
        self.rightpeak_num_combobox = QtWidgets.QComboBox()
        self.rightpeak_num_combobox.addItems(["1"])
        plot_grid.addWidget(self.rightpeak_num_combobox, 0, 3)
        # plot kinetics
        self.loopkinetics_pushbutton = QtWidgets.QPushButton("Plot Loop Kinetics")
        self.loopVsmol_pushbutton = QtWidgets.QPushButton("Plot Loop Vs Mol")
        self.moving_window_spinbox = QtWidgets.QSpinBox()
        self.moving_window_spinbox.setRange(1, int(1e3))
        self.moving_window_spinbox.setValue(51)
        self.moving_window_spinbox.setKeyboardTracking(False)
        plot_grid.addWidget(self.loopkinetics_pushbutton, 1, 0)
        plot_grid.addWidget(self.loopVsmol_pushbutton, 1, 1)
        plot_grid.addWidget(QtWidgets.QLabel("moving window length:"), 1, 2)
        plot_grid.addWidget(self.moving_window_spinbox, 1, 3)
        # include Force
        self.force_checkbox = QtWidgets.QCheckBox("Include Force")
        self.force_combobox = QtWidgets.QComboBox()
        self.force_combobox.addItems(["Analytical", "Interpolation"])
        plot_grid.addWidget(self.force_checkbox, 2, 0)
        plot_grid.addWidget(self.force_combobox, 2, 1)
        # choose and plot types
        self.plottype_pushbutton = QtWidgets.QPushButton("Plot")
        self.plottype_combobox = QtWidgets.QComboBox()
        self.plottype_combobox.addItems(["MSDmoving", "MSDsavgol", "MSDlagtime",
                                         "MSDlagtime-AllPeaks",
                                         "LoopSizeVsPosition",                                         
                                         "TimeTraceCol1", "TimeTraceCol2",
                                         "AvTimeTraceCol1", "AvTimeTraceCol2",
                                         "FitKinetics",
                                         "LoopSizeVsPositionDetailed"
                                         ])
        plot_grid.addWidget(self.plottype_pushbutton, 2, 2)
        plot_grid.addWidget(self.plottype_combobox, 2, 3)

    def connect_signals(self):
        self.prominence_spinbox.valueChanged.connect(self.on_prominence_spinbox_changed)
        self.prominence_slider.sliderReleased.connect(self.on_prominence_slider_changed)
        self.preview_checkbox.stateChanged.connect(self.on_prominence_spinbox_changed)
        self.loopcorrection_checkbox.stateChanged.connect(self.on_prominence_spinbox_changed)
        self.minwidth_spinbox.valueChanged.connect(self.on_peakwidth_spinbox_changed)
        self.maxwidth_spinbox.valueChanged.connect(self.on_peakwidth_spinbox_changed)
        self.DNAlength_spinbox.valueChanged.connect(self.on_prominence_spinbox_changed)
        self.DNApuncta_spinbox.valueChanged.connect(self.on_prominence_spinbox_changed)
        self.smoothlength_spinbox.valueChanged.connect(self.on_smoothlength_spinbox_changed)
        self.peakthreshold_checkbox.stateChanged.connect(self.on_threshold_all_peaks_changed)
        self.peakthreshold_spinbox.valueChanged.connect(self.on_threshold_all_peaks_changed)
        self.SxOconc_spinbox.valueChanged.connect(self.on_SxOconc_spinbox_changed)
        self.smol_prominence_spinbox.valueChanged.connect(self.on_smol_prominence_spinbox_changed)
        self.smol_prominence_slider.sliderReleased.connect(self.on_smol_prominence_slider_changed)
        self.smol_preview_checkbox.stateChanged.connect(self.on_smol_prominence_spinbox_changed)
        self.linkplot_pushbutton.clicked.connect(self.on_clicking_linkplot_pushbutton)
        self.merge_tracks_button.clicked.connect(self.window.merge_tracks)
        self.loopVsmol_pushbutton.clicked.connect(self.on_clicking_loopVsmol_pushbutton)
        self.loopkinetics_pushbutton.clicked.connect(self.on_clicking_loopkinetics_pushbutton)
        self.plottype_pushbutton.clicked.connect(self.window.plottype_multipeak)

    def disconnect_signals(self):
        self.prominence_spinbox.valueChanged.disconnect()
        self.prominence_slider.sliderReleased.disconnect()
        self.preview_checkbox.stateChanged.disconnect()
        self.loopcorrection_checkbox.stateChanged.disconnect()
        self.minwidth_spinbox.valueChanged.disconnect()
        self.maxwidth_spinbox.valueChanged.disconnect()
        self.DNAlength_spinbox.valueChanged.disconnect()
        self.SxOconc_spinbox.valueChanged.disconnect()
        self.DNApuncta_spinbox.valueChanged.disconnect()
        self.smoothlength_spinbox.valueChanged.disconnect()
        self.smol_prominence_spinbox.valueChanged.disconnect()
        self.smol_prominence_slider.sliderReleased.disconnect()
        self.smol_preview_checkbox.stateChanged.disconnect()
        self.linkplot_pushbutton.clicked.disconnect()
        self.merge_tracks_button.clicked.disconnect()
        self.loopVsmol_pushbutton.clicked.disconnect()
        self.loopkinetics_pushbutton.clicked.disconnect()
        self.plottype_pushbutton.clicked.disconnect()

    def on_prominence_spinbox_changed(self):
        self.window.compute_contour_length() # if DNA length is changed, the contour length also changes
        value = self.prominence_spinbox.value()
        self.prominence_slider.setValue(value)
        self.window.params_change_loop_detection()

    def on_prominence_slider_changed(self):
        value = self.prominence_slider.value()
        self.prominence_spinbox.setValue(value)

    def on_peakwidth_spinbox_changed(self):
        self.window.params_change_loop_detection()
        self.window.params_change_smol_detection()

    def on_smoothlength_spinbox_changed(self):
        if self.smoothlength_spinbox.value() % 2 != 0: # only processing odd values
            self.window.params_change_loop_detection()
            self.window.params_change_smol_detection()

    def on_threshold_all_peaks_changed(self):
        self.window.params_change_loop_detection()
        self.window.params_change_smol_detection()

    def on_SxOconc_spinbox_changed(self):
        self.window.compute_contour_length()
        self.window.compute_persistence_length()

    def on_smol_prominence_spinbox_changed(self):
        value = self.smol_prominence_spinbox.value()
        self.smol_prominence_slider.setValue(value)
        self.window.params_change_smol_detection()

    def on_smol_prominence_slider_changed(self):
        value = self.smol_prominence_slider.value()
        self.smol_prominence_spinbox.setValue(value)

    def on_clicking_linkplot_pushbutton(self):
        self.window.matplot_all_peaks()
    
    def on_clicking_loopkinetics_pushbutton(self):
        self.window.matplot_loop_kinetics()

    def on_clicking_loopVsmol_pushbutton(self):
        self.window.matplot_loop_vs_sm()

    def on_close_event(self):
        settings = io.load_user_settings()
        settings["kymograph"]["MultiPeak"] = {}
        settings["kymograph"]["MultiPeak"]["prominence"] = self.prominence_spinbox.value()
        settings["kymograph"]["MultiPeak"]["loop_correction"] = self.loopcorrection_checkbox.isChecked()
        settings["kymograph"]["MultiPeak"]["preview"] = self.preview_checkbox.isChecked()
        settings["kymograph"]["MultiPeak"]["DNAlength"] = self.DNAlength_spinbox.value()
        settings["kymograph"]["MultiPeak"]["DNAcontourlength"] = self.DNAcontourlength_spinbox.value()
        settings["kymograph"]["MultiPeak"]["DNApuncta"] = self.DNApuncta_spinbox.value()
        settings["kymograph"]["MultiPeak"]["smoothlength"] = self.smoothlength_spinbox.value()
        settings["kymograph"]["MultiPeak"]["minwidth"] = self.minwidth_spinbox.value()
        settings["kymograph"]["MultiPeak"]["maxwidth"] = self.maxwidth_spinbox.value()
        settings["kymograph"]["MultiPeak"]["SxOconcentration"] = self.SxOconc_spinbox.value()
        settings["kymograph"]["MultiPeak"]["smol_prominence"] = self.smol_prominence_spinbox.value()
        settings["kymograph"]["MultiPeak"]["smol_preview"] = self.smol_preview_checkbox.isChecked()
        settings["kymograph"]["MultiPeak"]["searchrange"] = self.searchrange_spinbox.value()
        settings["kymograph"]["MultiPeak"]["memory"] = self.memory_spinbox.value()
        settings["kymograph"]["MultiPeak"]["filterlen"] = self.filterlen_spinbox.value()
        settings["kymograph"]["MultiPeak"]["smol_preview"] = self.filterlen_spinbox.value()
        settings["kymograph"]["MultiPeak"]["link_col1col2"] = self.link_col1col2_checkbox.isChecked()
        settings["kymograph"]["MultiPeak"]["max_frame_diff"] = self.max_frame_diff_spinbox.value()
        settings["kymograph"]["MultiPeak"]["max_pix_diff"] = self.max_pix_diff_spinbox.value()
        settings["kymograph"]["MultiPeak"]["min_coloc_diff"] = self.min_coloc_diff_spinbox.value()
        settings["kymograph"]["MultiPeak"]["force"] = self.force_checkbox.isChecked()
        io.save_user_settings(settings)
        self.settings = settings
        return self.settings

    def on_start_event(self, settings=None):
        if settings is None:
            settings = io.load_user_settings()
        try:
            self.prominence_spinbox.setValue(settings["kymograph"]["MultiPeak"]["prominence"])
            self.prominence_slider.setValue(settings["kymograph"]["MultiPeak"]["prominence"])
            self.loopcorrection_checkbox.setChecked(settings["kymograph"]["MultiPeak"]["loop_correction"])
            self.preview_checkbox.setChecked(settings["kymograph"]["MultiPeak"]["preview"])
            self.DNAlength_spinbox.setValue(settings["kymograph"]["MultiPeak"]["DNAlength"])
            self.DNAcontourlength_spinbox.setValue(settings["kymograph"]["MultiPeak"]["DNAcontourlength"])
            self.DNApuncta_spinbox.setValue(settings["kymograph"]["MultiPeak"]["DNApuncta"])
            self.smoothlength_spinbox.setValue(settings["kymograph"]["MultiPeak"]["smoothlength"])
            self.minwidth_spinbox.setValue(settings["kymograph"]["MultiPeak"]["minwidth"])
            self.maxwidth_spinbox.setValue(settings["kymograph"]["MultiPeak"]["maxwidth"])
            self.SxOconc_spinbox.setValue(settings["kymograph"]["MultiPeak"]["SxOconcentration"])
            self.smol_prominence_spinbox.setValue(settings["kymograph"]["MultiPeak"]["smol_prominence"])
            self.smol_prominence_slider.setValue(settings["kymograph"]["MultiPeak"]["smol_prominence"])
            self.smol_preview_checkbox.setChecked(settings["kymograph"]["MultiPeak"]["smol_preview"])
            self.searchrange_spinbox.setValue(settings["kymograph"]["MultiPeak"]["searchrange"])
            self.memory_spinbox.setValue(settings["kymograph"]["MultiPeak"]["memory"])
            self.filterlen_spinbox.setValue(settings["kymograph"]["MultiPeak"]["filterlen"])
            self.link_col1col2_checkbox.setChecked(settings["kymograph"]["MultiPeak"]["link_col1col2"])
            self.max_frame_diff_spinbox.setValue(settings["kymograph"]["MultiPeak"]["max_frame_diff"])
            self.max_pix_diff_spinbox.setValue(settings["kymograph"]["MultiPeak"]["max_pix_diff"])
            self.min_coloc_diff_spinbox.setValue(settings["kymograph"]["MultiPeak"]["min_coloc_diff"])
            self.force_checkbox.setChecked(settings["kymograph"]["MultiPeak"]["force"])
        except Exception as e:
            print(e)
            pass
        self.settings = settings
        return self.settings

class SuperGaussFittingDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.window = parent
        self.setWindowTitle("Find DNA ends")

        # a figure instance to plot on
        self.figure = plt.figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        statusSuperGaussGroupBox = QtWidgets.QGroupBox("Supergauss fit output")
        hLayoutSuperGaussOutput = QtWidgets.QHBoxLayout(statusSuperGaussGroupBox)
        self.superGaussStatusLabel = QtWidgets.QLabel("Idle")
        hLayoutSuperGaussOutput.addWidget(self.superGaussStatusLabel)

        # spin boxes and labels for supergauss parameters        
        superGaussGroupBox = QtWidgets.QGroupBox("Supergauss parameters")
        hLayoutSuperGauss = QtWidgets.QHBoxLayout(superGaussGroupBox)

        self.doSuperGaussFitCheckBox = QtWidgets.QCheckBox("Fit supergauss?")
        self.doSuperGaussFitCheckBox.setChecked(False)
        self.doSuperGaussFitCheckBox.setStyleSheet("color: red")
        hLayoutSuperGauss.addWidget(self.doSuperGaussFitCheckBox)

        self.superGaussWidthLabel = QtWidgets.QLabel("Width:")
        hLayoutSuperGauss.addWidget(self.superGaussWidthLabel)

        self.superGaussWidthSpinBox = QtWidgets.QSpinBox()        
        self.superGaussWidthSpinBox.setRange(1, int(1e5))
        self.superGaussWidthSpinBox.setValue(20)
        self.superGaussWidthSpinBox.setKeyboardTracking(False)
        hLayoutSuperGauss.addWidget(self.superGaussWidthSpinBox)

        self.superGaussOrderLabel = QtWidgets.QLabel("Order:")
        hLayoutSuperGauss.addWidget(self.superGaussOrderLabel)

        self.superGaussOrderSpinBox = QtWidgets.QSpinBox()        
        self.superGaussOrderSpinBox.setRange(1, int(99))
        self.superGaussOrderSpinBox.setValue(6)
        self.superGaussOrderSpinBox.setKeyboardTracking(False)
        hLayoutSuperGauss.addWidget(self.superGaussOrderSpinBox)

        self.PeakPeelingAmplitudeLabel = QtWidgets.QLabel("Fracion of max.:")
        hLayoutSuperGauss.addWidget(self.PeakPeelingAmplitudeLabel)

        self.superGaussFractionMaxSpinBox = QtWidgets.QDoubleSpinBox()        
        self.superGaussFractionMaxSpinBox.setRange(0.001, 1)
        self.superGaussFractionMaxSpinBox.setValue(0.98)
        self.superGaussFractionMaxSpinBox.setSingleStep(0.001)
        self.superGaussFractionMaxSpinBox.setDecimals(3)
        self.superGaussFractionMaxSpinBox.setKeyboardTracking(False)
        hLayoutSuperGauss.addWidget(self.superGaussFractionMaxSpinBox)

        self.superGaussWidthSpinBox.valueChanged.connect(self.find_dna_ends_supergauss)
        self.superGaussOrderSpinBox.valueChanged.connect(self.find_dna_ends_supergauss)
        self.superGaussFractionMaxSpinBox.valueChanged.connect(self.find_dna_ends_supergauss)
        self.doSuperGaussFitCheckBox.stateChanged.connect(self.find_dna_ends_supergauss)

        # spin boxes and labels for peak peeling parameters        
        peakPeelingGroupBox = QtWidgets.QGroupBox("Peak peeling parameters")
        hLayoutPeakPeeling = QtWidgets.QHBoxLayout(peakPeelingGroupBox)

        self.doPeakPeelingCheckBox = QtWidgets.QCheckBox("Peak peeling?")
        self.doPeakPeelingCheckBox.setStyleSheet("color: green")
        self.doPeakPeelingCheckBox.setChecked(True)
        hLayoutPeakPeeling.addWidget(self.doPeakPeelingCheckBox)

        self.PeakPeelingNoPeaksLabel = QtWidgets.QLabel("Max Peak No.:")
        hLayoutPeakPeeling.addWidget(self.PeakPeelingNoPeaksLabel)

        self.PeakPeelingNoPeaksSpinBox = QtWidgets.QSpinBox()        
        self.PeakPeelingNoPeaksSpinBox.setRange(1, int(100))
        self.PeakPeelingNoPeaksSpinBox.setValue(10)
        self.PeakPeelingNoPeaksSpinBox.setKeyboardTracking(False)
        hLayoutPeakPeeling.addWidget(self.PeakPeelingNoPeaksSpinBox)

        self.PeakPeelingPSFLabel = QtWidgets.QLabel("PSF/nm:")
        hLayoutPeakPeeling.addWidget(self.PeakPeelingPSFLabel)

        self.PeakPeelingPSFSpinBox = QtWidgets.QSpinBox()        
        self.PeakPeelingPSFSpinBox.setRange(1, 10000)
        self.PeakPeelingPSFSpinBox.setValue(300)
        self.PeakPeelingPSFSpinBox.setSingleStep(10)
        self.PeakPeelingPSFSpinBox.setKeyboardTracking(False)
        hLayoutPeakPeeling.addWidget(self.PeakPeelingPSFSpinBox)

        self.PeakPeelingAmplitudeLabel = QtWidgets.QLabel("amplitude:")
        hLayoutPeakPeeling.addWidget(self.PeakPeelingAmplitudeLabel)

        self.PeakPeelingAmplitudeSpinBox = QtWidgets.QDoubleSpinBox()        
        self.PeakPeelingAmplitudeSpinBox.setRange(0, 1)
        self.PeakPeelingAmplitudeSpinBox.setValue(0.9)
        self.PeakPeelingAmplitudeSpinBox.setSingleStep(0.05)
        self.PeakPeelingAmplitudeSpinBox.setDecimals(2)
        self.PeakPeelingAmplitudeSpinBox.setKeyboardTracking(False)
        hLayoutPeakPeeling.addWidget(self.PeakPeelingAmplitudeSpinBox)

        self.PeakPeelingResidueLabel = QtWidgets.QLabel("residual:")
        hLayoutPeakPeeling.addWidget(self.PeakPeelingResidueLabel)

        self.PeakPeelingResidueSpinBox = QtWidgets.QDoubleSpinBox()        
        self.PeakPeelingResidueSpinBox.setRange(0.001, 1)
        self.PeakPeelingResidueSpinBox.setValue(0.1)
        self.PeakPeelingResidueSpinBox.setSingleStep(0.05)
        self.PeakPeelingResidueSpinBox.setDecimals(2)
        self.PeakPeelingResidueSpinBox.setKeyboardTracking(False)
        hLayoutPeakPeeling.addWidget(self.PeakPeelingResidueSpinBox)

        self.PeakPeelingNoPeaksSpinBox.valueChanged.connect(self.find_dna_ends_peakPeeling)
        self.PeakPeelingPSFSpinBox.valueChanged.connect(self.find_dna_ends_peakPeeling)
        self.PeakPeelingAmplitudeSpinBox.valueChanged.connect(self.find_dna_ends_peakPeeling)
        self.PeakPeelingResidueSpinBox.valueChanged.connect(self.find_dna_ends_peakPeeling)
        self.doPeakPeelingCheckBox.stateChanged.connect(self.find_dna_ends_peakPeeling)

        # status for peak peeling
        statusPeakPeelingGroupBox = QtWidgets.QGroupBox("Peak peeling output")
        hLayoutPeakPeelingOutput = QtWidgets.QHBoxLayout(statusPeakPeelingGroupBox)
        self.peakPeelingStatusLabel = QtWidgets.QLabel("Peak peeling: Idle")
        hLayoutPeakPeelingOutput.addWidget(self.peakPeelingStatusLabel)

        # set the layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)        
        layout.addWidget(superGaussGroupBox)
        layout.addWidget(peakPeelingGroupBox)
        layout.addWidget(statusSuperGaussGroupBox)
        layout.addWidget(statusPeakPeelingGroupBox)
        self.setLayout(layout)

        # execute peak peeling as default method


    def init(self):
        try:
            non_loop_dna_avg = self.window.kymo_left_noLoop.mean(axis=0)
            plt.plot(non_loop_dna_avg)
            self.canvas.draw()
        except:
            pass
        # self.find_dna_ends_supergauss() # do it the first time with default parameters
        self.doPeakPeelingCheckBox.isChecked()
        self.find_dna_ends_peakPeeling() # do it the first time with default parameters
        self.show()


    def find_dna_ends_supergauss(self):
        if not self.doSuperGaussFitCheckBox.isChecked():
            return
        try:
            self.figure.clear()
            self.canvas.draw()        
            non_loop_dna_avg       = self.window.kymo_left_noLoop.mean(axis=0)
            self.gauss_length      = self.superGaussWidthSpinBox.value()
            self.gauss_order       = self.superGaussOrderSpinBox.value()
            self.fraction_of_max   = self.superGaussFractionMaxSpinBox.value()
            self.dna_ends, formula = kymograph.find_ends_supergauss(non_loop_dna_avg, gauss_length=self.gauss_length, \
                gauss_order=self.gauss_order, fraction_of_max=self.fraction_of_max, threshold_Imax=0.5, plotting=True)
            self.canvas.draw()
            if self.dna_ends is not None:
                self.window.dna_ends_changed()

                self.figure.suptitle(formula)
                dna_length = np.abs(np.diff(self.dna_ends))
                self.superGaussStatusLabel.setText("Supergauss fit found. DNA length: "+"{:.2f}".format(dna_length[0] * self.window.pixelSize)+" microns ("\
                    +"{:.2f}".format(dna_length[0])+" pixels)")
                self.superGaussStatusLabel.setStyleSheet("color: green")
                self.canvas.draw()
            
        except:
            self.superGaussStatusLabel.setText("No suitable supergauss fit found.")
            self.superGaussStatusLabel.setStyleSheet("color: red")
            pass
        self.canvas.draw()


    def find_dna_ends_peakPeeling(self):
        if not self.doPeakPeelingCheckBox.isChecked():
            return
        try:
            self.figure.clear()
            self.canvas.draw()        
            non_loop_dna_avg           = self.window.kymo_left_noLoop.mean(axis=0)
            self.peakPeeling_NoPeaks   = self.PeakPeelingNoPeaksSpinBox.value()
            self.peakPeeling_PSF       = self.PeakPeelingPSFSpinBox.value()
            self.peakPeeling_amplitude = self.PeakPeelingAmplitudeSpinBox.value()
            self.peakPeeling_residue   = self.PeakPeelingResidueSpinBox.value()
            self.dna_ends = kymograph.find_ends_peakPeeling(non_loop_dna_avg, noPeaks=self.peakPeeling_NoPeaks, \
                PSF=self.peakPeeling_PSF/(self.window.pixelSize*1e3), amplitude=self.peakPeeling_amplitude, \
                    residue=self.peakPeeling_residue, plotting=True)
            self.canvas.draw()
            if self.dna_ends is not None:
                self.window.dna_ends_changed()

                dna_length = np.abs(np.diff(self.dna_ends))
                self.peakPeelingStatusLabel.setText("Peak peeling result found. DNA length: "+"{:.2f}".format(dna_length[0] * self.window.pixelSize)+" microns ("\
                    +"{:.2f}".format(dna_length[0])+" pixels)")
                self.peakPeelingStatusLabel.setStyleSheet("color: green")
                self.canvas.draw()
            
        except:
            self.peakPeelingStatusLabel.setText("No suitable peak peeling result found.")
            self.peakPeelingStatusLabel.setStyleSheet("color: red")
            pass
        self.canvas.draw()

class KineticsFitting(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.window = parent
        self.setWindowTitle("Kinetics Fitting")

        self.figure = plt.figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        # set layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        # fit layout
        fit_layout = QtWidgets.QGridLayout()
        # fit loop kinetics button
        self.button_fit_kinetics = QtWidgets.QPushButton('Fit Loop Kinetics')
        fit_layout.addWidget(self.button_fit_kinetics, 0, 0)
        self.mintime_spinbox = QtWidgets.QSpinBox()
        self.mintime_spinbox.setRange(0, 1e3)
        self.maxtime_spinbox = QtWidgets.QSpinBox()
        self.maxtime_spinbox.setRange(1, 1e3)
        fit_layout.addWidget(self.mintime_spinbox, 0, 1)
        fit_layout.addWidget(self.maxtime_spinbox, 0, 2)
        layout.addLayout(fit_layout)
        self.setLayout(layout)

        self.button_fit_kinetics.clicked.connect(self.plot)
    
    def plot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(1, 1)
        self.canvas.draw()

class ROIDialog(QtWidgets.QDialog):
    """
    ROI analysis and manipulation
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.window = parent
        self.setWindowTitle("ROI Analysis / Control")
        # self.resize(300, 0)
        self.setModal(False)

        vbox = QtWidgets.QVBoxLayout(self)
        ## ROI analysis
        analysis_groupbox = QtWidgets.QGroupBox("ROI Analysis")
        vbox.addWidget(analysis_groupbox)
        analysis_grid = QtWidgets.QGridLayout(analysis_groupbox)
        # set left and right labels
        label_L = QtWidgets.QLabel("Left ROI Rect")
        label_L.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        label_R = QtWidgets.QLabel("Right ROI Rect")
        label_R.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        analysis_grid.addWidget(label_L, 0, 0)
        analysis_grid.addWidget(label_R, 0, 1)
        # extract left images
        self.L_roiextract_pushbutton = QtWidgets.QPushButton("Extract Images(L)")
        analysis_grid.addWidget(self.L_roiextract_pushbutton, 1, 0)
        self.L_roiextract_pushbutton.clicked.connect(lambda: self.window.extract_rectROI('left'))
        # extract right images
        self.R_roiextract_pushbutton = QtWidgets.QPushButton("Extract Images(R)")
        self.R_roiextract_pushbutton.clicked.connect(lambda: self.window.extract_rectROI('right'))
        analysis_grid.addWidget(self.R_roiextract_pushbutton, 1, 1)
        # extract merged images
        self.merged_roiextract_pushbutton = QtWidgets.QPushButton("Extract and Merge Two colors")
        self.merged_roiextract_pushbutton.clicked.connect(lambda: self.window.extract_rectROI('both'))
        analysis_grid.addWidget(self.merged_roiextract_pushbutton, 2, 0, 1, 2)
        # plot time trace : left
        self.L_roitrace_pushbutton = QtWidgets.QPushButton("Time Trace(L)")
        self.L_roitrace_pushbutton.clicked.connect(lambda: self.window.timetrace_rectROI('left'))
        analysis_grid.addWidget(self.L_roitrace_pushbutton, 3, 0)
        # plot time trace: right
        self.R_roitrace_pushbutton = QtWidgets.QPushButton("Time Trace(R)")
        self.R_roitrace_pushbutton.clicked.connect(lambda: self.window.timetrace_rectROI('right'))
        analysis_grid.addWidget(self.R_roitrace_pushbutton, 3, 1)
        ## ROI setting / control
        control_groupbox = QtWidgets.QGroupBox("ROI Control")
        vbox.addWidget(control_groupbox)
        control_grid = QtWidgets.QGridLayout(control_groupbox)

class ManagePropertiesDialog(QtWidgets.QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.window = parent
        self.setWindowTitle("Add/Remove Properties")
        self.resize(300, 400)
        self.setModal(True)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)

        self.layout = QtWidgets.QVBoxLayout(self)

        self.add_button = QtWidgets.QPushButton('+')
        
        self.layout.addWidget(self.add_button)
        self.scrollArea = QtWidgets.QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollLayout = QtWidgets.QFormLayout(self.scrollAreaWidgetContents)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.scrollArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.layout.addWidget(self.scrollArea)

        self.load_property_widgets()

        self.OK_button = QtWidgets.QPushButton('OK')
        self.OK_button.setAutoDefault(True)
        self.layout.addWidget(self.OK_button)

        self.add_button.clicked.connect(self.add_property)
        self.OK_button.clicked.connect(self.OK_button_callback)
    
    def load_property_widgets(self):
        if self.scrollLayout.rowCount() > 0:
            for i in reversed(range(self.scrollLayout.rowCount())):
                self.scrollLayout.removeRow(i)

        self.prop_widgets = []
        for row_num, prop in enumerate(self.window.ui.manualproperties):
            self.prop_widgets.append({
                'MainWidget': QtWidgets.QWidget(),
                'MainLayout': QtWidgets.QGridLayout(),
                'LabelWidget': QtWidgets.QLineEdit(),
                'TypeWidget': QtWidgets.QComboBox(),
                'DeleteButton': QtWidgets.QPushButton('X'),
                'Callback': lambda row_num=row_num: self.remove_property(row_num)
            })

        for row_num, prop in enumerate(self.window.ui.manualproperties):
            self.prop_widgets[row_num]['MainWidget'].setLayout(self.prop_widgets[row_num]['MainLayout'])
            self.prop_widgets[row_num]['LabelWidget'].setText(prop['Label'])
            self.prop_widgets[row_num]['TypeWidget'].addItems(['True/False', 'Value'])
            self.prop_widgets[row_num]['DeleteButton'].setToolTip('Delete this property')
            self.prop_widgets[row_num]['DeleteButton'].clicked.connect(self.prop_widgets[row_num]['Callback'])


            if prop['Type'] == 'text':
                self.prop_widgets[row_num]['TypeWidget'].setCurrentText('Value')
            elif prop['Type'] == 'checkbox':
                self.prop_widgets[row_num]['TypeWidget'].setCurrentText('True/False')
            else:
                raise(Exception('Unknown value type entered in manual properties dialog...'))

            self.prop_widgets[row_num]['MainLayout'].addWidget(self.prop_widgets[row_num]['LabelWidget'], 0, 0)
            self.prop_widgets[row_num]['MainLayout'].addWidget(self.prop_widgets[row_num]['TypeWidget'], 0, 1)
            self.prop_widgets[row_num]['MainLayout'].addWidget(self.prop_widgets[row_num]['DeleteButton'], 0, 2)
            self.scrollLayout.addRow(self.prop_widgets[row_num]['MainWidget'])

    def add_property(self):
        num = len(self.window.ui.manualproperties)
        self.window.ui.manualproperties.append({
                'Label': 'Property'+str(num),
                'Type': 'checkbox',
                'Value': 0,
                'Widget': None
            })

        self.load_property_widgets()
        

    def remove_property(self, row):
        print('Deleting row num '+str(row))
        del self.window.ui.manualproperties[row]
        # load property widgets function
        self.load_property_widgets()

    def OK_button_callback(self):
        for i, pw in enumerate(self.prop_widgets):
            self.window.ui.manualproperties[i]['Label'] = pw['LabelWidget'].text()
            if pw['TypeWidget'].currentText() == 'Value':
                self.window.ui.manualproperties[i]['Type'] = 'text'
            elif pw['TypeWidget'].currentText() == 'True/False':
                self.window.ui.manualproperties[i]['Type'] = 'checkbox'
            else:
                raise(Exception('Unknown value type entered in manual properties dialog...'))

        self.window.ui.updateManualPropertiesBar()

        self.close()

    def closeEvent(self, event):
        self.window.ui.updateManualPropertiesBar()
        self.hide()
        #TODO: Ask whether the user wants the settings to be saved
        
    def init(self):
        self.show()

class DoubleSlider(QtWidgets.QSlider):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decimals = 5
        self._max_int = 10 ** self.decimals

        super().setMinimum(0)
        super().setMaximum(self._max_int)

        self._min_value = 0.0
        self._max_value = 1.0

    @property
    def _value_range(self):
        return self._max_value - self._min_value

    def value(self):
        return float(super().value()) / self._max_int * self._value_range + self._min_value

    def setValue(self, value):
        super().setValue(int((value - self._min_value) / self._value_range * self._max_int))

    def setMinimum(self, value):
        if value > self._max_value:
            raise ValueError("Minimum limit cannot be higher than maximum")

        self._min_value = value
        self.setValue(self.value())

    def setMaximum(self, value):
        if value < self._min_value:
            raise ValueError("Minimum limit cannot be higher than maximum")

        self._max_value = value
        self.setValue(self.value())

    def minimum(self):
        return self._min_value

    def maximum(self):
        return self._max_value

class MainWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)
        # dockwidget home
        self.dockarea = pg_da.DockArea()
        self.layout.addWidget(self.dockarea)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.dockarea.sizePolicy().hasHeightForWidth())
        self.dockarea.setSizePolicy(sizePolicy)
        self.dockarea.setBaseSize(QtCore.QSize(0, 2))

        ## manual properties bar
        self.manPropScroll = QtWidgets.QScrollArea(self)
        self.manPropWidget = QtWidgets.QWidget()
        self.manPropLayout = QtWidgets.QHBoxLayout(self.manPropWidget)
        # self.manPropWidget.setLayout(self.manPropLayout)
        self.manPropScroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.manPropScroll.setWidgetResizable(True)
        self.manPropScroll.setWidget(self.manPropWidget)
        self.layout.addWidget(self.manPropScroll)

        ## manage manual properties
        self.manPropButton = QtWidgets.QPushButton("Add/Remove Properties")
        self.manPropLayout.addWidget(self.manPropButton)
        self.generateExcelButton = QtWidgets.QPushButton('Generate Excel')
        self.manPropLayout.addWidget(self.generateExcelButton)
        self.manualproperties = [
            {
                'Label': 'Ignore',
                'Type': 'checkbox',
                'Value': 0,
                'Widget': None
            },
            {
                'Label': 'Comment',
                'Type': 'text',
                'Value': '',
                'Widget': None
            }]
        self.updateManualPropertiesBar()

        ## bottom bar
        self.bottomwidget = QtWidgets.QWidget()
        self.layout.addWidget(self.bottomwidget)
        self.bottomlayout = QtWidgets.QGridLayout()
        self.bottomwidget.setLayout(self.bottomlayout)
        # Num colors
        grid_numc = QtWidgets.QGridLayout()
        self.bottomlayout.addLayout(grid_numc, 0, 0)
        grid_numc.addWidget(QtWidgets.QLabel("NumColors:"), 0, 0)
        self.numColorsComboBox = QtWidgets.QComboBox()
        self.numColorsComboBox.addItems(["1", "2", "3"])
        self.numColorsComboBox.setCurrentText(DEFAULTS["Number of colors"])
        grid_numc.addWidget(self.numColorsComboBox, 0, 1)
        # Current ROI width
        grid_roi = QtWidgets.QGridLayout()
        self.bottomlayout.addLayout(grid_roi, 1, 0)
        grid_numc.addWidget(QtWidgets.QLabel("CurrROIwidth"), 1, 0)
        self.currROIWidthLabel = QtWidgets.QLabel(
            str(DEFAULT_PARAMETERS["ROI Width"]) + " pixels")
        grid_numc.addWidget(self.currROIWidthLabel, 1, 1)
        # detect loops
        grid_btn1 = QtWidgets.QGridLayout()
        self.bottomlayout.addLayout(grid_btn1, 0, 1)
        self.detectLoopsBtn = QtWidgets.QPushButton("Detect loops")
        self.detectLoopsBtn.setShortcut("Ctrl+D")
        grid_btn1.addWidget(self.detectLoopsBtn, 0, 0, 1, 1)
        self.findDNAendsBtn = QtWidgets.QPushButton("Find DNA Ends")
        grid_btn1.addWidget(self.findDNAendsBtn, 0, 1, 1, 1)

        self.processImageCheckBox = QtWidgets.QCheckBox("Process Image")
        self.processImageCheckBox.setChecked(True)
        grid_btn1.addWidget(self.processImageCheckBox, 1, 0)
        self.processImageComboBox = QtWidgets.QComboBox()
        self.processImageComboBox.addItems(["Median", "Bckg", "N2V", "Sparse-SIM"])
        grid_btn1.addWidget(self.processImageComboBox, 1, 1)
        # Kymograph
        grid_kymo = QtWidgets.QGridLayout()
        self.bottomlayout.addLayout(grid_kymo, 0, 2)
        self.RealTimeKymoCheckBox = QtWidgets.QCheckBox("RealTimeKymo")
        self.RealTimeKymoCheckBox.setChecked(True)
        self.updateKymoBtn = QtWidgets.QPushButton("UpdateKymo")
        self.updateKymoBtn.setShortcut("Ctrl+U")
        grid_kymo.addWidget(self.RealTimeKymoCheckBox, 0, 0)
        grid_kymo.addWidget(self.updateKymoBtn, 1, 0)
        # Frame number: start and end
        grid_frame = QtWidgets.QGridLayout()
        self.bottomlayout.addLayout(grid_frame, 0, 3)
        self.frameStartSpinBox = QtWidgets.QSpinBox()
        self.frameStartSpinBox.setRange(0, int(1e5))
        self.frameStartSpinBox.setValue(0)
        self.frameStartSpinBox.setKeyboardTracking(False)
        self.frameEndSpinBox = QtWidgets.QSpinBox()
        self.frameEndSpinBox.setRange(-1, int(1e5))
        self.frameEndSpinBox.setValue(-1)
        self.frameEndSpinBox.setKeyboardTracking(False)
        grid_frame.addWidget(QtWidgets.QLabel("FrameStart:"), 0, 0)
        grid_frame.addWidget(QtWidgets.QLabel("FrameEnd:"), 0, 2)
        grid_frame.addWidget(self.frameStartSpinBox, 0, 1)
        grid_frame.addWidget(self.frameEndSpinBox, 0, 3)
        self.save_nth_frameSpinBox = QtWidgets.QSpinBox()
        self.save_nth_frameSpinBox.setRange(1, int(1e5))
        self.save_nth_frameSpinBox.setValue(1)
        self.save_nth_frameSpinBox.setKeyboardTracking(False)
        grid_frame.addWidget(QtWidgets.QLabel("Save nth frames:"), 1, 0, 1, 2)
        grid_frame.addWidget(self.save_nth_frameSpinBox, 1, 2)
        # save section
        grid_save = QtWidgets.QGridLayout()
        self.bottomlayout.addLayout(grid_save, 0, 5)
        self.saveSectionBtn = QtWidgets.QPushButton("Save section")
        self.saveSectionComboBox = QtWidgets.QComboBox()
        self.saveSectionComboBox.addItems(["d0left", "d0right",
                                           "ROI:tif",
                                           "d1left:tif", "d1right:tif",
                                           "d2left:tif", "d2right:tif",
                                           ])
        self.saveFormatComboBox = QtWidgets.QComboBox()
        self.saveFormatComboBox.addItems([".mp4", ".avi", ".gif",
                                          ".pdf", ".png", ".jpeg",
                                          ".tif", ".svg"])
        self.saveFramerateSpinBox = QtWidgets.QSpinBox()
        self.saveFramerateSpinBox.setRange(1, int(1e3))
        self.saveFramerateSpinBox.setValue(7)
        self.saveFramerateSpinBox.setSuffix(" fps")
        grid_save.addWidget(self.saveSectionBtn, 0, 0)
        grid_save.addWidget(self.saveSectionComboBox, 1, 0)
        grid_save.addWidget(self.saveFormatComboBox, 0, 1)
        grid_save.addWidget(self.saveFramerateSpinBox, 1, 1)
        
        # swap or merge colors
        grid_colors = QtWidgets.QGridLayout()
        self.bottomlayout.addLayout(grid_colors, 0, 6)
        self.swapColorsBtn = QtWidgets.QPushButton("Swap colors")
        self.mergeColorsCheckBox = QtWidgets.QCheckBox("Merge colors")
        self.swapColorsComboBox = QtWidgets.QComboBox()
        self.swapColorsComboBox.addItems(["1<->2", "2<->3", "1<->3"])
        grid_colors.addWidget(self.swapColorsBtn, 0, 0)
        grid_colors.addWidget(self.swapColorsComboBox, 0, 1)
        grid_colors.addWidget(self.mergeColorsCheckBox, 1, 0)
    
    def clear_manPropLayout(self, L = False):
        if not L:
            L = self.manPropLayout
        if L is not None:
            # print('L: ' + str(L.count()))
            while L.count():
                item = L.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    if not (widget == self.manPropButton or widget == self.generateExcelButton):
                        widget.deleteLater()
                else:
                    self.clear_manPropLayout(item.layout())

    def updateManualPropertiesBar(self):
        # Remove all existing property widgets
        if self.manPropLayout.count() >2: #>2 because of the "Add/Remove Property" and "Generate Excel" buttons
            self.clear_manPropLayout()
        
        self.manPropLayout.addWidget(self.manPropButton)
        self.manPropLayout.addWidget(self.generateExcelButton)
        print(self.manPropButton)
        # Add property widgets
        for i, prop in enumerate(self.manualproperties):
            if prop['Type'] == 'text':
                prop['Widget'] =  QtWidgets.QGridLayout()
                # prop['Widget'].setMaximumWidth(200)
                self.manPropLayout.addLayout(prop['Widget'])
                prop['Widget'].addWidget(QtWidgets.QLabel(prop['Label'], alignment=QtCore.Qt.AlignRight), 0, 0)
                textbox = QtWidgets.QLineEdit()
                textbox.setMinimumWidth(100)
                if prop['Label'] != 'Comment':
                    textbox.setMaximumWidth(100)
                prop['Widget'].addWidget(textbox, 0, 1)
                prop['Textbox'] = textbox
                prop['Textbox'].setText(prop['Value'])
            elif prop['Type'] == 'checkbox':
                prop['Widget'] = QtWidgets.QCheckBox(prop['Label'])
                self.manPropLayout.addWidget(prop['Widget'])
                prop['Widget'].setChecked(bool(prop['Value']))
                

class Window(QtWidgets.QMainWindow):
    """ The main window """

    def __init__(self):
        super().__init__()
        # Init GUI
        self.title_string = "LEADS : Kymograph Analysis"
        self.setWindowTitle(self.title_string)
        this_directory = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(this_directory, "assets", "kymograph_window_bar.png")
        icon = QtGui.QIcon(icon_path)
        self.setWindowIcon(icon)
        self.resize(1200, 900)
        self.ui = MainWidget()
        self.setCentralWidget(self.ui)
        self.dockarea = self.ui.dockarea
        self.parameters_dialog = ParametersDialog(self)
        self.multipeak_dialog = MultiPeakDialog(self)
        self.file_dialog = FileDialog(self)
        self.roi_dialog = ROIDialog(self)
        self.sparseSIM_dialog = SparseSIMDialog(self)
        self.manual_properties_dialog = ManagePropertiesDialog(self)
        self.init_menu_bar()
        # load params
        self.load_parameters()
        # add figures
        self.add_col1_imvs()
        if self.numColors == "2":
            self.add_col2_imvs()
        elif self.numColors == "3":
            self.add_col2_imvs()
            self.add_col3_imvs()
        self.defaultDockState = self.dockarea.saveState()
        self.load_user_settings()
        self.connect_signals_init()
        self.connect_signals()
        self.multipeak_dialog.connect_signals()

    def init_menu_bar(self):
        menu_bar = self.menuBar()

        """ File """
        file_menu = menu_bar.addMenu("File")
        open_action = file_menu.addAction("Open image stack (.tif/.tiff)")
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.load_img_stack)
        openFolder_action = file_menu.addAction("Open folder containing .tif/.tiff")
        openFolder_action.setShortcut("Ctrl+F")
        openFolder_action.triggered.connect(self.file_dialog.show)        
        previousFile_action = file_menu.addAction("Previous file")
        previousFile_action.setShortcut("F1")
        previousFile_action.triggered.connect(self.previous_file)
        nextFile_action = file_menu.addAction("Next file")
        nextFile_action.setShortcut("F2")
        nextFile_action.triggered.connect(self.next_file)
        save_action = file_menu.addAction("Save (yaml and hdf5)")
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save)
        save_as_action = file_menu.addAction("Save as .hdf5")
        save_as_action.setShortcut("Ctrl+Shift+S")
        save_as_action.triggered.connect(self.save_as)
        loadyaml_action = file_menu.addAction("Load .yaml params")
        loadyaml_action.setShortcut("Ctrl+Y")
        loadyaml_action.triggered.connect(self.load_yaml_params_external)
        saveyaml_as_action = file_menu.addAction("Save .yaml params As")
        saveyaml_as_action.setShortcut("Ctrl+Shift+Y")
        saveyaml_as_action.triggered.connect(self.save_yaml_as)
        open_cropping_GUI_action = file_menu.addAction("Open cropping GUI")
        open_cropping_GUI_action.setShortcut("Ctrl+Shift+O")
        open_cropping_GUI_action.triggered.connect(self.openCroppingGUI)
        close_all_mpl_figures = file_menu.addAction("Close all figures")
        close_all_mpl_figures.setShortcut("Ctrl+Shift+W")
        close_all_mpl_figures.triggered.connect(self.closeAllMPLFigures)
        """ View """
        view_menu = menu_bar.addMenu("View")
        default_action = view_menu.addAction("Default View State")
        default_action.setShortcut("Ctrl+A")
        default_action.triggered.connect(self.restore_default_dockstate)
        hide_roi_action = view_menu.addAction("Hide Rect ROIs")
        hide_roi_action.triggered.connect(lambda x: self.show_or_hide_roi("hide"))
        show_roi_action = view_menu.addAction("Show Rect ROIs")
        show_roi_action.triggered.connect(lambda x: self.show_or_hide_roi("show"))
        """ Analyze """
        analyze_menu = menu_bar.addMenu("Analyze")
        parameters_action = analyze_menu.addAction("Parameters")
        parameters_action.setShortcut("Ctrl+P")
        parameters_action.triggered.connect(self.parameters_dialog.show)
        multipeak_action = analyze_menu.addAction("Multi Peak Analysis")
        multipeak_action.setShortcut("Ctrl+M")
        multipeak_action.triggered.connect(self.multipeak_dialog.show)
        roi_action = analyze_menu.addAction("ROI Analysis / Control")
        roi_action.setShortcut("Ctrl+R")
        roi_action.triggered.connect(self.roi_dialog.show)
        sparseSIM_action = analyze_menu.addAction("Sparse SIM parameters")
        sparseSIM_action.triggered.connect(self.sparseSIM_dialog.show)
        
        """ Image """
        image_menu = menu_bar.addMenu("Image")
        normalize_img_action = image_menu.addAction("Normalize Images")
        normalize_img_action.triggered.connect(self.normalize_images)
        normalize_kymo_action = image_menu.addAction("Normalize Kymographs")
        normalize_kymo_action.triggered.connect(lambda x: print("coming soon! Work is in progress!"))
        crop_img_roi_action = image_menu.addAction("Crop image in the rect ROI")
        crop_img_roi_action.triggered.connect(self.crop_img_rect_roi)
        flip_img_menu = image_menu.addMenu("Flip Images")
        flip_vertical_action = flip_img_menu.addAction("Vertically")
        flip_vertical_action.triggered.connect(lambda x: self.flip_images(2))
        flip_horizonta_action = flip_img_menu.addAction("Horizontally")
        flip_horizonta_action.triggered.connect(lambda x: self.flip_images(1))
        """ Help """
        help_menu = menu_bar.addMenu("Help")
        keyboardshortcuts_action = help_menu.addAction(" Keyboard Shortcuts")
        keyboardshortcuts_action.triggered.connect(lambda x: print("coming soon! Work is in progress!"))
        project_site_action = help_menu.addAction("Go to the project site")
        project_site_action.triggered.connect(lambda x: webbrowser.open('https://github.com/biswajitSM/LEADS'))

    def connect_signals_init(self):
        self.ui.numColorsComboBox.currentIndexChanged.connect(self.change_num_colors)
        self.ui.processImageCheckBox.stateChanged.connect(self.processed_image_check)
        self.ui.processImageComboBox.currentIndexChanged.connect(self.processed_image_check)
        self.ui.mergeColorsCheckBox.stateChanged.connect(self.merge_colors)
        self.ui.swapColorsBtn.clicked.connect(self.swap_colors)
        self.ui.detectLoopsBtn.clicked.connect(self.detect_loops)
        self.ui.saveSectionBtn.clicked.connect(self.save_section)
        self.ui.frameStartSpinBox.valueChanged.connect(self.frames_changed) #keyboardTracking=False, so works when entered
        self.ui.frameEndSpinBox.valueChanged.connect(self.frames_changed)
        self.ui.RealTimeKymoCheckBox.stateChanged.connect(self.realtime_kymo)
        self.ui.updateKymoBtn.clicked.connect(self.update_kymo)
        self.ui.findDNAendsBtn.clicked.connect(self.freshly_initialize_supergauss_dialog)
        self.ui.manPropButton.clicked.connect(self.manual_properties_dialog.init)
        self.ui.generateExcelButton.clicked.connect(self.generate_overview_excel_properties)

    def freshly_initialize_supergauss_dialog(self):
        self.supergauss_dialog = SuperGaussFittingDialog(self)
        self.supergauss_dialog.init()

    def connect_signals(self):
        # self.roirect_left.sigRegionChanged.connect(self.roi_changed)
        self.roirect_left.sigRegionChangeFinished.connect(self.roi_changed)
        self.infline_left.sigPositionChanged.connect(self.infiline_left_update)
        self.imv00.sigTimeChanged.connect(self.on_frame_change_imv00)
        if self.numColors == "2" or self.numColors == "3":
            self.infline_right.sigPositionChanged.connect(self.infiline_right_update)
            self.imv01.sigTimeChanged.connect(self.on_frame_change_imv01)
            if self.numColors == "3":
                self.infline_col3.sigPositionChanged.connect(self.infiline_col3_update)
                # self.imv02.sigTimeChanged.connect(self.on_frame_change_imv02)

    def load_parameters(self):
        self.numColors = self.ui.numColorsComboBox.currentText()
        self.LineROIwidth = self.parameters_dialog.roi_spinbox.value()
        self.pixelSize = 1e-3 * self.parameters_dialog.pix_spinbox.value()
        self.acquisitionTime = 1e-3 * int(self.numColors) * self.parameters_dialog.aqt_spinbox.value()
        # plot elements
        self.d0_col3 = None
        # get paramters from multipeak dialog
        self.peak_prominence = self.multipeak_dialog.prominence_spinbox.value()
        self.dna_length_kb = self.multipeak_dialog.DNAlength_spinbox.value()
        self.dna_puncta_size = self.multipeak_dialog.DNApuncta_spinbox.value()
        self.correction_with_no_loop = self.multipeak_dialog.loopcorrection_checkbox.isChecked()
        # parameters for smol (2nd color)
        self.peak_prominence_smol = self.multipeak_dialog.smol_prominence_spinbox.value()
        # multi peak paramters from analyze/peak analysis dialog
        self.search_range_link = self.multipeak_dialog.searchrange_spinbox.value()
        self.memory_link = self.multipeak_dialog.memory_spinbox.value()
        self.filter_length_link = self.multipeak_dialog.filterlen_spinbox.value()
        # initialize parameters that don't exist yet
        self.folderpath = None
        self.scalebar_img = None
        self.kymo_left = None
        self.max_peak_dict = None
        self.max_smpeak_dict = None
        self.df_peaks_linked = None
        self.linkedpeaks_analyzed = None
        self.df_cols_linked = None
        self.reload_kymo_gui = False

    def add_col1_imvs(self):
        self.imv00 = pg.ImageView(name='color 1')
        self.imv00.setPredefinedGradient(DEFAULTS["ColorMap"])
        self.hide_imgv_roi_norm(self.imv00)
        self.imv00.fps = 7
        # self.imv00.imageItem.setLookupTable(colormaps.pgcmap_parula.getLookupTable())
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
        self.region3_Loop = pg.LinearRegionItem((120, 200), brush=QtGui.QBrush(QtGui.QColor(255, 0, 0, 50)))
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
        self.infline_loopkymo_top = pg.InfiniteLine(movable=False, pos=0, angle=0, pen=(3, 9), label='left={value:0.0f}',
            labelOpts={'position':0.05, 'color': (200,200,100), 'fill': (200,200,200,25), 'movable': True})
        self.infline_loopkymo_bottom = pg.InfiniteLine(movable=False, pos=40, angle=0, pen=(3, 9), label='right={value:0.0f}',
            labelOpts={'position':0.05, 'color': (200,200,100), 'fill': (200,200,200,25), 'movable': True})
        self.imv21.addItem(self.infline_loopkymo_top)
        self.imv21.addItem(self.infline_loopkymo_bottom)
        # invert Y-axis of kymos
        self.imv10.view.invertY(False)
        self.imv20.view.invertY(False)
        self.imv21.view.invertY(False)

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

        # add text (fps) to images
        self.imv00_text = pg.TextItem(text="")
        self.imv00_text.setParentItem(self.imv00.imageItem)

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
        self.infline_right = pg.InfiniteLine(movable=True, angle=90, pen=(3, 9), label='x={value:0.0f}',
            labelOpts={'position':0.75, 'color': (200,200,100), 'fill': (200,200,200,25), 'movable': True})
        self.infline_right.addMarker(marker='v', position=0.7, size=10)
        self.infline_right.addMarker(marker='^', position=0.3, size=10)
        self.imv11.addItem(self.infline_right)
        
        self.imv22 = pg.ImageView()
        self.imv22.setPredefinedGradient(DEFAULTS["ColorMap"])
        self.hide_imgv_cmap(self.imv22)
        self.imv23 = pg.ImageView()
        self.imv23.setPredefinedGradient(DEFAULTS["ColorMap"])
        self.hide_imgv_roi_norm(self.imv23)
        
        # invert Y-axis of kymos
        self.imv11.view.invertY(False)
        self.imv22.view.invertY(False)
        self.imv23.view.invertY(False)

        self.infline_loopkymo_top_col2 = pg.InfiniteLine(movable=False, pos=0, angle=0, pen=(3, 9), label='left={value:0.0f}',
            labelOpts={'position':0.05, 'color': (200,200,100), 'fill': (200,200,200,25), 'movable': True})
        self.infline_loopkymo_bottom_col2 = pg.InfiniteLine(movable=False, pos=40, angle=0, pen=(3, 9), label='right={value:0.0f}',
            labelOpts={'position':0.05, 'color': (200,200,100), 'fill': (200,200,200,25), 'movable': True})
        self.imv23.addItem(self.infline_loopkymo_top_col2)
        self.imv23.addItem(self.infline_loopkymo_bottom_col2)

        self.d0_right = pg_da.Dock("d0right-single molecule")
        self.d0_right.addWidget(self.imv01)
        self.d1_right = pg_da.Dock("d1right-Kymograph single molecule")
        self.d1_right.addWidget(self.imv11)
        self.d2_right = pg_da.Dock("d2right-single molecule on NoLoop and Loop")
        self.d2_right.addWidget(self.imv22, 0, 0, 1, 1)
        self.d2_right.addWidget(self.imv23, 0, 1, 1, 5)

        self.dockarea.addDock(self.d0_right, 'right')
        self.dockarea.addDock(self.d1_right, 'bottom', self.d0_right)
        self.dockarea.addDock(self.d2_right, 'bottom', self.d1_right)

        # add text (fps) to images
        self.imv01_text = pg.TextItem(text="")
        self.imv01_text.setParentItem(self.imv01.imageItem)

    def add_col3_imvs(self):
        self.imv02 = pg.ImageView(name='color 2')
        self.imv02.setPredefinedGradient(DEFAULTS["ColorMap"])
        self.hide_imgv_roi_norm(self.imv02)
        self.imv02.playRate = 7
        self.roirect_col3 = pg.LineROI([20, 20], [40, 20], width=11, pen=(3, 9))        
        self.imv02.addItem(self.roirect_col3)

        self.plot_kymo_col3 = pg.PlotItem(name='red_kymo')
        self.plot_kymo_col3.hideAxis('left')#; self.plot4.hideAxis('bottom')
        self.plot_kymo_col3.setXLink(self.plot3); self.plot_kymo_col3.setYLink(self.plot3)
        self.imv12 = pg.ImageView(view=self.plot_kymo_col3)
        self.imv12.setPredefinedGradient(DEFAULTS["ColorMap"])
        self.hide_imgv_cmap(self.imv12)
        self.infline_col3 = pg.InfiniteLine(movable=True, angle=90, pen=(3, 9), label='x={value:0.0f}',
            labelOpts={'position':0.75, 'color': (200,200,100), 'fill': (200,200,200,25), 'movable': True})
        self.infline_col3.addMarker(marker='v', position=0.7, size=10)
        self.infline_col3.addMarker(marker='^', position=0.3, size=10)
        self.imv12.addItem(self.infline_col3)
        
        self.imv24 = pg.ImageView()
        self.imv24.setPredefinedGradient(DEFAULTS["ColorMap"])
        self.hide_imgv_cmap(self.imv24)
        self.imv25 = pg.ImageView()
        self.imv25.setPredefinedGradient(DEFAULTS["ColorMap"])
        self.hide_imgv_roi_norm(self.imv25)
        
        # invert Y-axis of kymos
        self.imv12.view.invertY(False)
        self.imv24.view.invertY(False)
        self.imv25.view.invertY(False)

        self.d0_col3 = pg_da.Dock("d0col3-single molecule")
        self.d0_col3.addWidget(self.imv02)
        self.d1_col3 = pg_da.Dock("d1col3-Kymograph single molecule")
        self.d1_col3.addWidget(self.imv12)
        self.d2_col3 = pg_da.Dock("d2col3-single molecule on NoLoop and Loop")
        self.d2_col3.addWidget(self.imv24, 0, 0, 1, 1)
        self.d2_col3.addWidget(self.imv25, 0, 1, 1, 5)

        self.dockarea.addDock(self.d0_col3, 'right')
        self.dockarea.addDock(self.d1_col3, 'bottom', self.d0_col3)
        self.dockarea.addDock(self.d2_col3, 'bottom', self.d1_col3)

        # add text (fps) to images
        self.imv02_text = pg.TextItem(text="")
        self.imv02_text.setParentItem(self.imv02.imageItem)

    def remove_all_widgets(self):
        self.dockarea.clear()

    def restore_default_dockstate(self):
        self.dockarea.restoreState(self.defaultDockState)
        if self.kymo_left is not None:
            shape = self.kymo_left.shape
            self.imv10.view.autoRange()
            self.imv10.view.setXRange(0, shape[0])

    def set_scalebar(self):
        self.pixelSize = 1e-3 * self.parameters_dialog.pix_spinbox.value()
        self.acquisitionTime = 1e-3 * int(self.numColors) * self.parameters_dialog.aqt_spinbox.value() # converted to sec from ms
        if self.scalebar_img is not None:
            self.scalebar_img.size = 2/self.pixelSize
            self.scalebar_img.updateBar()
            if self.numColors == "2" or self.numColors == "3":
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
            if self.numColors == "2" or self.numColors == "3":
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


    def load_folder(self):
        if self.folderpath is None:
            self.folderpath = io.FileDialog('', "Select a folder",).openFolderNameDialog()
        else:
            self.folderpath = io.FileDialog(os.path.dirname(self.folderpath), "Select a folder",).openFolderNameDialog()
        filenames = sorted(glob.glob(self.folderpath + "/*.tif"), key=alphanumeric_key)
        if len(filenames)==0:
            filenames = sorted(glob.glob(self.folderpath + "/*.tiff"), key=alphanumeric_key)
        if len(filenames)==0:
            print('No .tif/.tiff files found in directory. Returning.')
            return

        # list containing extensions that leads itself produces - we usually dont want to find those here
        exclude = ['_left_kymo.tif', '_right_kymo.tif', '_left_selected_kymo.tif', '_right_selected_kymo.tif', '_ROI.tif', '_right_frame', '_right.tif', '_left_frame', '_left.tif']
        # find all files which carry a 'processed.tif' tag and omit those
        filenames_clean = []
        for filename in filenames:
            filename = filename.replace('_processed.tif', '.tif')
            if any([(exclusion in filename) for exclusion in exclude]):
                continue
            filenames_clean.append( filename )
        filenames_clean = sorted(list(set(filenames_clean)))
        self.filenames_all = filenames_clean
        self.numFiles = len(filenames_clean)
        self.searchROIs()        
        self.currentFile = 0
        settings = io.load_user_settings()
        settings["kymograph"]["PWD"] = self.folderpath
        settings["kymograph"]["Acquisiton Time"] = self.parameters_dialog.aqt_spinbox.value()
        settings["kymograph"]["Pixel Size"] = self.parameters_dialog.pix_spinbox.value()
        settings["kymograph"]['ROI width'] = self.LineROIwidth
        io.save_user_settings(settings)
        self.file_dialog.PathQLabel.setText(self.folderpath)
        self.updateFileDialog()
        # self.display_file()

    def display_file(self):
        if self.currentFile<(len(self.filenames)):
            if os.path.isfile(self.filenames[self.currentFile]):
                try:
                    self.load_img_stack(filepath=self.filenames[self.currentFile])
                except:
                    self.load_img_stack(filepath=self.filenames[self.currentFile].replace('.tif', '_processed.tif'))
            else:
                self.load_img_stack(filepath=self.filenames[self.currentFile].replace('.tif', '_processed.tif'))

    def next_file(self):
        if not hasattr(self, 'filenames'):
            return
        if self.currentFile == self.numFiles-1:
            print('Last .tif file in folder already displayed.')  
            return      
        self.currentFile += 1
        self.display_file()

    def previous_file(self):
        if not hasattr(self, 'filenames'):
            return
        if self.currentFile == 0:
            print('First .tif file in folder already displayed.')
            return
        self.currentFile -= 1
        self.display_file()

    def any_file(self):
        if not hasattr(self, 'filenames'):
            return        
        self.currentFile = int( self.file_dialog.fileNumberSpinBox.value()-1 )
        self.display_file()

    def searchROIs(self):
        if not hasattr(self, 'filenames_all'):
            return
        if len(self.file_dialog.ROI_key_lineedit.text())==0:
            self.numFiles = len(self.filenames_all)
            self.filenames = self.filenames_all
        else:
            filenames_filtered = []
            for filename in self.filenames_all:
                searchpattern = '.' + self.file_dialog.ROI_key_lineedit.text().replace('*', '.') + '.' # we allow both . and * for any character in the GUI, re only accepts . though
                if searchpattern[0:1] == '..':
                    searchpattern = searchpattern[1:]
                if searchpattern[-2:] == '..':
                    searchpattern = searchpattern[:-1]
                searchpattern = searchpattern.replace('.', '.*') # the star looks for one or more repetitions of the dot
                filename = filename.replace('.tif', '') # remove .tif since the . confuses re
                reResult = re.search(searchpattern, filename)
                if hasattr(reResult, 'span'): # then we found something
                    filenames_filtered.append(filename+'.tif')
            filenames_filtered = sorted(list(set(filenames_filtered)))
            self.numFiles = len(filenames_filtered)
            self.filenames = filenames_filtered
        self.updateFileDialog()
        self.currentFile = 0
        self.display_file()

    def print_all_filenames(self):
        if not hasattr(self, 'filenames_all'):
            return
        print('All files in folder '+os.path.dirname(self.filenames_all[0]))
        numFiles = len(self.filenames_all)
        numDigits = len(str(numFiles))
        for i in range(numFiles):
            print('['+str(i+1).zfill(numDigits)+'/'+str(numFiles)+'] '+os.path.basename(self.filenames_all[i]))
        print('\n')

    def print_filtered_filenames(self):
        if not hasattr(self, 'filenames'):
            return
        print('Filtered files in folder '+os.path.dirname(self.filenames[0]))
        numFiles = len(self.filenames)
        numDigits = len(str(numFiles))
        for i in range(numFiles):
            print('['+str(i+1).zfill(numDigits)+'/'+str(numFiles)+'] '+os.path.basename(self.filenames[i]))
        print('\n')

    def updateFileDialog(self):
        self.file_dialog.PathQLabel.setText(self.folderpath)
        self.file_dialog.fileNumberLabel.setText("/"+str(self.numFiles))
        self.file_dialog.fileNumberSpinBox.setMaximum(self.numFiles)

    def generate_overview_excel(self):
        if not hasattr(self, 'filenames'):
            return
        self.xlxsName = os.path.join(self.folderpath, str(int(time.time()*1e7))+'_'+os.path.basename(os.path.dirname(self.folderpath))+'_overview.xlsx')
        workbook = xlsxwriter.Workbook(self.xlxsName)
        worksheet = workbook.add_worksheet()
        manual_props_list = [p['Label'] for p in self.ui.manualproperties]
        keywords = ['#', 'Folderpath', 'Name'] + manual_props_list
        bold = workbook.add_format({'bold': True})
        row = 0
        for col in range(len(keywords)):
            # get_col_widths(dataframe)
            worksheet.write(row, col, keywords[col], bold)
            # worksheet.set_column(col, col, 2*int(col==0 or col==1 or col==2 or col==7 )+len(keywords[col]))
        count = 0
        for filename in self.filenames:
            row += 1
            count += 1
            worksheet.write(row, 0, count)
            worksheet.write(row, 2, os.path.basename(filename))
            worksheet.write(row, 1, os.path.dirname(filename))
            
            hdf5_file = filename.replace('.tif', '_analysis.hdf5')
            if os.path.isfile(hdf5_file) == True:
                with h5py.File(hdf5_file, 'r') as h5_analysis:
                    if 'manual_properties' in h5_analysis.keys():
                        for prop in h5_analysis['manual_properties']:
                            p = h5_analysis['manual_properties'][prop]
                            label = p['Label'][()].decode() # hdf5 files store as byte strings so we use decode...
                            index = int(p['Index'][()])
                            value = p['Value'][()]
                            if type(value) == type(b''):
                                value = value.decode()
                            var_type = p['Type'][()].decode()
                            
                            if label in manual_props_list:
                                worksheet.write(row, 3 + manual_props_list.index(label), value)
                

        worksheet.freeze_panes(1, 0) # fix the first row
        workbook.close()
        try:
            subprocess.Popen(r'explorer /select,"'+self.xlxsName.replace('/', '\\')+'"')
        except:
            pass

    def generate_overview_excel_properties(self):
        # print(self.filenames)
        continue_excel = True
        for f in self.filenames:
            filename = f.replace('.tif', '_analysis.hdf5')
            if os.path.isfile(filename) == False:
                continue_excel = False
                print('No file exists')
                break
            else:
                print(filename)
        if continue_excel == False:
            qm = QtWidgets.QMessageBox()
            ret = qm.question(self,'Excel generation warning', "Not all cropped ROIs have associated hdf5 files, would you still like to generate an Excel?", qm.Yes | qm.No)

            if ret == qm.Yes:
                self.generate_overview_excel()
        else:
            self.generate_overview_excel()

    def load_img_stack(self, filepath=None):
        folder_open = False
        self.sparseSIM_dialog.backup_saved = False
        if filepath is None or (not filepath):
            filepath = io.FileDialog(self.folderpath, "Open a tif file stack",
                                    "Tif file (*.tif)").openFileNameDialog()
            self.setWindowTitle(self.title_string + '-' + filepath)
        else:
            self.setWindowTitle(self.title_string + '-' + ' [' + str(self.currentFile+1) + '/' + str(self.numFiles) + '] '+ filepath)
                                
        # set paths and file names
        self.filepath = filepath
        self.folderpath = os.path.dirname(self.filepath)
        filename = os.path.basename(self.filepath)
        (self.filename_base, ext) = os.path.splitext(filename)
        # read the image
        self.image_meta = read_img_stack(self.filepath)
        self.frame_start = 0
        self.frame_end = -1
        self.set_img_stack()
        self.load_yaml_params()
        if self.ui.processImageCheckBox.isChecked() and ('_processed' != self.filename_base[-len('_processed'):]):
            self.image_meta = self.get_processed_image()
            self.set_img_stack()
        # disconnect the dependent signals
        self.multipeak_dialog.disconnect_signals()
        try:
            self.set_yaml_params()
        except Exception as e:
            print(e)
            pass
        # connect back the dependent signals
        self.multipeak_dialog.connect_signals()

        self.load_manual_properties()

    def set_img_stack(self):
        print("\nLoading and processing the image ...")
        start_time = time.time()
        if self.image_meta["num_colors"] == 1 and self.numColors == "2":
            print("Only one channel exists in the tif stack!\n \
                Can't display two colors")
            self.ui.numColorsComboBox.setCurrentText("1")
        elif self.image_meta["num_colors"] == 1 and self.numColors == "3":
            print("Only one(1) channels exists in the tif stack!\n \
                Can't display 3 colors")
            self.ui.numColorsComboBox.setCurrentText("1")
        elif self.image_meta["num_colors"] == 2 and self.numColors == "3":
            print("Only two(2) channels exists in the tif stack!\n \
                Can't display 3 colors")
            self.ui.numColorsComboBox.setCurrentText("2")
        if self.numColors == "3":
            self.imgarr_left = self.image_meta['img_arr_color_0'][self.frame_start:self.frame_end, ...]
            self.imgarr_right = self.image_meta['img_arr_color_1'][self.frame_start:self.frame_end, ...]
            self.imgarr_col3 = self.image_meta['img_arr_color_2'][self.frame_start:self.frame_end, ...]
            self.imv02.setImage(self.imgarr_col3)
            if self.ui.mergeColorsCheckBox.isChecked():
                arr_combined = np.concatenate((self.imgarr_right[:, :, :, np.newaxis],
                                            self.imgarr_left[:, :, :, np.newaxis],
                                            self.imgarr_col3[:, :, :, np.newaxis],),
                                            axis=3)
                self.imv01.setImage(arr_combined, levelMode='rgba')
            else:
                self.imv01.setImage(self.imgarr_right)
            self.imv01.showMaximized()
        elif self.numColors == "2":
            self.imgarr_left = self.image_meta['img_arr_color_0'][self.frame_start:self.frame_end, ...]
            self.imgarr_right = self.image_meta['img_arr_color_1'][self.frame_start:self.frame_end, ...]
            if self.ui.mergeColorsCheckBox.isChecked():
                arr_combined = np.concatenate((self.imgarr_right[:, :, :, np.newaxis],
                                            self.imgarr_left[:, :, :, np.newaxis],
                                            np.zeros_like(self.imgarr_right[:, :, :, np.newaxis])),
                                            axis=3)
                self.imv01.setImage(arr_combined, levelMode='rgba')
            else:
                self.imv01.setImage(self.imgarr_right)
            self.imv01.showMaximized()
        elif self.numColors == "1":
            self.imgarr_left = self.image_meta['img_arr_color_0'][self.frame_start:self.frame_end, ...]
        self.imv00.setImage(self.imgarr_left)
        self.imv00.showMaximized()
        self.roi_changed()
        self.region_Loop_changed()
        self.region_noLoop_changed()
        self.set_scalebar()
        # print("took %s seconds!" % (time.time() - start_time))
        print("Loading done. Displaying file "+self.filepath)

    def load_img_seq(self):
        return

    def load_yaml_params(self):
        self.filepath_yaml = os.path.join(self.folderpath, self.filename_base + '_params.yaml')
        if os.path.isfile(self.filepath_yaml):
            with open(self.filepath_yaml) as f:
                self.params_yaml = yaml.safe_load(f)
        else:
            with open(self.filepath_yaml, 'w') as f:
                self.params_yaml = {
                    'filepath' : self.filepath,
                    'folderpath' : self.folderpath,
                    'Pixel Size' : self.parameters_dialog.pix_spinbox.value(),
                    'ROI width' : None,
                    'Region Errbar': [10, 30],
                    'MultiPeak' : {},
                }
                shape = list(self.imgarr_left[0, ...].shape)
                self.params_yaml["Region Errbar"] = [1, max(shape)]
                yaml.dump(self.params_yaml, f)
        return self.params_yaml

    def load_yaml_params_external(self):
        filepath_yaml = io.FileDialog(self.filepath_yaml, "open yaml parameter file",
                                 "yaml (*.yaml)").openFileNameDialog()
        if os.path.isfile(filepath_yaml):
            with open(filepath_yaml) as f:
                self.params_yaml = yaml.safe_load(f)
        self.set_yaml_params()

    def save_yaml_as(self):
        yamlpath = io.FileDialog(self.filepath_yaml, 'Save parameters as',
                        'YAML File(*.yaml)').saveFileDialog()
        self.save_yaml_params(yamlpath)

    def save_yaml_params(self, filepath_yaml):
        self.params_yaml = self.load_yaml_params()
        self.params_yaml['Pixel Size'] = self.parameters_dialog.pix_spinbox.value()
        self.params_yaml['ROI width'] = self.parameters_dialog.roi_spinbox.text()
        self.params_yaml['roi1 state'] = {}
        self.params_yaml['roi1 state']['position'] = list(self.roirect_left.pos()) #[self.roirect_left.pos()[0], self.roirect_left.pos()[1]]
        self.params_yaml['roi1 state']['size'] = list(self.roirect_left.size())
        self.params_yaml['roi1 state']['angle'] = float(self.roirect_left.angle())
        self.params_yaml['region3_noLoop'] = list(self.region3_noLoop.getRegion())
        self.params_yaml['region3_Loop'] = list(self.region3_Loop.getRegion())
        if self.plot_loop_errbar is not None:
            self.params_yaml['Region Errbar'] = list(self.region_errbar.getRegion())
            self.params_yaml['dna ends'] = [float(x) for x in self.dna_ends]#list(self.dna_ends)
        multipeak_settings = self.multipeak_dialog.on_close_event()
        self.params_yaml["MultiPeak"] = multipeak_settings["kymograph"]["MultiPeak"]
        with open(filepath_yaml, 'w') as f:
            yaml.dump(self.params_yaml, f)

    def save(self):
        self.save_yaml_params(self.filepath_yaml)
        print("Parameters saved to yaml file")
        # check if we overwrite or save a new file
        if self.parameters_dialog.SaveNewAnalysisFile_checkbox.isChecked():
            suffix = self.parameters_dialog.SaveNewAnalysisFile_lineedit.text()
            if len(suffix)>0: suffix = ('_'+suffix).replace('__','_')
            existingAdditionalFiles = glob.glob(os.path.join(self.folderpath, self.filename_base+'*analysis_additionalFile'+suffix+'*'))
            if len(existingAdditionalFiles)==0:
                runningNumber = "{0:03}".format(1)
            else:
                # find position of '_additionalFile'
                numbers = []
                for file in existingAdditionalFiles:
                    numbers.append( int(file.split('additionalFile'+suffix)[-1].split('.hdf5')[0].replace('_','')) )
                runningNumber = "{0:03}".format(max(numbers)+1)
            filepath_hdf5 = os.path.join(self.folderpath, self.filename_base + '_analysis_additionalFile'+suffix+runningNumber+'.hdf5')
        else:
            filepath_hdf5 = os.path.join(self.folderpath, self.filename_base + '_analysis.hdf5')
        self.save_hdf5(filepath_hdf5)
        print("output and metadata saved to hdf5 file")

    def save_as(self):
        suggest_name = self.folderpath+'/'+self.filename_base + '_analysis.hdf5'
        hdf5path = io.FileDialog(suggest_name, 'Save analysis outputs as',
                        'HDF5 File(*.hdf5)').saveFileDialog()
        self.save_hdf5(hdf5path)
        (name_without_ext, _) = os.path.splitext(os.path.basename(hdf5path))
        yamlpath = os.path.join(os.path.dirname(hdf5path), name_without_ext + '.yaml')
        self.save_yaml_params(yamlpath)

    def set_yaml_params(self):
        if self.params_yaml['ROI width'] is not None:
            self.roirect_left.setPos(self.params_yaml['roi1 state']['position'])
            self.roirect_left.setSize(self.params_yaml['roi1 state']['size'])
            self.roirect_left.setAngle(self.params_yaml['roi1 state']['angle'])

            self.region3_Loop.setRegion(self.params_yaml['region3_Loop'])
            self.region3_noLoop.setRegion(self.params_yaml['region3_noLoop'])
        if "MultiPeak" in self.params_yaml and len(self.params_yaml["MultiPeak"])>0:
            multipeak_setting = {"kymograph" : {}}
            multipeak_setting["kymograph"]["MultiPeak"] = self.params_yaml["MultiPeak"]
            try:
                _ = self.multipeak_dialog.on_start_event(settings=multipeak_setting)
            except Exception as e:
                print(e)
                pass
        if "Region Errbar" in self.params_yaml:
            if self.plot_loop_errbar is None:
                self.set_loop_detection_widgets()
            self.region_errbar.setRegion(self.params_yaml['Region Errbar'])
            self.detect_loops()

    def closeAllMPLFigures(self):
        plt.close('all')

    def openCroppingGUI(self):
        crop_images_gui.main()

    def processed_image_check(self):
        if self.ui.processImageCheckBox.isChecked():
            self.image_meta = self.get_processed_image()
            self.set_img_stack()
        else:
            self.image_meta = read_img_stack(self.filepath)
            self.set_img_stack()

    def get_processed_image(self, preview=False):
        if self.ui.processImageComboBox.currentText() == "Median" and not preview:
            fpath_processed = os.path.join(self.folderpath, self.filename_base + '_processed.tif')
            if os.path.isfile(fpath_processed):
                self.image_meta = read_img_stack(fpath_processed)
            else:
                if self.numColors == "3":
                    self.imgarr_left = median_bkg_substration(self.imgarr_left)
                    self.imgarr_right = median_bkg_substration(self.imgarr_right)
                    self.imgarr_col3 = median_bkg_substration(self.imgarr_col3)
                    comb_arr = np.concatenate((self.imgarr_left[:,np.newaxis,:,:],
                                            self.imgarr_right[:,np.newaxis,:,:],
                                            self.imgarr_col3[:,np.newaxis,:,:]),
                                            axis=1)
                    imwrite(fpath_processed, comb_arr, imagej=True,
                            metadata={'axis': 'TCYX', 'channels': self.numColors,
                            'mode': 'composite',})
                elif self.numColors == "2":
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

        elif self.ui.processImageComboBox.currentText() == "Bckg" and not preview:
            fpath_processed = os.path.join(self.folderpath, self.filename_base + '_bckgprocessed.tif')
            if os.path.isfile(fpath_processed):
                self.image_meta = read_img_stack(fpath_processed)
            else:
                if self.numColors == "3":
                    self.imgarr_left = median_bkg_substration(self.imgarr_left, onlyBackground=True)
                    self.imgarr_right = median_bkg_substration(self.imgarr_right, onlyBackground=True)
                    self.imgarr_col3 = median_bkg_substration(self.imgarr_col3, onlyBackground=True)
                    comb_arr = np.concatenate((self.imgarr_left[:,np.newaxis,:,:],
                                            self.imgarr_right[:,np.newaxis,:,:],
                                            self.imgarr_col3[:,np.newaxis,:,:]),
                                            axis=1)
                    imwrite(fpath_processed, comb_arr, imagej=True,
                            metadata={'axis': 'TCYX', 'channels': self.numColors,
                            'mode': 'composite',})
                elif self.numColors == "2":
                    self.imgarr_left = median_bkg_substration(self.imgarr_left, onlyBackground=True)
                    self.imgarr_right = median_bkg_substration(self.imgarr_right, onlyBackground=True)
                    comb_arr = np.concatenate((self.imgarr_left[:,np.newaxis,:,:],
                                            self.imgarr_right[:,np.newaxis,:,:]),
                                            axis=1)
                    imwrite(fpath_processed, comb_arr, imagej=True,
                            metadata={'axis': 'TCYX', 'channels': self.numColors,
                            'mode': 'composite',})
                elif self.numColors == "1":
                    self.imgarr_left = median_bkg_substration(self.imgarr_left, onlyBackground=True)
                    imwrite(fpath_processed, self.imgarr_left, imagej=True,
                            metadata={'axis': 'TCYX', 'channels': self.numColors,
                            'mode': 'composite',})
                self.image_meta = read_img_stack(fpath_processed)

        elif self.ui.processImageComboBox.currentText() == "N2V":
            fpath_processed = os.path.join(self.folderpath, self.filename_base + '_N2V_processed.tif')
            if os.path.isfile(fpath_processed):
                self.image_meta = read_img_stack(fpath_processed)
            else:
                self.image_meta = read_img_stack(self.filepath)
                print("N2V processed file doesn't exist")

        elif self.ui.processImageComboBox.currentText() == "Sparse-SIM" or preview:
            if preview:
                fpath_processed = os.path.join(self.folderpath, self.filename_base + '_SparseSIM_processed_preview.tif')
            else:
                fpath_processed = os.path.join(self.folderpath, self.filename_base + '_SparseSIM_processed.tif')
            if os.path.isfile(fpath_processed) and not preview: # if preview is on, we are changing parameters, so re-compute
                self.image_meta = read_img_stack(fpath_processed)
            else:
                # obtain sparseSIM parameters
                sigma = self.sparseSIM_dialog.sigma_spinbox.value()
                numIter = self.sparseSIM_dialog.numIter_spinbox.value()
                fidelity = self.sparseSIM_dialog.fidelity_spinbox.value()
                sparsity = self.sparseSIM_dialog.sparsity_spinbox.value()
                tcontinuity = self.sparseSIM_dialog.tcontinuity_spinbox.value()
                background = self.sparseSIM_dialog.background_combobox.currentText()
                deconv_iter = self.sparseSIM_dialog.deconv_iter_spinbox.value()
                deconv_type = self.sparseSIM_dialog.deconv_type_combobox.currentText()
                if deconv_type=='No deconvolution':
                    deconv_type = 0
                elif deconv_type=='LandWeber deconvolution':
                    deconv_type = 1
                elif deconv_type=='Richardson-Lucy deconvolution':
                    deconv_type = 2

                if background=="No background":
                    background = 0
                elif background=="Weak background (HI)":
                    background = 1
                elif background=="Strong background (HI)":
                    background = 2
                elif background=="Weak background (LI)":
                    background = 3
                elif background=="With background (LI)":
                    background = 4
                elif background=="Strong background (LI)":
                    background = 5
                
                sparseSIMparameters = {
                    "sigma": sigma,
                    "numIter": numIter,
                    "fidelity": fidelity,
                    "sparsity": sparsity,
                    "tcontinuity": tcontinuity,
                    "background": background,
                    "deconv_iter": deconv_iter,
                    "deconv_type": deconv_type
                }

                if self.numColors == "3":
                    if preview:
                        imgarr_left = np.load(os.path.join(self.folderpath, 'imgarr_left.npy'))
                        imgarr_right = np.load(os.path.join(self.folderpath, 'imgarr_right.npy'))
                        imgarr_col3 = np.load(os.path.join(self.folderpath, 'imgarr_col3.npy'))
                        self.imgarr_left = applySparseSIM(imgarr_left, params=sparseSIMparameters, preview=preview)
                        self.imgarr_right = applySparseSIM(imgarr_right, params=sparseSIMparameters, preview=preview)
                        self.imgarr_col3 = applySparseSIM(imgarr_col3, params=sparseSIMparameters, preview=preview)
                    else:
                        self.imgarr_left = applySparseSIM(self.imgarr_left, params=sparseSIMparameters, preview=preview)
                        self.imgarr_right = applySparseSIM(self.imgarr_right, params=sparseSIMparameters, preview=preview)
                        self.imgarr_col3 = applySparseSIM(self.imgarr_col3, params=sparseSIMparameters, preview=preview)
                    comb_arr = np.concatenate((self.imgarr_left[:,np.newaxis,:,:],
                                            self.imgarr_right[:,np.newaxis,:,:],
                                            self.imgarr_col3[:,np.newaxis,:,:]),
                                            axis=1)
                    imwrite(fpath_processed, comb_arr, imagej=True,
                            metadata={'axis': 'TCYX', 'channels': self.numColors,
                            'mode': 'composite',})
                elif self.numColors == "2":
                    if preview:
                        imgarr_left = np.load(os.path.join(self.folderpath, 'imgarr_left.npy'))
                        imgarr_right = np.load(os.path.join(self.folderpath, 'imgarr_right.npy'))
                        self.imgarr_left = applySparseSIM(imgarr_left, params=sparseSIMparameters, preview=preview)
                        self.imgarr_right = applySparseSIM(imgarr_right, params=sparseSIMparameters, preview=preview)
                    else:
                        self.imgarr_left = applySparseSIM(self.imgarr_left, params=sparseSIMparameters, preview=preview)
                        self.imgarr_right = applySparseSIM(self.imgarr_right, params=sparseSIMparameters, preview=preview)
                    comb_arr = np.concatenate((self.imgarr_left[:,np.newaxis,:,:],
                                            self.imgarr_right[:,np.newaxis,:,:]),
                                            axis=1)
                    imwrite(fpath_processed, comb_arr, imagej=True,
                            metadata={'axis': 'TCYX', 'channels': self.numColors,
                            'mode': 'composite',})
                elif self.numColors == "1":
                    if preview:
                        imgarr_left = np.load(os.path.join(self.folderpath, 'imgarr_left.npy'))
                        self.imgarr_left = applySparseSIM(imgarr_left, params=sparseSIMparameters, preview=preview)
                    else:
                        self.imgarr_left = applySparseSIM(self.imgarr_left, params=sparseSIMparameters, preview=preview)
                    
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
        # self.kymo_left = self.kymo_left / np.max(self.kymo_left)
        self.imv10.setImage(self.kymo_left)
        if self.numColors == "3":
            # get kymo of color 2
            self.roirect_right.setState(self.roirect_left.getState())
            roi_data = self.roirect_right.getArrayRegion(self.imgarr_right,
                                                self.imv01.imageItem, axes=(1, 2))                                                       
            self.kymo_right = np.sum(roi_data, axis=2)
            self.kymo_right = self.kymo_right / np.max(self.kymo_right)
            # get kymo of color 2
            self.roirect_col3.setState(self.roirect_left.getState())
            roi_data = self.roirect_col3.getArrayRegion(self.imgarr_col3,
                                                self.imv02.imageItem, axes=(1, 2))
            self.kymo_col3 = np.sum(roi_data, axis=2)
            self.kymo_col3 = self.kymo_col3 / np.max(self.kymo_col3)
            self.imv12.setImage(self.kymo_col3)
            if self.ui.mergeColorsCheckBox.isChecked():
                self.kymo_comb = np.concatenate((self.kymo_right[:, :, np.newaxis],
                                                 self.kymo_left[:, :, np.newaxis],
                                                 self.kymo_col3[:, :, np.newaxis],),
                                                 axis=2)
                kymo_comb = self.kymo_comb
                for nChannel in range(kymo_comb.shape[2]):
                    temp = kymo_comb[:,:,nChannel]
                    temp /= np.max(temp)
                    kymo_comb[:,:,nChannel] = temp * (2**16-1)
                self.kymo_comb = kymo_comb

                self.imv11.setImage(self.kymo_comb, levelMode='rgba')
                self.imv11.ui.histogram.show()
            else:
                self.imv11.setImage(self.kymo_right)
        elif self.numColors == "2":
            self.roirect_right.setState(self.roirect_left.getState())
            roi_data = self.roirect_right.getArrayRegion(self.imgarr_right,
                                                self.imv01.imageItem, axes=(1, 2))                                                       
            self.kymo_right = np.sum(roi_data, axis=2)
            self.kymo_right = self.kymo_right / np.max(self.kymo_right)
            if self.ui.mergeColorsCheckBox.isChecked():
                self.kymo_comb = np.concatenate((self.kymo_right[:, :, np.newaxis],
                                    self.kymo_left[:, :, np.newaxis],
                                    np.zeros_like(self.kymo_right[:, :, np.newaxis])),
                                    axis=2)
                
                kymo_comb = self.kymo_comb[:,:,:-1]
                for nChannel in range(kymo_comb.shape[2]):
                    temp = kymo_comb[:,:,nChannel]
                    temp /= np.max(temp)
                    kymo_comb[:,:,nChannel] = temp * (2**16-1)
                self.kymo_comb[:,:,:-1] = kymo_comb
                
                self.imv11.setImage(self.kymo_comb, levelMode='rgba')
                self.imv11.ui.histogram.show()
            else:
                self.imv11.setImage(self.kymo_right)

    def crop_img_rect_roi(self):
        self.ui.mergeColorsCheckBox.setChecked(False)
        if self.numColors == "3":
            self.roirect_col3.setState(self.roirect_left.getState())
            self.roirect_right.setState(self.roirect_left.getState())
            self.imgarr_left = self.roirect_left.getArrayRegion(self.imgarr_left,
                                                self.imv00.imageItem, axes=(1, 2))
            self.imgarr_right = self.roirect_right.getArrayRegion(self.imgarr_right,
                                                self.imv01.imageItem, axes=(1, 2))
            self.imgarr_col3 = self.roirect_col3.getArrayRegion(self.imgarr_col3,
                                                self.imv02.imageItem, axes=(1, 2))
            self.imv00.setImage(self.imgarr_left)
            self.imv01.setImage(self.imgarr_right)
            self.imv02.setImage(self.imgarr_col3)
        elif self.numColors == "2":
            self.roirect_right.setState(self.roirect_left.getState())
            self.imgarr_left = self.roirect_left.getArrayRegion(self.imgarr_left,
                                                self.imv00.imageItem, axes=(1, 2))
            self.imgarr_right = self.roirect_right.getArrayRegion(self.imgarr_right,
                                                self.imv01.imageItem, axes=(1, 2))
            self.imv00.setImage(self.imgarr_left)
            self.imv01.setImage(self.imgarr_right)
        elif self.numColors == "1":
            self.imgarr_left = self.roirect_left.getArrayRegion(self.imgarr_left,
                                                self.imv00.imageItem, axes=(1, 2))
            self.imv00.setImage(self.imgarr_left)
    
    def flip_images(self, flip_axis=2):
        print(self.imgarr_left.ndim, self.imgarr_left.shape)
        if self.numColors == "3":
            self.imgarr_left = np.flip(self.imgarr_left, axis=flip_axis)
            self.imgarr_right = np.flip(self.imgarr_right, axis=flip_axis)
            self.imgarr_col3 = np.flip(self.imgarr_col3, axis=flip_axis)
            self.imv00.setImage(self.imgarr_left)
            self.imv01.setImage(self.imgarr_right)
            self.imv02.setImage(self.imgarr_col3)
        elif self.numColors == "2":
            self.imgarr_left = np.flip(self.imgarr_left, axis=flip_axis)
            self.imgarr_right = np.flip(self.imgarr_right, axis=flip_axis)
            self.imv00.setImage(self.imgarr_left)
            self.imv01.setImage(self.imgarr_right)
        elif self.numColors == "1":
            self.imgarr_left = np.flip(self.imgarr_left, axis=flip_axis)
            self.imv00.setImage(self.imgarr_left)

    def normalize_images(self):
        self.imgarr_left = 1000*self.imgarr_left/self.imgarr_left.max()
        self.imv00.setImage(self.imgarr_left)

    def on_frame_change_imv00(self):
        frame_imv00 = self.imv00.currentIndex
        self.imv00_text.setText(str(np.round(frame_imv00 * self.acquisitionTime, 1)) + ' s')
        self.infline_left.setValue(frame_imv00)
        if int(self.numColors) > 1:
            self.imv01.setCurrentIndex(frame_imv00)
            self.imv01_text.setText(str(np.round(frame_imv00 * self.acquisitionTime, 1)) + ' s')

    def on_frame_change_imv01(self):
        frame_imv01 = self.imv01.currentIndex
        self.imv00.setCurrentIndex(frame_imv01)
        self.imv01_text.setText(str(np.round(frame_imv01 * self.acquisitionTime, 1)) + ' s')

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
        roi1_state_update['size'][1] = self.parameters_dialog.roi_spinbox.value()
        self.roirect_left.setState(roi1_state_update)

    def show_or_hide_roi(self, kind="hide"):
        if kind == "hide":
            pen = None
        else:
            pen = (255, 255, 0)
        handles = self.roirect_left.getHandles()
        if self.numColors == "3":
            self.roirect_left.setPen(pen)
            self.roirect_right.setPen(pen)
            self.roirect_col3.setPen(pen)
        elif self.numColors == "2":
            self.roirect_left.setPen(pen)
            self.roirect_right.setPen(pen)
        elif self.numColors == "1":
            self.roirect_left.setPen(pen)

    def timetrace_rectROI(self, side='left'):
        if side == 'left':
            roi1_data = self.roirect_left.getArrayRegion(self.imgarr_left,
                                                self.imv00.imageItem, axes=(1, 2))
        elif side == 'right' and self.numColors == "2":
            roi1_data = self.roirect_right.getArrayRegion(self.imgarr_right,
                                                self.imv01.imageItem, axes=(1, 2))
        else:
            roi1_data = None
        win = pg.plot()
        win.setWindowTitle('Time trace of ROI: ' + str(side))
        if roi1_data is not None:
            timetrace = np.average(roi1_data, axis=(1,2))
            win.plot(timetrace)
            win.setYRange(0, np.max(timetrace))

    def extract_rectROI(self, side='left'):
        roi_data = None
        win = pg.image(title='Extracted images of ROI: ' + str(side))
        if side == 'left':
            roi_data = self.roirect_left.getArrayRegion(self.imgarr_left,
                                                self.imv00.imageItem, axes=(1, 2))
            win.setPredefinedGradient(DEFAULTS["ColorMap"])
            win.setImage(roi_data)
        elif side == 'right' and self.numColors == "2":
            roi_data = self.roirect_right.getArrayRegion(self.imgarr_right,
                                                self.imv01.imageItem, axes=(1, 2))
            win.setPredefinedGradient(DEFAULTS["ColorMap"])
            win.setImage(roi_data)
        elif side == 'both' and self.numColors == "2":
            roi_data_1 = self.roirect_left.getArrayRegion(self.imgarr_left,
                                                self.imv00.imageItem, axes=(1, 2))
            roi2_data_2 = self.roirect_right.getArrayRegion(self.imgarr_right,
                                                self.imv01.imageItem, axes=(1, 2))
            roi_data = np.concatenate((roi2_data_2[:, :, :, np.newaxis],
                                        roi_data_1[:, :, :, np.newaxis],
                                        np.zeros_like(roi2_data_2[:, :, :, np.newaxis])),
                                        axis=3)
            win.setImage(roi_data, levelMode='rgba')        

    def infiline_left_update(self):
        frame_numer = int(self.infline_left.value())
        pos = self.infline_left.getPos()
        self.imv00.setCurrentIndex(frame_numer)
        if self.numColors == "2" or self.numColors == "3":
            self.imv01.setCurrentIndex(frame_numer)
            self.infline_right.setPos(pos)
            if self.numColors == "3":
                self.imv02.setCurrentIndex(frame_numer)
                self.infline_col3.setPos(pos)

    def infiline_right_update(self):
        frame_numer = int(self.infline_right.value())
        pos = self.infline_right.getPos()
        self.imv00.setCurrentIndex(frame_numer)
        self.imv01.setCurrentIndex(frame_numer)
        self.infline_left.setPos(pos)
        if self.numColors == "3":
            self.imv02.setCurrentIndex(frame_numer)
            self.infline_col3.setPos(pos)
    
    def infiline_col3_update(self):
        frame_number = int(self.infline_col3.value())
        pos = self.infline_col3.getPos()
        self.imv00.setCurrentIndex(frame_number)
        self.imv01.setCurrentIndex(frame_number)
        self.imv02.setCurrentIndex(frame_number)
        self.infline_left.setPos(pos)
        self.infline_right.setPos(pos)

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
        self.reload_kymo_gui = True
        self.numColors = self.ui.numColorsComboBox.currentText()
        DEFAULTS["Number of colors"] = self.numColors
        self.close()
        main()
        
    def region_noLoop_changed(self):
        minX, maxX = self.region3_noLoop.getRegion()
        self.kymo_left_noLoop = self.kymo_left[int(minX):int(maxX), :]
        # background set to 0
        self.kymo_left_noLoop = self.kymo_left_noLoop - np.min(np.average(self.kymo_left_noLoop, axis=0))
        self.imv20.setImage(self.kymo_left_noLoop)
        if self.numColors == "3":
            self.kymo_right_noLoop = self.kymo_right[int(minX):int(maxX), :]
            self.kymo_right_noLoop = self.kymo_right_noLoop - np.min(np.average(self.kymo_right_noLoop, axis=0))
            self.kymo_col3_noLoop = self.kymo_col3[int(minX):int(maxX), :]
            self.kymo_col3_noLoop = self.kymo_col3_noLoop - np.min(np.average(self.kymo_col3_noLoop, axis=0))
            self.imv24.setImage(self.kymo_col3_noLoop)
            if self.ui.mergeColorsCheckBox.isChecked():
                kymo_noLoop_comb = np.concatenate((self.kymo_right_noLoop[:, :, np.newaxis],
                                        self.kymo_left_noLoop[:, :, np.newaxis],
                                        self.kymo_col3_noLoop[:, :, np.newaxis],),
                                        axis=2)

                for nChannel in range(kymo_noLoop_comb.shape[2]):
                    temp = kymo_noLoop_comb[:,:,nChannel]
                    temp /= np.max(temp)
                    kymo_noLoop_comb[:,:,nChannel] = temp * (2**16-1)

                self.imv22.setImage(kymo_noLoop_comb, levelMode='rgba')
            else:
                self.imv22.setImage(self.kymo_right_noLoop)
        elif self.numColors == "2":
            self.kymo_right_noLoop = self.kymo_right[int(minX):int(maxX), :]
            self.kymo_right_noLoop = self.kymo_right_noLoop - np.min(np.average(self.kymo_right_noLoop, axis=0))
            if self.ui.mergeColorsCheckBox.isChecked():
                kymo_noLoop_comb = np.concatenate((self.kymo_right_noLoop[:, :, np.newaxis],
                                        self.kymo_left_noLoop[:, :, np.newaxis],
                                        np.zeros_like(self.kymo_right_noLoop[:, :, np.newaxis])),
                                        axis=2)
                
                for nChannel in range(kymo_noLoop_comb.shape[2]-1):
                    temp = kymo_noLoop_comb[:,:,nChannel]
                    temp /= np.max(temp)
                    kymo_noLoop_comb[:,:,nChannel] = temp * (2**16-1)

                self.imv22.setImage(kymo_noLoop_comb, levelMode='rgba')
            else:
                self.imv22.setImage(self.kymo_right_noLoop)   

    def region_Loop_changed(self):
        minX, maxX = self.region3_Loop.getRegion()
        self.kymo_left_loop = self.kymo_left[int(minX):int(maxX), :]
        # background set to 0
        self.kymo_left_loop = self.kymo_left_loop - np.min(np.average(self.kymo_left_loop, axis=0))
        self.imv21.setImage(self.kymo_left_loop)
        if self.numColors == "3":
            self.kymo_col3_loop = self.kymo_col3[int(minX):int(maxX), :]
            self.kymo_col3_loop = self.kymo_col3_loop - np.min(np.average(self.kymo_col3_loop, axis=0))
            self.kymo_right_loop = self.kymo_right[int(minX):int(maxX), :]
            self.kymo_right_loop = self.kymo_right_loop - np.min(np.average(self.kymo_right_loop, axis=0))
            self.imv25.setImage(self.kymo_col3_loop)
            if self.ui.mergeColorsCheckBox.isChecked():
                kymo_loop_comb = np.concatenate((self.kymo_right_loop[:, :, np.newaxis],
                                        self.kymo_left_loop[:, :, np.newaxis],
                                        self.kymo_col3_loop[:, :, np.newaxis],),
                                        axis=2)

                for nChannel in range(kymo_loop_comb.shape[2]):
                    temp = kymo_loop_comb[:,:,nChannel]
                    temp /= np.max(temp)
                    kymo_loop_comb[:,:,nChannel] = temp * (2**16-1)

                self.imv23.setImage(kymo_loop_comb, levelMode='rgba')
            else:
                self.imv23.setImage(self.kymo_right_loop)
        elif self.numColors == "2":
            self.kymo_right_loop = self.kymo_right[int(minX):int(maxX), :]
            self.kymo_right_loop = self.kymo_right_loop - np.min(np.average(self.kymo_right_loop, axis=0))
            if self.ui.mergeColorsCheckBox.isChecked():
                kymo_loop_comb = np.concatenate((self.kymo_right_loop[:, :, np.newaxis],
                                        self.kymo_left_loop[:, :, np.newaxis],
                                        np.zeros_like(self.kymo_right_loop[:, :, np.newaxis])),
                                        axis=2)

                for nChannel in range(kymo_loop_comb.shape[2]-1):
                    temp = kymo_loop_comb[:,:,nChannel]
                    temp /= np.max(temp)
                    kymo_loop_comb[:,:,nChannel] = temp * (2**16-1)

                self.imv23.setImage(kymo_loop_comb, levelMode='rgba')
            else:
                self.imv23.setImage(self.kymo_right_loop)
        # also clear the loop position data point
        if self.plot_loop_errbar is not None:
            self.plotLoopPosData.clear()

    def merge_colors(self):
        # rgba images need to have 3 colors (arr[t, x, y, c]). c must be 3 
        if self.ui.mergeColorsCheckBox.isChecked():
            if self.numColors == "3":
                arr_combined = np.concatenate((self.imgarr_right[:, :, :, np.newaxis],
                                            self.imgarr_left[:, :, :, np.newaxis],
                                            self.imgarr_col3[:, :, :, np.newaxis],),
                                            axis=3)
                self.imv01.setImage(arr_combined, levelMode='rgba')
                self.imv01.showMaximized()
                self.kymo_comb = np.concatenate((self.kymo_right[:, :, np.newaxis],
                                        self.kymo_left[:, :, np.newaxis],
                                        self.kymo_col3[:, :, np.newaxis],),
                                        axis=2)

                kymo_comb = self.kymo_comb[:,:,:-1]
                for nChannel in range(kymo_comb.shape[2]):
                    temp = kymo_comb[:,:,nChannel]
                    temp /= np.max(temp)
                    kymo_comb[:,:,nChannel] = temp * (2**16-1)
                self.kymo_comb[:,:,:-1] = kymo_comb
                
                self.imv11.setImage(self.kymo_comb, levelMode='rgba')
                self.imv11.ui.histogram.show()
                self.kymo_loop_comb = np.concatenate((self.kymo_right_loop[:, :, np.newaxis],
                                        self.kymo_left_loop[:, :, np.newaxis],
                                        self.kymo_col3_loop[:, :, np.newaxis],),#CHANGE to proper array
                                        axis=2)

                kymo_loop_comb = self.kymo_loop_comb[:,:,:-1]
                for nChannel in range(kymo_loop_comb.shape[2]):
                    temp = kymo_loop_comb[:,:,nChannel]
                    temp /= np.max(temp)
                    kymo_loop_comb[:,:,nChannel] = temp * (2**16-1)
                self.kymo_loop_comb[:,:,:-1] = kymo_loop_comb

                kymo_noLoop_comb = np.concatenate((self.kymo_right_noLoop[:, :, np.newaxis],
                                        self.kymo_left_noLoop[:, :, np.newaxis],
                                        self.kymo_col3_noLoop[:, :, np.newaxis],),
                                        axis=2)

                for nChannel in range(kymo_noLoop_comb.shape[2]-1):
                    temp = kymo_noLoop_comb[:,:,nChannel]                    
                    temp /= np.max(temp)
                    kymo_noLoop_comb[:,:,nChannel] = temp * (2**16-1)

                self.imv22.setImage(kymo_noLoop_comb, levelMode='rgba')
                self.imv23.setImage(self.kymo_loop_comb, levelMode='rgba')
            if self.numColors == "2":
                arr_combined = np.concatenate((self.imgarr_right[:, :, :, np.newaxis],
                                            self.imgarr_left[:, :, :, np.newaxis],
                                            np.zeros_like(self.imgarr_right[:, :, :, np.newaxis])),
                                            axis=3)
                self.imv01.setImage(arr_combined, levelMode='rgba')
                self.imv01.showMaximized()
                self.kymo_comb = np.concatenate((self.kymo_right[:, :, np.newaxis],
                                        self.kymo_left[:, :, np.newaxis],
                                        np.zeros_like(self.kymo_right[:, :, np.newaxis])),
                                        axis=2)

                kymo_comb = self.kymo_comb[:,:,:-1]
                for nChannel in range(kymo_comb.shape[2]):
                    temp = kymo_comb[:,:,nChannel]
                    temp /= np.max(temp)
                    kymo_comb[:,:,nChannel] = temp * (2**16-1)
                self.kymo_comb[:,:,:-1] = kymo_comb

                self.imv11.setImage(self.kymo_comb, levelMode='rgba')
                self.imv11.ui.histogram.show()
                self.kymo_loop_comb = np.concatenate((self.kymo_right_loop[:, :, np.newaxis],
                                        self.kymo_left_loop[:, :, np.newaxis],
                                        np.zeros_like(self.kymo_right_loop[:, :, np.newaxis])),
                                        axis=2)
                kymo_noLoop_comb = np.concatenate((self.kymo_right_noLoop[:, :, np.newaxis],
                                        self.kymo_left_noLoop[:, :, np.newaxis],
                                        np.zeros_like(self.kymo_right_noLoop[:, :, np.newaxis])),
                                        axis=2)

                kymo_loop_comb = self.kymo_loop_comb[:,:,:-1]
                for nChannel in range(kymo_loop_comb.shape[2]):
                    temp = kymo_loop_comb[:,:,nChannel]
                    temp /= np.max(temp)
                    kymo_loop_comb[:,:,nChannel] = temp * (2**16-1)
                self.kymo_loop_comb[:,:,:-1] = kymo_loop_comb

                for nChannel in range(kymo_noLoop_comb.shape[2]-1):
                    temp = kymo_noLoop_comb[:,:,nChannel]
                    
                    temp /= np.max(temp)
                    kymo_noLoop_comb[:,:,nChannel] = temp * (2**16-1)

                self.imv22.setImage(kymo_noLoop_comb, levelMode='rgba')
                self.imv23.setImage(self.kymo_loop_comb, levelMode='rgba')
                # self.imv23.addItem(self.infline_loopkymo_top)
                # self.imv23.addItem(self.infline_loopkymo_bottom)
        else:
            # set back the imagedata
            self.imv01.setImage(self.imgarr_right, levelMode='mono')
            self.imv01.showMaximized()
            self.imv11.setImage(self.kymo_right, levelMode='mono')
            self.imv22.setImage(self.kymo_right_noLoop, levelMode='mono')
            self.imv23.setImage(self.kymo_right_loop, levelMode='mono')
        return

    def swap_colors(self):
        if self.numColors == "2"  or "3":
            if self.ui.swapColorsComboBox.currentText() == "1<->2":
                print("am at switching colors")
                temp_arr = self.image_meta['img_arr_color_0']
                self.image_meta['img_arr_color_0'] = self.image_meta['img_arr_color_1']
                self.image_meta['img_arr_color_1'] = temp_arr
                self.set_img_stack()
            elif self.ui.swapColorsComboBox.currentText() == "2<->3":
                temp_arr = self.image_meta['img_arr_color_1']
                self.image_meta['img_arr_color_1'] = self.image_meta['img_arr_color_2']
                self.image_meta['img_arr_color_2'] = temp_arr
                self.set_img_stack()
            elif self.ui.swapColorsComboBox.currentText() == "1<->3":
                temp_arr = self.image_meta['img_arr_color_2']
                self.image_meta['img_arr_color_2'] = self.image_meta['img_arr_color_0']
                self.image_meta['img_arr_color_0'] = temp_arr
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
        print('set region errbar')
        self.plot_loop_errbar.addItem(self.region_errbar, ignoreBounds=True)
        self.dna_ends = [2, 60]
        self.dna_infline_left = pg.InfiniteLine(movable=True, pos=self.dna_ends[0], angle=90, pen=(3, 9), label='dna left end ={value:0.0f}',
            labelOpts={'position':0.15, 'angle':90, 'color': (200,200,100), 'fill': (200,200,200,25), 'movable': True})
        self.plot_loop_errbar.addItem(self.dna_infline_left, ignoreBounds=True)
        self.dna_infline_right = pg.InfiniteLine(movable=True, pos=self.dna_ends[1], angle=90, pen=(3, 9), label='dna right end ={value:0.0f}',
            labelOpts={'position':0.15, 'angle':90, 'color': (200,200,100), 'fill': (200,200,200,25), 'movable': True})
        self.plot_loop_errbar.addItem(self.dna_infline_right, ignoreBounds=True)
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
            symbol='o', symbolSize=5, pen=pg.mkPen('r'), symbolPen=pg.mkPen(None))#symbolBrush=pg.mkPen('r')
        if self.numColors == "2" or self.numColors == "3":
            self.d3_right = pg_da.Dock("Single Molecule detections")
            self.dockarea.addDock(self.d3_right, 'bottom', self.d2_right)
            self.plot_loop_vs_sm = pg.PlotItem()
            self.imv31 = pg.ImageView(view=self.plot_loop_vs_sm)
            self.imv31.setPredefinedGradient(DEFAULTS["ColorMap"])
            self.plot_loop_vs_sm.getViewBox().invertY(False)
            self.hide_imgv_cmap(self.imv31)
            self.d3_right.addWidget(self.imv31)
            # self.d3_right.addWidget(self.plot_loop_vs_sm)
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
            self.plotSmolPosData = self.plot_loop_vs_sm.scatterPlot(
                    symbol='o', symbolSize=5, pen=pg.mkPen('r'), symbolPen=pg.mkPen(None))
            if self.numColors == "3":
                self.d3_right = pg_da.Dock("Color 3 detections")
                self.dockarea.addDock(self.d3_right, 'bottom', self.d2_col3)
        # change the default docking positions to the new one
        self.defaultDockState = self.dockarea.saveState()
        self.dockarea.restoreState(self.defaultDockState)

    def params_change_loop_detection(self):
        self.peak_prominence = self.multipeak_dialog.prominence_spinbox.value()
        self.dna_length_kb = self.multipeak_dialog.DNAlength_spinbox.value()
        self.dna_puncta_size = self.multipeak_dialog.DNApuncta_spinbox.value()
        self.correction_with_no_loop = self.multipeak_dialog.loopcorrection_checkbox.isChecked()
        if self.multipeak_dialog.preview_checkbox.isChecked():
            self.preview_maxpeak_on_params_change()

    def preview_maxpeak_on_params_change(self):
        if self.plot_loop_errbar is None:
            self.set_loop_detection_widgets()
        self.loop_region_left = int(self.region_errbar.getRegion()[0])
        self.loop_region_right = int(self.region_errbar.getRegion()[1])
        if self.loop_region_left < 0:
            self.loop_region_left = 0
            self.dna_ends[0] = self.loop_region_left + 5
            self.region_errbar.setRegion((self.loop_region_left, self.loop_region_right))
        if self.loop_region_right > self.kymo_left_loop.shape[1]:
            self.loop_region_right = self.kymo_left_loop.shape[1] - 1
            self.dna_ends[1] = self.loop_region_left - 5
            self.region_errbar.setRegion((self.loop_region_left, self.loop_region_right))
        self.dna_ends[0] = float(self.dna_infline_left.getXPos())
        self.dna_ends[1] = float(self.dna_infline_right.getXPos())
        self.infline_loopkymo_top.setPos(self.dna_ends[0])
        self.infline_loopkymo_bottom.setPos(self.dna_ends[1])
        if int(self.numColors) > 1:
            self.infline_loopkymo_top_col2.setPos(self.dna_ends[0])
            self.infline_loopkymo_bottom_col2.setPos(self.dna_ends[1])
        
        min_peak_width = self.multipeak_dialog.minwidth_spinbox.value()
        max_peak_width = self.multipeak_dialog.maxwidth_spinbox.value()
        self.peak_prominence = self.multipeak_dialog.prominence_spinbox.value()
        self.all_peaks_dict = peakfinder_savgol(self.kymo_left_loop.T,
                self.loop_region_left, -self.loop_region_right,
                prominence_min=self.peak_prominence,
                peak_width=(min_peak_width, max_peak_width),
                pix_width=self.dna_puncta_size,
                smooth_length=self.multipeak_dialog.smoothlength_spinbox.value(),
                threshold_glbal_peak=self.multipeak_dialog.peakthreshold_checkbox.isChecked(),
                threshold_value=self.multipeak_dialog.peakthreshold_spinbox.value(),
                plotting=False, kymo_noLoop=self.kymo_left_noLoop.T,
                correction_noLoop=self.correction_with_no_loop
                )
        self.max_peak_dict = analyze_maxpeak(self.all_peaks_dict['Max Peak'], smooth_length=7,
                frame_width = self.dna_ends[1] - self.dna_ends[0],
                dna_length=self.dna_length_kb, pix_width=self.dna_puncta_size,
                )
        if self.numColors == "2" or self.numColors == "3":
            self.all_smpeaks_dict = peakfinder_savgol(self.kymo_right_loop.T,
                self.loop_region_left, -self.loop_region_right,
                prominence_min=self.peak_prominence_smol,
                peak_width=(min_peak_width, max_peak_width),
                pix_width=self.dna_puncta_size,
                smooth_length=self.multipeak_dialog.smoothlength_spinbox.value(),
                threshold_glbal_peak=self.multipeak_dialog.peakthreshold_checkbox.isChecked(),
                threshold_value=self.multipeak_dialog.peakthreshold_spinbox.value(),
                plotting=False,
                )
        self.plotLoopPosData.setData(self.all_peaks_dict["All Peaks"]["FrameNumber"],
                                     self.all_peaks_dict["All Peaks"]["PeakPosition"])

    def params_change_smol_detection(self):
        if self.numColors == "2" or self.numColors == "3":
            self.peak_prominence_smol = self.multipeak_dialog.smol_prominence_spinbox.value()
            if self.multipeak_dialog.smol_preview_checkbox.isChecked():
                self.preview_smol_peaks_on_params_change()

    def preview_smol_peaks_on_params_change(self):
        if self.plot_loop_errbar is None:
            self.set_loop_detection_widgets()
        self.loop_region_left = int(self.region_errbar.getRegion()[0])
        self.loop_region_right = int(self.region_errbar.getRegion()[1])
        min_peak_width = self.multipeak_dialog.minwidth_spinbox.value()
        max_peak_width = self.multipeak_dialog.maxwidth_spinbox.value()
        if self.numColors == "2" or self.numColors == "3":
            self.all_smpeaks_dict = peakfinder_savgol(self.kymo_right_loop.T,
                self.loop_region_left, -self.loop_region_right,
                prominence_min=self.peak_prominence_smol,
                peak_width=(min_peak_width, max_peak_width),
                smooth_length=self.multipeak_dialog.smoothlength_spinbox.value(),
                threshold_glbal_peak=self.multipeak_dialog.peakthreshold_checkbox.isChecked(),
                threshold_value=self.multipeak_dialog.peakthreshold_spinbox.value(),
                pix_width=self.dna_puncta_size, plotting=False,)
            self.plot_loop_vs_sm_smdata.clear()
            self.plot_loop_vs_sm_loopdata.clear()
            self.imv31.setImage(self.kymo_right_loop)
            self.plotSmolPosData.setData(self.all_smpeaks_dict["All Peaks"]["FrameNumber"],
                                     self.all_smpeaks_dict["All Peaks"]["PeakPosition"])

    def compute_contour_length(self):
        SXO_nM = [0, 10, 50, 100, 200, 500]
        Lc_correction_factor = [1, 1.0258, 1.0523, 1.0649, 1.0948, 1.3829] # for non-coilable DNA
        SxOconc = self.multipeak_dialog.SxOconc_spinbox.value()
        DNAlength_kb = self.multipeak_dialog.DNAlength_spinbox.value()
        correction_factor = np.interp(SxOconc, SXO_nM, Lc_correction_factor)
        DNAcontourlength = DNAlength_kb * 0.342 * correction_factor
        self.multipeak_dialog.DNAcontourlength_spinbox.setValue(DNAcontourlength)

    def compute_persistence_length(self):
        SXO_nM = [0, 10, 50, 100, 200, 500]
        Lp = [46.1, 41.9, 36, 35.1, 37.1, 37.2] # for non-coilable DNA
        SxOconc = self.multipeak_dialog.SxOconc_spinbox.value()
        Lp_interp = np.interp(SxOconc, SXO_nM, Lp)
        self.multipeak_dialog.DNApersistencelength_spinbox.setValue(Lp_interp)

    def dna_ends_changed(self, fromSuperGaussWindow=True):
        if fromSuperGaussWindow: # if it came from the clicking on 'Detect loops', self.dna_ends is already populated
            self.dna_ends = self.supergauss_dialog.dna_ends
        self.dna_infline_left.setPos(self.dna_ends[0])
        self.dna_infline_right.setPos(self.dna_ends[1])
        self.infline_loopkymo_top.setPos(self.dna_ends[0])
        self.infline_loopkymo_bottom.setPos(self.dna_ends[1])
        self.infline_loopkymo_top_col2.setPos(self.dna_ends[0])
        self.infline_loopkymo_bottom_col2.setPos(self.dna_ends[1])

    def detect_loops(self):
        if self.plot_loop_errbar is None:
            self.set_loop_detection_widgets()
        # self.d3_left.setStretch(200, 200)
        # if self.numColors == "2":
        #     self.d3_right.setStretch(200, 200)
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
        self.preview_maxpeak_on_params_change()
        frame_no = self.max_peak_dict["Max Peak"]["FrameNumber"]
        peak_pos = self.max_peak_dict["Max Peak"]["PeakPosition"]
        int_peak = self.max_peak_dict["Max Peak"]["PeakIntensity"]
        int_peakup = self.max_peak_dict["Max Peak"]["PeakUpIntensity"]
        int_peakdown = self.max_peak_dict["Max Peak"]["PeakDownIntensity"]
        self.plotLoopPosData.setData(frame_no, peak_pos)
        self.plot_data_loop.setData(frame_no, int_peak)
        self.plot_data_loopUp.setData(frame_no, int_peakup)
        self.plot_data_loopDown.setData(frame_no, int_peakdown)
        self.plot_data_loop_filt.setData(frame_no,
                        self.max_peak_dict["Max Peak"]["PeakIntFiltered"])
        self.plot_data_loopUp_filt.setData(frame_no,
                        self.max_peak_dict["Max Peak"]["PeakIntUpFiltered"])
        self.plot_data_loopDown_filt.setData(frame_no,
                        self.max_peak_dict["Max Peak"]["PeakIntDownFiltered"])
        if self.numColors == "2" or self.numColors == "3":
            # self.all_smpeaks_dict = peakfinder_savgol(self.kymo_right_loop.T,
            #     self.loop_region_left, -self.loop_region_right,
            #     prominence_min=self.peak_prominence, pix_width=self.dna_puncta_size, plotting=False,
            #     # kymo_noLoop=self.kymo_left_noLoop.T, #use this carefully, safe way is to put it none or just comment this line
            #     )
            self.imv31.clear()
            self.plotSmolPosData.clear()
            self.max_smpeak_dict = analyze_maxpeak(self.all_smpeaks_dict['Max Peak'], smooth_length=7,
                    frame_width = self.dna_ends[1] - self.dna_ends[0],
                    dna_length=self.dna_length_kb, pix_width=self.dna_puncta_size,
                    )
            max_loop_sm_dict = loop_sm_dist(self.max_peak_dict, self.max_smpeak_dict, smooth_length=21)
            self.plot_sm_dist.setData(max_loop_sm_dict['FrameNumber'],
                                      max_loop_sm_dict['PeakDiffFiltered'])
            # update data in smVsLoop plot
            self.plot_loop_vs_sm_smdata.setData(
                    self.max_smpeak_dict['Max Peak']['FrameNumber'],
                    self.max_smpeak_dict['Max Peak']['PeakPosition'])
            self.plot_loop_vs_sm_loopdata.setData(
                    self.max_peak_dict['Max Peak']['FrameNumber'],
                    self.max_peak_dict['Max Peak']['PeakPosition'])
            self.plot_loop_vs_sm.setYRange(self.loop_region_left-5, self.loop_region_right+5)
            self.plot_loop_vs_sm_linetop.setValue(self.loop_region_left)
            self.plot_loop_vs_sm_linebottom.setValue(self.loop_region_right)

        # also by default try to find the DNA ends by peak peeling with default parameters
        if self.parameters_dialog.DefaultFindDNAends_checkbox.isChecked():
            self.dna_ends = kymograph.find_ends_peakPeeling(noLoop_avg, noPeaks=10, \
                    PSF=300/(self.pixelSize*1e3), amplitude=0.9, \
                        residue=0.1, plotting=False)
            if self.dna_ends is not None:
                self.dna_ends_changed(fromSuperGaussWindow=False)


    def matplot_all_peaks(self, usePrecomputed=None, usePrecomputedsm=None):
        plt.rcParams["savefig.directory"] = self.filepath # default saving dir is the path of the current file
        if len(self.filename_base)==0 and hasattr(self, 'filename_base_clipboard'):
            df=pd.DataFrame([self.filename_base_clipboard + '_'])            
        else:
            df=pd.DataFrame([self.filename_base + '_'])            
            self.filename_base_clipboard = self.filename_base
        df.to_clipboard(index=False,header=False) # copy file name to clipboard for easy figure saving

        self.preview_maxpeak_on_params_change()
        self.search_range_link = self.multipeak_dialog.searchrange_spinbox.value()
        self.memory_link = self.multipeak_dialog.memory_spinbox.value()
        self.filter_length_link = self.multipeak_dialog.filterlen_spinbox.value()
        if self.numColors == "2" or self.numColors == "3":
            result = kymograph.link_and_plot_two_color(
                    self.all_peaks_dict["All Peaks"], self.all_smpeaks_dict["All Peaks"],
                    acqTime=self.acquisitionTime, search_range=self.search_range_link, memory=self.memory_link,
                    filter_length=self.filter_length_link, plotting=True, 
                    DNA_ends=self.dna_ends, usePrecomputed=usePrecomputed, usePrecomputedsm=usePrecomputedsm)
            self.df_peaks_linked = result['df_peaks_linked']
            self.df_peaks_linked_sm = result['df_peaks_linked_sm']
            df_gb = self.df_peaks_linked.groupby("particle")
            gb_names = list(df_gb.groups.keys())
            for i in range(len(gb_names)):
                gb_names[i] = str(gb_names[i])
            self.multipeak_dialog.leftpeak_num_combobox.clear()
            self.multipeak_dialog.leftpeak_num_combobox.addItems(gb_names)
            df_gb = self.df_peaks_linked_sm.groupby("particle")
            gb_names = list(df_gb.groups.keys())
            for i in range(len(gb_names)):
                gb_names[i] = str(gb_names[i])
            self.multipeak_dialog.rightpeak_num_combobox.clear()
            self.multipeak_dialog.rightpeak_num_combobox.addItems(gb_names)
            if self.multipeak_dialog.link_col1col2_checkbox.isChecked():
                dna_contour_len = self.multipeak_dialog.DNAcontourlength_spinbox.value()
                self.linkedpeaks_analyzed = kymograph.analyze_multipeak(self.df_peaks_linked,
                        frame_width=self.dna_ends[1] - self.dna_ends[0],
                        dna_length=self.dna_length_kb, 
                        pix_width=self.dna_puncta_size, pix_size=self.pixelSize,
                        # interpolation=interpolation_bool,
                        SxOconc=self.multipeak_dialog.SxOconc_spinbox.value())
                delta_frames = self.multipeak_dialog.max_frame_diff_spinbox.value()
                delta_pixels = self.multipeak_dialog.max_pix_diff_spinbox.value()
                delta_colocalized = self.multipeak_dialog.min_coloc_diff_spinbox.value()
                self.df_cols_linked = kymograph.link_multipeaks_2colrs(
                    self.linkedpeaks_analyzed, self.df_peaks_linked_sm,
                    delta_frames=delta_frames, delta_pixels=delta_pixels,
                    delta_colocalized=delta_colocalized)
                for i in range(len(self.df_cols_linked.index)):
                    particle_i = self.df_cols_linked.loc[i]
                    xy_col1 = (particle_i["frame_col1"], particle_i["x_col1"])
                    xy_col2 = (particle_i["frame_col2"], particle_i["x_col2"])
                    from matplotlib.patches import ConnectionPatch
                    con = ConnectionPatch(xyA=xy_col1, xyB=xy_col2, coordsA="data", coordsB="data",
                                axesA=result["ax1"], axesB=result["ax2"], color="red")
                    result["ax2"].add_artist(con)
                print(self.df_cols_linked)
            result["ax1"].axhline(self.dna_ends[0], color='g', alpha=0.5)
            result["ax1"].axhline(self.dna_ends[1], color='g', alpha=0.5)
            result["ax2"].axhline(self.dna_ends[0], color='g', alpha=0.5)
            result["ax2"].axhline(self.dna_ends[1], color='g', alpha=0.5)
            result["ax3"].axhline(self.dna_ends[0], color='g', alpha=0.5)
            result["ax3"].axhline(self.dna_ends[1], color='g', alpha=0.5)
            result["ax4"].axhline(self.dna_ends[0], color='g', alpha=0.5)
            result["ax4"].axhline(self.dna_ends[1], color='g', alpha=0.5)

            result["ax1"].set_ylim(self.dna_ends[0]-5, self.dna_ends[1]+5)
            result["ax2"].set_ylim(self.dna_ends[0]-5, self.dna_ends[1]+5)
            result["ax3"].set_ylim(self.dna_ends[0]-5, self.dna_ends[1]+5)
            result["ax4"].set_ylim(self.dna_ends[0]-5, self.dna_ends[1]+5)
        else:
            self.df_peaks_linked = kymograph.link_peaks(
                    self.all_peaks_dict["All Peaks"],
                    search_range=self.search_range_link, memory=self.memory_link,
                    filter_length=self.filter_length_link, plotting=False, 
                    DNA_ends=self.dna_ends, usePrecomputed=usePrecomputed)
            fig = plt.figure(figsize=(10, 4))
            gs = fig.add_gridspec(1, 4)
            axis = fig.add_subplot(gs[0, :-1])
            axis.set_xlabel('Frames')
            axis.set_ylabel('Pixels')
            axis_r = fig.add_subplot(gs[0, -1:], sharey=axis)
            axis_r.set_xticklabels([])

            df_gb = self.df_peaks_linked.groupby("particle")
            if len(df_gb) > 0:
                gb_names = list(df_gb.groups.keys())
                for i in range(len(gb_names)):
                    name = gb_names[i]
                    gp_sel = df_gb.get_group(name)
                    axis.plot(gp_sel["frame"], gp_sel["x"], label=str(name), alpha=0.8)
                    axis.text(gp_sel["frame"].values[0], np.average(gp_sel["x"].values[:10]), str(name))
                    gb_names[i] = str(gb_names[i])
                self.multipeak_dialog.leftpeak_num_combobox.clear()
                self.multipeak_dialog.leftpeak_num_combobox.addItems(gb_names)
            axis_r.hist(self.df_peaks_linked["PeakPosition"], orientation='horizontal')
            axis.axhline(self.dna_ends[0], 0, self.all_peaks_dict["shape_kymo"][1], color='g', alpha=0.5)
            axis.axhline(self.dna_ends[1], 0, self.all_peaks_dict["shape_kymo"][1], color='g', alpha=0.5)
            axis.set_ylim(0, self.all_peaks_dict["shape_kymo"][0])
            plt.gcf().show()

    def matplot_loop_kinetics(self, ax=None):
        left_peak_no = int(self.multipeak_dialog.leftpeak_num_combobox.currentText())
        right_peak_no = int(self.multipeak_dialog.rightpeak_num_combobox.currentText())
        # self.loop_region_left = int(self.region_errbar.getRegion()[0])
        # self.loop_region_right = int(self.region_errbar.getRegion()[1])
        df_gb = self.df_peaks_linked.groupby("particle")
        group_sel = df_gb.get_group(left_peak_no)
        group_sel = group_sel.reset_index(drop=True)
        peak_analyzed_dict = analyze_maxpeak(group_sel, smooth_length=7,
                frame_width = self.dna_ends[1] - self.dna_ends[0],
                dna_length=self.dna_length_kb, pix_width=self.dna_puncta_size,)
        if ax is None:
            _, ax = plt.subplots()
        df_peak_analyzed = peak_analyzed_dict["Max Peak"]
        n_moving = self.multipeak_dialog.moving_window_spinbox.value()
        if n_moving%2 == 0:
            n_moving = n_moving + 1
        n_order = 2
        # loop
        ax.plot(df_peak_analyzed["FrameNumber"] * self.acquisitionTime,
                df_peak_analyzed["PeakIntensity"], '.g', label=r'$I_{loop}$')
        ax.plot(df_peak_analyzed["FrameNumber"] * self.acquisitionTime,
                savgol_filter(df_peak_analyzed["PeakIntensity"].values,  window_length=n_moving, polyorder=n_order),
                'g', label=r'$I_{loop} filtered$')
        # Above loop
        ax.plot(df_peak_analyzed["FrameNumber"] * self.acquisitionTime,
                df_peak_analyzed["PeakUpIntensity"], '.r', label=r'$I_{up}$')
        ax.plot(df_peak_analyzed["FrameNumber"] * self.acquisitionTime,
                savgol_filter(df_peak_analyzed["PeakUpIntensity"].values,  window_length=n_moving, polyorder=n_order),
                'r', label=r'$I_{up} filtered$')
        # Below loop
        ax.plot(df_peak_analyzed["FrameNumber"] * self.acquisitionTime,
                df_peak_analyzed["PeakDownIntensity"], '.b', label=r'$I_{down}$')
        ax.plot(df_peak_analyzed["FrameNumber"] * self.acquisitionTime,
                savgol_filter(df_peak_analyzed["PeakDownIntensity"].values,  window_length=n_moving, polyorder=n_order),
                'b', label=r'$I_{down} filtered$')
        if self.numColors == "2" or self.numColors == "3":
            df_gb = self.df_peaks_linked_sm.groupby("particle")
            group_sel = df_gb.get_group(right_peak_no)
            group_sel = group_sel.reset_index(drop=True)
            peak_analyzed_dict_sm = analyze_maxpeak(group_sel, smooth_length=7,
                    frame_width = self.dna_ends[1] - self.dna_ends[0],
                    dna_length=self.dna_length_kb, pix_width=self.dna_puncta_size,)
            sel_loop_sm_dict = loop_sm_dist(peak_analyzed_dict, peak_analyzed_dict_sm, smooth_length=n_moving)
            ax.plot(sel_loop_sm_dict["FrameNumber"] * self.acquisitionTime,
                    sel_loop_sm_dict["PeakDiffFiltered"], 'm', label='SMol')

        if self.multipeak_dialog.force_checkbox.isChecked():
            if self.multipeak_dialog.force_combobox.currentText() == "Interpolation":
                interpolation_bool = True
            else: interpolation_bool = False
            persistence_length = self.multipeak_dialog.DNApersistencelength_spinbox.value()
            self.linkedpeaks_analyzed = kymograph.analyze_multipeak(self.df_peaks_linked,
                    frame_width=self.dna_ends[1] - self.dna_ends[0],
                    dna_length=self.dna_length_kb, 
                    pix_width=self.dna_puncta_size, pix_size=self.pixelSize,
                    interpolation=interpolation_bool, dna_persistence_length=persistence_length, 
                    SxOconc=self.multipeak_dialog.SxOconc_spinbox.value())
            df_gb = self.linkedpeaks_analyzed.groupby("particle")
            group_sel = df_gb.get_group(left_peak_no)
            group_sel = group_sel.reset_index(drop=True)
            print(group_sel)
            ax_f = ax.twinx()
            ax_f.plot(group_sel["FrameNumber"] * self.acquisitionTime,
                      savgol_filter(group_sel["Force"].values, window_length=11, polyorder=2),
                      '.', label='Force')
            ax_f.set_ylabel('Force / pN')
            ax_f.legend(loc='lower right')
        ax.set_xlabel('time/s')
        ax.set_ylabel('DNA/kb')
        ax.legend()
        plt.gcf().show()

    def fit_loop_kinetics(self):
        # from .kinetics_fitting import KineticsFitting
        fitting_dialog = KineticsFitting(self)
        fitting_dialog.show()

    def matplot_loop_vs_sm(self):
        left_peak_no = int(self.multipeak_dialog.leftpeak_num_combobox.currentText())
        right_peak_no = int(self.multipeak_dialog.rightpeak_num_combobox.currentText())
        self.loop_region_left = int(self.region_errbar.getRegion()[0])
        self.loop_region_right = int(self.region_errbar.getRegion()[1])
        _, ax = plt.subplots()
        # left peak
        df_gb = self.df_peaks_linked.groupby("particle")
        group_sel = df_gb.get_group(left_peak_no)
        group_sel = group_sel.reset_index(drop=True)
        ax.plot(group_sel["FrameNumber"], group_sel["x"], 'g', label='DNA')
        if self.numColors == "2" or self.numColors == "3":
            df_gb = self.df_peaks_linked_sm.groupby("particle")
            group_sel = df_gb.get_group(right_peak_no)
            group_sel = group_sel.reset_index(drop=True)
            ax.plot(group_sel["FrameNumber"], group_sel["x"], 'm', label='Single Molecule')
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('pixels')
        ax.legend()
        plt.gcf().show()


    def merge_tracks_fun(self, tracks, df):
        tracks = tracks.replace(' ', '')
        tracks = sorted( list( set( [int(track) for track in tracks.split(',')] ) ) )
        replot = False
        if len(tracks)>1:
            replot = True
            minTrackNo = tracks[0]
            maxTrackNo = tracks[-1]
            selected_columns = df['particle']
            particles = selected_columns.copy()
            particles = particles.values
            selected_columns = df['frame']
            frame = selected_columns.copy()
            frame = frame.values
            changedParticleID = np.zeros(particles.shape)
            for track in tracks:
                changedParticleID[particles==track] = 1
                particles[particles==track] = minTrackNo                
            
            remove = []
            u, c = np.unique(frame[changedParticleID.astype(bool)], return_counts=True)
            if any(c>1):
                duplicateIndices = np.where(c>1)
                for duplicateIndex in duplicateIndices[0]:
                    # see where frame assumes the duplicate value
                    remove_tmp = np.where( np.logical_and(frame==u[duplicateIndex], particles==minTrackNo) )
                    remove.append( remove_tmp[0][0:-1] )
            remove = np.array(remove).flatten()
            particles[particles!=minTrackNo] -= maxTrackNo-1
            df['particle'] = particles
            df = df.drop(remove) # drop these rows
        return df, replot


    def merge_tracks(self):
        replot = False
        left_tracks = self.multipeak_dialog.merge_left_lineedit.text()
        if len(left_tracks)>0:
            self.df_peaks_linked, replot = self.merge_tracks_fun(left_tracks, self.df_peaks_linked)

        if hasattr(self, 'df_peaks_linked_sm'):
            right_tracks = self.multipeak_dialog.merge_right_lineedit.text()
            if len(right_tracks)>0:
                self.df_peaks_linked_sm, replot = self.merge_tracks_fun(right_tracks, self.df_peaks_linked_sm)
               
        if replot:
            if hasattr(self, 'df_peaks_linked_sm'):
                self.matplot_all_peaks(usePrecomputed=self.df_peaks_linked, usePrecomputedsm=self.df_peaks_linked_sm)
            else:
                self.matplot_all_peaks(usePrecomputed=self.df_peaks_linked)



    def plottype_multipeak(self):
        plt.rcParams["savefig.directory"] = self.filepath # default saving dir is the path of the current file
        if len(self.filename_base)==0 and hasattr(self, 'filename_base_clipboard'):
            df=pd.DataFrame([self.filename_base_clipboard + '_' + self.multipeak_dialog.plottype_combobox.currentText()])            
        else:
            df=pd.DataFrame([self.filename_base + '_' + self.multipeak_dialog.plottype_combobox.currentText()])            
            self.filename_base_clipboard = self.filename_base
        df.to_clipboard(index=False,header=False) # copy file name to clipboard for easy figure saving

        left_peak_no = int(self.multipeak_dialog.leftpeak_num_combobox.currentText())
        right_peak_no = int(self.multipeak_dialog.rightpeak_num_combobox.currentText())
        if not ("AvTimeTraceCol" in self.multipeak_dialog.plottype_combobox.currentText()):
            if self.df_peaks_linked is None:
                print('No tracked particles yet.')
                return
            df_gb = self.df_peaks_linked.groupby("particle")
            group_sel_col1 = df_gb.get_group(left_peak_no)
            group_sel_col1 = group_sel_col1.reset_index(drop=True)
            if self.numColors == "2" or self.numColors == "3":
                df_gb = self.df_peaks_linked_sm.groupby("particle")
                group_sel_col2 = df_gb.get_group(right_peak_no)
                group_sel_col2 = group_sel_col2.reset_index(drop=True)
        if self.multipeak_dialog.plottype_combobox.currentText() == "MSDmoving":
            print("plot MSD")
            _, ax = plt.subplots()
            n=self.multipeak_dialog.moving_window_spinbox.value()
            n_savgol = n
            if n%2 != 0:
                n = n+1
            else:
                n_savgol += 1
            ind = int(n/2)
            msd_moving = kymograph.msd_moving(group_sel_col1['x'].values, n=n)
            frames = group_sel_col1['FrameNumber'].values[ind:-ind]
            peak_analyzed_dict = kymograph.analyze_maxpeak(group_sel_col1, smooth_length=7,
                    frame_width = self.dna_ends[1] - self.dna_ends[0],
                    dna_length=self.dna_length_kb, pix_width=self.dna_puncta_size,)
            # ax.plot(frames, msd_moving, 'g', label='color_1')
            if self.numColors == "2" or self.numColors == "3":
                msd_moving = kymograph.msd_moving(group_sel_col2['x'].values, n=n)
                frames = group_sel_col2['FrameNumber'].values[ind:-ind]
                ax.plot(frames* self.acquisitionTime, msd_moving, 'm', label='MSD particle')
                peak_analyzed_dict_sm = kymograph.analyze_maxpeak(group_sel_col2, smooth_length=7,
                    frame_width = self.dna_ends[1] - self.dna_ends[0],
                    dna_length=self.dna_length_kb, pix_width=self.dna_puncta_size,)
                sel_loop_sm_dict = kymograph.loop_sm_dist(peak_analyzed_dict, peak_analyzed_dict_sm, smooth_length=7)
                pos_diff_kb = sel_loop_sm_dict['PositionDiff_kb']
                if n_savgol>=len(pos_diff_kb):
                    n_savgol = len(pos_diff_kb)-2
                if n_savgol%2 == 0:
                    n_savgol += 1
                pos_diff_kb_smooth = savgol_filter(pos_diff_kb, window_length=n_savgol, polyorder=1)
                if pos_diff_kb_smooth.min() < 0:
                    pos_diff_kb = pos_diff_kb - pos_diff_kb_smooth.min()
                    pos_diff_kb_smooth = pos_diff_kb_smooth - pos_diff_kb_smooth.min()
                ax_right = ax.twinx()
                ax_right.plot(sel_loop_sm_dict['FrameNumber'] * self.acquisitionTime,
                              pos_diff_kb,
                              '.r', label='Position Differnece')                
                ax_right.plot(sel_loop_sm_dict['FrameNumber'] * self.acquisitionTime,
                              pos_diff_kb_smooth,
                              'r', label='Position Differnece')
                ax_right.set_ylabel("Kilobases", color="r")
                ax_right.tick_params(axis='y', colors='r')
                ax_right.spines["right"].set_color("r")
                ax_right.legend(loc='center right')

            ax.set_xlabel("time/s", color='m')
            ax.tick_params(axis='y', colors='m')
            ax.spines["left"].set_color("m")
            ax.set_ylabel("Moving MSD(" + str(n) + " points)")
            ax.legend()
            plt.gcf().show()
        elif self.multipeak_dialog.plottype_combobox.currentText() == "MSDsavgol":
            print("plot MSD savgol")
            _, ax = plt.subplots(3,1, figsize=(6,8), sharex=True)
            axLoopSize = ax[1]
            axLoopSizeMSD = ax[2]
            ax = ax[0]
            n_savgol = self.multipeak_dialog.moving_window_spinbox.value()
            n = n_savgol
            if n_savgol%2 == 0:
                n_savgol = n_savgol + 1
            else:
                n = n+1
            n_order = 1
            n_savgol = 11
            ind = int(n/2)
            msd_moving = kymograph.msd_moving(group_sel_col1['x'].values, n=n)
            frames = group_sel_col1['FrameNumber'].values[ind:-ind]
            peak_analyzed_dict = kymograph.analyze_maxpeak(group_sel_col1, smooth_length=7,
                    frame_width = self.dna_ends[1] - self.dna_ends[0],
                    dna_length=self.dna_length_kb, pix_width=self.dna_puncta_size,)
            # ax.plot(frames, savgol_filter(msd_moving, window_length=n_savgol, polyorder=n_order), 'g', label='color_1')
            if self.numColors == "2" or self.numColors == "3":
                msd_moving = kymograph.msd_moving(group_sel_col2['x'].values, n=n)
                frames = group_sel_col2['FrameNumber'].values[ind:-ind]
                ax.plot(frames * self.acquisitionTime,
                        msd_moving * self.pixelSize**2, #converted to µm²
                        '.', color='darkslategrey', label='MSD particle')
                ax.plot(frames * self.acquisitionTime,
                        savgol_filter(msd_moving, window_length=n_savgol, polyorder=n_order) * self.pixelSize**2, #converted to µm²
                        color='darkslategrey', label='')
                peak_analyzed_dict_sm = kymograph.analyze_maxpeak(group_sel_col2, smooth_length=7,
                    frame_width = self.dna_ends[1] - self.dna_ends[0],
                    dna_length=self.dna_length_kb, pix_width=self.dna_puncta_size,)
                sel_loop_sm_dict = kymograph.loop_sm_dist(peak_analyzed_dict, peak_analyzed_dict_sm, smooth_length=7)
                pos_diff_kb = sel_loop_sm_dict['PositionDiff_kb']
                pos_diff_kb_smooth = savgol_filter(pos_diff_kb, window_length=n_savgol, polyorder=1)
                if pos_diff_kb_smooth.min() < 0:
                    pos_diff_kb = pos_diff_kb - pos_diff_kb_smooth.min()
                    pos_diff_kb_smooth = pos_diff_kb_smooth - pos_diff_kb_smooth.min()
                ax_right = ax.twinx()
                ax_right.plot(sel_loop_sm_dict['FrameNumber'] * self.acquisitionTime,
                              pos_diff_kb,
                              '.', color='darkorange', label='Distance')
                ax_right.plot(sel_loop_sm_dict['FrameNumber'] * self.acquisitionTime,
                              pos_diff_kb_smooth,
                              color='darkorange', label='')
                ax_right.set_ylabel("Distance/kb", color='darkorange')
                ax_right.tick_params(axis='y', colors='darkorange')
                ax_right.spines["right"].set_color('darkorange')
                ax_right.legend(loc='center right', labelcolor='linecolor')

                # plot loop Size vs MSD
                MSD = msd_moving * self.pixelSize**2
                left_peak_no = int(self.multipeak_dialog.leftpeak_num_combobox.currentText())
            
                df_gb = self.df_peaks_linked.groupby("particle")
                group_sel = df_gb.get_group(left_peak_no)
                group_sel = group_sel.reset_index(drop=True)
                peak_analyzed_dict = analyze_maxpeak(group_sel, smooth_length=7,
                        frame_width = self.dna_ends[1] - self.dna_ends[0],
                        dna_length=self.dna_length_kb, pix_width=self.dna_puncta_size,)
                df_peak_analyzed = peak_analyzed_dict["Max Peak"]
                n_moving = self.multipeak_dialog.moving_window_spinbox.value()
                if n_moving%2 == 0:
                    n_moving = n_moving + 1
                n_order = 2
                # loop
                loopSize = df_peak_analyzed["PeakIntensity"]
                # loopSizeSavGol = savgol_filter(df_peak_analyzed["PeakIntensity"].values,  window_length=n_moving, polyorder=n_order)
                framesLoopSize = df_peak_analyzed["FrameNumber"] * self.acquisitionTime
                framesMSD = frames * self.acquisitionTime
                minFrame = np.max([np.min(framesLoopSize), np.min(framesMSD)])+self.acquisitionTime
                maxFrame = np.min([np.max(framesLoopSize), np.max(framesMSD)])-self.acquisitionTime
                framesInt = np.arange(minFrame, maxFrame, self.acquisitionTime)
                loopSizeInterpolator = interp1d(framesLoopSize, loopSize)
                MSDInterpolator = interp1d(framesMSD, MSD)
                loopSizeInt = loopSizeInterpolator(framesInt)
                MSDInt = MSDInterpolator(framesInt)
                loopSizeIntSavgol = savgol_filter(loopSizeInt,  window_length=n_moving, polyorder=n_order)
                MSDIntSavgol = savgol_filter(MSDInt,  window_length=n_moving, polyorder=n_order)
                axLoopSize.plot(framesLoopSize, loopSize, '.g', alpha=0.2)
                axLoopSize.plot(framesInt, loopSizeIntSavgol, '-g')
                # axLoopSizeMSD.plot(MSDInt, loopSizeInt, '.')
                axLoopSizeMSD.plot(MSDIntSavgol[0], loopSizeIntSavgol[0], '.')
                jetColormap = plt.get_cmap('jet')
                kymograph.colorline(MSDIntSavgol, loopSizeIntSavgol, z=None, cmap=jetColormap, linewidth=2, ax=axLoopSizeMSD)

                from scipy.stats import spearmanr
                corr, pval = spearmanr(MSDIntSavgol, loopSizeIntSavgol)
                string = r"$\rho_{Spear}$ = "+"{:.2f}".format(corr)+'\n'+r"$p=$"+"{:.2e}".format(pval)
                axLoopSizeMSD.annotate(string, (np.max(MSDIntSavgol), np.max(loopSizeIntSavgol)), 
                    horizontalalignment='right', verticalalignment='top')
                axLoopSizeMSD.set_xlim(left=np.min(MSDIntSavgol)*0.95, right=np.max(MSDIntSavgol)*1.05)
                axLoopSizeMSD.set_ylim(bottom=np.min(loopSizeIntSavgol)*0.95, top=np.max(loopSizeIntSavgol)*1.05)

            ax.set_xlabel("time/s")
            axLoopSize.set_xlabel("time/s")
            axLoopSizeMSD.set_xlabel(r"MSD(${\mu} m^2$)")

            ax.tick_params(axis='y', colors='darkslategrey')
            ax.spines["left"].set_color("darkslategrey")

            ax.set_ylabel(r"MSD(${\mu} m^2$)", color='darkslategrey')
            axLoopSize.set_ylabel(r"Loop size [kbp]")
            axLoopSizeMSD.set_ylabel(r"Loop size [kbp]")
            
            ax.legend(labelcolor='linecolor')
            plt.tight_layout()
            plt.gcf().show()
        elif self.multipeak_dialog.plottype_combobox.currentText() == "MSDlagtime":
            print("plot MSD")
            _, ax = plt.subplots()
            msd = kymograph.msd_1d_nb1(group_sel_col1['x'].values)
            plt.plot(msd, 'g', label="MSD color-1")
            if self.numColors == "2" or self.numColors == "3":
                msd = kymograph.msd_1d_nb1(group_sel_col2['x'].values)
                plt.plot(msd, 'm', label="MSD color-1")
            ax.set_xlabel("Frame Number")
            ax.set_ylabel("MSD")
            ax.set_yscale('log')
            ax.legend()
            plt.gcf().show()
        elif self.multipeak_dialog.plottype_combobox.currentText() == "MSDlagtime-AllPeaks":
            if self.numColors == "1":
                fig,(ax1) = plt.subplots(nrows=1, ncols=1)
                _ = kymograph.msd_lagtime_allpeaks(self.df_peaks_linked,
                                pixelsize = self.pixelSize,
                                fps=int(self.numColors) * (1/self.acquisitionTime),
                                max_lagtime=100, axis=ax1)
                ax1.set_title("Color 1")
            elif self.numColors == "2" or self.numColors == "3":
                fig,(ax1, ax2) = plt.subplots(nrows=1, ncols=2)
                _ = kymograph.msd_lagtime_allpeaks(self.df_peaks_linked,
                                pixelsize = self.pixelSize,
                                fps=int(self.numColors) * (1/self.acquisitionTime),
                                max_lagtime=100, axis=ax1)
                _ = kymograph.msd_lagtime_allpeaks(self.df_peaks_linked_sm,
                                pixelsize = self.pixelSize,
                                fps=int(self.numColors) * (1/self.acquisitionTime),
                                max_lagtime=100, axis=ax2)
                ax1.set_title("Color 1")
                ax2.set_title("Color 2")
            plt.gcf().show()
        elif self.multipeak_dialog.plottype_combobox.currentText() == "LoopSizeVsPosition" or self.multipeak_dialog.plottype_combobox.currentText() == "LoopSizeVsPositionDetailed":
            print("plot LoopSizeVsPosition")
            _, ax = plt.subplots()
            x = group_sel_col1['x'].values
            frames = group_sel_col1["FrameNumber"].values.astype(int)
            peakIndex = np.vstack((x-3, x-2, x-1, x, x+1, x+2, x+3))
            peakIndex[peakIndex>self.kymo_left_loop.shape[1]] = self.kymo_left_loop.shape[1]
            peakIndex[peakIndex<0] = 0
            peakIndexArray = np.zeros(self.kymo_left_loop.shape)
            peakIndexArray[frames, peakIndex.astype(int)] = 1
            peakIndexArray[peakIndexArray==0] = np.nan
            trace_col1 = np.nanmean( self.kymo_left_loop*peakIndexArray, axis=1)            
            trace_col1_bg = np.nanmean( self.kymo_left_loop*~np.isnan(peakIndexArray), axis=1)
            trace_col1_bg = trace_col1_bg[~np.isnan(trace_col1)]
            trace_col1 = trace_col1[~np.isnan(trace_col1)]
            
            loopSize = trace_col1#/(trace_col1+trace_col1_bg) * self.multipeak_dialog.DNAlength_spinbox.value() #trace_col1 - trace_col1_bg
            loopPosition = (x-self.dna_ends[0])/self.dna_ends[1] * self.multipeak_dialog.DNAlength_spinbox.value()#x


            # savgol filtering
            n_savgol = self.multipeak_dialog.moving_window_spinbox.value()
            if n_savgol>=len(loopSize):
                n_savgol = len(loopSize)-2
            if n_savgol%2 == 0:
                n_savgol += 1  
            n_savgol_lo = np.floor( n_savgol/5 )
            if n_savgol_lo>=len(loopSize):
                n_savgol_lo = len(loopSize)-2
            if n_savgol_lo%2 == 0:
                n_savgol_lo += 1 
            n_savgol_hi = np.ceil( n_savgol*2.5 )
            if n_savgol_hi>=len(loopSize):
                n_savgol_hi = len(loopSize)-2
            if n_savgol_hi%2 == 0:
                n_savgol_hi += 1 
            n_savgol    = int(n_savgol)
            n_savgol_lo = int(n_savgol_lo)
            n_savgol_hi = int(n_savgol_hi)

            fig = plt.figure(figsize=(10, 6))
            jetColormap = plt.get_cmap('jet')
            grid = plt.GridSpec(1,3, wspace=0.0, hspace=0.0)
            ax = fig.add_subplot(grid[0,0])            
            
            ax.scatter(loopSize, loopPosition, s=2, c=group_sel_col1["FrameNumber"].values, cmap=jetColormap)
            loopPosition_savgol = savgol_filter(loopPosition, window_length=n_savgol_lo, polyorder=1)
            loopSize_savgol = savgol_filter(loopSize, window_length=n_savgol_lo, polyorder=1)
            kymograph.colorline(loopSize_savgol, loopPosition_savgol, z=None, cmap=jetColormap, linewidth=2, ax=ax)
            ax.set_ylabel('Position [kb]')
            ax.set_xlim([min(loopSize), max(loopSize)])
            ax.set_ylim([min(loopPosition), max(loopPosition)])
            x_text = min(loopSize) + (max(loopSize)-min(loopSize))/2
            y_text = max(loopPosition) + (max(loopPosition)-min(loopPosition))*0.05
            s = 'window length '+str(n_savgol_lo)+' frames'
            ax.text(x_text, y_text, s, ha='center')
            
            ax = fig.add_subplot(grid[0,1], yticklabels=[])            
            ax.scatter(loopSize, loopPosition, s=2, c=group_sel_col1["FrameNumber"].values, cmap=jetColormap)
            loopPosition_savgol = savgol_filter(loopPosition, window_length=n_savgol, polyorder=1)
            loopSize_savgol = savgol_filter(loopSize, window_length=n_savgol, polyorder=1)
            kymograph.colorline(loopSize_savgol, loopPosition_savgol, z=None, cmap=jetColormap, linewidth=2, ax=ax)
            ax.set_xlabel('Loop size [a.u.]')
            ax.set_xlim([min(loopSize), max(loopSize)])
            ax.set_ylim([min(loopPosition), max(loopPosition)])
            s = 'window length '+str(n_savgol)+' frames'
            ax.text(x_text, y_text, s, ha='center')
                      
            ax = fig.add_subplot(grid[0,2], yticklabels=[])            
            ax.scatter(loopSize, loopPosition, s=2, c=group_sel_col1["FrameNumber"].values, cmap=jetColormap)
            loopPosition_savgol = savgol_filter(loopPosition, window_length=n_savgol_hi, polyorder=1)
            loopSize_savgol = savgol_filter(loopSize, window_length=n_savgol_hi, polyorder=1)
            kymograph.colorline(loopSize_savgol, loopPosition_savgol, z=None, cmap=jetColormap, linewidth=2, ax=ax)
            ax.set_xlim([min(loopSize), max(loopSize)])
            ax.set_ylim([min(loopPosition), max(loopPosition)])
            s = 'window length '+str(n_savgol_hi)+' frames'
            ax.text(x_text, y_text, s, ha='center')

            plt.gcf().show()

            if self.multipeak_dialog.plottype_combobox.currentText() == "LoopSizeVsPositionDetailed":
                # 3-point interpolation
                x_3pt = np.zeros(x.shape)
                y_3pt = np.zeros(x.shape)
                x_3pt_smooth = np.zeros(x.shape)
                y_3pt_smooth = np.zeros(x.shape)
                smooth_length = self.multipeak_dialog.smoothlength_spinbox.value()
                for ind in range(len(x)):
                    peakIndex_x = peakIndex[1:-1,ind].astype(int)
                    peakIndex_x = peakIndex[:,ind].astype(int)
                    y_3pt[ind], x_3pt[ind] = kymograph.qinterp_max(self.kymo_left_loop[frames[ind], peakIndex_x], x=peakIndex_x)
                    
                    line1d = self.kymo_left_loop[frames[ind], :]
                    if smooth_length > 2:
                        line1d_smth = savgol_filter(line1d, window_length=smooth_length, polyorder=1)
                    else:
                        line1d_smth = line1d

                    y_3pt_smooth[ind], x_3pt_smooth[ind] = kymograph.qinterp_max(line1d_smth[peakIndex_x], x=peakIndex_x)



                from scipy.signal import butter,filtfilt
                fs = 5       # sample rate, Hz
                cutoff = 0.5      # desired cutoff frequency of the filter, Hz
                nyq = 0.5 * fs  # Nyquist Frequency
                order = 4       # sin wave can be approx represented as quadratic
                def butter_lowpass_filter(data, cutoff, fs, order):
                    normal_cutoff = cutoff / nyq
                    # Get the filter coefficients 
                    b, a = butter(order, normal_cutoff, btype='low', analog=False)
                    y = filtfilt(b, a, data)
                    return y

                # plot original smoothing of integer pixel values
                loopPosition_savgol = savgol_filter(loopPosition, window_length=n_savgol, polyorder=1)
                loopSize_savgol = savgol_filter(loopSize, window_length=n_savgol, polyorder=1)
                fig = plt.figure(figsize=(10, 6))
                grid = plt.GridSpec(2, 4, wspace=0.0, hspace=0.0)
                ax = fig.add_subplot(grid[0,0:1], xticklabels=[])
                ax.plot(frames, loopPosition)
                ax.plot(frames, loopPosition_savgol)
                ax.set_ylabel('Loop position [kb]')
                ax = fig.add_subplot(grid[1,0:1])
                ax.plot(frames, loopSize)
                ax.plot(frames, loopSize_savgol)
                ax.set_xlabel('Frames')
                ax.set_ylabel('Loop size [kb]')
                ax = fig.add_subplot(grid[:,2:])
                ax.scatter(loopSize, loopPosition, s=2, c=group_sel_col1["FrameNumber"].values, cmap=jetColormap)
                kymograph.colorline(loopSize_savgol, loopPosition_savgol, z=None, cmap=jetColormap, linewidth=2, ax=ax)
                ax.set_ylabel('Loop position [kb]')
                ax.set_xlabel('Loop size [kb]')
                ax.set_title('Savgol smoothing of integer pixel values')
                plt.gcf().show()

                # plot subpixel localisation and savgol
                loopPosition_subpix_savgol = savgol_filter(x_3pt_smooth, window_length=n_savgol, polyorder=1)
                loopSize_subpix_savgol = savgol_filter(y_3pt_smooth, window_length=n_savgol, polyorder=1)
                fig = plt.figure(figsize=(10, 6))
                grid = plt.GridSpec(2, 4, wspace=0.0, hspace=0.0)
                ax = fig.add_subplot(grid[0,0:1], xticklabels=[])
                ax.plot(frames, x_3pt_smooth)
                ax.plot(frames, loopPosition_subpix_savgol)
                ax.set_ylabel('Loop position [kb]')
                ax = fig.add_subplot(grid[1,0:1])
                ax.plot(frames, y_3pt_smooth)
                ax.plot(frames, loopSize_subpix_savgol)
                ax.set_xlabel('Frames')
                ax.set_ylabel('Loop size [kb]')
                ax = fig.add_subplot(grid[:,2:])
                ax.scatter(y_3pt_smooth, x_3pt_smooth, s=2, c=group_sel_col1["FrameNumber"].values, cmap=jetColormap)
                kymograph.colorline(loopSize_subpix_savgol, loopPosition_subpix_savgol, z=None, cmap=jetColormap, linewidth=2, ax=ax)
                ax.set_ylabel('Loop position [kb]')
                ax.set_xlabel('Loop size [kb]')
                ax.set_title('Savgol smoothing of subpixel values')
                plt.gcf().show()
                
                # plot subpixel localisation and butterworth
                cutoff = 0.5      # desired cutoff frequency of the filter, Hz
                loopPosition_subpix_butterworth = butter_lowpass_filter(x_3pt_smooth, cutoff, fs, order)
                loopSize_subpix_butterworth = butter_lowpass_filter(y_3pt_smooth, cutoff, fs, order)
                fig = plt.figure(figsize=(10, 6))
                grid = plt.GridSpec(2, 4, wspace=0.0, hspace=0.0)
                ax = fig.add_subplot(grid[0,0:1], xticklabels=[])
                ax.plot(frames, x_3pt_smooth)
                ax.plot(frames, loopPosition_subpix_butterworth)
                ax.set_ylabel('Loop position [kb]')
                ax = fig.add_subplot(grid[1,0:1])
                ax.plot(frames, y_3pt_smooth)
                ax.plot(frames, loopSize_subpix_butterworth)
                ax.set_xlabel('Frames')
                ax.set_ylabel('Loop size [kb]')
                ax = fig.add_subplot(grid[:,2:])
                ax.scatter(y_3pt_smooth, x_3pt_smooth, s=2, c=group_sel_col1["FrameNumber"].values, cmap=jetColormap)
                kymograph.colorline(loopSize_subpix_butterworth, loopPosition_subpix_butterworth, z=None, cmap=jetColormap, linewidth=2, ax=ax)
                ax.set_ylabel('Loop position [kb]')
                ax.set_xlabel('Loop size [kb]')
                ax.set_title('0.5 Hz cutoff smoothing of subpixel values')
                plt.gcf().show()

                # plot subpixel localisation and butterworth
                cutoff = 0.25      # desired cutoff frequency of the filter, Hz
                loopPosition_subpix_butterworth = butter_lowpass_filter(x_3pt_smooth, cutoff, fs, order)
                loopSize_subpix_butterworth = butter_lowpass_filter(y_3pt_smooth, cutoff, fs, order)
                fig = plt.figure(figsize=(10, 6))
                grid = plt.GridSpec(2, 4, wspace=0.0, hspace=0.0)
                ax = fig.add_subplot(grid[0,0:1], xticklabels=[])
                ax.plot(frames, x_3pt_smooth)
                ax.plot(frames, loopPosition_subpix_butterworth)
                ax.set_ylabel('Loop position [kb]')
                ax = fig.add_subplot(grid[1,0:1])
                ax.plot(frames, y_3pt_smooth)
                ax.plot(frames, loopSize_subpix_butterworth)
                ax.set_xlabel('Frames')
                ax.set_ylabel('Loop size [kb]')
                ax = fig.add_subplot(grid[:,2:])
                ax.scatter(y_3pt_smooth, x_3pt_smooth, s=2, c=group_sel_col1["FrameNumber"].values, cmap=jetColormap)
                kymograph.colorline(loopSize_subpix_butterworth, loopPosition_subpix_butterworth, z=None, cmap=jetColormap, linewidth=2, ax=ax)
                ax.set_ylabel('Loop position [kb]')
                ax.set_xlabel('Loop size [kb]')
                ax.set_title('0.25 Hz cutoff smoothing of subpixel values')
                plt.gcf().show()


        elif self.multipeak_dialog.plottype_combobox.currentText() == "TimeTraceCol1":
            print("plot TimeTrace")
            _, ax = plt.subplots()
            x = group_sel_col1['x'].values
            peakIndex = np.vstack((x-2, x-1, x, x+1, x+2))
            peakIndex[peakIndex>self.kymo_left_loop.shape[1]] = self.kymo_left_loop.shape[1]
            peakIndex[peakIndex<0] = 0
            peakIndexArray = np.zeros(self.kymo_left_loop.shape)
            peakIndexArray[group_sel_col1["FrameNumber"].values.astype(int), peakIndex.astype(int)] = 1
            peakIndexArray[peakIndexArray==0] = np.nan
            trace_col1 = np.nanmean( self.kymo_left_loop*peakIndexArray, axis=1)            
            trace_col1_bg = np.nanmean( self.kymo_left_loop*~np.isnan(peakIndexArray), axis=1)
            trace_col1_bg = trace_col1_bg[~np.isnan(trace_col1)]
            trace_col1 = trace_col1[~np.isnan(trace_col1)]
            ax.plot(group_sel_col1["FrameNumber"], trace_col1, '-', color=(0.8, 0.8, 0.8), label="Peak")
            ax.plot(group_sel_col1["FrameNumber"], trace_col1_bg, '--', color=(0.8, 0.8, 0.8), label="Background")
            ax.plot(group_sel_col1["FrameNumber"], trace_col1-trace_col1_bg, 'k-', label="Subtracted")

            # sIntensity = int( np.ceil( np.max(trace_col1-trace_col1_bg)*0.35 ) )
            # sSpatial = int( len(trace_col1)/5 )
            # if sSpatial==0: sSpatial = int( len(trace_col1)/2 )
            # subtracted_smooth = kymograph.bilateralFtr1D(trace_col1-trace_col1_bg, sSpatial = sSpatial, sIntensity = sIntensity) # +/- pixels further than 3*sSpatial pixels will have approx 0 effect
            # ax.plot(group_sel_col1["FrameNumber"], subtracted_smooth, 'r-', label="Subtracted smooth")
            # # attempt to find steps. ref https://github.com/thomasbkahn/step-detect
            # step_detect_data = subtracted_smooth#trace_col1-trace_col1_bg
            # p2  = step_detect.mz_fwt(step_detect_data, n=2)
            # p2 /= np.abs(p2).max()
            # try:
            #     stepPositions = step_detect.find_steps(np.abs(p2), 0.5)
            #     minVal = min(step_detect_data)
            #     maxVal = max(step_detect_data)
            #     minFrame = np.min( group_sel_col1["FrameNumber"].values )
            #     for ii in range(len(stepPositions)):
            #         plt.plot((stepPositions[ii]+minFrame, stepPositions[ii]+minFrame), (minVal, maxVal), 'r')
            # except:
            #     pass

            ax.set_xlabel("Frame Number")
            ax.set_ylabel("Intensity")
            ax.legend()
            plt.gcf().show()
        elif self.multipeak_dialog.plottype_combobox.currentText() == "TimeTraceCol2" and self.numColors == "2":
            print("plot TimeTrace")
            _, ax = plt.subplots()
            x = group_sel_col2['x'].values
            peakIndex = np.vstack((x-2, x-1, x, x+1, x+2))
            peakIndex[peakIndex>self.kymo_right_loop.shape[1]] = self.kymo_right_loop.shape[1]
            peakIndex[peakIndex<0] = 0
            peakIndexArray = np.zeros(self.kymo_right_loop.shape)
            peakIndexArray[group_sel_col2["FrameNumber"].values.astype(int), peakIndex.astype(int)] = 1
            peakIndexArray[peakIndexArray==0] = np.nan
            trace_col2 = np.nanmean( self.kymo_right_loop*peakIndexArray, axis=1)            
            trace_col2_bg = np.nanmean( self.kymo_right_loop*~np.isnan(peakIndexArray), axis=1)
            trace_col2_bg = trace_col2_bg[~np.isnan(trace_col2)]
            trace_col2 = trace_col2[~np.isnan(trace_col2)]
            ax.plot(group_sel_col2["FrameNumber"], trace_col2, '-', color=(0.8, 0.8, 0.8), label="Peak")
            ax.plot(group_sel_col2["FrameNumber"], trace_col2_bg, '--', color=(0.8, 0.8, 0.8), label="Background")
            ax.plot(group_sel_col2["FrameNumber"], trace_col2-trace_col2_bg, 'k-', label="Subtracted")

            # sIntensity = int( np.ceil( np.max(trace_col2-trace_col2_bg)*0.35 ) )
            # sSpatial = int( len(trace_col2)/5 )
            # if sSpatial==0: sSpatial = int( len(trace_col2)/2 )
            # subtracted_smooth = kymograph.bilateralFtr1D(trace_col2-trace_col2_bg, sSpatial = sSpatial, sIntensity = sIntensity) # +/- pixels further than 3*sSpatial pixels will have approx 0 effect
            # ax.plot(group_sel_col2["FrameNumber"], subtracted_smooth, 'r-', label="Subtracted smooth")
            # # attempt to find steps. ref https://github.com/thomasbkahn/step-detect
            # step_detect_data = subtracted_smooth#trace_col2-trace_col2_bg
            # p2  = step_detect.mz_fwt(step_detect_data, n=2)
            # p2 /= np.abs(p2).max()
            # try:
            #     stepPositions = step_detect.find_steps(np.abs(p2), 0.5)
            #     minVal = min(step_detect_data)
            #     maxVal = max(step_detect_data)
            #     minFrame = np.min( group_sel_col2["FrameNumber"].values )
            #     for ii in range(len(stepPositions)):
            #         plt.plot((stepPositions[ii]+minFrame, stepPositions[ii]+minFrame), (minVal, maxVal), 'r')
            # except:
            #     pass
            ax.set_xlabel("Frame Number")
            ax.set_ylabel("Intensity")
            ax.legend()
            plt.gcf().show()

        elif self.multipeak_dialog.plottype_combobox.currentText() == "AvTimeTraceCol1":
            print("plot average TimeTrace")
            _, ax = plt.subplots()
            trace_col1 = np.mean( self.kymo_left_loop, axis=1) * 1000
            FrameNumber = np.arange(1, len(trace_col1)+1)      
            ax.plot(FrameNumber, trace_col1, 'k-', label="Intensity")

            # sIntensity = int( np.ceil( np.max(trace_col1)*0.35 ) )
            # sSpatial = int( len(trace_col1)/5 )
            # if sSpatial==0: sSpatial = int( len(trace_col1)/2 )
            # subtracted_smooth = kymograph.bilateralFtr1D(trace_col1, sSpatial = sSpatial, sIntensity = sIntensity) # +/- pixels further than 3*sSpatial pixels will have approx 0 effect
            # ax.plot(FrameNumber, subtracted_smooth, 'r-', label="Intensity smooth")

            ax.set_xlabel("Frame Number")
            ax.set_ylabel("Intensity")
            ax.legend()
            plt.gcf().show()

        elif self.multipeak_dialog.plottype_combobox.currentText() == "AvTimeTraceCol2" and self.numColors == "2":
            print("plot average TimeTrace")
            _, ax = plt.subplots()
            trace_col2 = np.mean( self.kymo_right_loop, axis=1) * 1000  
            FrameNumber = np.arange(1, len(trace_col2)+1)      
            ax.plot(FrameNumber, trace_col2, 'k-', label="Intensity")

            # sIntensity = int( np.ceil( np.max(trace_col2)*0.5 ) )
            # sIntensity = int( np.ceil( np.max(trace_col2)*0.35 ) )
            # sSpatial = int( len(trace_col2)/5 )
            # if sSpatial==0: sSpatial = int( len(trace_col2)/2 )
            # subtracted_smooth = kymograph.bilateralFtr1D(trace_col2, sSpatial = sSpatial, sIntensity = sIntensity) # +/- pixels further than 3*sSpatial pixels will have approx 0 effect

            # ax.plot(FrameNumber, subtracted_smooth, 'r-', label="Intensity smooth")
            # # attempt to find steps. ref https://github.com/thomasbkahn/step-detect
            # step_detect_data = subtracted_smooth#trace_col2
            # p2  = step_detect.mz_fwt(step_detect_data, n=2)
            # p2 /= np.abs(p2).max()
            # try:
            #     stepPositions = step_detect.find_steps(np.abs(p2), 0.5)
            #     minVal = min(step_detect_data)
            #     maxVal = max(step_detect_data)
            #     for ii in range(len(stepPositions)):
            #         plt.plot((stepPositions[ii], stepPositions[ii]), (minVal, maxVal), 'r')
            # except:
            #     pass
            ax.set_xlabel("Frame Number")
            ax.set_ylabel("Intensity")
            ax.legend()
            plt.gcf().show()

    def save_hdf5(self, filepath_hdf5):
        # Update self.ui.manualproperties
        for prop in self.ui.manualproperties:
            if prop['Type'] == 'checkbox':
                prop['Value'] = prop['Widget'].isChecked()
            elif prop['Type'] == 'text':
                prop['Value'] = prop['Textbox'].text()
            else:
                raise(Exception('Property type not recognized'))

        with h5py.File(filepath_hdf5, 'w') as h5_analysis:
            # save parameters
            params_group = h5_analysis.create_group("parameters")
            hdf5dict.dump(self.params_yaml, params_group)
            h5_analysis["filepath"] = self.filepath
            h5_analysis["DNA ends"] = self.dna_ends
            # if self.kymo_left is not None:
            h5_analysis["Left Image Array"] = self.roirect_left.getArrayRegion(
                                self.imgarr_left,
                                self.imv00.imageItem, axes=(1, 2))
            h5_analysis["Left Kymograph"] = self.kymo_left.T
            h5_analysis["Left Kymograph Loop"] = self.kymo_left_loop.T
            h5_analysis["Left Kymograph No Loop"] = self.kymo_left_noLoop.T
            if self.numColors == "2" or self.numColors == "3":
                h5_analysis["Right Image Array"] = self.roirect_right.getArrayRegion(
                                self.imgarr_right,
                                self.imv01.imageItem, axes=(1, 2))
                h5_analysis["Right Kymograph"] = self.kymo_right.T
                h5_analysis["Right Kymograph Loop"] = self.kymo_right_loop.T
                h5_analysis["Right Kymograph No Loop"] = self.kymo_right_noLoop.T
                if self.numColors == "3":
                    h5_analysis[" Col3 Image Array"] = self.roirect_col3.getArrayRegion(
                                self.imgarr_col3,
                                self.imv02.imageItem, axes=(1, 2))
                    h5_analysis["Col3 Kymograph"] = self.kymo_col3.T
                    h5_analysis["Col3 Kymograph Loop"] = self.kymo_col3_loop.T
                    h5_analysis["Col3 Kymograph No Loop"] = self.kymo_col3_noLoop.T
            if self.max_peak_dict is not None:
                h5_analysis["Left Max Peaks"] = self.max_peak_dict["Max Peak"].to_records()
                if self.numColors == "2" and self.max_smpeak_dict is not None:
                    h5_analysis["Right Max Peaks"] = self.max_smpeak_dict["Max Peak"].to_records()
            if self.df_peaks_linked is not None:
                h5_analysis["Left Linked Peaks"] = self.df_peaks_linked.to_records()
                if self.numColors == "2":
                    h5_analysis["Right Linked Peaks"] = self.df_peaks_linked_sm.to_records()
            if self.linkedpeaks_analyzed is not None:
                h5_analysis["Left Linked Peaks Analyzed"] = self.linkedpeaks_analyzed.to_records()
            if self.df_cols_linked is not None and len(self.df_cols_linked.index)>0:
                h5_analysis["Two Colors Linked"] = self.df_cols_linked.to_records()
            
            # Save manual properties
            # Copy all manual properties except the Widget, which cannot be saved to hdf5 files (and is not required to be saved...)
            props = []
            man_props = h5_analysis.create_group('manual_properties')
            for index, p in enumerate(self.ui.manualproperties):
                property = man_props.create_group(p['Label'])
                for key, value in p.items():
                    if key not in ['Widget', 'Textbox']:
                        property[key] = value
                if 'Index' not in property.keys() :
                    property['Index'] = index ## TODO: Fix this!!
            print(man_props)


    def load_manual_properties(self):
        '''
        Loads the manual properties set in the current hdf5 file. 
        The loaded properties are inserted the self.ui.manualproperties variable which contains all 
        current manual properties, their types and values.
        '''
        filepath_hdf5 = os.path.join(self.folderpath, self.filename_base + '_analysis.hdf5')
        if os.path.isfile(filepath_hdf5):
            h5_man_props = []
            # First clear out the manual property values
            for i, pp in enumerate(self.ui.manualproperties):
                if pp['Type'] == 'checkbox':
                    pp['Value'] = 0
                elif pp['Type'] == 'text':
                    pp['Value'] = ''
            
            # Then read the saved manual properties and fill in when given
            with h5py.File(filepath_hdf5, 'r') as h5_analysis_load:
                man_props = h5_analysis_load['manual_properties']
                if 'manual_properties' in list(h5_analysis_load.keys()):
                    for i, prop in enumerate(man_props.keys()):
                        p = h5_analysis_load['manual_properties'][prop]
                        label = p['Label'][()].decode() # hdf5 files store as byte strings so we use decode...
                        if 'Index' in p.keys():
                            index = int(p['Index'][()])
                        else:
                            index = i
                        value = p['Value'][()]
                        if type(value) == type(b''):
                            value = value.decode()
                        var_type = p['Type'][()].decode()
                        for i, pp in enumerate(self.ui.manualproperties):
                            if label == pp['Label']:
                                self.ui.manualproperties[i] = deepcopy({
                                    'Label': deepcopy(label),
                                    'Type' : deepcopy(var_type),
                                    'Value': deepcopy(value),
                                    'Widget': None,
                                    'Index': deepcopy(index)
                                })            
            self.ui.updateManualPropertiesBar()
        else:
            print(filepath_hdf5 + ' :(')
            print('No hdf5 file found, proceed as normal')
        

    def save_section(self):
        prev_state = self.ui.RealTimeKymoCheckBox.isChecked()
        self.ui.RealTimeKymoCheckBox.setChecked(False)
        temp_folder = os.path.abspath(os.path.join(self.folderpath, 'temp'))
        if not os.path.isdir(temp_folder):
            os.mkdir(temp_folder)
        nth_frame_to_save = self.ui.save_nth_frameSpinBox.value()
        current_index = self.imv00.currentIndex
        # save color0 video : d0left
        if self.ui.saveSectionComboBox.currentText() == "d0left" and self.ui.saveFormatComboBox.currentText() in [".mp4", ".avi", ".gif"]:
            roi_state = self.roirect_left.getState()
            self.roirect_left.setPos((-100, -100)) # move away from the imageItem
            print("Converting to video ...")
            pbar = tqdm.tqdm(total = 1 + int(self.imgarr_left.shape[0]/nth_frame_to_save))
            i = 0
            while i < self.imgarr_left.shape[0]:
                self.imv00.setCurrentIndex(i)
                exporter = pyqtgraph.exporters.ImageExporter(self.imv00.imageItem)
                exporter.export(os.path.join(temp_folder, 'temp_'+str(i)+'.png'))
                # self.imv00.jumpFrames(1)
                i += nth_frame_to_save
                pbar.update(1)
            self.roirect_left.setState(roi_state) #set back to its previous state
            filelist_png = glob.glob(temp_folder+'/temp_*.png')
            frame_rate = str(self.ui.saveFramerateSpinBox.value())
            extension = self.ui.saveFormatComboBox.currentText()
            filename = os.path.join(self.folderpath, self.filename_base + '_left' + extension)
            os.chdir(temp_folder)
            makevideo.png_to_video_cv2(temp_folder, filename, fps=int(frame_rate), scaling=4)
            for file in filelist_png:
                os.remove(file)
            os.rmdir(temp_folder)
            pbar.close()
            self.imv00.setCurrentIndex(current_index)
            print("Video conversion FINISHED")
        # save color0 current image : d0right
        elif self.ui.saveSectionComboBox.currentText() == "d0left" and self.ui.saveFormatComboBox.currentText() in [".svg", ".png", ".jpeg", ".tif"]:
            roi_state = self.roirect_left.getState()
            self.roirect_left.setPos((-100, -100)) # move away from the imageItem
            extension = self.ui.saveFormatComboBox.currentText()
            filename = os.path.join(self.folderpath, self.filename_base + '_left_frame' + str(current_index) + extension)
            exporter = pyqtgraph.exporters.ImageExporter(self.imv00.imageItem)
            exporter.params.param('width').setValue(int(exporter.params['width'] * 4))
            exporter.export(filename)
            self.roirect_left.setState(roi_state) #set back to its previous state
        # save color1 video : d0right
        elif self.ui.saveSectionComboBox.currentText() == "d0right" and self.ui.saveFormatComboBox.currentText() in [".mp4", ".avi", ".gif"]:
            roi_state = self.roirect_right.getState()
            self.roirect_right.setPos((-100, -100)) # move away from the imageItem
            print("Converting to video ...")
            pbar = tqdm.tqdm(total = 1 + int(self.imgarr_right.shape[0]/nth_frame_to_save))
            i = 0
            while i < self.imgarr_right.shape[0]:
                self.imv01.setCurrentIndex(i)
                exporter = pyqtgraph.exporters.ImageExporter(self.imv01.imageItem)
                exporter.export(os.path.join(temp_folder, 'temp_'+str(i)+'.png'))
                # self.imv00.jumpFrames(1)
                i += nth_frame_to_save
                pbar.update(1)
            self.roirect_right.setState(roi_state) #set back to its previous state
            filelist_png = glob.glob(temp_folder+'/temp_*.png')
            frame_rate = str(self.ui.saveFramerateSpinBox.value())
            extension = self.ui.saveFormatComboBox.currentText()
            filename = os.path.join(self.folderpath, self.filename_base + '_right' + extension)
            os.chdir(temp_folder)
            makevideo.png_to_video_cv2(temp_folder, filename, fps=int(frame_rate), scaling=4)
            for file in filelist_png:
                os.remove(file)
            os.rmdir(temp_folder)
            pbar.close()
            self.imv00.setCurrentIndex(current_index)
            print("Video conversion FINISHED")
        # save color1 current image : d0right
        elif self.ui.saveSectionComboBox.currentText() == "d0right" and self.ui.saveFormatComboBox.currentText() in [".svg", ".png", ".jpeg", ".tif"]:
            roi_state = self.roirect_left.getState()
            self.roirect_left.setPos((-100, -100)) # move away from the imageItem
            extension = self.ui.saveFormatComboBox.currentText()
            filename = os.path.join(self.folderpath, self.filename_base + '_right_frame' + str(current_index) + extension)
            exporter = pyqtgraph.exporters.ImageExporter(self.imv01.imageItem)
            exporter.params.param('width').setValue(int(exporter.params['width'] * 4))
            exporter.export(filename)
            self.roirect_left.setState(roi_state) #set back to its previous state
        # save ROIleft as tif
        elif self.ui.saveSectionComboBox.currentText() == "ROI:tif":
            filename = self.folderpath+'/'+self.filename_base + '_ROI.tif'
            if self.numColors == "1":
                roi_data = self.roirect_left.getArrayRegion(
                                    self.imgarr_left,
                                    self.imv00.imageItem, axes=(1, 2))
            elif self.numColors == "2":
                roi_data_1 = self.roirect_left.getArrayRegion(self.imgarr_left,
                                                    self.imv00.imageItem, axes=(1, 2))
                roi_data_2 = self.roirect_right.getArrayRegion(self.imgarr_right,
                                                    self.imv01.imageItem, axes=(1, 2))
                roi_data = np.concatenate((
                                          roi_data_2[:, np.newaxis, :, :],
                                          roi_data_1[:, np.newaxis, :, :],
                                          ), axis=1)
            imwrite(filename, roi_data.astype(np.uint16), imagej=True,
                    metadata={'axis': 'TCYX', 'channels': self.numColors, 'mode': 'composite',})
        # save left full kymo : d1left
        elif self.ui.saveSectionComboBox.currentText() == "d1left:tif":
            filename = self.folderpath+'/'+self.filename_base + '_left_kymo.tif'
            imwrite(filename, self.kymo_left.T.astype(np.uint16), imagej=True,
                    metadata={'axis': 'TCYX', 'channels': self.numColors, 'mode': 'composite',})
        # save right full kymo : d1right
        elif self.ui.saveSectionComboBox.currentText() == "d1right:tif":
            filename = self.folderpath+'/'+self.filename_base + '_right_kymo.tif'
            if self.ui.mergeColorsCheckBox.isChecked() and self.numColors == "2":
                kymo_comb = self.kymo_comb[:,:,:-1]
                for nChannel in range(kymo_comb.shape[2]):
                    temp = kymo_comb[:,:,nChannel]
                    temp /= np.max(temp)
                    kymo_comb[:,:,nChannel] = temp * (2**16-1)
                imwrite(filename, kymo_comb.T.astype(np.uint16), imagej=True,
                        metadata={'axis': 'TCYX', 'channels': self.numColors, 'mode': 'composite',})
                exporter = pyqtgraph.exporters.ImageExporter(self.imv11.imageItem)
                exporter.export(filename.replace('.tif', '.png'))
            else:
                imwrite(filename, self.kymo_right.T.astype(np.uint16), imagej=True,
                        metadata={'axis': 'TCYX', 'channels': self.numColors, 'mode': 'composite',})
        # save left selected kymo : d2left
        elif self.ui.saveSectionComboBox.currentText() == "d2left:tif":
            filename = self.folderpath+'/'+self.filename_base + '_left_selected_kymo.tif'
            imwrite(filename, self.kymo_left_loop.T.astype(np.uint16), imagej=True,
                    metadata={'axis': 'TCYX', 'channels': self.numColors, 'mode': 'composite',})
        # save right selected kymo : d2right
        elif self.ui.saveSectionComboBox.currentText() == "d2right:tif":
            filename = self.folderpath+'/'+self.filename_base + '_right_selected_kymo.tif'
            if self.ui.mergeColorsCheckBox.isChecked() and self.numColors == "2":
                kymo_loop_comb = self.kymo_loop_comb[:,:,:-1]
                for nChannel in range(kymo_loop_comb.shape[2]):
                    temp = kymo_loop_comb[:,:,nChannel]
                    temp /= np.max(temp)
                    kymo_loop_comb[:,:,nChannel] = temp * (2**16-1)
                imwrite(filename, kymo_loop_comb.T.astype(np.uint16), imagej=True,
                        metadata={'axis': 'TCYX', 'channels': self.numColors, 'mode': 'composite',})
                exporter = pyqtgraph.exporters.ImageExporter(self.imv23.imageItem)
                exporter.export(filename.replace('.tif', '.png'))
            else:
                imwrite(filename, self.kymo_right_loop.T.astype(np.uint16), imagej=True,
                        metadata={'axis': 'TCYX', 'channels': self.numColors, 'mode': 'composite',})
        self.ui.RealTimeKymoCheckBox.setChecked(prev_state)

    def frames_changed(self):
        print("Changing the frames and resetting plts...")
        start_time = time.time()
        self.frame_start = self.ui.frameStartSpinBox.value()
        self.frame_end = self.ui.frameEndSpinBox.value()
        self.set_img_stack()
        print("took %s seconds to reset!" % (time.time() - start_time))
        print("DONE:Changing the frames.")

    def closeEvent(self, event):
        self.reload_kymo_gui = False
        settings = io.load_user_settings()
        if self.folderpath is not None:
            settings["kymograph"]["PWD"] = self.folderpath
            settings["kymograph"]["Acquisiton Time"] = self.parameters_dialog.aqt_spinbox.value()
            settings["kymograph"]["Pixel Size"] = self.parameters_dialog.pix_spinbox.value()
            settings["kymograph"]['ROI width'] = self.LineROIwidth
        io.save_user_settings(settings)
        self.multipeak_dialog.on_close_event()
        QtWidgets.qApp.closeAllWindows()

    def load_user_settings(self):
        settings = io.load_user_settings()
        try:
            self.folderpath = settings["kymograph"]["PWD"]
            self.parameters_dialog.aqt_spinbox.setValue(settings["kymograph"]["Acquisiton Time"])
            self.parameters_dialog.pix_spinbox.setValue(settings["kymograph"]["Pixel Size"])
            self.parameters_dialog.roi_spinbox.setValue(settings["kymograph"]['ROI width'])
        except Exception as e:
            print(e)
            pass
        try:
            if len(self.folderpath) == 0:
                self.folderpath = None
        except:
            self.folderpath = None
        self.multipeak_dialog.on_start_event()


def main():
    win = Window()
    win.show()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    win = main()
    sys.exit(app.exec_())
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtWidgets.QApplication.instance().exec_()