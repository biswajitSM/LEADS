import os
import glob
import numpy as np
from scipy import ndimage, misc
import skimage as sk
from skimage.transform import rotate
from tifffile import imread, imsave, imwrite
from roifile import ImagejRoi
import PySimpleGUI as sg
import pims
from tqdm import trange 

def get_rect_params(rect, printing=False):
    length = int(np.sqrt((rect[0][0] - rect[3][0])**2 + (rect[0][1] - rect[3][1])**2))
    width = int(np.sqrt((rect[0][0] - rect[1][0])**2 + (rect[0][1] - rect[1][1])**2))
    dy = rect[3][1] - rect[0][1]
    dx = rect[3][0] - rect[0][0]
    if dx == 0:
        angle = 0
    else:
        m1 = (dy/dx);   # tan(A)
        A = np.arctan(m1) * 180 / np.pi;
        angle = 270-A#-(A-90)
    x_cent = int((rect[0, :][0] + rect[2, :][0])/2)
    y_cent = int((rect[0, :][1] + rect[2, :][1])/2)
    if printing:
        print('length:', length, ', width:', width, ', angle:', angle)
    rect_params = {
        'length': length,
        'width' : width,
        'angle' : int(angle),
        'x_cent' : x_cent,
        'y_cent' : y_cent,}
    return rect_params

def crop_rect(img, rect):
    # rect : array for rectanglular roi as in napari
    rect_params = get_rect_params(rect)
    if rect_params['angle'] != 0:
        img_rot = rotate(img, angle=rect_params['angle'],
                         center=(rect_params['y_cent'], rect_params['x_cent']))
    else:
        img_rot = img
    x = int(rect_params['x_cent'] - rect_params['width']/2)
    y = int(rect_params['y_cent'] - rect_params['length']/2)
    img_croped = img_rot[x:x+rect_params['width'], y:y+rect_params['length']]
    return sk.util.img_as_uint(img_croped)

def rect_shape_to_roi(arr_from_napari):
    arr_from_napari = np.flip(arr_from_napari)
    roi_coord = np.array([arr_from_napari[0], arr_from_napari[3],
                          arr_from_napari[2], arr_from_napari[1]], dtype=np.float32)
    return roi_coord

def save_rectshape_as_imageJroi(shape_layer):
    folder_to_save = sg.tkinter.filedialog.askdirectory(title="give the name of the folder to save")
    if not os.path.isdir(folder_to_save):
        print("folder to save doesn't exist")
    rect_shape = np.rint(shape_layer.data)
    for i in range(len(rect_shape)):
        rect_arr = rect_shape[i]
        rect_params = get_rect_params(rect_arr)
        rect_0 = rect_arr[0].astype(int)
        if i < 10: sl_no = str(0) + str(i)
        else: sl_no = str(i)
        nam = sl_no + '-x' + str(rect_0[0]) + '-y' + str(rect_0[1]) +\
              '-l' + str(rect_params['length']) + '-w' + str(rect_params['width']) +\
              '-a' + str(rect_params['angle']) + 'd'
        roi_coord = rect_shape_to_roi(rect_arr)
        roi_ij = ImagejRoi.frompoints(roi_coord)
        roi_ij.tofile(os.path.join(folder_to_save, nam+'.roi'))
    return

def roi_to_rect_shape(roi_imagej):
    roi = ImagejRoi.fromfile(roi_imagej)
    roi_coord = np.flip(roi.coordinates())
    arr_rect_napari = np.array([roi_coord[2], roi_coord[1], roi_coord[0], roi_coord[3]], dtype=np.float32)    
    return arr_rect_napari

def addroi_to_shapelayer(shape_layer, roi_file_list):
    roi_arr_list = []
    for roi_file in roi_file_list:
        roi_arr = roi_to_rect_shape(roi_file)
        roi_arr_list.append(roi_arr)
    shape_layer.data = roi_arr_list
    return roi_arr_list

def crop_rect_shapes(image_meta, shape_layer, dir_to_save=None,
                     frame_start=None, frame_end=None):
    '''
    shape_layer should be a shape object from napari viewer 
    '''
    folderpath = image_meta['folderpath']
    if frame_start is None: frame_start = 0
    if frame_end is None: frame_end = image_meta['num_frames']  // image_meta['num_colors']
    num_frames_update = image_meta['num_colors'] * ((frame_end - frame_start) // image_meta['num_colors'])
    frame_end = frame_start + num_frames_update
    
    rect_shape_temp = np.rint(shape_layer.data)
    shape_layer.data = rect_shape_temp
    if len(rect_shape_temp) == 1:
        if shape_layer.shape_type[0] != 'rectangle':
            print('The shape is not rectangular, please only give rectngular shapes')
        else: rect_shape = rect_shape_temp
    else:
        rect_shape = []
        for i in range(len(rect_shape_temp)):
            if shape_layer.shape_type[i] == 'rectangle':
                rect_shape.append(rect_shape_temp[i])
    print('len of rectshape: ', len(rect_shape))
    
    if dir_to_save is None:
        dir_to_save = os.path.join(folderpath+'_analysis')
        if not os.path.isdir(dir_to_save):
            os.makedirs(dir_to_save)
    names_roi_tosave = []
    img_array_all = {}
    for i in range(len(rect_shape)):
        rect = rect_shape[i]
        rect_params = get_rect_params(rect)
        key = 'arr' + str(i)
        img_array_all[key] = np.zeros((num_frames_update, image_meta['num_colors'],
                                       rect_params['width'], rect_params['length']),
                                       dtype=np.uint16)        
        # name = pix_x-pix_y-length-width-angle-frame_start-frame_end
        rect_0 = rect[0].astype(int)
        if i < 10: sl_no = str(0) + str(i)
        else: sl_no = str(i)
        nam = 'x' + str(rect_0[0]) + '-y' + str(rect_0[1]) +\
              '-l' + str(rect_params['length']) + '-w' + str(rect_params['width']) +\
              '-a' + str(rect_params['angle']) + 'd-f' + str(frame_start) + '-f' +\
              str(frame_end)
        names_roi_tosave.append(nam)
    rect_keys = list(img_array_all.keys())   
    for col in range(image_meta['num_colors']):
        print('Corpping color: {} ...'.format(col))
        imgseq = pims.ImageSequence(image_meta['filenames_color_'+str(col)])[frame_start:frame_end]
        for i in trange(num_frames_update, desc='cropping images'):
            img = np.array(imgseq[i], dtype=np.uint16)
            for j in range(len(rect_shape)):
                img_croped = crop_rect(img, rect_shape[j])
                img_array_all[rect_keys[j]][i, col, :, :] = img_croped

    for i in range(len(rect_shape)):
        roi_ij = ImagejRoi.frompoints(rect_shape_to_roi(rect_shape[i]))
        roi_ij.tofile(os.path.join(dir_to_save, names_roi_tosave[i]+'.roi'))
        imwrite(os.path.join(dir_to_save, names_roi_tosave[i]+'.tif'),
                img_array_all[rect_keys[i]], imagej=True,
                metadata={'axis': 'TCYX', 'channels': image_meta['num_colors'],
                'mode': 'composite',})
    print(" croping is finished")
    return


from skimage.io import imread
from skimage.io.collection import alphanumeric_key
from dask import delayed
import dask.array as da
folderpath = ''
def daskread_img_seq(num_colors=1, bkg_subtraction=False, path=""):
    '''
    Import image sequences (saved individually in a folder)
    num_colors : integer
    returns: dask arrays and 
    '''
    image_meta = {}
    image_meta['num_colors'] = num_colors
    global folderpath
    if path:
        folderpath = path
    else:
        filepath = sg.tkinter.filedialog.askopenfilename(title = "Select tif file/s",
                                                        filetypes = (("tif files","*.tif"),("all files","*.*")))
        folderpath = os.path.dirname(filepath)
    image_meta['folderpath'] = folderpath
    filenames = sorted(glob.glob(folderpath + "/*.tif"), key=alphanumeric_key)
    image_meta['filenames'] = filenames
    # read the first file to get the shape and dtype
    # ASSUMES THAT ALL FILES SHARE THE SAME SHAPE/TYPE
    sample = imread(filenames[0])
    if bkg_subtraction:
        sample = bkg_substration(sample)
    print('minimum intensity: {}, maximum intensity: {}'.format(sample.min(), sample.max()))

    lazy_imread = delayed(imread)  # lazy reader
    lazy_arrays = [lazy_imread(fn) for fn in filenames] # read delayed images for all filenames
    dask_arrays = [
        da.from_delayed(delayed_reader, shape=sample.shape, dtype=sample.dtype)
        for delayed_reader in lazy_arrays
    ]

    # Stack into one large dask.array
    stack = da.stack(dask_arrays, axis=0)
    print('Whole Stack shape: ', stack.shape)  # (nfiles, nz, ny, nx)

    # background subtraction
    if bkg_subtraction:
        stack = stack.map_blocks(bkg_substration)
   

    num_frames = num_colors * (len(filenames)//num_colors)
    image_meta['num_frames'] = num_frames
    for i in range(num_colors):
        image_meta['filenames_color_'+str(i)] = filenames[i:num_frames:num_colors]
        image_meta['stack_color_'+str(i)] = stack[i:num_frames:num_colors]        
        sample = imread(filenames[i])
        image_meta['min_int_color_'+str(i)] = sample.min()
        image_meta['max_int_color_'+str(i)] = sample.max()        
    return image_meta


from scipy.ndimage import white_tophat, black_tophat
def bkg_substration(txy_array, size_bgs=150, light_bg=False):
    array_processed = np.zeros_like(txy_array)
    if array_processed.ndim>2:
        for i in range(txy_array.shape[0]):
            img = txy_array[i]
            if light_bg:
                img_bgs = black_tophat(img, size=size_bgs)
            else:
                img_bgs = white_tophat(img, size=size_bgs)
            array_processed[i, :, :] = img_bgs
    else:
        if light_bg:
            array_processed = black_tophat(txy_array, size=size_bgs)
        else:
            array_processed = white_tophat(txy_array, size=size_bgs)
    return array_processed
