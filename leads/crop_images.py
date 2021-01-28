import os
import glob
import numpy as np
from scipy import ndimage, misc
import skimage as sk
from skimage.filters import threshold_otsu
from skimage.transform import rotate
from tifffile import imread, imsave, imwrite
from roifile import ImagejRoi
import pims
import yaml
from tqdm import trange
from . import io

# ---------------------------------------------------------------------
def get_rect_params(rect, printing=False):
    length = int(np.sqrt((rect[0][0] - rect[3][0])**2 + (rect[0][1] - rect[3][1])**2))
    width = int(np.sqrt((rect[0][0] - rect[1][0])**2 + (rect[0][1] - rect[1][1])**2))
    dy = rect[3][1] - rect[0][1]
    dx = rect[3][0] - rect[0][0]
    if dx == 0:
        angle = 0
    else:
        m1 = (dy/dx);   # tan(A)
        A = np.arctan(m1) * 180 / np.pi
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

# ---------------------------------------------------------------------
def crop_rect(img, rect):
    # rect : array for rectanglular roi as in napari
    rect_params = get_rect_params(rect)
    # if rect_params['angle'] != 0:
    #     img_rot = rotate(img, angle=rect_params['angle'],
    #                      center=(rect_params['y_cent'], rect_params['x_cent']))
    # else:
    #     img_rot = img
    x = int(rect_params['x_cent'] - rect_params['width']/2)
    y = int(rect_params['y_cent'] - rect_params['length']/2)
    img_croped = img[x:x+rect_params['width'], y:y+rect_params['length']]
    return sk.util.img_as_uint(img_croped)

# ---------------------------------------------------------------------
def rect_shape_to_roi(arr_from_napari):
    arr_from_napari = np.flip(arr_from_napari)
    roi_coord = np.array([arr_from_napari[0], arr_from_napari[3],
                          arr_from_napari[2], arr_from_napari[1]], dtype=np.float32)
    return roi_coord

# ---------------------------------------------------------------------
def save_rectshape_as_imageJroi(shape_layer, folder_to_save=None, label=None, defaultShape=None):
    if folder_to_save is None:
        folder_to_save = io.FileDialog().openDirectoryDialog()
    if not os.path.isdir(folder_to_save):
        os.makedirs(folder_to_save)
    label = cleanLabel(label)
    
    rect_shape = np.rint(shape_layer.data)
    if len(rect_shape)==0:
        return
    numShapes = len(rect_shape)
    # remove column with all 0 entries
    rect_shape = [rect_shape[i][:,(rect_shape[i]!=0).any(axis=0)] for i in range(len(rect_shape))]
    names = []
    for i in range(numShapes):
        rect_arr = rect_shape[i]
        # if the ROI is at the default position, continue; we dont save it        
        if (rect_shape[i]==defaultShape).all():
            continue
        rect_params = get_rect_params(rect_arr)
        rect_0 = rect_arr[0].astype(int)
        # sl_no = ''
        # if numShapes>1:
        #     if i < 10: sl_no = str(0) + str(i)
        #     else: sl_no = str(i)
        #     sl_no = sl_no + '_'
        nam = 'x' + str(rect_0[0]) + '-y' + str(rect_0[1]) +\
              '-l' + str(rect_params['length']) + '-w' + str(rect_params['width']) +\
              '-a' + str(rect_params['angle']) + label.lower()

        roi_coord = rect_shape_to_roi(rect_arr)
        roi_ij = ImagejRoi.frompoints(roi_coord)
        roi_ij.tofile(os.path.join(folder_to_save, nam+'.roi'))
        names.append(nam)
    return names

# ---------------------------------------------------------------------
def roi_to_rect_shape(roi_imagej):
    roi = ImagejRoi.fromfile(roi_imagej)
    roi_coord = np.flip(roi.coordinates())
    arr_rect_napari = np.array([roi_coord[2], roi_coord[1], roi_coord[0], roi_coord[3]], dtype=np.float32)    
    return arr_rect_napari

# ---------------------------------------------------------------------
def addroi_to_shapelayer(shape_layer, roi_file_list):
    roi_arr_list = []
    for roi_file in roi_file_list:
        roi_arr = roi_to_rect_shape(roi_file)
        roi_arr_list.append(roi_arr)
    shape_layer.data = roi_arr_list
    return roi_arr_list

# ---------------------------------------------------------------------
def cleanLabel(label):
    if label is not None:
        label = label.replace(' ', '_')
        label = label.replace('-', '_')
        label = label.replace(']', '')
        label = label.replace('[', '')
        if '_' in label:
            fractions = label.rsplit("_")
            label = label.replace('_'+fractions[-1],'') # the last entry is the number of the layer. we remove this
        # in case the last entry was not actually a number, then we have now nothing left
        if len(label) == 0:
            label = fractions[-1]
        if label == 'Shapes': # 'Shapes' is the default name when the user creates a new layer manually
            label = 'ROI'
        # if label dont have a leading underscore, add it
        if label[0] != '_':
            label = '_' + label
    else:
        label = ''
    return label

# ---------------------------------------------------------------------
def readROIassociatedYamlFile(roi_file, num_colors):
    # for all colors in the yaml file
    # if we cannot find it, set all to zero
    yamlFileName = roi_file[:-4] + '.yaml'
    shift_yaml = LoadShiftYamlFile(yamlFileName, 0, 0, 0, num_colors)
    angle = []
    shift_x = []
    shift_y = []
    for col in range(len(shift_yaml['x'])):
        if "angle" in shift_yaml:
            angle.append(shift_yaml['angle']['col'+str(int(col))])
        else:
            angle.append(0)
        shift_x.append(shift_yaml['x']['col'+str(int(col))])
        shift_y.append(shift_yaml['y']['col'+str(int(col))])
    return angle, shift_x, shift_y 

# ---------------------------------------------------------------------
def LoadShiftYamlFile(yamlFileName, xShift, yShift, angle, numColors):
    if os.path.isfile(yamlFileName): # if it exists, load it. assume there is only one such file
        try:
            yaml_file = open(yamlFileName, "r")
        except FileNotFoundError:
            shift_yaml = MakeShiftYamlFile(xShift, yShift, angle, numColors)
            return shift_yaml
        try:
            shift_yaml = yaml.load(yaml_file, Loader=yaml.FullLoader)
            yaml_file.close()
            shift_yaml = io.AutoDict(shift_yaml)
        except:
            shift_yaml = MakeShiftYamlFile(xShift, yShift, angle, numColors)
    else: # if is doesnt exist, we create all structures from scratch                
        shift_yaml = MakeShiftYamlFile(xShift, yShift, angle, numColors) 
    return shift_yaml

# ---------------------------------------------------------------------
def MakeShiftYamlFile(xShift, yShift, angle, numColors):
    shift_yaml = {}
    shift_yaml["x"] = {}
    shift_yaml["y"] = {}
    shift_yaml["angle"] = {}
    for nColor in range(numColors):
        shift_yaml["x"]["col"+str(int(nColor))] = xShift
        shift_yaml["y"]["col"+str(int(nColor))] = yShift
        shift_yaml["angle"]["col"+str(int(nColor))] = angle
    return shift_yaml

# ---------------------------------------------------------------------
def crop_rect_shapes(image_meta, shape_layers, dir_to_save=None,
                     frame_start=None, frame_end=None,
                     geometric_transform=False,
                     shift_x=0, shift_y=0, angle=0, label=None, 
                     defaultShape=None, numColors=0):
    '''
    shape_layer should be a shape object from napari viewer 
    '''
    
    if frame_start is None: frame_start = 0
    if frame_end is None: frame_end = image_meta['num_frames']  // image_meta['num_colors']
    num_frames_update = image_meta['num_colors'] * ((frame_end - frame_start) // image_meta['num_colors'])
    frame_end = frame_start + num_frames_update
    
    if type(shape_layers) is not list: # if only a single layer was supplied
        shape_layers = [shape_layers]
    if type(label) is not list: # if only a single layer was supplied
        label = [label]

    # loop through layers and extract rect_shape's
    numLayers = len(shape_layers)
    rect_shape = []
    labels     = []
    for nLayer in range(numLayers):
        # first see if there is any shape in the layer. if not, continue
        rect_shape_temp = np.rint(shape_layers[nLayer].data)
        if len(rect_shape_temp)==0:
            continue    

        # first get the label and clean up
        label[nLayer] = cleanLabel(label[nLayer])

        # now get the rect_shape
        # in case the shape was added by the user manually, it contains by default a 
        # third (z) dimension which is all 0 for 2D images. We need to remove this
        # in order not to confuse ImagejRoi.frompoints
        # for layers and shapes we added ourselves, layer.data is empty. we need a special case for that
        # if len(shape_layers[nLayer].data)==0: 
            # print('if worked')
        if len(rect_shape_temp) == 1: # check if we have a rectangle
            if rect_shape_temp[0].shape[0] != 4:
                print('The shape is not rectangular, please only give rectngular shapes')
            else:
                # remove column with all 0 entries
                rect_shape_temp = [rect_shape_temp[i][:,(rect_shape_temp[i]!=0).any(axis=0)] for i in range(len(rect_shape_temp))]#rect_shape_temp[0][:,(rect_shape_temp[0]!=0).any(axis=0)]
                if not (rect_shape_temp[0]==defaultShape).all():
                    rect_shape.append(rect_shape_temp[0])
                    labels.append(label[nLayer])
        else:
            # remove column with all 0 entries
            rect_shape_temp = [rect_shape_temp[i][:,(rect_shape_temp[i]!=0).any(axis=0)] for i in range(len(rect_shape_temp))]
            for i in range(len(rect_shape_temp)):
                if rect_shape_temp[i].shape[0] == 4:
                    # check if the current shape is the default shape. if so, skip
                    if (rect_shape_temp[i]==defaultShape).all():
                        continue
                    rect_shape.append(rect_shape_temp[i])
                    labels.append(label[nLayer])
        # else: # loaded ROI. nothing we can use shape_layers[nLayer].shape_type
        #     shape_layers[nLayer].data = rect_shape_temp
        #     if len(rect_shape_temp) == 1:
        #         if shape_layers[nLayer].shape_type[0] != 'rectangle':
        #             print('The shape is not rectangular, please only give rectngular shapes')
        #         else:
        #             # remove column with all 0 entries
        #             rect_shape_temp = [rect_shape_temp[i][:,(rect_shape_temp[i]!=0).any(axis=0)] for i in range(len(rect_shape_temp))]
        #             # check if the current shape is the default shape. if so, skip
        #             if not (rect_shape_temp[0]==defaultShape).all():
        #                 rect_shape.append(rect_shape_temp[0])
        #                 labels.append(label[nLayer])
        #     else:
        #         # remove column with all 0 entries
        #         rect_shape_temp = [rect_shape_temp[i][:,(rect_shape_temp[i]!=0).any(axis=0)] for i in range(len(rect_shape_temp))]
        #         for i in range(len(rect_shape_temp)):
        #             if shape_layers[nLayer].shape_type[i] == 'rectangle':
        #                 # check if the current shape is the default shape. if so, skip
        #                 if (rect_shape_temp[i]==defaultShape).all():
        #                     continue
        #                 rect_shape.append(rect_shape_temp[i])
        #                 labels.append(label[nLayer])

    # now labels and rect_shape have the same length
    # if there are none to process => only shapes with the default position, return without doing anything
    if len(labels) == 0:
        print('Found only ROIs at the default position. Move ROIs before cropping')
        return

    folderpath = image_meta['folderpath']
    if dir_to_save is None:
        dir_to_save = os.path.join(folderpath+'_analysis')
        if not os.path.isdir(dir_to_save):
            os.makedirs(dir_to_save)

    names_roi_tosave = []
    names_tif_tosave = []
    img_array_all    = {}
    imgseq = pims.ImageSequence(image_meta['filenames_color_'+str(0)])[0]        
    img = np.array(imgseq, dtype=np.uint16)
    imgSize = img.shape
    for i in range(len(rect_shape)):
        # first get the names, based on the real ROI as it was saved
        rect = rect_shape[i]
        rect_params = get_rect_params(rect)
        rect_0 = rect[0].astype(int)
        shift_text = ''
        if geometric_transform:
            shift_text = shift_text + '_shifted_' + str(int(shift_x[0])) + 'dx_' + str(int(shift_y[0])) + 'dy'
        nam = 'x' + str(rect_0[0]) + '-y' + str(rect_0[1]) +\
              '-l' + str(rect_params['length']) + '-w' + str(rect_params['width']) +\
              '-a' + str(rect_params['angle']) + '-f' + str(frame_start) + '-f' +\
              str(frame_end) + shift_text + labels[i].lower()
        names_tif_tosave.append(nam) # with the -f flags
        nam = 'x' + str(rect_0[0]) + '-y' + str(rect_0[1]) +\
              '-l' + str(rect_params['length']) + '-w' + str(rect_params['width']) +\
              '-a' + str(rect_params['angle']) + shift_text + labels[i].lower()
        names_roi_tosave.append(nam) # without the -f flags

        # now after having the name, correct if the ROI is outside the image dimension
        rect_shape[i] = [[np.max((x, 1)) for x in y] for y in rect_shape[i]]
        rect_shape[i] = np.asarray(
            [
            np.array(
                (
                    np.min((y[0],imgSize[0])), 
                    np.min((y[1],imgSize[1]))
                )
            ) 
                for y in rect_shape[i]
            ]
        )
        rect = rect_shape[i]
        rect_params = get_rect_params(rect)
        key = 'arr' + str(i)
        img_array_all[key] = np.zeros((num_frames_update, image_meta['num_colors'],
                                       rect_params['width'], rect_params['length']),
                                       dtype=np.uint16)        


    rect_keys = list(img_array_all.keys())   
    for col in range(image_meta['num_colors']):
        imgseq = pims.ImageSequence(image_meta['filenames_color_'+str(col)])[frame_start:frame_end]
        for i in trange(num_frames_update, desc='Cropping color: {} '.format(col)):
            img = np.array(imgseq[i], dtype=np.uint16)
            # when the first image is loaded, determine if we better shift
            # the crop only for speed (as long as no shift > 10% of image 
            # dimension) or if we have to shift the whole image, which is slower
            # but we dont lose half of the crop from shifting
            if i==0:
                minWidth = float('inf')
                minLength = float('inf')
                for nRect in range(len(rect_shape)):
                    rect_params = get_rect_params(rect_shape[nRect])
                    minWidth = np.min([minWidth, rect_params['width']])
                    minLength = np.min([minLength, rect_params['length']])
                percentage = 0.1 # 10% of the smallest crop
                if (shift_x[col]/minLength>percentage) or (shift_y[col]/minWidth>percentage):
                    shift_wholeImage = True
                else:
                    shift_wholeImage = False                

            # shift whole image
            if shift_wholeImage and geometric_transform and ((angle[col]!=0) or (shift_x[col]!=0) or (shift_y[col]!=0)):
                img = geometric_shift(img, angle=angle[col],
                                    shift_x=shift_x[col], shift_y=shift_y[col])
            for j in range(len(rect_shape)):                
                img_croped = crop_rect(img, rect_shape[j])
                # shift crop if true
                if (not shift_wholeImage) and geometric_transform and ((angle[col]!=0) or (shift_x[col]!=0) or (shift_y[col]!=0)):
                    img_croped = geometric_shift(img_croped, angle=angle[col],
                                    shift_x=shift_x[col], shift_y=shift_y[col])
                # append the image to the stack
                img_array_all[rect_keys[j]][i, col, :, :] = img_croped

    # save ROIs and associated yaml files
    yamlFileName = os.path.join(folderpath, 'shift.yaml')                           
    shift_yaml = LoadShiftYamlFile(yamlFileName, shift_x, shift_y, angle, image_meta['num_colors'])
    shift_yaml["numColors"] = image_meta['num_colors']
    for nColor in range(image_meta['num_colors']):
        shift_yaml["x"]["col"+str(int(nColor))] = shift_x[nColor]
        shift_yaml["y"]["col"+str(int(nColor))] = shift_y[nColor]            
    # save the yaml file
    shift_yaml = io.to_dict_walk(shift_yaml)
    for i in range(len(rect_shape)):
        roi_ij = ImagejRoi.frompoints(rect_shape_to_roi(rect_shape[i]))
        roi_ij.tofile(os.path.join(dir_to_save, names_roi_tosave[i]+'.roi'))
        imwrite(os.path.join(dir_to_save, names_tif_tosave[i]+'.tif'),
                img_array_all[rect_keys[i]], imagej=True,
                metadata={'axis': 'TCYX', 'channels': image_meta['num_colors'],
                'mode': 'composite',})

        yamlFileName = os.path.join(dir_to_save, names_roi_tosave[i]+'.yaml')
        with open(yamlFileName, "w") as shift_file:
            yaml.dump(dict(shift_yaml), shift_file, default_flow_style=False)
    return


from skimage.io import imread
from skimage.io.collection import alphanumeric_key
from dask import delayed
import dask.array as da
import random
folderpath = ''
# ---------------------------------------------------------------------
def daskread_img_seq(num_colors=1, bkg_subtraction=False, path="", RotationAngle=None):
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
        for file in os.listdir(folderpath):
            if file.endswith(".tif") or file.endswith(".tiff"):
                filepath = file
                break
    else:
        settings = io.load_user_settings()
        try:
            folderpath = settings["crop"]["PWD"]
            if os.path.isfile(folderpath):
                folderpath = os.path.dirname(folderpath)
            folderpath = os.path.dirname(folderpath)
            # filepath = io.FileDialog(folderpath, "Select a folder with tif",
            #                      "Tif File (*.tif *.tiff)").openFileNameDialog()
            filepath = io.FileDialog(folderpath, "Select a folder containing .tif files",).openFolderNameDialog()
        except Exception as e:
            print(e)
            # filepath = io.FileDialog(None, "open a tif file stack",
            #                      "Tif File (*.tif *.tiff)").openFileNameDialog()
            filepath = io.FileDialog(None, "Select a folder containing .tif files",).openFolderNameDialog()
            pass
        if not filepath:
            image_meta = {}
            return image_meta
        if os.path.isfile(filepath):
            folderpath = os.path.dirname(filepath)
        else:
            folderpath = filepath
    image_meta['folderpath'] = folderpath
    if filepath.endswith('tif'):
        filenames = sorted(glob.glob(folderpath + "/*.tif"), key=alphanumeric_key)
    elif filepath.endswith('tiff'):
        filenames = sorted(glob.glob(folderpath + "/*.tiff"), key=alphanumeric_key)
    else:
        filenames = sorted(glob.glob(folderpath + "/*.tif"), key=alphanumeric_key)
        if len(filenames)==0:
            filenames = sorted(glob.glob(folderpath + "/*.tiff"), key=alphanumeric_key)
    image_meta['filenames'] = filenames
    # read the first file to get the shape and dtype
    # ASSUMES THAT ALL FILES SHARE THE SAME SHAPE/TYPE
    sample = imread(filenames[0])
    if bkg_subtraction:
        sample = bkg_substration(sample)
    # print('minimum intensity: {}, maximum intensity: {}'.format(sample.min(), sample.max()))

    lazy_imread = delayed(imread)  # lazy reader
    lazy_arrays = [lazy_imread(fn) for fn in filenames] # read delayed images for all filenames
    dask_arrays = [
        da.from_delayed(delayed_reader, shape=sample.shape, dtype=sample.dtype)
        for delayed_reader in lazy_arrays
    ]

    # save current folderpath for the next time when the dialog is opened
    settings = io.load_user_settings()
    if folderpath is not None:
        settings["crop"]["PWD"] = folderpath
    io.save_user_settings(settings)

    # Stack into one large dask.array
    stack = da.stack(dask_arrays, axis=0)
    # print('Whole Stack shape: ', stack.shape)  # (nfiles, nz, ny, nx)

    # background subtraction
    if bkg_subtraction:
        stack = stack.map_blocks(bkg_substration)
   
    # apply rotation
    if RotationAngle is not None:
        if any([x != 0 for x in RotationAngle]):
            stack = stack.map_blocks(applyRotation, RotationAngle=RotationAngle, num_colors=num_colors)

    num_frames = num_colors * (len(filenames)//num_colors)
    image_meta['num_frames'] = num_frames
    minVal = []; maxVal = []
    minCon = []; maxCon = []
    sampling = random.choices(range(0, num_frames, num_colors), k=10)
    for i in range(num_colors):
        image_meta['filenames_color_'+str(i)] = filenames[i:num_frames:num_colors]
        image_meta['stack_color_'+str(i)] = stack[i:num_frames:num_colors]
        for k in sampling:
            sample = imread(filenames[k+i])
            if bkg_subtraction:
                sample = bkg_substration(sample)
            minVal.append(sample.min())
            maxVal.append(sample.max())
            minConTemp, maxConTemp = AutoAdjustContrast(sample)
            minCon.append(minConTemp)
            maxCon.append(maxConTemp)
        image_meta['min_int_color_'+str(i)] = np.mean(minVal)
        image_meta['max_int_color_'+str(i)] = np.mean(maxVal)        
        image_meta['min_contrast_color_'+str(i)] = np.mean(minCon)
        image_meta['max_contrast_color_'+str(i)] = np.mean(maxCon)
        # print('Channel {}: adjusted min intensity {}, adjusted max intensity {}'.format(i, round(np.mean(minCon)), round(np.mean(maxCon))))    
    
    return image_meta

# ---------------------------------------------------------------------
def applyRotation(txy_array, RotationAngle=[0], num_colors=1):
    array_processed = np.zeros_like(txy_array)
    if array_processed.ndim > 2:
        for i in range(txy_array.shape[0]):
            img = txy_array[i]
            ### i tried here to compute which channel we are currently pocessing, but
            ### this function gets only fed one image at a time, so I dont know
            ### how to decide which image from the series is shown and, respectively,
            ### which color it is
            # col = np.mod(i,num_colors)
            img_rot = ndimage.rotate(img, RotationAngle[0], reshape=False)
            array_processed[i, :, :] = img_rot
    else:
        array_processed = ndimage.rotate(txy_array, RotationAngle[0], reshape=False)
        # imutils.rotate(txy_array, angle=RotationAngle)
    return array_processed

# ---------------------------------------------------------------------
def AutoAdjustContrast(img):
    size = img.shape
    pixelCount = size[0]*size[1]
    limit = pixelCount/10
    AUTO_THRESHOLD = 5000/2
    threshold = pixelCount/AUTO_THRESHOLD
    hist = np.histogram(img, bins=255, range=(img.min(), img.max()))
    i = -1
    found = False
    while (not found) and (i<255):
        i = i+1
        count = hist[0][i] #hist[0] are histogram counts
        if count>limit: count = 0
        found = count>threshold
    hmin = i
    found = False
    i = 255
    while (not found) and (i>0):
        i = i-1
        count = hist[0][i]
        if (count>limit): count = 0
        found = count > threshold
    hmax = i
    if (hmax<hmin):
        hmin = img.min()
        hmax = img.max()
    else: # map back to real pixel values
        hmin = hist[1][hmin]
        hmax = hist[1][hmax]
    return hmin, hmax

# ---------------------------------------------------------------------
from scipy.ndimage import white_tophat, black_tophat
def bkg_substration(txy_array, size_bgs=50, light_bg=False):
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

# ---------------------------------------------------------------------
def geometric_shift(image2d, angle, shift_x, shift_y):
    if angle!=0:
        image2d = ndimage.rotate(image2d, angle, reshape=False)
    if float(shift_y)!=0 or float(shift_x)!=0:
        image2d = ndimage.shift(image2d, (float(shift_y), float(shift_x)))
    return image2d