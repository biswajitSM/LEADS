import os
import glob
import numpy as np
from scipy import ndimage, misc, optimize, special
import skimage as sk
from skimage.transform import rotate
from tifffile import imread, imsave, imwrite
from roifile import ImagejRoi
import pims
import yaml
import re
from tqdm import trange
from . import io
from scipy.spatial import distance as dist
from matplotlib import pyplot as plt
from PyQt5.QtWidgets import QMessageBox

# ---------------------------------------------------------------------
def get_rect_params(rect, printing=False):
    length = int(np.sqrt((rect[0][0] - rect[3][0])**2 + (rect[0][1] - rect[3][1])**2))
    width = int(np.sqrt((rect[0][0] - rect[1][0])**2 + (rect[0][1] - rect[1][1])**2))
    dy = rect[3][1] - rect[0][1]
    dx = rect[3][0] - rect[0][0]
    if dx == 0 or dy == 0:
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
def crop_rect(img, rect, angle=0):
    # rect : array for rectanglular roi as in napari
    rect_params = get_rect_params(rect)
    if rect_params['angle'] != 0 or angle!=0:
        img_rot = rotate(img, angle=rect_params['angle']+angle)
    else:
        img_rot = img
    x = int(rect_params['x_cent'] - rect_params['width']/2)
    y = int(rect_params['y_cent'] - rect_params['length']/2)
    img_cropped = img_rot[x:x+rect_params['width'], y:y+rect_params['length']]
    return sk.util.img_as_uint(img_cropped)

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
    for iRect in range(numShapes):
        if (rect_shape[iRect]==defaultShape).all():
            continue
        rect = np.array(rect_shape[iRect])
        rect = order_points(rect)

        rect_0 = rect[0].astype(int)
                            
        dy = rect[3][1] - rect[0][1]
        dx = rect[3][0] - rect[0][0]
        if dx == 0 or dy == 0:
            angle_roi = int(0)
        else:
            angle_roi = int(np.arctan(dy/dx) * 180 / np.pi)
        width_x = int( np.ceil( np.sqrt(np.sum((rect[3,:]-rect[2,:])**2)) ) )
        height_y = int( np.ceil( np.sqrt(np.sum((rect[2,:]-rect[1,:])**2)) ) )
        nam = 'x' + str(rect_0[0]) + '-y' + str(rect_0[1]) +\
            '-l' + str(height_y) + '-w' + str(width_x) +\
            '-a' + str(angle_roi) + label.lower()
        roi_ij = ImagejRoi.frompoints(rect_shape_to_roi(rect))
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
    file_corrected = False

    for col in range(shift_yaml['numColors']):
            if not 'col'+str(int(col)) in shift_yaml['angle'] \
                or not 'col'+str(int(col)) in shift_yaml['x'] \
                or not 'col'+str(int(col)) in shift_yaml['y']:
                shift_yaml['numColors'] = col
                file_corrected = True
                continue
            if "angle" in shift_yaml:
                angle.append(shift_yaml['angle']['col'+str(int(col))])
            else:
                angle.append(0)
            shift_x.append(shift_yaml['x']['col'+str(int(col))])
            shift_y.append(shift_yaml['y']['col'+str(int(col))])

    if file_corrected:
        shift_yaml = io.to_dict_walk(shift_yaml)
        os.makedirs(os.path.dirname(yamlFileName), exist_ok=True)
        with open(yamlFileName, "w") as shift_file:
            yaml.dump(dict(shift_yaml), shift_file, default_flow_style=False)
    return angle, shift_x, shift_y, shift_yaml['numColors']

# ---------------------------------------------------------------------
def LoadShiftYamlFile(yamlFileName, xShift, yShift, angle, numColors):
    if os.path.isfile(yamlFileName): # if it exists, load it. assume there is only one such file
        try:
            yaml_file = open(yamlFileName, "r")
        except FileNotFoundError:
            print('Not found: '+yamlFileName)
            shift_yaml = MakeShiftYamlFile(xShift, yShift, angle, numColors)
            return shift_yaml
        try:
            shift_yaml = yaml.load(yaml_file, Loader=yaml.FullLoader)
            yaml_file.close()
            shift_yaml = io.AutoDict(shift_yaml)
        except:
            print('Found but not readable: '+yamlFileName)
            # it might be here that yaml files are corrupt. This is especially crucial for cropping.
            # If we encounter such a file in a posXX_analysis folder, try to look at the corresponding folder with '_analysis'
            # our folders have usually the names pos or default since we use micromanager
            pos_analysis_folder = re.search("pos\d+_analysis", yamlFileName.lower())        
            default_analysis_folder = re.search("default\d+_analysis", yamlFileName.lower())
            if (pos_analysis_folder is not None) or (default_analysis_folder is not None):
                yamlFileInImageFolder = os.path.join(os.path.dirname(yamlFileName)[:-len('_analysis')], 'shift.yaml')
                try:
                    yaml_file = open(yamlFileInImageFolder, "r")
                except FileNotFoundError:
                    print('Alternative yaml file not found: '+yamlFileInImageFolder)
                    shift_yaml = MakeShiftYamlFile(xShift, yShift, angle, numColors)
                try:                    
                    shift_yaml = yaml.load(yaml_file, Loader=yaml.FullLoader)
                    yaml_file.close()
                    shift_yaml = io.AutoDict(shift_yaml)
                    print('Alternative yaml file loaded: '+yamlFileInImageFolder)
                except:
                    print('Alternative yaml file found but not readable: '+yamlFileInImageFolder)
                    shift_yaml = MakeShiftYamlFile(xShift, yShift, angle, numColors)
            else:
                print('No potential alternative yaml file present.')
                shift_yaml = MakeShiftYamlFile(xShift, yShift, angle, numColors)
    else: # if is doesnt exist, we create all structures from scratch      
        print('Given file does not exist: '+yamlFileName)          
        shift_yaml = MakeShiftYamlFile(xShift, yShift, angle, numColors) 
    return shift_yaml

# ---------------------------------------------------------------------
def MakeShiftYamlFile(xShift, yShift, angle, numColors):
    shift_yaml = {}
    shift_yaml["x"] = {}
    shift_yaml["y"] = {}
    shift_yaml["angle"] = {}
    shift_yaml["numColors"] = numColors
    for nColor in range(numColors):
        shift_yaml["x"]["col"+str(int(nColor))] = xShift
        shift_yaml["y"]["col"+str(int(nColor))] = yShift
        shift_yaml["angle"]["col"+str(int(nColor))] = angle
    return shift_yaml

# ---------------------------------------------------------------
def ShiftROI(coord, dy, dx):

    y = np.array( [x[0] for x in coord] )
    x = np.array( [x[1] for x in coord] )
    return np.array( [[y[i]+dy, x[i]+dx] for i in range(len(coord))] )

# ---------------------------------------------------------------------
def order_points(pts):
    # sorts points clockwise
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl])

# ---------------------------------------------------------------------
def rotatePoint(point, centerPoint, angleDeg):
    angleRad = angleDeg * np.pi/180
    return np.array([(point[0]-centerPoint[0]) * np.cos(angleRad) - (point[1]-centerPoint[1]) * np.sin(angleRad),
    (point[0]-centerPoint[0]) * np.sin(angleRad) + (point[1]-centerPoint[1]) * np.cos(angleRad)]) + centerPoint

# ---------------------------------------------------------------------
def rotateShiftROI(rect, img, imgSize, shift_x, shift_y, angle, fixedShape=None):
    rect = order_points(rect) # toplleft, topright, bottomrrigt, bottomleft

    # compute angle of the ROI which the user defined (not the angle by whcih the image is rotated)
    # angle between rect and surrounding_rect. surrounding_rect is straight. 
    # We measure COUNTERCLOCKWISE since skimage.transform.rotate rotates counter-clockwise
    dy = rect[3][1] - rect[0][1]
    dx = rect[3][0] - rect[0][0]
    if dx == 0 or dy == 0:
        angle_rect_surrounding_rect = 0
    else:
        angle_rect_surrounding_rect = np.arctan(dy/dx) * 180 / np.pi

    # we first rotate the ROI according to the rotation of the image
    if angle != 0:
        centerPoint = np.array(imgSize)/2 
        rect = np.array([
            rotatePoint(rect[0], centerPoint, -angle), 
            rotatePoint(rect[1], centerPoint, -angle), 
            rotatePoint(rect[2], centerPoint, -angle), 
            rotatePoint(rect[3], centerPoint, -angle)
        ])
        

    # plt.figure()
    # plt.imshow(img)
    # # plt.plot(surrounding_rect[:,1], surrounding_rect[:,0])
    # plt.plot(rect[:,1], rect[:,0])
    # # plt.plot(rot_rect[:,1], rot_rect[:,0])                
    # # plt.show()

    # plt.figure()
    # plt.imshow(rotate(img, angle=angle))
    # # plt.plot(surrounding_rect[:,1], surrounding_rect[:,0])
    # plt.plot(rect[:,1], rect[:,0])
    # plt.plot(rot_rect[:,1], rot_rect[:,0])                
    # plt.show()

    
    if (shift_y!=0 or shift_x!=0):
        rect[:,0] = rect[:,0]-shift_y
        rect[:,1] = rect[:,1]-shift_x
    rect = np.round(rect)


    # rect[rect<0] = 0
    # rect[:,0][rect[:,0]>imgSize[0]] = imgSize[0]
    # rect[:,1][rect[:,1]>imgSize[1]] = imgSize[1]
    # if col==0 and i==0: # check again if dimensions are > 0. otherwise skip
    #     if not (np.sqrt(np.sum((rect[3,:]-rect[2,:])**2))>0) and not (np.sqrt(np.sum((rect[2,:]-rect[1,:])**2))>0):
    #         print('Skipping ROI ['+str(iRect+1)+'/'+str(len(rect_shape))+']')
    #         continue

    # obtain min and max values of the surrounding rectangle
    minx = np.min(rect[:,0], axis=0)
    maxx = np.max(rect[:,0], axis=0)
    miny = np.min(rect[:,1], axis=0)
    maxy = np.max(rect[:,1], axis=0)
    surrounding_rect = np.array([  
        [minx, miny], \
        [maxx, miny], \
        [maxx, maxy], \
        [minx, maxy], \
        ]) # topleft, topright, bottomrrigt, bottomleft
        # the centers of the rectangle and surrounding rectangle are the same

    # plt.figure()
    # plt.imshow(img)
    # plt.plot(surrounding_rect[:,1], surrounding_rect[:,0])
    # plt.plot(rect[:,1], rect[:,0])
    # plt.show()

    # now we cropped the surrounding rectangle, let's take away this rotation from the ROI again to leave only with the rotation that the user defined on the ROI itself, not on the image
    # if angle != 0:
    #     centerPoint = np.mean(rect,axis=0) # we rotate the ROI around its center
    #     rect = np.array([
    #         rotatePoint(rect[0], centerPoint, angle), # note that we rotate by -angle above
    #         rotatePoint(rect[1], centerPoint, angle), 
    #         rotatePoint(rect[2], centerPoint, angle), 
    #         rotatePoint(rect[3], centerPoint, angle)
    #     ])

    # plt.figure()
    # plt.imshow(img)
    # plt.plot(surrounding_rect[:,1], surrounding_rect[:,0])
    # plt.plot(rect[:,1], rect[:,0])
    # plt.show()

    # angle between rect and surrounding_rect. surrounding_rect is straight. 
    # We measure COUNTERCLOCKWISE since skimage.transform.rotate rotates counter-clockwise
    dy = rect[3][1] - rect[0][1]
    dx = rect[3][0] - rect[0][0]
    if dx == 0 or dy == 0:
        angle_rect_surrounding_rect = 0
    else:
        angle_rect_surrounding_rect = np.arctan(dy/dx) * 180 / np.pi    

    minx_backup = minx
    miny_backup = miny
    minx_append = 0
    miny_append = 0
    maxx_append = 0
    maxy_append = 0
    if maxx>img.shape[0]:
        if minx>img.shape[0]: # if the ROI is entirely out of the image
            maxx_append = int( maxx-minx )
        else:
            maxx_append =int( maxx-img.shape[0] )
            maxx = img.shape[0]
    if maxy>img.shape[1]:
        if miny>img.shape[1]: # if the ROI is entirely out of the image
            maxy_append = int( maxy-miny )
        else:
            maxy_append = int( maxy-img.shape[1] )
            maxy = img.shape[1]
    if minx<0:
        if maxx<0:
            minx_append = int( maxx-minx )
            minx = img.shape[0]+1
            maxx = minx+minx_append
        else:
            minx_append = int( np.abs(minx) )
            minx = 0
    if miny<0:
        if maxy<0:
            miny_append = int( maxy-miny )
            miny = img.shape[1]+1
            maxy = miny+miny_append
        else:
            miny_append = int( np.abs(miny) )
            miny = 0


    img_surrounding = img[int(minx):int(maxx), int(miny):int(maxy)]
    # rotationCenter  = ((int(maxy)-int(miny)) / 2 - 0.5, (int(maxx)-int(minx)) / 2 - 0.5)
    dtype = np.dtype(img[0][0])
    if minx_append>0:
        img_surrounding = np.concatenate( (np.zeros((minx_append, img_surrounding.shape[1]), dtype=dtype), img_surrounding), axis=0)
    if maxx_append>0:
        img_surrounding = np.concatenate( (img_surrounding, np.zeros((maxx_append, img_surrounding.shape[1]), dtype=dtype)), axis=0)
    if miny_append>0:
        img_surrounding = np.concatenate( (np.zeros((img_surrounding.shape[0], miny_append), dtype=dtype), img_surrounding), axis=1)
    if maxy_append>0:
        img_surrounding = np.concatenate( (img_surrounding, np.zeros((img_surrounding.shape[0], maxy_append), dtype=dtype)), axis=1)

    if angle_rect_surrounding_rect != 0 and angle_rect_surrounding_rect!=360:
        img_surrounding_rot = rotate(img_surrounding, angle=360-angle_rect_surrounding_rect) # we have to rotate the other way as the ROI is                

        rect_straight = rect
        rect_straight[:,0] = rect_straight[:,0]-minx_backup
        rect_straight[:,1] = rect_straight[:,1]-miny_backup
        centerPoint = np.mean(rect_straight,axis=0) # we rotate the ROI around its center
        rect_straight = np.array([
            rotatePoint(rect_straight[0], centerPoint, -angle_rect_surrounding_rect), # now the rectangle is straight again
            rotatePoint(rect_straight[1], centerPoint, -angle_rect_surrounding_rect), 
            rotatePoint(rect_straight[2], centerPoint, -angle_rect_surrounding_rect), 
            rotatePoint(rect_straight[3], centerPoint, -angle_rect_surrounding_rect)
        ])


        # plt.figure()
        # plt.imshow(img_surrounding_rot)
        # plt.plot(rect_straight[:,1], rect_straight[:,0])
        # plt.show()
        tranposeImage = False
        minx = np.min(rect_straight[:,0], axis=0)
        maxx = np.max(rect_straight[:,0], axis=0)
        miny = np.min(rect_straight[:,1], axis=0)
        maxy = np.max(rect_straight[:,1], axis=0)
        if minx<0:
            minx = 0
        if miny<0:
            miny = 0
        if maxx>img_surrounding_rot.shape[0]:
            maxx = img_surrounding_rot.shape[0]
        if maxy>img_surrounding_rot.shape[1]:
            maxy = img_surrounding_rot.shape[1]
        if fixedShape is not None:
            currentShape = np.array([int(maxx)-int(minx), int(maxy)-int(miny)])
            if (currentShape != fixedShape).all():
                if sum(np.abs(currentShape-fixedShape)) > sum(np.abs(currentShape[-1::-1]-fixedShape)):
                    fixedShape = fixedShape[-1::-1]
                    tranposeImage = True
                
                maxx = minx + fixedShape[0]
                maxy = miny + fixedShape[1]
                maxx = int(maxx)
                maxy = int(maxy)
                if img_surrounding_rot.shape[0]<maxx: # if the image is too small, extend it with zeros
                    img_surrounding_rot = np.concatenate((img_surrounding_rot, \
                        np.zeros((maxx-img_surrounding_rot.shape[0], img_surrounding_rot.shape[1]))), axis=0)
                if img_surrounding_rot.shape[1]<maxy: # if the image is too small, extend it with zeros
                    img_surrounding_rot = np.concatenate((img_surrounding_rot, \
                        np.zeros((img_surrounding_rot.shape[0], maxy-img_surrounding_rot.shape[1]))), axis=1)
        img_crop = img_surrounding_rot[int(minx):int(maxx), int(miny):int(maxy)]
        if tranposeImage:
            img_crop = np.transpose(img_crop)
        
        # plt.figure()
        # plt.imshow(img_crop)
        # plt.show()
        # plt.close("all")
    else:
        img_crop = img_surrounding
    if fixedShape is not None:
        if (img_crop.shape != fixedShape):
            a=1
    if (img_crop.shape[0]==0) or (img_crop.shape[1]==0):
        a=1
    return sk.util.img_as_uint(img_crop)

# ---------------------------------------------------------------------
def crop_rect_shapes(image_meta, shape_layers, dir_to_save=None,
                     frame_start=None, frame_end=None,
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
    img_array_all    = {}
    for col in range(image_meta['num_colors']):
        imgseq = pims.ImageSequence(image_meta['filenames_color_'+str(col)])[frame_start:frame_end]
        for i in trange(num_frames_update, desc='Cropping color: {} '.format(col)):
            # load image
            img = np.array(imgseq[i], dtype=np.uint16)
            imgSize = img.shape

            ## first determine if all crops are valid or if some lie outside the image region after applying all shifts/rotations
            if i == 0:
                numRectanglesOutside = 0
                for col2 in range(image_meta['num_colors']):
                    for iRect in range(len(rect_shape)):
                        # rectangle shape
                        rect = np.array(rect_shape[iRect])
                        rect = order_points(rect) # toplleft, topright, bottomrrigt, bottomleft                        

                        if angle[col] != 0:
                            centerPoint = np.array(imgSize)/2 # np.array([0, 0])#
                            rect = np.array([
                                rotatePoint(rect[0], centerPoint, -angle[col]), 
                                rotatePoint(rect[1], centerPoint, -angle[col]), 
                                rotatePoint(rect[2], centerPoint, -angle[col]), 
                                rotatePoint(rect[3], centerPoint, -angle[col])
                            ])

                        if (shift_y[col]!=0 or shift_x[col]!=0):
                            rect[:,0] = rect[:,0]-shift_y[col]
                            rect[:,1] = rect[:,1]-shift_x[col]
                        rect = np.round(rect)
            
            ## all crops are valid, thus proceed
            for iRect in range(len(rect_shape)):
                # rectangle shape
                rect = np.array(rect_shape[iRect])
                # for the first image, we'll do it for all colors and see if the shapes for all colors are the same
                # in order to add them all to the same array later
                if i==0:
                    for col2 in range(image_meta['num_colors']):
                        if col2 == 0:                            
                            img_crop = rotateShiftROI(rect, img, imgSize, shift_x[col2], shift_y[col2], angle[col2])
                            img_crop_shape = img_crop.shape
                        else:
                            img_crop = rotateShiftROI(rect, img, imgSize, shift_x[col2], shift_y[col2], angle[col2], fixedShape=img_crop_shape)

                    # construct names of ROIs to save
                    rect = order_points(rect) # toplleft, topright, bottomrrigt, bottomleft
                    rect_0 = rect[0].astype(int)
                        
                    dy = rect[3][1] - rect[0][1]
                    dx = rect[3][0] - rect[0][0]
                    if dx == 0 or dy == 0:
                        angle_roi = int(0)
                    else:
                        angle_roi = int(np.arctan(dy/dx) * 180 / np.pi)
                    width_x = int( np.ceil( np.sqrt(np.sum((rect[3,:]-rect[2,:])**2)) ) )
                    height_y = int( np.ceil( np.sqrt(np.sum((rect[2,:]-rect[1,:])**2)) ) )
                    nam = 'x' + str(rect_0[0]) + '-y' + str(rect_0[1]) +\
                          '-l' + str(height_y) + '-w' + str(width_x) +\
                          '-a' + str(angle_roi) + labels[i].lower()
                    names_roi_tosave.append(nam) # without the -f flags
                
                key = 'arr' + str(iRect)
                if col == 0:
                    img_crop = rotateShiftROI(rect, img, imgSize, shift_x[col], shift_y[col], angle[col])
                else:
                    img_crop = rotateShiftROI(rect, img, imgSize, shift_x[col], shift_y[col], angle[col], fixedShape=img_array_all[key].shape[2:4])


                # initialize the array holding all crops based on the size of the just cropped crop                
                if col==0 and i==0:                    
                    img_array_all[key] = np.zeros((num_frames_update, image_meta['num_colors'],
                                                img_crop.shape[0], img_crop.shape[1]),
                                                dtype=np.uint16) 

                # write crop into img_array_all
                img_array_all[key][i, col, :, :] = img_crop

    # save ROIs and associated yaml files
    rect_keys = list(img_array_all.keys())   
    yamlFileName = os.path.join(folderpath, 'shift.yaml')                           
    shift_yaml = LoadShiftYamlFile(yamlFileName, shift_x, shift_y, angle, image_meta['num_colors'])
    shift_yaml["numColors"] = image_meta['num_colors']
    for nColor in range(image_meta['num_colors']):
        shift_yaml["x"]["col"+str(int(nColor))] = shift_x[nColor]
        shift_yaml["y"]["col"+str(int(nColor))] = shift_y[nColor]
        shift_yaml["angle"]["col"+str(int(nColor))] = angle[nColor]
    # save the yaml file
    shift_yaml = io.to_dict_walk(shift_yaml)
    for i in range(len(rect_shape)):
        roi_ij = ImagejRoi.frompoints(rect_shape_to_roi(rect_shape[i]))
        roi_ij.tofile(os.path.join(dir_to_save, names_roi_tosave[i]+'.roi'))
        imwrite(os.path.join(dir_to_save, names_roi_tosave[i]+'.tif'),
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
def daskread_img_seq(num_colors=1, bkg_subtraction=False, mean_subtraction=False, path="", RotationAngle=None):
    '''
    Import image sequences (saved individually in a folder)
    num_colors : integer
    returns: dask arrays and 
    # 20210330: mean subtraction currently without function. The reason is that the mean image has to be computed over 
    every n'th image only for n colors, while dask arrays only allow the same operation on each element in the array
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

    # background subtraction
    if bkg_subtraction:
        stack = stack.map_blocks(bkgSubtraction)
    # if mean_subtraction:
    #     stackMean = stack.mean(axis=0).persist()
    #     stack = stack - stackMean
        # stack.map_blocks(meanSubtraction)
   
    # apply rotation. For that, see if we can load an already existing yaml file from which we read the previously applied rotation angle
    if len(folderpath)==0:
        yamlFileName = ''
    else:
        yamlFileName = os.path.join(folderpath, 'shift.yaml')
    shift_yaml = LoadShiftYamlFile(yamlFileName, 0, 0, 0, num_colors)
    RotationAngle = [0] * shift_yaml["numColors"]
    for nColor in range(shift_yaml["numColors"]):
        try:
            RotationAngle[nColor] = shift_yaml["angle"]["col"+str(int(nColor))]
        except:
            RotationAngle[nColor] = 0
    if (RotationAngle[:-1] != RotationAngle[1:]):
        print('Not all read rotation angles are equal. Not applying any rotation.')
        RotationAngle = None

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
                sample = bkgSubtraction(sample)
            # if mean_subtraction:
            #     sample -= sample
            minVal.append(sample.min())
            maxVal.append(sample.max())            
            minConTemp, maxConTemp = AutoAdjustContrastIJ(sample)
            minCon.append(minConTemp)
            maxCon.append(maxConTemp)
        image_meta['min_int_color_'+str(i)] = np.mean(minVal)
        image_meta['max_int_color_'+str(i)] = np.mean(maxVal)        
        image_meta['min_contrast_color_'+str(i)] = np.mean(minCon)#AutoAdjustContrastSorting(sample)#
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
def AutoAdjustContrastIJ(img):
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
def invnormcdf(x, mu, sigma):
    # noise model: mu is the mean value, sigma is the sigma of the gaussian noise
    return np.real(mu+sigma*np.sqrt(2)*special.erfinv(x))

# ---------------------------------------------------------------------
def AutoAdjustContrastSorting(img):
    img = img.astype('float')
    size = img.shape
    img_bckg = bkgSubtraction(img)
    img_bckg_flat = img_bckg.flatten()
    # get rid of zeros, that messes with the fitting once the intensity values are sorted
    if (img_bckg_flat==0).any():
        img_bckg_flat[img_bckg_flat==0] = np.nan
        sumNaNs = np.sum(np.isnan(img_bckg_flat))
        N = size[0]*size[1] - sumNaNs
    else:
        sumNaNs = 0
        N = size[0]*size[1]
    x = np.linspace(-0.999, 0.999, N)

    # sort intensity values
    sort_ind = np.argsort(img_bckg_flat)
    img_bckg_sorted = img_bckg_flat[sort_ind]
    
    # if there are nan's, take them away now
    if sumNaNs > 0:
        img_bckg_sorted = img_bckg_sorted[~np.isnan(img_bckg_sorted)]
    
    # subtract mean value and normalize data
    img_bckg_sorted -= np.mean(img_bckg_sorted)
    img_bckg_sorted /= max(img_bckg_sorted)
    
    # fit to noise model
    initial_guess = np.array([0, 0.2])
    LB = np.array([-0.2, 0.001])
    UB = np.array([0.2, 0.35])
    
    popt, pcov = optimize.curve_fit(invnormcdf, x[0:int(np.round(N/2))], \
        img_bckg_sorted[0:int(np.round(N/2))], p0=initial_guess, bounds=(LB, UB))

    noise_subtracted = img_bckg_sorted - invnormcdf(x, popt[0], popt[1])
    noise_subtracted /= np.max(noise_subtracted.flatten())
    
    ind = np.argmax(noise_subtracted > 0.01)
    sorted_im = np.sort(img.flatten())
    threshold = sorted_im[ind]

    # from matplotlib import pyplot as plt
    # plt.figure()
    # plt.plot(x, img_bckg_sorted)
    # plt.plot(x, invnormcdf(x, popt[0], popt[1]))
    # # plt.show()
    # plt.savefig('foo.png')

    return threshold

# ---------------------------------------------------------------------
from scipy.ndimage import white_tophat, black_tophat
def bkgSubtraction(txy_array, size_bgs=50, light_bg=False):
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
# def meanSubtraction(txy_array):
#     array_processed = np.zeros_like(txy_array)
#     if array_processed.ndim>2:
#         for i in range(txy_array.shape[0]):
#             img = txy_array[i]
#             if light_bg:
#                 img_bgs = black_tophat(img, size=size_bgs)
#             else:
#                 img_bgs = white_tophat(img, size=size_bgs)
#             array_processed[i, :, :] = img_bgs
#     else:
#         if light_bg:
#             array_processed = black_tophat(txy_array, size=size_bgs)
#         else:
#             array_processed = white_tophat(txy_array, size=size_bgs)
#     return array_processed
# def meanSubtraction(txy_array):
#     array_processed = np.zeros_like(txy_array)
#     if array_processed.ndim>2:
#         # meanImage = np.nanmean(txy_array, axis=0, keepdims=True)
#         meanImage = sum(txy_array) / len(txy_array)
#         if np.isnan(meanImage.flatten()).any():
#             return txy_array
#         # plt.figure(), plt.imshow(meanImage), plt.show()
#         for i in range(txy_array.shape[0]):
#             array_processed[i, :, :] = txy_array[i] - meanImage
#     # for i in range(txy_array.shape[0]):
#         # array_processed = txy_array - meanImage
#         # array_processed[array_processed<0] = 0
#         print('shape: ')
#         print(txy_array.shape)
#         print('meanImage: ')
#         print(meanImage)
#         asdkjgasd
#         print(np.nanmean(meanImage.flatten()))
#     return array_processed

# ---------------------------------------------------------------------
def geometric_shift(image2d, angle, shift_x, shift_y):
    if angle!=0:
        image2d = ndimage.rotate(image2d, angle, reshape=False)
    if float(shift_y)!=0 or float(shift_x)!=0:
        image2d = ndimage.shift(image2d, (float(shift_y), float(shift_x)))
    return image2d