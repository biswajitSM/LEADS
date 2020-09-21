import os
import glob
import numpy as np
from scipy import ndimage, misc
import skimage as sk
from skimage.transform import rotate
from tifffile import imread, imsave, imwrite
from roifile import ImagejRoi
# import PySimpleGUI as sg
import pims
from tqdm import trange
from . import io
from PyQt5.QtWidgets import QApplication
import time

# ---------------------------------------------------------------
def FindAllSubdirectories(dirname, count=0):
    subfolders= [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        count += 1
        print('Found '+str(count)+' subdirectories.', end="\r", flush=True)
        subfolders.extend(FindAllSubdirectories(dirname, count))
    return subfolders

# ---------------------------------------------------------------
def IdentifyRelevantDirectories(directory):
    # scan directory for folders
    sub_dirs = FindAllSubdirectories(directory)
    print()

    # check if there are any ambiguities we have to ask the user about
    name_list = []
    sub_dirs_updated = []
    for i in trange(len(sub_dirs), desc='Check for ambiguities'):
        roi_file_list = glob.glob(sub_dirs[i] + "/**/*.roi", recursive = True)

        if roi_file_list: # only consider if there are any roi files
            sub_dirs_updated.append(sub_dirs[i])
            subfolders = [f.path for f in os.scandir(sub_dirs[i]) if f.is_dir()]
            for j in range(len(subfolders)):
                name = os.path.basename(subfolders[j])
                if (not("analysis" in name) and (name not in name_list)):
                    name_list.append(name)
    sub_dirs = sub_dirs_updated

    # if there is a choice to be made                
    if len(name_list)>1:
        print("Found more than one potential folder to crop images from. Which one to take?")
        for i in range(len(name_list)):
            print("[", i, "] ", name_list[i])
        choice = int(input())
        name_list = name_list[choice]

        # remove all directories which don't fit in the user selected scheme
        sub_dirs_updated = []
        for i in trange(len(sub_dirs), desc='Remove unselected subdirectories'):
            subfolders = [f.path for f in os.scandir(sub_dirs[i]) if f.is_dir()]
            for j in range(len(subfolders)):
                name = os.path.basename(subfolders[j])
                if (name_list in name) and (sub_dirs[i] not in sub_dirs_updated):
                    sub_dirs_updated.append(sub_dirs[i])
        sub_dirs = sub_dirs_updated    
    else:
        name_list = name_list[0]
        
    # remove all subdirectories which contain a roi file which has 2x "-f" in the name since those are files which are saved together with the crop    
    sub_dirs_updated = []
    for i in trange(len(sub_dirs), desc='Remove already processed ROIs'):
        subfolders = [f.path for f in os.scandir(sub_dirs[i]) if f.is_dir()]
        for j in range(len(subfolders)):
            roi_file_list = glob.glob(sub_dirs[i] + "/**/*.roi", recursive = True)
            for k in range(len(roi_file_list)):
                if (roi_file_list[k].count('-f')<2) and (sub_dirs[i] not in sub_dirs_updated):
                    sub_dirs_updated.append(sub_dirs[i])
    sub_dirs = sub_dirs_updated
    
    print("Cropping from folder", name_list, ": Found", len(sub_dirs), "director(y/ies).")
    return sub_dirs, name_list

# ---------------------------------------------------------------
def get_rect_params(rect, printing=False):
    length = int(np.sqrt((rect[0][0] - rect[3][0])**2 + (rect[0][1] - rect[3][1])**2))
    width = int(np.sqrt((rect[0][0] - rect[1][0])**2 + (rect[0][1] - rect[1][1])**2))
    dy = rect[3][1] - rect[0][1]
    dx = rect[1][0] - rect[0][0]
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

# ---------------------------------------------------------------
def crop_rect(img, rect):
    # rect : array for rectanglular roi as in napari
    rect_params = get_rect_params(rect)
    if rect_params['angle'] != 0:
        img_rot = rotate(img, angle=rect_params['angle'],
                         center=(rect_params['y_cent'], rect_params['x_cent']))
    else:
        img_rot = img
    x = int(rect_params['x_cent'] - rect_params['length']/2)
    y = int(rect_params['y_cent'] - rect_params['width']/2)
    img_croped = img_rot[x:x+rect_params['length'], y:y+rect_params['width']]
    return sk.util.img_as_uint(img_croped)

# ---------------------------------------------------------------
def rect_shape_to_roi(arr_from_napari):
    arr_from_napari = np.flip(arr_from_napari)
    roi_coord = np.array([arr_from_napari[0], arr_from_napari[3],
                          arr_from_napari[2], arr_from_napari[1]], dtype=np.float32)
    return roi_coord

# ---------------------------------------------------------------
def crop(sub_dir, tag, roi_coord_list, num_colors, nDir, numDirs):
    
    folderpath = os.path.join(sub_dir+os.path.sep+tag)
    
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
    img_array_all = {}
    for i in range(len(roi_coord_list)):  
        rect = roi_coord_list[i]
        rect_params = get_rect_params(rect) 
        key = 'arr' + str(i)
        img_array_all[key] = np.zeros((round(num_frames_update/num_colors), num_colors,
                                       rect_params['length'], rect_params['width']),
                                       dtype=np.uint16)        
        # name = pix_x-pix_y-length-width-angle-frame_start-frame_end
        rect_0 = rect[2].astype(int)
        if i < 10: sl_no = str(0) + str(i)
        else: sl_no = str(i)
        nam = 'x' + str(rect_0[0]) + '-y' + str(rect_0[1]) +\
              '-l' + str(rect_params['width']) + '-w' + str(rect_params['length']) +\
              '-a' + str(rect_params['angle']) + 'd-f' + str(frame_start) + '-f' +\
              str(frame_end)
        names_roi_tosave.append(nam)
    rect_keys = list(img_array_all.keys())

    # go through each roi name and see if it already exists. If yes, skip it
    skipROI = []
    skipIMG = []
    for i in range(len(roi_coord_list)):
        ROIpath = os.path.join(dir_to_save, names_roi_tosave[i]+'.roi')
        IMGpath = os.path.join(dir_to_save, names_roi_tosave[i]+'.tif')
        if os.path.isfile(ROIpath):
            skipROI.append(True)
        else:
            skipROI.append(False)
        if os.path.isfile(IMGpath):
            skipIMG.append(True)
        else:
            skipIMG.append(False)
    if all(skipIMG):
        print('All crops already exist. Returning.')
        for i in range(len(roi_coord_list)):
            if (not skipROI[i]):
                roi_ij = ImagejRoi.frompoints(rect_shape_to_roi(roi_coord_list[i]))
                roi_ij.tofile(os.path.join(dir_to_save, names_roi_tosave[i]+'.roi'))
        return


    #imgseq = imgseq[frame_start:frame_end]
    for col in range(num_colors):        
        
        for i in trange(round(num_frames_update/num_colors), desc='Cropping color '+str(col+1)+'/'+str(num_colors)+' for directory '+str(nDir+1)+'/'+str(numDirs)+'...'):
            frame_index = i*num_colors+col
            if col==0:
                frame_index += 1
            elif col==1:
                frame_index -= 1
            img = np.array(imgseq[frame_index], dtype=np.uint16)
            for j in range(len(roi_coord_list)):
                if (not skipIMG[j]):
                    img_croped = crop_rect(img, roi_coord_list[j])
                    img_array_all[rect_keys[j]][i, col, :, :] = img_croped            
    
    for i in range(len(roi_coord_list)):
        if (not skipROI[i]):
            roi_ij = ImagejRoi.frompoints(rect_shape_to_roi(roi_coord_list[i]))
            roi_ij.tofile(os.path.join(dir_to_save, names_roi_tosave[i]+'.roi'))
        if (not skipIMG[i]):
            imwrite(os.path.join(dir_to_save, names_roi_tosave[i]+'.tif'),
                    img_array_all[rect_keys[i]], imagej=True,
                    metadata={'axis': 'TCYX', 'channels': num_colors,
                    'mode': 'composite',})
    

def crop_from_directory(directory=None):
    '''
    crops from all the subfolders with images and corresponding ROIs
    '''
    if directory is None:
        settings = io.load_user_settings()
        try:
            directory = settings["crop"]["PWD BATCH"]
        except:
            directory = None
            pass
        _ = QApplication([])
        directory = io.FileDialog(directory, 'Please select a directory').openDirectoryDialog()
        print('Directory:\n'+directory+'\n...')
        settings["crop"]["PWD BATCH"] = directory
        io.save_user_settings(settings)
    # with open('.tkinter_lastdir', 'w') as f: f.write(directory)

    num_colors = 2

    # Get relevant subdirectories
    sub_dirs, tag = IdentifyRelevantDirectories(directory)

    # go through each sub-directory and get all .roi files
    for i in range(len(sub_dirs)):
        roi_file_list = glob.glob(sub_dirs[i] + "/**/*.roi", recursive = True)
        
        roi_file_list_updated = []
        for j in range(len(roi_file_list)):    
            if (roi_file_list[j].count('-f')<2):
                roi_file_list_updated.append(roi_file_list[j])
        roi_file_list = roi_file_list_updated
        
        roi_coord_list = []
        for roi_file in roi_file_list:
            roi = ImagejRoi.fromfile(roi_file)
            roi_coord = np.flip(roi.coordinates())
            roi_coord[roi_coord<0] = 1
            roi_coord_list.append(roi_coord)
            
        print('Cropping in '+sub_dirs[i]+os.path.sep+tag)
        crop(sub_dirs[i], tag, roi_coord_list, num_colors, i, len(sub_dirs))
    print('Cropping is finished.')

if __name__ == "__main__":
    crop_from_directory()