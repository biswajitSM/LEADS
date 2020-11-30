import os
import sys
import glob
import numpy as np
from pathlib import Path
import shutil
from scipy import ndimage, misc
import skimage as sk
from skimage.transform import rotate
from tifffile import imread, imsave, imwrite
from roifile import ImagejRoi
import pims
from tqdm import trange
from . import io
from PyQt5.QtWidgets import QApplication
from leads.crop_images import get_rect_params, crop_rect, rect_shape_to_roi
import time

# ---------------------------------------------------------------
def FindAllSubdirectories(dirname, count=0):
    subfolders = [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        count += 1
        print('Found '+str(count)+' subdirectories...', end="\r", flush=True)
        subfolders.extend(FindAllSubdirectories(dirname, count))
    return subfolders

# ---------------------------------------------------------------
def IdentifyRelevantDirectories(directory, num_colors=None):
    
    # scan directory for folders
    sub_dirs = FindAllSubdirectories(directory)
    sub_dirs.append(directory) 
    print()

    # check if there are any ambiguities we have to ask the user about
    # basename_list = []
    sub_dirs_updated = []
    for i in trange(len(sub_dirs), desc='Check for ROI files'):
        roi_file_list = glob.glob(sub_dirs[i] + '/*.roi', recursive = False)

         # only consider if there are any roi files and if these roi files do 
         # not contain two "-f" in their name (those are the processed ones)
        if roi_file_list:
            if (not any(roi_file_list_item.count('-f')>1 for roi_file_list_item in roi_file_list)):
                sub_dirs_updated.append(sub_dirs[i])
    sub_dirs = sub_dirs_updated

    # if there is a choice to be made                
    if len(sub_dirs)>1:
        print("Found more than one potential folder to crop images from. Which one to take?")
        for i in range(len(sub_dirs)):
            print("[", i, "] ", sub_dirs[i])
        choice = [int(item) for item in input("Enter the list items (separated by blank, type -1 for all): ").split()]
        if choice[0]==-1:
            choice = range(0, len(sub_dirs))
        sub_dirs = [sub_dirs[i] for i in choice]
    elif not sub_dirs:
        print('No directories found. Try again.')
        crop_from_directory(directory=None, num_colors=num_colors)
        
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
    
    print()
    print("Cropping from the following folders: ")
    for j in range(len(sub_dirs)):
        print("- ", sub_dirs[j])
    print()
        
    return sub_dirs

# ---------------------------------------------------------------
def GetROIDescription(roi_file_list, descriptions=[]):
    # the format is something like xxxxxxxx_descrip_ti_on.roi
    # find from the underscore to the dot to get the description and convert to small letters
    for k in range(len(roi_file_list)):      
        split_list = os.path.basename( roi_file_list[k] ).rsplit("_") # we only want the filename
        merged_list = ""
        for l in range(1, len(split_list)):
            merged_list += split_list[l]
            if l < len(split_list)-1:
                merged_list += '_'
        descriptions.append( merged_list.split(".")[-2].lower() )
    return descriptions

# ---------------------------------------------------------------
def crop(dir, sub_dir, roi_coord_list, roi_file_list, roi_original_names, 
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
        rect_params = get_rect_params(rect) 
        key = 'arr' + str(i)
        img_array_all[key] = np.zeros((round(num_frames_update/num_colors), num_colors,
                                       rect_params['width'], rect_params['length']),
                                       dtype=np.uint16)        
        # name = pix_x-pix_y-length-width-angle-frame_start-frame_end
        rect_0 = rect[0].astype(int)
        # if i < 10: sl_no = str(0) + str(i)
        # else: sl_no = str(i)
        nam = 'x' + str(rect_0[0]) + '-y' + str(rect_0[1]) +\
              '-l' + str(rect_params['length']) + '-w' + str(rect_params['width']) +\
              '-a' + str(rect_params['angle']) + 'd-f' + str(frame_start) + '-f' +\
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
    if sort_FOVs:
        sorted_dir = []
        for i in range(len(roi_coord_list)):
            child  = Path(roi_file_list[i]).parent
            parent = child.parent.parts[-1]
            child = child.parts[-1]
            # create a dir to save the FOV to if it doesnt exist yet
            sorted_dir.append( os.path.join(sort_directory, ROIdescriptions[i]) )
            if not os.path.isdir(sorted_dir[-1]):
                os.makedirs(sorted_dir[-1])
            name_sorted.append( 
                os.path.join(sorted_dir[-1],\
                parent + '_' + child + '_' +\
                names_roi_tosave_no_frames[i]) )

    # go through each roi name and see if it already exists. If yes, skip it
    skipROI  = []
    skipIMG  = []
    skipSORT = []
    for i in range(len(roi_coord_list)):
        ROIpath = os.path.join(dir_to_save, names_roi_tosave[i]+'.roi')
        IMGpath = os.path.join(dir_to_save, names_roi_tosave[i]+'.tif')
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
                roi_ij = ImagejRoi.frompoints(rect_shape_to_roi(roi_coord_list[i]))
                roi_ij.tofile(os.path.join(dir_to_save, names_roi_tosave[i]+'.roi')) # saving the ROI in the subfolder
                roi_ij.tofile(os.path.join(dir, names_roi_tosave_no_frames[i]+'.roi')) # saving the ROI in the folder where the ROI was found but with new name
                os.remove(os.path.join(dir, roi_original_names[i])) # remove the original file
            if (not skipSORT[i]):
                roi_ij = ImagejRoi.frompoints(rect_shape_to_roi(roi_coord_list[i]))
                roi_ij.tofile(name_sorted[i]+'.roi') # saving the ROI in the sorted dir
                # copying the images from the dir to the sorted one
                shutil.copy2(os.path.join(dir_to_save, names_roi_tosave[i]+'.tif'),\
                     name_sorted[i]+'.tif') # copy the file to the sorted dir
        return

    #imgseq = imgseq[frame_start:frame_end]
    for col in range(num_colors):        
        
        for i in trange(round(num_frames_update/num_colors), desc='Cropping color '+str(col+1)+'/'+str(num_colors)+' in subfolder '+str(nDir+1)+'/'+str(numDirs)+'...'):
            frame_index = i*num_colors+col
            if num_colors>1:
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
            roi_ij.tofile(os.path.join(dir_to_save, names_roi_tosave[i]+'.roi')) # saving the ROI in the subfolder
            roi_ij.tofile(os.path.join(dir, names_roi_tosave_no_frames[i]+'.roi')) # saving the ROI in the folder where the ROI was found but with new name
            os.remove(os.path.join(dir,roi_original_names[i])) # remove the original file        
        if (not skipIMG[i]):
            imwrite(os.path.join(dir_to_save, names_roi_tosave[i]+'.tif'),
                    img_array_all[rect_keys[i]], imagej=True,
                    metadata={'axis': 'TCYX', 'channels': num_colors,
                    'mode': 'composite',})
        if (not skipSORT[i]):
                roi_ij.tofile(name_sorted[i]+'.roi') # saving the ROI in the sorted dir
                shutil.copy2(os.path.join(dir_to_save, names_roi_tosave[i]+'.tif'),\
                     name_sorted[i]+'.tif') # copy the file to the sorted dir

    
# ---------------------------------------------------------------
def crop_from_directory(directory=None, num_colors=None, sort_FOVs=None):
    '''
    crops from all the subfolders with images and corresponding ROIs
    FOVs can be sorted by their description into another folder while keeping their name and source path
    '''
    if directory is None:
        directory = ChooseDirectory_Crop()
        if not os.path.isdir(directory):
                os.makedirs(directory)
    
    if num_colors is None: 
        num_colors = inputNumber("How many colors? ")

    sort_directory = None
    if sort_FOVs is None: 
        sort_FOVs = inputBool("Sort FOVs by description [0 or 1]? ")
        if sort_FOVs:
            sort_directory = ChooseDirectory_Sort()
            if not os.path.isdir(sort_directory):
                    os.makedirs(sort_directory)

    # Get relevant subdirectories
    # sub_dirs, tags = IdentifyRelevantDirectories(directory, num_colors)
    sub_dirs = IdentifyRelevantDirectories(directory, num_colors)

    # go through each sub-directory and get all .roi files
    print('Starting cropping routine...')
    for i in range(len(sub_dirs)):
        roi_file_list = glob.glob(sub_dirs[i] + "/*.roi", recursive = False)
        ROIdescriptions = GetROIDescription(roi_file_list)
        roi_coord_list, roi_original_names = LoadROIs(roi_file_list)

        # there might be several subfolders to which this ROI is to be applied. 
        # Get all of those subfolders 
        # (but only the ones which are exactly one level lower in the hierarchy and
        # which contain .tif images)
        subfolders_temp = [f.path for f in os.scandir(sub_dirs[i]) if f.is_dir()]
        subfolders = []
        for subfolder in subfolders_temp:
            files         = glob.glob(subfolder + '/*.tif', recursive=False) # look for tif files
            roi_subfolder = glob.glob(subfolder + "/*.roi", recursive = False) # look for roi files in this subfolder. These are likely to be the prcessed ones already
            if len(files)>0 and (not any(roi_subfolder_item.count('-f')>1 for roi_subfolder_item in roi_subfolder)):
                if not subfolders: # if subfolders is still empty (=first iteration)
                    print('[', str(i+1), '/', str(len(sub_dirs)), '] ', 
                    'Applying ROIs in "', sub_dirs[i], '" to:')
                subfolders.append(subfolder)
                print('- ', subfolder)

        for sf in range(len(subfolders)):
            crop(sub_dirs[i], subfolders[sf], roi_coord_list, roi_file_list, roi_original_names,
            ROIdescriptions, sort_directory, num_colors, sort_FOVs, sf, len(subfolders))
        print()
        
    print('Cropping is finished.')

# ---------------------------------------------------------------
def LoadROIs(roi_file_list):
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
def ChooseDirectory_Crop():
    settings = io.load_user_settings()
    try:
        directory = settings["crop"]["PWD BATCH"]
    except:
        directory = None
        pass
    _ = QApplication([])
    directory = io.FileDialog(directory, 'Please select a directory to crop from').openDirectoryDialog()
    if not directory:
        sys.exit("No direcotry specified. Exiting.")
    print('Directory to crop from: '+directory+'\n')
    settings["crop"]["PWD BATCH"] = directory
    io.save_user_settings(settings)
    return directory

# ---------------------------------------------------------------
def ChooseDirectory_Sort():
    settings = io.load_user_settings()
    try:
        directory = settings["crop"]["PWD BATCH SORT"]
    except:
        directory = None
        pass
    _ = QApplication([])
    directory = io.FileDialog(directory, 'Please select a directory to store sorted FOVs').openDirectoryDialog()
    if not directory:
        sys.exit("No direcotry specified. Exiting.")
    print('Directory to save sorted FOVs: '+directory+'\n')
    settings["crop"]["PWD BATCH SORT"] = directory
    io.save_user_settings(settings)
    return directory
    
# ---------------------------------------------------------------
def inputNumber(message):
  while True:
    try:
       userInput = int(input(message))       
    except ValueError:
       print("Not an integer! Try again.")
       continue
    else:
       return userInput 
       break 

# ---------------------------------------------------------------
def inputBool(message):
  while True:
    try:
       userInput = int(input(message))       
    except ValueError:
       print("Not an integer! Write either 0 or 1.")
       continue
    else:
        if userInput>0:
            userInput = True
        else:
            userInput = False
        return userInput 
        break 

# ---------------------------------------------------------------    
if __name__ == "__main__":
    crop_from_directory()