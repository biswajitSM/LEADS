import os
import time
import numpy as np
from tifffile import imwrite, imread
from leads.kymograph import read_img_stack, median_bkg_substration
from leads import io
from PyQt5.QtWidgets import QApplication

# _ = QApplication([])
# directory = io.FileDialog(None, 'Please select a directory').openDirectoryDialog()

def get_file_list(directory, extension=".tif", exclude_ext="_processed.tif"):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension) and not file.endswith(exclude_ext):
                fpath = os.path.join(root, file)
                file_list.append(fpath)
    return file_list


def process_directory(directory, rewrite=False):
    time_start = time.time()
    file_list = get_file_list(directory)
    file_list
    for tif_stack_path in file_list:
        out_tif_stack_path = tif_stack_path[:-4]+"_processed.tif"
        if not os.path.isfile(out_tif_stack_path) or rewrite:
            print(out_tif_stack_path)
            # process_tifstack(tif_stack_path)
            median_tifstack(tif_stack_path)
    print(f"total time took for the directory to predict : {time.time() - time_start} secs\n")
    return

def process_tifstack(fpath):
    time_start = time.time()
    outputdir = os.path.dirname(fpath)
    base_filename = os.path.basename(fpath)
    fpath_processed = os.path.join(outputdir, base_filename[:-4]+"_processed.tif")
    image_meta = read_img_stack(fpath)
    if image_meta["num_colors"] == 1:
        col_0 = median_bkg_substration(image_meta["img_arr_color_0"])
        comb_arr = median_bkg_substration(col_0)
    elif image_meta["num_colors"] == 2:
        col_0 = median_bkg_substration(image_meta["img_arr_color_0"])
        col_1 = median_bkg_substration(image_meta["img_arr_color_1"])
        col_0 = median_bkg_substration(col_0)
        col_1 = median_bkg_substration(col_1)
        comb_arr = np.concatenate((col_0[:,np.newaxis,:,:],
                                    col_1[:,np.newaxis,:,:]),
                                    axis=1)
    elif image_meta["num_colors"] == 3:
        col_0 = median_bkg_substration(image_meta["img_arr_color_0"])
        col_1 = median_bkg_substration(image_meta["img_arr_color_1"])
        col_2 = median_bkg_substration(image_meta["img_arr_color_2"])
        col_0 = median_bkg_substration(col_0)
        col_1 = median_bkg_substration(col_1)
        col_2 = median_bkg_substration(col_2)
        comb_arr = np.concatenate((col_0[:,np.newaxis,:,:],
                                    col_1[:,np.newaxis,:,:],
                                    col_2[:,np.newaxis,:,:]),
                                    axis=1)
    imwrite(fpath_processed, comb_arr, imagej=True,
            metadata={'axis': 'TCYX', 'channels': image_meta["num_colors"],
            'mode': 'composite',})
    # print("processed: " + fpath_processed)
    print(f"time took for {base_filename} to process : {time.time() - time_start} secs\n")

def median_tifstack(tif_stack_path):
    '''
    filters a tifstack file with dimension 3 or 4;
    and save the processed file with an extension '{filename}_processed.tif'
    '''
    outputdir = os.path.dirname(tif_stack_path)
    base_filename = os.path.basename(tif_stack_path)
    fpath_processed = os.path.join(outputdir, base_filename[:-4]+"_processed.tif")
    img_arr = imread(tif_stack_path)
    ndim = img_arr.ndim
    if ndim == 3:
        num_colors = 1
        img_arr_processed = median_bkg_substration(img_arr)
    elif ndim == 4:
        # make it work for any umber of colors e.g. 3
        num_colors = img_arr.shape[1]
        img_arr_processed = np.zeros_like(img_arr) #, dtype=np.float32
        for color in range(num_colors):
            img_arr_processed[:, color, :, :] = median_bkg_substration(img_arr[:, color, :, :])
    imwrite(fpath_processed, img_arr_processed, imagej=True,
            metadata={'axis': 'TCYX', 'channels': num_colors,
            'mode': 'composite',})

if __name__ == '__main__':
    _ = QApplication([])
    directory = io.FileDialog(None, 'Please select a directory').openDirectoryDialog()
    process_directory(directory, rewrite=False)
