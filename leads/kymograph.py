import os, glob
import numpy as np
import pandas as pd
import h5py
from scipy.ndimage import median_filter, white_tophat, black_tophat
from scipy.signal import find_peaks, savgol_filter, peak_prominences
import PySimpleGUI as sg
import tifffile
from skimage.io.collection import alphanumeric_key
import pims
import trackpy
import matplotlib.pyplot as plt
plt.style.use('seaborn')


def read_img_stack(filepath):
    # filepath = sg.tkinter.filedialog.askopenfilename(title = "Select tif file/s",
    #                                                 filetypes = (("tif files","*.tif"),("all files","*.*")))

    image_meta = {}
    image_meta['filepath'] = os.path.abspath(filepath)
    with tifffile.TiffFile(filepath) as tif:
        imagej_hyperstack = tif.asarray()
        imagej_metadata = tif.imagej_metadata
    if imagej_hyperstack.ndim == 3:
        num_colors = 1
        image_meta["num_colors"] = num_colors
        image_meta['img_arr_color_' + str(0)] = imagej_hyperstack
    elif imagej_hyperstack.ndim == 4:
        num_colors = imagej_metadata['channels']
        image_meta["num_colors"] = num_colors
        for i in range(num_colors):
            image_meta['img_arr_color_' + str(i)] = imagej_hyperstack[:, i, :, :]
    return image_meta


def read_img_seq(num_colors=2):
    filepath = sg.tkinter.filedialog.askopenfilename(title = "Select tif file/s",
                                                    filetypes = (("tif files","*.tif *.tiff"),("all files","*.*")))
    image_meta = {}
    folderpath = os.path.dirname(filepath)
    image_meta['folderpath'] = folderpath
    if filepath.endswith('tif'):
        filenames = sorted(glob.glob(folderpath + "/*.tif"), key=alphanumeric_key)
    elif filepath.endswith('tiff'):
        filenames = sorted(glob.glob(folderpath + "/*.tif"), key=alphanumeric_key)
    image_meta['filenames'] = filenames
    num_frames = num_colors * (len(filenames)//num_colors)
    image_meta['num_frames'] = num_frames
    for i in range(num_colors):
        image_meta['filenames_color_'+str(i)] = filenames[i:num_frames:num_colors]
        image_meta['img_arr_color_'+str(i)] = pims.ImageSequence(image_meta['filenames_color_'+str(i)])
    return image_meta


def median_bkg_substration(txy_array, size_med=5, size_bgs=10, light_bg=False):
    array_processed = np.zeros_like(txy_array)
    for i in range(txy_array.shape[0]):
        img = txy_array[i]
        img_med = median_filter(img, size=size_med)
        if light_bg:
            img_med_bgs = black_tophat(img_med, size=size_bgs)
        else:
            img_med_bgs = white_tophat(img_med, size=size_bgs)
        array_processed[i, :, :] = img_med_bgs
    return array_processed


def peakfinder_savgol(kym_arr, skip_left=None, skip_right=None,
                      prominence_min=1/2, pix_width=11, plotting=False,
                      kymo_noLoop=None, correction_noLoop=True):
    '''
    prominence_min : minimum peak height with respect to max
    signal in the kymo.
    skip_left, skip_right : both postive integer.
    kymo_noLoop: needs have same number of rows as kym_arr
    '''
    if skip_left is None: skip_left = 0
    if skip_right is 0 or None: skip_right = 1
    if kymo_noLoop is not None and correction_noLoop:
        kymo_noLoop_avg = np.sum(kymo_noLoop, axis=1)
        kymo_noLoop_avg = 1 + (kymo_noLoop_avg/max(kymo_noLoop_avg))
        kym_arr = kym_arr / kymo_noLoop_avg[:, None]
    kym_arr_cropped = kym_arr[skip_left : -skip_right, :]
    pix_rad = int(pix_width/2)
    maxpeak_pos_list = []
    pos_list = []
    prominence_list = []
    for i in range(kym_arr_cropped.shape[1]):
        line1d = kym_arr_cropped[:, i]
        line1d_smth = savgol_filter(line1d, window_length=7, polyorder=1)
        peaks, properties = find_peaks(line1d_smth,
                prominence=min(line1d_smth))
        prom_bool = properties['prominences'] > (prominence_min * max(line1d_smth))
        prominences = properties['prominences'][prom_bool]
        peaks_sel = peaks[prom_bool]
        if len(peaks_sel) != 0:
            for peak_pos in peaks_sel:
                peak_int = np.sum(line1d_smth[peak_pos-pix_rad : peak_pos+pix_rad])
                peak_up_int = np.sum(line1d_smth[:peak_pos-pix_rad])
                peak_down_int = np.sum(line1d_smth[peak_pos+pix_rad:])
                pos_list.append([i, peak_pos, peak_int, peak_up_int, peak_down_int])
            prominence_list.append(prominences)
            # max peak pos and val
            max_prominence = max(prominences) #max(line1d_smth[peaks])
            maxpeak_pos = peaks_sel[prominences == max_prominence][-1]
            maxpeak_val = np.sum(line1d_smth[maxpeak_pos-pix_rad : maxpeak_pos+pix_rad])
            peak_up_int = np.sum(line1d_smth[:maxpeak_pos-pix_rad])
            peak_down_int = np.sum(line1d_smth[maxpeak_pos+pix_rad:])
            maxpeak_pos_list.append([i, maxpeak_pos, maxpeak_val, peak_up_int, peak_down_int])
    pos_list = np.array(pos_list)
    pos_list[:, 1] = pos_list[:, 1] + skip_left
    maxpeak_pos_list = np.array(maxpeak_pos_list)
    maxpeak_pos_list[:, 1] = maxpeak_pos_list[:, 1] + skip_left
    df_pks = pd.DataFrame(pos_list,
            columns=["FrameNumber", "PeakPosition", "PeakIntensity", "PeakUpIntensity", "PeakDownIntensity"])
    df_max_pks = pd.DataFrame(maxpeak_pos_list,
            columns=["FrameNumber", "PeakPosition", "PeakIntensity", "PeakUpIntensity", "PeakDownIntensity"])
    peak_dict = {
        "All Peaks": df_pks,
        "Max Peak" : df_max_pks,
        "Peak Prominences": prominence_list,
        "skip_left" : skip_left,
        "skip_right" : skip_right,
        "shape_kymo" : kym_arr.shape
    }
    if plotting:
        plt.figure(figsize=(15,5))
        plt.imshow(kym_arr, cmap='inferno')
        plt.plot(peak_dict["Max Peak"]["FrameNumber"],
                 peak_dict["Max Peak"]["PeakPosition"], '.r', alpha=0.5)
    return peak_dict


def analyze_maxpeak(df_maxpeak, smooth_length=11, fitting=False, fit_lim=[0, 30],
                convert_to_kb=True, frame_width=None, dna_length=48, pix_width=11):
    '''
    df_maxpeak is the dataframe obtained from the peak_finder analysis
    dna_length: in kilobases
    pix_width: should be same as in peakfinder function to get more accurate estimation
    '''
    if frame_width is None: frame_width = df_maxpeak["PeakPosition"].max() - df_maxpeak["PeakPosition"].min()
    pix_rad = pix_width/2
    df_analyzed = df_maxpeak.copy(deep=True)
    if convert_to_kb:
        total_int = df_maxpeak['PeakIntensity'] + df_maxpeak['PeakUpIntensity'] + df_maxpeak['PeakDownIntensity']
        avg_int_frame = total_int / frame_width
        df_analyzed['PeakIntensity'] = dna_length * (df_maxpeak['PeakIntensity'] - pix_width*avg_int_frame) / total_int
        df_analyzed['PeakUpIntensity'] = dna_length * (df_maxpeak['PeakUpIntensity'] + pix_rad*avg_int_frame) / total_int
        df_analyzed['PeakDownIntensity'] = dna_length * (df_maxpeak['PeakDownIntensity'] + pix_rad*avg_int_frame) / total_int
        df_analyzed['TotalIntensity'] = df_analyzed['PeakIntensity'] + df_analyzed['PeakUpIntensity'] + df_analyzed['PeakDownIntensity']
    df_analyzed["PeakIntFiltered"] = savgol_filter(df_analyzed["PeakIntensity"].values,
                                                window_length=smooth_length, polyorder=2)
    df_analyzed["PeakIntUpFiltered"] = savgol_filter(df_analyzed["PeakUpIntensity"].values,
                                                window_length=smooth_length, polyorder=2)
    df_analyzed["PeakIntDownFiltered"] = savgol_filter(df_analyzed["PeakDownIntensity"].values,
                                                window_length=smooth_length, polyorder=2)

    maxpeak_dict = {
        "Max Peak" : df_analyzed,
        "smooth length" : smooth_length,
        "fitting": fitting,
        "fit limit": fit_lim,
        "frame width": frame_width,
        "dna length": dna_length,
    }
    if fitting:
        pass
    return maxpeak_dict


def loop_sm_dist(maxpeak_dict, smpeak_dict, plotting=False, smooth_length=11):
    '''
    Compares the positions (frame numbers) of two colors and 
    returns the values where both colors have the same frame numbers
    '''
    df_loop = maxpeak_dict["Max Peak"]
    df_sm = smpeak_dict['Max Peak']
    df_sm = df_sm.loc[df_sm['FrameNumber'].isin(df_loop['FrameNumber'])]
    df_loop = df_loop.loc[df_loop['FrameNumber'].isin(df_sm['FrameNumber'])]
    df_sm.reset_index(drop=True, inplace=True)
    df_loop.reset_index(drop=True, inplace=True)
    pos_diff = (df_loop['PeakPosition'] - df_sm['PeakPosition']).values
    if np.average(pos_diff[:5]) < 0:
        pos_diff = -pos_diff
    no_loop_dna = df_loop["PeakUpIntensity"] + df_loop["PeakDownIntensity"]
    pos_diff_kb = no_loop_dna * pos_diff / maxpeak_dict['frame width']
    pos_diff_kb_shift = pos_diff_kb + df_loop['PeakIntensity']
    frame_number = df_loop['FrameNumber'].values
    peak_diff = pos_diff_kb_shift.values
    peak_diff_filt = savgol_filter(peak_diff, window_length=smooth_length, polyorder=2)
    loop_sm_dict = {
        "FrameNumber": frame_number,
        "PeakDiff" : peak_diff, # difference in peak position in kilobases
        "PeakDiffFiltered" : peak_diff_filt, #smoothed peaks
    }
    if plotting:
        frame_no = maxpeak_dict["Max Peak"]["FrameNumber"]
        int_peakdown = maxpeak_dict["Max Peak"]["PeakDownIntensity"]
        plt.plot(frame_no, maxpeak_dict["Max Peak"]["PeakIntensity"], '.g', label='loop')
        plt.plot(frame_no, maxpeak_dict["Max Peak"]["PeakIntFiltered"], 'g')
        plt.plot(frame_no, maxpeak_dict["Max Peak"]["PeakUpIntensity"], '.b', label='up')
        plt.plot(frame_no, maxpeak_dict["Max Peak"]["PeakIntUpFiltered"], 'b')
        plt.plot(frame_no, maxpeak_dict["Max Peak"]["PeakIntDownFiltered"], '.r', label='down')
        plt.plot(frame_no, maxpeak_dict["Max Peak"]["PeakIntDownFiltered"], 'r')
        # this func specific
        plt.plot(frame_number, df_loop['PeakIntensity'], 'm', label='road block')
        plt.xlabel('Frame number')
        plt.ylabel('DNA/kb')
        plt.legend()
    return loop_sm_dict

def link_peaks(df_peaks, df_peaks_sm=None, search_range=10, memory=5, filter_length=10,
               plotting=True, axis=None,):
    plt.style.use('seaborn')
    df_peaks["x"] = df_peaks["PeakPosition"]
    if df_peaks_sm is None:
        df_peaks["y"] = df_peaks["FrameNumber"]
    else:
        df_peaks["y"] = df_peaks_sm["PeakPosition"]
    df_peaks["frame"] = df_peaks["FrameNumber"]
    t = trackpy.link(df_peaks, search_range=search_range,
                memory=memory, pos_columns=['y', 'x'])
    t_filt = trackpy.filter_stubs(t, filter_length)
    t_filt_gb = t_filt.groupby("particle")
    gb_names = list(t_filt_gb.groups.keys())
    peaks_linked = t_filt.copy(deep=True)
    particle_index = 1
    for name in gb_names:
        peaks_linked['particle'][t_filt['particle']==name] = particle_index
        particle_index += 1
    peaks_linked_gb = peaks_linked.groupby("particle")
    gb_names = list(peaks_linked_gb.groups.keys())
    if plotting:
        if axis is None:
            fig, axis = plt.subplots()
        for name in gb_names:
            gp_sel = peaks_linked_gb.get_group(name)
            axis.plot(gp_sel["frame"], gp_sel["x"], label=str(name), alpha=0.8)
            axis.text(gp_sel["frame"].values[0], gp_sel["x"].values[0], str(name))
        plt.show()
    peaks_linked = peaks_linked.reset_index(drop=True)
    return peaks_linked


def link_and_plot_two_color(peak_dict, peak_dict_sm,
            search_range=10, memory=5, filter_length=10,
            plotting=True):
    df_peaks = peak_dict["All Peaks"]
    df_peaks_sm = peak_dict_sm["All Peaks"]
    # set axes and figsize
    fig = plt.figure()
    gs = fig.add_gridspec(3, 3)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_xticks([])
    ax2 = fig.add_subplot(gs[1, :])
    ax2.set_xticks([])
    ax3 = fig.add_subplot(gs[2, :])
    # link and plot data to it
    df_peaks_linked = link_peaks(df_peaks, search_range=search_range,
                          memory=memory, filter_length=filter_length,
                          plotting=plotting, axis=ax1)
    df_peaks_linked_sm = link_peaks(df_peaks_sm, search_range=search_range,
                          memory=memory, filter_length=filter_length,
                          plotting=plotting, axis=ax2)
    ax3.plot(df_peaks_linked['frame'], df_peaks_linked['x'], '.g', alpha=0.8, label='DNA')
    ax3.plot(df_peaks_linked_sm['frame'], df_peaks_linked_sm['x'], '.m', alpha=0.8, label='SM')
    ax3.legend(loc=4)
    dna_height = peak_dict["shape_kymo"][0]
    kymo_length = peak_dict["shape_kymo"][1]
    ax_list = [ax1, ax2, ax3]
    for ax in ax_list:
        ax.hlines([0, dna_height], 0, kymo_length, 'g', alpha=0.5)
        ax.set_ylim([0 - 0.05*dna_height, dna_height + 0.05*dna_height])
        ax.set_xlim(0, kymo_length)
        ax.set_ylabel('pixels')
    ax3.set_xlabel('Frame Numbers')
    ax1.text(kymo_length-0.55*kymo_length, 0.9*dna_height, 'DNA punctas')
    ax2.text(kymo_length-0.55*kymo_length, 0.9*dna_height, 'Single molecules')
    ax3.text(kymo_length-0.55*kymo_length, 0.9*dna_height, 'DNA punctas and SM')
    fig.tight_layout()
    plt.show()
    return df_peaks_linked, df_peaks_linked_sm


if __name__ == "__main__":
    print('kymograph module imported')
    