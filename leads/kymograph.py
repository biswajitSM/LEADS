import os, glob
import numpy as np
from numba import jit
import pandas as pd
from tqdm import trange
from .utils.sparseDeconv.sparse_recon import sparse_deconv
from scipy.optimize import curve_fit
from scipy.ndimage import median_filter, white_tophat, black_tophat, gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter
from scipy.spatial import cKDTree
from scipy import interpolate
import tifffile
from skimage.io.collection import alphanumeric_key
import pims
import trackpy
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
from . import io
from .utils import figure_params
plt.rcParams.update(figure_params.params_dict)

def read_img_stack(filepath):
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
    filepath = io.FileDialog(None, 'Open Tif File', "Tif File (*.tif *.tiff)").openFileNameDialog()
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


def median_bkg_substration(txy_array, size_med=5, size_bgs=10, light_bg=False, onlyBackground=False):
    array_processed = np.zeros_like(txy_array)
    for i in range(txy_array.shape[0]):
        img = txy_array[i]
        if onlyBackground:
            img_med = img
        else:
            img_med = median_filter(img, size=size_med)
        if light_bg:
            img_med_bgs = black_tophat(img_med, size=size_bgs)
        else:
            img_med_bgs = white_tophat(img_med, size=size_bgs)
        array_processed[i, :, :] = img_med_bgs
    return array_processed

def applySparseSIM(txy_array, params=None, preview=False):
    # apply sparseSIM (https://doi.org/10.1038/s41587-021-01092-2)
    if params is None:
        params = {
            "sigma": 5,
            "numIter": 100,
            "fidelity": 150,
            "sparsity": 10,
            "tcontinuity": 0.5,
            "background": 0,
            "deconv_iter": 7,
            "deconv_type": 1
        }

    array_processed = np.zeros_like(txy_array)
    # get longer axis
    maxShape = np.max(txy_array.shape[1:])
    boundary = 0
    if maxShape%2!=0:
        maxShape += 1
        boundary = 1
    maxShapeAxis = np.argmax(txy_array.shape[1:])
    minShape = np.min(txy_array.shape[1:])
    paddedImg = np.zeros((maxShape, maxShape))
    if preview:
        numFrames = 5
    else:
        numFrames = txy_array.shape[0]
    for i in trange(numFrames):
        img = txy_array[i]
        if maxShapeAxis==0:
            paddedImg[:-boundary, 0:minShape] = img
        elif maxShapeAxis==1:
            paddedImg[0:minShape, :-boundary] = img
            
        img_recon = sparse_deconv.sparse_deconv(paddedImg, np.array([params["sigma"], params["sigma"]]), \
                sparse_iter = params["numIter"], fidelity = params["fidelity"], \
                sparsity = params["sparsity"], tcontinuity = params["tcontinuity"],
                background = params["background"], deconv_iter = params["deconv_iter"], \
                deconv_type = params["deconv_type"])

        if maxShapeAxis==0:
            img_recon = img_recon[:-boundary, 0:minShape]
        elif maxShapeAxis==1:
            img_recon = img_recon[0:minShape, :-boundary]
        array_processed[i, :, :] = img_recon
    return array_processed


def bkg_substration(txy_array, size_bgs=10, light_bg=False):
    '''
    Background substraction
    '''
    array_processed = np.zeros_like(txy_array)
    for i in range(txy_array.shape[0]):
        img = txy_array[i]
        if light_bg:
            img_bgs = black_tophat(img, size=size_bgs)
        else:
            img_bgs = white_tophat(img, size=size_bgs)
        array_processed[i, :, :] = img_bgs
    return array_processed


def find_ends_canny(dna_kym, sigma=2, threshold_min=0.1, threshold_max=None, plotting=False):
    if threshold_max is not None:
        threshold_max = threshold_max * dna_kym.max()
    if threshold_min is not None:
        threshold_min = threshold_min*dna_kym.mean()
    edges = canny(dna_kym, sigma=sigma, low_threshold=threshold_min, high_threshold=threshold_max)
    left_ends = []
    right_ends = []
    for i in range(edges.shape[1]):
        bool_i = edges[:, i]
        bool_i_inds = np.where(bool_i)
        if bool_i_inds[0].shape[0] == 0:
            left_ends.append(0)
            right_ends.append(0)
        else:
            left_ends.append(bool_i_inds[0][0] - 1)
            right_ends.append(bool_i_inds[0][-1] + 1)
    left_ends[0] = left_ends[1]
    right_ends[0] = right_ends[1]
    left_ends[-1] = left_ends[-2]
    right_ends[-1] = right_ends[-2]
    if plotting:
        fig,(ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15,10))
        ax1.imshow(dna_kym)
        ax2.imshow(edges)
        ax1.plot(right_ends, '.y')
        ax1.plot(left_ends, '.w')
    return left_ends, right_ends


def find_ends_supergauss(line_data, gauss_length=10, gauss_order=6, fraction_of_max=1, threshold_Imax=0.5, plotting=True):
    
    xdata = np.arange(len(line_data))
    line_data_temp = line_data
    line_data = np.array([float(value)/max(line_data_temp) for value in line_data_temp])
    
    plt.plot(xdata, line_data)
    
    initial_guess = [.2,1.,np.median(xdata),gauss_length, gauss_order]
    constraints =([0, 0, 0, gauss_length, gauss_order],[np.inf, np.inf, np.inf, np.inf, np.inf])
    popt, pcov = curve_fit(super_gauss_function, xdata, line_data,
                           p0 = initial_guess, bounds = constraints)
    fine_scale_x = np.linspace(xdata[0],xdata[-1],len(xdata)*1000)
    maximum = max(super_gauss_function(xdata, *popt))
    #--- this is the threshold
    # intersection_value = maximum-(maximum - popt[0])*threshold_Imax
    # linedata = [intersection_value for x in fine_scale_x]
    # index = np.argwhere(np.diff(np.sign(linedata - super_gauss_function(fine_scale_x, *popt)))).flatten()
    
    # just take where the supergauss is fraction_of_max*popt[0]
    intersection_value = maximum-(maximum - popt[0])*(1-fraction_of_max)
    linedata = [intersection_value for x in fine_scale_x]
    index = np.argwhere(np.diff(np.sign(linedata - super_gauss_function(fine_scale_x, *popt)))).flatten()
    if len(index) != 2:
        print('WARNING: no sensible fit found which intersects with data.')    
    # index_intersection = [int(fine_scale_x[index[0]]), int(fine_scale_x[index[1]])]
    #-- Here is where we ensure two crossings and select for length
    plt.plot(fine_scale_x, super_gauss_function(fine_scale_x, *popt), 'r') 
    if len(index)==2 and plotting:
        index_intersection = [fine_scale_x[index[0]], fine_scale_x[index[1]]]
        plt.plot(index_intersection,
            super_gauss_function([fine_scale_x[index[0]],fine_scale_x[index[1]]], *popt), 'ro')
        plt.plot([index_intersection[0], index_intersection[0]], [0, 1], 'r')
        plt.plot([index_intersection[1], index_intersection[1]], [0, 1], 'r')
        formula = '$'\
            +str(np.round(popt[1], decimals=1))\
            +'\cdot e^{-(\\frac{(x-'\
            +str(np.round(popt[2], decimals=1))\
            +')^2}{2'\
            +str(np.round(popt[3], decimals=1))\
            +'^2})^{'\
            +str(np.round(popt[4], decimals=1))\
            +'}}+'\
            +str(np.round(popt[0], decimals=1))\
            +'$'
        return index_intersection, formula
    else:
        return None


def super_gauss_function(x, floor, amplitude, mean, sigma, power):
    '''https://en.wikipedia.org/wiki/Gaussian_function'''
    return floor + amplitude*np.exp((-((x-mean)**2/(2*sigma**2))**power))

def OneGaussian(x,mu,sigma):
    # This is the equation for a normalized1D gaussian peak value one
    return np.exp(-((x-mu)**2)/(2*sigma**2))

def find_ends_peakPeeling(line_data, noPeaks=10, PSF=4, amplitude=0.9, residue=0.1, plotting=True):
    # PSF is given in pixel of the original supplied data
    
    xdata = np.arange(len(line_data))
    line_data_temp = line_data
    line_data = np.array([float(value)/max(line_data_temp) for value in line_data_temp])
    splineParameters = interpolate.splrep(xdata, line_data)
    numInterpolationPoints = len(xdata)*100
    xdata_interpolated = np.linspace(xdata[0], xdata[-1], numInterpolationPoints, endpoint=True)
    line_data = interpolate.splev(xdata_interpolated, splineParameters, der=0)
    if residue < 1e-6:
        residue = 0.005
        print("WARNING PEAK PEELING: Residue has been set to low value. Danger of an infinite loop. Setting it to 0.005.")
    if noPeaks > 50:
        noPeaks = 50
        print("WARNING PEAK PEELING: Max number of peaks has been set to a large value. Danger of an infinite loop. Setting it to 50.")
    
    if plotting:
        plt.plot(xdata_interpolated, line_data)
    

    stopit = False
    PeelCurve = line_data
    buildcurve = np.zeros(len(line_data))
    peakcount = 0
    CoveredFraction = []
    Xpos = []
    singleGaussians = []


    while not stopit:
        PeakVal = np.max(PeelCurve)
        Xpos.append( xdata_interpolated[np.argmax(PeelCurve)] )
        OneGaussianCurve = amplitude * PeakVal * OneGaussian(xdata_interpolated, Xpos[-1], PSF)    
        PeelCurve  = PeelCurve  - OneGaussianCurve
        buildcurve = buildcurve + OneGaussianCurve
        singleGaussians.append( OneGaussianCurve )
        CoveredFraction.append( np.sum(buildcurve)/np.sum(line_data) )
        if peakcount > 0:               
            RelChange = (CoveredFraction[-1]-CoveredFraction[-2]) / CoveredFraction[-1]
            stopit = ( (RelChange<residue) or (peakcount>noPeaks) and (peakcount>=1) )
        
        peakcount += 1
        if (peakcount >= noPeaks):
            stopit = True
    ResidualCurve = line_data - buildcurve    


    if len(Xpos) > 1:
        dna_ends_xdata_interpolated_px = np.array([\
            np.min(Xpos), \
            np.max(Xpos)]) 
        if plotting:
            plt.plot(xdata_interpolated, buildcurve, 'g')
            plt.plot(xdata_interpolated, ResidualCurve, 'k--', alpha=0.5)
            for singleGaussian in singleGaussians:
                plt.plot(xdata_interpolated, singleGaussian, 'k', alpha=0.5)
            plt.plot([dna_ends_xdata_interpolated_px[0], dna_ends_xdata_interpolated_px[0]], [0, 1], 'g')
            plt.plot([dna_ends_xdata_interpolated_px[1], dna_ends_xdata_interpolated_px[1]], [0, 1], 'g')
            lower_end = np.where(xdata_interpolated==dna_ends_xdata_interpolated_px[0])
            upper_end = np.where(xdata_interpolated==dna_ends_xdata_interpolated_px[1])
            plt.plot(dna_ends_xdata_interpolated_px, [line_data[lower_end], line_data[upper_end]], 'go')
        
        return dna_ends_xdata_interpolated_px
    else:
        return None


def bilateralFtr1D(y, sSpatial = 5, sIntensity = 1):
    '''
    version: 1.0
    author: akshay: https://gitlab.com/-/snippets/16327
    date: 10/03/2016
    
    The equation of the bilateral filter is
    
            (       dx ^ 2       )       (         dI ^2        )
    F = exp (- ----------------- ) * exp (- ------------------- )
            (  sigma_spatial ^ 2 )       (  sigma_Intensity ^ 2 )
        ~~~~~~~~~~~~~~~~~~~~~~~~~~
        This is a guassian filter!
        dx - The 'geometric' distance between the 'center pixel' and the pixel
         to sample
    dI - The difference between the intensity of the 'center pixel' and
         the pixel to sample
    sigma_spatial and sigma_Intesity are constants. Higher values mean
    that we 'tolerate more' higher value of the distances dx and dI.
    
    Dependencies: numpy, scipy.ndimage.gaussian_filter1d
    
    calc gaussian kernel size as: filterSize = (2 * radius) + 1; radius = floor (2 * sigma_spatial)
    y - input data
    '''

    # gaussian filter and parameters
    radius = int( np.floor (2 * sSpatial) )
    filterSize = ((2 * radius) + 1)
    ftrArray = np.zeros (int(filterSize))
    ftrArray[radius] = 1
    
    # Compute the Gaussian filter part of the Bilateral filter
    gauss = gaussian_filter1d(ftrArray, sSpatial)

    # 1d data dimensions
    width = y.size

    # 1d resulting data
    ret = np.zeros (width)

    for i in range(width):

        ## To prevent accessing values outside of the array
        # The left part of the lookup area, clamped to the boundary
        xmin = max (i - radius, 1)
        # How many columns were outside the image, on the left?
        dxmin = xmin - (i - radius)

        # The right part of the lookup area, clamped to the boundary
        xmax = min (i + radius, width)
        # How many columns were outside the image, on the right?
        dxmax = (i + radius) - xmax

        # # The actual range of the array we will look at
        # area = y [xmin:xmax]

        # # The center position
        # center = y [i]

        # The left expression in the bilateral filter equation
        # We take only the relevant parts of the matrix of the
        # Gaussian weights - we use dxmin, dxmax, dymin, dymax to
        # ignore the parts that are outside the image
        expS = gauss[(1+dxmin):(filterSize-dxmax)]

        # The right expression in the bilateral filter equation
        dy = y [xmin:xmax] - y [i]
        dIsquare = (dy * dy)
        expI = np.exp (- dIsquare / (sIntensity * sIntensity))

        # The bilater filter (weights matrix)
        F = expI * expS

        # Normalized bilateral filter
        Fnormalized = F / sum(F)

        # Multiply the area by the filter
        tempY = y [xmin:xmax] * Fnormalized

        # The resulting pixel is the sum of all the pixels in
        # the area, according to the weights of the filter
        # ret(i,j,R) = sum (tempR(:))
        ret[i] = sum (tempY)
    
    return ret

def peakfinder_savgol(kym_arr, skip_left=None, skip_right=None,
                      smooth_length=7, prominence_min=1/2, peak_width=(None, None),
                      threshold_glbal_peak=True, threshold_value=1, #threshold: threshold vlue in percentage that compare all peaks in the kymo
                      pix_width=11, plotting=False,
                      kymo_noLoop=None, correction_noLoop=True):
    '''
    prominence_min : minimum peak height with respect to max
    signal in the kymo.
    skip_left, skip_right : both postive integer.
    kymo_noLoop: needs have same number of rows as kym_arr
    '''
    if skip_left is None: skip_left = 0
    if skip_right is None or skip_right == 0: skip_right = 1
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
        if smooth_length > 2:
            line1d_smth = savgol_filter(line1d, window_length=smooth_length, polyorder=1)
        else:
            line1d_smth = line1d
        peaks, properties = find_peaks(line1d_smth,
                prominence=(None, None), width=peak_width) #min(line1d_smth), width=(pix_rad, 3*pix_rad)
        prom_bool = properties['prominences'] > (prominence_min * max(line1d_smth))
        line1d_smth = line1d # set here the line on which we do the actual intensity quantification to the non-smoothed version, which is closer to the real value
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
    if threshold_glbal_peak:
        df_pks = threshold_all_peaks(df_pks, threshold_value)
        df_max_pks = threshold_all_peaks(df_max_pks, threshold_value)
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


def threshold_all_peaks(df_peaks, threshold=1):
    '''threshold: threshold vlue in percentage that compare all peaks in the kymo
    df_peaks    
    '''
    int_peaks = df_peaks["PeakIntensity"].median()
    int_threshold = 1e-2*threshold*int_peaks
    mask = df_peaks["PeakIntensity"] > int_threshold
    df_peaks_threshold = df_peaks[mask]
    return df_peaks_threshold.reset_index(drop=True)


def analyze_maxpeak(df_maxpeak, smooth_length=11, fitting=False, fit_lim=[0, 30],
                convert_to_kb=True, frame_width=None, dna_length=48, pix_width=11):
    '''
    df_maxpeak is the dataframe obtained from the peak_finder analysis
    dna_length: in kilobases
    pix_width: should be same as in peakfinder function to get more accurate estimation
    '''
    if frame_width is None: frame_width = df_maxpeak["PeakPosition"].max() - df_maxpeak["PeakPosition"].min()
    pix_rad = int( pix_width/2 )
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
    if smooth_length > len(peak_diff):
        smooth_length = len(peak_diff)
        if smooth_length%2==0: smooth_length -= 1
    if smooth_length <= 1:
        peak_diff_filt = peak_diff
    else:        
        peak_diff_filt = savgol_filter(peak_diff, window_length=smooth_length, polyorder=2)
    loop_sm_dict = {
        "FrameNumber" : frame_number,
        "PositionDiff" : pos_diff,
        "PositionDiff_kb" : pos_diff_kb,
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

def link_peaks(df_peaks, acqTime=None, df_peaks_sm=None, search_range=10, memory=5, filter_length=10,
               plotting=True, axis=None, DNA_ends=None, usePrecomputed=None):
    xLabelIsFrames = False
    if acqTime is None:
        acqTime = 1
        xLabelIsFrames = True
    df_peaks["x"] = df_peaks["PeakPosition"]
    if df_peaks_sm is None:
        df_peaks["y"] = df_peaks["FrameNumber"]
    else:
        df_peaks["y"] = df_peaks_sm["PeakPosition"]
    df_peaks["frame"] = df_peaks["FrameNumber"]
    if usePrecomputed is None:
        trackpy.quiet(suppress=True)
        print('Linking peaks... hang on!')
        t = trackpy.link(df_peaks, search_range=search_range,
                    memory=memory, pos_columns=['y', 'x'])
        trackpy.quiet(suppress=False)
        t_filt = trackpy.filter_stubs(t, filter_length)
    else:
        t_filt = usePrecomputed
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
            fig = plt.figure(figsize=(10, 4))
            gs = fig.add_gridspec(1, 4)
            axis = fig.add_subplot(gs[0, :-1])
            if xLabelIsFrames:
                axis.set_xlabel('Frames')
            else:
                axis.set_xlabel('time/s')
            axis.set_ylabel('Pixels')
            axis_r = fig.add_subplot(gs[0, -1:])
            axis_r.set_xticks([])
            axis_r.set_yticks([])

            axis_r.hist(df_peaks["PeakPosition"], orientation='horizontal')
        for name in gb_names:
            gp_sel = peaks_linked_gb.get_group(name)
            axis.plot(gp_sel["frame"]*acqTime, gp_sel["x"], label=str(name), alpha=0.8)
            relPosition = (np.average(gp_sel["x"].values) - np.min(DNA_ends)) / (np.max(DNA_ends)-np.min(DNA_ends))
            track_name_and_relPosition = str(name) + ": " + "{:.2f}".format(relPosition)
            axis.text(gp_sel["frame"].values[0]*acqTime, np.average(gp_sel["x"].values[:10]), track_name_and_relPosition)
        plt.gcf().show()
    peaks_linked = peaks_linked.reset_index(drop=True)
    return peaks_linked


def link_and_plot_two_color(df_peaks, df_peaks_sm, acqTime=None,
            search_range=10, memory=5, filter_length=10,
            plotting=True, DNA_ends=None, usePrecomputed=None, usePrecomputedsm=None):
    # set axes and figsize
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(4, 4)
    ax1 = fig.add_subplot(gs[0, :-1])
    ax1.set_xticks([])
    ax2 = fig.add_subplot(gs[1, :-1])
    ax2.set_xticks([])
    ax3 = fig.add_subplot(gs[2, :-1])
    ax3.set_xticks([])
    ax4 = fig.add_subplot(gs[3, :-1])
    # add axs for histogram
    ax1_r = fig.add_subplot(gs[0, -1:], sharey=ax1)
    ax1_r.set_xticklabels([])
    ax2_r = fig.add_subplot(gs[1, -1:], sharey=ax2)
    ax2_r.set_xticklabels([])
    ax3_r = fig.add_subplot(gs[2, -1:], sharey=ax3)
    ax3_r.set_xticklabels([])

    xLabelIsFrames = False
    if acqTime is None:
        acqTime = 1
        xLabelIsFrames = True
        
    # link and plot data to it
    df_peaks_linked = link_peaks(df_peaks, acqTime=acqTime, search_range=search_range,
                          memory=memory, filter_length=filter_length,
                          plotting=plotting, axis=ax1, DNA_ends=DNA_ends, usePrecomputed=usePrecomputed)
    ax1_r.hist(df_peaks["PeakPosition"], orientation='horizontal')
    df_peaks_linked_sm = link_peaks(df_peaks_sm, acqTime=acqTime, search_range=search_range,
                          memory=memory, filter_length=filter_length,
                          plotting=plotting, axis=ax2, DNA_ends=DNA_ends, usePrecomputed=usePrecomputedsm)
    ax2_r.hist(df_peaks_sm["PeakPosition"], orientation='horizontal')
    ax3_r.hist(df_peaks["PeakPosition"], histtype='step', orientation='horizontal')
    ax3_r.hist(df_peaks_sm["PeakPosition"], histtype='step', orientation='horizontal')

    ax3.plot(df_peaks_linked_sm['frame']*acqTime, df_peaks_linked_sm['x'], '.m', alpha=0.5, label='SM')
    ax3.plot(df_peaks_linked['frame']*acqTime, df_peaks_linked['x'], '.g', alpha=0.3, label='DNA')
    ax3.legend(loc=4)
    ax4_sc = ax4.scatter(df_peaks_linked['frame']*acqTime, df_peaks_linked['x'], marker='.',
                         c=df_peaks_linked["PeakIntensity"].values, cmap='jet')
    ax4_cbar = plt.colorbar(ax4_sc)
    l, b, w, h = ax4.get_position().bounds
    ll, bb, ww, hh = ax4_cbar.ax.get_position().bounds
    ax4_cbar.ax.set_position([ll- ll*0.1, b, ww - ww*0.1, h])
    # ax4_cbar.set_ticks([])
    ax_list = [ax1, ax2, ax3, ax4]
    max_frame_no = df_peaks['FrameNumber'].max()
    max_frame_no_sm = df_peaks_sm['FrameNumber'].max()
    xmax = acqTime * max(max_frame_no, max_frame_no_sm)
    for ax in ax_list:
        ax.set_ylim(0, None)
        ax.set_xlim(0, xmax)
        ax.set_ylabel('pixels')
    if xLabelIsFrames:
        ax4.set_xlabel('Frame Numbers')
    else:
        ax4.set_xlabel('time/s')
    ax1.text(0.55, 0.9, 'DNA punctas')
    ax2.text(0.55, 0.9, 'Single molecules')
    ax3.text(0.55, 0.9, 'DNA punctas and SM')
    # fig.tight_layout()
    plt.gcf().show()
    result = {
        'df_peaks_linked' : df_peaks_linked,
        'df_peaks_linked_sm' : df_peaks_linked_sm,
        'ax1' : ax1,
        'ax2' : ax2,
        'ax3' : ax3,
        'ax4' : ax4,
    }
    return result


def analyze_multipeak(df_linked, convert_to_kb=True, frame_width=None,
                      dna_length=48.5, pix_width=11, pix_size=0.115,
                     interpolation=True, dna_persistence_length=50, SxOconc=100):
    '''
    frame_width: same as the DA end to end distance
    pix_size: in micrometer
    '''
    if frame_width is None: frame_width = df_linked["PeakPosition"].max() - df_linked["PeakPosition"].min()
    pix_rad = int( pix_width/2 )
    df_linked_cp = df_linked.copy(deep=True)
    df_linked_cp["TotalIntensity"] = df_linked["PeakIntensity"] + df_linked["PeakUpIntensity"] + df_linked["PeakDownIntensity"]
    df_linked_cp["NonPeakIntensity"] = df_linked["PeakUpIntensity"] + df_linked["PeakDownIntensity"]
    df_linked_gb = df_linked_cp.groupby("FrameNumber")
    gb_names = df_linked_gb.groups.keys()
    for name in gb_names:
        gb_sel = df_linked_gb.get_group(name)
        # gb_sel.apply(lambda x: x)
        num_peaks = len(gb_sel)
        if num_peaks > 1:
            total_avg_int = np.sum(gb_sel["TotalIntensity"][...]) / num_peaks
            non_peak_int = total_avg_int - np.sum(gb_sel["PeakIntensity"][...])
            df_linked_cp["NonPeakIntensity"][df_linked_cp["FrameNumber"]==name] = non_peak_int
    df_linked_gb = df_linked_cp.copy(deep=True)
    if convert_to_kb:
        avg_int_pix = df_linked_gb["TotalIntensity"] / frame_width # intensity per pixel
        df_linked_gb['PeakIntensity'] = dna_length * (df_linked_gb['PeakIntensity'] - 2*pix_rad*avg_int_pix) / df_linked_gb["TotalIntensity"]
        df_linked_gb['PeakUpIntensity'] = dna_length * (df_linked_gb['PeakUpIntensity'] + pix_rad*avg_int_pix) / df_linked_gb["TotalIntensity"]
        df_linked_gb['PeakDownIntensity'] = dna_length * (df_linked_gb['PeakDownIntensity'] + pix_rad*avg_int_pix) / df_linked_gb["TotalIntensity"]        
        # df_linked_gb['NonPeakIntensity'] = dna_length * (df_linked_gb['NonPeakIntensity'] + 2*pix_rad*avg_int_pix) / df_linked_gb["TotalIntensity"]
        df_linked_gb['NonPeakIntensity'] = df_linked_gb['PeakUpIntensity'] + df_linked_gb['PeakDownIntensity']
        df_linked_gb['TotalIntensity'] = dna_length * df_linked_gb['TotalIntensity'] / df_linked_gb["TotalIntensity"]

    # plt.figure()
    # plt.plot(df_linked_gb['PeakIntensity'], label='peak')
    # plt.plot(df_linked_gb['PeakUpIntensity'], label='up')
    # plt.plot(df_linked_gb['PeakDownIntensity']+df_linked_gb['PeakUpIntensity'], label='sum u+d')
    # plt.plot(df_linked_gb['PeakDownIntensity']+df_linked_gb['PeakUpIntensity']+df_linked_gb['PeakIntensity'], label='sum u+d+p')
    # plt.plot(df_linked_gb['PeakDownIntensity'], label='down')
    # plt.plot(df_linked_gb['NonPeakIntensity'], label='nonpeak')
    # plt.plot(df_linked_gb['TotalIntensity'], label='total')
    # plt.legend()
    # plt.show()

    SxO_nM = [0, 10, 50, 100, 200, 500]
    Lc_correction_factor = [1, 1.0258, 1.0523, 1.0649, 1.0948, 1.3829] # for non-coilable DNA
    # interpolate correction factor for the SxO concentration we are working at
    correction_factor = np.interp(SxOconc, SxO_nM, Lc_correction_factor)


    # compute the contour length of the piece of DNA which is not in the DNA
    DNA_contour_length_nonPeakFraction = df_linked_gb['NonPeakIntensity'] * 0.342 * correction_factor    
    # um_per_kb = DNA_contour_length/dna_length
    # non_peak_dna_um = um_per_kb * df_linked_gb['NonPeakIntensity'].values
    # nonpeak_rel_ext = frame_width * pix_size / non_peak_dna_um
    # divide the end-2-end tether length of the DNA molecule by the contour length to obtain the rel. extension
    nonpeak_rel_ext = frame_width * pix_size / DNA_contour_length_nonPeakFraction
    df_linked_gb["NonPeakRelativeExtension"] = nonpeak_rel_ext
    if interpolation:
        F = np.interp(nonpeak_rel_ext, RELATIVE_EXTENSION, FORCE)
    else:
        # F = force_wlc(nonpeak_rel_ext, Plen=dna_persistence_length)
        F = force_wlc7(nonpeak_rel_ext, dna_persistence_length/1e3)
    df_linked_gb["Force"] = F
    return df_linked_gb


def link_multipeaks_2colrs(
                df_peaks_linked_col1, df_peaks_linked_col2,
                delta_frames=10, delta_pixels=10,
                delta_colocalized=5, plotting=False):
    coord_col1 = df_peaks_linked_col1[['x', 'y']].values
    coord_col2 = df_peaks_linked_col2[['x', 'y']].values
    if plotting:
        fig, ax = plt.subplots()    
    tree_col1 = cKDTree(coord_col1)
    particle_names = df_peaks_linked_col2.particle.unique()
    print(particle_names)
    columns = ["particle_col1", "particle_col2", "int_col1", "int_col2",
               "frame_col1", "frame_col2", "x_col1", "x_col2",
               "len_col1", "len_col2"
               ]
    df_cols_linked = pd.DataFrame(columns=columns)
    for col2_particle_no in particle_names:
        col2_particle_i = df_peaks_linked_col2[df_peaks_linked_col2["particle"]==col2_particle_no][['x', 'y']].values
        dist, indexes = tree_col1.query(col2_particle_i)
        paired_indexes = clean_duplicate_maxima(dist, indexes)
        left_linked_indexes = []
        for i, j in paired_indexes:
            pix1, frame1 = coord_col1[i]
            pix2, frame2 = col2_particle_i[j]
            distance = np.sqrt( (pix2 - pix1)**2 + (frame2 - frame1)**2 )
            width = coord_col1.shape[1]
            # if distance < distance_max:
            if abs(pix2-pix1) < delta_pixels and abs(frame2-frame1) < delta_frames:
                left_linked_indexes.append(i)
            if plotting:
                tmp_color = np.random.uniform(0,1,3)
                ax.plot(frame1, pix1, color=tmp_color, marker='+')
                ax.plot(frame2, pix2 + width, color=tmp_color, marker='+')
                ax.plot([frame1, frame2], [pix1, pix2 + width], color=tmp_color)
        col1_paritcles = df_peaks_linked_col1.iloc[left_linked_indexes]['particle'][...]
        if len(col1_paritcles.index) > delta_colocalized:
            col1_particle_no = np.argmax(np.bincount(col1_paritcles))
            part_int_col1 = df_peaks_linked_col1[df_peaks_linked_col1['particle']==col1_particle_no]['PeakIntensity'][:10].mean()
            part_int_col2 = df_peaks_linked_col2[df_peaks_linked_col2['particle']==col2_particle_no]['PeakIntensity'][:10].mean()
            frame_col1 = df_peaks_linked_col1[df_peaks_linked_col1['particle']==col1_particle_no]["FrameNumber"].values[0]
            frame_col2 = df_peaks_linked_col2[df_peaks_linked_col2['particle']==col2_particle_no]["FrameNumber"].values[0]
            x_col1 = df_peaks_linked_col1[df_peaks_linked_col1['particle']==col1_particle_no]["x"].values[0]
            x_col2 = df_peaks_linked_col2[df_peaks_linked_col2['particle']==col2_particle_no]["x"].values[0]
            len_col1 = len(df_peaks_linked_col1[df_peaks_linked_col1['particle']==col1_particle_no].index)
            len_col2 = len(df_peaks_linked_col2[df_peaks_linked_col2['particle']==col2_particle_no].index)
            df_cols_linked = df_cols_linked.append(pd.DataFrame([[
                                float(col1_particle_no), float(col2_particle_no), part_int_col1, part_int_col2,
                                frame_col1, frame_col2, x_col1, x_col2,
                                float(len_col1), float(len_col2),
                                ]], columns=columns), ignore_index=True)
    if plotting:
        ax.plot(coord_col1[:, 1], coord_col1[:, 0], '.b', alpha=0.1)
        ax.plot(coord_col2[:, 1], coord_col2[:, 0], '.r', alpha=0.1)
        plt.gcf().show()
    return df_cols_linked


def clean_duplicate_maxima(dist, indexes):
    paired_indexes = []
    count = -1
    for i in set(indexes):
        tmp_dist = np.inf
        tmp = None
        for j, k in zip(indexes, dist):
            if i == j:
                count += 1
                if k < tmp_dist:
                    tmp = [j, count]
                    tmp_dist = k
            else:
                pass
        if tmp is not None:
            paired_indexes.append(tmp)
    return paired_indexes

def qinterp_max(y, x=None, extras=False):
    '''
    Find the maximum point in an array, and if it's an interior point,
    interpolate among it and its two nearest neighbors to predict
    the interpolated maximum.
    
    Returns that interpolated maximum,
    if extras=True, also returns the coefficients
    and the index and value of the sample maximum.
    source: https://www.embeddedrelated.com/showarticle/855.php
    '''
    if x is None:
        x = np.arange(0, len(y))
    imax = 0
    ymax = -float('inf')
    # run through the points to find the maximum
    for i,y_i in enumerate(y):
        if y_i > ymax:
            imax = i
            ymax = y_i
    # no interpolation if at the ends
    # print(i)
    if imax == 0 or imax == i:
        # print('conditions rue')
        return ymax, x[imax]
    # otherwise, y[imax] >= either of its neighbors,
    # and we use quadratic interpolation:
    # print(imax)
    c = y[imax]
    b = (y[imax+1]-y[imax-1])/2.0
    a = (y[imax+1]+y[imax-1])/2.0 - c
    yinterp = c - b*b/4.0/a
    xinterp = (x[1]-x[0])*(-b/2.0/a) + x[1]
    if extras:
        return yinterp, (a,b,c), imax, ymax
    else:
        return yinterp, xinterp


def colorline(
    x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0, ax=None):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """
    x = list(x)
    y = list(y)
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    if ax is None:
        ax = plt.gca()
    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments
    
def force_wlc(rel_ext, Plen=50):
    '''
    rel_ext : relative extension
    Plen : Persistence Length in nm
    '''
    kbT = 4.114 # unit: pN⋅nm
    sqt = 1 / (4 * (1 - rel_ext)**2)
    F = (kbT/Plen)*(sqt - 0.25 + rel_ext)
    return F


def force_wlc7(rel_ext, Lp):
    ### Use the model by Bouchiat, et al. Biophys J 76:409 (1999) ###
    # Given a vector of extensions and the parameter
    # # Lp = persistence length  (in micro-m) 
    # # Lc = contour length      (in micro-m) 
    # 
    # # T  = absolute temperature (in Kelvin) # This function returns the forces computed from a 
    # # 7 parameter model of the WLC, # using the model by Bouchiat, et al. Biophys J 76:409 (1999)

    T = 295 # 21.5 °C in Kelvin
    kT = 1.3806503 *10**(-23)*T/(10**(-18)) # k_B T in units of pN micro-m 
    # z_scaled = ext/Lc

    a  = np.zeros(7)
    a[0] = 1
    a[1] = -0.5164228
    a[2] = -2.737418
    a[3] = 16.07497
    a[4] = -38.87607
    a[5] = 39.49944
    a[6] = -14.17718


    Fwlc = 1/(4*(1 - rel_ext)**2)  - 1/4
    for i in range(len(a)):
        Fwlc += a[i] * rel_ext**(i+1) # the i+1 is due to the fact that python starts with i=0 (matlab starts with i+1)

    Fwlc = Fwlc * kT/Lp
    return Fwlc



def msd_moving(x, n=5):
    diff = np.diff(x)
    diff_sq = diff**2
    MSD = moving_average(diff_sq, n)
    return MSD


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def msd_lagtime_allpeaks(df_linked_peaks, pixelsize, fps, max_lagtime=100, axis=None):
    df_linked_peaks['y'] = 1
    imsd = trackpy.imsd(df_linked_peaks, mpp=pixelsize, fps=fps, max_lagtime=max_lagtime)
    if axis is None:
        fig, axis = plt.subplots()
    for col in imsd.columns:
        axis.plot(imsd.index, imsd[col], label=col)
    axis.set_yscale('log')
    axis.set_xscale('log')
    axis.set_xlabel('lag time [s]')
    axis.set_ylabel(r'$\angle \Delta r^2 \rangle$ [$\mu$m$^2$]')
    axis.legend()
    return imsd

@jit(nopython = True)
def msd_1d_nb1(x):
    result = np.zeros_like(x)
    for delta in range(1,len(x)):
        thisresult = 0
        for i in range(delta,len(x)):
            thisresult += (x[i] - x[i-delta])**2
        result[delta] = thisresult / (len(x) - delta)
    return result


RELATIVE_EXTENSION = np.array([0.23753, 0.26408, 0.37554, 0.49173, 0.63215, 0.70054, 0.73801,
                               0.76414, 0.77522, 0.78494, 0.79982, 0.81246, 0.82065, 0.82519,
                               0.83706, 0.84505, 0.84763, 0.85419, 0.86057, 0.8663 , 0.8697 ,
                               0.87641, 0.87783, 0.87796, 0.88391, 0.88784, 0.88949, 0.8944 ,
                               0.89656, 0.89834, 0.89997, 0.90261, 0.90348, 0.90675, 0.90767,
                               0.91085, 0.91311, 0.9126 , 0.91601, 0.91726, 0.91791, 0.91901,
                               0.92   , 0.92409, 0.92469, 0.92648, 0.92733, 0.92774, 0.92833,
                               0.93037])

FORCE = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. , 1.1, 1.2, 1.3,
                  1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2. , 2.1, 2.2, 2.3, 2.4, 2.5, 2.6,
                  2.7, 2.8, 2.9, 3. , 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9,
                  4. , 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5. ])


if __name__ == "__main__":
    print('kymograph module imported')
    