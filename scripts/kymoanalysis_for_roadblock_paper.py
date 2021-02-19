import sys
import numpy as np
from scipy.signal import savgol_filter
from PyQt5 import QtWidgets, QtCore, QtGui
import qdarkstyle
from leads.gui.kymograph_gui import Window
from leads import kymograph
import matplotlib.pyplot as plt


class NewWindow(Window):
    def __init__(self):
        super().__init__()

    def plottype_multipeak(self):
        left_peak_no = int(self.multipeak_dialog.leftpeak_num_combobox.currentText())
        right_peak_no = int(self.multipeak_dialog.rightpeak_num_combobox.currentText())
        df_gb = self.df_peaks_linked.groupby("particle")
        group_sel_col1 = df_gb.get_group(left_peak_no)
        group_sel_col1 = group_sel_col1.reset_index(drop=True)
        if self.numColors == "2" or "3":
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
                    frame_width = self.loop_region_right - self.loop_region_left,
                    dna_length=self.dna_length_kb, pix_width=self.dna_puncta_size,)
            # ax.plot(frames, msd_moving, 'g', label='color_1')
            if self.numColors == "2":
                msd_moving = kymograph.msd_moving(group_sel_col2['x'].values, n=n)
                frames = group_sel_col2['FrameNumber'].values[ind:-ind]
                ax.plot(frames* self.acquisitionTime, msd_moving, 'm', label='MSD particle')
                peak_analyzed_dict_sm = kymograph.analyze_maxpeak(group_sel_col2, smooth_length=7,
                    frame_width = self.loop_region_right - self.loop_region_left,
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
                              '.r', label='Position Differnece')                
                ax_right.plot(sel_loop_sm_dict['FrameNumber'] * self.acquisitionTime,
                              pos_diff_kb_smooth,
                              'r', label='Position Differnece')
                ax_right.set_ylabel("Kilobases", color="r")
                ax_right.tick_params(axis='y', colors='r')
                ax_right.spines["right"].set_color("r")
                ax_right.legend(loc='center right')

            ax.set_xlabel("Frame Number", color='m')
            ax.tick_params(axis='y', colors='m')
            ax.spines["left"].set_color("m")
            ax.set_ylabel("Moving MSD(" + str(n) + " points)")
            ax.legend()
            plt.show()
        elif self.multipeak_dialog.plottype_combobox.currentText() == "MSDsavgol":
            print("plot MSD savgol")
            _, ax = plt.subplots()
            n_savgol=self.multipeak_dialog.moving_window_spinbox.value()
            n = n_savgol
            if n_savgol%2 == 0:
                n_savgol = n_savgol + 1
            else:
                n = n+1
            n_order = 1
            n_savgol = 11
            # n_order = 1
            # n=12
            # if n%2 != 0:
            #     n = n+1
            ind = int(n/2)
            msd_moving = kymograph.msd_moving(group_sel_col1['x'].values, n=n)
            frames = group_sel_col1['FrameNumber'].values[ind:-ind]
            peak_analyzed_dict = kymograph.analyze_maxpeak(group_sel_col1, smooth_length=7,
                    frame_width = self.loop_region_right - self.loop_region_left,
                    dna_length=self.dna_length_kb, pix_width=self.dna_puncta_size,)
            # ax.plot(frames, savgol_filter(msd_moving, window_length=n_savgol, polyorder=n_order), 'g', label='color_1')
            if self.numColors == "2":
                msd_moving = kymograph.msd_moving(group_sel_col2['x'].values, n=n)
                frames = group_sel_col2['FrameNumber'].values[ind:-ind]
                ax.plot(frames * self.acquisitionTime, msd_moving, '.m', label='MSD particle')
                ax.plot(frames * self.acquisitionTime, savgol_filter(msd_moving, window_length=n_savgol, polyorder=n_order), 'm', label='MSD particle')
                peak_analyzed_dict_sm = kymograph.analyze_maxpeak(group_sel_col2, smooth_length=7,
                    frame_width = self.loop_region_right - self.loop_region_left,
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
                              '.r', label='Position Differnece')
                ax_right.plot(sel_loop_sm_dict['FrameNumber'] * self.acquisitionTime,
                              pos_diff_kb_smooth,
                              'r', label='Position Differnece')
                ax_right.set_ylabel("Distance/kb", color='r')
                ax_right.tick_params(axis='y', colors='r')
                ax_right.spines["right"].set_color("r")
                ax_right.legend(loc='center right')
            ax.set_xlabel("time/s")
            ax.tick_params(axis='y', colors='m')
            ax.spines["left"].set_color("m")
            ax.set_ylabel("MSD moving average(" + str(n) + " points)")
            ax.legend()
            plt.show()
        elif self.multipeak_dialog.plottype_combobox.currentText() == "MSDlagtime":
            print("plot MSD")
            _, ax = plt.subplots()
            msd = kymograph.msd_1d_nb1(group_sel_col1['x'].values)
            plt.plot(msd, 'g', label="MSD color-1")
            if self.numColors == "2" or "3":
                msd = kymograph.msd_1d_nb1(group_sel_col2['x'].values)
                plt.plot(msd, 'm', label="MSD color-1")
            ax.set_xlabel("Frame Number")
            ax.set_ylabel("MSD")
            ax.set_yscale('log')
            ax.legend()
            plt.show()
        elif self.multipeak_dialog.plottype_combobox.currentText() == "MSDlagtime-AllPeaks":
            if self.numColors == "1":
                fig,(ax1) = plt.subplots(nrows=1, ncols=1)
                _ = kymograph.msd_lagtime_allpeaks(self.df_peaks_linked,
                                pixelsize = self.pixelSize,
                                fps=int(self.numColors) * (1/self.acquisitionTime),
                                max_lagtime=100, axis=ax1)
                ax1.set_title("Color 1")
            elif self.numColors == "2" or "3":
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
            plt.show()
        elif self.multipeak_dialog.plottype_combobox.currentText() == "TimeTraceCol1":
            print("plot TimeTrace")
            _, ax = plt.subplots()
            trace_col1 = 7 * np.average(self.kymo_left_loop, axis=1)
            trace_col1_bg = trace_col1# - np.average(group_sel_col1["PeakIntensity"].values)
            ax.plot(group_sel_col1["FrameNumber"], group_sel_col1["PeakIntensity"], label="Peak")
            ax.plot(trace_col1_bg, label="Background")
            ax.set_xlabel("Frame Number")
            ax.set_ylabel("Intensity")
            ax.legend()
            plt.show()
        elif self.multipeak_dialog.plottype_combobox.currentText() == "TimeTraceCol2" and self.numColors == "2":
            print("plot TimeTrace")
            _, ax = plt.subplots()
            trace_col2 = 7 * np.average(self.kymo_right_loop, axis=1)
            trace_col2_bg = trace_col2# - np.average(group_sel_col2["PeakIntensity"].values)
            ax.plot(group_sel_col2["FrameNumber"], group_sel_col2["PeakIntensity"], label="Peak")
            ax.plot(trace_col2_bg, label="Background")
            ax.set_xlabel("Frame Number")
            ax.set_ylabel("Intensity")
            ax.legend()
            plt.show()

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = NewWindow()
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    win.show()
    sys.exit(app.exec_())
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtWidgets.QApplication.instance().exec_()

if __name__ == '__main__':
    main()