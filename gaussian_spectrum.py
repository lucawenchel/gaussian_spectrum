import numpy as np
from typing import Sequence, Tuple
import matplotlib.pyplot as plt
import tqdm
from utils import *


class GaussianSpectrum:

    # figure settings
    cm = 1 / 2.54
    figwidth = 20 * cm
    plt.rcParams['figure.figsize'] = [figwidth, figwidth]
    plt.rcParams['font.size'] = 20
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'STIXGeneral'

    def __init__(self, peak_list_path: str, min_ppm: Sequence, max_ppm: Sequence, pts: Sequence, ppm_sigma: Sequence):
        self.pl = read_from_xeasy(peak_list_path)
        self.inter_pl = self.filter_peaks_by_assignment(mode="inter")
        self.intra_pl = self.filter_peaks_by_assignment(mode="intra")
        self.spans = [(min_ppm[i], max_ppm[i]) for i in range(len(pts))]
        self.ppm_axes = [np.linspace(min_ppm[i], max_ppm[i], pts[i]) for i in range(len(pts))]
        self.ppm_sigma = ppm_sigma
        self.pts = pts

    def filter_peaks_by_assignment(self, mode="intra", res_numbers=np.empty(0)):

        # check if there are multiple chains, otherwise bomb out
        new_data = []
        assignments = []

        for i, p in enumerate(self.pl):
            assignment = self.pl.get_peak_assignment(i)

            if mode == "inter":

                if assignment[0][1] != assignment[1][1]:
                    new_data.append(p)
                    assignments.append(list(assignment))

            elif mode == "intra" and res_numbers.shape[0] == 0:

                if assignment[0][1] == assignment[1][1]:
                    new_data.append(p)
                    assignments.append(list(assignment))

            else:

                if (int(assignment[0][1]) in res_numbers) or (assignment[0][1] == assignment[1][1] and int(assignment[0][1]) in res_numbers):
                    new_data.append(p)
                    assignments.append(list(assignment))

        return PeakList(new_data, self.pl.axes_names, self.pl.unit, info={"ASSIGNMENT": np.array(assignments)})


    def gaussian_peak(self, ppm_coord: Sequence):

        gaussians = []

        for axis, coord, sigma in zip(self.ppm_axes, ppm_coord, self.ppm_sigma):

            gaussian = np.exp(- (axis - coord) * (axis - coord) / (2 * sigma * sigma))
            gaussians.append(gaussian)

        gaussian_peak = gaussians[0].reshape((len(gaussians[0]), 1)) @ gaussians[1].reshape((1, len(gaussians[1])))

        if np.max(gaussian_peak) == 0.:
            norm_gaussian_peak = np.zeros(self.pts)

        else:
            norm_gaussian_peak = gaussian_peak / np.max(gaussian_peak) * 10

        return norm_gaussian_peak


    def spectrum(self, mode: str):

        # preallocate gaussian spectrum
        gaussian_spectrum = np.zeros(self.pts)

        if mode == "inter":

            if len(self.inter_pl.data) == 0:
                print('No inter-chain interactions present.')
                return np.zeros(self.pts)

            else:

                for ppm_coord in tqdm.tqdm(self.inter_pl.data):
                    gaussian_spectrum += self.gaussian_peak(ppm_coord=ppm_coord)

                return gaussian_spectrum

        elif mode == "intra":

            for ppm_coord in tqdm.tqdm(self.intra_pl.data):
                gaussian_spectrum += self.gaussian_peak(ppm_coord=ppm_coord)

            return gaussian_spectrum

        else:
            raise ValueError("Mode is not avaiable. Available modes: intra, inter")


    def custom_spectrum(self, peak_coords: np.ndarray):

        # preallocate gaussian spectrum
        gaussian_spectrum = np.zeros(self.pts)

        for ppm_coord in tqdm.tqdm(peak_coords):
            gaussian_spectrum += self.gaussian_peak(ppm_coord=ppm_coord)

        return gaussian_spectrum




if __name__ == '__main__':
    peak_list_path_monomer = '/home/luca/Documents/project_3_darr_spectrum/humanlysc_monomer_DARR_intrares_exp.peaks'
    peak_list_path_fibril = '/home/luca/Documents/project_3_darr_spectrum/humanlysc_1layer_DARR_intrares_exp_no_chains.peaks'


    # peak_list_path_monomer = '/home/luca/Documents/project_3_darr_spectrum/humanlysc_monomer_NCA_intrares_exp.peaks'
    # peak_list_path_fibril = '/home/luca/Documents/project_3_darr_spectrum/humanlysc_1layer_NCA_intrares_exp.peaks'

    min_ppm = [0, 0]
    max_ppm = [140, 140]
    pts = [2048, 2048]
    gaussian_spectrum_monomer = GaussianSpectrum(peak_list_path=peak_list_path_monomer, min_ppm=min_ppm, max_ppm=max_ppm, pts=pts, ppm_sigma=[0.88, 0.88])
    gaussian_spectrum_fibril = GaussianSpectrum(peak_list_path=peak_list_path_fibril, min_ppm=min_ppm, max_ppm=max_ppm, pts=pts, ppm_sigma=[0.88, 0.88])


    ppm_axes = gaussian_spectrum_monomer.ppm_axes
    PPM1, PPM2 = np.meshgrid(*ppm_axes)

    res_to_mark_monomer = np.arange(0, 130)#np.concatenate((np.arange(0, 40), np.arange(85, 130)))
    res_to_mark_fibril = np.arange(0, 128)#np.arange(37+22, 80+22)

    # res_to_mark_complement = []
    # for res_num in np.arange(130):
    #     if res_num not in res_to_mark:
    #         res_to_mark_complement.append(res_num)
    # res_to_mark_complement = np.array(res_to_mark_complement)

    # intra_spectrum_bmrb = budget_spectrum_bmrb.spectrum(mode="machdasespasst")
    # monomer_spectrum = budget_spectrum_monomer.spectrum(mode="intra")
    # fibril_spectrum = budget_spectrum_fibril.spectrum(mode="intra")

    # custom spectra
    pl_to_mark_monomer = gaussian_spectrum_monomer.filter_peaks_by_assignment(mode="intra", res_numbers=res_to_mark_monomer)
    pl_to_mark_fibril = gaussian_spectrum_fibril.filter_peaks_by_assignment(mode="intra", res_numbers=res_to_mark_fibril)

    # spectrum_to_mark_monomer = budget_spectrum_monomer.custom_spectrum(pl_to_mark_monomer.data)
    spectrum_to_mark_fibril = gaussian_spectrum_fibril.custom_spectrum(pl_to_mark_fibril.data)

    fig, ax = plt.subplots()
    contour_levels = [1 * 1.2 ** n for n in range(0, 30)]


    # ax.contour(PPM1, PPM2, spectrum_to_mark_monomer, levels=contour_levels, colors='red', linewidths=0.25)
    ax.contour(PPM1, PPM2, spectrum_to_mark_fibril, levels=contour_levels, colors='red', linewidths=0.25)
    # ax.set_ylim([99, 136])
    # ax.set_xlim([44, 71])
    ax.set_ylim([4, 81])
    ax.set_xlim([4, 81])
    ax.invert_xaxis()
    ax.invert_yaxis()

    for spine in ax.spines.values():
        spine.set_linewidth(2)

    # ticks DARR spectrum
    ticks_with_annotations = list(np.arange(10, 81, 10))
    ticks_without_annotations = list(np.arange(5, 76, 10))
    ax.set_xlabel(r'$\mathrm{\delta_2(^{13}C)} \:/\: \mathrm{ppm}$')
    ax.set_ylabel(r'$\mathrm{\delta_1(^{13}C)} \:/\: \mathrm{ppm} $')
    ax.set_xticks(ticks_with_annotations)
    ax.set_xticks(ticks_without_annotations, minor=True)#, labels=[str(tick) if tick in ticks_with_annotations else "" for tick in ticks])
    ax.set_yticks(ticks_with_annotations)
    ax.set_yticks(ticks_without_annotations, minor=True)

    # # ticks NCA spectrum
    # yticks_with_annotations = list(np.arange(100, 131, 10))
    # yticks_without_annotations = list(np.arange(105, 136, 10))
    # xticks_with_annotations = list(np.arange(50, 71, 10))
    # xticks_without_annotations = list(np.arange(45, 66, 10))
    # ax.set_xlabel(r'$\mathrm{\delta_2(^{13}C)} \:/\: \mathrm{ppm}$')
    # ax.set_ylabel(r'$\mathrm{\delta_1(^{15}N)} \:/\: \mathrm{ppm} $')
    # ax.set_xticks(xticks_with_annotations)
    # ax.set_xticks(xticks_without_annotations, minor=True)#, labels=[str(tick) if tick in ticks_with_annotations else "" for tick in ticks])
    # ax.set_yticks(yticks_with_annotations)
    # ax.set_yticks(yticks_without_annotations, minor=True)

    ax.tick_params(which='major', length=10, width=2)
    ax.tick_params(which='minor', length=5, width=2)

    fig.savefig("/home/luca/Documents/project_3_darr_spectrum/DARR_spectrum_fibril_intra_all.pdf") # monomer_1to40_87to146

    plt.show()





