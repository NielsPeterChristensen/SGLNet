# -*- coding: utf-8 -*-
"""
Library for manual basic threshold segmentation.

@author: niels, 2025
"""


# =============================================================================
# Packages
# =============================================================================


import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
import tkinter as tk
import SGLNet.Plotting.plot_utils as plot_utils
import SGLNet.Plotting.cmap_func as cmap_func
import skimage as ski


# =============================================================================
# Classes
# =============================================================================


class BasicThresholdSegmentation:
    """Class for manual threshold segmentation on unwrapped images"""
    
    def __init__(self, unwrapped_image: np.ndarray, 
                 wrapped_image: np.ndarray = None, min_area: int = 100):
        """
        Parameters
        ----------
        unwrapped_image : np.ndarray
            Unwrapped phase image.
        wrapped_image : np.ndarray, optional
            Wrapped phase image. The default is None
        min_area : int, optional
            Minimum pixel area for lowpass filter. The default is 100.
        """
        self.unwrapped_image = unwrapped_image
        self.wrapped_image = wrapped_image
        self.min_area = min_area
        self._pre_proc()
        self.set_threshold()
        self._intermediate_proc()
        self.select_labels()
        self._post_proc()
        
    def __repr__(self):
        return ("BasicThresholdSegmentation object for manual segmentation.\n"
                f"  Image shape = {self.unwrapped_image.shape}\n"
                f"  Number of regions = {self.count}\n"
                f"  Region areas = {[region.area for region in self._regionprops]}\n"
                f"  Positive threshold = {self.positive_threshold}\n"
                f"  Negative threshold = {self.negative_threshold}\n")
    
    @property
    def attributes(self):
        """Return class attributes"""
        keys = [key for key in self.__dict__.keys()
                if not key.startswith(('_', '__'))]
        return keys
        
    def _pre_proc(self):
        """
        Pre-processing method. Applies Gaussian blur, decomposes image
        into positive and negative phase components, makes first-order 
        threshold estimate with Otsu two-peak, and subsequently runs 
        self._iterative_proc_positive() and self._iterative_proc_negative().
        """
        #-- Gaussian blur
        self.blurred_image = ski.filters.gaussian(self.unwrapped_image)
        #-- Decompose positive and negative
        self._pos_mask = self.blurred_image >= 0
        self._neg_mask = self.blurred_image < 0
        pos = np.copy(self.blurred_image)
        neg = np.copy(self.blurred_image)
        pos[self._neg_mask] = np.nan
        neg[self._pos_mask] = np.nan
        self.positive_component = pos
        self.negative_component = neg
        del pos
        del neg
        #-- Otsu two-peak automatic threshold
        self.positive_threshold = ski.filters.threshold_otsu(
            self.positive_component[self._pos_mask])
        self.negative_threshold = ski.filters.threshold_otsu(
            self.negative_component[self._neg_mask])
        #-- Mask, filter, label and make contours
        self._iterative_proc_positive()
        self._iterative_proc_negative()
        
    def _iterative_proc_positive(self):
        """
        Processing of positive phase component for iterative refinement of
        threshold mask. Computes binary mask from threshold, applies
        lowpass filter, labels individual regions, and extracts contours.
        """
        #-- Binary threshold mask
        self.positive_mask = self.positive_component > self.positive_threshold
        #-- Lowpass filter
        self.positive_mask = ski.morphology.remove_small_objects(
            self.positive_mask, min_size=self.min_area)
        #-- Label
        self._positive_labels, self._positive_count = ski.measure.label(
            self.positive_mask, connectivity=2, return_num=True)
        #-- Contours
        self._positive_contours = ski.measure.find_contours(
            self._positive_labels, level=0.5)
        
    def _iterative_proc_negative(self):
        """
        Processing of negative phase component for iterative refinement of
        threshold mask. Computes binary mask from threshold, applies
        lowpass filter, labels individual regions, and extracts contours.
        """
        #-- Binary threshold mask
        self.negative_mask = self.negative_component < self.negative_threshold
        #-- Lowpass filter
        self.negative_mask = ski.morphology.remove_small_objects(
            self.negative_mask, min_size=self.min_area)
        #-- Label
        self._negative_labels, self._negative_count = ski.measure.label(
            self.negative_mask, connectivity=2, return_num=True)
        #-- Contours
        self._negative_contours = ski.measure.find_contours(
            self._negative_labels, level=0.5)
            
    def set_threshold(self):
        """Display interactive canvas for threshold selection."""
        fig, ax = plt.subplots()
        plot_utils.plot_maximized()
        plt.subplots_adjust(bottom=0.25)
        if self.wrapped_image is None:
            _ = self._plot_phu(ax)
        else:
            _ = self._plot_phr(ax)
        self._plot_positive_contours(ax)
        self._plot_negative_contours(ax)
        #-- Add sliders
        ax_slider1 = plt.axes([0.1, 0.05, 0.6, 0.03], facecolor='lightgray')
        slider1 = Slider(ax_slider1, 'Threshold 1', 0, np.nanmax(self.positive_component), valinit=self.positive_threshold, valstep=0.01)
        ax_slider2 = plt.axes([0.1, 0.0, 0.6, 0.03], facecolor='lightgray')
        slider2 = Slider(ax_slider2, 'Threshold 2', np.nanmin(self.negative_component), 0, valinit=self.negative_threshold, valstep=0.01)
        #-- Link sliders to update functions
        slider1.on_changed(lambda val: self._update_pos(val, fig, ax, slider1))
        slider2.on_changed(lambda val: self._update_neg(val, fig, ax, slider2))
        #-- Get user input
        self._proceed = False
        ax_button = plt.axes([0.8, 0.025, 0.1, 0.075])
        Button(ax_button, 'Continue')
        fig.canvas.mpl_connect('button_press_event', lambda event: self._on_click1(event, ax_button, fig))
        plot_utils.TkAgg_focus()
        #-- Wait until user continues
        while self._proceed is False:
            plt.pause(0.1)
    
    def _on_click1(self, event, button_axes, fig):
        """Action when button is clicked in interactive threshold canvas."""
        if event.inaxes == button_axes:
            self._proceed = True
            plt.close(fig)
            
    def _update_pos(self, val, fig, ax, slider):
        """
        Action when positive threshold slider is changed in 
        interactive canvas.
        """
        self.positive_threshold = slider.val
        self._iterative_proc_positive()
        for line in ax.lines: line.remove()
        self._plot_positive_contours(ax)
        self._plot_negative_contours(ax)
        fig.canvas.draw_idle()
        
    def _update_neg(self, val, fig, ax, slider):
        """
        Action when negative threshold slider is changed
        in interactive canvas.
        """
        self.negative_threshold = slider.val
        self._iterative_proc_negative()
        for line in ax.lines: line.remove()
        self._plot_positive_contours(ax)
        self._plot_negative_contours(ax)
        fig.canvas.draw_idle()  
        
    def _intermediate_proc(self):
        """
        Intermediate processing after final threshold selection.
        Unifies binary masks, labels individual regions, and extracts
        morphometric properties.
        """
        #-- Combine binary masks
        self.combined_mask = np.bitwise_or(
            self.positive_mask, self.negative_mask)
        #-- Label
        self._all_labels, self._total_count = ski.measure.label(
            self.combined_mask, connectivity=2, return_num=True)
        #-- Morphometrics
        self._regionprops = ski.measure.regionprops(self._all_labels)

    def select_labels(self):
        """Display interactive canvas for segmentation region selection."""
        color_labels = ski.color.label2rgb(self._all_labels)
        fig, ax = plt.subplots()
        plot_utils.plot_maximized()
        ax.imshow(color_labels)
        plot_utils.remove_axis_elements(ax)
        ax.set_title(f'Total number of regions = {self._total_count}')
        #-- Overlay the area of each region
        for region in self._regionprops:
            y, x = region.centroid
            area = region.area
            ax.text(x, y, f'{area}', color='white', fontsize=8, ha='center', va='center', 
                    bbox=dict(facecolor='black', edgecolor='none', alpha=0.7, boxstyle='round'))
        #-- Define new variables
        self.click_x = []
        self.click_y = []
        self.click_label = []
        #-- Get user input
        self._proceed = False
        ax_button = plt.axes([0.4, 0.025, 0.2, 0.075])
        Button(ax_button, 'Continue')
        fig.canvas.mpl_connect('button_press_event', lambda event: self._on_click2(event, ax, ax_button, fig))
        plot_utils.TkAgg_focus()
        #-- Wait until user continues
        while self._proceed is False:
            plt.pause(0.1)

    def _on_click2(self, event, plot_axes, button_axes, fig):
        """
        Action when button is clicked in interactive segmentation 
        region selection canvas.
        """
        if event.inaxes == button_axes:
            self._proceed = True
            plt.close(fig)  
        if event.inaxes == plot_axes:
            #-- Get click coordinates
            x, y = int(event.xdata), int(event.ydata)
            #-- Get corresponding label
            click_label = self._all_labels[y, x]
            #-- Store input
            if click_label == 0:
                print(f'Pixel ({x}, {y}) is not within a valid region!' \
                      + 'Try again.')
            else:
                self.click_x.append(x)
                self.click_y.append(y)
                self.click_label.append(click_label)
                print(f"Pixel ({x}, {y}) with label ({click_label}) stored!")
                plot_axes.plot(x, y, 'wx')
            
    def _post_proc(self):
        """
        Post processing after final segmentationr regions are selected.
        Labels individual regions, counts the final number of regions,
        extracts binary mask, and computes contours.
        """
        #-- Filter labels
        self.labels = np.where(np.isin(self._all_labels, self.click_label), 
                               self._all_labels, 0)
        #-- Update morphometrics
        self._regionprops = ski.measure.regionprops(self.labels)
        #-- Update count
        self.count = len(self.click_label)
        #-- Update binary mask
        self.mask = self.labels > 0
        #-- Update contour
        self.contours = ski.measure.find_contours(self.labels, level=0.5)

    def _plot_positive_contours(self, ax):
        """Plot contour for positive phase component segmentations in red."""
        lines = []
        for contour in self._positive_contours:
            line, = ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='k')
            lines.append(line)
        self._positive_lines = lines
        del lines
    
    def _plot_negative_contours(self, ax):
        """Plot contour for negative phase component segmentations in blue."""
        lines = []
        for contour in self._negative_contours:
            line, = ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='k')
            lines.append(line)
        self._negative_lines = lines
        del lines

    def _plot_phu(self, ax):
        """Plot unwrapped phase with asymmetric colormap"""
        asym_seismic = cmap_func.asymmetric_cmap(self.blurred_image, plt.get_cmap('seismic'))
        im = ax.imshow(self.blurred_image, cmap=asym_seismic)
        return im
    
    def _plot_phr(self, ax):
        """Plot wrapped phase with gist_rainbow colormap"""
        im = ax.imshow(self.wrapped_image, cmap='gist_rainbow')
        return im