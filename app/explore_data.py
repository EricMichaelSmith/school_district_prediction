#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 14:36:33 2015

@author: Eric
"""

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats

import config
reload(config)
import join_data
reload(join_data)
import utilities
reload(utilities)



def main():
    """ Explore various aspects of the data. """
    
    plot_feature_histograms()
    plot_cross_correlations_wrapper()
    plot_pairwise_correlations(change=True)
    plot_pairwise_correlations(change=False)
    
    
    
def make_colorbar(ax, color_t_t, color_value_t, label_s):
    """ Creates a colorbar with the given axis handle ax; the colors are defined according to color_t_t and the values are mapped according to color_value_t. color_t_t and color_value_t must currently both be of length 3. The colorbar is labeled with label_s. """

    # Create the colormap for the colorbar    
    colormap = make_colormap(color_t_t, color_value_t)    
    
    # Create the colorbar
    norm = mpl.colors.Normalize(vmin=color_value_t[0], vmax=color_value_t[2])
    color_bar_handle = mpl.colorbar.ColorbarBase(ax, cmap=colormap,
                                               norm=norm,
                                               orientation='horizontal')
    color_bar_handle.set_label(label_s)
    
    
    
def make_colormap(color_t_t, color_value_t):
    """ Given colors defined in color_t_t and values defined in color_value_t, creates a LinearSegmentedColormap object. Works with only three colors and corresponding values for now. """
        
    # Find how far the second color is from the first and third
    second_value_fraction = float(color_value_t[1] - color_value_t[0]) / \
        float(color_value_t[2] - color_value_t[0])
    
    # Create the colormap
    color_s_l = ['red', 'green', 'blue']
    color_map_entry = lambda color_t_t, i_color: \
        ((0.0, color_t_t[0][i_color], color_t_t[0][i_color]),
         (second_value_fraction, color_t_t[1][i_color], color_t_t[1][i_color]),
         (1.0, color_t_t[2][i_color], color_t_t[2][i_color]))
    color_d = {color_s: color_map_entry(color_t_t, i_color) for i_color, color_s
              in enumerate(color_s_l)}
    colormap = LinearSegmentedColormap('ShapePlotColorMap', color_d)
    
    return colormap
    
    

def plot_cross_correlations(plot_data_a_d, plot_name):
    """ Plot the cross-correlations of all features, averaged over schools. """
    
    feature_s_l = sorted(plot_data_a_d.keys())
    fig = plt.figure(figsize=(2*len(feature_s_l), 2*len(feature_s_l)))
    for i, feature_i_s in enumerate(feature_s_l):
        for j, feature_j_s in enumerate(feature_s_l):
            xcorr_a = np.ndarray((plot_data_a_d[feature_i_s].shape[0],
                                  2*plot_data_a_d[feature_i_s].shape[1]-1))
            for k_row in range(plot_data_a_d[feature_i_s].shape[0]):
                xcorr_a[k_row, :] = np.correlate(plot_data_a_d[feature_i_s][k_row, :],
                                              plot_data_a_d[feature_j_s][k_row, :],
                                              mode='full')
            xcorr_a = xcorr_a[~np.any(np.isnan(xcorr_a), axis=1), :]
            print('{0} and {1}: {2:d} schools'.format(feature_i_s, feature_j_s, xcorr_a.shape[0]))
            plot_number = len(feature_s_l)*i + j + 1
            ax = fig.add_subplot(len(feature_s_l), len(feature_s_l), plot_number)
            ax.plot(np.mean(xcorr_a, axis=0))
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            if i == len(feature_s_l)-1:
                ax.set_xlabel(feature_j_s)
            if j == 0:
                ax.set_ylabel(feature_i_s)
    plt.savefig(os.path.join(save_path, plot_name + '.png'))
    
    
    
def plot_cross_correlations_control_wrapper():
    """ Plot the cross-correlations of fake data with the mean and standard deviation of observed features. """
    
    feature_s_l = sorted(data_a_d.keys())
    mean_d = {feature_s : np.mean(data_a_d[feature_s].reshape(-1))\
        for feature_s in feature_s_l} 
    std_d = {feature_s : np.std(data_a_d[feature_s].reshape(-1))\
        for feature_s in feature_s_l}
    fake_data_a_d = {}
    for feature_s in feature_s_l:
        fake_data_a_d[feature_s] = mean_d[feature_s] + \
            np.random.randn(1e5, data_a_d[feature_s].shape[1]) * std_d[feature_s]
    plot_cross_correlations(fake_data_a_d, 'cross_correlations_control')
    

    
def plot_cross_correlations_wrapper():
    """ Plot the cross-correlations of all features, averaged over schools. """
    
    plot_cross_correlations(data_a_d, 'cross_correlations')
            
    
    
def plot_feature_histograms():
    """ Plot histograms of all features. """

    con = utilities.connect_to_sql('joined')
    with con:    
        cur = con.cursor()
        for database_s in database_s_l:
            field_s_l = ['ENTITY_CD'] + \
                ['{0}_{1:d}'.format(database_s, year) for year in config.year_l]
            raw_data_a = utilities.select_data(con, cur, field_s_l, 'master',
                                      output_type='np_array')
            data_a = raw_data_a[:, 1:]
            valid_la = ~np.isnan(data_a)
                        
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for i, year in enumerate(config.year_l):
                col_a = data_a[:, i]
                ax.hist(col_a[valid_la[:, i]], bins=20,
                        color=config.year_plot_color_d[year],
                        histtype='step')
            ax.set_xlabel(database_s)
            ax.set_ylabel('Frequency')
            ax.ticklabel_format(useOffset=False)
            plt.savefig(os.path.join(save_path, database_s + '.png'))
            
            
            
def plot_pairwise_correlations(change=True):
    
    # Create heatmap from pairwise correlations
    heat_map_a = np.ndarray((len(data_a_d), len(data_a_d)))
    feature_name_s_l = sorted(data_a_d.keys(), reverse=True)
    for i_feature1, feature1_s in enumerate(feature_name_s_l):
        for i_feature2, feature2_s in enumerate(feature_name_s_l):
            if change:
                feature1_a = 2 * (data_a_d[feature1_s][:, -1] - data_a_d[feature1_s][:, 0]) / \
                    (data_a_d[feature1_s][:, -1] + data_a_d[feature1_s][:, 0])
                feature2_a = 2 * (data_a_d[feature2_s][:, -1] - data_a_d[feature2_s][:, 0]) / \
                    (data_a_d[feature2_s][:, -1] + data_a_d[feature2_s][:, 0])
                suffix_s = '_change'
            else:
                feature1_a = data_a_d[feature1_s][:, -1]
                feature2_a = data_a_d[feature2_s][:, -1]
                suffix_s = ''
            is_none_b_a = np.isnan(feature1_a) | \
                np.isnan(feature2_a)
            feature1_a = np.array(feature1_a[~is_none_b_a])
            feature2_a = np.array(feature2_a[~is_none_b_a])
            heat_map_a[i_feature1, i_feature2] = \
                stats.linregress(np.array(feature1_a.tolist()),
                                 np.array(feature2_a.tolist()))[2]
    
    # Create figure and heatmap axes
    fig = plt.figure(figsize=(10, 11))
    heatmap_ax = fig.add_axes([0.43, 0.10, 0.55, 0.55])
    
    # Show image
    color_t_t = ((1, 0, 0), (1, 1, 1), (0, 1, 0))
    max_magnitude = 1.0
    colormap = make_colormap(color_t_t, (-max_magnitude, 0, max_magnitude))
    heatmap_ax.imshow(heat_map_a,
                      cmap=colormap,
                      aspect='equal',
                      interpolation='none',
                      vmin=-max_magnitude,
                      vmax=max_magnitude)
    
    # Format axes
    heatmap_ax.xaxis.set_tick_params(labelbottom='off', labeltop='on')
    heatmap_ax.set_xlim([-0.5, len(feature_name_s_l)-0.5])
    heatmap_ax.set_xticks(range(len(feature_name_s_l)))
    heatmap_ax.set_xticklabels(feature_name_s_l, rotation=90)
    heatmap_ax.invert_xaxis()
    heatmap_ax.set_ylim([-0.5, len(feature_name_s_l)-0.5])
    heatmap_ax.set_yticks(range(len(feature_name_s_l)))
    heatmap_ax.set_yticklabels(feature_name_s_l)
    
    # Add colorbar
    color_ax = fig.add_axes([0.25, 0.06, 0.50, 0.02])
    color_bar_s = "Correlation strength (Pearson's r)"
    make_colorbar(color_ax, color_t_t, (-max_magnitude, 0, max_magnitude),
                  color_bar_s)
    
    plt.savefig(os.path.join(save_path, 'pearsons_r_heatmap' + suffix_s + '.png'))
    
    

save_path = os.path.join(config.plot_path, 'explore_data')
if not os.path.isdir(save_path):
    os.mkdir(save_path)

database_s_l = []
for Database in join_data.Database_l + join_data.DistrictDatabase_l:
    Instance = Database()
    database_s_l.append(Instance.new_table_s)
    
    
data_a_d = {}
con = utilities.connect_to_sql('joined')
with con:    
    cur = con.cursor()
    for database_s in database_s_l:
        field_s_l = ['ENTITY_CD'] + \
            ['{0}_{1:d}'.format(database_s, year) for year in config.year_l]
        raw_data_a = utilities.select_data(con, cur, field_s_l, 'master',
                                  output_type='np_array')
        data_a_d[database_s] = raw_data_a[:, 1:]
    
    

if __name__ == '__main__':
    main()