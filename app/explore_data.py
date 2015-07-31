#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 14:36:33 2015

@author: Eric

Perform descriptive analysis of the data and create various plots, without worrying about prediction.
"""

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import signal, stats

import config
reload(config)
import join_data
reload(join_data)
import utilities
reload(utilities)



def main():
    """ Explore various aspects of the data. """

#    plot_feature_histograms()
    plot_cross_correlations_wrapper()
#    plot_pairwise_correlations(change=True)
#    plot_pairwise_correlations(change=False)
#    make_change_scatter_plot('student_retention_rate', 'teacher_number')
#    make_change_scatter_plot('budget', 'discount_lunch')



def make_change_scatter_plot(feature1_s, feature2_s):
    """ Make a scatter plot of the change in one feature over time vs. the change in another feature over time """

    feature1_a = 2 * (data_a_d[feature1_s][:, -1] - data_a_d[feature1_s][:, 0]) / \
                    (data_a_d[feature1_s][:, -1] + data_a_d[feature1_s][:, 0])
    feature2_a = 2 * (data_a_d[feature2_s][:, -1] - data_a_d[feature2_s][:, 0]) / \
                    (data_a_d[feature2_s][:, -1] + data_a_d[feature2_s][:, 0])
    all_database_stats_d = join_data.collect_database_stats()

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_axes(0.1, 0.1, 0.8, 0.8)
    ax = make_scatter_plot(ax, feature1_a, feature2_a, 'k', plot_regression_b=True,
                           print_stats_b=True)
    ax.set_xlabel(all_database_stats_d[feature1_s]['description_s'])
    ax.set_ylabel(all_database_stats_d[feature2_s]['description_s'])
    plt.savefig(os.path.join(save_path, 'change_scatter_plot.png'))



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


def make_scatter_plot(ax, x_a, y_a, color_t, plot_axes_at_zero_b=False,
                                               plot_regression_b=False,
                                               print_stats_b=False):
    """
    Creates a scatter plot.
    """

    # Plot all data
    ax.scatter(x_a, y_a,
                    c=color_t,
                    edgecolors='none')

    # Plot x=0 and y=0 lines
    if plot_axes_at_zero_b:
        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')

    # Plot regression line (one set of points only)
    if plot_regression_b:
        plot_regression(ax, x_a, y_a)

    if print_stats_b:
        slope, intercept, r_value, p_value, std_err = \
            stats.linregress(np.array(x_a.tolist()),
                             np.array(y_a.tolist()))
        ax.set_title('r-value = %0.2f, p-value = %0.3g' % \
                     (r_value, p_value))

    return ax



def plot_cross_correlations(plot_data_a_d, plot_name):
    """ Plot the cross-correlations of all features, averaged over schools. """

    feature_s_l = sorted(plot_data_a_d.keys())
    fig = plt.figure(figsize=(2*len(feature_s_l), 2*len(feature_s_l)))
    for i, feature_i_s in enumerate(feature_s_l):
        for j, feature_j_s in enumerate(feature_s_l):

            # For each school, standardize its time trace
            num_years = plot_data_a_d[feature_i_s].shape[1]
            per_school_mean_1_a = np.nanmean(plot_data_a_d[feature_i_s], axis=1).reshape(-1, 1)
            per_school_std_1_a = np.nanstd(plot_data_a_d[feature_i_s], axis=1).reshape(-1, 1)
            standard1_a = (plot_data_a_d[feature_i_s] -
                           np.tile(per_school_mean_1_a, (1, num_years))) / \
                          np.tile(per_school_std_1_a, (1, num_years))
            per_school_mean_2_a = np.nanmean(plot_data_a_d[feature_j_s], axis=1).reshape(-1, 1)
            per_school_std_2_a = np.nanstd(plot_data_a_d[feature_j_s], axis=1).reshape(-1, 1)
            standard2_a = (plot_data_a_d[feature_j_s] -
                           np.tile(per_school_mean_2_a, (1, num_years))) / \
                          np.tile(per_school_std_2_a, (1, num_years))

            # Simulate iid data
            simulated1_a = np.random.normal(0, 1,
                (1e4, standard1_a.shape[1]))
            if i == j:
                simulated2_a = simulated1_a
            else:
                simulated2_a = np.random.normal(0, 1,
                    (1e4, standard2_a.shape[1]))

            xcorr_a = np.ndarray((standard1_a.shape[0],
                                  2*standard1_a.shape[1]-1))
            simulated_xcorr_a = np.ndarray((simulated1_a.shape[0],
                                            2*simulated1_a.shape[1]-1))
            for k_row in range(standard1_a.shape[0]):
                xcorr_a[k_row, :] = np.correlate(standard1_a[k_row, :],
                                              standard2_a[k_row, :],
                                              mode='full')
            for k_row in range(simulated1_a.shape[0]):
                simulated_xcorr_a[k_row, :] = \
                    np.correlate(simulated1_a[k_row, :],
                                 simulated2_a[k_row, :],
                                 mode='full')

            xcorr_a = xcorr_a[~np.any(np.isnan(xcorr_a), axis=1), :]

            # Calculate if correlation is significant
            _, ttest_p_a = stats.ttest_ind(xcorr_a, simulated_xcorr_a, axis=0)

            # Correct for the size of the overlap
            len_trace = standard1_a.shape[1]
            normalization_a = np.array(range(1, len_trace+1) +
                                       range(len_trace-1, 0, -1)).astype(float)
            norm_mean_xcorr_a = np.mean(xcorr_a, axis=0) / normalization_a
            norm_mean_simulated_xcorr_a = np.mean(simulated_xcorr_a, axis=0) / normalization_a

            print('{0} and {1}: {2:d} schools'.format(feature_i_s, feature_j_s, xcorr_a.shape[0]))
            plot_number = len(feature_s_l)*i + j + 1
            ax = fig.add_subplot(len(feature_s_l), len(feature_s_l), plot_number)
            for x, val in enumerate(ttest_p_a):
                if val < 1e-6:
                    ax.axvspan(x-0.5, x+0.5, facecolor=(1,0.8,0.8), alpha=1,
                               linewidth=0)
                elif val < 1e-4:
                    ax.axvspan(x-0.5, x+0.5, facecolor=(1,0.9,0.9), alpha=1,
                               linewidth=0)
                elif val < 1e-2:
                    ax.axvspan(x-0.5, x+0.5, facecolor=(1,0.95,0.95), alpha=1,
                               linewidth=0)
            ax.plot(np.zeros(norm_mean_xcorr_a.shape), color='k')
            ax.plot(norm_mean_simulated_xcorr_a, color='r')
            ax.plot(norm_mean_xcorr_a, color='b')
            ax.set_ylim([1.2*min(np.concatenate((norm_mean_xcorr_a,
                                                    norm_mean_simulated_xcorr_a),
                                                   axis=1)),
                          1.2*max(np.concatenate((norm_mean_xcorr_a,
                                                    norm_mean_simulated_xcorr_a),
                                                   axis=1))])
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
    """ Create a colored correlation matrix between every pair of features """

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
            slope, intercept, r_value, p_value, std_err = \
                stats.linregress(np.array(feature1_a.tolist()),
                                 np.array(feature2_a.tolist()))
            heat_map_a[i_feature1, i_feature2] = r_value
            print('{0} and {1}:\n\tr-value {2:0.3f}, p-value {3:0.3g}\n\t{4:d} schools matched'.format(feature1_s, feature2_s, r_value, p_value, sum(is_none_b_a)))

    # Create figure and heatmap axes
    fig = plt.figure(figsize=(10, 11))
    heatmap_ax = fig.add_axes([0.43, 0.10, 0.55, 0.55])

    # Show image
    color_t_t = ((1, 0, 0), (1, 1, 1), (0, 0, 1))
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



def plot_regression(ax, x_a, y_a):
    """
    Plots a regression line on the current plot. The stretching of the
    regression line and resetting of the axis limits is kind of a hack.
    """

    # Find correct axis limits
    axis_limits_t = ax.axis()

    # Calculate and plot regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_a, y_a)
    plt.plot([2*axis_limits_t[0], 2*axis_limits_t[1]],
             [slope*2*axis_limits_t[0]+intercept,
              slope*2*axis_limits_t[1]+intercept],
             'r')

    # Reset axis limits
    ax.set_xlim(axis_limits_t[0], axis_limits_t[1])
    ax.set_ylim(axis_limits_t[2], axis_limits_t[3])

    return ax



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
