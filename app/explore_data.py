#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 14:36:33 2015

@author: Eric
"""

import matplotlib.pyplot as plt
import numpy as np
import os

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
    plot_cross_correlations_control_wrapper()
    
    

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
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for i, year in enumerate(config.year_l):
                ax.hist(data_a[:, i], bins=20, color=config.year_plot_color_d[year],
                        histtype='step')
            ax.set_xlabel(database_s)
            ax.set_ylabel('Frequency')
            ax.ticklabel_format(useOffset=False)
            plt.savefig(os.path.join(save_path, database_s + '.png'))
    
    

save_path = os.path.join(config.plot_path, 'explore_data')
if not os.path.isdir(save_path):
    os.mkdir(save_path)

database_s_l = []
for Database in join_data.Database_l:
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