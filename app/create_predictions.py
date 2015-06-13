#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:44:57 2015

@author: Eric

Just visualize the data
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas
reload(pandas)
import pandas as pd
import statsmodels.api as sm

import config
reload(config)
import utilities
reload(utilities)



def main():
    
    
    ## Read in data
    con = utilities.connect_to_sql('joined')
    with con:
        cur = con.cursor()     
        field_s_l = ['ENTITY_CD'] + \
            ['`AVG(fraction_passing_{:d})`'.format(year) for year in config.year_l]
        data_a = utilities.select_data(con, cur, field_s_l, 'regents_pass_rate',
                                  output_type='np_array')                          
    data_df = pd.DataFrame(data_a, columns=['ENTITY_CD'] + config.year_l)
    to_string = lambda x: '{:012.0f}'.format(x)
    data_df['ENTITY_CD'] = data_df['ENTITY_CD'].map(to_string)
    
    
    ## Run regression models, validate and predict future scores, and run controls    
    data_no_na_df = data_df.dropna()
    # Delete NaN values, currently (as of 2007--2014 data) only from Greenburgh Eleven Union Free School / Greenburgh Eleven High School (660411020000 and 04)    
    data_a = data_no_na_df.iloc[:, 1:].as_matrix()
    # Take out ENTITY_CD so that all columns are test scores    
    
    all_results_d = {}
    lag_l = range(1, 6)
    
    # Run autoregression with different lags on raw test scores
    for lag in lag_l:
        model_s = 'raw_lag{:d}'.format(lag)
        all_results_d[model_s] = fit_and_predict(data_a, AutoRegression,
                                                 diff=False, lag=lag)
    
    # Run autogression with different lags on diff of test scores w.r.t. year
    for lag in lag_l:
        model_s = 'diff_lag{:d}'.format(lag)
        all_results_d[model_s] = fit_and_predict(data_a, AutoRegression,
                                                 diff=True, lag=lag)

    # Run control: prediction is same as mean over years in training set
    model_s = 'mean_over_years_score_control'
    all_results_d[model_s] = fit_and_predict(data_a, MeanOverYears)
    
    # Run control: prediction is same as previous year's data
    model_s = 'same_as_last_year_score_control'
    all_results_d[model_s] = fit_and_predict(data_a, SameAsLastYear)
    
    # Run control: prediction is same as previous year's data
    model_s = 'same_change_as_last_year_score_control'
    all_results_d[model_s] = fit_and_predict(data_a, SameChangeAsLastYear)
    
    all_mses_d = {key: value['last_data_year_mse'] for (key, value) in all_results_d.iteritems()}
    for key, value in all_mses_d.iteritems():
        print('{0}: \n\t{1:1.5g}'.format(key, value))
    
    
    ## Plot MSEs of all regression models
    model_s_l = ['raw_lag1', 'raw_lag2', 'raw_lag3', 'raw_lag4', 'raw_lag5', 'diff_lag1', 'diff_lag2', 'diff_lag3', 'diff_lag4', 'diff_lag5', 'mean_over_years_score_control', 'same_as_last_year_score_control', 'same_change_as_last_year_score_control']
    mse_l = [all_mses_d[key] for key in model_s_l]
    fig = plt.figure()
    ax = fig.add_subplot(111) 
    ax.bar(range(len(mse_l)), mse_l)
    ax.set_title('Comparison of MSE of autoregression algorithms vs. controls')
    ax.set_xticks(np.arange(len(mse_l)))
    ax.set_xticklabels(model_s_l, rotation=45)
    ax.set_ylabel('Mean squared error')
    ax.axhline(y=all_mses_d['mean_over_years_score_control'], color='r')
    plt.savefig('mse_all_models.png')
                   
    ## {{{Then, save them all to a database.}}}
                   
                   
    ## Exploratory plots
    fig = plt.figure()
    ax = fig.add_subplot(111)                   
    ax.plot(config.year_l, data_a.transpose()*100)
    ax.set_xlabel('Year')
    ax.set_ylabel('Percent passing Regents exam\n(averaged over subjects)')
    ax.set_ylim([0, 100])
    ax.ticklabel_format(useOffset=False)
    plt.savefig('all_schools.png')
    
    ## Exploratory plots
    significant_rising_la = (data_a[:, -1] - data_a[:, 0] > 0.2)
    to_plot_a = data_a[significant_rising_la, :].transpose()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(config.year_l, to_plot_a*100)
    ax.set_xlabel('Year')
    ax.set_ylabel('Percent passing Regents exam\n(averaged over subjects)')
    ax.set_ylim([0, 100])
    ax.ticklabel_format(useOffset=False)
    plt.savefig('significant_rising.png')
    
    ## Exploratory plots
    significant_falling_la = (data_a[:, -1] - data_a[:, 0] < -0.2)
    to_plot_a = data_a[significant_falling_la, :].transpose()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(config.year_l, to_plot_a*100)
    ax.set_xlabel('Year')
    ax.set_ylabel('Percent passing Regents exam\n(averaged over subjects)')
    ax.set_ylim([0, 100])
    ax.ticklabel_format(useOffset=False)
    plt.savefig('significant_falling.png')



class AutoRegression(object):

    def __init__(self):
        pass

    def fit(self, raw_array, diff=False, lag=1):
        """ Performs an auto-regression of a given lag on the input array. Axis 0 indexes observations (schools) and axis 1 indexes years. """

        # Apply optional parameters
        if diff:
            array = np.diff(raw_array, 1, axis=1)
        else:
            array = raw_array
        
        # Create model and fit parameters
        Y = array[:, lag:].reshape(-1)
        X = np.ndarray((Y.shape[0], 0))
        for i in range(lag):
            X = np.concatenate((X, array[:, i:-lag+i].reshape(-1, 1)), axis=1)
        X = sm.add_constant(X)
        # Y = X_t = A_1 * X_(t-lag) + A_2 * X_(t-lag+1)) + ... + A_lag * X_(t-1) + A_(lag+1)
        model = sm.OLS(Y, X)
        results = model.fit()
        print('Lag of {0:d}:'.format(lag))
        print(results.params)
        
        return results
        
    def predict(self, raw_array, results, diff=False, lag=1):
        """ Given the input results model, predicts the year of data immediately succeeding the last year of the input array. Axis 0 indexes observations (schools) and axis 1 indexes years. """
        
        if diff:
            array = np.diff(raw_array, 1, axis=1)
            X = array[:, -lag:]
            X = sm.add_constant(X)
            predicted_change_a = results.predict(X)
            prediction_a = raw_array[:, -1] + predicted_change_a
        else:
            array = raw_array
            X = array[:, -lag:]
            X = sm.add_constant(X)
            prediction_a = results.predict(X)            
            
        return prediction_a.reshape((-1, 1))
    
    
    
class MeanOverYears(object):
    
    def __init__(self):
        pass
    
    def fit(self, array, **kwargs):
        return None
        
    def predict(self, array, results, **kwargs):
        mean_over_years_a = np.mean(array, axis=1)
        return mean_over_years_a.reshape((-1, 1))



class SameAsLastYear(object):
    
    def __init__(self):
        pass
    
    def fit(self, array, **kwargs):
        return None
        
    def predict(self, array, results, **kwargs):
        last_year_a = array[:, -1]
        return last_year_a.reshape((-1, 1))
        
        
        
class SameChangeAsLastYear(object):
    
    def __init__(self):
        pass
    
    def fit(self, array, **kwargs):
        return None
        
    def predict(self, array, results, **kwargs):
        change_over_last_year_a = array[:, -1] - array[:, -2]
        extrapolation_from_last_year_a = array[:, -1] + change_over_last_year_a
        return extrapolation_from_last_year_a.reshape((-1, 1))
        
        
        
def find_mse(Y, prediction_Y):
    return np.sum((Y - prediction_Y)**2)
    
    
    
def fit_and_predict(array, Class, **kwargs):
    """ Given an array, fits it using Class and the optional options. """
    
    num_years_to_predict = 3    
    
    instance = Class()
    results_d = {}
        
    results_d['result_object'] = instance.fit(array[:, :-1], **kwargs)
    last_fitted_year_prediction_a = \
        instance.predict(array[:, :-2], results_d['result_object'],
                         **kwargs)
    results_d['last_fitted_year_mse'] = find_mse(array[:, -2],
                                              last_fitted_year_prediction_a)
    last_data_year_prediction_a = \
        instance.predict(array[:, :-1], results_d['result_object'],
                         **kwargs)
    results_d['last_data_year_mse'] = find_mse(array[:, -1],
                                            last_data_year_prediction_a)    
    future_prediction_a = np.ndarray((array.shape[0], 0))
    for year in range(num_years_to_predict):
        combined_data_a = np.concatenate((array, future_prediction_a), axis=1)
        new_year_prediction_a = instance.predict(combined_data_a,
                                                 results_d['result_object'],
                                                 **kwargs)
        future_prediction_a = np.concatenate((future_prediction_a,
                                              new_year_prediction_a), axis=1)
    results_d['prediction_a'] = future_prediction_a
    
    return results_d
    
    
    
if __name__ == '__main__':
    main()