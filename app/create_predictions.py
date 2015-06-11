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
    
    con = utilities.connect_to_sql('joined')
    with con:
        cur = con.cursor()     
        field_s_l = ['ENTITY_CD'] + \
            ['`AVG(percent_passing_{:d})`'.format(year) for year in config.year_l]
        data_a = utilities.select_data(con, cur, field_s_l, 'regents_pass_rate',
                                  output_type='np_array')                          
    data_df = pd.DataFrame(data_a, columns=['ENTITY_CD'] + config.year_l)
    to_string = lambda x: '{:012.0f}'.format(x)
    data_df['ENTITY_CD'] = data_df['ENTITY_CD'].map(to_string)
    
    data_no_na_df = data_df.dropna()
    # Delete NaN values, currently (as of 2007--2014 data) only from Greenburgh Eleven Union Free School / Greenburgh Eleven High School (660411020000 and 04)
    data_t_df = data_no_na_df.iloc[:, 1:].transpose()
    # Take out ENTITY_CD and transpose the remaining data
    diff_df = data_t_df.diff().dropna()
    year_to_predict_diff_df = diff_df.iloc[-1, :]
    
    # Exploratory plot
#        plt.plot(data_df.columns, data_df.transpose())
#        plt.show()
    
    # Fit AR(1) to all data except for 2014, in order to predict it
    X = diff_df.iloc[:-1, :].as_matrix().reshape((-1, 1))
    X = sm.add_constant(X)
    AR1_diff_Y = diff_df.iloc[1:, :].as_matrix().reshape((-1, 1))
    AR1_diff_model = sm.OLS(AR1_diff_Y, X)
    AR1_diff_results = AR1_diff_model.fit()
    print(AR1_diff_results.params)
    last_fitted_year_diff_df = diff_df.iloc[-2, :]

    # Find MSE of fitted model
    X = last_fitted_year_diff_df.as_matrix()
    X = sm.add_constant(X)
    AR1_diff_prediction_a = AR1_diff_results.predict(X)
    print('AR1:\n\t{:f}'.format(find_mse(year_to_predict_diff_df.as_matrix(),
                                     AR1_diff_prediction_a)))
    
    # Find MSE of model that assumes no change for any school in the coming year
    X = last_fitted_year_diff_df.as_matrix()
    X = sm.add_constant(X)
    no_change_Y = np.zeros(last_fitted_year_diff_df.as_matrix().shape)
    no_change_model = sm.OLS(no_change_Y, X)
    no_change_results = no_change_model.fit()
    print(no_change_results.params)
    no_change_prediction_a = no_change_results.predict(X)
    print('No change from previous year:\n\t{:f}'\
          .format(find_mse(year_to_predict_diff_df.as_matrix(),
                           no_change_prediction_a)))
                                                              
    # Predict the test scores of the next year (currently 2015)
    X = diff_df.iloc[-1, :].as_matrix()
    # The latest year for which we have data
    X = sm.add_constant(X)
    AR1_diff_next_year_prediction_a = AR1_diff_results.predict(X)
    AR1_next_year_prediction_a = data_t_df.iloc[-1, :].as_matrix() + \
        AR1_diff_next_year_prediction_a
                                                                  
    # Save data in database
    new_column_s = 'avg_percent_passing_prediction_{:d}'.format(config.year_l[-1]+1)
    AR1_next_year_prediction_df = pd.DataFrame(AR1_next_year_prediction_a,
                                               index=data_no_na_df['ENTITY_CD'],
                                               columns=[new_column_s])
    utilities.write_to_sql_table(AR1_next_year_prediction_df,
                                 'regents_pass_rate_prediction', 'joined')
             

    ## Run regression models, validate and predict future scores, and run controls
    data_a = data_no_na_df[:, 1:].as_matrix()
    # Take out ENTITY_CD so that all columns are test scores
    last_fitted_year_actual_a = data_a[:, -2]
    last_data_year_actual_a = data_a[:, -1]
    
    diff_a = np.diff(data_a, n=1, axis=1)
    last_fitted_year_actual_diff_a = diff_a[:, -2]
    last_data_year_actual_diff_a = diff_a[:, -1]
    
    all_results_d = {}
    lag_l = range(1, 6)
    num_years_to_predict = 3
    
    # Run autoregression with different lags on raw test scores
    for lag in lag_l:
        model_s = 'raw_lag{:d}'.format(lag)
        results_d = {}
        
        results_d.result_object = perform_auto_regression(data_a[:, :-1], lag)
        last_fitted_year_prediction_a = \
            predict_given_auto_regression(data_a[:, :-2], results_d.result_object)
        results_d.last_fitted_year_mse = find_mse(last_fitted_year_actual_a,
                                                  last_fitted_year_prediction_a)
        last_data_year_prediction_a = \
            predict_given_auto_regression(data_a[:, :-1], results_d.result_object)
        results_d.last_data_year_mse = find_mse(last_data_year_actual_a,
                                                last_data_year_prediction_a)    
        future_prediction_a = np.ndarray((data_a.shape[0], 0))
        for year in range(num_years_to_predict):
            combined_data_a = np.concatenate((data_a, future_prediction_a), axis=1)
            new_year_prediction_a = predict_given_auto_regression(combined_data_a)
            future_prediction_a = np.concatenate((future_prediction_a,
                                                  new_year_prediction_a), axis=1)
        results_d.prediction_a = future_prediction_a
        
        all_results_d[model_s] = results_d
    
    # Run autogression with different lags on diff of test scores w.r.t. year
    for lag in lag_l:
        model_s = 'diff_lag{:d}'.format(lag)
        results_d = {}
        
        results_d.result_object = perform_auto_regression(diff_a[:, :-1], lag)
        last_fitted_year_prediction_a = \
            predict_given_auto_regression(diff_a[:, :-2], results_d.result_object)
        results_d.last_fitted_year_mse = find_mse(last_fitted_year_actual_diff_a,
                                                  last_fitted_year_prediction_a)
        last_data_year_prediction_a = \
            predict_given_auto_regression(diff_a[:, :-1], results_d.result_object)
        results_d.last_data_year_mse = find_mse(last_data_year_actual_diff_a,
                                                last_data_year_prediction_a)    
        future_prediction_a = np.ndarray((diff_a.shape[0], 0))
        for year in range(num_years_to_predict):
            combined_data_a = np.concatenate((diff_a, future_prediction_a), axis=1)
            new_year_prediction_a = predict_given_auto_regression(combined_data_a)
            future_prediction_a = np.concatenate((future_prediction_a,
                                                  new_year_prediction_a), axis=1)
        results_d.prediction_a = future_prediction_a
        
        all_results_d[model_s] = results_d
        
    # Run control: prediction is same as mean over years in training set
    model_s = 'mean_score_control'
    results_d = {}
    results_d.result_object = None
    results_d.last_fitted_year_mse = find_mse(last_fitted_year_actual_a,
                                              np.mean(data_a[:, :-1], axis=1))
    results_d.last_data_year_mse = find_mse(last_data_year_actual_a,
                                            np.mean(data_a[:, :-1], axis=1))
    results_d.prediction_a = np.tile(np.mean(data_a[:, 1:], axis=1),
                                     (1, num_years_to_predict))
 
    # Run control: prediction is same as previous year's data
    model_s = 'last_years_score_control'
    results_d = {}
    results_d.result_object = None
    results_d.last_fitted_year_mse = find_mse(last_fitted_year_actual_a,
                                              data_a[:, -2])
    results_d.last_data_year_mse = find_mse(last_data_year_actual_a,
                                            data_a[:, -2])
    results_d.prediction_a = np.tile(data_a[:, -1], (1, num_years_to_predict))
                   
    # {{{call perform_auto_regression a bunch of times, both on the differentiated and undifferentiated data, and then do four controls: mean of pass rate, median of pass rate, propagate last past rate, propagate last change in pass rate. For each of these, perform the regression on 2007--2013 and get back the results object; "predict" 2013 and calculate the MSE; "predict" 2014 and calculate the MSE; and predict 2015. Plot these 14 measures or at least some of them on a graph, but see which ones are highest/lowest.}}}



class AutoRegression(object):

    def __init__(self):
        pass

    def fit(self, array, options_d=None):
        """ Performs an auto-regression of a given lag on the input array. Axis 0 indexes observations (schools) and axis 1 indexes years. """

        lag = options_d['lag']        
        
        # Create model and fit parameters
        Y = array[:, lag:].reshape(-1)
        X = np.ndarray((Y.shape[0], 0))
        for i in range(lag):
            X = np.concatenate((X, array[:, i:-lag+i]).reshape(-1), axis=1)
        X = sm.add_constant(X)
        # Y = X_t = A_1 * X_(t-lag) + A_2 * X_(t-lag+1)) + ... + A_lag * X_(t-1) + A_(lag+1)
        model = sm.OLS(Y, X)
        results = model.fit()
        print('Lag of {0:d}:'.format(lag))
        print(results.params)
        
        return results
        
    def predict(self, array, results):
        """ Given the input results model, predicts the year of data immediately succeeding the last year of the input array. Axis 0 indexes observations (schools) and axis 1 indexes years. """
    
    lag = len(results.params)-1
    X = array[:, -lag:]
    X = sm.add_constant(X)
        
    return results.predict(X)
        
        
        
def find_mse(Y, prediction_Y):
    return np.sum((Y - prediction_Y)**2)
    
    
    
def fit_and_predict(array, Class, options_d={}):
    """ Given an array, fits it using Class and the optional options. """
    
    num_years_to_predict = 3    
    
    results_d = {}
        
    results_d.result_object = Class.fit(array[:, :-1], options_d)
    last_fitted_year_prediction_a = \
        Class.predict(array[:, :-2], results_d.result_object)
    results_d.last_fitted_year_mse = find_mse(data_a[:, -2],
                                              last_fitted_year_prediction_a)
    last_data_year_prediction_a = \
        Class.predict(array[:, :-1], results_d.result_object)
    results_d.last_data_year_mse = find_mse(data_a[:, -1],
                                            last_data_year_prediction_a)    
    future_prediction_a = np.ndarray((array.shape[0], 0))
    for year in range(num_years_to_predict):
        combined_data_a = np.concatenate((array, future_prediction_a), axis=1)
        new_year_prediction_a = Class.predict(combined_data_a, results_d.result_object)
        future_prediction_a = np.concatenate((future_prediction_a,
                                              new_year_prediction_a), axis=1)
    results_d.prediction_a = future_prediction_a
    
    return results_d
    
    
    
if __name__ == '__main__':
    main()