#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:44:57 2015

@author: Eric

Just visualize the data
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import statsmodels.api as sm

import config
reload(config)
import join_data
reload(join_data)
import utilities
reload(utilities)



def main():
    
    PrimaryFeature = join_data.RegentsPassRate()
    primary_feature_s = PrimaryFeature.new_table_s
    
    
    ## Read in data    
    con = utilities.connect_to_sql('joined')
    with con:
        cur = con.cursor() 
        field_s_l = ['ENTITY_CD'] + \
            ['{0}_{1:d}'.format(primary_feature_s, year) for year in config.year_l]
        raw_data_a = utilities.select_data(con, cur, field_s_l, 'master',
                                  output_type='np_array')
        aux_data_a_d = {}
        aux_Database_l = join_data.Database_l
        for Database in aux_Database_l:
            Instance = Database()
            feature_s = Instance.new_table_s
            if feature_s != primary_feature_s:
                field_s_l = ['ENTITY_CD'] + \
                    ['{0}_{1:d}'.format(feature_s, year) for year in config.year_l]
                aux_data_a_d[feature_s] = utilities.select_data(con, cur, field_s_l,
                                                                'master',
                                                                output_type='np_array')

    
    ## Format data
    data_a = raw_data_a[:, 1:]
    print(data_a.shape[0])
    # Drop the ENTITY_CD column
    for feature_s in aux_data_a_d.iterkeys():
        aux_data_a_d[feature_s] = aux_data_a_d[feature_s][:, 1:]
        print(aux_data_a_d[feature_s].shape[0])
    
    
    ## Run regression models, validate and predict future scores, and run controls    
    all_results_d = {}
    lag_l = range(1, 6)
    
    # Run autoregression with different lags on raw test scores
    for lag in lag_l:
        model_s = 'raw_lag{:d}'.format(lag)
        all_results_d[model_s] = fit_and_predict(data_a, AutoRegression,
                                                 aux_data_a_d=aux_data_a_d,
                                                 diff=False, lag=lag)
    
    # Run autogression with different lags on diff of test scores w.r.t. year
    for lag in lag_l:
        model_s = 'diff_lag{:d}'.format(lag)
        all_results_d[model_s] = fit_and_predict(data_a, AutoRegression,
                                                 aux_data_a_d=aux_data_a_d,
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

    all_train_mses_d = {key: value['last_fitted_year_mse'] for (key, value) in all_results_d.iteritems()}    
    all_test_mses_d = {key: value['last_data_year_mse'] for (key, value) in all_results_d.iteritems()}
    for key, value in all_test_mses_d.iteritems():
        print('{0}: \n\t{1:1.5g}'.format(key, value))
    
    
    ## Plot MSEs of all regression models     
    model_s_l = ['raw_lag1', 'raw_lag2', 'raw_lag3', 'raw_lag4', 'raw_lag5', 'diff_lag1', 'diff_lag2', 'diff_lag3', 'diff_lag4', 'diff_lag5', 'mean_over_years_score_control', 'same_as_last_year_score_control', 'same_change_as_last_year_score_control']
    train_mse_l = [all_train_mses_d[key] for key in model_s_l]
    test_mse_l = [all_test_mses_d[key] for key in model_s_l]
    bar_width = 0.35
    train_index_a = np.arange(len(train_mse_l))
    test_index_a = np.arange(bar_width, len(test_mse_l)+bar_width)
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.3, 0.8, 0.6]) 
    ax.bar(train_index_a, train_mse_l, bar_width, color='r', label='Training')
    ax.bar(test_index_a, test_mse_l, bar_width, color='b', label='Test')
    ax.set_title('Comparison of MSE of autoregression algorithms vs. controls')
    ax.set_xticks(np.arange(len(test_mse_l))+bar_width)
    ax.set_xticklabels(model_s_l, rotation=90)
    ax.set_ylabel('Mean squared error')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12))
    ax.axhline(y=all_test_mses_d['mean_over_years_score_control'], color='k')
    plt.savefig(os.path.join(config.plot_path, 'mse_all_models.png'))
               
               
    ## Exploratory plots
    fig = plt.figure()
    ax = fig.add_subplot(111)                   
    ax.plot(config.year_l, data_a.transpose()*100)
    ax.set_xlabel('Year')
    ax.set_ylabel('Percent passing Regents exam\n(averaged over subjects)')
    ax.set_ylim([0, 100])
    ax.ticklabel_format(useOffset=False)
    plt.savefig(os.path.join(config.plot_path, 'all_schools.png'))
    
    significant_rising_la = (data_a[:, -1] - data_a[:, 0] > 0.2)
    to_plot_a = data_a[significant_rising_la, :].transpose()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(config.year_l, to_plot_a*100)
    ax.set_xlabel('Year')
    ax.set_ylabel('Percent passing Regents exam\n(averaged over subjects)')
    ax.set_ylim([0, 100])
    ax.ticklabel_format(useOffset=False)
    plt.savefig(os.path.join(config.plot_path, 'significant_rising.png'))
    
    significant_falling_la = (data_a[:, -1] - data_a[:, 0] < -0.2)
    to_plot_a = data_a[significant_falling_la, :].transpose()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(config.year_l, to_plot_a*100)
    ax.set_xlabel('Year')
    ax.set_ylabel('Percent passing Regents exam\n(averaged over subjects)')
    ax.set_ylim([0, 100])
    ax.ticklabel_format(useOffset=False)
    plt.savefig(os.path.join(config.plot_path, 'significant_falling.png'))
    
    
    ## Save data to the SQL database
    model_to_save_s = 'raw_lag1'
    new_column_s_l = ['{0}_prediction_{1:d}'.format(primary_feature_s, year) \
                      for year in config.prediction_year_l]
    prediction_df = pd.DataFrame(all_results_d[model_to_save_s]['prediction_a'],
                                 index=raw_data_a[:, 0],
                                 columns=new_column_s_l)
    utilities.write_to_sql_table(prediction_df,
                                 '{0}_prediction'.format(primary_feature_s), 'joined')    



# Let's think about applying statsmodels' ARIMA model later: I'm thinking that the way to do this is to bring in all data except for one high school's time series data using the "exog" keyword, but I don't think that's truly cross-sectional: "exog" seems to be more useful for predicting future test pass rates from past past rates and funding level, for example, not from past pass rates of the school in question and all other schools. And Des seems to be right that vector ARIMA isn't the way to go here, because it seems like each sample in the cross-section still gets its own fitted variables, right?
#class ARIMA(object):
#    
#    def __init__(self):
#        pass
#
#    def fit(self, raw_array, order_t=(0,0,0)):
#        """ Performs an ARIMA on raw_array given (p,d,q) as order_t. Axis 0 indexes observations (schools) and axis 1 indexes years. """
#
#        results = sm.tsa.ARIMA(raw_array.transpose(), order_t).fit()
#        print('(p, d, q) = ({0:d}, {0:d}, {0:d}):'.format(order_t[0],
#                                                          order_t[1],
#                                                          order_t[2]))
#        print(results.params)
#        
#        return results
#                
#    def predict(self, raw_array, results, order_t=(0,0,0))
#        """ Given the input results model, predicts the year of data immediately succeeding the last year of the input array. Axis 0 indexes observations (schools) and axis 1 indexes years. """
#
#        # {{{}}}        
        
    

class AutoRegression(object):

    def __init__(self):
        pass

    def fit(self, raw_array, aux_data_a_d=None, diff=False, lag=1):
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
            # Y = X_t = A_1 * X_(t-lag) + A_2 * X_(t-lag+1)) + ... + A_lag * X_(t-1) + A_(lag+1)
        if aux_data_a_d:
            for feature_s in aux_data_a_d.iterkeys():
                if diff:
                    array = np.diff(aux_data_a_d[feature_s], 1, axis=1)
                else:
                    array = aux_data_a_d[feature_s]
                for i in range(lag):
                    X = np.concatenate((X, array[:, i:-lag+i].reshape(-1, 1)), axis=1)
        X = sm.add_constant(X)
        
        model = sm.OLS(Y, X)
        results = model.fit()
        print('Lag of {0:d}:'.format(lag))
        print(results.params)
        
        return results
        
    def predict(self, raw_array, results, aux_data_a_d=None, diff=False, lag=1):
        """ Given the input results model, predicts the year of data immediately succeeding the last year of the input array. Axis 0 indexes observations (schools) and axis 1 indexes years. """
        
        if diff:
            array = np.diff(raw_array, 1, axis=1)
            X = array[:, -lag:]
            if aux_data_a_d:
                for feature_s in aux_data_a_d.iterkeys():
                    array = np.diff(aux_data_a_d[feature_s], 1, axis=1)
                    X = np.concatenate((X, array[:, -lag:]), axis=1)
            X = sm.add_constant(X)
            predicted_change_a = results.predict(X)
            prediction_a = raw_array[:, -1] + predicted_change_a
        else:
            array = raw_array
            X = array[:, -lag:]
            if aux_data_a_d:
                for feature_s in aux_data_a_d.iterkeys():
                    array = aux_data_a_d[feature_s]
                    X = np.concatenate((X, array[:, -lag:]), axis=1)
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
    
    num_years_to_predict = len(config.prediction_year_l)    
    
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