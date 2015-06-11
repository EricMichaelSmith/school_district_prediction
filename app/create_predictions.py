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
                                 
    # {{{call perform_auto_regression a bunch of times, both on the differentiated and undifferentiated data, and then do four controls: mean of pass rate, median of pass rate, propagate last past rate, propagate last change in pass rate. Plot these 14 measures or at least some of them on a graph, but see which ones are highest/lowest.}}}

        
        
def find_mse(Y, prediction_Y):
    return np.sum((Y - prediction_Y)**2)
    
    
    
def perform_auto_regression(df, lag):
    """ Performs an auto-regression of lag lag on DataFrame df. """
    
    Y = df.iloc[lag:, :].as_matrix().reshape(-1)
    X = np.ndarray((Y.shape[0], 0))
    for i in range(lag):
        X = np.concatenate((X, df.iloc[i:-lag+i]), axis=1)
    X = sm.add_constant(X)
    # Y = X_t = A_1 * X_(t-lag) + A_2 * X_(t-lag+1)) + ... + A_lag * X_(t-1) + A_(lag+1)
    model = sm.OLS(Y, X)
    results = model.fit()
    print('Lag of {0:d}:'.format(lag))
    print(results.params)
    # {{{output: the prediction for the current year, the output for the next year, and the coefficients. Remember that, when you're assessing the validity of the models, you'll have to set Y to be *2013* to train so that you can test the models on 2014 data.}}}
    
    
    
if __name__ == '__main__':
    main()