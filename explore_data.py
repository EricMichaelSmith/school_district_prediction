# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:44:57 2015

@author: Eric

Just visualize the data
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

import config
reload(config)
import select_data
reload(select_data)
import utilities
reload(utilities)



def main():
    
    con = utilities.connect_to_sql('joined')
    with con:
        cur = con.cursor()
        
        field_s_l = ['`AVG(percent_passing_{:d})`'.format(year) for year in config.year_l]
        data_a = select_data.main(con, cur, field_s_l, 'regents_pass_rate',
                                  output_type='np_array')
        data_df = pd.DataFrame(data_a, columns=config.year_l)
        
        # Delete NaN values, currently (as of 2007--2014 data) only from Greenburgh Eleven Union Free School / Greenburgh Eleven High School (660411020000 and 04)
        data_t_df = data_df.dropna().transpose()
        diff_df = data_t_df.diff().dropna()
        year_to_predict_diff_df = diff_df.iloc[-1, :]
        
        # Exploratory plot
#        plt.plot(data_df.columns, data_df.transpose())
#        plt.show()
        
        # Fit AR(1) to all data except for 2014, in order to predict it
        X = diff_df.iloc[:-1, :].as_matrix().reshape((-1, 1))
        X = sm.add_constant(X)
        AR1_Y = diff_df.iloc[1:, :].as_matrix().reshape((-1, 1))
        AR1_model = sm.OLS(AR1_Y, X)
        AR1_results = AR1_model.fit()
        print(AR1_results.params)
        last_fitted_year_diff_df = diff_df.iloc[-2, :]

        # Find MSE of fitted model
        X = last_fitted_year_diff_df.as_matrix()
        X = sm.add_constant(X)
        AR1_prediction_Y = AR1_results.predict(X)
        print('AR1:\n\t{:f}'.format(find_mse(year_to_predict_diff_df.as_matrix(),
                                         AR1_prediction_Y)))
        
        # Find MSE of model that assumes no change for any school in the coming year
        X = last_fitted_year_diff_df.as_matrix()
        X = sm.add_constant(X)
        no_change_Y = np.zeros(last_fitted_year_diff_df.as_matrix().shape)
        no_change_model = sm.OLS(no_change_Y, X)
        no_change_results = no_change_model.fit()
        print(no_change_results.params)
        no_change_prediction_Y = no_change_results.predict(X)
        print('No change from previous year:\n\t{:f}'.format(find_mse(year_to_predict_diff_df.as_matrix(),
                                                                  no_change_prediction_Y)))      
        
        
        
def find_mse(Y, prediction_Y):
    return np.sum((Y - prediction_Y)**2)