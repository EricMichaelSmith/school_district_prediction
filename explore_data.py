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
        data_df = data_df.dropna()
        
        # Exploratory plot
#        plt.plot(data_df.columns, data_df.transpose())
#        plt.show()
        
        # AR(1)
        X = data_df.transpose()[:-1].as_matrix().reshape((-1, 1))
        X = sm.add_constant(X)
        Y = data_df.transpose()[1:].as_matrix().reshape((-1, 1))
        model = sm.OLS(Y, X)
        results = model.fit()
        print(results.params)
        print(results.tvalues)