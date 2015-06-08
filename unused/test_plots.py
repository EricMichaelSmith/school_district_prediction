# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 18:06:01 2015

@author: Eric
"""

from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def main():
    
    # Reading in data
    base_data_path = \
        '/Users/Eric/Dropbox/Computing/Personal/Code/school_district_prediction/' + \
        'data/report_cards/'
    year_l = range(2012, 2015)
    selected_rows_df_d = {}
    for year in year_l:
        file_name = 'Regents Examination Annual Results.csv'
        data_path = base_data_path + 'SRC' + str(year) + \
            '_Exported/' + file_name
        raw_df = pd.read_csv(data_path)
        selected_rows_df_d[year] = raw_df[(raw_df['SUBJECT'] == 'REG_ENG') & \
        (raw_df['YEAR'] == year) & \
        (raw_df['SUBGROUP_NAME'] == 'All Students') & \
        (raw_df['PER_65-84'] != 's') & (raw_df['PER_85-100'] != 's') & \
        (raw_df['ENTITY_NAME'] == raw_df['ENTITY_NAME'].str.upper())]
        # The last requirement is to filter out entries that don't correspond to valid school districts, based on the assumption that valid school districts are in all caps. I need to check this.
        selected_rows_df_d[year]['PER_65-100'] = \
            selected_rows_df_d[year]['PER_65-84'].convert_objects(convert_numeric=True) + \
            selected_rows_df_d[year]['PER_85-100'].convert_objects(convert_numeric=True)
#    print(selected_rows_df_d[year_l[0]][:5])
#    print('Number of matching rows: %d' % len(selected_rows_df_d[year_l[0]]))
    
    
    # Merging
    merged_df = selected_rows_df_d[year_l[0]]
    for i, year in enumerate(year_l[1:]):
        year_i = i+1
        merged_df = pd.merge(merged_df, selected_rows_df_d[year], how='inner', \
        on='ENTITY_CD', left_index=True, \
        suffixes=('_{!s}'.format(year_l[year_i-1]), '_{!s}'.format(year)))
    if 'PER_65-100' in merged_df.columns:
        merged_df.rename(columns={'PER_65-100':'PER_65-100_{!s}'.format(year_l[-1])},
                         inplace=True)
    # Take care of the case when there is an odd number of years
    merged_df = merged_df.set_index('ENTITY_CD')
    
    
    # Plotting: all time series
    time_series_l_l = []
    print(merged_df.index)
    print(merged_df.columns)
    for i in merged_df.index:
        time_series_l = []
        for year in year_l:
            time_series_l.append(merged_df.loc[i, 'PER_65-100_{!s}'.format(year)])
        time_series_l_l.append(time_series_l)
    for time_series_l in time_series_l_l:
            print(year_l)
            print(time_series_l)
            plt.plot(year_l, time_series_l, c=np.random.rand(3))
    plt.xlabel('Year')
    plt.ylabel('Percent passing English Regents exam')
    ax = plt.gca()
    ax.ticklabel_format(useOffset=False)
    plt.show()
        
    

if __name__ == '__main__':
    main()