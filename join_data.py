"""
Join all data in SQL databases
"""

import os

import config
reload(config)
import utilities
reload(utilities)



def main():

    # Read file of source paths
    with open(os.path.join(config.code_path, 'data_sources.txt'), 'r') as f:
        data_source_path_l = f.read().split('\n')

    con = utilities.connect_to_sql('joined')
    with con:
        cur = con.cursor()      
        
        # {{{call join_regents_pass_rate}}}



def join_regents_pass_rate():
    """ Join the percentage of students passing the Regents exam """
            
    # {{{do this}}}
            
            

if __name__ == '__main__':
    main()