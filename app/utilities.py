"""
Provides utility functions for school_district_prediction
"""

import numpy as np
import pymysql as mdb
from sqlalchemy import create_engine

import config_unsynced

    
    
def connect_to_sql(database_s):
    """ Connect to MySQL server. Returns connection object and cursor. """

    con = mdb.connect('localhost', 'root', config_unsynced.pword_s, database_s) #host, user, password, #database
    
    return con
    


def select_data(con, cur, field_s_l, table_s, output_type='np_array'):
    """
    Selects fields specified in field_s_l from table table_s; returns a dictionary where each field is a key.
    """
    
    # Grab data in the form of a tuple of dictionaries
    command_s = 'SELECT '
    for field_s in field_s_l:
        command_s += field_s + ', '
    command_s = command_s[:-2] + ' FROM ' + table_s + ';'
#    print(command_s)
    cur.execute(command_s)
    output_t_t = cur.fetchall()
    
    # Convert to appropriate type
    if output_type == 'np_array':
        
        fields = np.ndarray(shape=(len(output_t_t), len(field_s_l)))
        for l_row, output_t in enumerate(output_t_t):
            fields[l_row, :] = output_t
    
    else:
        raise ValueError('Output type not recognized.')
    
    return fields
    
    
    
def write_to_sql_table(df, table_s, database_s):
    """ Writes DataFrame df as a new table table_s in SQL. Assumes no existing connection to a MySQL database!"""
    
    engine = create_engine('mysql+pymysql://root:{0}@localhost/'\
                           .format(config_unsynced.pword_s) + \
                           database_s,
                           echo=False)
    engine.execute('DROP TABLE IF EXISTS {0};'.format(table_s))
    cnx = engine.raw_connection()
    df.to_sql(name=table_s, con=cnx, flavor='mysql', if_exists='replace')