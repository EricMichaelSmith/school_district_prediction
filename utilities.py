"""
Provides utility functions for school_district_prediction
"""

import pymysql as mdb



def connect_to_sql(database_s):
    """ Connect to MySQL server. Returns connection object and cursor. """

    con = mdb.connect('localhost', 'root', '', database_s) #host, user, password, #database
    
    return con