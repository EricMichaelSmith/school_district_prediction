#!/usr/bin/env python
"""
Join all data in SQL databases
"""

import numpy as np
import os
import pandas as pd


import config
reload(config)
import utilities
reload(utilities)



def main():
    
    find_school_key()    
    
    for Database in Database_l:
        instance = Database()
        con = utilities.connect_to_sql('temp')    
        with con:
            cur = con.cursor()
            for year in config.year_l:
                instance.extract(cur, year)
        con = utilities.connect_to_sql('joined')
        with con:
            cur = con.cursor()
            join_years(cur, instance.new_table_s, 'ENTITY_CD')
            
    for Database in DistrictDatabase_l:
        instance = Database()
        for year in config.year_l:
            instance.extract(year)
        con = utilities.connect_to_sql('joined')
        with con:
            cur = con.cursor()
            join_years(cur, instance.new_table_s, 'district')
            
    con = utilities.connect_to_sql('joined')
    with con:
        cur = con.cursor()
        join_databases(cur, Database_l, DistrictDatabase_l)
        
        
        
def collect_database_stats():
    all_Database_l = Database_l + DistrictDatabase_l
    all_database_stats_d = {}
    for Database in all_Database_l:
        instance = Database()
        d = {}
        d['description_s'] = instance.description_s
        d['explanatory_name'] = instance.explanatory_name
        d['multiplier'] = instance.multiplier
        d['range_l'] = instance.range_l
        d['output_format_1_s'] = instance.output_format_1_s
        d['output_format_2_s'] = instance.output_format_2_s
        d['new_table_s'] = instance.new_table_s
        d['orig_table_s_d'] = instance.orig_table_s_d
        d['allow_prediction'] = instance.allow_prediction
        d['in_metric'] = instance.in_metric
        d['metric_weight'] = instance.metric_weight
        d['bar_plot_s'] = instance.bar_plot_s
        all_database_stats_d[instance.new_table_s] = d
    return all_database_stats_d



def extract_table_name(Class):
    Instance = Class()
    return Instance.new_table_s



def find_school_key():
    """ Creates a table of each school ID and name """
    
    con = utilities.connect_to_sql('joined')    
    with con:
        cur = con.cursor()
        command_s = 'DROP TABLE IF EXISTS school_key;'
        cur.execute(command_s)
        command_s = """CREATE TABLE school_key
SELECT ENTITY_CD, ENTITY_NAME FROM SRC{0:d}.`{1}`
WHERE YEAR = {0:d}
AND SUBJECT = 'REG_ENG'
AND SUBGROUP_NAME = 'General Education'
AND ENTITY_CD NOT LIKE '%0000'
AND ENTITY_CD NOT LIKE '00000000000%'
AND ENTITY_CD != '111111111111'
AND ENTITY_CD != '240901040001'
AND ENTITY_CD != '241001060003'"""
        # The REG_ENG is kind of a hack; and I had to remove 240901040001 and 241001060003 because the rows were multiplying exponentially in the database like a virus
        instance = RegentsPassRate()
        command_s = command_s.format(config.year_l[-1],
                                     instance.orig_table_s_d[config.year_l[-1]])
        cur.execute(command_s)
        command_s = """ALTER TABLE school_key ADD district CHAR(6)"""
        cur.execute(command_s)
        command_s = """UPDATE school_key SET district = SUBSTRING(ENTITY_CD, 1, 6);"""
        cur.execute(command_s)
        command_s = """ALTER TABLE school_key
ADD INDEX ENTITY_CD (ENTITY_CD)"""
        cur.execute(command_s)
        
        

def join_databases(cur, Database_l, DistrictDatabase_l):
    """ Join all databases """
    
    master_database_s = 'master'
    individual_database_s_l = [extract_table_name(Database) for Database in Database_l]
    individual_district_database_s_l = [extract_table_name(Database) for Database in DistrictDatabase_l]
    
    print('Starting join_databases')
    
    cur.execute('DROP TABLE IF EXISTS {0}'.format(master_database_s))
    command_s = """CREATE TABLE {0}
SELECT * FROM school_key""".format(master_database_s)
    for individual_database_s in individual_database_s_l:
        this_table_command_s = """
LEFT JOIN (SELECT ENTITY_CD_{0}, """.format(individual_database_s)
        for year in config.year_l:
            this_table_command_s += '{0}_{1:d}, '.format(individual_database_s, year)
        this_table_command_s = this_table_command_s[:-2]
        this_table_command_s += """ FROM {0}) AS {0}
ON school_key.ENTITY_CD = {0}.ENTITY_CD_{0}"""
        this_table_command_s = this_table_command_s.format(individual_database_s)
        command_s += this_table_command_s
    for individual_database_s in individual_district_database_s_l:
        this_table_command_s = """
LEFT JOIN (SELECT district_{0}, """.format(individual_database_s)
        for year in config.year_l:
            this_table_command_s += '{0}_{1:d}, '.format(individual_database_s, year)
        this_table_command_s = this_table_command_s[:-2]
        this_table_command_s += """ FROM {0}) AS {0}
ON school_key.district = {0}.district_{0}"""
        this_table_command_s = this_table_command_s.format(individual_database_s)
        command_s += this_table_command_s
    command_s += ';'
    cur.execute(command_s)
    
    print('Database {0} created.'.format(master_database_s))
        
        
        
def join_years(cur, new_table_s, join_s):
    """ Join separate years of a database that was just extracted into the temp database """
    
    print('Starting join_years for {0}'.format(new_table_s))
    
    cur.execute('DROP TABLE IF EXISTS {0};'.format(new_table_s))
    command_s = """CREATE TABLE {0}
SELECT * FROM school_key""".format(new_table_s)
    for year in config.year_l:
        this_table_command_s = """
INNER JOIN temp.temp{0:d}_final
ON school_key.{1} = temp.temp{0:d}_final.{1}_{0:d}"""
        this_table_command_s = this_table_command_s.format(year, join_s)
        command_s += this_table_command_s
    command_s += ';'
    cur.execute(command_s)
    command_s = """ALTER TABLE {0} CHANGE {1} {1}_{0} CHAR(12);"""
    cur.execute(command_s.format(new_table_s, join_s))
    
    print('Database {0} created.'.format(new_table_s))
    
    
    
class Budget(object):
    """ Yearly budget """
    
    def __init__(self):
        self.description_s = 'Annual budget of school district\n(thousands of dollars)'
        self.explanatory_name = 'Budget'
        self.multiplier = 0.001
        self.range_l = [0, np.inf]
        self.output_format_1_s = 'school district budget'
        self.output_format_2_s = '{:0.0f} thousand dollars'
        self.new_table_s = 'budget'
        self.orig_table_s_d = {year:'NYSDOB_Enacted_SchoolAid__{0:d}'.format(year-1)
                               for year in range(2007, 2015)}
        self.orig_table_s_d[2011] = 'NYSDOB_Enacted_SchoolAid__2011'
        self.column_s = {year:'{0:d}-{1:02.0f}'.format(year-1, year-2000)
                         for year in range(2007, 2015)}
        self.allow_prediction = True
        self.in_metric = False
        self.metric_weight = None
        self.bar_plot_s = None
        
                             
    def extract(self, year):

        print('Creating {0} for year {1:d}'.format(self.new_table_s, year))

        df = pd.read_excel(os.path.join(config.data_path, 'budgets',
                                        self.orig_table_s_d[year] + '.xlsx'))
        filtered_df = df[df['Aid Category'] == 'Sum of Above Aid Categories']
        trimmed_df = filtered_df.loc[:, ['BEDS Code', self.column_s[year]]]
        trimmed_df.columns = ['district_{0}'.format(year),
                              '{1}_{0:d}'.format(year, self.new_table_s)]
        final_df = trimmed_df.set_index('district_{0}'.format(year))
        utilities.write_to_sql_table(final_df, 'temp{0:d}_final'.format(year),
                                     'temp')
    
    

class DiscountLunch(object):
    """ Fraction of students receiving reduced or free lunch """
    
    def __init__(self):
        self.description_s = 'Students receiving reduced-price or free lunch (%)'
        self.explanatory_name = 'Reduced or free lunch'
        self.multiplier = 100
        self.range_l = [0, 1]
        self.output_format_1_s = 'percent of students receiving reduced-price or free lunch'
        self.output_format_2_s = '{:0.0f}%'
        self.new_table_s = 'discount_lunch'
        self.orig_table_s_d = {year:'Demographic Factors' for year in range(2007, 2015)}
        self.allow_prediction = True
        self.in_metric = False
        self.metric_weight = None
        self.bar_plot_s = None
        
        
    def extract(self, cur, year):
        """ Returns an N-by-2 of the ENTITY_CD and value """
        
        assert(year >= 2007)        
        
        print('Creating {0} for year {1:d}'.format(self.new_table_s, year))
        
        command_s = 'DROP TABLE IF EXISTS temp{0:d};'
        cur.execute(command_s.format(year))
        command_s = 'CREATE TABLE temp{0:d} SELECT * FROM SRC{0:d}.`{1}`;'
        cur.execute(command_s.format(year, self.orig_table_s_d[year]))
        command_s = """DELETE FROM temp{0:d}
WHERE PER_FREE_LUNCH = 's' OR PER_REDUCED_LUNCH = 's';"""
        cur.execute(command_s.format(year))
        command_s = """ALTER TABLE temp{0:d} CHANGE ENTITY_CD ENTITY_CD_{0:d} CHAR(12);"""
        cur.execute(command_s.format(year))
        command_s = """ALTER TABLE temp{0:d} ADD {1}_{0:d} FLOAT(12);"""
        cur.execute(command_s.format(year, self.new_table_s))
        command_s = """UPDATE temp{0:d}
SET {1}_{0:d} = (PER_FREE_LUNCH + PER_REDUCED_LUNCH) / 100;"""
        cur.execute(command_s.format(year, self.new_table_s))
            
        command_s = 'DROP TABLE IF EXISTS temp{0:d}_final;'
        cur.execute(command_s.format(year))
        command_s = """CREATE TABLE temp{0:d}_final
SELECT ENTITY_CD_{0:d}, {1}_{0:d} FROM temp{0:d}
WHERE YEAR = {0:d};"""
        cur.execute(command_s.format(year, self.new_table_s))
        command_s = """ALTER TABLE temp{0:d}_final
ADD INDEX ENTITY_CD_{0:d} (ENTITY_CD_{0:d});"""
        cur.execute(command_s.format(year))
        
        
class Dropout(object):
    """ Fraction of students who dropped out of school """
    
    def __init__(self):
        self.description_s = 'Dropout rate (%)'
        self.explanatory_name = 'Dropout rate'
        self.multiplier = 100
        self.range_l = [0, 1]
        self.output_format_1_s = 'dropout rate'
        self.output_format_2_s = '{:0.1f}%'
        self.new_table_s = 'dropout'
        self.orig_table_s_d = {year:'High School Noncompleters' for year in range(2007, 2015)}
        self.allow_prediction = True
        self.in_metric = True
        self.metric_weight = -1
        self.bar_plot_s = 'Dropout\nrate (%)'
        
        
    def extract(self, cur, year):
        """ Returns an N-by-2 of the ENTITY_CD and value """
        
        assert(year >= 2007)        
        
        print('Creating {0} for year {1:d}'.format(self.new_table_s, year))
        
        command_s = 'DROP TABLE IF EXISTS temp{0:d};'
        cur.execute(command_s.format(year))
        command_s = """CREATE TABLE temp{0:d} SELECT * FROM SRC{0:d}.`{1}`
WHERE YEAR = {0:d} AND SUBGROUP_NAME = 'General Education' AND PER_DROPOUT != 's';"""
        cur.execute(command_s.format(year, self.orig_table_s_d[year]))
        command_s = """ALTER TABLE temp{0:d} CHANGE ENTITY_CD ENTITY_CD_{0:d} CHAR(12);"""
        cur.execute(command_s.format(year))
        command_s = """ALTER TABLE temp{0:d} ADD {1}_{0:d} FLOAT(12);"""
        cur.execute(command_s.format(year, self.new_table_s))
        command_s = """UPDATE temp{0:d} SET {1}_{0:d} = PER_DROPOUT / 100;"""
        cur.execute(command_s.format(year, self.new_table_s))
        command_s = 'DROP TABLE IF EXISTS temp{0:d}_final;'
        cur.execute(command_s.format(year))
        command_s = """CREATE TABLE temp{0:d}_final
SELECT ENTITY_CD_{0:d}, {1}_{0:d} FROM temp{0:d}
WHERE YEAR = {0:d};"""
        cur.execute(command_s.format(year, self.new_table_s))
        command_s = """ALTER TABLE temp{0:d}_final
ADD INDEX ENTITY_CD_{0:d} (ENTITY_CD_{0:d});"""
        cur.execute(command_s.format(year))
        
        

class EighthELAScore(object):
    """ Mean score on 8th grade English exams """
    
    def __init__(self):
        self.new_table_s = 'eighth_ela_score'
        self.orig_table_s_d = {year:'ELA8 Subgroup Results' for year in range(2007, 2015)}
        
    def extract(self, cur, year):
        """ Returns an N-by-2 of the ENTITY_CD and value """
        
        assert(year >= 2007)        
        
        print('Creating {0} for year {1:d}'.format(self.new_table_s, year))
        
        command_s = 'DROP TABLE IF EXISTS temp{0:d};'
        cur.execute(command_s.format(year))
        command_s = """CREATE TABLE temp{0:d} SELECT * FROM SRC{0:d}.`{1}`
WHERE YEAR = {0:d} AND SUBGROUP_NAME = 'General Education';"""
        cur.execute(command_s.format(year, self.orig_table_s_d[year]))
        command_s = """DELETE FROM temp{0:d}
WHERE MEAN_SCORE = 's';"""
        cur.execute(command_s.format(year))
        command_s = """ALTER TABLE temp{0:d} CHANGE ENTITY_CD ENTITY_CD_{0:d} CHAR(12);"""
        cur.execute(command_s.format(year))
        command_s = """ALTER TABLE temp{0:d} ADD {1}_{0:d} FLOAT(12);"""
        cur.execute(command_s.format(year, self.new_table_s))
        command_s = """UPDATE temp{0:d} SET {1}_{0:d} = MEAN_SCORE;"""
        cur.execute(command_s.format(year, self.new_table_s))
        command_s = 'DROP TABLE IF EXISTS temp{0:d}_final;'
        cur.execute(command_s.format(year))
        command_s = """CREATE TABLE temp{0:d}_final
SELECT ENTITY_CD_{0:d}, {1}_{0:d} FROM temp{0:d}
WHERE YEAR = {0:d};"""
        cur.execute(command_s.format(year, self.new_table_s))
        command_s = """ALTER TABLE temp{0:d}_final
ADD INDEX ENTITY_CD_{0:d} (ENTITY_CD_{0:d});"""
        cur.execute(command_s.format(year))
        
        

class EighthMathScore(object):
    """ Mean score on 8th grade math exams """
    
    def __init__(self):
        self.new_table_s = 'eighth_math_score'
        self.orig_table_s_d = {year:'Math8 Subgroup Results' for year in range(2007, 2015)}
        
    def extract(self, cur, year):
        """ Returns an N-by-2 of the ENTITY_CD and value """
        
        assert(year >= 2007)        
        
        print('Creating {0} for year {1:d}'.format(self.new_table_s, year))
        
        command_s = 'DROP TABLE IF EXISTS temp{0:d};'
        cur.execute(command_s.format(year))
        command_s = """CREATE TABLE temp{0:d} SELECT * FROM SRC{0:d}.`{1}`
WHERE YEAR = {0:d} AND SUBGROUP_NAME = 'General Education';"""
        cur.execute(command_s.format(year, self.orig_table_s_d[year]))
        command_s = """DELETE FROM temp{0:d}
WHERE MEAN_SCORE = 's';"""
        cur.execute(command_s.format(year))
        command_s = """ALTER TABLE temp{0:d} CHANGE ENTITY_CD ENTITY_CD_{0:d} CHAR(12);"""
        cur.execute(command_s.format(year))
        command_s = """ALTER TABLE temp{0:d} ADD {1}_{0:d} FLOAT(12);"""
        cur.execute(command_s.format(year, self.new_table_s))
        command_s = """UPDATE temp{0:d} SET {1}_{0:d} = MEAN_SCORE;"""
        cur.execute(command_s.format(year, self.new_table_s))
        command_s = 'DROP TABLE IF EXISTS temp{0:d}_final;'
        cur.execute(command_s.format(year))
        command_s = """CREATE TABLE temp{0:d}_final
SELECT ENTITY_CD_{0:d}, {1}_{0:d} FROM temp{0:d}
WHERE YEAR = {0:d};"""
        cur.execute(command_s.format(year, self.new_table_s))
        command_s = """ALTER TABLE temp{0:d}_final
ADD INDEX ENTITY_CD_{0:d} (ENTITY_CD_{0:d});"""
        cur.execute(command_s.format(year))
        
        
        
class EighthScienceScore(object):
    """ Mean score on 8th grade science exams """
    
    def __init__(self):
        self.new_table_s = 'eighth_science_score'
        self.orig_table_s_d = {year:'Science8 Subgroup Results' for year in range(2007, 2015)}
        
    def extract(self, cur, year):
        """ Returns an N-by-2 of the ENTITY_CD and value """
        
        assert(year >= 2007)        
        
        print('Creating {0} for year {1:d}'.format(self.new_table_s, year))
        
        command_s = 'DROP TABLE IF EXISTS temp{0:d};'
        cur.execute(command_s.format(year))
        command_s = """CREATE TABLE temp{0:d} SELECT * FROM SRC{0:d}.`{1}`
WHERE YEAR = {0:d} AND SUBGROUP_NAME = 'General Education';"""
        cur.execute(command_s.format(year, self.orig_table_s_d[year]))
        command_s = """DELETE FROM temp{0:d}
WHERE MEAN_SCORE = 's';"""
        cur.execute(command_s.format(year))
        command_s = """ALTER TABLE temp{0:d} CHANGE ENTITY_CD ENTITY_CD_{0:d} CHAR(12);"""
        cur.execute(command_s.format(year))
        command_s = """ALTER TABLE temp{0:d} ADD {1}_{0:d} FLOAT(12);"""
        cur.execute(command_s.format(year, self.new_table_s))
        command_s = """UPDATE temp{0:d} SET {1}_{0:d} = MEAN_SCORE;"""
        cur.execute(command_s.format(year, self.new_table_s))
        command_s = 'DROP TABLE IF EXISTS temp{0:d}_final;'
        cur.execute(command_s.format(year))
        command_s = """CREATE TABLE temp{0:d}_final
SELECT ENTITY_CD_{0:d}, {1}_{0:d} FROM temp{0:d}
WHERE YEAR = {0:d};"""
        cur.execute(command_s.format(year, self.new_table_s))
        command_s = """ALTER TABLE temp{0:d}_final
ADD INDEX ENTITY_CD_{0:d} (ENTITY_CD_{0:d});"""
        cur.execute(command_s.format(year))
        
        
        
class PopTwelfth(object):
    """ Population of 12th grade """
    
    def __init__(self):
        self.description_s = '12th grade population'
        self.explanatory_name = 'Population of 12th grade'
        self.multiplier = 1
        self.range_l = [0, np.inf]
        self.output_format_1_s = '12th grade population'
        self.output_format_2_s = '{:0.0f}'
        self.new_table_s = 'pop_twelfth'
        self.orig_table_s_d = {year:'BEDS Day Enrollment' for year in range(2007, 2015)}
        self.allow_prediction = True
        self.in_metric = False
        self.metric_weight = None
        self.bar_plot_s = None

        
    def extract(self, cur, year):
        """ Returns an N-by-2 of the ENTITY_CD and value """
        
        assert(year >= 2007)        
        
        print('Creating {0} for year {1:d}'.format(self.new_table_s, year))
        
        command_s = 'DROP TABLE IF EXISTS temp{0:d};'
        cur.execute(command_s.format(year))
        command_s = """CREATE TABLE temp{0:d} SELECT * FROM SRC{0:d}.`{1}`
WHERE YEAR = {0:d};"""
        cur.execute(command_s.format(year, self.orig_table_s_d[year]))
        command_s = """ALTER TABLE temp{0:d} CHANGE ENTITY_CD ENTITY_CD_{0:d} CHAR(12);"""
        cur.execute(command_s.format(year))
        command_s = """ALTER TABLE temp{0:d} ADD {1}_{0:d} FLOAT(12);"""
        cur.execute(command_s.format(year, self.new_table_s))
        command_s = """UPDATE temp{0:d} SET {1}_{0:d} = `12`;"""
        cur.execute(command_s.format(year, self.new_table_s))
        command_s = 'DROP TABLE IF EXISTS temp{0:d}_final;'
        cur.execute(command_s.format(year))
        command_s = """CREATE TABLE temp{0:d}_final
SELECT ENTITY_CD_{0:d}, {1}_{0:d} FROM temp{0:d}
WHERE YEAR = {0:d};"""
        cur.execute(command_s.format(year, self.new_table_s))
        command_s = """ALTER TABLE temp{0:d}_final
ADD INDEX ENTITY_CD_{0:d} (ENTITY_CD_{0:d});"""
        cur.execute(command_s.format(year))
        
        
        
class PostSecondary(object):
    """ Fraction of students receiving some sort of post-secondary education after high school """
    
    def __init__(self):
        self.description_s = 'Percentage of graduates attending college\nor other post-secondary education'
        self.explanatory_name = 'Post-secondary education'
        self.multiplier = 100
        self.range_l = [0, 1]
        self.output_format_1_s = 'percentage of graduates attending college or other post-secondary education'
        self.output_format_2_s = '{:0.0f}%'
        self.new_table_s = 'post_secondary'
        self.orig_table_s_d = {year:'High School Post-Graduation Plans of Completers' for year in range(2009, 2015)}
        self.orig_table_s_d[2007] = 'High School Post-Graduation Plans of Graduates'
        self.orig_table_s_d[2008] = 'High School Post-Graduation Plans of Graduates'
        self.allow_prediction = True
        self.in_metric = True
        self.metric_weight = 1
        self.bar_plot_s = 'College / post-secondary\nattendance rate (%)'

        
    def extract(self, cur, year):
        """ Returns an N-by-2 of the ENTITY_CD and value """
        
        assert(year >= 2007)        
        
        print('Creating {0} for year {1:d}'.format(self.new_table_s, year))
        
        command_s = 'DROP TABLE IF EXISTS temp{0:d};'
        cur.execute(command_s.format(year))
        command_s = """CREATE TABLE temp{0:d} SELECT * FROM SRC{0:d}.`{1}`
WHERE YEAR = {0:d} AND SUBGROUP_NAME = 'General Education'"""
        if year < 2014:
            command_s += """ AND PER_4YR_COLLEGE_IN_STATE NOT LIKE '%s%'
AND PER_4YR_COLLEGE_OUT_STATE NOT LIKE '%s%'
AND PER_2YR_COLLEGE_IN_STATE NOT LIKE '%s%'
AND PER_2YR_COLLEGE_OUT_STATE NOT LIKE '%s%'
AND PER_POST_SECONDARY_IN_STATE NOT LIKE '%s%'
AND PER_POST_SECONDARY_OUT_STATE NOT LIKE '%s%';"""
        else:
            command_s += """ AND PER_4YR_COLLEGE != 's'
AND PER_2YR_COLLEGE != 's' AND PER_POST_SECONDARY != 's';"""
        cur.execute(command_s.format(year, self.orig_table_s_d[year]))
        if year < 2014:
            command_s = """DELETE FROM temp{0:d} WHERE PER_4YR_COLLEGE_IN_STATE LIKE '%s%'
OR PER_4YR_COLLEGE_OUT_STATE LIKE '%s%' OR PER_2YR_COLLEGE_IN_STATE LIKE '%s%'
OR PER_2YR_COLLEGE_OUT_STATE LIKE '%s%' OR PER_POST_SECONDARY_IN_STATE LIKE '%s%'
OR PER_POST_SECONDARY_OUT_STATE LIKE '%s%';"""
            cur.execute(command_s.format(year))
        else:
            command_s = """DELETE FROM temp{0:d} WHERE PER_4YR_COLLEGE LIKE '%s%'
OR PER_2YR_COLLEGE LIKE '%s%' OR PER_POST_SECONDARY LIKE '%s%';"""
            cur.execute(command_s.format(year))
        command_s = """ALTER TABLE temp{0:d} CHANGE ENTITY_CD ENTITY_CD_{0:d} CHAR(12);"""
        cur.execute(command_s.format(year))
        command_s = """ALTER TABLE temp{0:d} ADD {1}_{0:d} FLOAT(12);"""
        cur.execute(command_s.format(year, self.new_table_s))
        if year < 2014:
            command_s = """UPDATE temp{0:d} SET {1}_{0:d} = (PER_4YR_COLLEGE_IN_STATE + PER_4YR_COLLEGE_OUT_STATE + PER_2YR_COLLEGE_IN_STATE + PER_2YR_COLLEGE_OUT_STATE + PER_POST_SECONDARY_IN_STATE + PER_POST_SECONDARY_OUT_STATE) / 100;"""
        else:
            command_s = """UPDATE temp{0:d} SET {1}_{0:d} = (PER_4YR_COLLEGE + PER_2YR_COLLEGE + PER_POST_SECONDARY) / 100;"""
        cur.execute(command_s.format(year, self.new_table_s))
        command_s = 'DROP TABLE IF EXISTS temp{0:d}_final;'
        cur.execute(command_s.format(year))
        command_s = """CREATE TABLE temp{0:d}_final
SELECT ENTITY_CD_{0:d}, {1}_{0:d} FROM temp{0:d}
WHERE YEAR = {0:d};"""
        cur.execute(command_s.format(year, self.new_table_s))
        command_s = """ALTER TABLE temp{0:d}_final
ADD INDEX ENTITY_CD_{0:d} (ENTITY_CD_{0:d});"""
        cur.execute(command_s.format(year))
        


class RegentsPassRate(object):
    
    def __init__(self):
        self.description_s = 'Percent passing Regents Exams\n(averaged over subjects)'
        self.explanatory_name = 'Regents Exams pass rate'
        self.multiplier = 100
        self.range_l = [0, 1]
        self.output_format_1_s = 'percent of students passing Regents Exams (averaged over subjects)'
        self.output_format_2_s = '{:0.0f}%'
        self.new_table_s = 'regents_pass_rate'
        self.orig_table_s_d = {2004: 'Regents',
                          2005: 'Regents',
                          2006: 'Regents Results 2005-06',
                          2007: 'Regents Examination Annual Results',
                          2008: 'Regents Examination Annual Results',
                          2009: 'Regents Examination Annual Results',
                          2010: 'Regents Examination Annual Results',
                          2011: 'Regents Examination Annual Results',
                          2012: 'Regents Examination Annual Results',
                          2013: 'Regents Examination Annual Results',
                          2014: 'Regents Examination Annual Results'}
        self.allow_prediction = True
        self.in_metric = True
        self.metric_weight = 1
        self.bar_plot_s = 'Regents Exams\npass rate (%)'


    def extract(self, cur, year):
        """ Returns an N-by-3 of the ENTITY_CD, SUBJECT, and pass rate """
        
        print('Creating {0} for year {1:d}'.format(self.new_table_s, year))
        
        command_s = 'DROP TABLE IF EXISTS temp{0:d};'
        cur.execute(command_s.format(year))
        command_s = 'CREATE TABLE temp{0:d} SELECT * FROM SRC{0:d}.`{1}`;'
        cur.execute(command_s.format(year, self.orig_table_s_d[year]))
        if year >= 2007:
            command_s = """DELETE FROM temp{0:d}
WHERE TESTED = 's' OR `NUM_65-84` = 's' OR `NUM_85-100` = 's';"""
            cur.execute(command_s.format(year))
            command_s = """ALTER TABLE temp{0:d} CHANGE ENTITY_CD ENTITY_CD_{0:d} CHAR(12);"""
            cur.execute(command_s.format(year))
            command_s = """ALTER TABLE temp{0:d} CHANGE SUBJECT SUBJECT_{0:d} CHAR(12);"""
            cur.execute(command_s.format(year))
            command_s = """ALTER TABLE temp{0:d}
CHANGE TESTED TESTED INT;"""
            cur.execute(command_s.format(year))
            command_s = """ALTER TABLE temp{0:d}
CHANGE `NUM_65-84` `NUM_65-84` INT;"""
            cur.execute(command_s.format(year))
            command_s = """ALTER TABLE temp{0:d}
CHANGE `NUM_85-100` `NUM_85-100` INT;"""
            cur.execute(command_s.format(year))
            command_s = """ALTER TABLE temp{0:d} ADD fraction_passing_{0:d} FLOAT(23);"""
            cur.execute(command_s.format(year))
            command_s = """UPDATE temp{0:d}
SET fraction_passing_{0:d} = (`NUM_65-84` + `NUM_85-100`) / TESTED;"""
            cur.execute(command_s.format(year))
        else:
            command_s = """DELETE FROM temp{0:d}
WHERE Tested = '#' OR `65-100` = '#';"""
            cur.execute(command_s.format(year))
            command_s = """ALTER TABLE temp{0:d} CHANGE BEDS_CD ENTITY_CD_{0:d} CHAR(12);"""
            cur.execute(command_s.format(year))
            command_s = """ALTER TABLE temp{0:d} CHANGE SUBJECT_CD SUBJECT_{0:d} CHAR(12);"""
            cur.execute(command_s.format(year))
            if year == 2006:
                command_s = """ALTER TABLE temp{0:d}
CHANGE GROUP_NAME SUBGROUP_NAME CHAR(30);"""
                cur.execute(command_s.format(year))
            command_s = """ALTER TABLE temp{0:d}
CHANGE Tested Tested INT;"""
            cur.execute(command_s.format(year))
            command_s = """ALTER TABLE temp{0:d}
CHANGE `65-100` `65-100` INT;"""
            cur.execute(command_s.format(year))
            command_s = """ALTER TABLE temp{0:d} ADD fraction_passing_{0:d} FLOAT(23);"""
            cur.execute(command_s.format(year))
            command_s = """UPDATE temp{0:d}
SET fraction_passing_{0:d} = `65-100` / Tested;"""
            cur.execute(command_s.format(year))
        command_s = 'DROP TABLE IF EXISTS temp{0:d}_filtered;'
        cur.execute(command_s.format(year))
        print('Starting to filter for year {:d}'.format(year))
        if year >= 2006:
            command_s = """CREATE TABLE temp{0:d}_filtered
SELECT ENTITY_CD_{0:d}, SUBJECT_{0:d}, fraction_passing_{0:d} FROM temp{0:d}
WHERE YEAR = {0:d}
AND SUBGROUP_NAME = 'General Education'
AND ENTITY_CD_{0:d} NOT LIKE '%0000'
AND ENTITY_CD_{0:d} NOT LIKE '00000000000%'
AND ENTITY_CD_{0:d} != '111111111111';"""
        else:
            command_s = """CREATE TABLE temp{0:d}_filtered
SELECT ENTITY_CD_{0:d}, SUBJECT_{0:d}, fraction_passing_{0:d} FROM temp{0:d}
WHERE YEAR = {0:d}
AND ENTITY_CD_{0:d} NOT LIKE '%0000'
AND ENTITY_CD_{0:d} NOT LIKE '00000000000%'
AND ENTITY_CD_{0:d} != '111111111111';"""
        cur.execute(command_s.format(year))
        command_s = 'DROP TABLE IF EXISTS temp{0:d}_final;'
        cur.execute(command_s.format(year))
        command_s = """CREATE TABLE temp{0:d}_final
SELECT ENTITY_CD_{0:d}, AVG(fraction_passing_{0:d}) FROM temp{0:d}_filtered
WHERE SUBJECT_{0:d} IN ('REG_GLHIST', 'REG_USHG_RV', 'REG_ENG', 'REG_INTALG', 'REG_ESCI_PS', 'REG_LENV', 'REG_MATHA')
GROUP BY ENTITY_CD_{0:d};"""
    # At some point REG_MATHA disappeared and got replaced by REG_INTALG
        cur.execute(command_s.format(year))
        command_s = """ALTER TABLE temp{0:d}_final
CHANGE `AVG(fraction_passing_{0:d})` {1}_{0:d} FLOAT;"""
        cur.execute(command_s.format(year, self.new_table_s))
        command_s = """ALTER TABLE temp{0:d}_final
ADD INDEX ENTITY_CD_{0:d} (ENTITY_CD_{0:d});"""
        cur.execute(command_s.format(year))
        
        
        
class TeacherNumber(object):
    """ Number of teachers """
    
    def __init__(self):
        self.description_s = 'Number of teachers'
        self.explanatory_name = 'Number of teachers'
        self.multiplier = 1
        self.range_l = [0, np.inf]
        self.output_format_1_s = 'number of teachers'
        self.output_format_2_s = '{:0.0f}'
        self.new_table_s = 'teacher_number'
        self.orig_table_s_d = {year:'Staff' for year in range(2007, 2015)}
        self.allow_prediction = True
        self.in_metric = False
        self.metric_weight = None
        self.bar_plot_s = None

        
    def extract(self, cur, year):
        """ Returns an N-by-2 of the ENTITY_CD and value """
        
        assert(year >= 2007)        
        
        print('Creating {0} for year {1:d}'.format(self.new_table_s, year))
        
        command_s = 'DROP TABLE IF EXISTS temp{0:d};'
        cur.execute(command_s.format(year))
        command_s = """CREATE TABLE temp{0:d} SELECT * FROM SRC{0:d}.`{1}`
WHERE YEAR = {0:d};"""
        cur.execute(command_s.format(year, self.orig_table_s_d[year]))
        command_s = """ALTER TABLE temp{0:d} CHANGE ENTITY_CD ENTITY_CD_{0:d} CHAR(12);"""
        cur.execute(command_s.format(year))
        command_s = """ALTER TABLE temp{0:d} ADD {1}_{0:d} FLOAT(12);"""
        cur.execute(command_s.format(year, self.new_table_s))
        command_s = """UPDATE temp{0:d} SET {1}_{0:d} = NUM_TEACH;"""
        cur.execute(command_s.format(year, self.new_table_s))
        command_s = 'DROP TABLE IF EXISTS temp{0:d}_final;'
        cur.execute(command_s.format(year))
        command_s = """CREATE TABLE temp{0:d}_final
SELECT ENTITY_CD_{0:d}, {1}_{0:d} FROM temp{0:d}
WHERE YEAR = {0:d};"""
        cur.execute(command_s.format(year, self.new_table_s))
        command_s = """ALTER TABLE temp{0:d}_final
ADD INDEX ENTITY_CD_{0:d} (ENTITY_CD_{0:d});"""
        cur.execute(command_s.format(year))
        
        
        
class TenthClassSize(object):
    """ Mean class size of 10-grade English, math, science, and social studies """
    
    def __init__(self):
        self.description_s = 'Size of 10th grade classes\n(averaged over English, math, science, and social studies)'
        self.explanatory_name = 'Average classroom size'
        self.multiplier = 1
        self.range_l = [0, np.inf]
        self.output_format_1_s = 'average classroom size'
        self.output_format_2_s = '{:0.0f}'
        self.new_table_s = 'tenth_class_size'
        self.orig_table_s_d = {year:'Average Class Size' for year in range(2007, 2015)}
        self.allow_prediction = False
        self.in_metric = True
        self.metric_weight = -0.01
        self.bar_plot_s = 'Avg. class size\n'

        
    def extract(self, cur, year):
        """ Returns an N-by-2 of the ENTITY_CD and value """
        
        assert(year >= 2007)        
        
        print('Creating {0} for year {1:d}'.format(self.new_table_s, year))
        
        command_s = 'DROP TABLE IF EXISTS temp{0:d};'
        cur.execute(command_s.format(year))
        command_s = """CREATE TABLE temp{0:d} SELECT * FROM SRC{0:d}.`{1}`
WHERE YEAR = {0:d};"""
        cur.execute(command_s.format(year, self.orig_table_s_d[year]))
        command_s = """ALTER TABLE temp{0:d} CHANGE ENTITY_CD ENTITY_CD_{0:d} CHAR(12);"""
        cur.execute(command_s.format(year))
        command_s = """ALTER TABLE temp{0:d} ADD {1}_{0:d} FLOAT(12);"""
        cur.execute(command_s.format(year, self.new_table_s))
        command_s = """UPDATE temp{0:d} SET {1}_{0:d} = (GRADE_10_ENGLISH + GRADE_10_MATH + GRADE_10_SCI + GRADE_10_SS) / 4;"""
        cur.execute(command_s.format(year, self.new_table_s))
        command_s = 'DROP TABLE IF EXISTS temp{0:d}_final;'
        cur.execute(command_s.format(year))
        command_s = """CREATE TABLE temp{0:d}_final
SELECT ENTITY_CD_{0:d}, {1}_{0:d} FROM temp{0:d}
WHERE YEAR = {0:d};"""
        cur.execute(command_s.format(year, self.new_table_s))
        command_s = """ALTER TABLE temp{0:d}_final
ADD INDEX ENTITY_CD_{0:d} (ENTITY_CD_{0:d});"""
        cur.execute(command_s.format(year))
        
        
        
class TurnoverRate(object):
    """ Turnover rate of all teachers """
    
    def __init__(self):
        self.description_s = 'Percentage of teacher turnover in a given year'
        self.explanatory_name = 'Turnover rate'
        self.multiplier = 100
        self.range_l = [0, 1]
        self.output_format_1_s = 'yearly teacher turnover rate'
        self.output_format_2_s = '{:0.0f}%'
        self.new_table_s = 'turnover_rate'
        self.orig_table_s_d = {year:'Staff' for year in range(2007, 2015)}
        self.allow_prediction = False
        self.in_metric = True
        self.metric_weight = -1
        self.bar_plot_s = 'Yearly turnover\nrate (%)'

        
    def extract(self, cur, year):
        """ Returns an N-by-2 of the ENTITY_CD and value """
        
        assert(year >= 2007)        
        
        print('Creating {0} for year {1:d}'.format(self.new_table_s, year))
        
        command_s = 'DROP TABLE IF EXISTS temp{0:d};'
        cur.execute(command_s.format(year))
        command_s = 'CREATE TABLE temp{0:d} SELECT * FROM SRC{0:d}.`{1}`;'
        cur.execute(command_s.format(year, self.orig_table_s_d[year]))
        command_s = """DELETE FROM temp{0:d}
WHERE PER_TURN_ALL = 's';"""
        cur.execute(command_s.format(year))
        command_s = """ALTER TABLE temp{0:d} CHANGE ENTITY_CD ENTITY_CD_{0:d} CHAR(12);"""
        cur.execute(command_s.format(year))
        command_s = """ALTER TABLE temp{0:d} ADD {1}_{0:d} FLOAT(12);"""
        cur.execute(command_s.format(year, self.new_table_s))
        command_s = """UPDATE temp{0:d}
SET {1}_{0:d} = PER_TURN_ALL / 100;"""
        cur.execute(command_s.format(year, self.new_table_s))
            
        command_s = 'DROP TABLE IF EXISTS temp{0:d}_final;'
        cur.execute(command_s.format(year))
        command_s = """CREATE TABLE temp{0:d}_final
SELECT ENTITY_CD_{0:d}, {1}_{0:d} FROM temp{0:d}
WHERE YEAR = {0:d};"""
        cur.execute(command_s.format(year, self.new_table_s))
        command_s = """ALTER TABLE temp{0:d}_final
ADD INDEX ENTITY_CD_{0:d} (ENTITY_CD_{0:d});"""
        cur.execute(command_s.format(year))
        
        

Database_l = [DiscountLunch,
              Dropout,
              PopTwelfth,
              PostSecondary,
              RegentsPassRate,
              TeacherNumber,
              TenthClassSize,
              TurnoverRate]
DistrictDatabase_l = [Budget]
# Features removed: EighthELAScore, EighthMathScore, EighthScienceScore
        
            

if __name__ == '__main__':
    main()