#!/usr/bin/env python
"""
Join all data in SQL databases
"""

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

            join_years(cur, instance.new_table_s)
            
    con = utilities.connect_to_sql('joined')
    with con:
        cur = con.cursor()
        join_databases(cur, Database_l)



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
AND ENTITY_CD != '111111111111'"""
        # The REG_ENG is kind of a hack
        instance = RegentsPassRate()
        command_s = command_s.format(config.year_l[-1],
                                     instance.orig_table_s_d[config.year_l[-1]])
        cur.execute(command_s)
        command_s = """ALTER TABLE school_key
ADD INDEX ENTITY_CD (ENTITY_CD)"""
        cur.execute(command_s)
        
        

def join_databases(cur, Database_l):
    """ Join all databases """
    
    master_database_s = 'master'
    individual_database_s_l = [extract_table_name(Database) for Database in Database_l]
    
    print('Starting join_databases')
    
    cur.execute('DROP TABLE IF EXISTS {0}'.format(master_database_s))
    command_s = """CREATE TABLE {0}
SELECT * FROM school_key""".format(master_database_s)
    for individual_database_s in individual_database_s_l:
        this_table_command_s = """
INNER JOIN (SELECT ENTITY_CD_{0}, """.format(individual_database_s)
        for year in config.year_l:
            this_table_command_s += '{0}_{1:d}, '.format(individual_database_s, year)
        this_table_command_s = this_table_command_s[:-2]
        this_table_command_s += """ FROM {0}) AS {0}
ON school_key.ENTITY_CD = {0}.ENTITY_CD_{0}"""
        this_table_command_s = this_table_command_s.format(individual_database_s)
        command_s += this_table_command_s
    command_s += ';'
    cur.execute(command_s)
    
    print('Database {0} created.'.format(master_database_s))
        
        
        
def join_years(cur, new_table_s):
    """ Join separate years of a database that was just extracted into the temp database """
    
    print('Starting join_years for {0}'.format(new_table_s))
    
    cur.execute('DROP TABLE IF EXISTS {0};'.format(new_table_s))
    command_s = """CREATE TABLE {0}
SELECT * FROM school_key""".format(new_table_s)
    for year in config.year_l:
        this_table_command_s = """
INNER JOIN temp.temp{0:d}_final
ON school_key.ENTITY_CD = temp.temp{0:d}_final.ENTITY_CD_{0:d}"""
        this_table_command_s = this_table_command_s.format(year)
        command_s += this_table_command_s
    command_s += ';'
    cur.execute(command_s)
    command_s = """ALTER TABLE {0} CHANGE ENTITY_CD ENTITY_CD_{0} CHAR(12);"""
    cur.execute(command_s.format(new_table_s))
    
    print('Database {0} created.'.format(new_table_s))
    
    

class DiscountLunch(object):
    """ Fraction of students receiving reduced or free lunch """
    
    def __init__(self):
        self.new_table_s = 'discount_lunch'
        self.orig_table_s_d = {year:'Demographic Factors' for year in range(2007, 2015)}
        
    def extract(self, cur, year):
        """ Returns an N-by-2 of the ENTITY_CD and value """
        
        assert(year >= 2007)        
        
        print('Creating {0} for year {1:d}'.format(self.new_table_s, year))
        
        command_s = 'DROP TABLE IF EXISTS temp{0:d};'
        cur.execute(command_s.format(year))
        command_s = 'CREATE TABLE temp{0:d} SELECT * FROM SRC{0:d}.`{1}`;'
        cur.execute(command_s.format(year, self.orig_table_s_d[year]))
        command_s = """DELETE FROM temp{0:d}
WHERE PER_FREE_LUNCH = 's' OR PER_FREE_LUNCH IS NULL
OR PER_REDUCED_LUNCH = 's' OR PER_REDUCED_LUNCH IS NULL;"""
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
WHERE MEAN_SCORE = 's' OR MEAN_SCORE IS NULL;"""
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
WHERE MEAN_SCORE = 's' OR MEAN_SCORE IS NULL;"""
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
WHERE MEAN_SCORE = 's' OR MEAN_SCORE IS NULL;"""
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
        


class RegentsPassRate(object):
    
    def __init__(self):
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
        
        
        
class TurnoverRate(object):
    """ Turnover rate of all teachers """
    
    def __init__(self):
        self.new_table_s = 'turnover_rate'
        self.orig_table_s_d = {year:'Staff' for year in range(2007, 2015)}
        
    def extract(self, cur, year):
        """ Returns an N-by-2 of the ENTITY_CD and value """
        
        assert(year >= 2007)        
        
        print('Creating {0} for year {1:d}'.format(self.new_table_s, year))
        
        command_s = 'DROP TABLE IF EXISTS temp{0:d};'
        cur.execute(command_s.format(year))
        command_s = 'CREATE TABLE temp{0:d} SELECT * FROM SRC{0:d}.`{1}`;'
        cur.execute(command_s.format(year, self.orig_table_s_d[year]))
        command_s = """DELETE FROM temp{0:d}
WHERE PER_TURN_ALL = 's' OR PER_TURN_ALL IS NULL;"""
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
        
        
        
Database_l = [DiscountLunch, EighthELAScore, EighthMathScore, EighthScienceScore,
              RegentsPassRate, TurnoverRate]
        
            

if __name__ == '__main__':
    main()