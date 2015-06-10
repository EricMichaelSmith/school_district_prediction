#!/usr/bin/env
"""
Join all data in SQL databases
"""

import config
reload(config)
import utilities
reload(utilities)



def main():
    
    con = utilities.connect_to_sql('temp')    
    with con:
        cur = con.cursor()

        find_regents_school_key(cur)
        for year in config.year_l:
            find_regents_pass_rate(cur, year, config.regents_table_s_d[year])
    
    con = utilities.connect_to_sql('joined')
    with con:
        cur = con.cursor()

        join_regents_pass_rate(cur)



def find_regents_school_key(cur):
    """ Creates a table of each school ID and name """
    
    command_s = 'DROP TABLE IF EXISTS school_key;'
    cur.execute(command_s)
    command_s = """CREATE TABLE school_key
SELECT ENTITY_CD, ENTITY_NAME FROM SRC{0:d}.`{1}`
WHERE YEAR = {0:d} AND SUBJECT = 'REG_ENG' AND SUBGROUP_NAME = 'General Education'"""
    # The REG_ENG is kind of a hack
    command_s = command_s.format(config.year_l[-1],
                                 config.regents_table_s_d[config.year_l[-1]])
    cur.execute(command_s)
    command_s = """ALTER TABLE school_key
ADD INDEX ENTITY_CD (ENTITY_CD)"""
    cur.execute(command_s)
    


def find_regents_pass_rate(cur, year, table_s):
    """ Returns an N-by-3 of the ENTITY_CD, SUBJECT, and pass rate """
    
    print('Starting find_regents_pass_rate for year {:d}'.format(year))
    
    command_s = 'DROP TABLE IF EXISTS temp{0:d};'
    cur.execute(command_s.format(year))
    command_s = 'CREATE TABLE temp{0:d} SELECT * FROM SRC{0:d}.`{1}`;'
    cur.execute(command_s.format(year, table_s))
    command_s = "DELETE FROM temp{0:d} WHERE `PER_65-84` = 's' OR `PER_85-100` = 's';"
    cur.execute(command_s.format(year))
    command_s = """ALTER TABLE temp{0:d} CHANGE ENTITY_CD ENTITY_CD_{0:d} CHAR(12);"""
    cur.execute(command_s.format(year))
    command_s = """ALTER TABLE temp{0:d} CHANGE SUBJECT SUBJECT_{0:d} CHAR(12);"""
    cur.execute(command_s.format(year))
    command_s = """ALTER TABLE temp{0:d} CHANGE `PER_65-84` `PER_65-84` INT;"""
    cur.execute(command_s.format(year))
    command_s = """ALTER TABLE temp{0:d} CHANGE `PER_85-100` `PER_85-100` INT;"""
    cur.execute(command_s.format(year))
    command_s = """ALTER TABLE temp{0:d} ADD percent_passing_{0:d} INT;"""
    cur.execute(command_s.format(year))
    command_s = """UPDATE temp{0:d}
SET percent_passing_{0:d} = `PER_65-84` + `PER_85-100`;"""
    cur.execute(command_s.format(year))
    command_s = 'DROP TABLE IF EXISTS temp{0:d}_filtered;'
    cur.execute(command_s.format(year))
    print('Starting to filter for year {:d}'.format(year))
    command_s = """CREATE TABLE temp{0:d}_filtered
SELECT ENTITY_CD_{0:d}, SUBJECT_{0:d}, percent_passing_{0:d} FROM temp{0:d}
WHERE YEAR = {0:d} AND SUBGROUP_NAME = 'General Education';"""
    cur.execute(command_s.format(year))
    command_s = 'DROP TABLE IF EXISTS temp{0:d}_averaged;'
    cur.execute(command_s.format(year))
    command_s = """CREATE TABLE temp{0:d}_averaged
SELECT ENTITY_CD_{0:d}, AVG(percent_passing_{0:d}) FROM temp{0:d}_filtered
WHERE SUBJECT_{0:d} IN ('REG_GLHIST', 'REG_USHG_RV', 'REG_ENG', 'REG_INTALG', 'REG_ESCI_PS', 'REG_LENV', 'REG_MATHA')
GROUP BY ENTITY_CD_{0:d};"""
# At some point REG_MATHA disappeared and got replaced by REG_INTALG
    cur.execute(command_s.format(year))
    command_s = """ALTER TABLE temp{0:d}_averaged
ADD INDEX ENTITY_CD_{0:d} (ENTITY_CD_{0:d})"""
    cur.execute(command_s.format(year))
        


def join_regents_pass_rate(cur):
    """ Join the percentage of students passing the Regents exam """
    
    print('Starting join_regents_pass_rate')
    
    cur.execute('DROP TABLE IF EXISTS regents_pass_rate')
    command_s = """CREATE TABLE regents_pass_rate
SELECT * FROM temp.school_key"""
    for year in config.year_l:
        this_table_command_s = """
INNER JOIN temp.temp{2:d}_averaged
ON temp.school_key.ENTITY_CD = temp.temp{2:d}_averaged.ENTITY_CD_{2:d}"""
        this_table_command_s = this_table_command_s.format(config.year_l[-1],
            config.regents_table_s_d[config.year_l[-1]],
            year)
        command_s += this_table_command_s
    command_s += ';'
    cur.execute(command_s)
    
    print('Regents pass rate database created.')
            
            

if __name__ == '__main__':
    main()