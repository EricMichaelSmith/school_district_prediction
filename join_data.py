"""
Join all data in SQL databases
"""

import config
reload(config)
import utilities
reload(utilities)



def main():

    con = utilities.connect_to_sql('joined')
    with con:
        cur = con.cursor()

        join_regents_pass_rate(cur)



def find_regents_pass_rate(cur, year, table_s):
    """ Returns an N-by-3 of the ENTITY_CD, SUBJECT, and pass rate """
    
    command_s = 'DROP TABLE IF EXISTS temp.SRC{0:d};'
    cur.execute(command_s.format(year))
    command_s = 'INSERT INTO temp.SRC{0:d} SELECT * FROM SRC{0:d}.{1};'
    cur.execute(command_s.format(year, table_s))
    command_s = """ALTER TABLE temp.SRC{0:d}
CHANGE ENTITY_CD ENTITY_CD_{0:d}
CHANGE SUBJECT SUBJECT_{0:d}
CHANGE `PER_65-84` `PER_65-84` INT
CHANGE `PER_85-100` `PER_85-100` INT
ADD percent_passing_{0:d};"""
    cur.execute(command_s.format(year))
    command_s = """UPDATE temp.SRC{0:d}
SET percent_passing_{0:d} = `PER_65-84` + `PER_85-100`;"""
    cur.execute(command_s.format(year))
    command_s = """CREATE TABLE temp.SRC{0:d}_filtered
SELECT ENTITY_CD, SUBJECT, percent_passing FROM temp
WHERE YEAR = {0:d} AND SUBGROUP_NAME = 'All Students';"""
    cur.execute(command_s.format(year))
    
        


def join_regents_pass_rate(cur):
    """ Join the percentage of students passing the Regents exam """
    
    cur.execute('DROP TABLE IF EXISTS regents_pass_rate')
    for year in config.year_l:
        find_regents_pass_rate(cur, year, config.regents_table_s_d[year])
    command_s = """CREATE TABLE regents_pass_rate
SELECT * FROM temp.SRC{0:d}_filtered""".format(config.year_l[0])
    for year in config.year_l[1:]:
        this_table_command_s = """
INNER JOIN temp.SRC{1:d}_filtered
ON temp.SRC{0:d}_filtered.ENTITY_CD_{0:d} = temp.SRC{1:d}_filtered.ENTITY_CD_{1:d}""".format(config.year_l[0], year)
        command_s += this_table_command_s
    command_s += ';'
    cur.execute(command_s)
    
    print('Regents pass rate database created.')
            
            

if __name__ == '__main__':
    main()