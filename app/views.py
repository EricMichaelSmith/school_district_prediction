from a_Model import ModelIt
from flask import render_template, request
from app import app
import pymysql as mdb

db= mdb.connect(user="root", host="localhost", db="joined", charset='utf8')

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html",
        title = 'Home',
        )

@app.route('/input')
def cities_input():
  return render_template("input.html")

@app.route('/output')
def cities_output():
  #pull 'ID' from input field and store it
  ID1 = request.args.get('ID1')

  with db:
    cur = db.cursor()
    #just select the city from 'joined' that the user inputs
    command_s = """SELECT ENTITY_CD, ENTITY_NAME, `AVG(percent_passing_2014)`
FROM regents_pass_rate WHERE ENTITY_CD='{0}';"""
    cur.execute(command_s.format(ID1))
    query_results = cur.fetchall()

  schools = []
  for result in query_results:
    schools.append(dict(id=result[0], name=result[1], pass_rate_2014=result[2]))

  #call a function from a_Model package. note we are only pulling one result in the query
  pass_rate_input = schools[0]['pass_rate_2014']
  the_result = ModelIt(ID1, pass_rate_input)
  return render_template("output.html", schools = schools, the_result = the_result)