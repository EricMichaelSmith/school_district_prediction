from app import app
from flask import make_response, render_template, request
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
import pymysql as mdb
import StringIO

import config
import join_data

db = mdb.connect(user="root", host="localhost", db="joined", charset='utf8')

plot_instance = join_data.RegentsPassRate()



@app.route('/')
@app.route('/index')
@app.route('/input')
def schools_input():
  return render_template("input.html")



@app.route('/output')
def schools_output():
    #pull 'Name1' and 'Name2' from input field and store them
    Name1 = request.args.get('Name1')
    Name2 = request.args.get('Name2')

    schools1_past = query_past_scores(name=Name1)
    schools1_prediction = query_prediction_scores(ID=schools1_past[0]['school_id'])
    schools2_past = query_past_scores(name=Name2)
    schools2_prediction = query_prediction_scores(ID=schools2_past[0]['school_id']) 
    school1_predicted_difference = schools1_prediction[0]['score_l'][-1] - \
        schools2_prediction[0]['score_l'][-1]
    num_years_prediction = len(schools1_prediction[0]['score_l'])
    if num_years_prediction == 1:
        year_s = 'year'
    else:
        year_s = 'years'
    if school1_predicted_difference > 0:
        output_message_s = "Prediction: School 1's average passing rate will be {0:0.1f}% higher in {1:d} {2}.".format(school1_predicted_difference*100,
                             num_years_prediction, year_s)
        output_message_color_s = 'red'
    elif school1_predicted_difference < 0:
        output_message_s = "Prediction: School 2's average passing rate will be {0:0.1f}% higher in {1:d} {2}.".format(-school1_predicted_difference*100,
                             num_years_prediction, year_s)
        output_message_color_s = 'blue'
    else:
        output_message_s = "Prediction: both schools' test scores will be equal in {0:d} {1}.".format(num_years_prediction, year_s)
        output_message_color_s = 'black'

    return render_template("output.html",
                           schools1_past = schools1_past,
                           schools2_past = schools2_past,
                           output_message_s = output_message_s,
                           output_message_color_s = output_message_color_s)
        
                         

@app.route('/plot')
def plot():

    ID1 = request.args.get('ID1')
    ID2 = request.args.get('ID2')
    
    schools1_past = query_past_scores(ID=ID1)
    schools1_prediction = query_prediction_scores(ID=ID1)
    schools2_past = query_past_scores(ID=ID2)
    schools2_prediction = query_prediction_scores(ID=ID2)        

    fig = plt.Figure()
    fig.patch.set_facecolor('white')
    axis = fig.add_axes([0.1, 0.23, 0.8, 0.67])
 
    curve1, = axis.plot(config.year_l, np.array(schools1_past[0]['score_l'])*100, 'r')
    curve2, = axis.plot(config.year_l, np.array(schools2_past[0]['score_l'])*100, 'b')
    schools1_extrapolation_l = schools1_past[0]['score_l'][-1:] + \
        schools1_prediction[0]['score_l']
    schools2_extrapolation_l = schools2_past[0]['score_l'][-1:] + \
        schools2_prediction[0]['score_l']
    axis.plot(config.year_l[-1:]+config.prediction_year_l,
              np.array(schools1_extrapolation_l)*100, 'r--')
    axis.plot(config.year_l[-1:]+config.prediction_year_l,
              np.array(schools2_extrapolation_l)*100, 'b--')
    axis.set_xlabel('Year')
    axis.set_xlim([config.year_l[0], config.prediction_year_l[-1]])
    axis.set_title(plot_instance.description_s)
    axis.ticklabel_format(useOffset=False)
    axis.legend([curve1, curve2], [schools1_past[0]['name'], schools2_past[0]['name']],
                loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False)
    canvas = FigureCanvas(fig)
    output = StringIO.StringIO()
    canvas.print_png(output)
    response = make_response(output.getvalue())
    response.mimetype = 'image/png'
    return response
    
    

def query_past_scores(ID=None, name=None):
    with db:
        cur = db.cursor()
        #just select the city from 'master' that the user inputs
        command_s = 'SELECT ENTITY_CD, ENTITY_NAME, '
        for year in config.year_l:
            command_s += '{0}_{1:d}, '.format(plot_instance.new_table_s, year)
        command_s = command_s[:-2]
        if ID:
            command_s += " FROM master WHERE ENTITY_CD='{0}';".format(ID)            
        elif name:
            command_s += " FROM master WHERE ENTITY_NAME LIKE '{0}%';"\
                .format(name.upper())
        cur.execute(command_s)
        query_results = cur.fetchall()

    schools = []
    for result in query_results:
        schools.append(dict(school_id=result[0], name=result[1],
                            score_l=result[2:]))
    
    return schools
    
    
    
def query_prediction_scores(ID):
    with db:
        cur = db.cursor()
        #just select the city from 'master' that the user inputs
        command_s = 'SELECT ENTITY_CD, '
        for year in config.prediction_year_l:
            command_s += '{0}_prediction_{1:d}, '.format(plot_instance.new_table_s,
                                                           year)
        command_s = command_s[:-2]
        command_s += " FROM {0}_prediction WHERE ENTITY_CD='{1}';"\
            .format(plot_instance.new_table_s, ID)
        cur.execute(command_s.format(ID))
        query_results = cur.fetchall()

    schools = []
    for result in query_results:
        schools.append(dict(school_id=result[0],
                            score_l=result[1:]))
    
    return schools