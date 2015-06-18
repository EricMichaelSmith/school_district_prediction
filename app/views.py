from app import app
from flask import make_response, render_template, request
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
import pymysql as mdb
import StringIO

import config
import join_data



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
    Feature = request.args.get('Feature')
    
    for database_s, val in all_database_stats_d.iteritems():
        if all_database_stats_d[database_s]['explanatory_name'] == Feature:
            feature_s = database_s


    ## Information for dropdown list
    dropdown_s_l = []
    for database_s in all_database_stats_d.iterkeys():
        if all_database_stats_d[database_s]['allow_prediction']:
            dropdown_s_l.append(all_database_stats_d[database_s]['explanatory_name'])
    dropdown_s_l = sorted(dropdown_s_l)


    ## Information for time trace plot and for output text
    schools1_past = query_past_scores(feature_s, name=Name1)
    schools1_prediction = query_prediction_scores(feature_s, 
                                                  ID=schools1_past[0]['school_id'])
    schools2_past = query_past_scores(feature_s, name=Name2)
    schools2_prediction = query_prediction_scores(feature_s,
                                                  ID=schools2_past[0]['school_id']) 
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
                           dropdown_s_l = dropdown_s_l,
                           feature_s = feature_s,
                           schools1_past = schools1_past,
                           schools2_past = schools2_past,
                           output_message_s = output_message_s,
                           output_message_color_s = output_message_color_s)
        
                         

@app.route('/plot')
def plot():

    ID1 = request.args.get('ID1')
    ID2 = request.args.get('ID2')
    feature_s = request.args.get('Feature')
    
    schools1_past = query_past_scores(feature_s, ID=ID1)
    schools1_prediction = query_prediction_scores(feature_s, ID=ID1)
    schools2_past = query_past_scores(feature_s, ID=ID2)
    schools2_prediction = query_prediction_scores(feature_s, ID=ID2)   
    feature_d = all_database_stats_d[feature_s]     

    fig = plt.Figure()
    fig.patch.set_facecolor('white')
    axis = fig.add_axes([0.1, 0.23, 0.8, 0.67])
 
    curve1, = axis.plot(config.year_l,
        np.array(schools1_past[0]['score_l'])*feature_d['multiplier'], 'r')
    curve2, = axis.plot(config.year_l,
        np.array(schools2_past[0]['score_l'])*feature_d['multiplier'], 'b')
    schools1_extrapolation_l = schools1_past[0]['score_l'][-1:] + \
        schools1_prediction[0]['score_l']
    schools2_extrapolation_l = schools2_past[0]['score_l'][-1:] + \
        schools2_prediction[0]['score_l']
    axis.plot(config.year_l[-1:]+config.prediction_year_l,
              np.array(schools1_extrapolation_l)*feature_d['multiplier'], 'r--')
    axis.plot(config.year_l[-1:]+config.prediction_year_l,
              np.array(schools2_extrapolation_l)*feature_d['multiplier'], 'b--')
    axis.set_xlabel('Year')
    axis.set_xlim([config.year_l[0], config.prediction_year_l[-1]])
    axis.set_title(feature_d['description_s'])
    axis.ticklabel_format(useOffset=False)
    axis.legend([curve1, curve2], [schools1_past[0]['name'], schools2_past[0]['name']],
                loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False)
    canvas = FigureCanvas(fig)
    output = StringIO.StringIO()
    canvas.print_png(output)
    response = make_response(output.getvalue())
    response.mimetype = 'image/png'
    return response
        
        

def query_past_scores(feature_s, ID=None, name=None):
    with db:
        cur = db.cursor()
        #just select the city from 'master' that the user inputs
        cur.execute('USE joined;')
        command_s = 'SELECT ENTITY_CD, ENTITY_NAME, '
        for year in config.year_l:
            command_s += '{0}_{1:d}, '.format(feature_s, year)
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
    
    
    
def query_prediction_scores(feature_s, ID):
    with db:
        cur = db.cursor()
        #just select the city from 'master' that the user inputs
        cur.execute('USE joined;')
        command_s = 'SELECT ENTITY_CD, '
        for year in config.prediction_year_l:
            command_s += '{0}_prediction_{1:d}, '.format(feature_s,
                                                           year)
        command_s = command_s[:-2]
        command_s += " FROM {0}_prediction WHERE ENTITY_CD='{1}';"\
            .format(feature_s, ID)
        cur.execute(command_s.format(ID))
        query_results = cur.fetchall()

    schools = []
    for result in query_results:
        schools.append(dict(school_id=result[0],
                            score_l=result[1:]))
    
    return schools
    
    
    
db = mdb.connect(user="root", host="localhost", db="joined", charset='utf8')
all_database_stats_d = join_data.collect_database_stats()