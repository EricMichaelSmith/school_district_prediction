from app import app
from flask import make_response, render_template, request
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
import pymysql as mdb
import StringIO

import config
import config_unsynced
import join_data



@app.route('/')
@app.route('/index')
@app.route('/input')
def schools_input():
  return render_template("input.html",
                         dropdown_s_l = dropdown_s_l)
                         
                         

@app.route('/bar_plot')
def bar_plot():
    
    ID1 = request.args.get('ID1')
    ID2 = request.args.get('ID2')
    feature_s_l = []
    loop = True
    while loop:
        arg_s = 'feature{0:d}'.format(len(feature_s_l)+1)
        if arg_s in request.args:
            feature_s_l.append(request.args.get(arg_s))
        else:
            loop = False
    overall_score1 = request.args.get('score1')
    overall_score2 = request.args.get('score2')

    fig = plt.Figure(figsize=(10,6))
    fig.patch.set_facecolor('white')
    
    for i_feature, feature_s in enumerate(feature_s_l):
        school1_prediction = query_prediction_scores(feature_s, ID=ID1)
        school2_prediction = query_prediction_scores(feature_s, ID=ID2)
        
        axis = fig.add_subplot(1, len(feature_s)+1, i_feature+1)
        score1 = school1_prediction[0]['score_l'][-1] * \
            all_database_stats_d[feature_s]['multiplier']
        score2 = school2_prediction[0]['score_l'][-1] * \
            all_database_stats_d[feature_s]['multiplier']
        axis.bar(0, score1, 1, 'r')
        axis.bar(1, score2, 1, 'b')
        axis.set_xlim([-0.25, 1.25])
        axis.set_xlabel(all_database_stats_d[feature_s]['bar_plot_s'])
        
    axis = fig.add_subplot(1, len(feature_s)+1, len(feature_s)+1)
    axis.bar(0, overall_score1, 1, 'r')
    axis.bar(1, overall_score2, 1, 'b')
    axis.set_xlim([-0.25, 1.25])
    axis.set_xlabel('Overall score', fontweight='bold')
    
    canvas = FigureCanvas(fig)
    output = StringIO.StringIO()
    canvas.print_png(output)
    response = make_response(output.getvalue())
    response.mimetype = 'image/png'
    return response



@app.route('/error')
def error():
    return render_template('error.html')



@app.route('/output')
def schools_output():
    #pull 'Name1' and 'Name2' from input field and store them
    Name1 = request.args.get('Name1')
    Name2 = request.args.get('Name2')
    if 'Feature' in request.args:
        Feature = request.args.get('Feature')
    else:
        Feature = None
    
    
    ## Check if inputs correspond to valid schools
    schools1 = query_past_scores(None, name=Name1)
    schools2 = query_past_scores(None, name=Name2)
    
    if (not schools1) or (not schools2):
        return render_template('error.html')
    else:


        ## Query inputs on all features
        past_1_d = {}
        prediction_1_d = {}
        past_2_d = {}
        prediction_2_d = {}
        bar_plot_s_d = {}
        metric_weight_d = {}
        range_l_d = {}
        for feature_s, value in all_database_stats_d.iteritems():
            if all_database_stats_d[feature_s]['in_metric']:
                past_1_d_l = query_past_scores(feature_s, ID=schools1[0]['school_id'])
                prediction_1_d_l = query_prediction_scores(feature_s, ID=schools1[0]['school_id'])
                past_2_d_l = query_past_scores(feature_s, ID=schools2[0]['school_id'])
                prediction_2_d_l = query_prediction_scores(feature_s, ID=schools2[0]['school_id'])
                if not np.any(np.isnan(past_1_d_l[0]['score_l'])) and \
                    not np.any(np.isnan(prediction_1_d_l[0]['score_l'])) and \
                    not np.any(np.isnan(past_2_d_l[0]['score_l'])) and \
                    not np.any(np.isnan(prediction_2_d_l[0]['score_l'])):
                        past_1_d[feature_s] = past_1_d_l[0]['score_l']
                        prediction_1_d[feature_s] = prediction_1_d_l[0]['score_l']
                        past_2_d[feature_s] = past_2_d_l[0]['score_l']
                        prediction_2_d[feature_s] = prediction_2_d_l[0]['score_l']
                        bar_plot_s_d[feature_s] = \
                            all_database_stats_d[feature_s]['bar_plot_s']
                        metric_weight_d[feature_s] = \
                            all_database_stats_d[feature_s]['metric_weight']
                        range_l_d[feature_s] = \
                            all_database_stats_d[feature_s]['range_l']
        features_to_plot_s_l = sorted([key for key in bar_plot_s_d.iterkeys()])
        
        
        ## Calculate overall scores
        raw_score1 = 0
        raw_score2 = 0
        max_possible_score = 0
        for feature_s in features_to_plot_s_l:
            if metric_weight_d[feature_s] > 0:
                raw_score1 += past_1_d[feature_s][-1] * metric_weight_d[feature_s]
                raw_score2 += past_2_d[feature_s][-1] * metric_weight_d[feature_s]
                max_possible_score += range_l_d[feature_s][1] * metric_weight_d[feature_s]
            else:
                max_this_feature = -range_l_d[feature_s][1] * metric_weight_d[feature_s]
                raw_score1 += past_1_d[feature_s][-1] * metric_weight_d[feature_s] \
                    + max_this_feature
                raw_score2 += past_2_d[feature_s][-1] * metric_weight_d[feature_s] \
                    + max_this_feature
                max_possible_score += max_this_feature
        norm_score1 = raw_score1 / max_possible_score * 100
        norm_score2 = raw_score2 / max_possible_score * 100
                
    
        ## Information for time trace plot and for output text
        if Feature:
            for database_s, val in all_database_stats_d.iteritems():
                if all_database_stats_d[database_s]['explanatory_name'] == Feature:
                    feature_s = database_s
            feature_d = all_database_stats_d[feature_s]
            default_dropdown_s = feature_d['explanatory_name']
            schools1_past = query_past_scores(feature_s, name=Name1)
            schools2_past = query_past_scores(feature_s, name=Name2)
            schools1_prediction = query_prediction_scores(feature_s, 
                                                          ID=schools1_past[0]['school_id'])
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
                output_message_s = "Prediction: The {0} of {4} will be {1} higher in {2:d} {3}.".format(feature_d['output_format_1_s'],
                    feature_d['output_format_2_s'],
                    num_years_prediction,
                    year_s, Name1)
                output_message_s = output_message_s.format(feature_d['multiplier']*abs(school1_predicted_difference))
                output_message_color_s = 'red'
            elif school1_predicted_difference < 0:
                output_message_s = "Prediction: The {0} of {4} will be {1} higher in {2:d} {3}.".format(feature_d['output_format_1_s'],
                    feature_d['output_format_2_s'],
                    num_years_prediction,
                    year_s, Name2)
                output_message_s = output_message_s.format(feature_d['multiplier']*abs(school1_predicted_difference))
                output_message_color_s = 'blue'
            else:
                output_message_s = "Prediction: the {0} of both schools will be equal in {2:d} {3}.".format(feature_d['output_format_1_s'],
                    feature_d['output_format_2_s'],
                    num_years_prediction,
                    year_s)
                output_message_s = output_message_s.format(feature_d['multiplier']*abs(school1_predicted_difference))
                output_message_color_s = 'black'
        else:
            default_dropdown_s = ''
            feature_s = ''
            schools1_past = []
            schools2_past = []
            output_message_s = ''
            output_message_color_s = ''
                

        return render_template("output.html",
                               dropdown_s_l = dropdown_s_l,
                               Name1 = Name1,
                               Name2 = Name2,
                               features_to_plot_s_l = features_to_plot_s_l,
                               score1 = norm_score1,
                               score2 = norm_score2,
                               default_dropdown_s = default_dropdown_s,
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
    
    feature_d = all_database_stats_d[feature_s]
    schools1_past = query_past_scores(feature_s, ID=ID1)
    schools1_prediction = query_prediction_scores(feature_s, ID=ID1)
    schools2_past = query_past_scores(feature_s, ID=ID2)
    schools2_prediction = query_prediction_scores(feature_s, ID=ID2)   

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
        if feature_s:
            for year in config.year_l:
                command_s += '{0}_{1:d}, '.format(feature_s, year)
        command_s = command_s[:-2]
        if ID:
            command_s += " FROM master WHERE ENTITY_CD='{0}';".format(ID)            
        elif name:
            command_s += " FROM master WHERE ENTITY_NAME LIKE '%{0}%';"\
                .format(name.upper())
        cur.execute(command_s)
        query_results = cur.fetchall()

    schools = []
    for result in query_results:
        if feature_s:
            schools.append(dict(school_id=result[0], name=result[1],
                                score_l=result[2:]))
        else:
            schools.append(dict(school_id=result[0], name=result[1]))
    
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
    
    
    
db = mdb.connect(user="root", passwd=config_unsynced.pword_s,
                 host="localhost", db="joined", charset='utf8')
all_database_stats_d = join_data.collect_database_stats()

## Information for dropdown list
dropdown_s_l = []
for database_s in all_database_stats_d.iterkeys():
    if all_database_stats_d[database_s]['allow_prediction']:
        dropdown_s_l.append(all_database_stats_d[database_s]['explanatory_name'])
dropdown_s_l = sorted(dropdown_s_l)