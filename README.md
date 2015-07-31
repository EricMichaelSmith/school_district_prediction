school_district_prediction (used for schoolstradamus.com)
==========================

Most important scripts:
* app/templates/index.html: the single HTML file that serves as the front-end for the prediction service, receiving inputs and displaying graphs and information
* app/config.py: configuration settings for the analysis
* app/create_predictions.py: creates all predictions and outputs/plots the results in various ways for diagnostic purposes
* app/explore_data.py: performs descriptive analysis of the data and creates various plots, without worrying about prediction
* app/join_data.py: joins all data in SQL databases
* app/utilities.py: provides utility functions, such as selecting data from SQL and writing to it
* app/views.py: Flask decorators binding website pages to functions passing variables
