# Variables
#year_l = range(2004, 2015)
year_l = range(2007, 2015)
# The schema was completely overhauled in 2007, so it's easiest to just look from then for now
year_plot_color_d = {2007: (0.75, 0, 0),
                     2008: (0.75, 0.38, 0),
                     2009: (0.75, 0.75, 0),
                     2010: (0, 0.75, 0),
                     2011: (0, 0.75, 0.75),
                     2012: (0, 0, 0.75),
                     2013: (0.75, 0, 0.75),
                     2014: (0, 0, 0)}
prediction_year_l = range(2015, 2018)

# Paths
code_path = '/Users/Eric/GitHub/EricMichaelSmith/school_district_prediction'
data_path = '/Users/Eric/Dropbox/Computing/Personal/Code/school_district_prediction/data'
plot_path = '/Users/Eric/Dropbox/Computing/Personal/Code/school_district_prediction/plots'