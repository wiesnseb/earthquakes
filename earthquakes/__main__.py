from import_csv import import_csv
from plots import *
from features import *
from preprocessing import *
from model_stats import *
from models import *
import matplotlib.pyplot as plt



#Import csv file
path = 'earthquakes/data/earthquakes_turkey.csv'
df = import_csv(path)


# <====================== EXPLORATION ======================>
#explore_plot(df)



# <====================== FEATURE ENGINEERING ======================>
#Split Date Columns
df = add_split_date(df, split = 'year')
df = add_split_date(df, split = 'month')
df = add_split_date(df, split = 'day')

#Time between events
df = add_inter_event_duration(df)

window_size = 15
#Add rolling mean to mag column 
df = add_rolling_statistic(df, col_name='magnitude', window_size=window_size, fill_value="mean", statistic='mean')

#Add rolling avg to mag column 
df = add_rolling_statistic(df, col_name='magnitude', window_size=window_size, fill_value="mean", statistic='avg')

#Add rolling std to mag column 
df = add_rolling_statistic(df, col_name='magnitude', window_size=window_size, fill_value="mean", statistic='std')

#Add rolling max to mag column 
df = add_rolling_statistic(df, col_name='magnitude', window_size=window_size, fill_value="mean", statistic='max')

#Add rolling min to mag column 
df = add_rolling_statistic(df, col_name='magnitude', window_size=window_size, fill_value="mean", statistic='min')

#Drop remaining Na
df = df.dropna()

#Set date as index column
#df = set_index(df, col_name='date')

#Add cluster for Lat and Long
df = add_kmeans(df, num_clusters=20)

#Add mean magnitude in location radius
#df = add_mean_mag_location(df, 1)


#Drop columns
#df = drop_col()


# <====================== PREPROCESSING ======================>

#Normalizing dataset
df = normalize_columns(df, ['depth'
                            , 'magnitude'
                            , 'inter_event_duration'
                            , 'rolling_15_magnitude_mean'
                            , 'rolling_15_magnitude_avg'
                            , 'rolling_15_magnitude_std'
                            , 'rolling_15_magnitude_max'
                            , 'rolling_15_magnitude_min'])


#Train Test Split
train, test = split_data(df, 0.2)

#Create datasets

target_variable = 'magnitude'
feature_variables = ['cluster_id'
                            , 'depth'
                            , 'magnitude'
                            #, 'year'
                            #, 'month'
                            #, 'day'
                            , 'inter_event_duration'
                            , 'rolling_15_magnitude_mean'
                            , 'rolling_15_magnitude_avg'
                            , 'rolling_15_magnitude_std'
                            , 'rolling_15_magnitude_max'
                            , 'rolling_15_magnitude_min']

#feature_variables = ['depth', 'magnitude', 'inter_event_duration']


time_steps = 20
X_train, y_train = create_dataset(train, feature_variables, target_variable, time_steps=time_steps)
X_test, y_test = create_dataset(test, feature_variables, target_variable, time_steps=time_steps)


print(f"Shape X_train: {X_train.shape}, Shape y_train: {y_train.shape}")
print(f"Shape X_test: {X_test.shape}, Shape y_test: {y_test.shape}")



# <====================== LSTM ======================>
#history, model = run_LSTM(50, X_train, y_train)

# <====================== GRU ======================>
history, model = run_GRU(50, X_train, y_train)


# <====================== MODEL ANALYSIS ======================>$
#Plot graph of loss and validation loss over time
plot_loss(history)

#Fit scaler for the training set
target_scaler = fit_scaler(train, target_variable)

#Store model predictions in df
result_df = get_predictions(model, X_test, y_test, target_scaler)

#Plot model prediction again true values
plot_true_pred(result_df, y_test = 'y_test_mag', y_pred = 'y_pred_mag')

#Statistics about model performance
evaluate_model(result_df['y_test_mag'], result_df['y_pred_mag'])