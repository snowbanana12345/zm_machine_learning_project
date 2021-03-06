from sklearn.ensemble import RandomForestClassifier
from src.data_base_module.data_retrival import instance as db
from typing import List, Dict, Set, Tuple
import pandas as pd
import src.data_base_module.data_blocks as data
import src.machine_learning_module.feature_generators as feature_gen_mod
import src.machine_learning_module.label_generators as label_gen_mod
import src.machine_learning_module.filter_generators as filter_gen_mod
import src.machine_learning_module.pipeline as pipe_mod
import src.machine_learning_module.performance_metrics as perf_mod

import src.machine_learning_module.feature_generator_data.talib_momentum as talib_mom_data
"""
This is a basic model script that sketches out the basic format of scripts that make use of models

Labeling : abs change labeling
Filtering : every kth filtering
Sampling : Volume  
ml model : random forest classifier
Features : talib momentum indicators
"""
# ------ user inputs ------
symbol : str = "NHK17"
sampling_volume : int = 20
price_series_used : data.BarDataColumns = data.BarDataColumns.CLOSE
look_ahead : int = 15
threshold : float = 15
period : int = 5
shift : int = 0
num_datasets = 40
num_test_sets = 8
cross_val_shift = 0

# ----- #define --------
MORNING : data.IntraDayPeriod = data.IntraDayPeriod.MORNING
AFTERNOON : data.IntraDayPeriod = data.IntraDayPeriod.AFTERNOON

# ------ parameter dictionaries -------
rfr_params : dict = {
    "n_estimators" : 100,
    "criterion" : "entropy",
    "max_depth" : 5,
    "min_samples_split" : 10,
    "min_weight_fraction_leaf" : 0.1,
    "class_weight" : "balanced",
    "max_samples" : 0.75,
    "max_features" : 5
}

label_generator_params : dict = {
    "look_ahead" : look_ahead,
    "threshold" : threshold,
    "criteria" : price_series_used,
}

filter_generator_params : dict = {
    "period" : period,
    "shift" : shift,
    "criteria" : price_series_used
}

talib_momentum_param_dict : Dict[str, List[Dict[str, float]]] = {
    talib_mom_data.ADX : [{'timeperiod' : 14}, {'timeperiod' : 20}],
    talib_mom_data.ADXR : [{'timeperiod' : 8}],
    talib_mom_data.APO : [{'fastperiod' : 10, 'slowperiod' : 14}],
    talib_mom_data.AROON_UP : [{'timeperiod' : 14}],
    talib_mom_data.AROON_DOWN :  [{'timeperiod' : 14}],
    talib_mom_data.AROONOSC : [{'timeperiod' : 14}],
    talib_mom_data.CCI : [{'timeperiod' : 14}],
    talib_mom_data.CMO : [{'timeperiod' : 14}],
    talib_mom_data.MACD: [{'fastperiod' : 14, 'slowperiod' : 16, 'signalperiod' : 18}],
    talib_mom_data.MACD_SIGNAL : [{'fastperiod' : 10, 'slowperiod' : 14, 'signalperiod' : 18}],
    talib_mom_data.MACD_HIST: [{'fastperiod' : 13, 'slowperiod' : 11, 'signalperiod' : 24}],
    talib_mom_data.MOM : [{'timeperiod' : 2}],
    talib_mom_data.PPO : [{'fastperiod' : 10, 'slowperiod' : 14}],
    talib_mom_data.ROC : [{'timeperiod' : 14}],
    talib_mom_data.ROCP: [{'timeperiod' : 14}],
    talib_mom_data.RSI : [{'timeperiod' : 14}],
    talib_mom_data.STOCH_SLOWK : [{'fastk_period' : 10, 'slowk_period' : 15}],
    talib_mom_data.STOCH_SLOWD : [{'fastk_period' : 10, 'slowk_period' : 15}],
    talib_mom_data.STOCHF_FASTK : [{'fastk_period' : 10, 'fastd_period' : 15}],
    talib_mom_data.STOCHF_FASTD : [{'fastk_period' : 10, 'fastd_period' : 15}],
    talib_mom_data.TRIX : [{'timeperiod' : 14}],
    talib_mom_data.ULTOSC : [{'timeperiod1' : 7, 'timeperiod2' : 14, 'timeperiod3' : 21}],
    talib_mom_data.WILLR : [{'timeperiod' : 5}],
}

# ----- objects ------
feature_generator : feature_gen_mod.FeatureGenerator = feature_gen_mod.TalibMomentum(parameters_dict = talib_momentum_param_dict)
label_generator : label_gen_mod.LabelGenerator = label_gen_mod.AbsoluteChangeLabel(**label_generator_params)
filter_generator : filter_gen_mod.FilterGenerator = filter_gen_mod.EveryKthFilter(**filter_generator_params)


# ------ date list -------
""" 40 data sets in total, 20 days, morning and afternoon """
date_lst : [data.Date] = [
    data.Date(day=25, month=1, year=2017),
    data.Date(day=26, month=1, year=2017),
    data.Date(day=27, month=1, year=2017),
    data.Date(day=31, month=1, year=2017),
    data.Date(day=1, month=2, year=2017),
    data.Date(day=2, month=2, year=2017),
    data.Date(day=6, month=2, year=2017),
    data.Date(day=7, month=2, year=2017),
    data.Date(day=8, month=2, year=2017),
    data.Date(day=9, month=2, year=2017),
    data.Date(day=10, month=2, year=2017),
    data.Date(day=14, month=2, year=2017),
    data.Date(day=15, month=2, year=2017),
    data.Date(day=16, month=2, year=2017),
    data.Date(day=17, month=2, year=2017),
    data.Date(day=20, month=2, year=2017),
    data.Date(day=21, month=2, year=2017),
    data.Date(day=22, month=2, year=2017),
    data.Date(day=23, month=2, year=2017),
    data.Date(day=24, month=2, year=2017)]
# ------ cross validation set splits -------
""" List of tuples in the format (train set, test set)"""
cross_val_indices : List[Tuple[List[int], List[int]]] = pipe_mod.find_cross_val_sets_indices(num_datasets = num_datasets, num_test_sets = num_test_sets, start= cross_val_shift)

# ------ load data ------
morning_bar_wrapper_lst : List[data.BarDataFrame] = [db.get_sampled_volume_bar(symbol = symbol, date = date, sampling_volume = sampling_volume, intra_day_period = MORNING) for date in date_lst]
afternoon_bar_wrapper_lst : List[data.BarDataFrame] = [db.get_sampled_volume_bar(symbol = symbol, date = date, sampling_volume = sampling_volume, intra_day_period = AFTERNOON) for date in date_lst]
bar_wrapper_lst = []
for morning_bar_wrapper, afternoon_bar_wrapper in zip(morning_bar_wrapper_lst, afternoon_bar_wrapper_lst):
    bar_wrapper_lst.append(morning_bar_wrapper)
    bar_wrapper_lst.append(afternoon_bar_wrapper)

# ----- get description for each data set for easier tracking -----
bar_data_description_lst : List[str] = [str(bar_wrapper) for bar_wrapper in bar_wrapper_lst]
# ----- create features ------
feature_df_lst : List[feature_gen_mod.FeatureDataFrame] = [feature_generator.create_features_for_data_bar(bar_wrapper) for bar_wrapper in bar_wrapper_lst]
# ----- create labels -----
label_df_lst : List[label_gen_mod.LabelDataFrame] = [label_generator.create_labels_for_data_bar(bar_wrapper) for bar_wrapper in bar_wrapper_lst]
# ----- create filters ------
filter_array_lst : List[filter_gen_mod.FilterArray] = [filter_generator.create_filter_for_data_bar(bar_wrapper) for bar_wrapper in bar_wrapper_lst]

# ----- apply filter and remove missing columns -----
filter_output_lst : List[Tuple[feature_gen_mod.FeatureDataFrame, label_gen_mod.LabelDataFrame]] = [pipe_mod.filter_features_and_labels(feature_df, label_df, filter_array, description)
                                                           for feature_df, label_df, filter_array,description in zip(feature_df_lst, label_df_lst, filter_array_lst, bar_data_description_lst)]
filtered_feature_df_lst : List[feature_gen_mod.FeatureDataFrame] = []
filtered_label_array_lst : List[label_gen_mod.LabelDataFrame] = []
for feature_df, label_array in filter_output_lst:
    filtered_feature_df_lst.append(feature_df)
    filtered_label_array_lst.append(label_array)

# ----- cross validation ------

ml_model_template = RandomForestClassifier(**rfr_params)
train_results, test_results = pipe_mod.cross_validate_binary_classification(
        feature_wrapper_lst = filtered_feature_df_lst,
        label_array_lst = filtered_label_array_lst,
        ml_model_template = ml_model_template,
        cross_val_indices = cross_val_indices,
)

# ----- options ------
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
# ----- print cross validation results ------
train_collated_results : perf_mod.CollatedBinaryClassificationMetrics = perf_mod.CollatedBinaryClassificationMetrics(train_results)
test_collated_results : perf_mod.CollatedBinaryClassificationMetrics = perf_mod.CollatedBinaryClassificationMetrics(test_results)
print(" ------- training set results --------- ")
train_collated_results.print()
print(" ------- test set results ------------- ")
test_collated_results.print()


