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
In this script we test the abstraction of parts of this code into functions

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
label_generator : label_gen_mod.ClassificationLabelGenerator = label_gen_mod.AbsoluteChangeLabel(**label_generator_params)
filter_generator : filter_gen_mod.FilterGenerator = filter_gen_mod.EveryKthFilter(**filter_generator_params)

