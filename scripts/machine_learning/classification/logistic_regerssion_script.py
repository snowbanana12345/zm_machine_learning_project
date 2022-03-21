import sklearn.linear_model
from sklearn.linear_model import LogisticRegression
import src.data_base_module.data_blocks as data
from src.data_base_module.data_retrival import instance as db
from typing import Dict, List, Any, Set, Tuple, Iterator
import src.machine_learning_module.utils.random_feature_generators as rand_feat_gen_mod
import src.machine_learning_module.feature_generators as feature_gen_mod
import src.machine_learning_module.utils.hyper_parameter_tuning as hyper_mod
import src.machine_learning_module.label_generators as label_gen_mod
import src.machine_learning_module.filter_generators as filter_gen_mod
import src.machine_learning_module.pipeline as pipe_mod
import src.machine_learning_module.performance_metrics as perf_mod
import src.machine_learning_module.feature_generator_data.talib_momentum as talib_mom_data
"""
In this script, we attempt to use logistic regression

-> absolute change labeling 
-> every kth filtering
-> hyper parameter tuning
-> random feature generation
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
num_feat_generation = 10

# ----- #define --------
MORNING : data.IntraDayPeriod = data.IntraDayPeriod.MORNING
AFTERNOON : data.IntraDayPeriod = data.IntraDayPeriod.AFTERNOON

# ------ parameter dictionaries -------
possible_param_values : Dict[str, List[Any]] = {
    "penalty" : ['l1', 'l2', 'elasticnet', 'none'],
    'tol' : [1e-3, 1e-4, 1e-5],
    'C' : [1.5, 1.0, 0.5, 0.25, 0.1],
    'fit_intercept' : [True, False],
    'class_weights' : ['balanced'],
    'solver' : ['saga'],
}

label_generator_params : Dict[str, Any] = {
    "look_ahead" : look_ahead,
    "threshold" : threshold,
    "criteria" : price_series_used,
}

filter_generator_params : Dict[str, Any] = {
    "period" : period,
    "shift" : shift,
    "criteria" : price_series_used
}

# ---- random gen feature parameters -----
talib_mom_arg_dict_gen_param_dict : Dict[str, Dict[str, int]] = {
    talib_mom_data.ADX : {'timeperiod_min' : 10, 'timeperiod_max' : 30},
    talib_mom_data.ADXR : {'timeperiod_min' : 10, 'timeperiod_max' : 30},
    talib_mom_data.APO : {"fast_min" : 10, "fast_max" : 30, "slow_min" : 10, "slow_max" : 30},
    talib_mom_data.AROON_UP : {'timeperiod_min' : 10, 'timeperiod_max' : 30},
    talib_mom_data.AROON_DOWN :  {'timeperiod_min' : 10, 'timeperiod_max' : 30},
    talib_mom_data.AROONOSC : {'timeperiod_min' : 10, 'timeperiod_max' : 30},
    talib_mom_data.CCI : {'timeperiod_min' : 10, 'timeperiod_max' : 30},
    talib_mom_data.CMO : {'timeperiod_min' : 10, 'timeperiod_max' : 30},
    talib_mom_data.MACD: {"fast_min" : 10, "fast_max" : 30, "slow_min" : 10, "slow_max" : 30, "signal_min" : 10, "signal_max" : 30},
    talib_mom_data.MACD_SIGNAL : {"fast_min" : 10, "fast_max" : 30, "slow_min" : 10, "slow_max" : 30, "signal_min" : 10, "signal_max" : 30},
    talib_mom_data.MACD_HIST: {"fast_min" : 10, "fast_max" : 30, "slow_min" : 10, "slow_max" : 30, "signal_min" : 10, "signal_max" : 30},
    talib_mom_data.MOM : {'timeperiod_min' : 10, 'timeperiod_max' : 30},
    talib_mom_data.PPO : {"fast_min" : 10, "fast_max" : 30, "slow_min" : 10, "slow_max" : 30},
    talib_mom_data.ROC : {'timeperiod_min' : 10, 'timeperiod_max' : 30},
    talib_mom_data.ROCP: {'timeperiod_min' : 10, 'timeperiod_max' : 30},
    talib_mom_data.RSI : {'timeperiod_min' : 10, 'timeperiod_max' : 30},
    talib_mom_data.STOCH_SLOWK : {'fastk_min' : 10, 'fastk_max' : 30, 'slowk_min' : 10, "slowk_max" : 30},
    talib_mom_data.STOCH_SLOWD : {'fastk_min' : 10, 'fastk_max' : 30, 'slowk_min' : 10, "slowk_max" : 30},
    talib_mom_data.STOCHF_FASTK : {'fastk_min' : 10, 'fastk_max' : 30, 'fastd_min' : 10, "fastd_max" : 30},
    talib_mom_data.STOCHF_FASTD : {'fastk_min' : 10, 'fastk_max' : 30, 'fastd_min' : 10, "fastd_max" : 30},
    talib_mom_data.TRIX : {'timeperiod_min' : 10, 'timeperiod_max' : 30},
    talib_mom_data.ULTOSC : {'tp1_min' : 10, 'tp1_max' : 30, 'tp2_min' : 10, 'tp2_max' : 30, 'tp3_min' : 10, "tp3_max" : 30},
    talib_mom_data.WILLR : {'timeperiod_min' : 10, 'timeperiod_max' : 30}
}

talib_mom_repeats_dict = {
    talib_mom_data.ADX: 2,
    talib_mom_data.ADXR: 2,
    talib_mom_data.APO: 2,
    talib_mom_data.AROON_UP:  2,
    talib_mom_data.AROON_DOWN: 2,
    talib_mom_data.AROONOSC: 2,
    talib_mom_data.CCI: 2,
    talib_mom_data.CMO: 2,
    talib_mom_data.MACD: 2,
    talib_mom_data.MACD_SIGNAL: 2,
    talib_mom_data.MACD_HIST: 2,
    talib_mom_data.MOM: 2,
    talib_mom_data.PPO: 2,
    talib_mom_data.ROC: 2,
    talib_mom_data.ROCP: 2,
    talib_mom_data.RSI: 2,
    talib_mom_data.STOCH_SLOWK: 2,
    talib_mom_data.STOCH_SLOWD: 2,
    talib_mom_data.STOCHF_FASTK: 2 ,
    talib_mom_data.STOCHF_FASTD: 2,
    talib_mom_data.TRIX: 2,
    talib_mom_data.ULTOSC: 2,
    talib_mom_data.WILLR: 2
}

# ---- create objects ------
label_generator : label_gen_mod.ClassificationLabelGenerator = label_gen_mod.AbsoluteChangeLabel(**label_generator_params)
filter_generator : filter_gen_mod.FilterGenerator = filter_gen_mod.EveryKthFilter(**filter_generator_params)
talib_mom_param_dict_gen : rand_feat_gen_mod.RandomParaDictGenerator = rand_feat_gen_mod.create_random_param_gen(arg_dict_gen_param_dict = talib_mom_arg_dict_gen_param_dict,
                                                                                                                 feature_name_list = talib_mom_data.FEATURE_NAME_LIST,
                                                                                                                 arg_dict_gen_func = talib_mom_data.UNIFORM_ARG_DICT_GEN_GEN_FUNC,
                                                                                                                 repeats_dict = talib_mom_repeats_dict)
model_hyper_param_iterator : Iterator[Dict[str, Any]] = hyper_mod.generate_combinations(param_values_dict = possible_param_values)

# ----- classification classes -----
classification_classes : Set[int] = label_generator.classification_classes
default_class : int = label_generator.default_class
non_default_classes : Set[int] = {cls for cls in classification_classes if cls is not default_class}

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
# ----- create labels -----
label_df_lst : List[label_gen_mod.LabelDataFrame] = [label_generator.create_labels_for_data_bar(bar_wrapper) for bar_wrapper in bar_wrapper_lst]
# ----- create filters ------
filter_array_lst : List[filter_gen_mod.FilterArray] = [filter_generator.create_filter_for_data_bar(bar_wrapper) for bar_wrapper in bar_wrapper_lst]

# ----- loop through features and hyper parameters -----
for ml_param_dict in model_hyper_param_iterator:
    ml_model = LogisticRegression(**ml_param_dict)
    for _ in range(num_feat_generation):
        talib_mom_param : Dict[str, List[Dict[str, Any]]] = talib_mom_param_dict_gen.generate_param_dict()
        feature_gen : feature_gen_mod.FeatureGenerator = feature_gen_mod.TalibMomentum(parameters_dict = talib_mom_param)
        feature_wrapper_lst = [feature_gen.create_features_for_data_bar(bar_wrapper = bar_wrapper) for bar_wrapper in bar_wrapper_lst]
        # ----- apply filter and remove missing columns -----
        filter_output_lst: List[Tuple[feature_gen_mod.FeatureDataFrame, label_gen_mod.LabelDataFrame]] = [pipe_mod.filter_features_and_labels(feature_wrapper, label_df, filter_array, description)
        for feature_wrapper, label_df, filter_array, description in zip(feature_wrapper_lst, label_df_lst, filter_array_lst, bar_data_description_lst)]
        filtered_feature_df_lst: List[feature_gen_mod.FeatureDataFrame] = []
        filtered_label_array_lst: List[label_gen_mod.LabelDataFrame] = []
        for feature_df, label_array in filter_output_lst:
            filtered_feature_df_lst.append(feature_df)
            filtered_label_array_lst.append(label_array)
        # ----- convert labels into 1 vs all labels --------
        padded_label_array_dict_lst: List[Dict[int, label_gen_mod.LabelDataFrame]] = [
            pipe_mod.convert_to_one_vs_all_labeling(label_wrapper=label_wrapper, classes=classification_classes, default_class=default_class) for
            label_wrapper in filtered_label_array_lst]
        # ----- reorganize the label arrays -------
        padded_label_array_lst_dict: Dict[int, List[label_gen_mod.LabelDataFrame]] = {}
        for cls in non_default_classes:
            array_lst: List[label_gen_mod.LabelDataFrame] = []
            for label_dict in padded_label_array_dict_lst:
                array_lst.append(label_dict[cls])
            padded_label_array_lst_dict[cls] = array_lst
        # ----- cross validation ------
        train_cross_val_result_dict: Dict[int, List[perf_mod.BinaryClassificationMetrics]] = {}
        test_cross_val_result_dict: Dict[int, List[perf_mod.BinaryClassificationMetrics]] = {}
        for cls in non_default_classes:
            padded_label_array_lst = padded_label_array_lst_dict[cls]
            train_cross_val_result_lst, test_cross_val_result_lst = pipe_mod.cross_validate_binary_classification(
                feature_wrapper_lst=filtered_feature_df_lst,
                label_array_lst=padded_label_array_lst,
                ml_model_template=ml_model,
                cross_val_indices=cross_val_indices,
            )
            train_cross_val_result_dict[cls] = train_cross_val_result_lst
            test_cross_val_result_dict[cls] = test_cross_val_result_lst
        # ----- aggregate the results ------
        for cls in non_default_classes:
            train_cross_val_result_lst: List[perf_mod.BinaryClassificationMetrics] = train_cross_val_result_dict[cls]
            test_cross_val_result_lst: List[perf_mod.BinaryClassificationMetrics] = test_cross_val_result_dict[cls]
            train_collated_results: perf_mod.CollatedBinaryClassificationMetrics = perf_mod.CollatedBinaryClassificationMetrics(train_cross_val_result_lst)
            test_collated_results: perf_mod.CollatedBinaryClassificationMetrics = perf_mod.CollatedBinaryClassificationMetrics(test_cross_val_result_lst)
        # ----- keep track of the best performing configurations ------


