import pandas as pd
import src.data_base_module.data_blocks as data
from typing import List, Dict
from src.data_base_module.data_retrival import instance as db
import src.machine_learning_module.feature_generators as feature_gen_mod
import src.plotting_module.base_plotting_functions as base_plot_mod
import matplotlib.pyplot as plt

# ------ import feature generator data -------
import src.machine_learning_module.feature_generator_data.talib_momentum as talib_mom_data
import src.machine_learning_module.feature_generator_data.talib_volatility as talib_vol_data
import src.machine_learning_module.feature_generator_data.talib_pattern as talib_pat_data

# ----- user inputs ----
symbol = "NHK17"
sampling_volume = 20
date = data.Date(day = 25, month = 1, year = 2017)

# ----- load data -----
bar_data_wrapper = db.get_sampled_volume_bar(symbol = symbol, date = date, intra_day_period = data.IntraDayPeriod.MORNING, sampling_volume = sampling_volume)

# ---- feature parameters -----
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

talib_vol_param_dict : Dict[str, List[Dict[str, float]]] = {
    talib_vol_data.ATR : [{'timeperiod' : 15}],
    talib_vol_data.NATR : [{'timeperiod' : 16}],
    talib_vol_data.TRANGE : [{}],
}

talib_pat_param_dict :  Dict[str, List[Dict[str, float]]] = {feature_name : [{}] for feature_name in talib_pat_data.FEATURE_NAME_LIST}
# ---- generate feature -----
#feature_gen : feature_gen_mod.FeatureGenerator = feature_gen_mod.TalibMomentum(parameters_dict = talib_momentum_param_dict)
#feature_gen : feature_gen_mod.FeatureGenerator = feature_gen_mod.TalibVolatility(parameters_dict = talib_vol_param_dict)
feature_gen : feature_gen_mod.FeatureGenerator = feature_gen_mod.TalibPattern(parameters_dict = talib_pat_param_dict)
feature_wrapper : feature_gen_mod.FeatureDataFrame = feature_gen.create_features_for_data_bar(bar_wrapper = bar_data_wrapper)
feature_df : pd.DataFrame = feature_wrapper.get_feature_df()

# ---- print parameters -----
for feat_name, argument_dict_lst in feature_wrapper.parameter_dict.items():
    print(f"Feature : {feat_name}")
    for argument_dict in argument_dict_lst:
        print(argument_dict)

print(feature_wrapper.dataset_description_lst)
print(feature_wrapper.feature_generator_name_lst)

# ---- plot -----
for feat_name, feat_ser in feature_df.items():
    plt.plot(range(len(feat_ser)),feat_ser)
    plt.title(feat_name)
    plt.show()

