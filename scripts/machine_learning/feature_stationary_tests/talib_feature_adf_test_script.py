import pandas as pd
from typing import List
from statsmodels.tsa.stattools import adfuller
import src.data_base_module.data_blocks as data
from src.data_base_module.data_retrival import instance as db
import src.machine_learning_module.feature_generators as feature_gen_mod
import src.data_processing_module.data_cleaning as data_cleaner

# ----- user inputs ----
symbol = "NHK17"
sampling_seconds = 60
sampling_ticks = 200
sampling_volume = 20
sampling_dollar = 500000

# ----- date list -----
#date_lst = [data.Date(day=25, month = 1, year = 2017)]

date_lst : [data.Date] = [
    data.Date(day=25, month = 1, year = 2017),
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
    data.Date(day=24, month=2, year=2017),
    data.Date(day=27, month=2, year=2017),
    data.Date(day=28, month=2, year=2017),
    data.Date(day=1, month=3, year=2017),
    data.Date(day=2, month=3, year=2017),
    data.Date(day=3, month=3, year=2017),
    data.Date(day=6, month=3, year=2017),
    data.Date(day=7, month=3, year=2017)]

# ----- load data -----
# ------- time sampled bars -------
#morning_bar_data_wrapper_lst: List[data.BarDataFrame] = [db.get_sampled_time_bar(symbol=symbol, date=date, intra_day_period=data.IntraDayPeriod.MORNING,sampling_seconds=sampling_seconds) for date in date_lst]
#afternoon_bar_data_wrapper_lst: List[data.BarDataFrame] = [db.get_sampled_time_bar(symbol=symbol, date=date, intra_day_period=data.IntraDayPeriod.AFTERNOON,sampling_seconds=sampling_seconds) for date in date_lst]
# ------- tick sampled bars ------
#morning_bar_data_wrapper_lst: List[data.BarDataFrame] = [db.get_sampled_tick_bar(symbol=symbol, date=date, intra_day_period=data.IntraDayPeriod.MORNING,sampling_ticks=sampling_ticks) for date in date_lst]
#afternoon_bar_data_wrapper_lst: List[data.BarDataFrame] = [db.get_sampled_tick_bar(symbol=symbol, date=date, intra_day_period=data.IntraDayPeriod.AFTERNOON,sampling_ticks=sampling_ticks) for date in date_lst]
# ------- volume sampled bars ------
#morning_bar_data_wrapper_lst: List[data.BarDataFrame] = [db.get_sampled_volume_bar(symbol=symbol, date=date, intra_day_period=data.IntraDayPeriod.MORNING,sampling_volume=sampling_volume) for date in date_lst]
#afternoon_bar_data_wrapper_lst: List[data.BarDataFrame] = [db.get_sampled_volume_bar(symbol=symbol, date=date, intra_day_period=data.IntraDayPeriod.AFTERNOON,sampling_volume=sampling_volume) for date in date_lst]
# ------ dollar sampled bars -------
morning_bar_data_wrapper_lst: List[data.BarDataFrame] = [db.get_sampled_dollar_bar(symbol=symbol, date=date, intra_day_period=data.IntraDayPeriod.MORNING,sampling_dollar=sampling_dollar) for date in date_lst]
afternoon_bar_data_wrapper_lst: List[data.BarDataFrame] = [db.get_sampled_dollar_bar(symbol=symbol, date=date, intra_day_period=data.IntraDayPeriod.AFTERNOON,sampling_dollar=sampling_dollar) for date in date_lst]

bar_data_wrapper_lst = []
for morning_bar, afternoon_bar in zip(morning_bar_data_wrapper_lst, afternoon_bar_data_wrapper_lst):
    bar_data_wrapper_lst.append(morning_bar)
    bar_data_wrapper_lst.append(afternoon_bar)

# ----- interpolate zero rows  ------
bar_data_wrapper_lst : List[data.BarDataFrame] = [data_cleaner.interpolate_bar_zero_prices(bar_wrapper = bar_wrapper) for bar_wrapper in bar_data_wrapper_lst]

# ----- create features ------
#feature_gen : feature_gen_mod.FeatureGenerator = feature_gen_mod.TalibMomentum()
feature_gen : feature_gen_mod.FeatureGenerator = feature_gen_mod.TalibVolatility()
feature_data_frame_lst = [feature_gen.create_features_for_data_bar(bar_data) for bar_data in bar_data_wrapper_lst]
feature_data_frame_lst = [feature_df.dropna() for feature_df in feature_data_frame_lst]
compiled_feature_df = pd.concat(feature_data_frame_lst, axis = 0, ignore_index = True)

# ---- stationary tests -------
print("# ------ checking for stationarity ------ #")
for col_name in compiled_feature_df.columns:
    feature_series = compiled_feature_df[col_name]
    result = adfuller(feature_series)
    print(col_name + " -- " + str(result[0]))

