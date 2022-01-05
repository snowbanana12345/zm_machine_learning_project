from statsmodels.tsa.stattools import adfuller
import src.data_base_module.data_blocks as data
from src.data_base_module.data_retrival import instance as db
import src.machine_learning_module.feature_generators as feature_gen
import src.data_processing_module.data_cleaning as data_cleaner
import pandas as pd

# ----- user inputs ----
symbol = "NHK17"
sampling_seconds = 60
sampling_ticks = 200
sampling_volume = 20
sampling_dollar = 500000
d = 0.6
window_size = 40

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
morning_time_bar_data_wrapper_lst = [db.get_sampled_time_bar(symbol = symbol, date = date,intra_day_period = data.IntraDayPeriod.MORNING,
                                            sampling_seconds = sampling_seconds) for date in date_lst]
afternoon_time_bar_data_wrapper_lst = [db.get_sampled_time_bar(symbol = symbol, date = date,intra_day_period = data.IntraDayPeriod.AFTERNOON,
                                            sampling_seconds = sampling_seconds) for date in date_lst]
morning_tick_bar_data_wrapper_lst = [db.get_sampled_tick_bar(symbol = symbol, date = date,intra_day_period = data.IntraDayPeriod.MORNING,
                                                          sampling_ticks = sampling_ticks) for date in date_lst]
afternoon_tick_bar_data_wrapper_lst = [db.get_sampled_tick_bar(symbol = symbol, date = date,intra_day_period = data.IntraDayPeriod.AFTERNOON,
                                                          sampling_ticks = sampling_ticks) for date in date_lst]
morning_vol_bar_data_wrapper_lst = [db.get_sampled_volume_bar(symbol = symbol, date = date,intra_day_period = data.IntraDayPeriod.MORNING,
                                                          sampling_volume = sampling_volume) for date in date_lst]
afternoon_vol_bar_data_wrapper_lst = [db.get_sampled_volume_bar(symbol = symbol, date = date,intra_day_period = data.IntraDayPeriod.AFTERNOON,
                                                          sampling_volume = sampling_volume) for date in date_lst]
morning_dollar_bar_data_wrapper_lst = [db.get_sampled_dollar_bar(symbol = symbol, date = date,intra_day_period = data.IntraDayPeriod.MORNING,
                                                          sampling_dollar = sampling_dollar) for date in date_lst]
afternoon_dollar_bar_data_wrapper_lst = [db.get_sampled_dollar_bar(symbol = symbol, date = date,intra_day_period = data.IntraDayPeriod.AFTERNOON,
                                                          sampling_dollar = sampling_dollar) for date in date_lst]

# ----- merge morning and after noon lists -----
time_bar_data_wrapper_lst  = []
for morning_bar, afternoon_bar in zip(morning_time_bar_data_wrapper_lst, afternoon_time_bar_data_wrapper_lst):
    time_bar_data_wrapper_lst.append(morning_bar)
    time_bar_data_wrapper_lst.append(afternoon_bar)

tick_bar_data_wrapper_lst = []
for morning_bar, afternoon_bar in zip(morning_tick_bar_data_wrapper_lst, afternoon_tick_bar_data_wrapper_lst):
    tick_bar_data_wrapper_lst.append(morning_bar)
    tick_bar_data_wrapper_lst.append(afternoon_bar)

vol_bar_data_wrapper_lst = []
for morning_bar, afternoon_bar in zip(morning_vol_bar_data_wrapper_lst, afternoon_vol_bar_data_wrapper_lst):
    vol_bar_data_wrapper_lst.append(morning_bar)
    vol_bar_data_wrapper_lst.append(afternoon_bar)

dollar_bar_data_wrapper_lst = []
for morning_bar, afternoon_bar in zip(morning_dollar_bar_data_wrapper_lst, afternoon_dollar_bar_data_wrapper_lst):
    dollar_bar_data_wrapper_lst.append(morning_bar)
    dollar_bar_data_wrapper_lst.append(afternoon_bar)
# ----- interpolate zero rows in time and tick data frames ------
time_bar_data_wrapper_lst = [data_cleaner.interpolate_bar_zero_prices(bar_wrapper) for bar_wrapper in time_bar_data_wrapper_lst]
tick_bar_data_wrapper_lst = [data_cleaner.interpolate_bar_zero_prices(bar_wrapper) for bar_wrapper in tick_bar_data_wrapper_lst]
vol_bar_data_wrapper_lst = [data_cleaner.interpolate_bar_zero_prices(bar_wrapper) for bar_wrapper in vol_bar_data_wrapper_lst]
dollar_bar_data_wrapper_lst = [data_cleaner.interpolate_bar_zero_prices(bar_wrapper) for bar_wrapper in dollar_bar_data_wrapper_lst]

# ----- perform fractional differencing -------
frac_diff_time_bar_wrapper_lst = [feature_gen.frac_diff_bar_data_frame_wrapper(bar_df_wrapper = time_bar_df, d = d, window_size = window_size)
                          for time_bar_df in time_bar_data_wrapper_lst]
frac_diff_tick_bar_wrapper_lst = [feature_gen.frac_diff_bar_data_frame_wrapper(bar_df_wrapper = tick_bar_df, d = d, window_size = window_size)
                          for tick_bar_df in tick_bar_data_wrapper_lst ]
frac_diff_vol_bar_wrapper_lst = [feature_gen.frac_diff_bar_data_frame_wrapper(bar_df_wrapper = vol_bar_df, d = d, window_size = window_size)
                          for vol_bar_df in vol_bar_data_wrapper_lst ]
frac_diff_dollar_bar_wrapper_lst = [feature_gen.frac_diff_bar_data_frame_wrapper(bar_df_wrapper = dollar_bar_df, d = d, window_size = window_size)
                          for dollar_bar_df in dollar_bar_data_wrapper_lst]

# ----- drop missing rows and merge data frames ------
frac_diff_time_bar_lst : [pd.DataFrame] = [frac_diff_bar.get_bar_data_reference().dropna() for frac_diff_bar in frac_diff_time_bar_wrapper_lst]
frac_diff_tick_bar_lst : [pd.DataFrame] = [frac_diff_bar.get_bar_data_reference().dropna() for frac_diff_bar in frac_diff_tick_bar_wrapper_lst]
frac_diff_vol_bar_lst : [pd.DataFrame] = [frac_diff_bar.get_bar_data_reference().dropna() for frac_diff_bar in frac_diff_vol_bar_wrapper_lst]
frac_diff_dollar_bar_lst : [pd.DataFrame] = [frac_diff_bar.get_bar_data_reference().dropna() for frac_diff_bar in frac_diff_dollar_bar_wrapper_lst]

frac_diff_time_bar_df = pd.concat(frac_diff_time_bar_lst, axis = 0, ignore_index = True)
frac_diff_tick_bar_df = pd.concat(frac_diff_tick_bar_lst, axis = 0, ignore_index = True)
frac_diff_vol_bar_df = pd.concat(frac_diff_vol_bar_lst, axis = 0, ignore_index = True)
frac_diff_dollar_bar_df = pd.concat(frac_diff_dollar_bar_lst, axis = 0, ignore_index = True)

# ----- perform adf tests ------
print("----- ADF TEST on Time bars -- d = " + str(d) + " -- window size : " + str(window_size) + " ------")
for col in frac_diff_time_bar_df.columns:
    frac_diff_series = frac_diff_time_bar_df[col]
    adf_result = adfuller(frac_diff_series)
    print(col + " -- " + str(adf_result[0]))

print("----- ADF TEST on Tick bars -- d = " + str(d) + " -- window size : " + str(window_size) + " ------")
for col in frac_diff_tick_bar_df.columns:
    frac_diff_series = frac_diff_tick_bar_df[col]
    adf_result = adfuller(frac_diff_series)
    print(col + " -- " + str(adf_result[0]))

print("----- ADF TEST on Volume bars -- d = " + str(d) + " -- window size : " + str(window_size) + " ------")
for col in frac_diff_vol_bar_df.columns:
    frac_diff_series = frac_diff_vol_bar_df[col]
    adf_result = adfuller(frac_diff_series)
    print(col + " -- " + str(adf_result[0]))

print("----- ADF TEST on Dollar bars -- d = " + str(d) + " -- window size : " + str(window_size) + " ------")
for col in frac_diff_dollar_bar_df.columns:
    frac_diff_series = frac_diff_dollar_bar_df[col]
    adf_result = adfuller(frac_diff_series)
    print(col + " -- " + str(adf_result[0]))








