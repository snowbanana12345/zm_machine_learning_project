from typing import List
import src.data_base_module.data_blocks as data
from src.data_base_module.data_retrival import instance as db
import src.machine_learning_module.feature_generators as feature_gen
import src.data_processing_module.data_cleaning as data_cleaner
import pandas as pd
import src.plotting_module.base_plotting_functions as base_plot_mod
import matplotlib.pyplot as plt

# ----- user inputs ----
symbol = "NHK17"
sampling_seconds = 60
sampling_ticks = 200
sampling_volume = 20
sampling_dollar = 500000
d = 0.1
window_size = 40

# ----- date list -----
# date_lst = [data.Date(day=25, month = 1, year = 2017)]

date_lst: [data.Date] = [
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

# ----- merge morning and after noon lists -----
bar_data_wrapper_lst : List[data.BarDataFrame] = []
for morning_bar, afternoon_bar in zip(morning_bar_data_wrapper_lst, afternoon_bar_data_wrapper_lst):
    bar_data_wrapper_lst.append(morning_bar)
    bar_data_wrapper_lst.append(afternoon_bar)

# ----- interpolate zero rows  ------
bar_data_wrapper_lst : List[data.BarDataFrame] = [data_cleaner.interpolate_bar_zero_prices(bar_wrapper = bar_wrapper) for bar_wrapper in bar_data_wrapper_lst]

# ----- perform fractional differencing -------
frac_diff_bar_wrapper_lst : List[feature_gen.FracDiffBarDataFrame] = [feature_gen.frac_diff_bar_data_frame_wrapper(bar_wrapper, window_size = window_size, d = d)
                                  for bar_wrapper in bar_data_wrapper_lst]

# ----- drop missing rows and merge data frames ------
frac_diff_bar_lst: [pd.DataFrame] = [frac_diff_bar.get_bar_data_reference().dropna() for frac_diff_bar in
                                          frac_diff_bar_wrapper_lst]
frac_diff_bar_df = pd.concat(frac_diff_bar_lst, axis=0, ignore_index=True)

# ----- plotting ------
for col_name in frac_diff_bar_df.columns:
    series: pd.Series = frac_diff_bar_df[col_name]
    base_plot_mod.plot_xy(series.index.values, series.values, x_label="bar no", y_label=col_name,
                          title="Series : " + col_name)
    plt.show()
