import numpy as np
import src.data_base_module.data_blocks as data
from src.data_base_module.data_retrival import instance as db
import src.machine_learning_module.filter_generators as filter_gen_mod
import src.plotting_module.machine_learning_plotter as ml_plt
import matplotlib.pyplot as plt

# ----- user inputs ----
symbol = "NHK17"
sampling_seconds = 60
sampling_ticks = 200
sampling_volume = 20
sampling_dollar = 500000
date = data.Date(day = 26, month = 1, year = 2017)
intra_day_period = data.IntraDayPeriod.MORNING
col_name : data.BarDataColumns = data.BarDataColumns.CLOSE
filter_period : int = 100
filter_shift : int = 75

# ----- load data ------
bar_wrapper : data.BarDataFrame = db.get_sampled_volume_bar(symbol = symbol, sampling_volume = sampling_volume, date = date, intra_day_period = intra_day_period)

# ----- filter class ------
filter_gen : filter_gen_mod.FilterGenerator = filter_gen_mod.EveryKthFilter(period = filter_period, shift = filter_shift, criteria = col_name)

# ----- create filter ------
filter_array : np.array = filter_gen.create_filter_for_data_bar(bar_wrapper = bar_wrapper)

# ----- compute and print filter statistics -----
filter_stats : filter_gen_mod.FilterStats = filter_gen_mod.find_filter_stats(filter_array = filter_array, filter_name = filter_gen.name)
filter_gen_mod.print_filter_stats(filter_stats = filter_stats)

# ----- plot series -----
ml_plt.plot_price_series_with_filter(price_series = bar_wrapper.get_column(col_name), filter_array = filter_array, price_series_name = col_name.value, filter_name = filter_gen.name)
plt.show()


