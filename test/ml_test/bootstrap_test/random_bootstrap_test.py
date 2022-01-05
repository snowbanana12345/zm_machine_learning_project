import src.machine_learning_module.bootstrapping as boot_mod
import src.machine_learning_module.label_generators as label_mod
import src.machine_learning_module.filter_generators as filter_mod
from src.data_base_module.data_retrival import instance as db
import src.data_base_module.data_blocks as data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sample_portion = 0.6
symbol = "NHK17"
date = data.Date(day = 25, month = 1, year = 2017)
MORNING = data.IntraDayPeriod.MORNING
sampling_volume = 20
look_ahead = 20
threshold = 20
num_boot_straps = 10
column_used = data.BarDataColumns.CLOSE

# ----- generators -----
test_bootstrap_gen : boot_mod.BootStrapGenerator = boot_mod.RandomBootStrapGenerator(sample_portion = sample_portion)
label_gen : label_mod.LabelGenerator = label_mod.AbsoluteChangeLabel(look_ahead = look_ahead, threshold = threshold,criteria = column_used)
filter_gen : filter_mod.FilterGenerator = filter_mod.IdentityFilter()

# ----- load data -----
vol_bar : data.BarDataFrame = db.get_sampled_volume_bar(symbol = symbol, date = date, sampling_volume = sampling_volume, intra_day_period = MORNING)

# ----- create label ------
label_wrapper : label_mod.LabelDataFrame = label_gen.create_labels_for_data_bar(bar_wrapper = vol_bar)
filter_wrapper : filter_mod.FilterArray = filter_gen.create_filter_for_data_bar(bar_wrapper = vol_bar)

# ----- do bootstrapping -----
boot_strap_matrix : boot_mod.BootStrapMatrix = test_bootstrap_gen.generate_bootstrap_array(label_wrapper = label_wrapper, filter_wrapper = filter_wrapper, num_bootstraps = num_boot_straps)

# ----- plot bootstrap -----
filter_arr : np.array = filter_wrapper.get_filter_array_ref()
price_series : pd.Series = vol_bar.get_column(col_name = column_used)
filter_points : pd.Series = price_series.loc[filter_arr]
for boot_strap_row in boot_strap_matrix.get_rows_it():
    select_arr : np.array = np.logical_and(boot_strap_row, filter_arr)
    select_points : pd.Series = price_series.loc[select_arr]
    plt.plot(price_series.index, price_series, label = "price series")
    plt.scatter(filter_points.index, filter_points, label = "filter points")
    plt.scatter(select_points.index, select_points, label = "selected points")
    plt.ylabel("price")
    plt.xlabel("bar number")
    plt.legend()
    plt.show()
    plt.clf()







