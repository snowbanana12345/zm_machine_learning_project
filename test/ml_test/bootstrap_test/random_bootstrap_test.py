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

# ----- check if bootstrapping only picks up points that are filtered -----
filter_arr : np.array = filter_wrapper.get_filter_array_ref()
for i,row in enumerate(boot_strap_matrix.get_matrix()):
    if any(row[np.logical_not(filter_arr)]):
        print(f"Row {i} : has included a entry on the bootstrap that has not passed the filter")

# ----- find statistics on bootstrap -----
# ----- find % of the total filtered points that are taken -----
percentages = []
for i, row in enumerate(boot_strap_matrix.get_matrix()):
    filtered_row = row[filter_arr]
    percentage = sum(filtered_row) / len(filtered_row) * 100
    print(f"Row {i} : has {percentage} % of the data set selected")
    percentages.append(percentage)
mean_pct = sum(percentages) / len(percentages)

# ----- find the number of times each valid example has been selected -----
count_arr = np.zeros(len(filter_arr))
for row in boot_strap_matrix.get_matrix():
    count_arr += row
count_arr = count_arr[filter_arr]
print(f"occupancy of each bootstrapped example : mean : {np.mean(count_arr)} -- sd : {np.std(count_arr)}")


