import src.data_base_module.data_blocks as data
from src.data_base_module.data_retrival import instance as db
import src.machine_learning_module.label_generators as label_gen_mod
import src.machine_learning_module.filter_generators as filter_gen_mod
import src.machine_learning_module.bootstrapping as boot_mod
from typing import List
import numpy as np

"""
In this series of scripts, we test the label uniqueness of various boot strapping schemes on different labeling schemes

Bars : volume sampled at 20
Filtering scheme : every kth filter
Labeling scheme : absolute price change labeling
Bootstrap scheme : random 

Inputs : 
symbol : stock symbol, only data avaliable right now NHK17
sampling volume : only data avalible right now is 20
price_series_used : open close high or low, default use the close


Outupts : 
Average separation of filtering scheme
Average percentage of labels taken in each boot strap
Average uniqueness across all of the data sets weighted by the number of examples selected
"""
# ------ data loading inputs ------
symbol : str = "NHK17"
sampling_volume : int = 20
num_datasets : int = 40

# ----- labeling -----
look_ahead : int = 30
threshold : float = 25
price_series_used : data.BarDataColumns = data.BarDataColumns.CLOSE
label_generator : label_gen_mod.LabelGenerator = label_gen_mod.AbsoluteChangeLabel(look_ahead = look_ahead, threshold = threshold, criteria = price_series_used)

# ----- filtering -----
filter_period : int = 6
filter_shift : int = 0
filter_name = "every kth"
filter_generator : filter_gen_mod.FilterGenerator = filter_gen_mod.EveryKthFilter(period = filter_period, shift = filter_shift, criteria = price_series_used)

# ----- boot strapping -----
sample_portion : float = 0.75
num_bootstraps : int = 50
boot_generator : boot_mod.BootStrapGenerator = boot_mod.RandomBootStrapGenerator(sample_portion = sample_portion)

# ----- define -----
MORNING = data.IntraDayPeriod.MORNING
AFTERNOON = data.IntraDayPeriod.AFTERNOON

# ----- dates of data used -----
""" 40 data sets in total, 20 days, morning and afternoon """
date_lst : [data.Date] = [
    data.Date(day=25, month = 1, year = 2017),
    data.Date(day=26, month=1, year=2017),]
    #data.Date(day=27, month=1, year=2017),
    #data.Date(day=31, month=1, year=2017),
    #data.Date(day=1, month=2, year=2017),
    #data.Date(day=2, month=2, year=2017),
    #data.Date(day=6, month=2, year=2017),
    #data.Date(day=7, month=2, year=2017),
    #data.Date(day=8, month=2, year=2017),
    #data.Date(day=9, month=2, year=2017),
    #data.Date(day=10, month=2, year=2017),
    #data.Date(day=14, month=2, year=2017),
    #data.Date(day=15, month=2, year=2017),
    #data.Date(day=16, month=2, year=2017),
    #data.Date(day=17, month=2, year=2017),
    #data.Date(day=20, month=2, year=2017),
    #data.Date(day=21, month=2, year=2017),
    #data.Date(day=22, month=2, year=2017),
    #data.Date(day=23, month=2, year=2017),
    #data.Date(day=24, month=2, year=2017)]

# ------ load data ------
morning_bar_wrapper_lst : List[data.BarDataFrame] = [db.get_sampled_volume_bar(symbol = symbol, date = date, sampling_volume = sampling_volume, intra_day_period = MORNING) for date in date_lst]
afternoon_bar_wrapper_lst : List[data.BarDataFrame] = [db.get_sampled_volume_bar(symbol = symbol, date = date, sampling_volume = sampling_volume, intra_day_period = AFTERNOON) for date in date_lst]
bar_wrapper_lst : List[data.BarDataFrame] = [None] * (len(morning_bar_wrapper_lst) + len(afternoon_bar_wrapper_lst))
bar_wrapper_lst[::2] = morning_bar_wrapper_lst
bar_wrapper_lst[1::2] = afternoon_bar_wrapper_lst

# ----- create labels -----
label_df_lst : List[label_gen_mod.LabelDataFrame] = [label_generator.create_labels_for_data_bar(bar_wrapper) for bar_wrapper in bar_wrapper_lst]
# ----- create filters ------
filter_array_lst : List[filter_gen_mod.FilterArray] = [filter_generator.create_filter_for_data_bar(bar_wrapper) for bar_wrapper in bar_wrapper_lst]
# ----- create bootstraps ------
bootstrap_matrix_lst : List[boot_mod.BootStrapMatrix] = [boot_generator.generate_bootstrap_array(label_wrapper = label_wrapper, filter_wrapper = filter_wrapper,num_bootstraps = num_bootstraps)
                                                         for label_wrapper, filter_wrapper in zip(label_df_lst, filter_array_lst)]

# ----- calculate mean number of examples selected for each dataset -----
mean_num_selected_lst : List[float] = []
for bs_matrix, filter_wrapper in zip(bootstrap_matrix_lst, filter_array_lst):
    num_selected_lst : List[int] = []
    for bs_row in bs_matrix.get_rows_it():
        num_selected = len(bs_row[bs_row])
        num_selected_lst.append(num_selected)
    mean_num_selected = sum(num_selected_lst) / len(num_selected_lst) if len(num_selected_lst) else 0
    mean_num_selected_lst.append(mean_num_selected)

# ----- calculate uniqueness for each boot strap -------
uniqueness_lst : List[float] = []
for bs_matrix, label_wrapper, filter_wrapper in zip(bootstrap_matrix_lst, label_df_lst, filter_array_lst):
    single_uni_lst : List[float] = []
    for i,bs_row in enumerate(bs_matrix.get_rows_it()):
        uniqueness = boot_mod.find_average_uniqueness(label_wrapper = label_wrapper, filter_wrapper = filter_wrapper, boot_strap_row = bs_row, bootstrap_description = f"trial : {i + 1}")
        single_uni_lst.append(uniqueness)
    mean_uniqueness = sum(single_uni_lst) / len(single_uni_lst) if len(single_uni_lst) else 0
    uniqueness_lst.append(mean_uniqueness)

# ----- find proportion of examples selected in each boot strap -----
num_examples_lst : List[int] = []
for filter_wrapper in filter_array_lst:
    filter_arr : np.array = filter_wrapper.get_filter_array_ref()
    num_examples : int = len(filter_arr[filter_arr])
    num_examples_lst.append(num_examples)
proportion_lst : List[float] = [(selected / total if total > 0 else 0) for selected,total in zip(mean_num_selected_lst, num_examples_lst)]

# ----- find the weighted average uniqueness across all datasets ------
total_selected : float = sum(mean_num_selected_lst)
weighted_avg_uniqueness : float = np.dot(uniqueness_lst, mean_num_selected_lst) / total_selected if total_selected else 0

# ----- find the weighted proportion across all datasets ------
total_examples : int = sum(num_examples_lst)
weighted_avg_proportion : float = np.dot(proportion_lst, num_examples_lst) / total_examples if total_examples else 0

# ----- filter array stats -----
filter_stats_lst : List[filter_gen_mod.FilterStats] = [filter_gen_mod.find_filter_stats(filter_wrapper, filter_name) for filter_wrapper in filter_array_lst]
mean_separation_lst : List[float] = [filter_stat.mean_spacing for filter_stat in filter_stats_lst]
weighted_avg_separation : float = np.dot(mean_separation_lst, num_examples_lst) / total_examples if total_examples else 0

# ----- print out results ------
print(f"weighted average uniqueness : {weighted_avg_uniqueness}")
print(f"weighted average proportion : {weighted_avg_proportion}")
print(f"weighted average filter separation : {weighted_avg_separation}")

# ----- print out detailed lists ------
print(f"uniqueness list           : {uniqueness_lst}")
print(f"mean number selected list : {mean_num_selected_lst}")
print(f"number examples list      : {num_examples_lst}")
print(f"mean proportion list      : {proportion_lst}")
print(f"filter separation list    : {mean_separation_lst}")

