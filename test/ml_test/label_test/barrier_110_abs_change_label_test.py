import src.machine_learning_module.label_generators as label_gen_mod
import src.data_base_module.data_blocks as data
import pandas as pd
import numpy as np

# ----- test data ------
test_symbol : str = "TEST"

open_str : str = data.BarDataColumns.OPEN.value
close_str : str = data.BarDataColumns.CLOSE.value
high_str : str = data.BarDataColumns.HIGH.value
low_str : str = data.BarDataColumns.LOW.value
timestamp_str : str = data.BarDataColumns.TIMESTAMP.value
vol_str : str = data.BarDataColumns.VOLUME.value
vwap_str : str = data.BarDataColumns.VWAP.value

bar1 = {open_str : 25, close_str : 25, high_str : 25, low_str : 25, timestamp_str : 1 , vol_str : 412, vwap_str : 25}
bar2 = {open_str : 29, close_str : 27.99, high_str : 28.01, low_str : 25, timestamp_str : 2 , vol_str : 592, vwap_str : 25}
bar3 = {open_str : 25, close_str : 22, high_str : 21.99, low_str : 27.9, timestamp_str : 3 , vol_str : 239, vwap_str : 35}
bar4 = {open_str : 22, close_str : 25, high_str : 25, low_str : 22, timestamp_str : 4 , vol_str : 352, vwap_str : 20}
bar5 = {open_str : 25, close_str : 22.01, high_str : 21.99, low_str : 22, timestamp_str : 5 , vol_str : 439, vwap_str : 20}
bar6 = {open_str : 25, close_str : 28, high_str : 28.01, low_str : 19.1, timestamp_str : 6 , vol_str : 239, vwap_str : 22}
bar7 = {open_str : 25, close_str : 25, high_str : 25, low_str : 25, timestamp_str : 7 , vol_str : 322, vwap_str : 10}
bar8 = {open_str : 25, close_str : 25, high_str : 25, low_str : 25, timestamp_str : 8 , vol_str : 401, vwap_str : 10}
bar9 = {open_str : 29, close_str : 25, high_str : 25, low_str : 25, timestamp_str : 9 , vol_str : 401, vwap_str : 10}
bar10 = {open_str : 25, close_str : 25, high_str : 25, low_str : 25, timestamp_str : 10 , vol_str : 401, vwap_str : 10}
bar11 = {open_str : 25, close_str : 25, high_str : 25, low_str : 25, timestamp_str : 11, vol_str : 401, vwap_str : 10}
bar12 = {open_str : 22, close_str : 28, high_str : 22, low_str : 27.999, timestamp_str : 12 , vol_str : 401, vwap_str : 100}

bar_df : pd.DataFrame = pd.DataFrame([bar1, bar2, bar3, bar4, bar5, bar6, bar7, bar8, bar9, bar10, bar11, bar12])
bar_wrapper = data.BarDataFrame(symbol = test_symbol)
bar_wrapper.set_data_frame(bar_df)

# ------ answers -------
# ------------- test case for open series, upper barrier = 4, lower barrier = 3 --------
answer_open_label_series = pd.Series([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 0])
answer_open_look_ahead_series = pd.Series([1, 1, 1, 5, 4, 3, 2, 1, 1, 2, 1, 0])

# ------------ test case for close series, upper barrier = 3, lower barrier = 3 --------
answer_close_label_series = pd.Series([-1, -1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 0])
answer_close_look_ahead_series = pd.Series([2, 1, 1, 2, 1, 1, 5, 4, 3, 2, 1, 0])

# ------------ test case for high series, upper barrier = 3, lower barrier = 3 ---------
answer_high_label_series = pd.Series([1, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 0])
answer_high_look_ahead_series = pd.Series([1, 1, 1, 1, 1, 1, 5, 4, 3, 2, 1, 0])

# ------------ test case for low series, upper barrier = 3, lower barrier = 3 ----------
answer_low_label_series = pd.Series([-1, -1, -1, 1, 1, 1, 0 ,0 , 0, 0, 0, 0])
answer_low_look_ahead_series = pd.Series([3, 2, 1, 3, 2, 1, 5, 4, 3, 2, 1, 0])

# ----------- test case for vwap series, lower barrier = 3, upper barrier = 3 ----------
answer_vwap_label_series = pd.Series([1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 0])
answer_vwap_look_ahead_series = pd.Series([2, 1, 1, 3, 2, 1, 5, 4, 3, 2, 1, 0])

# ------ initialize label generators --------
label_gen_open : label_gen_mod.LabelGenerator = label_gen_mod.Barrier110AbsChangeLabel(upper_barrier = 4.0, lower_barrier = 3.0, criteria = data.BarDataColumns.OPEN)
label_gen_close : label_gen_mod.LabelGenerator = label_gen_mod.Barrier110AbsChangeLabel(upper_barrier = 3.0, lower_barrier = 3.0, criteria = data.BarDataColumns.CLOSE)
label_gen_high : label_gen_mod.LabelGenerator = label_gen_mod.Barrier110AbsChangeLabel(upper_barrier = 3.0, lower_barrier = 3.0, criteria = data.BarDataColumns.HIGH)
label_gen_low : label_gen_mod.LabelGenerator = label_gen_mod.Barrier110AbsChangeLabel(upper_barrier = 3.0, lower_barrier = 3.0, criteria = data.BarDataColumns.LOW)
label_gen_vwap : label_gen_mod.LabelGenerator = label_gen_mod.Barrier110AbsChangeLabel(upper_barrier = 3.0, lower_barrier = 3.0, criteria = data.BarDataColumns.VWAP)

# ------ get labeling results ------
label_result_open : label_gen_mod.LabelDataFrame = label_gen_open.create_labels_for_data_bar(bar_wrapper)
label_result_close : label_gen_mod.LabelDataFrame = label_gen_close.create_labels_for_data_bar(bar_wrapper)
label_result_high : label_gen_mod.LabelDataFrame = label_gen_high.create_labels_for_data_bar(bar_wrapper)
label_result_low : label_gen_mod.LabelDataFrame = label_gen_low.create_labels_for_data_bar(bar_wrapper)
label_result_vwap : label_gen_mod.LabelDataFrame = label_gen_vwap.create_labels_for_data_bar(bar_wrapper)

# ------ compare results with answer -------
open_compare_df = pd.DataFrame({
    "result_label" : label_result_open.get_label_df_ref().loc[:, label_result_open.label_col_name],
    "result_look_ahead" : label_result_open.get_label_df_ref().loc[:, label_result_open.look_ahead_col_name],
    "answer_label" : answer_open_label_series,
    "answer_look_ahead" : answer_open_look_ahead_series
})

close_compare_df = pd.DataFrame({
    "result_label": label_result_close.get_label_df_ref().loc[:, label_result_close.label_col_name],
    "result_look_ahead": label_result_close.get_label_df_ref().loc[:, label_result_close.look_ahead_col_name],
    "answer_label": answer_close_label_series,
    "answer_look_ahead": answer_close_look_ahead_series
})

high_compare_df = pd.DataFrame({
    "result_label": label_result_high.get_label_df_ref().loc[:, label_result_high.label_col_name],
    "result_look_ahead": label_result_high.get_label_df_ref().loc[:, label_result_high.look_ahead_col_name],
    "answer_label": answer_high_label_series,
    "answer_look_ahead": answer_high_look_ahead_series
})

low_compare_df = pd.DataFrame({
    "result_label": label_result_low.get_label_df_ref().loc[:, label_result_low.label_col_name],
    "result_look_ahead": label_result_low.get_label_df_ref().loc[:, label_result_low.look_ahead_col_name],
    "answer_label": answer_low_label_series,
    "answer_look_ahead": answer_low_look_ahead_series
})

vwap_compare_df = pd.DataFrame({
    "result_label": label_result_vwap.get_label_df_ref().loc[:, label_result_vwap.label_col_name],
    "result_look_ahead": label_result_vwap.get_label_df_ref().loc[:, label_result_vwap.look_ahead_col_name],
    "answer_label": answer_vwap_label_series,
    "answer_look_ahead": answer_vwap_look_ahead_series
})

# ------ print out compare data frames ------
print(" ----- OPEN test case ------ ")
print(open_compare_df)
print(" ----- CLOSE test case ------ ")
print(close_compare_df)
print(" ----- HIGH test case ------ ")
print(high_compare_df)
print(" ----- LOW test case ------ ")
print(low_compare_df)
print(" ----- VWAP test case ------ ")
print(vwap_compare_df)