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

bar1 = {open_str : 25, close_str : 29, high_str : 31, low_str : 31, timestamp_str : 1 , vol_str : 412, vwap_str : 25}
bar2 = {open_str : 26, close_str : 29, high_str : 30, low_str : 30, timestamp_str : 2 , vol_str : 592, vwap_str : 26}
bar3 = {open_str : 31.3, close_str : 29, high_str : 29, low_str : 29, timestamp_str : 3 , vol_str : 239, vwap_str : 27}
bar4 = {open_str : 28.9, close_str : 29, high_str : 28, low_str : 28.0001, timestamp_str : 4 , vol_str : 352, vwap_str : 28}
bar5 = {open_str : 29, close_str : 29, high_str : 27, low_str : 28, timestamp_str : 5 , vol_str : 439, vwap_str : 22}
bar6 = {open_str : 31, close_str : 31.9, high_str : 31, low_str : 22, timestamp_str : 6 , vol_str : 239, vwap_str : 20}
bar7 = {open_str : 32, close_str : 26.1, high_str : 32, low_str : 23, timestamp_str : 7 , vol_str : 322, vwap_str : 19}
bar8 = {open_str : 26, close_str : 26, high_str : 28, low_str : 24, timestamp_str : 8 , vol_str : 401, vwap_str : 22}
bar9 = {open_str : 29, close_str : 32, high_str : 27, low_str : 24.9999, timestamp_str : 9 , vol_str : 401, vwap_str : 25}
bar10 = {open_str : 35, close_str : 23.0001, high_str : 32, low_str : 25, timestamp_str : 10 , vol_str : 401, vwap_str : 22}
bar11 = {open_str : 26, close_str : 29, high_str : 33, low_str : 25, timestamp_str : 11, vol_str : 401, vwap_str : 22}
bar12 = {open_str : 30, close_str : 29, high_str : 34, low_str : 25, timestamp_str : 12 , vol_str : 401, vwap_str : 19}

bar_df : pd.DataFrame = pd.DataFrame([bar1, bar2, bar3, bar4, bar5, bar6, bar7, bar8, bar9, bar10, bar11, bar12])
bar_wrapper = data.BarDataFrame(symbol = test_symbol)
bar_wrapper.set_data_frame(bar_df)

# ------ answers -------
# ------------- test case for open series, upper barrier = 3, lower barrier = 3, max holding period = 4 --------
answer_open_label_series = pd.Series([1, 1, 0, 1, 1, -1, -1, 1, 1, -1, 1, 0])
answer_open_look_ahead_series = pd.Series([2, 1, 4, 3, 2, 2, 1, 1, 1, 1, 1, 0])

# ------------ test case for close series, upper barrier = 3, lower barrier = 3, max holding period = 4 --------
answer_close_label_series = pd.Series([0, 0, 0, -1, -1, -1, 1, 1, -1, 1, 0, 0])
answer_close_look_ahead_series = pd.Series([4, 4, 4, 4, 3, 1, 2, 1, 1, 1, 1, 0])

# ------------ test case for high series, upper barrier = 1, lower barrier = 4, max holding period = 4 ---------
answer_high_label_series = pd.Series([-1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 0])
answer_high_look_ahead_series = pd.Series([4, 4, 3, 2, 1, 1, 1, 2, 1, 1, 1, 0])

# ------------ test case for low series, upper barrier = 3, lower barrier = 3, max holding period = 4 ----------
answer_low_label_series = pd.Series([-1, -1, -1, -1, -1, 1, 0, 0, 0, 0, 0, 0])
answer_low_look_ahead_series = pd.Series([4, 4, 3, 2, 1, 4, 4, 4, 3, 2, 1, 0])

# ----------- test case for vwap series, lower barrier = 3, upper barrier = 3, max holding period = 3 ----------
answer_vwap_label_series = pd.Series([1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 0])
answer_vwap_look_ahead_series = pd.Series([3, 3, 2, 1, 2, 3, 1, 1, 1, 2, 1, 0])

# ---------- test case 2 for vwap series, lower barrier = 2, upper barrier = 2, max holding period = 1 ----------
answer_vwap2_label_series = pd.Series([0,0,0,-1, -1,0,1,1,-1,0,-1,0])
answer_vwap2_look_ahead_series = pd.Series([1,1,1,1,1,1,1,1,1,1,1,0])

# ------ initialize label generators --------
label_gen_open : label_gen_mod.LabelGenerator = label_gen_mod.Barrier111AbsChangeLabel(upper_barrier = 3.0, lower_barrier = 3.0, max_holding_period = 4, criteria = data.BarDataColumns.OPEN)
label_gen_close : label_gen_mod.LabelGenerator = label_gen_mod.Barrier111AbsChangeLabel(upper_barrier = 3.0, lower_barrier = 3.0, max_holding_period = 4, criteria = data.BarDataColumns.CLOSE)
label_gen_high : label_gen_mod.LabelGenerator = label_gen_mod.Barrier111AbsChangeLabel(upper_barrier = 1.0, lower_barrier = 4.0, max_holding_period = 4, criteria = data.BarDataColumns.HIGH)
label_gen_low : label_gen_mod.LabelGenerator = label_gen_mod.Barrier111AbsChangeLabel(upper_barrier = 3.0, lower_barrier = 3.0, max_holding_period = 4, criteria = data.BarDataColumns.LOW)
label_gen_vwap : label_gen_mod.LabelGenerator = label_gen_mod.Barrier111AbsChangeLabel(upper_barrier = 3.0, lower_barrier = 3.0, max_holding_period = 3, criteria = data.BarDataColumns.VWAP)
label_gen_vwap2 : label_gen_mod.LabelGenerator = label_gen_mod.Barrier111AbsChangeLabel(upper_barrier = 2.0, lower_barrier = 2.0, max_holding_period = 1, criteria = data.BarDataColumns.VWAP)

# ------ get labeling results ------
label_result_open : label_gen_mod.LabelDataFrame = label_gen_open.create_labels_for_data_bar(bar_wrapper)
label_result_close : label_gen_mod.LabelDataFrame = label_gen_close.create_labels_for_data_bar(bar_wrapper)
label_result_high : label_gen_mod.LabelDataFrame = label_gen_high.create_labels_for_data_bar(bar_wrapper)
label_result_low : label_gen_mod.LabelDataFrame = label_gen_low.create_labels_for_data_bar(bar_wrapper)
label_result_vwap : label_gen_mod.LabelDataFrame = label_gen_vwap.create_labels_for_data_bar(bar_wrapper)
label_result_vwap2 : label_gen_mod.LabelDataFrame = label_gen_vwap2.create_labels_for_data_bar(bar_wrapper)


# ------ compare results with answer -------
open_compare_df = pd.DataFrame({
    "result_label" : label_result_open.get_label_series_ref(),
    "result_look_ahead" : label_result_open.get_label_df_ref().loc[:, label_result_open.look_ahead_col_name],
    "answer_label" : answer_open_label_series,
    "answer_look_ahead" : answer_open_look_ahead_series
})

close_compare_df = pd.DataFrame({
    "result_label": label_result_close.get_label_series_ref(),
    "result_look_ahead": label_result_close.get_look_ahead_series_ref(),
    "answer_label": answer_close_label_series,
    "answer_look_ahead": answer_close_look_ahead_series
})

high_compare_df = pd.DataFrame({
    "result_label": label_result_high.get_label_series_ref(),
    "result_look_ahead": label_result_high.get_look_ahead_series_ref(),
    "answer_label": answer_high_label_series,
    "answer_look_ahead": answer_high_look_ahead_series
})

low_compare_df = pd.DataFrame({
    "result_label": label_result_low.get_label_series_ref(),
    "result_look_ahead": label_result_low.get_look_ahead_series_ref(),
    "answer_label": answer_low_label_series,
    "answer_look_ahead": answer_low_look_ahead_series
})

vwap_compare_df = pd.DataFrame({
    "result_label": label_result_vwap.get_label_series_ref(),
    "result_look_ahead": label_result_vwap.get_look_ahead_series_ref(),
    "answer_label": answer_vwap_label_series,
    "answer_look_ahead": answer_vwap_look_ahead_series
})

vwap2_compare_df = pd.DataFrame({
    "result_label": label_result_vwap2.get_label_series_ref(),
    "result_look_ahead": label_result_vwap2.get_look_ahead_series_ref(),
    "answer_label": answer_vwap2_label_series,
    "answer_look_ahead": answer_vwap2_look_ahead_series
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
print(" ----- VWAP test case 2 ------ ")
print(vwap2_compare_df)


