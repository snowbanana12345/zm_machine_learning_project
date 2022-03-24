import src.machine_learning_module.label_generators as label_gen_mod
import src.data_base_module.data_blocks as data
import pandas as pd
import numpy as np
import unittest

open_str: str = data.BarDataColumns.OPEN.value
close_str: str = data.BarDataColumns.CLOSE.value
high_str: str = data.BarDataColumns.HIGH.value
low_str: str = data.BarDataColumns.LOW.value
timestamp_str: str = data.BarDataColumns.TIMESTAMP.value
vol_str: str = data.BarDataColumns.VOLUME.value
vwap_str: str = data.BarDataColumns.VWAP.value

class TestAbsChangeLabel(unittest.TestCase):
    def setUp(self):
        self.test_symbol: str = "TEST"
        self.bar1 = {open_str: 25, close_str: 29, high_str: 31, low_str: 15, timestamp_str: 1, vol_str: 412, vwap_str: 25}
        self.bar2 = {open_str: 27, close_str: 29, high_str: 31, low_str: 22, timestamp_str: 2, vol_str: 592, vwap_str: 26}
        self.bar3 = {open_str: 28, close_str: 29, high_str: 34.1, low_str: 15, timestamp_str: 3, vol_str: 239, vwap_str: 29}
        self.bar4 = {open_str: 29.9, close_str: 29, high_str: 31.2, low_str: 22, timestamp_str: 4, vol_str: 352,vwap_str: 26}
        self.bar5 = {open_str: 31.1, close_str: 29, high_str: 35, low_str: 15, timestamp_str: 5, vol_str: 439, vwap_str: 25}
        self.bar6 = {open_str: 26.8, close_str: 29, high_str: 35, low_str: 22, timestamp_str: 6, vol_str: 239,vwap_str: 22.1}
        self.bar7 = {open_str: 28.1, close_str: 29, high_str: 32.1, low_str: 15, timestamp_str: 7, vol_str: 322,vwap_str: 25}
        self.bar8 = {open_str: 23.9, close_str: 29, high_str: 32, low_str: 22, timestamp_str: 8, vol_str: 401,vwap_str: 28.1}
        self.bar_df: pd.DataFrame = pd.DataFrame([self.bar1, self.bar2, self.bar3, self.bar4, self.bar5, self.bar6, self.bar7, self.bar8])
        self.bar_wrapper = data.BarDataFrame(symbol=self.test_symbol)
        self.bar_wrapper.set_data_frame(self.bar_df)

        # ------ initialize label generators --------
        self.label_gen_open: label_gen_mod.LabelGenerator = label_gen_mod.AbsoluteChangeLabel(2, 3.0,data.BarDataColumns.OPEN)
        self.label_gen_close: label_gen_mod.LabelGenerator = label_gen_mod.AbsoluteChangeLabel(5, 3.0,data.BarDataColumns.CLOSE)
        self.label_gen_high: label_gen_mod.LabelGenerator = label_gen_mod.AbsoluteChangeLabel(2, 3.0,data.BarDataColumns.HIGH)
        self.label_gen_low: label_gen_mod.LabelGenerator = label_gen_mod.AbsoluteChangeLabel(2, 3.0, data.BarDataColumns.LOW)
        self.label_gen_vwap: label_gen_mod.LabelGenerator = label_gen_mod.AbsoluteChangeLabel(1, 3.0, data.BarDataColumns.VWAP)

    def test_case_1(self):
        label_result_open: label_gen_mod.LabelDataFrame = self.label_gen_open.create_labels_for_data_bar(self.bar_wrapper)
        answer_open_label_series = pd.Series([1, 0, 1, -1, -1, 0, np.nan, np.nan])
        answer_open_look_ahead_series = pd.Series([2, 2, 2, 2, 2, 2, np.nan, np.nan])
        self.assertTrue(label_result_open.get_label_series_ref().equals(answer_open_label_series))
        self.assertTrue(label_result_open.get_look_ahead_series_ref().equals(answer_open_look_ahead_series))

    def test_case_2(self):
        label_result_close: label_gen_mod.LabelDataFrame = self.label_gen_close.create_labels_for_data_bar(self.bar_wrapper)
        answer_close_label_series = pd.Series([0, 0, 0, np.nan, np.nan, np.nan, np.nan, np.nan])
        answer_close_look_ahead_series = pd.Series([5, 5, 5, np.nan, np.nan, np.nan, np.nan, np.nan])
        self.assertTrue(label_result_close.get_label_series_ref().equals(answer_close_label_series))
        self.assertTrue(label_result_close.get_look_ahead_series_ref().equals(answer_close_look_ahead_series))

    def test_case_3(self):
        label_result_high: label_gen_mod.LabelDataFrame = self.label_gen_high.create_labels_for_data_bar(self.bar_wrapper)
        answer_high_label_series = pd.Series([1,0,0,1,0,-1, np.nan, np.nan])
        answer_high_look_ahead_series = pd.Series([2,2,2,2,2,2,np.nan, np.nan])
        self.assertTrue(label_result_high.get_label_series_ref().equals(answer_high_label_series))
        self.assertTrue(label_result_high.get_look_ahead_series_ref().equals(answer_high_look_ahead_series))

    def test_case_4(self):
        label_result_low: label_gen_mod.LabelDataFrame = self.label_gen_low.create_labels_for_data_bar(self.bar_wrapper)
        answer_low_label_series = pd.Series([0, 0, 0, 0, 0, 0, np.nan, np.nan])
        answer_low_look_ahead_series = pd.Series([2, 2, 2, 2, 2, 2, np.nan, np.nan])
        self.assertTrue(label_result_low.get_label_series_ref().equals(answer_low_label_series))
        self.assertTrue(label_result_low.get_look_ahead_series_ref().equals(answer_low_look_ahead_series))

    def test_case_5(self):
        label_result_vwap: label_gen_mod.LabelDataFrame = self.label_gen_vwap.create_labels_for_data_bar(self.bar_wrapper)
        answer_vwap_label_series = pd.Series([0,1,-1,0,0,0,1,np.nan])
        answer_vwap_look_ahead_series = pd.Series([1,1,1,1,1,1,1,np.nan])
        self.assertTrue(label_result_vwap.get_label_series_ref().equals(answer_vwap_label_series))
        self.assertTrue(label_result_vwap.get_look_ahead_series_ref().equals(answer_vwap_look_ahead_series))

if __name__ == '__main__':
    unittest.main()


