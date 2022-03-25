import src.data_processing_module.sampling as sampler
import pandas as pd
from src.data_base_module.data_blocks import TickDataColumns, TickDataFrame, Date, IntraDayPeriod, TickBarDataFrame, BarDataColumns
import numpy as np
import unittest

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

class TickSamplingTest(unittest.TestCase):
    def setUp(self):
        # -------- test case , sample into bars of 4 ticks --------
        self.test_date = Date(day=17, month=7, year=2012)

        # -------- bar 1 VWAP = 29.5------
        tick1 = {TickDataColumns.TIMESTAMP_NANO.value: 101, TickDataColumns.LAST_PRICE.value: 20,
                 TickDataColumns.LAST_QUANTITY.value: 2}
        tick2 = {TickDataColumns.TIMESTAMP_NANO.value: 256, TickDataColumns.LAST_PRICE.value: 30,
                 TickDataColumns.LAST_QUANTITY.value: 3}
        tick3 = {TickDataColumns.TIMESTAMP_NANO.value: 312, TickDataColumns.LAST_PRICE.value: 25,
                 TickDataColumns.LAST_QUANTITY.value: 1}
        tick4 = {TickDataColumns.TIMESTAMP_NANO.value: 498, TickDataColumns.LAST_PRICE.value: 35,
                 TickDataColumns.LAST_QUANTITY.value: 4}

        # ------ bar 2 (empty) VWAP = 0 -------
        tick5 = {TickDataColumns.TIMESTAMP_NANO.value: 685, TickDataColumns.LAST_PRICE.value: 0,
                 TickDataColumns.LAST_QUANTITY.value: 0}
        tick6 = {TickDataColumns.TIMESTAMP_NANO.value: 795, TickDataColumns.LAST_PRICE.value: 0,
                 TickDataColumns.LAST_QUANTITY.value: 0}
        tick7 = {TickDataColumns.TIMESTAMP_NANO.value: 982, TickDataColumns.LAST_PRICE.value: 0,
                 TickDataColumns.LAST_QUANTITY.value: 0}
        tick8 = {TickDataColumns.TIMESTAMP_NANO.value: 1109, TickDataColumns.LAST_PRICE.value: 0,
                 TickDataColumns.LAST_QUANTITY.value: 0}

        # ------ bar 3 (mixture of empty ticks) VWAP = 45 -----
        tick9 = {TickDataColumns.TIMESTAMP_NANO.value: 1203, TickDataColumns.LAST_PRICE.value: 0,
                 TickDataColumns.LAST_QUANTITY.value: 0}
        tick10 = {TickDataColumns.TIMESTAMP_NANO.value: 1376, TickDataColumns.LAST_PRICE.value: 35,
                  TickDataColumns.LAST_QUANTITY.value: 2}
        tick11 = {TickDataColumns.TIMESTAMP_NANO.value: 1416, TickDataColumns.LAST_PRICE.value: 50,
                  TickDataColumns.LAST_QUANTITY.value: 4}
        tick12 = {TickDataColumns.TIMESTAMP_NANO.value: 1531, TickDataColumns.LAST_PRICE.value: 0,
                  TickDataColumns.LAST_QUANTITY.value: 0}

        # ----- bar 4 (2 ticks) left over ticks VWAP = 45 -------
        tick13 = {TickDataColumns.TIMESTAMP_NANO.value: 1785, TickDataColumns.LAST_PRICE.value: 35,
                  TickDataColumns.LAST_QUANTITY.value: 2}
        tick14 = {TickDataColumns.TIMESTAMP_NANO.value: 1982, TickDataColumns.LAST_PRICE.value: 50,
                  TickDataColumns.LAST_QUANTITY.value: 4}

        # ----- tick data frame -------
        self.raw_tick_df = pd.DataFrame(
            [tick1, tick2, tick3, tick4, tick5, tick6, tick7, tick8, tick9, tick10, tick11, tick12, tick13, tick14])
        # ----- fill in the limit order book columns -------
        for column in [TickDataColumns.ASK1P.value,
                       TickDataColumns.ASK2P.value,
                       TickDataColumns.ASK3P.value,
                       TickDataColumns.ASK4P.value,
                       TickDataColumns.ASK5P.value,
                       TickDataColumns.ASK1Q.value,
                       TickDataColumns.ASK2Q.value,
                       TickDataColumns.ASK3Q.value,
                       TickDataColumns.ASK4Q.value,
                       TickDataColumns.ASK5Q.value,
                       TickDataColumns.BID1P.value,
                       TickDataColumns.BID2P.value,
                       TickDataColumns.BID3P.value,
                       TickDataColumns.BID4P.value,
                       TickDataColumns.BID5P.value,
                       TickDataColumns.BID1Q.value,
                       TickDataColumns.BID2Q.value,
                       TickDataColumns.BID3Q.value,
                       TickDataColumns.BID4Q.value,
                       TickDataColumns.BID5Q.value]:
            self.raw_tick_df[column] = np.zeros(14)

        # ------ create gibberish column to test redundant column removal of TickDataFrame Wrapper class -----
        self.raw_tick_df["alalabaarababababa_bubrabubra"] = 0
        # ------ create tick data frame wrapper object -------
        self.tick_df_wrapper = TickDataFrame(tick_df=self.raw_tick_df, date=self.test_date, intra_day_period=IntraDayPeriod.WHOLE_DAY,
                                        symbol="TEST")

    def test_case(self):
        # ------ expected answer ---------
        bar1 = {BarDataColumns.TIMESTAMP.value: 101, BarDataColumns.OPEN.value: 20.0, BarDataColumns.CLOSE.value: 35.0,
                BarDataColumns.HIGH.value: 35.0, BarDataColumns.LOW.value: 20.0, BarDataColumns.VWAP.value: 29.5,
                BarDataColumns.VOLUME.value: 10}
        bar2 = {BarDataColumns.TIMESTAMP.value: 685, BarDataColumns.OPEN.value: 0, BarDataColumns.CLOSE.value: 0,
                BarDataColumns.HIGH.value: 0, BarDataColumns.LOW.value: 0, BarDataColumns.VWAP.value: 0,
                BarDataColumns.VOLUME.value: 0}
        bar3 = {BarDataColumns.TIMESTAMP.value: 1203, BarDataColumns.OPEN.value: 35, BarDataColumns.CLOSE.value: 50,
                BarDataColumns.HIGH.value: 50, BarDataColumns.LOW.value: 35, BarDataColumns.VWAP.value: 45,
                BarDataColumns.VOLUME.value: 6}
        bar4 = {BarDataColumns.TIMESTAMP.value: 1785, BarDataColumns.OPEN.value: 35, BarDataColumns.CLOSE.value: 50,
                BarDataColumns.HIGH.value: 50, BarDataColumns.LOW.value: 35, BarDataColumns.VWAP.value: 45,
                BarDataColumns.VOLUME.value: 6}
        expected_bar_df = pd.DataFrame([bar1, bar2, bar3, bar4])
        # ------ create gibberish column to test redundant column removal of VolumeDataFrame Wrapper class ------
        expected_bar_df["hachoooo_hachoooo"] = 1
        expected_bar_wrapper = TickBarDataFrame(bar_df = expected_bar_df, sampling_ticks=4, date=self.test_date,
                                                 intra_day_period=IntraDayPeriod.WHOLE_DAY, symbol = "TEST", deep_copy = False)
        result_bar_wrapper = sampler.tick_sampling(self.tick_df_wrapper, sampling_ticks=4)
        try :
            self.assertEqual(expected_bar_wrapper, result_bar_wrapper)
        except AssertionError:
            expected_df = expected_bar_wrapper.get_bar_data_reference()
            result_df = result_bar_wrapper.get_bar_data_reference()
            expected_df.columns = [header + "_expected" for header in expected_df.columns]
            result_df.columns = [header + "_result" for header in result_df.columns]
            combine_dict = {}
            for expected_name, result_name, in zip(expected_df.columns, result_df.columns):
                combine_dict[result_name] = result_df[result_name]
                combine_dict[expected_name] = expected_df[expected_name]
            combined_df = pd.DataFrame(combine_dict)
            print(combined_df)

