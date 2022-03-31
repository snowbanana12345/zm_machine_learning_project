import src.data_processing_module.sampling as sampler
import pandas as pd
from src.data_base_module.data_blocks import TickDataColumns, TickDataFrame, Date, IntraDayPeriod, BarDataFrame, \
    BarDataColumns, BarInfo, Sampling, TickInfo
import numpy as np
import unittest

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

class TimeSamplingTest(unittest.TestCase):
    def setUp(self):
        # -------- test case data --------
        self.test_date = Date(day=12, month=5, year=2021)

        # ------- bar 1 VWAP = 25.0-------
        tick1 = {TickDataColumns.TIMESTAMP_NANO.value: 0, TickDataColumns.LAST_PRICE.value: 10,
                 TickDataColumns.LAST_QUANTITY.value: 1}
        tick2 = {TickDataColumns.TIMESTAMP_NANO.value: 2 * 1E9, TickDataColumns.LAST_PRICE.value: 20,
                 TickDataColumns.LAST_QUANTITY.value: 1}
        tick3 = {TickDataColumns.TIMESTAMP_NANO.value: 7 * 1E9, TickDataColumns.LAST_PRICE.value: 30,
                 TickDataColumns.LAST_QUANTITY.value: 1}
        tick4 = {TickDataColumns.TIMESTAMP_NANO.value: 14 * 1E9, TickDataColumns.LAST_PRICE.value: 40,
                 TickDataColumns.LAST_QUANTITY.value: 1}

        # ------ bar 2 (empty) all zeros -------
        # ------ bar 3 VWAP = 3.9375 -------
        tick5 = {TickDataColumns.TIMESTAMP_NANO.value: 35 * 1E9, TickDataColumns.LAST_PRICE.value: 5,
                 TickDataColumns.LAST_QUANTITY.value: 5}
        tick6 = {TickDataColumns.TIMESTAMP_NANO.value: 35 * 1E9 + 1, TickDataColumns.LAST_PRICE.value: 4,
                 TickDataColumns.LAST_QUANTITY.value: 7}
        tick7 = {TickDataColumns.TIMESTAMP_NANO.value: 35 * 1E9 + 2, TickDataColumns.LAST_PRICE.value: 3,
                 TickDataColumns.LAST_QUANTITY.value: 2}
        tick8 = {TickDataColumns.TIMESTAMP_NANO.value: 35 * 1E9 + 3, TickDataColumns.LAST_PRICE.value: 2,
                 TickDataColumns.LAST_QUANTITY.value: 2}

        # ----- bar 4 VWAP = 19 ------
        tick9 = {TickDataColumns.TIMESTAMP_NANO.value: 45 * 1E9, TickDataColumns.LAST_PRICE.value: 19,
                 TickDataColumns.LAST_QUANTITY.value: 15}

        # ----- bar 5 VWAP = 81 -------
        tick10 = {TickDataColumns.TIMESTAMP_NANO.value: 75 * 1E9 - 1, TickDataColumns.LAST_PRICE.value: 81,
                  TickDataColumns.LAST_QUANTITY.value: 25}

        # ----- bar 6 VWAP = 109 -------
        tick11 = {TickDataColumns.TIMESTAMP_NANO.value: 75 * 1E9 + 1, TickDataColumns.LAST_PRICE.value: 109,
                  TickDataColumns.LAST_QUANTITY.value: 79}

        # ----- bar 7 VWAP = -------
        tick12 = {TickDataColumns.TIMESTAMP_NANO.value: 90 * 1E9 + 91286, TickDataColumns.LAST_PRICE.value: 205,
                  TickDataColumns.LAST_QUANTITY.value: 40}
        tick13 = {TickDataColumns.TIMESTAMP_NANO.value: 90 * 1E9 + 1348722, TickDataColumns.LAST_PRICE.value: 210,
                  TickDataColumns.LAST_QUANTITY.value: 30}
        tick14 = {TickDataColumns.TIMESTAMP_NANO.value: 90 * 1E9 + 14257883, TickDataColumns.LAST_PRICE.value: 215,
                  TickDataColumns.LAST_QUANTITY.value: 30}

        # ------------ set up data frames -----------
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
        tick_info = TickInfo(date = self.test_date, symbol = "TEST", intra_day_period = IntraDayPeriod.WHOLE_DAY)
        self.tick_wrapper = TickDataFrame(tick_df=self.raw_tick_df, tick_info = tick_info)

    def test_case(self):
        # --------- expected answer --------
        bar1 = {BarDataColumns.TIMESTAMP.value: 0, BarDataColumns.OPEN.value: 10.0, BarDataColumns.CLOSE.value: 40.0,
                BarDataColumns.HIGH.value: 40.0, BarDataColumns.LOW.value: 10.0, BarDataColumns.VWAP.value: 25.5,
                BarDataColumns.VOLUME.value: 4}
        bar2 = {BarDataColumns.TIMESTAMP.value: 0, BarDataColumns.OPEN.value: 0.0, BarDataColumns.CLOSE.value: 0.0,
                BarDataColumns.HIGH.value: 0.0, BarDataColumns.LOW.value: 0.0, BarDataColumns.VWAP.value: 0.0,
                BarDataColumns.VOLUME.value: 0}
        bar3 = {BarDataColumns.TIMESTAMP.value: 35 * 1E9, BarDataColumns.OPEN.value: 5.0, BarDataColumns.CLOSE.value: 2.0,
                BarDataColumns.HIGH.value: 5.0, BarDataColumns.LOW.value: 2.0, BarDataColumns.VWAP.value: 3.9375,
                BarDataColumns.VOLUME.value: 16}
        bar4 = {BarDataColumns.TIMESTAMP.value: 45 * 1E9, BarDataColumns.OPEN.value: 19.0, BarDataColumns.CLOSE.value: 19.0,
                BarDataColumns.HIGH.value: 19.0, BarDataColumns.LOW.value: 19.0, BarDataColumns.VWAP.value: 19.0,
                BarDataColumns.VOLUME.value: 15}
        bar5 = {BarDataColumns.TIMESTAMP.value: 75 * 1E9 - 1, BarDataColumns.OPEN.value: 81.0, BarDataColumns.CLOSE.value: 81.0,
                BarDataColumns.HIGH.value: 81.0, BarDataColumns.LOW.value: 81.0, BarDataColumns.VWAP.value: 81.0,
                BarDataColumns.VOLUME.value: 25}
        bar6 = {BarDataColumns.TIMESTAMP.value: 75 * 1E9 + 1, BarDataColumns.OPEN.value: 109.0,
                BarDataColumns.CLOSE.value: 109.0,
                BarDataColumns.HIGH.value: 109.0, BarDataColumns.LOW.value: 109.0, BarDataColumns.VWAP.value: 109.0,
                BarDataColumns.VOLUME.value: 79}
        bar7 = {BarDataColumns.TIMESTAMP.value: 90 * 1E9 + 91286, BarDataColumns.OPEN.value: 205.0,
                BarDataColumns.CLOSE.value: 215.0,
                BarDataColumns.HIGH.value: 215.0, BarDataColumns.LOW.value: 205.0, BarDataColumns.VWAP.value: 209.5,
                BarDataColumns.VOLUME.value: 100}

        expected_bar_df = pd.DataFrame([bar1, bar2, bar3, bar4, bar5, bar6, bar7])
        expected_bar_df["hachoooo_hachoooo"] = 1
        expected_bar_info = BarInfo(symbol = "TEST", date = self.test_date, intra_day_period = IntraDayPeriod.WHOLE_DAY,
                                    sampling_level = 15, sampling_type = Sampling.TIME)
        expected_bar_wrapper = BarDataFrame(bar_data = expected_bar_df, bar_info = expected_bar_info)

        # ------ sampling -----
        result_bar_wrapper = sampler.time_sampling(tick_wrapper=self.tick_wrapper, sampling_seconds=15)

        self.assertTrue(all(expected_bar_wrapper.bar_data == result_bar_wrapper.bar_data))


