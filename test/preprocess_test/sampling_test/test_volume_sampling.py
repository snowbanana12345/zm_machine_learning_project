import src.data_processing_module.sampling as sampler
import pandas as pd
from src.data_base_module.data_blocks import TickDataColumns, TickDataFrame, Date, IntraDayPeriod, VolumeBarDataFrame, BarDataColumns
import numpy as np
import unittest

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

class VolumeSamplingTest(unittest.TestCase):
    def setUp(self):
        # -------- test case , sample into bars of volume 10 --------
        self.test_date = Date(day=17, month=7, year=2012)
        # -------- bar 1 VWAP = 29.5 -------
        tick1 = {TickDataColumns.TIMESTAMP_NANO.value: 101, TickDataColumns.LAST_PRICE.value: 20,
                 TickDataColumns.LAST_QUANTITY.value: 2}
        tick2 = {TickDataColumns.TIMESTAMP_NANO.value: 256, TickDataColumns.LAST_PRICE.value: 30,
                 TickDataColumns.LAST_QUANTITY.value: 3}
        tick3 = {TickDataColumns.TIMESTAMP_NANO.value: 312, TickDataColumns.LAST_PRICE.value: 25,
                 TickDataColumns.LAST_QUANTITY.value: 1}
        tick4 = {TickDataColumns.TIMESTAMP_NANO.value: 498, TickDataColumns.LAST_PRICE.value: 35,
                 TickDataColumns.LAST_QUANTITY.value: 4}

        # ------ bar 2 (over flow by 2) VWAP = 29.5 ------
        tick5 = {TickDataColumns.TIMESTAMP_NANO.value: 501, TickDataColumns.LAST_PRICE.value: 20,
                 TickDataColumns.LAST_QUANTITY.value: 2}
        tick6 = {TickDataColumns.TIMESTAMP_NANO.value: 612, TickDataColumns.LAST_PRICE.value: 30,
                 TickDataColumns.LAST_QUANTITY.value: 3}
        tick7 = {TickDataColumns.TIMESTAMP_NANO.value: 757, TickDataColumns.LAST_PRICE.value: 25,
                 TickDataColumns.LAST_QUANTITY.value: 1}
        tick8 = {TickDataColumns.TIMESTAMP_NANO.value: 897, TickDataColumns.LAST_PRICE.value: 35,
                 TickDataColumns.LAST_QUANTITY.value: 6}

        # ----- bar 3 (together with overflow from previous bar) VWAP = 29.5 -------
        tick9 = {TickDataColumns.TIMESTAMP_NANO.value: 923, TickDataColumns.LAST_PRICE.value: 40,
                 TickDataColumns.LAST_QUANTITY.value: 4}
        tick10 = {TickDataColumns.TIMESTAMP_NANO.value: 1045, TickDataColumns.LAST_PRICE.value: 15,
                  TickDataColumns.LAST_QUANTITY.value: 3}
        tick11 = {TickDataColumns.TIMESTAMP_NANO.value: 1123, TickDataColumns.LAST_PRICE.value: 20,
                  TickDataColumns.LAST_QUANTITY.value: 1}

        # ----- bar 4, 5, 6 (large tick, enough to fill 3 bars and over flow by 2) VWAP = 35 ------
        tick12 = {TickDataColumns.TIMESTAMP_NANO.value: 1211, TickDataColumns.LAST_PRICE.value: 35,
                  TickDataColumns.LAST_QUANTITY.value: 32}

        # ----- bar 7 (same as bar 3, test if multi-bar overflow is working, start with 2 over flow, over flow by 1) VWAP = 29.5-----
        tick13 = {TickDataColumns.TIMESTAMP_NANO.value: 1332, TickDataColumns.LAST_PRICE.value: 40,
                  TickDataColumns.LAST_QUANTITY.value: 4}
        tick14 = {TickDataColumns.TIMESTAMP_NANO.value: 1445, TickDataColumns.LAST_PRICE.value: 15,
                  TickDataColumns.LAST_QUANTITY.value: 3}
        tick15 = {TickDataColumns.TIMESTAMP_NANO.value: 1523, TickDataColumns.LAST_PRICE.value: 20,
                  TickDataColumns.LAST_QUANTITY.value: 2}

        # ----- bar 8,9 (single tick, test if bar overflow working starting from over flow of 1)  -------
        tick16 = {TickDataColumns.TIMESTAMP_NANO.value: 1675, TickDataColumns.LAST_PRICE.value: 55,
                  TickDataColumns.LAST_QUANTITY.value: 19}

        # ----- residue ticks, not enough to form a bar ------
        tick17 = {TickDataColumns.TIMESTAMP_NANO.value: 1895, TickDataColumns.LAST_PRICE.value: 100,
                  TickDataColumns.LAST_QUANTITY.value: 1}
        tick18 = {TickDataColumns.TIMESTAMP_NANO.value: 1927, TickDataColumns.LAST_PRICE.value: 100,
                  TickDataColumns.LAST_QUANTITY.value: 1}
        tick19 = {TickDataColumns.TIMESTAMP_NANO.value: 1998, TickDataColumns.LAST_PRICE.value: 100,
                  TickDataColumns.LAST_QUANTITY.value: 1}

        # ------------ set up data frames -----------
        # ----- tick data frame -------
        self.raw_tick_df = pd.DataFrame(
            [tick1, tick2, tick3, tick4, tick5, tick6, tick7, tick8, tick9, tick10, tick11, tick12, tick13, tick14,
             tick15, tick16, tick17, tick18, tick19])

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
            self.raw_tick_df[column] = np.zeros(19)
        # ------ create gibberish column to test redundant column removal of TickDataFrame Wrapper class -----
        self.raw_tick_df["alalabaarababababa_bubrabubra"] = 0

        # ------ create tick data frame wrapper object -------
        self.tick_wrapper = TickDataFrame(tick_df=self.raw_tick_df, date=self.test_date, intra_day_period=IntraDayPeriod.WHOLE_DAY, symbol="TEST")

    def test_case(self):
        bar1 = {BarDataColumns.TIMESTAMP.value : 101, BarDataColumns.OPEN.value : 20.0, BarDataColumns.CLOSE.value : 35.0,
                BarDataColumns.HIGH.value : 35.0, BarDataColumns.LOW.value : 20.0, BarDataColumns.VWAP.value : 29.5}
        bar2 = {BarDataColumns.TIMESTAMP.value : 501, BarDataColumns.OPEN.value : 20.0, BarDataColumns.CLOSE.value : 35.0,
                BarDataColumns.HIGH.value : 35.0, BarDataColumns.LOW.value : 20.0, BarDataColumns.VWAP.value : 29.5}
        bar3 = {BarDataColumns.TIMESTAMP.value : 897, BarDataColumns.OPEN.value : 35.0, BarDataColumns.CLOSE.value : 20.0,
                BarDataColumns.HIGH.value : 40.0, BarDataColumns.LOW.value : 15.0, BarDataColumns.VWAP.value : 29.5}
        bar4 = {BarDataColumns.TIMESTAMP.value : 1211, BarDataColumns.OPEN.value : 35.0, BarDataColumns.CLOSE.value : 35.0,
                BarDataColumns.HIGH.value : 35.0, BarDataColumns.LOW.value : 35.0, BarDataColumns.VWAP.value : 35.0}
        bar5 = {BarDataColumns.TIMESTAMP.value : 1211, BarDataColumns.OPEN.value : 35.0, BarDataColumns.CLOSE.value : 35.0,
                BarDataColumns.HIGH.value : 35.0, BarDataColumns.LOW.value : 35.0, BarDataColumns.VWAP.value : 35.0}
        bar6 = {BarDataColumns.TIMESTAMP.value : 1211, BarDataColumns.OPEN.value : 35.0, BarDataColumns.CLOSE.value : 35.0,
                BarDataColumns.HIGH.value : 35.0, BarDataColumns.LOW.value : 35.0, BarDataColumns.VWAP.value : 35.0}
        bar7 = {BarDataColumns.TIMESTAMP.value : 1211, BarDataColumns.OPEN.value : 35.0, BarDataColumns.CLOSE.value : 20.0,
                BarDataColumns.HIGH.value : 40.0, BarDataColumns.LOW.value : 15.0, BarDataColumns.VWAP.value : 29.5}
        bar8 = {BarDataColumns.TIMESTAMP.value : 1523, BarDataColumns.OPEN.value : 20.0, BarDataColumns.CLOSE.value : 55.0,
                BarDataColumns.HIGH.value : 55.0, BarDataColumns.LOW.value : 20.0, BarDataColumns.VWAP.value : 51.5}
        bar9 = {BarDataColumns.TIMESTAMP.value : 1675, BarDataColumns.OPEN.value : 55.0, BarDataColumns.CLOSE.value : 55.0,
                BarDataColumns.HIGH.value : 55.0, BarDataColumns.LOW.value : 55.0, BarDataColumns.VWAP.value : 55.0}
        expected_bar_df = pd.DataFrame([bar1, bar2, bar3, bar4, bar5, bar6, bar7, bar8, bar9])
        expected_bar_df[BarDataColumns.VOLUME.value] = 10
        # ------ create gibberish column to test redundant column removal of VolumeDataFrame Wrapper class ------
        expected_bar_df["hachoooo_hachoooo"] = 1
        # ------ create answer bar df wrapper -------
        expected_bar_wrapper = VolumeBarDataFrame(bar_df=expected_bar_df, sampling_volume=10, date=self.test_date,
                                                   intra_day_period=IntraDayPeriod.WHOLE_DAY, symbol="TEST",
                                                   deep_copy=False)
        # ------ sampling ------
        result_bar_wrapper = sampler.volume_sampling(self.tick_wrapper, sampling_volume = 10)

        self.assertTrue(all(expected_bar_wrapper.get_bar_data_reference() == result_bar_wrapper.get_bar_data_reference()))

