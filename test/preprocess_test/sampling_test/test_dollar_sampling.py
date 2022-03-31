import src.data_processing_module.sampling as sampler
import pandas as pd
import src.data_base_module.data_blocks as dat_blocks
import numpy as np
import unittest

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

class DollarSamplingTest(unittest.TestCase):
    def setUp(self):
        # -------- test case , sample into bars of dollar 100 --------
        self.test_date = dat_blocks.Date(day=17, month=7, year=2012)

        # -------- bar 1 volume = 10 (no over flow) -------
        tick1 = {dat_blocks.TickDataColumns.TIMESTAMP_NANO.value: 175, dat_blocks.TickDataColumns.LAST_PRICE.value: 10,
                 dat_blocks.TickDataColumns.LAST_QUANTITY.value: 2}
        tick2 = {dat_blocks.TickDataColumns.TIMESTAMP_NANO.value: 323, dat_blocks.TickDataColumns.LAST_PRICE.value: 15,
                 dat_blocks.TickDataColumns.LAST_QUANTITY.value: 1}
        tick3 = {dat_blocks.TickDataColumns.TIMESTAMP_NANO.value: 477, dat_blocks.TickDataColumns.LAST_PRICE.value: 20,
                 dat_blocks.TickDataColumns.LAST_QUANTITY.value: 2}
        tick4 = {dat_blocks.TickDataColumns.TIMESTAMP_NANO.value: 596, dat_blocks.TickDataColumns.LAST_PRICE.value: 5,
                 dat_blocks.TickDataColumns.LAST_QUANTITY.value: 5}

        # ------- bar 2 volume = 6.6 (over flow by 24) ------
        tick5 = {dat_blocks.TickDataColumns.TIMESTAMP_NANO.value: 700, dat_blocks.TickDataColumns.LAST_PRICE.value: 13,
                 dat_blocks.TickDataColumns.LAST_QUANTITY.value: 3}
        tick6 = {dat_blocks.TickDataColumns.TIMESTAMP_NANO.value: 775, dat_blocks.TickDataColumns.LAST_PRICE.value: 11,
                 dat_blocks.TickDataColumns.LAST_QUANTITY.value: 1}
        tick7 = {dat_blocks.TickDataColumns.TIMESTAMP_NANO.value: 857, dat_blocks.TickDataColumns.LAST_PRICE.value: 22,
                 dat_blocks.TickDataColumns.LAST_QUANTITY.value: 2}
        tick8 = {dat_blocks.TickDataColumns.TIMESTAMP_NANO.value: 1010, dat_blocks.TickDataColumns.LAST_PRICE.value: 10,
                 dat_blocks.TickDataColumns.LAST_QUANTITY.value: 3}

        # ------ bar 3 (carry over 24, 2.4 volume from previous bar ) volume = 5.4 -------
        tick9 = {dat_blocks.TickDataColumns.TIMESTAMP_NANO.value: 1331, dat_blocks.TickDataColumns.LAST_PRICE.value: 29,
                 dat_blocks.TickDataColumns.LAST_QUANTITY.value: 2}
        tick10 = {dat_blocks.TickDataColumns.TIMESTAMP_NANO.value: 1511, dat_blocks.TickDataColumns.LAST_PRICE.value: 18,
                  dat_blocks.TickDataColumns.LAST_QUANTITY.value: 1}

        # ------ bar 4 (mixture of empty ticks) ----------
        tick11 = {dat_blocks.TickDataColumns.TIMESTAMP_NANO.value: 1545, dat_blocks.TickDataColumns.LAST_PRICE.value: 10,
                  dat_blocks.TickDataColumns.LAST_QUANTITY.value: 2}
        tick12 = {dat_blocks.TickDataColumns.TIMESTAMP_NANO.value: 1687, dat_blocks.TickDataColumns.LAST_PRICE.value: 0,
                  dat_blocks.TickDataColumns.LAST_QUANTITY.value: 0}
        tick13 = {dat_blocks.TickDataColumns.TIMESTAMP_NANO.value: 1778, dat_blocks.TickDataColumns.LAST_PRICE.value: 15,
                  dat_blocks.TickDataColumns.LAST_QUANTITY.value: 1}
        tick14 = {dat_blocks.TickDataColumns.TIMESTAMP_NANO.value: 1928, dat_blocks.TickDataColumns.LAST_PRICE.value: 0,
                  dat_blocks.TickDataColumns.LAST_QUANTITY.value: 0}
        tick15 = {dat_blocks.TickDataColumns.TIMESTAMP_NANO.value: 1999, dat_blocks.TickDataColumns.LAST_PRICE.value: 20,
                  dat_blocks.TickDataColumns.LAST_QUANTITY.value: 2}
        tick16 = {dat_blocks.TickDataColumns.TIMESTAMP_NANO.value: 2012, dat_blocks.TickDataColumns.LAST_PRICE.value: 0,
                  dat_blocks.TickDataColumns.LAST_QUANTITY.value: 0}
        tick17 = {dat_blocks.TickDataColumns.TIMESTAMP_NANO.value: 2304, dat_blocks.TickDataColumns.LAST_PRICE.value: 5,
                  dat_blocks.TickDataColumns.LAST_QUANTITY.value: 5}

        # ------ left over ticks --------
        tick18 = {dat_blocks.TickDataColumns.TIMESTAMP_NANO.value: 2399, dat_blocks.TickDataColumns.LAST_PRICE.value: 5,
                  dat_blocks.TickDataColumns.LAST_QUANTITY.value: 5}
        tick19 = {dat_blocks.TickDataColumns.TIMESTAMP_NANO.value: 2454, dat_blocks.TickDataColumns.LAST_PRICE.value: 5,
                  dat_blocks.TickDataColumns.LAST_QUANTITY.value: 5}

        # ----- tick data frame -------
        self.raw_tick_df = pd.DataFrame(
            [tick1, tick2, tick3, tick4, tick5, tick6, tick7, tick8, tick9, tick10, tick11, tick12, tick13, tick14,
             tick15, tick16, tick17, tick18, tick19])

        # ----- fill in the limit order book columns -------
        for column in [dat_blocks.TickDataColumns.ASK1P.value,
                       dat_blocks.TickDataColumns.ASK2P.value,
                       dat_blocks.TickDataColumns.ASK3P.value,
                       dat_blocks.TickDataColumns.ASK4P.value,
                       dat_blocks.TickDataColumns.ASK5P.value,
                       dat_blocks.TickDataColumns.ASK1Q.value,
                       dat_blocks.TickDataColumns.ASK2Q.value,
                       dat_blocks.TickDataColumns.ASK3Q.value,
                       dat_blocks.TickDataColumns.ASK4Q.value,
                       dat_blocks.TickDataColumns.ASK5Q.value,
                       dat_blocks.TickDataColumns.BID1P.value,
                       dat_blocks.TickDataColumns.BID2P.value,
                       dat_blocks.TickDataColumns.BID3P.value,
                       dat_blocks.TickDataColumns.BID4P.value,
                       dat_blocks.TickDataColumns.BID5P.value,
                       dat_blocks.TickDataColumns.BID1Q.value,
                       dat_blocks.TickDataColumns.BID2Q.value,
                       dat_blocks.TickDataColumns.BID3Q.value,
                       dat_blocks.TickDataColumns.BID4Q.value,
                       dat_blocks.TickDataColumns.BID5Q.value]:
            self.raw_tick_df[column] = np.zeros(19)
        # ------ create gibberish column to test redundant column removal of TickDataFrame Wrapper class -----
        self.raw_tick_df["alalabaarababababa_bubrabubra"] = 0
        # ------ create tick data frame wrapper object -------
        self.tick_info = dat_blocks.TickInfo(symbol = "TEST", date = self.test_date, intra_day_period = dat_blocks.IntraDayPeriod.WHOLE_DAY)
        self.tick_df_wrapper = dat_blocks.TickDataFrame(tick_df=self.raw_tick_df, tick_info = self.tick_info)


    def test_case(self):
        # --------- correct answer --------
        bar1 = {dat_blocks.BarDataColumns.TIMESTAMP.value: 175, dat_blocks.BarDataColumns.OPEN.value: 10, dat_blocks.BarDataColumns.CLOSE.value: 5,
                dat_blocks.BarDataColumns.HIGH.value: 20, dat_blocks.BarDataColumns.LOW.value: 5, dat_blocks.BarDataColumns.VWAP.value: 100 / 10,
                dat_blocks.BarDataColumns.VOLUME.value: 10}
        bar2 = {dat_blocks.BarDataColumns.TIMESTAMP.value: 700, dat_blocks.BarDataColumns.OPEN.value: 13, dat_blocks.BarDataColumns.CLOSE.value: 10,
                dat_blocks.BarDataColumns.HIGH.value: 22, dat_blocks.BarDataColumns.LOW.value: 10, dat_blocks.BarDataColumns.VWAP.value: 100 / 6.6,
                dat_blocks.BarDataColumns.VOLUME.value: 6.6}
        bar3 = {dat_blocks.BarDataColumns.TIMESTAMP.value: 1010, dat_blocks.BarDataColumns.OPEN.value: 10, dat_blocks.BarDataColumns.CLOSE.value: 18,
                dat_blocks.BarDataColumns.HIGH.value: 29, dat_blocks.BarDataColumns.LOW.value: 10, dat_blocks.BarDataColumns.VWAP.value: 100 / 5.4,
                dat_blocks.BarDataColumns.VOLUME.value: 5.4}
        bar4 = {dat_blocks.BarDataColumns.TIMESTAMP.value: 1545, dat_blocks.BarDataColumns.OPEN.value: 10, dat_blocks.BarDataColumns.CLOSE.value: 5,
                dat_blocks.BarDataColumns.HIGH.value: 20, dat_blocks.BarDataColumns.LOW.value: 5, dat_blocks.BarDataColumns.VWAP.value: 100 / 10,
                dat_blocks.BarDataColumns.VOLUME.value: 10}
        expected_bar_df = pd.DataFrame([bar1, bar2, bar3, bar4])
        # ------ create gibberish column to test redundant column removal of VolumeDataFrame Wrapper class ------
        expected_bar_df["hachoooo_hachoooo"] = 1
        sampling_level = 100
        expected_bar_info = dat_blocks.BarInfo(symbol = "TEST", date = self.test_date, intra_day_period = dat_blocks.IntraDayPeriod.WHOLE_DAY
                                               , sampling_type = dat_blocks.Sampling.DOLLAR, sampling_level = sampling_level)
        expected_bar_wrapper = dat_blocks.BarDataFrame(bar_data = expected_bar_df, bar_info = expected_bar_info)
        result_bar_wrapper = sampler.dollar_sampling(self.tick_df_wrapper, sampling_dollar=sampling_level)

        self.assertTrue(all(expected_bar_wrapper.bar_data == result_bar_wrapper.bar_data))
