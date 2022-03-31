import unittest
import src.data_base_module.data_blocks as dat
import pandas as pd

class DataBlockEqualityTest(unittest.TestCase):
    def setUp(self):
        self.date1 = dat.Date(day=12, month=5, year=2019)
        self.date2 = dat.Date(day=12, month=5, year=2019)
        self.date3 = dat.Date(day=12, month=6, year=2019)
        self.date4 = dat.Date(day=11, month=6, year=2019)

        self.MORNING = dat.IntraDayPeriod.MORNING
        self.AFTERNOON = dat.IntraDayPeriod.AFTERNOON
        self.symbol = "TEST"

        self.df = pd.DataFrame({
            dat.BarDataColumns.TIMESTAMP.value : [10, 20, 30, 40],
            dat.BarDataColumns.OPEN.value :  [12, 15, 16, 17],
            dat.BarDataColumns.CLOSE.value : [15, 16, 19, 17],
            dat.BarDataColumns.HIGH.value :  [22, 25, 26, 29],
            dat.BarDataColumns.LOW.value :   [5,6,7,8],
            dat.BarDataColumns.VOLUME.value : [10, 30, 25, 19],
            dat.BarDataColumns.VWAP.value : [16, 17, 19, 15]
        })

        self.df2 = pd.DataFrame({
            dat.BarDataColumns.TIMESTAMP.value : [10, 20, 30, 40],
            dat.BarDataColumns.OPEN.value : [15, 15, 16, 17],
            dat.BarDataColumns.CLOSE.value : [15, 16, 19, 17],
            dat.BarDataColumns.HIGH.value : [22, 25, 26, 29],
            dat.BarDataColumns.LOW.value : [5, 6, 7, 9],
            dat.BarDataColumns.VOLUME.value : [10, 30, 25, 19],
            dat.BarDataColumns.VWAP.value : [16, 17, 13, 15]
        })

        self.tick_df = pd.DataFrame({
            dat.TickDataColumns.TIMESTAMP_NANO.value : [12, 15, 17, 19],
            dat.TickDataColumns.LAST_PRICE.value : [15, 17, 19, 20],
            dat.TickDataColumns.LAST_QUANTITY.value : [10, 12, 13, 15],
            dat.TickDataColumns.ASK1P.value : [10, 11, 12, 13],
            dat.TickDataColumns.ASK2P.value: [11, 12, 13, 14],
            dat.TickDataColumns.ASK3P.value: [12, 13, 14, 15],
            dat.TickDataColumns.ASK4P.value: [13, 14, 15, 16],
            dat.TickDataColumns.ASK5P.value: [14, 15, 16, 17],
            dat.TickDataColumns.ASK1Q.value: [1, 1, 1, 1],
            dat.TickDataColumns.ASK2Q.value: [2, 2, 2, 2],
            dat.TickDataColumns.ASK3Q.value: [3, 3, 3, 3],
            dat.TickDataColumns.ASK4Q.value: [2, 3, 2, 3],
            dat.TickDataColumns.ASK5Q.value: [4, 4, 2, 2],
            dat.TickDataColumns.BID1P.value: [9, 9, 9, 9],
            dat.TickDataColumns.BID2P.value: [8, 8, 8, 8],
            dat.TickDataColumns.BID3P.value: [7, 7, 7, 7],
            dat.TickDataColumns.BID4P.value: [6, 6, 6, 6],
            dat.TickDataColumns.BID5P.value: [5, 5, 5, 5],
            dat.TickDataColumns.BID1Q.value: [3, 3, 4, 4],
            dat.TickDataColumns.BID2Q.value: [5, 5, 4, 3],
            dat.TickDataColumns.BID3Q.value: [3, 3, 2, 2],
            dat.TickDataColumns.BID4Q.value: [2, 2, 2, 2],
            dat.TickDataColumns.BID5Q.value: [1, 1, 1, 1],
        })

        self.tick_df_2 = pd.DataFrame({
            dat.TickDataColumns.TIMESTAMP_NANO.value: [15, 17, 19, 20],
            dat.TickDataColumns.LAST_PRICE.value: [15, 17, 19, 20],
            dat.TickDataColumns.LAST_QUANTITY.value: [10, 12, 13, 15],
            dat.TickDataColumns.ASK1P.value: [10, 11, 12, 13],
            dat.TickDataColumns.ASK2P.value: [11, 12, 13, 14],
            dat.TickDataColumns.ASK3P.value: [12, 13, 14, 15],
            dat.TickDataColumns.ASK4P.value: [13, 14, 15, 16],
            dat.TickDataColumns.ASK5P.value: [14, 15, 16, 17],
            dat.TickDataColumns.ASK1Q.value: [5, 5, 5, 5],
            dat.TickDataColumns.ASK2Q.value: [2, 2, 2, 2],
            dat.TickDataColumns.ASK3Q.value: [3, 3, 3, 3],
            dat.TickDataColumns.ASK4Q.value: [2, 3, 2, 3],
            dat.TickDataColumns.ASK5Q.value: [4, 4, 2, 2],
            dat.TickDataColumns.BID1P.value: [9, 9, 9, 9],
            dat.TickDataColumns.BID2P.value: [8, 8, 8, 8],
            dat.TickDataColumns.BID3P.value: [7, 7, 7, 7],
            dat.TickDataColumns.BID4P.value: [6, 6, 6, 6],
            dat.TickDataColumns.BID5P.value: [5, 5, 5, 5],
            dat.TickDataColumns.BID1Q.value: [3, 3, 4, 4],
            dat.TickDataColumns.BID2Q.value: [5, 5, 4, 3],
            dat.TickDataColumns.BID3Q.value: [3, 3, 2, 2],
            dat.TickDataColumns.BID4Q.value: [2, 2, 2, 2],
            dat.TickDataColumns.BID5Q.value: [1, 1, 1, 1],
        })

    def tearDown(self):
        pass

    def test_date_equality(self):
        self.assertEqual(self.date1, self.date2)
        self.assertNotEqual(self.date1, self.date3)
        self.assertNotEqual(self.date1, self.date4)
        self.assertNotEqual(self.date3, self.date4)

    def test_bar_equality(self):
        bar_info_1 = dat.BarInfo(symbol = self.symbol, date = self.date1, intra_day_period = self.MORNING, sampling_level = 10, sampling_type = dat.Sampling.VOLUME)
        bar_info_2 = dat.BarInfo(symbol = self.symbol, date = self.date2, intra_day_period = self.MORNING, sampling_level = 10, sampling_type = dat.Sampling.VOLUME)
        bar_info_3 = dat.BarInfo(symbol = self.symbol, date = self.date3, intra_day_period = self.AFTERNOON, sampling_level = 10, sampling_type = dat.Sampling.VOLUME)
        bar_info_4 = dat.BarInfo(symbol = self.symbol, date = self.date3, intra_day_period = self.AFTERNOON, sampling_level = 10, sampling_type = dat.Sampling.VOLUME)

        bar_1 = dat.BarDataFrame(bar_data = self.df, bar_info = bar_info_1)
        bar_2 = dat.BarDataFrame(bar_data=self.df, bar_info=bar_info_2)
        bar_3 = dat.BarDataFrame(bar_data=self.df, bar_info=bar_info_3)
        bar_4 = dat.BarDataFrame(bar_data=self.df2, bar_info=bar_info_4)

        self.assertEqual(bar_1, bar_2)
        self.assertNotEqual(bar_1, bar_3)
        self.assertNotEqual(bar_1, bar_4)
        self.assertNotEqual(bar_3, bar_4)

    def test_tick_equality(self):
        tick_info_1 = dat.TickInfo(symbol = self.symbol, date = self.date1, intra_day_period = self.MORNING)
        tick_info_2 = dat.TickInfo(symbol = self.symbol, date = self.date2, intra_day_period = self.MORNING)
        tick_info_3 = dat.TickInfo(symbol = self.symbol, date = self.date3, intra_day_period = self.AFTERNOON)
        tick_info_4 = dat.TickInfo(symbol = self.symbol, date = self.date3, intra_day_period = self.AFTERNOON)

        tick_1 = dat.TickDataFrame(tick_df = self.tick_df, tick_info = tick_info_1)
        tick_2 = dat.TickDataFrame(tick_df = self.tick_df, tick_info = tick_info_2)
        tick_3 = dat.TickDataFrame(tick_df = self.tick_df, tick_info = tick_info_3)
        tick_4 = dat.TickDataFrame(tick_df=self.tick_df_2, tick_info = tick_info_4)
        self.assertEqual(tick_1, tick_2)
        self.assertNotEqual(tick_1, tick_3)
        self.assertNotEqual(tick_1, tick_4)
        self.assertNotEqual(tick_3, tick_4)

if __name__ == '__main__':
    unittest.main()