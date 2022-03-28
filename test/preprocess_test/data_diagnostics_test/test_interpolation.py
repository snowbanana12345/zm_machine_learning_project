import unittest
import src.data_processing_module.data_cleaning as dat_clean
import src.data_base_module.data_blocks as dat_blocks
import pandas as pd

MORNING = dat_blocks.IntraDayPeriod.MORNING
AFTERNOON = dat_blocks.IntraDayPeriod.AFTERNOON
WHOLE_DAY = dat_blocks.IntraDayPeriod.WHOLE_DAY

class TestInterpolate(unittest.TestCase):
    def setUp(self):
        time_stamp_row = [12, 25, 58, 90, 107, 111, 150, 160, 179, 185]
        time_stamp_row = [ts * 3600 * 1E9 for ts in time_stamp_row]
        trade_price_row = [0,192, 0, 193, 195, 196, 0, 198, 199, 195]
        trade_quantity_row = [0, 1, 0, 2, 4, 3, 0, 1, 2, 3]
        ask1p_row = [191, 192, 193, 193, 195, 196, 197, 198, 199, 198]
        ask2p_row = [192, 193, 194, 195, 196, 197, 198, 199, 200, 199]
        ask3p_row = [193, 194, 195, 196, 197, 198, 199, 200, 201, 200]
        ask4p_row = [194, 195, 196, 197, 198, 199, 200, 201, 202, 201]
        ask5p_row = [195, 196, 197, 198, 199, 200, 201, 202, 203, 202]
        ask1q_row = [2, 4, 5, 6, 10, 9, 5, 4, 3, 2]
        ask2q_row = [3, 4, 6, 9, 5, 11, 13, 1, 3, 3]
        ask3q_row = [4, 5, 6, 1, 2, 3, 4, 5, 6, 9]
        ask4q_row = [5, 6, 1, 2, 3, 5, 6, 9, 7, 5]
        ask5q_row = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        bid1p_row = [190, 191, 192, 192, 194, 195, 197, 197, 198, 197]
        bid2p_row = [189, 190, 191, 191, 193, 194, 196, 196, 197, 196]
        bid3p_row = [188, 189, 190, 190, 192, 193, 194, 195, 195, 194]
        bid4p_row = [187, 188, 189, 189, 191, 192, 193, 194, 194, 193]
        bid5p_row = [186, 187, 188, 187, 190, 191, 192, 193, 193, 192]
        bid1q_row = [5, 3, 10, 9, 4, 7, 8, 5, 4, 2]
        bid2q_row = [4, 3, 2, 1, 5, 7, 8, 9, 10, 11]
        bid3q_row = [10, 9, 5, 6, 2, 1, 4, 5, 6, 8]
        bid4q_row = [5, 6, 9, 8, 1, 2, 3, 4, 5, 10]
        bid5q_row = [1, 1, 3, 2, 4, 5, 6, 9, 8, 7]

        self.example_tick_df = pd.DataFrame({
            dat_blocks.TickDataColumns.TIMESTAMP_NANO.value : time_stamp_row,
            dat_blocks.TickDataColumns.LAST_PRICE.value: trade_price_row,
            dat_blocks.TickDataColumns.LAST_QUANTITY.value:trade_quantity_row,
            dat_blocks.TickDataColumns.ASK1P.value: ask1p_row,
            dat_blocks.TickDataColumns.ASK2P.value: ask2p_row,
            dat_blocks.TickDataColumns.ASK3P.value: ask3p_row,
            dat_blocks.TickDataColumns.ASK4P.value: ask4p_row,
            dat_blocks.TickDataColumns.ASK5P.value: ask5p_row,
            dat_blocks.TickDataColumns.ASK1Q.value: ask1q_row,
            dat_blocks.TickDataColumns.ASK2Q.value: ask2q_row,
            dat_blocks.TickDataColumns.ASK3Q.value: ask3q_row,
            dat_blocks.TickDataColumns.ASK4Q.value: ask4q_row,
            dat_blocks.TickDataColumns.ASK5Q.value: ask5q_row,
            dat_blocks.TickDataColumns.BID1P.value: bid1p_row,
            dat_blocks.TickDataColumns.BID2P.value: bid2p_row,
            dat_blocks.TickDataColumns.BID3P.value: bid3p_row,
            dat_blocks.TickDataColumns.BID4P.value: bid4p_row,
            dat_blocks.TickDataColumns.BID5P.value: bid5p_row,
            dat_blocks.TickDataColumns.BID1Q.value: bid1q_row,
            dat_blocks.TickDataColumns.BID2Q.value: bid2q_row,
            dat_blocks.TickDataColumns.BID3Q.value: bid3q_row,
            dat_blocks.TickDataColumns.BID4Q.value: bid4q_row,
            dat_blocks.TickDataColumns.BID5Q.value: bid5q_row,
        })

        self.test_date = dat_blocks.Date(day = 1, month = 1, year = 1000)
        self.test_symbol = "TEST"


    def test_morning_afternoon_split(self):
        tick_wrapper = dat_blocks.TickDataFrame(tick_df = self.example_tick_df, date = self.test_date, symbol = self.test_symbol, intra_day_period = WHOLE_DAY)
        test_split_point = 96
        morning_wrapper, afternoon_wrapper = dat_clean.morning_after_noon_split(tick_wrapper, test_split_point)

        exp_morn_time_stamp_row = [12, 25, 58, 90, 107]
        exp_morn_time_stamp_row = [ts * 3600 * 1E9 for ts in exp_morn_time_stamp_row]
        exp_morn_trade_price_row = [0,192, 0, 193, 195]
        exp_morn_trade_quantity_row = [0, 1, 0, 2, 4]
        exp_morn_ask1p_row = [191, 192, 193, 193, 195]
        exp_morn_ask2p_row = [192, 193, 194, 195, 196]
        exp_morn_ask3p_row = [193, 194, 195, 196, 197]
        exp_morn_ask4p_row = [194, 195, 196, 197, 198]
        exp_morn_ask5p_row = [195, 196, 197, 198, 199]
        exp_morn_ask1q_row = [2, 4, 5, 6, 10]
        exp_morn_ask2q_row = [3, 4, 6, 9, 5]
        exp_morn_ask3q_row = [4, 5, 6, 1, 2]
        exp_morn_ask4q_row = [5, 6, 1, 2, 3]
        exp_morn_ask5q_row = [1, 2, 3, 4, 5]
        exp_morn_bid1p_row = [190, 191, 192, 192, 194]
        exp_morn_bid2p_row = [189, 190, 191, 191, 193]
        exp_morn_bid3p_row = [188, 189, 190, 190, 192]
        exp_morn_bid4p_row = [187, 188, 189, 189, 191]
        exp_morn_bid5p_row = [186, 187, 188, 187, 190]
        exp_morn_bid1q_row = [5, 3, 10, 9, 4]
        exp_morn_bid2q_row = [4, 3, 2, 1, 5]
        exp_morn_bid3q_row = [10, 9, 5, 6, 2]
        exp_morn_bid4q_row = [5, 6, 9, 8, 1]
        exp_morn_bid5q_row = [1, 1, 3, 2, 4]
        exp_morn_tick_df = pd.DataFrame({
            dat_blocks.TickDataColumns.TIMESTAMP_NANO.value : exp_morn_time_stamp_row,
            dat_blocks.TickDataColumns.LAST_PRICE.value: exp_morn_trade_price_row,
            dat_blocks.TickDataColumns.LAST_QUANTITY.value:exp_morn_trade_quantity_row,
            dat_blocks.TickDataColumns.ASK1P.value: exp_morn_ask1p_row,
            dat_blocks.TickDataColumns.ASK2P.value: exp_morn_ask2p_row,
            dat_blocks.TickDataColumns.ASK3P.value: exp_morn_ask3p_row,
            dat_blocks.TickDataColumns.ASK4P.value: exp_morn_ask4p_row,
            dat_blocks.TickDataColumns.ASK5P.value: exp_morn_ask5p_row,
            dat_blocks.TickDataColumns.ASK1Q.value: exp_morn_ask1q_row,
            dat_blocks.TickDataColumns.ASK2Q.value: exp_morn_ask2q_row,
            dat_blocks.TickDataColumns.ASK3Q.value: exp_morn_ask3q_row,
            dat_blocks.TickDataColumns.ASK4Q.value: exp_morn_ask4q_row,
            dat_blocks.TickDataColumns.ASK5Q.value: exp_morn_ask5q_row,
            dat_blocks.TickDataColumns.BID1P.value: exp_morn_bid1p_row,
            dat_blocks.TickDataColumns.BID2P.value: exp_morn_bid2p_row,
            dat_blocks.TickDataColumns.BID3P.value: exp_morn_bid3p_row,
            dat_blocks.TickDataColumns.BID4P.value: exp_morn_bid4p_row,
            dat_blocks.TickDataColumns.BID5P.value: exp_morn_bid5p_row,
            dat_blocks.TickDataColumns.BID1Q.value: exp_morn_bid1q_row,
            dat_blocks.TickDataColumns.BID2Q.value: exp_morn_bid2q_row,
            dat_blocks.TickDataColumns.BID3Q.value: exp_morn_bid3q_row,
            dat_blocks.TickDataColumns.BID4Q.value: exp_morn_bid4q_row,
            dat_blocks.TickDataColumns.BID5Q.value: exp_morn_bid5q_row,
        })
        exp_morn_wrapper = dat_blocks.TickDataFrame(tick_df = exp_morn_tick_df, date = self.test_date, symbol = self.test_symbol, intra_day_period = MORNING)

        exp_aft_time_stamp_row = [111, 150, 160, 179, 185]
        exp_aft_time_stamp_row = [ts * 3600 * 1E9 for ts in exp_aft_time_stamp_row]
        exp_aft_trade_price_row = [196, 0, 198, 199, 195]
        exp_aft_trade_quantity_row = [3, 0, 1, 2, 3]
        exp_aft_ask1p_row = [196, 197, 198, 199, 198]
        exp_aft_ask2p_row = [197, 198, 199, 200, 199]
        exp_aft_ask3p_row = [198, 199, 200, 201, 200]
        exp_aft_ask4p_row = [199, 200, 201, 202, 201]
        exp_aft_ask5p_row = [200, 201, 202, 203, 202]
        exp_aft_ask1q_row = [9, 5, 4, 3, 2]
        exp_aft_ask2q_row = [11, 13, 1, 3, 3]
        exp_aft_ask3q_row = [3, 4, 5, 6, 9]
        exp_aft_ask4q_row = [5, 6, 9, 7, 5]
        exp_aft_ask5q_row = [6, 7, 8, 9, 10]
        exp_aft_bid1p_row = [195, 197, 197, 198, 197]
        exp_aft_bid2p_row = [194, 196, 196, 197, 196]
        exp_aft_bid3p_row = [193, 194, 195, 195, 194]
        exp_aft_bid4p_row = [192, 193, 194, 194, 193]
        exp_aft_bid5p_row = [191, 192, 193, 193, 192]
        exp_aft_bid1q_row = [7, 8, 5, 4, 2]
        exp_aft_bid2q_row = [7, 8, 9, 10, 11]
        exp_aft_bid3q_row = [1, 4, 5, 6, 8]
        exp_aft_bid4q_row = [2, 3, 4, 5, 10]
        exp_aft_bid5q_row = [5, 6, 9, 8, 7]

        exp_aft_tick_df = pd.DataFrame({
            dat_blocks.TickDataColumns.TIMESTAMP_NANO.value : exp_aft_time_stamp_row,
            dat_blocks.TickDataColumns.LAST_PRICE.value: exp_aft_trade_price_row,
            dat_blocks.TickDataColumns.LAST_QUANTITY.value:exp_aft_trade_quantity_row,
            dat_blocks.TickDataColumns.ASK1P.value: exp_aft_ask1p_row,
            dat_blocks.TickDataColumns.ASK2P.value: exp_aft_ask2p_row,
            dat_blocks.TickDataColumns.ASK3P.value: exp_aft_ask3p_row,
            dat_blocks.TickDataColumns.ASK4P.value: exp_aft_ask4p_row,
            dat_blocks.TickDataColumns.ASK5P.value: exp_aft_ask5p_row,
            dat_blocks.TickDataColumns.ASK1Q.value: exp_aft_ask1q_row,
            dat_blocks.TickDataColumns.ASK2Q.value: exp_aft_ask2q_row,
            dat_blocks.TickDataColumns.ASK3Q.value: exp_aft_ask3q_row,
            dat_blocks.TickDataColumns.ASK4Q.value: exp_aft_ask4q_row,
            dat_blocks.TickDataColumns.ASK5Q.value: exp_aft_ask5q_row,
            dat_blocks.TickDataColumns.BID1P.value: exp_aft_bid1p_row,
            dat_blocks.TickDataColumns.BID2P.value: exp_aft_bid2p_row,
            dat_blocks.TickDataColumns.BID3P.value: exp_aft_bid3p_row,
            dat_blocks.TickDataColumns.BID4P.value: exp_aft_bid4p_row,
            dat_blocks.TickDataColumns.BID5P.value: exp_aft_bid5p_row,
            dat_blocks.TickDataColumns.BID1Q.value: exp_aft_bid1q_row,
            dat_blocks.TickDataColumns.BID2Q.value: exp_aft_bid2q_row,
            dat_blocks.TickDataColumns.BID3Q.value: exp_aft_bid3q_row,
            dat_blocks.TickDataColumns.BID4Q.value: exp_aft_bid4q_row,
            dat_blocks.TickDataColumns.BID5Q.value: exp_aft_bid5q_row,
        })

        exp_aft_wrapper = dat_blocks.TickDataFrame(tick_df = exp_aft_tick_df, date = self.test_date, symbol = self.test_symbol, intra_day_period = AFTERNOON)

        self.assertEqual(morning_wrapper.symbol, exp_morn_wrapper.symbol, "morning symbol comparison")
        self.assertEqual(morning_wrapper.date, exp_morn_wrapper.date, "morning date comparison")
        self.assertTrue(morning_wrapper.tick_data.equals(exp_morn_wrapper.tick_data), "morning tick_data comparison")
        self.assertEqual(morning_wrapper.intra_day_period, morning_wrapper.intra_day_period, "morning intra_day_period comparison")
        self.assertEqual(morning_wrapper, exp_morn_wrapper, "morning comparison")

        self.assertEqual(afternoon_wrapper.symbol, exp_aft_wrapper.symbol, "afternoon symbol comparison")
        self.assertEqual(afternoon_wrapper.date, exp_aft_wrapper.date, "morning date comparison")
        self.assertTrue(afternoon_wrapper.tick_data.equals(exp_aft_wrapper.tick_data), "morning tick_data comparison")
        self.assertEqual(afternoon_wrapper.intra_day_period, exp_aft_wrapper.intra_day_period, "morning intra_day_period comparison")
        self.assertEqual(afternoon_wrapper, exp_aft_wrapper, "afternoon comparison")

    def test_interpolate_zero_bid_ask_prices(self):
        expected_tick_df = self.example_tick_df.copy()
        self.example_tick_df.loc[:, dat_blocks.TickDataColumns.ASK1P.value] = [191.0, 192.0, 193.0, 193.0, 195.0, 0.0, 197.0, 198.0, 199.0, 198.0]
        self.example_tick_df.loc[:, dat_blocks.TickDataColumns.BID1P.value] = [0.0, 191.0, 192.0, 192.0, 194.0, 195.0, 197.0, 197.0, 198.0, 0.0]
        tick_wrapper = dat_blocks.TickDataFrame(tick_df = self.example_tick_df, date = self.test_date, symbol = self.test_symbol, intra_day_period = MORNING)
        dat_clean.interpolate_zero_bid_ask_prices(tick_df_wrapper = tick_wrapper)
        expected_tick_df.loc[:, dat_blocks.TickDataColumns.ASK1P.value] = [191.0, 192.0, 193.0, 193.0, 195.0, 196.0, 197.0, 198.0, 199.0, 198.0]
        expected_tick_df.loc[:, dat_blocks.TickDataColumns.BID1P.value] = [191.0, 191.0, 192.0, 192.0, 194.0, 195.0, 197.0, 197.0, 198.0, 198.0]
        expected_tick_wrapper = dat_blocks.TickDataFrame(tick_df = expected_tick_df, date = self.test_date, symbol = self.test_symbol, intra_day_period = MORNING)
        self.assertEqual(tick_wrapper, expected_tick_wrapper, "Intepolate_zero_bid_ask_prices")

    def test_interpolate_zero_bid_ask_quantities(self):
        expected_tick_df = self.example_tick_df.copy()
        self.example_tick_df.loc[:, dat_blocks.TickDataColumns.ASK1Q.value] = [2.0, 4.0, 0.0, 0.0, 10.0, 9.0, 5.0, 4.0, 3.0, 2.0]
        self.example_tick_df.loc[:, dat_blocks.TickDataColumns.BID4Q.value] = [0.0, 6.0, 9.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
        tick_wrapper = dat_blocks.TickDataFrame(tick_df=self.example_tick_df, date=self.test_date,symbol=self.test_symbol, intra_day_period=MORNING)
        dat_clean.interpolate_zero_bid_ask_quantities(tick_df_wrapper=tick_wrapper)
        expected_tick_df.loc[:, dat_blocks.TickDataColumns.ASK1Q.value] = [2.0, 4.0, 6.0, 8.0, 10.0, 9.0, 5.0, 4.0, 3.0, 2.0]
        expected_tick_df.loc[:, dat_blocks.TickDataColumns.BID4Q.value] = [6.0, 6.0, 9.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
        expected_tick_wrapper = dat_blocks.TickDataFrame(tick_df=expected_tick_df, date=self.test_date,symbol=self.test_symbol, intra_day_period=MORNING)
        self.assertEqual(tick_wrapper, expected_tick_wrapper, "Intepolate_zero_bid_ask_quantites")

    def test_interpolate_bid_ask_price_outliers(self):
        """ the function is not modifying the input tick data frame """
        upper_limit = 300
        lower_limit = 100
        tick_df = self.example_tick_df.copy()
        expected_tick_df = self.example_tick_df.copy()
        tick_df.loc[:, dat_blocks.TickDataColumns.ASK3P.value] = [191.0, 192.0, 301.0, 193.0, 195.0, 196.0 ,197.0, 99.0, 199.0, 198.0]
        tick_df.loc[:, dat_blocks.TickDataColumns.BID5P.value] = [50.0, 191.0, 192.0, 192.0, 194.0, 195.0,197.0, 197.0, 198.0, 400.0]
        tick_wrapper = dat_blocks.TickDataFrame(tick_df=tick_df, date=self.test_date,symbol=self.test_symbol, intra_day_period=MORNING)
        dat_clean.interpolate_bid_ask_price_outliers(tick_df_wrapper=tick_wrapper, lower_threshold = lower_limit, upper_threshold = upper_limit)
        expected_tick_df.loc[:, dat_blocks.TickDataColumns.ASK3P.value] = [191.0, 192.0, 192.5, 193.0, 195.0, 196.0 ,197.0, 198.0, 199.0, 198.0]
        expected_tick_df.loc[:, dat_blocks.TickDataColumns.BID5P.value] = [191.0, 191.0, 192.0, 192.0, 194.0, 195.0,197.0, 197.0, 198.0, 198.8]
        expected_tick_wrapper = dat_blocks.TickDataFrame(tick_df=expected_tick_df, date=self.test_date,symbol=self.test_symbol, intra_day_period=MORNING)
        self.assertTrue(all(tick_df == expected_tick_df))

    def test_interpolate_bid_ask_quantity_outliers(self):
        """ the function is not modifying the input tick data frame """
        upper_limit = 30
        lower_limit = 5
        tick_df = self.example_tick_df.copy()
        expected_tick_df = self.example_tick_df.copy()
        tick_df.loc[:, dat_blocks.TickDataColumns.ASK4Q.value] = [15.0, 16.0, 17.0, 20.0, 5.0, 10.0, 12.0, 13.0, 30.0, 20.0]
        tick_df.loc[:, dat_blocks.TickDataColumns.BID1Q.value] = [35.0, 16.0, 17.0, 20.0, 13.0, 10.0, 12.0, 13.0, 17.0, 1.0]
        tick_wrapper = dat_blocks.TickDataFrame(tick_df=tick_df, date=self.test_date,symbol=self.test_symbol, intra_day_period=MORNING)
        dat_clean.interpolate_bid_ask_price_outliers(tick_df_wrapper=tick_wrapper, lower_threshold = lower_limit, upper_threshold = upper_limit)
        expected_tick_df.loc[:, dat_blocks.TickDataColumns.ASK4Q.value] = [15.0, 16.0, 17.0, 20.0, 13.0, 10.0, 12.0, 13.0, 17.0, 20.0]
        expected_tick_df.loc[:, dat_blocks.TickDataColumns.BID1Q.value] = [16.0, 16.0, 17.0, 20.0, 13.0, 10.0, 12.0, 13.0, 17.0, 17.0]
        expected_tick_wrapper = dat_blocks.TickDataFrame(tick_df=expected_tick_df, date=self.test_date,symbol=self.test_symbol, intra_day_period=MORNING)
        self.assertTrue(all(tick_df == expected_tick_df))


    def test_interpolate_trade_price_outliers(self):
        upper_limit = 315
        lower_limit = 160
        tick_df = self.example_tick_df.copy()
        expected_tick_df = self.example_tick_df.copy()
        tick_df.loc[:, dat_blocks.TickDataColumns.LAST_PRICE.value] = [190.0, 187.0, 0.0, 150.0, 186.0, 190.0, 0.0, 320.0, 0.0, 200.0]
        tick_wrapper = dat_blocks.TickDataFrame(tick_df=tick_df, date=self.test_date, symbol=self.test_symbol,intra_day_period=MORNING)
        dat_clean.interpolate_bid_ask_price_outliers(tick_df_wrapper=tick_wrapper, lower_threshold=lower_limit,upper_threshold=upper_limit)
        expected_tick_df.loc[:, dat_blocks.TickDataColumns.LAST_PRICE.value] = [190.0, 187.0, 0.0, 186.5, 186.0, 190.0, 0.0 , 195.0, 0.0, 200.0]
        expected_tick_wrapper = dat_blocks.TickDataFrame(tick_df=expected_tick_df, date=self.test_date,symbol=self.test_symbol, intra_day_period=MORNING)
        self.assertTrue(all(tick_df == expected_tick_df))

    def test_interpolate_trade_volume_outliers(self):
        upper_limit = 100.0
        lower_limit = 30.0
        tick_df = self.example_tick_df.copy()
        expected_tick_df = self.example_tick_df.copy()
        tick_df.loc[:, dat_blocks.TickDataColumns.LAST_QUANTITY.value] = [50.0, 0.0, 20.0, 45.0, 0.0, 55.0, 65.0, 75.0, 150.0, 65.0]
        tick_wrapper = dat_blocks.TickDataFrame(tick_df=tick_df, date=self.test_date, symbol=self.test_symbol,intra_day_period=MORNING)
        dat_clean.interpolate_bid_ask_price_outliers(tick_df_wrapper=tick_wrapper, lower_threshold=lower_limit, upper_threshold=upper_limit)
        expected_tick_df.loc[:, dat_blocks.TickDataColumns.LAST_QUANTITY.value] = [50.0, 0.0, 47.5, 45.0, 0.0, 55.0, 65.0, 75.0, 70.0, 65.0]
        expected_tick_wrapper = dat_blocks.TickDataFrame(tick_df=expected_tick_df, date=self.test_date,symbol=self.test_symbol, intra_day_period=MORNING)
        self.assertTrue(all(tick_df == expected_tick_df))

    def test_interpolate_bar_zero_prices(self):
        timestamps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        open_prices = [0, 10, 12, 13, 9, 7, 5, 8, 9, 10]
        close_prices = [6, 7, 8, 10, 12, 13, 16, 5, 4, 3]
        high_prices = [4, 5, 6, 10, 0, 7, 5, 4, 2, 3]
        low_prices = [5, 6, 2, 3, 4, 5, 9, 8, 7, 0]
        vwap_prices = [20, 21, 22, 0, 24, 25, 26, 27, 28, 29]
        volumes = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        open_prices = [float(x) for x in open_prices]
        close_prices = [float(x) for x in close_prices]
        high_prices = [float(x) for x in high_prices]
        low_prices = [float(x) for x in low_prices]
        vwap_prices = [float(x) for x in vwap_prices]
        bar_df = pd.DataFrame({
            dat_blocks.BarDataColumns.TIMESTAMP.value : timestamps,
            dat_blocks.BarDataColumns.OPEN.value: open_prices,
            dat_blocks.BarDataColumns.CLOSE.value: close_prices,
            dat_blocks.BarDataColumns.HIGH.value: high_prices,
            dat_blocks.BarDataColumns.LOW.value: low_prices,
            dat_blocks.BarDataColumns.VWAP.value: vwap_prices,
            dat_blocks.BarDataColumns.VOLUME.value: volumes,
        })

        bar_wrapper = dat_blocks.BarDataFrame(symbol = self.test_symbol)
        bar_wrapper.set_data_frame(bar_df)
        result_bar_wrapper = dat_clean.interpolate_bar_zero_prices(bar_wrapper)

        exp_timestamps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        exp_open_prices = [10, 10, 12, 13, 9, 7, 5, 8, 9, 10]
        exp_close_prices = [6, 7, 8, 10, 12, 13, 16, 5, 4, 3]
        exp_high_prices = [4, 5, 6, 10, 8.5, 7, 5, 4, 2, 3]
        exp_low_prices = [5, 6, 2, 3, 4, 5, 9, 8, 7, 7]
        exp_vwap_prices = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        exp_volumes = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        exp_open_prices = [float(x) for x in exp_open_prices]
        exp_close_prices = [float(x) for x in exp_close_prices]
        exp_high_prices = [float(x) for x in exp_high_prices]
        exp_low_prices = [float(x) for x in exp_low_prices]
        exp_vwap_prices = [float(x) for x in exp_vwap_prices]

        exp_bar_df = pd.DataFrame({
            dat_blocks.BarDataColumns.TIMESTAMP.value : exp_timestamps,
            dat_blocks.BarDataColumns.OPEN.value: exp_open_prices,
            dat_blocks.BarDataColumns.CLOSE.value: exp_close_prices,
            dat_blocks.BarDataColumns.HIGH.value: exp_high_prices,
            dat_blocks.BarDataColumns.LOW.value: exp_low_prices,
            dat_blocks.BarDataColumns.VWAP.value: exp_vwap_prices,
            dat_blocks.BarDataColumns.VOLUME.value: exp_volumes,
        })

        exp_bar_wrapper = dat_blocks.BarDataFrame(symbol = self.test_symbol)
        exp_bar_wrapper.set_data_frame(exp_bar_df)
        self.assertTrue(all(result_bar_wrapper.get_bar_data_reference() == exp_bar_df))

if __name__ == '__main__':
    unittest.main()


