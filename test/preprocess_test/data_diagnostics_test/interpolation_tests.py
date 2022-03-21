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
        self.assertTrue(True)

    def interpolate_bid_ask_price_outliers(self):
        self.assertTrue(True)

    def interpolate_bid_ask_quantity_outliers(self):
        self.assertTrue(True)

    def interpolate_trade_price_outliers(self):
        pass

    def interpolate_trade_volume_outliers(self):
        pass

    def interpolate_bar_zero_prices(self):
        pass

if __name__ == '__main__':
    unittest.main()

