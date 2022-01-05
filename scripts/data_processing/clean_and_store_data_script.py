from src.data_base_module.data_retrival import instance as db
import src.data_base_module.data_blocks as data
import src.data_processing_module.data_cleaning as data_cleaner

# ------- retrieve raw tick data -------
tick_df_wrapper = db.get_raw_tick_data("NHK17", data.Date(day = 7, month = 3, year = 2017))
# ------- split tick data between morning and afternoon to deal with mid day gapping ------
morning_df_wrapper, afternoon_df_wrapper = data_cleaner.morning_after_noon_split(tick_df_wrapper, 7)

# ------ clean morning data -------
data_cleaner.interpolate_zero_bid_ask_prices(morning_df_wrapper)
data_cleaner.interpolate_zero_bid_ask_quantities(morning_df_wrapper)
data_cleaner.interpolate_bid_ask_quantity_outliers(morning_df_wrapper, outlier_threshold = 1000)
data_cleaner.interpolate_bid_ask_price_outliers(morning_df_wrapper, lower_threshold = 18000, upper_threshold = 20000)
data_cleaner.interpolate_trade_price_outliers(morning_df_wrapper, lower_threshold = 18000, upper_threshold = 20000)
data_cleaner.interpolate_trade_volume_outliers(morning_df_wrapper, outlier_threshold = 200)

# ------ clean afternoon data ------
data_cleaner.interpolate_zero_bid_ask_prices(afternoon_df_wrapper)
data_cleaner.interpolate_zero_bid_ask_quantities(afternoon_df_wrapper)
data_cleaner.interpolate_bid_ask_quantity_outliers(afternoon_df_wrapper, outlier_threshold = 1000)
data_cleaner.interpolate_bid_ask_price_outliers(afternoon_df_wrapper, lower_threshold = 18000, upper_threshold = 20000)
data_cleaner.interpolate_trade_price_outliers(afternoon_df_wrapper, lower_threshold = 18000, upper_threshold = 20000)
data_cleaner.interpolate_trade_volume_outliers(afternoon_df_wrapper, outlier_threshold = 200)

# ------ insert cleaned data into data base -------
db.insert_clean_tick_data(morning_df_wrapper)
db.insert_clean_tick_data(afternoon_df_wrapper)
