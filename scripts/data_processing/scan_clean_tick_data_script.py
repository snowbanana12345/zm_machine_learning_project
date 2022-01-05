from src.data_base_module.data_retrival import instance as db
import src.data_base_module.data_blocks as data
import src.data_processing_module.data_diagnostics as data_diagnostics
import src.data_processing_module.data_cleaning as data_cleaner

# ------- retrieve raw tick data -------
morning_df_wrapper = db.get_clean_tick_data(symbol = "NHK17", date = data.Date(day = 7, month = 3, year = 2017), intra_day_period = data.IntraDayPeriod.MORNING)
afternoon_df_wrapper = db.get_clean_tick_data(symbol = "NHK17", date = data.Date(day = 7, month = 3, year = 2017), intra_day_period = data.IntraDayPeriod.AFTERNOON)

# ------ scan cleaned morning data -------
data_diagnostics.scan_tick_missing_values(tick_df_wrapper = morning_df_wrapper)
data_diagnostics.scan_tick_bid_ask_prices_zero_entries(tick_df_wrapper = morning_df_wrapper)
data_diagnostics.scan_tick_bid_ask_quantities_zero_entries(tick_df_wrapper = morning_df_wrapper)
data_diagnostics.scan_time_stamp_duplicates(tick_df_wrapper = morning_df_wrapper)
data_diagnostics.scan_both_non_zero_trade_price_and_quantity(tick_df_wrapper = morning_df_wrapper)
data_diagnostics.scan_both_non_zero_bid_ask_price_and_quantity(tick_df_wrapper = morning_df_wrapper)
data_diagnostics.scan_ask_in_ascending_order(tick_df_wrapper = morning_df_wrapper)
data_diagnostics.scan_bid_in_descending_order(tick_df_wrapper = morning_df_wrapper)
data_diagnostics.scan_bid_ask_price_outliers(tick_df_wrapper = morning_df_wrapper, lower_threshold = 18000, upper_threshold = 20000)
data_diagnostics.scan_bid_ask_quantity_outliers(tick_df_wrapper = morning_df_wrapper, lower_threshold =  0, upper_threshold = 1000)

# ------ scan cleaned afternoon data -------
data_diagnostics.scan_tick_missing_values(tick_df_wrapper = afternoon_df_wrapper)
data_diagnostics.scan_tick_bid_ask_prices_zero_entries(tick_df_wrapper = afternoon_df_wrapper)
data_diagnostics.scan_tick_bid_ask_quantities_zero_entries(tick_df_wrapper = afternoon_df_wrapper)
data_diagnostics.scan_time_stamp_duplicates(tick_df_wrapper = afternoon_df_wrapper)
data_diagnostics.scan_both_non_zero_trade_price_and_quantity(tick_df_wrapper = afternoon_df_wrapper)
data_diagnostics.scan_both_non_zero_bid_ask_price_and_quantity(tick_df_wrapper = afternoon_df_wrapper)
data_diagnostics.scan_ask_in_ascending_order(tick_df_wrapper = afternoon_df_wrapper)
data_diagnostics.scan_bid_in_descending_order(tick_df_wrapper = afternoon_df_wrapper)
data_diagnostics.scan_bid_ask_price_outliers(tick_df_wrapper = afternoon_df_wrapper, lower_threshold = 18000, upper_threshold = 20000)
data_diagnostics.scan_bid_ask_quantity_outliers(tick_df_wrapper = afternoon_df_wrapper, lower_threshold =  0, upper_threshold = 1000)
