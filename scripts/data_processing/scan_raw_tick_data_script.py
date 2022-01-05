from src.data_base_module.data_retrival import instance as db
import src.data_base_module.data_blocks as data
import src.data_processing_module.data_diagnostics as data_diagnostics

tick_df_wrapper : data.TickDataFrame = db.get_raw_tick_data("NHK17", data.Date(day = 2, month = 2, year = 2017))
data_diagnostics.scan_tick_missing_values(tick_df_wrapper = tick_df_wrapper)
data_diagnostics.scan_tick_bid_ask_prices_zero_entries(tick_df_wrapper = tick_df_wrapper)
data_diagnostics.scan_tick_bid_ask_quantities_zero_entries(tick_df_wrapper = tick_df_wrapper)
data_diagnostics.scan_time_stamp_duplicates(tick_df_wrapper = tick_df_wrapper)
data_diagnostics.scan_both_non_zero_trade_price_and_quantity(tick_df_wrapper = tick_df_wrapper)
data_diagnostics.scan_both_non_zero_bid_ask_price_and_quantity(tick_df_wrapper = tick_df_wrapper)
data_diagnostics.scan_ask_in_ascending_order(tick_df_wrapper = tick_df_wrapper)
data_diagnostics.scan_bid_in_descending_order(tick_df_wrapper = tick_df_wrapper)
data_diagnostics.scan_trade_quantity_outliers(tick_df_wrapper = tick_df_wrapper, outlier_threshold = 200)