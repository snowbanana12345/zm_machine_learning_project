from src.data_base_module.data_retrival import instance as db
import src.data_base_module.data_blocks as data
import src.plotting_module.tick_data_plotter as tick_plot


# ------- retrieve raw tick data -------
morning_df_wrapper = db.get_clean_tick_data(symbol = "NHK17", date = data.Date(day = 7, month = 3, year = 2017), intra_day_period = data.IntraDayPeriod.MORNING)
afternoon_df_wrapper = db.get_clean_tick_data(symbol = "NHK17", date = data.Date(day = 7, month = 3, year = 2017), intra_day_period = data.IntraDayPeriod.AFTERNOON)

# ------- plot morning data --------
tick_plot.plot_and_save_tick_trade_volume(tick_df_wrapper = morning_df_wrapper, sampling_seconds = 300, is_raw_data = False)
tick_plot.plot_and_save_tick_trade_price(tick_df_wrapper = morning_df_wrapper, is_raw_data = False)
tick_plot.plot_and_save_bid_ask_prices(tick_df_wrapper = morning_df_wrapper, is_raw_data = False)
tick_plot.plot_and_save_bid_ask_spread(tick_df_wrapper = morning_df_wrapper, is_raw_data = False)
tick_plot.plot_and_save_bid_ask_quantities(tick_df_wrapper = morning_df_wrapper, is_raw_data = False)
tick_plot.plot_and_save_tick_count(tick_df_wrapper = morning_df_wrapper, sampling_seconds = 300, is_raw_data = False)
tick_plot.plot_and_save_tick_avg_bid_ask_spread(tick_df_wrapper = morning_df_wrapper, sampling_seconds = 300, is_raw_data = False)

# ------- plot afternoon data -------
tick_plot.plot_and_save_tick_trade_volume(tick_df_wrapper = afternoon_df_wrapper, sampling_seconds = 300, is_raw_data = False)
tick_plot.plot_and_save_tick_trade_price(tick_df_wrapper = afternoon_df_wrapper, is_raw_data = False)
tick_plot.plot_and_save_bid_ask_prices(tick_df_wrapper = afternoon_df_wrapper, is_raw_data = False)
tick_plot.plot_and_save_bid_ask_spread(tick_df_wrapper = afternoon_df_wrapper, is_raw_data = False)
tick_plot.plot_and_save_bid_ask_quantities(tick_df_wrapper = afternoon_df_wrapper, is_raw_data = False)
tick_plot.plot_and_save_tick_count(tick_df_wrapper = afternoon_df_wrapper, sampling_seconds = 300, is_raw_data = False)
tick_plot.plot_and_save_tick_avg_bid_ask_spread(tick_df_wrapper = afternoon_df_wrapper, sampling_seconds = 300, is_raw_data = False)

