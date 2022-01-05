from src.data_base_module.data_retrival import instance as db
import src.data_base_module.data_blocks as data
import src.plotting_module.tick_data_plotter as tick_plot

tick_df_wrapper = db.get_raw_tick_data("NHK17", data.Date(day = 7, month = 3, year = 2017))
#tick_plot.plot_and_save_tick_trade_volume(tick_df_wrapper = tick_df_wrapper, sampling_seconds = 300, is_raw_data = True)
#tick_plot.plot_and_save_tick_trade_price(tick_df_wrapper = tick_df_wrapper, is_raw_data = True)
#tick_plot.plot_and_save_bid_ask_prices(tick_df_wrapper = tick_df_wrapper, is_raw_data = True)
#tick_plot.plot_and_save_bid_ask_spread(tick_df_wrapper = tick_df_wrapper, is_raw_data = True)
#tick_plot.plot_and_save_bid_ask_quantities(tick_df_wrapper = tick_df_wrapper, is_raw_data = True)
#tick_plot.plot_and_save_tick_count(tick_df_wrapper = tick_df_wrapper, sampling_seconds = 300, is_raw_data = True)
tick_plot.plot_and_save_tick_avg_bid_ask_spread(tick_df_wrapper = tick_df_wrapper, sampling_seconds = 300, is_raw_data = True)