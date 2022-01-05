import logging
import definitions
import os
import src.data_base_module.data_blocks as data

# ------- initialize a logger for this specific module -------
logger = logging.getLogger("plotting_module")
logger.setLevel(level=logging.INFO)
file_handler = logging.FileHandler(os.path.join(definitions.LOGS_FOLDER_PATH, "plotting.log"))
formatter = logging.Formatter("[%(asctime)s] : %(levelname)s : [%(name)s] : %(message)s ")
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# ------ tick plotting logging -------
def get_tick_df_wrapper_str(tick_df_wrapper : data.TickDataFrame) -> str:
    return tick_df_wrapper.symbol + " -- " + tick_df_wrapper.date.get_str_format_2() + " -- " + tick_df_wrapper.intra_day_period.value


def log_plot_and_save_tick_vol(tick_df_wrapper : data.TickDataFrame, sampling_seconds : int, is_raw_data : bool) -> None:
    message = get_tick_df_wrapper_str(tick_df_wrapper = tick_df_wrapper)
    message += " -- " + " plotted and saved volume traded in " + str(sampling_seconds) + "S intervals" + " on " + ("raw" if is_raw_data else "cleaned")
    logger.info(message)


def log_plot_and_save_tick_trade_price(tick_df_wrapper : data.TickDataFrame, is_raw_data : bool) -> None:
    message = get_tick_df_wrapper_str(tick_df_wrapper=tick_df_wrapper)
    message += " -- " + " plotted and saved traded price " + " on " + ("raw" if is_raw_data else "cleaned")
    logger.info(message)


def log_plot_and_save_tick_bid_ask_price(tick_df_wrapper : data.TickDataFrame, bid_ask_col_name : str, is_raw_data : bool) -> None:
    message = get_tick_df_wrapper_str(tick_df_wrapper = tick_df_wrapper)
    message += " -- " + " plotted and saved " + bid_ask_col_name + " on " + ("raw" if is_raw_data else "cleaned")
    logger.info(message)


def log_plot_and_save_bid_ask_spread(tick_df_wrapper : data.TickDataFrame, is_raw_data : bool) -> None:
    message = get_tick_df_wrapper_str(tick_df_wrapper = tick_df_wrapper)
    message += " -- " + " plotted and saved bid ask spread " + " on " + ("raw" if is_raw_data else "cleaned")
    logger.info(message)


def log_plot_and_save_tick_bid_ask_quantity(tick_df_wrapper : data.TickDataFrame, bid_ask_qty_col_name : str, is_raw_data : bool) -> None:
    message = get_tick_df_wrapper_str(tick_df_wrapper = tick_df_wrapper)
    message += " -- " + " plotted and saved " + bid_ask_qty_col_name + " quantity " + " on " + ("raw" if is_raw_data else "cleaned")
    logger.info(message)


def log_plot_and_save_tick_counts(tick_df_wrapper : data.TickDataFrame, sampling_seconds : int, is_raw_data : bool):
    message = get_tick_df_wrapper_str(tick_df_wrapper=tick_df_wrapper)
    message += " -- " + " plotted and saved " + " number of tick counts in intervals of " + str(sampling_seconds) + "S on " + ("raw" if is_raw_data else "cleaned")
    logger.info(message)


def log_plot_and_save_avg_bid_ask_spread(tick_df_wrapper : data.TickDataFrame, sampling_seconds : int, is_raw_data : bool):
    message = get_tick_df_wrapper_str(tick_df_wrapper=tick_df_wrapper)
    message += " -- " + " plotted and saved " + " tick average bid ask spread in intervals of " + str(sampling_seconds) + "S on " + ("raw" if is_raw_data else "cleaned")
    logger.info(message)

# ------ bar data plotting functions -----
def log_plot_and_save_time_bar(time_bar_wrapper : data.TimeBarDataFrame, bar_col : data.BarDataColumns):
    message = str(time_bar_wrapper)
    message += " -- Plotted and saved : " + bar_col.value
    logger.info(message)


def log_plot_and_save_tick_bar(tick_bar_wrapper : data.TickBarDataFrame, bar_col : data.BarDataColumns):
    message = str(tick_bar_wrapper)
    message += " -- Plotted and saved : " + bar_col.value
    logger.info(message)


def log_plot_and_save_volume_bar(volume_bar_wrapper : data.VolumeBarDataFrame, bar_col : data.BarDataColumns):
    message = str(volume_bar_wrapper)
    message += " -- Plotted and saved : " + bar_col.value
    logger.info(message)


def log_plot_and_save_dollar_bar(dollar_bar_wrapper : data.DollarBarDataFrame, bar_col : data.BarDataColumns):
    message = str(dollar_bar_wrapper)
    message += " -- Plotted and saved : " + bar_col.value
    logger.info(message)

# ------ machine learning plotting functions -----
def log_plot_price_series_with_filter(price_series_name : str, filter_name : str):
    message = "Plotted : " + price_series_name + " with filter : " + filter_name + " applied"
    logger.info(message)