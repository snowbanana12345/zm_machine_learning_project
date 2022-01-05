import logging
import definitions
import os
import src.data_base_module.data_blocks as data

# ------- initialize a logger for this specific module -------
LOGGER = logging.getLogger("data_processing_module")
LOGGER.setLevel(level=logging.INFO)
file_handler = logging.FileHandler(os.path.join(definitions.LOGS_FOLDER_PATH, "data_processing.log"))
formatter = logging.Formatter("[%(asctime)s] : %(levelname)s : [%(name)s] : %(message)s ")
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
LOGGER.addHandler(file_handler)
LOGGER.addHandler(stream_handler)


# ------- logging functions for sampling ---------
def log_time_sampling_start(ticker_symbol : str, sampling_seconds: int, date : data.Date,
                            intra_day_period : data.IntraDayPeriod) -> None:
    message = "Started sampling : " + ticker_symbol + " -- " + date.get_str_format_2() + " -- " + intra_day_period.value \
              + " -- into " + str(sampling_seconds) + "S time bars "
    LOGGER.info(message)


def log_time_sampling_end(ticker_symbol : str, sampling_seconds: int, date: data.Date,
                          intra_day_period: data.IntraDayPeriod) -> None:
    message = "Finished sampling : " + ticker_symbol + " -- " + date.get_str_format_2() + " -- " + intra_day_period.value \
              + " -- into " + str(sampling_seconds) + "S time bars "
    LOGGER.info(message)


def log_volume_sampling_start(ticker_symbol : str, sampling_volume : int, date: data.Date, intra_day_period: data.IntraDayPeriod) -> None:
    message = "Started sampling : " + ticker_symbol + " -- " + date.get_str_format_2() + " -- " + intra_day_period.value \
              + " -- into " + str(sampling_volume) + " volume bars "
    LOGGER.info(message)


def log_volume_sampling_end(ticker_symbol : str, sampling_volume : int, date: data.Date, intra_day_period: data.IntraDayPeriod) -> None:
    message = "Finished sampling : " + ticker_symbol + " -- " + date.get_str_format_2() + " -- " + intra_day_period.value \
              + " -- into " + str(sampling_volume) + " volume bars "
    LOGGER.info(message)


def log_tick_sampling_start(ticker_symbol : str, sampling_ticks : int, date : data.Date,
                            intra_day_period : data.IntraDayPeriod) -> None:
    message = "Started sampling : " + ticker_symbol + " -- " + date.get_str_format_2() + " -- " + intra_day_period.value \
              + " -- into " + str(sampling_ticks) + "_tick tick bars "
    LOGGER.info(message)


def log_tick_sampling_end(ticker_symbol : str, sampling_ticks : int, date : data.Date,
                            intra_day_period : data.IntraDayPeriod) -> None:
    message = "Finished sampling : " + ticker_symbol + " -- " + date.get_str_format_2() + " -- " + intra_day_period.value \
              + " -- into " + str(sampling_ticks) + "_tick tick bars "
    LOGGER.info(message)


def log_dollar_sampling_start(ticker_symbol : str, sampling_dollar : int , date : data.Date, intra_day_period : data.IntraDayPeriod) -> None:
    message = "Started sampling : " + ticker_symbol + " -- " + date.get_str_format_2() + " -- " + intra_day_period.value \
              + " -- into " + str(sampling_dollar) + "_dollar tick bars "
    LOGGER.info(message)


def log_dollar_sampling_end(ticker_symbol : str, sampling_dollar : int , date : data.Date, intra_day_period : data.IntraDayPeriod) -> None:
    message = "Finished sampling : " + ticker_symbol + " -- " + date.get_str_format_2() + " -- " + intra_day_period.value \
              + " -- into " + str(sampling_dollar) + "_dollar tick bars "
    LOGGER.info(message)


# -------- logging functions for data cleaning -------
def log_morning_after_noon_split(tick_df_wrapper : data.TickDataFrame, split_hours : int) -> None:
    message = str(tick_df_wrapper) + " -- data split into morning and afternoon periods -- split point -- " + str(split_hours) + " hours from start of day"
    LOGGER.info(message)


def log_interpolate_zero_bid_ask_prices(tick_df_wrapper : data.TickDataFrame, col_name : str) -> None:
    message = str(tick_df_wrapper) + " -- Interpolate zero values for -- " + col_name
    LOGGER.info(message)


def log_interpolate_outlier_bid_ask_quantities(tick_df_wrapper : data.TickDataFrame, col_name : str, outlier_threshold : int)->None:
    message = str(tick_df_wrapper) + " -- Interpolate outliers for -- " + col_name + " -- threshold : " + str(outlier_threshold)
    LOGGER.info(message)


def log_interpolate_zero_bid_ask_quantities(tick_df_wrapper : data.TickDataFrame, col_name : str) -> None:
    message = str(tick_df_wrapper) + " -- Interpolate zero values for -- " + col_name
    LOGGER.info(message)


def log_interpolate_outlier_bid_ask_prices(tick_df_wrapper : data.TickDataFrame, col_name : str, lower_threshold : float, upper_threshold : float):
    message = str(tick_df_wrapper) + " -- Interpolate outliers for -- " + col_name + " -- lower threshold : " \
              + str(lower_threshold) + " -- upper threshold -- " + str(upper_threshold)
    LOGGER.info(message)


def log_interpolate_outlier_trade_prices(tick_df_wrapper : data.TickDataFrame, lower_threshold : float, upper_threshold : float):
    message = str(tick_df_wrapper) + " -- Interpolate outlier for -- traded price  -- " + " lower threshold : " + str(lower_threshold)\
    + " -- " + str(upper_threshold)
    LOGGER.info(message)


def log_interpolate_outlier_trade_quantities(tick_df_wrapper : data.TickDataFrame, outlier_threshold : int):
    message = str( tick_df_wrapper) + " -- Interpolate outlier for -- traded quantity -- " + " outier_threshold threshold : " + str(outlier_threshold)
    LOGGER.info(message)

# ------ bar data logging -----
def log_interpolate_zeros_bar(bar_df_wrapper : data.BarDataFrame):
    message = str(bar_df_wrapper) + " -- Interpolate zeros  -- "
    LOGGER.info(message)