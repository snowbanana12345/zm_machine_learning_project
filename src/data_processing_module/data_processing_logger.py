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
def log_time_sampling_start(tick_info : data.TickInfo, sampling_seconds: int) -> None:
    message = f"Started sampling : {tick_info.symbol}  -- {tick_info.date.get_str_format_2()} -- {tick_info.intra_day_period.value} -- into {sampling_seconds}S time bars "
    LOGGER.info(message)

def log_time_sampling_end(tick_info : data.TickInfo, sampling_seconds: int) -> None:
    message = f"Finished sampling : {tick_info.symbol} -- {tick_info.date.get_str_format_2()} -- {tick_info.intra_day_period.value} \
               -- into {sampling_seconds}S time bars "
    LOGGER.info(message)

def log_volume_sampling_start(tick_info : data.TickInfo, sampling_volume : int) -> None:
    message = f"Started sampling : {tick_info.symbol} -- {tick_info.date.get_str_format_2()} -- {tick_info.intra_day_period.value} \
               -- into {sampling_volume} volume bars "
    LOGGER.info(message)

def log_volume_sampling_end(tick_info : data.TickInfo, sampling_volume : int) -> None:
    message = f"Finished sampling : {tick_info.symbol} -- {tick_info.date.get_str_format_2()} -- {tick_info.intra_day_period.value} \
               -- into {sampling_volume} volume bars "
    LOGGER.info(message)

def log_tick_sampling_start(tick_info : data.TickInfo, sampling_ticks : int) -> None:
    message = f"Started sampling : {tick_info.symbol} -- {tick_info.date.get_str_format_2()} -- {tick_info.intra_day_period.value}-- into {sampling_ticks} tick tick bars "
    LOGGER.info(message)


def log_tick_sampling_end(tick_info : data.TickInfo, sampling_ticks : int) -> None:
    message = f"Finished sampling : {tick_info.symbol} -- {tick_info.date.get_str_format_2()} -- {tick_info.intra_day_period.value} \
              -- into {sampling_ticks}_tick tick bars "
    LOGGER.info(message)

def log_dollar_sampling_start(tick_info : data.TickInfo, sampling_dollar : int) -> None:
    message = f"Started sampling : {tick_info.symbol} -- {tick_info.date.get_str_format_2()} -- {tick_info.intra_day_period.value} -- into {sampling_dollar}_dollar tick bars "
    LOGGER.info(message)

def log_dollar_sampling_end(tick_info : data.TickInfo, sampling_dollar : int) -> None:
    message = f"Finished sampling : {tick_info.symbol} -- {tick_info.date.get_str_format_2()} -- {tick_info.intra_day_period.value} -- into {sampling_dollar}_dollar tick bars "
    LOGGER.info(message)

# -------- logging functions for data cleaning -------
def log_morning_after_noon_split(tick_wrapper : data.TickDataFrame, split_hours : int) -> None:
    message = f"{tick_wrapper} -- data split into morning and afternoon periods -- split point -- {split_hours} hours from start of day"
    LOGGER.info(message)

def log_interpolate_zero_bid_ask_prices(tick_wrapper : data.TickDataFrame, col_name : str) -> None:
    message = f"{tick_wrapper} -- Interpolate zero values for -- {col_name}"
    LOGGER.info(message)

def log_interpolate_outlier_bid_ask_quantities(tick_wrapper : data.TickDataFrame, col_name : str, outlier_threshold : int)->None:
    message = f"{tick_wrapper} -- Interpolate outliers for -- {col_name} -- threshold : {outlier_threshold}"
    LOGGER.info(message)

def log_interpolate_zero_bid_ask_quantities(tick_wrapper : data.TickDataFrame, col_name : str) -> None:
    message = f"{tick_wrapper} -- Interpolate zero values for -- {col_name}"
    LOGGER.info(message)

def log_interpolate_outlier_bid_ask_prices(tick_wrapper : data.TickDataFrame, col_name : str, lower_threshold : float, upper_threshold : float):
    message = f"{tick_wrapper} -- Interpolate outliers for -- {col_name} -- lower threshold : {lower_threshold} -- upper threshold -- {upper_threshold}"
    LOGGER.info(message)

def log_interpolate_outlier_trade_prices(tick_wrapper : data.TickDataFrame, lower_threshold : float, upper_threshold : float):
    message = f"{tick_wrapper} -- Interpolate outlier for -- traded price  -- lower threshold :  {lower_threshold} -- {upper_threshold}"
    LOGGER.info(message)

def log_interpolate_outlier_trade_quantities(tick_wrapper : data.TickDataFrame, outlier_threshold : int):
    message = f"{tick_wrapper} -- Interpolate outlier for -- traded quantity -- outier_threshold threshold : {outlier_threshold}"
    LOGGER.info(message)

# ------ bar data logging -----
def log_interpolate_zeros_bar(bar_wrapper : data.BarDataFrame):
    message = f"{bar_wrapper} -- Interpolate zeros  -- "
    LOGGER.info(message)