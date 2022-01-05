import logging
import definitions
import os
import src.data_base_module.data_blocks as data

# ------- initialize a logger for this specific module -------
logger = logging.getLogger("data_base_module")
logger.setLevel(level=logging.INFO)
file_handler = logging.FileHandler(os.path.join(definitions.LOGS_FOLDER_PATH, "data_base.log"))
formatter = logging.Formatter("[%(asctime)s] : %(levelname)s : [%(name)s] : %(message)s ")
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


# ------- tick data logging -----------

def log_raw_tick_data_access(symbol: str, date: data.Date) -> None:
    logger.info("Retrieved raw tick data : " + symbol + " : " + date.get_str_format_2())

def log_clean_tick_data_insertion(symbol : str, date : data.Date, intra_day_period : data.IntraDayPeriod):
    logger.info("Insert clean tick data : " + symbol + " : " + date.get_str_format_2() + " : " + intra_day_period.value)

def log_clean_tick_data_access(symbol : str, date : data.Date, intra_day_period : data.IntraDayPeriod):
    logger.info("Retrieved clean tick data : " + symbol + " : " + date.get_str_format_2() + " : " + intra_day_period.value)

# ------- bar data logging -------
def log_insert_time_bar(time_bar_data_frame : data.TimeBarDataFrame):
    message = "Inserted Time Bar Data Frame : " + str(time_bar_data_frame)
    logger.info(message)

def log_insert_tick_bar(tick_bar_data_frame : data.TickBarDataFrame):
    message = "Inserted Tick Bar Data Frame : " + str(tick_bar_data_frame)
    logger.info(message)

def log_insert_volume_bar(volume_bar_data_frame : data.VolumeBarDataFrame):
    message = "Inserted Volume Bar Data Frame : " + str(volume_bar_data_frame)
    logger.info(message)

def log_insert_dollar_bar(dollar_bar_data_frame : data.DollarBarDataFrame):
    message = "Inserted dollar Bar Data Frame : " + str(dollar_bar_data_frame)
    logger.info(message)

# ------ bar data access -------
def log_retrieved_time_bar(time_bar_data_frame: data.TimeBarDataFrame):
    message = "Retrieved Time Bar Data Frame : " + str(time_bar_data_frame)
    logger.info(message)


def log_retrieved_tick_bar(tick_bar_data_frame: data.TickBarDataFrame):
    message = "Retrieved Tick Bar Data Frame : " + str(tick_bar_data_frame)
    logger.info(message)


def log_retrieved_volume_bar(volume_bar_data_frame: data.VolumeBarDataFrame):
    message = "Retrieved Volume Bar Data Frame : " + str(volume_bar_data_frame)
    logger.info(message)


def log_retrieved_dollar_bar(dollar_bar_data_frame: data.DollarBarDataFrame):
    message = "Retrieved dollar Bar Data Frame : " + str(dollar_bar_data_frame)
    logger.info(message)