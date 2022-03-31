import logging
import definitions
import os
import src.data_base_module.data_blocks as dat_blocks

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

def log_raw_tick_data_access(tick_info : dat_blocks.TickInfo) -> None:
    logger.info(f"Retrieved raw tick data : {tick_info.symbol} : {tick_info.date.get_str_format_2()}")

def log_clean_tick_data_insertion(tick_info : dat_blocks.TickInfo):
    logger.info(f"Insert clean tick data : {tick_info.symbol} : {tick_info.date.get_str_format_2()} : {tick_info.intra_day_period.value()}")

def log_clean_tick_data_access(tick_info : dat_blocks.TickInfo):
    logger.info(f"Retrieved clean tick data : {tick_info.symbol} : {tick_info.date.get_str_format_2()} : {tick_info.intra_day_period.value()}")

# ------- bar data logging -------
def log_insert_bar(bar_info : dat_blocks.BarInfo):
    if bar_info.sampling_type == dat_blocks.Sampling.TIME:
        logger.info("Inserted Bar Data Frame : " + str(bar_info))
    elif bar_info.sampling_type == dat_blocks.Sampling.TICK:
        logger.info("Inserted Tick Bar Data Frame : " + str(bar_info))
    elif bar_info.sampling_type == dat_blocks.Sampling.VOLUME:
        logger.info("Inserted Volume Bar Data Frame : " + str(bar_info))
    elif bar_info.sampling_type == dat_blocks.Sampling.DOLLAR:
        logger.info("Inserted Dollar Bar Data Frame : " + str(bar_info))
    else :
        raise NotImplementedError()

# ------ bar data access -------
def log_retrieved_bar(bar_info : dat_blocks.BarInfo):
    if bar_info.sampling_type == dat_blocks.Sampling.TIME:
        logger.info("Retrieved Time Bar Data Frame : " + str(bar_info))
    elif bar_info.sampling_type == dat_blocks.Sampling.TICK:
        logger.info("Retrieved Tick Bar Data Frame : " + str(bar_info))
    elif bar_info.sampling_type == dat_blocks.Sampling.VOLUME:
        logger.info("Retrieved Volume Bar Data Frame : " + str(bar_info))
    elif bar_info.sampling_type == dat_blocks.Sampling.DOLLAR:
        logger.info("Retrieved dollar Bar Data Frame : " + str(bar_info))
    else :
        raise NotImplementedError()
