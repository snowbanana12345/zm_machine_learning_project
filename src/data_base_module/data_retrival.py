import definitions
import os
import pandas as pd
import src.data_base_module.data_blocks as dat_blocks
import src.data_base_module.data_base_logger as db_logger


class DataBase:
    def __init__(self):
        """ Makeshift database that works with csv files and folders
        Issues :
        -> no way to check what is in the data base
        -> very hard to validate inputs
        -> will simply throw a FileNotFound Exception if attempting to retrieve data that is not in the data base
        """
        self.bar_data_folder_path = os.path.join(definitions.DATA_FOLDER_PATH, "data_storage", "data_bar")
        self.bar_data_wlimit_folder_path = os.path.join(definitions.DATA_FOLDER_PATH, "data_storage", "data_bar_wlimit")
        self.tick_data_folder_path = os.path.join(definitions.DATA_FOLDER_PATH, "data_storage", "data_raw_tick")
        self.tick_clean_data_folder = os.path.join(definitions.DATA_FOLDER_PATH, "data_storage", "data_clean_tick")

    # ------- get raw tick data -------
    def get_raw_tick_data(self, ticker_symbol: str, date: dat_blocks.Date) -> dat_blocks.TickDataFrame:
        """ This function reads tick files in the name format : ModelDepthProto_[date].csv """
        tick_info = dat_blocks.TickInfo(symbol=ticker_symbol, date=date,
                                        intra_day_period=dat_blocks.IntraDayPeriod.WHOLE_DAY)
        file_path = os.path.join(self.tick_data_folder_path, ticker_symbol,
                                 "ModelDepthProto_" + date.get_str() + ".csv")
        db_logger.log_raw_tick_data_access(tick_info=tick_info)
        retrieved_tick_df: pd.DataFrame = pd.read_csv(file_path)
        return dat_blocks.TickDataFrame(tick_df=retrieved_tick_df, tick_info=tick_info)

    # ------ clean tick data file path constructor -----
    def get_clean_tick_file_path(self, tick_info: dat_blocks.TickInfo) -> str:
        """ creates file path to store cleaned tick data in the format [symbol]_[date]_[intra day period]"""
        file_name = tick_info.symbol
        file_name += "_" + tick_info.date.get_str()
        file_name += "_" + tick_info.intra_day_period.value + ".csv"
        return os.path.join(self.tick_clean_data_folder, file_name)

    # ------ get clean tick data ------
    def get_clean_tick_data(self, symbol: str, date: dat_blocks.Date,
                            intra_day_period: dat_blocks.IntraDayPeriod) -> dat_blocks.TickDataFrame:
        """ retrieved stored cleaned tick data and wrap it into a TickDataFrame class """
        tick_info = dat_blocks.TickInfo(symbol=symbol, date=date, intra_day_period=intra_day_period)
        file_path = self.get_clean_tick_file_path(tick_info=tick_info)
        tick_df = pd.read_csv(file_path)
        db_logger.log_clean_tick_data_access(tick_info=tick_info)
        return dat_blocks.TickDataFrame(tick_df=tick_df, tick_info=tick_info)

    # ------ insert clean tick data --------
    def insert_clean_tick_data(self, tick_wrapper: dat_blocks.TickDataFrame) -> None:
        """ Store a cleaned tick data frame """
        file_path = self.get_clean_tick_file_path(tick_info=tick_wrapper.tick_info)
        tick_wrapper.tick_data.to_csv(file_path, index=False)
        db_logger.log_clean_tick_data_insertion(tick_info=tick_wrapper.tick_info)

    # ------- Manager bar data sampled from tick data ---------
    # ----------- file_path generating functions ------------
    def get_sampled_bar_file_path(self, bar_info: dat_blocks.BarInfo):
        file_name = f"{bar_info.date.get_str()}_{bar_info.intra_day_period.value}"
        folder_name = None
        if bar_info.sampling_type == dat_blocks.Sampling.TIME:
            file_name += f"_time_{bar_info.sampling_level}S_sampled.csv"
            folder_name = "time_sampled"
        elif bar_info.sampling_type == dat_blocks.Sampling.VOLUME:
            file_name += f"_volume_{bar_info.sampling_level}_sampled.csv"
            folder_name = "volume_sampled"
        elif bar_info.sampling_type == dat_blocks.Sampling.TICK:
            file_name += f"_tick_{bar_info.sampling_level}_sampled.csv"
            folder_name = "tick_sampled"
        elif bar_info.sampling_type == dat_blocks.Sampling.DOLLAR:
            file_name += f"_dollar_{bar_info.sampling_level}_sampled.csv"
            folder_name = "dollar_sampled"
        return os.path.join(self.bar_data_folder_path, bar_info.symbol, folder_name, file_name)

    # ----------- get bar data functions --------
    def get_sampled_bar(self, symbol: str, date: dat_blocks.Date, intra_day_period: dat_blocks.IntraDayPeriod,
                        sampling_level: int, sampling_type: dat_blocks.Sampling):
        bar_info = dat_blocks.BarInfo(symbol=symbol, date=date, intra_day_period=intra_day_period,
                                      sampling_level=sampling_level, sampling_type=sampling_type)
        file_path = self.get_sampled_bar_file_path(bar_info=bar_info)
        bar_df = pd.read_csv(file_path)
        db_logger.log_retrieved_bar(bar_info=bar_info)
        return dat_blocks.BarDataFrame(bar_data=bar_df, bar_info=bar_info)

    # ------ insert bar data functions ------
    def insert_sampled_bar(self, bar_wrapper: dat_blocks.BarDataFrame) -> None:
        file_path = self.get_sampled_bar_file_path(bar_wrapper.bar_info)
        bar_wrapper.bar_data.to_csv(file_path, index=False)
        db_logger.log_insert_bar(bar_info=bar_wrapper.bar_info)


instance: DataBase = DataBase()
