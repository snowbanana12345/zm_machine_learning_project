import definitions
import os
import pandas as pd
import src.data_base_module.data_blocks as data
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
    def get_raw_tick_data(self, ticker_symbol : str, date : data.Date) -> data.TickDataFrame :
        """ This function reads tick files in the name format : ModelDepthProto_[date].csv """
        file_path = os.path.join(self.tick_data_folder_path, ticker_symbol , "ModelDepthProto_" + date.get_str() + ".csv")
        db_logger.log_raw_tick_data_access(ticker_symbol, date)
        retrieved_tick_df : pd.DataFrame = pd.read_csv(file_path)
        return data.TickDataFrame(tick_df = retrieved_tick_df, symbol = ticker_symbol,  intra_day_period = data.IntraDayPeriod.WHOLE_DAY,
                                  date = date)

    # ------ clean tick data file path constructor -----
    def get_clean_tick_file_path(self, symbol : str, date : data.Date, intra_day_period : data.IntraDayPeriod) -> str:
        """ creates file path to store cleaned tick data in the format [symbol]_[date]_[intra day period]"""
        file_name = symbol
        file_name += "_" + date.get_str()
        file_name += "_" + intra_day_period.value + ".csv"
        return os.path.join(self.tick_clean_data_folder, file_name)

    # ------ get clean tick data ------
    def get_clean_tick_data(self, symbol : str, date : data.Date, intra_day_period : data.IntraDayPeriod) -> data.TickDataFrame:
        """ retrieved stored cleaned tick data and wrap it into a TickDataFrame class """
        file_path = self.get_clean_tick_file_path(symbol = symbol, date = date, intra_day_period = intra_day_period)
        tick_df = pd.read_csv(file_path)
        tick_df_wrapper = data.TickDataFrame(tick_df = tick_df, symbol = symbol, date = date, intra_day_period = intra_day_period)
        db_logger.log_clean_tick_data_access(symbol = symbol, date = date, intra_day_period = intra_day_period)
        return tick_df_wrapper

    # ------ insert clean tick data --------
    def insert_clean_tick_data(self, tick_df_wrapper : data.TickDataFrame) -> None:
        """ Store a cleaned tick data frame """
        file_path = self.get_clean_tick_file_path(symbol = tick_df_wrapper.symbol, date = tick_df_wrapper.date, intra_day_period = tick_df_wrapper.intra_day_period)
        tick_df_wrapper.get_tick_data().to_csv(file_path, index = False)
        db_logger.log_clean_tick_data_insertion(symbol = tick_df_wrapper.symbol, date = tick_df_wrapper.date, intra_day_period = tick_df_wrapper.intra_day_period)


    # ------- Manager bar data sampled from tick data ---------
    # ----------- file_path generating functions ------------
    def get_sampled_time_bar_file_path(self, symbol : str, date : data.Date, intra_day_period : data.IntraDayPeriod, sampling_seconds : int) -> str:
        """ create a file path in the format : [date]_[intra_day_period]_time_[sampling seconds] with [symbol] as the folder"""
        file_name = date.get_str() + "_" + intra_day_period.value + "_time_" + str(sampling_seconds) + "S_sampled.csv"
        return os.path.join(self.bar_data_folder_path, symbol, "time_sampled", file_name)

    def get_sampled_tick_bar_file_path(self, symbol : str, date : data.Date, intra_day_period : data.IntraDayPeriod, sampling_ticks : int) -> str:
        """ create a file path in the format : [date]_[intra day period]_tick_[sampling ticks] with [symbol] as the folder"""
        file_name = date.get_str() + "_" + intra_day_period.value + "_tick_" + str(sampling_ticks) + "_sampled.csv"
        return os.path.join(self.bar_data_folder_path, symbol, "tick_sampled", file_name)

    def get_sampled_volume_bar_file_path(self, symbol : str, date : data.Date, intra_day_period : data.IntraDayPeriod, sampling_volume : int) -> str:
        """ create a file path in the format : [date]_[intra day period]_volume_[sampling volume] with [symbol] as the folder"""
        file_name = date.get_str() + "_" + intra_day_period.value + "_" + "volume" + "_" + str(sampling_volume) + "_sampled.csv"
        return os.path.join(self.bar_data_folder_path, symbol, "volume_sampled", file_name)

    def get_sampled_dollar_bar_file_path(self, symbol : str, date : data.Date, intra_day_period : data.IntraDayPeriod, sampling_dollar : int) -> str:
        """ create a file path in the format : [date]_[intra day_period]_dollar_[sampling_dollar] with [symbol] as the folder"""
        file_name = date.get_str() + "_" + intra_day_period.value + "_" + "dollar" + "_" + str(sampling_dollar) + "_sampled.csv"
        return os.path.join(self.bar_data_folder_path, symbol, "dollar_sampled", file_name)

    # ----------- get bar data functions --------
    def get_sampled_time_bar(self, symbol : str, date : data.Date, intra_day_period : data.IntraDayPeriod, sampling_seconds : int) -> data.TimeBarDataFrame:
        """ retrieve a time sampled bar data with the given inputs """
        file_path = self.get_sampled_time_bar_file_path(symbol, date, intra_day_period, sampling_seconds)
        time_bar_df = pd.read_csv(file_path)
        time_bar_wrapper = data.TimeBarDataFrame(bar_df = time_bar_df,
                                     sampling_seconds = sampling_seconds,
                                     date = date,
                                     intra_day_period = intra_day_period,
                                     symbol = symbol)
        db_logger.log_retrieved_time_bar(time_bar_wrapper)
        return time_bar_wrapper

    def get_sampled_tick_bar(self, symbol : str, date : data.Date, intra_day_period : data.IntraDayPeriod, sampling_ticks : int) -> data.TickBarDataFrame:
        """ retrieve a tick sampled bar data with the given inputs """
        file_path = self.get_sampled_tick_bar_file_path(symbol, date, intra_day_period, sampling_ticks)
        tick_bar_df = pd.read_csv(file_path)
        tick_bar_wrapper = data.TickBarDataFrame(bar_df = tick_bar_df,
                                     sampling_ticks = sampling_ticks,
                                     date = date,
                                     intra_day_period = intra_day_period,
                                     symbol = symbol)
        db_logger.log_retrieved_tick_bar(tick_bar_wrapper)
        return tick_bar_wrapper

    def get_sampled_volume_bar(self, symbol : str, date : data.Date, intra_day_period : data.IntraDayPeriod, sampling_volume : int) -> data.VolumeBarDataFrame:
        """ retrieve a volume sampled bar data with the given inputs """
        file_path = self.get_sampled_volume_bar_file_path(symbol, date, intra_day_period, sampling_volume)
        volume_bar_df = pd.read_csv(file_path)
        volume_bar_wrapper = data.VolumeBarDataFrame(bar_df = volume_bar_df,
                                       sampling_volume = sampling_volume,
                                       date = date,
                                       intra_day_period = intra_day_period,
                                       symbol = symbol)
        db_logger.log_retrieved_volume_bar(volume_bar_wrapper)
        return volume_bar_wrapper

    def get_sampled_dollar_bar(self, symbol : str, date : data.Date, intra_day_period : data.IntraDayPeriod, sampling_dollar : int) -> data.DollarBarDataFrame:
        """ retrieve a dollar sampled bar data with the given inputs """
        file_path = self.get_sampled_dollar_bar_file_path(symbol, date, intra_day_period, sampling_dollar)
        dollar_bar_df = pd.read_csv(file_path)
        dollar_bar_wrapper = data.DollarBarDataFrame(bar_df = dollar_bar_df,
                                       sampling_dollar = sampling_dollar,
                                       date = date,
                                       intra_day_period = intra_day_period,
                                       symbol = symbol)
        db_logger.log_retrieved_dollar_bar(dollar_bar_wrapper)
        return dollar_bar_wrapper

    # ------ insert bar data functions ------
    def insert_sampled_time_bar(self, time_bar_data_frame : data.TimeBarDataFrame) -> None:
        """ stores time bar data """
        sampling_seconds : int = time_bar_data_frame.sampling_seconds
        symbol : str = time_bar_data_frame.symbol
        date : data.Date = time_bar_data_frame.date
        intra_day_period : data.IntraDayPeriod = time_bar_data_frame.intra_day_period
        file_path = self.get_sampled_time_bar_file_path(symbol, date, intra_day_period, sampling_seconds)
        time_bar_data_frame.get_bar_data_reference().to_csv(file_path, index = False)
        db_logger.log_insert_time_bar(time_bar_data_frame = time_bar_data_frame)

    def insert_sampled_tick_bar(self, tick_bar_data_frame : data.TickBarDataFrame) -> None:
        """ stores tick bar data """
        sampling_ticks : int = tick_bar_data_frame.sampling_ticks
        symbol : str = tick_bar_data_frame.symbol
        date : data.Date = tick_bar_data_frame.date
        intra_day_period : data.IntraDayPeriod = tick_bar_data_frame.intra_day_period
        file_path = self.get_sampled_tick_bar_file_path(symbol, date, intra_day_period, sampling_ticks)
        tick_bar_data_frame.get_bar_data_reference().to_csv(file_path, index = False)
        db_logger.log_insert_tick_bar(tick_bar_data_frame = tick_bar_data_frame)

    def insert_sampled_volume_bar(self, volume_bar_data_frame : data.VolumeBarDataFrame) -> None:
        """ stores volume bar data """
        sampling_volume : int = volume_bar_data_frame.sampling_volume
        symbol : str = volume_bar_data_frame.symbol
        date : data.Date = volume_bar_data_frame.date
        intra_day_period : data.IntraDayPeriod = volume_bar_data_frame.intra_day_period
        file_path = self.get_sampled_volume_bar_file_path(symbol, date, intra_day_period, sampling_volume)
        volume_bar_data_frame.get_bar_data_reference().to_csv(file_path, index = False)
        db_logger.log_insert_volume_bar(volume_bar_data_frame = volume_bar_data_frame)

    def insert_sampled_dollar_bar(self, dollar_bar_data_frame : data.DollarBarDataFrame) -> None:
        """ stores dollar bar data """
        sampling_dollar : int = dollar_bar_data_frame.sampling_dollar
        symbol : str = dollar_bar_data_frame.symbol
        date : data.Date = dollar_bar_data_frame.date
        intra_day_period : data.IntraDayPeriod = dollar_bar_data_frame.intra_day_period
        file_path = self.get_sampled_dollar_bar_file_path(symbol, date, intra_day_period, sampling_dollar)
        dollar_bar_data_frame.get_bar_data_reference().to_csv(file_path, index = False)
        db_logger.log_insert_dollar_bar(dollar_bar_data_frame = dollar_bar_data_frame)


instance = DataBase()


