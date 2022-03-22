from enum import Enum
import pandas as pd
from dataclasses import dataclass


# ------- date class --------
def front_pad_zeros(input_str : str, total_length : int) -> str:
    """ pads zeros infront of the input_str such that the length of the resulting string equals total_length"""
    pads = total_length - len(input_str)
    if pads > 0:
        return "0" * pads + input_str
    else :
        return input_str

@dataclass(frozen=True)
class Date:
    """
    Custom date class
    """
    day : int
    month : int
    year : int

    def __post_init__(self):
        if not 1 <= self.day <= 31:
            raise InvalidDateException(self.day, self.month, self.year)
        if not 1 <= self.month <= 12:
            raise InvalidDateException(self.day, self.month, self.year)
        if not 1 <= self.year <= 9999:
            raise InvalidDateException(self.day, self.month, self.year)

    def get_day(self):
        return self.day

    def get_month(self):
        return self.month

    def get_year(self):
        return self.year

    def get_str(self) -> str:
        """ standardized format for string is YYYYMMDD"""
        year_str = front_pad_zeros(str(self.year), 4)
        month_str = front_pad_zeros(str(self.month), 2)
        day_str = front_pad_zeros(str(self.day), 2)
        return year_str + month_str + day_str

    def get_str_format_2(self):
        """ get string in the format YYYY-MM-DD"""
        year_str = front_pad_zeros(str(self.year), 4)
        month_str = front_pad_zeros(str(self.month), 2)
        day_str = front_pad_zeros(str(self.day), 2)
        return year_str + "-" + month_str + "-" + day_str

    def __eq__(self, other):
        return (self.day == other.day) and (self.month == other.month) and (self.year == other.year)

# ------- Enums -------
class Sampling(Enum):
    TIME = "time_sampled"
    TICK = "tick_sampled"
    VOLUME = "volume_sampled"
    DOLLAR = "dollar_sampled"


class BarDataColumns(Enum):
    OPEN  = "open"
    CLOSE  = "close"
    HIGH  = "high"
    LOW  = "low"
    VOLUME  = "volume"
    VWAP = "VWAP"
    TIMESTAMP  = "timestamp"


class IntraDayPeriod(Enum):
    MORNING = "morning"
    AFTERNOON = "afternoon"
    MIDNIGHT = "midnight"
    WHOLE_DAY = "whole_day"


class TickDataColumns(Enum):
    TIMESTAMP_NANO = "timestampNano"
    LAST_PRICE = "lastPrice"
    LAST_QUANTITY = "lastQty"
    ASK1P = "ask1p"
    ASK2P = "ask2p"
    ASK3P = "ask3p"
    ASK4P = "ask4p"
    ASK5P = "ask5p"
    ASK1Q = "ask1q"
    ASK2Q = "ask2q"
    ASK3Q = "ask3q"
    ASK4Q = "ask4q"
    ASK5Q = "ask5q"
    BID1P = "bid1p"
    BID2P = "bid2p"
    BID3P = "bid3p"
    BID4P = "bid4p"
    BID5P = "bid5p"
    BID1Q = "bid1q"
    BID2Q = "bid2q"
    BID3Q = "bid3q"
    BID4Q = "bid4q"
    BID5Q = "bid5q"


# ------- bar data frames -------
""" The purpose of bar data frame classes is to encapsulate a pandas data frame containing bar data 
-> ensure that the wrapped data frame contains certain columns 
-> contain meta data about the bar data , sampling level, date, ticker symbol etc. """


class BarDataFrame:
    REQUIRED_COLUMNS = [BarDataColumns.OPEN,
                        BarDataColumns.CLOSE,
                        BarDataColumns.HIGH,
                        BarDataColumns.LOW,
                        BarDataColumns.TIMESTAMP,
                        BarDataColumns.VOLUME,
                        BarDataColumns.VWAP]
    """ Abstract class """
    def __init__(self, symbol: str):
        self.bar_data : pd.DataFrame = pd.DataFrame()
        self.symbol : str = symbol
        self.print_str : str =  "Bar OCHLV -- " + self.symbol

    def get_column(self, col_name : BarDataColumns) -> pd.Series:
        return self.bar_data[col_name.value]

    def create_empty_bar_df(self) -> pd.DataFrame:
        return pd.DataFrame(columns = [col.value for col in BarDataFrame.REQUIRED_COLUMNS])

    def create_empty_copy(self):
        """ creates an empty data frame with a copy of the meta data this class holds """
        return BarDataFrame(self.symbol)

    def set_data_frame(self, bar_df : pd.DataFrame, deep_copy : bool = False) -> None:
        self.bar_data = bar_df

    def get_bar_data_reference(self) -> pd.DataFrame:
        return self.bar_data

    def get_bar_data_copy(self) -> pd.DataFrame:
        """ returns a copy of the bar data frame that the class is wrapping
        copying instead of referencing is done to avoid setting with copy errors """
        return self.bar_data.copy(deep=True)

    def __str__(self) -> str:
        return f"Bar OCHLV -- {self.symbol}"

    def __len__(self) -> int:
        return len(self.bar_data)


class TimeBarDataFrame(BarDataFrame):
    """
    Wrapper for a pandas data frame that contains specifically time sampled intraday bar data
    Also contains meta data : sampling_seconds, intra_day_period, date
    """
    def __init__(self, bar_df: pd.DataFrame,
                 sampling_seconds: int,
                 date: Date,
                 intra_day_period: IntraDayPeriod,
                 symbol: str,
                 deep_copy=False):
        """
        creates a copy of the original bar data frame to avoid setting with copy errors
        :param bar_df: The bar data frame retrieved or sampled
        :param sampling_seconds: the level of sampling that was used
        :param date : the date, what else do you want?
        :param intra_day_period : morning afternoon or midnight or perhaps the whole day
        :param deep_copy: if set to true, the wrapped data frame will be a copy of the data frame used to initialize the object
        else, it will be a reference.
        Warning: This class will convert the input the dataframe to ranged indexing which may cause issues if object holds a reference instead of a copy
        """
        # ----- meta data -------
        super().__init__(symbol)
        self.sampling_seconds: int = sampling_seconds
        self.date: Date = date
        self.intra_day_period: IntraDayPeriod = intra_day_period
        self.set_data_frame(bar_df = bar_df, deep_copy = deep_copy)

    def set_data_frame(self, bar_df : pd.DataFrame, deep_copy: bool = False) -> None:
        # ----- check if the necessary data columns are found in the data frame ------
        for required_column in BarDataFrame.REQUIRED_COLUMNS:
            if required_column.value not in bar_df.columns:
                raise SamplingRequiredColumnNotFound(required_column, Sampling.TIME)
        # ----- do deep copying if specified to be true ------
        # ----- makes sure only required columns are stored in the data frame -------
        if deep_copy:
            self.bar_data = bar_df.loc[:, [column.value for column in BarDataFrame.REQUIRED_COLUMNS]].copy(deep=deep_copy)
        else:
            self.bar_data = bar_df.loc[:, [column.value for column in BarDataFrame.REQUIRED_COLUMNS]]
        # ----- enforce ranged indexing for data frame ------
        self.bar_data.index = pd.RangeIndex(len(self.bar_data))

    def create_empty_copy(self):
        return TimeBarDataFrame(bar_df = self.create_empty_bar_df(), symbol = self.symbol, sampling_seconds = self.sampling_seconds,
                                date = self.date, intra_day_period = self.intra_day_period)

    def __str__(self):
        return f"Time sampled bar -- {self.symbol} -- {self.date.get_str_format_2()} -- {self.intra_day_period.value} -- sampling seconds : {self.sampling_seconds}"

    def __eq__(self, other):
        """ NOTE : this equality condition fails when comparing floats with int even if the values are the same """
        if not isinstance(other, TimeBarDataFrame):
            return False
        data_frame_eq = self.bar_data.equals(other.bar_data)
        date_eq = self.date == other.date
        symbol_eq = self.symbol == other.symbol
        sampling_eq = self.sampling_seconds == other.sampling_seconds
        intraday_eq = self.intra_day_period == other.intra_day_period
        return data_frame_eq and date_eq and symbol_eq and intraday_eq and sampling_eq

class TickBarDataFrame(BarDataFrame):
    """
    Wrapper for a pandas data frame that contains specifically intraday tick sampled bar data
    Also contains meta data : sampling_ticks, intra_day_period, date
    """

    def __init__(self, bar_df: pd.DataFrame,
                 sampling_ticks: int,
                 date: Date,
                 intra_day_period: IntraDayPeriod,
                 symbol: str,
                 deep_copy=False):
        """
        creates a copy of the original bar data frame to avoid setting with copy errors
        :param bar_df: The bar data frame retrieved or sampled
        :param sampling_ticks: the level of sampling that was used
        :param date : the date, what else do you want?
        :param intra_day_period : morning afternoon or midnight or perhaps the whole day
        :param deep_copy: if set to true, the wrapped data frame will be a copy of the data frame used to initialize the object
        else, it will be a reference.
        Warning: This class will convert the input the dataframe to ranged indexing which may cause issues if object holds a reference instead of a copy
        """
        # ----- meta data -------
        super().__init__(symbol)
        self.sampling_ticks: int = sampling_ticks
        self.date: Date = date
        self.intra_day_period: IntraDayPeriod = intra_day_period
        self.set_data_frame(bar_df = bar_df, deep_copy = deep_copy)

    def set_data_frame(self, bar_df : pd.DataFrame, deep_copy : bool  = False):
        # ----- check if the necessary data columns are found in the data frame ------
        for required_column in BarDataFrame.REQUIRED_COLUMNS:
            if required_column.value not in bar_df.columns:
                raise SamplingRequiredColumnNotFound(required_column, Sampling.TICK)
        # ----- do deep copying if specified to be true ------
        # ----- makes sure only required columns are stored in the data frame -------
        if deep_copy:
            self.bar_data = bar_df.loc[:, [column.value for column in BarDataFrame.REQUIRED_COLUMNS]].copy(deep=deep_copy)
        else:
            self.bar_data = bar_df.loc[:, [column.value for column in BarDataFrame.REQUIRED_COLUMNS]]
        # ----- enforce ranged indexing for data frame ------
        self.bar_data.index = pd.RangeIndex(len(self.bar_data))

    def create_empty_copy(self):
        return TickBarDataFrame(bar_df = self.create_empty_bar_df(), symbol = self.symbol, sampling_ticks = self.sampling_ticks,
                                date = self.date, intra_day_period = self.intra_day_period)

    def __str__(self):
        return f"Tick sampled bar -- {self.symbol} --  {self.date.get_str_format_2()}  -- {self.intra_day_period.value}  -- sampling ticks : {self.sampling_ticks}"

    def __eq__(self, other):
        """ NOTE : this equality condition fails when comparing floats with int even if the values are the same """
        if not isinstance(other, TickBarDataFrame):
            return False
        data_frame_eq = self.bar_data.equals(other.bar_data)
        date_eq = self.date == other.date
        symbol_eq = self.symbol == other.symbol
        sampling_eq = self.sampling_ticks == other.sampling_ticks
        intraday_eq = self.intra_day_period == other.intra_day_period
        return data_frame_eq and date_eq and symbol_eq and intraday_eq and sampling_eq

class VolumeBarDataFrame(BarDataFrame):
    """
    Wrapper for a pandas data frame that contains specifically intraday volume sampled bar data
    Also contains meta data : sampling_volume, intra_day_period, date
    """
    def __init__(self, bar_df: pd.DataFrame,
                 sampling_volume: int,
                 date: Date,
                 intra_day_period: IntraDayPeriod,
                 symbol: str,
                 deep_copy=False):
        """
        creates a copy of the original bar data frame to avoid setting with copy errors
        :param bar_df: The bar data frame retrieved or sampled
        :param sampling_volume: the level of sampling that was used
        :param date : the date, what else do you want?
        :param intra_day_period : morning afternoon or midnight or perhaps the whole day
        :param deep_copy: if set to true, the wrapped data frame will be a copy of the data frame used to initialize the object
        else, it will be a reference.
        Warning: This class will convert the input the dataframe to ranged indexing which may cause issues if object holds a reference instead of a copy
        """
        # ----- meta data -------
        super().__init__(symbol)
        self.sampling_volume: int = sampling_volume
        self.date: Date = date
        self.intra_day_period: IntraDayPeriod = intra_day_period
        self.set_data_frame(bar_df = bar_df, deep_copy = deep_copy)

    def set_data_frame(self, bar_df : pd.DataFrame, deep_copy : bool = False) -> None:
        # ----- check if the necessary data columns are found in the data frame ------
        for required_column in VolumeBarDataFrame.REQUIRED_COLUMNS:
            if required_column.value not in bar_df.columns:
                raise SamplingRequiredColumnNotFound(required_column, Sampling.VOLUME)
        # ----- do deep copying if specified to be true ------
        # ----- makes sure only required columns are stored in the data frame -------
        if deep_copy:
            self.bar_data = bar_df.loc[:, [column.value for column in VolumeBarDataFrame.REQUIRED_COLUMNS]].copy(deep=deep_copy)
        else:
            self.bar_data = bar_df.loc[:, [column.value for column in VolumeBarDataFrame.REQUIRED_COLUMNS]]
        # ----- enforce ranged indexing for data frame ------
        self.bar_data.index = pd.RangeIndex(len(self.bar_data))

    def create_empty_copy(self):
        return VolumeBarDataFrame(bar_df = self.create_empty_bar_df(), symbol = self.symbol, sampling_volume = self.sampling_volume,
                                date = self.date, intra_day_period = self.intra_day_period)

    def __str__(self):
        return f"Volume sampled bar -- {self.symbol} -- {self.date.get_str_format_2()} -- {self.intra_day_period.value} -- sampling volume : {self.sampling_volume}"

    def __eq__(self, other):
        """ NOTE : this equality condition fails when comparing floats with int even if the values are the same """
        if not isinstance(other, VolumeBarDataFrame):
            return False
        data_frame_eq = self.bar_data.equals(other.bar_data)
        date_eq = self.date == other.date
        symbol_eq = self.symbol == other.symbol
        sampling_eq = self.sampling_volume == other.sampling_volume
        intraday_eq = self.intra_day_period == other.intra_day_period
        return data_frame_eq and date_eq and symbol_eq and intraday_eq and sampling_eq

class DollarBarDataFrame(BarDataFrame):
    """
    Wrapper for a pandas data frame that contains specifically intraday dollar sampled bar data
    Also contains meta data : sampling_dollar
    """
    def __init__(self, bar_df: pd.DataFrame,
                 sampling_dollar: int,
                 date: Date,
                 intra_day_period: IntraDayPeriod,
                 symbol: str,
                 deep_copy=False):
        """
        creates a copy of the original bar data frame to avoid setting with copy errors
        :param bar_df: The bar data frame retrieved or sampled
        :param sampling_dollar: the level of sampling that was used
        :param date : the date, what else do you want?
        :param intra_day_period : morning afternoon or midnight or perhaps the whole day
        :param deep_copy: if set to true, the wrapped data frame will be a copy of the data frame used to initialize the object
        else, it will be a reference.
        Warning: This class will convert the input the dataframe to ranged indexing which may cause issues if object holds a reference instead of a copy
        """
        # ----- meta data -------
        super().__init__(symbol)
        self.sampling_dollar: int = sampling_dollar
        self.date: Date = date
        self.intra_day_period: IntraDayPeriod = intra_day_period
        self.set_data_frame(bar_df = bar_df, deep_copy = deep_copy)


    def set_data_frame(self, bar_df : pd.DataFrame, deep_copy : bool = False):
        # ----- check if the necessary data columns are found in the data frame ------
        for required_column in BarDataFrame.REQUIRED_COLUMNS:
            if required_column.value not in bar_df.columns:
                raise SamplingRequiredColumnNotFound(required_column, Sampling.VOLUME)
        # ----- do deep copying if specified to be true ------
        # ----- makes sure only required columns are stored in the data frame -------
        if deep_copy:
            self.bar_data = bar_df.loc[:, [column.value for column in BarDataFrame.REQUIRED_COLUMNS]].copy(deep=deep_copy)
        else:
            self.bar_data = bar_df.loc[:, [column.value for column in BarDataFrame.REQUIRED_COLUMNS]]
        # ----- enforce ranged indexing for data frame ------
        self.bar_data.index = pd.RangeIndex(len(self.bar_data))

    def create_empty_copy(self):
        return DollarBarDataFrame(bar_df = self.create_empty_bar_df(), symbol = self.symbol, sampling_dollar = self.sampling_dollar,
                                date = self.date, intra_day_period = self.intra_day_period)

    def __str__(self):
        return f"Dollar sampled bar -- {self.symbol} -- {self.date.get_str_format_2()}  -- {self.intra_day_period.value} --  sampling dollars : {self.sampling_dollar}"

    def __eq__(self, other):
        """ NOTE : this equality condition fails when comparing floats with int even if the values are the same """
        if not isinstance(other, DollarBarDataFrame):
            return False
        data_frame_eq = self.bar_data.equals(other.bar_data)
        date_eq = self.date == other.date
        symbol_eq = self.symbol == other.symbol
        sampling_eq = self.sampling_dollar == other.sampling_dollar
        intraday_eq = self.intra_day_period == other.intra_day_period
        return data_frame_eq and date_eq and symbol_eq and intraday_eq and sampling_eq


# -------- Tick data frame --------
class TickDataFrame:
    """
    Wrapper for a pandas data frame that contains tick data for one intraday period
    Contains meta data : date, intraday period, ticker symbol
    A tick data frame requires and enforces the following columns
    KEY_SYMBOL TIMESTAMP_NANO LAST_PRICE LAST_QUANTITY ASK1P ASK2P ASK3P ASK4P ASK5P
    ASK1Q ASK2Q ASK3Q ASK4Q ASK5Q BID1P BID2P BID3P BID4P BID5P BID1Q BID2Q BID3Q BID4Q BID5Q
    Enforces ranged indexing for the underlying dataframe
    """
    REQUIRED_COLUMNS = [
        TickDataColumns.TIMESTAMP_NANO,
        TickDataColumns.LAST_PRICE,
        TickDataColumns.LAST_QUANTITY,
        TickDataColumns.ASK1P,
        TickDataColumns.ASK2P,
        TickDataColumns.ASK3P,
        TickDataColumns.ASK4P,
        TickDataColumns.ASK5P,
        TickDataColumns.ASK1Q,
        TickDataColumns.ASK2Q,
        TickDataColumns.ASK3Q,
        TickDataColumns.ASK4Q,
        TickDataColumns.ASK5Q,
        TickDataColumns.BID1P,
        TickDataColumns.BID2P,
        TickDataColumns.BID3P,
        TickDataColumns.BID4P,
        TickDataColumns.BID5P,
        TickDataColumns.BID1Q,
        TickDataColumns.BID2Q,
        TickDataColumns.BID3Q,
        TickDataColumns.BID4Q,
        TickDataColumns.BID5Q]

    def __init__(self, tick_df: pd.DataFrame,
                 date: Date,
                 intra_day_period: IntraDayPeriod,
                 symbol: str):
        # ----- meta data -------
        self.date: Date = date
        self.intra_day_period: IntraDayPeriod = intra_day_period
        self.symbol : str = symbol
        # ----- check if the necessary data columns are found in the data frame ------
        for required_column in TickDataFrame.REQUIRED_COLUMNS:
            if required_column.value not in tick_df.columns:
                raise TickRequiredColumnNotFound(required_column)
        # ----- makes sure only required columns are stored in the data frame -------
        self.tick_data : pd.DataFrame = tick_df.loc[:, [column.value for column in TickDataFrame.REQUIRED_COLUMNS]]
        # ----- enforce ranged indexing for data frame ------
        self.tick_data.index = pd.RangeIndex(len(self.tick_data))

    def get_tick_data(self) -> pd.DataFrame:
        """ return a reference to tick data frame """
        return self.tick_data

    def reset_all_columns_to_range_index(self):
        for col in self.tick_data.columns:
            self.tick_data[col].index = pd.RangeIndex(len(self.tick_data))

    def __str__(self):
        return self.symbol + " -- " + self.date.get_str_format_2() + " -- " + self.intra_day_period.value

    def __eq__(self, other) -> bool:
        """ NOTE : this equality condition fails when comparing floats with int even if the values are the same """
        if not isinstance(other, TickDataFrame):
            return False
        data_frame_eq = self.tick_data.equals(other.tick_data)
        date_eq = self.date == other.date
        symbol_eq = self.symbol == other.symbol
        intra_day_eq = self.intra_day_period == other.intra_day_period
        return data_frame_eq and date_eq and symbol_eq and intra_day_eq

# -------- custom exceptions ---------
class TickRequiredColumnNotFound(Exception):
    def __init__(self, missing_column : TickDataColumns):
        message = "Required column missing : " + missing_column.value
        super().__init__(message)

class SamplingRequiredColumnNotFound(Exception):
    def __init__(self, missing_column: BarDataColumns, sampling_type: Sampling):
        message = "Required column missing : " + missing_column.value + " : for sampling : " + sampling_type.value
        super().__init__(message)

class InvalidDateException(ValueError):
    def __init__(self, day : int, month : int, year : int):
        message = "Invalid day, month or year input : " + str(day) + " : " + str(month) + " : " + str(year)
        super().__init__(message)
