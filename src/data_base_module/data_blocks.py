from enum import Enum
import pandas as pd
from dataclasses import dataclass
from typing import ClassVar, Dict


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
    TIME = "Time sampled"
    TICK = "Tick sampled"
    VOLUME = "Volume sampled"
    DOLLAR = "Dollar sampled"


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

@dataclass
class BarInfo:
    symbol : str
    date : Date
    intra_day_period : IntraDayPeriod
    sampling_type : Sampling
    sampling_level : int

    def __str__(self):
        return f"{self.sampling_type.value} -- {self.symbol} -- {self.date.get_str_format_2()} -- {self.intra_day_period.value} -- sampling seconds : {self.sampling_level}"

class BarDataFrame:
    """ Abstract class """
    def __init__(self, bar_data : pd.DataFrame, bar_info : BarInfo):
        self.validate_bar_columns(bar_data)
        self.bar_data : pd.DataFrame = bar_data.loc[:, [col.value for col in BarDataColumns]]
        self.bar_info = bar_info

    def get_column(self, col_name : BarDataColumns) -> pd.Series:
        return self.bar_data[col_name.value]

    def validate_bar_columns(self, bar_data : pd.DataFrame):
        for required_column in BarDataColumns:
            if required_column.value not in bar_data.columns:
                raise BarRequiredColumnNotFound(required_column)

    def __str__(self) -> str:
        return str(self.bar_info)

    def __len__(self) -> int:
        return len(self.bar_data)

    def __eq__(self, other) -> bool:
        if not isinstance(other, BarDataFrame):
            return False
        return self.bar_data.equals(other.bar_data) and (self.bar_info == other.bar_info)

# -------- Tick data frame -------
@dataclass
class TickInfo:
    symbol : str
    date : Date
    intra_day_period : IntraDayPeriod

    def __str__(self):
        return self.symbol + " -- " + self.date.get_str_format_2() + " -- " + self.intra_day_period.value


class TickDataFrame:
    """
    Wrapper for a pandas data frame that contains tick data for one intraday period
    Contains meta data : date, intraday period, ticker symbol
    A tick data frame requires and enforces the following columns
    KEY_SYMBOL TIMESTAMP_NANO LAST_PRICE LAST_QUANTITY ASK1P ASK2P ASK3P ASK4P ASK5P
    ASK1Q ASK2Q ASK3Q ASK4Q ASK5Q BID1P BID2P BID3P BID4P BID5P BID1Q BID2Q BID3Q BID4Q BID5Q
    Enforces ranged indexing for the underlying dataframe
    """
    def __init__(self, tick_df: pd.DataFrame, tick_info : TickInfo):
        self.validate_tick_columns(tick_df)
        self.tick_info : TickInfo = tick_info
        self.tick_data : pd.DataFrame = tick_df.loc[:, [column.value for column in TickDataColumns]]
        self.tick_data.index = pd.RangeIndex(len(self.tick_data))

    def validate_tick_columns(self, tick_df : pd.DataFrame) -> None:
        for required_column in TickDataColumns:
            if required_column.value not in tick_df.columns:
                raise TickRequiredColumnNotFound(required_column)

    def __str__(self):
        return str(self.tick_info)

    def __eq__(self, other) -> bool:
        """ NOTE : this equality condition fails when comparing floats with int even if the values are the same """
        if not isinstance(other, TickDataFrame):
            return False
        return self.tick_data.equals(other.tick_data) and (self.tick_info == other.tick_info)

# -------- custom exceptions ---------
class TickRequiredColumnNotFound(Exception):
    def __init__(self, missing_column : TickDataColumns):
        message = "Required column missing : " + missing_column.value
        super().__init__(message)

class BarRequiredColumnNotFound(Exception):
    def __init__(self, missing_column: BarDataColumns):
        message = "Required column missing : " + missing_column.value
        super().__init__(message)

class InvalidDateException(ValueError):
    def __init__(self, day : int, month : int, year : int):
        message = "Invalid day, month or year input : " + str(day) + " : " + str(month) + " : " + str(year)
        super().__init__(message)
