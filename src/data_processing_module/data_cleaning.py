import pandas as pd
import numpy as np
import src.data_base_module.data_blocks as data
import src.data_processing_module.data_processing_logger as dp_logger


def morning_after_noon_split(tick_df_wrapper : data.TickDataFrame, split_points_hours : int) -> (data.TickDataFrame, data.TickDataFrame):
    """
    Splits the tick data along the mid day gap
    :param tick_df_wrapper: reference to the tick data frame
    :param split_points_hours: The time in hours from the time of the first tick to the gap
    :return: None
    """
    tick_df_ref : pd.DataFrame = tick_df_wrapper.get_tick_data()
    split_point : int = tick_df_ref[data.TickDataColumns.TIMESTAMP_NANO.value].iloc[0] + split_points_hours * 3600 * 1E9
    morning_points : [bool] = (tick_df_ref[data.TickDataColumns.TIMESTAMP_NANO.value] < split_point)
    afternoon_points : [bool] = np.logical_not(morning_points)
    morning_df : pd.DataFrame = tick_df_ref[morning_points]
    afternoon_df : pd.DataFrame = tick_df_ref[afternoon_points]
    morning_df_wrapper = data.TickDataFrame(tick_df = morning_df, symbol = tick_df_wrapper.symbol, date = tick_df_wrapper.date,
                                            intra_day_period = data.IntraDayPeriod.MORNING)
    afternoon_df_wrapper = data.TickDataFrame(tick_df = afternoon_df, symbol = tick_df_wrapper.symbol, date = tick_df_wrapper.date,
                                              intra_day_period = data.IntraDayPeriod.AFTERNOON)
    dp_logger.log_morning_after_noon_split(tick_df_wrapper = tick_df_wrapper, split_hours = split_points_hours)
    return morning_df_wrapper, afternoon_df_wrapper

# ----- tick_data_frame_cleaning --------
def interpolate_zero_bid_ask_prices(tick_df_wrapper : data.TickDataFrame) -> None:
    """
    Interpolate zero entries in the bid ask prices
    :param tick_df_wrapper: reference to the tick data frame
    :return: None
    """
    tick_df_ref : pd.DataFrame = tick_df_wrapper.get_tick_data()
    for col_name in [data.TickDataColumns.ASK1P.value,
                     data.TickDataColumns.ASK2P.value,
                     data.TickDataColumns.ASK3P.value,
                     data.TickDataColumns.ASK4P.value,
                     data.TickDataColumns.ASK5P.value,
                     data.TickDataColumns.BID1P.value,
                     data.TickDataColumns.BID2P.value,
                     data.TickDataColumns.BID3P.value,
                     data.TickDataColumns.BID4P.value,
                     data.TickDataColumns.BID5P.value]:
        limit_price_series = tick_df_ref.loc[:, col_name]
        tick_df_ref.loc[:, col_name] = limit_price_series.mask(limit_price_series == 0).interpolate().ffill().bfill()
        dp_logger.log_interpolate_zero_bid_ask_prices(tick_df_wrapper = tick_df_wrapper, col_name = col_name)


def interpolate_zero_bid_ask_quantities(tick_df_wrapper : data.TickDataFrame) -> None:
    """
    Interpolate zero entries in the bid ask quantities
    :param tick_df_wrapper: reference to the tick data frame
    :return: None
    """
    tick_df_ref : pd.DataFrame = tick_df_wrapper.get_tick_data()
    for col_name in [data.TickDataColumns.ASK1Q.value,
                     data.TickDataColumns.ASK2Q.value,
                     data.TickDataColumns.ASK3Q.value,
                     data.TickDataColumns.ASK4Q.value,
                     data.TickDataColumns.ASK5Q.value,
                     data.TickDataColumns.BID1Q.value,
                     data.TickDataColumns.BID2Q.value,
                     data.TickDataColumns.BID3Q.value,
                     data.TickDataColumns.BID4Q.value,
                     data.TickDataColumns.BID5Q.value]:
        limit_quantity_series = tick_df_ref.loc[:, col_name]
        tick_df_ref.loc[:, col_name] = limit_quantity_series.mask(limit_quantity_series == 0).interpolate().ffill().bfill()
        dp_logger.log_interpolate_zero_bid_ask_quantities(tick_df_wrapper = tick_df_wrapper, col_name = col_name)


def interpolate_bid_ask_price_outliers(tick_df_wrapper : data.TickDataFrame, lower_threshold : float, upper_threshold : float) -> None:
    tick_df_ref : pd.DataFrame = tick_df_wrapper.get_tick_data()
    for col_name in [data.TickDataColumns.ASK1P.value,
                     data.TickDataColumns.ASK2P.value,
                     data.TickDataColumns.ASK3P.value,
                     data.TickDataColumns.ASK4P.value,
                     data.TickDataColumns.ASK5P.value,
                     data.TickDataColumns.BID1P.value,
                     data.TickDataColumns.BID2P.value,
                     data.TickDataColumns.BID3P.value,
                     data.TickDataColumns.BID4P.value,
                     data.TickDataColumns.BID5P.value]:
        limit_price_series : pd.Series = tick_df_ref.loc[:, col_name]
        outlier_bool_arr : [bool] = np.logical_or((limit_price_series < lower_threshold), (limit_price_series > upper_threshold))
        tick_df_ref.loc[:, col_name] = limit_price_series.mask(outlier_bool_arr).interpolate().ffill().bfill()
        dp_logger.log_interpolate_outlier_bid_ask_prices(tick_df_wrapper = tick_df_wrapper, col_name = col_name, lower_threshold = lower_threshold, upper_threshold = upper_threshold)


def interpolate_bid_ask_quantity_outliers(tick_df_wrapper : data.TickDataFrame, outlier_threshold : int) -> None:
    tick_df_ref : pd.DataFrame = tick_df_wrapper.get_tick_data()
    for col_name in [data.TickDataColumns.ASK1Q.value,
                     data.TickDataColumns.ASK2Q.value,
                     data.TickDataColumns.ASK3Q.value,
                     data.TickDataColumns.ASK4Q.value,
                     data.TickDataColumns.ASK5Q.value,
                     data.TickDataColumns.BID1Q.value,
                     data.TickDataColumns.BID2Q.value,
                     data.TickDataColumns.BID3Q.value,
                     data.TickDataColumns.BID4Q.value,
                     data.TickDataColumns.BID5Q.value]:
        limit_quantity_series : pd.Series = tick_df_ref.loc[:, col_name]
        tick_df_ref.loc[:, col_name] = limit_quantity_series.mask(limit_quantity_series > outlier_threshold).interpolate().ffill().bfill()
        dp_logger.log_interpolate_outlier_bid_ask_quantities(tick_df_wrapper = tick_df_wrapper, col_name = col_name, outlier_threshold = outlier_threshold)

def interpolate_trade_price_outliers(tick_df_wrapper : data.TickDataFrame, lower_threshold : float, upper_threshold : float) -> None:
    tick_df_ref : pd.DataFrame = tick_df_wrapper.get_tick_data()
    trade_price_series : pd.Series = tick_df_ref[data.TickDataColumns.LAST_PRICE.value]
    # ----- pull out the ticks that have trades ------
    has_trade_bool_arr : [bool] = (trade_price_series > 0)
    trade_price_series = trade_price_series[has_trade_bool_arr]
    # ----- remove and interpolate outlier trade prices -------
    is_outlier_bool_arr : [bool] = np.logical_or((trade_price_series < lower_threshold), (trade_price_series > upper_threshold))
    trade_price_series = trade_price_series.mask(is_outlier_bool_arr).interpolate().ffill().bfill()
    # ----- put back the interpolate trade price series ------
    tick_df_ref.loc[:, data.TickDataColumns.LAST_PRICE.value] = trade_price_series
    tick_df_ref.loc[:, data.TickDataColumns.LAST_PRICE.value] = tick_df_ref.loc[:, data.TickDataColumns.LAST_PRICE.value].fillna(0)
    dp_logger.log_interpolate_outlier_trade_prices(tick_df_wrapper = tick_df_wrapper, lower_threshold = lower_threshold, upper_threshold = upper_threshold)

def interpolate_trade_volume_outliers(tick_df_wrapper : data.TickDataFrame, outlier_threshold : int) -> None:
    tick_df_ref : pd.DataFrame = tick_df_wrapper.get_tick_data()
    trade_quantity_series : pd.Series = tick_df_ref[data.TickDataColumns.LAST_QUANTITY.value]
    # ----- pull out the ticks that has trades ------
    has_trade_bool_arr : [bool] = (trade_quantity_series > 0)
    trade_quantity_series = trade_quantity_series[has_trade_bool_arr]
    # ----- remove and interpolate outlier trade quantities ------
    is_outlier_bool_arr : [bool] = (trade_quantity_series > outlier_threshold)
    trade_quantity_series = trade_quantity_series.mask(is_outlier_bool_arr).interpolate().ffill().bfill()
    # ----- put back the interpolate trade price series -----
    tick_df_ref.loc[:, data.TickDataColumns.LAST_QUANTITY.value] = trade_quantity_series
    tick_df_ref.loc[:, data.TickDataColumns.LAST_QUANTITY.value] = tick_df_ref.loc[:, data.TickDataColumns.LAST_QUANTITY.value].fillna(0)
    dp_logger.log_interpolate_outlier_trade_quantities(tick_df_wrapper = tick_df_wrapper, outlier_threshold = outlier_threshold)

# -------- interpolation functions for bar data ----------
def interpolate_bar_zero_prices(bar_wrapper : data.BarDataFrame) -> data.BarDataFrame:
    bar_copy : pd.DataFrame = bar_wrapper.get_bar_data_copy()
    for col_name in [data.BarDataColumns.OPEN.value,
                     data.BarDataColumns.CLOSE.value,
                     data.BarDataColumns.HIGH.value,
                     data.BarDataColumns.LOW.value,
                     data.BarDataColumns.VWAP.value,
                     data.BarDataColumns.VOLUME.value,
                     data.BarDataColumns.TIMESTAMP.value]:
        series : pd.Series = bar_copy[col_name]
        bar_copy[col_name] = series.mask(series == 0).interpolate().ffill().bfill()
    new_bar_wrapper : data.BarDataFrame = bar_wrapper.create_empty_copy()
    new_bar_wrapper.set_data_frame(bar_df = bar_copy, deep_copy = False)
    dp_logger.log_interpolate_zeros_bar(bar_df_wrapper = bar_wrapper)
    return new_bar_wrapper



