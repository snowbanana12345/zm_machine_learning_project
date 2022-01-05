from scipy import stats
import numpy as np
import pandas as pd
import src.data_base_module.data_blocks as data


# ------- print statistics on tick data -------
# ----------- print functions to check un cleaned raw tick data -------
def scan_tick_bid_ask_prices_zero_entries(tick_df_wrapper : data.TickDataFrame) -> None:
    print(" ---------- " + str(tick_df_wrapper) + " -- Scanning for zeros in bid ask prices ----------")
    tick_data_ref : pd.DataFrame = tick_df_wrapper.get_tick_data()
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
        series : pd.Series = tick_data_ref[col_name]
        print("Number of zeros in " + col_name + " : " + str((series == 0).sum()))

def scan_tick_bid_ask_quantities_zero_entries(tick_df_wrapper : data.TickDataFrame) -> None:
    print(" ---------- " + str(tick_df_wrapper) + " -- Scanning for zeros in bid ask quantity ----------")
    tick_data_ref : pd.DataFrame = tick_df_wrapper.get_tick_data()
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
        series : pd.Series = tick_data_ref[col_name]
        print("Number of zeros in " + col_name + " : " + str((series == 0).sum()))


def scan_tick_missing_values(tick_df_wrapper : data.TickDataFrame) -> None:
    print(" ---------- " + str(tick_df_wrapper) + " Scanning for missing values ---------------")
    tick_df_ref : pd.DataFrame = tick_df_wrapper.get_tick_data()
    for col_name in tick_df_ref.columns:
        series = tick_df_ref[col_name]
        num_missing_values = len(series[pd.isnull(series)])
        print("Column : " + col_name + " has " + str(num_missing_values) + " missing values")


def scan_time_stamp_duplicates(tick_df_wrapper : data.TickDataFrame) -> None:
    # ------ time stamp duplicate scanning -------
    tick_df_ref : pd.DataFrame = tick_df_wrapper.get_tick_data()
    print(str(tick_df_wrapper) + " -- has duplicate time stamps : " + str(tick_df_ref[data.TickDataColumns.TIMESTAMP_NANO.value].duplicated().any()))


def scan_both_non_zero_bid_ask_price_and_quantity(tick_df_wrapper: data.TickDataFrame) -> None:
    print(" ---------- " + str(tick_df_wrapper) + " scanning if both bid-ask price and quantity are non zero ---------------")
    tick_df_ref = tick_df_wrapper.get_tick_data()
    limit_price_column_names = [data.TickDataColumns.ASK1P.value,
                     data.TickDataColumns.ASK2P.value,
                     data.TickDataColumns.ASK3P.value,
                     data.TickDataColumns.ASK4P.value,
                     data.TickDataColumns.ASK5P.value,
                     data.TickDataColumns.BID1P.value,
                     data.TickDataColumns.BID2P.value,
                     data.TickDataColumns.BID3P.value,
                     data.TickDataColumns.BID4P.value,
                     data.TickDataColumns.BID5P.value]
    limit_quantity_column_names = [data.TickDataColumns.ASK1Q.value,
                     data.TickDataColumns.ASK2Q.value,
                     data.TickDataColumns.ASK3Q.value,
                     data.TickDataColumns.ASK4Q.value,
                     data.TickDataColumns.ASK5Q.value,
                     data.TickDataColumns.BID1Q.value,
                     data.TickDataColumns.BID2Q.value,
                     data.TickDataColumns.BID3Q.value,
                     data.TickDataColumns.BID4Q.value,
                     data.TickDataColumns.BID5Q.value]
    for price_col_name, quantity_col_name in zip(limit_price_column_names, limit_quantity_column_names):
        price_series = tick_df_ref[price_col_name]
        quantity_series = tick_df_ref[quantity_col_name]
        is_one_zero_series: [bool] = np.logical_xor((price_series.values == 0), (quantity_series.values == 0))
        print(price_col_name + " - " + quantity_col_name + " -- Number of ticks with non zero price but zero trade quantity or vice versa : " + str(len([bol for bol in is_one_zero_series if bol])))


def scan_both_non_zero_trade_price_and_quantity(tick_df_wrapper : data.TickDataFrame) -> None:
    print(" ---------- " + str(tick_df_wrapper) + " scanning if both trade price and quantity are non zero ---------------")
    tick_df_ref : pd.DataFrame = tick_df_wrapper.get_tick_data()
    last_price_series = tick_df_ref[data.TickDataColumns.LAST_PRICE.value]
    last_quantity_series = tick_df_ref[data.TickDataColumns.LAST_QUANTITY.value]
    is_one_zero_series : [bool]= np.logical_xor((last_price_series.values == 0), (last_quantity_series == 0))
    print(" Number of ticks with non zero trade price but zero trade quantity or vice versa : " + str(len([bol for bol in is_one_zero_series if bol])))

def is_ascending(values : [float]) -> bool:
    return values == sorted(values)

def is_descending(values : [float]) -> bool:
    return values == sorted(values, reverse = True)

def scan_ask_in_ascending_order(tick_df_wrapper : data.TickDataFrame) -> None:
    print(" ---------- " + str(tick_df_wrapper) + " scanning if ask prices are in ascending order ---------------")
    tick_df_ref : pd.DataFrame = tick_df_wrapper.get_tick_data()
    ask1p_series = tick_df_ref[data.TickDataColumns.ASK1P.value]
    ask2p_series = tick_df_ref[data.TickDataColumns.ASK2P.value]
    ask3p_series = tick_df_ref[data.TickDataColumns.ASK3P.value]
    ask4p_series = tick_df_ref[data.TickDataColumns.ASK4P.value]
    ask5p_series = tick_df_ref[data.TickDataColumns.ASK5P.value]
    ascending_bool_array = [is_ascending([ask1p, ask2p, ask3p, ask4p, ask5p]) for ask1p, ask2p, ask3p, ask4p, ask5p
                            in zip(ask1p_series, ask2p_series, ask3p_series, ask4p_series, ask5p_series)]
    print(" Number of rows that ask prices are not ascending : " + str(len([bol for bol in ascending_bool_array if not bol])))


def scan_bid_in_descending_order(tick_df_wrapper : data.TickDataFrame) -> None:
    print(" ---------- " + str(tick_df_wrapper) + " scanning if bid prices are in descending order --------------")
    tick_df_ref : pd.DataFrame = tick_df_wrapper.get_tick_data()
    bid1p_series = tick_df_ref[data.TickDataColumns.BID1P.value]
    bid2p_series = tick_df_ref[data.TickDataColumns.BID2P.value]
    bid3p_series = tick_df_ref[data.TickDataColumns.BID3P.value]
    bid4p_series = tick_df_ref[data.TickDataColumns.BID4P.value]
    bid5p_series = tick_df_ref[data.TickDataColumns.BID5P.value]
    descending_bool_array = [is_descending([bid1p, bid2p, bid3p, bid4p, bid5p]) for bid1p, bid2p, bid3p, bid4p, bid5p
                            in zip(bid1p_series, bid2p_series, bid3p_series, bid4p_series, bid5p_series)]
    print(" Number of rows that bid prices are not descending : " + str(len([bol for bol in descending_bool_array if not bol])))


def scan_bid_ask_price_outliers(tick_df_wrapper : data.TickDataFrame, lower_threshold : float, upper_threshold : float) -> None:
    print(" ---------- " + str(tick_df_wrapper) + " scanning for outliers in bid ask prices --------------")
    tick_data_ref = tick_df_wrapper.get_tick_data()
    limit_price_column_names = [data.TickDataColumns.ASK1P.value,
                     data.TickDataColumns.ASK2P.value,
                     data.TickDataColumns.ASK3P.value,
                     data.TickDataColumns.ASK4P.value,
                     data.TickDataColumns.ASK5P.value,
                     data.TickDataColumns.BID1P.value,
                     data.TickDataColumns.BID2P.value,
                     data.TickDataColumns.BID3P.value,
                     data.TickDataColumns.BID4P.value,
                     data.TickDataColumns.BID5P.value]
    for col_name in limit_price_column_names:
        limit_price_col = tick_data_ref[col_name]
        outlier_bool_arr = np.logical_or((limit_price_col < lower_threshold), (limit_price_col > upper_threshold))
        print(" Number of outliers in column : " + col_name + " : " + str(len([bol for bol in outlier_bool_arr if bol]))\
              + " : lower_threshold : " + str(lower_threshold) + " : upper_threshold : " + str(upper_threshold))


def scan_bid_ask_quantity_outliers(tick_df_wrapper : data.TickDataFrame, lower_threshold : float, upper_threshold : float):
    print(" ---------- " + str(tick_df_wrapper) + " scanning for outliers in bid ask quantities --------------")
    tick_data_ref = tick_df_wrapper.get_tick_data()
    limit_quantity_column_names = [data.TickDataColumns.ASK1Q.value,
                     data.TickDataColumns.ASK2Q.value,
                     data.TickDataColumns.ASK3Q.value,
                     data.TickDataColumns.ASK4Q.value,
                     data.TickDataColumns.ASK5Q.value,
                     data.TickDataColumns.BID1Q.value,
                     data.TickDataColumns.BID2Q.value,
                     data.TickDataColumns.BID3Q.value,
                     data.TickDataColumns.BID4Q.value,
                     data.TickDataColumns.BID5Q.value]
    for col_name in limit_quantity_column_names:
        limit_quantity_col = tick_data_ref[col_name]
        outlier_bool_arr = np.logical_or((limit_quantity_col < lower_threshold), (limit_quantity_col > upper_threshold))
        print(" Number of outliers in column : " + col_name + " : " + str(len([bol for bol in outlier_bool_arr if bol]))\
              + " : lower_threshold : " + str(lower_threshold) + " : upper_threshold : " + str(upper_threshold))

def scan_trade_quantity_outliers(tick_df_wrapper : data.TickDataFrame, outlier_threshold : float):
    print(" ---------- " + str(tick_df_wrapper) + " scanning for outliers in trade quantities quantities --------------")
    tick_df_ref = tick_df_wrapper.get_tick_data()
    trade_quantity_series = tick_df_ref[data.TickDataColumns.LAST_QUANTITY.value]
    outlier_bool_arr = (trade_quantity_series > outlier_threshold)
    print(" Number of outlier trade quantities : " + str(len([bol for bol in outlier_bool_arr if bol])) + " : threshold : " + str(outlier_threshold))

