import matplotlib.pyplot as plt
import pandas as pd
import src.data_base_module.data_blocks as data
import src.plotting_module.base_plotting_functions as base_plt
import src.plotting_module.plotting_logger as plt_logger
import definitions
import os

raw_tick_data_plots_folder = os.path.join(definitions.PLOT_FOLDER_PATH, "raw_tick_plots")
clean_tick_data_plots_folder = os.path.join(definitions.PLOT_FOLDER_PATH, "cleaned_tick_plots")

FIGURE_X_SIZE = 16
FIGURE_Y_SIZE = 12
DPI = 120

# ------- plotting functions -------
def plot_and_save_tick_count(tick_df_wrapper : data.TickDataFrame, sampling_seconds : int, is_raw_data : bool):
    # ------- sampling trade volume into time bars ----------
    tick_data_ref: pd.DataFrame = tick_df_wrapper.get_tick_data()
    time_stamp_series = tick_data_ref[data.TickDataColumns.TIMESTAMP_NANO.value]
    time_stamp_series.index = tick_data_ref[data.TickDataColumns.TIMESTAMP_NANO.value].apply(lambda ts: pd.Timestamp(ts))
    count_series = time_stamp_series.resample(str(sampling_seconds) + "S").count()
    time_stamp_series = time_stamp_series.resample(str(sampling_seconds) + "S").first()
    # ------ plotting -----
    plot_title : str = get_plot_title(tick_df_wrapper = tick_df_wrapper, plot_name = "num of ticks in intervals of " + str(sampling_seconds) + "S")
    plt.figure(figsize=(FIGURE_X_SIZE, FIGURE_Y_SIZE), dpi=DPI)
    base_plt.plot_xy_nano_time(time_stamp_series.values, count_series.values, y_label="tick counts", title = plot_title)
    # ----- saving the plot ------
    save_file_path = get_file_path(tick_df_wrapper = tick_df_wrapper, file_name = "tick_counts_" + str(sampling_seconds) + "S", is_raw_data = is_raw_data)
    plt.savefig(save_file_path)
    plt.close()
    # ----- reset all range indexes of the tick data frame -----
    tick_df_wrapper.reset_all_columns_to_range_index()
    # ----- log completion of function -----
    plt_logger.log_plot_and_save_tick_counts(tick_df_wrapper = tick_df_wrapper, sampling_seconds = sampling_seconds, is_raw_data = is_raw_data)


def plot_and_save_tick_avg_bid_ask_spread(tick_df_wrapper : data.TickDataFrame, sampling_seconds : int, is_raw_data : bool) -> None:
    # ------ find the difference between best bid and best ask and resample them  ---------
    tick_df_ref = tick_df_wrapper.get_tick_data()
    ask_col_names = [data.TickDataColumns.ASK1P, data.TickDataColumns.ASK2P, data.TickDataColumns.ASK3P,
                     data.TickDataColumns.ASK4P, data.TickDataColumns.ASK5P]
    bid_col_names = [data.TickDataColumns.BID1P, data.TickDataColumns.BID2P, data.TickDataColumns.BID3P,
                     data.TickDataColumns.BID4P, data.TickDataColumns.BID5P]
    ask_col_names = [ask.value for ask in ask_col_names]
    bid_col_names = [bid.value for bid in bid_col_names]
    best_ask_series : pd.Series = tick_df_ref.loc[:, ask_col_names].apply(min, axis=1)
    best_bid_series : pd.Series = tick_df_ref.loc[:, bid_col_names].apply(max, axis=1)
    bid_ask_spread_series : pd.Series = best_ask_series - best_bid_series
    time_stamp_series : pd.Series = tick_df_ref[data.TickDataColumns.TIMESTAMP_NANO.value]
    bid_ask_spread_series.index = time_stamp_series.apply(lambda ts: pd.Timestamp(ts))
    time_stamp_series.index = bid_ask_spread_series.index
    bid_ask_spread_series = bid_ask_spread_series.resample(rule = str(sampling_seconds) + "S").mean()
    time_stamp_series = time_stamp_series.resample(rule = str(sampling_seconds) + "S").first()
    # ------ plotting -----
    plot_title = get_plot_title(tick_df_wrapper = tick_df_wrapper, plot_name = "tick averaged bid ask spread in intervals of " + str(sampling_seconds) + "S")
    plt.figure(figsize=(FIGURE_X_SIZE, FIGURE_Y_SIZE), dpi=DPI)
    base_plt.plot_xy_nano_time(time_stamp_nano_series = time_stamp_series.values, values = bid_ask_spread_series.values, y_label = "average bid ask spread", title = plot_title)
    # ----- saving the plot ------
    save_file_path = get_file_path(tick_df_wrapper = tick_df_wrapper, file_name = "avg_bid_ask_spread_" + str(sampling_seconds) + "S", is_raw_data = is_raw_data)
    plt.savefig(save_file_path)
    plt.close()
    # ----- reset all range indexes of the tick data frame -----
    tick_df_wrapper.reset_all_columns_to_range_index()
    # ----- log completion of function -----
    plt_logger.log_plot_and_save_avg_bid_ask_spread(tick_df_wrapper = tick_df_wrapper, sampling_seconds = sampling_seconds, is_raw_data = is_raw_data)


def plot_and_save_bid_ask_spread(tick_df_wrapper : data.TickDataFrame, is_raw_data : bool) -> None:
    # ------ find the difference between best bid and best ask  ---------
    tick_df_ref = tick_df_wrapper.get_tick_data()
    ask_col_names = [data.TickDataColumns.ASK1P, data.TickDataColumns.ASK2P, data.TickDataColumns.ASK3P, data.TickDataColumns.ASK4P, data.TickDataColumns.ASK5P]
    bid_col_names = [data.TickDataColumns.BID1P, data.TickDataColumns.BID2P, data.TickDataColumns.BID3P, data.TickDataColumns.BID4P, data.TickDataColumns.BID5P]
    ask_col_names = [ask.value for ask in ask_col_names]
    bid_col_names = [bid.value for bid in bid_col_names]
    best_ask_series = tick_df_ref.loc[:, ask_col_names].apply(min, axis = 1)
    best_bid_series = tick_df_ref.loc[:, bid_col_names].apply(max, axis = 1)
    bid_ask_spread_series = best_ask_series - best_bid_series
    time_stamp_series = tick_df_ref[data.TickDataColumns.TIMESTAMP_NANO.value]
    # ------ plotting -----
    plot_title = get_plot_title(tick_df_wrapper = tick_df_wrapper, plot_name = "bid_ask_spread")
    plt.figure(figsize=(FIGURE_X_SIZE, FIGURE_Y_SIZE), dpi=DPI)
    base_plt.plot_xy_nano_time(time_stamp_nano_series = time_stamp_series.values, values = bid_ask_spread_series.values, y_label = "bid ask spread", title = plot_title)
    # -----saving the plot ------
    save_file_path = get_file_path(tick_df_wrapper = tick_df_wrapper, file_name = "bid_ask_spread", is_raw_data = is_raw_data)
    plt.savefig(save_file_path)
    plt.close()
    # ----- log completion of function -----
    plt_logger.log_plot_and_save_bid_ask_spread(tick_df_wrapper = tick_df_wrapper, is_raw_data = is_raw_data)

def plot_and_save_bid_ask_quantities(tick_df_wrapper : data.TickDataFrame, is_raw_data : bool) -> None:
    tick_df_ref: pd.DataFrame = tick_df_wrapper.get_tick_data()
    time_stamp_series: pd.Series = tick_df_ref[data.TickDataColumns.TIMESTAMP_NANO.value]
    for col_name in [data.TickDataColumns.ASK1Q.value,
                     data.TickDataColumns.ASK2Q.value,
                     data.TickDataColumns.ASK3Q.value,
                     data.TickDataColumns.ASK4Q.value,
                     data.TickDataColumns.ASK5Q.value,
                     data.TickDataColumns.BID1Q.value,
                     data.TickDataColumns.BID2Q.value,
                     data.TickDataColumns.BID3Q.value,
                     data.TickDataColumns.BID4Q.value,
                     data.TickDataColumns.BID5Q.value,
                     ]:
        bid_ask_series: pd.Series = tick_df_ref[col_name]
        # ------ plot individual ask_series --------
        plot_title = get_plot_title(tick_df_wrapper = tick_df_wrapper, plot_name = col_name + "_quantity")
        plt.figure(figsize=(FIGURE_X_SIZE, FIGURE_Y_SIZE), dpi=DPI)
        base_plt.plot_xy_nano_time(time_stamp_series.values, bid_ask_series.values, y_label=col_name, num_x_ticks=15, title=plot_title)
        # ------ saving the individual plot -------
        save_file_path = get_file_path(tick_df_wrapper = tick_df_wrapper, file_name = col_name, is_raw_data = is_raw_data)
        plt.savefig(save_file_path)
        plt.close()
        # ------ log completion of obtaining ask plot -----
        plt_logger.log_plot_and_save_tick_bid_ask_quantity(tick_df_wrapper = tick_df_wrapper, bid_ask_qty_col_name = col_name, is_raw_data = is_raw_data)


def plot_and_save_tick_trade_price(tick_df_wrapper: data.TickDataFrame, is_raw_data: bool) -> None:
    # ------ filter out only ticks that have non zero trade prices -------
    tick_data_ref: pd.DataFrame = tick_df_wrapper.get_tick_data()
    last_price_series: pd.Series = tick_data_ref[data.TickDataColumns.LAST_PRICE.value]
    time_stamp_series: pd.Series = tick_data_ref[data.TickDataColumns.TIMESTAMP_NANO.value]
    has_trade_bool_arr : [bool] = (last_price_series > 0)
    time_stamp_series = time_stamp_series[has_trade_bool_arr]
    last_price_series = last_price_series[has_trade_bool_arr]
    # ------ plotting -----
    plot_title = get_plot_title(tick_df_wrapper = tick_df_wrapper, plot_name = "traded_price")
    plt.figure(figsize=(FIGURE_X_SIZE, FIGURE_Y_SIZE), dpi=DPI)
    base_plt.plot_xy_nano_time(time_stamp_series.values, last_price_series.values, y_label="traded price", title = plot_title)
    # ------ saving the plot -------
    save_file_path = get_file_path(tick_df_wrapper = tick_df_wrapper, file_name = "trade_price", is_raw_data = is_raw_data)
    plt.savefig(save_file_path)
    plt.close()
    # ----- log completion of function -----
    plt_logger.log_plot_and_save_tick_trade_price(tick_df_wrapper=tick_df_wrapper, is_raw_data=is_raw_data)


def plot_and_save_tick_trade_volume(tick_df_wrapper: data.TickDataFrame, sampling_seconds: int,
                                    is_raw_data: bool) -> None:
    # ------- sampling trade volume into time bars ----------
    tick_data_ref: pd.DataFrame = tick_df_wrapper.get_tick_data()
    trade_qty_series = tick_data_ref[data.TickDataColumns.LAST_QUANTITY.value]
    time_stamp_series = tick_data_ref[data.TickDataColumns.TIMESTAMP_NANO.value]
    trade_qty_series.index = tick_data_ref[data.TickDataColumns.TIMESTAMP_NANO.value].apply(lambda ts: pd.Timestamp(ts))
    time_stamp_series.index = trade_qty_series.index
    trade_qty_series = trade_qty_series.resample(str(sampling_seconds) + "S").sum()
    time_stamp_series = time_stamp_series.resample(str(sampling_seconds) + "S").first()
    # ------- plotting --------
    plot_title = get_plot_title(tick_df_wrapper = tick_df_wrapper, plot_name =  " -- volume traded in intervals of -- " + str(sampling_seconds) + "S")
    plt.figure(figsize=(FIGURE_X_SIZE, FIGURE_Y_SIZE), dpi=DPI)
    base_plt.plot_xy_nano_time(time_stamp_series.values, trade_qty_series.values, y_label="volume traded",
                               num_x_ticks=15,
                               title=plot_title)
    # ------ saving the plot -------
    save_file_path = get_file_path(tick_df_wrapper = tick_df_wrapper, file_name = "volume_traded_" + str(sampling_seconds) + "S", is_raw_data = is_raw_data)
    plt.savefig(save_file_path)
    plt.close()
    # ----- reset all range indexes of the tick data frame -----
    tick_df_wrapper.reset_all_columns_to_range_index()
    # ------ log completion of function -----
    plt_logger.log_plot_and_save_tick_vol(tick_df_wrapper=tick_df_wrapper, sampling_seconds=sampling_seconds,
                                          is_raw_data=is_raw_data)


def plot_and_save_bid_ask_prices(tick_df_wrapper: data.TickDataFrame, is_raw_data: bool) -> None:
    tick_df_ref: pd.DataFrame = tick_df_wrapper.get_tick_data()
    time_stamp_series: pd.Series = tick_df_ref[data.TickDataColumns.TIMESTAMP_NANO.value]
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
        bid_ask_series: pd.Series = tick_df_ref[col_name]
        # ------ plot individual ask_series --------
        plot_title = get_plot_title(tick_df_wrapper = tick_df_wrapper, plot_name = col_name + "_price")
        plt.figure(figsize=(FIGURE_X_SIZE, FIGURE_Y_SIZE), dpi=DPI)
        base_plt.plot_xy_nano_time(time_stamp_series.values, bid_ask_series.values, y_label=col_name, num_x_ticks=15,
                                   title=plot_title)
        # ------ saving the individual plot -------
        save_file_path = get_file_path(tick_df_wrapper = tick_df_wrapper, file_name = col_name, is_raw_data = is_raw_data)
        plt.savefig(save_file_path)
        plt.close()
        # ------ log completion of obtaining ask plot -----
        plt_logger.log_plot_and_save_tick_bid_ask_price(tick_df_wrapper = tick_df_wrapper, bid_ask_col_name = col_name, is_raw_data = is_raw_data)

# ------ save function -------
def get_file_path(tick_df_wrapper : data.TickDataFrame, file_name : str, is_raw_data : bool) -> str:
    if is_raw_data:
        save_folder = os.path.join(raw_tick_data_plots_folder, tick_df_wrapper.symbol,tick_df_wrapper.date.get_str())
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        save_file_name = tick_df_wrapper.symbol + "_" + tick_df_wrapper.date.get_str() + "_" + tick_df_wrapper.intra_day_period.value + "_" + file_name + "_raw.png"
        save_file_path = os.path.join(save_folder, save_file_name)
    else:
        save_folder = os.path.join(clean_tick_data_plots_folder, tick_df_wrapper.symbol,tick_df_wrapper.date.get_str())
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        save_file_name = tick_df_wrapper.symbol + "_" + tick_df_wrapper.date.get_str() + "_" + tick_df_wrapper.intra_day_period.value + "_" + file_name + "_clean.png"
        save_file_path = os.path.join(save_folder, save_file_name)
    return save_file_path

def get_plot_title(tick_df_wrapper : data.TickDataFrame, plot_name : str) -> str:
    return tick_df_wrapper.symbol + " -- " + tick_df_wrapper.date.get_str_format_2() + " -- " + tick_df_wrapper.intra_day_period.value + " -- " + plot_name