import matplotlib.pyplot as plt
import src.data_base_module.data_blocks as data
import src.plotting_module.base_plotting_functions as base_plt
import src.plotting_module.plotting_logger as plt_logger
import definitions
import os
import pandas as pd

bar_data_plots_folder = os.path.join(definitions.PLOT_FOLDER_PATH, "bar_data_plots")

FIGURE_X_SIZE = 16
FIGURE_Y_SIZE = 12
DPI = 120

def get_time_bar_plot_file_path(time_bar_wrapper : data.TimeBarDataFrame, col : data.BarDataColumns) -> str:
    folder_path = os.path.join(bar_data_plots_folder, time_bar_wrapper.symbol, time_bar_wrapper.date.get_str())
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    folder_path = os.path.join(folder_path, "time_sampled")
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    file_name = time_bar_wrapper.symbol + "_" + time_bar_wrapper.date.get_str() + "_"
    file_name += "_" + time_bar_wrapper.intra_day_period.value + "_time_"
    file_name += str(time_bar_wrapper.sampling_seconds) + "S_" + col.value + ".png"
    return os.path.join(folder_path, file_name)


def get_time_bar_plot_title(time_bar_wrapper : data.TimeBarDataFrame, col : data.BarDataColumns) -> str:
    return str(time_bar_wrapper) + " -- " + col.value


def plot_and_save_time_bar(time_bar_wrapper : data.TimeBarDataFrame, interpolate_zeros : bool = True):
    time_bar_df_ref = time_bar_wrapper.get_bar_data_reference()
    time_seconds_series = [x * time_bar_wrapper.sampling_seconds for x in range(len(time_bar_df_ref))]
    for bar_data_col in [data.BarDataColumns.OPEN,
                         data.BarDataColumns.CLOSE,
                         data.BarDataColumns.HIGH,
                         data.BarDataColumns.LOW,
                         data.BarDataColumns.VWAP,
                         data.BarDataColumns.VOLUME]:
        plot_title : str = get_time_bar_plot_title(time_bar_wrapper, bar_data_col)
        plot_save_path : str = get_time_bar_plot_file_path(time_bar_wrapper, bar_data_col)
        plt.figure(figsize=(FIGURE_X_SIZE, FIGURE_Y_SIZE), dpi=DPI)
        series = time_bar_df_ref[bar_data_col.value]
        if interpolate_zeros:
            series = series.mask(series == 0).interpolate().ffill().bfill()
        base_plt.plot_xy(time_seconds_series, series.values, x_label = "time(seconds)",
                         y_label = bar_data_col.value, title = plot_title)
        plt.savefig(plot_save_path)
        plt.close()
        plt_logger.log_plot_and_save_time_bar(time_bar_wrapper = time_bar_wrapper, bar_col = bar_data_col)

def get_tick_bar_plot_file_path(tick_bar_wrapper : data.TickBarDataFrame, col : data.BarDataColumns) -> str:
    folder_path = os.path.join(bar_data_plots_folder, tick_bar_wrapper.symbol, tick_bar_wrapper.date.get_str(), "tick_sampled")
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    folder_path = os.path.join(folder_path, "tick_sampled")
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    file_name = tick_bar_wrapper.symbol + "_" + tick_bar_wrapper.date.get_str() + "_"
    file_name += "_" + tick_bar_wrapper.intra_day_period.value + "_tick_"
    file_name += str(tick_bar_wrapper.sampling_ticks) + "_" + col.value + ".png"
    return os.path.join(folder_path, file_name)


def get_tick_bar_plot_title(tick_bar_wrapper : data.TickBarDataFrame, col : data.BarDataColumns) -> str:
    return str(tick_bar_wrapper) + " -- " + col.value


def plot_and_save_tick_bar(tick_bar_wrapper : data.TickBarDataFrame, interpolate_zeros : bool = True) -> None:
    tick_bar_df_ref = tick_bar_wrapper.get_bar_data_reference()
    tick_count_series = [x * tick_bar_wrapper.sampling_ticks for x in range(len(tick_bar_df_ref))]
    for bar_data_col in [data.BarDataColumns.OPEN,
                         data.BarDataColumns.CLOSE,
                         data.BarDataColumns.HIGH,
                         data.BarDataColumns.LOW,
                         data.BarDataColumns.VWAP,
                         data.BarDataColumns.VOLUME]:
        plot_title: str = get_tick_bar_plot_title(tick_bar_wrapper, bar_data_col)
        plot_save_path: str = get_tick_bar_plot_file_path(tick_bar_wrapper, bar_data_col)
        plt.figure(figsize=(FIGURE_X_SIZE, FIGURE_Y_SIZE), dpi=DPI)
        series = tick_bar_df_ref[bar_data_col.value]
        if interpolate_zeros:
            series = series.mask(series == 0).interpolate().ffill().bfill()
        base_plt.plot_xy(tick_count_series, series.values, x_label="number_of_ticks_passed",
                         y_label=bar_data_col.value, title=plot_title)
        plt.savefig(plot_save_path)
        plt.close()
        plt_logger.log_plot_and_save_tick_bar(tick_bar_wrapper = tick_bar_wrapper, bar_col = bar_data_col)

def get_volume_bar_plot_file_path(volume_bar_wrapper : data.VolumeBarDataFrame, col : data.BarDataColumns) -> str:
    folder_path = os.path.join(bar_data_plots_folder, volume_bar_wrapper.symbol, volume_bar_wrapper.date.get_str())
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    folder_path = os.path.join(folder_path, "volume_sampled")
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    file_name = volume_bar_wrapper.symbol + "_" + volume_bar_wrapper.date.get_str() + "_"
    file_name += "_" + volume_bar_wrapper.intra_day_period.value + "_volume_"
    file_name += str(volume_bar_wrapper.sampling_volume) + "_" + col.value + ".png"
    return os.path.join(folder_path, file_name)


def get_volume_bar_plot_title(volume_bar_wrapper : data.VolumeBarDataFrame, col : data.BarDataColumns) -> str:
    return str(volume_bar_wrapper) + " -- " + col.value

def plot_and_save_volume_bar(volume_bar_wrapper : data.VolumeBarDataFrame, interpolate_zeros : bool = True) -> None:
    volume_bar_df_ref = volume_bar_wrapper.get_bar_data_reference()
    volume_count_series = [x * volume_bar_wrapper.sampling_volume for x in range(len(volume_bar_df_ref))]
    for bar_data_col in [data.BarDataColumns.OPEN,
                         data.BarDataColumns.CLOSE,
                         data.BarDataColumns.HIGH,
                         data.BarDataColumns.LOW,
                         data.BarDataColumns.VWAP]:
        plot_title: str = get_volume_bar_plot_title(volume_bar_wrapper, bar_data_col)
        plot_save_path: str = get_volume_bar_plot_file_path(volume_bar_wrapper, bar_data_col)
        plt.figure(figsize=(FIGURE_X_SIZE, FIGURE_Y_SIZE), dpi=DPI)
        series : pd.Series = volume_bar_df_ref[bar_data_col.value]
        if interpolate_zeros:
            series = series.mask(series == 0).interpolate().ffill().bfill()
        base_plt.plot_xy(volume_count_series, series.values, x_label="volume_traded",
                         y_label=bar_data_col.value, title=plot_title)
        plt.savefig(plot_save_path)
        plt.close()
        plt_logger.log_plot_and_save_volume_bar(volume_bar_wrapper, bar_col = bar_data_col)

def get_dollar_bar_plot_file_path(dollar_bar_wrapper : data.DollarBarDataFrame, col : data.BarDataColumns) -> str:
    folder_path = os.path.join(bar_data_plots_folder, dollar_bar_wrapper.symbol, dollar_bar_wrapper.date.get_str())
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    folder_path = os.path.join(folder_path, "dollar_sampled")
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    file_name = dollar_bar_wrapper.symbol + "_" + dollar_bar_wrapper.date.get_str() + "_"
    file_name += "_" + dollar_bar_wrapper.intra_day_period.value + "_dollar_"
    file_name += str(dollar_bar_wrapper.sampling_dollar) + "_" + col.value + ".png"
    return os.path.join(folder_path, file_name)


def get_dollar_bar_plot_title(dollar_bar_wrapper : data.DollarBarDataFrame, col : data.BarDataColumns) -> str:
    return str(dollar_bar_wrapper) + " -- " + col.value


def plot_and_save_dollar_bar(dollar_bar_wrapper : data.DollarBarDataFrame, interpolate_zeros : bool = True) -> None:
    dollar_bar_df_ref = dollar_bar_wrapper.get_bar_data_reference()
    dollar_count_series = [x * dollar_bar_wrapper.sampling_dollar for x in range(len(dollar_bar_df_ref))]
    for bar_data_col in [data.BarDataColumns.OPEN,
                         data.BarDataColumns.CLOSE,
                         data.BarDataColumns.HIGH,
                         data.BarDataColumns.LOW,
                         data.BarDataColumns.VWAP,
                         data.BarDataColumns.VOLUME]:
        plot_title: str = get_dollar_bar_plot_title(dollar_bar_wrapper, bar_data_col)
        plot_save_path: str = get_dollar_bar_plot_file_path(dollar_bar_wrapper, bar_data_col)
        plt.figure(figsize=(FIGURE_X_SIZE, FIGURE_Y_SIZE), dpi=DPI)
        series: pd.Series = dollar_bar_df_ref[bar_data_col.value]
        if interpolate_zeros:
            series = series.mask(series == 0).interpolate().ffill().bfill()
        base_plt.plot_xy(dollar_count_series, series.values, x_label="dollar_traded",
                         y_label=bar_data_col.value, title=plot_title)
        plt.savefig(plot_save_path)
        plt.close()
        plt_logger.log_plot_and_save_dollar_bar(dollar_bar_wrapper, bar_col = bar_data_col)

