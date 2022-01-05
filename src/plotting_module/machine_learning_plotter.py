import matplotlib.pyplot as plt
import src.plotting_module.plotting_logger as plt_logger
import pandas as pd
import numpy as np
from src.general_module.custom_exceptions import ArrayLengthMisMatchException

FIGURE_X_SIZE = 16
FIGURE_Y_SIZE = 12
DPI = 120


def plot_price_series_with_filter(price_series : pd.Series, filter_array : np.array, price_series_name : str = "price_series", filter_name : str = "filter_array"):
    """

    :param price_series:
    :param filter_array:
    :param price_series_name:
    :param filter_name:
    :return:
    """
    if len(price_series) != len(filter_array):
        raise ArrayLengthMisMatchException(len(price_series), len(filter_array), price_series_name, filter_name)
    plt.figure(figsize = (FIGURE_X_SIZE, FIGURE_Y_SIZE), dpi = DPI)
    # ------ plot price series ------
    plt.plot(range(len(price_series)), price_series.values, label = price_series_name)
    # ------ plot filter points -----
    filtered_price_series = price_series.mask(np.logical_not(filter_array)).dropna()
    plt.scatter(filtered_price_series.index, filtered_price_series.values, label = filter_name, color = "green", marker = "o")
    # ------ style the plot -----
    plt.xlabel("bar_number")
    plt.ylabel(price_series_name)
    plt.title(price_series_name + " with filter : " + filter_name + " applied")
    plt.grid(True)
    # ------ log completion of plotting -----
    plt_logger.log_plot_price_series_with_filter(price_series_name = price_series_name, filter_name = filter_name)






