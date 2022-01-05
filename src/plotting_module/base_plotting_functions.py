import matplotlib.pyplot as plt
from datetime import datetime


def plot_2d_scatter(x, y, z, title="Plot"):
    if len(x) == 0 or len(y) == 0:
        return
    plt.scatter(x, y, s=20, c=z, alpha=0.75)
    plt.title(title)


def plot_xy(x, y, x_label : str = "x", y_label : str = "y", title : str ="Plot") -> None:
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)


def plot_xy_multi(x, y_s, x_label="x", y_label="y", labels=None, title="Plot"):
    labels = labels if labels else ["y" + str(i) for i in range(len(y_s))]
    for y_dat, label in zip(y_s, labels):
        plt.plot(x, y_dat, label=label)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend()
    plt.title(title)


def plot_scatter(x, y, x_label='x', y_label='y', title="Plot"):
    if len(x) == 0 or len(y) == 0:
        return
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)


def save_plot(save_file_path, win_x_inches, win_y_inches, dpi=100):
    """ sandwich plt.plot between plt.figure() and this function """
    plt.gcf().set_size_inches(win_x_inches, win_y_inches)
    plt.savefig(save_file_path, dpi=dpi)
    plt.close()


def nano_time_to_day_time_str(time_stamp_nano: int) -> str:
    return str(datetime.fromtimestamp(time_stamp_nano // 1E9).time())

def nano_array_to_string_array(nano_time_array: [int], num_x_ticks: int) -> ([float], [str]):
    start_time, end_time = nano_time_array[0], nano_time_array[-1]
    div = (end_time - start_time) / num_x_ticks
    x_tick_labels = [nano_time_to_day_time_str(start_time + int(multi * div)) for multi in range(num_x_ticks + 1)]
    num_x_ticks = [start_time + int(multi * div) for multi in range(num_x_ticks + 1)]
    return num_x_ticks, x_tick_labels

def plot_xy_nano_time(time_stamp_nano_series: [float], values: [float], x_label="time (ns)", y_label="y", num_x_ticks=8,
                      title="Plot") -> None:
    """ plots with an additional axis containing date time, x is a nano time stamp array """
    x_ticks, x_tick_labels = nano_array_to_string_array(time_stamp_nano_series, num_x_ticks)
    plt.plot(time_stamp_nano_series, values, label=y_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(x_ticks, x_tick_labels)
    plt.legend()
    plt.title(title)




