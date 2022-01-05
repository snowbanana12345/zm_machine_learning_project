from dataclasses import dataclass
from typing import List, Set
from src.general_module.custom_exceptions import IncorrectDataTypeException
import pandas as pd
import numpy as np
import src.data_base_module.data_blocks as data
import src.machine_learning_module.machine_learning_logger as ml_logger


# ---- filter array -----
class FilterArray:
    """
    filter array, 1 if the example is chosen by the filter, 0 otherwise
    sampled indices, the set of indices that are sampled
    """

    def __init__(self, filter_array: np.array):
        if not filter_array.dtype == bool:
            raise IncorrectDataTypeException(expected=bool, actual=filter_array.dtype)
        self.filter_array: np.array = filter_array
        self.sampled_indices: List[int] = [i for i, fil in enumerate(self.filter_array) if fil]

    def get_filter_array_ref(self) -> np.array:
        return self.filter_array

    def get_sampled_indices(self) -> List:
        return self.sampled_indices

    def __len__(self):
        return len(self.filter_array)


# ---- filter generators ------
class FilterGenerator:
    def __init__(self):
        self.name: str = "[Filter]_Abstract_filter_generator_class"

    def create_filter_for_data_bar(self, bar_wrapper: data.BarDataFrame) -> FilterArray:
        """
        :param bar_wrapper:
        :return: a boolean numpy array, True or 1 means the bar is sampled, False or 0 otherwise
        NOTE : it is important that you apply .astype(np.bool_) to create a bool array explicitly,
        int32 will not work with pd.Series.mask
        """
        raise Exception("called abstract method : create_filters of abstract class : FilterGenerator")


class IdentityFilter(FilterGenerator):
    """
    Place holder class, use this to disable filtering, samples every bar from the input data frame
    """

    def __init__(self):
        super().__init__()
        self.name: str = "[Filter]_Identity"

    def create_filter_for_data_bar(self, bar_wrapper: data.BarDataFrame) -> np.array:
        filter_array = np.ones(len(bar_wrapper)).astype(np.bool_)
        ml_logger.log_bar_filter_creation(self.name, bar_wrapper)
        return FilterArray(filter_array=filter_array)


class EveryKthFilter(FilterGenerator):
    """
    Filters at constant intervals 
    period : the interval size in number of bars
    shift : the offset, 
    
    Example : period = 4 shift = 1 produces the array
    0,1,0,0,0,1,0,0,0,1,0 .....
    """

    def __init__(self, period: int, shift: int, criteria: data.BarDataColumns):
        super().__init__()
        self.period: int = period
        self.shift: int = shift
        self.name: str = "[Filter]_every_" + str(self.period) + "th"
        self.criteria: data.BarDataColumns = criteria

    def create_filter_for_data_bar(self, bar_wrapper: data.BarDataFrame) -> FilterArray:
        filter_array = np.array(
            [1 if (i + self.shift) % self.period == 0 else 0 for i in range(len(bar_wrapper))]).astype(np.bool_)
        ml_logger.log_bar_filter_creation(self.name, bar_wrapper)
        return FilterArray(filter_array=filter_array)


class SymmetricAbsPriceCumSum(FilterGenerator):
    def __init__(self, threshold: float, criteria: data.BarDataColumns):
        super().__init__()
        self.threshold: float = threshold
        self.criteria: data.BarDataColumns = criteria
        self.name: str = "[Filter]_symmetric_abs_price_cumsum_" + str(self.threshold) + "_threshold"

    def create_filter_for_data_bar(self, bar_wrapper: data.BarDataFrame) -> FilterArray:
        price_diff_array: np.array = bar_wrapper.get_column(self.criteria).diff().values
        filter_array: np.array = np.zeros(len(bar_wrapper)).astype(np.bool_)
        s_pos = 0
        s_neg = 0
        for i in range(1, len(bar_wrapper)):
            s_pos = max(0, s_pos + price_diff_array[i])
            s_neg = min(0, s_neg + price_diff_array[i])
            s_t = max(s_pos, - s_neg)
            if s_t > self.threshold:
                filter_array[i] = 1
                s_pos = 0
                s_neg = 0
        ml_logger.log_bar_filter_creation(self.name, bar_wrapper)
        return FilterArray(filter_array=filter_array)


# ------- filter statistics --------
@dataclass(frozen=True)
class FilterStats:
    filter_name: str
    total_bars: int
    sampled_bars: int
    mean_spacing: float
    sd_spacing: float
    skew_spacing: float
    kurt_spacing: float


def find_filter_stats(filter_wrapper: FilterArray, filter_name: str) -> FilterStats:
    filter_array = filter_wrapper.get_filter_array_ref()
    total_bars: int = len(filter_array)
    sampled_bars: int = np.count_nonzero(filter_array)
    mean_spacing = 0
    sd_spacing = 0
    skew_spacing = 0
    kurt_spacing = 0
    if sampled_bars == 0:
        return FilterStats(filter_name=filter_name, total_bars=total_bars, sampled_bars=sampled_bars,
                           mean_spacing=mean_spacing,
                           sd_spacing=sd_spacing, skew_spacing=skew_spacing, kurt_spacing=kurt_spacing)
    # ------ find the list of spacings --------
    spacings = []
    first_non_zero_index = -1
    for i in range(0, total_bars):
        if filter_array[i]:
            first_non_zero_index = i
            break
    last_bar = first_non_zero_index
    for i in range(first_non_zero_index + 1, total_bars):
        if filter_array[i]:
            spacings.append(i - last_bar)
            last_bar = i
    # ------ compute spacing statistics ------
    spacings_ser: pd.Series = pd.Series(spacings)
    mean_spacing = spacings_ser.mean()
    sd_spacing = spacings_ser.std()
    skew_spacing = spacings_ser.skew()
    kurt_spacing = spacings_ser.kurt()
    return FilterStats(filter_name=filter_name, total_bars=total_bars, sampled_bars=sampled_bars,
                       mean_spacing=mean_spacing,
                       sd_spacing=sd_spacing, skew_spacing=skew_spacing, kurt_spacing=kurt_spacing)


def print_filter_stats(filter_stats: FilterStats) -> None:
    print("Statistics for filter : " + filter_stats.filter_name)
    print("total bars : " + str(filter_stats.total_bars))
    print("sampled bars : " + str(filter_stats.sampled_bars))
    print("mean spacing : " + str(filter_stats.mean_spacing))
    print("sd spacing : " + str(filter_stats.sd_spacing))
    print("skew spacing : " + str(filter_stats.skew_spacing))
    print("kurt spacing : " + str(filter_stats.kurt_spacing))
