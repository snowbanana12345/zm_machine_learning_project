from src.general_module.custom_exceptions import IncorrectDataTypeException, ArrayLengthMisMatchException
from src.machine_learning_module.label_generators import LabelDataFrame
from src.machine_learning_module.filter_generators import FilterArray
import src.machine_learning_module.machine_learning_logger as ml_logger
import numpy as np
import pandas as pd
from typing import Iterator
import random


class BootStrapMatrix:
    def __init__(self, boot_strap_arr : np.array):
        """
        :param boot_strap_arr: a 2D array , rows represent individual boot straps, data type must be bool
        """
        # --- validation ----
        if not len(boot_strap_arr.shape) == 2:
            raise ValueError("Boot Strap Array requires a 2D numpy array")
        if not boot_strap_arr.dtype == bool:
            raise IncorrectDataTypeException(expected = bool, actual =  boot_strap_arr.dtype)
        # --- store ---
        self.boot_strap_arr : np.array = boot_strap_arr

    def get_rows_it(self) -> Iterator[np.array]:
        """
        :return: Iterator that produces a bootstrap array
        """
        return iter(self.boot_strap_arr)

MAX_BOOTSTRAPS_ALLOWED = 10000

class BootStrapGenerator:
    def __init__(self):
        pass

    def generate_bootstrap_array(self, label_wrapper : LabelDataFrame, filter_wrapper : FilterArray, num_bootstraps : int) -> BootStrapMatrix:
        """
        Takes in a label dataframe and a filter array
        Only include filtered examples in the boot strap
        :param filter_array:
        :param label_df:
        :param num_bootstraps:
        :return: BootStrapArray
        """
        if len(label_wrapper) != len(filter_wrapper):
            raise ArrayLengthMisMatchException(len(label_wrapper), len(filter_wrapper), "label", "filter")
        if num_bootstraps > MAX_BOOTSTRAPS_ALLOWED:
            raise ValueError(f"Maximum number of bootstraps exceeded : {num_bootstraps} -- Maximum : {MAX_BOOTSTRAPS_ALLOWED}")

class RandomBootStrapGenerator(BootStrapGenerator):

    def __init__(self, sample_portion : float):
        super().__init__()
        if not 0 <= sample_portion <= 1:
            raise ValueError("Acceptance probability should be a float between 0 and 1")
        self.sample_portion : float = sample_portion
        self.name = f"random boot strap generator : {self.sample_portion}"

    def generate_bootstrap_array(self, label_wrapper : LabelDataFrame, filter_wrapper : FilterArray, num_bootstraps : int) -> BootStrapMatrix:
        super().generate_bootstrap_array(label_wrapper, filter_wrapper, num_bootstraps)
        boot_strap_arr = np.zeros((num_bootstraps, len(label_wrapper)), dtype = np.bool_)
        filter_arr : np.array = filter_wrapper.get_filter_array_ref()
        look_ahead_arr : np.array = label_wrapper.get_look_ahead_series_ref()
        for i in range(num_bootstraps):
            for j in range(len(label_wrapper)):
                if filter_arr[j] and not pd.isnull(look_ahead_arr[j]):
                    boot_strap_arr[i, j] = True if random.uniform(0, 1) < self.sample_portion else False
        ml_logger.log_boot_strap(boot_strap_gen_name = self.name, label_data_frame_description = str(label_wrapper), filter_description = str(filter_wrapper))
        return BootStrapMatrix(boot_strap_arr)


# ----- CALCULATE INFORMATION OVERLAP ------
def find_average_uniqueness(label_wrapper : LabelDataFrame, filter_wrapper : FilterArray, boot_strap_row : np.array, bootstrap_description : str = "No Description") -> float:
    """
    Find information overlap using the formula
    :param label_wrapper: only the look ahead array is required for this function
    :param filter_wrapper: boolean array, 1 if the example has been filtered, else 0
    :param boot_strap_row: boolean array, 1 if the example is selected, else 0
    :return: the average overlap between each example
    NOTE : label wrapper can have null entries indicating a labeling was not possible at that bar
    """
    look_ahead_arr : np.array = label_wrapper.get_look_ahead_series_ref()
    filter_arr : np.array = filter_wrapper.get_filter_array_ref()
    if len(look_ahead_arr) != len(filter_arr):
        raise ArrayLengthMisMatchException(len(look_ahead_arr), len(filter_arr), "look_ahead", "filter")
    if len(filter_arr) != len(boot_strap_row):
        raise ArrayLengthMisMatchException(len(filter_arr), len(boot_strap_row), "filter", "bootstrap")
    selected_arr = np.logical_and(filter_arr, boot_strap_row)
    length : int = len(look_ahead_arr)
    overlap_arr = np.zeros(length).astype(np.int32)
    # ------- calculate the number of labels that uses each point in time, the overlap array -------
    for i in range(length):
        if pd.isnull(look_ahead_arr[i]):
            continue
        if not selected_arr[i]:
            continue
        for j in range(look_ahead_arr[i] + 1):
            if i + j < length:
                overlap_arr[i + j] += 1
    # ------- calculate uniqueness of each selected label --------
    uniqueness_arr = np.zeros(length)
    for i in range(length):
        if not selected_arr[i]:
            continue
        label_uni: float = 0
        for j in range(look_ahead_arr[i] + 1):
            if i + j < length and overlap_arr[i + j] > 0:
                label_uni += 1 / overlap_arr[i + j]
        uniqueness_arr[i] = label_uni / (look_ahead_arr[i] + 1)
    result : float = sum(uniqueness_arr[uniqueness_arr > 0]) / len(uniqueness_arr[uniqueness_arr > 0])
    # ------ log completion of function -------
    ml_logger.log_find_average_uniqueness(str(label_wrapper), str(filter_wrapper), result, bootstrap_description = bootstrap_description)
    # ------ return the average of the average of each uniqueness label -------
    return result





