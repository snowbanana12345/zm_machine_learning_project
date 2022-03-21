import pandas as pd
import numpy as np
import src.data_base_module.data_blocks as data
import src.machine_learning_module.machine_learning_logger as ml_logger
from typing import Set, Dict
from general_module.custom_exceptions import ArrayLengthMisMatchException

# ----- label data frame -------

class LabelDataFrame:
    """
    contains a label data array and a look ahead array
    could be regression or classification
    NOTE : data type for look_ahead_array should be int32
    NOTE : look_ahead_series will have missing entries at the end of the data frame where there is insufficient future
    data to label
    NOTE : DO NOT modify underlying numpy array objects / pandas series as we are passing these objects around by reference
    """
    def __init__(self, label_series: pd.Series, look_ahead_series: pd.Series, label_gen_name: str = "", bar_data_description : str = "",
                 classification_classes : Set[int] = None, default_class : int = None, deep_copy=False):
        # ---- check if label, look_ahead_series have the same length -----
        if not len(label_series) == len(look_ahead_series):
            raise ArrayLengthMisMatchException(len(label_series), len(look_ahead_series), "label", "look_ahead")
        self.label_gen_name: str = label_gen_name
        self.bar_data_description : str = bar_data_description
        self.label_series : pd.Series = label_series.copy() if deep_copy else label_series
        self.look_ahead_series = look_ahead_series.copy() if deep_copy else look_ahead_series
        # ----- enforce range indexing ------
        self.label_series.index = pd.RangeIndex(len(label_series))
        self.look_ahead_series.index = pd.RangeIndex(len(look_ahead_series))
        # ----- store classification classes if applicable ------
        self.classification_classes: Set[int] = classification_classes
        self.default_class: int = default_class

    def get_label_series_ref(self) -> pd.Series:
        return self.label_series

    def get_look_ahead_series_ref(self) -> pd.Series:
        return self.look_ahead_series

    def __str__(self) -> str:
        return "Label Data Frame : " + self.label_gen_name + " -- on : " + self.bar_data_description

    def __len__(self) -> int:
        return len(self.look_ahead_series)

    def update_series(self, label_series : pd.Series, look_ahead_series : pd.Series):
        if not len(label_series) == len(look_ahead_series):
            raise ArrayLengthMisMatchException(len(label_series), len(look_ahead_series), "label_series", "look_ahead_series")
        return LabelDataFrame(label_series = label_series, look_ahead_series = look_ahead_series, label_gen_name = self.label_gen_name)

    def get_one_vs_all(self) -> Dict[int, pd.Series]:
        if not self.classification_classes or not self.default_class:
            return {}
        one_v_all_dict : Dict[int, pd.Series] = {}
        for cls in self.classification_classes:
            if not cls == self.default_class:
                one_v_all_dict[cls] = pd.Series([1 if label == cls else 0 for label in self.label_series.values])
        return one_v_all_dict

# ------ label generators -------
class LabelGenerator:
    """ abstract class label generator """

    def __init__(self):
        pass

    def create_labels_for_data_bar(self, bar_wrapper: data.BarDataFrame) -> LabelDataFrame:
        raise Exception("called abstract method : abstract class : LabelGenerator")

# ------- absolute price change label generator -------
class AbsoluteChangeLabel(LabelGenerator):
    NAME = "[Label]_absolute_price_change"
    """
    labels based on absolute price change
    1 if the price change at the look ahead is equal to or exceeds the threshold
    -1 if the price change at the look ahead is equal to or lower than - threshold
    0 otherwise
    NOTE : this labeling can produce missing values 
    """

    def __init__(self, look_ahead: int, threshold: float, criteria: data.BarDataColumns):
        super().__init__()
        if criteria not in [data.BarDataColumns.OPEN, data.BarDataColumns.CLOSE, data.BarDataColumns.HIGH,
                            data.BarDataColumns.LOW, data.BarDataColumns.VWAP]:
            pass
        self.look_ahead: int = look_ahead
        self.threshold: float = threshold
        self.criteria: data.BarDataColumns = criteria
        self.classification_classes: Set[int] = {1, -1, 0}
        self.default_class = 0

    def create_labels_for_data_bar(self, bar_wrapper: data.BarDataFrame) -> LabelDataFrame:
        price_series: pd.Series = bar_wrapper.get_bar_data_reference()[self.criteria.value]
        future_price_series: pd.Series = price_series.shift(-self.look_ahead)
        diff_series = future_price_series - price_series
        label_series = diff_series.apply(self.label_function)
        look_ahead_series = pd.Series([self.look_ahead]).repeat(len(label_series))
        look_ahead_series.index = label_series.index
        look_ahead_series = look_ahead_series.mask(
            look_ahead_series.index > (len(look_ahead_series) - self.look_ahead - 1))
        label_series = label_series.mask(label_series.index > (len(label_series) - self.look_ahead - 1))
        ml_logger.log_bar_label_creation(label_gen_name=AbsoluteChangeLabel.NAME, bar_wrapper=bar_wrapper)
        return LabelDataFrame(label_series=label_series,
                                            look_ahead_series=look_ahead_series,
                                            label_gen_name=AbsoluteChangeLabel.NAME,
                                            classification_classes=self.classification_classes,
                                            default_class=self.default_class,
                                            bar_data_description = str(bar_wrapper))

    def label_function(self, price_difference: float) -> int:
        if price_difference >= self.threshold:
            return 1
        elif price_difference <= - self.threshold:
            return -1
        else:
            return 0


class Barrier111AbsChangeLabel(LabelGenerator):
    NAME = "[Label]_barrier_111"
    """ all 3 barriers are active
    labels 1 for hitting upper barrier first
    labels 0 for reaching the maximum holding period without hitting either barrier
    labels -1 for hitting lower barrier first
    NOTE : labels 0 if the end of the data set is reached without hitting either barrier 
    NOTE : if either barrier is hit on the max holding period, the label will be 1 or -1 depending on the barrier hit 
    NOTE : the very last bar is labeled 0 with 0 look ahead 
    NOTE : does not produce missing values """

    def __init__(self, upper_barrier: float, lower_barrier: float, max_holding_period: int,
                 criteria: data.BarDataColumns):
        """
        :param upper_barrier: the increase between the upper barrier and the current price level in absolute price
        :param lower_barrier: the decrease between the lower barrier and the current price level in absolute price
        :param max_holding_period: the maximum holding period in units of number of bars
        :param criteria: the price series used for generating labels, e.g. open, close, high, low, VWAP
        """
        super().__init__()
        self.upper_barrier: float = upper_barrier
        self.lower_barrier: float = lower_barrier
        self.max_holding_period: int = max_holding_period
        self.criteria: data.BarDataColumns = criteria
        self.classification_classes: Set[int] = {1, -1, 0}
        self.default_class : int = 0

    def create_labels_for_data_bar(self, bar_wrapper: data.BarDataFrame) -> LabelDataFrame:
        price_series: pd.Series = bar_wrapper.get_bar_data_reference()[self.criteria.value]
        # ------ calculate labeling and look ahead -------
        label_array: np.array = np.zeros(len(price_series)).astype(np.int_)
        price_array: np.array = price_series.values
        look_ahead_array: np.array = np.zeros(len(price_series)).astype(np.int_)
        for i in range(len(price_array)):
            upper_barrier_price = price_array[i] + self.upper_barrier
            lower_barrier_price = price_array[i] - self.lower_barrier
            expiry_bar_number: int = min(i + self.max_holding_period, len(price_array) - 1)
            label: int = 0
            stop_bar_number: int = i
            for j in range(i + 1, expiry_bar_number + 1):
                stop_bar_number = j
                if price_array[j] >= upper_barrier_price:
                    label = 1
                    break
                if price_array[j] <= lower_barrier_price:
                    label = -1
                    break
            label_array[i] = label
            look_ahead_array[i] = stop_bar_number - i
        ml_logger.log_bar_label_creation(label_gen_name=Barrier111AbsChangeLabel.NAME, bar_wrapper=bar_wrapper)
        return LabelDataFrame(label_series= pd.Series(label_array),
                                            look_ahead_series= pd.Series(look_ahead_array),
                                            label_gen_name=AbsoluteChangeLabel.NAME,
                                            classification_classes=self.classification_classes,
                                            default_class=self.default_class,
                                            bar_data_description = str(bar_wrapper))


class Barrier110AbsChangeLabel(LabelGenerator):
    NAME = "[Label]_barrier_110"
    """ no maximum holding period 
    labels 1 for hitting upper barrier first
    labels -1 for hitting lower barrier first
    NOTE : labels 0 if the end of the data set is reached without hitting either barrier 
    NOTE : labels 1 or -1 if the corresponding barrier is hit at the end of the data set
    NOTE : the very last bar is labeled 0 with 0 look ahead
    NOTE : does not produce missing values """

    def __init__(self, upper_barrier: float, lower_barrier: float, criteria: data.BarDataColumns):
        """
        :param upper_barrier: the increase between the upper barrier and the current price level in absolute price
        :param lower_barrier: the decrease between the lower barrier and the current price level in absolute price
        :param criteria: the price series used for generating labels, e.g. open, close, high, low, VWAP
        """
        super().__init__()
        self.upper_barrier: float = upper_barrier
        self.lower_barrier: float = lower_barrier
        self.criteria: data.BarDataColumns = criteria
        self.classification_classes: Set[int] = {1, -1, 0}
        self.default_class = 0

    def create_labels_for_data_bar(self, bar_wrapper: data.BarDataFrame) -> LabelDataFrame:
        price_series: pd.Series = bar_wrapper.get_bar_data_reference()[self.criteria.value]
        # ------ calculate labeling and look ahead -------
        label_array: np.array = np.zeros(len(price_series)).astype(np.int_)
        price_array: np.array = price_series.values
        look_ahead_array: np.array = np.zeros(len(price_series)).astype(np.int_)
        for i in range(len(price_array)):
            upper_barrier_price = price_array[i] + self.upper_barrier
            lower_barrier_price = price_array[i] - self.lower_barrier
            expiry_bar_number: int = len(price_array) - 1
            label: int = 0
            stop_bar_number: int = i
            for j in range(i + 1, expiry_bar_number + 1):
                stop_bar_number = j
                if price_array[j] >= upper_barrier_price:
                    label = 1
                    break
                if price_array[j] <= lower_barrier_price:
                    label = -1
                    break
            label_array[i] = label
            look_ahead_array[i] = stop_bar_number - i
        ml_logger.log_bar_label_creation(label_gen_name=Barrier110AbsChangeLabel.NAME, bar_wrapper=bar_wrapper)
        return LabelDataFrame(label_series = pd.Series(label_array),
                                            look_ahead_series = pd.Series(look_ahead_array),
                                            label_gen_name=AbsoluteChangeLabel.NAME,
                                            classification_classes=self.classification_classes,
                                            default_class=self.default_class,
                                            bar_data_description=str(bar_wrapper))


class Barrier101AbsChangeLabel(LabelGenerator):
    NAME = "[Label]_barrier_101"
    """ lower barrier disabled
    labels 1 for hitting upper barrier first
    labels 0 if maximum holding period is reached without hitting the upper barrier
    NOTE : labels 0 if the end of the data set is reached without hitting either barrier 
    NOTE : the very last bar is labeled 0 with 0 look ahead
    NOTE : does not produce missing values """

    def __init__(self, upper_barrier: float, max_holding_period: int, criteria: data.BarDataColumns):
        """
        :param upper_barrier: the increase between the upper barrier and the current price level in absolute price
        :param max_holding_period: the maximum holding period in units of number of bars
        :param criteria: the price series used for generating labels, e.g. open, close, high, low, VWAP
        """
        super().__init__()
        self.upper_barrier: float = upper_barrier
        self.max_holding_period: int = max_holding_period
        self.criteria: data.BarDataColumns = criteria
        self.classification_classes: Set[int] = {1, 0}
        self.default_class = 0

    def create_labels_for_data_bar(self, bar_wrapper: data.BarDataFrame) -> LabelDataFrame:
        price_series: pd.Series = bar_wrapper.get_bar_data_reference()[self.criteria.value]
        # ------ calculate labeling and look ahead -------
        label_array: np.array = np.zeros(len(price_series)).astype(np.int_)
        price_array: np.array = price_series.values
        look_ahead_array: np.array = np.zeros(len(price_series)).astype(np.int_)
        for i in range(len(price_array)):
            upper_barrier_price = price_array[i] + self.upper_barrier
            expiry_bar_number: int = min(i + self.max_holding_period, len(price_array) - 1)
            label: int = 0
            stop_bar_number: int = i
            for j in range(i + 1, expiry_bar_number + 1):
                stop_bar_number = j
                if price_array[j] >= upper_barrier_price:
                    label = 1
                    break
            label_array[i] = label
            look_ahead_array[i] = stop_bar_number - i
        ml_logger.log_bar_label_creation(label_gen_name=Barrier101AbsChangeLabel.NAME, bar_wrapper=bar_wrapper)
        return LabelDataFrame(label_series= pd.Series(label_array),
                                            look_ahead_series= pd.Series(look_ahead_array),
                                            label_gen_name=AbsoluteChangeLabel.NAME,
                                            classification_classes=self.classification_classes,
                                            default_class=self.default_class,
                                            bar_data_description=str(bar_wrapper))


class Barrier011AbsChangeLabel(LabelGenerator):
    NAME = "[Label]_barrier_011"
    """ upper_barrier disabled  
    labels -1 for hitting lower barrier first
    labels 0 if maximum holding period is reached without hitting the upper barrier
    NOTE : labels 0 if the end of the data set is reached without hitting either barrier 
    NOTE : the very last bar is labeled 0 with 0 look ahead 
    NOTE : does not produce missing values"""

    def __init__(self, lower_barrier: float, max_holding_period: int, criteria: data.BarDataColumns):
        super().__init__()
        self.lower_barrier: float = lower_barrier
        self.max_holding_period: int = max_holding_period
        self.criteria: data.BarDataColumns = criteria
        self.classification_classes: Set[int] = {-1, 0}
        self.default_class = 0

    def create_labels_for_data_bar(self, bar_wrapper: data.BarDataFrame) -> LabelDataFrame:
        price_series: pd.Series = bar_wrapper.get_bar_data_reference()[self.criteria.value]
        # ------ calculate labeling and look ahead -------
        label_array: np.array = np.zeros(len(price_series)).astype(np.int_)
        price_array: np.array = price_series.values
        look_ahead_array: np.array = np.zeros(len(price_series)).astype(np.int_)
        for i in range(len(price_array)):
            lower_barrier_price = price_array[i] - self.lower_barrier
            expiry_bar_number: int = min(i + self.max_holding_period, len(price_array) - 1)
            label: int = 0
            stop_bar_number: int = i
            for j in range(i + 1, expiry_bar_number + 1):
                stop_bar_number = j
                if price_array[j] <= lower_barrier_price:
                    label = -1
                    break
            label_array[i] = label
            look_ahead_array[i] = stop_bar_number - i
        ml_logger.log_bar_label_creation(label_gen_name=Barrier011AbsChangeLabel.NAME, bar_wrapper=bar_wrapper)
        return LabelDataFrame(label_series = pd.Series(label_array),
                                            look_ahead_series = pd.Series(look_ahead_array),
                                            label_gen_name=AbsoluteChangeLabel.NAME,
                                            classification_classes=self.classification_classes,
                                            default_class=self.default_class,
                                            bar_data_description=str(bar_wrapper))
