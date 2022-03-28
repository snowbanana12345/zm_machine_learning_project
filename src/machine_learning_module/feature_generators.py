import pandas as pd
import numpy as np
from typing import List, Dict, Set, Hashable, Tuple, Iterator
import itertools

from openpyxl.packaging.manifest import Override

import src.machine_learning_module.machine_learning_logger as ml_logger
import src.data_base_module.data_blocks as data
import random
from dataclasses import dataclass, field

# ------ import feature generator data -------
import src.machine_learning_module.feature_generator_data.talib_momentum as talib_mom_data
import src.machine_learning_module.feature_generator_data.talib_volatility as talib_vol_data
import src.machine_learning_module.feature_generator_data.talib_pattern as talib_pat_data

# ------ FEATURE DATA FRAME --------
@dataclass
class FeatureInfo:
    feature_gen_names : Set[str] = field(default_factory = set)
    bar_descriptions : Set[str] = field(default_factory = set)

    def merge_horizontal(self, other):
        if not isinstance(other, FeatureInfo):
            raise ValueError("FeatureInfo class can only merge with other FeatureInfo")
        new_gen_names = self.feature_gen_names.union(other.feature_gen_names)
        return FeatureInfo(feature_gen_names = new_gen_names, bar_descriptions = self.bar_descriptions)

    def merge_vertical(self, other):
        if not isinstance(other, FeatureInfo):
            raise ValueError("FeatureInfo class can only merge with other FeatureInfo")
        new_bar_descriptions = self.bar_descriptions.union(other.bar_descriptions)
        return FeatureInfo(feature_gen_names = self.feature_gen_names, bar_descriptions = new_bar_descriptions)

class FeatureDataFrame:
    """
    A parameter dict is in the format :
    Dict[ feature name , List[Dict[argument name, argument value]]
    """
    def __init__(self, feature_df: pd.DataFrame, feature_info : FeatureInfo):
        self.feature_df: pd.DataFrame = feature_df
        self.feature_info : FeatureInfo = feature_info

    def get_feature_df(self) -> pd.DataFrame:
        return self.feature_df

    def __len__(self):
        return len(self.feature_df)

    def merge_horizontal(self, other):
        if not isinstance(other, FeatureDataFrame):
            raise ValueError("FeatureDataFrame can only merge with other feature data frames")
        new_feature_df = pd.concat([self.feature_df, other.feature_df], axis = 1)
        new_feature_info = self.feature_info.merge_horizontal(other.feature_info)
        return FeatureDataFrame(feature_df = new_feature_df, feature_info = new_feature_info)

    def merge_vertical(self, other):
        if not isinstance(other, FeatureDataFrame):
            raise ValueError("FeatureDataFrame can only merge with other feature data frames")
        new_feature_df = pd.concat([self.feature_df, other.feature_df], axis = 0, ignore_index = True)
        new_feature_info = self.feature_info.merge_vertical(other.feature_info)
        return FeatureDataFrame(feature_df=new_feature_df, feature_info=new_feature_info)

    def update_dataframe(self, new_feature_df : pd.DataFrame):
        return FeatureDataFrame(feature_df = new_feature_df, feature_info = self.feature_info)

EMPTY_FEATURE_DATA_FRAME: FeatureDataFrame = FeatureDataFrame(feature_df=pd.DataFrame(), feature_info = FeatureInfo())


# ------- FEATURE GENERATORS  -------

class FeatureGenerator:
    """ abstract class feature generator """
    def __init__(self, parameters_dict: Dict[str, List[Dict[str, float]]], name: str, alias : str,
                 feature_name_list: List[str], feature_creation_func_dict: Dict[str, callable],
                 feature_arguments_dict: Dict[str, List[str]], argument_validation_func_dict: Dict[str, callable],
                 feature_argument_notes: Dict[str, str], window_size_func_dict: Dict[str, callable]):
        """

        :param parameters_dict:
        :param name: the main feature generator name
        :param alias: an alias to prevent column
        :param feature_name_list:
        :param feature_creation_func_dict:
        :param feature_arguments_dict:
        :param argument_validation_func_dict:
        :param feature_argument_notes:
        :param window_size_func_dict:
        """
        # ------ class level variables -----
        self.name : str = name
        self.feature_name_list: List[str] = feature_name_list
        self.alias : str = alias
        self.feature_creation_func_dict: Dict[str, callable] = feature_creation_func_dict
        self.feature_arguments_dict: Dict[str, List[str]] = feature_arguments_dict
        self.argument_validation_func_dict: Dict[str, callable] = argument_validation_func_dict
        self.feature_argument_notes: Dict[str, str] = feature_argument_notes
        self.window_size_func_dict: Dict[str, callable] = window_size_func_dict
        # ----- instance specific parameters ------
        self.parameters_dict: Dict[str, List[Dict[str, float]]] = self.remove_parameter_dict_duplicates(parameters_dict)

        # ----- perform checks -------
        self.check_parameter_dict(parameters_dict=parameters_dict, feature_name_list=self.feature_name_list,
                                  feature_arguments_dict=self.feature_arguments_dict,
                                  feature_gen_name=self.name)
        self.validate_parameter_dict(parameters_dict=parameters_dict,
                                     argument_validation_func_dict=self.argument_validation_func_dict,
                                     feature_argument_notes=self.feature_argument_notes)
        # ----- calculate window size ------
        self.required_window_size: int = self.calculate_window_size(parameters_dict = self.parameters_dict,
                                                                    feature_window_size_func_dict = self.window_size_func_dict)

    def create_features_for_data_bar(self, bar_wrapper: data.BarDataFrame) -> FeatureDataFrame:
        if len(bar_wrapper) < self.required_window_size:
            raise InSufficientWindowingData(num_data_rows=len(bar_wrapper), window_size=self.required_window_size)

    # ------ utility functions ------
    @staticmethod
    def generate_feature_name(generator_name: str, generator_alias : str, feature_name: str, arguments: Dict[str, float]) -> str:
        name_str: str = f"[{generator_name} -- {generator_alias}]--{feature_name}"
        for name, value in arguments.items():
            name_str += f"_{name}_{value}"
        return name_str

    @staticmethod
    def remove_parameter_dict_duplicates(parameters_dict: Dict[str, List[Dict[str, float]]]) -> Dict[
        str, List[Dict[str, float]]]:
        def remove_duplicates(items):
            unique_items = list()
            for item in items:
                if item not in unique_items:
                    yield item
                    unique_items.append(item)

        new_parameter_dict = {}
        for feat_name, arguments_dict_lst in parameters_dict.items():
            new_parameter_dict[feat_name] = list(remove_duplicates(arguments_dict_lst))
        return new_parameter_dict

    @staticmethod
    def check_parameter_dict(parameters_dict: Dict[str, List[Dict[str, float]]], feature_name_list: List[str],
                             feature_arguments_dict: Dict[str, List[str]], feature_gen_name: str) -> None:
        for feature_name, argument_dict_lst in parameters_dict.items():
            if feature_name not in feature_name_list:
                raise ValueError(f"{feature_name} is not valid for {feature_gen_name}")
            for argument_dict in argument_dict_lst:
                argument_names: List[str] = list(argument_dict.keys())
                expected_argument_names: List[str] = feature_arguments_dict[feature_name]
                if not argument_names == expected_argument_names:
                    raise ValueError(f"Expected arguments : {expected_argument_names} -- Recieved : {argument_names}")

    @staticmethod
    def validate_parameter_dict(parameters_dict: Dict[str, List[Dict[str, float]]],
                                argument_validation_func_dict: Dict[str, callable],
                                feature_argument_notes: Dict[str, str]):
        for feature_name, argument_dict_lst in parameters_dict.items():
            validation_func: callable = argument_validation_func_dict[feature_name]
            val_criteria: str = feature_argument_notes[feature_name]
            for argument_dict in argument_dict_lst:
                val_passed: bool = validation_func(**argument_dict)
                if not val_passed:
                    raise ValueError(
                        f"Validation failed -- Feature : {feature_name} -- Criteria : {val_criteria} -- Input : {argument_dict}")

    @staticmethod
    def calculate_window_size(parameters_dict: Dict[str, List[Dict[str, float]]],
                              feature_window_size_func_dict: Dict[str, callable]) -> int:
        window_size_lst = []
        for feature_name, argument_dict_lst in parameters_dict.items():
            window_size_func: callable = feature_window_size_func_dict[feature_name]
            for argument_dict in argument_dict_lst:
                window_size: int = window_size_func(**argument_dict)
                window_size_lst.append(window_size)
        return max(window_size_lst) if len(window_size_lst) > 0 else 0

# ----- TALIB FEATURE GENERARTORS ------
class TalibFeatureGenerator(FeatureGenerator):
    def __init__(self, parameters_dict: Dict[str, List[Dict[str, float]]], name: str, alias : str,
                 feature_name_list: List[str], feature_creation_func_dict: Dict[str, callable],
                 feature_arguments_dict: Dict[str, List[str]], argument_validation_func_dict: Dict[str, callable],
                 feature_argument_notes: Dict[str, str], window_size_func_dict: Dict[str, callable]):
        super().__init__(parameters_dict, name, alias, feature_name_list, feature_creation_func_dict,
                         feature_arguments_dict, argument_validation_func_dict, feature_argument_notes,
                         window_size_func_dict)

    @Override
    def create_features_for_data_bar(self, bar_wrapper: data.BarDataFrame) -> FeatureDataFrame:
        super().create_features_for_data_bar(bar_wrapper)
        bar_df_ref: pd.DataFrame = bar_wrapper.get_bar_data_reference()
        open_series: pd.Series = bar_df_ref[data.BarDataColumns.OPEN.value]
        high_series: pd.Series = bar_df_ref[data.BarDataColumns.HIGH.value]
        low_series: pd.Series = bar_df_ref[data.BarDataColumns.LOW.value]
        close_series: pd.Series = bar_df_ref[data.BarDataColumns.CLOSE.value]
        feature_df: pd.DataFrame = self.create_talib_features(open_series, high_series, low_series, close_series)
        feature_info = FeatureInfo(feature_gen_names ={f"{self.name} -- {self.alias}"},
                                   bar_descriptions = {str(bar_wrapper)})
        feature_wrapper: FeatureDataFrame = FeatureDataFrame(feature_df=feature_df, feature_info = feature_info)
        # ----- log feature creation completion ------
        ml_logger.log_bar_feature_creation(feature_gen_name=TalibMomentum.NAME, bar_wrapper=bar_wrapper)
        return feature_wrapper

    def create_talib_features(self, open_series: pd.Series, close_series: pd.Series, high_series: pd.Series,
                              low_series: pd.Series) -> pd.DataFrame:
        feature_df = pd.DataFrame(index=open_series.index)
        for feature_name, argument_dict_lst in self.parameters_dict.items():
            for argument_dict in argument_dict_lst:
                feat_name_str: str = FeatureGenerator.generate_feature_name(generator_name=self.name,
                                                                            generator_alias=self.alias,
                                                                            feature_name=feature_name,
                                                                            arguments=argument_dict)
                feat_gen_func: callable = self.feature_creation_func_dict[feature_name]
                feature_df[feat_name_str] = feat_gen_func(open=open_series, close=close_series, high=high_series,
                                                          low=low_series, **argument_dict)
        return feature_df


class TalibMomentum(TalibFeatureGenerator):
    NAME: str = "Talib Momentum Indicators"
    def __init__(self, parameters_dict: Dict[str, List[Dict[str, float]]], alias : str):
        super().__init__(parameters_dict = parameters_dict,
                         name= TalibMomentum.NAME,
                         alias= alias,
                         feature_name_list = talib_mom_data.FEATURE_NAME_LIST,
                         feature_creation_func_dict = talib_mom_data.FEATURE_CREATION_FUNCTION_DICT,
                         feature_arguments_dict = talib_mom_data.FEATURE_ARGUMENTS_DICT,
                         feature_argument_notes = talib_mom_data.FEATURE_ARGUMENT_NOTES,
                         argument_validation_func_dict = talib_mom_data.ARGUMENT_VALIDATION_FUNC_DICT,
                         window_size_func_dict = talib_mom_data.FEATURE_WINDOW_SIZE_FUNC_DICT)


class TalibVolatility(TalibFeatureGenerator):
    NAME: str = "Talib Volatility indicators"
    def __init__(self, parameters_dict : Dict[str, List[Dict[str, float]]], alias : str):
        super().__init__(parameters_dict = parameters_dict,
                         name= TalibVolatility.NAME,
                         alias= alias,
                         feature_name_list = talib_vol_data.FEATURE_NAME_LIST,
                         feature_creation_func_dict = talib_vol_data.FEATURE_CREATION_FUNCTION_DICT,
                         feature_arguments_dict = talib_vol_data.FEATURE_ARGUMENTS_DICT,
                         feature_argument_notes = talib_vol_data.FEATURE_ARGUMENT_NOTES,
                         argument_validation_func_dict = talib_vol_data.ARGUMENT_VALIDATION_FUNC_DICT,
                         window_size_func_dict = talib_vol_data.FEATURE_WINDOW_SIZE_FUNC_DICT)


class TalibPattern(TalibFeatureGenerator):
    NAME: str = "Talib Pattern generator "
    def __init__(self, parameters_dict: Dict[str, List[Dict[str, float]]], alias):
        super().__init__(parameters_dict = parameters_dict,
                         name= TalibPattern.NAME,
                         alias= alias,
                         feature_name_list = talib_pat_data.FEATURE_NAME_LIST,
                         feature_creation_func_dict = talib_pat_data.FEATURE_CREATION_FUNCTION_DICT,
                         feature_arguments_dict = talib_pat_data.FEATURE_ARGUMENTS_DICT,
                         feature_argument_notes = talib_pat_data.FEATURE_ARGUMENT_NOTES,
                         argument_validation_func_dict = talib_pat_data.ARGUMENT_VALIDATION_FUNC_DICT,
                         window_size_func_dict = talib_pat_data.FEATURE_WINDOW_SIZE_FUNC_DICT)


# ------ FRACTIONAL DIFFERENCING --------------
class FracDiffBarDataFrame:
    def __init__(self, bar_wrapper: data.BarDataFrame, window_size: int, d: float):
        self.bar_wrapper: data.BarDataFrame = bar_wrapper
        self.window_size: int = window_size
        self.d: float = d

    def get_bar_data_reference(self) -> pd.DataFrame:
        return self.bar_wrapper.get_bar_data_reference()

    def get_bar_data_copy(self) -> pd.DataFrame:
        return self.bar_wrapper.get_bar_data_copy()

    def __str__(self):
        return str(self.bar_wrapper) + " -- " + "Frac diff : " + " d = " + str(self.d) + " window_size = " + str(
            self.window_size)


def get_frac_weights(n: int, d: float) -> np.array:
    weights = []
    w = 1
    for k in range(1, n):
        weights.append(w)
        w = -w / k * (d - k + 1)
    weights.append(w)
    return np.array(weights)[::-1]


def frac_diff(series: pd.Series, n: int, d: float) -> pd.Series:
    """

    :param series: price series to be fractionally differentiated
    :param n: size of the rolling window
    :param d: fraction diff
    :return: fractionally differenced series
    NOTE : the series returned has missing values equal to the window required to fractionally difference
    """
    weights = get_frac_weights(n, d)
    return series.rolling(n).apply(lambda row: np.dot(row.values, weights))


def frac_diff_bar_data_frame(bar_df: pd.DataFrame, window_size: int, d: float) -> pd.DataFrame:
    new_bar_df: pd.DataFrame = bar_df.copy()
    new_bar_df[data.BarDataColumns.OPEN.value] = frac_diff(new_bar_df[data.BarDataColumns.OPEN.value], window_size, d)
    new_bar_df[data.BarDataColumns.CLOSE.value] = frac_diff(new_bar_df[data.BarDataColumns.CLOSE.value], window_size, d)
    new_bar_df[data.BarDataColumns.HIGH.value] = frac_diff(new_bar_df[data.BarDataColumns.HIGH.value], window_size, d)
    new_bar_df[data.BarDataColumns.LOW.value] = frac_diff(new_bar_df[data.BarDataColumns.LOW.value], window_size, d)
    new_bar_df[data.BarDataColumns.VWAP.value] = frac_diff(new_bar_df[data.BarDataColumns.VWAP.value], window_size, d)
    return new_bar_df


def frac_diff_bar_data_frame_wrapper(bar_df_wrapper: data.BarDataFrame, window_size: int,
                                     d: float) -> FracDiffBarDataFrame:
    bar_df_ref = bar_df_wrapper.get_bar_data_reference()
    new_bar_df = frac_diff_bar_data_frame(bar_df=bar_df_ref, window_size=window_size, d=d)
    new_bar_wrapper = bar_df_wrapper.create_empty_copy()
    new_bar_wrapper.set_data_frame(bar_df=new_bar_df, deep_copy=False)
    ml_logger.log_bar_frac_diff(bar_wrapper=bar_df_wrapper, window_size=window_size, d=d)
    return FracDiffBarDataFrame(bar_wrapper=new_bar_wrapper, window_size=window_size, d=d)


# ----- CUSTOM EXCEPTIONS -------
class InSufficientWindowingData(Exception):
    def __init__(self, num_data_rows: int, window_size: int):
        super().__init__(f"Required data rows for windowing : {window_size} -- provided data rows : {num_data_rows}")
