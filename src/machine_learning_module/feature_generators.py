import pandas as pd
import numpy as np
from typing import List, Dict, Set, Hashable, Tuple, Iterator
import itertools
import src.machine_learning_module.machine_learning_logger as ml_logger
import src.data_base_module.data_blocks as data
import random

# ------ import feature generator data -------
import src.machine_learning_module.feature_generator_data.talib_momentum as talib_mom_data
import src.machine_learning_module.feature_generator_data.talib_volatility as talib_vol_data
import src.machine_learning_module.feature_generator_data.talib_pattern as talib_pat_data

# ------ FEATURE DATA FRAME --------
class FeatureDataFrame:
    """
    A parameter dict is in the format :
    Dict[ feature name , List[Dict[argument name, argument value]]
    """

    def __init__(self, feature_df: pd.DataFrame, parameter_dict: Dict[str, List[Dict[str, float]]],
                 feature_generator_name_lst: List[str], dataset_description_lst: List[str]):
        self.feature_df: pd.DataFrame = feature_df
        self.parameter_dict: Dict[str, List[Dict[str, float]]] = parameter_dict
        self.feature_generator_name_lst: List[str] = feature_generator_name_lst
        self.dataset_description_lst: List[str] = dataset_description_lst
        self.num_features = len(feature_df.columns)
        self.feature_df.index = pd.RangeIndex(len(self.feature_df))

    def get_feature_df_ref(self) -> pd.DataFrame:
        return self.feature_df

    def get_feature_df_copy(self) -> pd.DataFrame:
        return self.feature_df.copy()

    def __len__(self):
        return len(self.feature_df)


EMPTY_FEATURE_DATA_FRAME: FeatureDataFrame = FeatureDataFrame(feature_df=pd.DataFrame(), parameter_dict={},
                                                              feature_generator_name_lst=[], dataset_description_lst=[])


def concat_feature_df_horizontal(feature_wrapper_lst: List[FeatureDataFrame]) -> FeatureDataFrame:
    """
    horizontally binds a list feature dataframes together
    The intented use for this function is to concatenate feature data frames that were created from the same set of datasets
    NOTE : when there are duplicate feature arguments, this function will remove duplicates in the parameter dict and take the first
    column in the input feature wrapper lst
    NOTE : use the same number of rows, this function produces a dataframe with missing values if dataframes with different rows are used.
    :param feature_wrapper_lst: List of FeatureDataFrames
    :return: FeatureDataFrame
    """
    # ----- check if the input provided is empty -----
    if len(feature_wrapper_lst) == 0:
        return EMPTY_FEATURE_DATA_FRAME
    # ----- unpack each component -----
    new_feature_df_lst: List[pd.DataFrame] = [feature_wrapper.get_feature_df_ref() for feature_wrapper in
                                              feature_wrapper_lst]
    new_feature_generator_name_lst: List[str] = list(
        itertools.chain(*[feature_wrapper.feature_generator_name_lst for feature_wrapper in feature_wrapper_lst]))
    new_parameter_dict_lst: List[Dict[str, List[Dict[str, float]]]] = [feature_wrapper.parameter_dict for
                                                                       feature_wrapper in feature_wrapper_lst]
    new_dataset_description_lst: List[str] = list(
        itertools.chain(*[feature_wrapper.dataset_description_lst for feature_wrapper in feature_wrapper_lst]))
    # ----- checks ------
    check_dataset_descriptions_equal(feature_wrapper_lst, function_name="concat_feature_df_horizontal")
    check_dataset_rows_equal(feature_wrapper_lst, function_name="concat_feature_df_horizontal")
    check_parameter_dict_duplicates(feature_wrapper_lst, function_name="concat_feature_df_horizontal")
    # ----- clear duplicates from feature generators name list and dataset description list -----
    new_feature_generator_name_lst = list(dict.fromkeys(new_feature_generator_name_lst))
    new_dataset_description_lst = list(dict.fromkeys(new_feature_generator_name_lst))
    # ----- join parameter dict together -------
    new_parameter_dict: Dict[str, List[Dict[str, float]]] = merge_parameter_dicts(new_parameter_dict_lst)
    # ----- join feature data frames ------
    new_feature_df: pd.DataFrame = merge_df_lst_without_duplicate_col_names(new_feature_df_lst)
    # ----- create the new feature data frame object ------
    new_feature_wrapper: FeatureDataFrame = FeatureDataFrame(feature_df=new_feature_df,
                                                             parameter_dict=new_parameter_dict,
                                                             feature_generator_name_lst=new_feature_generator_name_lst,
                                                             dataset_description_lst=new_dataset_description_lst)
    return new_feature_wrapper


def concat_feature_df_vertical(feature_wrapper_lst: List[FeatureDataFrame]) -> FeatureDataFrame:
    """
    vertically binds a feature dataframe together
    The intended use of this function is on feature dataframes that are produced from the same feature generators
    NOTE : the resulting feature dataframe will be ordered according to the order in the feature wrapper lst
    NOTE : using this function on feature dataframes that have different columns will result in missing values
    :param feature_wrapper_lst: List of FeatureDataFrames
    :return: FeatureDataFrame
    """
    # ----- check if the input provided is empty -----
    if len(feature_wrapper_lst) == 0:
        return EMPTY_FEATURE_DATA_FRAME
    # ----- unpack each component -----
    new_feature_df_lst: List[pd.DataFrame] = [feature_wrapper.get_feature_df_ref() for feature_wrapper in
                                              feature_wrapper_lst]
    new_feature_generator_name_lst: List[str] = list(
        itertools.chain(*[feature_wrapper.feature_generator_name_lst for feature_wrapper in feature_wrapper_lst]))
    new_parameter_dict_lst: List[Dict[str, List[Dict[str, float]]]] = [feature_wrapper.parameter_dict for
                                                                       feature_wrapper in feature_wrapper_lst]
    new_dataset_description_lst: List[str] = list(
        itertools.chain(*[feature_wrapper.dataset_description_lst for feature_wrapper in feature_wrapper_lst]))
    # ----- checks ------
    check_parameter_dicts_equal(feature_wrapper_lst, function_name="concat_feature_df_vertical")
    check_dataset_duplicates(feature_wrapper_lst, function_name="concat_feature_df_vertical")
    check_feature_gen_name_lst_equal(feature_wrapper_lst, function_name="concat_feature_df_vertical")
    # ----- clear duplicates from feature generators name list and dataset description list -----
    new_feature_generator_name_lst = list(dict.fromkeys(new_feature_generator_name_lst))
    new_dataset_description_lst = list(dict.fromkeys(new_feature_generator_name_lst))
    # ----- join parameter dict together -------
    new_parameter_dict: Dict[str, List[Dict[str, float]]] = merge_parameter_dicts(new_parameter_dict_lst)
    # ----- join feature data frames ------
    new_feature_df: pd.DataFrame = pd.concat(new_feature_df_lst, axis=0)
    # ----- create the new feature data frame object ------
    new_feature_wrapper: FeatureDataFrame = FeatureDataFrame(feature_df=new_feature_df,
                                                             parameter_dict=new_parameter_dict,
                                                             feature_generator_name_lst=new_feature_generator_name_lst,
                                                             dataset_description_lst=new_dataset_description_lst)
    return new_feature_wrapper


def check_dataset_descriptions_equal(feature_wrapper_lst: List[FeatureDataFrame],
                                     function_name: str = "function") -> None:
    for description_lst_1, description_lst_2 in itertools.combinations(
            [feature_wrapper.dataset_description_lst for feature_wrapper in feature_wrapper_lst], 2):
        if description_lst_1 != description_lst_2:
            ml_logger.warn_unequal_datasets(function_name=function_name)


def check_dataset_rows_equal(feature_wrapper_lst: List[FeatureDataFrame], function_name: str = "function") -> None:
    feature_df_size_lst: List[int] = [len(feature_wrapper) for feature_wrapper in feature_wrapper_lst]
    for num_row1, num_row2 in itertools.combinations(feature_df_size_lst, 2):
        if not num_row1 == num_row2:
            ml_logger.warn_datasets_num_rows_unequal(function_name=function_name)


def check_parameter_dict_duplicates(feature_wrapper_lst: List[FeatureDataFrame],
                                    function_name: str = "function") -> None:
    new_parameter_dict_lst: List[Dict[str, List[Dict[str, float]]]] = [feature_wrapper.parameter_dict for
                                                                       feature_wrapper in feature_wrapper_lst]
    for parameter_dict_1, parameter_dict_2 in itertools.combinations(new_parameter_dict_lst, 2):
        for (feat_name_1, arguments_lst_dict_1), (feat_name_2, arguments_lst_dict_2) in itertools.product(
                parameter_dict_1.items(), parameter_dict_2.items()):
            for argument_dict_1, argument_dict2 in itertools.product(arguments_lst_dict_1, arguments_lst_dict_2):
                if argument_dict_1 == argument_dict2 and feat_name_1 == feat_name_2:
                    ml_logger.warn_feature_param_duplicate(function_name=function_name, feature_name=feat_name_1,
                                                           argument_dict=argument_dict_1)


def merge_parameter_dicts(parameter_dict_lst: List[Dict[str, List[Dict[str, float]]]]) -> Dict[
    str, List[Dict[str, float]]]:
    """
    merges a list of parameter dictionaries and removes all duplicates
    :param parameter_dict_lst: list of parameter dicts
    :return: a single parameter_dict
    """
    feature_names: Set[str] = set([key for keyview in [dic.keys() for dic in parameter_dict_lst] for key in keyview])
    new_parameter_dict: Dict[str, List[Dict[str, float]]] = {feat_name: [] for feat_name in feature_names}
    for feat_name in feature_names:
        for parameter_dict in parameter_dict_lst:
            new_parameter_dict[feat_name].extend(parameter_dict[feat_name])
    for feat_name in feature_names:
        new_parameter_dict[feat_name] = list(dict.fromkeys(new_parameter_dict[feat_name]))
    return new_parameter_dict


def merge_df_lst_without_duplicate_col_names(df_lst: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge a list of pandas dataframes into a dataframe while removing duplicate columns keeping only the first dataframe in the list
    to have it.
    :param df_lst: a nonempty list of pandas data frames with possiblely duplicate column names
    :return: a single pandas dataframe
    """
    new_df = df_lst[0]
    for feature_df in df_lst[1:]:
        col_to_use: pd.Index = feature_df.columns.difference(new_df.columns)
        new_df = new_df.merge(right=new_df, left=feature_df[col_to_use], how="outer", axis=1)
    return new_df


def check_parameter_dicts_equal(feature_wrapper_lst: List[FeatureDataFrame], function_name: str = "function"):
    for parameter_dict_1, parameter_dict_2 in itertools.combinations(feature_wrapper_lst, 2):
        if not parameter_dict_1 == parameter_dict_2:
            ml_logger.warn_feature_param_unequal(function_name=function_name)
            break


def check_dataset_duplicates(feature_wrapper_lst: List[FeatureDataFrame], function_name: str = "function"):
    description_lst: List[List[str]] = [feature_wrapper.dataset_description_lst for feature_wrapper in
                                        feature_wrapper_lst]
    for lst1, lst2 in itertools.combinations(description_lst, 2):
        intersection: Set[str] = set(lst1) & set(lst2)
        if len(intersection) != 0:
            ml_logger.warn_dataset_duplicates(num_intersections=len(intersection), function_name=function_name)
            break


def check_feature_gen_name_lst_equal(feature_wrapper_lst: List[FeatureDataFrame], function_name: str = "function"):
    feature_gen_name_lst: List[List[str]] = [feature_wrapper.feature_generator_name_lst for feature_wrapper in
                                             feature_wrapper_lst]
    for lst1, lst2 in itertools.combinations(feature_gen_name_lst, 2):
        difference: Set[str] = (set(lst1) - set(lst2)).union(set(lst2) - set(lst1))
        if len(difference) != 0:
            ml_logger.warn_feature_gen_name_lst_unequal(difference=difference, function_name=function_name)


# ------- FEATURE GENERATORS  -------

class FeatureGenerator:
    """ abstract class feature generator """
    def __init__(self, parameters_dict: Dict[str, List[Dict[str, float]]],
                 feature_gen_name : str,
                 feature_name_list: List[str],
                 feature_creation_func_dict: Dict[str, callable],
                 feature_arguments_dict: Dict[str, List[str]],
                 argument_validation_func_dict : Dict[str, callable],
                 feature_argument_notes: Dict[str, str],
                 window_size_func_dict: Dict[str, callable]):
        # ------ class level variables -----
        self.feature_gen_name : str = feature_gen_name
        self.feature_name_list: List[str] = feature_name_list
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
                                  feature_gen_name=self.feature_gen_name)
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
    def generate_feature_name(generator_name: str, feature_name: str, arguments: Dict[str, float]) -> str:
        name_str: str = f"[{generator_name}]--{feature_name}"
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
    def __init__(self, parameters_dict: Dict[str, List[Dict[str, float]]], feature_gen_name: str,
                 feature_name_list: List[str], feature_creation_func_dict: Dict[str, callable],
                 feature_arguments_dict: Dict[str, List[str]], argument_validation_func_dict: Dict[str, callable],
                 feature_argument_notes: Dict[str, str], window_size_func_dict: Dict[str, callable]):
        super().__init__(parameters_dict, feature_gen_name, feature_name_list, feature_creation_func_dict,
                         feature_arguments_dict, argument_validation_func_dict, feature_argument_notes,
                         window_size_func_dict)

    def create_features_for_data_bar(self, bar_wrapper: data.BarDataFrame) -> FeatureDataFrame:
        super().create_features_for_data_bar(bar_wrapper)
        bar_df_ref: pd.DataFrame = bar_wrapper.get_bar_data_reference()
        open_series: pd.Series = bar_df_ref[data.BarDataColumns.OPEN.value]
        high_series: pd.Series = bar_df_ref[data.BarDataColumns.HIGH.value]
        low_series: pd.Series = bar_df_ref[data.BarDataColumns.LOW.value]
        close_series: pd.Series = bar_df_ref[data.BarDataColumns.CLOSE.value]
        feature_df: pd.DataFrame = self.create_talib_features(open_series, high_series, low_series, close_series)
        feature_wrapper: FeatureDataFrame = FeatureDataFrame(feature_df=feature_df, parameter_dict=self.parameters_dict,
                                                             feature_generator_name_lst=[TalibMomentum.NAME],
                                                             dataset_description_lst=[str(bar_wrapper)])
        # ----- log feature creation completion ------
        ml_logger.log_bar_feature_creation(feature_gen_name=TalibMomentum.NAME, bar_wrapper=bar_wrapper)
        return feature_wrapper

    def create_talib_features(self, open_series: pd.Series, close_series: pd.Series, high_series: pd.Series,
                              low_series: pd.Series) -> pd.DataFrame:
        feature_df = pd.DataFrame(index=open_series.index)
        for feature_name, argument_dict_lst in self.parameters_dict.items():
            for argument_dict in argument_dict_lst:
                feat_name_str: str = FeatureGenerator.generate_feature_name(generator_name=TalibMomentum.NAME,
                                                                            feature_name=feature_name,
                                                                            arguments=argument_dict)
                feat_gen_func: callable = self.feature_creation_func_dict[feature_name]
                feature_df[feat_name_str] = feat_gen_func(open=open_series, close=close_series, high=high_series,
                                                          low=low_series, **argument_dict)
        return feature_df


class TalibMomentum(TalibFeatureGenerator):
    NAME: str = "Talib Momentum Indicators"
    def __init__(self, parameters_dict: Dict[str, List[Dict[str, float]]]):
        super().__init__(parameters_dict = parameters_dict,
                         feature_gen_name = TalibMomentum.NAME,
                         feature_name_list = talib_mom_data.FEATURE_NAME_LIST,
                         feature_creation_func_dict = talib_mom_data.FEATURE_CREATION_FUNCTION_DICT,
                         feature_arguments_dict = talib_mom_data.FEATURE_ARGUMENTS_DICT,
                         feature_argument_notes = talib_mom_data.FEATURE_ARGUMENT_NOTES,
                         argument_validation_func_dict = talib_mom_data.ARGUMENT_VALIDATION_FUNC_DICT,
                         window_size_func_dict = talib_mom_data.FEATURE_WINDOW_SIZE_FUNC_DICT)


class TalibVolatility(TalibFeatureGenerator):
    NAME: str = "Talib Volatility indicators"
    def __init__(self, parameters_dict : Dict[str, List[Dict[str, float]]]):
        super().__init__(parameters_dict = parameters_dict,
                         feature_gen_name = TalibVolatility.NAME,
                         feature_name_list = talib_vol_data.FEATURE_NAME_LIST,
                         feature_creation_func_dict = talib_vol_data.FEATURE_CREATION_FUNCTION_DICT,
                         feature_arguments_dict = talib_vol_data.FEATURE_ARGUMENTS_DICT,
                         feature_argument_notes = talib_vol_data.FEATURE_ARGUMENT_NOTES,
                         argument_validation_func_dict = talib_vol_data.ARGUMENT_VALIDATION_FUNC_DICT,
                         window_size_func_dict = talib_vol_data.FEATURE_WINDOW_SIZE_FUNC_DICT)


class TalibPattern(TalibFeatureGenerator):
    NAME: str = "Talib Pattern generator "
    def __init__(self, parameters_dict: Dict[str, List[Dict[str, float]]]):
        super().__init__(parameters_dict = parameters_dict,
                         feature_gen_name = TalibPattern.NAME,
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







