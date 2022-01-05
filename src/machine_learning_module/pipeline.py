from typing import Set, Dict, List, Tuple
from src.general_module.custom_exceptions import ArrayLengthMisMatchException
import sklearn.base as sklearn_base
import pandas as pd
import numpy as np
import itertools
import src.machine_learning_module.feature_generators as feature_gen_mod
import src.machine_learning_module.label_generators as label_gen_mod
import src.machine_learning_module.filter_generators as filter_gen_mod
import src.machine_learning_module.machine_learning_logger as ml_logger
import src.machine_learning_module.performance_metrics as perf_mod


def filter_features_and_labels(feature_wrapper : feature_gen_mod.FeatureDataFrame, label_wrapper : label_gen_mod.LabelDataFrame, filter_array : filter_gen_mod.FilterArray,
                               dataset_description : str) -> (feature_gen_mod.FeatureDataFrame, label_gen_mod.LabelDataFrame):
    """
    Applies filter to both the feature data frame and label array
    Removes all row with missing values from feature data frame as these rows are due to windowing
    Removes all rows with
    :param feature_wrapper:
    :param label_wrapper:
    :param filter_array:
    :return: data frame containing filtered rows of the
    """
    # ------- check array sizing ------
    if len(feature_wrapper) != len(label_wrapper):
        raise ArrayLengthMisMatchException(array_1_length = len(feature_wrapper), array_2_length = len(label_wrapper),
                                           array_1_description = "feature data frame ", array_2_description = "label series")
    if len(feature_wrapper) != len(filter_array):
        raise ArrayLengthMisMatchException(array_1_length=len(feature_wrapper), array_2_length=len(filter_array),
                                           array_1_description="feature data frame ",
                                           array_2_description="filter array")
    # ------- compute set of indices that can be used as training examples ------
    feature_df : pd.DataFrame = feature_wrapper.get_feature_df_ref()
    label_series : pd.Series = label_wrapper.get_label_series_ref()
    look_ahead_series : pd.Series = label_wrapper.get_look_ahead_series_ref()
    valid_feature_df_indices : Set[int] = set(feature_df.dropna().index)
    valid_filter_indices : Set[int] = set(filter_array.get_sampled_indices())
    valid_label_indices : Set[int] = set(label_series.dropna().index)
    valid_indices : Set[int] = valid_feature_df_indices.intersection(valid_filter_indices).intersection(valid_label_indices)
    sorted_valid_indices : List[int] = list(sorted(valid_indices))
    # ------ apply filter --------
    filtered_feature_df : pd.DataFrame = feature_df.loc[sorted_valid_indices]
    filtered_label_series : pd.Series = label_series.loc[sorted_valid_indices]
    filtered_look_ahead_series : pd.Series = label_series.loc[sorted_valid_indices]
    # ----- logging -----
    ml_logger.log_filter_feature_label(dataset_description = dataset_description)
    # ----- rewrap filtered feature and label data frames -----
    new_filtered_feature_wrapper = feature_gen_mod.FeatureDataFrame(feature_df = filtered_feature_df, parameter_dict = feature_wrapper.parameter_dict)
    new_filtered_label_wrapper = label_gen_mod.LabelDataFrame(label_series = filtered_label_series, look_ahead_series = filtered_look_ahead_series
                                                              ,label_gen_name = label_wrapper.label_gen_name, )
    return filtered_feature_df, filtered_label_series

def convert_to_one_vs_all_labeling(label_array : np.array, classes : Set[int], default_class : int) -> Dict[int, np.array]:
    classes = classes.copy()
    classes.remove(default_class)
    result : Dict[int : np.array] = {}
    for cls in classes:
        result[cls] = np.array([1 if label == cls else 0 for label in label_array])
    return result

def find_cross_val_sets_indices(num_datasets : int, num_test_sets : int, shift : int) -> List[Tuple[List[int], List[int]]]:
    """
    This function assumes that the original data sets has been broken up into several subsets
    Example, numdataset = 10, num_testsets = 3, shift = 1 results in the following sets
    1 represents train set, 0 represents test set
    1000111111, 1111000111, 1111111000,
    :param num_datasets: num of data sets
    :param num_test_sets: num of test sets used for
    :param shift: the first position to start taking the test set
    :return: a list of tuples of sets of indices, the tuple is in the format (train set indices, test set indices )
    """
    num_cross_vals : int = (num_datasets - shift) // num_test_sets
    cross_val_sets = []
    for i in range(num_cross_vals):
        test_start_point : int = shift + i * num_test_sets
        test_end_point : int = test_start_point + num_test_sets
        cross_val_sets.append(([*itertools.chain(range(test_start_point), range(test_end_point, num_datasets))]
                               , [*range(test_start_point, test_end_point)]))
    return cross_val_sets

def cross_validate_binary_classification(feature_df_lst : List[feature_gen_mod.FeatureDataFrame]
                                         , label_array_lst : List[label_gen_mod.LabelDataFrame], ml_model_template
                                    , cross_val_indices : List[Tuple[List[int], List[int]]]) \
        -> Tuple[List[perf_mod.BinaryClassificationMetrics], List[perf_mod.BinaryClassificationMetrics]]:
    """

    :param feature_df_lst: list of feature data frames
    :param label_array_lst: list of the arrays of labels
    :param ml_model_template: an sklearn model, it has the two functions, fit() and predict(), copies of the ml model will be made
    :param cross_val_indices: a list of
    :return: a tuple of the train and test list of BinaryClassificationMetrics object
    NOTE : do not pass in an ml model that has already been fit as ml_model_template
    """
    train_cross_val_result_lst : List[perf_mod.BinaryClassificationMetrics] = []
    test_cross_val_result_lst : List[perf_mod.BinaryClassificationMetrics] = []
    for number_completed, (train_indices, test_indices) in enumerate(cross_val_indices):
        ml_model = sklearn_base.clone(ml_model_template)
        train_feature_df : pd.DataFrame = pd.concat([feature_df_lst[index].get_feature_df_ref() for index in train_indices], ignore_index=True, axis = 0)
        test_feature_df : pd.DataFrame = pd.concat([feature_df_lst[index].get_feature_df_ref() for index in test_indices], ignore_index=True, axis = 0)
        train_label_series : pd.Series = pd.concat([label_array_lst[index].get_label_series_ref() for index in train_indices], axis = 0)
        test_label_series : pd.Series = pd.concat([label_array_lst[index].get_label_series_ref() for index in test_indices], axis = 0)
        ml_model.fit(train_feature_df, train_label_series)
        train_prediction_array : np.array = ml_model.predict(train_feature_df)
        test_prediction_array : np.array = ml_model.predict(test_feature_df)
        train_cross_val_result = perf_mod.find_binary_classification_metrics(train_prediction_array, train_label_series)
        test_cross_val_result = perf_mod.find_binary_classification_metrics(test_prediction_array, test_label_series)
        train_cross_val_result_lst.append(train_cross_val_result)
        test_cross_val_result_lst.append(test_cross_val_result)
        ml_logger.log_cross_val_completion(number_completed = number_completed)
    return train_cross_val_result_lst, test_cross_val_result_lst



