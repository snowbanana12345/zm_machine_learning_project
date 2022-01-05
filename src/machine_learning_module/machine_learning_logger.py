import logging
import definitions
import os
import src.data_base_module.data_blocks as data
from typing import Dict, List, Set

# ------- initialize a logger for this specific module -------
LOGGER = logging.getLogger("machine_learning_module")
LOGGER.setLevel(level=logging.INFO)
file_handler = logging.FileHandler(os.path.join(definitions.LOGS_FOLDER_PATH, "machine_learning.log"))
formatter = logging.Formatter("[%(asctime)s] : %(levelname)s : [%(name)s] : %(message)s ")
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
LOGGER.addHandler(file_handler)
LOGGER.addHandler(stream_handler)


# ------- feature generator logger -------
feature_submodule_str = "SubModule : feature_generators --"
def log_bar_feature_creation(feature_gen_name : str, bar_wrapper : data.BarDataFrame) -> None:
    LOGGER.info(feature_submodule_str + f"{feature_gen_name} completed feature creation on : {bar_wrapper}")

def warn_unequal_datasets(function_name : str):
    LOGGER.warning(feature_submodule_str + f"Function : {function_name} -- datasets used are different")

def warn_datasets_num_rows_unequal(function_name : str):
    LOGGER.warning(feature_submodule_str + f"Function : {function_name} -- datasets do not have the same number of rows, rows with nan may be produced")

def warn_feature_param_duplicate(function_name : str, feature_name, argument_dict : Dict[str, List[float]]):
    LOGGER.warning(feature_submodule_str + f"Function : {function_name} -- feature name : {feature_name} -- params : {argument_dict}")

def warn_feature_param_unequal(function_name : str):
    LOGGER.warning(feature_submodule_str + f"Function : {function_name} -- dataset parameters")

def warn_dataset_duplicates(num_intersections, function_name):
    LOGGER.warning(feature_submodule_str + f"Function : {function_name} -- number of dataset intersections {num_intersections}")

def warn_feature_gen_name_lst_unequal(difference : Set[str], function_name : str):
    LOGGER.warning(feature_submodule_str + f"Function : {function_name} -- Difference : {difference}")

def log_bar_frac_diff(bar_wrapper : data.BarDataFrame, window_size : int, d : float) -> None:
    LOGGER.info(feature_submodule_str + "Frac diff : d = " + str(d) + " window_size = " + str(window_size) + " on : " + str(bar_wrapper))

# ------ label generator logger -------
label_submodule_str = "SubModule : label_generators --"
def log_bar_label_creation(label_gen_name : str, bar_wrapper : data.BarDataFrame) -> None:
    LOGGER.info(label_submodule_str + label_gen_name + " completed label creation on : " + str(bar_wrapper))

# ------ filter generator logger ------
filter_submodule_str = "SubModule : filter_generators --"
def log_bar_filter_creation(filter_gen_name : str, bar_wrapper : data.BarDataFrame) -> None:
    LOGGER.info(filter_submodule_str + filter_gen_name + " completed filter creation on : " + str(bar_wrapper))

# ----- pipeline logger ------
pipeline_submodule_str = "SubModule : pipeline"
def log_filter_feature_label(dataset_description : str):
    LOGGER.info(pipeline_submodule_str + f"complete feature and label filtering on : {dataset_description}")

def log_cross_val_completion(number_completed : int):
    LOGGER.info(pipeline_submodule_str + f"complete cross validating train test set : {number_completed}")

# ----- boot strap logger -----
bootstrap_submodule_str = "SubModule : bootstraping"
def log_boot_strap(boot_strap_gen_name : str, label_data_frame_description : str, filter_description : str):
    LOGGER.info(bootstrap_submodule_str + f"{boot_strap_gen_name} bootstrapped {label_data_frame_description} with filter : {filter_description}")

def log_find_average_uniqueness(label_description : str, filter_description : str, average_uniqueness : float, bootstrap_description : str):
    LOGGER.info(bootstrap_submodule_str + f"found average uniqueness : {average_uniqueness} -- bootstrap : {bootstrap_description} -- label : {label_description} -- filter : {filter_description}")

