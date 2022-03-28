import unittest
import src.machine_learning_module.feature_generators as feat_gen_mod
import src.data_base_module.data_blocks as dat_blocks
from typing import Dict, List
import pandas as pd

feature_name_list_stud : List[str] = ["Tom", "Jerry"]
feature_arguments_dict_stud : Dict[str, List[str]] = {"Tom" : ["clicks", "mice"], "Jerry" : ["milk", "cookies"]}

def tom_func(series, clicks, mice):
    return series * clicks - mice

def jerry_func(series, milk, cookies):
    return series + milk + cookies

feature_creation_func_dict_stud : Dict[str, callable] = {
    "Tom" : lambda series, clicks, mice : tom_func(series, clicks, mice),
    "Jerry" : lambda series, milk, cookies : jerry_func(series, milk, cookies)
}

argument_validation_func_dict_stud : Dict[str, callable] = {
    "Tom" : lambda clicks, mice : clicks != 0 and mice != 0,
    "Jerry" : lambda milk, cookies : milk != 0 and cookies != 0
}

feature_argument_notes_stud : Dict[str, str] = {
    "Tom" : "Tom likes mice",
    "Jerry": "Jerry likes milk"
}

window_size_func_stud : Dict[str, callable] = {
    "Tom" : lambda clicks, mice : 1,
    "Jerry" : lambda milk, cookies : 1,
}

class FeatureGeneratorSTUD(feat_gen_mod.FeatureGenerator):
    def __init__(self, parameters_dict: Dict[str, List[Dict[str, float]]], alias: str, price_series_used : dat_blocks.BarDataColumns):
        super().__init__(parameters_dict = parameters_dict,
                         name = "Tom and jerry",
                         alias = alias,
                         feature_name_list = feature_name_list_stud,
                         feature_creation_func_dict = feature_creation_func_dict_stud,
                         feature_arguments_dict = feature_arguments_dict_stud,
                         argument_validation_func_dict = feature_arguments_dict_stud,
                         feature_argument_notes = feature_argument_notes_stud,
                         window_size_func_dict = window_size_func_stud)
        self.price_series_used = price_series_used

class FeatureDataFrameTest(unittest.TestCase):
    def setUp(self):
        self.bar_df = pd.DataFrame({
            dat_blocks.BarDataColumns.TIMESTAMP : [1,2,3,4,5],
            dat_blocks.BarDataColumns.OPEN.value : [5,6,7,6,5],
            dat_blocks.BarDataColumns.CLOSE.value : [2,2,2,2,2],
            dat_blocks.BarDataColumns.HIGH.value: [3,3,3,3,3],
            dat_blocks.BarDataColumns.LOW.value: [4,5,3,1,3],
            dat_blocks.BarDataColumns.VOLUME.value: [6,7,5,3,4],
            dat_blocks.BarDataColumns.VWAP.value: [5,8,7,2,3],
        })
        
        self.test_parameters_dict_1 = {
            "Tom" : [{"clicks" : 1, "mice" : 3}],
            "Jerry" : [{"milk" : 3, "cookies" : 5}]
        }
        self.test_alias_1 = "test_1"
        self.test_price_series_used_1 = dat_blocks.BarDataColumns.OPEN.value

    def test_feature_creation_case_1(self):
        feat_gen_stud = FeatureGeneratorSTUD(self.test_parameters_dict_1, alias = self.test_alias_1, price_series_used = self.test_price_series_used_1)
        feature_wrapper = feat_gen_stud.create_features_for_data_bar()



