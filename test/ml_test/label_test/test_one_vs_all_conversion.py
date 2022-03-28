import unittest
import src.machine_learning_module.label_generators as label_gen_mod
import pandas as pd
import numpy as np


class OneVsAllConversionTest(unittest.TestCase):
    def setUp(self):
        test_name = "TEST_NAME"
        test_description = "TEST_DESCRIPTION"
        self.label_info_1 = label_gen_mod.LabelDataFrameInfo(label_gen_name = test_name,
                                                             bar_data_description = test_description,
                                                             classification_classes = {-2, -1, 0, 1, 2},
                                                             default_class = 0
                                                             )
        self.label_info_2 = self.label_info_1.replicate_pad()
        self.label_info_3 = label_gen_mod.LabelDataFrameInfo(label_gen_name = "blah",
                                                             bar_data_description = "blah blah",
                                                             classification_classes = {-2, -1, 0, 1, 2},
                                                             default_class = 0
                                                             )

    def test_equality(self):

        test_label_series_1 = pd.Series([2, 1, 0, -1, -2, -2, -1, 0, 1, 2, 2, 0, -2])
        test_look_ahead_series_1 = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
        test_label_series_2 = pd.Series([2, 1, 0, -1, -2, -2, -1, 0, 1, 2, 2, 0, -2])
        test_look_ahead_series_2 = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
        test_label_series_3 = pd.Series([2, 0, 0, -1, -2, 1, -1, 0, 1, 2, 2, 0, -2])
        test_look_ahead_series_3 = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])

        wrapper_1 = label_gen_mod.LabelDataFrame(
            label_series = test_label_series_1,
            look_ahead_series = test_look_ahead_series_1,
            label_info = self.label_info_1
        )

        wrapper_2 = label_gen_mod.LabelDataFrame(
            label_series = test_label_series_2,
            look_ahead_series = test_look_ahead_series_2,
            label_info = self.label_info_1
        )

        wrapper_3 = label_gen_mod.LabelDataFrame(
            label_series=test_label_series_3,
            look_ahead_series=test_look_ahead_series_3,
            label_info = self.label_info_1
        )

        wrapper_4 = label_gen_mod.LabelDataFrame(
            label_series=test_label_series_1,
            look_ahead_series = test_look_ahead_series_1,
            label_info = self.label_info_3
        )

        self.assertEqual(wrapper_1, wrapper_2)
        self.assertNotEqual(wrapper_1, wrapper_3)
        self.assertNotEqual(wrapper_1, wrapper_4)

    def test_case_1(self):
        test_label_series = pd.Series([2, 1, 0, -1, -2, -2, -1, 0, 1, 2, 2, 0, -2])
        test_look_ahead_series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
        test_label_wrapper : label_gen_mod.LabelDataFrame = label_gen_mod.LabelDataFrame(
            label_series = test_label_series,
            look_ahead_series = test_look_ahead_series,
            label_info = self.label_info_1
        )
        result = test_label_wrapper.get_one_vs_all()

        expected_class_2_label = pd.Series([1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0])
        expected_class_1_label = pd.Series([0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        expected_class_m1_label = pd.Series([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        expected_class_m2_label = pd.Series([0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1])

        expected_class_2_wrapper = label_gen_mod.LabelDataFrame(
            label_series = expected_class_2_label,
            look_ahead_series = test_look_ahead_series,
            label_info = self.label_info_2
        )
        expected_class_1_wrapper = label_gen_mod.LabelDataFrame(
            label_series = expected_class_1_label,
            look_ahead_series=test_look_ahead_series,
            label_info = self.label_info_2
        )
        expected_class_m1_wrapper = label_gen_mod.LabelDataFrame(
            label_series = expected_class_m1_label,
            look_ahead_series=test_look_ahead_series,
            label_info = self.label_info_2
        )
        expected_class_m2_wrapper = label_gen_mod.LabelDataFrame(
            label_series = expected_class_m2_label,
            look_ahead_series=test_look_ahead_series,
            label_info = self.label_info_2
        )

        self.assertEqual(result[2], expected_class_2_wrapper, "class 2")
        self.assertEqual(result[1], expected_class_1_wrapper, "class 1")
        self.assertEqual(result[-1], expected_class_m1_wrapper, "class -1")
        self.assertEqual(result[-2], expected_class_m2_wrapper, "class -2")

    def test_case_2(self):
        test_label_series = pd.Series([2, 1, 0, -1, -2, -2, -1, 0, 1, 2, np.nan, np.nan, np.nan])
        test_look_ahead_series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.nan, np.nan, np.nan])
        test_label_wrapper: label_gen_mod.LabelDataFrame = label_gen_mod.LabelDataFrame(
            label_series=test_label_series,
            look_ahead_series=test_look_ahead_series,
            label_info = self.label_info_1
        )
        result = test_label_wrapper.get_one_vs_all()

        expected_class_2_label = pd.Series([1, 0, 0, 0, 0, 0, 0, 0, 0, 1, np.nan, np.nan, np.nan])
        expected_class_1_label = pd.Series([0, 1, 0, 0, 0, 0, 0, 0, 1, 0, np.nan, np.nan, np.nan])
        expected_class_m1_label = pd.Series([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, np.nan, np.nan, np.nan])
        expected_class_m2_label = pd.Series([0, 0, 0, 0, 1, 1, 0, 0, 0, 0, np.nan, np.nan, np.nan])

        expected_class_2_wrapper = label_gen_mod.LabelDataFrame(
            label_series = expected_class_2_label,
            look_ahead_series = test_look_ahead_series,
            label_info = self.label_info_2
        )
        expected_class_1_wrapper = label_gen_mod.LabelDataFrame(
            label_series = expected_class_1_label,
            look_ahead_series=test_look_ahead_series,
            label_info = self.label_info_2
        )
        expected_class_m1_wrapper = label_gen_mod.LabelDataFrame(
            label_series = expected_class_m1_label,
            look_ahead_series=test_look_ahead_series,
            label_info = self.label_info_2
        )
        expected_class_m2_wrapper = label_gen_mod.LabelDataFrame(
            label_series = expected_class_m2_label,
            look_ahead_series=test_look_ahead_series,
            label_info = self.label_info_2
        )

        self.assertEqual(result[2], expected_class_2_wrapper, "class 2")
        self.assertEqual(result[1], expected_class_1_wrapper, "class 1")
        self.assertEqual(result[-1], expected_class_m1_wrapper, "class -1")
        self.assertEqual(result[-2], expected_class_m2_wrapper, "class -2")




