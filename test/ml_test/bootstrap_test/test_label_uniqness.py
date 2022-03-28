import src.machine_learning_module.bootstrapping as boot_mod
import src.machine_learning_module.label_generators as label_mod
import src.machine_learning_module.filter_generators as filter_mod
import numpy as np
import pandas as pd
import unittest


class LabelUniquenessTest(unittest.TestCase):
    def test_case_1(self):
        # ----- set up -----
        """
        index
        filter
        look ahead
        boot strap
        selected
        overlap
        0  1  2  3  4  5  6  7  8  9 10 11 12
        1  0  1  1  0  0  1  0  1  0  1  1  1
        4  2  1  2  4  5  3  5  4  3  2  1  0
        1  0  1  0  0  0  1  0  1  0  1  1  0
        1  0  1  0  0  0  1  0  1  0  1  1  0
        1  1  2  2  1  0  1  1  2  2  2  3  3
        """
        test_filter_arr : np.array = np.array([1,0,1,1,0,0,1,0,1,0,1,1,1]).astype(np.bool_)
        test_label_arr : np.array = np.array([1,0,1,0,1,1,1,0,0,0,1,0,1])
        test_look_ahead_arr : np.array = np.array([4,2,1,2,4,5,3,5,4,3,2,1,0])
        test_boot_row : np.array = np.array([1,0,1,0,0,0,1,0,1,0,1,1,0]).astype(np.bool_)

        test_label_series : pd.Series = pd.Series(test_label_arr)
        test_look_ahead_series : pd.Series = pd.Series(test_look_ahead_arr)

        test_filter_wrapper : filter_mod.FilterArray = filter_mod.FilterArray(filter_array = test_filter_arr)
        test_label_wrapper : label_mod.LabelDataFrame = label_mod.LabelDataFrame(label_series= test_label_series,
                                                                                 look_ahead_series= test_look_ahead_series,
                                                                                 label_gen_name = "TEST",
                                                                                 bar_data_description = "TEST")

        # ----- correct answer ------
        # correct_overlap_arr : np.array = np.array([[1, 1, 2, 2, 1, 0, 1, 1, 2, 2, 2, 3, 3]])
        correct_uniqueness_arr : np.array = np.zeros(13).astype(np.float_)
        correct_uniqueness_arr[0] = (1 + 1 + 1/2 + 1/2 + 1) / 5 # 0.80
        correct_uniqueness_arr[2] = (1/2 + 1/2) / 2 # 0.5
        correct_uniqueness_arr[6] = (1 + 1 + 1/2 + 1/2) / 4 # 0.75
        correct_uniqueness_arr[8] = (1/2 + 1/2 + 1/2 + 1/3 + 1/3) / 5 # 0.433
        correct_uniqueness_arr[10] = (1/2 + 1/3 + 1/3) / 3 # 0.389
        correct_uniqueness_arr[11] = (1/3 + 1/3) / 2 # 0.333
        correct_average_uniqueness = sum(correct_uniqueness_arr) / 6

        # ---- test -----
        test_average_uniqueness = boot_mod.find_average_uniqueness(filter_wrapper = test_filter_wrapper, label_wrapper = test_label_wrapper, boot_strap_row = test_boot_row)
        self.assertEqual(correct_average_uniqueness, test_average_uniqueness)

if __name__ == '__main__':
    unittest.main()
