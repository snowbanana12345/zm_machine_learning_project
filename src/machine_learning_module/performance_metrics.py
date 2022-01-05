from dataclasses import dataclass
from src.general_module.custom_exceptions import ArrayLengthMisMatchException
import numpy as np
import pandas as pd
from typing import List


@dataclass(frozen=True)
class BinaryClassificationMetrics:
    """
    Notes on binary classification performance
    Multiclass classification can always be decomposed into one vs all binary classifications
    Denote TP, FP, TN, FN as true position, false positive, true negative, false negative respectively
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)
    f1_score = 2 * precision * recall / (precision + recall)

    TP, FP, TN, FN rates are given as a number between 0 and 1
    """
    accuracy: float
    num_examples: int
    num_pred_positives: int
    num_pred_negatives: int
    num_actual_positives: int
    num_actual_negatives: int
    true_positive_rate: float
    false_positive_rate: float
    true_negative_rate: float
    false_negative_rate: float
    precision: float
    recall: float
    specificity: float
    f1_score: float

    def print(self):
        print(" ---- Binary classfication metrics ------- ")
        print("accuracy : " + str(self.accuracy))
        print("number of examples : " + str(self.num_examples))
        print("number of predicted positives : " + str(self.num_pred_positives))
        print("number of predicted negatives : " + str(self.num_pred_negatives))
        print("number of actual positives : " + str(self.num_actual_positives))
        print("number of actual negatives : " + str(self.num_actual_negatives))
        print("true positive rate : " + str(self.true_positive_rate))
        print("false positive rate : " + str(self.false_positive_rate))
        print("true negative rate : " + str(self.true_negative_rate))
        print("false negative rate : " + str(self.false_negative_rate))
        print("precision : " + str(self.precision))
        print("recall : " + str(self.recall))
        print("specificity : " + str(self.specificity))
        print("f1_score : " + str(self.f1_score))


def find_binary_classification_metrics(predicted: np.array, actual: np.array) -> BinaryClassificationMetrics:
    # ------- check input -------
    if len(predicted) != len(actual):
        raise ArrayLengthMisMatchException(array_1_length=len(predicted), array_2_length=len(actual),
                                           array_1_description="predicted", array_2_description="actual")
    for pred in predicted:
        if pred not in [0, 1]:
            print("Warning : predicted array has a value that is not in [0, 1] : " + str(pred))
            break
    for act in actual:
        if act not in [0, 1]:
            print("Warning : actual array has a value that is not in [0, 1] : " + str(act))
            break
    # ------- find number of positives and negatives -------
    num_examples = len(predicted)
    num_pred_positives = np.count_nonzero(predicted)
    num_pred_negatives = num_examples - num_pred_positives
    num_actual_positives = np.count_nonzero(actual)
    num_actual_negatives = num_examples - num_actual_positives
    # ----- accuracy, TP, FP, TN, FN -------
    accuracy = np.count_nonzero(predicted == actual) / num_examples
    true_positive_num = np.count_nonzero(np.logical_and(predicted == 1, actual == 1))
    false_positive_num = np.count_nonzero(np.logical_and(predicted == 1, actual == 0))
    true_negative_num = np.count_nonzero(np.logical_and(predicted == 0, actual == 0))
    false_negative_num = np.count_nonzero(np.logical_and(predicted == 0, actual == 1))
    true_positive_rate = true_positive_num / num_pred_positives if num_pred_positives > 0 else 0
    false_positive_rate = false_positive_num / num_pred_positives if num_pred_positives > 0 else 0
    true_negative_rate = true_negative_num / num_pred_negatives if num_pred_negatives > 0 else 0
    false_negative_rate = false_negative_num / num_pred_negatives if num_pred_negatives > 0 else 0
    # ----- precision, recall, specificity, f1 score ------
    precision = true_positive_num / num_pred_positives if num_pred_positives > 0 else 0
    recall = true_positive_num / (true_positive_num + false_negative_num) if (
                                                                                         true_positive_num + false_negative_num) > 0 else 0
    specificity = true_negative_num / (true_negative_num + false_positive_num) if (
                                                                                              true_negative_num + false_positive_num) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    # ----- return BinaryClassificationMetrics data class -----
    return BinaryClassificationMetrics(accuracy=accuracy, num_examples=num_examples,
                                       num_pred_positives=num_pred_positives,
                                       num_pred_negatives=num_pred_negatives, num_actual_positives=num_actual_positives,
                                       num_actual_negatives=num_actual_negatives, true_positive_rate=true_positive_rate,
                                       false_positive_rate=false_positive_rate, true_negative_rate=true_negative_rate,
                                       false_negative_rate=false_negative_rate, precision=precision, recall=recall,
                                       specificity=specificity, f1_score=f1_score)


class CollatedBinaryClassificationMetrics:
    def __init__(self, *metrics: BinaryClassificationMetrics):
        self.accuracy_lst: List[float] = []
        self.num_examples_lst: List[int] = []
        self.num_pred_positives_lst: List[int] = []
        self.num_pred_negatives_lst: List[int] = []
        self.num_actual_positives_lst: List[int] = []
        self.num_actual_negatives_lst: List[int] = []
        self.true_positive_rate_lst: List[float] = []
        self.false_positive_rate_lst: List[float] = []
        self.true_negative_rate_lst: List[float] = []
        self.false_negative_rate_lst: List[float] = []
        self.precision_lst: List[float] = []
        self.recall_lst: List[float] = []
        self.specificity_lst: List[float] = []
        self.f1_score_lst: List[float] = []

        for metric in metrics:
            self.accuracy_lst.append(metric.accuracy)
            self.num_examples_lst.append(metric.num_examples)
            self.num_pred_positives_lst.append(metric.num_pred_positives)
            self.num_pred_negatives_lst.append(metric.num_pred_negatives)
            self.num_actual_positives_lst.append(metric.num_actual_positives)
            self.num_actual_negatives_lst.append(metric.num_actual_negatives)
            self.true_positive_rate_lst.append(metric.true_positive_rate)
            self.false_positive_rate_lst.append(metric.false_positive_rate)
            self.true_negative_rate_lst.append(metric.true_negative_rate)
            self.false_negative_rate_lst.append(metric.false_negative_rate)
            self.precision_lst.append(metric.precision)
            self.recall_lst.append(metric.recall)
            self.specificity_lst.append(metric.specificity)
            self.f1_score_lst.append(metric.f1_score)

        self.raw_table: pd.DataFrame = pd.DataFrame({
            "accuracy": self.accuracy_lst,
            "number of examples": self.num_examples_lst,
            "positive predictions": self.num_pred_positives_lst,
            "negative predictions": self.num_pred_negatives_lst,
            "actual positives": self.num_actual_positives_lst,
            "actual negatives": self.num_actual_negatives_lst,
            "true positive rates": self.true_positive_rate_lst,
            "false positive rates": self.false_positive_rate_lst,
            "true negative rates": self.true_negative_rate_lst,
            "false negative rates": self.false_negative_rate_lst,
            "precision": self.precision_lst,
            "recall": self.recall_lst,
            "specificity": self.specificity_lst,
            "f1 score": self.f1_score_lst,
        })

        # ------ compute statistics --------
        self.mean_accuracy: float = float(np.mean(self.accuracy_lst))
        self.sd_accuracy: float = float(np.std(self.accuracy_lst))
        self.mean_true_positive_rate: float = float(np.mean(self.true_positive_rate_lst))
        self.sd_true_positive_rate: float = float(np.std(self.true_positive_rate_lst))
        self.mean_false_positive_rate: float = float(np.mean(self.false_positive_rate_lst))
        self.sd_false_positive_rate: float = float(np.std(self.false_positive_rate_lst))
        self.mean_true_negative_rate: float = float(np.mean(self.true_negative_rate_lst))
        self.sd_true_negative_rate: float = float(np.std(self.true_negative_rate_lst))
        self.mean_false_negative_rate: float = float(np.mean(self.false_negative_rate_lst))
        self.sd_false_negative_rate: float = float(np.std(self.false_negative_rate_lst))
        self.mean_precision: float = float(np.mean(self.precision_lst))
        self.sd_precision: float = float(np.std(self.precision_lst))
        self.mean_recall: float = float(np.mean(self.recall_lst))
        self.sd_recall: float = float(np.std(self.recall_lst))
        self.mean_specificity: float = float(np.mean(self.specificity_lst))
        self.sd_specificity: float = float(np.std(self.specificity_lst))
        self.mean_f1_score: float = float(np.mean(self.f1_score_lst))
        self.sd_f1_score: float = float(np.std(self.f1_score_lst))

        self.stats_table = pd.DataFrame({
            "mean": [self.mean_accuracy, self.mean_true_positive_rate, self.mean_false_positive_rate,
                     self.mean_true_negative_rate,
                     self.mean_false_negative_rate, self.mean_precision, self.mean_recall, self.mean_specificity,
                     self.mean_f1_score],
            "sd": [self.sd_accuracy, self.sd_true_positive_rate, self.sd_false_positive_rate,
                   self.sd_true_negative_rate,
                   self.sd_false_negative_rate, self.sd_precision, self.sd_recall, self.sd_specificity,
                   self.sd_f1_score],
        }, index=["accuracy",
                  "true positive rates",
                  "false positive rates",
                  "true negative rates",
                  "false negative rates",
                  "precision",
                  "recall",
                  "specificity",
                  "f1 score"])

    def print(self):
        print(" ------ mean and sd on metrics ------- ")
        print(self.stats_table)
        print(" ------ numbers for each run ------- ")
        print(self.raw_table)
