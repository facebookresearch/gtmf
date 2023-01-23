# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing as mp
import random
import time
from collections import Counter

from typing import Dict, List, NamedTuple, Optional

import numpy as np
import pandas as pd

import scipy.stats as stats

from plotly import graph_objects as go
from plotly.subplots import make_subplots


def get_prediction(df, score, pos_label, neg_label, threshold):
    """
    Get predicted labels based on predicted scores and threshold

    Params
    df:          a dataframe contains columns of predicted scores
    score:       column name of predicted scores
    pos_label:   name of positive label category
    neg_label:   name of negative label category
    threshold:   threshold determining positive and negative labels

    Return:
    predicted labels: get predicted labels
    """
    df["pred_label"] = np.where(df[score] > threshold, pos_label, neg_label)


def test(true_label, pred_label, pos_label):
    """
    Obtain test result for a single sample in binary classification:

    Params
    true_label:  true label
    pred_label:  predicted label
    pos_label:   the label category to be considered as positive

    Return:
    test result: test result in string
    """
    if pred_label != pos_label:
        if true_label != pos_label:
            return "TN"
        else:
            return "FN"
    else:
        if true_label == pos_label:
            return "TP"
        else:
            return "FP"


def get_test_result(df, true_label, pred_label, pos_label="1"):
    """
    Obtain confusion matrix and test results for several samples in binary classification:

    Params
    df:                 a dataframe contains columns of true labels and predicted labels
    true_label:         column name of true label
    pred_label:         column name of predicted label
    pos_label:          the label category to be considered as positive

    Return:
    confusion matrix:   a dictionary of numbers of different test results
    test result:        a list of test results of samples; each element is a string
    """

    test_result = [
        test(df[true_label][i], df[pred_label][i], pos_label) for i in range(len(df))
    ]
    return dict(Counter(test_result)), test_result


def get_sample_loss(df, true_label, pred_label, metric):
    """
    Obtain losses for several samples in regression/multi-label classification

    Params
    df:                 a dataframe contains columns of true labels and predicted labels
    true_label:         column name of true label
    pred_label:         column name of predicted label
    metric:             currently support: (1) mean absolute error, (2) mean square error and (3) Jaccard index

    Return:
    loss:               a numpy array of sample losses
    """

    if metric == "mean_absolute_error":
        return np.array(abs(df[true_label] - df[pred_label]))
    elif metric == "mean_square_error":
        return np.array((df[true_label] - df[pred_label]) ** 2)
    elif metric == "jaccard_index":
        return np.array(
            [
                len(set(df[pred_label][i]).intersection(df[true_label][i]))
                / len(set(df[pred_label][i]).union(df[true_label][i]))
                for i in range(len(df))
            ]
        )


def accuracy(test_result, bootstrap_sample, weight):
    """
    Calculate accuracy

    Params
    test_result:        a list of test results
    bootstrap_sample:   a list of indices in the bootstrap sample
    weight:             a list of sample weights

    Return:
    accuracy:           accuracy for this bootstrap sample
    """
    return sum(
        [
            weight[i]
            for i in bootstrap_sample
            if test_result[i] == "TP" or test_result[i] == "TN"
        ]
    ) / sum(weight[bootstrap_sample])


def fpr(test_result, bootstrap_sample, weight):
    total_fp = sum([weight[i] for i in bootstrap_sample if test_result[i] == "FP"])
    total_tn = sum([weight[i] for i in bootstrap_sample if test_result[i] == "TN"])
    return total_fp / (total_fp + total_tn)


def recall(test_result, bootstrap_sample, weight):
    total_tp = sum([weight[i] for i in bootstrap_sample if test_result[i] == "TP"])
    total_fn = sum([weight[i] for i in bootstrap_sample if test_result[i] == "FN"])
    return total_tp / (total_tp + total_fn)


def precision(test_result, bootstrap_sample, weight):
    total_tp = sum([weight[i] for i in bootstrap_sample if test_result[i] == "TP"])
    total_fp = sum([weight[i] for i in bootstrap_sample if test_result[i] == "FP"])
    return total_tp / (total_tp + total_fp)


def f1(test_result, bootstrap_sample, weight):
    total_tp = sum([weight[i] for i in bootstrap_sample if test_result[i] == "TP"])
    total_fp = sum([weight[i] for i in bootstrap_sample if test_result[i] == "FP"])
    total_fn = sum([weight[i] for i in bootstrap_sample if test_result[i] == "FN"])
    return 2 * total_tp / (2 * total_tp + total_fn + total_fp)


def fbeta(test_result, bootstrap_sample, weight, beta):
    total_tp = sum([weight[i] for i in bootstrap_sample if test_result[i] == "TP"])
    total_fp = sum([weight[i] for i in bootstrap_sample if test_result[i] == "FP"])
    total_fn = sum([weight[i] for i in bootstrap_sample if test_result[i] == "FN"])
    return (
        (1 + beta**2)
        * total_tp
        / ((1 + beta**2) * total_tp + beta**2 * total_fn + total_fp)
    )


def fbeta_variance(test_result, total_sample_size, weight, beta):
    """
    Calculate variance of f_beta metric

    Params
    test_result:        a list of test results
    total_sample_size:  total number of samples (scalar)
    weight:             a list of sample weights
    beta:               parameter beta (scalar)

    Return:
    variance:           variance of f_beta metric using Cochran approximation
    """
    tp_weights = np.array(
        [weight[i] for i in range(total_sample_size) if test_result[i] == "TP"]
    )
    fp_weights = np.array(
        [weight[i] for i in range(total_sample_size) if test_result[i] == "FP"]
    )
    fn_weights = np.array(
        [weight[i] for i in range(total_sample_size) if test_result[i] == "FN"]
    )
    total_tp = sum(tp_weights)
    total_fp = sum(fp_weights)
    total_fn = sum(fn_weights)
    fbeta_hat = (
        (1 + beta**2)
        * total_tp
        / ((1 + beta**2) * total_tp + total_fp + beta**2 * total_fn)
    )
    tp_wt_ssq, fp_wt_ssq, fn_wt_ssq = (
        sum(tp_weights**2),
        sum(fp_weights**2),
        sum(fn_weights**2),
    )

    return (
        (1 + beta**2) ** 2 * tp_wt_ssq * (1 - fbeta_hat) ** 2
        + (fp_wt_ssq + beta**4 * fn_wt_ssq) * fbeta_hat**2
    ) / ((1 + beta**2) * total_tp + total_fp + beta**2 * total_fn) ** 2


def f1_variance(test_result, total_sample_size, weight):
    tp_weights = np.array(
        [weight[i] for i in range(total_sample_size) if test_result[i] == "TP"]
    )
    fp_weights = np.array(
        [weight[i] for i in range(total_sample_size) if test_result[i] == "FP"]
    )
    fn_weights = np.array(
        [weight[i] for i in range(total_sample_size) if test_result[i] == "FN"]
    )
    total_tp = sum(tp_weights)
    total_fp = sum(fp_weights)
    total_fn = sum(fn_weights)
    f1_hat = 2 * total_tp / (2 * total_tp + total_fp + total_fn)
    tp_wt_ssq, fp_wt_ssq, fn_wt_ssq = (
        sum(tp_weights**2),
        sum(fp_weights**2),
        sum(fn_weights**2),
    )
    return (
        4 * tp_wt_ssq * (1 - f1_hat) ** 2 + (fp_wt_ssq + fn_wt_ssq) * f1_hat**2
    ) / (2 * total_tp + total_fp + total_fn) ** 2


def accuracy_variance(test_result, total_sample_size, weight):
    acc_hat = sum(
        [
            weight[i]
            for i in range(total_sample_size)
            if test_result[i] == "TP" or test_result[i] == "TN"
        ]
    )

    pos_wt_ssq = sum(
        [
            weight[i] ** 2
            for i in range(total_sample_size)
            if test_result[i] == "TP" or test_result[i] == "TN"
        ]
    )
    neg_wt_ssq = sum(
        [
            weight[i] ** 2
            for i in range(total_sample_size)
            if test_result[i] == "FP" or test_result[i] == "FN"
        ]
    )
    return (
        total_sample_size
        / (total_sample_size - 1)
        * (pos_wt_ssq * (1 - acc_hat) ** 2 + neg_wt_ssq * acc_hat**2)
    )


def recall_variance(test_result, total_sample_size, weight):
    tp_weights = np.array(
        [weight[i] for i in range(total_sample_size) if test_result[i] == "TP"]
    )
    fn_weights = np.array(
        [weight[i] for i in range(total_sample_size) if test_result[i] == "FN"]
    )
    total_tp = sum(tp_weights)
    total_fn = sum(fn_weights)
    rec_hat = total_tp / (total_tp + total_fn)

    return (
        sum(tp_weights**2) * (1 - rec_hat) ** 2 + sum(fn_weights**2) * rec_hat**2
    ) / (total_tp + total_fn) ** 2


def precision_variance(test_result, total_sample_size, weight):
    tp_weights = np.array(
        [weight[i] for i in range(total_sample_size) if test_result[i] == "TP"]
    )
    fp_weights = np.array(
        [weight[i] for i in range(total_sample_size) if test_result[i] == "FP"]
    )
    total_tp = sum(tp_weights)
    total_fp = sum(fp_weights)
    pre_hat = total_tp / (total_tp + total_fp)
    return (
        sum(tp_weights**2) * (1 - pre_hat) ** 2 + sum(fp_weights**2) * pre_hat**2
    ) / (total_tp + total_fp) ** 2


def fpr_variance(test_result, total_sample_size, weight):
    fp_weights = np.array(
        [weight[i] for i in range(total_sample_size) if test_result[i] == "FP"]
    )
    tn_weights = np.array(
        [weight[i] for i in range(total_sample_size) if test_result[i] == "TN"]
    )
    total_fp = sum(fp_weights)
    total_tn = sum(tn_weights)
    fpr_hat = total_fp / (total_fp + total_tn)

    return (
        sum(fp_weights**2) * (1 - fpr_hat) ** 2 + sum(tn_weights**2) * fpr_hat**2
    ) / (total_fp + total_tn) ** 2


def bootstrap_metric(test_result, total_sample_size, weight, metric, sample_size, beta):
    """
    Calculate bootstrap metric for classification metrics

    Params
    test_result:        a list of test results
    total_sample_size:  total number of samples (scalar)
    weight:             a list of sample weights
    metric:             name of the metric
    sample_size:        sample size for each bootstrap sample
    beta:               [optional] parameter beta (scalar)

    Return:
    bootstrap metric:   a scalar metric
    """
    bootstrap_sample = random.choices(range(total_sample_size), k=sample_size)
    fun_dict = {
        "accuracy": accuracy,
        "fpr": fpr,
        "f1": f1,
        "fbeta": fbeta,
        "precision": precision,
        "recall": recall,
    }
    if metric == "fbeta":
        return fun_dict[metric](test_result, bootstrap_sample, weight, beta)
    else:
        return fun_dict[metric](test_result, bootstrap_sample, weight)


def bootstrap_average(loss, total_sample_size, weight, sample_size):
    """
    Calculate bootstrap metric for regression metrics

    Params
    loss:               a list of losses
    total_sample_size:  total number of samples (scalar)
    weight:             a list of sample weights
    sample_size:        sample size for each bootstrap sample

    Return:
    bootstrap metric:   a scalar metric
    """
    bootstrap_sample = random.choices(range(total_sample_size), k=sample_size)

    return sum(loss[bootstrap_sample] * weight[bootstrap_sample]) / sum(
        weight[bootstrap_sample]
    )


def analytic_variance(test_result, total_sample_size, weight, metric, beta):
    """
    Calculate analytical variance of classification metrics

    Params
    test_results:       a list of test results of samples
    total_sample_size:  total number of samples (scalar)
    weight:             a list of sample weights
    metric:             name of the metric
    beta:               [optional] parameter beta (scalar)

    Return:
    variance:           variance of the metric using Cochran approximation
    """
    fun_dict = {
        "accuracy": accuracy_variance,
        "fpr": fpr_variance,
        "f1": f1_variance,
        "fbeta": fbeta_variance,
        "precision": precision_variance,
        "recall": recall_variance,
    }
    if metric == "fbeta":
        return fun_dict[metric](test_result, total_sample_size, weight, beta)
    else:
        return fun_dict[metric](test_result, total_sample_size, weight)


def bootstrap_classification(
    test_result,
    weight,
    metric,
    bootstrap_size,
    sample_size,
    n_processes,
    beta,
    random_state,
):
    """
    Warpper of bootstrapped classification metrics with multi-processing

    Params
    test_results:       a list of test results of samples
    weight:             a list of sample weights
    metric:             name of the metric
    bootstrap_size:     number of repeated bootstrap sampling
    sample_size:        number of samples in each bootstrap sampling
    n_processes:        number of parallel processes
    beta:               [optional] parameter beta (scalar)
    random_state:       [optional] random seed

    Return:
    bootstrap metrics:  a numpy array of bootstrapped metrics
    """
    random.seed(random_state)
    total_sample_size = len(weight)
    with mp.Pool(processes=n_processes) as p:
        bootstrap_metrics = p.starmap(
            bootstrap_metric,
            [
                (test_result, total_sample_size, weight, metric, sample_size, beta)
                for i in range(bootstrap_size)
            ],
        )

    return np.array(bootstrap_metrics)


def bootstrap_other(
    loss, weight, bootstrap_size, sample_size, n_processes, random_state
):
    """
    Warpper of bootstrapped regression metrics with multi-processing

    Params
    loss:               a list of test results of samples
    weight:             a list of sample weights
    bootstrap_size:     number of repeated bootstrap sampling
    sample_size:        number of samples in each bootstrap sampling
    n_processes:        number of parallel processes
    random_state:       [optional] random seed

    Return:
    bootstrap metrics:  a numpy array of bootstrapped metrics
    """
    random.seed(random_state)
    total_sample_size = len(weight)
    with mp.Pool(processes=n_processes) as p:
        bootstrap_metrics = p.starmap(
            bootstrap_average,
            [
                (loss, total_sample_size, weight, sample_size)
                for i in range(bootstrap_size)
            ],
        )

    return np.array(bootstrap_metrics)


def metric_interval_classification(
    df,
    true_label,
    pred_label,
    metric,
    method,
    bootstrap_size=1e3,
    sample_size=0,
    n_processes=2,
    random_state=None,
    pos_label="1",
    confidence_level=0.95,
    beta=1,
    weight=None,
):

    if sample_size == 0:
        sample_size = len(df)
    """
    Warpper of calculating point, variance and interval estimates of classification metrics

    Params
    df:                 a dataframe contains columns of true label and predicted label
    true_label:         name of true label column
    pred_label:         name of predicted label column
    metric:             name of the metric
    method:             bootstrap or normal approximation
    bootstrap_size:     number of repeated bootstrap sampling
    sample_size:        number of samples in each bootstrap sampling
    n_processes:        number of parallel processes
    random_state:       [optional] random seed
    pos_label:          name of label category to be considered as positive
    confidence_level:   scalar between 0 and 1
    beta:               [optional] parameter value of beta
    weight:             [optional] sample weights

    Return: a dictionary of
    estimate:           metric point estimate
    lb:                 lower bound of CI of the metric
    ub:                 upper bound of CI of the metric
    variance:           metric variance
    bootstrap metrics:  [only available for classification metrics with bootstrap] a numpy array of bootstrapped metrics
    """

    confusion, test_result = get_test_result(df, true_label, pred_label, pos_label)
    total_sample_size = len(df)

    if method == "bootstrap":
        if weight is None:
            weight = np.repeat(1 / total_sample_size, total_sample_size)
            bootstrap_samples = stats.multinomial.rvs(
                sample_size,
                p=np.array(
                    [confusion["TP"], confusion["FP"], confusion["FN"], confusion["TN"]]
                )
                / sum(confusion.values()),
                size=int(bootstrap_size),
                random_state=random_state,
            )
            if metric == "accuracy":
                bootstrap_metrics = (
                    bootstrap_samples[:, 0] + bootstrap_samples[:, 3]
                ) / sample_size
            elif metric == "fpr":
                bootstrap_metrics = bootstrap_samples[:, 1] / sample_size
            elif metric == "f1":
                bootstrap_metrics = (
                    2
                    * bootstrap_samples[:, 0]
                    / (
                        2 * bootstrap_samples[:, 0]
                        + bootstrap_samples[:, 2]
                        + bootstrap_samples[:, 1]
                    )
                )
            elif metric == "fbeta":
                bootstrap_metrics = (
                    (1 + beta**2)
                    * bootstrap_samples[:, 0]
                    / (
                        (1 + beta**2) * bootstrap_samples[:, 0]
                        + beta**2 * bootstrap_samples[:, 2]
                        + bootstrap_samples[:, 1]
                    )
                )
            elif metric == "precision":
                bootstrap_metrics = bootstrap_samples[:, 0] / (
                    bootstrap_samples[:, 0] + bootstrap_samples[:, 1]
                )
            elif metric == "recall":
                bootstrap_metrics = bootstrap_samples[:, 0] / (
                    bootstrap_samples[:, 0] + bootstrap_samples[:, 2]
                )
        else:
            bootstrap_metrics = bootstrap_classification(
                test_result,
                weight,
                metric,
                bootstrap_size,
                sample_size,
                n_processes,
                beta,
                random_state,
            )

    elif method == "normal_approx":
        if weight is None:
            weight = np.repeat(1 / total_sample_size, total_sample_size)

        variance = analytic_variance(
            test_result, total_sample_size, weight, metric, beta
        )

    fun_dict = {
        "accuracy": accuracy,
        "fpr": fpr,
        "f1": f1,
        "fbeta": fbeta,
        "precision": precision,
        "recall": recall,
    }

    if metric == "fbeta":
        estimate = fun_dict[metric](test_result, range(total_sample_size), weight, beta)
    else:
        estimate = fun_dict[metric](test_result, range(total_sample_size), weight)

    if method == "bootstrap":
        return dict(
            bs_estimate=estimate,
            bs_metrics=bootstrap_metrics,
            bs_lb=np.quantile(bootstrap_metrics, q=(1 - confidence_level) / 2),
            bs_ub=np.quantile(bootstrap_metrics, q=1 - (1 - confidence_level) / 2),
            bs_var=np.var(bootstrap_metrics),
            metric=metric,
        )
    else:
        qt_normal = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        sd = np.sqrt(variance)
        return dict(
            na_estimate=estimate,
            na_lb=estimate - qt_normal * sd,
            na_ub=estimate + qt_normal * sd,
            na_var=variance,
            metric=metric,
        )


def metric_interval_other(
    df,
    true_label,
    pred_label,
    metric,
    method,
    bootstrap_size,
    sample_size,
    n_processes,
    random_state=None,
    confidence_level=0.95,
    weight=None,
):

    if sample_size == 0:
        sample_size = len(df)
    total_sample_size = len(df)
    if weight is None:
        weight = np.repeat(1 / total_sample_size, total_sample_size)
    sample_loss = get_sample_loss(df, true_label, pred_label, metric)
    estimate = sum(weight * sample_loss)
    if method == "bootstrap":
        bootstrap_metrics = bootstrap_other(
            sample_loss, weight, bootstrap_size, sample_size, n_processes, random_state
        )
        return dict(
            bs_estimate=estimate,
            bs_metrics=bootstrap_metrics,
            bs_lb=np.quantile(bootstrap_metrics, q=(1 - confidence_level) / 2),
            bs_ub=np.quantile(bootstrap_metrics, q=1 - (1 - confidence_level) / 2),
            bs_var=np.var(bootstrap_metrics),
            metric=metric,
        )
    else:
        variance = sum(weight**2 * (sample_loss - estimate) ** 2)
        qt_normal = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        sd = np.sqrt(variance)
        return dict(
            na_estimate=estimate,
            na_lb=estimate - qt_normal * sd,
            na_ub=estimate + qt_normal * sd,
            na_var=variance,
            metric=metric,
        )


def metric_interval(
    df: pd.DataFrame,
    metric: str,
    method: str,
    true_label: str,
    pred_label: str,
    weight_col_name: str,
    pos_label: str,
    bootstrap_size: Optional[int] = 1000,
    sample_size: Optional[int] = 0,
    confidence_level: Optional[float] = 0.95,
    beta: Optional[int] = 1,
    n_processes: Optional[int] = None,
    random_seed: Optional[int] = None,
):

    if not n_processes:
        n_processes = int(mp.cpu_count() / 2)
    if not random_seed:
        random_seed = int(time.time())

    classification_metric = {"recall", "precision", "fpr", "f1", "fbeta", "accuracy"}
    other_metric = {"mean_absolute_error", "mean_square_error", "jaccard_index"}
    weight = (
        np.array(df[weight_col_name] / sum(df[weight_col_name]))
        if weight_col_name
        else None
    )

    if metric in classification_metric:
        if not pos_label:
            raise Exception(
                f"parameter pos_label is used to specify the label category to be considered as positive. It cannot be empty string or None when calculating classification metrics. Current value of pos_label is {pos_label}"
            )
            return
        result = metric_interval_classification(
            df,
            true_label,
            pred_label,
            metric,
            method,
            bootstrap_size,
            sample_size,
            n_processes,
            random_seed,
            pos_label,
            confidence_level,
            beta,
            weight,
        )
    elif metric in other_metric:
        result = metric_interval_other(
            df,
            true_label,
            pred_label,
            metric,
            method,
            bootstrap_size,
            sample_size,
            n_processes,
            random_seed,
            confidence_level,
            weight,
        )

    return result


def get_metric_result_bootstrap(
    df: pd.DataFrame,
    metric_list: str,
    true_label: str,
    pred_label: str,
    weight_col_name: str,
    pos_label: str,
    bootstrap_size: Optional[int] = 1000,
    sample_size: Optional[int] = 0,
    confidence_level: Optional[float] = 0.95,
    beta: Optional[int] = 1,
    n_processes: Optional[int] = None,
    random_seed: Optional[int] = None,
):
    ### bootstrap
    bs_results, bs_metrics = [], []
    for metric in metric_list:
        tmp_result = metric_interval(
            df=df,
            metric=metric,
            method="bootstrap",
            true_label=true_label,
            pred_label=pred_label,
            weight_col_name=weight_col_name,
            pos_label=pos_label,
            bootstrap_size=bootstrap_size,
            sample_size=sample_size,
            confidence_level=confidence_level,
            beta=beta,
            n_processes=n_processes,
            random_seed=random_seed,
        )

        bs_results.append(
            pd.DataFrame(
                {
                    k: [tmp_result[k]]
                    for k in ["metric", "bs_estimate", "bs_lb", "bs_ub", "bs_var"]
                }
            )
        )
        bs_metrics.append(
            pd.DataFrame({"bs_metrics_" + metric: tmp_result["bs_metrics"]})
        )

    return pd.concat(bs_results), pd.concat(bs_metrics, axis=1)


def get_metric_result_normal_approximation(
    df: pd.DataFrame,
    metric_list: str,
    true_label: str,
    pred_label: str,
    weight_col_name: str,
    pos_label: str,
    bootstrap_size: Optional[int] = 1000,
    sample_size: Optional[int] = 0,
    confidence_level: Optional[float] = 0.95,
    beta: Optional[int] = 1,
    n_processes: Optional[int] = None,
    random_seed: Optional[int] = None,
):
    ### normal approximation
    na_results = []
    for metric in metric_list:
        tmp_result = metric_interval(
            df=df,
            metric=metric,
            method="normal_approx",
            true_label=true_label,
            pred_label=pred_label,
            weight_col_name=weight_col_name,
            pos_label=pos_label,
            bootstrap_size=bootstrap_size,
            sample_size=sample_size,
            confidence_level=confidence_level,
            beta=beta,
            n_processes=n_processes,
            random_seed=random_seed,
        )
        na_results.append(
            pd.DataFrame(
                {
                    k: [tmp_result[k]]
                    for k in ["metric", "na_estimate", "na_lb", "na_ub", "na_var"]
                }
            )
        )
    # return na_results
    return pd.concat(na_results)


def get_bs_metrics_plot(df, metrics):

    fig = make_subplots(rows=len(metrics), cols=1, subplot_titles=metrics)
    for i, metric in enumerate(metrics):
        df_column = "bs_metrics_" + metric
        fig.add_trace(go.Histogram(x=df[df_column]), row=i + 1, col=1)
        # fig.update_xaxes(title_text="Bootstrap metrics", row=i + 1, col=1)
        fig.update_yaxes(title_text="Count", row=i + 1, col=1)
    fig.update_layout(
        title_text="Distribution of bootstrap metrics",
        height=400 * len(metrics),
        width=1200,
    )

    return fig


def get_metric_interval_results(
    df: pd.DataFrame,
    metric_list: List[str],
    methods: List[str],
    true_label: str,
    pred_label: str,
    pos_label: Optional[str] = None,
    bootstrap_size: Optional[int] = 1000,
    sample_size: Optional[int] = 0,
    confidence_level: Optional[float] = 0.95,
    beta: Optional[int] = 1,
    bootstrap_metrics_plot: Optional[bool] = False,
    n_processes: Optional[int] = None,
    random_seed: Optional[int] = None,
    isweight: Optional[bool] = False,
    weight_col_name: Optional[str] = None,
):

    if not n_processes:
        n_processes = int(mp.cpu_count() / 2)
    if not random_seed:
        random_seed = int(time.time())
    print(f"random seed used for {metric_list}{methods} is {random_seed}")
    weight_col_name = weight_col_name if isweight else None

    if "bootstrap" in methods:
        bs_result, bs_metrics = get_metric_result_bootstrap(
            df=df,
            metric_list=metric_list,
            true_label=true_label,
            pred_label=pred_label,
            weight_col_name=weight_col_name,
            pos_label=pos_label,
            bootstrap_size=bootstrap_size,
            sample_size=sample_size,
            confidence_level=confidence_level,
            beta=beta,
            n_processes=n_processes,
            random_seed=random_seed,
        )
    else:
        bs_result, bs_metrics = pd.DataFrame(), pd.DataFrame()

    na_results = (
        get_metric_result_normal_approximation(
            df=df,
            metric_list=metric_list,
            true_label=true_label,
            pred_label=pred_label,
            weight_col_name=weight_col_name,
            pos_label=pos_label,
            bootstrap_size=bootstrap_size,
            sample_size=sample_size,
            confidence_level=confidence_level,
            beta=beta,
            n_processes=n_processes,
            random_seed=random_seed,
        )
        if "normal_approx" in methods
        else pd.DataFrame()
    )

    assert not (
        bs_result.empty and na_results.empty
    ), "at least one method(bootstrap, normal_approx) should be selected"

    if bs_result.empty:
        result = na_results
    elif na_results.empty:
        result = bs_result
    else:
        result = pd.merge(bs_result, na_results, on="metric")
    result_columns = list(result.columns)
    possible_columns = [
        "metric",
        "bs_estimate",
        "na_estimate",
        "bs_lb",
        "na_lb",
        "bs_ub",
        "na_ub",
        "bs_var",
        "na_var",
    ]

    columns = [c for c in possible_columns if c in result_columns]
    fig = (
        get_bs_metrics_plot(bs_metrics, metric_list)
        if bootstrap_metrics_plot and len(bs_metrics) != 0
        else go.Figure()
    )
    return result[columns], bs_metrics, fig


# ------ Bento Kernel --------#
class Results(NamedTuple):
    results: pd.DataFrame
    bootstrap_metrics: Optional[pd.DataFrame] = None
    fig: Optional[go.Figure] = None


# Bento Kernel API
def choose_metrics_by_label_type() -> Dict[str, list]:
    """
    Guide to select metrics by type of the label

    Return a dict where values are the metrics that the aggregated variance dimension provides
    """
    metrics_dispatcher = {
        "metrics for single classification label": [
            "recall",
            "precision",
            "fpr",
            "f1",
            "fbeta",
            "accuracy",
        ],
        "metrics for regression labels": ["mean_absolute_error", "mean_square_error"],
        "metrics for multiple classification labels": ["jaccard_index"],
        "available methods": ["normal_approx", "bootstrap"],
    }
    return metrics_dispatcher


# Bento Kernel API
def get_single_metric_interval_results(
    df: pd.DataFrame,
    metric: str,
    method: str,
    true_label: str,
    pred_label: str,
    pos_label: Optional[str] = None,
    weight_col_name: Optional[str] = None,
    bootstrap_size: Optional[int] = 1000,
    sample_size: Optional[int] = 0,
    confidence_level: Optional[float] = 0.95,
    beta: Optional[int] = 1,
    bootstrap_metrics_plot: Optional[bool] = False,
    n_processes: Optional[int] = None,
    random_seed: Optional[int] = None,
    isweight: Optional[bool] = False,
) -> Results:
    """
    calculate the variance metrics

    return an object with attributes: metric results and bootstrap metrics if method is set as "bootstrap", also return histograms for bootstrap metrics if bootstrap_metrics_plot is set as True
    metric results : a dictionary of
    estimate:           metric point estimate
    lb:                 lower bound of CI of the metric
    ub:                 upper bound of CI of the metric
    variance:           metric variance

    bootstrap metrics:  a numpy array of bootstrapped metrics
    histograms for bootstrap metrics if bootstrap_metrics_plot is set as True, othervise None

    Parameters
    ------------
    df: pd.DataFrame
        a dataframe contains columns of true label and predicted label
    metric_list: List[str]
        a list of metrics names. To get more info, call choose_metrics_by_label_type().
    methods: List[str]
        a list of method. Two methods are provided: bootstrap, normal approximation
    true_label: str
        name of true label column
    pred_label: str
        name of predicted label column
    pos_label: str,
        the label category to be considered as positive
    weight_col_name: Optional, default is None.
        name of sample weights column. weight_col_name cannot be None if isweight is True
    bootstrap_size: Optional, default is 1000
        number of repeated bootstrap sampling
    sample_size: Optional, default is 0 where will use all data
        number of samples in each bootstrap sampling
    confidence_level: Optional, default is 0.95
        scalar between 0 and 1
    beta: Optional, default is 1
        parameter value of beta
    bootstrap_metrics_plot: Optional, default is False
        plot histograms of bootstrap metrics if set as True
    n_processes: Optional, default is 2
        number of parallel processes
    random_seed: Optional, default is None
        random seed
    isweight: Optional, default is False
        involve weights in metric calculation if set as True

    """

    weight_col_name = weight_col_name if isweight else None
    if not random_seed:
        random_seed = int(time.time())
    print(f"random seed used for {metric} - {method} is {random_seed}")
    result = metric_interval(
        df,
        metric,
        method,
        true_label,
        pred_label,
        weight_col_name,
        pos_label,
        bootstrap_size,
        sample_size,
        confidence_level,
        beta,
        n_processes,
        random_seed,
    )
    if method == "bootstrap":
        bs_metrics = pd.DataFrame(
            data=result.pop("bs_metrics", None), columns=["bs_metrics_" + metric]
        )
        fig = get_bs_metrics_plot(bs_metrics, [metric])
        return Results(results=result, bootstrap_metrics=bs_metrics, fig=fig)
    else:
        return Results(results=result)


# Bento Kernel API
def get_metrics_interval_results(
    df: pd.DataFrame,
    metric_list: List[str],
    methods: List[str],
    true_label: str,
    pred_label: str,
    pos_label: Optional[str] = None,
    bootstrap_size: Optional[int] = 1000,
    sample_size: Optional[int] = 0,
    confidence_level: Optional[float] = 0.95,
    beta: Optional[int] = 1,
    bootstrap_metrics_plot: Optional[bool] = False,
    n_processes: Optional[int] = None,
    random_seed: Optional[int] = None,
    isweight: Optional[bool] = False,
    weight_col_name: Optional[str] = None,
) -> Results:
    """
    calculate the variance metrics

    return an object with attributes: metric results and bootstrap metrics if method is set as "bootstrap", also return histograms for bootstrap metrics if bootstrap_metrics_plot is set as True

    metric results : a table of the following results for each metric
    estimate:           metric point estimate
    lb:                 lower bound of CI of the metric
    ub:                 upper bound of CI of the metric
    variance:           metric variance

    bootstrap metrics:  a numpy array of bootstrapped metrics
    histograms for bootstrap metrics if bootstrap_metrics_plot is set as True, othervise None

    Parameters
    ------------
    df: pd.DataFrame
        a dataframe contains columns of true label and predicted label
    metric_list: List[str]
        a list of metrics names. To get more info, call choose_metrics_by_label_type().
    methods: List[str]
        a list of method. Two methods are provided: bootstrap, normal approximation
    true_label: str
        name of true label column
    pred_label: str
        name of predicted label column
    pos_label: str,
        the label category to be considered as positive
    bootstrap_size: Optional, default is 1000
        number of repeated bootstrap sampling
    sample_size: Optional, default is 0 where will use all data
        number of samples in each bootstrap sampling
    confidence_level: Optional, default is 0.95
        scalar between 0 and 1
    beta: Optional, default is 1
        parameter value of beta
    bootstrap_metrics_plot: Optional, default is False
        plot histograms of bootstrap metrics if set as True
    n_processes: Optional, default is the half of count of cpus
        number of parallel processes
    random_seed: Optional, default is datetime.now()
        random seed
    isweight: Optional, default is False
        involve weights in metric calculation if set as True
    weight_col_name: Optional, default is None
        name of sample weights column. If isweight is True, weight_col_name cannot be empty

    """

    result, bs_metrics, fig = get_metric_interval_results(**locals())

    return (
        Results(results=result, bootstrap_metrics=bs_metrics, fig=fig)
        if "bootstrap" in methods
        else Results(results=result)
    )
