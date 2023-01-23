# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
from scipy.stats import chisquare, kendalltau, pearsonr, spearmanr, ttest_1samp
from sklearn.metrics import precision_recall_fscore_support
from statsmodels.stats.weightstats import ttest_ind

"""
The script includes the following metrics
- Individual level metrics(where the error of each individual item counts)
    a) precision, recall, F1  -> for binary(e.g. teen/adult) or categorical(e.g. ['13-17', '18-22', '23-29', '30+']) labels
    b) KL-Divergence, RMSE    -> for continuous labels(e.g. age)

- Aggregated level metrics (where we care about the aggregated bias):
    ASMD, t tests (one sample, two sample with equal and unequal variances) -> for all types(binary, categorical and continuous)

    Note that the p-values in the output are information to help understand how different the means are.
    It is up to the team to make the final decision on whether the means are close enough.
"""


# --- Individual level metrics --- #
# --- precision, recall, F1 --- #
def prec_recall_f1_report(
    df: pd.DataFrame,
    labels_to_measure_column: str,
    golden_labels_column: str,
    categorized_buckets: List[Any],
    weight_column: Optional[str] = None,
    sample_weighted=False,
) -> pd.DataFrame:

    """
    Calculate precision, recall, F1 for binary or categorical labels. Individual level metrics(where the error of each individual item counts)

    Return precision, recall and f1 results

    Parameters
    -----------
    df: pd.DataFrame
        dataset in Datafram format with required columns
    labels_to_measure_column: str
        the column name for the labels to be measured
    golden_labels_column: str
        the column name for golden labels
    categorized_buckets: List[Any]
        classes of labels
    weight_column: Optional[str] = None
        the column name of representative weights
    sample_weighted, default is False.
        If it is True/False, return the weighted/unweighted results of the metric
    """
    labels_to_measure = df[labels_to_measure_column]
    golden_labels = df[golden_labels_column]

    sample_weight = df[weight_column] if sample_weighted else None

    per_class_summary = precision_recall_fscore_support(
        y_true=golden_labels,
        y_pred=labels_to_measure,
        sample_weight=sample_weight,
        labels=categorized_buckets,
    )

    weighted_avg_by_support_summary = list(
        precision_recall_fscore_support(
            y_true=golden_labels,
            y_pred=labels_to_measure,
            sample_weight=sample_weight,
            labels=categorized_buckets,
            average="weighted",
        )
    )

    macro_summary = precision_recall_fscore_support(
        y_true=golden_labels,
        y_pred=labels_to_measure,
        sample_weight=sample_weight,
        labels=categorized_buckets,
        average="macro",
    )

    micro_summary = precision_recall_fscore_support(
        y_true=golden_labels,
        y_pred=labels_to_measure,
        sample_weight=sample_weight,
        labels=categorized_buckets,
        average="micro",
    )

    support_index = "support"
    if sample_weight is not None:
        support_index = "sample_weighted_support"

    report_df = pd.DataFrame(
        list(per_class_summary),
        index=["precision", "recall", "f1-score"] + [support_index],
    )
    report_df.columns = categorized_buckets

    total = report_df.loc[support_index].sum()

    report_df[
        "weighted average across classes by support"
    ] = weighted_avg_by_support_summary
    report_df["macro (average across classes)"] = macro_summary
    report_df["micro (regardless of classes)"] = micro_summary

    report = report_df.T
    report[support_index] = report[support_index] / total

    return report


# --- KL-Divergence, RMSE  ---#


def smoothed_cdf(cdf_x, cdf_y, x_sel):
    ind = max(
        np.where(cdf_x <= x_sel)[0]
    )  # index of the largest x that not bigger than x_sel
    if ind == len(cdf_x) - 1:
        y_sel = max(cdf_y)
    else:
        y_sel = cdf_y[ind] + (cdf_y[ind + 1] - cdf_y[ind]) / (
            cdf_x[ind + 1] - cdf_x[ind]
        ) * (x_sel - cdf_x[ind])
    return y_sel


def get_pdf(
    df,
    label_column,
    bucket_endpoints,
    cdf_step_size,
    weight_column,
    sample_weighted=False,
):

    l = df[label_column]
    # empirical cdf of l
    # unweighted version equivalent to ecdf = ECDF(s); ecdf_x_full = ecdf.x; ecdf_y_full = ecdf.y

    df_sorted = df.sort_values(by=label_column, ascending=True)
    ecdf_x_full = np.concatenate([[-float("inf")], np.array(df_sorted[label_column])])

    if sample_weighted:
        ecdf_y_full = np.concatenate([[0], np.cumsum(df_sorted[weight_column])])
        ecdf_y_full /= max(ecdf_y_full)
    else:
        ecdf_y_full = np.linspace(0, len(l), len(l) + 1) / len(l)

    # select fewer, i.e. 1000, end-points in cdf.
    # TODO: set unequal step size based on input dataset automatically
    n = len(ecdf_y_full)
    step = max(1, round(n / 1000)) if not cdf_step_size else cdf_step_size

    ecdf_x = np.concatenate(
        [[ecdf_x_full[0]], ecdf_x_full[1 : (n - 2) : step], [ecdf_x_full[n - 1]]]
    )
    ecdf_y = np.concatenate(
        [[ecdf_y_full[0]], ecdf_y_full[1 : (n - 2) : step], [ecdf_y_full[n - 1]]]
    )

    # smooth cdf
    def scdf(x_sel):
        return smoothed_cdf(ecdf_x, ecdf_y, x_sel)

    k = len(bucket_endpoints)

    p = [0] * k
    p[0] = scdf(bucket_endpoints[0])
    for i in range(1, k):
        p[i] = scdf(bucket_endpoints[i]) - scdf(bucket_endpoints[i - 1])

    return p


def rmse(
    df,
    labels_to_measure_column="labels",
    golden_labels_column="golden_labels",
    weight_column="weight",
    sample_weighted=False,
):
    if not sample_weighted:
        return np.sqrt(
            ((df[labels_to_measure_column] - df[golden_labels_column]) ** 2).mean()
        )
    else:
        return np.sqrt(
            (
                df[weight_column]
                * (df[labels_to_measure_column] - df[golden_labels_column]) ** 2
            ).sum()
            / df[weight_column].sum()
        )


def get_distribution_plot(
    df,
    id_column,
    labels_to_measure_column,
    golden_labels_column,
    bins=30,
    weight_column=None,
    sample_weighted=False,
):

    df_melt = pd.melt(
        df,
        id_vars=[id_column],
        value_vars=[labels_to_measure_column, golden_labels_column],
        var_name="source",
        value_name="labeled_data",
    )
    df_melt["labeled_data"] = df_melt["labeled_data"].astype(int)

    plt.figure(figsize=(10, 6))
    if not sample_weighted:
        sp = sns.displot(
            df_melt,
            x="labeled_data",
            hue="source",
            bins=bins,
            kde=True,
            palette=["#320ff0", "#ff0000"],
            alpha=0.5,
        )
    else:
        df_melt[weight_column] = df[weight_column].tolist() * 2
        sp = sns.displot(
            df_melt,
            x="labeled_data",
            hue="source",
            bins=bins,
            kde=True,
            fill=True,
            weights=df_melt[weight_column],
            palette=["#320ff0", "#ff0000"],
            alpha=0.5,
        )
    plt.legend()
    plt.xlabel("label value")

    return sp.fig


def get_continuous_accuracy_plot(
    df: pd.DataFrame,
    labels_to_measure_column: str,
    golden_labels_column: str,
    id_column: str,
    plot_bins: Optional[Any] = 30,
    weight_column: Optional[str] = None,
    sample_weighted: Optional[bool] = False,
):
    """
    Returns the PDF(probability density function) plot of continuous labels

    Parameters
    -----------
    df: pd.DataFrame
        dataset in Datafram format and with required columns
    labels_to_measure_column: str
        the column name of he labels to be measured.
    id_column: str
        the column name of job identifier
    plot_bins: default is 30
        bins used for PDF plot. If bins is an integer, it defines the number of equal-width bins in the range. If bins is a sequence, it defines the bin edges, including the left edge of the first bin and the right edge of the last bin. Please refer to seaborn.displot for detail.
    golden_labels_column: str
        the column name for golden labels
    weight_column: Optional[str] = None
        the column name of representative weights
    sample_weighted, default is False.
        If it is True/False, return the weighted/unweighted results of the metric

    """

    weight_column = weight_column if sample_weighted else None
    ax = get_distribution_plot(
        df=df,
        bins=plot_bins,
        id_column=id_column,
        labels_to_measure_column=labels_to_measure_column,
        golden_labels_column=golden_labels_column,
        weight_column=weight_column,
        sample_weighted=sample_weighted,
    )

    return ax


def hellinger_dist(
    df,
    labels_to_measure_column,
    golden_labels_column,
    bucket_endpoints,
    cdf_step_size,
    weight_column,
    sample_weighted=False,
):

    if bucket_endpoints:
        bucket_min, bucket_max, step = bucket_endpoints
    else:
        bucket_min, bucket_max, step = (
            math.floor(min(df[labels_to_measure_column], df[golden_labels_column])),
            math.ceil(max(df[labels_to_measure_column], df[golden_labels_column])),
            1,
        )
    bucket_endpoints = range(bucket_min, bucket_max, step)

    p_labels = get_pdf(
        df,
        label_column=labels_to_measure_column,
        bucket_endpoints=bucket_endpoints,
        cdf_step_size=cdf_step_size,
        weight_column=weight_column,
        sample_weighted=sample_weighted,
    )
    p_golden_labels = get_pdf(
        df,
        label_column=golden_labels_column,
        bucket_endpoints=bucket_endpoints,
        cdf_step_size=cdf_step_size,
        weight_column=weight_column,
        sample_weighted=sample_weighted,
    )

    return np.sqrt(
        np.sum((np.sqrt(p_labels) - np.sqrt(p_golden_labels)) ** 2)
    ) / np.sqrt(2)


def get_continuous_accuracy_report(
    df: pd.DataFrame,
    labels_to_measure_column: str,
    golden_labels_column: str,
    bucket_endpoints: List[float],
    cdf_step_size: int,
    weight_column: Optional[str] = None,
    sample_weighted=False,
):
    """
    Calculate the accuracy report for continuous label type
    Return the metric results of rmse and KL-Divergence

    Parameters
    -----------
    df: pd.DataFrame
        dataset in Datafram format and with required columns
    labels_to_measure_column: str
        the column name of he labels to be measured.
    golden_labels_column: str
        the column name for golden labels
    bucket_endpoints: List(float/int, float/int, int)
        a list including [min_value, max_value, step_size] of the labels
    cdf_step_size: int
        step size used to calculate cdf
    weight_column: Optional[str] = None
        the column name of representative weights
    sample_weighted, default is False.
        If it is True/False, return the weighted/unweighted results of the metric


    """

    weight_column = weight_column if sample_weighted else None

    rmse_metric = rmse(
        df=df,
        labels_to_measure_column=labels_to_measure_column,
        golden_labels_column=golden_labels_column,
        weight_column=weight_column,
        sample_weighted=sample_weighted,
    )
    hellinger_dist_metric = hellinger_dist(
        df=df,
        labels_to_measure_column=labels_to_measure_column,
        golden_labels_column=golden_labels_column,
        bucket_endpoints=bucket_endpoints,
        cdf_step_size=cdf_step_size,
        weight_column=weight_column,
        sample_weighted=sample_weighted,
    )

    report = pd.DataFrame(
        data={"accuracy_metric": [rmse_metric, hellinger_dist_metric]},
        index=["rmse", "hellinger_dist"],
    )

    return report


# --- Aggregated level metrics  --- #
# --- ASMD, t tests---#


def asmd(
    df: pd.DataFrame,
    labels_to_measure_column: str,
    golden_labels_column: str,
    label_type: str,
    weight_column: Optional[str] = None,
    sample_weighted=False,
) -> float:
    """
    calculate the asmd(Aggregated level metrics) metric
    Returns the asmd metric

    Parameters
    -----------
    df: pd.DataFrame
        dataset in Datafram format and with required columns
    labels_to_measure_column: str
        the column name of he labels to be measured.
    golden_labels_column: str
        the column name for golden labels
    label_type: str
        the type of label: "binary", "continuous", or "categorical"
    weight_column: Optional[str] = None
        the column name of representative weights
    sample_weighted, default is False.
        If it is True/False, return the weighted/unweighted results of the metric

    """

    weight_column = weight_column if sample_weighted else None

    x = df[labels_to_measure_column]
    y = df[golden_labels_column]

    if label_type != "continuous":

        x = pd.factorize(x, sort=True)[0]
        y = pd.factorize(y, sort=True)[0]

        k = np.unique(x).size

        if k != np.unique(x).size:
            print("classes numbers do not match")
            return None

        if k <= 1:
            print("invalid class number")
            return None

        if k > 2:

            asmd_metric = []

            for i in range(k):
                xi = np.where(x == i, 1, 0)
                yi = np.where(y == i, 1, 0)

                if not sample_weighted:
                    asmd_metric.append(np.abs(np.mean(xi - np.mean(yi)) / np.std(yi)))
                else:
                    asmd_metric.append(
                        np.abs(
                            np.mean(df[weight_column] * xi)
                            - np.mean(df[weight_column] * yi)
                        )
                        / np.std(df[weight_column] * yi)
                    )

            return asmd_metric

    # continuous or k = 2
    if not sample_weighted:
        return np.abs(np.mean(x) - np.mean(y)) / np.std(y)
    else:
        return np.abs(
            np.mean(df[weight_column] * x) - np.mean(df[weight_column] * y)
        ) / np.std(df[weight_column] * y)


def f_test(x, y, alt="two_sided"):
    """
    Performs the F-test to test equal variance or not.
    :param x: The first group of data
    :param y: The second group of data
    :param alt: The alternative hypothesis, one of "two_sided" (default), "greater" or "less"
    :return: a tuple with the F statistic value and the p-value.
    """
    df1 = len(x) - 1
    df2 = len(y) - 1
    f = x.var() / y.var()
    if alt == "greater":
        p = 1.0 - st.f.cdf(f, df1, df2)
    elif alt == "less":
        p = st.f.cdf(f, df1, df2)
    else:
        p = 2.0 * (1.0 - st.f.cdf(f, df1, df2))
    return f, p


def two_sample_ttest(
    df: pd.DataFrame,
    labels_to_measure_column: str,
    golden_labels_column: str,
    label_type: str,
    unequal_var_alpha: float = 0.05,
    weight_column: Optional[str] = None,
    sample_weighted=False,
):
    """
    calualate the two smaple t test(Aggregated level metrics)

    Returns metric result and p value

    Parameters
    -----------
    df: pd.DataFrame
        dataset in Datafram format and with required columns
    labels_to_measure_column: str
        the column name of he labels to be measured.
    golden_labels_column: str
        the column name for golden labels
    label_type: str
        the type of label: "binary", "continuous", or "categorical"
    unequal_var_alpha: default is 0.05
        unequal_val_alpha is used for two sample t test. The value will be used as a threshold and compared with the p value of f_test(x, y) to see if we should assume the standard deviation of the samples is same or not. Please refer to wiki and implementation for details.
    weight_column: Optional[str] = None
        the column name of representative weights
    sample_weighted, default is False.
        If it is True/False, return the weighted/unweighted results of the metric

    -----------
    Note that the p-values in the output are information to help understand how different the means are. It is up to the team to make the final decision on whether the means are close enough.

    """
    if sample_weighted:
        # Convert weight to frequency weight which is required by function ttest_ind
        # Please refer to the following link for more info(https://www.statsmodels.org/dev/generated/statsmodels.stats.weightstats.ttest_ind.html)
        df[weight_column] = (df[weight_column] / sum(df[weight_column])) * len(df)
    x = df[labels_to_measure_column]
    y = df[golden_labels_column]

    if label_type != "continuous":

        x = pd.factorize(x, sort=True)[0]
        y = pd.factorize(y, sort=True)[0]

        k = np.unique(x).size

        if k != np.unique(x).size:
            print("classes numbers do not match")
            return None

        if k <= 1:
            print("invalid class number")
            return None

        if k > 2:

            ttest_metric = []

            for i in range(k):

                xi = np.where(x == i, 1, 0)
                yi = np.where(y == i, 1, 0)

                f, p = f_test(xi, yi)

                if p < unequal_var_alpha:
                    usevar = "unequal"
                else:
                    usevar = "pooled"

                if sample_weighted:
                    ttest_metric.append(
                        ttest_ind(
                            xi,
                            yi,
                            usevar=usevar,
                            weights=(df[weight_column], df[weight_column]),
                        )[:2]
                    )
                else:
                    ttest_metric.append(ttest_ind(xi, yi, usevar=usevar)[:2])

            return ttest_metric

    # continuous or k = 2
    f, p = f_test(x, y)

    if p < unequal_var_alpha:
        usevar = "unequal"
    else:
        usevar = "pooled"

    if sample_weighted:
        ttest_metric = ttest_ind(
            x, y, usevar=usevar, weights=(df[weight_column], df[weight_column])
        )[:2]
    else:
        ttest_metric = ttest_ind(x, y, usevar=usevar)[:2]

    return ttest_metric


def get_weighted_true_mean_catogrical(df, golden_column_name, weight_column_name):
    weighted_g_m = []
    for e in df[golden_column_name].unique():
        weighted_g_m.append(sum(df[df[golden_column_name] == e][weight_column_name]))
    return weighted_g_m


def get_true_mean(label_type, df, golden_column_name, weight_column_name):
    if label_type == "continuous":
        return (
            sum(df[golden_column_name] * df[weight_column_name]) / len(df)
            if weight_column_name
            else sum(df[golden_column_name]) / len(df)
        )
    elif label_type == "binary":
        return (
            get_weighted_true_mean_catogrical(
                df, golden_column_name, weight_column_name
            )[0]
            if weight_column_name
            else min((df[golden_column_name].value_counts() / len(df)))
        )
    elif label_type == "categorical":
        return (
            get_weighted_true_mean_catogrical(
                df, golden_column_name, weight_column_name
            )
            if weight_column_name
            else (df[golden_column_name].value_counts() / len(df))
        )


def one_sample_ttest(
    df: pd.DataFrame,
    true_mean: float,
    labels_to_measure_column: str,
    label_type: str,
    weight_column: Optional[str] = None,
    sample_weighted=False,
):
    """
    calculate metric of one sample t test(Aggregated level metrics)

    Returns metric result and p value
    Parameters
    -----------
    df: pd.DataFrame
        dataset in Datafram format and with required columns
    true_mean: float
        True mean is the value that we compare with the weighted or unweighted sample mean from labeled data set
    labels_to_measure_column: str
        the column name of he labels to be measured.
    label_type: str
        the type of label: "binary", "continuous", or "categorical"
    weight_column: Optional[str] = None
        the column name of representative weights
    sample_weighted, default is False.
        If it is True/False, return the weighted/unweighted results of the metric

    -----------
    True mean is the value that we compare with the weighted or unweighted sample mean from labeled data set.  It can be
    - population mean of model prediction based labels or calibrated scores, or
    - (weighted) mean of another source of Ground Truth such as Survey data, or
    - a value that we believe to be the true mean (e.g. the mean setting in simulations, or from expert's information).

    Note that the p-values in the output are information to help understand how different the means are. It is up to the team to make the final decision on whether the means are close enough.

    """
    weight_column = weight_column if sample_weighted else None
    if sample_weighted:
        true_mean_parameter = "one_sample_ttest_true_mean_weighted"
    else:
        true_mean_parameter = "one_sample_ttest_true_mean"

    if not true_mean:
        raise Exception(
            f"{true_mean_parameter} should not be empty if one smaple t test is selected"
        )

    try:
        if label_type == "categorical":
            float(sum(true_mean))
        else:
            float(true_mean)
    except Exception:
        print(f"{true_mean_parameter} should be numeric value")

    x = df[labels_to_measure_column]

    if label_type != "continuous":

        x = pd.factorize(x, sort=True)[0]

        k = np.unique(x).size

        if k <= 1:
            print("invalid class number")
            return None

        if k > 2:
            if k != np.asarray(true_mean).size:
                print("classes numbers do not match with golden mean size")
                return None

            ttest_metric = []

            for i in range(k):
                # To confirm: xi is not used.
                # xi = np.where(x == i, 1, 0)

                if sample_weighted:
                    ttest_metric.append(
                        ttest_1samp(df[weight_column] * x, true_mean[i])[:2]
                    )
                else:
                    ttest_metric.append(ttest_1samp(x, true_mean[i])[:2])

            return ttest_metric

    # continuous or k = 2
    if sample_weighted:
        ttest = ttest_1samp(df[weight_column] * x, true_mean)[:2]
    else:
        ttest = ttest_1samp(x, true_mean)[:2]

    return ttest


# This metric is only for categorical labels with number of classes k > 2
def chisquare_test(
    df: pd.DataFrame,
    labels_to_measure_column: str,
    golden_labels_column: str,
    categorized_buckets: List[Any],
    weight_column: Optional[str] = None,
    sample_weighted=False,
):
    """
    Calculate the chisquare_test which is only for categorical labels with number of classes k > 2(Aggregated level metrics)

    Returns
    the value of chisquare_test metric

    Parameters
    -----------
    df: pd.DataFrame
        dataset in Datafram format and with required columns
    labels_to_measure_column: str
        the column name for the labels to be measured
    golden_labels_column: str
        the column name for golden labels
    categorized_buckets: List[Any]
        classes of labels
    weight_column: Optional[str] = None
        the column name of representative weights
    sample_weighted, default is False.
        If it is True/False, return the weighted/unweighted results of the metric

    """
    weight_column = weight_column if sample_weighted else None

    x_freq = []
    y_freq = []

    if not sample_weighted:
        x_freq_raw = df.groupby(labels_to_measure_column).count().iloc[:, 0]
        y_freq_raw = df.groupby(golden_labels_column).count().iloc[:, 0]

    else:
        x_freq_raw = (
            df[[labels_to_measure_column, weight_column]]
            .groupby(labels_to_measure_column)
            .count()
            .iloc[:, 0]
        )
        y_freq_raw = (
            df[[golden_labels_column, weight_column]]
            .groupby(golden_labels_column)
            .count()
            .iloc[:, 0]
        )

    # you can also difrectly use x_freq_raw and y_freq_raw, with risks on non_coveraged classes
    for i in range(len(categorized_buckets)):
        x_freq.append(x_freq_raw[categorized_buckets[i]])
        y_freq.append(y_freq_raw[categorized_buckets[i]])

    chisquare_test_metric = chisquare(x_freq, y_freq)

    return chisquare_test_metric


def get_aggregated_accuracy_report(
    df: pd.DataFrame,
    labels_to_measure_column: str,
    golden_labels_column: str,
    categorized_buckets: List[Any],
    label_type: str,
    true_mean: float,
    unequal_var_alpha: float = 0.05,
    weight_column: Optional[str] = None,
    sample_weighted=False,
):
    """
    Calculate the Aggregated level metrics (where we care about the aggregated bias):
    ASMD, t tests (one sample, two sample with equal and unequal variances) -> for all types(binary, categorical and continuous)
    chisquare_test for categorical label type

    Note that the p-values in the output are information to help understand how different the means are.
    It is up to the team to make the final decision on whether the means are close enough.

    Parameters
    -----------
    df: pd.DataFrame
        dataset in Datafram format and with required columns
    labels_to_measure_column: str
        the column name of he labels to be measured.
    golden_labels_column: str
        the column name for golden labels
    categorized_buckets: List[Any]
        classes of labels
    label_type: str
        the type of label: "binary", "continuous", or "categorical"
    true_mean: float
        True mean is the value that we compare with the weighted or unweighted sample mean from labeled data set
    unequal_var_alpha: default is 0.05
        unequal_val_alpha is used for two sample t test. The value will be used as a threshold and compared with the p value of f_test(x, y) to see if we should assume the standard deviation of the samples is same or not. Please refer to wiki and implementation for details.
    weight_column: Optional[str] = None
        the column name of representative weights
    sample_weighted, default is False.
        If it is True/False, return the weighted/unweighted results of the metric

    -----------
    Note that the p-values in the output are information to help understand how different the means are. It is up to the team to make the final decision on whether the means are close enough.

    True mean is the value that we compare with the weighted or unweighted sample mean from labeled data set.  It can be
    - population mean of model prediction based labels or calibrated scores, or
    - (weighted) mean of another source of Ground Truth such as Survey data, or
    - a value that we believe to be the true mean (e.g. the mean setting in simulations, or from expert's information).

    """

    """
    Note that the p-values of ttests in the output are information to help understand how different the means are.
    It is up to the team to make the final decision on whether the means are close enough.
    """

    weight_column = weight_column if sample_weighted else None
    asmd_metric = asmd(
        df=df,
        labels_to_measure_column=labels_to_measure_column,
        golden_labels_column=golden_labels_column,
        label_type=label_type,
        weight_column=weight_column,
        sample_weighted=sample_weighted,
    )

    one_sample_ttest_metric = one_sample_ttest(
        df=df,
        labels_to_measure_column=labels_to_measure_column,
        true_mean=true_mean,
        label_type=label_type,
        weight_column=weight_column,
        sample_weighted=sample_weighted,
    )

    two_sample_ttest_metric = two_sample_ttest(
        df=df,
        labels_to_measure_column=labels_to_measure_column,
        golden_labels_column=golden_labels_column,
        label_type=label_type,
        unequal_var_alpha=unequal_var_alpha,
        weight_column=weight_column,
        sample_weighted=sample_weighted,
    )

    if label_type == "categorical":
        chisquare_test_metric = chisquare_test(
            df=df,
            labels_to_measure_column=labels_to_measure_column,
            golden_labels_column=golden_labels_column,
            categorized_buckets=categorized_buckets,
            weight_column=weight_column,
            sample_weighted=sample_weighted,
        )

        report = pd.DataFrame(
            data={
                "aggregated_accuracy_metric": [asmd_metric]
                + [list(one_sample_ttest_metric)]
                + [list(two_sample_ttest_metric)]
                + [chisquare_test_metric]
            },
            index=[
                "asmd",
                "1_sample_ttest_with_pvalue",
                "2_sample_ttest_with_pvalue",
                "chisquare_test_with_pvalue",
            ],
        )
    else:
        report = pd.DataFrame(
            data={
                "aggregated_accuracy_metric": [asmd_metric]
                + [list(one_sample_ttest_metric)]
                + [list(two_sample_ttest_metric)]
            },
            index=["asmd", "1_sample_ttest_with_pvalue", "2_sample_ttest_with_pvalue"],
        )

    return report


# --- validity checks for jobs without the golden set --- #
def get_kendalltau(
    df, labels_to_measure_column="labels", proxy_labels_column="proxy_fof_adult_pct"
):
    df = df[df[proxy_labels_column].notnull()]
    return kendalltau(df[labels_to_measure_column], df[proxy_labels_column])


def get_spearmanr(
    df, labels_to_measure_column="labels", proxy_labels_column="proxy_fof_adult_pct"
):
    df = df[df[proxy_labels_column].notnull()]
    return spearmanr(df[labels_to_measure_column], df[proxy_labels_column])


def get_pearsonr(
    df, labels_to_measure_column="labels", proxy_labels_column="proxy_fof_adult_pct"
):
    df = df[df[proxy_labels_column].notnull()]
    return pearsonr(df[labels_to_measure_column], df[proxy_labels_column])


def get_validity_checks_report(
    df: pd.DataFrame, labels_to_measure_column: str, proxy_labels_columns: str
):
    """
    Calcualte validity checks for datasets without golden standard GT
    Returns
    ----------
    results: metrics results of kendalltau_with_pvalue, spearmanr_with_pvalue and pearsonr_with_pvalue. The p-values in the output are information to help understand how correlated the labels to the proxies. It is up to the team to make the final decision on whether the proxies and labels are correlated enough.

    Output labels that most conflict with proxies to deep dive.

    Parameters
    -----------
    df: pd.DataFrame
        Dataset with columns labels_to_measure_column and proxy_labels_columns
    labels_to_measure_column: str
        column name of labels to measure
    proxy_labels_columns
        column name of proxy labels that can reflect the groud truth of labeling job to some extent

    """

    report = pd.DataFrame(
        data={
            "validity_check_with_"
            + proxy_labels_columns[0]: [
                get_kendalltau(
                    df,
                    labels_to_measure_column=labels_to_measure_column,
                    proxy_labels_column=proxy_labels_columns[0],
                ),
                get_spearmanr(
                    df,
                    labels_to_measure_column=labels_to_measure_column,
                    proxy_labels_column=proxy_labels_columns[0],
                ),
                get_pearsonr(
                    df,
                    labels_to_measure_column=labels_to_measure_column,
                    proxy_labels_column=proxy_labels_columns[0],
                ),
            ],
            "validity_check_with_"
            + proxy_labels_columns[1]: [
                get_kendalltau(
                    df,
                    labels_to_measure_column=labels_to_measure_column,
                    proxy_labels_column=proxy_labels_columns[1],
                ),
                get_spearmanr(
                    df,
                    labels_to_measure_column=labels_to_measure_column,
                    proxy_labels_column=proxy_labels_columns[1],
                ),
                get_pearsonr(
                    df,
                    labels_to_measure_column=labels_to_measure_column,
                    proxy_labels_column=proxy_labels_columns[1],
                ),
            ],
        },
        index=[
            "kendalltau_with_pvalue",
            "spearmanr_with_pvalue",
            "pearsonr_with_pvalue",
        ],
    )

    df_with_rank = df.copy()

    for i in range(len(proxy_labels_columns)):
        df_with_rank = df_with_rank[df_with_rank[proxy_labels_columns[i]].notnull()]

    df_with_rank[labels_to_measure_column + "_rank"] = df_with_rank[
        labels_to_measure_column
    ].rank()
    df_with_rank["proxy_avg_rank"] = [0] * df_with_rank.shape[0]
    for i in range(len(proxy_labels_columns)):
        df_with_rank[proxy_labels_columns[i] + "_rank"] = df_with_rank[
            proxy_labels_columns[i]
        ].rank()
        df_with_rank["proxy_avg_rank"] += df_with_rank[
            proxy_labels_columns[i] + "_rank"
        ]

    df_with_rank["proxy_avg_rank"] = df_with_rank["proxy_avg_rank"] / len(
        proxy_labels_columns
    )

    df_with_rank["label_proxy_avg_rank_diff"] = abs(
        df_with_rank[labels_to_measure_column + "_rank"]
        - df_with_rank["proxy_avg_rank"]
    )

    labels_to_deep_dive = df_with_rank.sort_values(
        by="label_proxy_avg_rank_diff", ascending=False, na_position="first"
    ).head(20)

    return report.to_dict(), labels_to_deep_dive
