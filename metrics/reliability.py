# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numbers
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa
import simpledorff


# --- Basic metrics --- #
# Job agreement rate
# Annotator agreement rate


def job_agreement_rate(
    table: pd.DataFrame, job: str, decision: str, weight: Optional[str] = None
) -> float:
    """
    Calculate job agreement rate

    Params
    ----------
    table:     a dataframe with each row representing a job with one decision
    job:       column name of job id
    decision:  column name of label
    weight:    [optional] column name of sample weights; if None, all samples are assumed to have equal weights

    Return
    -----------
    job agreement rate

    """
    table_agg = table.groupby(job).agg(list)
    table_agg_len = len(table_agg)
    if not weight:
        wt = [1] * table_agg_len
    else:
        wt = [item[0] for item in table_agg[weight]]
    return sum(
        np.array([len(set(decisions)) == 1 for decisions in table_agg[decision]]) * wt
    ) / sum(wt)


def annotator_agreement_rate(table: pd.DataFrame, job: str, decision: str) -> float:
    """
    Calculate annotator agreement rate

    Params
    -----------
    table:     a dataframe with each row representing a job with one decision
    job:       column name of job id
    decision:  column name of label

    Return
    -----------
    annotator agreement rate

    """
    table_agg = table.groupby(job).agg(list)

    agree_pairs = [
        sum([k * (k - 1) // 2 for k in Counter(decisions).values()])
        for decisions in table_agg[decision]
    ]
    total_pairs = [
        len(decisions) * (len(decisions) - 1) // 2 for decisions in table_agg[decision]
    ]

    return sum(agree_pairs) / sum(total_pairs)


# --- Metrics for two annotators --- #
# Cohen’s κ  -- used for both ordinal and nominal labels.
# Kendall’s τ -- only be used for ordinal/continuous labels.
# Spearman’s ρ -- only be used for ordinal/continuous labels.
# Kendall's τ and Spearman's ρ are expected to be similar since they both assess the correlation between the age labels given by two annotators.
# Cohen's κ treats ordinal labels as nominal labels. Although it is similar to the other two metrics in this case, it can be much smaller than the other two metrics when the labels given by two annotators are close but not identical.


def cohen_kappa(
    table: pd.DataFrame, decision1: str, decision2: str, weight: Optional[str] = None
):
    """
    Calculate Cohen's kappa, can be applied on ordinal and nominal labels. This metric is only for the jobs with only two labelers.

    Params
    -----------
    table:      a dataframe with each row representing a job with double reviews
    decision1:  column name of decisions given by annotator 1
    decision2:  column name of decisions given by annotator 2
    weight:     [optional] column name of sample weights; if None, all samples are assumed to have equal weights

    Return
    -----------
    Cohen's kappa

    """
    if weight:
        weight = table[weight]

    # convet int32/bool to int64
    if is_numeric_dtype(table[decision1]):
        table[decision1] = table[decision1].astype(int)

    if is_numeric_dtype(table[decision2]):
        table[decision2] = table[decision2].astype(int)

    return cohen_kappa_score(table[decision1], table[decision2], sample_weight=weight)


def kendall_tau(table: pd.DataFrame, decision1: str, decision2: str):
    """
    Calculate Kendall's tau, can be applied on ordinal/continuous labels. This metric is only for the jobs with only two labelers.

    Params
    -----------
    table:      a dataframe with each row representing a job with double reviews
    decision1:  column name of decisions given by annotator 1
    decision2:  column name of decisions given by annotator 2

    Return:
    -----------
    Kendall's tau
    p-value of two-sided test

    """
    if (
        is_numeric_dtype(table[decision1])
        and is_numeric_dtype(table[decision2])
        and table[decision1].dtypes != bool
        and table[decision2].dtypes != bool
    ):
        return kendalltau(table[decision1], table[decision2])
    else:
        raise Exception(
            f"decisions should be numeric. Current type of decision1 is {table[decision1].dtypes}; Current type of decision2 is {table[decision2].dtypes}"
        )


def spearman_rho(table: pd.DataFrame, decision1: str, decision2: str):
    """
    Calculate Spearman's rho, can be applied on ordinal/continuous labels. This metric is only for the jobs with only two labelers.

    Params
    -----------
    table:      a dataframe with each row representing a job with double reviews
    decision1:  column name of decisions given by annotator 1
    decision2:  column name of decisions given by annotator 2

    Return
    -----------
    Spearman's rho
    p-value of two-sided test

    """
    if (
        is_numeric_dtype(table[decision1])
        and is_numeric_dtype(table[decision2])
        and table[decision1].dtypes != bool
        and table[decision2].dtypes != bool
    ):
        return spearmanr(table[decision1], table[decision2])
    else:
        raise Exception(
            f"decisions should be numeric. Current type of decision1 is {table[decision1].dtypes}; Current type of decision2 is {table[decision2].dtypes}"
        )


# --- Multi-annotator metrics --- #
# Fleiss’s κ
# Krippendorff's α


# Fleiss’s κ
def job_to_annotator_job_matrix(table, job, decision, annotator):
    """
    Convert job table into a annotator by job matrix: the element in this matrix is either
        1) a label if the annotator provides a label to the job
        2) NaN if the annotator does not provide any label to the job

    Params
    -----------
    table:      a dataframe with each row representing a job with a decision
    job:        column name of job id
    decision:   column name of label
    annotator:  column name of annotator id

    Return
    -----------
    A 'num_annotators * num_jobs' pandas matrix
    """
    return table.pivot_table(
        values=decision, index=annotator, columns=job, aggfunc="first"
    )

def row_counter(row, table_dict):

    vals = row.dropna().values
    index = row.name

    if index not in table_dict:
        table_dict[index] = Counter()
    for val in vals:
        table_dict[index][val] += 1
    return

def make_value_by_unit_table_dict(annotator_job_matrix):
    table_dict = {}
    data_by_exp = annotator_job_matrix.T.sort_index(axis=1).sort_index()
    data_by_exp.apply(lambda x: row_counter(x, table_dict), axis=1)
    return table_dict

def calculate_frequency_dicts(vbu_table_dict):

    vbu = (
        pd.DataFrame.from_dict(vbu_table_dict, orient="index")
        .T.sort_index(axis=0)
        .sort_index(axis=1)
        .fillna(0)
    )
    ubv = vbu.T
    vbu_masked = ubv.mask(ubv.sum(1) == 1, other=0).T
    return dict(
        ubv_masked=vbu_masked.T,
    )


def fleiss_kappa_helper(table, job, annotator_num):
    count_num_decisions = Counter(table[job])
    job_id_with_double_review = np.array(list(count_num_decisions.keys()))[
        [item == annotator_num for item in count_num_decisions.values()]
    ]
    label_all_job_with_double_review = table.iloc[
        [item in job_id_with_double_review for item in table[job]]
    ]

    return label_all_job_with_double_review


def fleiss_kappa_shadow(
    table: pd.DataFrame, job: str, decision: str, annotator: str, annotator_num: int = 2
) -> float:
    """
    Calculate Fleiss's kappa
    Note: All jobs must receive the same number of labels. Specify the number of labels via parameter annotator_num. The function will help filter out qualified jobs.

    Params
    -----------
    table:      a dataframe with each row representing a job with a decision
    job:        column name of job id
    decision:   column name of label
    annotator:  column name of annotator id
    annotator_num: the number of labels of dataset, default is 2

    Return
    -----------
    Fleiss's kappa

    """
    table = fleiss_kappa_helper(table, job, annotator_num)

    # Convert job table into a annotator by job matrix: the element in this matrix is either
    #     1) a label if the annotator provides a label to the job
    #     2) NaN if the annotator does not provide any label to the job
    # Return A 'num_annotators * num_jobs' pandas matrix
    table_to_matrix = table.pivot_table(
        values=decision, index=annotator, columns=job, aggfunc="first"
    )
    vbu_table_dict = make_value_by_unit_table_dict(table_to_matrix)
    frenquency_dict = calculate_frequency_dicts(vbu_table_dict)

    return fleiss_kappa(frenquency_dict["ubv_masked"])


# Krippendorff's α

# This metric is the most advanced and flexible metric among all metrics we present here.
# It can be used for any label data type since we can adjust the loss functions as shown in the cell below.
# It includes many metrics discussed in previous sections as special cases under certain circumstances. See wikipedia for more details.


# Define loss functions for different types of label data
# It is easy to create user-defined loss functions
def nominal_metric(x, y):
    return 1 if x != y else 0


def ratio_metric(x, y):
    assert isinstance(x, numbers.Number) and isinstance(
        y, numbers.Number
    ), f"When using ratio metric, the tool expects the decisions are numeric. Current type is {type(x)} and {type(y)}"

    return ((x - y) / (x + y)) ** 2


def interval_metric(x, y):
    assert isinstance(x, numbers.Number) and isinstance(
        y, numbers.Number
    ), f"When using interval metric, the tool expects the decisions are numeric. Current type is {type(x)} and {type(y)}"
    return (x - y) ** 2


def loss_function_dispatcher(loss_func: str):
    dispatcher = {
        "nominal": nominal_metric,
        "ratio": ratio_metric,
        "interval": interval_metric,
    }
    return dispatcher[loss_func]

def calculate_krippendorffs_alpha(
    table: pd.DataFrame,
    job: str,
    decision: str,
    annotator: str,
    metric_fn: str = "nominal",
) -> float:
    """
    Calculate Krippendorff's alpha. It is for jobs with multiple labelers.

    Params
    -----------
    table:      a dataframe with each row representing a job with a decision
    job:        column name of job id
    decision:   column name of label
    annotator:  column name of annotator id
    metric_fn:  user-defined metric function. Available options: "nominal", "ratio", or "interval"

    Return
    -----------
    Krippendorff's alpha

    """
    return simpledorff.calculate_krippendorffs_alpha_for_df(table, job, annotator, decision,loss_function_dispatcher(metric_fn))
    # return simpledorff.calculate_krippendorffs_alpha_for_df(table, =job, annotator_col=annotator, class_col=decision, metric_fn = metric_fn)
