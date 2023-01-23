# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Optional

import numpy as np
import pandas as pd


# --- Helper Functions --- #
def merge_label_commercial_datsets(
    label_data: pd.DataFrame,
    commercial_report: pd.DataFrame,
    common_cols: List[str],
    cost_per_second: str,
    handling_time_second: str,
) -> pd.DataFrame:
    """
    Merge the label dataset and the commercial report dataset based on the given common columns.

    Return merged dataset with the cpd(cost per decison) column calculated based on cost_per_second and handling_time_second

    Parameters
    ----------
    label_data: pd.DataFrame
        label dataset with decision/label values and handling times across each group (vertical, language, vendor, labeler_type).
    commercial_report: pd.DataFrame
        commercial report with cost per time unit for each group (program, project, language, vendor, labeler_type)
    common_cols: list[str]
        a list of common columns used to merge two tables
    cost_per_second: str
        column name of cost per second in commercial report
    handling_time_second: str
        column name of handling time in second in label dataset
    """
    merged_dat = pd.merge(
        label_data,
        commercial_report,
        how="left",
        on=common_cols,
    )
    # "cpd" cost per decision
    merged_dat["cpd"] = merged_dat[handling_time_second] * merged_dat[cost_per_second]

    return merged_dat


def get_cnt_converted(x: Any, non_converted_list: List[str]):
    return np.sum(~x.isin(non_converted_list))


def get_conversion_rate(x: Any, non_converted_list: List[str]):
    return np.mean(~x.isin(non_converted_list))


def get_conversion_rate_std(x: Any, non_converted_list: List[str]):
    return np.std(~x.isin(non_converted_list)) / np.sqrt(len(x))


# The goal of this function is to calculate the converted cost.
# input_column_names: column names for cost per attempted labels
# output_column_names: column names for the converted cost
def get_cost_per_converted_metrics(
    df: pd.DataFrame,
    input_column_names: List[str],
    output_column_names: List[str],
    conversion_rate_column: str,
) -> pd.DataFrame:

    for input_name, output_name in zip(input_column_names, output_column_names):
        df[output_name] = df[input_name] / df[conversion_rate_column]

    return df


def get_average_metric_std(x):
    return np.std(x) / np.sqrt(len(x))


# --- Metrics --- #

# ht: handling time in sec per decision
# cpd: usd cost per decision
# ht and cpd in cost summaries are all avg estimates, and std are std of avg estimate.
# non_converted_list: decision values that are not usable/convertable

# Task level
# to evaluate on the labelling cost per attempted/converted decision/label across the inputted general_group_list and additional_group_list of task level.


def get_task_level_cost_summary(
    dataset: pd.DataFrame,
    detailed_group_list: List[str],
    non_converted_list: List[str],
    decision_value_column: str,
    handling_time_second: str,
    cpd: str,
    time_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate the task (decision/response) level cost summary per attempted or per converted label across groups and labeler_types.

    Returns a table including cost per attempted label and cost per converted label. The table is grouped by the given parameter "detailed_group_list" and time if time_column is not specified.

    Parameters
    ----------
    dataset : pd.DataFrame
        A table includes columns of decision/label values, handling time in second, and cost per decision across each group (vertical, language, vendor, labeler_type).
    detailed_group_list : list of str
        A list of column names to specify how to group the data when calculating the metrics.
    non_converted_list : list
        A list of decision values that are not usable/convertable.
    decision_value_column : str
        The name of the column with decision values.
    handling_time_second : str
        Handling time per second per labeling task.
    cpd : str
        Cost per decision
    time_column : str, optional, default = None
        Column name of dates. If it's not None, the results will be also grouped by date.

    """

    if time_column:
        detailed_group_list = [time_column] + detailed_group_list

    task_level_summary = (
        dataset[
            detailed_group_list + [decision_value_column] + [handling_time_second, cpd]
        ]
        .groupby(detailed_group_list)
        .agg(
            cnt_attempted=(decision_value_column, "count"),
            cnt_converted=(
                decision_value_column,
                lambda x: get_cnt_converted(x, non_converted_list),
            ),
            conversion_rate=(
                decision_value_column,
                lambda x: get_conversion_rate(x, non_converted_list),
            ),
            conversion_rate_sd=(
                decision_value_column,
                lambda x: get_conversion_rate_std(x, non_converted_list),
            ),
            ht_attempted=(handling_time_second, "mean"),
            ht_attempted_std=(handling_time_second, get_average_metric_std),
            cpd_attempted=(cpd, "mean"),
            cpd_attempted_std=(cpd, get_average_metric_std),
        )
        .sort_values(by=detailed_group_list, ascending=True)
    )

    task_level_summary = get_cost_per_converted_metrics(
        task_level_summary,
        input_column_names=[
            "ht_attempted",
            "ht_attempted_std",
            "cpd_attempted",
            "cpd_attempted_std",
        ],
        output_column_names=[
            "ht_converted",
            "ht_converted_std",
            "cpd_converted",
            "cpd_converted_std",
        ],
        conversion_rate_column="conversion_rate",
    )
    task_level_summary = task_level_summary.loc[
        ~np.isnan(task_level_summary["cpd_attempted"])
    ]  # Optional: delete rows with na cpd

    return task_level_summary


# Survey
# to evaluate the labels gotten via on-platform survey. Since the cost cannot be evaluated by currency and assume the cost per attempt is 1 (1 attempted invitation/impression), no commercial report is required for this option.
def get_survey_cost_summary(
    dataset: pd.DataFrame,
    detailed_group_list: List[str],
    decision_value_column: str,
    non_converted_list: List[str],
    time_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate the cost summary for survey data.

    Returns a table including cost per converted label. The table is grouped by the given parameter "detailed_group_list" and time if time_column is not specified.

    Parameters
    ----------
    dataset : pd.DataFrame
        A table includes columns of decision/label values, handling time in second, and cost per decision across each group (vertical, language, vendor, labeler_type).
    detailed_group_list : list of str
        A list of column names to specify how to group the data when calculating the metrics.
    decision_value_column : str
        The name of the column with decision values.
    non_converted_list : list
        A list of decision values that are not usable/convertable.
    time_column : str, optional, default = None
        Column name of dates. If it's not None, the results will be also grouped by date.

    """
    if time_column:
        detailed_group_list = [time_column] + detailed_group_list

    survey_level_summary = (
        dataset[detailed_group_list + [decision_value_column]]
        .groupby(detailed_group_list)
        .agg(
            cnt_attempted=(decision_value_column, "count"),
            cnt_converted=(
                decision_value_column,
                lambda x: get_cnt_converted(x, non_converted_list),
            ),
            conversion_rate=(
                decision_value_column,
                lambda x: get_conversion_rate(x, non_converted_list),
            ),
            conversion_rate_sd=(
                decision_value_column,
                lambda x: get_conversion_rate_std(x, non_converted_list),
            ),
        )
        .sort_values(by=detailed_group_list, ascending=True)
    )

    # Assume cost_attempted is 1 (1 attempted invitation/impression)
    # therefore, cost_converted = cost_attempted / conversion_rate  = 1 / conversion_rate

    (
        survey_level_summary["cost_attempted"],
        survey_level_summary["cost_converted"],
    ) = zip(
        *survey_level_summary.apply(lambda row: (1, 1 / row["conversion_rate"]), axis=1)
    )

    return survey_level_summary


# target level
# to evaluate on labelling cost per final label across the inputted general_group_list and to combine cost of each target across multiple tasks with different types of labelers.
# cpd:  (usd) cost per decision
def get_target_level_cost_summary(
    dataset: pd.DataFrame,
    general_group_list: List[str],
    final_decision_column: str,
    ids: List[str],
    labeler_type: str,
    non_converted_list: List[str],
    decision_value_column: str,
    handling_time_second: str,
    cpd: str,
    filtered_out_type: Optional[List[str]] = [],
    time_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate the target (e.g. account/post/edge, i.e. the item to be labeled. Different from the task level, one target could be multi-reviewed by multiple labeling tasks.) level cost summary per attempted or per converted label across groups and labeler_types.

    Returns a table including cost per final label. The table is grouped by the given parameter "general_group_list" and time if time_column is not specified.

    Parameters
    ----------
    dataset : pd.DataFrame
        A table includes columns of decision/label values, handling time in second, and cost per decision across each group (vertical, language, vendor, labeler_type).
    general_group_list : list of str
        A list of column names to specify how to group the data when calculating the metrics.

    final_decision_column: str
        The name of the column with final decision values.
    ids: a list of str
        The column names for ids.
    labeler_type: str
        The labeler type column name
    non_converted_list : list
        A list of decision values that are not usable/convertable.
    decision_value_column : str
        The name of the column with decision values.
    handling_time_second : str
        Handling time per second per labeling task.
    cpd : str
        Cost per decision

    filtered_out_type: a list of str, optional, default = []
        Specify the labeler types that should not be considered in the cost of final labels

    time_column : str, optional, default = None
        Column name of dates. If it's not None, the results will be also grouped by date.

    """

    if time_column:
        general_group_list = [time_column] + general_group_list

    # TODO optimize the following code snippet to avoid deep copy
    dataset_sel = dataset.copy()
    if len(filtered_out_type) > 0:
        dataset_sel = dataset.loc[dataset[labeler_type] not in filtered_out_type]

    prep_for_final = dataset_sel.groupby(ids + general_group_list).agg(
        final_decision=(final_decision_column, "first"),
        cnt_task=(decision_value_column, "count"),
        ht=(handling_time_second, "sum"),
        cpd=(cpd, "sum"),
    )

    target_level_summary = prep_for_final.groupby(general_group_list).agg(
        cnt_task=("cnt_task", "sum"),
        cnt_target=(final_decision_column, "count"),
        cnt_final=(
            final_decision_column,
            lambda x: get_cnt_converted(x, non_converted_list),
        ),
        final_conversion_rate=(
            final_decision_column,
            lambda x: get_conversion_rate(x, non_converted_list),
        ),
        final_conversion_rate_std=(
            final_decision_column,
            lambda x: get_conversion_rate_std(x, non_converted_list),
        ),
        ht_target=("ht", "mean"),
        ht_target_std=("ht", get_average_metric_std),
        cpd_target=(cpd, "mean"),
        cpd_target_std=(cpd, get_average_metric_std),
    )

    target_level_summary = get_cost_per_converted_metrics(
        target_level_summary,
        input_column_names=[
            "ht_target",
            "ht_target_std",
            "cpd_target",
            "cpd_target_std",
        ],
        output_column_names=["ht_final", "ht_final_std", "cpd_final", "cpd_final_std"],
        conversion_rate_column="final_conversion_rate",
    )

    return target_level_summary
