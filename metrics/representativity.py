
#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Representativity dimension in GTMF is implemented via SQL/Hive. We also include this part in the Open Source package Balance. Please refer to following link for more information: https://import-balance.org/.

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from balance import Sample
from some_query_lib import Dataset, load, query, upload

some_query_lib_LOAD_MAX_SIZE = 3e10

DEFAULT_QUANTILES = [0.2, 0.4, 0.6, 0.8]


# Population Data Preparation
def get_population_features(population_data_hive_namespace, population_data_hive_table):
    population_features = Dataset(
        namespace=population_data_hive_namespace,
        table=population_data_hive_table,
    )
    return population_features


# bento kernel api
def discretize_covariates(
    population_data_hive_namespace: str,
    population_data_hive_table: str,
    primary_key: str,
    continuous_covars: List[str],
    continuous_covars_dataset: List[Dict[str, Any]],
    discrete_covars: List[str],
    holdout_covar: Optional[str] = None,
):
    """
    Discretize Covariates in the population feature table and construct new population
    feature table with new discretized covariates value column and other existing columns

    Params
    -------------
    population_data_hive_namespace:
        hive table namespace of dataset with all covars
    population_data_hive_table:
        hive table name of dataset with all covars
    primary_key:
        primary key for both population data table and sample data table. The primary key must be single column.
    continuous_covars:
        continues covariate columns need to discretize
    continuous_covars_dataset:
        continues covariate columns need to discretize and quantile and null value when continuous covariate is None
    discrete_covars:
        discrete_covars which need to construct new table

    --- Optional ---
    holdout_covar:
        holdout_covar which need to construct new table

    Return:
    a some_query_lib data set with new discretized covariates value column and other existing columns
    """

    population_features = get_population_features(
        population_data_hive_namespace,
        population_data_hive_table,
    )
    feature_quantiles = calculate_quantiles(
        population_features, continuous_covars, continuous_covars_dataset
    )
    return table_with_discretized_covariates(
        primary_key,
        population_features,
        feature_quantiles,
        continuous_covars,
        discrete_covars,
        holdout_covar,
    )


# Helper function for discretizing continuous covars
def quantiles(covariate, covar_dataset):
    null_value = covar_dataset.get("null_value", "")
    quantile_percentile = covar_dataset.get("quantile_percentile", DEFAULT_QUANTILES)
    quantile_percentile_str = ",".join(map(str, quantile_percentile))
    if null_value == "":
        return f"""
        ARRAY_DISTINCT(
            APPROX_PERCENTILE({covariate}, ARRAY[{quantile_percentile_str}])
        )
        """
    else:
        return f"""
        ARRAY_DISTINCT(
            {null_value} || APPROX_PERCENTILE(
                COALESCE({covariate}, {null_value}),
                ARRAY[{quantile_percentile_str}]
            )
        )
        """


# this function is only used for continuous covariates
def value_to_quantile(covariate):
    return f"""
    REDUCE(
        quantiles.{covariate},
        0,
        (s, x) -> IF(x < popn.{covariate}, s + 1, s),
        s -> CAST(s AS VARCHAR)
    )
    """


# Calculate quantiles for continuous covariates
def calculate_quantiles(
    population_features, continuous_covars, continuous_covars_dataset
):
    raw_sql = """
        SELECT
            {select_clause}
        FROM
            {{population_features}}
        """
    select_clause = ", ".join(["{" + x + "} AS " + x for x in continuous_covars])

    sql = raw_sql.format(select_clause=select_clause)

    feature_quantiles_refs_dict = dict(population_features=population_features)
    feature_quantiles_refs_dict["select_clause"] = query.fragment(sql=select_clause)
    for covar in continuous_covars:
        covar_dataset = next(x for x in continuous_covars_dataset if x["name"] is covar)
        if covar_dataset.get("quantile_literal"):
            arr = covar_dataset.get("quantile_literal")

            sub_sql = "ARRAY_DISTINCT(ARRAY[{arr}])".format(arr=",".join(map(str, arr)))
        else:
            sub_sql = quantiles(covar, covar_dataset)

        feature_quantiles_refs_dict[covar] = query.fragment(sql=sub_sql)

    return query(
        sql=sql,
        refs=feature_quantiles_refs_dict,
        custom_name="query_calculate_quantiles",
    )


# Discretize continuous covariates based on quantiles
def table_with_discretized_covariates(
    primary_key: str,
    popn_features,
    feature_quantiles,
    continuous_covars: List[str],
    discrete_covars: List[str],
    holdout_covar: Optional[str] = None,
):
    raw_sql = """
    SELECT
        {select_clause}
    FROM
        {{popn_features}} popn
    JOIN
        {{feature_quantiles}} quantiles on True
    """

    select_values = []
    select_values.append("popn." + primary_key)
    select_values.extend(["{" + x + "} AS " + x for x in continuous_covars])
    select_values.extend(["popn." + x for x in discrete_covars])
    if holdout_covar:
        select_values.append("popn." + holdout_covar)

    select_clause = ", ".join(select_values)
    sql = raw_sql.format(select_clause=select_clause)

    refs_dict = dict(
        popn_features=popn_features,
        feature_quantiles=feature_quantiles,
        select_clause=select_clause,
    )

    for covar in continuous_covars:
        refs_dict[covar] = query.fragment(sql=value_to_quantile(covar))

    return query(
        sql=sql, refs=refs_dict, custom_name="query_table_with_discretized_covariates"
    )


# bento kernel api

# Create segments corresponding to unique values of the discretized covariates
# Weights in the population sum to 1, and for each segment is proportional to the number of users in that segments
# Scores is a TDIGEST (histogram) of the holdout covariate
# Currently this only works for continuous holdout covariate.
def summarize_target_population(
    popn_quantiles,
    continuous_covars: List[str],
    discrete_covars: List[str],
    holdout_covar: Optional[str] = None,
    weight_col: Optional[str] = "weight",
):
    """
    Create segments corresponding to unique values of the discretized covariates

    Params
    -------------
    popn_quantiles:
        a some_query_lib dataset of population with discretized covariates value column
    continuous_covars:
        continues covariate columns need to discretize
    discrete_covars:
        discrete_covars which need to construct new table

    --- Optional ---
    holdout_covar
        holdout_covar which need to construct new table
    weight_col
        column name of weight

    Return:
    -------------
    a some_query_lib dataset with segments corresponding to unique values of the discretized covariates.
    The weight column represents the weight of each segment instance. SUM(weight * num_records) = 1
    Scores is a TDIGEST (histogram) of the holdout covariate

    """
    select_values = []
    select_values.extend(continuous_covars)
    select_values.extend(discrete_covars)
    select_values.append(
        "1.0 * COUNT(*) / ARBITRARY(popn_counts.num_records) AS {weight}".format(
            weight=weight_col
        )
    )
    select_values.append("COUNT(*) AS num_records")
    if holdout_covar:
        select_values.append(
            "CAST(TDIGEST_AGG({holdout_covar}, 1, 500) AS VARBINARY) AS {holdout_covar}".format(
                holdout_covar=holdout_covar
            )
        )
    select_clause = ", ".join(select_values)

    group_by_clause = ",".join(continuous_covars + discrete_covars)

    target_popn = query(
        sql="""
        WITH popn_counts AS (
            SELECT
                COUNT(*) AS num_records
            FROM {popn_quantiles}
        )
        SELECT
            {select_clause}
        FROM
            {popn_quantiles}
        JOIN popn_counts on TRUE
        GROUP BY
            {group_by_clause}
        """,
        refs=dict(
            select_clause=query.fragment(sql=select_clause),
            group_by_clause=query.fragment(sql=group_by_clause),
            popn_quantiles=popn_quantiles,
        ),
        custom_name="query_summarize_target_population",
    )
    return target_popn


# bento kernel api
# Construct sampling table by joining population table
def sample_data_setting(
    sample_namespace,
    sample_hive_tablename,
    primary_key,
    popn_quantiles,
    continuous_covars,
    discrete_covars,
    holdout_covar=None,
    inclusive_columns_in_sample=None,
    equal_weight=False,  # False when not using balance
    weight_col: Optional[str] = "weight",
):
    """
    Construct sampling table by joining population table

    Params
    -------------
    sample_namespace:
        hive table namespace of sample dataset
    sample_hive_tablename:
        hive table name of sample dataset
    primary_key:
        primary key for both population data table and sample data table. The primary key must be single column.
    popn_quantiles:
        a some_query_lib dataset of population with discretized covariates value column
    continuous_covars:
        continues covariate columns need to discretize
    discrete_covars:
        discrete_covars which need to construct new table

    --- Optional ---
    holdout_covar:
        holdout_covar which need to construct new table
    inclusive_columns_in_sample:
        The column name for ground truth labels or
                                     the non-covar column in sample data
    equal_weight:
        Boolean whether to set the weight column to 1/n
    weight_col:
        (Optional) column name of weight

    Return:
    a some_query_lib data set with new discretized covariates value column and other existing columns

    """
    sample = Dataset(namespace=sample_namespace, tablename=sample_hive_tablename)
    select_values = []
    select_values.append("popn." + primary_key)
    select_values.extend(["popn." + x for x in continuous_covars])
    select_values.extend(["popn." + x for x in discrete_covars])
    if holdout_covar:
        select_values.append("popn." + holdout_covar)

    if equal_weight:
        if (
            inclusive_columns_in_sample
            and weight_col not in inclusive_columns_in_sample
            or not inclusive_columns_in_sample
        ):
            print("GTMF: Setting sample weight to 1/n...")
            select_values.append(
                "1.0 / {n} AS {weight}".format(n=sample.num_rows, weight=weight_col)
            )

    if inclusive_columns_in_sample:
        select_values.extend(["sample." + x for x in inclusive_columns_in_sample])
    select_clause = ", ".join(select_values)

    sample_features = query(
        sql="""
        SELECT
            {select_clause}
        FROM
            {popn_quantiles} popn
        JOIN {sample} sample
            ON popn.{primary_id_column} = sample.{primary_id_column}
        """,
        refs=dict(
            select_clause=query.fragment(sql=select_clause),
            primary_id_column=query.fragment(sql=primary_key),
            popn_quantiles=popn_quantiles,
            sample=sample,
        ),
        custom_name="query_sample_data_setting",
    )

    return sample_features


def format_sample_features(
    sample_features,
    continuous_covars,
    discrete_covars,
    holdout_covar=None,
    equal_weight=False,
    weight_col="weight",
):

    select_values = []
    select_values.append("*")
    if not equal_weight:
        select_values.append(f"1.0 AS {weight_col}")
        # otherwise keep them at 1/n from sample_features

    select_clause = ", ".join(select_values)

    sample_features_formatted_ds = query(
        sql="""
        SELECT
            {select_clause}
        FROM
            {sample_features}
        """,
        refs=dict(
            select_clause=query.fragment(sql=select_clause),
            sample_features=sample_features,
        ),
        custom_name="query_format_sample_features",
    )

    return load(
        dataset=sample_features_formatted_ds,
        max_size=some_query_lib_LOAD_MAX_SIZE,
        custom_name="load_format_sample_features",
    )


def format_target_popn(
    target_popn,
    continuous_covars,
    discrete_covars,
    primary_key,
    weight_col: Optional[str] = "weight",
):
    select_values = []
    select_values.extend(continuous_covars)
    select_values.extend(discrete_covars)
    select_values.append(weight_col)
    select_values.append(
        "-1 * ROW_NUMBER() OVER(ORDER BY 1) AS {primary_id_column}".format(
            primary_id_column=primary_key
        )
    )
    select_clause = ", ".join(select_values)

    format_target_popn = query(
        sql="""
        SELECT
            {select_clause}
        FROM
            {target_popn}
        """,
        refs=dict(
            select_clause=query.fragment(sql=select_clause), target_popn=target_popn
        ),
        custom_name="query_format_target_popn",
    )

    return load(
        dataset=format_target_popn,
        max_size=some_query_lib_LOAD_MAX_SIZE,
        custom_name="load_format_target_popn",
    )


def process_sample_data_combined(
    sample_features,
    target_popn,
    primary_key,
    continuous_covars,
    discrete_covars,
    interactions,
    balance_arguments,
    holdout_covar=None,
    output_sample_weight_balance_table=None,
    equal_weight=False,
    weight_col: Optional[str] = "weight",
):
    covars = continuous_covars + discrete_covars

    # Load sample into Python for balance
    print("GTMF: Loading sample features...")
    sample_features_formatted = format_sample_features(
        sample_features,
        continuous_covars,
        discrete_covars,
        holdout_covar,
        equal_weight,
        weight_col,
    )

    # Load population into Python for balance
    print("GTMF: Loading target popn...  -- ", weight_col)

    target_popn_formatted = format_target_popn(
        target_popn, continuous_covars, discrete_covars, primary_key, weight_col
    )

    return process_sample_data(
        sample_features_formatted,
        target_popn_formatted,
        primary_key,
        covars,
        interactions,
        balance_arguments,
        output_sample_weight_balance_table,
        weight_col,
    )


def process_sample_data(
    sample_features_formatted,
    target_popn_formatted,
    primary_key,
    covars,
    interactions,
    balance_arguments,
    output_sample_weight_balance_table=None,
    weight_col: Optional[str] = "weight",
):
    print("GTMF: generate sample object, weight column - ", weight_col)
    sampleobj = Sample.from_frame(
        sample_features_formatted[covars + [primary_key, weight_col]],
        id_column=primary_key,
        weight_column=weight_col,
    )

    print("GTMF: generate target object, weight column - ", weight_col)
    targetobj = Sample.from_frame(
        target_popn_formatted[covars + [primary_key, weight_col]],
        id_column=primary_key,
        weight_column=weight_col,
    )

    sampleobj = sampleobj.set_target(targetobj)

    print("GTMF: getting balance weight...")
    if balance_arguments:
        balance_kwargs = balance_arguments
    else:
        balance_kwargs = _build_default_balance_kwargs(covars, interactions)

    balance_adjust = sampleobj.adjust(**balance_kwargs)

    # Send data back to Hive in order to calculate representativity metrics
    # With existing weights, this section can be skipped
    sample_features_formatted[weight_col] = balance_adjust.weights().df

    if not output_sample_weight_balance_table:
        output_sample_weight_balance_table = {}
    output_sample_weight_balance_table[
        "namespace"
    ] = output_sample_weight_balance_table.get("namespace", "datascience")
    output_sample_weight_balance_table[
        "oncall"
    ] = output_sample_weight_balance_table.get("oncall", "pdgt_oncall")

    print("GTMF: uploading sample weight to hive:")
    print(output_sample_weight_balance_table)

    output_sample_weight_balance_table["dataframe"] = sample_features_formatted

    sample_weights = upload(**output_sample_weight_balance_table)
    print(
        f"GTMF: sample weight uploaded to {sample_weights.tablename}:{sample_weights.namespace}"
    )
    return sample_weights


def _build_default_balance_kwargs(covars, interactions):
    return dict(
        method="ipw",
        formula=[
            "~ "
            + "+".join(["*".join(x) for x in interactions])
            + "+"
            + "+".join(covars)
        ],
        transformations="default",
        weight_trimming_mean_ratio=0.5,
    )


def get_design_effect(
    sample_weight,
    weight_col: Optional[str] = "weight",
):

    sample_features_formatted = load(
        dataset=query(
            sql="""
                SELECT
                    {weight}
                FROM {sample_weight}
            """,
            refs=dict(
                sample_weight=sample_weight, weight=query.fragment(sql=weight_col)
            ),
            custom_name="query_get_design_effect",
        ),
        max_size=some_query_lib_LOAD_MAX_SIZE,
        custom_name="load_get_design_effect",
    )
    return calulate_design_effect(sample_features_formatted, weight_col)


# bento kernel api
def calulate_design_effect(sample_features_formatted, weight_col):
    return (
        len(sample_features_formatted[weight_col])
        * sum(pow(sample_features_formatted[weight_col], 2))
        / pow(sum(sample_features_formatted[weight_col]), 2)
    )


# bento kernel api
def get_covariate_balance(
    target_popn, sample_weights, covars, weight_col: Optional[str] = "weight"
):
    return query(
        sql="""
        WITH popn_prop AS (
            SELECT
                {covars},
                SUM({weight}) AS popn_prop
            FROM {target_popn}
            GROUP BY
                GROUPING SETS ({covars_grouping_sets})
        ),
        sample_prop AS (
            SELECT
                {covars},
                SUM({weight}) AS sample_prop
            FROM {sample_weights}
            GROUP BY
                GROUPING SETS ({covars_grouping_sets})
        )
        SELECT
            {covars_p},
            popn_prop,
            COALESCE(sample_prop, 0) AS sample_prop
        FROM popn_prop p
        LEFT JOIN sample_prop s
            ON {covars_join}
        """,
        refs=dict(
            target_popn=target_popn,
            sample_weights=sample_weights,
            weight=query.fragment(sql=weight_col),
            covars=query.fragment(sql=", ".join(covars)),
            covars_grouping_sets=query.fragment(sql="(" + "), (".join(covars) + ")"),
            covars_p=query.fragment(sql="p." + ", p.".join(covars)),
            covars_join=query.fragment(
                sql=" OR ".join([f"p.{covar} = s.{covar}" for covar in covars])
            ),
        ),
        custom_name="query_get_covariate_balance",
    )


# if the weighted sample proportion / population proportion < cutoff
# we will consider the covariate-value to be underrepresented in the sample
def get_underrepresented(covariate_balance, coverage_cutoff):
    return query(
        sql="""
        SELECT
            *
        FROM {covariate_balance}
        WHERE
            sample_prop / popn_prop < {coverage_cutoff}
        """,
        refs=dict(covariate_balance=covariate_balance, coverage_cutoff=coverage_cutoff),
        custom_name="query_get_underrepresented",
    )


def get_underrepresented_segments(
    covariate_balance, coverage_cutoff, target_popn, covars, weight_col
):
    underrepresented = get_underrepresented(covariate_balance, coverage_cutoff)
    select_values = []
    select_values.extend(["t." + x for x in covars])
    select_values.append(f"ARBITRARY(t.{weight_col}) AS {weight_col}")

    select_clause = ", ".join(select_values)

    group_by_clause = ",".join(["t." + x for x in covars])

    return query(
        sql="""
        SELECT
            {select_clause}
        FROM
            {target_popn} t
        JOIN {underrepresented} u
            ON {covars_join}
        GROUP BY
            {group_by_clause}
        """,
        refs=dict(
            select_clause=query.fragment(sql=select_clause),
            group_by_clause=query.fragment(sql=group_by_clause),
            underrepresented=underrepresented,
            target_popn=target_popn,
            covars_join=query.fragment(
                sql=" OR ".join([f"t.{covar} = u.{covar}" for covar in covars])
            ),
        ),
        custom_name="query_get_underrepresented_segments",
    )


# bento kernel api
def get_mau_coverage(
    covariate_balance,
    coverage_cutoff,
    target_popn,
    covars,
    weight_col: Optional[str] = "weight",
):
    # Calculate segments in the population with any covariate-value that is underrepresented
    underrepresented_segments = get_underrepresented_segments(
        covariate_balance, coverage_cutoff, target_popn, covars, weight_col
    )
    mau_coverage = load(
        query(
            sql="""
            SELECT
                1 - SUM({weight}) AS prop
            FROM {underrepresented_segments}
            """,
            refs=dict(
                weight=query.fragment(sql=weight_col),
                underrepresented_segments=underrepresented_segments,
            ),
            custom_name="query_get_mau_coverage",
        ),
        max_size=some_query_lib_LOAD_MAX_SIZE,
        custom_name="load_get_mau_coverage",
    )

    return mau_coverage.iloc[0][0]


# bento kernel api
def get_holdout_distn_df(
    target_popn,
    sample_weights,
    holdout_covar,
    weight_col: Optional[str] = "weight",
):
    distn_df = load(
        query(
            sql="""
            WITH popn AS (
                SELECT
                    MERGE(CAST({holdout_covar} AS TDIGEST(DOUBLE))) AS popn_distn
                FROM {target_popn}
            ),
            discrete_weights AS (
                SELECT
                    {holdout_covar},
                    CAST(
                        ROUND(
                            100 * {weight} / MIN({weight}) OVER (
                                PARTITION BY
                                    1
                            )
                        ) AS BIGINT
                    ) AS discrete_weight
                FROM {sample_weights}
            ),
            sample AS (
                SELECT
                    TDIGEST_AGG({holdout_covar}, discrete_weight, 500) AS sample_distn
                FROM discrete_weights
            ),
            popn_values AS (
                SELECT
                    VALUES_AT_QUANTILES(popn_distn, {quantiles}) AS p_values
                FROM popn
            ),
            sample_values AS (
                SELECT
                    VALUES_AT_QUANTILES(sample_distn, {quantiles}) AS s_values
                FROM sample
            )
            SELECT
                'popn' AS variable,
                value
            FROM popn_values
            CROSS JOIN UNNEST(p_values) AS t1 (value)

            UNION ALL

            SELECT
                'sample' AS variable,
                value
            FROM sample_values
            CROSS JOIN UNNEST(s_values) AS t2 (value)
            """,
            refs=dict(
                target_popn=target_popn,
                sample_weights=sample_weights,
                weight=query.fragment(sql=weight_col),
                holdout_covar=query.fragment(sql=holdout_covar),
                quantiles=query.fragment(
                    sql="TRANSFORM(SEQUENCE(1, 999), x -> 0.001 * x)"
                ),
            ),
            custom_name="query_get_holdout_distn_df",
        ),
        max_size=some_query_lib_LOAD_MAX_SIZE,
        custom_name="load_get_holdout_distn_df",
    )

    return distn_df


# bento kernel api
def get_asmd(distn_df: pd.DataFrame) -> float:
    '''
    Return the result of asmd metric

    Params
    -------------
    holdout_distn_df: pd.DataFrame
        hold-out covariate for both the population and sample
    '''
    target_mean = np.mean(distn_df[distn_df["variable"] == "popn"].value)
    target_std = np.std(distn_df[distn_df["variable"] == "popn"].value)
    sample_mean = np.mean(distn_df[distn_df["variable"] == "sample"].value)

    return abs(target_mean - sample_mean) / target_std


def _thresholds(min_value, max_value, step):
    return f"""TRANSFORM(SEQUENCE({int(round(min_value/step,0))+1}, {int(round(max_value/step,0))-1}), x -> {step} * x)"""


def _pdf_from_cdf(array):
    return f"""
    REDUCE(
        {array},
        CAST(ARRAY[] AS ARRAY<DOUBLE>),
        (s, x) -> CONCAT(s, x - ARRAY_SUM(s)),
        s -> s
    )
    """


def _hellinger_dist(popn, test):
    return f"""
    SQRT(
        ARRAY_SUM(
            ZIP_WITH(
            {popn},
            {test}, (x, y) -> POW(SQRT(x) - SQRT(y), 2))
        )/2
    )
    """


# bento kernel api
def get_hellinger(
    target_popn,
    sample_weights,
    holdout_covar,
    hellinger_arguments,
    weight_col: Optional[str] = "weight",
):
    min_value = hellinger_arguments.get("min_value")
    max_value = hellinger_arguments.get("max_value")
    step = hellinger_arguments.get("step")
    return load(
        query(
            sql="""
            WITH popn AS (
                SELECT
                    MERGE(CAST({holdout_covar} AS TDIGEST(DOUBLE))) AS popn_distn
                FROM {target_popn}
            ),
            discrete_weights AS (
                SELECT
                    {holdout_covar},
                    CAST(
                        ROUND(
                            100 * {weight} / MIN({weight}) OVER (
                                PARTITION BY
                                    1
                            )
                        ) AS BIGINT
                    ) AS discrete_weight
                FROM {sample_weights}
            ),
            sample AS (
                SELECT
                    TDIGEST_AGG({holdout_covar}, discrete_weight, 500) AS sample_distn
                FROM discrete_weights
            ),
            popn_values AS (
                SELECT
                    QUANTILES_AT_VALUES(popn_distn, {thresholds}) AS cdf
                FROM popn
            ),
            sample_values AS (
                SELECT
                    QUANTILES_AT_VALUES(sample_distn, {thresholds}) AS cdf
                FROM sample
            )
            SELECT
                {hellinger_dist} AS hellinger_dist
            FROM popn_values
            JOIN sample_values
                ON TRUE
            """,
            refs=dict(
                target_popn=target_popn,
                sample_weights=sample_weights,
                weight=query.fragment(sql=weight_col),
                holdout_covar=query.fragment(sql=holdout_covar),
                thresholds=query.fragment(sql=_thresholds(min_value, max_value, step)),
                hellinger_dist=query.fragment(
                    sql=_hellinger_dist(
                        _pdf_from_cdf("popn_values.cdf"),
                        _pdf_from_cdf("sample_values.cdf"),
                    )
                ),
            ),
            custom_name="query_get_hellinger",
        ),
        max_size=some_query_lib_LOAD_MAX_SIZE,
        custom_name="load_get_hellinger",
    )


# bento kernel api
def get_covariate_balance_df(covariate_balance):
    return load(
        covariate_balance,
        max_size=some_query_lib_LOAD_MAX_SIZE,
        custom_name="load_get_covariate_balance_df",
    )


def gen_plots_of_covariates(covariate_balance, covars):
    covariate_balance_df = load(
        covariate_balance,
        max_size=some_query_lib_LOAD_MAX_SIZE,
        custom_name="load_gen_plots_of_covariates",
    )

    return proportion_plot(covariate_balance_df, covars)


# bento kernel api
def proportion_plot(covariate_balance_df: pd.DataFrame, covars: List[str]):
    """
    Return the proportion plot of covariate balance for each feature of both sample data and population data

    Params
    -------------
    covariate_balance_df: pd.DataFrame
        covariate balance of sample and population dataset
    covars: List[str]
        features that you would like to plot for
    """
    plots = []
    for covar in covars:
        fig, ax = plt.subplots()
        balance_df = covariate_balance_df.loc[
            :, (covar, "popn_prop", "sample_prop")
        ].dropna()
        balance_df = pd.melt(
            balance_df, id_vars=[covar], value_vars=["popn_prop", "sample_prop"]
        )
        balance_df.sort_values(covar, inplace=True)

        sns.barplot(data=balance_df, x=covar, y="value", hue="variable", ax=ax)
        ax.set_xlabel(covar)
        ax.set_ylabel("Proportion")
        ax.set_title("")

        plots.append(fig)

    return plots


# bento kernel api
def gen_plot_distribution_of_hold_out(
    holdout_distn_df: pd.DataFrame, holdout_covar: str
):

    """
    Return a plot of the distribution of the hold-out covariate for both the population and sample

    Params
    -------------
    holdout_distn_df: pd.DataFrame
        hold-out covariate for both the population and sample

    """
    fig, ax = plt.subplots()

    sns.histplot(
        data=holdout_distn_df,
        x="value",
        hue="variable",
        element="step",
        bins=20,
        stat="density",
        ax=ax,
    )
    ax.set_xlabel(holdout_covar)
    ax.set_ylabel("Density")
    ax.set_title("")

    return fig


# bento kernel api
def weigh_sample_data_from_external_source(
    sample_feature,
    sample_weights_path_namespace,
    sample_weights_path_tablename,
    primary_key,
    weight_col: Optional[str] = "weight",
):
    sample_with_weight = Dataset(
        namespace=sample_weights_path_namespace,
        tablename=sample_weights_path_tablename,
    )
    print(
        f"GTMF: It will re-normalize the weight to sum to 1. Renormalized weight column: {weight_col}_normalized"
    )
    return query(
        sql="""
        WITH raw_weight AS (
            SELECT
                sample_feature.*,
                COALESCE(sample_with_weight.{weight_col}, 0) AS {weight_col}
            FROM {sample_feature} sample_feature
            LEFT JOIN {sample_with_weight} sample_with_weight
                ON sample_feature.{primary_id_column}
                    = sample_with_weight.{primary_id_column}
        )
        SELECT
            *,
            {weight_col}/ SUM({weight_col}) OVER() AS {weight_col_n}
        FROM raw_weight
        """,
        refs=dict(
            sample_feature=sample_feature,
            sample_with_weight=sample_with_weight,
            primary_id_column=query.fragment(sql=primary_key),
            weight_col=query.fragment(sql=weight_col),
            weight_col_n=query.fragment(sql=weight_col + "_normalized"),
        ),
        custom_name="query_weigh_sample_data_from_external_source",
    )


# bento kernel api
def parameters(parameter_name: Optional[str] = None) -> Any:
    """
    Instruction for the parameters used in representativeness dimension

    Returns a dict with all parameter instruction or particular parameter instruction if the parameter name is given.

    Parameters
    ----------
    parameter: str, default is None
        parameter name, e.g. "discrete_covars"
    """
    parameters = {
        "population_data_path": "--- dict(namespace, tablename) Hive table of population data.",
        "sample_data_path": "--- dict(namespace, tablename) Hive table of sample data.",
        "continuous_covars_dataset": "dict name quantile_literal quantil_percentile null_value \n - continuous covariates which need to be discretized. \n - default quantile_percentage: [0, 0.2, 0.4, 0.6, 0.8] \n - null_value: null",
        "discrete_covars": "--- list(str) covar no need to quantile",
        "holdout_covar": "--- (Optional) holdout covar is used to generate \n - plot distribution of the hold-out covariate for both the population and sample \n - Hellinger Distance* currently we just support continuous holdout covariate, will support discrete in the future",
        "primary_key": "--- primary key for both population data table and sample data table. The primary key must be single column. If there are more than one in the raw data, user can construct composite key and dedup earlier",
        "hellinger_arguments": "--- dict(min_value, max_value, step) arguments are used to discretize holdout covar",
        "interaction_covar_paris": " --- (Optional) list[list] covar pairs used for weighting the sample data with default method ipx, list of covars pairs.",
        "balance_arguments": "--- (Optional) dict() For weight calculation, \n -adjusting sample to population by balance. sample.adjust. User can provide method and arguments used in this function.
        "non_covar_columns": "--- (Optional) list[str] The column name for ground truth labels or the non-covar column in sample data",
        "external_sample_weights_path": "--- (Optional) dict(namespace, tablename) external sample weight source.",
        "output_covariate_balance_df": "--- default is False, output covariate balance in dataframe format if it is set as True.",
        "output_holdout_distn_df": " ---  default is False, output holdout_distribution in dataframe format if is set as True.",
        "coverage_cutoff": "--- (Optional) used for population coverage culcation. If the weighted sample proportion / population proportion < cutoff, we will consider the covariate-value to be underrepresented in the sample.",
        "weight_col": "(Optional) column name of weight",
    }

    return parameters[parameter_name] if parameter_name else parameters
