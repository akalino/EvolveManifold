import argparse
import os

import numpy as np
import pandas as pd


def load_run_summary(path="run_level_summary.csv"):
    df = pd.read_csv(path)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def compute_lead_distributions(df):
    """
    Positive lead means PH detects earlier than spectral metric.

    lead = t_detect(spectral) - t_detect(PH)
    """
    spectral_cols = [
        "collapse_effective_rank_t_detect",
        "collapse_top_k_variance_fraction_t_detect",
    ]

    ph_cols = [
        "collapse_total_persistence_h1_t_detect",
        "collapse_betti_curve_area_h1_t_detect",
        "collapse_betti_curve_peak_h1_t_detect",
        "collapse_betti_curve_change_h1_t_detect",
    ]

    group_cols = [
        "ph_mode",
        "geometry",
        "mechanism",
        "schedule",
        "severity",
        "mover_frac",
        "noise",
        "n",
        "d",
    ]

    rows = []

    for spectral_col in spectral_cols:
        if spectral_col not in df.columns:
            continue

        for ph_col in ph_cols:
            if ph_col not in df.columns:
                continue

            tmp = df[group_cols + ["seed", spectral_col, ph_col]].copy()
            tmp = tmp.dropna(subset=[spectral_col, ph_col])

            if tmp.empty:
                continue

            tmp["spectral_metric"] = spectral_col.replace("collapse_", "").replace("_t_detect", "")
            tmp["ph_metric"] = ph_col.replace("collapse_", "").replace("_t_detect", "")
            tmp["lead"] = tmp[spectral_col] - tmp[ph_col]
            tmp["ph_leads"] = tmp["lead"] > 0
            tmp["tie"] = tmp["lead"] == 0
            tmp["spectral_leads"] = tmp["lead"] < 0

            rows.append(tmp)

    if not rows:
        raise ValueError("No valid detection-time columns found.")

    lead_long = pd.concat(rows, ignore_index=True)

    summary = (
        lead_long
        .groupby(group_cols + ["spectral_metric", "ph_metric"], dropna=False)
        .agg(
            n_runs=("lead", "count"),
            mean_lead=("lead", "mean"),
            median_lead=("lead", "median"),
            std_lead=("lead", "std"),
            min_lead=("lead", "min"),
            max_lead=("lead", "max"),
            frac_ph_leads=("ph_leads", "mean"),
            frac_ties=("tie", "mean"),
            frac_spectral_leads=("spectral_leads", "mean"),
        )
        .reset_index()
    )

    return lead_long, summary


def overall_lead_summary(lead_long):
    return (
        lead_long
        .groupby(["spectral_metric", "ph_metric"], dropna=False)
        .agg(
            n_runs=("lead", "count"),
            mean_lead=("lead", "mean"),
            median_lead=("lead", "median"),
            std_lead=("lead", "std"),
            min_lead=("lead", "min"),
            max_lead=("lead", "max"),
            frac_ph_leads=("ph_leads", "mean"),
            frac_ties=("tie", "mean"),
            frac_spectral_leads=("spectral_leads", "mean"),
        )
        .reset_index()
        .sort_values(["frac_ph_leads", "median_lead"], ascending=False)
    )


def mechanism_lead_summary(lead_long):
    return (
        lead_long
        .groupby(["mechanism", "spectral_metric", "ph_metric"], dropna=False)
        .agg(
            n_runs=("lead", "count"),
            mean_lead=("lead", "mean"),
            median_lead=("lead", "median"),
            frac_ph_leads=("ph_leads", "mean"),
            frac_ties=("tie", "mean"),
            frac_spectral_leads=("spectral_leads", "mean"),
        )
        .reset_index()
        .sort_values(["mechanism", "frac_ph_leads", "median_lead"], ascending=[True, False, False])
    )


def find_money_examples(lead_long, min_lead=10):
    """
    Find runs where PH detects much earlier than spectral metrics.
    """
    cols = [
        "ph_mode",
        "geometry",
        "mechanism",
        "schedule",
        "severity",
        "mover_frac",
        "noise",
        "n",
        "d",
        "seed",
        "spectral_metric",
        "ph_metric",
        "lead",
    ]

    return (
        lead_long.loc[lead_long["lead"] >= min_lead, cols]
        .sort_values("lead", ascending=False)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        default="metric_outputs",
        help="points to run_level_summary.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="metric_summaries",
    )
    args = parser.parse_args()
    df = load_run_summary(args.input_file)

    lead_long, lead_by_config = compute_lead_distributions(df)
    overall = overall_lead_summary(lead_long)
    by_mechanism = mechanism_lead_summary(lead_long)
    examples = find_money_examples(lead_long, min_lead=10)

    lead_long.to_csv(os.path.join(args.output_dir, "lead_distribution_long.csv"), index=False)
    lead_by_config.to_csv(os.path.join(args.output_dir,"lead_distribution_by_config.csv"), index=False)
    overall.to_csv(os.path.join(args.output_dir,"lead_distribution_overall.csv"), index=False)
    by_mechanism.to_csv(os.path.join(args.output_dir,"lead_distribution_by_mechanism.csv"), index=False)
    examples.to_csv(os.path.join(args.output_dir,"lead_distribution_money_examples.csv"), index=False)

    print("\nOverall PH lead summary:")
    print(overall)

    print("\nBest examples where PH leads by >= 10 epochs/checkpoints:")
    print(examples.head(20))