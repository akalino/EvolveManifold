import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_top_tables(run_df, out_dir):
    cols = [
        "mechanism",
        "geometry",
        "schedule",
        "severity",
        "mover_frac",
        "noise",
        "seed",
    ]

    ranking_metrics = [
        "collapse_effective_rank_end",
        "collapse_projection_residual_end",
        "collapse_total_persistence_h1_end",
        "collapse_effective_rank_auc",
        "collapse_projection_residual_auc",
        "collapse_total_persistence_h1_auc",
    ]

    for metric in ranking_metrics:
        if metric not in run_df.columns:
            continue

        top = (
            run_df[cols + [metric]]
            .sort_values(metric, ascending=False)
            .head(20)
        )
        top.to_csv(
            os.path.join(out_dir, f"top20_{metric}.csv"),
            index=False,
        )


def save_mover_frac_table(seed_agg_df, out_dir):
    keep_cols = [
        "mechanism",
        "geometry",
        "schedule",
        "severity",
        "noise",
        "mover_frac",
        "collapse_effective_rank_end__mean",
        "collapse_projection_residual_end__mean",
        "collapse_total_persistence_h1_end__mean",
        "collapse_effective_rank_auc__mean",
        "collapse_projection_residual_auc__mean",
    ]

    keep_cols = [c for c in keep_cols if c in seed_agg_df.columns]

    df = seed_agg_df[keep_cols].copy()
    df.to_csv(
        os.path.join(out_dir, "mover_frac_comparison.csv"),
        index=False,
    )


def save_mechanism_geometry_tables(seed_agg_df, out_dir):
    metric_map = {
        "collapse_effective_rank_end__mean": "endpoint_rank_collapse",
        "collapse_projection_residual_end__mean": "endpoint_projection_collapse",
        "collapse_total_persistence_h1_end__mean": "endpoint_h1_collapse",
        "collapse_effective_rank_auc__mean": "auc_rank_collapse",
        "collapse_projection_residual_auc__mean": "auc_projection_collapse",
    }

    for metric, stem in metric_map.items():
        if metric not in seed_agg_df.columns:
            continue

        pivot = seed_agg_df.pivot_table(
            index="mechanism",
            columns="geometry",
            values=metric,
            aggfunc="mean",
        )
        pivot.to_csv(os.path.join(out_dir, f"{stem}_table.csv"))


def plot_metric_trajectories(epoch_df, out_dir):
    plot_specs = [
        ("effective_rank", "effective_rank_over_epoch.png"),
        ("top_k_variance_fraction", "top_k_variance_fraction_over_epoch.png"),
        ("mean_pairwise_distance", "mean_pairwise_distance_over_epoch.png"),
        ("projection_residual", "projection_residual_over_epoch.png"),
        ("total_persistence_h1", "total_persistence_h1_over_epoch.png"),
    ]

    for metric, fname in plot_specs:
        if metric not in epoch_df.columns:
            continue

        plt.figure(figsize=(10, 6))
        for mech, df_mech in epoch_df.groupby("mechanism"):
            df_mech = df_mech.sort_values("epoch")
            y = df_mech.groupby("epoch")[metric].mean()
            plt.plot(y.index, y.values, label=mech)

        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.title(metric.replace("_", " "))
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=200)
        plt.close()


def plot_collapse_trajectories(epoch_df, out_dir):
    plot_specs = [
        ("collapse_effective_rank", "collapse_effective_rank_over_epoch.png"),
        ("collapse_top_k_variance_fraction",
         "collapse_top_k_variance_fraction_over_epoch.png"),
        ("collapse_mean_pairwise_distance",
         "collapse_mean_pairwise_distance_over_epoch.png"),
        ("collapse_projection_residual",
         "collapse_projection_residual_over_epoch.png"),
        ("collapse_total_persistence_h1",
         "collapse_total_persistence_h1_over_epoch.png"),
    ]

    for metric, fname in plot_specs:
        if metric not in epoch_df.columns:
            continue

        plt.figure(figsize=(10, 6))
        for mech, df_mech in epoch_df.groupby("mechanism"):
            df_mech = df_mech.sort_values("epoch")
            y = df_mech.groupby("epoch")[metric].mean()
            plt.plot(y.index, y.values, label=mech)

        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.title(metric.replace("_", " "))
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=200)
        plt.close()


def plot_mover_frac_effect(epoch_df, out_dir):
    key_metrics = [
        "collapse_effective_rank",
        "collapse_projection_residual",
        "collapse_total_persistence_h1",
    ]

    for (mechanism, geometry), df_sub in epoch_df.groupby(["mechanism", "geometry"]):
        for metric in key_metrics:
            if metric not in df_sub.columns:
                continue

            plt.figure(figsize=(9, 5))
            for mp, df_mp in df_sub.groupby("mover_frac"):
                df_mp = df_mp.sort_values("epoch")
                y = df_mp.groupby("epoch")[metric].mean()
                plt.plot(y.index, y.values, label=f"mp={mp}")

            plt.xlabel("Epoch")
            plt.ylabel(metric)
            plt.title(f"{mechanism} | {geometry} | {metric}")
            plt.legend()
            plt.tight_layout()

            safe_name = f"{mechanism}__{geometry}__{metric}.png".replace("/", "_")
            plt.savefig(
                os.path.join(out_dir, "mover_frac", safe_name),
                dpi=200,
            )
            plt.close()


def plot_heatmaps(seed_agg_df, out_dir):
    ensure_dir(os.path.join(out_dir, "heatmaps"))

    metric_map = {
        "collapse_effective_rank_end__mean": "heatmap_rank_endpoint.png",
        "collapse_projection_residual_end__mean": "heatmap_projection_endpoint.png",
        "collapse_total_persistence_h1_end__mean": "heatmap_h1_endpoint.png",
    }

    for metric, fname in metric_map.items():
        if metric not in seed_agg_df.columns:
            continue

        pivot = seed_agg_df.pivot_table(
            index="mechanism",
            columns="geometry",
            values=metric,
            aggfunc="mean",
        )

        plt.figure(figsize=(10, 5))
        plt.imshow(pivot.values, aspect="auto")
        plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.colorbar(label=metric)
        plt.title(metric.replace("_", " "))
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "heatmaps", fname), dpi=200)
        plt.close()

def plot_overlay_per_mechanism(epoch_df, out_dir):
    """
    Makes one overlay plot per mechanism using baseline-normalized collapse
    scores for:
      - effective_rank
      - projection_residual
      - total_persistence_h1
    """
    metric_map = {
        "effective_rank": "Effective rank",
        "projection_residual": "Projection residual",
        "total_persistence_h1": "Total persistence H1",
    }

    for mechanism, df_mech in epoch_df.groupby("mechanism"):
        df_mech = df_mech.sort_values("epoch").copy()

        avg_df = (
            df_mech.groupby("epoch")[
                ["effective_rank",
                 "projection_residual",
                 "total_persistence_h1"]
            ]
            .mean()
            .reset_index()
        )

        for col in metric_map:
            if col not in avg_df.columns:
                continue

            x0 = avg_df[col].iloc[0]
            if pd.notna(x0) and abs(x0) > 1e-12:
                avg_df[f"collapse_{col}"] = 1.0 - avg_df[col] / x0
            else:
                avg_df[f"collapse_{col}"] = np.nan

        plt.figure(figsize=(8, 5))

        for col, label in metric_map.items():
            ccol = f"collapse_{col}"
            if ccol not in avg_df.columns:
                continue
            plt.plot(avg_df["epoch"], avg_df[ccol], label=label)

        plt.xlabel("Epoch")
        plt.ylabel("Relative collapse score")
        plt.title(f"{mechanism}: normalized metric overlay")
        plt.legend()
        plt.tight_layout()

        safe_name = f"{mechanism}__metric_overlay.png".replace("/", "_")
        plt.savefig(os.path.join(out_dir, safe_name), dpi=200)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary_dir",
        type=str,
        default="metric_summaries",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="summary_assets",
    )
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    ensure_dir(os.path.join(args.out_dir, "tables"))
    ensure_dir(os.path.join(args.out_dir, "plots"))
    ensure_dir(os.path.join(args.out_dir, "plots", "mover_frac"))

    run_df = pd.read_csv(
        os.path.join(args.summary_dir, "run_level_summary.csv")
    )
    seed_agg_df = pd.read_csv(
        os.path.join(args.summary_dir, "summary_aggregated_over_seeds.csv")
    )
    epoch_df = pd.read_csv(
        os.path.join(args.summary_dir, "epochwise_mean_over_seeds.csv")
    )

    save_top_tables(run_df, os.path.join(args.out_dir, "tables"))
    save_mover_frac_table(seed_agg_df, os.path.join(args.out_dir, "tables"))
    save_mechanism_geometry_tables(seed_agg_df,
                                   os.path.join(args.out_dir, "tables"))

    plot_metric_trajectories(epoch_df, os.path.join(args.out_dir, "plots"))
    plot_collapse_trajectories(epoch_df, os.path.join(args.out_dir, "plots"))
    plot_mover_frac_effect(epoch_df, os.path.join(args.out_dir, "plots"))
    plot_heatmaps(seed_agg_df, os.path.join(args.out_dir, "plots"))
    plot_overlay_per_mechanism(
        epoch_df,
        os.path.join(args.out_dir, "plots"),
    )



if __name__ == "__main__":
    main()
