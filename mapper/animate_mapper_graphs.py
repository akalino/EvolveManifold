"""
Animate saved giotto-tda Mapper graphs over checkpoint time.

Config-driven usage:
    python evaluation/mapper_analysis/animate_mapper_graphs.py \
      --config mapper_canonical.json

The script reads:
- paths.base_root_env
- paths.base_root_fallback_env
- paths.base_root_default
- paths.mapper_output_subdir
- visualization.*

Expected input layout:
    <resolved mapper_output_subdir>/
      mapper_summary.csv
      graphs/
        mapper_000.pkl
        mapper_001.pkl
        ...
"""

from pathlib import Path
import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def resolve_base_root(paths_cfg):
    """Resolve base root using env var, fallback env var, then default path."""
    env_name = paths_cfg.get("base_root_env")
    fallback_env_name = paths_cfg.get("base_root_fallback_env")
    default = paths_cfg.get("base_root_default", ".")

    if env_name and os.getenv(env_name):
        return Path(os.getenv(env_name)).expanduser().resolve()

    if fallback_env_name and os.getenv(fallback_env_name):
        return Path(os.getenv(fallback_env_name)).expanduser().resolve()

    return Path(default).expanduser().resolve()


def resolve_mapper_dir(cfg):
    """
    Resolve mapper output directory.

    Priority:
    1. visualization.mapper_dir, if provided
    2. paths.mapper_output_subdir resolved under base root
    """
    paths_cfg = cfg.get("paths", {})
    vis_cfg = cfg.get("visualization", {})

    explicit = vis_cfg.get("mapper_dir")
    if explicit:
        mapper_dir = Path(explicit).expanduser()
        if not mapper_dir.is_absolute():
            mapper_dir = resolve_base_root(paths_cfg) / mapper_dir
        return mapper_dir.resolve()

    base_root = resolve_base_root(paths_cfg)
    mapper_subdir = paths_cfg.get("mapper_output_subdir", "mapper_outputs")
    mapper_dir = Path(mapper_subdir).expanduser()

    if not mapper_dir.is_absolute():
        mapper_dir = base_root / mapper_dir

    return mapper_dir.resolve()


def resolve_out_path(cfg, mapper_dir):
    """
    Resolve output animation path.

    visualization.out may be:
    - relative filename, e.g. mapper_evolution.gif
    - relative path, e.g. animations/mapper_evolution.gif
    - absolute path
    """
    vis_cfg = cfg.get("visualization", {})
    out = vis_cfg.get("out", "mapper_evolution.gif")

    out_path = Path(out).expanduser()
    if not out_path.is_absolute():
        out_path = mapper_dir / out_path

    return out_path.resolve()


def load_graph(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def graph_to_edges(graph):
    """Return edge list from an igraph-style Mapper graph."""
    if not hasattr(graph, "es"):
        return []

    return [tuple(e.tuple) for e in graph.es]


def get_node_sizes(graph):
    """
    Best-effort extraction of Mapper node sizes.

    giotto-tda / igraph vertex attributes may vary by version.
    """
    sizes = []

    if not hasattr(graph, "vs"):
        return sizes

    for v in graph.vs:
        attrs = v.attributes()

        if "node_elements" in attrs:
            sizes.append(len(attrs["node_elements"]))

        elif "indices" in attrs:
            sizes.append(len(attrs["indices"]))

        else:
            sizes.append(1)

    return sizes


def circle_layout(n):
    """Fallback deterministic layout."""
    if n == 0:
        return np.empty((0, 2))

    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(theta), np.sin(theta)])


def stable_spring_layout(graph):
    """
    Try igraph's layout_fruchterman_reingold.

    This gives a clean per-graph layout, but node identities are not
    matched across time. For first visual inspection, that is okay.
    """
    n = graph.vcount() if hasattr(graph, "vcount") else 0

    if n == 0:
        return np.empty((0, 2))

    try:
        layout = graph.layout_fruchterman_reingold()
        xy = np.asarray(layout.coords, dtype=float)

    except Exception:
        xy = circle_layout(n)

    if xy.shape[0] != n:
        xy = circle_layout(n)

    xy = xy - xy.mean(axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(xy, axis=1))
    if scale > 0:
        xy = xy / scale

    return xy


def draw_graph(ax, graph, title="", node_size_scale=25.0):
    ax.clear()
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.axis("off")

    n = graph.vcount() if hasattr(graph, "vcount") else 0
    m = graph.ecount() if hasattr(graph, "ecount") else 0

    if n == 0:
        ax.text(0.5, 0.5, "empty graph", ha="center", va="center")
        return

    xy = stable_spring_layout(graph)
    edges = graph_to_edges(graph)
    node_sizes = get_node_sizes(graph)

    if node_sizes:
        ns = np.asarray(node_sizes, dtype=float)
        sizes = node_size_scale * np.sqrt(np.maximum(ns, 1.0))
    else:
        sizes = np.full(n, 80.0)

    for u, v in edges:
        if u < n and v < n:
            ax.plot(
                [xy[u, 0], xy[v, 0]],
                [xy[u, 1], xy[v, 1]],
                linewidth=0.8,
                alpha=0.45,
            )

    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=sizes,
        alpha=0.85,
        linewidths=0.5,
        edgecolors="black",
    )

    ax.text(
        0.02,
        0.02,
        f"nodes={n}  edges={m}",
        transform=ax.transAxes,
        fontsize=9,
        ha="left",
        va="bottom",
    )


def get_graph_paths(summary, graph_dir):
    if "graph_pickle" in summary.columns:
        paths = [Path(p) for p in summary["graph_pickle"].dropna()]
        if paths:
            return paths

    return sorted(graph_dir.glob("mapper_*.pkl"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_json(args.config)
    vis_cfg = cfg.get("visualization", {})

    mapper_dir = resolve_mapper_dir(cfg)
    summary_path = mapper_dir / "mapper_summary.csv"
    graph_dir = mapper_dir / "graphs"
    out_path = resolve_out_path(cfg, mapper_dir)

    fps = int(vis_cfg.get("fps", 2))
    dpi = int(vis_cfg.get("dpi", 140))
    figsize = tuple(vis_cfg.get("figsize", [6.0, 5.0]))
    node_size_scale = float(vis_cfg.get("node_size_scale", 25.0))

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary: {summary_path}")

    if not graph_dir.exists():
        raise FileNotFoundError(f"Missing graph dir: {graph_dir}")

    summary = pd.read_csv(summary_path)
    graph_paths = get_graph_paths(summary, graph_dir)

    if not graph_paths:
        raise FileNotFoundError(f"No graph pickle files found in {graph_dir}")

    graphs = [load_graph(p) for p in graph_paths]

    # Keep summary aligned to loaded graph count.
    summary = summary.iloc[: len(graphs)].reset_index(drop=True)

    print(f"Mapper directory: {mapper_dir}")
    print(f"Loaded graphs: {len(graphs)}")
    print(f"Output: {out_path}")

    fig, ax = plt.subplots(figsize=figsize)

    def update(i):
        row = summary.iloc[i] if i < len(summary) else {}
        step = row.get("checkpoint_step_guess", i)
        path = row.get("checkpoint_path", "")
        short_path = Path(path).name if path else f"frame {i}"

        title = f"Mapper graph evolution | frame {i} | step {step}\n{short_path}"

        draw_graph(
            ax,
            graphs[i],
            title=title,
            node_size_scale=node_size_scale,
        )

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(graphs),
        interval=1000 / max(fps, 1),
        repeat=True,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = out_path.suffix.lower()
    if suffix == ".gif":
        ani.save(out_path, writer="pillow", fps=fps, dpi=dpi)
    elif suffix == ".mp4":
        ani.save(out_path, writer="ffmpeg", fps=fps, dpi=dpi)
    else:
        raise ValueError("visualization.out must end in .gif or .mp4")

    plt.close(fig)
    print(f"Wrote animation: {out_path}")


if __name__ == "__main__":
    main()
