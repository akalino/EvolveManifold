import glob
import os

import numpy as np

import umap
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE


def load_run_checkpoints_and_labels(_run_stem, _checkpoint_root, _label_root):
    label_path = os.path.join(_label_root, f"{_run_stem}_labels.npy")
    labels = np.load(label_path)

    ckpt_pattern = os.path.join(_checkpoint_root, f"{_run_stem}__epoch*.npy")
    checkpoint_paths = sorted(glob.glob(ckpt_pattern))

    checkpoints = []
    for path in checkpoint_paths:
        x = np.load(path)
        checkpoints.append((path, x))

    return checkpoints, labels


def embed_tsne(_x, _seed=0):
    return TSNE(n_components=2, random_state=_seed).fit_transform(_x)


def embed_umap(_x, _seed=0):
    reducer = umap.UMAP(n_components=2, random_state=_seed)
    return reducer.fit_transform(_x)





def plot_embedding(z, labels, title):
    plt.figure(figsize=(6, 6))
    plt.scatter(z[:, 0], z[:, 1], c=labels, s=8)
    plt.title(title)
    plt.tight_layout()
    plt.show()
