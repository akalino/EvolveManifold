"""
Experiments geared toward not having to compute full VR or DTM at each checkpoint.
"""
import numpy as np
import gudhi as gd

from scipy.spatial.distance import pdist, squareform

from complex_persistence import compute_vr_diagrams



def _empty_diagram():
    """
    Return an empty persistence diagram with the standard ``(0, 2)`` shape.
    """
    return np.empty((0, 2), dtype=float)


def _as_diagram_array(_dgm, _dim):
    """
    Convert one diagram-like object into a ``(n_bars, 2)`` NumPy array.

    :param _dgm: Diagram-like object.
    :param _dim: Homology dimension, used only for error messages.
    :return: Diagram array with two columns.
    """
    if _dgm is None:
        return _empty_diagram()

    arr = np.asarray(_dgm, dtype=float)

    if arr.size == 0:
        return _empty_diagram()

    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            "Persistence diagram for dimension {} must have shape (n, 2); got {}".format(
                _dim,
                arr.shape,
            )
        )

    return arr


def _normalize_diagrams(_raw_dgms, _max_dim):
    """
    Normalize PH backend outputs to the workflow contract.

    The public contract for this workflow is always:

    ``[H0_array, H1_array, ..., Hmax_array]``

    where each array has shape ``(n_bars, 2)``.  Some backends in this repo return
    a ``dict[int, np.ndarray]`` instead, so this helper converts dictionaries and
    validates list/tuple outputs.  It intentionally raises on ambiguous outputs
    rather than hiding a backend contract mismatch.

    :param _raw_dgms: Raw backend diagram output.
    :param _max_dim: Maximum homology dimension.
    :return: List of diagram arrays.
    """
    max_dim = int(_max_dim)

    if isinstance(_raw_dgms, dict):
        return [
            _as_diagram_array(_raw_dgms.get(dim, _empty_diagram()), dim)
            for dim in range(max_dim + 1)
        ]

    # Be permissive for a common wrapper pattern like ``(dgms, metadata)`` or
    # ``(metadata, dgms)``, but only when one item clearly looks like diagrams.
    if isinstance(_raw_dgms, tuple) and len(_raw_dgms) == 2:
        first, second = _raw_dgms
        if isinstance(first, (dict, list)):
            return _normalize_diagrams(first, max_dim)
        if isinstance(second, (dict, list)):
            return _normalize_diagrams(second, max_dim)

    if isinstance(_raw_dgms, (list, tuple)):
        if len(_raw_dgms) < max_dim + 1:
            raise ValueError(
                "Expected at least {} diagram arrays, got {}".format(
                    max_dim + 1,
                    len(_raw_dgms),
                )
            )
        return [
            _as_diagram_array(_raw_dgms[dim], dim)
            for dim in range(max_dim + 1)
        ]

    raise TypeError(
        "Unsupported persistence-diagram return type: {}".format(type(_raw_dgms).__name__)
    )


def compute_max_edge(_x, _sz_cut):
    """
    _x: Point cloud.
    _sz_cut: Cut max_edge_len to reduce memory usage.
    """
    d = pdist(_x)
    qs = np.quantile(d, [0.01, 0.05, 0.1, 0.5, 0.75, 0.95, 0.99])
    if _sz_cut:
        max_edge_len = qs[0]
    else:
        max_edge_len = qs[1]
    print(f"[VR MAX EDGE LENGTH] {max_edge_len}")
    return max_edge_len


def furthest_point_subsample(_x, _n_landmarks, _seed=17):
    """
    Greedy sampling of furthest points on rows of _x.

    _x: Point cloud.
    _n_landmarks: Number of landmarks.
    _seed: Seed.

    return: landmark indices.
    """
    n = _x.shape[0]
    if _n_landmarks >= n:
        return np.arange(n)

    r_num = np.random.default_rng(_seed)
    idx = np.empty(_n_landmarks, dtype=int)
    idx[0] = r_num.integers(0, n)

    d2 = np.sum((_x - _x[idx[0]]) ** 2, axis=1)
    for i in range (1, _n_landmarks):
        idx[i] = np.argmax(d2)
        new_d2 = np.sum((_x - _x[idx[i]]) ** 2, axis=1)
        d2 = np.minimum(d2, new_d2)
    return idx


def _pairwise_dist(_x):
    return squareform(pdist(_x))


def _radius_edges(_x, _max_edge_length):
    d_mat = _pairwise_dist(_x)
    ii, jj = np.where(np.triu(d_mat <= _max_edge_length, k=1))
    edges = [(int(i), int(j)) for i,j in zip(ii, jj)]
    return edges, d_mat


def _knn_edges(_x, _k):
    d_mat = _pairwise_dist(_x)
    n = d_mat.shape[0]
    nbrs = np.argsort(d_mat, axis=1)[:, 1:_k + 1]

    edge_set = set()
    for i in range(n):
        for j in nbrs[i]:
            a = min(i, int(j))
            b = max(i, int(j))
            edge_set.add((a, b))

    edges = sorted(edge_set)
    return edges, d_mat


def _build_clique_tree(_n, _edges, _edge_filtration, _max_dim):
    st = gd.SimplexTree() # pylint: disable=no-member
    for i in range(_n):
        st.insert([i], filtration=0.0)
    for i, j in _edges:
        st.insert([i, j], filtration=float(_edge_filtration[(i,j)]))
    st.expansion(_max_dim +1)
    st.make_filtration_non_decreasing()
    return st


def _update_tree_filtration(_st, _n, _edge_filtration):
    for i in range(_n):
        _st.assign_filtration([i], 0.0)
    for (i, j), val in _edge_filtration.items():
        _st.assign_filtration([i, j], float(val))
    _st.make_filtration_non_decreasing()


def _edge_filtration_from_d_mat(_edges, _d_mat, _max_edge_len=None):
    filt = {}
    for i, j in _edges:
        val = float(_d_mat[i, j])
        if _max_edge_len is not None:
            val = min(val, _max_edge_len)
        filt[(i, j)] = val
    return filt


def _simplex_change_score(_x, _x_prev):
    dx = np.linalg.norm(_x - _x_prev, axis=1).mean()
    scale = np.linalg.norm(_x_prev, axis=1).mean() + 1e-12
    return float(dx / scale)


def _mean_relative_edge_change(_edge_filt_new, _edge_filt_old):
    vals = []
    for key, new_val in _edge_filt_new.items():
        old_val = _edge_filt_old[key]
        denom = abs(old_val) + 1e-12
        vals.append(abs(new_val - old_val) / denom)
    return float(np.mean(vals)) if vals else 0.0


def _compute_knn_indices(_x, _k):
    """
    Returns kNN neighbor indices from the point cloud.

    :param _x: Input point cloud.
    :param _k: Number of nearest neighbors.
    :return: np array with indices of k nearest neighbors.
    """
    d_mat = _pairwise_dist(_x)
    n = d_mat.shape[0]

    if _k <= 0:
        raise ValueError("k cannot be negative or 0")
    if _k >= n:
        raise ValueError("k must be smaller than number of points n")
    d_mat = d_mat.copy()
    np.fill_diagonal(d_mat, np.inf)

    nn_idx = np.argsort(d_mat, axis=1)[:, :_k]
    return nn_idx


def _knn_identity_drift(_knn_old, _knn_new):
    """
    Computes the fraction of changed neighbor indices.

    :param _knn_old: Old knn indices.
    :param _knn_new: New knn indices.
    :return: Fraction of changed neighbor indices.
    """
    _, k = _knn_old.shape
    changed_frac = []
    for old_row, new_row in zip(_knn_old, _knn_new):
        old_set = set(old_row.tolist())
        new_set = set(new_row.tolist())
        overlap = len(old_set & new_set)
        frac_changed = 1.0 - (overlap / k)
        changed_frac.append(frac_changed)
    return float(np.mean(changed_frac))


def _knn_rank_drift(_knn_old, _knn_new):
    """
    Among neighbors that are still present, how much did their local ordering change?
    Mean normalized rank displacement for shared kNN neighbors.

    :param _knn_old:
    :param _knn_new:
    :return: Returns a value in [0, 1] approximately:
      0 means shared neighbors keep the same ranks;
      1 means either no neighbors are shared or shared neighbors move maximally.
    """
    _knn_old = np.asarray(_knn_old)
    _knn_new = np.asarray(_knn_new)

    if _knn_old.shape != _knn_new.shape:
        raise ValueError("kNN arrays must have the same shape")

    _, k = _knn_old.shape
    if k <= 1:
        return 0.0

    drifts = []

    for old_row, new_row in zip(_knn_old, _knn_new):
        old_rank = {int(idx): r for r, idx in enumerate(old_row)}
        new_rank = {int(idx): r for r, idx in enumerate(new_row)}

        shared = set(old_rank) & set(new_rank)

        if not shared:
            drifts.append(1.0)
            continue

        # Normalize by maximum possible rank displacement k - 1
        rank_moves = [
            abs(old_rank[j] - new_rank[j]) / float(k - 1)
            for j in shared
        ]

        # Penalize missing neighbors too, so this complements identity drift
        missing_frac = 1.0 - (len(shared) / float(k))
        shared_rank_drift = float(np.mean(rank_moves)) if rank_moves else 0.0

        # Conservative combination.
        drifts.append(max(shared_rank_drift, missing_frac))

    return float(np.mean(drifts))


def _nearest_landmark_assignment(_x, _landmarks, _return_dist=False):
    """
    Assigns each point to nearest landmark.

    :param _x: Input point cloud.
    :param _landmarks: Landmark points.
    :param _return_dist: Bool to return distances.
    :return:
    """
    diff = _x[:, None, :] - _landmarks[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    assign = np.argmin(d2, axis=1)
    if not _return_dist:
        return assign.astype(int)
    min_dist = np.sqrt(np.min(d2, axis=1))
    return assign.astype(int), min_dist


def _assignment_drift(_assign_prev, _assign_new):
    """
    Fraction of points with changed nearest-landmark assignment.

    :param _assign_prev: Prior point to landmark assignment.
    :param _assign_new: New point to landmark assignment.
    :return: Fraction of points whose nearest landmark changed.
    """
    _assign_prev = np.asarray(_assign_prev)
    _assign_new = np.asarray(_assign_new)
    changed = _assign_prev != _assign_new
    return float(np.mean(changed))


def _landmark_coverage_stats(_x, _landmarks):
    """
    Mean, max, quantiles of distance to nearest landmarks.

    :param _x: Point cloud.
    :param _landmarks: Landmark points.
    :return: Dict of landmark change stats.
    """
    diff = _x[:, None, :] - _landmarks[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    min_dist = np.sqrt(np.min(d2, axis=1))
    q = np.quantile(min_dist, [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
    return {
        "mean": float(np.mean(min_dist)),
        "std": float(np.std(min_dist)),
        "min": float(q[0]),
        "q25": float(q[1]),
        "median": float(q[2]),
        "q75": float(q[3]),
        "q90": float(q[4]),
        "q95": float(q[5]),
        "q99": float(q[6]),
        "max": float(q[7]),
    }


def _support_edge_recall(_support_edges, _x, _k):
    """
    Compare cached support edges to current kNN candidate edges.
    Recall is computed as |support intersect current_knn| / |current_knn|.

    :param _support_edges: Cached edges.
    :param _x: Current point cloud.
    :param _k: Number of nearest neighbors.
    :return: Support edge recall in [0,1].
    """
    n = _x.shape[0]

    support_edges = set()
    for i, j in _support_edges:
        a = min(int(i), int(j))
        b = max(int(i), int(j))
        if a != b:
            support_edges.add((a, b))

    knn_idx = _compute_knn_indices(_x, _k)
    curr_set = set()
    for i in range(n):
        for j in knn_idx[i]:
            a = min(i, int(j))
            b = max(i, int(j))
            if a != b:
                curr_set.add((a, b))
    if len(curr_set) == 0:
        return 1.0
    overlap = len(curr_set & support_edges)
    return float(overlap / len(curr_set))


def _support_edge_precision(_support_edges, _x, _k):
    """
    Compare cached support edges to current kNN candidate edges.

    Precision is |support intersect current_knn| / |support|.
    High precision means most cached support edges are still locally relevant.
    :param _support_edges:
    :param _x:
    :param _k:
    :return:
    """
    support_edges = set()
    for i, j in _support_edges:
        a = min(int(i), int(j))
        b = max(int(i), int(j))
        if a != b:
            support_edges.add((a, b))

    if len(support_edges) == 0:
        return 1.0

    knn_idx = _compute_knn_indices(_x, _k)
    curr_set = set()

    for i in range(_x.shape[0]):
        for j in knn_idx[i]:
            a = min(i, int(j))
            b = max(i, int(j))
            if a != b:
                curr_set.add((a, b))

    overlap = len(support_edges & curr_set)
    return float(overlap / len(support_edges))


def _compute_event_diagnostics(_x_prev, _x_new, _k,
                               _landmarks_prev, _landmarks_new,
                               _support_edges):
    """
    Returns a dictionary of event diagnostics.

    :param _x_prev: Prior point cloud.
    :param _x_new: Next step point cloud.
    :param _k: Nearest neighbors.
    :param _landmarks_prev: Prior landmark points.
    :param _landmarks_new: Current landmark points.
    :param _support_edges: Cached edges.
    :return:
    """
    out = {}

    cloud_diff = np.linalg.norm(_x_new - _x_prev, axis=1)
    base = np.linalg.norm(_x_prev, axis=1)

    out["mean_point_displacement"] = float(np.mean(cloud_diff))
    out["max_point_displacement"] = float(np.max(cloud_diff))
    out["relative_cloud_change"] = float(
        np.mean(cloud_diff) / (np.mean(base) + 1e-12)
    )

    if _k is not None:
        knn_prev = _compute_knn_indices(_x_prev, _k)
        knn_new = _compute_knn_indices(_x_new, _k)
        out['knn_identity_drift'] = _knn_identity_drift(knn_prev, knn_new)
        out['knn_rank_drift'] = _knn_rank_drift(knn_prev, knn_new)

        if _support_edges is not None:
            out['support_edge_recall'] = _support_edge_recall(_support_edges,
                                                              _x_new, _k)
            out["support_edge_precision"] = _support_edge_precision(_support_edges,
                                                                    _x_new, _k)

    if _landmarks_prev is not None and _landmarks_new is not None:
        assign_prev = _nearest_landmark_assignment(_x_prev, _landmarks_prev)
        assign_new = _nearest_landmark_assignment(_x_new, _landmarks_new)
        out["assignment_drift"] = _assignment_drift(assign_prev, assign_new)

        cov_prev = _landmark_coverage_stats(_x_prev, _landmarks_prev)
        cov_new = _landmark_coverage_stats(_x_new, _landmarks_new)

        out["coverage_mean_prev"] = cov_prev["mean"]
        out["coverage_mean_new"] = cov_new["mean"]
        out["coverage_mean_drift"] = float(cov_new["mean"] - cov_prev["mean"])

        out["coverage_q95_prev"] = cov_prev["q95"]
        out["coverage_q95_new"] = cov_new["q95"]
        out["coverage_q95_drift"] = float(cov_new["q95"] - cov_prev["q95"])

        out["coverage_max_prev"] = cov_prev["max"]
        out["coverage_max_new"] = cov_new["max"]
        out["coverage_max_drift"] = float(cov_new["max"] - cov_prev["max"])

    return out


def _compute_composite_event_score(_rx_dict,
                                   _weights=None,
                                   _use_abs_coverage_drift=True,
                                   _clip_nonnegative=True):
    """
    Compute composite event score.

    :param _rx_dict: Diagnostic metric dictionary.
    :param _weights: Dictionary of alpha, beta, gamma.
    :param _use_abs_coverage_drift: Absolute value of coverage drifts.
    :param _clip_nonnegative: Clip each term in composite score.
    :return: Composite event score.
    """
    default_weights = {
        "edge_drift": 1.0,
        "knn_identity_drift": 1.0,
        "coverage_drift": 1.0
    }
    if _weights is None:
        weights = default_weights
    else:
        weights = default_weights.copy()
        weights.update(_weights)

    edge_drift = float(_rx_dict["edge_drift"])
    knn_drift = float(_rx_dict["knn_identity_drift"])
    landmark_drift = float(_rx_dict["coverage_drift"])

    composite = (weights["edge_drift"] * edge_drift +
                 weights["knn_identity_drift"] * knn_drift +
                 weights["coverage_drift"] * landmark_drift)
    return float(composite)


def _should_refresh_support(_rx_dict, _score_thresh,
                            _weights=None, _min_support_recall=None):
    """
    Decide to refresh the support graph.

    :param _rx_dict: Diagnostic metric dictionary.
    :param _score_thresh: Threshold of composite score to trigger refresh event.
    :param _weights: Dictionary of alpha, beta, gamma.
    :param _min_support_recall: Optional floor on recall.
    :return: If true, perform a refresh.
    """
    score = _compute_composite_event_score(_rx_dict, _weights)

    if score >= _score_thresh:
        return True

    if _min_support_recall is not None:
        recall = _rx_dict["support_edge_recall"]
        if recall is not None and float(recall) < float(_min_support_recall):
            return True

    return False


def _refresh_landmark_support(_x_landmark, _k, _max_edge_length=None):
    """
    Updates the landmark support points.

    :param _x_landmark: Landmark points.
    :param _k: Neighborhood size.
    :param _max_edge_length: Maximum edge length.
    :return: Edges, distance matrix, edge filtration.
    """
    n = _x_landmark.shape[0]
    d_mat = _pairwise_dist(_x_landmark)

    d_work = d_mat.copy()
    np.fill_diagonal(d_work, np.inf)

    n_brs = np.argsort(d_work, axis=1)[:, :_k]

    edge_set = set()
    for i in range(n):
        for j in n_brs[i]:
            a = min(i, int(j))
            b = max(i, int(j))
            if a != b:
                edge_set.add((a, b))

    edges = sorted(edge_set)

    edge_filt = {}
    for i, j in edges:
        val = float(d_mat[i, j])
        if _max_edge_length is not None:
            val = min(val, float(_max_edge_length))
        edge_filt[(i, j)] = val

    return edges, d_mat, edge_filt


class _VRPHWorkflow:
    """
    Handles the complex / PH logic separately from metric computation.

    Modes
    -----
    full_vr:
        Recompute VR on the full checkpoint cloud each epoch.

    landmark_vr:
        Choose a fixed landmark subset at epoch 1 and compute VR only on
        those landmarks at each later epoch.

    skip_vr:
        Recompute VR only every _skip_every epochs. Between recomputations,
        reuse the previous diagrams.

    fixed_support_vr:
        Build epoch 1 radius graph support once, expand the clique complex once,
        and later epochs only update edge filtrations on the fixed support.

    fixed_knn_vr:
        Build epoch 1 symmetric knn graph once, expand clique complex once,
        and later epochs only update edge filtrations on the fixed support.

    event_driven:
        Recompute VR only when the cloud changes are meaningful, otherwise reuse
        prior diagrams.

    online_landmark_event:
        Choose fixed landmarks and a fixed knn support at epoch 1.
        Update edge weights every epoch, only recompute PH when the
        landmark support geometry has changed enough.

    online_landmark_dynamic_support:
        Choose fixed landmarks at epoch 1, but don't freeze the kNN support.
        Update edge weights at every epoch and refresh the landmark support
        graph when the drift diagnostics exceed the composite score threshold.
    """

    def __init__(self,
                 _mode="full_vr",
                 _max_dim=1,
                 _sparse=0.2,
                 _too_big=False,
                 _n_landmarks=250,
                 _seed=17,
                 _skip_every=2,
                 _knn_k=12,
                 _event_thresh=0.02,
                 _event_max_skip=5,
                 _force_every=10):
        self.mode = _mode
        self.max_dim = _max_dim
        self.sparse = _sparse
        self.too_big = _too_big
        self.n_landmarks = _n_landmarks
        self.seed = _seed
        self.skip_every = _skip_every
        self.knn_k = _knn_k
        self.event_thresh = _event_thresh
        self.event_max_skip = _event_max_skip

        self.force_every = _force_every
        self.last_force = None

        self.max_edge_len = None
        self.landmark_idx = None

        self.prev_dgms = None
        self.prev_epoch = None
        self.prev_x_use = None
        self.prev_x_full = None


        self.support_edges = None
        self.simplex_tree = None
        self.support_n = None

        self.last_recomputed = False

        self.edge_filt_prev = None
        self.last_event_score = None
        self.min_support_recall = 0.8
        self.support_edge_age = {}
        self.support_max_age = 3

    def _fit_epoch1(self, _x):
        self.max_edge_len = compute_max_edge(_x, self.too_big)

        if self.mode == "landmark_vr":
            self.landmark_idx = furthest_point_subsample(
                _x,
                self.n_landmarks,
                self.seed,
            )
            # lm_l = len(self.landmark_idx)
            # print(f"[PH LANDMARKS] {lm_l}")

        x_use = self._cloud_for_epoch(_x)

        if self.mode == "fixed_support_vr":
            edges, d_mat = _radius_edges(x_use, self.max_edge_len)
            edge_filt = _edge_filtration_from_d_mat(edges, d_mat, self.max_edge_len)
            self.support_edges = edges
            self.support_n = x_use.shape[0]
            self.simplex_tree = _build_clique_tree(self.support_n,
                                                   self.support_edges,
                                                   edge_filt,
                                                   self.max_dim)
            # el = len(edges)
            # print(f"[PH FIXED SUPPORT EDGES] {el}")

        if self.mode == "fixed_knn_vr":
            edges, d_mat = _knn_edges(x_use, self.knn_k)
            edge_filt = _edge_filtration_from_d_mat(edges, d_mat, self.max_edge_len)
            self.support_edges = edges
            self.support_n = x_use.shape[0]
            self.simplex_tree = _build_clique_tree(self.support_n,
                                                   self.support_edges,
                                                   edge_filt,
                                                   self.max_dim)
            # el = len(edges)
            # print(f"[PH FIXED KNN EDGES] {el}")

        if self.mode == "online_landmark_event":
            self.landmark_idx = furthest_point_subsample(_x,
                                                         self.n_landmarks,
                                                         self.seed)
            x_use = self._cloud_for_epoch(_x)
            edges, d_mat = _knn_edges(x_use, self.knn_k)
            edge_filt = _edge_filtration_from_d_mat(edges, d_mat, self.max_edge_len)
            self.support_edges = edges
            self.support_n = x_use.shape[0]
            self.simplex_tree = _build_clique_tree(self.support_n,
                                                   self.support_edges,
                                                   edge_filt,
                                                   self.max_dim)
            self.edge_filt_prev = edge_filt
            # print("[PH ONLINE LANDMARK EVENT] {} edges".format(len(edges)))

        if self.mode == "online_landmark_dynamic_support":
            self.landmark_idx = furthest_point_subsample(
                _x,
                self.n_landmarks,
                self.seed,
            )
            x_use = self._cloud_for_epoch(_x)
            edges, d_mat, edge_filt = _refresh_landmark_support(
                x_use,
                self.knn_k,
                self.max_edge_len,
            )
            self.support_edges = edges
            self.support_edge_age = {e: 0 for e in edges}
            self.support_n = x_use.shape[0]
            self.simplex_tree = _build_clique_tree(
                self.support_n,
                self.support_edges,
                edge_filt,
                self.max_dim,
            )
            self.edge_filt_prev = edge_filt
            # print("[PH ONLINE LANDMARK DYNAMIC SUPPORT] {} edges".format(len(edges)))

    def _cloud_for_epoch(self, _x):
        if self.landmark_idx is not None:
            return _x[self.landmark_idx]
        return _x

    def _compute_full_vr(self, _x_use):
        raw_dgms = compute_vr_diagrams(_x_use,
                                       _max_dim=self.max_dim,
                                       _max_edge_length=self.max_edge_len,
                                       _sparse=self.sparse)
        return _normalize_diagrams(raw_dgms, self.max_dim)

    def _compute_fixed_support(self, _x_use):
        if self.support_edges is None:
            raise ValueError("Support edges is None")
        d_mat = _pairwise_dist(_x_use)
        edge_filt = _edge_filtration_from_d_mat(self.support_edges,
                                                d_mat,
                                                _max_edge_len=self.max_edge_len)
        _update_tree_filtration(self.simplex_tree,
                                self.support_n,
                                edge_filt)
        self.simplex_tree.persistence()
        return [np.array(self.simplex_tree.persistence_intervals_in_dimension(d))
                for d in range(self.max_dim +1)]

    def _should_recompute_event(self, _x_use, _epoch):
        if self.prev_dgms is None:
            return True
        if self.prev_x_use is None:
            return True
        if self.prev_epoch is None:
            return True

        if (_epoch - self.prev_epoch) >= self.event_max_skip:
            return True

        score = _simplex_change_score(_x_use, self.prev_x_use)
        check = score >= self.event_thresh
        if check:
            print(f"[PH EVENT DRIVEN] change event occurred at epoch {_epoch}")
        return check

    def _event_reason(self, _rx_dict, _epoch):
        reasons = []
        score = _compute_composite_event_score(_rx_dict)
        if score >= self.event_thresh:
            reasons.append("score_thresh")

        recall = _rx_dict["support_edge_recall"]
        if (self.min_support_recall is not None and
                recall is not None and
                float(recall) < float(self.min_support_recall)):
            reasons.append("low_support_recall")
        if self.prev_epoch is not None:
            if _epoch - self.prev_epoch >= self.event_max_skip:
                reasons.append("max_skip")
        if self.last_force is not None:
            if _epoch - self.last_force >= self.force_every:
                reasons.append("force_every")
        return reasons

    def _should_recompute_ph(self, _rx_dict, _epoch):
        refresh_support = _should_refresh_support(_rx_dict,
                                                  _score_thresh=self.event_thresh,
                                                  _min_support_recall=self.min_support_recall)
        if refresh_support:
            return True
        if self.prev_epoch is not None:
            if _epoch - self.prev_epoch >= self.event_max_skip:
                return True
        if self.last_force is not None:
            if _epoch - self.last_force >= self.force_every:
                return True
        return False

    def _merge_support_edges(self, _old_edges, _new_edges):
        old_set = set(_old_edges)
        new_set = set(_new_edges)
        merged = sorted(old_set | new_set)
        for e in new_set:
            self.support_edge_age[e] = 0

        for e in old_set - new_set:
            if e not in self.support_edge_age:
                self.support_edge_age[e] = 0
        return merged

    def _prune_support_edges(self, _edges, _new_edges, _max_age):
        new_set = set(_new_edges)
        kept = []

        for e in _edges:
            if e in new_set:
                self.support_edge_age[e] = 0
            else:
                self.support_edge_age[e] += 1

            if self.support_edge_age[e] <= _max_age:
                kept.append(e)
        kept = sorted(kept)
        keep_set = set(kept)
        age_keys = list(self.support_edge_age.keys())
        for e in age_keys:
            if e not in keep_set:
                del self.support_edge_age[e]
        return kept

    def _build_simplex_tree_from_support(self, _x_landmarks, _edges):
        d_mat = _pairwise_dist(_x_landmarks)
        edge_filt = _edge_filtration_from_d_mat(_edges,
                                                d_mat,
                                                self.max_edge_len)
        st = _build_clique_tree(_x_landmarks.shape[0],
                                _edges,
                                edge_filt,
                                self.max_dim)
        return st, d_mat, edge_filt

    def _compute_online_landmark_event(self, _x_use, _epoch):
        d_mat = _pairwise_dist(_x_use)
        edge_filt_new = _edge_filtration_from_d_mat(self.support_edges,
                                                    d_mat,
                                                    self.max_edge_len)
        if self.prev_dgms is None:
            recompute = True
            event_score = np.inf
        elif self.prev_epoch is None:
            recompute = True
            event_score = np.inf
        else:
            event_score = _mean_relative_edge_change(edge_filt_new,
                                                     self.edge_filt_prev)
            recompute = (event_score >= self.event_thresh
                         or (_epoch - self.prev_epoch) >= self.event_max_skip
                         or (_epoch - self.last_force) >= self.force_every)
        self.last_event_score = event_score

        if not recompute:
            self.last_recomputed = False
            self.edge_filt_prev = edge_filt_new
            return self.prev_dgms

        _update_tree_filtration(self.simplex_tree,
                                self.support_n,
                                edge_filt_new)
        self.simplex_tree.persistence()
        dgms = [np.array(self.simplex_tree.persistence_intervals_in_dimension(d))
                for d in range(self.max_dim +1)]
        self.last_recomputed = True
        self.last_force = _epoch
        self.edge_filt_prev = edge_filt_new
        return dgms

    def _compute_online_landmark_dynamic_support(self, _x_full, _x_use, _epoch):
        """
        Event-driven PH on a landmark cloud with dynamic support refresh.
        """
        if self.support_edges is None or self.simplex_tree is None:
            edges, d_mat, edge_filt = _refresh_landmark_support(
                _x_use,
                _k=self.knn_k,
                _max_edge_length=self.max_edge_len,
            )
            self.support_edges = edges
            self.support_edge_age = {e: 0 for e in edges}
            self.support_n = _x_use.shape[0]
            self.simplex_tree = _build_clique_tree(
                self.support_n,
                self.support_edges,
                edge_filt,
                self.max_dim,
            )
            self.edge_filt_prev = edge_filt

        d_mat_curr = _pairwise_dist(_x_use)
        edge_filt_new = _edge_filtration_from_d_mat(
            self.support_edges,
            d_mat_curr,
            self.max_edge_len,
        )

        if self.prev_dgms is None or self.prev_x_use is None or self.prev_epoch is None:
            _update_tree_filtration(
                self.simplex_tree,
                self.support_n,
                edge_filt_new,
            )
            self.simplex_tree.persistence()
            dgms = [
                np.array(self.simplex_tree.persistence_intervals_in_dimension(d))
                for d in range(self.max_dim + 1)
            ]

            self.last_recomputed = True
            self.last_force = _epoch
            self.edge_filt_prev = edge_filt_new
            self.last_event_score = np.inf
            self.prev_x_full = _x_full.copy()
            return dgms

        edge_drift = _mean_relative_edge_change(edge_filt_new, self.edge_filt_prev)

        knn_prev = _compute_knn_indices(self.prev_x_use, self.knn_k)
        knn_new = _compute_knn_indices(_x_use, self.knn_k)

        cov_prev = _landmark_coverage_stats(self.prev_x_full, self.prev_x_use)
        cov_new = _landmark_coverage_stats(_x_full, _x_use)

        support_precision = _support_edge_precision(self.support_edges, _x_use, self.knn_k)

        diag = {
            "edge_drift": float(edge_drift),
            "knn_identity_drift": float(_knn_identity_drift(knn_prev, knn_new)),
            "knn_rank_drift": float(_knn_rank_drift(knn_prev, knn_new)),
            "coverage_drift": float(abs(cov_new["q95"] - cov_prev["q95"])),
            "support_edge_recall": float(
                _support_edge_recall(self.support_edges, _x_use, self.knn_k)
            ),
            "support_edge_precision": float(support_precision)
        }

        event_score = _compute_composite_event_score(diag)
        self.last_event_score = event_score

        reason_code = self._event_reason(diag, _epoch)
        refresh_support = (
            "score_thresh" in reason_code or
            "low_support_recall" in reason_code
        )
        recompute = self._should_recompute_ph(diag, _epoch)

        if refresh_support:
            new_edges, d_mat_curr, _ = _refresh_landmark_support(
                _x_use,
                _k=self.knn_k,
                _max_edge_length=self.max_edge_len
            )
            merged_edges = self._merge_support_edges(self.support_edges, new_edges)
            pruned_edges = self._prune_support_edges(
                merged_edges,
                new_edges,
                self.support_max_age
            )
            self.support_edges = pruned_edges
            self.support_n = _x_use.shape[0]
            self.simplex_tree, d_mat_curr, edge_filt_new = (
                self._build_simplex_tree_from_support(_x_use, self.support_edges))
            # print("[PH DYNAMIC SUPPORT] old {}, new {}, merged {}, kept {}".format(
            #     len(self.support_edges),
            #    len(new_edges),
            #    len(merged_edges),
            #    len(pruned_edges)
            # ))
        if not recompute:
            self.last_recomputed = False
            self.edge_filt_prev = edge_filt_new
            return self.prev_dgms

        if refresh_support:
            self.simplex_tree.persistence()
        else:
            _update_tree_filtration(
                self.simplex_tree,
                self.support_n,
                edge_filt_new,
            )
            self.simplex_tree.persistence()

        dgms = [
            np.array(self.simplex_tree.persistence_intervals_in_dimension(d))
            for d in range(self.max_dim + 1)
        ]

        self.last_recomputed = True
        self.last_force = _epoch
        self.edge_filt_prev = edge_filt_new
        return dgms

    def diagrams(self, _x, _epoch):
        """
        Returns persistence diagrams for this checkpoint.
        """
        if self.max_edge_len is None:
            self._fit_epoch1(_x)

        if (self.mode in ["fixed_support_vr",
                         "fixed_knn_vr",
                         "online_landmark_dynamic_support"] and
                self.support_edges is None):
            self._fit_epoch1(_x)

        x_use = self._cloud_for_epoch(_x)

        if self.mode == "full_vr":
            dgms = self._compute_full_vr(x_use)

        elif self.mode == "landmark_vr":
            dgms = self._compute_full_vr(x_use)

        elif self.mode == "skip_vr":
            if (self.prev_dgms is not None and
                    (_epoch % self.skip_every != 0)):
                return self.prev_dgms
            dgms = self._compute_full_vr(x_use)

        elif self.mode == "fixed_support_vr":
            dgms = self._compute_fixed_support(x_use)

        elif self.mode == "fixed_knn_vr":
            dgms = self._compute_fixed_support(x_use)

        elif self.mode == "event_driven":
            if self._should_recompute_event(x_use, _epoch):
                self.last_recomputed = True
                dgms = self._compute_full_vr(x_use)
            else:
                self.last_recomputed = False
                return self.prev_dgms
        elif self.mode == "online_landmark_event":
            dgms = self._compute_online_landmark_event(x_use, _epoch)
        elif self.mode == "online_landmark_dynamic_support":
            dgms = self._compute_online_landmark_dynamic_support(_x, x_use, _epoch)
        else:
            raise ValueError("Unknown mode")

        self.prev_dgms = dgms
        self.prev_epoch = _epoch
        self.prev_x_use = x_use.copy()
        self.prev_x_full = _x.copy()
        return dgms


# -----------------------------------------------------------------------------
# DTM-aware alternate workflow layer.
# -----------------------------------------------------------------------------
# This section intentionally leaves the original VR workflow above intact.  The
# public class name PHWorkflow is reintroduced below as a subclass that delegates
# all VR modes to _VRPHWorkflow and handles the new DTM modes locally.

from scipy.spatial.distance import cdist


_DTM_MODE_ALIASES = {
    "full_vr": "full_dtm",
    "landmark_vr": "landmark_dtm",
    "skip_vr": "skip_dtm",
    "fixed_support_vr": "fixed_support_dtm",
    "fixed_knn_vr": "fixed_knn_dtm",
    "event_driven": "event_driven_dtm",
    "online_landmark_event": "online_landmark_dtm_event",
    "online_landmark_dynamic_support": "online_landmark_dtm_dynamic_support",
}

_DTM_MODES = set(_DTM_MODE_ALIASES.values()) | {
    "full_dtm",
    "landmark_dtm",
    "skip_dtm",
    "fixed_support_dtm",
    "fixed_knn_dtm",
    "event_driven_dtm",
    "online_landmark_dtm_event",
    "online_landmark_dtm_dynamic_support",
    "online_landmark_dynamic_support_dtm",
}


def _safe_knn_k(_k, _n):
    """
    Return a k value that is valid for a point cloud with _n points.

    :param _k: Requested neighbor count.
    :param _n: Number of points.
    :return: Valid neighbor count.
    """
    if _n <= 1:
        raise ValueError("Need at least two points for nearest-neighbor support")
    return int(max(1, min(int(_k), _n - 1)))


def _compute_dtm_values(_x_ref, _x_query, _k=16, _exclude_zero=True):
    """
    Compute empirical distance-to-measure values for query points.

    This uses the square-root of the mean squared distance to the k nearest
    reference points.  When a query point is present in the reference set, the
    leading zero self-distance is skipped when possible.

    :param _x_ref: Reference point cloud of shape (n_ref, d).
    :param _x_query: Query point cloud of shape (n_query, d).
    :param _k: Number of neighbors used for the empirical DTM estimate.
    :param _exclude_zero: Whether to skip a leading zero self-distance.
    :return: DTM values of shape (n_query,).
    """
    x_ref = np.asarray(_x_ref, dtype=float)
    x_query = np.asarray(_x_query, dtype=float)

    if x_ref.ndim != 2 or x_query.ndim != 2:
        raise ValueError("DTM inputs must be two-dimensional point-cloud arrays")
    if x_ref.shape[1] != x_query.shape[1]:
        raise ValueError("Reference and query point clouds must share dimension")
    if x_ref.shape[0] == 0 or x_query.shape[0] == 0:
        raise ValueError("DTM inputs must be nonempty")

    d2 = cdist(x_query, x_ref, metric="sqeuclidean")
    d2.sort(axis=1)

    k = int(max(1, _k))
    vals = np.empty(x_query.shape[0], dtype=float)

    for row_idx, row in enumerate(d2):
        start = 0
        if _exclude_zero and row.shape[0] > 1 and row[0] <= 1e-24:
            start = 1
        stop = min(row.shape[0], start + k)
        if stop <= start:
            start = 0
            stop = min(row.shape[0], k)
        vals[row_idx] = np.sqrt(np.mean(row[start:stop]))

    return vals


def _dtm_vertex_stats(_dtm_vals):
    """
    Summarize DTM vertex values.

    :param _dtm_vals: DTM values.
    :return: Dictionary of scalar summaries.
    """
    vals = np.asarray(_dtm_vals, dtype=float)
    if vals.size == 0:
        return {
            "dtm_mean": 0.0,
            "dtm_std": 0.0,
            "dtm_median": 0.0,
            "dtm_q90": 0.0,
            "dtm_q95": 0.0,
            "dtm_max": 0.0,
        }
    q = np.quantile(vals, [0.5, 0.9, 0.95, 1.0])
    return {
        "dtm_mean": float(np.mean(vals)),
        "dtm_std": float(np.std(vals)),
        "dtm_median": float(q[0]),
        "dtm_q90": float(q[1]),
        "dtm_q95": float(q[2]),
        "dtm_max": float(q[3]),
    }


def _mean_relative_value_change(_new, _old):
    """
    Mean relative drift between two same-shaped value arrays.

    :param _new: New values.
    :param _old: Old values.
    :return: Mean relative change.
    """
    new = np.asarray(_new, dtype=float)
    old = np.asarray(_old, dtype=float)
    if new.shape != old.shape:
        raise ValueError("Value arrays must have the same shape")
    denom = np.abs(old) + 1e-12
    return float(np.mean(np.abs(new - old) / denom))


def _rank_value_drift(_new, _old):
    """
    Normalized rank drift between two same-shaped value arrays.

    :param _new: New values.
    :param _old: Old values.
    :return: Value in [0, 1] up to numerical error.
    """
    new = np.asarray(_new, dtype=float).reshape(-1)
    old = np.asarray(_old, dtype=float).reshape(-1)
    if new.shape != old.shape:
        raise ValueError("Value arrays must have the same shape")
    n = new.shape[0]
    if n <= 1:
        return 0.0
    old_rank = np.empty(n, dtype=float)
    new_rank = np.empty(n, dtype=float)
    old_rank[np.argsort(old)] = np.arange(n, dtype=float)
    new_rank[np.argsort(new)] = np.arange(n, dtype=float)
    return float(np.mean(np.abs(new_rank - old_rank)) / float(n - 1))


def _dtm_edge_filtration_from_d_mat(_edges, _d_mat, _dtm_vals,
                                    _max_edge_len=None,
                                    _rule="max",
                                    _dtm_weight=1.0,
                                    _edge_weight=1.0):
    """
    Convert support edges into DTM-aware filtration values.

    :param _edges: Edge list.
    :param _d_mat: Pairwise distance matrix on the support vertices.
    :param _dtm_vals: DTM vertex values on the support vertices.
    :param _max_edge_len: Optional cap on the distance part of the edge value.
    :param _rule: Combination rule: max, sqrt_sum, additive, or dtm_only.
    :param _dtm_weight: Weight applied to DTM vertex values.
    :param _edge_weight: Weight applied to metric edge lengths.
    :return: Edge-filtration dictionary.
    """
    vals = np.asarray(_dtm_vals, dtype=float)
    filt = {}
    for i, j in _edges:
        dist = float(_d_mat[i, j])
        if _max_edge_len is not None:
            dist = min(dist, float(_max_edge_len))
        dist = float(_edge_weight) * dist
        vi = float(_dtm_weight) * float(vals[i])
        vj = float(_dtm_weight) * float(vals[j])

        if _rule == "max":
            val = max(dist, vi, vj)
        elif _rule == "sqrt_sum":
            val = float(np.sqrt(dist * dist + vi * vi + vj * vj))
        elif _rule == "additive":
            val = dist + 0.5 * (vi + vj)
        elif _rule == "dtm_only":
            val = max(vi, vj)
        else:
            raise ValueError("Unknown DTM edge rule: {}".format(_rule))
        filt[(i, j)] = float(val)
    return filt


def _build_dtm_clique_tree(_n, _edges, _edge_filtration, _vertex_filtration,
                           _max_dim):
    """
    Build a clique complex with nonzero vertex filtrations.

    :param _n: Number of vertices.
    :param _edges: Edge list.
    :param _edge_filtration: Edge-filtration dictionary.
    :param _vertex_filtration: Vertex-filtration values.
    :param _max_dim: Maximum homology dimension.
    :return: Gudhi SimplexTree.
    """
    st = gd.SimplexTree() # pylint: disable=no-member
    vertex_vals = np.asarray(_vertex_filtration, dtype=float)
    if vertex_vals.shape[0] != _n:
        raise ValueError("Vertex filtration length must match number of vertices")

    for i in range(_n):
        st.insert([i], filtration=float(vertex_vals[i]))
    for i, j in _edges:
        edge_val = max(float(_edge_filtration[(i, j)]),
                       float(vertex_vals[i]),
                       float(vertex_vals[j]))
        st.insert([i, j], filtration=edge_val)
    st.expansion(_max_dim + 1)
    st.make_filtration_non_decreasing()
    return st


def _update_dtm_tree_filtration(_st, _n, _edge_filtration, _vertex_filtration):
    """
    Update vertex and edge filtrations for a DTM-aware SimplexTree.

    :param _st: Gudhi SimplexTree.
    :param _n: Number of vertices.
    :param _edge_filtration: Edge-filtration dictionary.
    :param _vertex_filtration: Vertex-filtration values.
    """
    vertex_vals = np.asarray(_vertex_filtration, dtype=float)
    if vertex_vals.shape[0] != _n:
        raise ValueError("Vertex filtration length must match number of vertices")

    for i in range(_n):
        _st.assign_filtration([i], float(vertex_vals[i]))
    for (i, j), val in _edge_filtration.items():
        edge_val = max(float(val), float(vertex_vals[i]), float(vertex_vals[j]))
        _st.assign_filtration([i, j], edge_val)
    _st.make_filtration_non_decreasing()


def _refresh_landmark_dtm_support(_x_landmark, _x_ref, _k, _dtm_k,
                                  _max_edge_length=None,
                                  _rule="max",
                                  _dtm_weight=1.0,
                                  _edge_weight=1.0):
    """
    Refresh a landmark support graph and DTM-aware filtrations.

    :param _x_landmark: Landmark point cloud.
    :param _x_ref: Reference cloud used to estimate DTM values.
    :param _k: Neighborhood size for support edges.
    :param _dtm_k: Number of neighbors used for DTM values.
    :param _max_edge_length: Optional cap on the distance part of edge values.
    :param _rule: DTM edge-combination rule.
    :param _dtm_weight: Weight applied to DTM values.
    :param _edge_weight: Weight applied to metric edge lengths.
    :return: Edges, distance matrix, edge filtration, vertex filtration.
    """
    k_use = _safe_knn_k(_k, _x_landmark.shape[0])
    edges, d_mat = _knn_edges(_x_landmark, k_use)
    vertex_filt = _compute_dtm_values(_x_ref, _x_landmark, _dtm_k)
    edge_filt = _dtm_edge_filtration_from_d_mat(
        edges,
        d_mat,
        vertex_filt,
        _max_edge_len=_max_edge_length,
        _rule=_rule,
        _dtm_weight=_dtm_weight,
        _edge_weight=_edge_weight,
    )
    return edges, d_mat, edge_filt, vertex_filt


class PHWorkflow(_VRPHWorkflow):
    """
    Drop-in PH workflow with DTM-aware alternatives to the VR speedup modes.

    Existing VR modes are delegated unchanged to the original workflow.  DTM can
    be requested either by using an explicit DTM mode, such as
    ``online_landmark_dtm_dynamic_support``, or by passing ``_filtration="dtm"``
    with an existing VR mode name.

    DTM modes
    ---------
    full_dtm:
        Recompute a DTM-weighted clique filtration on the current cloud.

    landmark_dtm:
        Use fixed landmarks but estimate landmark DTM values against the full
        current cloud by default.

    skip_dtm:
        Recompute DTM PH every ``_skip_every`` epochs and reuse diagrams between
        recomputations.

    fixed_support_dtm / fixed_knn_dtm:
        Build a radius or kNN support once, then update DTM vertex values and
        DTM-weighted edge filtrations on that fixed support.

    event_driven_dtm:
        Recompute full DTM PH only when the cloud changes enough.

    online_landmark_dtm_event:
        Fixed landmarks and fixed kNN support; update DTM-aware filtrations each
        epoch and recompute PH only when edge or DTM drift is large enough.

    online_landmark_dtm_dynamic_support:
        Fixed landmarks with dynamic kNN support refresh driven by metric,
        support, coverage, and DTM-value drift.
    """

    def __init__(self,
                 _mode="full_vr",
                 _max_dim=1,
                 _sparse=0.2,
                 _too_big=False,
                 _n_landmarks=250,
                 _seed=17,
                 _skip_every=2,
                 _knn_k=12,
                 _event_thresh=0.02,
                 _event_max_skip=5,
                 _force_every=10,
                 _filtration="vr",
                 _dtm_k=16,
                 _dtm_rule="max",
                 _dtm_weight=1.0,
                 _dtm_edge_weight=1.0,
                 _dtm_use_full_cloud=True,
                 _dtm_event_weights=None):
        super().__init__(
            _mode=_mode,
            _max_dim=_max_dim,
            _sparse=_sparse,
            _too_big=_too_big,
            _n_landmarks=_n_landmarks,
            _seed=_seed,
            _skip_every=_skip_every,
            _knn_k=_knn_k,
            _event_thresh=_event_thresh,
            _event_max_skip=_event_max_skip,
            _force_every=_force_every,
        )
        self.filtration = _filtration
        self.dtm_k = _dtm_k
        self.dtm_rule = _dtm_rule
        self.dtm_weight = _dtm_weight
        self.dtm_edge_weight = _dtm_edge_weight
        self.dtm_use_full_cloud = _dtm_use_full_cloud
        self.dtm_event_weights = _dtm_event_weights or {
            "edge_drift": 1.0,
            "knn_identity_drift": 1.0,
            "coverage_drift": 1.0,
            "dtm_value_drift": 1.0,
            "dtm_rank_drift": 0.25,
        }
        self.vertex_filt_prev = None
        self.last_dtm_stats = None
        self.last_dtm_diag = None

    def _dtm_mode(self):
        if self.mode in _DTM_MODES:
            if self.mode == "online_landmark_dynamic_support_dtm":
                return "online_landmark_dtm_dynamic_support"
            return self.mode
        if self.filtration == "dtm":
            return _DTM_MODE_ALIASES.get(self.mode, self.mode)
        return None

    def _is_dtm_active(self):
        return self._dtm_mode() in _DTM_MODES

    def _dtm_reference_cloud(self, _x_full, _x_use):
        if self.dtm_use_full_cloud:
            return _x_full
        return _x_use

    def _dtm_values_for_epoch(self, _x_full, _x_use):
        x_ref = self._dtm_reference_cloud(_x_full, _x_use)
        vals = _compute_dtm_values(x_ref, _x_use, self.dtm_k)
        self.last_dtm_stats = _dtm_vertex_stats(vals)
        return vals

    def _dtm_edge_filtration(self, _edges, _d_mat, _vertex_filt):
        return _dtm_edge_filtration_from_d_mat(
            _edges,
            _d_mat,
            _vertex_filt,
            _max_edge_len=self.max_edge_len,
            _rule=self.dtm_rule,
            _dtm_weight=self.dtm_weight,
            _edge_weight=self.dtm_edge_weight,
        )

    def _fit_epoch1(self, _x):
        if not self._is_dtm_active():
            return super()._fit_epoch1(_x)

        self.max_edge_len = compute_max_edge(_x, self.too_big)
        mode = self._dtm_mode()

        if mode in [
                "landmark_dtm",
                "online_landmark_dtm_event",
                "online_landmark_dtm_dynamic_support"]:
            self.landmark_idx = furthest_point_subsample(
                _x,
                self.n_landmarks,
                self.seed,
            )

        x_use = self._cloud_for_epoch(_x)
        x_ref = self._dtm_reference_cloud(_x, x_use)

        if mode == "fixed_support_dtm":
            edges, d_mat = _radius_edges(x_use, self.max_edge_len)
            vertex_filt = _compute_dtm_values(x_ref, x_use, self.dtm_k)
            edge_filt = self._dtm_edge_filtration(edges, d_mat, vertex_filt)
            self.support_edges = edges
            self.support_n = x_use.shape[0]
            self.simplex_tree = _build_dtm_clique_tree(
                self.support_n,
                self.support_edges,
                edge_filt,
                vertex_filt,
                self.max_dim,
            )
            self.edge_filt_prev = edge_filt
            self.vertex_filt_prev = vertex_filt

        if mode == "fixed_knn_dtm":
            edges, d_mat = _knn_edges(x_use, _safe_knn_k(self.knn_k, x_use.shape[0]))
            vertex_filt = _compute_dtm_values(x_ref, x_use, self.dtm_k)
            edge_filt = self._dtm_edge_filtration(edges, d_mat, vertex_filt)
            self.support_edges = edges
            self.support_n = x_use.shape[0]
            self.simplex_tree = _build_dtm_clique_tree(
                self.support_n,
                self.support_edges,
                edge_filt,
                vertex_filt,
                self.max_dim,
            )
            self.edge_filt_prev = edge_filt
            self.vertex_filt_prev = vertex_filt

        if mode == "online_landmark_dtm_event":
            edges, d_mat = _knn_edges(x_use, _safe_knn_k(self.knn_k, x_use.shape[0]))
            vertex_filt = _compute_dtm_values(x_ref, x_use, self.dtm_k)
            edge_filt = self._dtm_edge_filtration(edges, d_mat, vertex_filt)
            self.support_edges = edges
            self.support_n = x_use.shape[0]
            self.simplex_tree = _build_dtm_clique_tree(
                self.support_n,
                self.support_edges,
                edge_filt,
                vertex_filt,
                self.max_dim,
            )
            self.edge_filt_prev = edge_filt
            self.vertex_filt_prev = vertex_filt

        if mode == "online_landmark_dtm_dynamic_support":
            edges, d_mat, edge_filt, vertex_filt = _refresh_landmark_dtm_support(
                x_use,
                x_ref,
                self.knn_k,
                self.dtm_k,
                self.max_edge_len,
                self.dtm_rule,
                self.dtm_weight,
                self.dtm_edge_weight,
            )
            self.support_edges = edges
            self.support_edge_age = {e: 0 for e in edges}
            self.support_n = x_use.shape[0]
            self.simplex_tree = _build_dtm_clique_tree(
                self.support_n,
                self.support_edges,
                edge_filt,
                vertex_filt,
                self.max_dim,
            )
            self.edge_filt_prev = edge_filt
            self.vertex_filt_prev = vertex_filt

        return None

    def _persistence_from_dtm_tree(self):
        self.simplex_tree.persistence()
        return [
            np.array(self.simplex_tree.persistence_intervals_in_dimension(d))
            for d in range(self.max_dim + 1)
        ]

    def _compute_full_dtm(self, _x_full, _x_use):
        edges, d_mat = _radius_edges(_x_use, self.max_edge_len)
        vertex_filt = self._dtm_values_for_epoch(_x_full, _x_use)
        edge_filt = self._dtm_edge_filtration(edges, d_mat, vertex_filt)
        self.simplex_tree = _build_dtm_clique_tree(
            _x_use.shape[0],
            edges,
            edge_filt,
            vertex_filt,
            self.max_dim,
        )
        self.support_edges = edges
        self.support_n = _x_use.shape[0]
        self.edge_filt_prev = edge_filt
        self.vertex_filt_prev = vertex_filt
        return self._persistence_from_dtm_tree()

    def _compute_fixed_dtm_support(self, _x_full, _x_use):
        if self.support_edges is None:
            raise ValueError("Support edges is None")
        d_mat = _pairwise_dist(_x_use)
        vertex_filt = self._dtm_values_for_epoch(_x_full, _x_use)
        edge_filt = self._dtm_edge_filtration(self.support_edges, d_mat, vertex_filt)
        _update_dtm_tree_filtration(
            self.simplex_tree,
            self.support_n,
            edge_filt,
            vertex_filt,
        )
        self.edge_filt_prev = edge_filt
        self.vertex_filt_prev = vertex_filt
        return self._persistence_from_dtm_tree()

    def _should_recompute_dtm_event(self, _x_full, _x_use, _epoch):
        if self.prev_dgms is None:
            return True
        if self.prev_x_use is None or self.prev_epoch is None:
            return True
        if (_epoch - self.prev_epoch) >= self.event_max_skip:
            return True
        return _simplex_change_score(_x_use, self.prev_x_use) >= self.event_thresh

    def _dtm_event_score(self, _diag):
        score = 0.0
        for key, weight in self.dtm_event_weights.items():
            score += float(weight) * float(_diag.get(key, 0.0))
        return float(score)

    def _dtm_event_reason(self, _diag, _epoch):
        reasons = []
        score = self._dtm_event_score(_diag)
        if score >= self.event_thresh:
            reasons.append("score_thresh")
        recall = _diag.get("support_edge_recall")
        if (self.min_support_recall is not None and recall is not None and
                float(recall) < float(self.min_support_recall)):
            reasons.append("low_support_recall")
        if self.prev_epoch is not None and (_epoch - self.prev_epoch) >= self.event_max_skip:
            reasons.append("max_skip")
        if self.last_force is not None and (_epoch - self.last_force) >= self.force_every:
            reasons.append("force_every")
        return reasons

    def _compute_online_landmark_dtm_event(self, _x_full, _x_use, _epoch):
        d_mat = _pairwise_dist(_x_use)
        vertex_filt_new = self._dtm_values_for_epoch(_x_full, _x_use)
        edge_filt_new = self._dtm_edge_filtration(
            self.support_edges,
            d_mat,
            vertex_filt_new,
        )

        if self.prev_dgms is None or self.prev_epoch is None:
            recompute = True
            event_score = np.inf
        else:
            edge_drift = _mean_relative_edge_change(edge_filt_new, self.edge_filt_prev)
            dtm_drift = _mean_relative_value_change(vertex_filt_new, self.vertex_filt_prev)
            event_score = edge_drift + dtm_drift
            max_skip = (_epoch - self.prev_epoch) >= self.event_max_skip
            force = self.last_force is not None and ((_epoch - self.last_force) >= self.force_every)
            recompute = event_score >= self.event_thresh or max_skip or force

        self.last_event_score = event_score

        if not recompute:
            self.last_recomputed = False
            self.edge_filt_prev = edge_filt_new
            self.vertex_filt_prev = vertex_filt_new
            return self.prev_dgms

        _update_dtm_tree_filtration(
            self.simplex_tree,
            self.support_n,
            edge_filt_new,
            vertex_filt_new,
        )
        dgms = self._persistence_from_dtm_tree()
        self.last_recomputed = True
        self.last_force = _epoch
        self.edge_filt_prev = edge_filt_new
        self.vertex_filt_prev = vertex_filt_new
        return dgms

    def _build_dtm_simplex_tree_from_support(self, _x_full, _x_landmarks, _edges):
        d_mat = _pairwise_dist(_x_landmarks)
        vertex_filt = self._dtm_values_for_epoch(_x_full, _x_landmarks)
        edge_filt = self._dtm_edge_filtration(_edges, d_mat, vertex_filt)
        st = _build_dtm_clique_tree(
            _x_landmarks.shape[0],
            _edges,
            edge_filt,
            vertex_filt,
            self.max_dim,
        )
        return st, d_mat, edge_filt, vertex_filt

    def _compute_online_landmark_dtm_dynamic_support(self, _x_full, _x_use, _epoch):
        """
        Event-driven PH on a landmark cloud with dynamic DTM-aware support.
        """
        if self.support_edges is None or self.simplex_tree is None:
            x_ref = self._dtm_reference_cloud(_x_full, _x_use)
            edges, _, edge_filt, vertex_filt = _refresh_landmark_dtm_support(
                _x_use,
                x_ref,
                self.knn_k,
                self.dtm_k,
                self.max_edge_len,
                self.dtm_rule,
                self.dtm_weight,
                self.dtm_edge_weight,
            )
            self.support_edges = edges
            self.support_edge_age = {e: 0 for e in edges}
            self.support_n = _x_use.shape[0]
            self.simplex_tree = _build_dtm_clique_tree(
                self.support_n,
                self.support_edges,
                edge_filt,
                vertex_filt,
                self.max_dim,
            )
            self.edge_filt_prev = edge_filt
            self.vertex_filt_prev = vertex_filt

        d_mat_curr = _pairwise_dist(_x_use)
        vertex_filt_new = self._dtm_values_for_epoch(_x_full, _x_use)
        edge_filt_new = self._dtm_edge_filtration(
            self.support_edges,
            d_mat_curr,
            vertex_filt_new,
        )

        if self.prev_dgms is None or self.prev_x_use is None or self.prev_epoch is None:
            _update_dtm_tree_filtration(
                self.simplex_tree,
                self.support_n,
                edge_filt_new,
                vertex_filt_new,
            )
            dgms = self._persistence_from_dtm_tree()
            self.last_recomputed = True
            self.last_force = _epoch
            self.edge_filt_prev = edge_filt_new
            self.vertex_filt_prev = vertex_filt_new
            self.last_event_score = np.inf
            self.prev_x_full = _x_full.copy()
            return dgms

        edge_drift = _mean_relative_edge_change(edge_filt_new, self.edge_filt_prev)
        dtm_value_drift = _mean_relative_value_change(vertex_filt_new, self.vertex_filt_prev)
        dtm_rank_drift = _rank_value_drift(vertex_filt_new, self.vertex_filt_prev)

        knn_prev = _compute_knn_indices(self.prev_x_use, _safe_knn_k(self.knn_k, self.prev_x_use.shape[0]))
        knn_new = _compute_knn_indices(_x_use, _safe_knn_k(self.knn_k, _x_use.shape[0]))

        cov_prev = _landmark_coverage_stats(self.prev_x_full, self.prev_x_use)
        cov_new = _landmark_coverage_stats(_x_full, _x_use)

        diag = {
            "edge_drift": float(edge_drift),
            "knn_identity_drift": float(_knn_identity_drift(knn_prev, knn_new)),
            "knn_rank_drift": float(_knn_rank_drift(knn_prev, knn_new)),
            "coverage_drift": float(abs(cov_new["q95"] - cov_prev["q95"])),
            "support_edge_recall": float(_support_edge_recall(self.support_edges, _x_use, _safe_knn_k(self.knn_k, _x_use.shape[0]))),
            "support_edge_precision": float(_support_edge_precision(self.support_edges, _x_use, _safe_knn_k(self.knn_k, _x_use.shape[0]))),
            "dtm_value_drift": float(dtm_value_drift),
            "dtm_rank_drift": float(dtm_rank_drift),
        }
        self.last_dtm_diag = diag
        event_score = self._dtm_event_score(diag)
        self.last_event_score = event_score

        reason_code = self._dtm_event_reason(diag, _epoch)
        refresh_support = (
            "score_thresh" in reason_code or
            "low_support_recall" in reason_code
        )
        recompute = bool(reason_code)

        if refresh_support:
            x_ref = self._dtm_reference_cloud(_x_full, _x_use)
            new_edges, _, _, _ = _refresh_landmark_dtm_support(
                _x_use,
                x_ref,
                self.knn_k,
                self.dtm_k,
                self.max_edge_len,
                self.dtm_rule,
                self.dtm_weight,
                self.dtm_edge_weight,
            )
            merged_edges = self._merge_support_edges(self.support_edges, new_edges)
            pruned_edges = self._prune_support_edges(
                merged_edges,
                new_edges,
                self.support_max_age,
            )
            self.support_edges = pruned_edges
            self.support_n = _x_use.shape[0]
            self.simplex_tree, d_mat_curr, edge_filt_new, vertex_filt_new = (
                self._build_dtm_simplex_tree_from_support(
                    _x_full,
                    _x_use,
                    self.support_edges,
                )
            )

        if not recompute:
            self.last_recomputed = False
            self.edge_filt_prev = edge_filt_new
            self.vertex_filt_prev = vertex_filt_new
            return self.prev_dgms

        if refresh_support:
            dgms = self._persistence_from_dtm_tree()
        else:
            _update_dtm_tree_filtration(
                self.simplex_tree,
                self.support_n,
                edge_filt_new,
                vertex_filt_new,
            )
            dgms = self._persistence_from_dtm_tree()

        self.last_recomputed = True
        self.last_force = _epoch
        self.edge_filt_prev = edge_filt_new
        self.vertex_filt_prev = vertex_filt_new
        return dgms

    def diagrams(self, _x, _epoch):
        """
        Return persistence diagrams for a checkpoint.

        VR modes are delegated to the original workflow.  DTM modes use a
        density-aware clique filtration with DTM vertex values and DTM-weighted
        edge values.
        """
        if not self._is_dtm_active():
            dgms = super().diagrams(_x, _epoch)
            dgms = _normalize_diagrams(dgms, self.max_dim)
            self.prev_dgms = dgms
            return dgms

        if self.max_edge_len is None:
            self._fit_epoch1(_x)

        mode = self._dtm_mode()
        if (mode in [
                "fixed_support_dtm",
                "fixed_knn_dtm",
                "online_landmark_dtm_event",
                "online_landmark_dtm_dynamic_support"] and
                self.support_edges is None):
            self._fit_epoch1(_x)

        x_use = self._cloud_for_epoch(_x)

        if mode == "full_dtm":
            dgms = self._compute_full_dtm(_x, x_use)

        elif mode == "landmark_dtm":
            dgms = self._compute_full_dtm(_x, x_use)

        elif mode == "skip_dtm":
            if self.prev_dgms is not None and (_epoch % self.skip_every != 0):
                return self.prev_dgms
            dgms = self._compute_full_dtm(_x, x_use)

        elif mode == "fixed_support_dtm":
            dgms = self._compute_fixed_dtm_support(_x, x_use)

        elif mode == "fixed_knn_dtm":
            dgms = self._compute_fixed_dtm_support(_x, x_use)

        elif mode == "event_driven_dtm":
            if self._should_recompute_dtm_event(_x, x_use, _epoch):
                self.last_recomputed = True
                dgms = self._compute_full_dtm(_x, x_use)
            else:
                self.last_recomputed = False
                return self.prev_dgms

        elif mode == "online_landmark_dtm_event":
            dgms = self._compute_online_landmark_dtm_event(_x, x_use, _epoch)

        elif mode == "online_landmark_dtm_dynamic_support":
            dgms = self._compute_online_landmark_dtm_dynamic_support(_x, x_use, _epoch)

        else:
            raise ValueError("Unknown DTM mode: {}".format(mode))

        self.prev_dgms = dgms
        self.prev_epoch = _epoch
        self.prev_x_use = x_use.copy()
        self.prev_x_full = _x.copy()
        return dgms
