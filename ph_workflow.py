"""
Experiments geared toward not having to compute full VR at each checkpoint.
"""
import numpy as np
import gudhi as gd

from scipy.spatial.distance import pdist, squareform

from complex_persistence import compute_vr_diagrams


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
    print("[VR MAX EDGE LENGTH] {}".format(max_edge_len))
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
    st = gd.SimplexTree()
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
    n, k = _knn_old.shape
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

    :param _knn_old:
    :param _knn_new:
    :return:
    """
    pass


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
    changed = (_assign_prev != _assign_new)
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
    pass

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

        if _support_edges is not None:
            out['support_edge_recall'] = _support_edge_recall(_support_edges,
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


class PHWorkflow:
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
            print("[PH LANDMARKS] {}".format(len(self.landmark_idx)))

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
            print("[PH FIXED SUPPORT EDGES] {}".format(len(edges)))

        if self.mode == "fixed_knn_vr":
            edges, d_mat = _knn_edges(x_use, self.knn_k)
            edge_filt = _edge_filtration_from_d_mat(edges, d_mat, self.max_edge_len)
            self.support_edges = edges
            self.support_n = x_use.shape[0]
            self.simplex_tree = _build_clique_tree(self.support_n,
                                                   self.support_edges,
                                                   edge_filt,
                                                   self.max_dim)
            print("[PH FIXED KNN EDGES] {}".format(len(edges)))

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
            print("[PH ONLINE LANDMARK EVENT] {} edges".format(len(edges)))

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
            print("[PH ONLINE LANDMARK DYNAMIC SUPPORT] {} edges".format(len(edges)))

    def _cloud_for_epoch(self, _x):
        if self.landmark_idx is not None:
            return _x[self.landmark_idx]
        return _x

    def _compute_full_vr(self, _x_use):
        return compute_vr_diagrams(_x_use,
                                   _max_dim=self.max_dim,
                                   _max_edge_length=self.max_edge_len,
                                   _sparse=self.sparse)

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
            print("[PH EVENT DRIVEN] change event occurred at epoch {}".format(_epoch))
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

        diag = {
            "edge_drift": float(edge_drift),
            "knn_identity_drift": float(_knn_identity_drift(knn_prev, knn_new)),
            "coverage_drift": float(abs(cov_new["q95"] - cov_prev["q95"])),
            "support_edge_recall": float(
                _support_edge_recall(self.support_edges, _x_use, self.knn_k)
            ),
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
            print("[PH DYNAMIC SUPPORT] old {}, new {}, merged {}, kept {}".format(
                len(self.support_edges),
                len(new_edges),
                len(merged_edges),
                len(pruned_edges)
            ))
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
