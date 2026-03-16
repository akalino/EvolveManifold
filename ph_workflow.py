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
                 _event_max_skip=5):
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

        self.max_edge_len = None
        self.landmark_idx = None

        self.prev_dgms = None
        self.prev_epoch = None
        self.prev_x_use = None

        self.support_edges = None
        self.simplex_tree = None
        self.support_n = None

        self.last_recomputed = False

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
            edge_filt = _edge_filtration_from_d_mat(edges, d_mat, self.knn_k)
            self.support_edges = edges
            self.support_n = x_use.shape[0]
            self.simplex_tree = _build_clique_tree(self.support_n,
                                                   self.support_edges,
                                                   edge_filt,
                                                   self.max_dim)
            print("[PH FIXED KNN EDGES] {}".format(len(edges)))

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

    def diagrams(self, _x, _epoch):
        """
        Returns persistence diagrams for this checkpoint.
        """
        if self.max_edge_len is None:
            self._fit_epoch1(_x)

        if self.mode in ["fixed_support_vr",
                         "fixed_knn_vr"] and self.support_edges is None:
            self._fit_epoch1(_x)

        x_use = self._cloud_for_epoch(_x)

        if self.mode == "full_vr":
            dgms = self._compute_full_vr(x_use)

        elif self.mode == "landmark_vr":
            dgms = self._compute_full_vr(x_use)

        elif self.mode == "skip_vr":
            if self.prev_dgms is not None and (_epoch % self.skip_every != 0):
                return self.prev_dgms
            dgms = self._compute_fixed_support(x_use)

        elif self.mode == "fixed_support_vr":
            dgms = self._compute_fixed_support(x_use)

        elif self.mode == "fixed_knn_vr":
            dgms = self._compute_fixed_support(x_use)

        elif self.mode == "event_driven":
            if self._should_recompute_event(x_use, _epoch):
                self.last_recomputed = True
                dgms = self._compute_full_vr(x_use)
            else:
                return self.prev_dgms

        else:
            raise ValueError("Unknown mode")

        self.prev_dgms = dgms
        self.prev_epoch = _epoch
        self.prev_x_use = x_use.copy()
        return dgms