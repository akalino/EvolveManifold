from itertools import combinations


POINTS = {
    "a": (0,0),
    "b": (1,0),
    "c": (0,1),
    "d": (1,1),
    "e": (0,5),
    "f": (5,5),
    "g": (5,0),
}

VERTICES = list(POINTS.keys())


def taxicab(_p, _q):
    """
    Computes taxicab distance between two points.
    :param _p: first point.
    :param _q: second point.
    :return: distance between two points.
    """
    return abs(_p[0] - _q[0]) + abs(_p[1] - _q[1])


def gen_dist_dict(_vertices):
    """
    Generates dictionary of pairwise distances.
    Could later add the distance function as an input.

    :param _vertices: list of distinct vertices.
    :return: dict of pairwise distances.
    """
    dist = {}
    for u, v in combinations(_vertices, 2):
        dist[(u, v)] = taxicab(POINTS[u], POINTS[v])
    return dist


def print_distance_table(_vertices, _dist):
    """
    Pretty-print the pairwise taxicab distance table.
    """
    # column width: max label length or distance length
    width = 5

    # header
    print("\nTaxicab distance table:")
    print(" " * width + "".join(f"{v:>{width}}" for v in _vertices))

    # rows
    for u in _vertices:
        row = [f"{u:>{width}}"]
        for v in _vertices:
            if u == v:
                d = 0
            else:
                d = _dist[tuple(sorted((u, v)))]
            row.append(f"{d:>{width}}")
        print("".join(row))


def rips(_vertices, _r, _dist):
    """
    Return all simplices in R_r(p).
    Per the problem, imclude up to dim 2 to compute beta_0 and beta_1.

    :param _vertices: list of distinct vertices.
    :param _r: scale for Rips computation.
    :param _dist: pairwise distance dictionary.
    :return:
    """
    simplices_0 = [(v,) for v in _vertices]

    simplices_1 = []
    for u, v in combinations(_vertices, 2):
        if _dist[tuple(sorted((u, v)))] <= _r:  # sorting to maintain order
            simplices_1.append(tuple(sorted((u, v))))

    simplices_2 = []
    for tri in combinations(_vertices, 3):
        edges = combinations(tri, 2)
        if all(_dist[tuple(sorted(e))] <= _r for e in edges):
            simplices_2.append(tuple(sorted(tri)))

    return simplices_0, simplices_1, simplices_2


def rank_f2(_mat):
    """
    Compute rank of matrix _mat over F_2 via Gaussian elimination.

    :param _mat: Martix (here list of lists, but could/should swap to numpy).
    :return: the ranks.
    """
    if not _mat:
        return 0

    if not _mat[0]:
        return 0

    a = [row[:] for row in _mat]
    m = len(a)
    n = len(a[0])
    rank = 0
    col = 0

    while rank < m and col < n:
        pivot = None
        for i in range(rank, m):
            if a[i][col] == 1:
                pivot = i
                break

        if pivot is None:
            col += 1
            continue

        a[rank], a[pivot] = a[pivot], a[rank]

        for i in range(m):
            if i != rank and a[i][col] == 1:
                a[i] = [(x ^ y) for x, y in zip(a[i], a[rank])]

        rank += 1
        col += 1

    return rank


def betti_nums(_vertices, _r, _dist):
    """
    Compute Betti numbers at each step r.

    :param _vertices: list of distinct vertices.
    :param _r: Critical value for the "radius" (using Rips, but could use Cech).
    :param _dist: pairwise distance dictionary.
    :return: beta_0, beta_1
    """
    s0, s1, s2 = rips(_vertices, _r, _dist)

    # doing each boundary map separately
    # d_1: C_1 -> C_0
    d1 = []
    for v in s0:
        row = []
        for edge in s1:
            row.append(1 if v[0] in edge else 0)
        d1.append(row)

    rank_d1 = rank_f2(d1)

    # now d_2: C_2 -> C_1
    d2 = []
    for edge in s1:
        row = []
        for tri in s2:
            tri_edges = [tuple(sorted(e)) for e in combinations(tri, 2)]
            row.append(1 if edge in tri_edges else 0)
        d2.append(row)

    rank_d2 = rank_f2(d2)

    n0 = len(s0)
    n1 = len(s1)

    beta_0 = n0 - rank_d1
    beta_1 = n1 - rank_d1 - rank_d2

    return beta_0, beta_1


def main(_vertices, _dist):
    """
    Compute Betti numbers at each step r.
    :param _vertices: list of distinct vertices.
    :param _dist: pairwise distance dictionary.
    :return: None, prints to console.
    """

    critical_vals = sorted(set([0] + list(_dist.values())))
    print(f"critical_vals: {critical_vals}")

    print_distance_table(_vertices, _dist)

    for r in critical_vals:
        beta_0, beta_1 = betti_nums(_vertices, r, _dist)
        print(f"*** r={r} | beta_0={beta_0} | beta_1={beta_1}")


if __name__ == "__main__":
    dist = gen_dist_dict(VERTICES)
    main(VERTICES, dist)
