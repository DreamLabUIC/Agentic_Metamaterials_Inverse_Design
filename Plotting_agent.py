import numpy as np
import matplotlib.pyplot as plt
from math import comb
from scipy.spatial import cKDTree


import re
import pandas as pd


ELEMENT_COLORS = {
    "Fe": "#9C1C1C", "Al": "#A7A9B7", "Cu": "#B66E41", "Ni": "#C4C4C4",
    "Ti": "#FF6347", "Au": "#F1C40F", "Pb": "#4A4A4A", "Mg": "#00FF7F",
    "Zn": "#FF1493", "Pt": "#E5E4E2", "Ag": "#C0C0C0", "Co": "#1D4B8C",
    "Sn": "#F5DEB3", "Be": "#7EC8E3", "W": "#3B3B3B", "Pd": "#B0B7B4",
    "Nb": "#4C9E9A", "Zr": "#BCC6CC", "Ta": "#6A4E3B", "Mo": "#3B3C36",
}
DEFAULT_COLOR_HEX = "#A4A9B0"


def match_material(props,
                   propertiesFile="data/Materials_Properties.csv",
                   verbose=True):
    # ---- accept either a numeric [E nu] or an 'E,nu' string ----
    if isinstance(props, str):
        parts = [float(x) for x in re.split(r"[,\s\t]+", props.strip()) if x]
        props = np.array(parts, dtype=float)
    else:
        props = np.asarray(props, dtype=float).ravel()

    E = props[0]        # Young's modulus [GPa]
    nu = props[1]       # Poisson's ratio

    # ---- read reference materials ----
    T = pd.read_csv(propertiesFile)

    # Moduli are positive, so floor(x + 0.5) reproduces it exactly.
    E_ref = np.floor(T["YoungsModulus_Mpsi"].to_numpy() * 6.8947572932 + 0.5)
    nu_ref = T["PoissonsRatio"].to_numpy(dtype=float)

    # element abbreviation from strings like "Cr (Chromium)" -> "Cr"
    baseMat = T["BaseMaterial"].astype(str).to_numpy()
    elem_ref = []
    for s in baseMat:
        m = re.match(r"[A-Za-z]+", s.strip())
        elem_ref.append(m.group(0) if m else "")
    elem_ref = np.array(elem_ref, dtype=object)

    # ---- nearest neighbour in std-scaled (E, nu) space ----
    ref = np.column_stack([E_ref, nu_ref])
    scale = np.std(ref, axis=0, ddof=0)        # population std (numpy default)
    scale[scale == 0] = 1.0
    q = np.array([E, nu]) / scale
    refScaled = ref / scale
    d2 = np.sum((refScaled - q) ** 2, axis=1)
    idx = int(np.argmin(d2))

    # ---- outputs ----
    elem = elem_ref[idx]
    matchedMaterial = str(baseMat[idx])
    if elem and elem in ELEMENT_COLORS:
        color = ELEMENT_COLORS[elem]
    else:
        color = DEFAULT_COLOR_HEX

    # ---- report ----
    if verbose:
        print("Input (E, nu)      : (%.4f GPa, %.4f)" % (E, nu))
        print("Closest material   : %s   (element %s)" % (matchedMaterial, elem))
        print("Reference (E, nu)  : (%.0f GPa, %.4f)" % (E_ref[idx], nu_ref[idx]))
        print("Print color        : %s" % color)

    return color, elem, matchedMaterial



def _bezier(control_points, t):
    """Evaluate a Bezier curve of arbitrary degree at parameter values t.

    Parameters
    ----------
    control_points : (k, 3) array-like
        Control points P0 ... P(k-1). Degree = k - 1.
    t : (m,) array
        Parameter values in [0, 1].

    Returns
    -------
    (m, 3) ndarray of curve points.
    """
    P = np.asarray(control_points, dtype=float)
    n = len(P) - 1
    t = np.asarray(t, dtype=float)
    curve = np.zeros((t.size, 3))
    for i in range(n + 1):
        # Bernstein basis polynomial b_{i,n}(t)
        b = comb(n, i) * (t ** i) * ((1.0 - t) ** (n - i))
        curve += np.outer(b, P[i])
    return curve


def draw_unitcell_bezier_colored(
    node,
    strut,
    control_point_offset_ratio,
    R,
    color="#4A4A4A",
    ax=None,
    n_samples=100,
    one_indexed=True,
    node_color="k",
):
    """Draw a truss unit cell with Bezier-curved struts.

    Parameters
    ----------
    node : (N, 3) array-like
        Nodal coordinates.
    strut : (M, 2) array-like
        Connectivity; each row is a pair of node indices.
    control_point_offset_ratio : float
        Bow amount as a fraction of the strut length.
    R : float
        Line width for struts (also scales node marker size).
    color : str, optional
        Hex code ('#4A4A4A') or Matplotlib color name ('k', 'r', ...). Default '#4A4A4A'.
    ax : mpl_toolkits.mplot3d.Axes3D, optional
        Existing 3D axes to draw on. A new figure/axes is created if None.
    n_samples : int, optional
        Number of points sampled along each Bezier curve. Default 100.
    one_indexed : bool, optional
        Set True if `strut` uses 1-based indices. Default True.

    Returns
    -------
    ax : the Axes3D used for plotting.
    """
    node = np.asarray(node, dtype=float)
    strut = np.asarray(strut, dtype=int)
    offset = 1 if one_indexed else 0

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")


    ax.scatter(
        node[:, 0], node[:, 1], node[:, 2],
        c=node_color, s=R * 10, edgecolors=node_color, depthshade=False,
    )

    t = np.linspace(0.0, 1.0, n_samples)

    for s in strut:
        pts = node[[s[0] - offset, s[1] - offset], :]

        # sortrows(pts, [3 1 2]) -> primary key z, then x, then y.
        # np.lexsort takes keys in reverse priority order.
        order = np.lexsort((pts[:, 1], pts[:, 0], pts[:, 2]))
        sp = pts[order]

        if pts[0, 2] == 0 and pts[1, 2] == 0:
            # Both nodes on bottom face -> bow upward, quadratic Bezier.
            p1, p3 = sp[0], sp[1]
            d = p3 - p1
            mid = p1 + 0.5 * d
            off = control_point_offset_ratio * np.linalg.norm(d)
            p2 = mid + np.array([0.0, 0.0, off])
            curve = _bezier([p1, p2, p3], t)

        elif pts[0, 2] == 1 and pts[1, 2] == 1:
            # Both nodes on top face -> bow downward, quadratic Bezier.
            p1, p3 = sp[0], sp[1]
            d = p3 - p1
            mid = p1 + 0.5 * d
            off = control_point_offset_ratio * np.linalg.norm(d)
            p2 = mid - np.array([0.0, 0.0, off])
            curve = _bezier([p1, p2, p3], t)

        else:
            # General strut -> cubic Bezier with two control points.
            p1, p4 = sp[0], sp[1]
            d = p4 - p1
            off = control_point_offset_ratio * np.linalg.norm(d)
            quarter = p1 + 0.25 * d
            three_quarter = p1 + 0.75 * d
            p2 = quarter + np.array([0.0, 0.0, off])
            p3 = three_quarter - np.array([0.0, 0.0, off])
            curve = _bezier([p1, p2, p3, p4], t)

        ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color=color, linewidth=R)

    ax.set_axis_off()
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    return ax



_ROUND = 9  # decimals used to merge coincident points robustly


# --------------------------------------------------------------------------- #
#  Geometry helpers
# --------------------------------------------------------------------------- #
def _plane(V0, V1, V2):
    """Plane through 3 points -> (A, B, C, D) with Ax+By+Cz+D = 0."""
    a1, b1, c1 = V1 - V0
    a2, b2, c2 = V2 - V0
    A = b1 * c2 - b2 * c1
    B = a2 * c1 - a1 * c2
    C = a1 * b2 - b1 * a2
    D = -A * V0[0] - B * V0[1] - C * V0[2]
    return A, B, C, D


def _mirror(coeffs, P):
    """Reflect point P across the plane (A, B, C, D)."""
    A, B, C, D = coeffs
    k = (-A * P[0] - B * P[1] - C * P[2] - D) / (A * A + B * B + C * C)
    x2 = A * k + P[0]
    y2 = B * k + P[1]
    z2 = C * k + P[2]
    return np.array([2 * x2 - P[0], 2 * y2 - P[1], 2 * z2 - P[2]])


def _reflect_range(rows, coeffs, i0, i1):

    src = [rows[i] for i in range(i0, i1)]
    for s in src:
        rows.append(_mirror(coeffs, s))


def _reflect_cascade(rows, coeffs, count):

    for i in range(count):
        rows.append(_mirror(coeffs, rows[i]))


def _interp(p_rows, chunk):

    n = len(p_rows)
    pin = [p_rows[0]]
    if n >= 2:
        wvals = np.linspace(0.0, 1.0, chunk)
        for i in range(n - 1):
            for w0 in wvals:
                pin.append(w0 * p_rows[i] + (1.0 - w0) * p_rows[i + 1])
    return pin[1:]


def checkConnectivity(Con):

    Con = np.asarray(Con, dtype=int)
    if Con.size == 0:
        return False

    # Adjacency matrix A (n = max node index, as in the original)
    n = int(Con.max())
    A = np.zeros((n, n))
    for i in range(Con.shape[0]):
        a, b = Con[i, 0] - 1, Con[i, 1] - 1   
        A[a, b] = 1
        A[b, a] = 1

    # Degree matrix D and Laplacian L = D - A
    D = np.diag(A.sum(axis=1))
    L = D - A

    # Laplacian eigenvalues (ascending, real -- L is symmetric)
    eigenvalues = np.linalg.eigvalsh(L)

    # Second eigenvalue > 0  ->  connected
    return round(float(eigenvalues[1]), 4) > 0


def _finalize(p_rows, pin_rows, chunk):
    """Deduplicate points, build the edge list from pin chunk-endpoints,
    map coordinates to node indices, and check connectivity.

    Returns (points Nx3, Con2 Mx2 int [1-based], connected 0/1).
    """
    P = np.round(np.vstack(p_rows), _ROUND)
    points = np.unique(P, axis=0)  # sorted unique rows

    if len(pin_rows) == 0:
        return points, np.empty((0, 2), dtype=int), 0

    PIN = np.round(np.vstack(pin_rows), _ROUND)
    starts = range(0, len(PIN), chunk)
    con = np.array(
        [np.concatenate([PIN[s], PIN[s + chunk - 1]]) for s in starts]
    )
    con = np.round(con, _ROUND)
    Con1 = np.unique(con, axis=0)

    tree = cKDTree(points)

    def idx_of(coord):
        _, i = tree.query(coord)
        return int(i) + 1  

    Con2 = np.array(
        [[idx_of(row[0:3]), idx_of(row[3:6])] for row in Con1], dtype=int
    )
    connected = checkConnectivity(Con2)
    return points, Con2, connected


# --------------------------------------------------------------------------- #
#  Point generators  (each returns an ordered dict label -> np.array(3,))
# --------------------------------------------------------------------------- #
def _v(x, y, z):
    return np.array([float(x), float(y), float(z)])


def tetr_points(w0, w1, w2, w3, w4, w5,
                w01, w02, w03, w11, w12, w13,
                w21, w22, w23, w31, w32, w33):
    """Cubic group base points."""
    v0 = _v(0, 0, 0); v1 = _v(0, 0, .5); v2 = _v(0, .5, .5); v3 = _v(.5, .5, .5)
    e0 = w0 * v0 + (1 - w0) * v1
    e1 = w1 * v1 + (1 - w1) * v2
    e2 = w2 * v2 + (1 - w2) * v3
    e3 = w3 * v3 + (1 - w3) * v1
    e4 = w4 * v3 + (1 - w3) * v0          # NB: (1-w3) as in the original
    e5 = w5 * v2 + (1 - w5) * v0
    f0 = w01 * v0 + w02 * v1 + w03 * v2
    f1 = w11 * v1 + w12 * v2 + w13 * v3
    f2 = w21 * v0 + w22 * v1 + w23 * v3
    f3 = w31 * v0 + w32 * v2 + w33 * v3
    return dict(v0=v0, v1=v1, v2=v2, v3=v3,
                e0=e0, e1=e1, e2=e2, e3=e3, e4=e4, e5=e5,
                f0=f0, f1=f1, f2=f2, f3=f3)


def prism_points(w0, w1, w2, w3, w4, w5, w6, w7, w8,
                 w01, w02, w03, w11, w12, w13,
                 w21, w22, w23, w31, w32, w33, w41, w42, w43):
    """Tetragonal group base points."""
    v0 = _v(0, 0, 0); v1 = _v(0, .5, 0); v2 = _v(0, .5, .5)
    v3 = _v(0, 0, .5); v4 = _v(.5, 0, .5); v5 = _v(.5, .5, .5)
    e0 = w0 * v0 + (1 - w0) * v1
    e1 = w1 * v1 + (1 - w1) * v2
    e2 = w2 * v2 + (1 - w2) * v3
    e3 = w3 * v0 + (1 - w3) * v3
    e4 = w4 * v0 + (1 - w4) * v4
    e5 = w5 * v1 + (1 - w5) * v5
    e6 = w6 * v3 + (1 - w6) * v4
    e7 = w7 * v2 + (1 - w7) * v5
    e8 = w8 * v4 + (1 - w8) * v5
    f0 = w01 * v0 + w02 * v1 + w03 * v2
    f1 = w11 * v2 + w12 * v3 + w13 * v4
    f2 = w21 * v0 + w22 * v3 + w23 * v4
    f3 = w31 * v1 + w32 * v2 + w33 * v5
    f4 = w41 * v0 + w42 * v1 + w43 * v4
    return dict(v0=v0, v1=v1, v2=v2, v3=v3, v4=v4, v5=v5,
                e0=e0, e1=e1, e2=e2, e3=e3, e4=e4, e5=e5, e6=e6, e7=e7, e8=e8,
                f0=f0, f1=f1, f2=f2, f3=f3, f4=f4)


def hex_points(w0, w1, w2, w3, w4, w5, w6, w7, w8,
               w01, w02, w03, w11, w12, w13,
               w21, w22, w23, w31, w32, w33, w41, w42, w43):
    """Hexagonal group base points."""
    A = np.sqrt(2 * 0.5 ** 2 - 2 * 0.5 ** 2 * np.cos(np.deg2rad(30)))
    c15 = np.cos(np.deg2rad(15)); s15 = np.sin(np.deg2rad(15))
    v0 = _v(0.5 - A * c15, 0, A * s15)
    v1 = _v(0.5, 0, 0)
    v2 = _v(0.5, 0, 0.5)
    v3 = _v(0.5 - A * c15, 0.5, A * s15)
    v4 = _v(0.5, 0.5, 0)
    v5 = _v(0.5, 0.5, 0.5)
    e0 = w0 * v0 + (1 - w0) * v1
    e1 = w1 * v1 + (1 - w1) * v2
    e2 = w2 * v0 + (1 - w2) * v2
    e3 = w3 * v0 + (1 - w3) * v3
    e4 = w4 * v1 + (1 - w4) * v4
    e5 = w5 * v2 + (1 - w5) * v5
    e6 = w6 * v3 + (1 - w6) * v4
    e7 = w7 * v4 + (1 - w7) * v5
    e8 = w8 * v3 + (1 - w8) * v5
    f0 = w01 * v0 + w02 * v1 + w03 * v2
    f1 = w11 * v0 + w12 * v1 + w13 * v3
    f2 = w21 * v0 + w22 * v2 + w23 * v3
    f3 = w31 * v1 + w32 * v2 + w33 * v4
    f4 = w41 * v3 + w42 * v4 + w43 * v5
    return dict(v0=v0, v1=v1, v2=v2, v3=v3, v4=v4, v5=v5,
                e0=e0, e1=e1, e2=e2, e3=e3, e4=e4, e5=e5, e6=e6, e7=e7, e8=e8,
                f0=f0, f1=f1, f2=f2, f3=f3, f4=f4)


def triagonal_points(w0, w1, w2, w3, w4, w5, w6, w7, w8,
                     w01, w02, w03, w11, w12, w13,
                     w21, w22, w23, w31, w32, w33, w41, w42, w43):
    """Trigonal group base points."""
    c30 = np.cos(np.deg2rad(30)); s30 = np.sin(np.deg2rad(30))
    v0 = _v(0.5 - 0.5 * c30, 0, 0.5 - 0.5 * s30)
    v1 = _v(0.5, 0, 0.5)
    v2 = _v(0.5 - 0.5 * c30, 0, 0.5 + 0.5 * s30)
    v3 = _v(0.5 - 0.5 * c30, 1, 0.5 - 0.5 * s30)
    v4 = _v(0.5, 1, 0.5)
    v5 = _v(0.5 - 0.5 * c30, 1, 0.5 + 0.5 * s30)
    e0 = w0 * v0 + (1 - w0) * v1
    e1 = w1 * v1 + (1 - w1) * v2
    e2 = w2 * v0 + (1 - w2) * v2
    e3 = w3 * v0 + (1 - w3) * v3
    e4 = w4 * v1 + (1 - w4) * v4
    e5 = w5 * v2 + (1 - w5) * v5
    e6 = w6 * v3 + (1 - w6) * v4
    e7 = w7 * v4 + (1 - w7) * v5
    e8 = w8 * v3 + (1 - w8) * v5
    f0 = w01 * v0 + w02 * v1 + w03 * v2
    f1 = w11 * v0 + w12 * v1 + w13 * v3
    f2 = w21 * v0 + w22 * v2 + w23 * v3
    f3 = w31 * v1 + w32 * v2 + w33 * v4
    f4 = w41 * v3 + w42 * v4 + w43 * v5
    return dict(v0=v0, v1=v1, v2=v2, v3=v3, v4=v4, v5=v5,
                e0=e0, e1=e1, e2=e2, e3=e3, e4=e4, e5=e5, e6=e6, e7=e7, e8=e8,
                f0=f0, f1=f1, f2=f2, f3=f3, f4=f4)


def Mono_points(w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, ww11,
                w01, w02, w03, w11, w12, w13, w21, w22, w23,
                w31, w32, w33, w41, w42, w43, w51, w52, w53):
    """Monoclinic group base points."""
    v0 = _v(0, 0, 0); v1 = _v(0, 1, 0); v2 = _v(0, 1, .5); v3 = _v(0, 0, .5)
    v4 = _v(1, 0, .5); v5 = _v(1, 1, .5); v6 = _v(1, 0, 0); v7 = _v(1, 1, 0)
    e0 = w0 * v0 + (1 - w0) * v1
    e1 = w1 * v1 + (1 - w1) * v2
    e2 = w2 * v2 + (1 - w2) * v3
    e3 = w3 * v0 + (1 - w3) * v3
    e4 = w4 * v0 + (1 - w4) * v6
    e5 = w5 * v1 + (1 - w5) * v7
    e6 = w6 * v3 + (1 - w6) * v4
    e7 = w7 * v2 + (1 - w7) * v5
    e8 = w8 * v6 + (1 - w8) * v7
    e9 = w9 * v4 + (1 - w9) * v6
    e10 = w10 * v4 + (1 - w10) * v5
    e11 = ww11 * v5 + (1 - ww11) * v7
    f0 = w01 * v0 + w02 * v1 + w03 * v2
    f1 = w11 * v2 + w12 * v3 + w13 * v4
    f2 = w21 * v0 + w22 * v3 + w23 * v4
    f3 = w31 * v1 + w32 * v2 + w33 * v5
    f4 = w41 * v0 + w42 * v1 + w43 * v6
    f5 = w51 * v4 + w52 * v5 + w53 * v6
    return dict(v0=v0, v1=v1, v2=v2, v3=v3, v4=v4, v5=v5, v6=v6, v7=v7,
                e0=e0, e1=e1, e2=e2, e3=e3, e4=e4, e5=e5, e6=e6, e7=e7,
                e8=e8, e9=e9, e10=e10, e11=e11,
                f0=f0, f1=f1, f2=f2, f3=f3, f4=f4, f5=f5)


def Ortho_points(w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, ww11,
                 w01, w02, w03, w11, w12, w13, w21, w22, w23,
                 w31, w32, w33, w41, w42, w43, w51, w52, w53):
    """Orthotropic group base points."""
    v0 = _v(0, 0, 0); v1 = _v(0, .5, 0); v2 = _v(0, .5, .5); v3 = _v(0, 0, .5)
    v4 = _v(.5, 0, .5); v5 = _v(.5, .5, .5); v6 = _v(.5, 0, 0); v7 = _v(.5, .5, 0)
    e0 = w0 * v0 + (1 - w0) * v1
    e1 = w1 * v1 + (1 - w1) * v2
    e2 = w2 * v2 + (1 - w2) * v3
    e3 = w3 * v0 + (1 - w3) * v3
    e4 = w4 * v0 + (1 - w4) * v6
    e5 = w5 * v1 + (1 - w5) * v7
    e6 = w6 * v3 + (1 - w6) * v4
    e7 = w7 * v2 + (1 - w7) * v5
    e8 = w8 * v6 + (1 - w8) * v7
    e9 = w9 * v4 + (1 - w9) * v6
    e10 = w10 * v4 + (1 - w10) * v5
    e11 = ww11 * v5 + (1 - ww11) * v7
    f0 = w01 * v0 + w02 * v1 + w03 * v2
    f1 = w11 * v2 + w12 * v3 + w13 * v4
    f2 = w21 * v0 + w22 * v3 + w23 * v4
    f3 = w31 * v1 + w32 * v2 + w33 * v5
    f4 = w41 * v0 + w42 * v1 + w43 * v6
    f5 = w51 * v4 + w52 * v5 + w53 * v6
    return dict(v0=v0, v1=v1, v2=v2, v3=v3, v4=v4, v5=v5, v6=v6, v7=v7,
                e0=e0, e1=e1, e2=e2, e3=e3, e4=e4, e5=e5, e6=e6, e7=e7,
                e8=e8, e9=e9, e10=e10, e11=e11,
                f0=f0, f1=f1, f2=f2, f3=f3, f4=f4, f5=f5)


# --------------------------------------------------------------------------- #
#  Reflection routines
# --------------------------------------------------------------------------- #
def reflections2(p):
    """Cubic group (8 mirror planes)."""
    v0 = _v(0, 0, 0); v1 = _v(0, 0, .5); v2 = _v(0, .5, .5); v3 = _v(.5, .5, .5)
    v4 = _v(.5, 0, .5); v5 = _v(.5, 0, 0); v6 = _v(0, .5, 0)  # v7 unused here
    chunk = 11
    p_rows = [np.asarray(r, float) for r in p]
    n = len(p_rows)
    pin_rows = _interp(p_rows, chunk)
    l = len(pin_rows)

    _reflect_range(p_rows, _plane(v0, v1, v3), 0, n)
    _reflect_range(pin_rows, _plane(v0, v1, v3), 0, l)

    _reflect_range(p_rows, _plane(v0, v2, v3), 0, n)
    _reflect_range(pin_rows, _plane(v0, v2, v3), 0, l)

    _reflect_range(p_rows, _plane(v0, v4, v3), n, 2 * n)
    _reflect_range(pin_rows, _plane(v0, v4, v3), l, 2 * l)

    _reflect_range(p_rows, _plane(v0, v3, v6), 2 * n, 3 * n)
    _reflect_range(pin_rows, _plane(v0, v3, v6), 2 * l, 3 * l)

    _reflect_range(p_rows, _plane(v0, v3, v5), 3 * n, 4 * n)
    _reflect_range(pin_rows, _plane(v0, v3, v5), 3 * l, 4 * l)

    for pl in (_plane(v3, v4, v5), _plane(v2, v3, v6), _plane(v2, v3, v4)):
        _reflect_range(p_rows, pl, 0, len(p_rows))
        _reflect_range(pin_rows, pl, 0, len(pin_rows))

    return _finalize(p_rows, pin_rows, chunk)


def reflect_hex(p):
    """Hexagonal group (5 mirror planes)."""
    A = np.sqrt(2 * 0.5 ** 2 - 2 * 0.5 ** 2 * np.cos(np.deg2rad(30)))
    c15 = np.cos(np.deg2rad(15)); s15 = np.sin(np.deg2rad(15))
    v0 = _v(0.5 - A * c15, 0, A * s15)
    v1 = _v(0.5, 0, 0)
    v2 = _v(0.5, 0, 0.5)
    v3 = _v(0.5 - A * c15, 0.5, A * s15)
    v4 = _v(0.5, 0.5, 0)
    v5 = _v(0.5, 0.5, 0.5)
    v6 = _v(0.5 + A * c15, 0, A * s15)
    v7 = _v(0, 0, 0.5)
    chunk = 11
    p_rows = [np.asarray(r, float) for r in p]
    n = len(p_rows)
    pin_rows = _interp(p_rows, chunk)
    l = len(pin_rows)

    _reflect_range(p_rows, _plane(v1, v2, v4), 0, n)
    _reflect_range(pin_rows, _plane(v1, v2, v4), 0, l)

    _reflect_range(p_rows, _plane(v2, v5, v6), 0, 2 * n)
    _reflect_range(pin_rows, _plane(v2, v5, v6), 0, 2 * l)

    _reflect_range(p_rows, _plane(v0, v2, v5), 0, 2 * n)
    _reflect_range(pin_rows, _plane(v0, v2, v5), 0, 2 * l)

    for pl in (_plane(v3, v4, v5), _plane(v2, v5, v7)):
        _reflect_range(p_rows, pl, 0, len(p_rows))
        _reflect_range(pin_rows, pl, 0, len(pin_rows))

    return _finalize(p_rows, pin_rows, chunk)


def reflect_trigonal(p):
    """Trigonal group (3 mirror planes)."""
    c30 = np.cos(np.deg2rad(30)); s30 = np.sin(np.deg2rad(30))
    v0 = _v(0.5 - 0.5 * c30, 0, 0.5 - 0.5 * s30)
    v1 = _v(0.5, 0, 0.5)
    v2 = _v(0.5 - 0.5 * c30, 0, 0.5 + 0.5 * s30)
    v3 = _v(0.5 - 0.5 * c30, 1, 0.5 - 0.5 * s30)
    v4 = _v(0.5, 1, 0.5)
    # v5 unused in plane defs
    v6 = _v(0.5, 0, 0)
    chunk = 11
    p_rows = [np.asarray(r, float) for r in p]
    n = len(p_rows)
    pin_rows = _interp(p_rows, chunk)
    l = len(pin_rows)

    _reflect_range(p_rows, _plane(v0, v1, v3), 0, n)
    _reflect_range(pin_rows, _plane(v0, v1, v3), 0, l)

    _reflect_range(p_rows, _plane(v1, v2, v4), 0, n)
    _reflect_range(pin_rows, _plane(v1, v2, v4), 0, l)

    pl = _plane(v1, v4, v6)
    _reflect_range(p_rows, pl, 0, len(p_rows))
    _reflect_range(pin_rows, pl, 0, len(pin_rows))

    return _finalize(p_rows, pin_rows, chunk)


def reflections_ortho(p):
    """Orthotropic group (3 cascade reflections)."""
    v1 = _v(0, .5, 0); v2 = _v(0, .5, .5); v3 = _v(0, 0, .5)
    v4 = _v(.5, 0, .5); v5 = _v(.5, .5, .5); v6 = _v(.5, 0, 0)
    chunk = 51
    p_rows = [np.asarray(r, float) for r in p]
    pin_rows = _interp(p_rows, chunk)

    for pl in (_plane(v4, v5, v6), _plane(v1, v2, v5), _plane(v2, v3, v4)):
        L = len(pin_rows)                 
        _reflect_cascade(p_rows, pl, L)   # p cascades (len(p) < L)
        _reflect_cascade(pin_rows, pl, L) # pin doubles

    return _finalize(p_rows, pin_rows, chunk)


def reflections_Mono(p):
    """Monoclinic group (1 cascade reflection)."""
    v2 = _v(0, 1, .5); v3 = _v(0, 0, .5); v4 = _v(1, 0, .5)
    chunk = 51
    p_rows = [np.asarray(r, float) for r in p]
    pin_rows = _interp(p_rows, chunk)

    pl = _plane(v2, v3, v4)
    L = len(pin_rows)
    _reflect_cascade(p_rows, pl, L)
    _reflect_cascade(pin_rows, pl, L)

    return _finalize(p_rows, pin_rows, chunk)


def reflections_tetragonal(p):
    """Tetragonal group (4 cascade reflections)."""
    v0 = _v(0, 0, 0); v1 = _v(0, .5, 0); v2 = _v(0, .5, .5); v3 = _v(0, 0, .5)
    v4 = _v(.5, 0, .5); v5 = _v(.5, .5, .5); v6 = _v(.5, 0, 0)
    chunk = 51
    p_rows = [np.asarray(r, float) for r in p]
    pin_rows = _interp(p_rows, chunk)

    planes = (_plane(v0, v1, v5), _plane(v4, v5, v6),
              _plane(v1, v2, v5), _plane(v2, v3, v4))
    for pl in planes:
        L = len(pin_rows)
        _reflect_cascade(p_rows, pl, L)
        _reflect_cascade(pin_rows, pl, L)

    return _finalize(p_rows, pin_rows, chunk)


# --------------------------------------------------------------------------- #
#  build_lattice dispatcher
# --------------------------------------------------------------------------- #
_THIRD = 1.0 / 3.0

_GROUPS = {
    "Tetragonal": dict(
        pts=lambda: prism_points(
            .5, .5, .5, .5, .5, .5, .5, .5, .5,
            .5, 0, .5, .5, 0, .5, _THIRD, _THIRD, _THIRD,
            _THIRD, _THIRD, _THIRD, 0, .5, .5),
        labels=['v0', 'v1', 'v2', 'v3', 'v4', 'v5',
                'e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8',
                'f0', 'f1', 'f2', 'f3', 'f4'],
        reflect=reflections_tetragonal),
    "Monoclinic": dict(
        pts=lambda: Mono_points(
            .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5,
            .5, 0, .5, .5, 0, .5, .5, 0, .5, .5, 0, .5,
            0, .5, .5, 0, .5, .5),
        labels=['v0', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7',
                'e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7',
                'e8', 'e9', 'e10', 'e11',
                'f0', 'f1', 'f2', 'f3', 'f4', 'f5'],
        reflect=reflections_Mono),
    "Hexagonal": dict(
        pts=lambda: hex_points(
            .5, .5, .5, .5, .5, .5, .5, .5, .5,
            _THIRD, _THIRD, _THIRD, 0, .5, .5, 0, .5, .5,
            0, .5, .5, _THIRD, _THIRD, _THIRD),
        labels=['v0', 'v1', 'v2', 'v3', 'v4', 'v5',
                'e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8',
                'f0', 'f1', 'f2', 'f3', 'f4'],
        reflect=reflect_hex),
    "Orthotropic": dict(
        pts=lambda: Ortho_points(
            .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5,
            .5, 0, .5, .5, 0, .5, .5, 0, .5, .5, 0, .5,
            0, .5, .5, 0, .5, .5),
        labels=['v0', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7',
                'e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7',
                'e8', 'e9', 'e10', 'e11',
                'f0', 'f1', 'f2', 'f3', 'f4', 'f5'],
        reflect=reflections_ortho),
    "Cubic": dict(
        pts=lambda: tetr_points(
            .5, .5, .5, .5, .5, .5,
            _THIRD, _THIRD, _THIRD, _THIRD, _THIRD, _THIRD,
            _THIRD, _THIRD, _THIRD, _THIRD, _THIRD, _THIRD),
        labels=['v0', 'v1', 'v2', 'v3',
                'e0', 'e1', 'e2', 'e3', 'e4', 'e5',
                'f0', 'f1', 'f2', 'f3'],
        reflect=reflections2),
    "Trigonal": dict(
        pts=lambda: triagonal_points(
            .5, .5, .5, .5, .5, .5, .5, .5, .5,
            _THIRD, _THIRD, _THIRD, 0, .5, .5, 0, .5, .5,
            0, .5, .5, _THIRD, _THIRD, _THIRD),
        labels=['v0', 'v1', 'v2', 'v3', 'v4', 'v5',
                'e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8',
                'f0', 'f1', 'f2', 'f3', 'f4'],
        reflect=reflect_trigonal),
}


def build_lattice(dataLine):
    """Turn one data line into a reflected unit cell.

    Parameters
    ----------
    dataLine : str
        First token = symmetry group, remaining tokens = node labels,
        e.g. 'Tetragonal e8 f0 e4 e6 e1'.

    Returns
    -------
    points : (N, 3) ndarray      -- unique nodal coordinates
    Con    : (M, 2) int ndarray  -- struts, 1-based indices into `points`
    connected : int              -- 1 if the cell is a single component else 0
    p      : (k, 3) ndarray      -- the selected seed nodes (before reflection)
    """
    tokens = dataLine.strip().split()
    class_name, sel = tokens[0], tokens[1:]

    if class_name not in _GROUPS:
        raise ValueError('Unknown symmetry group: "%s"' % class_name)

    g = _GROUPS[class_name]
    pts = g["pts"]()
    labels = g["labels"]

    p = np.zeros((len(sel), 3))
    for k, name in enumerate(sel):
        if name not in labels or name not in pts:
            raise ValueError(
                'Label "%s" is not defined for the %s group.'
                % (name, class_name))
        p[k, :] = np.ravel(pts[name])

    points, Con, connected = g["reflect"](p)
    return points, Con, connected, p






# --------------------------------------------------------------------------- #
#  Spider plot of the 21 stiffness components (port of radar_compare_groups_C.m)
# --------------------------------------------------------------------------- #
# Order of the 21-vectors (as returned by the models):
#   C11 C12 C13 C14 C15 C16 C22 C23 C24 C25 C26 C33 C34 C35 C36 C44 C45 C46 C55 C56 C66
#   (Voigt 4 = 12, 5 = 13, 6 = 23)
STIFFNESS_NAMES = ["C%d%d" % (i + 1, j + 1) for i in range(6) for j in range(i, 6)]

# Families (spokes clockwise from 12 o'clock); signed families get a symmetric scale [-m, m]
STIFFNESS_FAMILIES = [
    ("Normal stiffness",       ["11", "22", "33"],                                   False),
    ("Shear stiffness",        ["44", "55", "66"],                                   False),
    ("Normal-normal coupling", ["12", "13", "23"],                                   True),
    ("Normal-shear coupling",  ["14", "15", "16", "24", "25", "26", "34", "35", "36"], True),
    ("Shear-shear coupling",   ["45", "46", "56"],                                   True),
]


def _nice_ceil(v):
    """Smallest 'nice' number (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8 x 10^k) >= v."""
    if not np.isfinite(v) or v <= 0:
        return 1e-3
    e = np.floor(np.log10(v))
    for m in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if m * 10 ** e >= v * (1 - 1e-12):
            return float(m * 10 ** e)


def default_radar_scales(*vectors):
    """5 x 2 [lo, hi] per family (GPa) that contain all given 21-vectors:
    [0, hi] for the stiffness families, [-m, m] for the coupling families.
    Pass the same scales to every frame of a set (e.g. a diffusion path)."""
    V = np.vstack([np.asarray(v, float).ravel() for v in vectors if v is not None])
    scales = np.zeros((len(STIFFNESS_FAMILIES), 2))
    for g, (_, members, signed) in enumerate(STIFFNESS_FAMILIES):
        cols = [STIFFNESS_NAMES.index("C" + m) for m in members]
        m = _nice_ceil(1.05 * np.abs(V[:, cols]).max())
        scales[g] = (-m, m) if signed else (0.0, m)
    return scales


def radar_compare_groups_C(t, p, matColor=None, scales=None, ax=None, predColor=None, predEdgeColor=None):
    """Spider plot of the 21 stiffness components, 5 family sectors, each family
    on its own real-unit (GPa) radial scale. Returns the Axes used.

    t, p     : 21-vectors - target and predicted C (GPa), in STIFFNESS_NAMES order.
    predColor, predEdgeColor : optional colours of the predicted polygon (default paper green).
               t may be None (forward simulation: predicted polygon only).
    matColor : material colour - hex ('#9C1C1C') or RGB triple, from match_material.
               The five sectors are five shades of this colour (dark -> light).
    scales   : 5 x 2 [lo hi] per family (GPa) in the STIFFNESS_FAMILIES order;
               default: default_radar_scales(t, p).

    Signed families have zero at mid radius (dashed ring). Components that are zero
    by symmetry are still drawn (on the zero ring of their family). Values beyond a
    family's scale are drawn on the boundary with a hollow marker. The outermost tick
    of each sector carries the unit (GPa).
    Target -> solid black outline + markers; predicted -> filled area, paper green.
    """
    has_target = t is not None
    p = np.asarray(p, float).ravel()
    t = np.asarray(t, float).ravel() if has_target else None
    scales = default_radar_scales(t, p) if scales is None else np.asarray(scales, float)

    # spoke order and family of every spoke
    order, famOf = [], []
    for g, (_, members, _) in enumerate(STIFFNESS_FAMILIES):
        order += [STIFFNESS_NAMES.index("C" + m) for m in members]
        famOf += [g] * len(members)
    order, famOf = np.array(order), np.array(famOf)
    P = len(order)                                                       # 21
    labels = [r"$\mathrm{C}_{%s}$" % STIFFNESS_NAMES[k][1:] for k in order]

    # ---------- colours ----------
    green = np.array([0.173, 0.443, 0.243])                              # predicted fill (default)
    to_rgb = lambda c: np.array([int(c.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]) / 255.0 \
        if isinstance(c, str) else np.asarray(c, float)
    fill_c = green if predColor is None else to_rgb(predColor)
    edge_c = fill_c if predEdgeColor is None else to_rgb(predEdgeColor)
    if matColor is None:
        matColor = np.array([0.078, 0.373, 0.420])
    elif isinstance(matColor, str):
        hx = matColor.replace("#", "")
        matColor = np.array([int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)]) / 255.0
    else:
        matColor = np.asarray(matColor, float).ravel()
    # five shades of the material colour: two darker, the colour, two lighter
    shade = [matColor * 0.55, matColor * 0.78, matColor,
             matColor + (1 - matColor) * 0.25, matColor + (1 - matColor) * 0.45]
    txtCol = tuple(matColor * 0.5)

    # ---------- font sizes ----------
    FS_prop, FS_tick = 15, 11
    nTicks = 4

    # ---------- geometry ----------
    th = np.pi / 2 - 2 * np.pi * np.arange(P) / P
    halfStep = np.pi / P
    lo, hi = scales[famOf, 0], scales[famOf, 1]
    radius = lambda v: (v[order] - lo) / (hi - lo)
    rP = radius(p)
    clipP = (rP < 0) | (rP > 1)
    rP = np.clip(rP, 0, 1)
    if has_target:
        rT = radius(t)
        clipT = (rT < 0) | (rT > 1)
        rT = np.clip(rT, 0, 1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
        fig.patch.set_facecolor("w")
    ax.set_aspect("equal")
    ax.axis("off")

    # ---------- background sectors ----------
    for g in range(len(STIFFNESS_FAMILIES)):
        j = np.where(famOf == g)[0]
        ang = np.linspace(th[j[0]] + halfStep, th[j[-1]] - halfStep, 80)
        ax.fill(np.concatenate([[0.0], np.cos(ang)]), np.concatenate([[0.0], np.sin(ang)]),
                facecolor=(*shade[g], 0.30), edgecolor="none")
        ax.fill(np.concatenate([np.cos(ang), 0.82 * np.cos(ang[::-1])]),
                np.concatenate([np.sin(ang), 0.82 * np.sin(ang[::-1])]),
                facecolor=(*shade[g], 0.30), edgecolor="none")
        # zero ring for signed families
        if scales[g, 0] < 0:
            r0 = -scales[g, 0] / (scales[g, 1] - scales[g, 0])
            ax.plot(r0 * np.cos(ang), r0 * np.sin(ang), "--", color=txtCol, lw=1.0)

    # ---------- rings + spokes ----------
    full = np.linspace(0, 2 * np.pi, 360)
    for rg in np.arange(1, nTicks + 1) / nTicks:
        ax.plot(rg * np.cos(full), rg * np.sin(full), "-", color=(0.92, 0.92, 0.92), lw=0.6)
    for j in range(P):
        ax.plot([0, np.cos(th[j])], [0, np.sin(th[j])], "-", color=(0.92, 0.92, 0.92), lw=0.6)

    # ---------- per-sector real tick values on the middle spoke; unit on the outer one ----------
    for g in range(len(STIFFNESS_FAMILIES)):
        j = np.where(famOf == g)[0]
        a = th[j[int(np.ceil(len(j) / 2)) - 1]]
        for k in range(1, nTicks + 1):
            rr = k / nTicks
            val = scales[g, 0] + rr * (scales[g, 1] - scales[g, 0])
            txt = "0" if val == 0 else "%.3g" % val
            if k == nTicks:
                txt += " GPa"
            ax.text(rr * np.cos(a), rr * np.sin(a), txt, fontsize=FS_tick, color=txtCol,
                    fontweight="bold", ha="center", va="center")

    # ---------- polygons ----------
    close_it = lambda v: np.concatenate([v, v[:1]])
    XP = np.array([close_it(rP * np.cos(th)), close_it(rP * np.sin(th))])
    ax.fill(XP[0], XP[1], facecolor=(*fill_c, 0.65), edgecolor=(*edge_c, 0.95), linewidth=1.4)
    if clipP.any():
        ax.scatter(XP[0, :-1][clipP], XP[1, :-1][clipP], s=46, facecolors="none",
                   edgecolors=[edge_c], linewidths=1.4, zorder=5)         # hollow = clipped
    if has_target:
        XT = np.array([close_it(rT * np.cos(th)), close_it(rT * np.sin(th))])
        ax.plot(XT[0], XT[1], "-", color=(0, 0, 0, 0.95), lw=2.2)
        ax.scatter(XT[0, :-1][~clipT], XT[1, :-1][~clipT], s=30, c="k", zorder=6)
        if clipT.any():
            ax.scatter(XT[0, :-1][clipT], XT[1, :-1][clipT], s=46, facecolors="none",
                       edgecolors="k", linewidths=1.4, zorder=6)

    # ---------- component labels ----------
    for j in range(P):
        ha, va = _anchor(th[j])
        ax.text(1.03 * np.cos(th[j]), 1.03 * np.sin(th[j]), labels[j], ha=ha, va=va,
                fontsize=FS_prop, fontweight="bold")

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.3, 1.3)
    return ax


def _anchor(theta):
    c, s = np.cos(theta), np.sin(theta)
    ha = "left" if c > 0.35 else ("right" if c < -0.35 else "center")
    va = "bottom" if s > 0.35 else ("top" if s < -0.35 else "center")
    return ha, va
