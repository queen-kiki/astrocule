import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment, minimize
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u


def get_estradiol_graph():
    """
    Returns the exact 2D planar vertex coordinates (unit bond length = 1.0)
    and edge list for the 20 heavy atoms of 17beta-estradiol (C18H24O2).
    """
    s3 = np.sqrt(3.0)
    c18, s18 = np.cos(np.deg2rad(18)), np.sin(np.deg2rad(18))
    c54, s54 = np.cos(np.deg2rad(54)), np.sin(np.deg2rad(54))
    c72, s72 = np.cos(np.deg2rad(72)), np.sin(np.deg2rad(72))

    # 20 heavy atoms: C1..C17 (steroid core), O3 (phenolic OH), C18 (methyl), O17 (17b-OH)
    atoms = {
        "C1":  (0.0, 1.0),
        "C2":  (-0.5 * s3, 0.5),
        "C3":  (-0.5 * s3, -0.5),
        "C4":  (0.0, -1.0),
        "C5":  (0.5 * s3, -0.5),
        "C10": (0.5 * s3, 0.5),
        "C6":  (s3, -1.0),
        "C7":  (1.5 * s3, -0.5),
        "C8":  (1.5 * s3, 0.5),
        "C9":  (s3, 1.0),
        "C11": (s3, 2.0),
        "C12": (1.5 * s3, 2.5),
        "C13": (2.0 * s3, 2.0),
        "C14": (2.0 * s3, 1.0),
        "C15": (2.0 * s3 + c18, 1.0 - s18),
        "C16": (2.0 * s3 + c18 + c54, 1.5),
        "C17": (2.0 * s3 + c18, 2.0 + s18),
        "O3":  (-s3, -1.0),
        "C18": (2.0 * s3, 3.0),
        "O17": (2.0 * s3 + c18 + c72, 2.0 + s18 + s72),
    }

    names = list(atoms.keys())
    idx = {name: i for i, name in enumerate(names)}
    coords = np.array([atoms[n] for n in names], dtype=float)
    coords -= coords.mean(axis=0)  # Center at origin

    bond_pairs = [
        # Ring A
        ("C1", "C2"), ("C2", "C3"), ("C3", "C4"),
        ("C4", "C5"), ("C5", "C10"), ("C10", "C1"),
        # Ring B
        ("C5", "C6"), ("C6", "C7"), ("C7", "C8"),
        ("C8", "C9"), ("C9", "C10"),
        # Ring C
        ("C9", "C11"), ("C11", "C12"), ("C12", "C13"),
        ("C13", "C14"), ("C14", "C8"),
        # Ring D
        ("C14", "C15"), ("C15", "C16"), ("C16", "C17"), ("C17", "C13"),
        # Functional groups
        ("C3", "O3"), ("C13", "C18"), ("C17", "O17"),
    ]
    edges = [(idx[a], idx[b]) for a, b in bond_pairs]
    double_bonds = [(idx["C1"], idx["C2"]), (idx["C3"], idx["C4"]), (idx["C5"], idx["C10"])]
    return names, coords, edges, double_bonds


def fetch_star_field(ra0_deg=305.5, dec0_deg=42.0, radius_deg=10.0, mag_limit=8.0):
    """
    Queries the Hipparcos catalog (I/239/hip_main) around (ra0_deg, dec0_deg).
    Default center is the rich Cygnus Milky Way field (near Sadr / Deneb),
    which transits directly overhead at ~47°N latitude.
    """
    print(f"Querying Hipparcos within {radius_deg}° of RA={ra0_deg}°, Dec={dec0_deg}° (Vmag < {mag_limit})...")
    v = Vizier(
        columns=["HIP", "_RAJ2000", "_DEJ2000", "Vmag"],
        column_filters={"Vmag": f"<{mag_limit}"},
        row_limit=-1,
    )
    center = SkyCoord(ra=ra0_deg * u.deg, dec=dec0_deg * u.deg, frame="icrs")
    result = v.query_region(center, radius=radius_deg * u.deg, catalog="I/239/hip_main")
    table = result[0]

    ra = np.array(table["_RAJ2000"], dtype=float)
    dec = np.array(table["_DEJ2000"], dtype=float)
    hip = np.array(table["HIP"], dtype=int)
    vmag = np.array(table["Vmag"], dtype=float)

    # Local tangent-plane projection (degrees)
    cos_dec0 = np.cos(np.deg2rad(dec0_deg))
    xi = (ra - ra0_deg) * cos_dec0
    eta = dec - dec0_deg
    star_xy = np.column_stack([xi, eta])
    print(f"Loaded {len(hip)} stars from Hipparcos.")
    return star_xy, hip, ra, dec, vmag, (ra0_deg, dec0_deg, cos_dec0)


def transform_points(template, params):
    """Applies similarity transform (tx, ty, theta, scale) to (N, 2) template."""
    tx, ty, theta, scale = params
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    return scale * (template @ rot.T) + np.array([tx, ty])


def find_best_asterism(template, star_xy, n_samples=60000, scale_bounds=(0.35, 0.85), seed=42):
    """
    Vectorized KD-Tree global sweep followed by Hungarian 1-to-1 assignment
    and Nelder-Mead local refinement.
    """
    rng = np.random.default_rng(seed)
    tree = cKDTree(star_xy)
    n_atoms = len(template)

    # Bound translation so the molecule stays well inside the star field
    xy_min, xy_max = star_xy.min(axis=0) + 1.5, star_xy.max(axis=0) - 1.5
    tx = rng.uniform(xy_min[0], xy_max[0], n_samples)
    ty = rng.uniform(xy_min[1], xy_max[1], n_samples)
    theta = rng.uniform(0, 2 * np.pi, n_samples)
    scale = rng.uniform(scale_bounds[0], scale_bounds[1], n_samples)

    # Vectorized rigid + scale transformation: shape (n_samples, 20, 2)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.stack([np.stack([c, -s], axis=-1), np.stack([s, c], axis=-1)], axis=-2)
    transformed = scale[:, None, None] * np.einsum("nij,kj->nki", rot, template)
    transformed += np.stack([tx, ty], axis=-1)[:, None, :]

    # Query KD-Tree in bulk
    dists, indices = tree.query(transformed.reshape(-1, 2), workers=-1)
    dists = dists.reshape(n_samples, n_atoms)
    indices = indices.reshape(n_samples, n_atoms)

    # Score by normalized RMSD ( penalizing any duplicate star assignments )
    rmsd = np.sqrt(np.mean(dists ** 2, axis=1)) / scale
    top_candidates = np.argsort(rmsd)[:150]

    best_score = np.inf
    best_params = None
    best_star_idx = None

    for cand in top_candidates:
        p0 = np.array([tx[cand], ty[cand], theta[cand], scale[cand]])

        def objective(p):
            if not (scale_bounds[0] <= p[3] <= scale_bounds[1]):
                return 1e6
            pts = transform_points(template, p)
            d, idx = tree.query(pts, k=1)
            # Penalty if two atoms snap to the same star
            unique_penalty = (n_atoms - len(np.unique(idx))) * 0.5
            return np.sqrt(np.mean(d ** 2)) / p[3] + unique_penalty

        res = minimize(objective, p0, method="Nelder-Mead", options={"maxiter": 400})
        pts = transform_points(template, res.x)

        # Guarantee strict 1-to-1 star assignment via Hungarian algorithm on k=8 nearest stars
        d_knn, idx_knn = tree.query(pts, k=8)
        candidate_stars = np.unique(idx_knn.ravel())
        cost_matrix = np.linalg.norm(pts[:, None, :] - star_xy[candidate_stars][None, :, :], axis=-1)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_idx = candidate_stars[col_ind]

        actual_rmsd_deg = np.sqrt(np.mean(cost_matrix[row_ind, col_ind] ** 2))
        norm_score = actual_rmsd_deg / res.x[3]

        if norm_score < best_score:
            best_score = norm_score
            best_params = res.x
            best_star_idx = matched_idx

    return best_params, best_star_idx


def plot_tattoo_chart(names, template, edges, double_bonds, star_xy, hip, ra, dec, vmag, best_params, best_star_idx):
    matched_xy = star_xy[best_star_idx]
    ideal_xy = transform_points(template, best_params)
    rmsd_arcmin = np.sqrt(np.mean(np.sum((matched_xy - ideal_xy) ** 2, axis=1))) * 60.0

    # Print catalog table
    print("\n" + "=" * 72)
    print(f"BEST ESTRADIOL ASTERISM MATCH (Bond scale = {best_params[3]:.3f}°, RMSD = {rmsd_arcmin:.2f} arcmin)")
    print("=" * 72)
    print(f"{'Atom':<6} {'HIP ID':<10} {'RA (J2000 deg)':<16} {'Dec (J2000 deg)':<16} {'Vmag':<6} {'Offset (arcmin)'}")
    print("-" * 72)
    for i, name in enumerate(names):
        s_idx = best_star_idx[i]
        err_arcmin = np.linalg.norm(matched_xy[i] - ideal_xy[i]) * 60.0
        print(f"{name:<6} HIP {hip[s_idx]:<6d} {ra[s_idx]:<16.5f} {dec[s_idx]:<16.5f} {vmag[s_idx]:<6.2f} {err_arcmin:.2f}'")

    center_ra = np.mean(ra[best_star_idx])
    center_dec = np.mean(dec[best_star_idx])
    c_coord = SkyCoord(ra=center_ra * u.deg, dec=center_dec * u.deg)
    coord_label = c_coord.to_string("hmsdms", precision=1)
    print("-" * 72)
    print(f"Tattoo Center Coordinates (J2000): {coord_label}\n")

    # Plot dark-mode celestial chart
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0b0e14")
    ax.set_facecolor("#0b0e14")

    # Filter background stars inside bounding box
    margin = 1.2
    xmin, ymin = matched_xy.min(axis=0) - margin
    xmax, ymax = matched_xy.max(axis=0) + margin
    mask = (star_xy[:, 0] >= xmin) & (star_xy[:, 0] <= xmax) & (star_xy[:, 1] >= ymin) & (star_xy[:, 1] <= ymax)

    bg_xy = star_xy[mask]
    bg_mag = vmag[mask]
    bg_sizes = 35.0 * (10 ** (-0.35 * (bg_mag - 5.0)))
    ax.scatter(bg_xy[:, 0], bg_xy[:, 1], s=bg_sizes, c="#cfd8dc", alpha=0.35, edgecolors="none", zorder=1)

    # Draw solid steroid backbone bonds between real stars
    for i, j in edges:
        p1, p2 = matched_xy[i], matched_xy[j]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#e2f1ff", lw=1.6, alpha=0.9, zorder=3)

    # Draw dotted inner aromatic double bonds in Ring A
    ring_a_center = matched_xy[:6].mean(axis=0)
    for i, j in double_bonds:
        p1 = matched_xy[i] + 0.18 * (ring_a_center - matched_xy[i])
        p2 = matched_xy[j] + 0.18 * (ring_a_center - matched_xy[j])
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#e2f1ff", lw=1.2, ls="--", alpha=0.85, zorder=3)

    # Highlight the 20 constellation stars (brighter starbursts for O3, O17, C18)
    special = {"O3", "O17", "C18"}
    for i, name in enumerate(names):
        x, y = matched_xy[i]
        if name in special:
            ax.scatter([x], [y], s=130, marker="*", c="#ffffff", zorder=5)
            ax.scatter([x], [y], s=260, c="#80d8ff", alpha=0.25, edgecolors="none", zorder=4)
        else:
            ax.scatter([x], [y], s=42, c="#ffffff", zorder=5)
            ax.scatter([x], [y], s=110, c="#80d8ff", alpha=0.18, edgecolors="none", zorder=4)

    ax.set_xlim(xmax, xmin)  # Astronomical convention: East (higher RA) to the left
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.set_title(
        f"17β-Estradiol Asterism  •  J2000 {coord_label}",
        color="#e2f1ff", fontsize=12, fontfamily="monospace", pad=16
    )
    plt.tight_layout()
    plt.savefig("estradiol_constellation.png", dpi=300, facecolor=fig.get_facecolor())
    print("Saved high-res chart to 'estradiol_constellation.png'.")
    plt.show()


if __name__ == "__main__":
    names, template, edges, double_bonds = get_estradiol_graph()
    star_xy, hip, ra, dec, vmag, proj = fetch_star_field(
        ra0_deg=305.5, dec0_deg=42.0, radius_deg=10.0, mag_limit=8.0
    )
    best_params, best_star_idx = find_best_asterism(template, star_xy)
    plot_tattoo_chart(
        names, template, edges, double_bonds,
        star_xy, hip, ra, dec, vmag, best_params, best_star_idx
    )
