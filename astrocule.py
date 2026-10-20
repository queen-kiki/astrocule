#!/usr/bin/env python3
"""
astrocule: find a molecule's skeletal formula hidden in the real night sky.

Takes any molecule (built-in name, SMILES, or a PubChem name lookup), lays out
its heavy-atom skeleton in 2D, and searches a Hipparcos star catalog for the
set of real stars that traces that skeleton with the least distortion. The
best matches are rendered as star charts with their J2000 coordinates.

    python astrocule.py estradiol
    python astrocule.py "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" --title Caffeine
    python astrocule.py estradiol --mag 8 --bond 0.35 0.85 --region 305.5 42 10
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit
from scipy.spatial import ConvexHull, Delaunay

DEG = np.pi / 180.0


# ---------------------------------------------------------------------------
# Molecules
# ---------------------------------------------------------------------------

KNOWN_MOLECULES = {
    "estradiol": ("17β-Estradiol", "C[C@]12CC[C@H]3[C@H]([C@@H]1CC[C@@H]2O)CCC4=C3C=CC(=C4)O"),
    "estradiol enanthate": (
        "Estradiol enanthate",
        "CCCCCCC(=O)O[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CCC4=C3C=CC(=C4)O)C",
    ),
    "testosterone": ("Testosterone", "C[C@]12CC[C@H]3[C@H]([C@@H]1CC[C@@H]2O)CCC4=CC(=O)CC[C@]34C"),
    "progesterone": ("Progesterone", "CC(=O)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CCC4=CC(=O)CC[C@]34C)C"),
    "spironolactone": (
        "Spironolactone",
        "CC(=O)S[C@@H]1CC2=CC(=O)CC[C@@]2([C@@H]3[C@@H]1[C@@H]4CC[C@]5([C@]4(CC3)C)CCC(=O)O5)C",
    ),
    "serotonin": ("Serotonin", "C1=CC2=C(C=C1O)C(=CN2)CCN"),
    "dopamine": ("Dopamine", "C1=CC(=C(C=C1CCN)O)O"),
    "caffeine": ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
}


@dataclass
class Molecule:
    name: str
    smiles: str
    symbols: list          # element symbol per heavy atom
    z: np.ndarray          # 2D layout as complex numbers, unit bond length, centroid at 0
    bonds: list            # (i, j, order) with aromatic rings kekulized to 1/2
    rings: list            # tuples of atom indices

    @property
    def n(self):
        return len(self.symbols)

    def degree(self, i):
        return sum(i in (a, b) for a, b, _ in self.bonds)


def _pubchem_smiles(name):
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{urllib.parse.quote(name)}/property/IsomericSMILES,Title/JSON"
    )
    with urllib.request.urlopen(url, timeout=20) as r:
        props = json.load(r)["PropertyTable"]["Properties"][0]
    smiles = props.get("IsomericSMILES") or props.get("SMILES")
    return props.get("Title", name), smiles


def load_molecule(query, title=None):
    """Resolve a built-in name, a SMILES string, or a PubChem name to a 2D skeleton."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdDepictor

    RDLogger.DisableLog("rdApp.*")
    key = query.strip().lower()
    if key in KNOWN_MOLECULES:
        name, smiles = KNOWN_MOLECULES[key]
    elif Chem.MolFromSmiles(query) is not None:
        name, smiles = query, query
    else:
        try:
            name, smiles = _pubchem_smiles(query)
        except Exception as e:
            raise SystemExit(
                f"'{query}' is not a built-in name or valid SMILES, and the PubChem lookup failed ({e}). "
                "Pass a SMILES string instead."
            )

    mol = Chem.RemoveHs(Chem.MolFromSmiles(smiles))
    Chem.Kekulize(mol, clearAromaticFlags=True)
    rdDepictor.SetPreferCoordGen(True)
    rdDepictor.Compute2DCoords(mol)
    pos = mol.GetConformer().GetPositions()[:, :2]

    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), int(b.GetBondTypeAsDouble())) for b in mol.GetBonds()]
    if not bonds:
        raise SystemExit("Need a molecule with at least one bond between heavy atoms.")
    z = pos[:, 0] + 1j * pos[:, 1]
    z /= np.mean([abs(z[i] - z[j]) for i, j, _ in bonds])
    z -= z.mean()

    return Molecule(
        name=title or name,
        smiles=smiles,
        symbols=[a.GetSymbol() for a in mol.GetAtoms()],
        z=z,
        bonds=bonds,
        rings=[tuple(r) for r in mol.GetRingInfo().AtomRings()],
    )


# ---------------------------------------------------------------------------
# Star catalog (Hipparcos stars via the HYG database)
# ---------------------------------------------------------------------------

HYG_URL = "https://raw.githubusercontent.com/astronexus/HYG-Database/main/hyg/CURRENT/hygdata_v41.csv"
CACHE_DIR = Path(os.environ.get("ASTROCULE_CACHE", Path.home() / ".cache" / "astrocule"))

CONSTELLATIONS = dict(
    item.split(":")
    for item in (
        "and:Andromeda ant:Antlia aps:Apus aqr:Aquarius aql:Aquila ara:Ara ari:Aries aur:Auriga boo:Boötes "
        "cae:Caelum cam:Camelopardalis cnc:Cancer cvn:Canes_Venatici cma:Canis_Major cmi:Canis_Minor "
        "cap:Capricornus car:Carina cas:Cassiopeia cen:Centaurus cep:Cepheus cet:Cetus cha:Chamaeleon "
        "cir:Circinus col:Columba com:Coma_Berenices cra:Corona_Australis crb:Corona_Borealis crv:Corvus "
        "crt:Crater cru:Crux cyg:Cygnus del:Delphinus dor:Dorado dra:Draco equ:Equuleus eri:Eridanus "
        "for:Fornax gem:Gemini gru:Grus her:Hercules hor:Horologium hya:Hydra hyi:Hydrus ind:Indus "
        "lac:Lacerta leo:Leo lmi:Leo_Minor lep:Lepus lib:Libra lup:Lupus lyn:Lynx lyr:Lyra men:Mensa "
        "mic:Microscopium mon:Monoceros mus:Musca nor:Norma oct:Octans oph:Ophiuchus ori:Orion pav:Pavo "
        "peg:Pegasus per:Perseus phe:Phoenix pic:Pictor psc:Pisces psa:Piscis_Austrinus pup:Puppis "
        "pyx:Pyxis ret:Reticulum sge:Sagitta sgr:Sagittarius sco:Scorpius scl:Sculptor sct:Scutum "
        "ser:Serpens sex:Sextans tau:Taurus tel:Telescopium tri:Triangulum tra:Triangulum_Australe "
        "tuc:Tucana uma:Ursa_Major umi:Ursa_Minor vel:Vela vir:Virgo vol:Volans vul:Vulpecula"
    ).split()
)


def radec_to_xyz(ra_deg, dec_deg):
    ra, dec = np.asarray(ra_deg) * DEG, np.asarray(dec_deg) * DEG
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=-1)


def xyz_to_radec(v):
    v = np.asarray(v, dtype=float)
    ra = np.arctan2(v[..., 1], v[..., 0]) / DEG % 360.0
    dec = np.arcsin(np.clip(v[..., 2] / np.linalg.norm(v, axis=-1), -1, 1)) / DEG
    return ra, dec


@dataclass
class Catalog:
    hip: np.ndarray
    ra: np.ndarray     # J2000, degrees
    dec: np.ndarray
    mag: np.ndarray    # V magnitude
    name: np.ndarray   # proper name or ""
    con: np.ndarray    # constellation abbreviation or ""
    ci: np.ndarray = None  # B-V color index (nan if unknown)

    def __post_init__(self):
        self.xyz = radec_to_xyz(self.ra, self.dec)
        if self.ci is None:
            self.ci = np.full(len(self.hip), np.nan)

    @property
    def temp(self):
        return bv_to_temp(self.ci)

    def subset(self, mask):
        return Catalog(self.hip[mask], self.ra[mask], self.dec[mask], self.mag[mask], self.name[mask],
                       self.con[mask], self.ci[mask])


def load_catalog(mag_limit, csv_path=None):
    """All Hipparcos stars brighter than mag_limit. Downloads and caches HYG v4.1 on first use."""
    npz = CACHE_DIR / "hyg_v41_hip_v2.npz"
    if csv_path is None and npz.exists():
        d = np.load(npz)
    else:
        if csv_path is None:
            csv_path = CACHE_DIR / "hygdata_v41.csv"
            if not csv_path.exists():
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                print(f"Downloading HYG star catalog to {csv_path} ...")
                urllib.request.urlretrieve(HYG_URL, csv_path)
        cols = {k: [] for k in ("hip", "ra", "dec", "mag", "name", "con", "ci")}
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if not row["hip"] or not row["mag"]:
                    continue
                cols["hip"].append(int(row["hip"]))
                cols["ra"].append(float(row["ra"]) * 15.0)  # HYG stores RA in hours
                cols["dec"].append(float(row["dec"]))
                cols["mag"].append(float(row["mag"]))
                cols["name"].append(row["proper"])
                cols["con"].append(row["con"])
                cols["ci"].append(float(row["ci"]) if row["ci"] else np.nan)
        d = {k: np.array(v) for k, v in cols.items()}
        if csv_path == CACHE_DIR / "hygdata_v41.csv":
            np.savez(npz, **d)
    cat = Catalog(d["hip"], d["ra"], d["dec"], d["mag"], d["name"], d["con"], d["ci"])
    return cat.subset(cat.mag <= mag_limit)


def bv_to_temp(bv):
    """Effective temperature (K) from the B-V color index (Ballesteros 2012). Unknown -> Sun-like."""
    bv = np.clip(np.nan_to_num(np.asarray(bv, dtype=float), nan=0.65), -0.4, 2.0)
    return 4600.0 * (1.0 / (0.92 * bv + 1.7) + 1.0 / (0.92 * bv + 0.62))


def temp_to_rgb(temp, saturation=1.0):
    """Approximate color of a blackbody at temp (K) as sRGB in 0..1 (Tanner Helland's fit).
    saturation > 1 exaggerates the hue, as astrophotos usually do."""
    t = np.clip(np.asarray(temp, dtype=float), 1000, 40000) / 100.0
    hot = t > 66
    r = np.where(hot, 329.698727446 * np.maximum(t - 60, 1e-9) ** -0.1332047592, 255.0)
    g = np.where(hot, 288.1221695283 * np.maximum(t - 60, 1e-9) ** -0.0755148492,
                 99.4708025861 * np.log(t) - 161.1195681661)
    b = np.where(t >= 66, 255.0,
                 np.where(t <= 19, 0.0, 138.5177312231 * np.log(np.maximum(t - 10, 1e-9)) - 305.0447927307))
    rgb = np.clip(np.stack([r, g, b], axis=-1) / 255.0, 0, 1)
    grey = rgb.mean(axis=-1, keepdims=True)
    return np.clip(grey + (rgb - grey) * saturation, 0, 1)


# ---------------------------------------------------------------------------
# Sky projection (stereographic: conformal, so it keeps molecule shapes honest)
# ---------------------------------------------------------------------------

def tangent_frame(center):
    c = center / np.linalg.norm(center)
    e = np.cross([0.0, 0.0, 1.0], c)
    if np.linalg.norm(e) < 1e-9:  # at a celestial pole
        e = np.array([0.0, 1.0, 0.0])
    e /= np.linalg.norm(e)
    return c, e, np.cross(c, e)


def project(xyz, center):
    """Stereographic projection around center, in degrees. x grows to the west so that
    plotting x to the right gives the view of the sky from below (east on the left)."""
    c, e, n = tangent_frame(center)
    k = 2.0 / (1.0 + xyz @ c) / DEG
    return -k * (xyz @ e) + 1j * k * (xyz @ n)


def unproject(w, center):
    c, e, n = tangent_frame(center)
    x, y = -w.real * DEG, w.imag * DEG
    r2 = x * x + y * y
    return ((4 * x / (4 + r2))[..., None] * e + (4 * y / (4 + r2))[..., None] * n
            + ((4 - r2) / (4 + r2))[..., None] * c)


def fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    zc = 1 - 2 * i / n
    phi = np.pi * (1 + 5 ** 0.5) * i
    r = np.sqrt(1 - zc * zc)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), zc])


def tile_cells(centers):
    """Exact Voronoi cell radius (degrees) and Voronoi neighbors of every tile center. On a sphere
    the Delaunay triangulation is the convex hull, and each cell's farthest points are the
    circumcenters of its triangles, i.e. the outward facet normals."""
    hull = ConvexHull(centers)
    radius = np.zeros(len(centers))
    neighbors = [set() for _ in centers]
    for tri, eq in zip(hull.simplices, hull.equations):
        ang = np.arccos(np.clip(centers[tri] @ eq[:3], -1, 1)) / DEG
        np.maximum.at(radius, tri, ang)
        for v in tri:
            neighbors[v].update(int(u) for u in tri if u != v)
    return radius, [np.array(sorted(nb)) for nb in neighbors]


def _xy(w):
    return np.column_stack([w.real, w.imag])


# ---------------------------------------------------------------------------
# Numba kernels (the hot loops of the search)
# ---------------------------------------------------------------------------

@njit(nogil=True, cache=True)
def _lsap(cost):
    """Minimum-cost assignment of every row to a distinct column (rows <= cols).
    Shortest augmenting path Hungarian algorithm, O(rows^2 * cols)."""
    n, m = cost.shape
    inf = np.inf
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, np.int64)    # p[j]: row (1-based) assigned to column j, 0 = free
    way = np.zeros(m + 1, np.int64)
    minv = np.empty(m + 1)
    used = np.empty(m + 1, np.bool_)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv[:] = inf
        used[:] = False
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    out = np.empty(n, np.int64)
    for j in range(1, m + 1):
        if p[j] != 0:
            out[p[j] - 1] = j - 1
    return out


@njit(nogil=True, cache=True)
def _fit_nb(src, dst, smin, smax):
    sc = src.mean()
    dc = dst.mean()
    num = 0j
    den = 0.0
    for i in range(len(src)):
        zs = src[i] - sc
        num += np.conj(zs) * (dst[i] - dc)
        den += zs.real * zs.real + zs.imag * zs.imag
    a = num / den
    s = abs(a)
    a = a * (min(max(s, smin), smax) / s)
    return a, dc - a * sc


@njit(nogil=True, cache=True)
def _refine_nb(T, P_all, a, b, smin, smax, skip, forced, iters, k):
    # Only stars that can plausibly be among an atom's k nearest: within the figure's radius plus
    # 1.5 bond lengths of where it starts. Brute force over those beats a k-d tree here.
    reach = abs(a) * (np.abs(T).max() + 1.5)
    reach2 = reach * reach
    n_loc = 0
    local = np.empty(len(P_all), np.int64)
    for j in range(len(P_all)):
        d = P_all[j] - b
        if d.real * d.real + d.imag * d.imag <= reach2:
            local[n_loc] = j
            n_loc += 1
    n = len(T)
    if n_loc < n:
        return np.full(n, -1, np.int64), a, b, np.inf
    Px = np.empty(n_loc)
    Py = np.empty(n_loc)
    for j in range(n_loc):
        Px[j] = P_all[local[j]].real
        Py[j] = P_all[local[j]].imag
    k = min(k, n_loc)
    n_drop = skip
    for i in range(n):
        if forced[i]:
            n_drop += 1
    stars = np.full(n, -1, np.int64)
    state = np.full(n, -2, np.int64)
    mark = np.zeros(n_loc, np.bool_)
    cand = np.empty(n_loc, np.int64)
    best_d = np.empty(k)
    best_j = np.empty(k, np.int64)
    use = np.ones(n, np.bool_)
    rows = np.empty(n, np.int64)
    resid = np.empty(n)
    predx = np.empty(n)
    predy = np.empty(n)
    cost = np.empty((n, min(n * k, n_loc)))
    for _ in range(iters):
        for i in range(n):
            w = a * T[i] + b
            predx[i] = w.real
            predy[i] = w.imag
        # k nearest stars of every predicted atom position; their union are the candidates
        mark[:] = False
        for i in range(n):
            best_d[:] = np.inf
            for j in range(n_loc):
                dx = Px[j] - predx[i]
                dy = Py[j] - predy[i]
                d = dx * dx + dy * dy
                if d < best_d[k - 1]:
                    q = k - 1
                    while q > 0 and best_d[q - 1] > d:
                        best_d[q] = best_d[q - 1]
                        best_j[q] = best_j[q - 1]
                        q -= 1
                    best_d[q] = d
                    best_j[q] = j
            for q in range(k):
                mark[best_j[q]] = True
        n_cand = 0
        for j in range(n_loc):
            if mark[j]:
                cand[n_cand] = j
                n_cand += 1
        if n_cand < n:
            return stars, a, b, np.inf
        C = cost[:, :n_cand]
        for i in range(n):
            for c in range(n_cand):
                dx = predx[i] - Px[cand[c]]
                dy = predy[i] - Py[cand[c]]
                C[i, c] = dx * dx + dy * dy
        use[:] = True
        if n_drop > 0:
            cols = _lsap(C)
            for i in range(n):
                resid[i] = np.inf if forced[i] else C[i, cols[i]]
            order = np.argsort(-resid)
            for q in range(n_drop):
                use[order[q]] = False
        n_rows = 0
        for i in range(n):
            if use[i]:
                rows[n_rows] = i
                n_rows += 1
        cols = _lsap(C[rows[:n_rows]])
        stars[:] = -1
        for r in range(n_rows):
            stars[rows[r]] = cand[cols[r]]
        same = True
        for i in range(n):
            if stars[i] != state[i]:
                same = False
                break
        if same:
            break
        state[:] = stars
        a, b = _fit_nb(T[rows[:n_rows]], Px[stars[rows[:n_rows]]] + 1j * Py[stars[rows[:n_rows]]], smin, smax)
    sq = 0.0
    cnt = 0
    for i in range(n):
        if stars[i] >= 0:
            w = a * T[i] + b
            dx = w.real - Px[stars[i]]
            dy = w.imag - Py[stars[i]]
            sq += dx * dx + dy * dy
            cnt += 1
            stars[i] = local[stars[i]]
    return stars, a, b, np.sqrt(sq / cnt) / abs(a)


@njit(nogil=True, cache=True)
def _grid_build(P, g):
    """Bucket points into square cells of size g (CSR layout) for radius queries."""
    x0 = P.real.min() - g
    y0 = P.imag.min() - g
    nx = int((P.real.max() - x0) / g) + 2
    ny = int((P.imag.max() - y0) / g) + 2
    cell = np.empty(len(P), np.int64)
    for i in range(len(P)):
        cell[i] = int((P[i].imag - y0) / g) * nx + int((P[i].real - x0) / g)
    order = np.argsort(cell, kind="mergesort")
    start = np.zeros(nx * ny + 1, np.int64)
    for i in range(len(P)):
        start[cell[i] + 1] += 1
    for c in range(nx * ny):
        start[c + 1] += start[c]
    return order, start, x0, y0, nx, ny


@njit(nogil=True, cache=True)
def _grid_nearest(P, grid, g, w, R):
    """Distance from w to the nearest point of P within R, or inf if there is none."""
    order, start, x0, y0, nx, ny = grid
    ix0 = max(int((w.real - R - x0) / g), 0)
    ix1 = min(int((w.real + R - x0) / g), nx - 1)
    iy0 = max(int((w.imag - R - y0) / g), 0)
    iy1 = min(int((w.imag + R - y0) / g), ny - 1)
    best = R * R
    found = False
    for iy in range(iy0, iy1 + 1):
        for ix in range(ix0, ix1 + 1):
            c = iy * nx + ix
            for q in range(start[c], start[c + 1]):
                d = P[order[q]] - w
                d2 = d.real * d.real + d.imag * d.imag
                if d2 <= best:
                    best = d2
                    found = True
    return np.sqrt(best) if found else np.inf


@njit(nogil=True, cache=True)
def _hypotheses_nb(Ts, ia, ib, order, P, grid, g, cand_i, smin, smax, tol, max_rel, skip,
                   frame, near_centers, region_c, region_cos):
    """Score every anchor-pair hypothesis of one tile, for each template in Ts (the drawing and
    its mirror image share the star pairs). Returns (a, b, sse), each (len(Ts), H), with
    sse = inf for hypotheses not owned by this tile or pruned."""
    D = abs(Ts[0, ib] - Ts[0, ia])
    r0 = (smin * D) ** 2
    r1 = (smax * D) ** 2
    M = len(P)

    # Star pairs (i, j) whose separation fits the anchor atoms' distance at an allowed scale.
    counts = np.zeros(len(cand_i), np.int64)
    for q in range(len(cand_i)):
        pi = P[cand_i[q]]
        cnt = 0
        for j in range(M):
            d = P[j] - pi
            d2 = d.real * d.real + d.imag * d.imag
            if r0 <= d2 <= r1:
                cnt += 1
        counts[q] = cnt
    offs = np.zeros(len(cand_i) + 1, np.int64)
    for q in range(len(cand_i)):
        offs[q + 1] = offs[q] + counts[q]
    H = offs[-1]
    ii = np.empty(H, np.int64)
    jj = np.empty(H, np.int64)
    for q in range(len(cand_i)):
        pi = P[cand_i[q]]
        pos = offs[q]
        for j in range(M):
            d = P[j] - pi
            d2 = d.real * d.real + d.imag * d.imag
            if r0 <= d2 <= r1:
                ii[pos] = cand_i[q]
                jj[pos] = j
                pos += 1

    c, e, nv = frame[0], frame[1], frame[2]
    n = Ts.shape[1] - skip
    limit = n * max_rel ** 2
    n_t = Ts.shape[0]
    a_out = np.empty((n_t, H), np.complex128)
    b_out = np.empty((n_t, H), np.complex128)
    sse = np.full((n_t, H), np.inf)
    for h in range(n_t * H):
        t = h // H
        hh = h % H
        T = Ts[t]
        a = (P[jj[hh]] - P[ii[hh]]) / (T[ib] - T[ia])
        b = P[ii[hh]] - a * T[ia]  # template centroid is 0, so b is where the figure's center lands
        a_out[t, hh] = a
        b_out[t, hh] = b
        # Ownership: the figure's center must fall in this tile's Voronoi cell (and the region).
        x = -b.real * (np.pi / 180.0)
        y = b.imag * (np.pi / 180.0)
        r2 = x * x + y * y
        f1 = 4.0 * x / (4.0 + r2)
        f2 = 4.0 * y / (4.0 + r2)
        f3 = (4.0 - r2) / (4.0 + r2)
        v0 = f1 * e[0] + f2 * nv[0] + f3 * c[0]
        v1 = f1 * e[1] + f2 * nv[1] + f3 * c[1]
        v2 = f1 * e[2] + f2 * nv[2] + f3 * c[2]
        own_dot = v0 * c[0] + v1 * c[1] + v2 * c[2]
        owned = True
        for u in range(near_centers.shape[0]):
            if v0 * near_centers[u, 0] + v1 * near_centers[u, 1] + v2 * near_centers[u, 2] > own_dot + 1e-12:
                owned = False
                break
        if owned and region_cos > -2.0:
            owned = v0 * region_c[0] + v1 * region_c[1] + v2 * region_c[2] >= region_cos
        if not owned:
            continue
        s = abs(a)
        R = tol * s
        acc = 0.0
        misses = 0
        alive = True
        for k in order:
            d = _grid_nearest(P, grid, g, a * T[k] + b, R)
            if d == np.inf:
                misses += 1
                if misses > skip:
                    alive = False
                    break
            else:
                acc += (d / s) ** 2
                if acc > limit:
                    alive = False
                    break
        if alive:
            sse[t, hh] = acc
    return a_out, b_out, sse


@njit(nogil=True, cache=True)
def _refine_batch(T, P, A, B, smin, smax, skip, iters, k):
    H = len(A)
    stars = np.empty((H, len(T)), np.int64)
    a_out = np.empty(H, np.complex128)
    b_out = np.empty(H, np.complex128)
    score = np.empty(H)
    no_force = np.zeros(len(T), np.bool_)
    for h in range(H):
        s, a, b, sc = _refine_nb(T, P, A[h], B[h], smin, smax, skip, no_force, iters, k)
        stars[h] = s
        a_out[h] = a
        b_out[h] = b
        score[h] = sc
    return stars, a_out, b_out, score


@njit(nogil=True, cache=True)
def _refine_each_forced(T, P, a, b, smin, smax, extra, iters, k):
    """Refine once per atom, with that atom forced starless (plus `extra` greedy drops)."""
    n = len(T)
    stars = np.empty((n, n), np.int64)
    a_out = np.empty(n, np.complex128)
    b_out = np.empty(n, np.complex128)
    score = np.empty(n)
    for i in range(n):
        f = np.zeros(n, np.bool_)
        f[i] = True
        s, a2, b2, sc = _refine_nb(T, P, a, b, smin, smax, extra, f, iters, k)
        stars[i] = s
        a_out[i] = a2
        b_out[i] = b2
        score[i] = sc
    return stars, a_out, b_out, score


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def fit_similarity(src, dst, smin=0.0, smax=np.inf):
    """Least-squares w = a*z + b (rotation + uniform scale + shift, no reflection),
    with |a| clamped to [smin, smax]."""
    sc, dc = src.mean(), dst.mean()
    a = np.vdot(src - sc, dst - dc) / np.vdot(src - sc, src - sc).real
    s = abs(a)
    a *= np.clip(s, smin, smax) / s
    return a, dc - a * sc


def refine(T, P, a, b, smin, smax, skip=0, forced=(), iters=12, k=6):
    """Alternate one-to-one star assignment (Hungarian) and similarity refit until stable.
    Candidates are the k nearest stars of every predicted atom position.

    The atoms in `forced`, plus the `skip` worst-fitting others, go without a star: they are left
    out of the assignment and the fit and just become line corners on the chart (trimmed ICP).
    Returns (stars, a, b, score) with stars[i] = -1 for starless atoms, or None."""
    f = np.zeros(len(T), dtype=np.bool_)
    f[list(forced)] = True
    stars, a, b, score = _refine_nb(T, P, complex(a), complex(b), smin, smax, skip, f, iters, k)
    if not np.isfinite(score):
        return None
    return stars, a, b, score


@dataclass
class Match:
    stars: np.ndarray   # catalog index for each atom, -1 where the atom has no star
    mirrored: bool
    center: np.ndarray  # projection center (unit vector)
    a: complex          # template -> chart plane: w = a*z + b, in degrees
    b: complex
    score: float        # RMSD of the starred atoms, in units of bond length (shape distortion)

    @property
    def used(self):
        return self.stars >= 0

    @property
    def n_skipped(self):
        return int(np.sum(~self.used))

    @property
    def bond_deg(self):
        return abs(self.a)

    @property
    def rmsd_arcmin(self):
        return self.score * self.bond_deg * 60.0


def _hypotheses(Ts, anchors, order, P, grid, g, cand_i, frame, near_centers, region,
                smin, smax, tol, max_rel, keep, skip):
    """Every star pair (i, j) that could host the anchor atoms defines one similarity
    transform. Score all of them atom by atom against the nearest stars, pruning early.
    Up to `skip` atoms may miss (no star within tol) without killing the hypothesis.
    Returns, per template in Ts, the `keep` best surviving (a, b) as two arrays."""
    region_c, region_cos = region if region is not None else (np.zeros(3), -3.0)
    a, b, sse = _hypotheses_nb(Ts, anchors[0], anchors[1], np.asarray(order, dtype=np.int64), P, grid, g,
                               cand_i, smin, smax, tol, max_rel, skip, frame, near_centers,
                               region_c, region_cos)
    out = []
    for t in range(len(Ts)):
        alive = np.flatnonzero(np.isfinite(sse[t]))
        best = alive[np.argsort(sse[t, alive], kind="stable")[:keep]]
        out.append((a[t, best], b[t, best]))
    return out


def _polish(T, mirrored, stars, cat_xyz, smin, smax, skip):
    """Re-project around the matched figure itself and refine there, so the final fit and
    the chart share one nearly distortion-free frame. Returns the best variant with exactly
    k starless atoms for every k up to `skip`; for k >= 1 every atom is tried as the first
    one to drop, the rest are picked greedily."""
    used = stars[stars >= 0]
    center = cat_xyz[used].mean(axis=0)
    center /= np.linalg.norm(center)
    r = np.arccos(np.clip(cat_xyz[used] @ center, -1, 1)).max() + 2 * smax * DEG
    field = np.flatnonzero(cat_xyz @ center >= np.cos(min(r, 80 * DEG)))
    P = project(cat_xyz[field], center)
    keep = stars >= 0
    a, b = fit_similarity(T[keep], P[np.searchsorted(field, stars[keep])], smin, smax)

    out = []
    for k in range(skip + 1):
        if k == 0:
            res = refine(T, P, a, b, smin, smax)
            if res is None:
                continue
            s, a2, b2, score = res
        else:
            S, A2, B2, scores = _refine_each_forced(T, P, complex(a), complex(b), smin, smax, k - 1, 12, 6)
            i = int(np.argmin(scores))
            if not np.isfinite(scores[i]):
                continue
            s, a2, b2, score = S[i], A2[i], B2[i], scores[i]
        out.append(Match(np.where(s >= 0, field[np.maximum(s, 0)], -1), mirrored, center, a2, b2, score))
    return out


def _anchor_pairs(z, count):
    """`count` disjoint atom pairs, each the farthest-apart pair among the atoms left, so that
    any `count - 1` starless atoms leave at least one pair fully starred."""
    left = list(range(len(z)))
    pairs = []
    for _ in range(count):
        if len(left) < 2:
            break
        sub = z[left]
        d = np.abs(sub[:, None] - sub[None, :])
        i, j = np.unravel_index(np.argmax(d), d.shape)
        pairs.append((left[i], left[j]))
        left = [x for x in left if x not in (left[i], left[j])]
    return pairs


def find_asterisms(mol, cat, bond=(1.0, 3.0), region=None, mirror=True, skip=0, top=10,
                   tol=0.6, max_rel=0.4, keep=150, polish=600, tile_deg=6.0, workers=None, verbose=True):
    """
    Search the sky for the star sets that best trace mol's skeleton.

    bond:    allowed bond length on the sky in degrees (min, max)
    region:  optional (ra_deg, dec_deg, radius_deg) to restrict where the figure's center may lie
    mirror:  also allow the mirror image of the 2D drawing
    skip:    allow up to this many atoms without a star (drawn as plain line corners)
    workers: threads to use (default: all cores)
    polish:  how many of the best raw candidates to re-fit in their own frame. Raw scores come from
             tile projections not centered on the figure, so be generous here.
    Returns up to `top` matches sorted by distortion, at least one per number of starless atoms,
    each using mostly different stars from the others with the same number of starless atoms.
    """
    smin, smax = bond
    if len(cat.hip) < mol.n:
        raise SystemExit(f"Only {len(cat.hip)} stars available, need at least {mol.n}.")
    if not 0 <= skip <= mol.n - 3:
        raise SystemExit(f"--skip must be between 0 and {mol.n - 3} for this molecule.")
    templates = [(False, mol.z)] + ([(True, np.conj(mol.z))] if mirror else [])
    Ts = np.array([T for _, T in templates])

    anchor_sets = []
    for anchors in _anchor_pairs(mol.z, skip + 1):
        order = [k for k in np.argsort(-np.abs(mol.z)) if k not in anchors]
        anchor_sets.append((anchors, order, np.abs(mol.z[list(anchors)]).max()))
    r_fig = np.abs(mol.z).max()

    xyz = cat.xyz
    n_tiles = int(np.ceil(2.0 / (1.0 - np.cos(tile_deg * DEG))))
    centers = fibonacci_sphere(n_tiles)
    cell_radius, cell_neighbors = tile_cells(centers)
    tiles = np.arange(n_tiles)
    if region is not None:
        # Same small tiles as the whole-sky search (so the per-tile candidate cap means the same
        # thing), restricted to tiles that can own a figure center inside the region.
        ra0, dec0, rad = region
        region_c = radec_to_xyz(ra0, dec0)
        sep = np.arccos(np.clip(centers @ region_c, -1, 1)) / DEG
        tiles = np.flatnonzero(sep <= rad + cell_radius + 1e-6)

    region_info = None if region is None else (region_c, np.cos(rad * DEG))
    workers = workers or os.cpu_count() or 1
    g = tol * smax / 2  # grid cell size for the star lookups
    def search_tile(t):
        """All figures whose center falls in tile t, as {(mirrored, stars): score}."""
        out = {}
        c = centers[t]
        cell_r = cell_radius[t] + 1e-6
        near = cell_neighbors[t]  # a point is in this tile's cell iff it's closer than to these
        field_r = min(cell_r + smax * r_fig + smax, 80.0)
        field = np.flatnonzero(xyz @ c >= np.cos(field_r * DEG))
        if len(field) < mol.n:
            return out
        P = project(xyz[field], c)
        grid = _grid_build(P, g)
        frame = np.array(tangent_frame(c))
        # Plane distance of the cell edge (stereographic stretches by 2 tan(θ/2) / θ).
        cell_plane = 2.0 * np.tan(cell_r * DEG / 2.0) / DEG
        for anchors, order, r_anchor in anchor_sets:
            # The anchor atom can sit up to smax * r_anchor from the figure's center.
            cand_i = np.flatnonzero(np.abs(P) <= cell_plane + smax * r_anchor)
            hyps = _hypotheses(Ts, anchors, order, P, grid, g, cand_i, frame, centers[near],
                               region_info, smin, smax, tol, max_rel, keep, skip)
            for (mirrored, T), (A_, B_) in zip(templates, hyps):
                if len(A_) == 0:
                    continue
                S, _, _, scores = _refine_batch(T, P, A_, B_, smin, smax, skip, 12, 6)
                ok = np.isfinite(scores)
                S, scores = S[ok], scores[ok]
                S = np.where(S >= 0, field[np.maximum(S, 0)], -1)
                for stars, score in zip(S.tolist(), scores.tolist()):
                    key = (mirrored, tuple(stars))
                    if score < out.get(key, np.inf):
                        out[key] = score
        return out

    # Tiles run on a thread pool; the numba kernels release the GIL, so this uses every core with
    # coarse, evenly balanced chunks of work.
    t0 = time.time()
    found = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, fut in enumerate(as_completed([pool.submit(search_tile, t) for t in tiles]), 1):
            for key, score in fut.result().items():
                if score < found.get(key, np.inf):
                    found[key] = score
            if verbose and (done % 25 == 0 or done == len(tiles)):
                print(f"\r  searched {done}/{len(tiles)} sky tiles, {len(found)} candidate figures "
                      f"({time.time() - t0:.0f}s)", end="", flush=True)
        if verbose:
            print()

        best = sorted(found.items(), key=lambda kv: kv[1])[: max(polish, 10 * top)]
        tmpl = dict(templates)
        jobs = [pool.submit(_polish, tmpl[mirrored], mirrored, np.array(stars), xyz, smin, smax, skip)
                for (mirrored, stars), _ in best]
        polished = {}
        for fut in jobs:
            for m in fut.result():
                key = (m.mirrored, tuple(m.stars))
                if key not in polished or m.score < polished[key].score:
                    polished[key] = m

    ranked = sorted(polished.values(), key=lambda m: m.score)
    results = []

    def distinct(m):
        return all(len(set(m.stars[m.used]) & set(r.stars[r.used])) <= 0.3 * mol.n
                   for r in results if r.n_skipped == m.n_skipped)

    for m in ranked:
        if len(results) < top and distinct(m):
            results.append(m)
    # Always show the best figure for every number of starless atoms, so they can be compared.
    for k in range(skip + 1):
        if not any(r.n_skipped == k for r in results):
            results += [m for m in ranked if m.n_skipped == k][:1]
    return sorted(results, key=lambda m: m.score)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_ra(deg, tenths=True):
    if not tenths:
        t = int(round(deg / 15.0 * 3600)) % (24 * 3600)
        h, rem = divmod(t, 3600)
        m, sec = divmod(rem, 60)
        return f"{h:02d}h {m:02d}m {sec:02d}s"
    t = int(round(deg / 15.0 * 36000)) % (24 * 36000)  # tenths of a second of time
    h, rem = divmod(t, 36000)
    m, sec = divmod(rem, 600)
    return f"{h:02d}h {m:02d}m {sec / 10:04.1f}s"


def format_dec(deg):
    sign = "-" if deg < 0 else "+"
    t = int(round(abs(deg) * 3600))
    d, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{sign}{d:02d}° {m:02d}′ {s:02d}″"


def vertices(mol, cat, m):
    """Chart-plane position of every atom: its star, or the ideal spot for starless atoms."""
    ideal = m.a * (np.conj(mol.z) if m.mirrored else mol.z) + m.b
    pos = ideal.copy()
    pos[m.used] = project(cat.xyz[m.stars[m.used]], m.center)
    return pos, ideal


def describe(mol, cat, m):
    ra, dec = xyz_to_radec(m.center)
    used = m.stars[m.used]
    cons = [c for c in cat.con[used] if c]
    con_counts = {c: cons.count(c) for c in dict.fromkeys(cons)}
    main_con = max(con_counts, key=con_counts.get) if con_counts else ""
    pos, ideal = vertices(mol, cat, m)
    offsets = np.abs(pos - ideal) * 60.0

    near = np.flatnonzero(cat.xyz @ m.center >= np.cos(30 * DEG))
    others = np.setdiff1d(near, used)
    inside = Delaunay(_xy(pos)).find_simplex(_xy(project(cat.xyz[others], m.center))) >= 0
    intruders = int(np.sum(inside & (cat.mag[others] < np.median(cat.mag[used]))))

    stars = []
    for i, s in enumerate(m.stars):
        entry = {"atom": f"{mol.symbols[i]}{i + 1}", "starless": bool(s < 0)}
        if s >= 0:
            entry.update(
                hip=int(cat.hip[s]),
                name=str(cat.name[s]),
                ra_deg=round(float(cat.ra[s]), 5),
                dec_deg=round(float(cat.dec[s]), 5),
                mag=round(float(cat.mag[s]), 2),
                offset_arcmin=round(float(offsets[i]), 2),
            )
        stars.append(entry)

    return {
        "molecule": mol.name,
        "smiles": mol.smiles,
        "ra_deg": round(float(ra), 5),
        "dec_deg": round(float(dec), 5),
        "ra": format_ra(ra),
        "dec": format_dec(dec),
        "ra_hms": format_ra(ra, tenths=False),
        "constellation": CONSTELLATIONS.get(main_con.lower(), main_con).replace("_", " "),
        "constellations": [CONSTELLATIONS.get(c.lower(), c).replace("_", " ") for c in con_counts],
        "distortion": round(float(m.score), 4),
        "rmsd_arcmin": round(float(m.rmsd_arcmin), 2),
        "bond_deg": round(float(m.bond_deg), 3),
        "mirrored": bool(m.mirrored),
        "starless_atoms": [s["atom"] for s in stars if s["starless"]],
        "brightest_mag": round(float(cat.mag[used].min()), 2),
        "faintest_mag": round(float(cat.mag[used].max()), 2),
        "brighter_intruders": intruders,
        "stars": stars,
    }


def print_summary(infos):
    print(f"\n{'#':>3}  {'distortion':>10}  {'RMSD':>6}  {'bond':>5}  {'starless':<10} {'RA':<14} {'Dec':<13} "
          f"{'mags':<9} {'mir':<3}  constellation")
    for rank, info in enumerate(infos, 1):
        print(f"{rank:>3}  {info['distortion']:>10.3f}  {info['rmsd_arcmin']:>5.1f}′  {info['bond_deg']:>4.2f}°  "
              f"{','.join(info['starless_atoms']) or '-':<10} {info['ra']:<14} {info['dec']:<13} "
              f"{info['brightest_mag']:.1f}–{info['faintest_mag']:.1f}  {'yes' if info['mirrored'] else '':<3}  "
              f"{info['constellation']}")


def print_report(info, rank):
    print(f"\n#{rank}  {info['ra']}  {info['dec']}  ({info['constellation']})")
    print(f"    distortion {info['distortion']:.3f} bond lengths (RMSD {info['rmsd_arcmin']:.1f}′), "
          f"bond {info['bond_deg']:.2f}°, mags {info['brightest_mag']:.1f}–{info['faintest_mag']:.1f}, "
          f"{info['brighter_intruders']} stars inside the figure brighter than its median star"
          + (", mirrored drawing" if info["mirrored"] else ""))
    print(f"    {'atom':<6}{'HIP':>8}  {'RA':>10} {'Dec':>10} {'mag':>5}  {'off':>6}  name")
    for s in info["stars"]:
        if s["starless"]:
            print(f"    {s['atom']:<6}{'-':>8}  (no star, drawn as a plain line corner)")
            continue
        print(f"    {s['atom']:<6}{s['hip']:>8}  {s['ra_deg']:>10.4f} {s['dec_deg']:>+10.4f} {s['mag']:>5.2f}"
              f"  {s['offset_arcmin']:>5.1f}′  {s['name']}")


# ---------------------------------------------------------------------------
# Real sky background (optional): any equirectangular all-sky image
# ---------------------------------------------------------------------------

# ICRS (J2000 equatorial) -> galactic rotation (Hipparcos definition).
ICRS_TO_GAL = np.array([
    [-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
    [+0.4941094278755837, -0.4448296299600112, +0.7469822444972189],
    [-0.8676661490190047, -0.1980763734312015, +0.4559837761750669],
])


@dataclass
class SkyImage:
    """An all-sky equirectangular image and how its pixels map onto the sky."""
    rgb: np.ndarray       # (h, w, 3) floats in 0..1
    diffuse: np.ndarray   # same image with the stars filtered out
    frame: str            # "equatorial" or "galactic"
    lon_sign: int         # +1: longitude grows to the right, -1: to the left
    lon_offset: float     # longitude (deg) at the left edge, before the sign is applied

    def sample(self, xyz, layer="diffuse"):
        """Bilinear sample of the image at unit vectors xyz (..., 3) (ICRS)."""
        from scipy.ndimage import map_coordinates

        v = xyz @ ICRS_TO_GAL.T if self.frame == "galactic" else xyz
        lon = np.arctan2(v[..., 1], v[..., 0]) / DEG
        lat = np.arcsin(np.clip(v[..., 2], -1, 1)) / DEG
        img = self.diffuse if layer == "diffuse" else self.rgb
        h, w = img.shape[:2]
        x = (((self.lon_sign * lon - self.lon_offset) / 360.0) % 1.0) * w - 0.5
        y = (90.0 - lat) / 180.0 * h - 0.5
        coords = [y.ravel(), x.ravel()]
        out = [map_coordinates(img[..., c], coords, order=1, mode="grid-wrap") for c in range(3)]
        return np.stack(out, axis=-1).reshape(xyz.shape[:-1] + (3,))


def load_sky_image(path, cat, max_width=4096):
    """Load an equirectangular all-sky map (jpg/png/tif) and work out its orientation by checking
    which convention puts bright pixels where the catalog's bright stars are."""
    from PIL import Image
    from scipy.ndimage import median_filter

    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(path)
    if im.format == "JPEG":
        im.draft("RGB", (max_width, max_width // 2))
    im = im.convert("RGB")
    if im.width > max_width:
        im = im.resize((max_width, max_width * im.height // im.width), Image.LANCZOS)
    rgb = np.asarray(im, dtype=np.float32) / 255.0
    # Stars are a few pixels wide at this size; a median filter keeps only the diffuse glow.
    diffuse = np.stack([median_filter(rgb[..., c], size=7, mode="wrap") for c in range(3)], axis=-1)

    bright = cat.xyz[cat.mag < 4.0]
    rng = np.random.default_rng(0)
    rand = rng.normal(size=(4000, 3))
    rand /= np.linalg.norm(rand, axis=1)[:, None]
    best, best_ratio = None, 0.0
    for frame in ("equatorial", "galactic"):
        for sign in (1, -1):
            for offset in (0.0, 180.0):
                sky = SkyImage(rgb, diffuse, frame, sign, offset)
                lum_star = sky.sample(bright, "rgb").mean()
                lum_rand = sky.sample(rand, "rgb").mean()
                ratio = lum_star / max(lum_rand, 1e-6)
                if ratio > best_ratio:
                    best, best_ratio = sky, ratio
    if best_ratio < 1.5:
        raise SystemExit(f"Couldn't match {path} to the sky (best contrast {best_ratio:.2f}). It needs to be "
                         "an equirectangular all-sky map in equatorial or galactic coordinates.")
    print(f"Sky image: {best.frame} coordinates, longitude {'right' if best.lon_sign > 0 else 'left'}ward "
          f"from {best.lon_offset:.0f}° (bright-star contrast {best_ratio:.1f}x)")
    return best


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def _draw_bonds(ax, mol, pos, color, lw, inner_lw, alpha=1.0, zorder=3, gap=None):
    """Skeletal-formula bonds between the points `pos` (complex): double bonds in rings get an
    inner parallel line toward the ring's center, other double/triple bonds parallel lines.
    gap[i] leaves that much space (data units) between atom i and its bonds."""
    def line(p, q, width):
        ax.plot([p.real, q.real], [p.imag, q.imag], color=color, lw=width, alpha=alpha,
                solid_capstyle="round", zorder=zorder)

    for i, j, order in mol.bonds:
        p, q = pos[i], pos[j]
        if gap is not None:
            u = (q - p) / abs(q - p)
            p, q = p + gap[i] * u, q - gap[j] * u
        d = q - p
        normal = 1j * d / abs(d)
        if order == 1:
            line(p, q, lw)
            continue
        # For a fused bond, draw the inner line inside the ring with the most double bonds
        # (the aromatic one), then prefer the smaller ring.
        ring = min((rg for rg in mol.rings if i in rg and j in rg), default=None,
                   key=lambda rg: (-sum(o == 2 and x in rg and y in rg for x, y, o in mol.bonds), len(rg)))
        if order == 2 and ring is not None:
            inward = pos[list(ring)].mean() - (p + q) / 2
            normal *= np.sign((np.conj(normal) * inward).real) or 1
            off = 0.2 * abs(d) * normal
            line(p, q, lw)
            line(p + 0.17 * d + off, q - 0.17 * d + off, inner_lw)
        else:
            for k in ((-0.09, 0.09) if order == 2 else (-0.15, 0.0, 0.15)):
                line(p + k * abs(d) * normal, q + k * abs(d) * normal, inner_lw)


def _pick_font():
    from matplotlib import font_manager

    names = {f.name for f in font_manager.fontManager.ttflist}
    return next((f for f in ("Inter", "Inter Display", "DejaVu Sans") if f in names), "sans-serif")


def _spaced(text, gap=" "):
    """Letter-spaced caps, poster style."""
    caps = "".join(c.upper() if c.isascii() else c for c in text)  # keep β a β, not a Β
    return gap.join(caps).replace(gap + " " + gap, "   ")


def _render_sky(pos, mags, temps, boost, featured, size_px, lo, hi, mag_ref, haze_pos, haze_mag,
                backdrop=None):
    """Additive-light render of stars into a float RGB image.

    pos/mags/temps: stars to draw (complex chart-plane positions, magnitudes, temperatures)
    boost:          per-star brightness factor (the figure's stars are drawn brighter)
    featured:       stars of the figure; they get a minimum size and a wider colored halo
    haze_pos/mag:   all stars in view, smoothed into a faint Milky Way glow
    backdrop:       optional (H, W, 3) diffuse light from a real sky image, used instead of the haze"""
    from scipy.ndimage import gaussian_filter

    W, H = size_px
    img = np.zeros((H, W, 3))
    sx = W / (hi[0] - lo[0])
    px = (pos.real - lo[0]) * sx
    py = (hi[1] - pos.imag) * sx
    u = W / 2400.0  # resolution-independent pixel unit

    # Milky Way haze: flux-weighted star density blurred to a gentle gradient. Only the excess over
    # the frame's sparsest parts glows, so it traces where the sky is genuinely richer.
    hx = ((haze_pos.real - lo[0]) * sx).astype(int)
    hy = ((hi[1] - haze_pos.imag) * sx).astype(int)
    ok = (hx >= 0) & (hx < W) & (hy >= 0) & (hy < H)
    dens = np.zeros((H, W))
    np.add.at(dens, (hy[ok], hx[ok]), (10 ** (-0.4 * (haze_mag[ok] - mag_ref))) ** 0.5)
    haze = gaussian_filter(dens, 170 * u)
    p10, p99 = np.percentile(haze, [10, 99.5])
    if backdrop is not None:
        # Real sky: keep its colors, lift it so the Milky Way's structure shows, floor at black.
        lum = backdrop.mean(axis=-1)
        b10, b99 = np.percentile(lum, [5, 99.7])
        img += 0.55 * np.clip((backdrop - b10) / max(b99 - b10, 1e-6), 0, None) ** 1.2
    elif p99 > p10:
        haze = np.clip((haze - p10) / (p99 - p10), 0, 1) ** 1.4
        img += 0.22 * haze[..., None] * np.array([0.38, 0.42, 0.95])

    colors = temp_to_rgb(temps, saturation=1.7)
    order = np.argsort(-mags)  # faint first; addition is order-independent but keeps it tidy
    for i in order:
        flux = 10 ** (-0.4 * (mags[i] - mag_ref)) * boost[i]
        level = np.log10(1 + flux)  # perceptual brightness, ~0 for the faintest stars
        if featured[i]:
            level = max(level, 1.3)
        col = colors[i]
        core_s = u * (0.9 + 1.1 * level)
        halo_s = core_s * (6.5 if featured[i] else 4.5)
        spike = u * 38 * level ** 1.6 if level > 0.6 else 0.0
        R = int(max(4 * halo_s, spike * 1.2, 3))
        x0, y0 = int(px[i]), int(py[i])
        xa, xb = max(x0 - R, 0), min(x0 + R + 1, W)
        ya, yb = max(y0 - R, 0), min(y0 + R + 1, H)
        if xa >= xb or ya >= yb:
            continue
        dx = np.arange(xa, xb) - px[i]
        dy = np.arange(ya, yb) - py[i]
        gx_c, gy_c = np.exp(-dx ** 2 / (2 * core_s ** 2)), np.exp(-dy ** 2 / (2 * core_s ** 2))
        gx_h, gy_h = np.exp(-dx ** 2 / (2 * halo_s ** 2)), np.exp(-dy ** 2 / (2 * halo_s ** 2))
        # Faint stars get dimmer, not just smaller, so the deep field reads as fine dust.
        dimming = 1.0 if featured[i] else min(1.0, 1.5 * np.sqrt(flux))
        core = np.outer(gy_c, gx_c) * (1.6 + 1.2 * level) * dimming
        halo = np.outer(gy_h, gx_h) * (0.10 + 0.22 * level) * (1.6 if featured[i] else dimming)
        patch = core[..., None] * (0.55 + 0.45 * col) + halo[..., None] * col
        if spike > 0:
            # Diffraction spikes: a thin cross, fading exponentially along each arm.
            w = u * 0.7
            arm_x = np.outer(np.exp(-dy ** 2 / (2 * w ** 2)), np.exp(-np.abs(dx) / (spike * 0.35)))
            arm_y = np.outer(np.exp(-np.abs(dy) / (spike * 0.35)), np.exp(-dx ** 2 / (2 * w ** 2)))
            patch += ((arm_x + arm_y) * (0.35 + 0.25 * level))[..., None] * (0.5 + 0.5 * col)
        img[ya:yb, xa:xb] += patch

    # Soft vignette, then a filmic curve so bright cores burn to white but halos keep their color.
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot((xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2))
    img *= (1 - 0.35 * np.clip(r - 0.4, 0, None) ** 1.5)[..., None]
    base = np.array([0.020, 0.024, 0.043])  # night-sky blue-black, same as the frame
    return np.clip(base + 1 - np.exp(-1.35 * img), 0, 1)


def _sky_scene(mol, cat, m, rot, lo, hi, W, H, sky_image=None, sky_gain=0.55):
    """Render the sky for the chart-plane box lo..hi at W x H pixels. Returns the image, the atom
    positions and, per atom, the gap (data units) bonds should leave around its star."""
    verts = vertices(mol, cat, m)[0] * rot
    r = np.hypot(*(hi - lo)) / 2 + np.abs(((lo + hi) / 2) @ [1, 1j]) + 1.0
    near = np.flatnonzero(cat.xyz @ m.center >= np.cos(min(r, 85) * DEG))
    w = project(cat.xyz[near], m.center) * rot
    inbox = (w.real >= lo[0]) & (w.real <= hi[0]) & (w.imag >= lo[1]) & (w.imag <= hi[1])
    near, w = near[inbox], w[inbox]

    used = m.stars[m.used]
    featured = np.isin(near, used)
    boost = np.where(featured, 5.0, 0.35)  # the figure's stars stand out
    mag_ref = float(np.max(cat.mag[used]))

    backdrop = None
    if sky_image is not None:
        # Sample the real sky at quarter resolution (it's diffuse light) and scale up.
        from scipy.ndimage import zoom

        qw, qh = W // 4, H // 4
        gx = lo[0] + (np.arange(qw) + 0.5) / qw * (hi[0] - lo[0])
        gy = hi[1] - (np.arange(qh) + 0.5) / qh * (hi[1] - lo[1])
        plane = (gx[None, :] + 1j * gy[:, None]) / rot  # undo the upright rotation
        small = sky_image.sample(unproject(plane, m.center))
        backdrop = sky_gain / 0.55 * zoom(small, (H / qh, W / qw, 1), order=1)[:H, :W]
    sky = _render_sky(w, cat.mag[near], cat.temp[near], boost, featured, (W, H), lo, hi, mag_ref,
                      w, cat.mag[near], backdrop)

    # Bonds stop just short of each star so the line doesn't cut through its core (corners of
    # starless atoms still join up). The gap follows the star's rendered core size.
    star_mag = np.full(mol.n, mag_ref)
    star_mag[m.used] = cat.mag[m.stars[m.used]]
    level = np.maximum(np.log10(1 + 10 ** (-0.4 * (star_mag - mag_ref)) * 5.0), 1.3)
    gap_px = (W / 2400.0) * (5.0 + 4.0 * level)
    gap = np.where(m.used, gap_px * (hi[0] - lo[0]) / W, 0.0)
    return sky, verts, gap, near, w


def plot_chart(mol, cat, m, info, path, north_up=False, labels=False, star_names=False, sky_image=None,
               sky_gain=0.55):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    font = _pick_font()
    ink, dim = "#e8eefc", "#8d97b5"
    rot = 1.0 if north_up else np.conj(m.a) / abs(m.a)  # turn the molecule upright on the page

    s = m.bond_deg
    v0 = vertices(mol, cat, m)[0] * rot
    lo = np.array([v0.real.min(), v0.imag.min()]) - 1.6 * s
    hi = np.array([v0.real.max(), v0.imag.max()]) + 1.6 * s
    W = 3600
    H = int(W * (hi[1] - lo[1]) / (hi[0] - lo[0]))
    sky, verts, gap, near, w = _sky_scene(mol, cat, m, rot, lo, hi, W, H, sky_image, sky_gain)

    fig_w = 12.0
    fig = plt.figure(figsize=(fig_w, fig_w * H / W + 1.6), facecolor="#05060b")  # = base color of the sky render
    top = 0.8 / (fig_w * H / W + 1.6)
    ax = fig.add_axes([0, 0.8 / (fig_w * H / W + 1.6), 1, 1 - 2 * top])
    ax.imshow(sky, extent=(lo[0], hi[0], lo[1], hi[1]), origin="upper", interpolation="bilinear")
    _draw_bonds(ax, mol, verts, "#d6e2ff", lw=1.0, inner_lw=0.8, alpha=0.7, gap=gap)

    if labels:
        for i in range(mol.n):
            if mol.symbols[i] != "C":
                ax.annotate(mol.symbols[i], (verts[i].real, verts[i].imag), xytext=(9, 9),
                            textcoords="offset points", color=dim, fontsize=9, fontfamily=font)
    if star_names:
        for idx, p in zip(near, w):
            if cat.name[idx]:
                ax.annotate(cat.name[idx], (p.real, p.imag), xytext=(8, -12), textcoords="offset points",
                            color=dim, fontsize=7.5, fontfamily=font)

    # Compass: where north and east really point on this chart.
    base = complex(hi[0] - 0.7 * s, lo[1] + 0.9 * s)
    for label, direction in (("N", 1j), ("E", -1.0)):
        tip = base + 0.42 * s * direction * rot
        ax.annotate("", xy=(tip.real, tip.imag), xytext=(base.real, base.imag),
                    arrowprops=dict(arrowstyle="-|>", color=dim, lw=0.8, mutation_scale=8))
        txt = base + 0.6 * s * direction * rot
        ax.text(txt.real, txt.imag, label, color=dim, fontsize=7.5, ha="center", va="center", fontfamily=font)

    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.axis("off")

    fig.text(0.5, 1 - 0.42 * top, _spaced(mol.name), color=ink, fontsize=17, ha="center", va="center",
             fontfamily=font, fontweight="light")
    fig.text(0.5, 0.62 * top, f"{info['ra']}     {info['dec']}", color=ink, fontsize=10.5,
             ha="center", va="center", fontfamily=font)
    fig.text(0.5, 0.28 * top, _spaced(f"{info['constellation']}  ·  J2000", " "), color=dim,
             fontsize=7.5, ha="center", va="center", fontfamily=font)

    fig.savefig(path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_wallpaper(mol, cat, m, info, path, size=(3840, 2160), north_up=False, sky_image=None,
                   sky_gain=0.55):
    """Full-bleed 16:9 wallpaper: sky to the edges, the molecule a little right of center at about
    half the width, a faint glow on the bonds and one quiet line of text in the corner."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    font = _pick_font()
    W, H = size
    rot = 1.0 if north_up else np.conj(m.a) / abs(m.a)
    v0 = vertices(mol, cat, m)[0] * rot
    fig_lo = np.array([v0.real.min(), v0.imag.min()])
    fig_hi = np.array([v0.real.max(), v0.imag.max()])
    # Degrees per pixel so the figure spans ~52% of the width and at most ~62% of the height.
    dpp = max((fig_hi[0] - fig_lo[0]) / (0.52 * W), (fig_hi[1] - fig_lo[1]) / (0.62 * H))
    cx, cy = (fig_lo + fig_hi) / 2
    lo = np.array([cx - 0.57 * W * dpp, cy - 0.5 * H * dpp])
    hi = lo + np.array([W, H]) * dpp
    sky, verts, gap, _, _ = _sky_scene(mol, cat, m, rot, lo, hi, W, H, sky_image, sky_gain)
    # Quiet the bottom-left corner a little so the caption sits on calm sky.
    yy, xx = np.mgrid[0:H, 0:W]
    shade = np.exp(-(((xx - 0.0 * W) / (0.32 * W)) ** 2 + ((yy - H) / (0.30 * H)) ** 2))
    sky = sky * (1 - 0.55 * shade)[..., None]

    fig = plt.figure(figsize=(W / 300, H / 300), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(sky, extent=(lo[0], hi[0], lo[1], hi[1]), origin="upper", interpolation="bilinear")
    # Glow first (wide, faint), then the line itself.
    _draw_bonds(ax, mol, verts, "#9fb8ff", lw=4.0, inner_lw=3.0, alpha=0.08, gap=gap)
    _draw_bonds(ax, mol, verts, "#e2eaff", lw=0.9, inner_lw=0.7, alpha=0.75, gap=gap)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.axis("off")

    fig.text(0.04, 0.095, _spaced(mol.name), color="#e8eefc", fontsize=12, ha="left", va="bottom",
             fontfamily=font, fontweight="light")
    fig.text(0.04, 0.062, f"{info['ra']}   {info['dec']}   ·   {info['constellation']}",
             color="#a3adc9", fontsize=7, ha="left", va="bottom", fontfamily=font)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _segment_distance(pts, a, b):
    """Distance from each point in pts to the segment a-b (all complex)."""
    d = b - a
    t = np.clip(((pts - a) * np.conj(d)).real / abs(d) ** 2, 0.0, 1.0)
    return np.abs(pts - (a + t * d))


def plot_tattoo(mol, cat, m, info, path_base, north_up=False, n_background=8, text=True):
    """Stencil-ready line art: black bonds, star dots in four sizes by magnitude, a few of the
    brightest nearby stars and the coordinates underneath. Writes <path_base>.svg (transparent,
    text kept as editable text) and <path_base>.png (white, 600 dpi)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    ink = "#000000"
    rot = 1.0 if north_up else np.conj(m.a) / abs(m.a)
    pos = vertices(mol, cat, m)[0] * rot
    s = m.bond_deg
    lo = np.array([pos.real.min(), pos.imag.min()]) - 0.9 * s
    hi = np.array([pos.real.max(), pos.imag.max()]) + 0.9 * s

    def radius(mag):
        # Four size classes keep the dots readable on skin; brighter stars get bigger dots.
        return s * np.select([mag < 4.0, mag < 5.0, mag < 6.0], [0.15, 0.12, 0.095], 0.075)

    # The brightest other stars in the frame, kept clear of the lines and dots so nothing merges.
    used = m.stars[m.used]
    near = np.flatnonzero(cat.xyz @ m.center >= np.cos(min(np.abs(hi - lo).max(), 80) * DEG))
    near = np.setdiff1d(near, used)
    w = project(cat.xyz[near], m.center) * rot
    ok = (w.real >= lo[0]) & (w.real <= hi[0]) & (w.imag >= lo[1]) & (w.imag <= hi[1])
    near, w = near[ok], w[ok]
    clear = np.full(len(w), np.inf)
    for i, j, _ in mol.bonds:
        clear = np.minimum(clear, _segment_distance(w, pos[i], pos[j]))
    edge = np.minimum.reduce([w.real - lo[0], hi[0] - w.real, w.imag - lo[1], hi[1] - w.imag])
    keep = (clear > 0.4 * s) & (edge > 0.15 * s)
    near, w = near[keep], w[keep]
    pick = np.argsort(cat.mag[near])[:n_background]
    bg_w, bg_mag = w[pick], cat.mag[near[pick]]

    width = 8.0
    height = width * (hi[1] - lo[1]) / (hi[0] - lo[0])
    fig, ax = plt.subplots(figsize=(width, height + (0.6 if text else 0)))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    _draw_bonds(ax, mol, pos, ink, lw=1.5, inner_lw=1.2)
    for i in np.flatnonzero(m.used):
        ax.add_patch(Circle((pos[i].real, pos[i].imag), radius(cat.mag[m.stars[i]]), color=ink, lw=0, zorder=4))
    for p, mag in zip(bg_w, bg_mag):
        # Background stars stay small so the molecule reads first.
        ax.add_patch(Circle((p.real, p.imag), 0.45 * radius(mag), color=ink, lw=0, zorder=4))

    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.axis("off")
    if text:
        dec = info["dec"].replace("-", "−")
        ax.text(0.5, -0.03, f"{info['ra_hms']}   ·   {dec}", transform=ax.transAxes, color=ink, fontsize=13,
                ha="center", va="top", fontfamily="DejaVu Sans")

    with plt.rc_context({"svg.fonttype": "none"}):
        fig.savefig(f"{path_base}.svg", transparent=True, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(f"{path_base}.png", dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description="Find a molecule's skeletal formula among real stars.")
    p.add_argument("molecule", help=f"built-in name ({', '.join(KNOWN_MOLECULES)}), SMILES, or a PubChem name")
    p.add_argument("--title", help="name to print on the chart")
    p.add_argument("--mag", type=float, default=6.5,
                   help="faintest star magnitude allowed in the figure (default 6.5, roughly naked eye)")
    p.add_argument("--bond", type=float, nargs=2, default=(1.0, 3.0), metavar=("MIN", "MAX"),
                   help="allowed bond length on the sky in degrees (default 1 3)")
    p.add_argument("--region", type=float, nargs=3, metavar=("RA", "DEC", "RADIUS"),
                   help="only search for figures centered within RADIUS degrees of RA/DEC (default: whole sky)")
    p.add_argument("--no-mirror", action="store_true", help="don't allow the mirror image of the drawing")
    p.add_argument("--skip", type=int, default=1,
                   help="let up to this many atoms go without a star; they become plain line corners (default 1)")
    p.add_argument("--top", type=int, default=10, help="how many candidates to list, sorted by distortion")
    p.add_argument("--charts", type=int, default=5, help="how many of the best candidates to chart in detail")
    p.add_argument("--out", default="out", help="output directory (default ./out)")
    p.add_argument("--north-up", action="store_true", help="draw charts north-up instead of molecule-upright")
    p.add_argument("--labels", action="store_true", help="label heteroatoms with their element")
    p.add_argument("--star-names", action="store_true", help="label named background stars")
    p.add_argument("--tattoo", action="store_true",
                   help="also write stencil-ready line art (SVG + PNG) for every charted candidate")
    p.add_argument("--tattoo-stars", type=int, default=8, metavar="N",
                   help="how many of the brightest nearby stars to add to the tattoo art (default 8)")
    p.add_argument("--no-tattoo-text", action="store_true", help="leave the coordinates off the tattoo art")
    p.add_argument("--wallpaper", action="store_true", help="also write a 3840x2160 wallpaper per charted candidate")
    p.add_argument("--sky-gain", type=float, default=0.55, help="brightness of the --sky-image backdrop (default 0.55)")
    p.add_argument("--workers", type=int, help="threads to use (default: all cores)")
    p.add_argument("--sky-image", metavar="PATH",
                   help="equirectangular all-sky image (e.g. NASA SVS Deep Star Maps) for a real Milky Way "
                        "behind the charts; its orientation is detected automatically")
    p.add_argument("--catalog", help="path to a local hygdata_v41.csv instead of downloading it")
    args = p.parse_args(argv)

    mol = load_molecule(args.molecule, args.title)
    print(f"{mol.name}: {mol.n} heavy atoms, {len(mol.bonds)} bonds")
    full = load_catalog(args.mag + 2.5, args.catalog)  # fainter stars only as chart background
    sel = np.flatnonzero(full.mag <= args.mag)
    cat = full.subset(sel)
    print(f"{len(cat.hip)} stars brighter than mag {args.mag}")

    matches = find_asterisms(mol, cat, bond=tuple(args.bond), region=args.region,
                             mirror=not args.no_mirror, skip=args.skip, top=args.top, workers=args.workers)
    if not matches:
        raise SystemExit("No match found. Try a fainter --mag, a wider --bond range or a bigger --region.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", mol.name.lower()).strip("-")[:40] or "molecule"
    sky_image = load_sky_image(args.sky_image, full) if args.sky_image else None
    report = []
    for m in matches:
        # Index into the full catalog, which also holds the fainter background stars.
        m.stars = np.where(m.used, sel[np.maximum(m.stars, 0)], -1)
        report.append(describe(mol, full, m))
    print_summary(report)
    for rank, (m, info) in enumerate(zip(matches, report), 1):
        info["rank"] = rank
        if rank <= args.charts:
            print_report(info, rank)
            png = out / f"{slug}_{rank}.png"
            plot_chart(mol, full, m, info, png, args.north_up, args.labels, args.star_names, sky_image,
                       args.sky_gain)
            if args.wallpaper:
                wp = out / f"{slug}_{rank}_wallpaper.png"
                plot_wallpaper(mol, full, m, info, wp, north_up=args.north_up, sky_image=sky_image,
                               sky_gain=args.sky_gain)
                info["wallpaper"] = str(wp)
            info["chart"] = str(png)
            if args.tattoo:
                base = out / f"{slug}_{rank}_tattoo"
                plot_tattoo(mol, full, m, info, base, args.north_up, args.tattoo_stars, not args.no_tattoo_text)
                info["tattoo"] = [f"{base}.svg", f"{base}.png"]
    (out / f"{slug}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nCharts and {slug}.json written to {out}/")


if __name__ == "__main__":
    main()
