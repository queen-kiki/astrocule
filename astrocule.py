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
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import Delaunay, cKDTree

DEG = np.pi / 180.0


# ---------------------------------------------------------------------------
# Molecules
# ---------------------------------------------------------------------------

KNOWN_MOLECULES = {
    "estradiol": ("17β-Estradiol", "C[C@]12CC[C@H]3[C@H]([C@@H]1CC[C@@H]2O)CCC4=C3C=CC(=C4)O"),
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

    def __post_init__(self):
        self.xyz = radec_to_xyz(self.ra, self.dec)

    def subset(self, mask):
        return Catalog(self.hip[mask], self.ra[mask], self.dec[mask], self.mag[mask], self.name[mask], self.con[mask])


def load_catalog(mag_limit, csv_path=None):
    """All Hipparcos stars brighter than mag_limit. Downloads and caches HYG v4.1 on first use."""
    npz = CACHE_DIR / "hyg_v41_hip.npz"
    if csv_path is None and npz.exists():
        d = np.load(npz)
    else:
        if csv_path is None:
            csv_path = CACHE_DIR / "hygdata_v41.csv"
            if not csv_path.exists():
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                print(f"Downloading HYG star catalog to {csv_path} ...")
                urllib.request.urlretrieve(HYG_URL, csv_path)
        cols = {k: [] for k in ("hip", "ra", "dec", "mag", "name", "con")}
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
        d = {k: np.array(v) for k, v in cols.items()}
        if csv_path == CACHE_DIR / "hygdata_v41.csv":
            np.savez(npz, **d)
    cat = Catalog(d["hip"], d["ra"], d["dec"], d["mag"], d["name"], d["con"])
    return cat.subset(cat.mag <= mag_limit)


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


def _xy(w):
    return np.column_stack([w.real, w.imag])


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


def refine(T, P, tree, a, b, smin, smax, skip=0, forced=(), iters=12, k=6):
    """Alternate one-to-one star assignment (Hungarian) and similarity refit until stable.

    The atoms in `forced`, plus the `skip` worst-fitting others, go without a star: they are left
    out of the assignment and the fit and just become line corners on the chart (trimmed ICP).
    Returns (stars, a, b, score) with stars[i] = -1 for starless atoms."""
    n = len(T)
    forced = list(forced)
    n_drop = len(forced) + skip
    k = min(k, len(P))
    state = None
    for _ in range(iters):
        pred = a * T + b
        _, nn = tree.query(_xy(pred), k=k)
        cand = np.unique(nn)
        if len(cand) < n:
            return None
        cost = np.abs(pred[:, None] - P[cand][None, :]) ** 2
        use = np.ones(n, dtype=bool)
        if n_drop:
            _, cols = linear_sum_assignment(cost)
            resid = cost[np.arange(n), cols]
            resid[forced] = np.inf
            use[np.argsort(-resid)[:n_drop]] = False
        _, cols = linear_sum_assignment(cost[use])
        stars = np.full(n, -1)
        stars[use] = cand[cols]
        if state is not None and np.array_equal(stars, state):
            break
        state = stars
        a, b = fit_similarity(T[use], P[stars[use]], smin, smax)
    use = stars >= 0
    score = np.sqrt(np.mean(np.abs(a * T[use] + b - P[stars[use]]) ** 2)) / abs(a)
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


def _hypotheses(T, anchors, order, P, tree, cand_i, in_cell, smin, smax, tol, max_rel, keep, skip):
    """Every star pair (i, j) that could host the anchor atoms defines one similarity
    transform. Score all of them atom by atom against the nearest stars, pruning early.
    Up to `skip` atoms may miss (no star within tol) without killing the hypothesis."""
    ia, ib = anchors
    D = abs(T[ib] - T[ia])
    nbrs = tree.query_ball_point(_xy(P[cand_i]), r=smax * D)
    counts = np.array([len(x) for x in nbrs])
    if counts.sum() == 0:
        return []
    ii = np.repeat(cand_i, counts)
    jj = np.concatenate([np.asarray(x, dtype=int) for x in nbrs])
    ok = np.abs(P[jj] - P[ii]) >= smin * D
    ii, jj = ii[ok], jj[ok]

    a = (P[jj] - P[ii]) / (T[ib] - T[ia])
    b = P[ii] - a * T[ia]  # template centroid is 0, so b is where the figure's center lands
    owned = in_cell(b)
    a, b = a[owned], b[owned]
    if len(a) == 0:
        return []

    n = len(T) - skip
    sse = np.zeros(len(a))
    misses = np.zeros(len(a), dtype=int)
    alive = np.arange(len(a))
    for k in order:
        d, _ = tree.query(_xy(a[alive] * T[k] + b[alive]), workers=-1)
        rel2 = (d / np.abs(a[alive])) ** 2
        miss = rel2 > tol ** 2
        misses[alive] += miss
        sse[alive] += np.where(miss, 0.0, rel2)
        alive = alive[(misses[alive] <= skip) & (sse[alive] <= n * max_rel ** 2)]
        if len(alive) == 0:
            return []
    best = alive[np.argsort(sse[alive])[:keep]]
    return list(zip(a[best], b[best]))


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
    tree = cKDTree(_xy(P))
    keep = stars >= 0
    a, b = fit_similarity(T[keep], P[np.searchsorted(field, stars[keep])], smin, smax)

    out = []
    for k in range(skip + 1):
        tries = [((), 0)] if k == 0 else [((i,), k - 1) for i in range(len(T))]
        best = None
        for forced, extra in tries:
            res = refine(T, P, tree, a, b, smin, smax, skip=extra, forced=forced)
            if res is not None and (best is None or res[3] < best[3]):
                best = res
        if best is not None:
            s, a2, b2, score = best
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
                   tol=0.6, max_rel=0.4, keep=40, tile_deg=6.0, verbose=True):
    """
    Search the sky for the star sets that best trace mol's skeleton.

    bond:    allowed bond length on the sky in degrees (min, max)
    region:  optional (ra_deg, dec_deg, radius_deg) to restrict where the figure's center may lie
    mirror:  also allow the mirror image of the 2D drawing
    skip:    allow up to this many atoms without a star (drawn as plain line corners)
    Returns up to `top` matches sorted by distortion, at least one per number of starless atoms,
    each using mostly different stars from the others with the same number of starless atoms.
    """
    smin, smax = bond
    if len(cat.hip) < mol.n:
        raise SystemExit(f"Only {len(cat.hip)} stars available, need at least {mol.n}.")
    if not 0 <= skip <= mol.n - 3:
        raise SystemExit(f"--skip must be between 0 and {mol.n - 3} for this molecule.")
    templates = [(False, mol.z)] + ([(True, np.conj(mol.z))] if mirror else [])

    anchor_sets = []
    for anchors in _anchor_pairs(mol.z, skip + 1):
        order = [k for k in np.argsort(-np.abs(mol.z)) if k not in anchors]
        anchor_sets.append((anchors, order, np.abs(mol.z[list(anchors)]).max()))
    r_fig = np.abs(mol.z).max()

    xyz = cat.xyz
    if region is not None:
        ra0, dec0, rad = region
        centers = radec_to_xyz(ra0, dec0)[None, :]
        cell_r = rad
    else:
        n_tiles = int(np.ceil(2.0 / (1.0 - np.cos(tile_deg * DEG))))
        centers = fibonacci_sphere(n_tiles)
        cell_r = 2.0 * np.arccos(1.0 - 2.0 / n_tiles) / DEG  # generous bound on cell radius

    t0 = time.time()
    found = {}
    for t, c in enumerate(centers):
        if region is not None:
            def in_cell(w):
                return np.abs(w) <= cell_r
        else:
            near = np.flatnonzero(centers @ c >= np.cos(3 * cell_r * DEG))

            def in_cell(w, c=c, near=near):
                v = unproject(w, c)
                return (v @ centers[near].T).max(axis=1) <= v @ c + 1e-12

        field_r = min(cell_r + smax * r_fig + smax, 80.0)
        field = np.flatnonzero(xyz @ c >= np.cos(field_r * DEG))
        if len(field) < mol.n:
            continue
        P = project(xyz[field], c)
        tree = cKDTree(_xy(P))

        for anchors, order, r_anchor in anchor_sets:
            cand_i = np.flatnonzero(np.abs(P) <= cell_r + smax * r_anchor)
            for mirrored, T in templates:
                hyps = _hypotheses(T, anchors, order, P, tree, cand_i, in_cell,
                                   smin, smax, tol, max_rel, keep, skip)
                for a, b in hyps:
                    res = refine(T, P, tree, a, b, smin, smax, skip=skip)
                    if res is None:
                        continue
                    stars, _, _, score = res
                    key = (mirrored, tuple(np.where(stars >= 0, field[np.maximum(stars, 0)], -1)))
                    if score < found.get(key, np.inf):
                        found[key] = score
        if verbose and (t % 25 == 0 or t == len(centers) - 1):
            print(f"\r  searched {t + 1}/{len(centers)} sky tiles, {len(found)} candidate figures "
                  f"({time.time() - t0:.0f}s)", end="", flush=True)
    if verbose:
        print()

    best = sorted(found.items(), key=lambda kv: kv[1])[: max(50, 10 * top)]
    tmpl = dict(templates)
    polished = {}
    for (mirrored, stars), _ in best:
        for m in _polish(tmpl[mirrored], mirrored, np.array(stars), xyz, smin, smax, skip):
            key = (mirrored, tuple(m.stars))
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

def format_ra(deg):
    t = int(round(deg / 15.0 * 36000)) % (24 * 36000)  # tenths of a second of time
    h, rem = divmod(t, 36000)
    m, s = divmod(rem, 600)
    return f"{h:02d}h {m:02d}m {s / 10:04.1f}s"


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
# Chart
# ---------------------------------------------------------------------------

def plot_chart(mol, cat, m, info, path, north_up=False, labels=False, star_names=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bg_col, line_col, glow_col = "#0b0e14", "#e2f1ff", "#80d8ff"
    rot = 1.0 if north_up else np.conj(m.a) / abs(m.a)  # turn the molecule upright on the page

    stars = vertices(mol, cat, m)[0] * rot
    s = m.bond_deg
    lo = np.array([stars.real.min(), stars.imag.min()]) - 1.6 * s
    hi = np.array([stars.real.max(), stars.imag.max()]) + 1.6 * s

    r = np.abs(hi - lo).max()
    near = np.flatnonzero(cat.xyz @ m.center >= np.cos(min(r, 80) * DEG))
    bg = project(cat.xyz[near], m.center) * rot
    inbox = (bg.real >= lo[0]) & (bg.real <= hi[0]) & (bg.imag >= lo[1]) & (bg.imag <= hi[1])
    near, bg = near[inbox], bg[inbox]

    def size(mag):
        return 30.0 * 10 ** (-0.32 * (mag - 5.0))

    fig, ax = plt.subplots(figsize=(10, 10 * (hi[1] - lo[1]) / (hi[0] - lo[0])), facecolor=bg_col)
    ax.set_facecolor(bg_col)
    ax.scatter(bg.real, bg.imag, s=size(cat.mag[near]), c="#cfd8dc", alpha=0.35, edgecolors="none", zorder=1)

    if star_names:
        for idx, w in zip(near, bg):
            if cat.name[idx] and idx not in m.stars[m.used]:
                ax.annotate(cat.name[idx], (w.real, w.imag), xytext=(5, -9), textcoords="offset points",
                            color="#8a96a3", fontsize=7, fontfamily="monospace", zorder=2)

    def line(p, q, **kw):
        ax.plot([p.real, q.real], [p.imag, q.imag], color=line_col, solid_capstyle="round", **kw)

    for i, j, order in mol.bonds:
        p, q = stars[i], stars[j]
        d = q - p
        if order == 1:
            line(p, q, lw=1.6, alpha=0.9, zorder=3)
            continue
        # For a fused bond, draw the inner line inside the ring with the most double bonds
        # (the aromatic one), then prefer the smaller ring.
        ring = min((rg for rg in mol.rings if i in rg and j in rg), default=None,
                   key=lambda rg: (-sum(o == 2 and x in rg and y in rg for x, y, o in mol.bonds), len(rg)))
        if order == 2 and ring is not None:
            # Inner parallel line toward the ring's center, like a textbook skeletal formula.
            inward = stars[list(ring)].mean() - (p + q) / 2
            normal = 1j * d / abs(d)
            normal *= np.sign((np.conj(normal) * inward).real) or 1
            off = 0.2 * abs(d) * normal
            line(p, q, lw=1.6, alpha=0.9, zorder=3)
            line(p + 0.17 * d + off, q - 0.17 * d + off, lw=1.2, alpha=0.85, zorder=3)
        else:
            normal = 1j * d / abs(d)
            shifts = (-0.09, 0.09) if order == 2 else (-0.15, 0.0, 0.15)
            for k in shifts:
                line(p + k * abs(d) * normal, q + k * abs(d) * normal, lw=1.3, alpha=0.9, zorder=3)

    for i, w in enumerate(stars):
        if not m.used[i]:
            continue  # starless atom: the bonds just meet in a plain corner
        special = mol.symbols[i] != "C" or mol.degree(i) == 1
        core = max(size(cat.mag[m.stars[i]]) * 1.4, 30.0)
        ax.scatter([w.real], [w.imag], s=core * 2.8, c=glow_col, alpha=0.22, edgecolors="none", zorder=4)
        if special:
            ax.scatter([w.real], [w.imag], s=core * 2.6, marker="*", c="#ffffff", edgecolors="none", zorder=5)
        else:
            ax.scatter([w.real], [w.imag], s=core, c="#ffffff", edgecolors="none", zorder=5)
        if labels and mol.symbols[i] != "C":
            ax.annotate(mol.symbols[i], (w.real, w.imag), xytext=(7, 7), textcoords="offset points",
                        color=glow_col, fontsize=9, fontfamily="monospace", zorder=6)

    # Compass: where north and east really point on this chart.
    base = complex(hi[0] - 0.55 * s, lo[1] + 0.55 * s)
    for label, direction in (("N", 1j), ("E", -1.0)):
        tip = base + 0.45 * s * direction * rot
        ax.annotate("", xy=(tip.real, tip.imag), xytext=(base.real, base.imag),
                    arrowprops=dict(arrowstyle="-|>", color="#8a96a3", lw=0.9), zorder=2)
        txt = base + 0.62 * s * direction * rot
        ax.text(txt.real, txt.imag, label, color="#8a96a3", fontsize=8, ha="center", va="center",
                fontfamily="monospace")

    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(mol.name, color=line_col, fontsize=15, fontfamily="monospace", pad=14)
    ax.text(0.5, -0.02, f"RA {info['ra']}   Dec {info['dec']}   ·   J2000   ·   {info['constellation']}",
            transform=ax.transAxes, color=line_col, fontsize=11, ha="center", va="top", fontfamily="monospace")
    fig.savefig(path, dpi=300, facecolor=bg_col, bbox_inches="tight", pad_inches=0.3)
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
    p.add_argument("--catalog", help="path to a local hygdata_v41.csv instead of downloading it")
    args = p.parse_args(argv)

    mol = load_molecule(args.molecule, args.title)
    print(f"{mol.name}: {mol.n} heavy atoms, {len(mol.bonds)} bonds")
    full = load_catalog(args.mag + 1.0, args.catalog)  # fainter stars only as chart background
    sel = np.flatnonzero(full.mag <= args.mag)
    cat = full.subset(sel)
    print(f"{len(cat.hip)} stars brighter than mag {args.mag}")

    matches = find_asterisms(mol, cat, bond=tuple(args.bond), region=args.region,
                             mirror=not args.no_mirror, skip=args.skip, top=args.top)
    if not matches:
        raise SystemExit("No match found. Try a fainter --mag, a wider --bond range or a bigger --region.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", mol.name.lower()).strip("-")[:40] or "molecule"
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
            plot_chart(mol, full, m, info, png, args.north_up, args.labels, args.star_names)
            info["chart"] = str(png)
    (out / f"{slug}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nCharts and {slug}.json written to {out}/")


if __name__ == "__main__":
    main()
