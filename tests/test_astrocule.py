import numpy as np
import pytest

import astrocule as A


def test_known_molecules_load_with_unit_bonds():
    for key in A.KNOWN_MOLECULES:
        mol = A.load_molecule(key)
        lengths = [abs(mol.z[i] - mol.z[j]) for i, j, _ in mol.bonds]
        assert np.mean(lengths) == pytest.approx(1.0)
        assert abs(mol.z.mean()) < 1e-9
    assert A.load_molecule("estradiol").n == 20  # C18 + O2 heavy atoms


def test_smiles_input_and_kekulized_bonds():
    mol = A.load_molecule("c1ccccc1O", title="Phenol")
    assert mol.name == "Phenol" and mol.n == 7
    assert sorted(o for *_, o in mol.bonds) == [1, 1, 1, 1, 2, 2, 2]


def test_fit_similarity_recovers_transform():
    rng = np.random.default_rng(0)
    z = rng.normal(size=12) + 1j * rng.normal(size=12)
    a, b = 1.7 * np.exp(0.8j), 3 - 2j
    fa, fb = A.fit_similarity(z, a * z + b)
    assert fa == pytest.approx(a) and fb == pytest.approx(b)
    fa, _ = A.fit_similarity(z, a * z + b, smax=1.0)
    assert abs(fa) == pytest.approx(1.0)


def test_projection_roundtrip():
    center = A.radec_to_xyz(123.0, -40.0)
    v = A.radec_to_xyz(np.array([120.0, 130.0, 123.0]), np.array([-35.0, -48.0, -40.0]))
    w = A.project(v, center)
    assert np.allclose(A.unproject(w, center), v)
    assert abs(w[2]) < 1e-9
    east = A.project(A.radec_to_xyz(124.0, -40.0), center)
    assert east.real < 0  # east is to the left on the chart


def test_format_coordinates():
    assert A.format_ra(305.5) == "20h 22m 00.0s"
    assert A.format_dec(-44.23667) == "-44° 14′ 12″"


def _planted_catalog(mol, shape, seed=7, drop=None):
    """400 random stars within 12 deg of (200, +30) plus a slightly jittered copy of shape.
    Atom `drop`, if given, gets no star."""
    rng = np.random.default_rng(seed)
    center = A.radec_to_xyz(200.0, 30.0)
    n_bg = 400
    bg = 12.0 * np.sqrt(rng.uniform(size=n_bg)) * np.exp(1j * rng.uniform(0, 2 * np.pi, n_bg))
    planted = 1.5 * np.exp(0.6j) * shape + (1 - 2j) + rng.normal(scale=0.05, size=mol.n) * (1 + 1j)
    if drop is not None:
        planted = np.delete(planted, drop)
    ra, dec = A.xyz_to_radec(A.unproject(np.concatenate([bg, planted]), center))
    n = len(ra)
    cat = A.Catalog(np.arange(n), ra, dec, np.full(n, 5.0), np.full(n, ""), np.full(n, "Boo"))
    return cat, np.arange(n_bg, n)


@pytest.mark.parametrize("mirrored", [False, True])
def test_finds_planted_molecule(mirrored):
    """Hide a slightly jittered molecule among random stars and check the search recovers it."""
    mol = A.load_molecule("serotonin")
    shape = np.conj(mol.z) if mirrored else mol.z
    cat, planted = _planted_catalog(mol, shape)

    best = A.find_asterisms(mol, cat, bond=(1.0, 2.0), region=(200.0, 30.0, 8.0), top=1, verbose=False)[0]
    # A random star can land closer to an ideal atom position than the jittered planted one,
    # so require the search to do at least as well as the planted figure and mostly reuse it.
    truth = A._polish(shape, mirrored, planted, cat.xyz, 1.0, 2.0, skip=0)[0]
    assert best.mirrored == mirrored
    assert best.n_skipped == 0
    assert best.score <= truth.score * 1.001
    assert len(set(best.stars) & set(planted)) >= mol.n - 2
    assert best.score < 0.1
    assert best.bond_deg == pytest.approx(1.5, rel=0.05)


@pytest.mark.parametrize("drop", [0, 6])
def test_skip_leaves_missing_atom_starless(drop):
    """With one planted star missing, skip=1 should find the rest and leave that atom starless."""
    mol = A.load_molecule("serotonin")
    cat, planted = _planted_catalog(mol, mol.z, drop=drop)

    matches = A.find_asterisms(mol, cat, bond=(1.0, 2.0), region=(200.0, 30.0, 8.0), mirror=False,
                               skip=1, top=3, verbose=False)
    assert [m.score for m in matches] == sorted(m.score for m in matches)
    assert {m.n_skipped for m in matches} == {0, 1}
    best = min((m for m in matches if m.n_skipped == 1), key=lambda m: m.score)
    assert best.score < 0.1
    assert np.flatnonzero(~best.used).tolist() == [drop]
    assert len(set(best.stars[best.used]) & set(planted)) >= mol.n - 3
    assert best.score < min(m.score for m in matches if m.n_skipped == 0)


def test_lsap_matches_scipy():
    from scipy.optimize import linear_sum_assignment
    rng = np.random.default_rng(3)
    for n, m in [(1, 1), (5, 5), (7, 12), (20, 60)]:
        cost = rng.uniform(size=(n, m))
        rows, cols = linear_sum_assignment(cost)
        ours = A._lsap(cost)
        assert len(set(ours)) == n
        assert cost[np.arange(n), ours].sum() == pytest.approx(cost[rows, cols].sum())


def test_refine_forced_skip():
    rng = np.random.default_rng(1)
    T = rng.normal(size=8) + 1j * rng.normal(size=8)
    P = np.concatenate([2 * T + 1, rng.normal(size=30) * 6 + 1j * rng.normal(size=30) * 6])
    stars, a, b, score = A.refine(T, P, 2.0, 1.0, 0.5, 4.0, forced=(3,))
    assert stars[3] == -1
    assert all(stars[i] == i for i in range(8) if i != 3)
    assert score == pytest.approx(0.0, abs=1e-9)
