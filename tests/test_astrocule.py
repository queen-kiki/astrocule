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


@pytest.mark.parametrize("mirrored", [False, True])
def test_finds_planted_molecule(mirrored):
    """Hide a slightly jittered molecule among random stars and check the search recovers it."""
    rng = np.random.default_rng(7)
    mol = A.load_molecule("serotonin")
    center = A.radec_to_xyz(200.0, 30.0)

    # 400 random stars within 12 deg of the center, plus the planted figure.
    n_bg = 400
    r = 12.0 * np.sqrt(rng.uniform(size=n_bg))
    phi = rng.uniform(0, 2 * np.pi, n_bg)
    bg = r * np.exp(1j * phi)
    shape = np.conj(mol.z) if mirrored else mol.z
    planted = 1.5 * np.exp(0.6j) * shape + (1 - 2j) + rng.normal(scale=0.05, size=mol.n) * (1 + 1j)
    v = A.unproject(np.concatenate([bg, planted]), center)
    ra, dec = A.xyz_to_radec(v)
    n = len(ra)
    cat = A.Catalog(np.arange(n), ra, dec, np.full(n, 5.0), np.full(n, ""), np.full(n, "Boo"))

    matches = A.find_asterisms(mol, cat, bond=(1.0, 2.0), region=(200.0, 30.0, 8.0), top=1, verbose=False)
    best = matches[0]
    # A random star can land closer to an ideal atom position than the jittered planted one,
    # so require the search to do at least as well as the planted figure and mostly reuse it.
    truth = A._polish(shape, mirrored, np.arange(n_bg, n), cat.xyz, 1.0, 2.0)
    assert best.mirrored == mirrored
    assert best.score <= truth.score * 1.001
    assert len(set(best.stars) & set(range(n_bg, n))) >= mol.n - 2
    assert best.score < 0.1
    assert best.bond_deg == pytest.approx(1.5, rel=0.05)
