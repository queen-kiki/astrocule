# astrocule

Find a molecule's skeletal formula hidden among real stars.

Give it any molecule and it lays out the heavy-atom skeleton in 2D, searches the whole sky
(Hipparcos stars) for the set of stars that traces that skeleton with the least distortion,
and renders the best matches as star charts with their J2000 coordinates.

![17β-Estradiol in Vela](examples/17-estradiol_1.png)

## Usage

```sh
pip install -r requirements.txt

python astrocule.py estradiol                       # built-in name
python astrocule.py "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" --title Caffeine   # any SMILES
python astrocule.py melatonin                       # anything else is looked up on PubChem
```

Built-in names: estradiol, testosterone, progesterone, spironolactone, serotonin, dopamine, caffeine.

The first run downloads the [HYG database](https://github.com/astronexus/HYG-Database) (about 34 MB)
to `~/.cache/astrocule`. A whole-sky search takes a minute or two. Results go to `./out`: one chart
per match plus a JSON file listing every star (HIP number, RA/Dec, magnitude) and how far it sits from
its ideal atom position.

| option | default | what it does |
| --- | --- | --- |
| `--mag` | 6.5 | faintest star allowed in the figure. 6.5 is roughly naked eye under a dark sky |
| `--bond MIN MAX` | 1 3 | bond length on the sky in degrees |
| `--region RA DEC R` | whole sky | only figures centered within R degrees of RA/DEC |
| `--no-mirror` | | don't allow the mirror image of the drawing |
| `--top` | 5 | number of distinct matches to report and chart |
| `--north-up` | | draw charts north-up instead of turning the molecule upright |
| `--labels` | | label heteroatoms (O, N, ...) with their element |
| `--star-names` | | label named background stars |

## How it works

1. **Molecule.** RDKit parses the SMILES, kekulizes aromatic rings and draws a CoordGen 2D layout,
   scaled to unit bond length.
2. **Hypotheses.** The sky is split into tiles. In each tile, every pair of stars that could hold the
   molecule's two farthest-apart atoms defines one rotation + scale + shift. All atoms of that
   hypothesis are checked against the nearest stars, and hypotheses are pruned atom by atom as soon as
   they can't end up good.
3. **Refinement.** Survivors alternate a strict one-star-per-atom assignment (Hungarian algorithm) with
   a least-squares similarity fit until the assignment stops changing.
4. **Polish.** Each candidate is re-projected around its own center and refit, so the score and the
   chart share one nearly distortion-free frame. Projection is stereographic, which preserves shapes
   locally.

The score is the RMSD between the stars and the ideal atom positions, in units of the bond length.
0.1 means stars are off by a tenth of a bond on average.

### Things to know

- Bigger figures always score better, because the same star density gives you more freedom relative
  to the bond length. So the best match will usually use close to the largest `--bond` you allow.
  Choose the size range you actually want.
- Dense parts of the Milky Way (Vela, Carina, Cygnus) tend to win for the same reason. The report's
  "stars inside the figure brighter than its median star" count tells you how much the figure would
  have to compete with other stars to the eye.
- Mirror images are allowed by default. A flat drawing has no stereochemistry, so a mirrored skeleton
  is still the same molecule as far as the drawing goes. Use `--no-mirror` if you care.
- Charts are rotated so the molecule sits upright. The N/E arrows show where north and east really are.
  East is on the left, as when looking up at the sky.
- Coordinates are J2000 and stellar proper motion is ignored, which is far below what you can see.

## Tests

```sh
python -m pytest tests
```

## Credits

Star data: [HYG database v4.1](https://github.com/astronexus/HYG-Database) by David Nash
(CC BY-SA 4.0), built on the Hipparcos catalog.
