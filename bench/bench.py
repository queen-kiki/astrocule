"""Benchmark the asterism search. Prints timing and a fingerprint of the results so
optimizations can be checked for identical output.

    python bench/bench.py            # regional benchmark (fast, for iterating)
    python bench/bench.py --sky      # whole sky (the real workload)
"""
import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import astrocule as A  # noqa: E402

sky = "--sky" in sys.argv
mol = A.load_molecule("estradiol")
cat = A.load_catalog(6.5)
region = None if sky else (135.0, -50.0, 20.0)

t = time.perf_counter()
res = A.find_asterisms(mol, cat, bond=(1.0, 3.0), region=region, skip=1, top=10, verbose=False)
dt = time.perf_counter() - t

fp = hashlib.sha1(repr([(m.mirrored, tuple(int(s) for s in m.stars)) for m in res]).encode()).hexdigest()[:10]
print(f"{'sky' if sky else 'region'}: {dt:.2f}s  best={res[0].score:.4f}  "
      f"scores={[round(m.score, 4) for m in res[:5]]}  fingerprint={fp}")
