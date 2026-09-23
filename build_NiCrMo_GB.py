#!/usr/bin/env python3
"""
build_NiCrMo_GB.py — turn a relaxed Ni grain boundary into a random Ni-Cr-Mo (Hastelloy-N)
alloy boundary, for the realistic-matrix Te segregation study.

Hastelloy-N ≈ Ni-16Mo-7Cr wt%  ≈  Ni-10Mo-8Cr at%  (Fe and minor elements dropped).
Generates several random realizations (seeds) so you can average Te segregation over the
chemical disorder (Wagih-style spectrum) rather than one arbitrary arrangement.

Usage:
    python build_NiCrMo_GB.py Ni_S5_021_relaxed.vasp --mo 0.10 --cr 0.08 --seeds 5
Outputs NiCrMo_S5_seed0.vasp ... and prints the achieved composition.
Then run segregation_energy_mace.py / cosegregation_mace.py on each.
"""
import argparse, numpy as np
from ase.io import read, write

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("structure")
    ap.add_argument("--mo", type=float, default=0.10, help="Mo atomic fraction")
    ap.add_argument("--cr", type=float, default=0.08, help="Cr atomic fraction")
    ap.add_argument("--seeds", type=int, default=5, help="number of random realizations")
    ap.add_argument("--protect", type=int, nargs="*", default=[],
                    help="atom indices to keep as Ni (e.g. a Te test site)")
    a = ap.parse_args()

    base = read(a.structure)
    N = len(base)
    nMo = int(round(a.mo*N)); nCr = int(round(a.cr*N))
    pool = [i for i in range(N) if i not in set(a.protect)]
    tag = a.structure.rsplit('.', 1)[0].split('/')[-1].replace('Ni_', '').replace('_relaxed', '')

    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        pick = rng.choice(pool, size=nMo+nCr, replace=False)
        mo_idx, cr_idx = pick[:nMo], pick[nMo:]
        at = base.copy(); s = at.get_chemical_symbols()
        for i in mo_idx: s[i] = 'Mo'
        for i in cr_idx: s[i] = 'Cr'
        at.set_chemical_symbols(s)
        syms = at.get_chemical_symbols()
        comp = {e: 100*syms.count(e)/N for e in ['Ni', 'Cr', 'Mo']}
        out = f"NiCrMo_{tag}_seed{seed}.vasp"
        write(out, at, format='vasp')
        print(f"{out}: Ni {comp['Ni']:.1f} / Cr {comp['Cr']:.1f} / Mo {comp['Mo']:.1f} at%  ({N} atoms)")
    print("\nNext: run segregation_energy_mace.py on each seed and average the Te spectrum;")
    print("compare to the pure-Ni value (-0.944 eV @ Σ5) to see how the alloy matrix shifts Te.")

if __name__ == "__main__":
    main()
