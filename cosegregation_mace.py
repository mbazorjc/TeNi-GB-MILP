#!/usr/bin/env python3
"""
cosegregation_mace.py  — Te–Cr and Te–Mo co-segregation at a Ni grain boundary (MACE-MP-0).

Question this answers (your core contribution): does the Hastelloy-N matrix (Cr, Mo)
PROMOTE or BLOCK tellurium at the grain boundary?

Pairwise binding energy of a Te–X pair at the GB (substitutional Ni->Te, Ni->X):
    E_bind = E(GB+Te+X) + E(GB_clean) - E(GB+Te) - E(GB+X)
    E_bind < 0  -> attractive : X co-segregates WITH Te  -> PROMOTES Te embrittlement
    E_bind > 0  -> repulsive  : X avoids Te              -> BLOCKS / mitigates
This connects to the experiments (Fu 2017, Han 2020) where GB chemistry changes Te behaviour.

Run on the SAME relaxed boundary that passed Stage 0:
    python cosegregation_mace.py Ni_S5_021_relaxed.vasp --normal 0 --te-site 539 --rmax 5.0
IMPORTANT: use the SAME --normal axis your segregation_energy_mace.py used (axis 0 / x),
since that run reproduced Σ5 ≫ Σ3. (omit --te-site to auto-pick the strongest Te site).
Use float64 for publication numbers.
"""
import argparse, numpy as np
from ase.io import read
from ase.optimize import FIRE
import torch
torch.backends.cuda.preferred_linalg_library('magma')

def get_calc():
    from mace.calculators import mace_mp
    return mace_mp(model="medium", dispersion=False, default_dtype="float64") #GPU
    #return mace_mp(model="medium", dispersion=False, default_dtype="float64", device='cpu') #CPU

def relax(atoms, calc, fmax=0.03, steps=300):
    a = atoms.copy(); a.calc = calc
    FIRE(a, logfile=None).run(fmax=fmax, steps=steps)
    return a.get_potential_energy()

def substitute(atoms, **idx_sym):
    a = atoms.copy(); s = a.get_chemical_symbols()
    for idx, sym in idx_sym.items():
        s[int(idx)] = sym
    a.set_chemical_symbols(s); return a

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("structure")
    ap.add_argument("--normal", type=int, default=0, help="GB-normal axis (match your segregation run = 0/x)")
    ap.add_argument("--te-site", type=int, default=-1, help="atom index for Te (default: auto strongest)")
    ap.add_argument("--rmax", type=float, default=5.0, help="max Te-X separation to scan (Å)")
    ap.add_argument("--gbwidth", type=float, default=3.0, help="GB slab half-width (Å)")
    a = ap.parse_args()

    calc = get_calc()
    gb = read(a.structure)
    pos = gb.get_positions(); n = a.normal; L = gb.get_cell().lengths()
    gb_plane = L[n]/2.0
    gb_sites = np.where(np.abs(pos[:, n] - gb_plane) < a.gbwidth)[0]

    E_clean = relax(gb, calc)
    print(f"# E(clean GB) = {E_clean:.4f} eV ; {len(gb_sites)} GB sites")

    # Te anchor
    if a.te_site >= 0:
        te = a.te_site
    else:
        best = None
        for i in gb_sites:
            e = relax(substitute(gb, **{str(i): 'Te'}), calc)
            if best is None or e < best[1]: best = (int(i), e)
        te = best[0]
    E_Te = relax(substitute(gb, **{str(te): 'Te'}), calc)
    print(f"# Te anchor site {te}  x={pos[te,n]:.2f} Å  E(GB+Te)={E_Te:.4f} eV")

    # candidate X sites: GB sites within rmax of Te
    d = pos - pos[te]
    for k in range(3): d[:, k] -= L[k]*np.round(d[:, k]/L[k])
    dist = np.linalg.norm(d, axis=1)
    cand = [int(i) for i in gb_sites if i != te and dist[i] <= a.rmax]
    print(f"# scanning {len(cand)} neighbour sites for Cr/Mo within {a.rmax} Å of Te\n")

    summary = {}; allrows = []
    for X in ["Cr", "Mo"]:
        print(f"=== Te–{X} co-segregation ===")
        print(f"{'X-site':>7}{'dist(Å)':>9}{'E_bind(eV)':>12}  effect")
        results = []
        for ix in cand:
            E_X   = relax(substitute(gb, **{str(ix): X}), calc)
            E_TeX = relax(substitute(gb, **{str(te): 'Te', str(ix): X}), calc)
            Eb = E_TeX + E_clean - E_Te - E_X
            results.append((ix, dist[ix], Eb)); allrows.append((X, ix, dist[ix], Eb))
        results.sort(key=lambda r: r[2])
        for ix, dd, Eb in results:
            eff = "PROMOTES (attract)" if Eb < -0.02 else ("BLOCKS (repel)" if Eb > 0.02 else "neutral")
            print(f"{ix:7d}{dd:9.2f}{Eb:12.3f}  {eff}")
        summary[X] = results[0]
        print(f">> strongest Te–{X} binding: {results[0][2]:.3f} eV at d={results[0][1]:.2f} Å\n")

    with open("cosegregation_results.csv", "w") as f:
        f.write("X,site,distance_A,E_bind_eV\n")
        for X, ix, dd, Eb in allrows:
            f.write(f"{X},{ix},{dd:.3f},{Eb:.4f}\n")
    print("wrote cosegregation_results.csv")

    print("# VERDICT")
    for X, (ix, dd, Eb) in summary.items():
        verb = "promotes" if Eb < -0.02 else ("blocks" if Eb > 0.02 else "is neutral to")
        print(f"  {X} {verb} Te segregation (strongest E_bind = {Eb:+.3f} eV)")

if __name__ == "__main__":
    main()
