#!/usr/bin/env python3
"""
work_of_separation2.py  — Te embrittlement of the Ni Sigma5 GB (MACE-MP-0), robust version.

Fixes the earlier PBC-cleave bug by building a proper (021) FREE-SURFACE SLAB (finite along the
GB normal, vacuum on both ends) instead of opening a gap in a periodic cell. Computes:
  gamma_GB, gamma_surf  ->  W_sep(clean) = 2*gamma_surf - gamma_GB
  Rice-Wang embrittlement:  dW_sep * A = E_seg^surface - E_seg^GB
     = [E(Te@surf) - E(clean surf)] - [E(Te@GB) - E(clean GB)]
  dW_sep < 0  => Te lowers the work of separation => embrittler.

Run:  python work_of_separation2.py Ni_S5_021_relaxed.vasp --normal 0 --te-site 539 --vac 12
(float32 is fine for these differences; set --calc emt only to test geometry on pure Ni.)
"""
import argparse, numpy as np
from ase.io import read
from ase.optimize import FIRE
from ase.lattice.cubic import FaceCenteredCubic

def get_calc(name="mace"):
    if name == "emt":
        from ase.calculators.emt import EMT; return EMT()
    from mace.calculators import mace_mp
    return mace_mp(model="medium", dispersion=False, default_dtype="float32")

def relax(atoms, calc, fmax=0.03, steps=300):
    a = atoms.copy(); a.calc = calc
    FIRE(a, logfile=None).run(fmax=fmax, steps=steps)
    return a.get_potential_energy(), a

def sub(atoms, idx, sym):
    a = atoms.copy(); s = a.get_chemical_symbols(); s[int(idx)] = sym; a.set_chemical_symbols(s); return a

def make_surface_slab(bicrystal, n, vac=12.0):
    """Keep the lower grain (fractional [0,0.5) along n) and expose its (021) faces with vacuum."""
    at = bicrystal.copy(); sp = at.get_scaled_positions()
    slab = at[(sp[:, n] >= 0.0) & (sp[:, n] < 0.5)]
    cell = slab.get_cell().copy(); axis = cell[n] / np.linalg.norm(cell[n])
    pos = slab.get_positions()
    ext = pos[:, n].max() - pos[:, n].min()
    cell[n] = axis * (ext + 2*vac); slab.set_cell(cell)
    pos[:, n] += (vac - pos[:, n].min()); slab.set_positions(pos)
    pbc = list(slab.get_pbc()); pbc[n] = False; slab.set_pbc(pbc)
    return slab

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("structure"); ap.add_argument("--normal", type=int, default=0)
    ap.add_argument("--te-site", type=int, default=-1)
    ap.add_argument("--vac", type=float, default=12.0)
    ap.add_argument("--gbwidth", type=float, default=3.0)
    ap.add_argument("--calc", default="mace", choices=["mace", "emt"])
    a = ap.parse_args()
    calc = get_calc(a.calc); J = 16.0218; n = a.normal

    # bulk reference energy per atom
    b = FaceCenteredCubic('Ni', size=(3,3,3), latticeconstant=3.52); b.calc = calc
    mu = b.get_potential_energy()/len(b)

    gb0 = read(a.structure)
    L = gb0.get_cell().lengths(); ip = [i for i in range(3) if i != n]; A = L[ip[0]]*L[ip[1]]
    pos = gb0.get_positions()

    # clean GB and clean surface slab
    E_gb, gb = relax(gb0, calc)
    slab0 = make_surface_slab(gb, n, a.vac)
    E_sf, slab = relax(slab0, calc)
    g_gb   = (E_gb - len(gb)*mu)/(2*A)*J
    g_surf = (E_sf - len(slab)*mu)/(2*A)*J
    W_clean = 2*g_surf - g_gb
    print(f"# gamma_GB = {g_gb:.3f} J/m^2 | gamma_surf = {g_surf:.3f} J/m^2 | W_sep(clean) = {W_clean:.3f} J/m^2")

    # Te at GB site
    if a.te_site >= 0: te = a.te_site
    else:
        sites = np.where(np.abs(pos[:, n] - L[n]/2) < a.gbwidth)[0]
        best=None
        for i in sites:
            e,_ = relax(sub(gb, int(i), 'Te'), calc)
            if best is None or e<best[1]: best=(int(i), e)
        te = best[0]
    E_gb_Te,_ = relax(sub(gb, te, 'Te'), calc)

    # Te on the surface: substitute the top-surface atom (highest coord along n) of the slab
    spos = slab.get_positions()[:, n]
    top = int(np.argmax(spos))
    E_sf_Te,_ = relax(sub(slab, top, 'Te'), calc)

    dW_A = (E_sf_Te - E_sf) - (E_gb_Te - E_gb)      # = E_seg^surf - E_seg^GB  (eV)
    dW = dW_A / A * J
    W_Te = W_clean + dW
    print(f"# E_seg^GB (binding) = {E_gb_Te - E_gb:.3f} eV | E_seg^surf = {E_sf_Te - E_sf:.3f} eV")
    print(f"# W_sep(clean) = {W_clean:.3f} | W_sep(Te) = {W_Te:.3f} J/m^2")
    print(f"# EMBRITTLEMENT dW_sep = {dW:+.3f} J/m^2 ; potency = {dW_A:+.3f} eV/atom")
    print(f"# dW < 0 => Te embrittles (prefers the free surface over the GB).")

if __name__ == "__main__":
    main()
