#!/usr/bin/env python3
"""
segregation_energy_mace.py
Te grain-boundary segregation energy in Ni using a FOUNDATION ML potential (MACE-MP-0).

This is the core calculation for Manuscript 1. It runs on the structures built by
build_ni_sigma5_gb.py. No DFT required.

Segregation energy (substitutional, consistent supercell):
    E_seg(site) = [E(Te@GB_site) - E(GB, pure Ni)] - [E(Te@bulk_site) - E(bulk-region, pure Ni)]
Because we substitute one Ni->Te in the SAME cell, the pure-Ni reference cancels and
    E_seg(site) = E(Te@GB_site) - E(Te@bulk_ref)
A negative value = Te prefers the grain boundary (segregation).

VALIDATION GATE (Week 1): the most-favourable GB site should give a clearly negative
E_seg, and the trend Sigma5 (strong) vs Sigma3 (weak) must match the DFT literature:
  - Liu et al., Comput. Mater. Sci. 88 (2014) 22-27  (Te embrittles Ni Sigma5 GB)
  - 'Diffusion of Te at Ni grain boundaries', RSC Adv. (2017): Te segregation strongest
    at Sigma5(021), weakest at Sigma3(111).
If MACE-MP-0 reproduces this ordering and a sensible magnitude, proceed; else fine-tune.

INSTALL (ideally on a GPU machine):
    pip install mace-torch ase
    # MACE-MP-0 weights download automatically on first use (model='medium').
RUN:
    python segregation_energy_mace.py Ni_S5_210_GB_relaxed.vasp
"""
import sys, numpy as np
from ase.io import read
from ase.optimize import FIRE
import torch

# def get_calc():
#     from mace.calculators import mace_mp
#     # foundation model; dispersion off; float64 for energies
#     return mace_mp(model="medium", dispersion=False, default_dtype="float64")

def get_calc():
    from mace.calculators import mace_mp
    torch.backends.cuda.preferred_linalg_library('magma')
    #device = 'cpu'

    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #return mace_mp(model="medium", dispersion=False, default_dtype="float64", device=device)
    return mace_mp(model="medium", dispersion=False, default_dtype="float64", batch_size=1)



def relax(atoms, calc, fmax=0.03, steps=300):
    atoms = atoms.copy(); atoms.calc = calc
    FIRE(atoms, logfile=None).run(fmax=fmax, steps=steps)
    torch.cuda.empty_cache()
    return atoms.get_potential_energy(), atoms

def main(gb_file):
    calc = get_calc()
    gb = read(gb_file)
    pos = gb.get_positions(); x = pos[:,0]; Lx = gb.get_cell().lengths()[0]

    # candidate GB sites: atoms within ~3 A of the GB plane (x = Lx/2)
    gb_mask = np.abs(x - Lx/2) < 3.0
    gb_sites = np.where(gb_mask)[0]
    # bulk reference site: nearest to quarter cell (far from both GBs)
    bulk_ref = int(np.argmin(np.abs(x - Lx/4)))

    # reference: Te at the bulk-like site
    a = gb.copy(); s = a.get_chemical_symbols(); s[bulk_ref] = 'Te'; a.set_chemical_symbols(s)
    E_bulk_Te, _ = relax(a, calc)
    print(f"# bulk-ref Te site idx={bulk_ref}  E={E_bulk_Te:.4f} eV")
    print(f"# scanning {len(gb_sites)} GB sites")
    print(f"{'site':>6} {'x(A)':>8} {'E_seg(eV)':>12}")

    results = []
    for idx in gb_sites:
        a = gb.copy(); s = a.get_chemical_symbols(); s[int(idx)] = 'Te'; a.set_chemical_symbols(s)
        E_gb_Te, _ = relax(a, calc)
        e_seg = E_gb_Te - E_bulk_Te
        results.append((int(idx), float(x[idx]), e_seg))
        print(f"{int(idx):6d} {x[idx]:8.2f} {e_seg:12.3f}")

    results.sort(key=lambda r: r[2])
    print("\n# SEGREGATION-ENERGY SPECTRUM (most favourable first)")
    for idx, xi, es in results[:10]:
        print(f"  site {idx:5d}  x={xi:6.2f}  E_seg={es:7.3f} eV")
    print(f"\n# strongest Te segregation: {results[0][2]:.3f} eV "
          f"(should be clearly negative; compare to Liu 2014 / RSC 2017)")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Ni_S5_210_GB.vasp")
