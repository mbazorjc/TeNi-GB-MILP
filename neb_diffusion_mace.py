#!/usr/bin/env python3
"""
neb_diffusion_mace.py — vacancy-mediated Te migration barrier at a Ni grain boundary
(and in bulk) via climbing-image NEB with MACE-MP-0. Reproduces/extends Huai 2017
(Te diffuses fast along some GBs: barrier lower than bulk).

Mechanism: Te sits substitutionally at site A with a vacancy at an adjacent site B;
the Te hops into the vacancy (A -> B). Endpoints share the SAME atoms (N-1); only the
Te position differs, so NEB interpolates cleanly.

    # GB path (auto-picks nearest in-plane GB neighbour of the Te site):
    python neb_diffusion_mace.py Ni_S5_021_relaxed.vasp --normal 0 --te-site 539 --images 5
    # bulk reference path:
    python neb_diffusion_mace.py Ni_S5_021_relaxed.vasp --bulk --images 5
Compare the GB barrier to the bulk barrier: lower at the GB => fast GB diffusion.
Use float64 for publication numbers.
"""
import argparse, numpy as np
from ase.io import read
from ase.optimize import FIRE
from ase.mep import NEB
import torch

def get_calc(name="mace"):
    if name == "emt":
        from ase.calculators.emt import EMT
        return EMT()
    from mace.calculators import mace_mp
    return mace_mp(model="medium", dispersion=False, default_dtype="float32", batch_size=1, device='cpu')

def relax(atoms, calc, fmax=0.03, steps=300):
    a = atoms.copy(); a.calc = calc
    FIRE(a, logfile=None).run(fmax=fmax, steps=steps)
    torch.cuda.empty_cache()
    return a

def build_endpoints(clean, A, B, species):
    """initial: species@A + vacancy@B ; final: species@B + vacancy@A. Same atom list."""
    posA = clean.positions[A].copy(); posB = clean.positions[B].copy()
    init = clean.copy()
    s = init.get_chemical_symbols(); s[A] = species; init.set_chemical_symbols(s)
    del init[B]                                   # vacancy at B
    tidx = [i for i, sy in enumerate(init.get_chemical_symbols()) if sy == species][0]
    final = init.copy()
    final.positions[tidx] = posB                  # Te hops A -> B (vacancy now at A)
    return init, final

def run_neb(init, final, calc, nimages, fmax=0.05):
    images = [init] + [init.copy() for _ in range(nimages)] + [final]
    neb = NEB(images, climb=True, allow_shared_calculator=True)
    neb.interpolate(mic=True)
    for im in images: im.calc = calc
    FIRE(neb, logfile=None).run(fmax=fmax, steps=300)
    E = np.array([im.get_potential_energy() for im in images])
    return E

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("structure")
    ap.add_argument("--normal", type=int, default=0)
    ap.add_argument("--te-site", type=int, default=-1)
    ap.add_argument("--to-site", type=int, default=-1, help="destination site (default: nearest GB neighbour)")
    ap.add_argument("--images", type=int, default=5)
    ap.add_argument("--gbwidth", type=float, default=3.0)
    ap.add_argument("--bulk", action="store_true", help="do a bulk reference path instead of GB")
    ap.add_argument("--species", default="Te")
    ap.add_argument("--calc", default="mace", choices=["mace", "emt"])
    a = ap.parse_args()

    calc = get_calc(a.calc)
    clean = relax(read(a.structure), calc)          # relax the clean cell first
    pos = clean.get_positions(); n = a.normal; L = clean.get_cell().lengths()

    if a.bulk:
        A = int(np.argmin(np.abs(pos[:, n] - L[n]/4)))   # a bulk-region atom
    elif a.te_site >= 0:
        A = a.te_site
    else:
        gb_sites = np.where(np.abs(pos[:, n] - L[n]/2) < a.gbwidth)[0]
        A = int(gb_sites[len(gb_sites)//2])

    # destination B: nearest neighbour of A (in the GB plane for GB path)
    d = pos - pos[A]
    for k in range(3): d[:, k] -= L[k]*np.round(d[:, k]/L[k])
    dd = np.linalg.norm(d, axis=1)
    if a.to_site >= 0:
        B = a.to_site
    else:
        order = np.argsort(dd)
        B = next(int(i) for i in order if i != A and dd[i] > 0.1)
    print(f"# path: {a.species} {A} -> {B}   (d = {dd[B]:.2f} Å, {'bulk' if a.bulk else 'GB'})")

    init, final = build_endpoints(clean, A, B, a.species)
    init = relax(init, calc); final = relax(final, calc)
    E = run_neb(init, final, calc, a.images)
    Ef = E - E[0]
    barrier_fwd = E.max() - E[0]
    barrier_rev = E.max() - E[-1]
    print("# image energies (eV, relative to initial):")
    print("  " + "  ".join(f"{e:+.3f}" for e in Ef))
    print(f"# MIGRATION BARRIER (forward)  = {barrier_fwd:.3f} eV")
    print(f"# MIGRATION BARRIER (reverse)  = {barrier_rev:.3f} eV")
    print("# Compare GB vs bulk: lower barrier at the GB => fast grain-boundary diffusion (Huai 2017).")

if __name__ == "__main__":
    main()
