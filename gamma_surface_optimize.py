#!/usr/bin/env python3
"""
gamma_surface_optimize.py
Minimize a grain-boundary energy over its microscopic degrees of freedom:
  (1) rigid in-plane translation of one grain  (the gamma-surface: dy, dz)
  (2) rigid normal offset / interplanar spacing (dx)
  (3) atom-deletion distance for overlaps after translation  (SWEPT)
  (4) atomic relaxation + perpendicular box-length relaxation of the best candidate
This is the step that brings a raw CSL GB down to the literature GB energy.

IMPORTANT: run this on a PROPER CSL grain boundary (from generate_reference_GBs_aimsgb.py),
not on a hand-built mirror GB. The scan only polishes a correct coincidence structure.

Cheap-then-accurate workflow:
  - scan with EMT (pure Ni) or MACE-MP-0 to find the optimal translation+deletion,
  - then do ONE final ionic relaxation of the winner in VASP for the publication value.

Usage:
    python gamma_surface_optimize.py Ni_S5_021_POSCAR.vasp --normal 0 --grid 8 8 --calc emt
"""
import argparse, numpy as np
from ase.io import read, write
from ase.optimize import FIRE

def get_calc(name):
    if name == "mace":
        from mace.calculators import mace_mp
        return mace_mp(model="medium", dispersion=False, default_dtype="float64")
    from ase.calculators.emt import EMT
    return EMT()

def bulk_energy_per_atom(calc, a0=3.52):
    from ase.lattice.cubic import FaceCenteredCubic
    b = FaceCenteredCubic('Ni', size=(3,3,3), latticeconstant=a0); b.calc = calc
    return b.get_potential_energy()/len(b)

def remove_overlaps(atoms, rcut):
    pos = atoms.get_positions(); cell = atoms.get_cell().lengths()
    keep=[]; taken=np.zeros(len(pos), bool)
    for i in range(len(pos)):
        if taken[i]: continue
        keep.append(i); d = pos - pos[i]
        for k in range(3): d[:,k]-=cell[k]*np.round(d[:,k]/cell[k])
        r=np.linalg.norm(d,axis=1)
        for j in np.where(r<rcut)[0]:
            if j>i: taken[j]=True
    return atoms[keep]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("structure")
    ap.add_argument("--normal", type=int, default=0, help="GB normal axis: 0=x,1=y,2=z")
    ap.add_argument("--grid", nargs=2, type=int, default=[8,8], help="in-plane translation grid")
    ap.add_argument("--dx", nargs="+", type=float, default=[-0.3,0.0,0.3], help="normal offsets (Ang)")
    ap.add_argument("--rcut", nargs="+", type=float, default=[1.5,1.7,1.9,2.1],
                    help="atom-deletion distances to SWEEP (Ang)")
    ap.add_argument("--calc", default="emt", choices=["emt","mace"])
    ap.add_argument("--steps", type=int, default=400, help="max relaxation steps")
    ap.add_argument("--relax-cell", action="store_true",
                    help="also relax the box length perpendicular to the GB")
    a=ap.parse_args()

    base=read(a.structure)
    calc=get_calc(a.calc); ebulk=bulk_energy_per_atom(calc)
    n=a.normal; inplane=[i for i in range(3) if i!=n]
    L=base.get_cell().lengths(); A=L[inplane[0]]*L[inplane[1]]
    J=16.0218

    def gamma_of(at):
        at=at.copy(); at.calc=calc
        return (at.get_potential_energy()-len(at)*ebulk)/(2*A)*J  # 2 GBs per cell

    sp=base.get_scaled_positions(); upper=sp[:,n]>=0.5
    best=None
    for iy in range(a.grid[0]):
        for iz in range(a.grid[1]):
            ty=iy/a.grid[0]*L[inplane[0]]; tz=iz/a.grid[1]*L[inplane[1]]
            for dxn in a.dx:
                for rc in a.rcut:
                    at=base.copy(); p=at.get_positions()
                    p[upper, inplane[0]]+=ty; p[upper, inplane[1]]+=tz; p[upper, n]+=dxn
                    at.set_positions(p); at.wrap()
                    at=remove_overlaps(at, rc)
                    try: g=gamma_of(at)
                    except Exception: continue
                    if best is None or g<best[0]: best=(g, ty, tz, dxn, rc, at)

    if best is None:
        print("No valid candidate found — check --normal and structure."); return
    print(f"best rigid (unrelaxed) gamma = {best[0]:.3f} J/m^2  "
          f"at dy={best[1]:.2f} dz={best[2]:.2f} dx={best[3]:.2f} rcut={best[4]:.2f}")

    win=best[5].copy(); win.calc=calc
    if a.relax_cell:
        from ase.filters import FrechetCellFilter
        voigt={0:0,1:1,2:2}[n]
        mask=[1 if i==voigt else 0 for i in range(6)]   # free ONLY the normal-axis length
        FIRE(FrechetCellFilter(win, mask=mask), logfile=None).run(fmax=0.05, steps=a.steps)
        tag="atoms + perpendicular box"
    else:
        FIRE(win, logfile=None).run(fmax=0.05, steps=a.steps)
        tag="atoms (fixed cell)"
    g_relaxed=(win.get_potential_energy()-len(win)*ebulk)/(2*A)*J
    print(f"RELAXED gamma ({tag}) = {g_relaxed:.3f} J/m^2   (atoms={len(win)}, calc={a.calc})")

    out=a.structure.rsplit('.',1)[0]+"_gbopt.vasp"
    write(out, win, format='vasp')
    print(f"saved optimized GB: {out}")
    print("Next: one final ionic relaxation of this structure in VASP for the publication value.")

if __name__=="__main__":
    main()
