#!/usr/bin/env python3
"""
inspect_gb.py — confirm a grain-boundary structure is what you think it is.

Two checks, no guessing:
  (A) If OVITO's python module is installed: Polyhedral Template Matching ->
      reports % FCC / HCP / other. A COHERENT Sigma3(111) twin = bulk FCC with a
      SINGLE plane of HCP atoms. (This is the definitive test.)
  (B) Always: a centrosymmetry-parameter (CSP) side-view image. Bulk FCC -> CSP~0
      (blue); GB atoms -> high CSP (red). The GB plane shows as a sharp line of
      red atoms; a coherent twin is a thin, flat, ordered line.

Usage:
    python inspect_gb.py Ni_S3_111_POSCAR.vasp --normal 0
Outputs <name>_inspect.png and prints structure-type fractions if OVITO is present.
"""
import argparse, numpy as np
from ase.io import read
from ase.neighborlist import NeighborList
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def centrosymmetry(atoms, nn=12, cutoff=3.1):
    nl = NeighborList([cutoff/2]*len(atoms), self_interaction=False, bothways=True)
    nl.update(atoms)
    pos = atoms.get_positions(); cell = atoms.get_cell()
    csp = np.zeros(len(atoms))
    for i in range(len(atoms)):
        idx, off = nl.get_neighbors(i)
        if len(idx) < nn:
            csp[i] = np.nan; continue
        v = pos[idx] + off @ cell - pos[i]
        d = np.linalg.norm(v, axis=1); order = np.argsort(d)[:nn]
        v = v[order]
        used = np.zeros(nn, bool); s = 0.0
        for a in range(nn):
            if used[a]: continue
            # pair with the most anti-parallel remaining vector
            best=-1; bestval=1e9
            for b in range(nn):
                if b==a or used[b]: continue
                val = np.sum((v[a]+v[b])**2)
                if val < bestval: bestval=val; best=b
            if best>=0:
                used[a]=used[best]=True; s += bestval
        csp[i] = s
    return csp

def try_ovito(fn):
    try:
        from ovito.io import import_file
        from ovito.modifiers import PolyhedralTemplateMatchingModifier as PTM
    except Exception:
        return None
    pipe = import_file(fn)
    ptm = PTM(); ptm.output_orientation = False
    # enable FCC/HCP/BCC structure identification
    for t in ptm.structures: t.enabled = True
    pipe.modifiers.append(ptm)
    data = pipe.compute()
    counts = data.attributes
    out = {}
    for k in ["PolyhedralTemplateMatching.counts.FCC",
              "PolyhedralTemplateMatching.counts.HCP",
              "PolyhedralTemplateMatching.counts.OTHER",
              "PolyhedralTemplateMatching.counts.BCC"]:
        out[k.split('.')[-1]] = counts.get(k, 0)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("structure")
    ap.add_argument("--normal", type=int, default=0, help="GB normal axis 0=x,1=y,2=z")
    a=ap.parse_args()
    atoms=read(a.structure)
    n=a.normal; ip=[i for i in range(3) if i!=n]
    csp=centrosymmetry(atoms)

    # OVITO structure typing (definitive) if available
    ov=try_ovito(a.structure)
    if ov:
        N=len(atoms)
        print("OVITO PTM structure fractions:")
        for k,v in ov.items(): print(f"  {k:6s}: {v:5d}  ({100*v/N:4.1f}%)")
        print("Coherent Sigma3(111) twin -> mostly FCC + a thin HCP plane.")
    else:
        print("(OVITO python module not found — using CSP image only. "
              "For the definitive FCC/HCP test, open the file in the OVITO GUI "
              "and apply Common Neighbor Analysis / PTM.)")

    # CSP side-view: project onto (normal, first in-plane) axis
    pos=atoms.get_positions(); c=csp.copy()
    finite=np.isfinite(c); c[~finite]=np.nanmax(c[finite])
    plt.figure(figsize=(8,4.5))
    sc=plt.scatter(pos[:,n], pos[:,ip[0]], c=c, cmap="coolwarm", s=12,
                   vmin=0, vmax=np.nanpercentile(c[finite],98))
    plt.colorbar(sc,label="centrosymmetry (bulk~0, GB high)")
    plt.xlabel(f"GB-normal axis {n} (Å)"); plt.ylabel(f"in-plane axis {ip[0]} (Å)")
    plt.title(f"{a.structure}: GB planes appear as red lines of high-CSP atoms")
    plt.tight_layout()
    out=a.structure.rsplit('.',1)[0]+"_inspect.png"
    plt.savefig(out,dpi=160); print(f"saved {out}")

if __name__=="__main__":
    main()
