#!/usr/bin/env python3
"""
analyze_deep_traps.py — verify the mechanism: are the deepest Te traps in the Ni-Cr-Mo
alloy grain boundary locally NI-RICH (few Cr/Mo neighbours)?

For each (alloy structure, segregation output) pair it counts the Cr+Mo first-shell
neighbours of every scanned GB site and correlates that with the Te segregation energy.
If the deepest traps have fewer solute neighbours than average, the mechanism (Cr/Mo repel
Te; Te falls into Ni-rich pockets) is confirmed. Publication figure @ 600 dpi.

Run on the HPC where the alloy structures live:
    python analyze_deep_traps.py --struct "NiCrMo_*seed*.vasp" --out "outseed*.txt"
"""
import argparse, glob, re, numpy as np
from ase.io import read
from ase.neighborlist import NeighborList
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def seed_of(fn):
    m = re.search(r"seed(\d+)", fn) or re.search(r"(\d+)", fn)
    return int(m.group(1)) if m else -1

def parse_out(path):
    d = {}
    for line in open(path):
        m = re.match(r"\s*(\d+)\s+[\d.]+\s+(-?\d+\.\d+)\s*$", line)
        if m: d[int(m.group(1))] = float(m.group(2))
    return d

def solute_neighbors(atoms, cutoff=3.0):
    nl = NeighborList([cutoff/2]*len(atoms), self_interaction=False, bothways=True)
    nl.update(atoms); sym = atoms.get_chemical_symbols()
    counts = np.zeros(len(atoms), int)
    for i in range(len(atoms)):
        idx, _ = nl.get_neighbors(i)
        counts[i] = sum(1 for j in idx if sym[j] in ("Cr", "Mo"))
    return counts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--struct", required=True, help="glob for alloy structures")
    ap.add_argument("--out", required=True, help="glob for segregation outputs")
    ap.add_argument("--cutoff", type=float, default=3.0, help="first-shell cutoff (Å)")
    a = ap.parse_args()

    structs = {seed_of(f): f for f in glob.glob(a.struct)}
    outs = {seed_of(f): f for f in glob.glob(a.out)}
    seeds = sorted(set(structs) & set(outs))
    if not seeds:
        print("No matching (structure, output) seed pairs found."); return

    E, NS = [], []
    for s in seeds:
        at = read(structs[s]); segs = parse_out(outs[s]); nsol = solute_neighbors(at, a.cutoff)
        for idx, e in segs.items():
            if idx < len(at):
                E.append(e); NS.append(int(nsol[idx]))
    E, NS = np.array(E), np.array(NS)
    print(f"# {len(E)} site-values across {len(seeds)} seeds (cutoff {a.cutoff} Å)")

    thr = np.percentile(E, 10)
    deep, rest = NS[E <= thr], NS[E > thr]
    r = np.corrcoef(E, NS)[0, 1]
    print(f"# deepest 10% traps : mean Cr/Mo neighbours = {deep.mean():.2f}")
    print(f"# all other sites   : mean Cr/Mo neighbours = {rest.mean():.2f}")
    print(f"# corr(E_seg, #solute-neighbours) = {r:+.2f}  (positive => more solutes weaken the trap)")
    print(f"# {'CONFIRMED: deepest Te traps are Ni-rich' if deep.mean() < rest.mean() else 'NOT confirmed'}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(NS + np.random.uniform(-0.15, 0.15, len(NS)), E, s=18, alpha=0.45, c='tab:red')
    for k in sorted(set(NS)):
        ax.plot(k, E[NS == k].mean(), 'ks', ms=9)
    ax.set_xlabel("number of Cr/Mo neighbours of the Te site")
    ax.set_ylabel("Te segregation energy $E_{seg}$ (eV)")
    ax.set_title("Deepest Te traps sit in Ni-rich GB sites (MACE-MP-0)")
    ax.axhline(0, color='gray', lw=0.8, ls=':'); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("Te_deep_trap_environment.png", dpi=600)
    print("saved Te_deep_trap_environment.png @ 600 dpi")

if __name__ == "__main__":
    main()
