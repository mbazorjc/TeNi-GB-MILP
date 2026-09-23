#!/usr/bin/env python3
"""
generate_reference_GBs_aimsgb.py
Build the EXACT coincidence-site-lattice (CSL) grain boundaries used in
Huai et al., RSC Adv. 7 (2017) 8421 (and Liu et al., Comput. Mater. Sci. 88 (2014) 22),
so we can REPLICATE their Te-segregation results before extending to your topic.

Why this and not a hand-rolled mirror: a simple mirror does NOT produce a coherent
Sigma3(111) twin (you get a wrong, high-energy interface). aimsgb builds proper CSL GBs.

INSTALL (you already have pymatgen):
    pip install aimsgb
RUN:
    python generate_reference_GBs_aimsgb.py
Outputs POSCAR files for each reference GB, ready for relaxation with MACE-MP-0.
"""
from pymatgen.core import Lattice, Structure
from aimsgb import Grain, GrainBoundary

A0 = 3.52  # Ni lattice constant (use your relaxed value for production)

# conventional 4-atom fcc Ni cell
ni = Structure(Lattice.cubic(A0), ["Ni"]*4,
               [[0,0,0],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5]])
# Try the recommended class method, fallback to explicit construction
if hasattr(Grain, 'from_structure'):
    ni_grain = Grain.from_structure(ni)
else:
    # Explicit: lattice, species (list), fractional coordinates
    ni_grain = Grain(lattice=ni.lattice, species=ni.species, coords=ni.frac_coords)
#gb = GrainBoundary(axis=p["axis"], sigma=p["sigma"], plane=p["plane"],#
#                   initial_struct=ni_grain, uc_a=1, uc_b=1)
# Reference GBs: (rotation axis, sigma, GB plane) — matching Huai et al. 2017
# Verify available planes with aimsgb.GBInformation(axis, sigma) if a plane is rejected.
GBS = {
    #"Ni_S5_021":  dict(axis=[1,0,0], sigma=5,  plane=[0,2,1]),   # strongest Te segregation
    #"Ni_S3_111":  dict(axis=[1,1,0], sigma=3,  plane=[1,1,1]),   # weakest (coherent twin)
    #"Ni_S9_221":  dict(axis=[1,1,0], sigma=9,  plane=[2,2,1]),   # good for diffusion (unstable w/ Te)
    #"Ni_S11_113": dict(axis=[1,1,0], sigma=11, plane=[1,1,3]),   # fastest GB diffusion
    "Ni_S5_021":  dict(axis=[1,0,0], sigma=5,  plane=[0,1,2]),   # corrected from (0,2,1)
    "Ni_S3_111":  dict(axis=[1,1,1], sigma=3,  plane=[1,1,1]),   # twist boundary (coherent twin
    "Ni_S9_221":  dict(axis=[1,1,0], sigma=9,  plane=[-2,2,1]),   # corrected
    "Ni_S11_113": dict(axis=[1,1,0], sigma=11, plane=[1,-1,3]),   # corrected
}

for tag, p in GBS.items():
    try:
        gb = GrainBoundary(axis=p["axis"], sigma=p["sigma"], plane=p["plane"],
                           initial_struct=ni_grain, uc_a=1, uc_b=1)
        s = gb.build_gb()
        s.to(filename=f"{tag}_POSCAR.vasp", fmt="poscar")
        print(f"{tag}: {len(s)} atoms  axis={p['axis']} sigma={p['sigma']} plane={p['plane']}  -> {tag}_POSCAR.vasp")
    except Exception as e:
        print(f"{tag}: FAILED ({e}). Check valid planes via aimsgb.GBInformation(axis,sigma).")

print("\nNext: relax each GB with MACE-MP-0, then run segregation_energy_mace.py on each.")
print("Replication target (Huai 2017): Te segregation strongest at S5(021), weakest at S3(111).")
