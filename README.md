# simassign

`simassign` is a python package designed to loosely wrap around the DESI pacakages [`fiberassign`](https://github.com/desihub/fiberassign) and [`desitarget`](https://github.com/desihub/desitarget/) to simulate the fiber assignment / Merged Target Ledger (MTL) update loop.  `simassign` provides functionality to simulate target assignments for different fields, randomly distributed targets, and target types outside of those defined in the DESI survey (c.f. the DESI [targetmask](https://github.com/desihub/desitarget/blob/main/py/desitarget/data/targetmask.yaml) for more details on DESI targets). `simassign` also allows simulation of DESI targets with different survey strategies than those chosen in DESI.

## Design Philosophy
`simassign` should be straightforward and well documented.

## How it Works

`simassign` is a package of code, but most of the package functions are designed to support a set of scripts included in bin/.
The general code flow of the scripts is outlined below.

### 1. Generate a Tiling File
First you want to generate a tiling file:
```
python generate_tiling_file.py --npass 20 -o tiles-desi2-20pass-offset-tiles-5000deg-superset.ecsv --trim [--fourex] [--collapse]
```

This script will load the DESI tiling, and generate up to a maximum of ~60 passes of the entire skyof tilings with unique tile centers.
These are generated as the first 15 passes defined in the DESI survey design, and an additional 45 defined by further rotations on a grid.

--trim ensures that the catalog is "trimmed" such that IN_DESI is only True for tiles that lie within the nominal DESI-II area.
For more details on what --fourex and --collapse do see the script's help message.

### Other steps:
TBW