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

This script will load the DESI tiling, and generate up to a maximum of ~60 passes of the entire skyof tilings with unique tile centers. These are generated as the first 15 passes defined in the DESI survey design, and an additional 45 defined by further rotations on a grid.

--trim ensures that the catalog is "trimmed" such that IN_DESI is only True for tiles that lie within the nominal DESI-II area. For more details on what --fourex and --collapse do see the script's help message.

The generated tile file is structured for interop with surveysim, and includes one dummy BRIGHT and one dummy BACKUP tile. The tile file can be passed directly to surveysim without any further processing.

### 2. Simulate a Survey
See `surveysim` to do this step.

### 3. Postprocess Simulated Survey
The output of the simulated survey needs to be minorly post processed to be run through `simassign`.

```
 python process_exposures_table.py -o exposures_processed_offset-tiles-5000deg-20pass.fits -e exposures_offset-tiles-500deg-20pass.fits -t tiles-desi2-20pass-offset-tiles-5000deg-superset.ecsv
```

`process_exposures_table` will take in the output exposures of the simulated survey (through `-e`/`--exposures`) and match the tiles to the master list of tiles, `-t`/`--tiles`. It propogates from the tiles file some tile details lost in the exposures file, namely `OBSCONDITIONS`. The processing also adds three columns necessary for simulated fiber assignming and mtl updates: `TIMESTAMP` and `TIMESTAMP_YMD` (calculated from the julien date of the observation, using astropy). The former is used for loading focalplane state in fiber assignment while the latter is used for multiprocessing over observation nights. `TILEDONE` is calculated from the `SNR2FRAC`, and is used to determine when the results of that tile are used for mtl updates.

## 4. Simulate Fiber Assignment

Do the thing.


### Other steps:
TBW