# simassign

`simassign` is a python package designed to loosely wrap around the DESI pacakages [`fiberassign`](https://github.com/desihub/fiberassign) and [`desitarget`](https://github.com/desihub/desitarget/) to simulate the fiber assignment / Merged Target Ledger (MTL) update loop.  `simassign` provides functionality to simulate target assignments for different fields, randomly distributed targets, and target types outside of those defined in the DESI survey (c.f. the DESI [targetmask](https://github.com/desihub/desitarget/blob/main/py/desitarget/data/targetmask.yaml) for more details on DESI targets). `simassign` also allows simulation of DESI targets with different survey strategies than those chosen in DESI.

## Design Philosophy
`simassign` should be straightforward and well documented.

## How it Works

`simassign` is both a package of helper functions and a selection of scripts. The main processing script is `run_survey_mp.py`, which will run and simulate a survey using python multiprocessing for the fiberassignment. `simassign` tries to accurately mimic the true survey loop as accurately as possible, which means that it takes the input catalog of objects and generates a true merged targeting ledger (MTL) which is updated every "day" with the results of the previous "night"'s fiberassignment[^1].

`run_survey_mp` requires only three things to fully simulate a survey:
1. A file of exposures. This file can be either an ecsv or a fits file, and should be a tabular dataset. The exposures file must include a few specific columns, but can contain any set of columns that are a superset of these:
    - `TIMESTAMP`: The timestamp at which the tile exposed; passed through to fiberassign for focal plane status
    - `DESIGNHA`: Design hour angle of the tile; passed through to fiberassign
    - `TIMESTAMP_YMD`: The night the tile is exposed on; used for deduplicating multiple exposures of the same tile on any given night.
    - `TILEID`: The TILEID exposed in this exposure.
    - `PROGRAM`: e.g. "dark", "bright". Used by fiberassign to determine targets.
    - `OBSCONDITIONS`: Integer value corresponding to the `PROGRAM` type. Used by fiberassign to determine targets.
    - `TILEDONE`: whether this tile is completed in this exposure or not.
    - `RA` and `DEC`: center of the tile exposed

    `run_fiberassign_mp.py` deduplicates the exposure file by `(TILEID, TIMESTAMP_YMD)` pairs to ensure that the multiprocessing does not fiberassign the same tile on multiple nights. Tiles are allowed to appear on multiple nights, use `TILEDONE` to indicate whether this is a precompletion night or not. The general principal is that a tile will be fiberassigned on a night (if the fiberassign file does not already exist) any time that `TILEID` occurs on a unique `TIMSTAMP_YMD`, but the results of that "observation" are only incorporated into MTL updates on nights where `TILEDONE`=TRUE. This mimics true survey behaviour, where a tile is fiber assigned when it is first observed, but only QAd after it finishes its entire effective time.

    A helper script, `process_exposure_table.py`, is provided to process the output of a `surveysim` simulation into a form readable by `simassign`. It takes as input the output `exposures` file produced by  `surveysim` as well as the input `tiles` file to add any additional necessary columns outlined above.
2. A configuration file. This configuration file must be structured like the desi `targetmask.yaml` but can nominally have any name. It must include definitions for each target, its required number of exposures, and a priority progression. Right now, `simassign` only supports three priority values: `DONE`, `UNOBS` and `MORE_ZGOOD`. Other values in the configuration file will be ignored.
3. A catalog of targets. The catalog of targets needs the following three columns, although as usual more can be provided and ignored by `simassign`:
    - `RA`, `DEC`: Self explanatory
    - `DESITARGET`: The DESI targeting bit, whose information is stored in the configuration file.

    Optionally, the file can include the columns `QSO_MASK`, `LBG_MASK` or `XLG_MASK` to define subdivisions within each target class. These columns will be ingested by `simassign` with the requisite information propagated to the MTLs, but only if such information is also included in the configuration file.

`run_survey_mp` has a variety of other optional requirements, some of the more notable ones are briefly detailed here:
- `--stds $VAL`: A catalog of standard stars to use in fiberassignment. This improves the accuracy of the fiberassignment.
- `--catalog_b`: An optional second catalog of objects, with the same data model as the first. If provided, it is also required to pass `--b_start_date`, a YMD timestamp on which to add these targets to the full catalog. This feature allows targets to be "turned on" after a specific date.

[^1]: Due to some limitations and design choices, targets are not split by PROGRAM, instead all targets enter a single MTL stored in `hp/main/dark/`. If this doesn't mean anything to you, then don't worry, because it's not important to you.

### Worked Example
In this subsection I will briefly breakdown the following `simassign` call:

```python run_survey_mp.py -o $OUTPARENT --catalog $CATALOG --nproc 32 --tiles $TILES --stds $STANDARDS --danger --config targetmask.yaml```

This call will run the survey using 32 multiprocessing processes (`--nproc 32`). The tile exposures are saved in `$TILES`, and the input target catalog is stored in `$CATALOG`. The configuration file is `targetmask.yaml`. The catalog of standard stars to propagate to fiberassign is stored in `$STDS`. `--danger` means that the script should run in danger mode, that is, use algorithms that are as fast as possible, but also slightly unstable or otherwise dangerous. Right now, this switch only turns off saving the full MTL every day, instead saving it only once a year.

## Etc.
`simassign` also provides some other helper scripts to ease generating the required inputs. See individual scripts for more details, documentation of those scripts in the readme is TBW.