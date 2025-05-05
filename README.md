# Validation of the helical parameter of the helical assembly entries in the EMDB

This repository contains the validated value of the helical parameter in the EMDB and the code of automatic curation. 

## Column Name

**emdb_id**: The EMD entry accession code with the format of EMD-XXXXX

**group**: This group labels whether it is an amyloid structure or not based on the title name in the EMDB meta data

**resolution (Å)**: The deposited resolution in the EMDB (not the validated resolution)
**rise_deposited (Å)**: The origianl deposited rise value

**twist_deposited (°)**: The original deposited twist value

**csym_deposited**: The original deposited axis symmetry

**rise_curated (Å)**: The curated rise value

**twist_curated (°)**: The curated twist value

**csym_curated**: The curated cyclic symmetry

**vector difference**: The vector difference between the original helical parameter and deposited helical parameter based on the estimated radius

**axes order**: Determine whether the z axis is aligned with the helical axis

**cc_emdb**: The cross correlation score between the symmetrized map using the original helical parameter with the original density

**cc_curated**: The cross correlation score between the symmetrized map using the curated helical parameter with the original density

**validated**: Whether this map can be validated or not, some cases using focus reconstruction or only one asym unit is deposited can not be validated

**update**: Whether there is significant difference of the cc value using the original helical parameter of curated helical parameter

**reason**: The potential error of the map.

## Reasons

**deposited**: the curated value is aligned well with the original value

## How to curate the EMDB helical database

Below shows the steps of estimating the helical parameters in the database. Before running the code, please source the environment of [helicon](https://github.com/jianglab/helicon)

### Find the EMDB accession number that need to be curated

```
python ./code/check_emdb.py
```

This will create csv file /file/need_curation.csv listing the EMD entries that need to be curated

### Run the auto HI3D

```
python ./code/auto_hi3d.py
```

This will create two csv files /file/validated.csv and /file/non_validated.csv, the validated.csv include all the entries that is consistent with the original helical parameters, non_validated.csv include the entries that is not consistent with the original helical parameters that need to be validated. 

### Manual validation using the HI3D web app

Please manually check through the EMDB entries in the /file/non_validated.csv using [HI3D](https://jiang.bio.purdue.edu/hi3d/)
