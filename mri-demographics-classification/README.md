# MRI Demographics Classification (Gender + Age×Gender)

This repository provides classical ML pipelines to classify demographic traits from structural MRI:

## Tasks
### 1) Gender classification (2-class)
- Classes: Female (0), Male (1)

### 2) Age×Gender classification (4-class)
- Classes:
  - 0: young_M
  - 1: young_F
  - 2: old_M
  - 3: old_F

## Feature families
### A) ROI features (Atlas-based)
Uses Harvard-Oxford atlas and `NiftiLabelsMasker` to extract regional mean intensities.

### B) PCA voxel features
Resamples MRIs to 2mm isotropic, flattens voxel intensities, then applies PCA.

### C) Morphometric features (FreeSurfer-like proxies)
Simple morphometry-style features computed from the MRI volume:
- cortical-shell proportion (proxy thickness)
- left/right inner (eroded) brain volume
- total brain volume

## Label source
Labels are read from:
`participants_LSD_andLEMON.csv`

Required columns:
- `participant_id`
- `gender` (M/F)
- `age` (bins like 20-25, 60-65, etc.)

Age bins used:
- young: 20–30 → (20-25, 25-30)
- old: 60–75 → (60-65, 65-70, 70-75)

## MRI filename requirement
Participant IDs are extracted from MRI filename prefix:
`subj_id = filename.split('_')[0]`

## How to run
Example:
```bash
python scripts/gender_roi_svm.py --data_folder "C:\research\MRI\structural_MRI" --csv_path "C:\research\MRI\participants_LSD_andLEMON.csv"
