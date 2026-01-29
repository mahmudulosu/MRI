# MRI Age-Group Classification (Young vs Old)

This repository contains two pipelines to classify **age group** (*young* vs *old*) using structural MRI:

## 1) Classical ML (Random Forest) on Voxel Count Features
Uses a CSV of precomputed voxel counts:
- `WM_voxels` (white matter)
- `GM_voxels` (gray matter)
- `CSF_voxels` (CSF)

Workflow:
1. Load voxel feature CSV
2. Load participant CSV with `participant_id` and `age`
3. Map age ranges into classes: `young` vs `old`
4. Merge on `participant_id`
5. Train/test split and RandomForest training
6. Print accuracy and classification report; save model + metrics

## 2) Deep Learning (3D-CNN) on Full MRI Volumes (NIfTI)
Uses `.nii` / `.nii.gz` volumes and trains a simple 3D CNN.

Workflow:
1. Load participant CSV with `participant_id` and `age`
2. Map age ranges to `young` vs `old`, then encode to labels
3. Match MRI filenames to participant IDs:
   - participant_id is extracted as: `filename.split('_')[0]`
4. Load NIfTI volumes, normalize, resize to a target 3D shape
5. Train a 3D CNN; evaluate and save the model

---

## Age mapping used

Valid age bins:
- `20-25`, `25-30` → **young**
- `60-65`, `65-70`, `70-75` → **old**

---

## Requirements

- Python 3.9+ recommended
- Key packages: numpy, pandas, scikit-learn, matplotlib, seaborn, nibabel, scipy, tqdm
- For 3D-CNN: tensorflow (GPU strongly recommended)

Install:
```bash
pip install -r requirements.txt
