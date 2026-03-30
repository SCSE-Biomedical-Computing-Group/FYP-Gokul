"""
Create individualized visual ROI masks.
Takes voxels from ICSC-identified visual modules, ranks it according to R2 score, 
and selects top N to match NSDGeneral dimensions for pretrained MindAligner.

Usage:
    python create_individualized_masks.py --subjects 1
    python create_individualized_masks.py --subjects 1 2 5 7
"""

import argparse
import h5py
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path


REQUIRED_VOXEL_COUNTS = {
    1: 15724,
    2: 14278,
    5: 13039,
    7: 12682
}

VISUAL_MODULES = {
    1: [1, 7],
    2: [2, 9],
    5: [6],
    7: [0, 6]
}

SUBJECT_MAP = {0: 1, 1: 2, 2: 5, 3: 7}


# ranking methods

def rank_by_r2(r2_values, visual_voxel_mask):
    scores = r2_values.copy()
    scores[~visual_voxel_mask] = -np.inf
    ranked = np.argsort(scores)[::-1]
    return ranked[visual_voxel_mask[ranked]]


def rank_by_variance(task_betas, visual_voxel_mask):
    scores = np.var(task_betas, axis=0)
    scores[~visual_voxel_mask] = -np.inf
    ranked = np.argsort(scores)[::-1]
    return ranked[visual_voxel_mask[ranked]]


def rank_by_mean_abs_beta(task_betas, visual_voxel_mask):
    scores = np.mean(np.abs(task_betas), axis=0)
    scores[~visual_voxel_mask] = -np.inf
    ranked = np.argsort(scores)[::-1]
    return ranked[visual_voxel_mask[ranked]]



# processing functions

def load_visual_module_voxels(subject_id, module_labels, atlas_path, brain_mask_path):
    
    # load brain mask and atlas
    brain_mask = np.load(str(brain_mask_path))
    atlas_flat = nib.load(str(atlas_path)).get_fdata().astype(int)[brain_mask]

    # load ids of rois in visual modules
    visual_roi_ids = np.where(np.isin(module_labels, VISUAL_MODULES[subject_id]))[0] + 1

    # mask with only voxels in visual modules
    visual_voxel_mask = np.isin(atlas_flat, visual_roi_ids)

    print(f"Visual modules: {VISUAL_MODULES[subject_id]}, {len(visual_roi_ids)} ROIs, {visual_voxel_mask.sum():,} voxels")
    return visual_voxel_mask


def load_r2_values(r2_path, brain_mask_path):

    # get R2 values from NSD raw data
    brain_mask = np.load(str(brain_mask_path))
    r2_flat = nib.load(str(r2_path)).get_fdata()[brain_mask]
    print(f"R2 range: [{r2_flat.min():.4f}, {r2_flat.max():.4f}], mean: {r2_flat.mean():.4f}")
    return r2_flat


def select_top_voxels(ranking_method, visual_voxel_mask, r2_values, task_betas, num_voxels_needed):
    
    # default is r2
    if ranking_method == 'r2':
        ranked = rank_by_r2(r2_values, visual_voxel_mask)
    elif ranking_method == 'variance':
        ranked = rank_by_variance(task_betas, visual_voxel_mask)
    elif ranking_method == 'mean_abs':
        ranked = rank_by_mean_abs_beta(task_betas, visual_voxel_mask)
    elif ranking_method == 'combined':
        ranked = rank_by_combined(r2_values, task_betas, visual_voxel_mask)
    else:
        raise ValueError(f"Unknown ranking method: {ranking_method}")

    selected = ranked[:num_voxels_needed]
    selected_r2 = r2_values[selected]

    # stats of selected voxels
    print(f"Selected {len(selected):,} voxels — R2 mean: {selected_r2.mean():.4f}, "
          f"median: {np.median(selected_r2):.4f}, range: [{selected_r2.min():.4f}, {selected_r2.max():.4f}]")
    return selected


def apply_mask_and_save(subject_id, selected_indices, task_data_path, output_dir, ranking_method):
    subj_str = f"subj{subject_id:02d}"

    # apply mask on task data
    with h5py.File(str(task_data_path), 'r') as f_in:
        individualized_betas = f_in['betas'][:, selected_indices]
    print(f"Masked shape: {individualized_betas.shape}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"individualized_{ranking_method}_{subj_str}_unordered.hdf5"

    with h5py.File(str(output_path), 'w') as f_out:
        dset = f_out.create_dataset('betas', data=individualized_betas, dtype=np.float32,
                                    compression='gzip', compression_opts=4)
        
        dset.attrs['subject'] = subject_id
        dset.attrs['num_trials'] = individualized_betas.shape[0]
        dset.attrs['num_voxels'] = individualized_betas.shape[1]
        dset.attrs['ranking_method'] = ranking_method
        dset.attrs['visual_modules'] = VISUAL_MODULES[subject_id]
        dset.attrs['description'] = (
            f"Individualized visual ROI mask for subject {subject_id:02d}, "
            f"selected using {ranking_method} ranking."
        )

        # save the selected voxel indices
        f_out.create_dataset('selected_voxel_indices', data=selected_indices, compression='gzip')

    size_mb = output_path.stat().st_size / (1024**2) # size in MB
    print(f"Saved {output_path.name} — {size_mb:.1f} MB")
    return output_path


def process_subject(subject_id, icsc_results, atlas_dir, brain_masks_dir,
                    r2_dir, task_data_dir, output_dir, ranking_method='r2'):
    
    subj_str = f"subj{subject_id:02d}"
    print(f"\nSubject {subject_id:02d}, ranking method: {ranking_method}")

    run_id = {j: i for i, j in SUBJECT_MAP.items()}[subject_id]
    module_labels = icsc_results.iloc[run_id, 5:].values.astype(int)

    # paths to all needed files
    atlas_path = atlas_dir / f"subject_{subject_id}" / "HCP_MMP1.nii.gz"
    brain_mask_path = brain_masks_dir / f"brain_mask_{subj_str}.npy"
    r2_path = r2_dir / subj_str / "R2.nii.gz"
    task_data_path = task_data_dir / f"full_brain_{subj_str}_all.hdf5"

    # check all files exist
    for path, name in [
        (atlas_path, "Atlas"),
        (brain_mask_path, "Brain mask"),
        (r2_path, "R2 file"),
        (task_data_path, "Task data")
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")
    
    # load visual modules voxels
    visual_voxel_mask = load_visual_module_voxels(subject_id, module_labels, atlas_path, brain_mask_path)
    
    # get R2 values
    r2_values = load_r2_values(r2_path, brain_mask_path)

    task_betas = None
    if ranking_method in ['variance', 'mean_abs', 'combined']:
        with h5py.File(str(task_data_path), 'r') as f:
            task_betas = f['betas'][:]

    selected_indices = select_top_voxels(
        ranking_method, visual_voxel_mask, r2_values, task_betas, REQUIRED_VOXEL_COUNTS[subject_id]
    )

    return apply_mask_and_save(subject_id, selected_indices, task_data_path, output_dir, ranking_method)


def main():
    parser = argparse.ArgumentParser(description="Create individualized visual ROI masks")
    parser.add_argument('--subjects', type=int, nargs='+', required=True, choices=[1, 2, 5, 7])
    parser.add_argument('--ranking_method', type=str, default='r2',
                        choices=['r2', 'variance', 'mean_abs', 'combined'])
    parser.add_argument('--icsc_results', type=str,
                        default=r'\path\ICSC_data\glasser_ICSC_data\subject_level_results\ICSC_subject_level_final_iter.csv')
    parser.add_argument('--atlas_dir', type=str, default=r'\path\atlas\Glasser')
    parser.add_argument('--brain_masks_dir', type=str, default=r'\path\processed\brain_masks')
    parser.add_argument('--r2_dir', type=str, default=r'\path\raw_nsd')
    parser.add_argument('--task_data_dir', type=str, default=r'\path\processed')
    parser.add_argument('--output_dir', type=str, default=r'\path\individualized_masks\Glasser')
    args = parser.parse_args()

    icsc_results = pd.read_csv(args.icsc_results, header=None)

    for subject_id in args.subjects:
        process_subject(
            subject_id=subject_id,
            icsc_results=icsc_results,
            atlas_dir=Path(args.atlas_dir),
            brain_masks_dir=Path(args.brain_masks_dir),
            r2_dir=Path(args.r2_dir),
            task_data_dir=Path(args.task_data_dir),
            output_dir=Path(args.output_dir),
            ranking_method=args.ranking_method
        )


if __name__ == "__main__":
    main()