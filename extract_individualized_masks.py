"""
Extract boolean masks from individualized HDF5 files.

Usage:
    python extract_individualized_masks.py
    python extract_individualized_masks.py --ranking_method r2
"""

import argparse
import h5py
import numpy as np
from pathlib import Path


def extract_mask_from_hdf5(subject_id, individualized_hdf5_path, brain_mask_path, output_dir):
    subj_str = f"subj{subject_id:02d}"

    brain_mask = np.load(str(brain_mask_path))
    num_brain_voxels = int(brain_mask.sum())
    print(f"Subject {subject_id:02d}: {num_brain_voxels:,} brain voxels")

    with h5py.File(str(individualized_hdf5_path), 'r') as f:
        # load indices
        selected_voxel_indices = f['selected_voxel_indices'][:]
    print(f"  {len(selected_voxel_indices):,} selected voxels, "
          f"index range [{selected_voxel_indices.min()}, {selected_voxel_indices.max()}]")

    # initialize mask with 0/False
    individualized_mask = np.zeros(num_brain_voxels, dtype=bool)

    # set selected voxels to 1/True
    individualized_mask[selected_voxel_indices] = True

    # check
    assert individualized_mask.sum() == len(selected_voxel_indices), \
        "Mask True count doesn't match number of indices!"

    output_dir.mkdir(parents=True, exist_ok=True)

    # save mask
    mask_output_path = output_dir / f"glasser_betas_all_{subj_str}_fp32_renorm_mask.npy"
    np.save(str(mask_output_path), individualized_mask)

    size_kb = mask_output_path.stat().st_size / 1024
    print(f"  Saved {mask_output_path.name} — {size_kb:.1f} KB")


def main():
    parser = argparse.ArgumentParser(description="Extract boolean masks from individualized HDF5 files")
    parser.add_argument('--ranking_method', type=str, default='r2',
                        choices=['r2', 'variance', 'mean_abs'])
    parser.add_argument('--individualized_dir', type=str,
                        default=r'\path\individualized_masks\Glasser')
    parser.add_argument('--brain_masks_dir', type=str,
                        default=r'\path\processed\brain_masks')
    parser.add_argument('--output_dir', type=str,
                        default=r'\path\individualized_masks\boolean_masks')
    args = parser.parse_args()

    for subject_id in [1, 2, 5, 7]:
        subj_str = f"subj{subject_id:02d}"
        individualized_hdf5 = Path(args.individualized_dir) / f"glasser_betas_all_{subj_str}_fp32_renorm.hdf5"
        brain_mask_path = Path(args.brain_masks_dir) / f"brain_mask_{subj_str}.npy"

        if not individualized_hdf5.exists() or not brain_mask_path.exists():
            print(f"Skipping subject {subject_id:02d} — files not found")
            continue

        extract_mask_from_hdf5(subject_id, individualized_hdf5, brain_mask_path, Path(args.output_dir))


if __name__ == "__main__":
    main()