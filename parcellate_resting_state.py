"""

Parcellate resting state fMRI data to ROIs using HCP_MMP1 atlas.
Reduces voxel-level data to ROI-level for running ICSC algorithm.

Input:
    - resting_state_subj0X_all.hdf5: [timepoints, num_voxels]
    - HCP_MMP1.nii.gz atlas: (spatial_dims) with ROI labels 1-180
    - brain_mask_subj0X.npy: (spatial_dims) boolean mask

Output:
    - resting_state_subj0X_parcellated.hdf5: [timepoints, 180]

Usage:
    python parcellate_resting_state.py --subjects 1
    python compute_session_correlations.py --subjects 1 2 5 7

"""

import argparse
import h5py
import nibabel as nib
import numpy as np
from pathlib import Path


def parcellate_to_rois(subject, resting_hdf5_path, atlas_path, brain_mask_path, output_dir):
    
    subj_str = f"subj{subject:02d}"

    # load brain mask
    brain_mask = np.load(str(brain_mask_path))
    spatial_dims = brain_mask.shape
    num_voxels = int(brain_mask.sum())
    print(f"Brain mask: {spatial_dims}, {num_voxels:,} voxels")

    # load the HCP_MMP1 atlas
    atlas_img = nib.load(str(atlas_path))
    atlas_data = atlas_img.get_fdata().astype(int)

    # ensure atlas and mask have the same shape
    if atlas_data.shape != spatial_dims:
        raise ValueError(f"Atlas shape {atlas_data.shape} doesn't match brain mask shape {spatial_dims}")

    # get roi labels according to the atlas
    unique_rois = np.unique(atlas_data)
    unique_rois = unique_rois[unique_rois > 0]
    num_rois = len(unique_rois)
    print(f"Atlas: {num_rois} ROIs, labels {unique_rois.min()} to {unique_rois.max()}")

    # flatten atlas
    atlas_flat = atlas_data[brain_mask]
    print(f"Flattened atlas: {atlas_flat.shape}")

    # load resting state data
    with h5py.File(str(resting_hdf5_path), 'r') as f:
        voxel_ts = f['betas'][:]
        num_timepoints = voxel_ts.shape[0]
        print(f"Time series shape: {voxel_ts.shape}")

    # check dimensions match
    if voxel_ts.shape[1] != num_voxels:
        raise ValueError(f"Voxel count mismatch: HDF5 has {voxel_ts.shape[1]}, brain mask has {num_voxels}")

    print(f"\nParcellating to {num_rois} ROIs:")

    # initialize                                    
    roi_ts = np.zeros((num_timepoints, num_rois), dtype=np.float32)
    rois_present, rois_empty = [], []

    for roi_idx, roi_label in enumerate(unique_rois):
        
        # find voxels in roi
        voxels_in_roi = (atlas_flat == roi_label)
        if voxels_in_roi.sum() > 0:
            # get average of all voxels in this roi
            roi_ts[:, roi_idx] = voxel_ts[:, voxels_in_roi].mean(axis=1)
            rois_present.append(roi_label)
        else:
            rois_empty.append(roi_label)

        if (roi_idx + 1) % 50 == 0:
            print(f"  {roi_idx + 1}/{num_rois} ROIs done...")

    print(f"ROIs with data: {len(rois_present)}, empty: {len(rois_empty)}")
    if rois_empty:
        print(f"Empty ROI labels: {rois_empty[:10]}{'...' if len(rois_empty) > 10 else ''}")

    # save parcellated data
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"resting_state_{subj_str}_parcellated.hdf5"

    with h5py.File(str(output_path), 'w') as f_out:
        dset = f_out.create_dataset('roi_timeseries', data=roi_ts, dtype=np.float32,
                                    compression='gzip', compression_opts=4)
        
        dset.attrs['subject'] = subject
        dset.attrs['num_timepoints'] = num_timepoints
        dset.attrs['num_rois'] = num_rois
        dset.attrs['atlas_name'] = 'HCP_MMP1'
        dset.attrs['rois_present'] = len(rois_present)
        dset.attrs['rois_empty'] = len(rois_empty)
        dset.attrs['description'] = (
            f"Resting state fMRI parcellated to {num_rois} ROIs using HCP_MMP1 atlas. "
        )

        f_out.create_dataset('roi_labels', data=unique_rois, compression='gzip')
        f_out['roi_labels'].attrs['description'] = (
            "HCP_MMP1 ROI labels (1-180). Index i in roi_timeseries corresponds to roi_labels[i]."
        )
        f_out.create_dataset('rois_present_mask',
                             data=np.array([roi in rois_present for roi in unique_rois], dtype=bool),
                             compression='gzip')

    # check for NaN values
    num_nans = np.isnan(roi_ts).sum()
    if num_nans > 0:
        print(f"Warning: {num_nans} NaN values detected")

    file_size_mb = output_path.stat().st_size / (1024**2) # MB size
    print(f"Done. {output_path} — shape ({num_timepoints}, {num_rois}), {file_size_mb:.2f} MB")
    print(f"Stats: mean {roi_ts.mean():.4f}, std {roi_ts.std():.4f}, min {roi_ts.min():.4f}, max {roi_ts.max():.4f}")


def main():
    parser = argparse.ArgumentParser(description="Parcellate resting state fMRI to ROIs using HCP_MMP1 atlas")
    parser.add_argument('--subjects', type=int, required=True, choices=[1, 2, 5, 7])
    parser.add_argument('--resting_dir', type=str, default=r'\path\resting_processed')
    parser.add_argument('--atlas_dir', type=str, default=r'\path\atlas')
    parser.add_argument('--brain_masks_dir', type=str, default=r'\path\processed\brain_masks')
    parser.add_argument('--output_dir', type=str, default=r'\path\resting_parcellated_glasser')
    args = parser.parse_args()

    subj_str = f"subj{args.subject:02d}"
    resting_hdf5 = Path(args.resting_dir) / f"resting_state_{subj_str}_all.hdf5"
    atlas_file = Path(args.atlas_dir) / "Glasser" / subj_str / "HCP_MMP1.nii.gz"
    brain_mask = Path(args.brain_masks_dir) / f"brain_mask_{subj_str}.npy"

    for filepath, name in [(resting_hdf5, "Resting state HDF5"), (atlas_file, "HCP_MMP1 atlas"), (brain_mask, "Brain mask")]:
        if not filepath.exists():
            raise FileNotFoundError(f"{name} not found: {filepath}")

    parcellate_to_rois(
        subject=args.subject,
        resting_hdf5_path=resting_hdf5,
        atlas_path=atlas_file,
        brain_mask_path=brain_mask,
        output_dir=Path(args.output_dir)
    )


if __name__ == "__main__":
    main()