"""
Process resting state fMRI data for ICSC algorithm.
Extracts voxel time series from resting state NIfTI files using the
existing task-data brain mask, then saves to HDF5.

Usage:
    python process_resting_state.py --subject 1
    python process_resting_state.py --subject 1 --brain_masks_dir \\path\\processed\\brain_masks
"""

import argparse
import time
from pathlib import Path
import os
import h5py
import nibabel as nib
import numpy as np


def load_brain_mask(brain_masks_dir, subject):
    subj_str = f"subj{subject:02d}"
    mask_path = brain_masks_dir / f"brain_mask_{subj_str}.npy"

    # check mask exists
    if mask_path.exists():
        mask = np.load(str(mask_path))
        print(f"Loaded brain mask from {mask_path}, shape {mask.shape}, {mask.sum():,} active voxels")
        return mask

    print(f"Brain mask not found at {mask_path}")
    return None



def filter_resting_niftis(nifti_files, expected_spatial):
    
    # Keep only the files that match the same dimensions as brain mask
    valid = []
    for f in nifti_files:
        img = nib.load(str(f))
        shape = img.shape
        # file must be 4D
        if len(shape) != 4:
            continue
        # file must match brain_mask dimensions
        if shape[:3] != expected_spatial:
            continue
        # file must have more than 2 timepoints
        if shape[3] <= 2:
            continue
        valid.append(f)
    return sorted(valid)


def process_resting_state(subject, resting_dir, brain_masks_dir, output_dir):
    subj_str = f"subj{subject:02d}"
    subj_resting_path = resting_dir / subj_str

    print(f"Processing resting state data for subject {subject:02d}")
    print(f"Input:  {subj_resting_path}")
    print(f"Output: {output_dir}")

    all_niftis = sorted(subj_resting_path.glob("*.nii.gz"))
    if not all_niftis:
        raise FileNotFoundError(f"No NIfTI files found in {subj_resting_path}")
    print(f"Found {len(all_niftis)} total NIfTI files")

    # load in the brain mask
    brain_mask = load_brain_mask(brain_masks_dir, subject)
    expected_spatial = brain_mask.shape
    resting_niftis = filter_resting_niftis(all_niftis, expected_spatial)
    if not resting_niftis: 
        raise FileNotFoundError(f"No valid resting state files matching spatial dims {expected_spatial}")


    num_voxels = int(brain_mask.sum())

    # count total timepoints in resting state data
    total_timepoints = 0
    for f in resting_niftis:
        img = nib.load(str(f))
        n_trs = img.shape[3]
        total_timepoints += n_trs
        print(f"  {f.name}: {n_trs} TRs")
    print(f"{len(resting_niftis)} valid sessions, {total_timepoints:,} total timepoints")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"resting_state_{subj_str}_all.hdf5"
    estimated_gb = (total_timepoints * num_voxels * 4) / (1024 ** 3) # estimation for file size of output
    print(f"Output: {output_path}, estimated {estimated_gb:.2f} GB uncompressed")

    with h5py.File(str(output_path), 'w') as f_out:
        dset = f_out.create_dataset(
            'betas',
            shape=(total_timepoints, num_voxels),
            dtype=np.float32,
            chunks=(min(125, total_timepoints), min(10000, num_voxels)),
            compression='gzip',
            compression_opts=4
        )

        dset.attrs['subject'] = subject
        dset.attrs['num_sessions'] = len(resting_niftis)
        dset.attrs['num_timepoints'] = total_timepoints
        dset.attrs['num_voxels'] = num_voxels
        dset.attrs['spatial_dims'] = list(expected_spatial)
        dset.attrs['description'] = (
            f"Resting state fMRI betas for Subject {subject:02d} extracted using brain mask. "
        )

        f_out.create_dataset('brain_mask', data=brain_mask, compression='gzip')
        f_out['brain_mask'].attrs['description'] = (
            f"Boolean brain mask, shape {expected_spatial}. True = voxel included in 'betas'."
        )

        t_start = time.time()
        write_offset = 0

        # extract voxels session by session
        for sess_idx, nifti_file in enumerate(resting_niftis):
            t_sess = time.time()
            print(f"Session {sess_idx + 1}/{len(resting_niftis)}: {nifti_file.name}")

            img = nib.load(str(nifti_file))
            data = img.get_fdata(dtype=np.float32)

            if data.shape[:3] != expected_spatial:
                print(f"  Unexpected shape {data.shape}, expected {expected_spatial}. Skipping.")
                continue

            n_trs = data.shape[3]
            session_betas = np.zeros((n_trs, num_voxels), dtype=np.float32)

            # extract within each session
            for tr_idx in range(n_trs):
                session_betas[tr_idx] = data[:, :, :, tr_idx][brain_mask]

            # ensure shape is the same
            assert session_betas.shape == (n_trs, num_voxels), \
                f"Shape mismatch: got {session_betas.shape}, expected ({n_trs}, {num_voxels})"

            dset[write_offset:write_offset + n_trs, :] = session_betas
            write_offset += n_trs

            elapsed = time.time() - t_sess
            total_elapsed = time.time() - t_start
            pct = (sess_idx + 1) / len(resting_niftis) * 100
            print(f"  Written rows {write_offset - n_trs}:{write_offset} | {elapsed:.1f}s | {total_elapsed:.0f}s total | {pct:.0f}%")

            del data, session_betas # to reduce memory usage

    # check first few samples
    with h5py.File(str(output_path), 'r') as f_val:
        betas = f_val['betas']
        sample = betas[0, :10]
        nonzero_pct = np.count_nonzero(betas[0]) / betas.shape[1] * 100
        print(f"Validation — shape: {betas.shape}, dtype: {betas.dtype}")
        print(f"First 10 voxels at t=0: {np.round(sample, 3)}")
        print(f"Non-zero voxels at t=0: {nonzero_pct:.1f}%")

    file_size_gb = os.path.getsize(str(output_path)) / (1024 ** 3)
    print(f"Done. {output_path} — {file_size_gb:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Process resting state fMRI data for ICSC algorithm")
    parser.add_argument('--subjects', type=int, nargs='+', required=True, choices=[1, 2, 5, 7])
    parser.add_argument('--resting_dir', type=str, default=r'\path\resting_state')
    parser.add_argument('--brain_masks_dir', type=str, default=r'\path\processed\brain_masks')
    parser.add_argument('--output_dir', type=str, default=r'\path\resting_processed')
    args = parser.parse_args()

    process_resting_state(
        subject=args.subject,
        resting_dir=Path(args.resting_dir),
        brain_masks_dir=Path(args.brain_masks_dir),
        output_dir=Path(args.output_dir)
    )


if __name__ == "__main__":
    main()