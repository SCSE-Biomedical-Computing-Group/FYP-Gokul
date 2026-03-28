"""
Process raw NSD NIfTI files into a single HDF5 file with session-wise z-scoring.
Processes one session at a time to avoid memory issues.

Usage:
    python process_nsd_data.py --subject 1
    python process_nsd_data.py --subject 1 --nsd_dir /path/raw_nsd --output_dir /path/processed
"""

import argparse
import h5py
import nibabel as nib
import numpy as np
from pathlib import Path
from tqdm import tqdm


def process_subject(subject, nsd_dir, output_dir):
    subj_str = f"subj{subject:02d}"
    subj_path = Path(nsd_dir) / subj_str

    nifti_files = sorted(subj_path.glob("betas_session*.nii.gz"))
    
    # check there are the nifti files
    if not nifti_files:
        raise FileNotFoundError(f"No NIfTI files found in {subj_path}")
    print(f"Found {len(nifti_files)} session files")

    # brain mask from first session
    img = nib.load(str(nifti_files[0]))
    data = img.get_fdata(dtype=np.float32)
    brain_mask = np.any(data != 0, axis=-1)
    num_voxels = brain_mask.sum()
    print(f"Brain shape: {brain_mask.shape}, voxels: {num_voxels:,}")
    del data, img

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"full_brain_subj{subject:02d}_all.hdf5" # name of the output file

    # ensure 30k trials
    total_trials = min(sum(nib.load(str(f)).shape[-1] for f in nifti_files), 30000) 
    print(f"Allocating ({total_trials}, {num_voxels}) dataset...")

    with h5py.File(output_file, 'w') as hf:
        dset = hf.create_dataset('betas', shape=(total_trials, num_voxels), dtype=np.float32,
                                 compression='gzip', compression_opts=4)

        trial_offset = 0
        for session_idx, nifti_file in enumerate(tqdm(nifti_files, desc="Processing sessions")):
            if trial_offset >= 30000:
                break

            img = nib.load(str(nifti_file))
            data = img.get_fdata(dtype=np.float32)
            num_trials = data.shape[-1]
            session_betas = np.zeros((num_trials, num_voxels), dtype=np.float32)

            for trial_idx in range(num_trials):
                trial_volume = data[:, :, :, trial_idx]
                session_betas[trial_idx] = trial_volume[brain_mask]

            # session-wise z-score
            session_mean = session_betas.mean(axis=0, keepdims=True)
            session_std = session_betas.std(axis=0, keepdims=True)
            session_std[session_std == 0] = 1.0
            session_betas_zscored = (session_betas - session_mean) / session_std

            if session_idx == 0:
                print(f"Session 1 z-score check: range [{session_betas_zscored.min():.3f}, {session_betas_zscored.max():.3f}], "
                      f"mean {session_betas_zscored.mean():.3f}, std {session_betas_zscored.std():.3f}")

            # only write a batch at a time due to memory contraints
            trials_to_write = min(num_trials, 30000 - trial_offset)
            dset[trial_offset:trial_offset + trials_to_write] = session_betas_zscored[:trials_to_write]
            trial_offset += trials_to_write

            del data, img, session_betas, session_betas_zscored

        hf['betas'].attrs['subject'] = subject
        hf['betas'].attrs['num_trials'] = trial_offset
        hf['betas'].attrs['num_voxels'] = num_voxels
        hf['betas'].attrs['processing'] = 'session-wise z-scored'
        hf['betas'].attrs['description'] = 'Full brain voxels with session-wise z-scoring, trials in NSD order'

    # size in GB
    size_gb = output_file.stat().st_size / (1024**3)
    print(f"Done. {output_file} — shape ({trial_offset}, {num_voxels}), {size_gb:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Process raw NSD data with session-wise z-scoring")
    parser.add_argument('--subjects', type=int, nargs='+', required=True, choices=[1, 2, 5, 7])
    parser.add_argument('--nsd_dir', type=str, default=r'\path\raw_nsd')
    parser.add_argument('--output_dir', type=str, default=r'\path\processed')
    args = parser.parse_args()

    process_subject(args.subject, args.nsd_dir, args.output_dir)


if __name__ == "__main__":
    main()