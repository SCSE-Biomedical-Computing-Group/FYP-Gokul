"""
Compute session-wise correlation matrices for ICSC.
Splits parcellated resting state data into sessions and computes
a correlation matrix per session.

Input:
    - resting_state_subj0X_parcellated.hdf5: [timepoints, 180 ROIs]

Output:
    - <output_dir>/subj0X/S01_corr.npy: [180, 180]
    - <output_dir>/subj0X/S02_corr.npy: [180, 180]
    - ...

Usage:
    python compute_session_correlations.py --subjects 1
    python compute_session_correlations.py --subjects 1 2 5 7
"""

import argparse
import h5py
import numpy as np
from pathlib import Path

# trials per session, it is constant across all subjects
TRS_PER_SESSION = 125


def compute_session_correlations(subject, parcellated_dir, output_base_dir):
    subj_str = f"subj{subject:02d}"

    # load data
    hdf5_path = parcellated_dir / f"resting_state_{subj_str}_parcellated.hdf5"
    if not hdf5_path.exists():
        raise FileNotFoundError(f"Parcellated data not found: {hdf5_path}")

    with h5py.File(str(hdf5_path), 'r') as f:
        roi_timeseries = f['roi_timeseries'][:]
        num_rois = f['roi_timeseries'].attrs['num_rois']
    
    print(f"Subject {subject:02d}: loaded {roi_timeseries.shape}, {num_rois} ROIs")

    num_timepoints = roi_timeseries.shape[0]
    num_sessions = num_timepoints // TRS_PER_SESSION
    leftover = num_timepoints % TRS_PER_SESSION # sessions that dont match the correct trials per session
    if leftover:
        print(f"  Warning: {leftover} leftover timepoints won't be used")
    print(f"  {num_sessions} sessions x {TRS_PER_SESSION} TRs")

    output_dir = output_base_dir / subj_str
    output_dir.mkdir(parents=True, exist_ok=True)

    for session_idx in range(num_sessions):
        # extract this session's data
        start_tr = session_idx * TRS_PER_SESSION
        end_tr = start_tr + TRS_PER_SESSION
        
        session_data = roi_timeseries[start_tr:end_tr, :]  # [125, num_rois]
        
        # compute correlation matrix
        # np.corrcoef expects variables as rows, so transpose
        corr_matrix = np.corrcoef(session_data.T)  # [num_rois, num_rois]
        
        # to handle potential NaN values
        if np.isnan(corr_matrix).any():
            print(f"  ⚠ Session {session_idx+1}: NaN values detected, replacing with 0")
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        
        # set diagonal to 0
        np.fill_diagonal(corr_matrix, 0)
        
        # save as .npy file (ICSC expects this format)
        output_file = output_dir / f"S{session_idx+1:02d}_corr.npy"
        np.save(str(output_file), corr_matrix)
        
        # for showing progress
        if (session_idx + 1) % 5 == 0 or (session_idx + 1) == num_sessions:
            print(f"  Session {session_idx+1}/{num_sessions}: {output_file.name}")

    # validate using first session
    first = np.load(str(output_dir / "S01_corr.npy"))
    print(f"  S01_corr.npy: shape {first.shape}, range [{first.min():.4f}, {first.max():.4f}], "
          f"symmetric: {np.allclose(first, first.T)}, diagonal sum: {np.diag(first).sum():.4f}")

    npy_files = list(output_dir.glob("S*_corr.npy"))
    total_size_mb = sum(f.stat().st_size for f in npy_files) / (1024**2)
    print(f"  {len(npy_files)} files saved to {output_dir}, {total_size_mb:.2f} MB total")


def main():
    parser = argparse.ArgumentParser(description="Compute session-wise correlation matrices for ICSC")
    parser.add_argument('--subjects', type=int, nargs='+', required=True, choices=[1, 2, 5, 7])
    parser.add_argument('--parcellated_dir', type=str, default=r'\path\resting_parcellated_glasser')
    parser.add_argument('--output_base_dir', type=str, default=r'\path\ICSC_data\glasser_ICSC_data')
    args = parser.parse_args()

    for subject in args.subjects:
        compute_session_correlations(
            subject=subject,
            parcellated_dir=Path(args.parcellated_dir),
            output_base_dir=Path(args.output_base_dir)
        )


if __name__ == "__main__":
    main()