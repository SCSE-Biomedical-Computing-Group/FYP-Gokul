"""
Download NSD resting state beta files from AWS S3.

Usage:
    python download_resting_state.py --subjects 1 2 5 7
    python download_resting_state.py --subjects 1 --output ./data
"""

import subprocess
import argparse
from pathlib import Path
from datetime import datetime


def download_subject_resting_state(subject_id, output_dir):
    subj_str = f"subj{subject_id:02d}"
    s3_path = f"s3://natural-scenes-dataset/nsddata_betas/ppdata/{subj_str}/func1pt8mm/restingbetas_fithrf/"
    local_path = Path(output_dir) / subj_str
    local_path.mkdir(parents=True, exist_ok=True)

    print(f"\nSubject {subject_id:02d}: {s3_path} -> {local_path}")

    cmd = ['aws', 's3', 'sync', s3_path, str(local_path), '--no-sign-request', '--region', 'us-east-1']
    print(f"Running: {' '.join(cmd)}\n")

    start = datetime.now()
    try:
        subprocess.run(cmd, check=True)
        elapsed = (datetime.now() - start).total_seconds() / 60
        nii_files = list(local_path.glob("*.nii.gz"))
        hdf5_files = list(local_path.glob("*.hdf5"))
        print(f"Done in {elapsed:.1f} min. {len(nii_files)} NIfTI, {len(hdf5_files)} HDF5 files downloaded.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download NSD resting state beta files from AWS S3")
    parser.add_argument('--subjects', type=int, nargs='+', required=True, choices=[1, 2, 5, 7])
    parser.add_argument('--output', type=str, default='/path/raw_nsd_resting')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for sid in args.subjects:
        download_subject_resting_state(sid, output_dir)


if __name__ == "__main__":
    main()