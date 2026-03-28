"""
Extract and save brain masks for NSD subjects, using the first task
session NIfTI file per subject.

Usage:
    python extract_brain_masks.py --subjects 1
    python extract_brain_masks.py --subjects 1 2 5 7
"""

import argparse
import numpy as np
import nibabel as nib
from pathlib import Path


def extract_brain_mask(subject, nsd_dir, output_dir):
    subj_str = f"subj{subject:02d}"
    subj_path = nsd_dir / subj_str


    nifti_files = sorted(subj_path.glob("betas_session*.nii.gz"))
    if not nifti_files:
        raise FileNotFoundError(f"No NIfTI files found in {subj_path}")

    print(f"Subject {subject:02d}: {len(nifti_files)} session files, using {nifti_files[0].name}")

    img = nib.load(str(nifti_files[0])) # first session file
    data = img.get_fdata(dtype=np.float32)
    brain_mask = np.any(data != 0, axis=-1) # method for making the file
    
    del data, img

    print(f"  Shape: {brain_mask.shape}, active voxels: {int(brain_mask.sum()):,}")

    mask_dir = output_dir / "brain_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    save_path = mask_dir / f"brain_mask_{subj_str}.npy"
    np.save(str(save_path), brain_mask)
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract and save brain masks from NSD task data")
    parser.add_argument('--subjects', type=int, nargs='+', required=True, choices=[1, 2, 5, 7])
    parser.add_argument('--nsd_dir', type=str, default=r'\path\raw_nsd')
    parser.add_argument('--output_dir', type=str, default=r'\path\processed')
    args = parser.parse_args()

    nsd_dir = Path(args.nsd_dir)
    output_dir = Path(args.output_dir)

    for subj in args.subjects:
        extract_brain_mask(subj, nsd_dir, output_dir)


if __name__ == "__main__":
    main()