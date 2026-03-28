"""
Simple NSD Data Processing Script
Since raw NSD is already in MindAligner's order, just load and save!

Usage:
    python simple_process_nsd.py --subject 1
"""

import argparse
import h5py
import nibabel as nib
import numpy as np
from pathlib import Path
from tqdm import tqdm


def process_subject(subject, nsd_dir, output_dir):
    """
    Process raw NSD data into full-brain HDF5 matching MindAligner's order
    
    Args:
        subject: Subject ID (1-8)
        nsd_dir: Directory with raw NSD files (contains subj01/, subj02/, etc.)
        output_dir: Where to save processed file
    """
    subj_str = f"subj{subject:02d}"
    subj_path = Path(nsd_dir) / subj_str
    
    print("="*70)
    print(f"Processing Subject {subject:02d}")
    print("="*70)
    print(f"Input:  {subj_path}")
    print(f"Output: {output_dir}")
    print("="*70)
    
    # Find all session files
    nifti_files = sorted(subj_path.glob("betas_session*.nii.gz"))
    print(f"\nFound {len(nifti_files)} session files")
    
    # Step 1: Create brain mask from first session
    print("\nStep 1: Creating brain mask...")
    img = nib.load(str(nifti_files[0]))
    data = img.get_fdata(dtype=np.float32)
    brain_mask = np.any(data != 0, axis=-1)
    
    num_voxels = brain_mask.sum()
    print(f"  Brain shape: {brain_mask.shape}")
    print(f"  Brain voxels: {num_voxels:,}")
    
    del data, img
    
    # Step 2: Process all sessions
    print("\nStep 2: Processing all sessions...")
    all_trials = []
    
    for nifti_file in tqdm(nifti_files, desc="Loading sessions"):
        # Load session
        img = nib.load(str(nifti_file))
        data = img.get_fdata(dtype=np.float32)
        
        num_trials = data.shape[-1]
        session_betas = np.zeros((num_trials, num_voxels), dtype=np.float32)
        
        # Extract brain voxels for each trial
        for trial_idx in range(num_trials):
            trial_volume = data[:, :, :, trial_idx]
            session_betas[trial_idx] = trial_volume[brain_mask]
        
        all_trials.append(session_betas)
        del data, img
    
    # Step 3: Concatenate all sessions
    print("\nStep 3: Concatenating sessions...")
    full_brain_betas = np.vstack(all_trials)
    print(f"  Total shape: {full_brain_betas.shape}")
    
    # Step 4: Take first 30,000 trials (MindAligner's subset)
    if full_brain_betas.shape[0] > 30000:
        print(f"\nStep 4: Trimming to first 30,000 trials...")
        full_brain_betas = full_brain_betas[:30000]
        print(f"  Final shape: {full_brain_betas.shape}")
    elif full_brain_betas.shape[0] < 30000:
        print(f"\n⚠ WARNING: Only {full_brain_betas.shape[0]} trials found (expected 30000)")
    
    # Step 5: Save
    print(f"\nStep 5: Saving to HDF5...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"full_brain_subj{subject:02d}_all.hdf5"
    
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('betas', data=full_brain_betas, dtype=np.float32, compression='gzip')
        f['betas'].attrs['subject'] = subject
        f['betas'].attrs['num_trials'] = full_brain_betas.shape[0]
        f['betas'].attrs['num_voxels'] = full_brain_betas.shape[1]
        f['betas'].attrs['description'] = 'Full brain voxels, trials in MindAligner order'
    
    file_size_gb = output_file.stat().st_size / (1024**3)
    
    print(f"\n{'='*70}")
    print("✓ Processing Complete!")
    print(f"{'='*70}")
    print(f"Output file: {output_file}")
    print(f"Shape: {full_brain_betas.shape}")
    print(f"Size: {file_size_gb:.2f} GB")
    print(f"{'='*70}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Process raw NSD data (simple version)")
    
    parser.add_argument('--subject', type=int, required=True, choices=range(1, 9))
    parser.add_argument('--nsd_dir', type=str, default=r'G:\NSDdata\raw_nsd',
                       help='Directory with raw NSD NIfTI files')
    parser.add_argument('--output_dir', type=str, default=r'G:\NSDdata\processed',
                       help='Output directory')
    
    args = parser.parse_args()
    
    process_subject(args.subject, args.nsd_dir, args.output_dir)


if __name__ == "__main__":
    main() 