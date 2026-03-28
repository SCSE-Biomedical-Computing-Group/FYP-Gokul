"""
Verify Resting State Data Structure
Inspects downloaded resting state files to understand their structure.

Usage:
    python verify_resting_structure.py --subject 1
"""

import argparse
import h5py
import nibabel as nib
import numpy as np
from pathlib import Path


def verify_resting_state_structure(subject, resting_dir):
    """
    Inspect resting state data structure for a subject
    
    Args:
        subject: Subject ID (1-8)
        resting_dir: Directory containing resting state data
    """
    subj_str = f"subj{subject:02d}"
    subj_path = Path(resting_dir) / subj_str
    
    print("="*70)
    print(f"Resting State Data Structure - Subject {subject:02d}")
    print("="*70)
    print(f"Directory: {subj_path}")
    print("="*70)
    
    # Check if directory exists
    if not subj_path.exists():
        print(f"✗ Directory not found: {subj_path}")
        return
    
    # Find all files
    all_files = list(subj_path.iterdir())
    nifti_files = sorted(subj_path.glob("*.nii.gz"))
    hdf5_files = sorted(subj_path.glob("*.hdf5"))
    
    print(f"\n📁 Files Found:")
    print(f"  Total files: {len(all_files)}")
    print(f"  NIfTI files (.nii.gz): {len(nifti_files)}")
    print(f"  HDF5 files (.hdf5): {len(hdf5_files)}")
    
    # List all files
    print(f"\n📋 File List:")
    for i, file in enumerate(sorted(all_files)[:20]):  # Show first 20
        print(f"  {i+1}. {file.name}")
    if len(all_files) > 20:
        print(f"  ... and {len(all_files) - 20} more files")
    
    # Inspect NIfTI files
    if nifti_files:
        print(f"\n{'='*70}")
        print("NIfTI File Analysis")
        print(f"{'='*70}")
        
        for i, nifti_file in enumerate(nifti_files[:5]):  # Inspect first 5
            print(f"\n[File {i+1}] {nifti_file.name}")
            try:
                img = nib.load(str(nifti_file))
                data_shape = img.shape
                data_dtype = img.get_data_dtype()
                
                print(f"  Shape: {data_shape}")
                print(f"  Data type: {data_dtype}")
                
                if len(data_shape) == 4:
                    print(f"  Spatial dimensions: {data_shape[:3]}")
                    print(f"  Timepoints (TRs): {data_shape[3]}")
                elif len(data_shape) == 3:
                    print(f"  3D volume (no temporal dimension)")
                
                # Load first volume to check values
                if len(data_shape) == 4:
                    first_vol = img.get_fdata(dtype=np.float32)[:, :, :, 0]
                else:
                    first_vol = img.get_fdata(dtype=np.float32)
                
                non_zero = np.any(first_vol != 0)
                print(f"  Contains data: {non_zero}")
                print(f"  Value range: [{first_vol.min():.4f}, {first_vol.max():.4f}]")
                
            except Exception as e:
                print(f"  ✗ Error loading: {e}")
        
        if len(nifti_files) > 5:
            print(f"\n... and {len(nifti_files) - 5} more NIfTI files")
        
        # Summary statistics
        print(f"\n{'='*70}")
        print("Summary Statistics")
        print(f"{'='*70}")
        
        total_timepoints = 0
        all_shapes = []
        
        for nifti_file in nifti_files:
            try:
                img = nib.load(str(nifti_file))
                shape = img.shape
                all_shapes.append(shape)
                if len(shape) == 4:
                    total_timepoints += shape[3]
            except:
                pass
        
        print(f"Total NIfTI files: {len(nifti_files)}")
        print(f"Total timepoints across all files: {total_timepoints}")
        
        if all_shapes:
            print(f"\nShape variations:")
            unique_shapes = set(all_shapes)
            for shape in unique_shapes:
                count = all_shapes.count(shape)
                print(f"  {shape}: {count} file(s)")
    
    # Inspect HDF5 files
    if hdf5_files:
        print(f"\n{'='*70}")
        print("HDF5 File Analysis")
        print(f"{'='*70}")
        
        for i, hdf5_file in enumerate(hdf5_files):
            print(f"\n[File {i+1}] {hdf5_file.name}")
            try:
                with h5py.File(hdf5_file, 'r') as f:
                    print(f"  Keys: {list(f.keys())}")
                    
                    for key in f.keys():
                        dataset = f[key]
                        print(f"\n  Dataset: '{key}'")
                        print(f"    Shape: {dataset.shape}")
                        print(f"    Data type: {dataset.dtype}")
                        
                        # Check attributes
                        if dataset.attrs:
                            print(f"    Attributes:")
                            for attr_name, attr_val in dataset.attrs.items():
                                print(f"      {attr_name}: {attr_val}")
                        
                        # Sample data
                        if len(dataset.shape) > 0:
                            print(f"    First 10 values: {dataset[0][:10] if len(dataset.shape) > 1 else dataset[:10]}")
                
            except Exception as e:
                print(f"  ✗ Error loading: {e}")
    
    # Recommendations
    print(f"\n{'='*70}")
    print("Processing Recommendations")
    print(f"{'='*70}")
    
    if nifti_files and total_timepoints > 0:
        print(f"✓ Found {len(nifti_files)} resting state scan(s)")
        print(f"✓ Total timepoints: {total_timepoints}")
        print(f"\nNext steps:")
        print(f"  1. Use same brain mask from task data")
        print(f"  2. Extract voxels for all {total_timepoints} timepoints")
        print(f"  3. Concatenate into single array: [{total_timepoints}, num_voxels]")
        print(f"  4. Save as resting_state_subj{subject:02d}_all.hdf5")
    else:
        print(f"⚠ No suitable resting state data found")
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Verify resting state data structure")
    
    parser.add_argument('--subject', type=int, required=True, choices=range(1, 9))
    parser.add_argument('--resting_dir', type=str, default=r'G:\NSDdata\resting_state',
                       help='Directory with resting state data')
    
    args = parser.parse_args()
    
    verify_resting_state_structure(args.subject, args.resting_dir)


if __name__ == "__main__":
    main()