"""
Inspect .tar Files Using Standard tarfile Module
=================================================
Simpler approach that works on Windows.

Usage:
    python simple_inspect_tar.py
"""

import tarfile
import numpy as np
from pathlib import Path
import io


def inspect_tar_simple(tar_path: Path):
    """
    Inspect .tar file using standard tarfile module.
    """
    print("=" * 70)
    print(f"Inspecting: {tar_path.name}")
    print("=" * 70)
    
    with tarfile.open(str(tar_path), 'r') as tar:
        members = tar.getmembers()
        
        print(f"\nTotal files in tar: {len(members)}")
        
        # Group files by their base name (before extension)
        samples = {}
        for member in members:
            # Extract base name (e.g., "000000" from "000000.npy")
            base_name = member.name.split('.')[0]
            ext = '.' + member.name.split('.')[-1] if '.' in member.name else ''
            
            if base_name not in samples:
                samples[base_name] = {}
            samples[base_name][ext] = member
        
        print(f"Number of samples: {len(samples)}")
        
        # Show first few samples
        print("\n" + "─" * 70)
        print("Sample Structure")
        print("─" * 70)
        
        sample_names = sorted(samples.keys())[:3]  # First 3 samples
        
        for sample_name in sample_names:
            print(f"\nSample: {sample_name}")
            print(f"  Files:")
            
            for ext, member in sorted(samples[sample_name].items()):
                print(f"    {member.name} ({member.size:,} bytes)")
                
                # If it's a .npy file, load and inspect it
                if ext == '.npy':
                    try:
                        f = tar.extractfile(member)
                        data = np.load(io.BytesIO(f.read()))
                        
                        print(f"      Shape: {data.shape}")
                        print(f"      Dtype: {data.dtype}")
                        
                        if data.dtype in [np.float32, np.float64]:
                            print(f"      Range: [{data.min():.4f}, {data.max():.4f}]")
                            print(f"      Mean: {data.mean():.4f}")
                        
                        # Determine what this is
                        if len(data.shape) == 1:
                            voxel_count = data.shape[0]
                            print(f"      → 1D fMRI data with {voxel_count:,} voxels")
                            
                            if 13000 <= voxel_count <= 16000:
                                print(f"      ✓ ALREADY MASKED (NSDGeneral, ~15k voxels)")
                            elif 250000 <= voxel_count <= 400000:
                                print(f"      ✓ FULL BRAIN (~300-370k voxels)")
                            else:
                                print(f"      ? Unknown size")
                    except Exception as e:
                        print(f"      Error loading: {e}")
                
                # If it's an image
                elif ext in ['.jpg', '.jpeg', '.png']:
                    print(f"      → Image file")
                
                # If it's text/json
                elif ext in ['.json', '.txt']:
                    try:
                        f = tar.extractfile(member)
                        content = f.read().decode('utf-8')
                        if len(content) <= 200:
                            print(f"      Content: {content}")
                        else:
                            print(f"      Content (first 200 chars): {content[:200]}...")
                    except Exception as e:
                        print(f"      Error reading: {e}")
        
        # Summary of all extensions found
        print("\n" + "─" * 70)
        print("Summary of File Types")
        print("─" * 70)
        
        all_extensions = set()
        for sample in samples.values():
            all_extensions.update(sample.keys())
        
        print(f"Extensions found: {sorted(all_extensions)}")
        
        # Count by extension
        for ext in sorted(all_extensions):
            count = sum(1 for sample in samples.values() if ext in sample)
            print(f"  {ext}: {count} files")


def main():
    tar_path = Path(r"C:\Users\Gokul\Desktop\Visual Studio Code\MindAligner\dataset\wds\subj01\train\0.tar")
    
    print("=" * 70)
    print("Simple .tar File Inspector")
    print("=" * 70)
    print(f"\nInspecting: {tar_path}")
    
    if not tar_path.exists():
        print(f"\n✗ File not found: {tar_path}")
        return
    
    inspect_tar_simple(tar_path)
    
    print("\n" + "=" * 70)
    print("CRITICAL QUESTION:")
    print("=" * 70)
    print("\nIs the fMRI data:")
    print("  A) ~15k voxels → Already masked with NSDGeneral")
    print("     → CANNOT use directly, must reprocess from raw NIfTI")
    print("\n  B) ~370k voxels → Full brain data")
    print("     → CAN apply individualized masks directly")
    print("=" * 70)


if __name__ == "__main__":
    main()