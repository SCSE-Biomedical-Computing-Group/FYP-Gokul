"""
Verify Image Alignment Between Reconstructions and Ground Truth
================================================================
Explores all_images.pt, COCO_73k_subj_indices.hdf5, and coco_images_224_float16.hdf5
to verify that reconstructions align with ground truth images.

Usage:
    python verify_image_alignment.py
"""

import h5py
import torch
import numpy as np
from pathlib import Path


def explore_all_images(data_path):
    """
    Explore what's in all_images.pt
    """
    print("=" * 70)
    print("1. EXPLORING all_images.pt")
    print("=" * 70)
    
    all_images_path = Path(data_path) / "evals" / "all_images.pt"
    
    if not all_images_path.exists():
        print(f"❌ File not found: {all_images_path}")
        return None
    
    all_images = torch.load(all_images_path)
    
    print(f"Shape: {all_images.shape}")
    print(f"Dtype: {all_images.dtype}")
    print(f"Min value: {all_images.min():.3f}")
    print(f"Max value: {all_images.max():.3f}")
    print(f"Mean: {all_images.mean():.3f}")
    
    print(f"\nFirst image shape: {all_images[0].shape}")
    print(f"Number of images: {len(all_images)}")
    
    return all_images


def explore_coco_indices(data_path):
    """
    Explore COCO_73k_subj_indices.hdf5
    """
    print("\n" + "=" * 70)
    print("2. EXPLORING COCO_73k_subj_indices.hdf5")
    print("=" * 70)
    
    indices_path = Path(data_path) / "COCO_73k_subj_indices.hdf5"
    
    if not indices_path.exists():
        print(f"❌ File not found: {indices_path}")
        return None
    
    with h5py.File(indices_path, 'r') as f:
        print(f"Keys in file: {list(f.keys())}")
        
        # Check each subject
        for key in f.keys():
            data = f[key][:]
            print(f"\n  {key}:")
            print(f"    Shape: {data.shape}")
            print(f"    Dtype: {data.dtype}")
            print(f"    Min COCO ID: {data.min()}")
            print(f"    Max COCO ID: {data.max()}")
            print(f"    First 10 COCO IDs: {data[:10]}")
            print(f"    Number of unique IDs: {len(np.unique(data))}")
        
        # Return subject 1 indices for further analysis
        subj01_indices = f['subj01'][:]
        return subj01_indices


def explore_coco_images(data_path):
    """
    Explore coco_images_224_float16.hdf5
    """
    print("\n" + "=" * 70)
    print("3. EXPLORING coco_images_224_float16.hdf5")
    print("=" * 70)
    
    images_path = Path(data_path) / "coco_images_224_float16.hdf5"
    
    if not images_path.exists():
        print(f"❌ File not found: {images_path}")
        return None
    
    with h5py.File(images_path, 'r') as f:
        print(f"Keys in file: {list(f.keys())}")
        
        images = f['images']
        print(f"\nImages dataset:")
        print(f"  Shape: {images.shape}")
        print(f"  Dtype: {images.dtype}")
        print(f"  Total images: {images.shape[0]}")
        
        # Sample a few images
        sample = images[0]
        print(f"\n  First image:")
        print(f"    Shape: {sample.shape}")
        print(f"    Min: {sample.min():.3f}")
        print(f"    Max: {sample.max():.3f}")
        
        return images.shape[0]


def verify_reconstruction_alignment(data_path):
    """
    Verify that reconstructed image IDs match ground truth order
    """
    print("\n" + "=" * 70)
    print("4. VERIFYING RECONSTRUCTION ALIGNMENT")
    print("=" * 70)
    
    # Load test data to get the COCO IDs that recon.py uses
    print("\nSimulating recon.py behavior...")
    
    # Load test behavioral data
    from my_utils.data_utils import get_test_dataloader
    
    test_dl, num_test = get_test_dataloader(data_path, subj=1, new_test=True)
    
    test_images_idx = []
    for test_i, (behav, _, _, _) in enumerate(test_dl):
        curr_images_idx = behav[:,0,0].cpu().numpy()
        test_images_idx.extend(curr_images_idx)
    
    test_images_idx = np.array(test_images_idx, dtype=int)
    
    print(f"Total test trials: {len(test_images_idx)}")
    print(f"Unique test images: {len(np.unique(test_images_idx))}")
    
    # This is what recon.py does
    unique_image_ids = np.unique(test_images_idx)
    total_images = min(len(unique_image_ids), 30)
    
    reconstructed_ids = unique_image_ids[:total_images]
    
    print(f"\nCOCO IDs that recon.py reconstructs (first 30):")
    print(reconstructed_ids)
    
    return reconstructed_ids, test_images_idx


def verify_all_images_order(data_path, reconstructed_ids):
    """
    Verify if all_images.pt follows the same order as reconstructed_ids
    """
    print("\n" + "=" * 70)
    print("5. CHECKING IF all_images.pt MATCHES RECONSTRUCTION ORDER")
    print("=" * 70)
    
    all_images_path = Path(data_path) / "evals" / "all_images.pt"
    
    if not all_images_path.exists():
        print("❌ all_images.pt not found - cannot verify!")
        return
    
    all_images = torch.load(all_images_path)
    
    print(f"\nall_images.pt has {len(all_images)} images")
    print(f"recon.py reconstructs {len(reconstructed_ids)} images")
    
    if len(all_images) == len(reconstructed_ids):
        print("✓ Same number of images!")
    else:
        print(f"⚠️  Different number of images!")
        print(f"   all_images.pt: {len(all_images)}")
        print(f"   Reconstructions: {len(reconstructed_ids)}")
    
    # Now we need to figure out: how is all_images.pt ordered?
    # It should match the sorted COCO IDs
    print(f"\nExpected COCO ID order (from recon.py):")
    print(reconstructed_ids)
    
    # Load COCO indices to map position → COCO ID
    indices_path = Path(data_path) / "COCO_73k_subj_indices.hdf5"
    
    if indices_path.exists():
        with h5py.File(indices_path, 'r') as f:
            subj01_indices = f['subj01'][:]
        
        print(f"\n✓ We can verify alignment by:")
        print(f"  1. all_images[i] should correspond to COCO ID = reconstructed_ids[i]")
        print(f"  2. We need to check if all_images.pt was created with the same sorting")
        
        # Try to find which COCO images correspond to all_images
        # This requires knowing the mapping
        
        print(f"\n⚠️  To fully verify alignment, we need to:")
        print(f"  - Check how MindAligner authors created all_images.pt")
        print(f"  - OR recreate all_images.pt ourselves with known COCO IDs")
    

def create_ground_truth_images(data_path, reconstructed_ids):
    """
    Create ground truth images in the correct order for evaluation
    """
    print("\n" + "=" * 70)
    print("6. CREATING GROUND TRUTH IMAGES FOR VERIFICATION")
    print("=" * 70)
    
    indices_path = Path(data_path) / "COCO_73k_subj_indices.hdf5"
    images_path = Path(data_path) / "coco_images_224_float16.hdf5"
    
    if not indices_path.exists() or not images_path.exists():
        print("❌ Cannot create ground truth - missing files")
        return None
    
    with h5py.File(indices_path, 'r') as idx_file, \
         h5py.File(images_path, 'r') as img_file:
        
        subj01_indices = idx_file['subj01'][:]  # Maps trial → COCO ID
        all_coco_images = img_file['images']     # [73000, 3, 224, 224]
        
        print(f"subj01_indices shape: {subj01_indices.shape}")
        print(f"all_coco_images shape: {all_coco_images.shape}")
        
        # For each reconstructed COCO ID, find the corresponding image
        ground_truth_images = []
        
        for coco_id in reconstructed_ids:
            # Find which position in subj01_indices has this COCO ID
            positions = np.where(subj01_indices == coco_id)[0]
            
            if len(positions) == 0:
                print(f"⚠️  COCO ID {coco_id} not found in subj01_indices!")
                continue
            
            # Take the first occurrence
            position = positions[0]
            
            # The image at this position in all_coco_images
            gt_image = all_coco_images[position]
            ground_truth_images.append(gt_image)
        
        ground_truth_tensor = torch.tensor(np.array(ground_truth_images))
        
        print(f"\n✓ Created ground truth tensor:")
        print(f"  Shape: {ground_truth_tensor.shape}")
        print(f"  This should match all_images.pt if alignment is correct!")
        
        return ground_truth_tensor


def main():
    # Update this to your data path
    data_path = r"C:\Users\Gokul\Desktop\Visual Studio Code\MindAligner\dataset"
    
    print("VERIFYING IMAGE ALIGNMENT FOR EVALUATION")
    print("=" * 70)
    print(f"Data path: {data_path}\n")
    
    # 1. Explore all_images.pt
    all_images = explore_all_images(data_path)
    
    # 2. Explore COCO indices
    subj01_indices = explore_coco_indices(data_path)
    
    # 3. Explore COCO images
    num_coco_images = explore_coco_images(data_path)
    
    # 4. Get the COCO IDs that recon.py reconstructs
    try:
        reconstructed_ids, test_images_idx = verify_reconstruction_alignment(data_path)
        
        # 5. Verify all_images.pt order
        verify_all_images_order(data_path, reconstructed_ids)
        
        # 6. Create ground truth for comparison
        created_gt = create_ground_truth_images(data_path, reconstructed_ids)
        
        # 7. Compare with all_images.pt if it exists
        if all_images is not None and created_gt is not None:
            print("\n" + "=" * 70)
            print("7. COMPARING all_images.pt WITH CREATED GROUND TRUTH")
            print("=" * 70)
            
            if all_images.shape == created_gt.shape:
                # Check if they're identical
                diff = torch.abs(all_images - created_gt).max()
                print(f"\nMax difference: {diff:.6f}")
                
                if diff < 0.001:
                    print("✓ ✓ ✓ PERFECT ALIGNMENT! all_images.pt matches reconstruction order!")
                else:
                    print("⚠️  Images don't match - possible alignment issue!")
            else:
                print(f"⚠️  Shape mismatch:")
                print(f"   all_images.pt: {all_images.shape}")
                print(f"   Created GT: {created_gt.shape}")
    
    except Exception as e:
        print(f"\n⚠️  Could not complete verification: {e}")
        print("This might be because data_utils import failed on Windows")
    
    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()