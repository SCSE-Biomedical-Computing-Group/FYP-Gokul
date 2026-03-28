"""
Verify that trial order matches between MindAligner and raw NSD
Uses image IDs to check alignment
"""

import argparse
import tarfile
import numpy as np
import io
from pathlib import Path
from scipy.io import loadmat

def load_nsd_image_sequence(subject, nsd_metadata_path='./nsd_metadata'):
    """Load which images were shown in which trials from NSD"""
    expdesign = loadmat(f'{nsd_metadata_path}/nsd_expdesign.mat')
    
    # subjectim: [subject, trial] -> image_id (1-indexed in MATLAB)
    # Convert to 0-indexed Python and get subject's sequence
    subj_sequence = expdesign['subjectim'][subject - 1] - 1  # Convert to 0-indexed
    
    print(f"NSD has {len(subj_sequence)} trials for subject {subject}")
    return subj_sequence


def load_mindaligner_image_sequence(subject, mindaligner_dir):
    """Load image IDs from MindAligner's WebDataset in order"""
    subj_str = f"subj{subject:02d}"
    wds_path = Path(mindaligner_dir) / "wds" / subj_str
    
    all_image_ids = []
    
    # Load training image IDs
    print("Loading training image IDs...")
    train_dir = wds_path / "train"
    train_files = sorted(train_dir.glob("*.tar"), key=lambda x: int(x.stem))
    
    for tar_file in train_files:
        with tarfile.open(tar_file, 'r') as tar:
            for member in tar.getmembers():
                if 'behav.npy' in member.name:
                    f = tar.extractfile(member)
                    behav = np.load(io.BytesIO(f.read()))
                    
                    # Column 0 = image IDs
                    img_ids = behav[:, 0]
                    valid_ids = img_ids[img_ids >= 0]  # Remove -1 padding
                    all_image_ids.extend(valid_ids.astype(int))
    
    train_count = len(all_image_ids)
    
    # Load test image IDs
    print("Loading test image IDs...")
    test_file = wds_path / "new_test" / "0.tar"
    
    with tarfile.open(test_file, 'r') as tar:
        for member in tar.getmembers():
            if 'behav.npy' in member.name:
                f = tar.extractfile(member)
                behav = np.load(io.BytesIO(f.read()))
                
                img_ids = behav[:, 0]
                valid_ids = img_ids[img_ids >= 0]
                all_image_ids.extend(valid_ids.astype(int))
    
    test_count = len(all_image_ids) - train_count
    
    print(f"MindAligner has {len(all_image_ids)} trials ({train_count} train + {test_count} test)")
    
    return np.array(all_image_ids), train_count, test_count


def verify_alignment(subject, mindaligner_dir):
    """Check if MindAligner's order matches raw NSD's order"""
    print("="*70)
    print(f"Verifying Trial Order - Subject {subject}")
    print("="*70)
    
    # Load sequences
    nsd_sequence = load_nsd_image_sequence(subject)
    ma_sequence, train_count, test_count = load_mindaligner_image_sequence(subject, mindaligner_dir)
    
    print(f"\n{'='*70}")
    print("Checking alignment...")
    print(f"{'='*70}\n")
    
    # Check if lengths match
    if len(ma_sequence) != len(nsd_sequence):
        print(f"⚠ WARNING: Length mismatch!")
        print(f"  NSD: {len(nsd_sequence)} trials")
        print(f"  MindAligner: {len(ma_sequence)} trials")
        print(f"\nUsing first {min(len(ma_sequence), len(nsd_sequence))} trials for comparison...")
    
    # Compare first N trials
    compare_length = min(len(ma_sequence), len(nsd_sequence), 30000)
    
    matches = 0
    mismatches = []
    
    for i in range(compare_length):
        if ma_sequence[i] == nsd_sequence[i]:
            matches += 1
        else:
            if len(mismatches) < 10:  # Store first 10 mismatches
                mismatches.append((i, ma_sequence[i], nsd_sequence[i]))
    
    match_rate = matches / compare_length * 100
    
    print(f"Match rate: {matches}/{compare_length} ({match_rate:.2f}%)")
    
    if match_rate > 99:
        print("\n✓ EXCELLENT: Nearly perfect alignment!")
        print("Your processed data will align correctly with MindAligner.")
    elif match_rate > 90:
        print("\n✓ GOOD: High alignment rate.")
        print("Minor discrepancies but should work.")
    elif match_rate > 50:
        print("\n⚠ MEDIUM: Partial alignment.")
        print("Some reordering may be needed.")
    else:
        print("\n✗ POOR: Low alignment.")
        print("Trial order does not match. Need different approach.")
    
    if mismatches:
        print(f"\nFirst {len(mismatches)} mismatches:")
        for idx, ma_img, nsd_img in mismatches:
            print(f"  Trial {idx}: MA image {ma_img} vs NSD image {nsd_img}")
    
    print(f"\n{'='*70}")
    
    return match_rate, matches, compare_length


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Verify trial order alignment")
    parser.add_argument('--subject', type=int, default=1, choices=range(1, 9))
    parser.add_argument('--mindaligner_dir', type=str, required=True)
    
    args = parser.parse_args()
    
    verify_alignment(args.subject, args.mindaligner_dir)