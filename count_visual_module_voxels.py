"""
Count voxels in visual modules and compare with required counts for pretrained MindAligner models.

Usage:
    python count_visual_module_voxels.py
"""

import pandas as pd
import numpy as np
import nibabel as nib
from pathlib import Path


REQUIRED_VOXEL_COUNTS = {
    1: 15724,
    2: 14278,
    5: 13039,
    7: 12682
}

VISUAL_CORTEX_TYPES = [
    "Primary_Visual",
    "Early_Visual",
    "Dorsal_Stream_Visual",
    "Ventral_Stream_Visual",
    "MT+_Complex_and_Neighboring_Visual_Areas"
]

SUBJECT_MAP = {0: 1, 1: 2, 2: 5, 3: 7}

# results from analyze_icsc_modules.py
VISUAL_MODULES = {
    1: [1, 7],
    2: [2, 9],
    5: [6],
    7: [0, 6],
}


def load_icsc_module_labels(icsc_results_path):

    # load the icsc results that show module assignments for all rois
    results = pd.read_csv(icsc_results_path, header=None)
    module_labels_dict = {}

    for idx, row in results.iterrows():
        subject_id = SUBJECT_MAP[int(row[0])]
        module_labels = row.iloc[5:].values.astype(int)
        module_labels_dict[subject_id] = module_labels
        print(f"Subject {subject_id:02d}: {len(module_labels)} ROI labels, {len(np.unique(module_labels))} modules")

    return module_labels_dict


def count_voxels_in_modules(subject_id, visual_module_ids, module_labels, atlas_path, brain_mask_path):
    
    # load brain mask
    brain_mask = np.load(str(brain_mask_path))
    total_brain_voxels = brain_mask.sum()

    # load atlas and flatten
    atlas_img = nib.load(str(atlas_path))
    atlas_data = atlas_img.get_fdata().astype(int)
    atlas_flat = atlas_data[brain_mask]

    # find which ROIs belong to visual modules
    visual_roi_mask = np.isin(module_labels, visual_module_ids)
    visual_roi_ids = np.where(visual_roi_mask)[0] + 1  # plus one since roi labels are 1-180
    
    # count number of voxels in visual modules
    num_visual_voxels = np.isin(atlas_flat, visual_roi_ids).sum()
    required = REQUIRED_VOXEL_COUNTS[subject_id]
    difference = num_visual_voxels - required

    # printing all results
    print(f"\nSubject {subject_id:02d}:")
    print(f"  Visual modules: {visual_module_ids}, {len(visual_roi_ids)} ROIs")
    print(f"  Brain voxels: {total_brain_voxels:,}, visual module voxels: {num_visual_voxels:,} "
          f"({num_visual_voxels / total_brain_voxels * 100:.1f}%)")
    print(f"  Required: {required:,}, available: {num_visual_voxels:,}, difference: {difference:+,}")

    if difference >= 0:
        print(f"  Sufficient — {(difference / required) * 100:.1f}% surplus")
    else:
        print(f"  Insufficient — need {abs(difference):,} more voxels")

    # voxels in each module
    print(f"  Per-module breakdown:")
    for module_id in visual_module_ids:
        rois = np.where(module_labels == module_id)[0] + 1
        voxels = np.isin(atlas_flat, rois).sum()
        print(f"    Module {module_id}: {voxels:,} voxels ({len(rois)} ROIs)")

    return {
        'subject_id': subject_id,
        'total_brain_voxels': int(total_brain_voxels),
        'visual_module_voxels': int(num_visual_voxels),
        'required_voxels': required,
        'difference': int(difference),
        'is_sufficient': difference >= 0,
    }


def main():
    icsc_results_path = Path(r'\path\ICSC_data\glasser_ICSC_data\subject_level_results\ICSC_subject_level_final_iter.csv')
    atlas_dir = Path(r'\path\atlas')
    brain_masks_dir = Path(r'\path\processed\brain_masks')

    module_labels_dict = load_icsc_module_labels(icsc_results_path)

    results = []
    for subject_id, visual_module_ids in VISUAL_MODULES.items():
        atlas_path = atlas_dir / 'Glasser' / f"subject_{subject_id}" / "HCP_MMP1.nii.gz"
        brain_mask_path = brain_masks_dir / f"brain_mask_subj{subject_id:02d}.npy"

        if not atlas_path.exists() or not brain_mask_path.exists():
            print(f"Skipping subject {subject_id:02d} — files not found")
            continue

        result = count_voxels_in_modules(
            subject_id=subject_id,
            visual_module_ids=visual_module_ids,
            module_labels=module_labels_dict[subject_id],
            atlas_path=atlas_path,
            brain_mask_path=brain_mask_path
        )
        results.append(result)

    # final summary table
    print(f"\n{'Subject':<10} {'Required':<12} {'Available':<12} {'Difference':<12} {'Status'}")
    print("-" * 60)
    for r in results:
        status = "GOOD" if r['is_sufficient'] else "INSUFFICIENT"
        print(f"{r['subject_id']:02d}        "
              f"{r['required_voxels']:<12,} "
              f"{r['visual_module_voxels']:<12,} "
              f"{r['difference']:>+12,} "
              f"{status}")


if __name__ == "__main__":
    main()