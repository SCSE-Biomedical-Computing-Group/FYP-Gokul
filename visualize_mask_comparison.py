"""
Visualize NSDGeneral vs individualized mask comparison.
Shows overlap, unique voxels, and spatial distribution across both masks.

Usage:
    python visualize_mask_comparison.py --subject 1
"""

import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from pathlib import Path
from nilearn import plotting
from nilearn.image import new_img_like


def create_comparison_mask(nsdgeneral_3d, individualized_3d):
    
    # 0 = neither, 1 = NSDGeneral only, 2 = individualized only, 3 = overlap
    comparison = np.zeros_like(nsdgeneral_3d, dtype=np.uint8)
    comparison[nsdgeneral_3d & ~individualized_3d] = 1 # NSDGeneral only
    comparison[~nsdgeneral_3d & individualized_3d] = 2 # Individualized only
    comparison[nsdgeneral_3d & individualized_3d] = 3 # Overlap
    return comparison


def print_statistics(nsdgeneral_3d, individualized_3d):
    
    # mask statistics
    nsd_num = int(nsdgeneral_3d.sum())
    indiv_num = int(individualized_3d.sum())
    overlap = int((nsdgeneral_3d & individualized_3d).sum())
    unique_nsd = int((nsdgeneral_3d & ~individualized_3d).sum())
    unique_ind = int((~nsdgeneral_3d & individualized_3d).sum())

    print(f"NSDGeneral: {nsd_num}, Individualized: {indiv_num:,}")
    print(f"Overlap: {overlap:,} ({100*overlap/nsd_num :.1f}% of NSDGeneral)")
    print(f"Unique to NSDGeneral: {unique_nsd:,} ({100*unique_nsd/nsd_num:.1f}%)")
    print(f"Unique to Individualized: {unique_ind:,} ({100*unique_ind/indiv_num:.1f}%)")


def visualize_with_nilearn(nsdgeneral_nii, individualized_nii, comparison_nii, subject, output_dir):
    print("Creating nilearn visualizations...")

    # axial + sagittal plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Subject {subject} — Mask Comparison", fontsize=14, fontweight='bold')

    for row, (display_mode, cut_label) in enumerate([('z', 'Axial'), ('x', 'Sagittal')]):
        plotting.plot_roi(nsdgeneral_nii, title=f'NSDGeneral ({cut_label})',
                          axes=axes[row, 0], display_mode=display_mode,
                          cut_coords=5, cmap='Reds', alpha=0.7)
        
        plotting.plot_roi(individualized_nii, title=f'Individualized ({cut_label})',
                          axes=axes[row, 1], display_mode=display_mode,
                          cut_coords=5, cmap='Blues', alpha=0.7)
        
        plotting.plot_roi(comparison_nii, title=f'Comparison ({cut_label})',
                          axes=axes[row, 2], display_mode=display_mode,
                          cut_coords=5, cmap='RdYlBu_r', alpha=0.7, vmin=0, vmax=3)

    plt.tight_layout()
    plt.savefig(output_dir / 'mask_comparison_nilearn.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved mask_comparison_nilearn.png")

    # glass brain
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"Subject {subject} — Glass Brain", fontsize=14, fontweight='bold')


    plotting.plot_glass_brain(nsdgeneral_nii, title='NSDGeneral', cmap='Reds',
                              alpha=0.8, threshold=0.5, vmax=1, colorbar=False, axes=axes[0])
    
    plotting.plot_glass_brain(individualized_nii, title='Individualized', cmap='Blues',
                              alpha=0.8, threshold=0.5, vmax=1, colorbar=False, axes=axes[1])
    
    plotting.plot_glass_brain(comparison_nii, title='Overlap', cmap='RdYlBu_r',
                              alpha=0.8, threshold=0.5, vmax=3, colorbar=False, axes=axes[2])

    plt.tight_layout()
    plt.savefig(output_dir / 'mask_comparison_glassbrain.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved mask_comparison_glassbrain.png")

    # seperate file for each of the three planes
    axial_cuts    = [-10, 0, 10]
    sagittal_cuts = [5, 15, 25]
    coronal_cuts  = [-70, -60, -50]

    for nii, label, cmap, fname in [
        (nsdgeneral_nii,     'NSDGeneral',     'Reds',      'ortho_nsdgeneral.png'),
        (individualized_nii, 'Individualized',  'PuBu',      'ortho_individualized.png'),
        (comparison_nii,     'Comparison',      'RdYlBu_r',  'ortho_comparison.png'),
    ]:
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        fig.suptitle(f"Subject {subject} — {label}", fontsize=14, fontweight='bold')

        plotting.plot_roi(nii, title='Axial',    axes=axes[0], display_mode='z',
                          cut_coords=axial_cuts,    cmap=cmap, alpha=0.8, vmin=0)
        
        plotting.plot_roi(nii, title='Coronal',  axes=axes[1], display_mode='y',
                          cut_coords=coronal_cuts,  cmap=cmap, alpha=0.8, vmin=0)
        
        plotting.plot_roi(nii, title='Sagittal', axes=axes[2], display_mode='x',
                          cut_coords=sagittal_cuts, cmap=cmap, alpha=0.8, vmin=0)
        
        plt.tight_layout()
        plt.savefig(output_dir / fname, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {fname}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=int, required=True, choices=[1, 2, 5, 7])
    args = parser.parse_args()
    subject = args.subject
    subj_str = f"subj{subject:02d}"

    nsdgeneral_path = Path(rf"\path\raw_nsd\nsdgeneral_masks\nsdgeneral_{subj_str}.nii.gz")
    individualized_mask_path = Path(rf"\path\individualized_masks\boolean_masks\glasser_betas_all_{subj_str}_fp32_mask.npy")
    brain_mask_path = Path(rf"\path\processed\brain_masks\brain_mask_{subj_str}.npy")
    output_dir = Path(rf"\path\visualizations\subject{subject}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Subject {subject}: {nsdgeneral_path.name} vs {individualized_mask_path.name}")

    nsdgeneral_nii = nib.load(str(nsdgeneral_path))
    nsdgeneral_3d  = nsdgeneral_nii.get_fdata(dtype=np.float32) > 0
    brain_shape    = nsdgeneral_3d.shape

    brain_mask_3d = np.load(str(brain_mask_path)).astype(bool)
    if brain_mask_3d.shape != brain_shape:
        raise ValueError(f"Brain mask shape {brain_mask_3d.shape} doesn't match NSDGeneral space {brain_shape}")

    individualized_1d = np.load(str(individualized_mask_path)).astype(bool)
    if individualized_1d.shape[0] != brain_mask_3d.sum():
        raise ValueError(f"Individualized mask length {individualized_1d.shape[0]} doesn't match brain voxel count {brain_mask_3d.sum()}")

    individualized_3d = np.zeros(brain_shape, dtype=bool)
    individualized_3d[brain_mask_3d] = individualized_1d

    print_statistics(nsdgeneral_3d, individualized_3d)
    comparison_3d = create_comparison_mask(nsdgeneral_3d, individualized_3d)

    nsdgeneral_clean = nsdgeneral_nii.get_fdata(dtype=np.float32).copy()
    nsdgeneral_clean[nsdgeneral_clean < 0] = 0
    visualize_with_nilearn(
        new_img_like(nsdgeneral_nii, nsdgeneral_clean),
        new_img_like(nsdgeneral_nii, individualized_3d.astype(np.float32)),
        new_img_like(nsdgeneral_nii, comparison_3d.astype(np.float32)),
        subject, output_dir
    )


    print(f"\nDone. Saved to {output_dir}")


if __name__ == "__main__":
    main()