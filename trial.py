"""
Visualize NSDGeneral vs Individualized Mask Comparison
======================================================
Creates publication-quality brain visualizations showing:
1. NSDGeneral mask
2. Individualized mask
3. Overlap (shared voxels)
4. Unique to NSDGeneral
5. Unique to Individualized

Usage:
    python visualize_mask_comparison.py --subject 1
    python visualize_mask_comparison.py --subject 1 --nsdgeneral_mask path/to/nsdgeneral.nii.gz --individualized_mask path/to/mask.npy --brain_ref path/to/betas_session01.nii.gz
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




def load_brain_mask(brain_mask_path):
    """
    Load pre-saved 3D brain mask .npy (shape: X, Y, Z).
    """
    return np.load(str(brain_mask_path)).astype(bool)


def create_comparison_mask(nsdgeneral_3d, individualized_3d):
    """
    Create a comparison mask showing different regions.

    Values:
        0 = Neither mask
        1 = Only NSDGeneral (unique to population)
        2 = Only Individualized (unique to subject)
        3 = Both (overlap)
    """
    comparison = np.zeros_like(nsdgeneral_3d, dtype=np.uint8)
    comparison[nsdgeneral_3d & ~individualized_3d] = 1   # NSDGeneral only
    comparison[~nsdgeneral_3d & individualized_3d] = 2   # Individualized only
    comparison[nsdgeneral_3d & individualized_3d] = 3    # Overlap
    return comparison


def print_statistics(nsdgeneral_3d, individualized_3d):
    """Print overlap statistics."""
    print("\n" + "=" * 70)
    print("MASK STATISTICS")
    print("=" * 70)

    nsdgeneral_count    = int(nsdgeneral_3d.sum())
    individualized_count = int(individualized_3d.sum())
    overlap_count       = int((nsdgeneral_3d & individualized_3d).sum())
    unique_nsdgeneral   = int((nsdgeneral_3d & ~individualized_3d).sum())
    unique_individualized = int((~nsdgeneral_3d & individualized_3d).sum())

    print(f"\nNSDGeneral voxels:        {nsdgeneral_count:,}")
    print(f"Individualized voxels:    {individualized_count:,}")
    print(f"\nOverlap:                  {overlap_count:,}  ({100*overlap_count/nsdgeneral_count:.1f}% of NSDGeneral)")
    print(f"Unique to NSDGeneral:     {unique_nsdgeneral:,}  ({100*unique_nsdgeneral/nsdgeneral_count:.1f}% of NSDGeneral)")
    # FIX: use individualized_count as denominator here, not nsdgeneral_count
    print(f"Unique to Individualized: {unique_individualized:,}  ({100*unique_individualized/individualized_count:.1f}% of Individualized)")


def visualize_with_nilearn(nsdgeneral_nii, individualized_nii, comparison_nii, subject, output_dir):
    """Create brain visualizations using nilearn."""
    print("\nCreating nilearn visualizations...")

    # ── Figure 1: axial + sagittal side-by-side ──────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Subject {subject} – Mask Comparison", fontsize=14, fontweight='bold')

    for row, display_mode, cut_label in [(0, 'z', 'Axial'), (1, 'x', 'Sagittal')]:
        plotting.plot_roi(nsdgeneral_nii,
                          title=f'NSDGeneral ({cut_label})',
                          axes=axes[row, 0],
                          display_mode=display_mode,
                          cut_coords=5,
                          cmap='Reds',
                          alpha=0.7)

        plotting.plot_roi(individualized_nii,
                          title=f'Individualized ({cut_label})',
                          axes=axes[row, 1],
                          display_mode=display_mode,
                          cut_coords=5,
                          cmap='Blues',
                          alpha=0.7)

        plotting.plot_roi(comparison_nii,
                          title=f'Comparison ({cut_label})',
                          axes=axes[row, 2],
                          display_mode=display_mode,
                          cut_coords=5,
                          cmap='RdYlBu_r',
                          alpha=0.7,
                          vmin=0, vmax=3)

    plt.tight_layout()
    out = output_dir / 'mask_comparison_nilearn.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {out}")
    plt.close()

    # ── Figure 2: glass brain ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"Subject {subject} – Glass Brain View", fontsize=14, fontweight='bold')

    # threshold=0.5 ensures only voxels with value > 0.5 (i.e. ==1) are rendered
    # nsdgeneral_nii here is already the cleaned version (no -1 values), passed from main()
    plotting.plot_glass_brain(nsdgeneral_nii,
                              title='NSDGeneral',
                              cmap='Reds', alpha=0.8,
                              threshold=0.5, vmax=1,
                              colorbar=False, axes=axes[0])
    plotting.plot_glass_brain(individualized_nii,
                              title='Individualized',
                              cmap='Blues', alpha=0.8,
                              threshold=0.5, vmax=1,
                              colorbar=False, axes=axes[1])
    plotting.plot_glass_brain(comparison_nii,
                              title='Overlap Comparison',
                              cmap='RdYlBu_r', alpha=0.8,
                              threshold=0.5, vmax=3,
                              colorbar=False, axes=axes[2])

    plt.tight_layout()
    out = output_dir / 'mask_comparison_glassbrain.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {out}")
    plt.close()

    # ── Figure 3a/3b/3c: one ortho file per mask ─────────────────────────────
    #axial_cuts    = [-10, 0, 10, 20, 30, 40]
    #sagittal_cuts = [-25, -15, -5, 5, 15, 25]
    #coronal_cuts  = [-90, -80, -70, -60, -50, -40]

    axial_cuts    = [-10, 0, 10]
    sagittal_cuts = [5, 15, 25]
    coronal_cuts  = [-70, -60, -50]

    for nii, label, cmap, fname in [
        (nsdgeneral_nii,     'NSDGeneral',    'Reds',      'ortho_nsdgeneral.png'),
        (individualized_nii, 'Individualized', 'PuBu',    'ortho_individualized.png'),
        (comparison_nii,     'Comparison',    'RdYlBu_r', 'ortho_comparison.png'),
    ]:
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        fig.suptitle(f"Subject {subject} – {label} (Visual Cortex)", fontsize=14, fontweight='bold')

        plotting.plot_roi(nii, title='Axial',
                          axes=axes[0], display_mode='z',
                          cut_coords=axial_cuts, cmap=cmap, alpha=0.8, vmin=0)
        plotting.plot_roi(nii, title='Coronal',
                          axes=axes[1], display_mode='y',
                          cut_coords=coronal_cuts, cmap=cmap, alpha=0.8, vmin=0)
        plotting.plot_roi(nii, title='Sagittal',
                          axes=axes[2], display_mode='x',
                          cut_coords=sagittal_cuts, cmap=cmap, alpha=0.8, vmin=0)

        plt.tight_layout()
        out = output_dir / fname
        plt.savefig(out, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {out}")
        plt.close()


def visualize_with_matplotlib(nsdgeneral_3d, individualized_3d, comparison_3d, subject, output_dir):
    """Create slice visualizations using matplotlib (fallback / always runs)."""
    print("\nCreating matplotlib slice visualizations...")

    # FIX: ListedColormap lives in matplotlib.colors, not plt.cm.colors
    cmap_comparison = ListedColormap(['black', 'red', 'blue', 'purple'])

    # Pick informative slices – centre-of-mass of the union mask
    union = nsdgeneral_3d | individualized_3d
    coords = np.argwhere(union)
    z_mid = int(np.median(coords[:, 2]))
    y_mid = int(np.median(coords[:, 1]))
    x_mid = int(np.median(coords[:, 0]))

    slices = {
        'Axial'   : (nsdgeneral_3d[:, :, z_mid].T,    individualized_3d[:, :, z_mid].T,    comparison_3d[:, :, z_mid].T),
        'Coronal' : (nsdgeneral_3d[:, y_mid, :].T,    individualized_3d[:, y_mid, :].T,    comparison_3d[:, y_mid, :].T),
        'Sagittal': (nsdgeneral_3d[x_mid, :, :].T,    individualized_3d[x_mid, :, :].T,    comparison_3d[x_mid, :, :].T),
    }

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f"Subject {subject} – Mask Comparison (Slices)", fontsize=14, fontweight='bold')

    for row, (plane, (nsd_sl, ind_sl, comp_sl)) in enumerate(slices.items()):
        axes[row, 0].imshow(nsd_sl,  cmap='Reds',  origin='lower')
        axes[row, 0].set_title(f'NSDGeneral ({plane})')
        axes[row, 0].axis('off')

        axes[row, 1].imshow(ind_sl,  cmap='Blues', origin='lower')
        axes[row, 1].set_title(f'Individualized ({plane})')
        axes[row, 1].axis('off')

        axes[row, 2].imshow(comp_sl, cmap=cmap_comparison, origin='lower', vmin=0, vmax=3)
        axes[row, 2].set_title(f'Comparison ({plane})')
        axes[row, 2].axis('off')

    # Shared legend for comparison panels
    legend_elements = [
        Patch(facecolor='red',    label='NSDGeneral only'),
        Patch(facecolor='blue',   label='Individualized only'),
        Patch(facecolor='purple', label='Overlap'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=12,
               bbox_to_anchor=(0.72, 0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = output_dir / 'mask_comparison_slices.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {out}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=int, required=True, choices=[1, 2, 5, 7])
    parser.add_argument('--nsdgeneral_mask', type=str, default=None,
                        help='Path to NSDGeneral NIfTI mask (.nii.gz)')
    parser.add_argument('--individualized_mask', type=str, default=None,
                        help='Path to individualized boolean mask (.npy)')
    parser.add_argument('--brain_mask', type=str, default=None,
                        help='Path to pre-saved brain mask .npy (1D boolean array)')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    subject = args.subject

    # ── Resolve paths ─────────────────────────────────────────────────────────
    subj_str = f"subj{subject:02d}"  # FIX: consistent zero-padding

    nsdgeneral_path = Path(args.nsdgeneral_mask) if args.nsdgeneral_mask else \
        Path(rf"G:\NSDdata\raw_nsd\nsdgeneral_masks\nsdgeneral_{subj_str}.nii.gz")

    individualized_mask_path = Path(args.individualized_mask) if args.individualized_mask else \
        Path(rf"G:\NSDdata\individualized_masks\boolean_masks\glasser_betas_all_{subj_str}_fp32_mask.npy")

    brain_mask_path = Path(args.brain_mask) if args.brain_mask else \
        Path(rf"G:\NSDdata\processed\brain_masks\brain_mask_{subj_str}.npy")

    output_dir = Path(args.output_dir) if args.output_dir else \
        Path(rf"G:\NSDdata\visualizations\subject{subject}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"VISUALIZING MASK COMPARISON FOR SUBJECT {subject}")
    print("=" * 70)
    print(f"NSDGeneral mask:      {nsdgeneral_path}")
    print(f"Individualized mask:  {individualized_mask_path}")
    print(f"Brain mask:           {brain_mask_path}")
    print(f"Output directory:     {output_dir}")

    # ── Load NSDGeneral ───────────────────────────────────────────────────────
    print(f"\nLoading NSDGeneral mask...")
    nsdgeneral_nii = nib.load(str(nsdgeneral_path))
    nsdgeneral_3d  = nsdgeneral_nii.get_fdata(dtype=np.float32) > 0
    brain_shape    = nsdgeneral_3d.shape  # use NSDGeneral header as canonical space

    # ── Load pre-saved brain mask ─────────────────────────────────────────────
    print(f"Loading brain mask...")
    brain_mask_3d = load_brain_mask(brain_mask_path)

    if brain_mask_3d.shape != brain_shape:
        raise ValueError(
            f"Brain mask shape {brain_mask_3d.shape} does not match "
            f"NSDGeneral space {brain_shape}."
        )

    # ── Load individualized mask and project into 3D ──────────────────────────
    print(f"Loading individualized mask...")
    individualized_1d = np.load(str(individualized_mask_path)).astype(bool)

    if individualized_1d.shape[0] != brain_mask_3d.sum():
        raise ValueError(
            f"Individualized mask length ({individualized_1d.shape[0]}) "
            f"does not match number of brain voxels in brain_mask ({brain_mask_3d.sum()}). "
            f"Ensure brain_mask_{subj_str}.npy was saved from the same processing run."
        )

    individualized_3d = np.zeros(brain_shape, dtype=bool)
    individualized_3d[brain_mask_3d] = individualized_1d

    # ── Statistics ────────────────────────────────────────────────────────────
    print_statistics(nsdgeneral_3d, individualized_3d)

    # ── Comparison mask ───────────────────────────────────────────────────────
    print(f"\nCreating comparison mask...")
    comparison_3d = create_comparison_mask(nsdgeneral_3d, individualized_3d)

    # ── Visualize ─────────────────────────────────────────────────────────────
    # ── Build nilearn images ──────────────────────────────────────────────────
    # The NSDGeneral NIfTI contains -1 values (non-brain fill) alongside 0 and 1.
    # plot_glass_brain projects absolute maximum along each axis, so -1 renders
    # as magnitude 1 and floods the silhouette. Zero them out before plotting.
    nsdgeneral_clean = nsdgeneral_nii.get_fdata(dtype=np.float32).copy()
    nsdgeneral_clean[nsdgeneral_clean < 0] = 0
    nsdgeneral_nii_clean = new_img_like(nsdgeneral_nii, nsdgeneral_clean)

    if NILEARN_AVAILABLE:
        individualized_nii = new_img_like(nsdgeneral_nii, individualized_3d.astype(np.float32))
        comparison_nii     = new_img_like(nsdgeneral_nii, comparison_3d.astype(np.float32))
        visualize_with_nilearn(nsdgeneral_nii_clean, individualized_nii, comparison_nii, subject, output_dir)

    visualize_with_matplotlib(nsdgeneral_3d, individualized_3d, comparison_3d, subject, output_dir)

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print("\nFiles created:")
    for f in sorted(output_dir.iterdir()):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()