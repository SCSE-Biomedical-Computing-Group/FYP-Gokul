"""
Reorder individualized voxels to match NSDGeneral positions based on
functional correspondence, using same HCP-MMP1 regions.

Usage:
    python smart_reorder_individualized_mask.py --subject 1
"""

import argparse
import numpy as np
import nibabel as nib
import h5py
from pathlib import Path


def load_hcp_mmp1_atlas(subject):

    # load atlas
    atlas_path = Path(rf"G:\NSDdata\atlas\Glasser\subject_{subject}\HCP_MMP1.nii.gz")
    if not atlas_path.exists():
        print(f"Atlas not found at {atlas_path}, falling back to spatial-only reordering")
        return None
    print(f"Loaded atlas: {atlas_path}")
    return nib.load(atlas_path).get_fdata()


def get_voxel_info(mask_3d, atlas_3d=None):
   
    # make dictionary with index, coords, region
    voxel_coords = np.array(np.where(mask_3d)).T  # [N, 3]
    voxel_info = []
    for i, (x, y, z) in enumerate(voxel_coords):
        info = {'index': i, 'coords': np.array([x, y, z])}
        if atlas_3d is not None:
            info['region'] = int(atlas_3d[x, y, z])
        voxel_info.append(info)
    return voxel_info


def smart_reorder(nsdgeneral_info, individualized_info):
    
    n_nsdgeneral = len(nsdgeneral_info)
    n_individualized = len(individualized_info)
    
    # check both lists have region info
    has_atlas = 'region' in nsdgeneral_info[0] and 'region' in individualized_info[0]

    print(f"NSDGeneral voxels: {n_nsdgeneral}, individualized voxels: {n_individualized}")
    
    if has_atlas:
        print("Using functional correspondence (HCP-MMP1 regions) + spatial proximity")
    else:
        print("Using spatial proximity only (no atlas found)")

    # to make sure already assigned voxels dont get reassigned
    used = np.zeros(n_individualized, dtype=bool)
    reorder_mapping = np.zeros(n_individualized, dtype=int)

    # process starting from lowest index
    for target_pos in range(min(n_nsdgeneral, n_individualized)):
        
        # save best score and its index
        nsd_voxel = nsdgeneral_info[target_pos]
        best_idx, best_score = None, float('inf')

        # loop through all voxels in individualized mask
        for ind_idx, ind_voxel in enumerate(individualized_info):

            # check voxel isnt already assigned    
            if used[ind_idx]:
                continue
            
            # calculate matching score (lower = better)
            if has_atlas and nsd_voxel['region'] == ind_voxel['region']:
                # same region, so use spatial distance
                spatial_dist = np.linalg.norm(
                    nsd_voxel['coords'] - ind_voxel['coords']
                )
                score = spatial_dist  # Prioritize same-region matches
            else:
                # if no atlas or different region
                spatial_dist = np.linalg.norm(
                    nsd_voxel['coords'] - ind_voxel['coords']
                )
                # penalty score so same region voxels are prioritised
                score = spatial_dist + 1000
            
            if score < best_score:
                best_score = score
                best_idx = ind_idx

        # save best index and mark it as used
        reorder_mapping[target_pos] = best_idx
        used[best_idx] = True

        # progress bar
        if (target_pos + 1) % 1000 == 0:
            print(f"  Mapped {target_pos + 1}/{min(n_nsdgeneral, n_individualized)}...")

    print("LOOP IS DONE")

    return reorder_mapping


def analyze_reordering(reorder_mapping, nsdgeneral_info, individualized_info):
    
    has_atlas = 'region' in nsdgeneral_info[0]
    n = min(len(nsdgeneral_info), len(reorder_mapping))

    if has_atlas:

        # for each pair, check if in same region
        same_region = sum(
            nsdgeneral_info[i]['region'] == individualized_info[reorder_mapping[i]]['region']
            for i in range(n)
        )
        print(f"Same HCP region: {same_region}/{n} ({100 * same_region / n:.1f}%)")

    # for each pair, get distance between
    dists = [np.linalg.norm(nsdgeneral_info[i]['coords'] - individualized_info[reorder_mapping[i]]['coords'])
             for i in range(n)]
    
    # statistics across all voxels
    print(f"Spatial displacement — mean: {np.mean(dists):.2f}, median: {np.median(dists):.2f}, max: {np.max(dists):.2f} voxels")


def apply_reordering(data_file, reorder_mapping, output_file):
    
    with h5py.File(data_file, 'r') as f_in:
        betas = f_in['betas'][:]  # [trials, voxels]
        
        print(f"Original data shape: {betas.shape}")
        
        # reorder voxels (second dimension)
        reordered_betas = betas[:, reorder_mapping]
        
        print(f"Reordered data shape: {reordered_betas.shape}")
        
        # save to new file
        with h5py.File(output_file, 'w') as f_out:
            f_out.create_dataset(
                'betas',
                data=reordered_betas,
                dtype=betas.dtype,
                compression='gzip'
            )
            
            # copy any other things as well
            for key in f_in.keys():
                if key != 'betas':
                    f_out.create_dataset(key, data=f_in[key][:])

    print(f"Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=int, required=True, choices=[1, 2, 5, 7])
    args = parser.parse_args()
    subject = args.subject

    nsdgeneral_path = Path(rf"\path\raw_nsd\nsdgeneral_masks\nsdgeneral_subj0{subject}.nii.gz")
    individualized_mask_path = Path(rf"\path\individualized_masks\boolean_masks\glasser_betas_all_subj0{subject}_fp32_mask.npy")
    data_file = Path(rf"\path\individualized_masks\Glasser\individualized_r2_subj0{subject}_unordered.hdf5")
    output_file = Path(rf"\path\individualized_masks\Glasser\glasser_betas_all_subj0{subject}_fp32_renorm_REORDERED.hdf5")
    mapping_file = Path(rf"\path\individualized_masks\boolean_masks\reorder_mapping_subj0{subject}.npy")

    nsdgeneral_3d = nib.load(nsdgeneral_path).get_fdata() > 0
    print(f"NSDGeneral voxels: {nsdgeneral_3d.sum()}")

    individualized_1d = np.load(individualized_mask_path)
    ref_nifti = nib.load(rf"G:\NSDdata\raw_nsd\subj0{subject}\betas_session01.nii.gz")
    brain_mask = np.any(ref_nifti.get_fdata() != 0, axis=-1)
    individualized_3d = np.zeros(ref_nifti.shape[:3], dtype=bool)
    individualized_3d[brain_mask] = individualized_1d
    print(f"Individualized voxels: {individualized_3d.sum()}")

    atlas_3d = load_hcp_mmp1_atlas(subject)

    nsdgeneral_info = get_voxel_info(nsdgeneral_3d, atlas_3d)
    individualized_info = get_voxel_info(individualized_3d, atlas_3d)

    reorder_mapping = smart_reorder(nsdgeneral_info, individualized_info)
    analyze_reordering(reorder_mapping, nsdgeneral_info, individualized_info)

    np.save(mapping_file, reorder_mapping)
    print(f"Saved reordering mapping: {mapping_file}")

    if data_file.exists():
        apply_reordering(data_file, reorder_mapping, output_file)
    else:
        print(f"Data file not found: {data_file}, mapping saved but data not reordered")


if __name__ == "__main__":
    main()