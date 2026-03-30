"""
Analyze ICSC modules and identify visual modules using HCP_MMP1 atlas labels.

Usage:
    python analyze_icsc_modules.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# types gotten from the HCP_MMP1 lookup table
VISUAL_CORTEX_TYPES = [
    "Primary_Visual",
    "Early_Visual",
    "Dorsal_Stream_Visual",
    "Ventral_Stream_Visual",
    "MT+_Complex_and_Neighboring_Visual_Areas"
]

# for the four subjects
SUBJECT_MAP = {0: 1, 1: 2, 2: 5, 3: 7}


def load_icsc_results(results_path):

    # load ICSC results
    results = pd.read_csv(results_path, header=None)
    print(f"ICSC results: {results.shape[0]} subjects, {results.shape[1]} columns")

    parsed_results = {}
    
    for idx, row in results.iterrows():
        run_id = int(row[0])
        iteration = int(row[1])
        consensus_cost = float(row[2])
        num_adjusted = int(row[3])
        num_modules = int(row[4])
        
        # extract module labels for all rois (starts after the first 4 columns)
        module_labels = row.iloc[5:].values.astype(int)
        
        parsed_results[run_id] = {
            'iteration': iteration,
            'consensus_cost': consensus_cost,
            'num_adjusted': num_adjusted,
            'num_modules': num_modules,
            'module_labels': module_labels
        }
        
        print(f"\nRun {run_id}:")
        print(f"  Converged at iteration: {iteration}")
        print(f"  Consensus cost: {consensus_cost:.4f}")
        print(f"  Number of modules: {num_modules}")
        print(f"  ROI labels shape: {module_labels.shape}")
        print(f"  Unique modules: {np.unique(module_labels)}")
        
        # verify number of rois
        assert len(module_labels) == 180, f"Expected 180 ROIs, got {len(module_labels)}"
        assert len(np.unique(module_labels)) == num_modules, \
            f"Expected {num_modules} unique modules, got {len(np.unique(module_labels))}"

    return parsed_results


def load_hcp_lookup(lookup_path):

    #load hcp lookup table
    lookup = pd.read_csv(lookup_path)
    print(f"Lookup table: {lookup.shape}, columns: {lookup.columns.tolist()}")
    print(f"RegionID range: {lookup['regionID'].min()} to {lookup['regionID'].max()}")

    print(f"\nCortex types ({len(lookup['cortex'].unique())}):")
    for cortex_type in sorted(lookup['cortex'].unique()):
        count = (lookup['cortex'] == cortex_type).sum()
        visual = "  <- visual" if cortex_type in VISUAL_CORTEX_TYPES else "" # to indicate in the output
        print(f"  {cortex_type}: {count}{visual}")

    return lookup


def analyze_module_composition(module_id, roi_indices, lookup):

    # find info about rois from lookup 
    module_rois = lookup[lookup['regionID'].isin(roi_indices)]
    total_rois = len(module_rois)

    # percentage of visual modules
    visual_rois = module_rois[module_rois['cortex'].isin(VISUAL_CORTEX_TYPES)]
    num_visual = len(visual_rois)
    visual_percent = num_visual / total_rois * 100 if total_rois > 0 else 0

    visual_systems = {}
    for vt in VISUAL_CORTEX_TYPES:
        count = int((module_rois['cortex'] == vt).sum())
        if count > 0:
            visual_systems[vt] = count

    return {
        'module_id': module_id,
        'total_rois': total_rois,
        'num_visual': num_visual,
        'visual_percent': visual_percent,
        'cortex_counts': module_rois['cortex'].value_counts().to_dict(),
        'lobe_counts': module_rois['Lobe'].value_counts().to_dict(),
        'visual_systems': visual_systems,
        'roi_indices': roi_indices.tolist(),
        'roi_names': module_rois['regionName'].tolist()
    }


def analyze_subject_modules(run_id, subject_data, lookup, visual_threshold=0.7):
    
    subject_id = SUBJECT_MAP[run_id]
    module_labels = subject_data['module_labels']
    num_modules = subject_data['num_modules']

    print(f"\nSubject {subject_id:02d} (run_id={run_id}): {num_modules} modules, "
          f"threshold {visual_threshold*100:.0f}%")
    
    # to make a table in the output
    print(f"{'Module':<8} {'ROIs':<6} {'Visual':<8} {'%Visual':<8} {'Status':<12} {'Main Systems'}")
    print("-" * 90)

    module_analyses = []
    visual_modules = []

    for module_id in range(num_modules):

        # find the rois in this module
        roi_indices = np.where(module_labels == module_id)[0] + 1

        
        analysis = analyze_module_composition(module_id, roi_indices, lookup)

        # determine if visual module or not
        is_visual = analysis['visual_percent'] >= visual_threshold * 100

        main_systems = sorted(analysis['cortex_counts'].items(), key=lambda x: x[1], reverse=True)[:3]
        main_systems_str = ", ".join([f"{name}({count})" for name, count in main_systems])
        status = "VISUAL" if is_visual else "non-visual"

        # print table
        print(f"{module_id:<8} {analysis['total_rois']:<6} {analysis['num_visual']:<8} "
              f"{analysis['visual_percent']:<8.1f} {status:<12} {main_systems_str[:50]}")

        if is_visual:
            visual_modules.append(module_id)
        module_analyses.append(analysis)

    print(f"\nVisual modules: {visual_modules} ({len(visual_modules)}/{num_modules})")

    return {
        'subject_id': subject_id,
        'run_id': run_id,
        'num_modules': num_modules,
        'visual_modules': visual_modules,
        'module_analyses': module_analyses,
        'consensus_cost': subject_data['consensus_cost']
    }


def main():
    icsc_results_path = Path(r'\path\ICSC_data\glasser_ICSC_data\subject_level_results\ICSC_subject_level_final_iter.csv')
    hcp_lookup_path = Path(r'\path\atlas\Glasser\HCPMMP1_UniqueRegionList.csv')
    output_dir = Path(r'\path\ICSC_data\glasser_ICSC_data\module_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    icsc_results = load_icsc_results(icsc_results_path)
    hcp_lookup = load_hcp_lookup(hcp_lookup_path)

    for run_id in sorted(icsc_results.keys()):

        # analysis can be used for exploration
        analysis = analyze_subject_modules(run_id, icsc_results[run_id], hcp_lookup)


    print("**Finished**")

if __name__ == "__main__":
    main()