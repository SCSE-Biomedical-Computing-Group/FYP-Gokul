# FYP-Gokul

## About
This project explores the use of individualized brain masks for fMRI-to-image reconstruction, 
building on the MindAligner model. The standard MindAligner uses a population-level visual ROI 
mask (NSDGeneral) which is the same across all subjects. This project instead derives 
subject-specific masks using the ICSC algorithm on resting-state fMRI data, which identifies 
functional brain modules unique to each subject. Voxels are then selected from the visual 
modules based on R² responsiveness scores, producing individualized masks that better reflect 
each subject's brain organization. The goal is to evaluate whether individualized masks improve 
reconstruction quality over the population-level baseline.

**ICSC Algorithm:** https://github.com/SCSE-Biomedical-Computing-Group/ICSC  
**MindAligner Model:** https://github.com/Da1yuqin/MindAligner

---

# How to Run

## 1.Install dependencies
Clone the repo and run the following in the root directory:

'''bash
pip install -r requirements.text
'''

## 2.Order of Codes

| Step | Script | Description |
|------|--------|-------------|
| 1 | `download_nsd.py` | Download raw task NSD data (~220GB) |
| 2 | `process_nsd.py` | Process raw task data, reducing voxels from ~700k to ~350k per subject using brain masks |
| 3 | `extract_brain_masks.py` | Extract and save brain masks per subject for downstream use |
| 4 | `download_resting_state.py` | Download resting state NSD data (~5GB) |
| 5 | `process_resting_state.py` | Process resting state data using the saved task brain masks |
| 6 | `parcellate_resting_state.py` | Parcellate processed resting state data into 180 ROIs using the HCP_MMP1 atlas |
| 7 | `create_ICSC_data.py` | Create per-session correlation matrices from resting state data for ICSC |
| 8 | `analyse_ICSC_modules.py` | Identify visual modules from ICSC output using the HCP_MMP1 atlas lookup table |
| 9 | `count_visual_module_voxels.py` | Count voxels in visual modules per subject and compare against NSDGeneral requirements |
| 10 | `create_individualized_data.py` | Select voxels from visual modules by R² score to create individualized masks and data |
| 11 | `extract_individualized_masks.py` | Extract the boolean masks used to create the individualized data |
| 12 | `individualized_mask_reorder.py` | Reorder individualized masks by functional correspondence and spatial proximity to align with NSDGeneral positions |

## 3.Training, Reconstruction, Evaluation

Training, reconstruction, and evaluation are performed using the MindAligner codebase at https://github.com/Da1yuqin/MindAligner. Refer to their instructions for using their model.
