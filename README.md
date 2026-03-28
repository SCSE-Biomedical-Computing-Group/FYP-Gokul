# FYP-Gokul

(passage)


ICSC Algorithm : https://github.com/SCSE-Biomedical-Computing-Group/ICSC
MindAligner Model : https://github.com/Da1yuqin/MindAligner

# Order of Use

### download_nsd.py

Used for downloading the raw NSD data. Size of raw data is ~220GB.

### process_nsd.py

Used for processing the downloaded raw NSD data. The file uses brain masks to cut down voxel count from ~700k to ~350k for each subject.

### extract_brain_masks.py

Extracts brain masks used for each subject and saves them for further use.

### download_resting_state.py

Used for downloading resting state NSD data for use in the ICSC algorithm. 

### process_resting_state.py

### paercellate_resting_state.py

### create_ICSC_data.py

### analyse_ICSC_modules.py

### count_visual_module_voxels.py

### create_individualized_data.py

### extract_individualized_masks.py

### individualized_mask_reorder.py
