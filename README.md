# FYP-Gokul

(passage)


ICSC Algorithm : https://github.com/SCSE-Biomedical-Computing-Group/ICSC
MindAligner Model : https://github.com/Da1yuqin/MindAligner

# Order of Use

### 1.download_nsd.py

Used for downloading the raw task NSD data. Size of raw data is ~220GB.

### 2.process_nsd.py

Used for processing the downloaded raw task NSD data. The file uses brain masks to cut down voxel count from ~700k to ~350k for each subject.

### 3.extract_brain_masks.py

Extracts brain masks used for each subject and saves them for further use.

### 4.download_resting_state.py

Used for downloading resting state NSD data for use in the ICSC algorithm. Size of raw data is ~5GB.

### 5.process_resting_state.py

Used for processing the resting state data. Uses the previously saved brain mask from task data to process this data.

### 6.paercellate_resting_state.py

Uses the HCP_MMP1 atlas provided by the authors to parcellate the processed resting state data into 180 ROIs.

### 7.create_ICSC_data.py

Creates correlation matrices from the processed resting state data, for use with the ICSC algorithm.

### 8.analyse_ICSC_modules.py

Determines which of the modules are visual, using the lookup table for the HCP_MMP1 atlas.

### 9.count_visual_module_voxels.py

Counts the number of voxels within the visual modules for each subject, against the required voxels from NSDGeneral

### 10.create_individualized_data.py

Uses the voxels from the visual modules, and chooses voxels according to R2 score to create a individualized mask. Then uses this mask to process the task NSD data to create individualized data for each subject.

### 11.extract_individualized_masks.py

Extracts the individualized masks that were used to create the individualized data.

### 12.individualized_mask_reorder.py

Performs reordering on the individualized masks according to functional corresspondence + spatial proximity and creates a reordered mask. Uses this mask to create reordered data for each subject.
