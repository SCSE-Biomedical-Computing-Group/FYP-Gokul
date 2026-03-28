import nibabel as nib
import numpy as np

atlas = nib.load("HCP_MMP1.nii.gz")
data = atlas.get_fdata().astype(int)
unique = np.unique(data)
print(unique[unique > 0])