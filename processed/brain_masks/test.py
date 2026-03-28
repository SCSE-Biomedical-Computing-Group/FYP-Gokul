# # import numpy as np, nibabel as nib
# # m = np.load(r"G:\NSDdata\processed\brain_masks\brain_mask_subj01.npy")
# # nsd = nib.load(r"G:\NSDdata\raw_nsd\nsdgeneral_masks\nsdgeneral_subj01.nii.gz")
# # print(m.shape, m.sum())   # should be 3D or 1D
# # print(nsd.shape)           # should be same spatial dims
# import numpy as np
# import nibabel as nib

# nii = nib.load(r"G:\NSDdata\raw_nsd\nsdgeneral_masks\nsdgeneral_subj01.nii.gz")
# data = nii.get_fdata(dtype=np.float32)

# print("Shape:", data.shape)
# print("Dtype:", data.dtype)
# print("Unique values (first 20):", np.unique(data)[:20])
# print("Total > 0:", (data > 0).sum())
# print("Total == 1:", (data == 1).sum())
# print("Min/Max:", data.min(), data.max())

import numpy as np, nibabel as nib
nii = nib.load(r"G:\NSDdata\raw_nsd\nsdgeneral_masks\nsdgeneral_subj01.nii.gz")
data = nii.get_fdata(dtype=np.float32)
print("Count of -1 values:", (data == -1).sum())
print("Count of  0 values:", (data ==  0).sum())
print("Count of  1 values:", (data ==  1).sum())