import nibabel as nib

img = nib.load(r'G:\NSDdata\raw_nsd\subj01\betas_session01.nii.gz')

print("Data type in file:", img.dataobj.dtype)
print("Slope:", img.dataobj.slope)
print("Intercept:", img.dataobj.inter)

# Load with automatic scaling
data = img.get_fdata()
print("After get_fdata():", data.min(), "to", data.max())