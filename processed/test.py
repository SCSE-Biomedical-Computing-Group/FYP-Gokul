# import h5py

# # Check what's in the full brain file
# with h5py.File('full_brain_subj01_all.hdf5', 'r') as f:
#     print("Keys:", list(f.keys()))
#     print("Shape:", f['betas'].shape)
#     print("Sample values:", f['betas'][0, :10])
#     print("Min:", f['betas'][0].min())
#     print("Max:", f['betas'][0].max())
import h5py

with h5py.File('full_brain_subj07_all.hdf5', 'r') as f:
    new_data = f['betas'][0]
    print(f"New individualized: [{new_data.min():.3f}, {new_data.max():.3f}]")
    print(f"Mean: {new_data.mean():.3f}, Std: {new_data.std():.3f}")