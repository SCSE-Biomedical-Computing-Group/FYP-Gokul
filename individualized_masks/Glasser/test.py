# Compare AFTER ROI selection
import h5py

with h5py.File('individualized_r2_subj07.hdf5', 'r') as f:
    new_data = f['betas'][0]
    print(f"New individualized: [{new_data.min():.3f}, {new_data.max():.3f}]")
    print(f"Mean: {new_data.mean():.3f}, Std: {new_data.std():.3f}")