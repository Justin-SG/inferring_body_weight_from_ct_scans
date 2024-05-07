# import nibabel as nib
# import numpy as np
# import os

# # Load NIfTI files
# file_paths = ["test/heart.nii.gz", "test/kidney_right.nii.gz", "test/kidney_left.nii.gz"]
# nii_objects = [nib.load(file_path) for file_path in file_paths]

# # Check compatibility (dimensions, orientations, voxel sizes, etc.)

# # Concatenate data along the appropriate axis
# data = np.concatenate([nii.get_fdata() for nii in nii_objects], axis=2)

# # Create a new NIfTI image with the concatenated data
# combined_nii = nib.Nifti1Image(data, affine=nii_objects[0].affine)

# os.makedirs("combined_test", exist_ok=True)

# # Save the combined NIfTI file
# nib.save(combined_nii, "combined_test/combined_scan.nii.gz")

import nibabel as nib
import numpy as np

# Load NIfTI files
file_paths = ["test/heart.nii.gz", "test/kidney_right.nii.gz", "test/kidney_left.nii.gz"]
nii_objects = [nib.load(file_path) for file_path in file_paths]

# Ensure compatibility (dimensions, orientations, voxel sizes, etc.)

# Get data from each NIfTI file
data_arrays = [nii.get_fdata() for nii in nii_objects]

# Initialize an empty array to hold the merged data
merged_data = np.zeros_like(data_arrays[0])

# Iterate over each voxel and combine the values from all images
for i in range(merged_data.shape[0]):
    for j in range(merged_data.shape[1]):
        for k in range(merged_data.shape[2]):
            voxel_values = [data_array[i, j, k] for data_array in data_arrays]
            merged_data[i, j, k] = np.mean(voxel_values)  # Example: averaging voxel values

# Create a new NIfTI image with the merged data
merged_nii = nib.Nifti1Image(merged_data, affine=nii_objects[0].affine)

# Save the merged NIfTI file
nib.save(merged_nii, "combined_test/combined_scan.nii.gz")
