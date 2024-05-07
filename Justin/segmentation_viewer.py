# Load nii.gz file and display slice in the middle of the volume

import os
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def display_nifti_files_2d(nifti_files):
    for file_path in nifti_files:
        img = nib.load(file_path)
        data = img.get_fdata()
        slice = data.shape[2] // 2
        plt.imshow(data[:, :, slice], cmap='gray')
        plt.title(file_path)
        plt.show()
        
# Example usage:
# Load all *.nii.gz files from the test folder
nifti_files = ["test/" + file for file in os.listdir("test") if file.endswith(".nii.gz")]

#nifti_files = ["test/heart.nii.gz", "test/kidney_left.nii.gz", "test/kidney_right.nii.gz"]
#display_nifti_files_2d(nifti_files)


# Combine all files into one 2D slice and color each file differently
import numpy as np

def display_nifti_files_2d_combined(nifti_files):
    data_combined = np.zeros((512, 512, 117), dtype=np.uint8)
    for i, file_path in enumerate(nifti_files):
        img = nib.load(file_path)
        data = img.get_fdata()
        slice = data.shape[2] // 2
        data_combined[:, :, i] = data[:, :, slice]
        
    plt.imshow(data_combined)
    plt.show()
    
# Example usage:
# Load all *.nii.gz files from the test folder
nifti_files = ["test/" + file for file in os.listdir("test") if file.endswith(".nii.gz")]

#nifti_files = ["test/heart.nii.gz", "test/kidney_left.nii.gz", "test/kidney_right.nii.gz"]
#display_nifti_files_2d_combined(nifti_files)

img = nib.load("combined_test/combined_scan.nii.gz")

data = img.get_fdata()

# Function to update the plot for each frame
def update(slice_index):
    plt.clf()  # Clear the current plot
    plt.imshow(data[:, :, slice_index], cmap='gray')  # Plot the current slice
    plt.title(f"Slice {slice_index}")
    plt.axis('off')  # Turn off axis
    plt.pause(0.0001)  # Pause to display the plot

# Create a figure
fig = plt.figure()

# Create an animation
ani = animation.FuncAnimation(fig, update, frames=data.shape[2], interval=50)

plt.show()

# # Process all niigz files in the test folder and create a new dataframe where each nifti file has a unique color in the new dataframe
# import pandas as pd

# def create_dataframe_from_nifti_files(nifti_files):
#     data_combined = np.zeros((512, 512, 3), dtype=np.uint8)
#     for i, file_path in enumerate(nifti_files):
#         img = nib.load(file_path)
#         data = img.get_fdata()
#         slice = data.shape[2] // 2
#         data_combined[:, :, i] = data[:, :, slice]
        
#     df = pd.DataFrame(data_combined.reshape(-1, len(nifti_files)), columns=[os.path.basename(file) for file in nifti_files])
#     return df

