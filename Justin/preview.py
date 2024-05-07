import os
import napari
import nibabel as nib

def display_nifti_files_3d(nifti_files):
    viewer = napari.Viewer()
    
    for file_path in nifti_files:
        img = nib.load(file_path)
        data = img.get_fdata()
        voxel_sizes = img.header['pixdim'][1:4]
        viewer.add_image(data, name=file_path, scale=voxel_sizes)

    napari.run()

# Example usage:
# Load all *.nii.gz files from the test folder
nifti_files = ["test/" + file for file in os.listdir("test") if file.endswith(".nii.gz")]

#nifti_files = ["test/heart.nii.gz", "test/kidney_left.nii.gz", "test/kidney_right.nii.gz"]
display_nifti_files_3d(nifti_files)
