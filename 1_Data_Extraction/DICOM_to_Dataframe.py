import os
import pandas as pd

from util.NativeTypeConverter import convert_to_native_type
from tqdm import tqdm
from pathlib import Path
from pydicom import dcmread, Sequence

# Function to get all DICOM files in the DICOMDIR structure
def get_dicom_files(dicomdir):
    dicom_files = []
    # Iterate over each patient in the DICOMDIR
    for patient in tqdm(dicomdir.patient_records, desc="CT-Scans"):
        # Get DICOM files for the current patient and extend the list
        dicom_files.extend(get_patient_dicom_files(patient))
    return dicom_files


# Function to get DICOM files for a specific patient
def get_patient_dicom_files(patient):
    dicom_files = []
    # Iterate over each study for the patient
    for study in patient.children:
        # Check if the record type is "STUDY"
        if study.DirectoryRecordType != "STUDY":
            continue  # Skip if it's not a study
        # Get DICOM files for the current study and extend the list
        dicom_files.extend(get_study_dicom_files(study))
    return dicom_files


# Function to get DICOM files for a specific study
def get_study_dicom_files(study):
    dicom_files = []
    # Iterate over each series in the study
    for series in study.children:
        # Check if the record type is "SERIES"
        if series.DirectoryRecordType != "SERIES":
            continue  # Skip if it's not a series
        # Get DICOM files for the current series and extend the list
        dicom_files.extend(get_series_dicom_files(series))
    return dicom_files


# Function to get DICOM files for a specific series
def get_series_dicom_files(series):
    dicom_files = []
    # Iterate over each image in the series
    for image in series.children:
        # Check if the record type is "IMAGE"
        if image.DirectoryRecordType != "IMAGE":
            continue  # Skip if it's not an image
        # Get DICOM file for the current image and append to the list
        dicom_files.append(get_dicom_for_image(image))
    return dicom_files


# Function to get DICOM information for a specific image
def get_dicom_for_image(image):
    # Read the DICOM file using pydicom
    instance = dcmread(os.path.join(root_dir, *image["ReferencedFileID"].value))
    # Extract DICOM metadata excluding the pixel data
    slice = get_fields_for_dataset(instance)
    # Add the file meta information
    slice.update(get_fields_for_dataset(instance.file_meta, 'FileMetaInformation'))
    # Add the pixel array to the metadata
    slice['PixelArrayFlat'] = instance.pixel_array.flatten()  # Reconstruction by Rows and Columns Fields
    return slice


# Function to extract all DICOM fields from a pydicom dataset
def get_fields_for_dataset(dataset, prefix=''):
    dataset_dict = dict()
    for field in dataset.dir():
        value = getattr(dataset, field, None)

        if field == 'PixelData':  # Skip PixelData field (added later)
            continue
        if field == 'ConvolutionKernel':  # = Multivalue of strings (feather can't serialize string lists)
            # Add all Kernels as separate fields
            for i, kernel in enumerate(value):
                dataset_dict[f'ConvolutionKernel_{i}'] = str(kernel)
            continue

        if type(value) is Sequence:  # List of Datasets
            for sub_dataset in value:
                dataset_dict.update(get_fields_for_dataset(sub_dataset, field))
        else:
            dataset_dict[f'{prefix}{field}'] = convert_to_native_type(value)
    return dataset_dict


# Read DICOMDIR file
dicomdir_path = '../../Scans/2022-01/DICOMDIR'
dicomdir = dcmread(dicomdir_path)  # Read the DICOMDIR file
root_dir = Path(dicomdir_path).resolve().parent  # Get the parent directory of DICOMDIR

# Get all the DICOM files in the DICOMDIR
scans = get_dicom_files(dicomdir)

# Convert the list of dictionaries to a pandas DataFrame
dicom_df = pd.DataFrame(scans)

print("Compressing the DataFrame...")
# Save the DataFrame to a Feather file (lightweight binary format)
dicom_df.to_feather(f'{root_dir}_dicom_df.feather',  version = 2, compression='zstd')
print("DataFrame compressed successfully!")


# TODO: Find out why ImageType (string list) can be serialized but ConvolutionKernel can not

