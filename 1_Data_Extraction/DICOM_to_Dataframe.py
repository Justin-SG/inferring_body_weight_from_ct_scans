import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from pydicom import dcmread, Sequence
from tqdm import tqdm

from util.NativeTypeConverter import convert_to_native_type
from util.Counter import Counter


def get_dicomdir_paths(path):
    """Retrieve all DICOMDIR paths from the specified directory."""
    return list(Path(path).rglob('DICOMDIR'))


def get_dicom_dataframe(path, read_images=True):
    """Create a DataFrame containing information from DICOMDIR files in the specified directory."""
    dicomdir_paths = get_dicomdir_paths(path)
    scans = []

    for dicomdir_path in dicomdir_paths:
        dicomdir = dcmread(dicomdir_path)
        scans.extend(extract_scans_from_dicomdir(dicomdir, read_images))

    return pd.DataFrame(scans)


def read_image(instance_path):
    """Read a DICOM image from the given path."""
    return dcmread(instance_path)


def extract_patient_data(patient, root_dir, dicomdir_filename, read_images=True):
    """Extract scan data from a patient record."""
    scans = []
    scan_base = {
        'Dicomdir': dicomdir_filename,
        'PatientId': patient.PatientID,
    }

    for study in patient.children:
        if study.DirectoryRecordType == "STUDY":
            scan_base['StudyId'] = study.StudyInstanceUID

            for series in study.children:
                if series.DirectoryRecordType == "SERIES":
                    scan_base['SeriesId'] = series.SeriesInstanceUID
                    scans.extend(extract_series_data(series, scan_base, root_dir, read_images))

    return scans


def extract_series_data(series, scan_base, root_dir, read_images=True):
    """Extract scan data from a series record."""
    scan = scan_base.copy()

    instance_paths = [
        os.path.join(root_dir, *image["ReferencedFileID"].value)
        for image in series.children if image.DirectoryRecordType == "IMAGE"
    ]

    if read_images:
        counter = Counter.count()
        with ThreadPoolExecutor() as executor:
            instances = list(executor.map(read_image, instance_paths))

        # Threading may cause out-of-order slices therefore
        # sort the instances by SliceLocation to ensure the slices are in the correct order again
        instances.sort(key=lambda instance: instance.SliceLocation)

        pixel_array_3d = np.stack([instance.pixel_array for instance in instances])
        scan.update(get_fields_for_dataset(instances[0]))
        scan['PixelArrayFile'] = f'Scan_{counter}.npy'
        store_pixel_array_to_file(f'./Data/PixelArray/{scan["PixelArrayFile"]}', pixel_array_3d)

    else:
        scan.update(get_fields_for_dataset(read_image(instance_paths[0])))

    scan['SliceCount'] = len(instance_paths)

    return [scan]


def store_pixel_array_to_file(file_path, pixel_array):
    """Store the pixel array of a scan to a file."""
    # create the directory if it does not exist
    create_path_if_not_exist(file_path)
    file_path = Path(file_path)
    np.save(file_path,
            pixel_array)

def extract_scans_from_dicomdir(dicomdir, read_images=True):
    """Extract CT scan information from a DICOMDIR object."""
    root_dir = Path(dicomdir.filename).resolve().parent
    scans = []

    for patient in tqdm(dicomdir.patient_records, desc=f'Getting CT-Scans from {dicomdir.filename}'):
        scans.extend(extract_patient_data(patient, root_dir, dicomdir.filename, read_images))

    return scans


def get_fields_for_dataset(dataset, prefix=''):
    """Extract all fields from a pydicom dataset into a dictionary."""
    dataset_dict = {}

    for field in dataset.dir():
        if field in ['PixelData', 'PatientID']:
            continue

        value = getattr(dataset, field, None)
        if field == 'ConvolutionKernel' and value is not None:
            for i, kernel in enumerate(value):
                dataset_dict[f'{prefix}ConvolutionKernel_{i}'] = str(kernel)
            continue

        if isinstance(value, Sequence):
            for sub_dataset in value:
                dataset_dict.update(get_fields_for_dataset(sub_dataset, f'{prefix}{field}.'))
        else:
            dataset_dict[f'{prefix}{field}'] = convert_to_native_type(value)

    return dataset_dict


def create_path_if_not_exist(directory):
    """Creates directories if they do not exist."""
    # get only the directory part of the path
    directory = os.path.dirname(directory)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

# Main script
if __name__ == "__main__":
    # Get root/destination directories and read files flag from program arguments
    if len(sys.argv) == 4:
        root_dir = sys.argv[1]
        destination = sys.argv[2]
        read_images = sys.argv[3] == 'True'
    else:
        print("Usage: python DICOM_to_Dataframe.py <root_dir> <destination> <read_images ('True' or 'False')>")
        sys.exit(1)

    df = get_dicom_dataframe(root_dir, read_images=bool(read_images))

    # Create the output directory if it does not exist
    create_path_if_not_exist(destination)

    print("Compressing the DataFrame...")
    df.to_feather(f'{destination}/dicom_df.feather', version=2, compression='zstd')
    print("DataFrame compressed successfully!")
