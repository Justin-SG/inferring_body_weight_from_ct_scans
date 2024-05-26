import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from pydicom import dcmread, Sequence
from tqdm import tqdm

from util.NativeTypeConverter import convert_to_native_type


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
        with ThreadPoolExecutor() as executor:
            instances = list(executor.map(read_image, instance_paths))

        # Threading may cause out-of-order slices therefore
        # sort the instances by SliceLocation to ensure the slices are in the correct order again
        instances.sort(key=lambda instance: instance.SliceLocation)

        pixel_array_flat = np.concatenate([instance.pixel_array.flatten(order='C') for instance in instances])
        scan.update(get_fields_for_dataset(instances[0]))
        scan['PixelArrayFlat'] = pixel_array_flat

    else:
        scan.update(get_fields_for_dataset(read_image(instance_paths[0])))

    scan['SliceCount'] = len(instance_paths)

    return [scan]


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


# from https://stackoverflow.com/questions/17315737/split-a-large-pandas-dataframe
def split_dataframe(df, chunk_size=100):
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i * chunk_size:(i + 1) * chunk_size])
    return chunks


def store_dataframe_chunks(chunks, filename, destination):
    for i, chunk in tqdm(enumerate(chunks), desc='Storing DataFrame chunks'):
        chunk.to_feather(f'{destination}/{filename}_{i}.feather', version=2, compression='zstd')


# Main script
if __name__ == "__main__":
    root_dir = '../../Scans'
    destination = 'Data/Metadata_Only'
    df = get_dicom_dataframe(root_dir, read_images=False)

    # Create the output directory if it does not exist
    if not os.path.exists(destination):
        os.makedirs(destination, exist_ok=True)

    print("Compressing the DataFrame...")
    store_dataframe_chunks(split_dataframe(df), 'dicom_df', destination)
    print("DataFrame compressed successfully!")
