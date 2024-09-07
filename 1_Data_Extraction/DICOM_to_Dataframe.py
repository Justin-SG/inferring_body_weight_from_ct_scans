import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import logging

import numpy as np
import pandas as pd
from pydicom import dcmread, Sequence
from tqdm import tqdm

from util.NativeTypeConverter import convert_to_native_type
from util.Counter import Counter

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def get_dicomdir_paths(path):
    """Retrieve all DICOMDIR paths from the specified directory."""
    logger.info(f"Searching for DICOMDIR paths in '{path}'")
    paths = list(Path(path).rglob('DICOMDIR'))
    logger.info(f"Found {len(paths)} DICOMDIR paths")
    return paths

def get_dicom_dataframe(path, read_images=False):
    """Create a DataFrame containing information from DICOMDIR files in the specified directory."""
    logger.info(f"Creating DataFrame from DICOMDIR files in '{path}'")
    dicomdir_paths = get_dicomdir_paths(path)
    scans = []

    for dicomdir_path in dicomdir_paths:
        logger.info(f"Extracting scans from '{dicomdir_path}'...")
        dicomdir = dcmread(dicomdir_path)
        extracted_scans = extract_scans_from_dicomdir(dicomdir, read_images)
        logger.info(f"Extracted {len(extracted_scans)} scans")
        scans.extend(extracted_scans)

    return pd.DataFrame(scans)

def read_image(instance_path):
    """Read a DICOM image from the given path."""
    return dcmread(instance_path)

def extract_patient_data(patient, root_dir, dicomdir_filename, read_images=False):
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

def extract_series_data(series, scan_base, root_dir, read_images=False):
    """Extract scan data from a series record."""
    scan = scan_base.copy()

    instance_paths = [
        os.path.join(root_dir, *image["ReferencedFileID"].value)
        for image in series.children if image.DirectoryRecordType == "IMAGE"
    ]

    project_dir = Path(__file__).resolve().parent.parent
    pixel_array_dir = project_dir / 'Data' / 'PixelArray'
    create_directory_if_not_exist(pixel_array_dir)

    # Generate the expected filename for the .npy file
    counter = Counter.count()
    npy_filename = f'Scan_{counter}.npy'
    npy_filepath = pixel_array_dir / npy_filename

    if read_images:
        if npy_filepath.exists():
            logger.info(f"Skipping pixel array reading file '{npy_filename}' already exists.")
            scan['PixelArrayFile'] = npy_filename
            scan.update(get_fields_for_dataset(read_image(instance_paths[0])))
        else:
            with ThreadPoolExecutor() as executor:
                instances = list(executor.map(read_image, instance_paths))

            # Sort the instances by SliceLocation to ensure correct order
            instances.sort(key=lambda instance: instance.SliceLocation)

            pixel_array_3d = np.stack([instance.pixel_array for instance in instances])
            scan.update(get_fields_for_dataset(instances[0]))

            # Save the pixel array to the PixelArray subfolder within the Data folder
            scan['PixelArrayFile'] = npy_filename
            store_pixel_array_to_file(npy_filepath, pixel_array_3d)
    else:
        scan.update(get_fields_for_dataset(read_image(instance_paths[0])))

    scan['SliceCount'] = len(instance_paths)

    return [scan]

def store_pixel_array_to_file(file_path, pixel_array):
    """Store the pixel array of a scan to a file."""
    file_path = Path(file_path)
    np.save(file_path, pixel_array)

def extract_scans_from_dicomdir(dicomdir, read_images=False):
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
        if field in ['PixelData', 'PatientID', 'ContrastFlowDuration', 'ContrastFlowRate']: # Contrast Flow Dur/Rate lead to an error
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

def create_directory_if_not_exist(path):
    """Creates directories if they do not exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Extract and process DICOM data into a DataFrame.")
    parser.add_argument('root_dir', type=str, help="The root directory containing DICOM files.")
    parser.add_argument('-r', '--read_images', action='store_true', help="Flag to read and process images.")

    args = parser.parse_args()

    # Generate DataFrame from DICOM data
    logger.info("Starting DICOM data extraction process")
    df = get_dicom_dataframe(args.root_dir, read_images=args.read_images)

    # Save the DataFrame to the Data folder within the project directory
    project_dir = Path(__file__).resolve().parent.parent
    data_dir = project_dir / 'Data'
    create_directory_if_not_exist(data_dir)
    df.to_feather(data_dir / 'dicom_df.feather', version=2, compression='zstd')
    logger.info(f"DataFrame saved to '{data_dir / 'dicom_df.feather'}' successfully!")

if __name__ == "__main__":
    main()
