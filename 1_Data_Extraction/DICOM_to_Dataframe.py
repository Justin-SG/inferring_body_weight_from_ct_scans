import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from pydicom import dcmread

raise NotImplementedError('This script is not yet complete')

# Adapted from https://pydicom.github.io/pydicom/stable/auto_examples/input_output/plot_read_dicom_directory.html#sphx-glr-auto-examples-input-output-plot-read-dicom-directory-py

# fetch the path to the test data
path = '../../Scans/2022-01/DICOMDIR'
dicomdir = dcmread(path)
root_dir = Path(dicomdir.filename).resolve().parent
print(f'Root directory: {root_dir}\n')

scans_df = list()

# Iterate through the PATIENT records
for patient in dicomdir.patient_records:
    print(
        f"PATIENT: PatientID={patient.PatientID}, "
        f"PatientName={patient.PatientName}"
    )

    # Find all the STUDY records for the patient
    studies = [
        ii for ii in patient.children if ii.DirectoryRecordType == "STUDY"
    ]
    for study in studies:
        descr = getattr(study, "StudyDescription",  "(no value available)")
        print(
            f"{'  ' * 1}STUDY: StudyID={study.StudyID}, "
            f"StudyDate={study.StudyDate}, StudyDescription={descr}"
        )

        # Find all the SERIES records in the study
        all_series = [
            ii for ii in study.children if ii.DirectoryRecordType == "SERIES"
        ]
        for series in all_series:
            # Find all the IMAGE records in the series
            images = [
                ii for ii in series.children
                if ii.DirectoryRecordType == "IMAGE"
            ]
            plural = ('', 's')[len(images) > 1]

            descr = getattr(
                series, "SeriesDescription", "(no value available)"
            )
            print(
                f"{'  ' * 2}SERIES: SeriesNumber={series.SeriesNumber}, "
                f"Modality={series.Modality}, SeriesDescription={descr} - "
                f"{len(images)} SOP Instance{plural}"
            )

            # Get the absolute file path to each instance
            #   Each IMAGE contains a relative file path to the root directory
            elems = [ii["ReferencedFileID"] for ii in images]
            # Make sure the relative file path is always a list of str
            paths = [[ee.value] if ee.VM == 1 else ee.value for ee in elems]
            paths = [Path(*p) for p in paths]

            # List the instance file paths
            for p in paths:
                # read the corresponding SOP Instance
                instance = dcmread(Path(root_dir) / p)

                # Get all attributes of the instance
                slice = dict()
                for field in instance.dir():
                    if (field == 'PixelData'):
                        continue
                    slice[field] = getattr(instance, field, None)
                slice['PixelArray'] = instance.pixel_array
                slice['ScanID'] = series.SeriesNumber
                scans_df.append(slice)


# Convert the list of dictionaries to a pandas DataFrame
scans_df = pd.DataFrame(scans_df)

# Save the DataFrame to a CSV file
scans_df.to_csv('scans_df.csv')
print(scans_df.head())
