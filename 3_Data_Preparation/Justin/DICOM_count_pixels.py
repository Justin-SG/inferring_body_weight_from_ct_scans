# Load DICOM data from cleaned_dicom_df.feather
import pandas as pd
from pathlib import Path
import sys
from skimage import io, filters, color, morphology, measure
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Configure the logger
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def calculate_mask(image_array_3d, slice_index):
    # Step 1: Thresholding and initial mask creation
    current_slice_data = image_array_3d[slice_index, :, :]
    tgray = current_slice_data > filters.threshold_otsu(current_slice_data)

    # Step 2: Remove small objects and holes
    keep_mask = morphology.remove_small_objects(tgray, min_size=463)
    keep_mask = morphology.remove_small_holes(keep_mask, area_threshold=1000)

    # Step 3: Identify and fill larger holes (lungs and patient table)
    # Label connected regions
    inverted_mask = np.logical_not(keep_mask)
    labeled_mask = measure.label(inverted_mask)

    
    # Find properties of the labeled regions
    regions = measure.regionprops(labeled_mask)

    # Create a mask for filling
    fill_mask = np.zeros_like(keep_mask)

    for region in regions:
        if region.area < 3000:
            # Fill the region in the mask
            coords = region.coords
            fill_mask[coords[:, 0], coords[:, 1]] = 1

    # Combine the fill mask with the original mask
    final_mask = np.logical_or(keep_mask, fill_mask)
            
    return final_mask

def main():
    # Load the DICOM metadata frames and concatenate them
    project_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = project_dir / 'Data'
    dicom_df_path = data_dir / 'cleaned_dicom_df.feather'
    logger.info(f"Loading DICOM metadata from '{dicom_df_path}'")
    dicom_df = pd.read_feather(data_dir / 'cleaned_dicom_df.feather')

    # print first 10 rows
    print(dicom_df.head(10))
    
    pixel_array_file_path = data_dir / 'PixelArray' / dicom_df.iloc[0]["PixelArrayFile"]
    
    # Read pixel array file using skimage
    image_array_1d = np.load(pixel_array_file_path)
    print(image_array_1d.shape)
    
    height = dicom_df.Rows.iloc[0]
    width = dicom_df.Columns.iloc[0]
    slice_count = dicom_df.SliceCount.iloc[0]

    # Create a 3D array with the pixel array as the first dimension
    image_array_3d = np.reshape(image_array_1d, (slice_count, width, height))
    
    non_zero_pixels = []
    
    for slice_index in range(slice_count):
        # Calculate the mask for the current slice
        mask = calculate_mask(image_array_3d, slice_index)
        non_zero_pixels.append(np.count_nonzero(mask))

    # Remove all outliers from the non-zero pixel counts
    non_zero_pixels = np.array(non_zero_pixels)
    non_zero_pixels = non_zero_pixels[np.abs(non_zero_pixels - np.mean(non_zero_pixels)) < 3 * np.std(non_zero_pixels)]

    # Plot non_zero_pixels as line graph
    plt.plot(non_zero_pixels)
    plt.show()
    
if __name__ == "__main__":
    main()