from pathlib import Path
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import pickle
import argparse

PATH_TO_PARENT_DIR = Path(__file__).resolve().parent.parent.parent
PATH_TO_DATA_DIR = PATH_TO_PARENT_DIR / "Data"

PATH_TO_CLEANED_DICOM_DF = PATH_TO_DATA_DIR / "cleaned_dicom_df.feather"

PATH_TO_SEGMENTATION_DF = PATH_TO_DATA_DIR / "segmentation_df.feather"
PATH_TO_BINCOUNT_HU_DF = PATH_TO_DATA_DIR / "bincount_HU_df.feather"
PATH_TO_BINCOUNT_STEP_75_DF = PATH_TO_DATA_DIR / "bincount_STEP_75_df.feather"
PATH_TO_BINCOUNT_STEP_150_DF = PATH_TO_DATA_DIR / "bincount_STEP_150_df.feather"
PATH_TO_TRAIN_TEST_SPLIT = PATH_TO_DATA_DIR / "train_test_split.feather"

PATH_TO_OUTPUT_DIR = PATH_TO_PARENT_DIR / "Model" / "Justin" / "Final"

def apply_voxel_volume(row, voxel_columns):
    return row[voxel_columns] * row['VoxelVolume']

# def transform_df(df, voxel_columns):
#     transformed_voxel_values = df.apply(lambda row: apply_voxel_volume(row, voxel_columns), axis=1)
#     transformed_voxel_values = pd.DataFrame(transformed_voxel_values.values.tolist(), columns=voxel_columns, index=df.index)
#     for col in voxel_columns:
#         df[col] = transformed_voxel_values[col]
    
#     df['PatientSex_encoded'] = df['PatientSex'].map({'F': 0, 'M': 1})
#     return df

def calculate_voxel_volumes(df, voxel_columns, replace_original_columns):
    transformed_voxel_values = df.apply(lambda row: apply_voxel_volume(row, voxel_columns), axis=1)
    transformed_voxel_values = pd.DataFrame(transformed_voxel_values.values.tolist(), columns=voxel_columns, index=df.index)

    if replace_original_columns:
        for col in voxel_columns:
            df[col] = transformed_voxel_values[col]
    else:
        new_columns = pd.DataFrame({f"volume_{col}": transformed_voxel_values[col] for col in voxel_columns})
        df = pd.concat([df, new_columns], axis=1)

    return df

def loadAndPrepareDataFrame(cleaned_dicom_df, actual_data_df, voxel_columns, columns_to_drop, calculate_voxel_volume, replace_original_columns):
    merged_with_meta_df = pd.merge(cleaned_dicom_df, actual_data_df, on="SeriesInstanceUID", how="left")
    merged_with_meta_df.loc[:, 'VoxelVolume'] = (merged_with_meta_df['PixelSpacing'] ** 2) * merged_with_meta_df['SliceThickness']
    if calculate_voxel_volume:
        merged_with_meta_df = calculate_voxel_volumes(merged_with_meta_df, voxel_columns, replace_original_columns)
        
    encoded_PatientSex_column = merged_with_meta_df['PatientSex'].map({'F': 0, 'M': 1})
    merged_with_meta_df = pd.concat([merged_with_meta_df, encoded_PatientSex_column.rename('PatientSex_encoded')], axis=1)
    final_df = merged_with_meta_df.drop(columns=columns_to_drop, errors='ignore')
    return final_df.copy()

def find_best_model_bayes(X, y):
    param_space = {
        'loss': Categorical(['squared_error', 'absolute_error']),
        'learning_rate': Real(0.01, 0.2, prior='log-uniform'),
        'n_estimators': Integer(50, 1000),
        'subsample': Real(0.5, 1.0, prior='uniform'),
        'min_samples_split': Integer(2, 10),
        'min_samples_leaf': Integer(1, 4),
        'min_weight_fraction_leaf': Real(0.0, 0.5, prior='uniform'),
        'max_depth': Integer(1, 9),
        'max_leaf_nodes': Integer(2, 10),
    }

    gb = GradientBoostingRegressor()
    bayes_search = BayesSearchCV(estimator=gb, search_spaces=param_space, n_iter=256, cv=8, n_jobs=-1, verbose=2, scoring='neg_mean_absolute_error')
    bayes_search.fit(X, y)
    
    print(bayes_search.best_params_)
    print(bayes_search.best_score_)
    return bayes_search.best_params_

def trainAndSaveResults(data_df, unique_name):
    df_for_training = data_df[data_df["set_type"] == "Train"]      
    target = df_for_training["PatientWeight"]
    df_for_training = df_for_training.drop(columns=["set_type", "PatientWeight", "SeriesInstanceUID"], errors='ignore')  
    
    best_params_bayes = find_best_model_bayes(df_for_training, target)
    best_model_bayes = GradientBoostingRegressor(**best_params_bayes)
    best_model_bayes.fit(df_for_training, target)

    with open(PATH_TO_OUTPUT_DIR / f'model_{unique_name}.pkl', 'wb') as f:
        pickle.dump(best_model_bayes, f)

    with open(PATH_TO_OUTPUT_DIR / f'best_params_{unique_name}.pkl', 'wb') as f:
        pickle.dump(best_params_bayes, f)
    
    target = data_df["PatientWeight"]
    df_for_predict = data_df.drop(columns=["set_type", "PatientWeight", "SeriesInstanceUID"], errors='ignore')
    predictions = best_model_bayes.predict(df_for_predict)
    
    results_df = pd.DataFrame(data_df["SeriesInstanceUID"])
    results_df["PredictedPatientWeight"] = predictions
    results_df.to_feather(PATH_TO_OUTPUT_DIR / f'predictions_{unique_name}.feather', version=2, compression="zstd")


parser = argparse.ArgumentParser(description="Train different GradientBoostingRegressors.")
parser.add_argument('-c', '--calculate_voxel_volume', action='store_true', help="Whether to calculate voxel volume.")
parser.add_argument('-r', '--replace_original_columns', action='store_true', help="Whether to replace the original count with the volume.")
parser.add_argument('-d', '--columns_to_drop_index', type=int, choices=[0, 1, 2], default=0, help="Index of columns to drop (0, 1, or 2).")
args = parser.parse_args()

if args.replace_original_columns and not args.calculate_voxel_volume:
    print("Error: --replace_original_columns requires --calculate_voxel_volume to be specified.")
    exit(-1)

calculate_voxel_volume = args.calculate_voxel_volume
columns_to_drop_index = args.columns_to_drop_index
replace_original_columns = args.replace_original_columns

cleaned_dicom_all_df = pd.read_feather(PATH_TO_CLEANED_DICOM_DF)
train_test_split_info_df = pd.read_feather(PATH_TO_TRAIN_TEST_SPLIT)
cleaned_dicom_df = pd.merge(cleaned_dicom_all_df, train_test_split_info_df, on="SeriesInstanceUID")

possible_columns_to_drop = [ ['PatientId','Rows', 'Columns', 'RescaleSlope', 'RescaleIntercept', 'SliceDirectory', 'PixelArrayFile', 'BodyPart', 'PixelSpacing', 'SliceThickness', 'VoxelVolume', 'PatientSex'], # These columns wont be used in training
                             ['PatientId','Rows', 'Columns', 'RescaleSlope', 'RescaleIntercept', 'SliceDirectory', 'PixelArrayFile', 'BodyPart', 'PixelSpacing', 'SliceThickness', 'PatientSex'], # These columns wont be used in training
                             ['PatientId','Rows', 'Columns', 'RescaleSlope', 'RescaleIntercept', 'SliceDirectory', 'PixelArrayFile', 'BodyPart', 'PatientSex'] ] # These columns wont be used in training

column_to_drop = possible_columns_to_drop[columns_to_drop_index]

segmentation_df = pd.read_feather(PATH_TO_SEGMENTATION_DF)
bincount_HU_df = pd.read_feather(PATH_TO_BINCOUNT_HU_DF)
bincount_STEP_75_df = pd.read_feather(PATH_TO_BINCOUNT_STEP_75_DF)
bincount_STEP_150_df = pd.read_feather(PATH_TO_BINCOUNT_STEP_150_DF)

merged_segmentation_Air_df = pd.merge(segmentation_df, bincount_HU_df[['Air', 'SeriesInstanceUID']], on="SeriesInstanceUID", how="left")
merged_segmentation_HU_df = pd.merge(segmentation_df, bincount_HU_df, on="SeriesInstanceUID", how="left")
merged_segmentation_75_df = pd.merge(segmentation_df, bincount_STEP_75_df, on="SeriesInstanceUID", how="left")
merged_segmentation_150_df = pd.merge(segmentation_df, bincount_STEP_150_df, on="SeriesInstanceUID", how="left")


voxel_columns_bin_75 = bincount_STEP_75_df.columns[0:-1].tolist()
voxel_columns_bin_150 = bincount_STEP_150_df.columns[0:-1].tolist()
voxel_columns_bin_HU = bincount_HU_df.columns[0:-1].tolist()
voxel_columns_segmentation = segmentation_df.columns[0:-1].tolist()
voxel_columns_seg_Air =  voxel_columns_segmentation + ['Air']
voxel_columns_seg_HU =  voxel_columns_segmentation + bincount_HU_df.columns[0:-1].tolist()
voxel_columns_seg_75 = voxel_columns_segmentation + bincount_STEP_75_df.columns[0:-1].tolist()
voxel_columns_seg_150 = voxel_columns_segmentation + bincount_STEP_150_df.columns[0:-1].tolist()

final_segmentation_df     = loadAndPrepareDataFrame(cleaned_dicom_df, segmentation_df           , voxel_columns_segmentation, column_to_drop, calculate_voxel_volume, replace_original_columns)
final_segmentation_Air_df = loadAndPrepareDataFrame(cleaned_dicom_df, merged_segmentation_Air_df, voxel_columns_seg_Air     , column_to_drop, calculate_voxel_volume, replace_original_columns)
final_segmentation_HU_df  = loadAndPrepareDataFrame(cleaned_dicom_df, merged_segmentation_HU_df , voxel_columns_seg_HU      , column_to_drop, calculate_voxel_volume, replace_original_columns)
final_segmentation_75_df  = loadAndPrepareDataFrame(cleaned_dicom_df, merged_segmentation_75_df , voxel_columns_seg_75      , column_to_drop, calculate_voxel_volume, replace_original_columns)
final_segmentation_150_df = loadAndPrepareDataFrame(cleaned_dicom_df, merged_segmentation_150_df, voxel_columns_seg_150     , column_to_drop, calculate_voxel_volume, replace_original_columns)
final_bin_HU_df           = loadAndPrepareDataFrame(cleaned_dicom_df, bincount_HU_df            , voxel_columns_bin_HU      , column_to_drop, calculate_voxel_volume, replace_original_columns)
final_bin_75_df           = loadAndPrepareDataFrame(cleaned_dicom_df, bincount_STEP_75_df       , voxel_columns_bin_75      , column_to_drop, calculate_voxel_volume, replace_original_columns)
final_bin_150_df          = loadAndPrepareDataFrame(cleaned_dicom_df, bincount_STEP_150_df      , voxel_columns_bin_150     , column_to_drop, calculate_voxel_volume, replace_original_columns)

calculate_voxel_volume_name = "volumeAndCount" if calculate_voxel_volume else "onlyCount"
columns_to_drop_name = "colDropAll" if columns_to_drop_index == 0 else "colDropVolume" if columns_to_drop_index == 1 else "colKeepAll"
replace_original_columns_name = "replaceOrig" if replace_original_columns else "keepOrig"


prefix = f"{calculate_voxel_volume_name}_{columns_to_drop_name}_{replace_original_columns_name}"

trainAndSaveResults(final_segmentation_df    , f'{prefix}_seg')
trainAndSaveResults(final_segmentation_Air_df, f'{prefix}_seg_air')
trainAndSaveResults(final_segmentation_HU_df , f'{prefix}_seg_HU')
trainAndSaveResults(final_segmentation_75_df , f'{prefix}_seg_75')
trainAndSaveResults(final_segmentation_150_df, f'{prefix}_seg_150')
trainAndSaveResults(final_bin_HU_df          , f'{prefix}_HU')
trainAndSaveResults(final_bin_75_df          , f'{prefix}_bincount_75')
trainAndSaveResults(final_bin_150_df         , f'{prefix}_bincount_150')