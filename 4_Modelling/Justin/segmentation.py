from pathlib import Path
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import pickle


PATH_TO_PARENT_DIR = Path(__file__).resolve().parent.parent.parent
PATH_TO_DATA_DIR = PATH_TO_PARENT_DIR / "Data"

PATH_TO_CLEANED_DICOM_DF = PATH_TO_DATA_DIR / "cleaned_dicom_df.feather"

PATH_TO_SEGMENTATION_DF = PATH_TO_DATA_DIR / "segmentation_df.feather"
PATH_TO_BINCOUNT_HU_DF = PATH_TO_DATA_DIR / "bincount_HU_df.feather"
PATH_TO_BINCOUNT_STEP_75_DF = PATH_TO_DATA_DIR / "bincount_STEP_75_df.feather"
PATH_TO_BINCOUNT_STEP_150_DF = PATH_TO_DATA_DIR / "bincount_STEP_150_df.feather"
PATH_TO_TRAIN_TEST_SPLIT = PATH_TO_DATA_DIR / "train_test_split.feather"

PATH_TO_MODEL_DIR = PATH_TO_PARENT_DIR / "Model"
PATH_TO_MODEL_SEG = PATH_TO_MODEL_DIR / "model_seg.pkl"
PATH_TO_MODEL_SEG_AIR = PATH_TO_MODEL_DIR / "model_seg_air.pkl"
PATH_TO_MODEL_SEG_75 = PATH_TO_MODEL_DIR / "model_seg_75.pkl"
PATH_TO_MODEL_SEG_150 = PATH_TO_MODEL_DIR / "model_seg_150.pkl"
PATH_TO_MODEL_SEG_HU = PATH_TO_MODEL_DIR / "model_seg_HU.pkl"



cleaned_dicom_df = pd.read_feather(PATH_TO_CLEANED_DICOM_DF)
train_test_split_df = pd.read_feather(PATH_TO_TRAIN_TEST_SPLIT)

train_test_split_df = train_test_split_df[train_test_split_df["set_type"] == "Train"]

cleaned_dicom_df = pd.merge(cleaned_dicom_df, train_test_split_df, on="SeriesInstanceUID")



segmentation_df = pd.read_feather(PATH_TO_SEGMENTATION_DF)
bincount_HU_df = pd.read_feather(PATH_TO_BINCOUNT_HU_DF)
bincount_STEP_75_df = pd.read_feather(PATH_TO_BINCOUNT_STEP_75_DF)
bincount_STEP_150_df = pd.read_feather(PATH_TO_BINCOUNT_STEP_150_DF)


merged_segmentation_df = pd.merge(cleaned_dicom_df, segmentation_df, on="SeriesInstanceUID", how="left")
merged_segmentation_Air_df = pd.merge(merged_segmentation_df, bincount_HU_df[['Air', 'SeriesInstanceUID']], on="SeriesInstanceUID", how="left")
merged_segmentation_HU_df = pd.merge(merged_segmentation_df, bincount_HU_df, on="SeriesInstanceUID", how="left")
merged_segmentation_75_df = pd.merge(merged_segmentation_df, bincount_STEP_75_df, on="SeriesInstanceUID", how="left")
merged_segmentation_150_df = pd.merge(merged_segmentation_df, bincount_STEP_150_df, on="SeriesInstanceUID", how="left")


target = merged_segmentation_df['PatientWeight']


voxel_columns_segmentation = segmentation_df.columns[0:-1].tolist()
voxel_columns_Air =  voxel_columns_segmentation + ['Air']
voxel_columns_HU =  voxel_columns_segmentation + bincount_HU_df.columns[0:-1].tolist()
voxel_columns_75 = voxel_columns_segmentation + bincount_STEP_75_df.columns[0:-1].tolist()
voxel_columns_150 = voxel_columns_segmentation + bincount_STEP_150_df.columns[0:-1].tolist()


merged_segmentation_df.loc[:, 'VoxelVolume'] = (merged_segmentation_df['PixelSpacing'] ** 2) * merged_segmentation_df['SliceThickness']
merged_segmentation_Air_df.loc[:, 'VoxelVolume'] = (merged_segmentation_Air_df['PixelSpacing'] ** 2) * merged_segmentation_Air_df['SliceThickness']
merged_segmentation_HU_df.loc[:, 'VoxelVolume'] = (merged_segmentation_HU_df['PixelSpacing'] ** 2) * merged_segmentation_HU_df['SliceThickness']
merged_segmentation_75_df.loc[:, 'VoxelVolume'] = (merged_segmentation_75_df['PixelSpacing'] ** 2) * merged_segmentation_75_df['SliceThickness']
merged_segmentation_150_df.loc[:, 'VoxelVolume'] = (merged_segmentation_150_df['PixelSpacing'] ** 2) * merged_segmentation_150_df['SliceThickness']


def apply_voxel_volume(row, voxel_columns):
    return row[voxel_columns] * row['VoxelVolume']

transformed_segmentation_df = merged_segmentation_df.copy()
transformed_segmentation_Air_df = merged_segmentation_Air_df.copy()
transformed_segmentation_HU_df = merged_segmentation_HU_df.copy()
transformed_segmentation_75_df = merged_segmentation_75_df.copy()
transformed_segmentation_150_df = merged_segmentation_150_df.copy()

transformed_segmentation_df[voxel_columns_segmentation] = merged_segmentation_df.apply(lambda row: apply_voxel_volume(row, voxel_columns_segmentation), axis=1)
transformed_segmentation_Air_df[voxel_columns_segmentation] = merged_segmentation_Air_df.apply(lambda row: apply_voxel_volume(row, voxel_columns_segmentation), axis=1)
transformed_segmentation_HU_df[voxel_columns_segmentation] = merged_segmentation_HU_df.apply(lambda row: apply_voxel_volume(row, voxel_columns_segmentation), axis=1)
transformed_segmentation_75_df[voxel_columns_segmentation] = merged_segmentation_75_df.apply(lambda row: apply_voxel_volume(row, voxel_columns_segmentation), axis=1)
transformed_segmentation_150_df[voxel_columns_segmentation] = merged_segmentation_150_df.apply(lambda row: apply_voxel_volume(row, voxel_columns_segmentation), axis=1)


transformed_segmentation_df['PatientSex_encoded'] =  transformed_segmentation_df['PatientSex'].map({'F': 0, 'M': 1}).copy()
transformed_segmentation_Air_df['PatientSex_encoded'] =  transformed_segmentation_Air_df['PatientSex'].map({'F': 0, 'M': 1}).copy()
transformed_segmentation_HU_df['PatientSex_encoded'] =  transformed_segmentation_HU_df['PatientSex'].map({'F': 0, 'M': 1}).copy()
transformed_segmentation_75_df['PatientSex_encoded'] =  transformed_segmentation_75_df['PatientSex'].map({'F': 0, 'M': 1}).copy()
transformed_segmentation_150_df['PatientSex_encoded'] =  transformed_segmentation_150_df['PatientSex'].map({'F': 0, 'M': 1}).copy()


base_columns_to_drop = ['PatientWeight', 'PatientId','Rows', 'Columns', 'RescaleSlope', 'RescaleIntercept', 'SeriesInstanceUID', 'SliceDirectory', 'PixelArrayFile', 'BodyPart', 'PixelSpacing', 'SliceThickness', 'PatientSex', 'set_type'] # These columns wont be used in training

cleaned_segmentation_df = transformed_segmentation_df.drop(columns=base_columns_to_drop, errors='ignore')
cleaned_segmentation_Air_df = transformed_segmentation_Air_df.drop(columns=base_columns_to_drop, errors='ignore')
cleaned_segmentation_HU_df = transformed_segmentation_HU_df.drop(columns=base_columns_to_drop, errors='ignore')
cleaned_segmentation_75_df = transformed_segmentation_75_df.drop(columns=base_columns_to_drop, errors='ignore')
cleaned_segmentation_150_df = transformed_segmentation_150_df.drop(columns=base_columns_to_drop, errors='ignore')



def find_best_model_bayes(X, y):
    param_space = {
        'loss': Categorical(['squared_error', 'absolute_error']),
        'learning_rate': Real(0.01, 0.2, prior='log-uniform'),
        'n_estimators': Integer(50, 500),
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



best_params_seg_bayes = find_best_model_bayes(cleaned_segmentation_df, target)
best_model_seg_bayes = GradientBoostingRegressor(**best_params_seg_bayes)
best_model_seg_bayes.fit(cleaned_segmentation_df, target)


with open(PATH_TO_MODEL_SEG, 'wb') as f:
    pickle.dump(best_model_seg_bayes, f)

with open(PATH_TO_MODEL_DIR / "best_params_seg.pkl", 'wb') as f:
    pickle.dump(best_params_seg_bayes, f)



best_params_seg_air_bayes = find_best_model_bayes(cleaned_segmentation_Air_df, target)
best_model_seg_air_bayes = GradientBoostingRegressor(**best_params_seg_air_bayes)
best_model_seg_air_bayes.fit(cleaned_segmentation_Air_df, target)

with open(PATH_TO_MODEL_SEG_AIR, 'wb') as f:
    pickle.dump(best_model_seg_air_bayes, f)

with open(PATH_TO_MODEL_DIR / "best_params_seg_air.pkl", 'wb') as f:
    pickle.dump(best_params_seg_air_bayes, f)



best_params_seg_HU_bayes = find_best_model_bayes(cleaned_segmentation_HU_df, target)
best_model_seg_HU_bayes = GradientBoostingRegressor(**best_params_seg_HU_bayes)
best_model_seg_HU_bayes.fit(cleaned_segmentation_HU_df, target)

with open(PATH_TO_MODEL_SEG_HU, 'wb') as f:
    pickle.dump(best_model_seg_HU_bayes, f)

with open(PATH_TO_MODEL_DIR / "best_params_seg_HU.pkl", 'wb') as f:
    pickle.dump(best_params_seg_HU_bayes, f)



best_params_seg_75_bayes = find_best_model_bayes(cleaned_segmentation_75_df, target)
best_model_seg_75_bayes = GradientBoostingRegressor(**best_params_seg_75_bayes)
best_model_seg_75_bayes.fit(cleaned_segmentation_75_df, target)

with open(PATH_TO_MODEL_SEG_75, 'wb') as f:
    pickle.dump(best_model_seg_75_bayes, f)

with open(PATH_TO_MODEL_DIR / "best_params_seg_75.pkl", 'wb') as f:
    pickle.dump(best_params_seg_75_bayes, f)



best_params_seg_150_bayes = find_best_model_bayes(cleaned_segmentation_150_df, target)
best_model_seg_150_bayes = GradientBoostingRegressor(**best_params_seg_150_bayes)
best_model_seg_150_bayes.fit(cleaned_segmentation_150_df, target)

with open(PATH_TO_MODEL_SEG_150, 'wb') as f:
    pickle.dump(best_model_seg_150_bayes, f)

with open(PATH_TO_MODEL_DIR / "best_params_seg_150.pkl", 'wb') as f:
    pickle.dump(best_params_seg_150_bayes, f)