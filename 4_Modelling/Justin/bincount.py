from pathlib import Path
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import pickle

PATH_TO_PARENT_DIR = Path(__file__).resolve().parent.parent.parent
PATH_TO_DATA_DIR = PATH_TO_PARENT_DIR / "Data"
PATH_TO_CLEANED_DICOM_DF = PATH_TO_DATA_DIR / "cleaned_dicom_df.feather"
PATH_TO_BINCOUNT_HU_DF = PATH_TO_DATA_DIR / "bincount_HU_df.feather"
PATH_TO_BINCOUNT_STEP_75_DF = PATH_TO_DATA_DIR / "bincount_STEP_75_df.feather"
PATH_TO_BINCOUNT_STEP_150_DF = PATH_TO_DATA_DIR / "bincount_STEP_150_df.feather"
PATH_TO_TRAIN_TEST_SPLIT = PATH_TO_DATA_DIR / "train_test_split.feather"


PATH_TO_MODEL_DIR = PATH_TO_PARENT_DIR / "Model"
PATH_TO_MODEL_BINCOUNT_75 = PATH_TO_MODEL_DIR / "model_bincount_75.pkl"
PATH_TO_MODEL_BINCOUNT_150 = PATH_TO_MODEL_DIR / "model_bincount_150.pkl"
PATH_TO_MODEL_HU = PATH_TO_MODEL_DIR / "model_bincount_HU.pkl"


cleaned_dicom_df = pd.read_feather(PATH_TO_CLEANED_DICOM_DF)
train_test_split_df = pd.read_feather(PATH_TO_TRAIN_TEST_SPLIT)
train_test_split_df = train_test_split_df[train_test_split_df["set_type"] == "Train"]
cleaned_dicom_df = pd.merge(cleaned_dicom_df, train_test_split_df, on="SeriesInstanceUID")


bincount_HU_df = pd.read_feather(PATH_TO_BINCOUNT_HU_DF)
bincount_STEP_75_df = pd.read_feather(PATH_TO_BINCOUNT_STEP_75_DF)
bincount_STEP_150_df = pd.read_feather(PATH_TO_BINCOUNT_STEP_150_DF)


merged_HU_df = pd.merge(cleaned_dicom_df, bincount_HU_df, on="SeriesInstanceUID", how="left")
merged_75_df = pd.merge(cleaned_dicom_df, bincount_STEP_75_df, on="SeriesInstanceUID", how="left")
merged_150_df = pd.merge(cleaned_dicom_df, bincount_STEP_150_df, on="SeriesInstanceUID", how="left")


target = merged_HU_df['PatientWeight']


merged_HU_df.loc[:, 'VoxelVolume'] = (merged_HU_df['PixelSpacing'] ** 2) * merged_HU_df['SliceThickness']
merged_75_df.loc[:, 'VoxelVolume'] = (merged_75_df['PixelSpacing'] ** 2) * merged_75_df['SliceThickness']
merged_150_df.loc[:, 'VoxelVolume'] = (merged_150_df['PixelSpacing'] ** 2) * merged_150_df['SliceThickness']


voxel_columns_HU = ["Air", "Fat", "Soft tissue on contrast CT", "Bone Cancellous", "Bone Cortical", "Lung Parenchyma", "Kidney", "Liver", "Lymph nodes", "Muscle", "Thymus (Children)", "Thymus (Adolescents)", "White matter", "Grey matter",]
voxel_columns_75 = [str(i) for i in range(0, 1500, 75)]
voxel_columns_150 = [str(i) for i in range(0, 1500, 150)]


def apply_voxel_volume(row, voxel_columns):
    return row[voxel_columns] * row['VoxelVolume']


transformed_HU_df = merged_HU_df.copy()
transformed_75_df = merged_75_df.copy()
transformed_150_df = merged_150_df.copy()


transformed_HU_df[voxel_columns_HU] = merged_HU_df.apply(lambda row: apply_voxel_volume(row, voxel_columns_HU), axis=1)
transformed_75_df[voxel_columns_75] = merged_75_df.apply(lambda row: apply_voxel_volume(row, voxel_columns_75), axis=1)
transformed_150_df[voxel_columns_150] = merged_150_df.apply(lambda row: apply_voxel_volume(row, voxel_columns_150), axis=1)


transformed_HU_df['PatientSex_encoded'] = transformed_HU_df['PatientSex'].map({'F': 0, 'M': 1})
transformed_75_df['PatientSex_encoded'] = transformed_75_df['PatientSex'].map({'F': 0, 'M': 1})
transformed_150_df['PatientSex_encoded'] = transformed_150_df['PatientSex'].map({'F': 0, 'M': 1})


base_columns_to_drop = ['PatientWeight', 'PatientId','Rows', 'Columns', 'RescaleSlope', 'RescaleIntercept', 'SeriesInstanceUID', 'SliceDirectory', 'BodyPart', 'PixelSpacing', 'SliceThickness', 'PatientSex', 'set_type'] # These columns wont be used in training

cleaned_HU_df = transformed_HU_df.drop(columns=base_columns_to_drop)
cleaned_75_df = transformed_75_df.drop(columns=base_columns_to_drop)
cleaned_150_df = transformed_150_df.drop(columns=base_columns_to_drop)



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
    return bayes_search.best_params_

best_params_HU_bayes = find_best_model_bayes(cleaned_HU_df, target)
best_model_HU_bayes = GradientBoostingRegressor(**best_params_HU_bayes)
best_model_HU_bayes.fit(cleaned_HU_df, target)

with open(PATH_TO_MODEL_HU, 'wb') as f:
    pickle.dump(best_model_HU_bayes, f)

with open(PATH_TO_MODEL_DIR / "best_params_HU.pkl", 'wb') as f:
    pickle.dump(best_params_HU_bayes, f)

print('Finding best model for 75')
best_params_75_bayes = find_best_model_bayes(cleaned_75_df, target)
best_model_75_bayes = GradientBoostingRegressor(**best_params_75_bayes)
best_model_75_bayes.fit(cleaned_75_df, target)

with open(PATH_TO_MODEL_BINCOUNT_75, 'wb') as f:
    pickle.dump(best_model_75_bayes, f)

with open(PATH_TO_MODEL_DIR / "best_params_75.pkl", 'wb') as f:
    pickle.dump(best_params_75_bayes, f)

print('Finding best model for 150')
best_params_150_bayes = find_best_model_bayes(cleaned_150_df, target)
best_model_150_bayes = GradientBoostingRegressor(**best_params_150_bayes)
best_model_150_bayes.fit(cleaned_150_df, target)

with open(PATH_TO_MODEL_BINCOUNT_150, 'wb') as f:
    pickle.dump(best_model_150_bayes, f)
    
with open(PATH_TO_MODEL_DIR / "best_params_150.pkl", 'wb') as f:
    pickle.dump(best_params_150_bayes, f)