"""
This Python script performs machine learning tasks using XGBoost for disease classification. 
The code includes:

- Importing necessary libraries such as torch, pickle, numpy, pandas, and xgboost.
- Defining parameters for the XGBoost model and hyperparameter search.
- Implementing functions for training the model, including data preprocessing, 
  feature selection, model tuning, and evaluation.
- Checking for the availability of CUDA (GPU acceleration) and handling the script accordingly.
- Utilizing repeated k-fold cross-validation to assess the model's performance and generating a classification report.

Please ensure the necessary data paths and file paths are provided before running the script.
"""
import torch
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import RepeatedKFold, RandomizedSearchCV, cross_validate

from feature_selection import significant_features
from utilities import check_cuda

seed = 42

vowels = [
    'a',
    'e',
    'i',
    'o',
    'u',
]

model_params = {
    "device": "cuda",
    "booster": "gbtree",
    "n_jobs": -1,
    "use_label_encoder": False,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "seed": seed,
}

param_grid = {
    "n_estimators": [100, 500, 1000],
    "learning_rate": [0.001, 0.01, 0.1, 0.2, 0.3],
    "gamma": [0, 0.10, 0.15, 0.25, 0.5],
    "max_depth": [4, 6, 8, 10, 12, 15],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bylevel": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
    'max_delta_step': [0, 1, 5, 10],
    "scale_pos_weight": [1, 5, 10, 20, 50, 100]
}

search_settings = {
    "param_distributions": param_grid,
    "scoring": 'balanced_accuracy',
    "n_jobs": -1,
    "n_iter": 100,
    "verbose": 20
}

def train_model(results_file_path, data_path, selected_features_path, save_model_path):
    for vowel in vowels:
        with open(results_file_path, "a+") as result_file:
            result_file.write(f"Vowel: {vowel}\n")

            data = pd.read_csv(data_path)
            data.drop(columns=['id'], inplace=True)

            # Prepare Data
            selected_feature_names = significant_features(data)

            X = data.drop(columns=['class'], axis=1)
            X = X[selected_feature_names]

            y = data['class']

            # Convert DataFrame to PyTorch tensors and move to GPU
            X_tensor = torch.tensor(X.values, dtype=torch.float32).cuda()
            y_tensor = torch.tensor(y.values, dtype=torch.float32).cuda()

            scaler = StandardScaler()
            X_scaled = torch.tensor(scaler.fit_transform(X_tensor.cpu().numpy()), dtype=torch.float32).cuda()

            # Save feature names to CSV file
            feature_names_df = pd.DataFrame({'Feature Names': selected_feature_names})
            feature_names_df.to_csv(selected_features_path, index=False)

            X_new_tensor = torch.tensor(X_scaled, dtype=torch.float32)

            # Model Initialization and Hyperparameter Tuning
            model = xgb.XGBClassifier(**model_params)
            kfolds = KFold(n_splits=5, shuffle=True, random_state=seed)
            random_search = RandomizedSearchCV(model,
                                               cv=kfolds.split(X_new_tensor.cpu().numpy(), y_tensor.cpu().numpy()),
                                               random_state=seed, **search_settings)
            random_search.fit(X_new_tensor.cpu().numpy(), y_tensor.cpu().numpy())

            # Best Model Parameters
            best_tuning_score = random_search.best_score_
            best_tuning_hyperparams = random_search.best_estimator_
            result_file.write(f"Best Model Parameters: {best_tuning_hyperparams}\n")

            # Save Trained Model to a Pickle File
            with open(save_model_path, 'wb') as model_file:
                pickle.dump(best_tuning_hyperparams, model_file)

            # Cross-Validation
            kfolds = RepeatedKFold(n_splits=5, n_repeats=20, random_state=seed)
            scoring = {"acc": make_scorer(accuracy_score, greater_is_better=True)}
            cv_results = cross_validate(model, X_new_tensor.cpu().numpy(),
                                        y_tensor.cpu().numpy(),
                                        scoring=scoring,
                                        cv=kfolds.split(X_new_tensor.cpu().numpy(), y_tensor.cpu().numpy()))

            # Performance Metric
            cls_report = {
                "acc_avg": round(float(np.mean(cv_results["test_acc"])), 4),
                "acc_std": round(float(np.std(cv_results["test_acc"])), 4),
            }
            acc = f"{cls_report['acc_avg']:.2f} +- {cls_report['acc_std']:.2f}"
            result_file.write(f"Classification Report: {acc}\n")

if __name__ == "__main__":
    results_file_path = ""
    data_path = ""
    selected_features_path = ""
    save_model_path = ""

    if not check_cuda():
        print("CUDA is not available. Exiting.")
        exit()
    else:
        train_model(results_file_path, data_path, selected_features_path, save_model_path)
