""" Runs a Bayesian hyperparameter optimisation search using Optuna 
to train and predict on a classicifation problems b."""

import pandas as pd
import numpy as np
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score
import optuna
import argparse
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
import mlflow.sklearn
import logging
from urllib.parse import urlparse
from optuna.integration.mlflow import MLflowCallback
from mlflow.data.pandas_dataset import PandasDataset
from random_forest import RandomForest
from support_vector_classifier import SVClassifier
import json

# Establish mlflow logging config
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Try optimising baseline model hyperparameters with Optuna - bayesian autoML methods
def objective(trial):

    with mlflow.start_run(run_name=str(trial.number)):

        # Features
        train_x_raw = pd.read_csv(data_params['X_train_path'])
        test_x_raw  = pd.read_csv(data_params['X_test_path'])

        # Labels
        train_y_raw  = pd.read_csv(data_params['y_train_path'])
        test_y_raw  = pd.read_csv(data_params['y_test_path'])
        
        # Sample these classifiers
        classifier_name = trial.suggest_categorical("classifier", ["SVC", "RandomForest"])

        # Iterate over model specific hyper-parameters 
        if classifier_name == "SVC":
            print("Initializing model")
            model = SVClassifier()
            X_train, y_train = model.reshape_data(train_x_raw, train_y_raw)
            X_test, y_test = model.reshape_data(test_x_raw , test_y_raw)

            # Define optimizable hyperparameters ranges: C
            lower_sample_c = model_params['svm']['c_lower_sample']
            upper_sample_c = model_params['svm']['c_upper_sample']

            # Define optimizable hyperparameters ranges: gamma
            lower_sample_gamma = model_params['svm']['gamma_lower_sample']
            upper_sample_gamma = model_params['svm']['gamma_upper_sample']

            # Bayesian search over sampled space
            svc_c = trial.suggest_float("svc_c", lower_sample_c, upper_sample_c, log=True)
            svc_gamma = trial.suggest_float("svc_gamma", lower_sample_gamma, upper_sample_gamma, log=True)
            classifier_obj = SVClassifier(C=svc_c, gamma=svc_gamma)

        elif classifier_name == "RandomForest":
            print("Initializing model")
            model = RandomForest()
            X_train, y_train = model.reshape_data(train_x_raw, train_y_raw)
            X_test, y_test = model.reshape_data(test_x_raw , test_y_raw)

            # Define optimizable hyperparameters ranges: max depth
            lower_sample_max_depth = model_params['random_forest']['max_depth_lower_sample']
            upper_sample_max_depth = model_params['random_forest']['max_depth_upper_sample']
            
            # Define optimizable hyperparameters ranges: n_estimators
            lower_sample_n_estimators = model_params['random_forest']['n_estimators_lower_sample']
            upper_sample_n_estimators = model_params['random_forest']['n_estimators_upper_sample']
            
            # Bayesian search over sampled space
            rf_max_depth = trial.suggest_int("rf_max_depth", lower_sample_max_depth, upper_sample_max_depth, log=True)
            rf_n_estimators = trial.suggest_int("rf_n_estimators", lower_sample_n_estimators, upper_sample_n_estimators, log=True)
            classifier_obj = RandomForest(max_depth=rf_max_depth, n_estimators=rf_n_estimators)

        # Predict on the train set
        predictions = model.fit_predict(X_train, y_train, X_test)
        test_acc = accuracy_score(y_test, predictions)

        # Log the datasets
        mlflow.log_artifacts('/Users/sum02dean/projects/wine_challenge/WINE/data')

        # Log the hyperparameters and trial metrics with mlflow
        signature = infer_signature(predictions, predictions)
        mlflow.log_params(trial.params)
        mlflow.log_metric('test_accuracy', test_acc)
        mlflow.sklearn.log_model(classifier_obj, classifier_name, signature=signature)
    return test_acc

# Run Main
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Read in the configs file
    parser.add_argument("--config_file", help="path to configs file",
                        type=str, default="WINE/configs/config_file.json")
    args = parser.parse_args()
    
    # Read in the parameters from configs file
    configs = json.load(open("/Users/sum02dean/projects/wine_challenge/WINE/configs/config_file.json"))
    mlflow_params = configs.get("mlflow_params")

    # Extract parameters
    mlflow_params = configs.get("mlflow_params")
    model_params = configs.get("model_params")
    data_params = configs.get("data_params")
    
    # Create Optuna study
    study = optuna.create_study(
        storage=mlflow_params["storage"],
        study_name=mlflow_params["study_name"], direction='maximize')
    
    # Optimize
    mlflow.set_experiment(experiment_name=mlflow_params["study_name"])
    study.optimize(objective, n_trials=mlflow_params["n_trials"])
    
    
# Run on terminal after running the prelimainary_model.py file:
# optuna-dashboard sqlite:///db.sqlite3 (be inside the src directory)
# mlflow ui (be inside the src directory)