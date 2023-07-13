import pandas as pd
import numpy as np
import sklearn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
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



# Establish mlflow logging config
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Try optimising baseline model hyperparameters with Optuna - bayesian autoML methods
def objective(trial):

    with mlflow.start_run(run_name=str(trial.number)):
        # Features have already been pre-processed
        train_x = pd.read_csv('/Users/sum02dean/projects/wine_challenge/WINE/data/scaled_unclipped_train_x.csv')
        test_x = pd.read_csv('/Users/sum02dean/projects/wine_challenge/WINE/data/scaled_unclipped_test_x.csv')

        # Labels
        train_y = pd.read_csv('/Users/sum02dean/projects/wine_challenge/WINE/data/train_y.csv')
        test_y = pd.read_csv('/Users/sum02dean/projects/wine_challenge/WINE/data/test_y.csv')

        # Reformat x & y
        train_labels = train_y.values.reshape(-1)
        train_features = train_x.values

        test_labels = test_y.values.reshape(-1)
        test_features = test_x.values

        # classifier_name = trial.suggest_categorical("classifier", ["SVC", "RandomForest"])
        classifier_name = 'RandomForest'

        # Iterate over model specific hyper-parameters 
        if classifier_name == "SVC":
            svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
            svc_gamma = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
            classifier_obj = sklearn.svm.SVC(C=svc_c, gamma=svc_gamma)
        else:
            rf_max_depth = trial.suggest_int("rf_max_depth", 2, 100, log=False)
            rf_n_estimators = trial.suggest_int("rf_n_estimators", 2, 500, log=False)
            classifier_obj = sklearn.ensemble.RandomForestClassifier(
                max_depth=rf_max_depth, n_estimators=rf_n_estimators
            )

        # Predict on the train set
        trained_model = classifier_obj.fit(X=train_features, y=train_labels)

        # Infer on the test set
        test_pred = trained_model.predict(test_features)
        test_acc = accuracy_score(test_labels, test_pred)

        # Log the hyperparameters and trial metrics with mlflow
        signature = infer_signature(test_pred, test_pred)
        mlflow.log_params(trial.params)
        mlflow.log_metric('test_accuracy', test_acc)
        mlflow.sklearn.log_model(classifier_obj, "random_forest", signature=signature)
    return test_acc

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--study_name", help="study name for optuna log",
                        type=str)
    args = parser.parse_args()
    print(args)

    # Setup dynamic logging
    study_name = args.study_name
    study = optuna.create_study(
        storage="sqlite:///db.sqlite3",
        study_name=study_name, direction='maximize')
    
    # Optimize
    mlflow.set_experiment(experiment_name='deans_run_2')
    study.optimize(objective, n_trials=5)
    
    
    # To run on terminal after training
    # optuna-dashboard sqlite:///db.sqlite3
    # mlflow ui