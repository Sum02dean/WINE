# Import the data


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

# Try optimising hyperparameters with optuna - bayesian methods

class Objective(object):
    def __init__(self, study_name):

        # Study name
        self.study_name = study_name

        # Features have already been pre-processed
        self.train_x = pd.read_csv('/Users/sum02dean/projects/wine_challenge/WINE/data/scaled_unclipped_train_x.csv')
        self.test_x = pd.read_csv('/Users/sum02dean/projects/wine_challenge/WINE/data/scaled_unclipped_test_x.csv')

        # Labels
        self.train_y = pd.read_csv('/Users/sum02dean/projects/wine_challenge/WINE/data/train_y.csv')
        self.test_y = pd.read_csv('/Users/sum02dean/projects/wine_challenge/WINE/data/test_y.csv')

        # Reformat x & y
        self.train_labels = self.train_y.values.reshape(-1)
        self.train_features = self.train_x.values

        self.test_labels = self.test_y.values.reshape(-1)
        self.test_features = self.test_x.values

    def __call__(self, trial):
        x, y = self.train_features, self.train_labels

        # classifier_name = trial.suggest_categorical("classifier", ["SVC", "RandomForest"])
        classifier_name = 'RandomForest'
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
        trained_model = classifier_obj.fit(X=self.train_features, y=self.train_labels)

        # Infer on the test set
        test_pred = trained_model.predict(self.test_features)
        test_acc = accuracy_score(self.test_labels, test_pred)
        return test_acc

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--study_name", help="study name for optuna log",
                        type=str)
    args = parser.parse_args()
    print(args)

    # Load the dataset in advance for reusing it each trial execution.
    study_name = args.study_name
    objective = Objective(study_name=study_name)
    study = optuna.create_study(
        storage="sqlite:///db.sqlite3",
        study_name=study_name, direction='maximize')
    
    #Optimize
    study.optimize(objective, n_trials=100)

# To run on terminal after training
# optuna-dashboard sqlite:///db.sqlite3