""" Runs a Bayesian hyperparameter optimisation search using Optuna
to train and predict on a classicifation problem set."""
import json
import logging
import numpy as np
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score
import optuna
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from random_forest import RandomForestModel
from support_vector_classifier import SVClassifierModel
from simple_torch_nn import SimmpleNetModel
from bayesian_neural_network import BayesModel
import os

# Establish mlflow logging config
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Try optimising baseline model hyperparameters with Optuna - bayesian autoML methods
def objective(trial):
    """
    Optimize the hyperparameters of a classifier using a Bayesian search algorithm.

    Parameters:
    - trial: An instance of the Trial class provided by Optuna, which represents a single run of the optimization algorithm.

    Returns:
    - test_acc: A float representing the accuracy of the model on the test set.
    """

    with mlflow.start_run(run_name=str(trial.number)):

        # Sample these classifiers
        classifier_name = trial.suggest_categorical("classifier",
                                                    ['SimpleNetModel',
                                                    'RandomForestModel',
                                                    'SVClassifierModel'])


        # Iterate over model specific hyper-parameters
        if classifier_name == "SVClassifierModel":

            # Define optimizable hyperparameters ranges: C
            lower_sample_c = model_params['svm']['c_lower_sample']
            upper_sample_c = model_params['svm']['c_upper_sample']

            # Define optimizable hyperparameters ranges: gamma
            lower_sample_gamma = model_params['svm']['gamma_lower_sample']
            upper_sample_gamma = model_params['svm']['gamma_upper_sample']

            # Bayesian search over sampled space
            svc_c = trial.suggest_float(
                "svc_c", lower_sample_c, upper_sample_c, log=True)

            svc_gamma = trial.suggest_float(
                "svc_gamma", lower_sample_gamma, upper_sample_gamma, log=True)

            # Build SVCClassifier by wrapping scikit-learn API
            model = SVClassifierModel(C=svc_c, gamma=svc_gamma)

            # Transform data
            x_train, y_train = model.transform_data(train_x_raw, train_y_raw)
            x_test, y_test = model.transform_data(test_x_raw , test_y_raw)

        if classifier_name == "RandomForestModel":

            # Define optimizable hyperparameters ranges: max depth
            lower_sample_max_depth = model_params['random_forest']['max_depth_lower_sample']
            upper_sample_max_depth = model_params['random_forest']['max_depth_upper_sample']

            # Define optimizable hyperparameters ranges: n_estimators
            lower_sample_n_estimators = model_params['random_forest']['n_estimators_lower_sample']
            upper_sample_n_estimators = model_params['random_forest']['n_estimators_upper_sample']

            # Bayesian search over sampled space
            rf_max_depth = trial.suggest_int("rf_max_depth", lower_sample_max_depth,
                                             upper_sample_max_depth, log=True)

            rf_n_estimators = trial.suggest_int("rf_n_estimators", lower_sample_n_estimators,
                                                upper_sample_n_estimators, log=True)

            # Build RandomForest by wrapping scikit-learn API
            model = RandomForestModel(max_depth=rf_max_depth, n_estimators=rf_n_estimators)

            # Transform data
            x_train, y_train = model.transform_data(train_x_raw, train_y_raw)
            x_test, y_test = model.transform_data(test_x_raw , test_y_raw)

        if classifier_name == "SimpleNetModel":

            # Define optimizable hyperparameters ranges: n_layers
            lower_sample_n_layers = model_params['neural_network']['n_layers_lower_sample']
            upper_sample_n_layers = model_params['neural_network']['n_layers_upper_sample']

            # Define optimizable hyperparater ranges: n_nodes
            lower_sample_n_nodes = model_params['neural_network']['n_nodes_lower_sample']
            upper_sample_n_nodes = model_params['neural_network']['n_nodes_upper_sample']

            # Define optimizable hyperparater ranges: lr
            lower_sample_lr = model_params['neural_network']['lr_lower_sample']
            upper_sample_lr = model_params['neural_network']['lr_upper_sample']


            # Bayesian search over sampled space
            nn_number_layers = trial.suggest_int("nn_layers", lower_sample_n_layers,
                                             upper_sample_n_layers, log=False)

            nn_layer_nodes = tuple([
                trial.suggest_int(f"layer_{i}_nodes", lower_sample_n_nodes,
                                  upper_sample_n_nodes, log=False)
                                  for i in range(nn_number_layers)])

            nn_lr = trial.suggest_float("lr", lower_sample_lr,
                                             upper_sample_lr, log=True)

            # Build simple neural network by wrapping Pytorch API
            n_epochs=model_params['neural_network']['n_epochs']
            model = SimmpleNetModel(
                in_dim=13, hidden_dims=nn_layer_nodes, final_dim=2,
                learning_rate=nn_lr, epochs=n_epochs)

            # Transform data
            x_train, y_train = model.transform_data(train_x_raw, train_y_raw)
            x_test, y_test = model.transform_data(test_x_raw , test_y_raw)

        if classifier_name == "BayesModel":

            # Define optimizable hyperparameters ranges: n_layers
            lower_sample_n_layers = model_params[classifier_name]['n_layers_lower_sample']
            upper_sample_n_layers = model_params[classifier_name]['n_layers_upper_sample']

            # Define optimizable hyperparater ranges: n_nodes
            lower_sample_n_nodes = model_params[classifier_name]['n_nodes_lower_sample']
            upper_sample_n_nodes = model_params[classifier_name]['n_nodes_upper_sample']

            # Define optimizable hyperparater ranges: lr
            lower_sample_lr = model_params[classifier_name]['lr_lower_sample']
            upper_sample_lr = model_params[classifier_name]['lr_upper_sample']


            # Bayesian search over sampled space
            nn_number_layers = trial.suggest_int("nn_layers", lower_sample_n_layers,
                                             upper_sample_n_layers, log=False)

            nn_layer_nodes = tuple([trial.suggest_int(f"layer_nodes_{str(i)}", lower_sample_n_nodes,
                                  upper_sample_n_nodes, log=False)
                                  for i in range(nn_number_layers)])

            nn_lr = trial.suggest_float("lr", lower_sample_lr,
                                             upper_sample_lr, log=False)
            
            # Only use the first 3 significant digits of the float
            nn_lr = round(nn_lr, 3)

            # Build simple neural network by wrapping Pytorch API
            n_epochs=model_params['neural_network']['n_epochs']
            model = BayesModel(
                in_dim=13, hidden_dims=nn_layer_nodes, final_dim=2,
                learning_rate=0.02, epochs=n_epochs,
                n_mcs= model_params[classifier_name]['n_mcs'])

            # Transform data
            x_train, y_train = model.transform_data(train_x_raw, train_y_raw)
            x_test, y_test = model.transform_data(test_x_raw , test_y_raw)
        
        # Predict on the train set
        predictions = model.fit_predict(x_train, y_train, x_test)
        test_acc = accuracy_score(y_test, predictions)
    
        
        # Log artifcats
        signature = infer_signature(predictions, predictions)
        mlflow.log_artifacts('/Users/sum02dean/projects/wine_challenge/WINE/data')
        mlflow.log_params(trial.params)
        mlflow.log_metric('test_accuracy', test_acc)
        mlflow.sklearn.log_model(model, classifier_name, signature=signature)

    return test_acc

# Run Main
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Read in the configs file
    FILE_NAME = "/Users/sum02dean/projects/wine_challenge/WINE/configs/config_file.json"
    OUTPUT_DIR_NAME = "/Users/sum02dean/projects/wine_challenge/WINE/output"
    parser.add_argument("--config_file", help="path to configs file", type=str, default=FILE_NAME)
    args = parser.parse_args()

    # Read in the parameters from configs file
    with open(args.config_file, encoding='utf-8') as f:
        configs = json.load(f)

    # Extract parameters
    mlflow_params = configs.get("mlflow_params")
    model_params = configs.get("model_params")
    data_params = configs.get("data_params")

    # Features
    train_x_raw = pd.read_csv(data_params['x_train_path'])
    test_x_raw  = pd.read_csv(data_params['x_test_path'])

    # Labels
    train_y_raw  = pd.read_csv(data_params['y_train_path'])
    test_y_raw  = pd.read_csv(data_params['y_test_path'])

    # Create Optuna study
    study = optuna.create_study(
        storage=mlflow_params["storage"],
        study_name=mlflow_params["study_name"], direction='maximize')

    # Optimize
    mlflow.set_experiment(experiment_name=mlflow_params["study_name"])
    study.optimize(objective, n_trials=mlflow_params["n_trials"])

    # Build outputdir if not existing
    if not os.path.exists(OUTPUT_DIR_NAME):
        os.makedirs(OUTPUT_DIR_NAME)

    # Save the best model parameters
    with open(os.path.join(OUTPUT_DIR_NAME, 
    f"best_params_{study.study_name}.json"), "w", encoding='utf-8') as f:
        json.dump(study.best_params, f)

# Run on terminal after running the prelimainary_model.py file:
# optuna-dashboard sqlite:///db.sqlite3 (be inside the src directory)
# mlflow ui (be inside the src directory).
