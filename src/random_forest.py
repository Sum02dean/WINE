""" Random forest classifier model"""
import sklearn
import numpy as np
from abc import abstractmethod, ABC
from utils import BaseModel
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

class RandomForest(BaseModel, ABC):
    """Build a random forest classifier
    """
    def __init__(self, max_depth: int=10, n_estimators: int=20):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.model = self.init_model()
        super().__init__()

    def fit(self, X, y) -> None:
        """
        Fit the model to the training data and return the fitted model.
        
        Parameters:
            X (array-like): The training input samples.
            y (array-like): The target values.
            
        Returns:
            self: The fitted model.
        """
        return  self.model.fit(X, y)
    
    def predict(self, X) -> np.array:
        """
        Predicts the output for a given input.

        Parameters:
            X (array-like): The input data for which the output needs to be predicted.

        Returns:
            array-like: The predicted output for the given input data.
        """
        return self.model.predict(X)
    
    
    def init_model(self) -> sklearn.ensemble.RandomForestClassifier:
        """
        Initializes and returns a trained Random Forest Classifier model.

        Returns:
            sklearn.ensemble.RandomForestClassifier: The trained Random Forest Classifier model.
        """
        model = RandomForestClassifier(max_depth=self.max_depth, n_estimators=self.n_estimators)
        return model
    
if __name__ == "__main__":

    # Features
    train_x_raw = pd.read_csv('/Users/sum02dean/projects/wine_challenge/WINE/data/scaled_unclipped_train_x.csv')
    test_x_raw  = pd.read_csv('/Users/sum02dean/projects/wine_challenge/WINE/data/scaled_unclipped_test_x.csv')

    # Labels
    train_y_raw  = pd.read_csv('/Users/sum02dean/projects/wine_challenge/WINE/data/train_y.csv')
    test_y_raw  = pd.read_csv('/Users/sum02dean/projects/wine_challenge/WINE/data/test_y.csv')

    print("Initializing model")
    model = RandomForest()
    
    X_train, y_train = model.reshape_data(train_x_raw, train_y_raw )
    X_test, y_test = model.reshape_data(test_x_raw , test_y_raw )

    predictions = model.fit_predict(X_train, y_train, X_test)
    accuracy = model.report_accuracy(y_test, predictions)
    print(f"Predictions: {predictions}")
    print(f"Accuracy: {accuracy}")

