""" Random forest classifier model"""
import sklearn
import numpy as np
from abc import ABC
from utils import BaseModel
from sklearn.ensemble import RandomForestClassifier

class RandomForest(BaseModel, ABC):
    """Build a random forest classifier
    """
    def __init__(self, max_depth: int=10, n_estimators: int=20):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.model = self.init_model()
        super().__init__()

    def fit(self, x: np.array, y: np.array) -> sklearn.ensemble.RandomForestClassifier:
        """
        Fit the model to the training data and return the fitted model.
        
        Parameters:
            x (array-like): The training input samples.
            y (array-like): The target values.
            
        Returns:
            self: The fitted model.
        """
        return  self.model.fit(x, y)

    def predict(self, x: np.array) -> np.array:
        """
        Predicts the output for a given input.

        Parameters:
            x (array-like): The input data for which the output needs to be predicted.

        Returns:
            array-like: The predicted output for the given input data.
        """
        return self.model.predict(x)

    def init_model(self) -> sklearn.ensemble.RandomForestClassifier:
        """
        Initializes and returns a trained Random Forest Classifier model.

        Returns:
            sklearn.ensemble.RandomForestClassifier: The trained Random Forest Classifier model.
        """
        model = RandomForestClassifier(max_depth=self.max_depth, n_estimators=self.n_estimators)
        return model

    def transform_data(self, x: np.array, y: np.array) -> (np.array, np.array):
        """
        A method that reshapes the data.

        Args:
            x (array-like): The input data.
            y (array-like): The target data.
        Returns:
            x (array-like): The reshaped input data.
            y (array-like): The reshaped target data.
        """

        x = x.to_numpy()
        y = y.to_numpy().reshape(-1)
        return x, y
