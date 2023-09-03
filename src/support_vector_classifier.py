""" SVM classifier model"""
import numpy as np
import sklearn
from abc import ABC
from utils import BaseModel
from sklearn import svm

class SVClassifierModel(BaseModel, ABC):
    """Build a SVM classifier
    """
    def __init__(self, C: float=1.0, gamma: float=1.0):
        self.C = C
        self.gamma = gamma
        self.model = self.init_model()
        super().__init__()

    def fit(self, x, y) -> None:
        """
        Fit the model to the training data and return the fitted model.
        
        Parameters:
            x (array-like): The training input samples.
            y (array-like): The target values.
            
        Returns:
            self: The fitted model.
        """
        return self.model.fit(x, y)

    def predict(self, x) -> np.array:
        """
        Predicts the output for a given input.
        """
        return self.model.predict(x)

    def init_model(self) -> sklearn.svm.SVC:
        """
        Initializes and returns a trained SVM model.

        Returns:
            sklearn.svm.SVC: The svm Classifier model.
        """
        svm_model = svm.SVC(C=self.C, gamma=self.gamma)
        return svm_model

    def transform_data(self, x: np.array, y: np.array) -> (np.array, np.array):
        """
        A method that reshapes or transforms the data.

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
    