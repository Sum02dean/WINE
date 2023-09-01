""" SVM classifier model"""
import sklearn
import numpy as np
from abc import ABC
from utils import BaseModel
import pandas as pd
from sklearn import svm

class SVClassifier(BaseModel, ABC):
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
        model = svm.SVC(C=self.C, gamma=self.gamma)
        return model

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
        X = x.to_numpy()
        y = y.to_numpy().reshape(-1)
        return X, y


if __name__ == "__main__":

     # Features
    train_x_raw = pd.read_csv('/Users/sum02dean/projects/wine_challenge/WINE/data/scaled_unclipped_train_x.csv')
    test_x_raw  = pd.read_csv('/Users/sum02dean/projects/wine_challenge/WINE/data/scaled_unclipped_test_x.csv')

    # Labels
    train_y_raw  = pd.read_csv('/Users/sum02dean/projects/wine_challenge/WINE/data/train_y.csv')
    test_y_raw  = pd.read_csv('/Users/sum02dean/projects/wine_challenge/WINE/data/test_y.csv')

    print("Initializing model")
    model = SVClassifier()
    X_train, y_train = model.transform_data(train_x_raw, train_y_raw )
    X_test, y_test = model.transform_data(test_x_raw , test_y_raw )

    predictions = model.fit_predict(X_train, y_train, X_test)
    accuracy = model.report_accuracy(y_test, predictions)
    print(f"Predictions: {predictions}")
    print(f"Accuracy: {accuracy}")

