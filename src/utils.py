# Here is where we keeep all of the models
import pandas as pd
from abc import abstractmethod, ABC
from sklearn.metrics import accuracy_score
import numpy as np
import torch

class BaseModel(ABC):
    """A base model to implement a machine learning model
    """
    @abstractmethod
    def init_model(self) -> None:
        """
        Initializes the model.
        """
        pass

    @abstractmethod
    def fit(self, x: np.array, y: np.array) -> None:
        pass

    @abstractmethod
    def predict(self, x: np.array) -> np.array:
        """
        Predicts the output for a given input.
        """
        pass

    @abstractmethod
    def transform_data(self, x: pd.DataFrame, y: pd.Series) -> None:
        """
        A method that reshapes or transforms the data.
        """
        pass

    def fit_predict(self, x: np.array, y: np.array, x_val: np.array) -> np.ndarray:
        self.fit(x, y)
        return self.predict(x_val)

    def report_accuracy(self, y_true: np.array, y_pred: np.array) -> float:
        test_acc = accuracy_score(y_true, y_pred)
        return test_acc

class MyDataset(torch.utils.data.Dataset):
    """Custom DataSet class for Pytorch models"""
    def __init__(self, features, labels=None):
        self.labels = labels
        self.features = features

    def __len__(self) -> int:
        'Denotes the total number of samples'
        return np.shape(self.features)[0]

    def __getitem__(self, index: int) -> (np.array, np.array):
        'Generates one sample of data'
        x = self.features[index]

        if self.labels is not None:
            y = self.labels[index]
        else:
            y = x

        return x, y
    