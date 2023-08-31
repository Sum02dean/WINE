# Here is where we keeep all of the models
from abc import abstractmethod, ABC
from sklearn.metrics import accuracy_score
import numpy as np
import torch

class BaseModel(ABC):

    @abstractmethod
    def init_model(self) -> None:
        """
        Initializes the model.
        """
        pass
    
    @abstractmethod
    def fit(self, x, y) -> None:
        pass
    
    @abstractmethod
    def predict(self, x) -> np.array:
        """
        Predicts the output for a given input.
        """
        pass
    
    @abstractmethod
    def transform_data(self, x, y) -> None:
        """
        A method that reshapes or transforms the data.
        """
        pass

    def fit_predict(self, x, y, x_val) -> None:
        self.fit(x, y)
        return self.predict(x_val)
    
    def report_accuracy(self, y_true, y_pred) -> float:
        test_acc = accuracy_score(y_true, y_pred)
        return test_acc 
    

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels=None):
        self.labels = labels
        self.features = features
    
    def __len__(self):
        'Denotes the total number of samples'
        return np.shape(self.features)[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        x = self.features[index]

        if self.labels is not None:
            y = self.labels[index]
        else:
            y = x

        return x, y
    