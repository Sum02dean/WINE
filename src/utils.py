# Here is where we keeep all of the models
from abc import abstractmethod, ABC
from sklearn.metrics import accuracy_score
import numpy as np

class BaseModel(ABC):

    @abstractmethod
    def init_model(self) -> None:
        """
        Initializes the model.

        :param self: The instance of the class.
        :return: None
        """
        pass
    
    @abstractmethod
    def fit(self, X, y) -> None:
        pass
    
    @abstractmethod
    def predict(self, X_val) -> np.array:
        pass

    def fit_predict(self, X, y, X_val) -> None:
        self.fit(X, y)
        return self.predict(X_val)
    
    def report_accuracy(self, y_true, y_pred) -> float:
        test_acc = accuracy_score(y_true, y_pred)
        return test_acc 
    
    def reshape_data(self, X, y) -> None:
        X = X.values
        y = y.values.reshape(-1)
        return X, y
   