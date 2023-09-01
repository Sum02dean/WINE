from abc import abstractclassmethod, ABC
from utils import BaseModel, MyDataset
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import pandas as pd
import typing
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import one_hot
from sklearn.metrics import accuracy_score

# Define simle adaptable neural network architecture
class SimpleNetwork(nn.Module):
    """A simple neural network"""

    def __init__(self, in_dim: int, hidden_dims: typing.Tuple[int,...],
                 final_dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim =hidden_dims
        self.final_dim = final_dim

        # Stack the layers into single tuple
        layer_sizes = (in_dim,) + hidden_dims + (final_dim,)
        num_affine_maps = len(layer_sizes) - 1
        
        # Build the network layers
        self.layers = nn.ModuleList([
            nn.Linear(layer_sizes[idx], layer_sizes[idx + 1], bias=True)
            for idx in range(num_affine_maps)
        ])

        # Activation functions
        self.activation = nn.ReLU()
        self.final_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""

        for idx, current_layer in enumerate(self.layers):
            x = current_layer(x)
            if idx < len(self.layers) - 1:
                x = self.activation(x)
        return self.final_activation(x)
        
# Build a model from SimpleNetwork and BaseModel
class SimmpleNetModel(BaseModel, ABC):
    """Build a simple neural network
    """
    def __init__(self, in_dim: int, hidden_dims: int, final_dim: int, learning_rate: float, epochs: int) -> None:
        
        # Set the device
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        torch.backends.cudnn.benchmark = True
        
        # Model args
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.final_dim = final_dim
        self.model = self.init_model()

        # Data loader args
        self.train_loader = None
        self.test_loader = None

        # Pytorch args
        self.learning_rate = learning_rate
        self.loss_fn = BCEWithLogitsLoss()
        self.optimiser = torch.optim.SGD(self.model.parameters(), self.learning_rate)
        self.max_epochs = epochs
        super().__init__()


    def fit(self, x: np.array, y: np.array) -> None:
        """
        Fit the model to the training data and return the fitted model.
        """
        # Initialize the train dataloaders
        train_dataset = MyDataset(features=x.values, labels=y)
        self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False)

        # Set configurations
        running_loss = 0

        # Loop over epochs
        self.model.train(True)
        for _, epoch in enumerate(range(self.max_epochs)):
            
            # Training
            for i, data in enumerate(self.train_loader):
                
                # Get the inputs & labels
                batch, labels = data

                # Convert the data-types
                batch = batch.type(torch.FloatTensor)
                labels = labels.type(torch.FloatTensor)

                # Transfer to device
                batch, labels = batch.to(self.device), labels.to(self.device)

                # Zero your gradients for every batch!
                self.optimiser.zero_grad()

                # Make predictions for this batch
                outputs = self.model(batch)
                #outputs = outputs.unsqueeze(1)

                # Compute the loss and its gradients
                loss = self.loss_fn(outputs, labels)
                loss.backward()

                # Adjust learning weights
                self.optimiser.step()

                # Gather data and report
                running_loss += loss.item()
                if i % 1000 == 999:
                    last_loss = running_loss / 1000 # loss per batch
                    print('  batch {} loss: {}'.format(i + 1, last_loss))
                    running_loss = 0.

    def predict(self, x: np.array) -> torch.Tensor:
        """
        Predicts the output for a given input.
        """

        # Initialize the train dataloaders
        test_dataset = MyDataset(features=x.values, labels=None)
        self.test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False)

        # Predictions
        preds = []
        all_logits = []

        # Predictions
        for i, data in enumerate(self.test_loader):
            
            # Get the inputs & labels
            batch, _ = data

            # Convert the data-types
            batch = batch.type(torch.FloatTensor)
            
            # Transfer to device
            batch = batch.to(self.device)

            with torch.no_grad():
                logits = self.model(batch)
                predictions = torch.where(logits > 0.5, 1, 0)
                # predictions = [np.argmax(x) for x in predictions.detach().numpy()]
                logits = [x[1] for x in logits.detach().numpy()]
                preds.append(predictions)
                all_logits.append(logits)

        predictions = np.array([item for sublist in preds for item in sublist])
        return predictions

    def init_model(self) -> nn.Module:
        """
        Initializes and returns a pytorch model.
        """
        model = SimpleNetwork(self.in_dim, self.hidden_dims, self.final_dim)
        return model
    
    def transform_data(self, x: np.array, y: np.array) -> (torch.Tensor, torch.Tensor):
        """
        A method that reshapes or transforms the data.
        """
        ohe_y = one_hot(torch.tensor(y.values)).numpy()
        return x, ohe_y.squeeze(1)

if __name__ == "__main__":

    # Read in the parameters from configs file
    import json
    configs = json.load(
        open(
            "/Users/sum02dean/projects/wine_challenge/WINE/configs/config_file.json",
              encoding='utf-8'))
    
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

    # Build the model 
    model = SimmpleNetModel(in_dim=13, hidden_dims=(64, 64), final_dim=2, learning_rate=0.02, epochs=2000)
    x_train, y_train = model.transform_data(x=train_x_raw, y=train_y_raw)
    x_test, y_test = model.transform_data(x=test_x_raw , y=test_y_raw)

    predictions = model.fit_predict(x_train, y_train, x_test)
    print(predictions.shape)
    print(y_test.shape)
    print(accuracy_score(y_test, predictions))
    