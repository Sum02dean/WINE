import typing
import numpy as np
import pandas as pd
from abc import ABC
from utils import BaseModel, MyDataset
import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import one_hot

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
    def __init__(self, in_dim: int, hidden_dims: int, 
                 final_dim: int, learning_rate: float, epochs: int) -> None:
        """Initialize the model

        Args:
            in_dim (int): Number of input features
            hidden_dims (int): Number of hidden layers
            final_dim (int): Number of output features
            learning_rate (float): Learning rate
            epochs (int): Number of epochs
        """

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

        # Loop over epochs
        self.model.train(True)
        for _, _ in enumerate(range(self.max_epochs)):

            # Training
            for _, data in enumerate(self.train_loader):

                # Get the input & labels
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

    def predict(self, x: np.array) -> torch.Tensor:
        """
        Predicts the output for a given input.
        """

        # Initialize the train dataloaders
        test_dataset = MyDataset(features=x.values, labels=None)
        self.test_loader = DataLoader(test_dataset, batch_size=64,
                                       shuffle=False, drop_last=False)

        # Predictions
        all_predictions = []
        all_logits = []

        # Predictions
        for _, data in enumerate(self.test_loader):

            # Get the inputs & labels
            batch, _ = data

            # Convert the data-types
            batch = batch.type(torch.FloatTensor)

            # Transfer to device
            batch = batch.to(self.device)

            with torch.no_grad():
                logits = self.model(batch)
                rounded_logits = torch.where(logits > 0.5, 1, 0)
                preds = [np.argmax(x) for x in rounded_logits.detach().numpy()]
                logits = [x[1] for x in logits.detach().numpy()]
                all_predictions.append(preds)
                all_logits.append(logits)

        all_predictions = np.array([item for sublist in preds for item in sublist])
        return all_predictions

    def init_model(self) -> nn.Module:
        """
        Initializes and returns a pytorch model.
        """
        simple_model = SimpleNetwork(self.in_dim, self.hidden_dims, self.final_dim)
        return simple_model

    def transform_data(self, x: np.array, y: np.array) -> (torch.Tensor, torch.Tensor):
        """
        A method that reshapes or transforms the data.
        """
        ohe_y = one_hot(torch.tensor(y.values)).numpy()
        return x, ohe_y.squeeze(1)
