""" Bayesian based neural network """
from abc import ABC
import warnings
import typing
import abc
import numpy as np
from tqdm import trange
import torch
import torch.optim
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
from utils import BaseModel

class ParameterDistribution(torch.nn.Module, metaclass=abc.ABCMeta):
    """
    Abstract class that models a distribution over model parameters,
    usable for Bayes by backprop.
    """

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def log_likelihood(self, values: torch.Tensor) -> torch.Tensor:
        """
        Calculate the log-likelihood of the given values
        :param values: Values to calculate the log-likelihood on
        :return: Log-likelihood
        """
        pass

    @abc.abstractmethod
    def sample(self) -> torch.Tensor:
        """
        Sample from this distribution.
        :return: Sample from this distribution.
        """
        pass

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Legacy method."""
        # DO NOT USE THIS METHOD
        warnings.warn('ParameterDistribution should not be called! Use its explicit methods!')
        return self.log_likelihood(values)
class MyDataset(torch.utils.data.Dataset):
    """Custom DataSet class for Pytorch models"""
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
class MultivariateDiagonalGaussian(ParameterDistribution):
    """
    Multivariate diagonal Gaussian distribution,
    i.e., assumes all elements to be independent Gaussians
    but with different means and standard deviations.
    This parameterizes the standard deviation via a parameter rho as
    sigma = softplus(rho).
    """

    def __init__(self, mu: torch.Tensor, rho: torch.Tensor):
        super(MultivariateDiagonalGaussian, self).__init__()  # always make sure to include the super-class init call!
        assert mu.size() == rho.size()
        self.mu = mu
        self.rho = rho
        self.sig = (F.softplus(rho)*0.05 + 1e-5).detach()

    def log_likelihood(self, values: torch.Tensor) -> torch.Tensor:
        dist = Normal(loc=self.mu, scale=self.sig)
        log_likelihood = dist.log_prob(values).sum()
        return log_likelihood

    def sample(self) -> torch.Tensor:
        epsilon = torch.distributions.Normal(0,1).sample(self.rho.size())
        return self.mu + self.sig*epsilon
class GaussianMixturePrior(ParameterDistribution):
    """
    Mixture of two Gaussian distributions as described in Bludell et al., 2015.
    """
    def __init__(self, mu_0: torch.Tensor, sigma_0: torch.Tensor, mu_1: torch.Tensor, sigma_1: torch.Tensor, pi: torch.Tensor):
        super(GaussianMixturePrior, self).__init__()
        self.mu_0 = mu_0 # mean of distribution 0
        self.sigma_0 = sigma_0 # std of distrinution 0
        self.mu_1 = mu_1 # mean of distribution 1
        self.sigma_1 = sigma_1 # std of distribution 1
        self.pi = pi # Probabilistic weight

    def log_likelihood(self, values: torch.Tensor) -> torch.Tensor:
        dist_0 = Normal(loc=self.mu_0, scale=self.sigma_0)
        dist_1 = Normal(loc=self.mu_1, scale=self.sigma_1)
        ll_0 = dist_0.log_prob(values)
        ll_1 = dist_1.log_prob(values)
        return torch.log(self.pi * torch.exp(ll_0) + (1 - self.pi) * torch.exp(ll_1)).sum()

    def sample(self) -> torch.Tensor:
        # Creates a mixture of the two distributions depending on the size parameter pi
        if np.random.rand() < self.pi:
            return Normal(loc=self.mu_0, scale=self.sigma_0).sample()
        else:
            return Normal(loc=self.mu_1, scale=self.sigma_1).sample()
class BayesMultiLoss():

    """ Computes the KLD + NLL multi-objective loss. KLD is computed as mean of n-bathches.
        The final loss is given as a mean over n-monte-carlo samples of the outputs returned by the forward pass of
        BayesNet."""
    def __init__(self, net_outputs, targets, log_posterior, log_prior,
                    batch_size, num_batches, method='exact'):

        # Define fields
        self.net_outputs = net_outputs # Forward pass outputs
        self.targets = targets         # y_batch targets
        self.log_posterior=log_posterior # Log post
        self.log_prior=log_prior     # Log prior
        self.batch_size = batch_size # Batch size
        self.num_batches = num_batches # Number of batches
        self.method = method # KLD method: exact or approx

    def __compute_kld_loss(self):
        """ Computes the kld loss"""

        if self.method == 'exact':
            kld = self.log_posterior - self.log_prior
            kld_scaled = kld / self.num_batches
            return kld_scaled

        elif self.method =='approx':
            log_ratio = self.log_prior - self.log_posterior
            kld = (log_ratio.exp() -1) - log_ratio
            kld_scaled = kld / self.num_batches
        return kld_scaled

    def __compute_nll_loss(self):
        """ Computes the NLL loss"""
        loss = F.nll_loss(F.log_softmax(self.net_outputs, dim=1), self.targets, reduction='sum')
        return loss

    def compute_loss(self):
        """ Computes the combined loss: KLD + NLL"""
        kld = self.__compute_kld_loss()
        nll = self.__compute_nll_loss()
        multi_loss = kld + nll
        return multi_loss
class BayesianLayer(nn.Module):
    """
    Module implementing a single Bayesian feedforward layer.
    It maintains a prior and variational posterior for the weights (and biases)
    and uses sampling to approximate the gradients via Bayes by backprop.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Create a BayesianLayer.

        :param in_features: Number of input features
        :param out_features: Number of output features
        :param bias: If true, use a bias term (i.e., affine instead of linear transformation)
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        # Set hyper priors: doubled because using guassian mixture distribution
        mu_0_prior =  torch.tensor(0.0)
        sigma_0_prior = torch.tensor(0.368)
        mu_1_prior = torch.tensor(0.0)
        sigma_1_prior = torch.tensor(0.00091)
        pi_prior = torch.tensor(0.5)

        # Define prior distribution
        self.prior = GaussianMixturePrior(mu_0 = mu_0_prior,
                                            sigma_0 = sigma_0_prior,
                                            mu_1 = mu_1_prior,
                                            sigma_1 = sigma_1_prior,
                                            pi = pi_prior
        )
        assert isinstance(self.prior, ParameterDistribution)
        assert not any(True for _ in self.prior.parameters()), 'Prior SHOULD NOT have parameters'

        # Set intitial hyper poteriors
        std_mu_init = torch.tensor(0.1) # Posterior distribution initial mean (multivariate diagonal guassian)
        std_rho_init = torch.tensor(1.) # Posterio distribution Parameterisation of Std (multivariate diagonal guassian)

        # Initialize weights by sampling from normal distributions
        # ... ( in_features = number of neurons in layer-1, out_features = number of connections going into layer)
        w_mu_init = Normal(torch.tensor(0.), std_mu_init).sample((out_features, in_features))
        w_rho_init = Normal(torch.tensor(0.), std_rho_init).sample((out_features, in_features))

        # Convert sampled weights into torch parameters for optimisation
        self.weights_var_posterior = MultivariateDiagonalGaussian(
            mu = torch.nn.Parameter(w_mu_init),
            rho = torch.nn.Parameter(w_rho_init)
        )

        # Error check
        assert isinstance(self.weights_var_posterior, ParameterDistribution)
        assert any(True for _ in self.weights_var_posterior.parameters()), 'Weight posterior must have parameters'

        if self.use_bias:
            # Initialize bias with zero mean and with parameterised std 
            b_mu_init = Normal(torch.tensor(0.), std_mu_init).sample((out_features,))
            b_rho_init = Normal(torch.tensor(0.), std_rho_init).sample((out_features,))

            # Use the same posterior family distribution and make them torch parameters for optimisation
            self.bias_var_posterior = MultivariateDiagonalGaussian(
            mu = torch.nn.Parameter(b_mu_init),
            rho = torch.nn.Parameter(b_rho_init)
        )
            assert isinstance(self.bias_var_posterior, ParameterDistribution)
            assert any(True for _ in self.bias_var_posterior.parameters()), 'Bias posterior must have parameters'
        else:
            self.bias_var_posterior = None

    def forward(self, inputs: torch.Tensor):
        """
        Perform one forward pass through this layer.
        """
        # Sample the weights (1st round is from initialised posterior)
        weights = self.weights_var_posterior.sample()

        # Generate the log-liklihood of the prior and log-posterior
        log_prior = self.prior.log_likelihood(weights)
        log_variational_posterior = self.weights_var_posterior.log_likelihood(weights)

        # As in standard machine learning, we simply add on the bias term to each output in the next adjacent layer
        if self.use_bias:
            # Sample the bias posterios and get prior log-likelihood
            bias = self.bias_var_posterior.sample()

            log_prior += self.prior.log_likelihood(bias)
            # Add on the terms to the variatinoal posterior
            log_variational_posterior += self.bias_var_posterior.log_likelihood(bias)
        else:
            bias = None
        # Compute the predictive outputs
        return F.linear(inputs, weights, bias), log_prior, log_variational_posterior
class BayesNet(nn.Module):
    """
    Module implementing a Bayesian feedforward neural network using BayesianLayer objects.
    """

    def __init__(self, in_features: int, hidden_features: typing.Tuple[int, ...], out_features: int):
        """
        Create a BNN.

        :param in_features: Number of input features
        :param hidden_features: Tuple where each entry corresponds to a (Bayesian) hidden layer with
            the corresponding number of features.
        :param out_features: Number of output features
        """

        super().__init__()
        # Dynamically build the number of layers and their sizes
        feature_sizes = (in_features,) + hidden_features + (out_features,)
        num_affine_maps = len(feature_sizes) - 1
        self.layers = nn.ModuleList([
            BayesianLayer(feature_sizes[idx], feature_sizes[idx + 1], bias=True)
            for idx in range(num_affine_maps)
        ])
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform one forward pass through the BNN using a single set of weights
        sampled from the variational posterior.

        :param x: Input features, float tensor of shape (batch_size, in_features)
        """
        # Initialize the "summed" log-prior likelihood and log-variational-posterior likelihood
        log_prior = torch.tensor(0.0)
        log_variational_posterior = torch.tensor(0.0)

        for idx, current_layer in enumerate(self.layers):
            x, log_prior_layer, log_variational_posterior_layer = current_layer(x)
            if idx < len(self.layers) - 1:
                x = self.activation(x)

            log_prior += log_prior_layer
            log_variational_posterior += log_variational_posterior_layer

        return x, log_prior, log_variational_posterior

    def predict_probabilities(self, x: torch.Tensor, num_mc_samples: int = 100) -> torch.Tensor:
        """
        Predict class probabilities for the given features by sampling from this BNN.

        :param x: Features to predict on, float tensor of shape (batch_size, in_features)
        :param num_mc_samples: Number of MC samples to take for prediction
        :return: Predicted class probabilities, float tensor of shape (batch_size, 10)
            such that the last dimension sums up to 1 for each row
        """
        probability_samples = torch.stack([F.softmax(self.forward(x)[0], dim=1) for _ in range(num_mc_samples)], dim=0)
        estimated_probability = torch.mean(probability_samples, dim=0)

        # assert estimated_probability.shape == (x.shape[0], 10)
        assert torch.allclose(torch.sum(estimated_probability, dim=1), torch.tensor(1.0)) # Make sure the probabilities add up to 1
        return estimated_probability, probability_samples

class BayesModel(object):
    """
    BNN using Bayes by backprop
    """

    def __init__(self, in_dim: int,  hidden_dims: typing.Tuple[int, ...],
                 final_dim: int, learning_rate: float, epochs: int, n_mcs: int):
        # Hyperparameters and general parameters
        self.num_epochs = epochs  # number of training epochs
        self.batch_size = 128  # training batch size
        self.learning_rate = learning_rate # training learning rates
        self.in_dim = in_dim
        self.hidden_layers = hidden_dims  # for each entry, creates a hidden layer with the corresponding number of units
        self.print_interval = 100  # number of batches until updated metrics are displayed during training
        self.n_mcs = n_mcs # Number of monte-carlo samples
        self.out_features = final_dim
        super().__init__()

        # BayesNet
        print('Using a BayesNet model')
        self.network = BayesNet(in_features=self.in_dim, hidden_features=self.hidden_layers,
                                out_features=self.out_features)

        # Optimizer for training
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)

    def fit(self,  x: np.array, y: np.array) -> None:
        """
        Train the neural network.
        :param dataset: Dataset you should use for training
        """
        dataset = MyDataset(features=x, labels=y)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

        # Place network into training mode
        self.network.train()

        progress_bar = trange(self.num_epochs)
        for _ in progress_bar:
            num_batches = len(data_loader)
            for batch_idx, (batch_x, batch_y) in enumerate(data_loader):
                # Convert the data-types
                self.network.zero_grad()

                # BayesNet training step via Bayes by backprop
                assert isinstance(self.network, BayesNet)

                loss = torch.tensor([0.0])
                for _ in range(self.n_mcs):
                    current_logits = self.network(batch_x)
                    outputs, log_prior, log_posterior = current_logits

                    # Compute the losses
                    batch_size = data_loader.batch_size
                    BML = BayesMultiLoss(net_outputs=outputs, targets=batch_y,
                                            log_posterior=log_posterior, log_prior=log_prior,
                                            batch_size=batch_size, num_batches=num_batches, method='approx')

                    loss += BML.compute_loss()

                # Backpropagate to get the gradients
                loss = loss/self.n_mcs
                loss.backward()

                # Step the gradients
                self.optimizer.step()

                # Update progress bar with accuracy occasionally
                if batch_idx % self.print_interval == 0:
                    current_logits, _, _ = self.network(batch_x)
                    current_accuracy = (current_logits.argmax(axis=1) == batch_y).float().mean()
                    progress_bar.set_postfix(loss=loss.item(), acc=current_accuracy.item())

    def predict(self, x: np.array) -> np.ndarray:
        """
        Predict the class probabilities for the given data
        :param data_loader: Data loader yielding the samples to predict on
        :return: (num_samples, 10) NumPy float array where the second dimension sums up to 1 for each row
        """

        # Instantaite the data loader
        dataset = MyDataset(features=x, labels=None)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, drop_last=False
        )
        self.network.eval()
        probability_batches = []
        predictive_probability_dist_batches = []
        for batch_x, _ in data_loader:

            current_probabilities, predictive_distribution = self.network.predict_probabilities(batch_x)
            current_probabilities  = current_probabilities.detach().numpy()
            predictive_distribution = predictive_distribution.detach().numpy()
            probability_batches.append(current_probabilities)
            predictive_probability_dist_batches.append(predictive_distribution)

        output = np.concatenate(probability_batches, axis=0)
        # output_pred_dist = predictive_probability_dist_batches
        assert isinstance(output, np.ndarray)
        assert np.allclose(np.sum(output, axis=1), 1.0)
        return output
    
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
        x = torch.FloatTensor(x.to_numpy())
        y = torch.LongTensor(y.to_numpy()).squeeze(-1)
        return x, y
    

    def fit_predict(self, x, y, x_val) -> (np.ndarray, np.ndarray):
        """Combines fit and predict functions

        Args:
            dataset (torch.utils.data.Dataset): Dataset object from pytorch

        Returns:
            (np.ndarray, np.ndarray): The MLE and predictive distribution
        """
        self.fit(x, y)
        x = self.predict(x_val)
        x = np.round(x)
        predictions = np.argmax(x, axis=1)
        return predictions
    
if __name__ == '__main__':
    pass