"""Model Parameters Module."""
import torch
import torch.nn as nn
import torch.optim as optim
from .search import SamplingSearch, GreedySearch, BeamSearch
from .utils import gaussian_mixture
from .loss_functions import (
    mse_cc_loss,
    correlation_coefficient_loss,
    kl_divergence_loss,
    BCEIgnoreNaN,
)

SEARCH_FACTORY = {
    "sampling": SamplingSearch,
    "greedy": GreedySearch,
    "beam": BeamSearch,
}

# LSTM(10, 20, 2) -> input has 10 features, 20 hidden size and 2 layers.
# NOTE: Make sure to set batch_first=True. Optionally set bidirectional=True
RNN_CELL_FACTORY = {"lstm": nn.LSTM, "gru": nn.GRU}

LOSS_FN_FACTORY = {
    "mse": nn.MSELoss(),
    "l1": nn.L1Loss(),
    "binary_cross_entropy": nn.BCELoss(),
    "mse_and_pearson": mse_cc_loss,
    "pearson": correlation_coefficient_loss,
    "kld": kl_divergence_loss,
    "binary_cross_entropy_ignore_nan_and_sum": BCEIgnoreNaN("sum"),
    "binary_cross_entropy_ignore_nan_and_mean": BCEIgnoreNaN("mean"),
}

OPTIMIZER_FACTORY = {
    "Adadelta": optim.Adadelta,
    "Adagrad": optim.Adagrad,
    "Adam": optim.Adam,
    "Sparseadam": optim.SparseAdam,
    "Adamax": optim.Adamax,
    "RMSprop": optim.RMSprop,
    "Rprop": optim.Rprop,
    "ASGD": optim.ASGD,
    "LBFGS": optim.LBFGS,
    "SGD": optim.SGD,
}

ACTIVATION_FN_FACTORY = {
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "selu": nn.SELU(),
    "tanh": nn.Tanh(),
    "lrelu": nn.LeakyReLU(),
    "elu": nn.ELU(),
    "celu": nn.CELU(),
}

AAE_DISTRIBUTION_FACTORY = {
    "Gaussian": torch.randn,
    "Uniform": torch.rand,
    "Gaussian_Mixture": gaussian_mixture,
}
