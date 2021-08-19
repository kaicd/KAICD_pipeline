"""Loss Functions."""
import math
from typing import Iterable

import torch
import torch.nn as nn


def vae_loss_function(
    decoder_loss: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_growth: float = 0.0015,
    step: int = None,
    eval_mode: bool = False,
) -> (torch.Tensor, torch.Tensor):
    """
    Loss Function for VAE.

    Reference:
        Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        https://arxiv.org/abs/1312.6114

    Args:
        decoder_loss (torch.Tensor): Reconstruction cross-entropy loss over the
            entire sequence of the input.
        mu (torch.Tensor): The latent mean, mu.
        logvar (torch.Tensor): Log of the latent variance.
        kl_growth (float): The rate at which the weight grows. Defaults to
            0.0015 resulting in a weight of 1 around step=9000.
        step (int): Global train step, not needed if eval_mode.
        eval_mode (bool): Set to True for model evaluation during test and
            validation. Defaults to False, for model training.

    Returns:
        (torch.Tensor, torch.Tensor): decoder loss and encoder loss.

        The VAE loss consisting of the cross-entropy (decoder)
        loss and KL divergence (encoder) loss.
    """
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    if eval_mode:
        kl_w = 1.0
    else:
        kl_w = kl_weight(step, growth_rate=kl_growth)
    return kl_w * kl_div + decoder_loss, kl_div


def mse_cc_loss(labels, predictions):
    """Compute loss based on MSE and Pearson correlation.

    The main assumption is that MSE lies in [0,1] range, i.e.: range is
    comparable with Pearson correlation-based loss.

    Args:
        labels (torch.Tensor): reference values
        predictions (torch.Tensor): predicted values

    Returns:
        torch.Tensor: A loss that computes the following:
        \$mse(labels, predictions) + 1 - r(labels, predictions)^2\$  # noqa
    """
    mse_loss_fn = nn.MSELoss()
    mse_loss = mse_loss_fn(predictions, labels)
    cc_loss = correlation_coefficient_loss(labels, predictions)
    return mse_loss + cc_loss


def kl_weight(step, growth_rate=0.004):
    """Kullback-Leibler weighting function.

    KL divergence weighting for better training of
    encoder and decoder of the VAE.

    Reference:
        https://arxiv.org/abs/1511.06349

    Args:
        step (int): The training step.
        growth_rate (float): The rate at which the weight grows.
            Defaults to 0.0015 resulting in a weight of 1 around step=9000.

    Returns:
        float: The weight of KL divergence loss term.
    """
    weight = 1 / (1 + math.exp((15 - growth_rate * step)))
    return weight


def kl_divergence_loss(mu, logvar):
    """KL Divergence loss from VAE paper.

    Reference:
        Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014

    Args:
        mu (torch.Tensor): Encoder output of means of shape
            `[batch_size, input_size]`.
        logvar (torch.Tensor): Encoder output of logvariances of shape
            `[batch_size, input_size]`.
    Returns:
        The KL Divergence of the thus specified distribution and a unit
        Gaussian.
    """
    # Increase precision (numerical underflow caused negative KLD).
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def pearsonr(x, y):
    """Compute Pearson correlation.

    Args:
        x (torch.Tensor): 1D vector
        y (torch.Tensor): 1D vector of the same size as y.

    Raises:
        TypeError: not torch.Tensors.
        ValueError: not same shape or at least length 2.

    Returns:
        Pearson correlation coefficient.
    """
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError("Function expects torch Tensors.")

    if len(x.shape) > 1 or len(y.shape) > 1:
        raise ValueError(" x and y must be 1D Tensors.")

    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    if len(x) < 2:
        raise ValueError("x and y must have length at least 2.")

    # If an input is constant, the correlation coefficient is not defined.
    if bool((x == x[0]).all()) or bool((y == y[0]).all()):
        raise ValueError("Constant input, r is not defined.")

    mx = x - torch.mean(x)
    my = y - torch.mean(y)
    cost = torch.sum(mx * my) / (
        torch.sqrt(torch.sum(mx ** 2)) * torch.sqrt(torch.sum(my ** 2))
    )
    return torch.clamp(cost, min=-1.0, max=1.0)


def correlation_coefficient_loss(labels, predictions):
    """Compute loss based on Pearson correlation.

    Args:
        labels (torch.Tensor): reference values
        predictions (torch.Tensor): predicted values

    Returns:
        torch.Tensor: A loss that when minimized forces high squared correlation coefficient:
        \$1 - r(labels, predictions)^2\$  # noqa
    """
    return 1 - pearsonr(labels, predictions) ** 2


def joint_loss(
    outputs, targets, reconstruction_loss, kld_loss, mu, logvar, alpha=0.5, beta=1.0
):
    """Loss Function from VAE paper.
    Reference:
        Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    Args:
        outputs (torch.Tensor): The decoder output of shape
            `[batch_size, input_size]`.
        targets (torch.Tensor): The encoder input of shape
            `[batch_size, input_size]`.
        alpha (float): Weighting of the 2 losses. Alpha in range [0, 1].
            Defaults to 0.5.
        beta (float): Scaling of the KLD in range [1., 100.] according to
            beta-VAE paper. Defaults to 1.0.
    Returns:
        (torch.Tensor, torch.Tensor, torch.Tensor): joint_loss, rec_loss, kld_loss  # noqa
        The VAE joint loss is a weighted combination of the
        reconstruction loss (e.g. L1, MSE) and the KL divergence of a
        multivariate unit Gaussian and the latent space representation.
        Reconstruciton loss is summed across input size and KL-Div is
        averaged across latent space.
        This comes from the fact that L2 norm is feature normalized
        and KL is z-dim normalized, s.t. alpha can be tuned for
        varying X, Z dimensions.
    """
    rec_loss = reconstruction_loss(outputs, targets)
    kld_loss = kld_loss(mu, logvar)
    joint_loss = alpha * rec_loss + (1 - alpha) * beta * kld_loss

    return joint_loss, rec_loss, kld_loss


def gg_loss(output: torch.Tensor, target_output: torch.Tensor) -> float:
    """
    The graph generation loss is the KL divergence between the target and predicted actions.

    Args:
    ----
        output (torch.Tensor) : Predicted APD tensor.
        target_output (torch.Tensor) : Target APD tensor.

    Returns:
    -------
        loss (float) : Average loss for this output.
    """
    # define activation function; note that one must use the softmax in the
    # KLDiv, never the sigmoid, as the distribution must sum to 1
    LogSoftmax = nn.LogSoftmax(dim=1)
    output = LogSoftmax(output)
    # normalize the target output (as can contain information on > 1 graph)
    target_output = target_output / torch.sum(target_output, dim=1, keepdim=True)
    # define loss function and calculate the los
    criterion = torch.nn.KLDivLoss(reduction="batchmean")
    loss = criterion(target=target_output, input=output)

    return loss


class BCEIgnoreNaN(nn.Module):
    """Wrapper for BCE function that ignores NaNs"""

    def __init__(self, reduction: str, class_weights: tuple = (1, 1)) -> None:
        """

        Args:
            reduction (str): Reduction applied in loss function. Either sum or mean.
            class_weights (tuple, optional): Class weights for loss function.
                Defaults to (1, 1), i.e. equal class weighhts.
        """
        super(BCEIgnoreNaN, self).__init__()
        self.loss = nn.BCELoss(reduction="none")

        if reduction != "sum" and reduction != "mean":
            raise ValueError(f"Chose reduction type as mean or sum, not {reduction}")
        self.reduction = reduction

        if not isinstance(class_weights, Iterable):
            raise TypeError(f"Pass iterable for weights, not: {type(class_weights)}")
        if not len(class_weights) == 2:
            raise ValueError(f"Class weight len should be 2, not: {len(class_weights)}")
        if not all(w > 0 for w in class_weights):
            raise ValueError(f"All weigths should be positive not: {class_weights}")

        self.class_weights = class_weights

    def forward(self, yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            y (torch.Tensor): Labels (1D or 2D).
            yhat (torch.Tensor): Predictions (1D or 2D).

        NOTE: This function has side effects (in-place modification of labels).
        This is needed since deepcopying the tensor destroys gradient flow. Just be
        aware that repeated calls of this function with the same variables lead to
        different results if and only if at least one nan is in y.

        Returns:
            torch.Tensor: BCE loss
        """

        # Find position of NaNs, set them to 0 to well-define BCE loss and
        # then filter them out
        nans = ~torch.isnan(y)
        y[y != y] = 0
        loss = self.loss(yhat, y) * nans.type(torch.float32)

        # Set a tensor with class weights, equal shape to labels.
        # NaNs are 0 in y now, but since loss is 0, downstream calc is unaffected.
        weight_tensor = torch.ones(y.shape).to(y.device)
        weight_tensor[y == 0.0] = self.class_weights[0]
        weight_tensor[y == 1.0] = self.class_weights[1]

        out = loss * weight_tensor

        if self.reduction == "mean":
            return torch.mean(out)
        elif self.reduction == "sum":
            return torch.sum(out)
