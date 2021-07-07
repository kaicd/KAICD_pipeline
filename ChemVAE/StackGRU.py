"""Stack Augmented GRU Implementation."""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


logger = logging.getLogger("stack_gru")


class StackGRU(nn.Module):
    """Stack Augmented Gated Recurrent Unit (GRU) class."""

    def __init__(self, params):
        """
        Initialization.
        Reference:
            GRU layers intended to help with the training of VAEs by weakening
            the decoder as proposed in: https://arxiv.org/abs/1511.06349.
        Args:
            params (dict): Hyperparameters.
        Items in params:
            embedding_size (int): The embedding size for the dict tokens
            rnn_cell_size (int): Hidden size of GRU.
            vocab_size (int): Output size of GRU (vocab size).
            stack_width (int): Number of stacks in parallel.
            stack_depth (int): Stack depth.
            n_layers (int): The number of GRU layer.
            dropout (float): Dropout on the output of GRU layers except the
                last layer.
            batch_size (int): Batch size.
            lr (float, optional): Learning rate default 0.01.
            optimizer (str, optional): Choice from OPTIMIZER_FACTORY.
                Defaults to 'adadelta'.
            padding_index (int, optional): Index of the padding token.
                Defaults to 0.
        """

        super(StackGRU, self).__init__()
        self.rnn_cell_size = params["rnn_cell_size"]
        self.vocab_size = params["vocab_size"]
        self.embedding_type = params.get("embedding", "learned")
        self.embedding_size = (
            params["embedding_size"]
            if self.embedding_type == "learned"
            else self.vocab_size
        )
        self.stack_width = params["stack_width"]
        self.stack_depth = params["stack_depth"]
        self.n_layers = params["n_layers"]
        self._update_batch_size(params["batch_size"])
        self.gru_input = self.embedding_size
        self.use_stack = params.get("use_stack", True)

        # Create the update function conditioned on whether stack is used.
        if self.use_stack:
            self.gru_input += self.stack_width
        else:
            logger.warning("Attention: No stack will be used")
        # Network
        self.stack_controls_layer = nn.Linear(
            in_features=self.rnn_cell_size, out_features=3
        )
        self.stack_input_layer = nn.Linear(
            in_features=self.rnn_cell_size, out_features=self.stack_width
        )
        self.embedding = nn.Embedding(
            self.vocab_size,
            self.embedding_size,
            padding_idx=params.get("pad_index", 0),
        )
        self.gru = nn.GRU(
            self.gru_input,
            self.rnn_cell_size,
            self.n_layers,
            bidirectional=False,
            dropout=params["dropout"],
        )
        self._check_params()
        # Plug in one hot-vectors and freeze weights
        if self.embedding_type == "one_hot":
            self.embedding.load_state_dict(
                {"weight": torch.nn.functional.one_hot(torch.arange(self.vocab_size))}
            )
            self.embedding.weight.requires_grad = False

    def forward(self, input_token, hidden, stack):
        """
        StackGRU forward function.
        Args:
            input_token (torch.Tensor): LongTensor containing
                indices of the input token of or `[1, batch_size]`.
            hidden (torch.Tensor): Hidden state of size
                `[n_layers, batch_size, rnn_cell_size]`.
            stack (torch.Tensor): Previous step's stack of size
                `[batch_size, stack_depth, stack_width]`.
        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor): output, hidden, stack.
            Output of size `[batch_size, vocab_size]`.
            Hidden state of size `[1, batch_size, rnn_cell_size]`.
            Stack of size `[batch_size, stack_depth, stack_width]`.
        """
        # Set device
        device = input_token.device
        embedded_input = self.embedding(input_token)
        if self.use_stack:
            # Stack update: Pre-gru stack update operations
            stack_controls = self.stack_controls_layer(hidden[-1, :, :])
            stack_controls = F.softmax(stack_controls, dim=-1)
            stack_input = self.stack_input_layer(hidden[-1, :, :].unsqueeze(0))
            stack_input = torch.tanh(stack_input)
            # Stack augmentation
            batch_size = stack.size(0)
            controls = stack_controls.view(-1, 3, 1, 1)
            zeros_at_the_bottom = torch.zeros(batch_size, 1, self.stack_width)
            zeros_at_the_bottom = Variable(zeros_at_the_bottom.to(device))
            a_push, a_pop, a_no_op = (controls[:, 0], controls[:, 1], controls[:, 2])
            # For unknown reasons, stack moves to cpu
            # Thus setting up stack's device separately.
            stack = stack.to(device)
            stack_down = torch.cat((stack[:, 1:], zeros_at_the_bottom), dim=1)
            stack_up = torch.cat((stack_input.permute(1, 0, 2), stack[:, :-1]), dim=1)
            new_stack = (a_no_op * stack) + (a_push * stack_up) + (a_pop * stack_down)
            stack_top = new_stack[:, 0, :].unsqueeze(0)
            gru_input = torch.cat((embedded_input, stack_top), dim=2)
        else:
            gru_input = input_token.to(device)
            new_stack = stack.to(device)

        output, hidden = self.gru(gru_input, hidden)
        return output, hidden, new_stack

    def _forward_pass_padded(self, *args):
        raise NotImplementedError

    def _forward_pass_packed(self, *args):
        raise NotImplementedError

    def _update_batch_size(self, batch_size: int, device: torch.device = None) -> None:
        """Updates the batch_size
        Arguments:
            batch_size (int): New batch size
        """
        self.batch_size = batch_size
        self.expected_shape = torch.tensor(
            [self.n_layers, self.batch_size, self.rnn_cell_size]
        )
        # Variable to initialize hidden state and stack
        self.init_hidden = Variable(
            torch.zeros(self.n_layers, self.batch_size, self.rnn_cell_size).to(device)
            if device is not None
            else torch.zeros(self.n_layers, self.batch_size, self.rnn_cell_size)
        )
        self.init_stack = Variable(
            torch.zeros(self.batch_size, self.stack_depth, self.stack_width).to(device)
            if device is not None
            else torch.zeros(self.batch_size, self.stack_depth, self.stack_width)
        )

    def _check_params(self):
        """
        Runs size checks on input parameter
        """
        if self.rnn_cell_size < self.embedding_size:
            logger.warning("Refrain from squashing embeddings in RNN cells")

    def _associate_language(self, language):
        """
        Raises:
            TypeError:
        """
        if isinstance(language, SMILESLanguage):
            self.smiles_language = language

        else:
            raise TypeError(
                "Please insert a smiles language (object of type "
                "pytoda.smiles.smiles_language.SMILESLanguage . Given was "
                f"{type(language)}"
            )