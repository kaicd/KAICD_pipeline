import torch as th
import torch.nn as nn
import pytoda
from paccmann_chemistry.models.stack_rnn import StackGRU


class StackGRUEncoder(StackGRU):
    """Stacked GRU Encoder."""

    def __init__(self, params, **kwargs):
        """
        Constructor.
        Args:
            params (dict): Hyperparameters.
        Items in params:
            latent_dim (int): Size of latent mean and variance.
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
            bidirectional (bool, optional): Whether to train a bidirectional
                GRU. Defaults to False.
        """
        super(StackGRUEncoder, self).__init__(params)

        self.bidirectional = params.get("bidirectional", False)
        self.n_directions = 2 if self.bidirectional else 1

        # In case of bidirectionality, we create a second StackGRU object that
        # will see the sequence in reverse direction
        if self.bidirectional:
            self.backward_stackgru = StackGRU(params)

        self.latent_dim = params["latent_dim"]
        self.hidden_to_mu = nn.Linear(
            in_features=self.rnn_cell_size * self.n_directions,
            out_features=self.latent_dim,
        )
        self.hidden_to_logvar = nn.Linear(
            in_features=self.rnn_cell_size * self.n_directions,
            out_features=self.latent_dim,
        )

    def encoder_train_step(self, input_seq):
        """
        The Encoder Train Step.
        Args:
            input_seq (torch.Tensor): the sequence of indices for the input
            of shape `[max batch sequence length +1, batch_size]`, where +1 is
            for the added start_index.
        Note: Input_seq is an output of sequential_data_preparation(batch) with
            batches returned by a DataLoader object.
        Returns:
            (torch.Tensor, torch.Tensor): mu, logvar
            mu is the latent mean of shape `[1, batch_size, latent_dim]`.
            logvar is the log of the latent variance of shape
                `[1, batch_size, latent_dim]`.
        """
        # Forward pass
        hidden = self.init_hidden
        stack = self.init_stack

        hidden = self._forward_fn(input_seq, hidden, stack)

        mu = self.hidden_to_mu(hidden)
        logvar = self.hidden_to_logvar(hidden)

        return mu, logvar

    def _forward_pass_padded(self, input_seq, hidden, stack):
        """
        The Encoder Train Step.
        Args:
            input_seq (torch.Tensor): the sequence of indices for the input
            of shape `[max batch sequence length +1, batch_size]`, where +1 is
            for the added start_index.
        Note: Input_seq is an output of sequential_data_preparation(batch) with
            batches returned by a DataLoader object.
        Returns:
            (torch.Tensor, torch.Tensor): mu, logvar
            mu is the latent mean of shape `[1, batch_size, latent_dim]`.
            logvar is the log of the latent variance of shape
                `[1, batch_size, latent_dim]`.
        """
        if isinstance(input_seq, nn.utils.rnn.PackedSequence) or not isinstance(
            input_seq, th.Tensor
        ):
            raise TypeError("Input is PackedSequence or is not a Tensor")
        expanded_input_seq = input_seq.unsqueeze(1)
        for input_entry in expanded_input_seq:
            _output, hidden, stack = self(input_entry, hidden, stack)

        hidden = self._post_gru_reshape(hidden)

        # Backward pass:
        if self.bidirectional:
            assert len(input_seq.shape) == 2, "Input Seq must be 2D Tensor."
            hidden_backward = self.backward_stackgru.init_hidden
            stack_backward = self.backward_stackgru.init_stack

            # [::-1] not yet implemented in torch.
            # We roll up time from end to start
            for input_entry_idx in range(len(expanded_input_seq) - 1, -1, -1):

                (
                    _output_backward,
                    hidden_backward,
                    stack_backward,
                ) = self.backward_stackgru(
                    expanded_input_seq[input_entry_idx], hidden_backward, stack_backward
                )
            # Concatenate forward and backward
            hidden_backward = self._post_gru_reshape(hidden_backward)
            hidden = th.cat([hidden, hidden_backward], dim=1)
        return hidden

    def _forward_pass_packed(self, input_seq, hidden, stack):
        """
        The Encoder Train Step.
        Args:
            input_seq (torch.nn.utls.rnn.PackedSequence): the sequence of
            indices for the input of shape.
        Note: Input_seq is an output of sequential_data_preparation(batch) with
            batches returned by a DataLoader object.
        Returns:
            (torch.Tensor, torch.Tensor): mu, logvar
            mu is the latent mean of shape `[1, batch_size, latent_dim]`.
            logvar is the log of the latent variance of shape
                `[1, batch_size, latent_dim]`.
        """
        if not isinstance(input_seq, nn.utils.rnn.PackedSequence):
            raise TypeError("Input is not PackedSequence")

        final_hidden = hidden.detach().clone()
        final_stack = stack.detach().clone()
        input_seq_packed, batch_sizes = utils.perpare_packed_input(input_seq)

        prev_batch = batch_sizes[0]

        for input_entry, batch_size in zip(input_seq_packed, batch_sizes):
            final_hidden, hidden = utils.manage_step_packed_vars(
                final_hidden, hidden, batch_size, prev_batch, batch_dim=1
            )
            final_stack, stack = utils.manage_step_packed_vars(
                final_stack, stack, batch_size, prev_batch, batch_dim=0
            )
            prev_batch = batch_size
            output, hidden, stack = self(
                input_entry.unsqueeze(0), hidden.contiguous(), stack
            )

        left_dims = hidden.shape[1]
        final_hidden[:, :left_dims, :] = hidden[:, :left_dims, :]
        final_stack[:left_dims, :, :] = stack[:left_dims, :, :]

        hidden = final_hidden
        stack = final_stack
        hidden = self._post_gru_reshape(hidden)

        # Backward pass:
        if self.bidirectional:
            # assert len(input_seq.shape) == 2, 'Input Seq must be 2D Tensor.'
            hidden_backward = self.backward_stackgru.init_hidden
            stack_backward = self.backward_stackgru.init_stack

            input_seq = utils.unpack_sequence(input_seq)

            for i, seq in enumerate(input_seq):
                idx = [i for i in range(len(seq) - 1, -1, -1)]
                idx = th.LongTensor(idx)
                input_seq[i] = seq.index_select(0, idx)

            input_seq = utils.repack_sequence(input_seq)
            input_seq_packed, batch_sizes = utils.perpare_packed_input(input_seq)

            final_hidden = hidden_backward.detach().clone()
            prev_batch = batch_sizes[0]

            for input_entry, batch_size in zip(input_seq_packed, batch_sizes):
                # for seq in input_seq:
                final_hidden, hidden_backward = utils.manage_step_packed_vars(
                    final_hidden, hidden_backward, batch_size, prev_batch, batch_dim=1
                )
                final_stack, stack_backward = utils.manage_step_packed_vars(
                    final_stack, stack_backward, batch_size, prev_batch, batch_dim=0
                )
                prev_batch = batch_size
                (
                    output_backward,
                    hidden_backward,
                    stack_backward,
                ) = self.backward_stackgru(
                    input_entry.unsqueeze(0), hidden_backward, stack_backward
                )
            left_dims = hidden_backward.shape[1]
            final_hidden[:, :left_dims, :] = hidden_backward[:, :left_dims, :]
            hidden_backward = final_hidden

            # Concatenate forward and backward
            hidden_backward = self._post_gru_reshape(hidden_backward)
            hidden = th.cat([hidden, hidden_backward], dim=1)
        return hidden

    def _post_gru_reshape(self, hidden: th.Tensor) -> th.Tensor:

        if not th.equal(th.tensor(hidden.shape), self.expected_shape):
            raise ValueError(
                f"GRU hidden layer has incorrect shape: {hidden.shape}. "
                f"Expected shape: {self.expected_shape}"
            )

        # Layers x Batch x Cell_size ->  B x C
        hidden = hidden[-1, :, :]

        return hidden
