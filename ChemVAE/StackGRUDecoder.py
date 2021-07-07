"""Model Classes Module."""
from itertools import takewhile

import torch as th
import torch.nn as nn
from torch.autograd import Variable

from Utility import utils
from Utility.search import BeamSearch, SamplingSearch
from StackGRU import StackGRU


class StackGRUDecoder(StackGRU):
    """Stack GRU Decoder."""

    def __init__(self, params, *args, **kwargs):
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
        super(StackGRUDecoder, self).__init__(params, *args, **kwargs)
        self.params = params
        self.latent_dim = params["latent_dim"]
        self.latent_to_hidden = nn.Linear(
            in_features=self.latent_dim, out_features=self.rnn_cell_size
        )
        self.output_layer = nn.Linear(self.rnn_cell_size, self.vocab_size)
        self.criterion = nn.CrossEntropyLoss()

    def _forward_pass_padded(self, input_seq, target_seq, hidden, stack):
        """The Decoder Train Step.
        Note: Input and target sequences are outputs of
            sequential_data_preparation(batch) with batches returned by
            a DataLoader object.
        Returns:
            The cross-entropy training loss for the decoder.
        """
        if isinstance(input_seq, nn.utils.rnn.PackedSequence) and not isinstance(
            input_seq, th.Tensor
        ):
            raise TypeError("Input is PackedSequence or is not a Tensor")

        loss = 0
        outputs = []
        for idx, (input_entry, target_entry) in enumerate(zip(input_seq, target_seq)):
            output, hidden, stack = self(input_entry.unsqueeze(0), hidden, stack)
            output = self.output_layer(output).squeeze(dim=0)
            loss += self.criterion(output, target_entry)
            outputs.append(output)

        # For monitoring purposes
        outputs = th.stack(outputs, -1)
        self.outputs = th.argmax(outputs, 1)

        return loss

    def _forward_pass_packed(self, input_seq, target_seq, hidden, stack):
        """The Decoder Train Step.
        Note: Input and target sequences are outputs of
            sequential_data_preparation(batch) with batches returned by a
            DataLoader object.
        Returns:
            The cross-entropy training loss for the decoder.
        """
        if not isinstance(input_seq, nn.utils.rnn.PackedSequence):
            raise TypeError("Input is not PackedSequence")

        loss = 0
        input_seq_packed, batch_sizes = utils.perpare_packed_input(input_seq)
        # Target sequence should have same batch_sizes as input_seq
        target_seq_packed, _ = utils.perpare_packed_input(target_seq)
        prev_batch = batch_sizes[0]
        outputs = []
        for idx, (input_entry, target_entry, batch_size) in enumerate(
            zip(input_seq_packed, target_seq_packed, batch_sizes)
        ):
            _, hidden = utils.manage_step_packed_vars(
                None, hidden, batch_size, prev_batch, batch_dim=1
            )
            _, stack = utils.manage_step_packed_vars(
                None, stack, batch_size, prev_batch, batch_dim=0
            )

            prev_batch = batch_size
            output, hidden, stack = self(
                input_entry.unsqueeze(0), hidden.contiguous(), stack
            )
            output = self.output_layer(output).squeeze(dim=0)

            loss += self.criterion(output, target_entry)
            outputs.append(th.argmax(output, -1))
        self.outputs = utils.packed_to_padded(outputs, target_seq_packed)
        return loss

    def generate(
        self, latent_z, prime_input, end_token, generate_len=100, search=SamplingSearch
    ):
        """
        Generate SMILES From Latent Z.
        Args:
            latent_z (torch.Tensor): The sampled latent representation
                of size `[1, batch_size, latent_dim]`.
            prime_input (torch.Tensor): Tensor of indices for the priming
                string. Must be of size [prime_input length].
                Example: `prime_input = torch.tensor([2, 4, 5])`
            end_token (torch.Tensor): End token for the generated molecule
                of shape [1].
                Example: `end_token = torch.LongTensor([3])`
            search (paccmann_chemistry.utils.search.Search): search strategy
                used in the decoder.
            generate_len (int): Length of the generated molecule.
        Returns:
            torch.Tensor: The sequence(s) for the generated molecule(s)
                of shape [batch_size, generate_len + len(prime_input)].
        Note: For each generated sequence all indices after the first
            end_token must be discarded.
        """
        self.batch_size = latent_z.shape[1]
        self.expected_shape = th.tensor(
            [self.n_layers, self.batch_size, self.rnn_cell_size]
        )
        # Variable to initialize hidden state and stack
        self.init_hidden = Variable(
            th.zeros(self.n_layers, self.batch_size, self.rnn_cell_size).to(
                latent_z.device
            )
        )
        self.init_stack = Variable(
            th.zeros(self.batch_size, self.stack_depth, self.stack_width).to(
                latent_z.device
            )
        )
        latent_z = latent_z.repeat(self.n_layers, 1, 1)
        hidden = self.latent_to_hidden(latent_z)
        stack = self.init_stack

        generated_seq = prime_input.repeat(self.batch_size, 1)
        prime_input = generated_seq.transpose(1, 0).unsqueeze(1)

        # use priming string to "build up" hidden state
        for prime_entry in prime_input[:-1]:
            _, hidden, stack = self(prime_entry, hidden, stack)
        input_token = prime_input[-1]

        # initialize beam search
        is_beam = isinstance(search, BeamSearch)
        if is_beam:
            beams = [[[list(), 0.0]]] * self.batch_size
            input_token = th.stack(
                [input_token]
                + [input_token.clone() for _ in range(search.beam_width - 1)]
            )
            hidden = th.stack(
                [hidden] + [hidden.clone() for _ in range(search.beam_width - 1)]
            )
            stack = th.stack(
                [stack] + [stack.clone() for _ in range(search.beam_width - 1)]
            )

        for idx in range(generate_len):
            if not is_beam:
                output, hidden, stack = self(input_token, hidden, stack)
                logits = self.output_layer(output).squeeze(dim=0)
                top_idx = search.step(logits)
                input_token = top_idx.view(1, -1).to(latent_z.device)
                generated_seq = th.cat((generated_seq, top_idx), dim=1)

                # if we don't generate in batches, we can do early stopping.
                if self.batch_size == 1 and top_idx == end_token:
                    break
            else:
                output, hidden, stack = zip(
                    *[
                        self(an_input_token, a_hidden, a_stack)
                        for an_input_token, a_hidden, a_stack in zip(
                            input_token, hidden, stack
                        )
                    ]
                )
                logits = th.stack([self.output_layer(o).squeeze() for o in output])
                hidden = th.stack(hidden)
                stack = th.stack(stack)
                input_token, beams = search.step(logits, beams)
                input_token = input_token.unsqueeze(1)

        if is_beam:
            generated_seq = th.stack(
                [
                    # get the list of tokens with the highest score
                    th.tensor(beam[0][0])
                    for beam in beams
                ]
            )

        molecule_gen = (
            takewhile(lambda x: x != end_token, molecule[1:])
            for molecule in generated_seq
        )
        molecule_map = map(list, molecule_gen)
        molecule_iter = iter(map(th.tensor, molecule_map))

        return molecule_iter
