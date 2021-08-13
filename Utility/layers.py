"""Custom layers implementation."""
import math
from typing import Callable
from collections import OrderedDict

import dgl
import dgl.function as fn
from dgl.ops import edge_softmax
from dgl.backend.pytorch.sparse import _gsddmm, _gspmm, gspmm, gsddmm
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from .utils import Squeeze, Unsqueeze, Temperature


def collapsing_layer(inputs, act_fn=nn.ReLU()):
    """Helper layer that collapses the last dimension of inputs.

    E.g. if cell line data includes CNV, inputs can be `[bs, 2048, 5]`
        and the outputs would be `[bs, 2048]`.

    Args:
        inputs (th.Tensor): of shape
            `[batch_size, *feature_sizes, hidden_size]`
        act_fn (callable): Nonlinearity to be used for collapsing.

    Returns:
        th.Tensor: Collapsed input of shape `[batch_size, *feature_sizes]`
    """
    collapse = nn.Sequential(
        OrderedDict(
            [
                ("dense", nn.Linear(inputs.shape[-1], 1)),
                ("act_fn", act_fn),
                ("squeeze", Squeeze()),
            ]
        )
    )
    return collapse(inputs)


def apply_dense_attention_layer(inputs, return_alphas=False):
    """Attention mechanism layer for dense inputs.

    Args:
        inputs (th.Tensor): Data input either of shape
            `[batch_size, feature_size]` or
            `[batch_size, feature_size, hidden_size]`.
        return_alphas (bool): Whether to return attention coefficients variable
            along with layer's output. Used for visualization purpose.
    Returns:
        th.Tensor or tuple(th.Tensor, th.Tensor):
            The tuple (outputs, alphas) if `return_alphas`
            else -by default- only outputs.
            Outputs are of shape `[batch_size, feature_size]`.
            Alphas are of shape `[batch_size, feature_size]`.
    """

    # If inputs have a hidden dimension, collapse them into a scalar
    inputs = collapsing_layer(inputs) if len(inputs.shape) == 3 else inputs
    assert len(inputs.shape) == 2

    attention_layer = nn.Sequential(
        OrderedDict(
            [
                ("dense", nn.Linear(inputs.shape[1], inputs.shape[1])),
                ("softmax", nn.Softmax()),
            ]
        )
    )
    alphas = attention_layer(inputs)
    output = alphas * inputs

    return (output, alphas) if return_alphas else output


def dense_layer(
    input_size, hidden_size, act_fn=nn.ReLU(), batch_norm=False, dropout=0.0
):
    return nn.Sequential(
        OrderedDict(
            [
                ("projection", nn.Linear(input_size, hidden_size)),
                (
                    "batch_norm",
                    nn.BatchNorm1d(hidden_size) if batch_norm else nn.Identity(),
                ),
                ("act_fn", act_fn),
                ("dropout", nn.Dropout(p=dropout)),
            ]
        )
    )


def dense_attention_layer(
    number_of_features: int, temperature: float = 1.0, dropout=0.0
) -> nn.Sequential:
    """Attention mechanism layer for dense inputs.

    Args:
        number_of_features (int): Size to allocate weight matrix.
        temperature (float): Softmax temperature parameter (0, inf). Lower
            temperature (< 1) result in a more descriminative/spiky softmax,
            higher temperature (> 1) results in a smoother attention.
    Returns:
        callable: a function that can be called with inputs.
    """
    return nn.Sequential(
        OrderedDict(
            [
                ("dense", nn.Linear(number_of_features, number_of_features)),
                ("dropout", nn.Dropout(p=dropout)),
                ("temperature", Temperature(temperature)),
                ("softmax", nn.Softmax(dim=-1)),
            ]
        )
    )


def convolutional_layer(
    num_kernel,
    kernel_size,
    act_fn=nn.ReLU(),
    batch_norm=False,
    dropout=0.0,
    input_channels=1,
):
    """Convolutional layer.

    Args:
        num_kernel (int): Number of convolution kernels.
        kernel_size (tuple[int, int]): Size of the convolution kernels.
        act_fn (callable): Functional of the nonlinear activation.
        batch_norm (bool): whether batch normalization is applied.
        dropout (float): Probability for each input value to be 0.
        input_channels (int): Number of input channels (defaults to 1).

    Returns:
        callable: a function that can be called with inputs.
    """
    return nn.Sequential(
        OrderedDict(
            [
                (
                    "convolve",
                    th.nn.Conv2d(
                        input_channels,  # channel_in
                        num_kernel,  # channel_out
                        kernel_size,  # kernel_size
                        padding=[kernel_size[0] // 2, 0],  # pad for valid conv.
                    ),
                ),
                ("squeeze", Squeeze()),
                ("act_fn", act_fn),
                ("dropout", nn.Dropout(p=dropout)),
                (
                    "batch_norm",
                    nn.BatchNorm1d(num_kernel) if batch_norm else nn.Identity(),
                ),
            ]
        )
    )


def gene_projection(num_genes, attention_size, ind_nonlin=nn.Sequential()):
    return nn.Sequential(
        OrderedDict(
            [
                ("projection", nn.Linear(num_genes, attention_size)),
                ("act_fn", ind_nonlin),
                ("expand", Unsqueeze(1)),
            ]
        )
    )


def smiles_projection(smiles_hidden_size, attention_size, ind_nonlin=nn.Sequential()):
    return nn.Sequential(
        OrderedDict(
            [
                ("projection", nn.Linear(smiles_hidden_size, attention_size)),
                ("act_fn", ind_nonlin),
            ]
        )
    )


def alpha_projection(attention_size):
    return nn.Sequential(
        OrderedDict(
            [
                ("projection", nn.Linear(attention_size, 1, bias=False)),
                ("squeeze", Squeeze()),
                ("softmax", nn.Softmax(dim=1)),
            ]
        )
    )


class ContextAttentionLayer(nn.Module):
    """
    Implements context attention as in the PaccMann paper (Figure 2C) in
    Molecular Pharmaceutics.
    With the additional option of having a hidden size in the context.
    NOTE:
    In tensorflow, weights were initialized from N(0,0.1). Instead, pytorch
    uses U(-stddev, stddev) where stddev=1./math.sqrt(weight.size(1)).
    """

    def __init__(
        self,
        reference_hidden_size: int,
        reference_sequence_length: int,
        context_hidden_size: int,
        context_sequence_length: int = 1,
        attention_size: int = 16,
        individual_nonlinearity: type = nn.Sequential(),
    ):
        """Constructor
        Arguments:
            reference_hidden_size (int): Hidden size of the reference input
                over which the attention will be computed (H).
            reference_sequence_length (int): Sequence length of the reference
                (T).
            context_hidden_size (int): This is either simply the amount of
                features used as context (G) or, if the context is a sequence
                itself, the hidden size of each time point.
            context_sequence_length (int): Hidden size in the context, useful
                if context is also textual data, i.e. coming from nn.Embedding.
                Defaults to 1.
            attention_size (int): Hyperparameter of the attention layer,
                defaults to 16.
            individual_nonlinearities (type): This is an optional
                nonlinearity applied to each projection. Defaults to
                nn.Sequential(), i.e. no nonlinearity. Otherwise it expects a
                nn activation function, e.g. nn.ReLU().
        """
        super().__init__()

        self.reference_sequence_length = reference_sequence_length
        self.reference_hidden_size = reference_hidden_size
        self.context_sequence_length = context_sequence_length
        self.context_hidden_size = context_hidden_size
        self.attention_size = attention_size
        self.individual_nonlinearity = individual_nonlinearity

        # Project the reference into the attention space
        self.reference_projection = nn.Sequential(
            OrderedDict(
                [
                    ("projection", nn.Linear(reference_hidden_size, attention_size)),
                    ("act_fn", individual_nonlinearity),
                ]
            )
        )  # yapf: disable

        # Project the context into the attention space
        self.context_projection = nn.Sequential(
            OrderedDict(
                [
                    ("projection", nn.Linear(context_hidden_size, attention_size)),
                    ("act_fn", individual_nonlinearity),
                ]
            )
        )  # yapf: disable

        # Optionally reduce the hidden size in context
        if context_sequence_length > 1:
            self.context_hidden_projection = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "projection",
                            nn.Linear(
                                context_sequence_length, reference_sequence_length
                            ),
                        ),
                        ("act_fn", individual_nonlinearity),
                    ]
                )
            )  # yapf: disable
        else:
            self.context_hidden_projection = nn.Sequential()

        self.alpha_projection = nn.Sequential(
            OrderedDict(
                [
                    ("projection", nn.Linear(attention_size, 1, bias=False)),
                    ("squeeze", Squeeze()),
                    ("softmax", nn.Softmax(dim=1)),
                ]
            )
        )

    def forward(self, reference: th.Tensor, context: th.Tensor):
        """
        Forward pass through a context attention layer
        Arguments:
            reference (th.Tensor): This is the reference input on which
                attention is computed.
                Shape: batch_size x ref_seq_length x ref_hidden_size
            context (th.Tensor): This is the context used for attention.
                Shape: batch_size x context_seq_length x context_hidden_size
        Returns:
            (output, attention_weights):  A tuple of two Tensors, first one
                containing the reference filtered by attention (shape:
                batch_size x context_hidden_size x 1) and the second one the
                attention weights (batch_size x context_sequence_length x 1).
        """
        assert len(reference.shape) == 3, "Reference tensor needs to be 3D"
        assert len(context.shape) == 3, "Context tensor needs to be 3D"

        reference_attention = self.reference_projection(reference)
        context_attention = self.context_hidden_projection(
            self.context_projection(context).permute(0, 2, 1)
        ).permute(0, 2, 1)
        alphas = self.alpha_projection(th.tanh(reference_attention + context_attention))
        output = th.sum(reference * th.unsqueeze(alphas, -1), 1)

        return output, alphas


class EnsembleLayer(nn.Module):
    """
    Following Lee at al (2015) we implement probability and score averaging
    model ensembles.
    """

    def __init__(self, typ, input_size, output_size, ensemble_size=5, fn=nn.ReLU()):
        """
        Args:
            typ    {str} from {'pron', 'score'} depending on whether the
                ensemble includes the activation function ('prob').
            input_size  {int} amount of input neurons
            output_size {int} amount of output neurons (# tasks/classes)
            ensemble_size {int} amount of parallel ensemble learners
            act_fn      {int} activation function used

        """
        super(EnsembleLayer, self).__init__()

        self.type = typ
        self.input_size = input_size
        self.output_size = output_size
        self.ensemble_size = ensemble_size
        self.act_fn = fn

        if typ == "prob":
            self.ensemble = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(input_size, output_size), fn)
                    for _ in range(ensemble_size)
                ]
            )

        elif typ == "score":
            self.ensemble = nn.ModuleList(
                [nn.Linear(input_size, output_size) for _ in range(ensemble_size)]
            )

        else:
            raise NotImplementedError("Choose type from {'score', 'prob'}")

    def forward(self, x):
        """Run forward pass through model ensemble

        Arguments:
            x {th.Tensor} -- shape: batch_size x input_size

        Returns:
            th.Tensor -- shape: batch_size x output_size
        """

        dist = [e(x) for e in self.ensemble]
        output = th.mean(th.stack(dist), dim=0)
        if self.type == "score":
            output = self.act_fn(output)

        return output


class GraphGather(nn.Module):
    """
    GGNN readout function.
    """

    def __init__(
        self,
        node_features: int,
        hidden_node_features: int,
        out_features: int,
        att_depth: int,
        att_hidden_dim: int,
        att_dropout_p: float,
        emb_depth: int,
        emb_hidden_dim: int,
        emb_dropout_p: float,
        big_positive: float,
    ) -> None:

        super().__init__()

        self.big_positive = big_positive

        self.att_nn = MLP(
            in_features=node_features + hidden_node_features,
            hidden_layer_sizes=[att_hidden_dim] * att_depth,
            out_features=out_features,
            dropout_p=att_dropout_p,
        )

        self.emb_nn = MLP(
            in_features=hidden_node_features,
            hidden_layer_sizes=[emb_hidden_dim] * emb_depth,
            out_features=out_features,
            dropout_p=emb_dropout_p,
        )

    def forward(
        self,
        hidden_nodes: th.Tensor,
        input_nodes: th.Tensor,
        node_mask: th.Tensor,
    ) -> th.Tensor:
        """
        Defines forward pass.
        """
        Softmax = nn.Softmax(dim=1)

        cat = th.cat((hidden_nodes, input_nodes), dim=2)
        energy_mask = (node_mask == 0).float() * self.big_positive
        energies = self.att_nn(cat) - energy_mask.unsqueeze(-1)
        attention = Softmax(energies)
        embedding = self.emb_nn(hidden_nodes)

        return th.sum(attention * embedding, dim=1)


class Set2Vec(nn.Module):
    """
    S2V readout function.
    """

    def __init__(
        self,
        node_features: int,
        hidden_node_features: int,
        lstm_computations: int,
        memory_size: int,
    ) -> None:

        super().__init__()

        self.lstm_computations = lstm_computations
        self.memory_size = memory_size
        self.embedding_matrix = nn.Linear(
            in_features=node_features + hidden_node_features,
            out_features=self.memory_size,
            bias=True,
        )
        self.lstm = nn.LSTMCell(
            input_size=self.memory_size, hidden_size=self.memory_size, bias=True
        )

    def forward(
        self,
        hidden_output_nodes: th.Tensor,
        input_nodes: th.Tensor,
        node_mask: th.Tensor,
    ) -> th.Tensor:
        """
        Defines forward pass.
        """
        softmax = th.nn.Softmax(dim=1)

        batch_size = input_nodes.shape[0]
        energy_mask = th.bitwise_not(node_mask).float() * self.C.big_negative

        lstm_input = th.zeros(batch_size, self.memory_size)

        cat = th.cat((hidden_output_nodes, input_nodes), dim=2)
        memory = self.embedding_matrix(cat)

        hidden_state = th.zeros(batch_size, self.memory_size)
        cell_state = th.zeros(batch_size, self.memory_size)

        for _ in range(self.lstm_computations):
            query, cell_state = self.lstm(lstm_input, (hidden_state, cell_state))

            # dot product query x memory
            energies = (query.view(batch_size, 1, self.memory_size) * memory).sum(
                dim=-1
            )
            attention = softmax(energies + energy_mask)
            read = (attention.unsqueeze(-1) * memory).sum(dim=1)

            hidden_state = query
            lstm_input = read

        cat = th.cat((query, read), dim=1)
        return cat


class MLP(nn.Module):
    """
    Multi-layer perceptron. Applies SELU after every linear layer.

    Args:
    ----
        in_features (int) : Size of each input sample.
        hidden_layer_sizes (list) : Hidden layer sizes.
        out_features (int) : Size of each output sample.
        dropout_p (float) : Probability of dropping a weight.
    """

    def __init__(
        self,
        in_features: int,
        hidden_layer_sizes: list,
        out_features: int,
        dropout_p: float,
    ) -> None:
        super().__init__()

        activation_function = nn.SELU

        # create list of all layer feature sizes
        fs = [in_features, *hidden_layer_sizes, out_features]

        # create list of linear_blocks
        layers = [
            self._linear_block(in_f, out_f, activation_function, dropout_p)
            for in_f, out_f in zip(fs, fs[1:])
        ]

        # concatenate modules in all sequentials in layers list
        layers = [module for sq in layers for module in sq.children()]

        # add modules to sequential container
        self.seq = nn.Sequential(*layers)

    def _linear_block(
        self, in_f: int, out_f: int, activation: nn.Module, dropout_p: float
    ) -> nn.Sequential:
        """
        Returns a linear block consisting of a linear layer, an activation function (SELU),
        and dropout (optional) stack.

        Args:
        ----
            in_f (int) : Size of each input sample.
            out_f (int) : Size of each output sample.
            activation (nn.Module) : Activation function.
            dropout_p (float) : Probability of dropping a weight.
        """
        # bias must be used in most MLPs in our models to learn from empty graphs
        linear = nn.Linear(in_f, out_f, bias=True)
        nn.init.xavier_uniform_(linear.weight)
        return nn.Sequential(linear, activation(), nn.AlphaDropout(dropout_p))

    def forward(self, layers_input: nn.Sequential) -> nn.Sequential:
        """
        Defines forward pass.
        """
        return self.seq(layers_input)


class GlobalReadout(nn.Module):
    """
    Global readout function class. Used to predict the action probability distributions (APDs)
    for molecular graphs.

    The first tier of two `MLP`s take as input, for each graph in the batch, the final transformed
    node feature vectors. These feed-forward networks correspond to the preliminary "f_add" and
    "f_conn" distributions.

    The second tier of three `MLP`s takes as input the output of the first tier of `MLP`s (the
    "preliminary" APDs) as well as the graph embeddings for all graphs in the batch. Output are
    the final APD components, which are then flattened and concatenated. No activation function
    is applied after the final layer, so that this can be done outside (e.g. in the loss function,
    and before sampling).
    """

    def __init__(
        self,
        f_add_elems: int,
        f_conn_elems: int,
        f_term_elems: int,
        mlp1_depth: int,
        mlp1_dropout_p: float,
        mlp1_hidden_dim: int,
        mlp2_depth: int,
        mlp2_dropout_p: float,
        mlp2_hidden_dim: int,
        graph_emb_size: int,
        max_n_nodes: int,
        node_emb_size: int,
    ) -> None:
        super().__init__()

        # preliminary f_add
        self.fAddNet1 = MLP(
            in_features=node_emb_size,
            hidden_layer_sizes=[mlp1_hidden_dim] * mlp1_depth,
            out_features=f_add_elems,
            dropout_p=mlp1_dropout_p,
        )

        # preliminary f_conn
        self.fConnNet1 = MLP(
            in_features=node_emb_size,
            hidden_layer_sizes=[mlp1_hidden_dim] * mlp1_depth,
            out_features=f_conn_elems,
            dropout_p=mlp1_dropout_p,
        )

        # final f_add
        self.fAddNet2 = MLP(
            in_features=(max_n_nodes * f_add_elems + graph_emb_size),
            hidden_layer_sizes=[mlp2_hidden_dim] * mlp2_depth,
            out_features=f_add_elems * max_n_nodes,
            dropout_p=mlp2_dropout_p,
        )

        # final f_conn
        self.fConnNet2 = MLP(
            in_features=(max_n_nodes * f_conn_elems + graph_emb_size),
            hidden_layer_sizes=[mlp2_hidden_dim] * mlp2_depth,
            out_features=f_conn_elems * max_n_nodes,
            dropout_p=mlp2_dropout_p,
        )

        # final f_term (only takes as input graph embeddings)
        self.fTermNet2 = MLP(
            in_features=graph_emb_size,
            hidden_layer_sizes=[mlp2_hidden_dim] * mlp2_depth,
            out_features=f_term_elems,
            dropout_p=mlp2_dropout_p,
        )

    def forward(
        self, node_level_output: th.Tensor, graph_embedding_batch: th.Tensor
    ) -> th.Tensor:
        """
        Defines forward pass.
        """
        # get preliminary f_add and f_conn
        f_add_1 = self.fAddNet1(node_level_output)
        f_conn_1 = self.fConnNet1(node_level_output)

        # reshape preliminary APDs into flattenened vectors (e.g. one vector per graph in batch)
        f_add_1_size = f_add_1.size()
        f_conn_1_size = f_conn_1.size()
        f_add_1 = f_add_1.view((f_add_1_size[0], f_add_1_size[1] * f_add_1_size[2]))
        f_conn_1 = f_conn_1.view(
            (f_conn_1_size[0], f_conn_1_size[1] * f_conn_1_size[2])
        )

        # get final f_add, f_conn, and f_term
        f_add_2 = self.fAddNet2(
            th.cat((f_add_1, graph_embedding_batch), dim=1).unsqueeze(dim=1)
        )
        f_conn_2 = self.fConnNet2(
            th.cat((f_conn_1, graph_embedding_batch), dim=1).unsqueeze(dim=1)
        )
        f_term_2 = self.fTermNet2(graph_embedding_batch)

        # flatten and concatenate
        cat = th.cat((f_add_2.squeeze(dim=1), f_conn_2.squeeze(dim=1), f_term_2), dim=1)

        return cat  # note: no activation function before returning


class GraphNorm(nn.Module):
    def __init__(self, norm_type, hidden_dim=300, print_info=None):
        super(GraphNorm, self).__init__()
        assert norm_type in ["bn", "gn", "ln", None]
        self.norm = None
        self.print_info = print_info
        if norm_type == "bn":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "ln":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm_type == "gn":
            self.norm = norm_type
            self.gnw = nn.Parameter(th.ones(hidden_dim))
            self.gnb = nn.Parameter(th.zeros(hidden_dim))

    def forward(self, x, batch_num_nodes):
        if self.norm is not None and type(self.norm) != str:
            return self.norm(x)
        elif self.norm is None:
            return x
        batch_size = len(batch_num_nodes)
        batch_list = th.as_tensor(batch_num_nodes, dtype=th.long, device=x.device)
        batch_index = th.arange(batch_size, device=x.device).repeat_interleave(
            batch_list
        )
        batch_index = batch_index.view((-1,) + (1,) * (x.dim() - 1)).expand_as(x)
        mean = th.zeros(batch_size, *x.shape[1:], device=x.device)
        mean = mean.scatter_add_(0, batch_index, x)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)
        sub = x - mean
        std = th.zeros(batch_size, *x.shape[1:], device=x.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        # return sub / std
        return self.gnw * sub / std + self.gnb


class GraphNormAndProj(nn.Module):
    def __init__(self, d_in, d_out, act, dropout, norm_type):
        super(GraphNormAndProj, self).__init__()
        self.norm = GraphNorm(norm_type, d_in)
        self.activation = act()
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_in, d_out)

    def forward(self, tensor, batch_num_graphs):
        x = self.norm(tensor, batch_num_graphs)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out(x)
        return x


class GraphNetBlock(nn.Module):
    def __init__(
        self,
        d_in: int = 256,
        qk_dim: int = 64,
        v_dim: int = 64,
        n_heads: int = 8,
        dropout: float = 0.2,
        attn_weight_norm="norm",
        act: Callable = nn.ReLU,
        norm_type="gn",
    ):
        """
        Initialize a multi-headed attention block compatible with DGLGraph
        inputs. Given a fully connected input graph with self loops,
        is analogous to original Transformer.

        Args:
            d_in: input dimension
            qk_dim: head dimension
            n_heads: number of heads
            dropout: dropout probability
            attn_weight_norm: attention pooling method, 'norm' or 'softmax'
        """
        super().__init__()
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.n_heads = n_heads
        self.d_hidden = n_heads * v_dim

        self.attn_weight_norm = {
            "norm": EdgeNormWithGainAndBias(n_heads),
            "softmax": edge_softmax,
        }[attn_weight_norm]
        self.attn_dropout = nn.Dropout(dropout)

        def pwff():
            return nn.Sequential(
                ActGLU(d_in, d_in * 2, act),
                nn.Dropout(dropout),
                nn.Linear(d_in * 2, d_in),
            )

        self.node_rezero = ReZero()
        self.edge_rezero = ReZero()
        self.node_ff = pwff()
        self.edge_ff = pwff()
        self.node_ff2 = pwff()
        self.edge_ff2 = pwff()

        self.q_proj = nn.Linear(d_in, self.qk_dim * self.n_heads)
        self.k_proj = nn.Linear(d_in, self.qk_dim * self.n_heads)
        self.v_proj = nn.Linear(d_in, self.v_dim * self.n_heads)
        self.eq_proj = nn.Linear(d_in, self.qk_dim * self.n_heads)
        self.ek_proj = nn.Linear(d_in, self.qk_dim * self.n_heads)
        self.ev_proj = nn.Linear(d_in, self.v_dim * self.n_heads)

        self.mix_nodes = GraphNormAndProj(
            n_heads * v_dim, d_in, act, dropout, norm_type
        )

    def forward(self, g: dgl.DGLGraph, n: th.Tensor, e: th.Tensor):
        # convection
        n = n + self.node_rezero(self.node_ff(n))
        e = e + self.edge_rezero(self.edge_ff(e))

        # diffusion (attn)
        q = rearrange(self.q_proj(n), "b (h qk) -> b h qk", h=self.n_heads)
        k = rearrange(self.k_proj(n), "b (h qk) -> b h qk", h=self.n_heads)
        v = rearrange(self.v_proj(n), "b (h v) -> b h v", h=self.n_heads)
        eq = rearrange(self.eq_proj(e), "b (h qk) -> b h qk", h=self.n_heads)
        ek = rearrange(self.ek_proj(e), "b (h qk) -> b h qk", h=self.n_heads)
        ev = rearrange(self.ev_proj(e), "b (h v) -> b h v", h=self.n_heads)

        g.ndata.update({"q": q, "k": k, "v": v})
        g.edata.update({"eq": eq, "ek": ek, "ev": ev})

        g.apply_edges(fn.v_dot_u("q", "k", "n2n"))  # n2n
        g.apply_edges(fn.v_dot_e("q", "ek", "n2e"))  # n2e
        g.apply_edges(fn.e_dot_u("eq", "k", "e2n"))  # e2n
        if self.attn_weight_norm == "softmax":
            scale = math.sqrt(self.qk_dim)
            g.edata["n2n"] /= scale
            g.edata["n2e"] /= scale
            g.edata["e2n"] /= scale
        n2n_attn = self.attn_dropout(self.attn_weight_norm(g, g.edata["n2n"]))
        n2e_attn = self.attn_dropout(self.attn_weight_norm(g, g.edata["n2e"]))
        e2n_attn = self.attn_dropout(self.attn_weight_norm(g, g.edata["n2e"]))

        # aggregate normalized weighted values per node
        g.apply_edges(
            lambda edge: {
                "wv": n2n_attn * edge.src["v"]
                + n2e_attn * edge.data["ev"]
                + e2n_attn * edge.src["v"]
            }
        )
        g.update_all(fn.copy_e("wv", "wv"), fn.sum("wv", "z"))

        n = n + self.node_rezero(
            self.mix_nodes(g.ndata["z"].view(-1, self.d_hidden), g.batch_num_nodes())
        )

        # convection
        n = n + self.node_rezero(self.node_ff2(n))
        e = e + self.edge_rezero(self.edge_ff2(e))

        return g, n, e


class GraphNet(nn.Module):
    def __init__(
        self,
        features=256,
        qk_dim=32,
        v_dim=64,
        n_layers=8,
        n_heads=8,
        dropout=0.2,
        attn_weight_norm="norm",
        act=nn.ReLU,
        norm_type="gn",
        pool_type="deepset",
    ):
        super(GraphNet, self).__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.attn_layers = nn.ModuleList()
        self.atom_enc = AtomEncoder(features)
        self.bond_enc = BondEncoder(features)

        for _ in range(n_layers):
            self.attn_layers.append(
                GraphNetBlock(
                    d_in=features,
                    qk_dim=qk_dim,
                    v_dim=v_dim,
                    dropout=dropout,
                    act=act,
                    norm_type=norm_type,
                    attn_weight_norm=attn_weight_norm,
                )
            )

        if pool_type == "deepset":
            self.readout = DeepSet(features, features, dropout=dropout)
        elif pool_type == "mean_max":
            self.readout = MeanMaxPool(features * 2)

    def forward(self, g: dgl.DGLGraph):
        n = self.atom_enc(g.ndata["feat"])
        e = self.bond_enc(g.edata["feat"])
        for i in range(self.n_layers):
            g, n, e = self.attn_layers[i](g, n, e)
        out = self.readout(g, n)
        return out


class ReZero(nn.Module):
    def __init__(self):
        super(ReZero, self).__init__()
        self.g = nn.Parameter(th.zeros(1))

    def forward(self, x):
        return x * self.g


class ActGLU(nn.Module):
    def __init__(self, d_in, d_out, act: Callable = nn.GELU):
        super(ActGLU, self).__init__()
        self.proj = nn.Linear(d_in, d_out * 2)
        self.act = act()

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.act(gate)


class MLP_IC(nn.Sequential):
    def __init__(self, *dims, norm=True, dropout=0.1, act=nn.ReLU):
        l = []
        for i in range(len(dims) - 2):
            l.extend(
                [
                    nn.Linear(dims[i], dims[i + 1]),
                    act(),
                    nn.BatchNorm1d(dims[i + 1]) if norm else nn.Identity(),
                    nn.Dropout(dropout),
                ]
            )
        l.append(nn.Linear(dims[-2], dims[-1]))
        super(MLP_IC, self).__init__(*l)


class DeepSet(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.1):
        super(DeepSet, self).__init__()
        self.glu = nn.Sequential(nn.Linear(d_in, d_in * 2), nn.GLU())
        self.agg = nn.Sequential(
            nn.BatchNorm1d(d_in), nn.Dropout(dropout), nn.Linear(d_in, d_out)
        )

    def forward(self, g, n):
        g.ndata["out"] = self.glu(n)
        readout = self.agg(dgl.readout_nodes(g, "out", op="sum"))

        return readout


class MeanMaxPool(nn.Module):
    def __init__(self, dim):
        super(MeanMaxPool, self).__init__()
        self.gain = nn.Parameter(th.ones(dim))
        self.bias = nn.Parameter(th.zeros(dim))

    def forward(self, g, n, key="out"):
        g.ndata[key] = n
        max = dgl.readout_nodes(g, key, op="max")
        mean = dgl.readout_nodes(g, key, op="mean")
        out = th.cat([max, mean], dim=-1)
        return out * self.gain + self.bias


@th.jit.script
def mish(input):
    return input * th.tanh(F.softplus(input))


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return mish(input)


def normalize(t: th.Tensor, eps=1e-8):
    """Normalize final dim"""
    t -= t.mean(dim=-1, keepdim=True)
    s = (t ** 2).mean(dim=-1, keepdim=True)
    return t * th.rsqrt(s + eps)


# noinspection PyMethodOverriding
class EdgeNorm(th.autograd.Function):
    r"""Apply normalization over signals of incoming edges.

    For a node :math:`i` of :math:`N`, head :math:`m`, EdgeNorm is an
    operation computing

    .. math::
        \textbf{\textit{a}}^i_m = \text{normalize}([l^{i,1}_m, ... , l^{i,N}_m])

        \text{normalize}(\textbf{\textit{x}})^j = g \cdot
        \frac{x^j - \mu_{x}}{\sigma_x} + b

    Adapted from dgl implementation of EdgeSoftmax:
    https://github.com/dmlc/dgl/blob/b36b6c268efb59b59046a74976067050141b1d6e/python/dgl/backend/pytorch/sparse.py#L180
    """

    @staticmethod
    def forward(ctx, gidx, score, eids=dgl.base.ALL):
        r"""
        Args:
            ctx: context to save cache intermediate values to
            gidx: graph index object
            score: edata shaped scores to normalize, first dimension should
                match length of eids
            eids: ids of edges to normalize

        Returns:
            edge score values normalized by destination node grouping.
        """
        # save graph to backward cache
        if not dgl.base.is_all(eids):
            gidx = gidx.edge_subgraph([eids], True).graph

        # graph statistics aggregated by target node: mu and stdev
        score_sums = _gspmm(gidx, "copy_rhs", "sum", None, score)[0]
        score_counts = _gspmm(gidx, "copy_rhs", "sum", None, th.ones_like(score))[0]
        means = score_sums / score_counts.clamp_min(1)
        residual = _gsddmm(gidx, "sub", score, means, "e", "v")
        var = th.pow(residual, 2)
        stdev = th.sqrt(_gspmm(gidx, "copy_rhs", "sum", None, var)[0] / score_counts)
        inv_stdev = 1.0 / stdev.clamp_min(1e-5)

        # rescale residuals
        normalized = _gsddmm(gidx, "mul", residual, inv_stdev, "e", "v")

        ctx.graph_cache = gidx  # cache non tensor obj
        ctx.save_for_backward(residual, inv_stdev)  # save tensors in ctx
        return normalized

    @staticmethod
    def backward(ctx, output_grad):
        """
        Args:
            ctx: cached context from forward pass
            output_grad: upstream derivatives, shape: (num_E,)

        Returns:
            As many elements as output_grad
        """
        gidx = ctx.graph_cache
        residual, inv_stdev = ctx.saved_tensors
        edge_counts = gspmm(gidx, "copy_rhs", "sum", None, th.ones_like(output_grad))
        dres = gsddmm(gidx, "mul", output_grad, inv_stdev, "e", "v")
        dinv_std = gspmm(gidx, "copy_rhs", "sum", None, output_grad * residual)
        dsqerr = dinv_std * -th.pow(inv_stdev, 2) * inv_stdev / 2 / edge_counts
        dres2 = 2 * gsddmm(gidx, "mul", residual, dsqerr, "e", "v")
        dresidual = -gspmm(gidx, "copy_rhs", "sum", None, dres + dres2)
        out = gsddmm(gidx, "add", dres + dres2, dresidual / edge_counts, "e", "v")

        return None, out, None


def edge_norm(gidx, scores, eids=dgl.base.ALL):
    return EdgeNorm.apply(gidx._graph, scores, eids)


class EdgeNormWithGainAndBias(nn.Module):
    """
    Edge normalization with gain and bias per head from Richter and
    Wattenhofer, 2020. https://arxiv.org/abs/2005.09561, adapted for graph
    input structures.
    """

    def __init__(self, nheads=1):
        super().__init__()
        # trainable gain and bias per head.
        self.gain = th.nn.Parameter(th.ones(nheads, 1))
        self.bias = th.nn.Parameter(th.zeros(nheads, 1))

    def forward(self, g, edge_scores, eids=dgl.base.ALL):
        eweights = edge_norm(g, edge_scores, eids)
        return self.gain * eweights + self.bias
