# load general packages and functions
import math
import torch as th
import torch.nn as nn

# load GraphINVENT-specific functions
from Utility.mpnns import AggregationMPNN, SummationMPNN, EdgeMPNN
from Utility.layers import GraphGather, Set2Vec, MLP, GlobalReadout


# defines specific MPNN implementations
class MNN(SummationMPNN):
    """
    The "message neural network" model.
    """

    def __init__(self) -> None:
        super(MNN, self).__init__()

        mlp1_hidden_dim = self.params.get("mlp1_hidden_dim", 500)
        mlp1_depth = self.params.get("mlp1_depth", 4)
        mlp1_dropout_p = self.params.get("mlp1_dropout_p", 0.0)
        mlp2_hidden_dim = self.params.get("mlp2_hidden_dim", 500)
        mlp2_depth = self.params.get("mlp2_depth", 4)
        mlp2_dropout_p = self.params.get("mlp2_dropout_p", 0.0)

        message_weights = th.Tensor(
            self.message_size,
            self.hidden_node_features,
            self.edge_features,
        )
        self.message_weights = nn.Parameter(message_weights)
        self.gru = nn.GRUCell(
            input_size=self.message_size,
            hidden_size=self.hidden_node_features,
            bias=True,
        )
        self.APDReadout = GlobalReadout(
            node_emb_size=self.hidden_node_features,
            graph_emb_size=self.hidden_node_features,
            mlp1_hidden_dim=mlp1_hidden_dim,
            mlp1_depth=mlp1_depth,
            mlp1_dropout_p=mlp1_dropout_p,
            mlp2_hidden_dim=mlp2_hidden_dim,
            mlp2_depth=mlp2_depth,
            mlp2_dropout_p=mlp2_dropout_p,
            f_add_elems=self.len_f_add_per_node,
            f_conn_elems=self.len_f_conn_per_node,
            f_term_elems=1,
            max_n_nodes=self.max_n_nodes,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdev = 1.0 / math.sqrt(self.message_weights.size(1))
        self.message_weights.data.uniform_(-stdev, stdev)

    def message_terms(
        self, nodes: th.Tensor, node_neighbours: th.Tensor, edges: th.Tensor
    ) -> th.Tensor:
        edges_view = edges.view(-1, 1, 1, self.n_edge_features)
        weights_for_each_edge = (edges_view * self.message_weights.unsqueeze(0)).sum(3)
        return th.matmul(weights_for_each_edge, node_neighbours.unsqueeze(-1)).squeeze()

    def update(self, nodes: th.Tensor, messages: th.Tensor) -> th.Tensor:
        return self.gru(messages, nodes)

    def readout(
        self,
        hidden_nodes: th.Tensor,
        input_nodes: th.Tensor,
        node_mask: th.Tensor,
    ) -> th.Tensor:
        graph_embeddings = th.sum(hidden_nodes, dim=1)
        output = self.APDReadout(hidden_nodes, graph_embeddings)
        return output


class S2V(SummationMPNN):
    """
    The "set2vec" model.
    """

    def __init__(self) -> None:
        super(S2V, self).__init__()

        enn_hidden_dim = self.params.get("enn_hidden_dim", 250)
        enn_depth = self.params.get("enn_depth", 4)
        enn_dropout_p = self.params.get("enn_dropout_p", 0.0)
        mlp1_hidden_dim = self.params.get("mlp1_hidden_dim", 500)
        mlp1_depth = self.params.get("mlp1_depth", 4)
        mlp1_dropout_p = self.params.get("mlp1_dropout_p", 0.0)
        mlp2_hidden_dim = self.params.get("mlp2_hidden_dim", 500)
        mlp2_depth = self.params.get("mlp2_depth", 4)
        mlp2_dropout_p = self.params.get("mlp2_dropout_p", 0.0)
        s2v_memory_size = self.params.get("s2v_memory_size", 100)
        s2v_lstm_computations = self.params.get("s2v_lstm_computations", 3)

        self.hidden_node_features = self.hidden_node_features
        self.enn = MLP(
            in_features=self.edge_features,
            hidden_layer_sizes=[enn_hidden_dim] * enn_depth,
            out_features=self.hidden_node_features * self.message_size,
            dropout_p=enn_dropout_p,
        )
        self.gru = nn.GRUCell(
            input_size=self.message_size,
            hidden_size=self.hidden_node_features,
            bias=True,
        )
        self.s2v = Set2Vec(
            node_features=self.node_features,
            hidden_node_features=self.hidden_node_features,
            lstm_computations=s2v_lstm_computations,
            memory_size=s2v_memory_size,
        )
        self.APDReadout = GlobalReadout(
            node_emb_size=self.hidden_node_features,
            graph_emb_size=s2v_memory_size * 2,
            mlp1_hidden_dim=mlp1_hidden_dim,
            mlp1_depth=mlp1_depth,
            mlp1_dropout_p=mlp1_dropout_p,
            mlp2_hidden_dim=mlp2_hidden_dim,
            mlp2_depth=mlp2_depth,
            mlp2_dropout_p=mlp2_dropout_p,
            f_add_elems=self.len_f_add_per_node,
            f_conn_elems=self.len_f_conn_per_node,
            f_term_elems=1,
            max_n_nodes=self.max_n_nodes,
        )

    def message_terms(
        self, nodes: th.Tensor, node_neighbours: th.Tensor, edges: th.Tensor
    ) -> th.Tensor:
        enn_output = self.enn(edges)
        matrices = enn_output.view(-1, self.message_size, self.hidden_node_features)
        msg_terms = th.matmul(matrices, node_neighbours.unsqueeze(-1)).squeeze(-1)
        return msg_terms

    def update(self, nodes: th.Tensor, messages: th.Tensor) -> th.Tensor:
        return self.gru(messages, nodes)

    def readout(
        self,
        hidden_nodes: th.Tensor,
        input_nodes: th.Tensor,
        node_mask: th.Tensor,
    ) -> th.Tensor:
        graph_embeddings = self.s2v(hidden_nodes, input_nodes, node_mask)
        output = self.APDReadout(hidden_nodes, graph_embeddings)
        return output


class AttentionS2V(AggregationMPNN):
    """
    The "set2vec with attention" model.
    """

    def __init__(self) -> None:
        super(AttentionS2V, self).__init__()

        att_hidden_dim = self.params.get("att_hidden_dim", 250)
        att_depth = self.params.get("att_depth", 4)
        att_dropout_p = self.params.get("att_dropout_p", 0.0)
        enn_hidden_dim = self.params.get("enn_hidden_dim", 250)
        enn_depth = self.params.get("enn_depth", 4)
        enn_dropout_p = self.params.get("enn_dropout_p", 0.0)
        mlp1_hidden_dim = self.params.get("mlp1_hidden_dim", 500)
        mlp1_depth = self.params.get("mlp1_depth", 4)
        mlp1_dropout_p = self.params.get("mlp1_dropout_p", 0.0)
        mlp2_hidden_dim = self.params.get("mlp2_hidden_dim", 500)
        mlp2_depth = self.params.get("mlp2_depth", 4)
        mlp2_dropout_p = self.params.get("mlp2_dropout_p", 0.0)
        s2v_memory_size = self.params.get("s2v_memory_size", 100)
        s2v_lstm_computations = self.params.get("s2v_lstm_computations", 3)

        self.enn = MLP(
            in_features=self.edge_features,
            hidden_layer_sizes=[enn_hidden_dim] * enn_depth,
            out_features=self.hidden_node_features * self.message_size,
            dropout_p=enn_dropout_p,
        )
        self.att_enn = MLP(
            in_features=self.hidden_node_features + self.edge_features,
            hidden_layer_sizes=[att_hidden_dim] * att_depth,
            out_features=self.message_size,
            dropout_p=att_dropout_p,
        )
        self.gru = nn.GRUCell(
            input_size=self.message_size,
            hidden_size=self.hidden_node_features,
            bias=True,
        )
        self.s2v = Set2Vec(
            node_features=self.node_features,
            hidden_node_features=self.hidden_node_features,
            lstm_computations=s2v_lstm_computations,
            memory_size=s2v_memory_size,
        )
        self.APDReadout = GlobalReadout(
            node_emb_size=self.hidden_node_features,
            graph_emb_size=s2v_memory_size * 2,
            mlp1_hidden_dim=mlp1_hidden_dim,
            mlp1_depth=mlp1_depth,
            mlp1_dropout_p=mlp1_dropout_p,
            mlp2_hidden_dim=mlp2_hidden_dim,
            mlp2_depth=mlp2_depth,
            mlp2_dropout_p=mlp2_dropout_p,
            f_add_elems=self.len_f_add_per_node,
            f_conn_elems=self.len_f_conn_per_node,
            f_term_elems=1,
            max_n_nodes=self.max_n_nodes,
        )

    def aggregate_message(
        self,
        nodes: th.Tensor,
        node_neighbours: th.Tensor,
        edges: th.Tensor,
        mask: th.Tensor,
    ) -> th.Tensor:
        softmax = nn.Softmax(dim=1)
        max_node_degree = node_neighbours.shape[1]
        enn_output = self.enn(edges)
        matrices = enn_output.view(
            -1,
            max_node_degree,
            self.message_size,
            self.hidden_node_features,
        )
        message_terms = th.matmul(matrices, node_neighbours.unsqueeze(-1)).squeeze()
        att_enn_output = self.att_enn(th.cat((edges, node_neighbours), dim=2))
        energies = att_enn_output.view(-1, max_node_degree, self.message_size)
        energy_mask = (1 - mask).float() * self.big_negative
        weights = softmax(energies + energy_mask.unsqueeze(-1))

        return (weights * message_terms).sum(1)

    def update(self, nodes: th.Tensor, messages: th.Tensor) -> th.Tensor:
        messages = messages + th.zeros(self.message_size)
        return self.gru(messages, nodes)

    def readout(
        self,
        hidden_nodes: th.Tensor,
        input_nodes: th.Tensor,
        node_mask: th.Tensor,
    ) -> th.Tensor:
        graph_embeddings = self.s2v(hidden_nodes, input_nodes, node_mask)
        output = self.APDReadout(hidden_nodes, graph_embeddings)
        return output


class GGNN(SummationMPNN):
    """
    The "gated-graph neural network" model.
    """

    def __init__(self) -> None:
        super(GGNN, self).__init__()

        enn_hidden_dim = self.params.get("enn_hidden_dim", 250)
        enn_depth = self.params.get("enn_depth", 4)
        enn_dropout_p = self.params.get("enn_dropout_p", 0.0)
        gather_width = self.params.get("gather_width", 100)
        gather_att_hidden_dim = self.params.get("gather_att_hidden_dim", 250)
        gather_att_depth = self.params.get("gather_att_depth", 4)
        gather_att_dropout_p = self.params.get("gather_att_dropout_p", 0.0)
        gather_emb_hidden_dim = self.params.get("gather_emb_hidden_dim", 250)
        gather_emb_depth = self.params.get("gather_emb_depth", 4)
        gather_emb_dropout_p = self.params.get("gather_emb_dropout_p", 0.0)
        mlp1_hidden_dim = self.params.get("mlp1_hidden_dim", 500)
        mlp1_depth = self.params.get("mlp1_depth", 4)
        mlp1_dropout_p = self.params.get("mlp1_dropout_p", 0.0)
        mlp2_hidden_dim = self.params.get("mlp2_hidden_dim", 500)
        mlp2_depth = self.params.get("mlp2_depth", 4)
        mlp2_dropout_p = self.params.get("mlp2_dropout_p", 0.0)

        self.msg_nns = nn.ModuleList()
        for _ in range(self.edge_features):
            self.msg_nns.append(
                MLP(
                    in_features=self.hidden_node_features,
                    hidden_layer_sizes=[enn_hidden_dim] * enn_depth,
                    out_features=self.message_size,
                    dropout_p=enn_dropout_p,
                )
            )
        self.gru = nn.GRUCell(
            input_size=self.message_size,
            hidden_size=self.hidden_node_features,
            bias=True,
        )
        self.gather = GraphGather(
            node_features=self.node_features,
            hidden_node_features=self.hidden_node_features,
            out_features=gather_width,
            att_depth=gather_att_depth,
            att_hidden_dim=gather_att_hidden_dim,
            att_dropout_p=gather_att_dropout_p,
            emb_depth=gather_emb_depth,
            emb_hidden_dim=gather_emb_hidden_dim,
            emb_dropout_p=gather_emb_dropout_p,
            big_positive=self.big_positive,
        )
        self.APDReadout = GlobalReadout(
            node_emb_size=self.hidden_node_features,
            graph_emb_size=gather_width,
            mlp1_hidden_dim=mlp1_hidden_dim,
            mlp1_depth=mlp1_depth,
            mlp1_dropout_p=mlp1_dropout_p,
            mlp2_hidden_dim=mlp2_hidden_dim,
            mlp2_depth=mlp2_depth,
            mlp2_dropout_p=mlp2_dropout_p,
            f_add_elems=self.len_f_add_per_node,
            f_conn_elems=self.len_f_conn_per_node,
            f_term_elems=1,
            max_n_nodes=self.max_n_nodes,
        )

    def message_terms(
        self, nodes: th.Tensor, node_neighbours: th.Tensor, edges: th.Tensor
    ) -> th.Tensor:
        edges_v = edges.view(-1, self.edge_features, 1)
        node_neighbours_v = edges_v * node_neighbours.view(
            -1, 1, self.hidden_node_features
        )
        terms_masked_per_edge = [
            edges_v[:, i, :] * self.msg_nns[i](node_neighbours_v[:, i, :])
            for i in range(self.edge_features)
        ]
        return sum(terms_masked_per_edge)

    def update(self, nodes: th.Tensor, messages: th.Tensor) -> th.Tensor:
        return self.gru(messages, nodes)

    def readout(
        self,
        hidden_nodes: th.Tensor,
        input_nodes: th.Tensor,
        node_mask: th.Tensor,
    ) -> th.Tensor:
        graph_embeddings = self.gather(hidden_nodes, input_nodes, node_mask)
        output = self.APDReadout(hidden_nodes, graph_embeddings)
        return output


class AttentionGGNN(AggregationMPNN):
    """
    The "GGNN with attention" model.
    """

    def __init__(self) -> None:
        super(AttentionGGNN, self).__init__()

        msg_hidden_dim = self.params.get("msg_hidden_dim", 250)
        msg_depth = self.params.get("msg_depth", 4)
        msg_dropout_p = self.params.get("msg_dropout_p", 0.0)
        att_hidden_dim = self.params.get("att_hidden_dim", 250)
        att_depth = self.params.get("att_depth", 4)
        att_dropout_p = self.params.get("att_dropout_p", 0.0)
        gather_width = self.params.get("gather_width", 100)
        gather_att_hidden_dim = self.params.get("gather_att_hidden_dim", 250)
        gather_att_depth = self.params.get("gather_att_depth", 4)
        gather_att_dropout_p = self.params.get("gather_att_dropout_p", 0.0)
        gather_emb_hidden_dim = self.params.get("gather_emb_hidden_dim", 250)
        gather_emb_depth = self.params.get("gather_emb_depth", 4)
        gather_emb_dropout_p = self.params.get("gather_emb_dropout_p", 0.0)
        mlp1_hidden_dim = self.params.get("mlp1_hidden_dim", 500)
        mlp1_depth = self.params.get("mlp1_depth", 4)
        mlp1_dropout_p = self.params.get("mlp1_dropout_p", 0.0)
        mlp2_hidden_dim = self.params.get("mlp2_hidden_dim", 500)
        mlp2_depth = self.params.get("mlp2_depth", 4)
        mlp2_dropout_p = self.params.get("mlp2_dropout_p", 0.0)

        self.msg_nns = nn.ModuleList()
        self.att_nns = nn.ModuleList()

        for _ in range(self.edge_features):
            self.msg_nns.append(
                MLP(
                    in_features=self.hidden_node_features,
                    hidden_layer_sizes=[msg_hidden_dim] * msg_depth,
                    out_features=self.message_size,
                    dropout_p=msg_dropout_p,
                )
            )
            self.att_nns.append(
                MLP(
                    in_features=self.hidden_node_features,
                    hidden_layer_sizes=[att_hidden_dim] * att_depth,
                    out_features=self.message_size,
                    dropout_p=att_dropout_p,
                )
            )
        self.gru = nn.GRUCell(
            input_size=self.message_size,
            hidden_size=self.hidden_node_features,
            bias=True,
        )
        self.gather = GraphGather(
            node_features=self.node_features,
            hidden_node_features=self.hidden_node_features,
            out_features=gather_width,
            att_depth=gather_att_depth,
            att_hidden_dim=gather_att_hidden_dim,
            att_dropout_p=gather_att_dropout_p,
            emb_depth=gather_emb_depth,
            emb_hidden_dim=gather_emb_hidden_dim,
            emb_dropout_p=gather_emb_dropout_p,
            big_positive=self.big_positive,
        )
        self.APDReadout = GlobalReadout(
            node_emb_size=self.hidden_node_features,
            graph_emb_size=gather_width,
            mlp1_hidden_dim=mlp1_hidden_dim,
            mlp1_depth=mlp1_depth,
            mlp1_dropout_p=mlp1_dropout_p,
            mlp2_hidden_dim=mlp2_hidden_dim,
            mlp2_depth=mlp2_depth,
            mlp2_dropout_p=mlp2_dropout_p,
            f_add_elems=self.len_f_add_per_node,
            f_conn_elems=self.len_f_conn_per_node,
            f_term_elems=1,
            max_n_nodes=self.max_n_nodes,
        )

    def aggregate_message(
        self,
        nodes: th.Tensor,
        node_neighbours: th.Tensor,
        edges: th.Tensor,
        mask: th.Tensor,
    ) -> th.Tensor:
        softmax = nn.Softmax(dim=1)
        energy_mask = (mask == 0).float() * self.big_positive
        embeddings_masked_per_edge = [
            edges[:, :, i].unsqueeze(-1) * self.msg_nns[i](node_neighbours)
            for i in range(self.n_edge_features)
        ]
        energies_masked_per_edge = [
            edges[:, :, i].unsqueeze(-1) * self.att_nns[i](node_neighbours)
            for i in range(self.n_edge_features)
        ]
        embedding = sum(embeddings_masked_per_edge)
        energies = sum(energies_masked_per_edge) - energy_mask.unsqueeze(-1)
        attention = softmax(energies)

        return th.sum(attention * embedding, dim=1)

    def update(self, nodes: th.Tensor, messages: th.Tensor) -> th.Tensor:
        return self.gru(messages, nodes)

    def readout(
        self,
        hidden_nodes: th.Tensor,
        input_nodes: th.Tensor,
        node_mask: th.Tensor,
    ) -> th.Tensor:
        graph_embeddings = self.gather(hidden_nodes, input_nodes, node_mask)
        output = self.APDReadout(hidden_nodes, graph_embeddings)
        return output


class EMN(EdgeMPNN):
    """
    The "edge memory network" model.
    """

    def __init__(self) -> None:
        super(EMN, self).__init__()

        edge_emb_hidden_dim = self.params.get("edge_emb_hidden_dim", 250)
        edge_emb_depth = self.params.get("edge_emb_depth", 4)
        edge_emb_dropout_p = self.params.get("edge_emb_dropout_p", 0.0)
        msg_hidden_dim = self.params.get("msg_hidden_dim", 250)
        msg_depth = self.params.get("msg_depth", 4)
        msg_dropout_p = self.params.get("msg_dropout_p", 0.0)
        att_hidden_dim = self.params.get("att_hidden_dim", 250)
        att_depth = self.params.get("att_depth", 4)
        att_dropout_p = self.params.get("att_dropout_p", 0.0)
        gather_width = self.params.get("gather_width", 100)
        gather_att_hidden_dim = self.params.get("gather_att_hidden_dim", 250)
        gather_att_depth = self.params.get("gather_att_depth", 4)
        gather_att_dropout_p = self.params.get("gather_att_dropout_p", 0.0)
        gather_emb_hidden_dim = self.params.get("gather_emb_hidden_dim", 250)
        gather_emb_depth = self.params.get("gather_emb_depth", 4)
        gather_emb_dropout_p = self.params.get("gather_emb_dropout_p", 0.0)
        mlp1_hidden_dim = self.params.get("mlp1_hidden_dim", 500)
        mlp1_depth = self.params.get("mlp1_depth", 4)
        mlp1_dropout_p = self.params.get("mlp1_dropout_p", 0.0)
        mlp2_hidden_dim = self.params.get("mlp2_hidden_dim", 500)
        mlp2_depth = self.params.get("mlp2_depth", 4)
        mlp2_dropout_p = self.params.get("mlp2_dropout_p", 0.0)

        self.embedding_nn = MLP(
            in_features=self.node_features * 2 + self.edge_features,
            hidden_layer_sizes=[edge_emb_hidden_dim] * edge_emb_depth,
            out_features=self.edge_emb_size,
            dropout_p=edge_emb_dropout_p,
        )
        self.emb_msg_nn = MLP(
            in_features=self.edge_emb_size,
            hidden_layer_sizes=[msg_hidden_dim] * msg_depth,
            out_features=self.edge_emb_size,
            dropout_p=msg_dropout_p,
        )
        self.att_msg_nn = MLP(
            in_features=self.edge_emb_size,
            hidden_layer_sizes=[att_hidden_dim] * att_depth,
            out_features=self.edge_emb_size,
            dropout_p=att_dropout_p,
        )
        self.gru = nn.GRUCell(
            input_size=self.edge_emb_size,
            hidden_size=self.edge_emb_size,
            bias=True,
        )
        self.gather = GraphGather(
            node_features=self.edge_emb_size,
            hidden_node_features=self.edge_emb_size,
            out_features=gather_width,
            att_depth=gather_att_depth,
            att_hidden_dim=gather_att_hidden_dim,
            att_dropout_p=gather_att_dropout_p,
            emb_depth=gather_emb_depth,
            emb_hidden_dim=gather_emb_hidden_dim,
            emb_dropout_p=gather_emb_dropout_p,
            big_positive=self.big_positive,
        )
        self.APDReadout = GlobalReadout(
            node_emb_size=self.edge_emb_size,
            graph_emb_size=gather_width,
            mlp1_hidden_dim=mlp1_hidden_dim,
            mlp1_depth=mlp1_depth,
            mlp1_dropout_p=mlp1_dropout_p,
            mlp2_hidden_dim=mlp2_hidden_dim,
            mlp2_depth=mlp2_depth,
            mlp2_dropout_p=mlp2_dropout_p,
            f_add_elems=self.len_f_add_per_node,
            f_conn_elems=self.len_f_conn_per_node,
            f_term_elems=1,
            max_n_nodes=self.max_n_nodes,
        )

    def preprocess_edges(
        self, nodes: th.Tensor, node_neighbours: th.Tensor, edges: th.Tensor
    ) -> th.Tensor:
        cat = th.cat((nodes, node_neighbours, edges), dim=1)
        return th.tanh(self.embedding_nn(cat))

    def propagate_edges(
        self,
        edges: th.Tensor,
        ingoing_edge_memories: th.Tensor,
        ingoing_edges_mask: th.Tensor,
    ) -> th.Tensor:
        softmax = nn.Softmax(dim=1)
        energy_mask = ((1 - ingoing_edges_mask).float() * self.big_negative).unsqueeze(
            -1
        )
        cat = th.cat((edges.unsqueeze(1), ingoing_edge_memories), dim=1)
        embeddings = self.emb_msg_nn(cat)
        edge_energy = self.att_msg_nn(edges)
        ing_memory_energies = self.att_msg_nn(ingoing_edge_memories) + energy_mask
        energies = th.cat((edge_energy.unsqueeze(1), ing_memory_energies), dim=1)
        attention = softmax(energies)
        # set aggregation of set of given edge feature and ingoing edge memories
        message = (attention * embeddings).sum(dim=1)

        return self.gru(message)  # return hidden state

    def readout(
        self,
        hidden_nodes: th.Tensor,
        input_nodes: th.Tensor,
        node_mask: th.Tensor,
    ) -> th.Tensor:
        graph_embeddings = self.gather(hidden_nodes, input_nodes, node_mask)
        output = self.APDReadout(hidden_nodes, graph_embeddings)
        return output
