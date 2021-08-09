# load general packages and functions
import json

import torch as th
import torch.nn as nn


class AggregationMPNN(nn.Module):
    """
    Abstract `AggregationMPNN` class. Specific models using this class are
    defined in `mpnn.py`; these are the attention networks AttS2V and AttGGNN.
    """

    def __init__(self, params: dict) -> None:
        super().__init__()
        # load parameters
        self.params = params
        self.hidden_node_features = self.params.get("hidden_node_features", 100)
        self.message_size = self.params.get("message_size", 100)
        self.message_passes = self.params.get("message_passes", 3)
        self.big_negative = self.params.get("big_negative", -1e6)
        self.node_features = self.params.get("node_features", 0)
        self.edge_features = self.params.get("edge_features", 0)
        self.len_f_add_per_node = self.params.get("len_f_add_per_node", 0)
        self.len_f_conn_per_node = self.params.get("len_f_conn_per_node", 0)
        self.max_n_nodes = self.params.get("max_n_nodes", 0)

    def aggregate_message(
        self,
        nodes: th.Tensor,
        node_neighbours: th.Tensor,
        edges: th.Tensor,
        mask: th.Tensor,
    ) -> None:
        """
        Message aggregation function, to be implemented in all `AggregationMPNN` subclasses.

        Args:
        ----
            nodes (th.Tensor) : Batch of node feature vectors.
            node_neighbours (th.Tensor) : Batch of node feature vectors for neighbors.
            edges (th.Tensor) : Batch of edge feature vectors.
            mask (th.Tensor) : Mask for non-existing neighbors, where elements are 1 if
              corresponding element exists and 0 otherwise.

        Shapes:
        ------
            nodes : (total N nodes in batch, N node features)
            node_neighbours : (total N nodes in batch, max node degree, N node features)
            edges : (total N nodes in batch, max node degree, N edge features)
            mask : (total N nodes in batch, max node degree)
        """
        raise NotImplementedError

    def update(self, nodes: th.Tensor, messages: th.Tensor) -> None:
        """
        Message update function, to be implemented in all `AggregationMPNN` subclasses.

        Args:
        ----
            nodes (th.Tensor) : Batch of node feature vectors.
            messages (th.Tensor) : Batch of incoming messages.

        Shapes:
        ------
            nodes : (total N nodes in batch, N node features)
            messages : (total N nodes in batch, N node features)
        """
        raise NotImplementedError

    def readout(
        self,
        hidden_nodes: th.Tensor,
        input_nodes: th.Tensor,
        node_mask: th.Tensor,
    ) -> None:
        """
        Local readout function, to be implemented in all `AggregationMPNN` subclasses.

        Args:
        ----
            hidden_nodes (th.Tensor) : Batch of node feature vectors.
            input_nodes (th.Tensor) : Batch of node feature vectors.
            node_mask (th.Tensor) : Mask for non-existing neighbors, where elements are 1
              if corresponding element exists and 0 otherwise.

        Shapes:
        ------
            hidden_nodes : (total N nodes in batch, N node features)
            input_nodes : (total N nodes in batch, N node features)
            node_mask : (total N nodes in batch, N features)
        """
        raise NotImplementedError

    def forward(self, nodes: th.Tensor, edges: th.Tensor) -> th.Tensor:
        """
        Defines forward pass.

        Args:
        ----
            nodes (th.Tensor) : Batch of node feature matrices.
            edges (th.Tensor) : Batch of edge feature tensors.

        Shapes:
        ------
            nodes : (batch size, N nodes, N node features)
            edges : (batch size, N nodes, N nodes, N edge features)

        Returns:
        -------
            output (th.Tensor) : This would normally be the learned graph representation,
              but in all MPNN readout functions in this work, the last layer is used to
              predict the action probability distribution for a batch of graphs from the learned
              graph representation.
        """
        adjacency = th.sum(edges, dim=3)

        # **note: "idc" == "indices", "nghb{s}" == "neighbour(s)"
        (
            edge_batch_batch_idc,
            edge_batch_node_idc,
            edge_batch_nghb_idc,
        ) = adjacency.nonzero(as_tuple=True)

        node_batch_batch_idc, node_batch_node_idc = adjacency.sum(-1).nonzero(
            as_tuple=True
        )
        node_batch_adj = adjacency[node_batch_batch_idc, node_batch_node_idc, :]

        node_batch_size = node_batch_batch_idc.shape[0]
        node_degrees = node_batch_adj.sum(-1).long()
        max_node_degree = node_degrees.max()

        node_batch_node_nghbs = th.zeros(
            node_batch_size,
            max_node_degree,
            self.hidden_node_features,
        )
        node_batch_edges = th.zeros(
            node_batch_size,
            max_node_degree,
            self.edge_features,
        )

        node_batch_nghb_nghb_idc = th.cat([th.arange(i) for i in node_degrees]).long()

        edge_batch_node_batch_idc = th.cat(
            [i * th.ones(degree) for i, degree in enumerate(node_degrees)]
        ).long()

        node_batch_node_nghb_mask = th.zeros(node_batch_size, max_node_degree)

        node_batch_node_nghb_mask[
            edge_batch_node_batch_idc, node_batch_nghb_nghb_idc
        ] = 1

        node_batch_edges[
            edge_batch_node_batch_idc, node_batch_nghb_nghb_idc, :
        ] = edges[edge_batch_batch_idc, edge_batch_node_idc, edge_batch_nghb_idc, :]

        # pad up the hidden nodes
        hidden_nodes = th.zeros(
            nodes.shape[0],
            nodes.shape[1],
            self.hidden_node_features,
        )
        hidden_nodes[
            : nodes.shape[0], : nodes.shape[1], : nodes.shape[2]
        ] = nodes.clone()

        for _ in range(self.message_passes):

            node_batch_nodes = hidden_nodes[
                node_batch_batch_idc, node_batch_node_idc, :
            ]
            node_batch_node_nghbs[
                edge_batch_node_batch_idc, node_batch_nghb_nghb_idc, :
            ] = hidden_nodes[edge_batch_batch_idc, edge_batch_nghb_idc, :]

            messages = self.aggregate_message(
                nodes=node_batch_nodes,
                node_neighbours=node_batch_node_nghbs.clone(),
                edges=node_batch_edges,
                mask=node_batch_node_nghb_mask,
            )

            hidden_nodes[node_batch_batch_idc, node_batch_node_idc, :] = self.update(
                node_batch_nodes.clone(), messages
            )

        node_mask = adjacency.sum(-1) != 0

        output = self.readout(hidden_nodes, nodes, node_mask)

        return output


class SummationMPNN(nn.Module):
    """
    Abstract `SummationMPNN` class. Specific models using this class are
    defined in `mpnn.py`; these are MNN, S2V, and GGNN.
    """

    def __init__(self, params: dict) -> None:
        super().__init__()
        # load parameters
        self.params = params
        self.hidden_node_features = self.params.get("hidden_node_features", 100)
        self.message_size = self.params.get("message_size", 100)
        self.message_passes = self.params.get("message_passes", 3)
        self.big_positive = self.params.get("big_positive", 1e6)
        self.node_features = self.params.get("node_features", 0)
        self.edge_features = self.params.get("edge_features", 0)
        self.len_f_add_per_node = self.params.get("len_f_add_per_node", 0)
        self.len_f_conn_per_node = self.params.get("len_f_conn_per_node", 0)
        self.max_n_nodes = self.params.get("max_n_nodes", 0)

    def message_terms(
        self, nodes: th.Tensor, node_neighbours: th.Tensor, edges: th.Tensor
    ) -> None:
        """
        Message passing function, to be implemented in all `SummationMPNN` subclasses.

        Args:
        ----
            nodes (th.Tensor) : Batch of node feature vectors.
            node_neighbours (th.Tensor) : Batch of node feature vectors for neighbors.
            edges (th.Tensor) : Batch of edge feature vectors.

        Shapes:
        ------
            nodes : (total N nodes in batch, N node features)
            node_neighbours : (total N nodes in batch, max node degree, N node features)
            edges : (total N nodes in batch, max node degree, N edge features)
        """
        raise NotImplementedError

    def update(self, nodes: th.Tensor, messages: th.Tensor) -> None:
        """
        Message update function, to be implemented in all `SummationMPNN` subclasses.

        Args:
        ----
            nodes (th.Tensor) : Batch of node feature vectors.
            messages (th.Tensor) : Batch of incoming messages.

        Shapes:
        ------
            nodes : (total N nodes in batch, N node features)
            messages : (total N nodes in batch, N node features)
        """
        raise NotImplementedError

    def readout(
        self,
        hidden_nodes: th.Tensor,
        input_nodes: th.Tensor,
        node_mask: th.Tensor,
    ) -> None:
        """
        Local readout function, to be implemented in all `SummationMPNN` subclasses.

        Args:
        ----
            hidden_nodes (th.Tensor) : Batch of node feature vectors.
            input_nodes (th.Tensor) : Batch of node feature vectors.
            node_mask (th.Tensor) : Mask for non-existing neighbors, where elements are 1
              if corresponding element exists and 0 otherwise.

        Shapes:
        ------
            hidden_nodes : (total N nodes in batch, N node features)
            input_nodes : (total N nodes in batch, N node features)
            node_mask : (total N nodes in batch, N features)
        """
        raise NotImplementedError

    def forward(self, nodes: th.Tensor, edges: th.Tensor) -> None:
        """
        Defines forward pass.

        Args:
        ----
            nodes (th.Tensor) : Batch of node feature matrices.
            edges (th.Tensor) : Batch of edge feature tensors.

        Shapes:
        ------
            nodes : (batch size, N nodes, N node features)
            edges : (batch size, N nodes, N nodes, N edge features)

        Returns:
        -------
            output (th.Tensor) : This would normally be the learned graph representation,
              but in all MPNN readout functions in this work, the last layer is used to
              predict the action probability distribution for a batch of graphs from the learned
              graph representation.
        """
        adjacency = th.sum(edges, dim=3)

        # **note: "idc" == "indices", "nghb{s}" == "neighbour(s)"
        (
            edge_batch_batch_idc,
            edge_batch_node_idc,
            edge_batch_nghb_idc,
        ) = adjacency.nonzero(as_tuple=True)

        (node_batch_batch_idc, node_batch_node_idc) = adjacency.sum(-1).nonzero(
            as_tuple=True
        )

        same_batch = node_batch_batch_idc.view(-1, 1) == edge_batch_batch_idc
        same_node = node_batch_node_idc.view(-1, 1) == edge_batch_node_idc

        # element ij of `message_summation_matrix` is 1 if `edge_batch_edges[j]`
        # is connected with `node_batch_nodes[i]`, else 0
        message_summation_matrix = (same_batch * same_node).float()

        edge_batch_edges = edges[
            edge_batch_batch_idc, edge_batch_node_idc, edge_batch_nghb_idc, :
        ]

        # pad up the hidden nodes
        hidden_nodes = th.zeros(
            nodes.shape[0],
            nodes.shape[1],
            self.hidden_node_features,
        )
        hidden_nodes[
            : nodes.shape[0], : nodes.shape[1], : nodes.shape[2]
        ] = nodes.clone()
        node_batch_nodes = hidden_nodes[node_batch_batch_idc, node_batch_node_idc, :]

        for _ in range(self.message_passes):
            edge_batch_nodes = hidden_nodes[
                edge_batch_batch_idc, edge_batch_node_idc, :
            ]

            edge_batch_nghbs = hidden_nodes[
                edge_batch_batch_idc, edge_batch_nghb_idc, :
            ]

            message_terms = self.message_terms(
                edge_batch_nodes, edge_batch_nghbs, edge_batch_edges
            )

            if len(message_terms.size()) == 1:  # if a single graph in batch
                message_terms = message_terms.unsqueeze(0)

            # the summation in eq. 1 of the NMPQC paper happens here
            messages = th.matmul(message_summation_matrix, message_terms)

            node_batch_nodes = self.update(node_batch_nodes, messages)
            hidden_nodes[
                node_batch_batch_idc, node_batch_node_idc, :
            ] = node_batch_nodes.clone()

        node_mask = adjacency.sum(-1) != 0

        output = self.readout(hidden_nodes, nodes, node_mask)

        return output


class EdgeMPNN(nn.Module):
    """
    Abstract `EdgeMPNN` class. A specific model using this class is defined
    in `mpnn.py`; this is the EMN.
    """

    def __init__(self, params: dict) -> None:
        super().__init__()
        # load parameters
        self.params = params
        self.hidden_node_features = self.params.get("hidden_node_features", 100)
        self.edge_emb_size = self.params.get("edge_emb_size", 100)
        self.big_positive = self.params.get("big_positive", 1e6)
        self.node_features = self.params.get("node_features", 0)
        self.edge_features = self.params.get("edge_features", 0)
        self.len_f_add_per_node = self.params.get("len_f_add_per_node", 0)
        self.len_f_conn_per_node = self.params.get("len_f_conn_per_node", 0)
        self.max_n_nodes = self.params.get("max_n_nodes", 0)

    def preprocess_edges(
        self, nodes: th.Tensor, node_neighbours: th.Tensor, edges: th.Tensor
    ) -> None:
        """
        Edge preprocessing step, to be implemented in all `EdgeMPNN` subclasses.

        Args:
        ----
            nodes (th.Tensor) : Batch of node feature vectors.
            node_neighbours (th.Tensor) : Batch of node feature vectors for neighbors.
            edges (th.Tensor) : Batch of edge feature vectors.
            max node degree, number of edge features}.

        Shapes:
        ------
            nodes : (total N nodes in batch, N node features)
            node_neighbours : (total N nodes in batch, max node degree, N node features)
            edges : (total N nodes in batch, max node degree, N edge features)
        """
        raise NotImplementedError

    def propagate_edges(
        self,
        edges: th.Tensor,
        ingoing_edge_memories: th.Tensor,
        ingoing_edges_mask: th.Tensor,
    ) -> None:
        """
        Edge propagation rule, to be implemented in all `EdgeMPNN` subclasses.

        Args:
        ----
            edges (th.Tensor) : Batch of edge feature tensors.
            ingoing_edge_memories (th.Tensor) : Batch of memories for all ingoing edges.
            ingoing_edges_mask (th.Tensor) : Mask for ingoing edges.

        Shapes:
        ------
            edges : (batch size, N nodes, N nodes, total N edge features)
            ingoing_edge_memories : (total N edges in batch, total N edge features)
            ingoing_edges_mask : (total N edges in batch, max node degree, total N edge features)
        """
        raise NotImplementedError

    def readout(
        self,
        hidden_nodes: th.Tensor,
        input_nodes: th.Tensor,
        node_mask: th.Tensor,
    ) -> None:
        """
        Local readout function, to be implemented in all `EdgeMPNN` subclasses.

        Args:
        ----
            hidden_nodes (th.Tensor) : Batch of node feature vectors.
            input_nodes (th.Tensor) : Batch of node feature vectors.
            node_mask (th.Tensor) : Mask for non-existing neighbors, where elements are 1
              if corresponding element exists and 0 otherwise.

        Shapes:
        ------
            hidden_nodes : (total N nodes in batch, N node features)
            input_nodes : (total N nodes in batch, N node features)
            node_mask : (total N nodes in batch, N features)
        """
        raise NotImplementedError

    def forward(self, nodes: th.Tensor, edges: th.Tensor) -> th.Tensor:
        """
        Defines forward pass.

        Args:
        ----
            nodes (th.Tensor) : Batch of node feature matrices.
            edges (th.Tensor) : Batch of edge feature tensors.

        Shapes:
        ------
            nodes : (batch size, N nodes, N node features)
            edges : (batch size, N nodes, N nodes, N edge features)

        Returns:
        -------
            output (th.Tensor) : This would normally be the learned graph representation,
              but in all MPNN readout functions in this work, the last layer is used to
              predict the action probability distribution for a batch of graphs from the learned
              graph representation.
        """
        adjacency = th.sum(edges, dim=3)

        # indices for finding edges in batch; `edges_b_idx` is batch index,
        # `edges_n_idx` is the node index, and `edges_nghb_idx` is the index
        # that each node in `edges_n_idx` is bound to
        edges_b_idx, edges_n_idx, edges_nghb_idx = adjacency.nonzero(as_tuple=True)

        n_edges = edges_n_idx.shape[0]
        adj_of_edge_batch_idc = adjacency.clone().long()

        # +1 to distinguish idx 0 from empty elements, subtracted few lines down
        r = th.arange(1, n_edges + 1)

        adj_of_edge_batch_idc[edges_b_idx, edges_n_idx, edges_nghb_idx] = r

        ingoing_edges_eb_idx = (
            th.cat(
                [
                    row[row.nonzero()]
                    for row in adj_of_edge_batch_idc[edges_b_idx, edges_nghb_idx, :]
                ]
            )
            - 1
        ).squeeze()

        edge_degrees = adjacency[edges_b_idx, edges_nghb_idx, :].sum(-1).long()
        ingoing_edges_igeb_idx = th.cat(
            [i * th.ones(d) for i, d in enumerate(edge_degrees)]
        ).long()
        ingoing_edges_ige_idx = th.cat([th.arange(i) for i in edge_degrees]).long()

        batch_size = adjacency.shape[0]
        n_nodes = adjacency.shape[1]
        max_node_degree = adjacency.sum(-1).max().int()
        edge_memories = th.zeros(n_edges, self.edge_emb_size)

        ingoing_edge_memories = th.zeros(
            n_edges,
            max_node_degree,
            self.edge_emb_size,
        )
        ingoing_edges_mask = th.zeros(n_edges, max_node_degree)

        edge_batch_nodes = nodes[edges_b_idx, edges_n_idx, :]
        # **note: "nghb{s}" == "neighbour(s)"
        edge_batch_nghbs = nodes[edges_b_idx, edges_nghb_idx, :]
        edge_batch_edges = edges[edges_b_idx, edges_n_idx, edges_nghb_idx, :]
        edge_batch_edges = self.preprocess_edges(
            nodes=edge_batch_nodes,
            node_neighbours=edge_batch_nghbs,
            edges=edge_batch_edges,
        )

        # remove h_ji:s influence on h_ij
        ingoing_edges_nghb_idx = edges_nghb_idx[ingoing_edges_eb_idx]
        ingoing_edges_receiving_edge_n_idx = edges_n_idx[ingoing_edges_igeb_idx]
        diff_idx = (
            ingoing_edges_receiving_edge_n_idx != ingoing_edges_nghb_idx
        ).nonzero()

        try:
            ingoing_edges_eb_idx = ingoing_edges_eb_idx[diff_idx].squeeze()
            ingoing_edges_ige_idx = ingoing_edges_ige_idx[diff_idx].squeeze()
            ingoing_edges_igeb_idx = ingoing_edges_igeb_idx[diff_idx].squeeze()
        except:
            pass

        ingoing_edges_mask[ingoing_edges_igeb_idx, ingoing_edges_ige_idx] = 1

        for _ in range(self.message_passes):
            ingoing_edge_memories[
                ingoing_edges_igeb_idx, ingoing_edges_ige_idx, :
            ] = edge_memories[ingoing_edges_eb_idx, :]
            edge_memories = self.propagate_edges(
                edges=edge_batch_edges,
                ingoing_edge_memories=ingoing_edge_memories.clone(),
                ingoing_edges_mask=ingoing_edges_mask,
            )

        node_mask = adjacency.sum(-1) != 0

        node_sets = th.zeros(
            batch_size,
            n_nodes,
            max_node_degree,
            self.edge_emb_size,
        )

        edge_batch_edge_memory_idc = th.cat(
            [th.arange(row.sum()) for row in adjacency.view(-1, n_nodes)]
        ).long()

        node_sets[
            edges_b_idx, edges_n_idx, edge_batch_edge_memory_idc, :
        ] = edge_memories
        graph_sets = node_sets.sum(2)

        output = self.readout(graph_sets, graph_sets, node_mask)
        return output
