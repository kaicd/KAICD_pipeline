# load general packages and functions
import torch as th
import torch.nn as nn


class AggregationMPNN(nn.Module):
    """
    Abstract `AggregationMPNN` class. Specific models using this class are
    defined in `mpnn.py`; these are the attention networks AttS2V and AttGGNN.
    """

    def __init__(
        self,
        hidden_node_features: int,
        n_edge_features: int,
        message_size: int,
        message_passes: int,
    ) -> None:
        super().__init__()

        self.hidden_node_features = hidden_node_features
        self.edge_features = n_edge_features
        self.message_size = message_size
        self.message_passes = message_passes

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
