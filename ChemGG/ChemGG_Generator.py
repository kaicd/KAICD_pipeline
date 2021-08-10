# load general packages and functions
import time
from typing import Tuple
import numpy as np
from tqdm import tqdm
import torch as th
import torch.nn as nn
import rdkit

# load GraphINVENT-specific functions
from ChemGG_MolGraph import ChemGG_GenerationGraph

# defines how to build molecular graphs using the following actions:
#  * "add" a node to graph
#  * "connect" existing nodes in graph
#  * "terminate" graph


class ChemGG_Generator:
    """
    Class for graph generation. Generates graphs in batches using the defined model.
    Optimized for quick generation on a GPU (sacrificed a bit of readability for speed here).
    """
    def __init__(
            self,
            model: nn.Module,
            batch_size: int,
            params: dict
    ) -> None:
        """
        Args:
        ----
            model (nn.Module) : Trained model.
            batch_size (int) : Generation batch size.
        """

        self.start_time = time.time()  # start the timer

        self.params = params
        self.batch_size = batch_size
        self.model = model

        # initializes `self.nodes`, `self.edges`, and `self.n_nodes`, which are
        # tensors for keeping track of the batch of graphs
        self.initialize_graph_batch()

        # allocate tensors for finished graphs; these will get filled in gradually
        # as graphs terminate: `self.generated_nodes`, `self.generated_edges`,
        # `self.generated_n_nodes`, `self.generated_nlls`, and `self.properly_terminated`
        self.allocate_graph_tensors()

    def sample(self) -> Tuple[list, th.Tensor, th.Tensor, th.Tensor]:
        """
        Samples the model for new molecular graphs, and cleans up after `build_graphs()`.
        Returns:
        -------
            graphs (list) : Generated molecular graphs (`GenerationGraphs`s).
            generated_nlls (th.Tensor) : Sampled NLLs per action for generated graphs.
            final_nlls (th.Tensor) : Final total NLLs (sum) for generated graphs.
            properly_terminated (th.Tensor) : Binary vector indicating if graphs were properly
              terminated (or not).
        """
        # build the graphs (these are stored as `self` attributes)
        n_generated_graphs = self.build_graphs()

        # get the time it took to generate graphs
        self.start_time = time.time() - self.start_time
        print(f"Generated {n_generated_graphs} molecules in {self.start_time:.4} s")
        print(f"--{n_generated_graphs/self.start_time:4.5} molecules/s")

        # convert the molecular graphs (currently separate node and edge features tensors) into
        # `GenerationGraph` objects; sometimes `n_generated_graphs` > `self.batch_size`, in
        # which case the extra generated graphs are simply discarded below
        graphs = [self.graph_to_graph(idx) for idx in range(self.batch_size)]

        # sum NLL per action to get the total NLL for each structure; remove extra zero padding
        final_nlls = th.sum(self.generated_nlls, dim=1)[:self.batch_size]

        # remove extra zero padding from NLLs
        generated_nlls = self.generated_nlls[self.generated_nlls != 0]

        # remove extra padding from `properly_terminated` tensor
        properly_terminated = self.properly_terminated[:self.batch_size]

        return graphs, generated_nlls, final_nlls, properly_terminated


    def build_graphs(self) ->  int:
        """
        Builds molecular graphs in batches, starting from empty graphs.
        Returns:
        -------
            n_generated_so_far (int) : Number molecules built (may be >
              `self.batch_size` due to buffer).
        """
        softmax = nn.Softmax(dim=1)

        # keep track of a few things...
        n_generated_so_far = 0
        t_bar = tqdm(total=self.batch_size)
        generation_round = 0

        # generate graphs in a batch, saving graphs when either the terminate action or an
        # invalid action is sampled, until `self.batch_size` number of graphs have been generated
        while n_generated_so_far < self.batch_size:

            # predict the APDs for this batch of graphs
            apd = softmax(self.model(self.nodes, self.edges))

            # sample the actions from the predicted APDs
            add, conn, term, invalid, nlls_just_sampled = self.get_actions(apd)

            # indicate (with a 1) the structures which have been properly terminated
            self.properly_terminated[n_generated_so_far:(n_generated_so_far + len(term))] = 1

            # collect the indices for all structures to write (and reset) this round
            termination_idc = th.cat((term, invalid))

            # never write out the dummy graph at index 0
            termination_idc = termination_idc[termination_idc != 0]

            # copy the graphs indicated by `terminated_idc` to the tensors for
            # finished graphs (i.e. `generated_{nodes/edges}`)
            n_generated_so_far = self.copy_terminated_graphs(termination_idc,
                                                             n_generated_so_far,
                                                             generation_round,
                                                             nlls_just_sampled)

            # apply actions to all graphs (note: applies dummy actions to terminated
            # graphs, since output will be reset anyways)
            self.apply_actions(add, conn, generation_round, nlls_just_sampled)

            # after actions are applied, reset graphs which were set to terminate this round
            self.reset_graphs(termination_idc)

            # update variables for tracking the progress
            t_bar.update(len(termination_idc))
            generation_round += 1

        # done generating
        t_bar.close()

        return n_generated_so_far

    def allocate_graph_tensors(self) -> None:
        """
        Allocates tensors for the node features, edge features, NLLs, and termination
        status for all graphs to be generated. These then get filled in during the
        graph generation process.
        """
        # define tensor shapes
        node_shape = (self.batch_size, *self.params["dim_nodes"])
        edge_shape = (self.batch_size, *self.params["dim_edges"])
        nlls_shape = (self.batch_size, self.params["max_n_nodes"] * 2)  # the 2 is arbitrary

        # allocate a buffer equal to the size of an extra batch
        n_allocate = self.batch_size * 2

        # create the placeholder tensors:

        # placeholder for node features tensor for all graphs
        self.generated_nodes = th.zeros((n_allocate, *node_shape[1:]),
                                           dtype=th.float32
                                        )

        # placeholder for edge features tensor for all graphs
        self.generated_edges = th.zeros((n_allocate, *edge_shape[1:]),
                                           dtype=th.float32
                                        )

        # placeholder for number of nodes per graph in all graphs
        self.generated_n_nodes = th.zeros(n_allocate, dtype=th.int8)

        # placeholder for sampled NLL per action for all graphs
        self.nlls = th.zeros(nlls_shape)

        # placeholder for sampled NLLs per action for all finished graphs
        self.generated_nlls = th.zeros((n_allocate, *nlls_shape[1:]))

        # placeholder for graph termination status (1 == properly terminated, 0 == improper)
        self.properly_terminated = th.zeros(n_allocate, dtype=th.int8)


    def apply_actions(self, add : Tuple[th.Tensor, ...],
                      conn : Tuple[th.Tensor, ...], generation_round : int,
                      nlls_sampled : th.Tensor) -> None:
        """
        Applies the batch of sampled actions (specified by `add` and `conn`) to
        the batch of graphs under construction. Also adds the NLLs for the newly
        sampled actions (`nlls_sampled`) to the running list of NLLs.
        Updates the following tensors:
            self.nodes (th.Tensor) : Updated node features tensor (batch).
            self.edges (th.Tensor) : Updated edge features tensor (batch).
            self.n_nodes (th.Tensor) : Updated number of nodes per graph (batch).
            self.nlls (th.Tensor) : Updated sampled NLL per action for graphs (batch).
        Args:
        ----
            add (tuple) : Indices for "add" actions sampled for batch of graphs.
            conn (tuple) : Indices for "connect" actions sampled for batch of graphs.
            generation_round (int) : Indicates current generation round.
            nlls_sampled (th.Tensor) : NLL per action sampled for the most recent
              set of actions.
        """
        def _add_nodes(add : Tuple[th.Tensor, ...], generation_round : int,
                       nlls_sampled : th.Tensor) -> None:
            """
            Adds new nodes to graphs which sampled the "add" action.
            Args:
            ----
                add (tuple) : Indices for "add" actions sampled for batch of graphs.
                generation_round (int) : Indicates current generation round.
                nlls_sampled (th.Tensor) : NLL per action sampled for the most
                  recent set of actions.
            """
            # get the action indices
            add = [idx.long() for idx in add]
            n_node_features = [self.params["n_atom_types"],
                               self.params["n_formal_charge"],
                               self.params["n_imp_H"],
                               self.params["n_chirality"]]

            if not self.params["use_explicit_H"] and not self.params["ignore_H"]:
                if self.params["use_chirality"]:
                    batch, bond_to, atom_type, charge, imp_h, chirality, bond_type, bond_from = add
                    # add the new nodes to the node features tensors
                    self.nodes[batch, bond_from, atom_type] = 1
                    self.nodes[batch, bond_from, charge + n_node_features[0]] = 1
                    self.nodes[batch, bond_from, imp_h + sum(n_node_features[0:2])] = 1
                    self.nodes[batch, bond_from, chirality + sum(n_node_features[0:3])] = 1
                else:
                    batch, bond_to, atom_type, charge, imp_h, bond_type, bond_from = add
                    # add the new nodes to the node features tensors
                    self.nodes[batch, bond_from, atom_type] = 1
                    self.nodes[batch, bond_from, charge + n_node_features[0]] = 1
                    self.nodes[batch, bond_from, imp_h + sum(n_node_features[0:2])] = 1
            elif self.params["use_chirality"]:
                batch, bond_to, atom_type, charge, chirality, bond_type, bond_from = add
                # add the new nodes to the node features tensors
                self.nodes[batch, bond_from, atom_type] = 1
                self.nodes[batch, bond_from, charge + n_node_features[0]] = 1
                self.nodes[batch, bond_from, chirality + sum(n_node_features[0:2])] = 1
            else:
                batch, bond_to, atom_type, charge, bond_type, bond_from = add
                # add the new nodes to the node features tensors
                self.nodes[batch, bond_from, atom_type] = 1
                self.nodes[batch, bond_from, charge + n_node_features[0]] = 1

            # mask dummy edges (self-loops) introduced from adding node to empty graph
            batch_masked = batch[th.nonzero(self.n_nodes[batch] != 0)]
            bond_to_masked = bond_to[th.nonzero(self.n_nodes[batch] != 0)]
            bond_from_masked = bond_from[th.nonzero(self.n_nodes[batch] != 0)]
            bond_type_masked = bond_type[th.nonzero(self.n_nodes[batch] != 0)]

            # connect newly added nodes to the graphs
            self.edges[batch_masked, bond_to_masked, bond_from_masked, bond_type_masked] = 1
            self.edges[batch_masked, bond_from_masked, bond_to_masked, bond_type_masked] = 1

            # keep track of the newly added node
            self.n_nodes[batch] += 1

            # include the NLLs for the add actions for this generation round
            self.nlls[batch, generation_round] = nlls_sampled[batch]


        def _conn_nodes(conn : Tuple[th.Tensor, ...], generation_round : int,
                        nlls_sampled : th.Tensor) -> None:
            """
            Connects nodes in graphs which sampled the "connect" action.
            Args:
            ----
                conn (tuple) : Indices for "connect" actions sampled for batch of graphs.
                generation_round (int) : Indicates current generation round.
                nlls_sampled (th.Tensor) : NLL per action sampled for the most
                  recent set of actions.
            """
            # get the action indices
            conn = [idx.long() for idx in conn]
            batch, bond_to, bond_type, bond_from = conn

            # apply the connect actions
            self.edges[batch, bond_from, bond_to, bond_type] = 1
            self.edges[batch, bond_to, bond_from, bond_type] = 1

            # include the NLLs for the connect actions for this generation round
            self.nlls[batch, generation_round] = nlls_sampled[batch]

        # first applies the "add" action to all graphs in batch (note: does nothing
        # if a graph did not sample "add")
        _add_nodes(add, generation_round, nlls_sampled)

        # then applies the "connect" action to all graphs in batch (note: does
        # nothing if a graph did not sample "connect")
        _conn_nodes(conn, generation_round, nlls_sampled)

    def copy_terminated_graphs(self, terminate_idc : th.Tensor, n_graphs_generated : int,
                               generation_round : int, nlls_sampled : th.Tensor) -> int:
        """
        Copies terminated graphs (either because "terminate" action sampled, or
        invalid action sampled) to `generated_nodes` and `generated_edges` before
        they are removed from the running batch of graphs being generated.
        Args:
        ----
            terminate_idc (th.Tensor) : Indices for graphs that will terminate this round.
            n_graphs_generated (int) : Number of graphs generated thus far (not including
              those about to be copied).
            generation_round (int) : Indicates the current generation round (running count).
            nlls_sampled (th.Tensor) : NLLs for the newest sampled action for each graph
              in a batch of graphs (not yet included in `nlls`).
        Returns:
        -------
            n_graphs_generated (int) : Number of graphs generated thus far.
        """
        # number of graphs to be terminated
        self.nlls[terminate_idc, generation_round] = nlls_sampled[terminate_idc]

        # number of graphs to be terminated
        n_done_graphs = len(terminate_idc)

        # copy the new graphs to the finished tensors
        nodes_local = self.nodes[terminate_idc]
        edges_local = self.edges[terminate_idc]
        n_nodes_local = self.n_nodes[terminate_idc]
        nlls_local = self.nlls[terminate_idc]

        begin_idx = n_graphs_generated
        end_idx = n_graphs_generated + n_done_graphs
        self.generated_nodes[begin_idx : end_idx] = nodes_local
        self.generated_edges[begin_idx : end_idx] = edges_local
        self.generated_n_nodes[begin_idx : end_idx] = n_nodes_local
        self.generated_nlls[begin_idx : end_idx] = nlls_local

        n_graphs_generated += n_done_graphs

        return n_graphs_generated

    def initialize_graph_batch(self) -> None:
        """
        Initializes a batch of empty graphs (zero `th.Tensor`s) to begin the
        generation process. Creates the following:
            self.nodes (th.Tensor) : Empty node features tensor (batch).
            self.edges (th.Tensor) : Empty edge features tensor (batch).
            self.n_nodes (th.Tensor) : Number of nodes per graph in (batch), currently all 0.
        Also, creates a dummy "non-empty" graph at index 0, so that the models do not
        freak out when they receive entirely zero th tensors as input (haven't found
        a more elegant solution to this problem; without the dummy non-empty graph, there
        is a silent error in the message update function of the MPNNs).
        """
        # define tensor shapes
        node_shape = ([self.batch_size] + self.params["dim_nodes"])
        edge_shape = ([self.batch_size] + self.params["dim_edges"])
        n_nodes_shape = [self.batch_size]

        # initialize tensors
        self.nodes = th.zeros(node_shape, dtype=th.float32)
        self.edges = th.zeros(edge_shape, dtype=th.float32)
        self.n_nodes = th.zeros(n_nodes_shape, dtype=th.int8)

        # add a dummy non-empty graph at idx 0, since models cannot receive purely empty graphs
        self.nodes[0] = th.ones(([1] + self.params["dim_nodes"]))
        self.edges[0, 0, 0, 0] = 1
        self.n_nodes[0] = 1

    def reset_graphs(self, idc : int) -> None:
        """
        Resets the `nodes` and `edges` tensors by reseting graphs which sampled
        invalid actions (indicated by `idc`). Updates the following:
            self.nodes_reset (th.Tensor) : Reset node features tensor (batch).
            self.edges_reset (th.Tensor) : Reset edge features tensor (batch).
            self.n_nodes_reset (th.Tensor) : Reset number of nodes per graph (batch).
            self.nlls_reset (th.Tensor) : Reset sampled NLL per action for graphs (batch).
        Args:
        ----
            idc (int) : Indices corresponding to graphs to reset.
        """
        node_shape = ([self.batch_size] + self.params["dim_nodes"])
        edge_shape = ([self.batch_size] + self.params["dim_edges"])
        n_nodes_shape = ([self.batch_size])
        nlls_shape = ([self.batch_size] + [self.params["max_n_nodes"] * 2])  # the 2 is arbitrary

        # reset the "bad" graphs with zero tensors
        if len(idc) > 0:
            self.nodes[idc] = th.zeros((len(idc), *node_shape[1:]),
                                          dtype=th.float32
                                       )
            self.edges[idc] = th.zeros((len(idc), *edge_shape[1:]),
                                          dtype=th.float32
                                       )
            self.n_nodes[idc] = th.zeros((len(idc), *n_nodes_shape[1:]),
                                            dtype=th.int8
                                         )
            self.nlls[idc] = th.zeros((len(idc), *nlls_shape[1:]),
                                         dtype=th.float32
                                      )

        # create a dummy non-empty graph
        self.nodes[0] = th.ones(([1] + self.params["dim_nodes"]))
        self.edges[0, 0, 0, 0] = 1
        self.n_nodes[0] = 1

    def get_actions(self, apds : th.Tensor) -> Tuple[th.Tensor, ...]:
        """
        Samples the input batch of APDs for a batch of actions to apply to the graphs,
        and separates the action indices.
        Args:
        ----
            apds (th.Tensor) : APDs for a batch of graphs.
        Returns:
        -------
            f_add_idc (th.Tensor) : Indices corresponding to "add" action.
            f_conn_idc (th.Tensor) : Indices corresponding to "connect" action.
            f_term_idc (th.Tensor) : Indices corresponding to "terminate" action.
            invalid_idc (th.Tensor) : Indices corresponding graphs which sampled
              an invalid action.
            nlls (th.Tensor) : NLLs per action corresponding to graphs in batch.
        """
        def _reshape_apd(apds : th.Tensor, batch_size : int) -> Tuple[th.Tensor, ...]:
            """
            Reshapes the input batch of APDs (inverse to flattening).
            Args:
            ----
                apds (th.Tensor) : APDs for a batch of graphs.
                batch_size (int) : Batch size.
            Returns:
            -------
                f_add (th.Tensor) : Reshaped APD segment for "add" action.
                f_conn (th.Tensor) : Reshaped APD segment for "connect" action.
                f_term (th.Tensor) : Reshaped APD segment for "terminate" action.
            """
            # get shapes of "add" and "connect" actions
            f_add_shape = (batch_size, *self.params["dim_f_add"])
            f_conn_shape = (batch_size, *self.params["dim_f_conn"])

            # get ilength of flattened segment of APD corresponding to "add" action
            f_add_size = np.prod(self.params["dim_f_add"])

            # reshape the various APD components
            f_add = th.reshape(apds[:, :f_add_size], f_add_shape)
            f_conn = th.reshape(apds[:, f_add_size:-1], f_conn_shape)
            f_term = apds[:, -1]

            return f_add, f_conn, f_term

        def _sample_apd(apds : th.Tensor, batch_size : int) -> Tuple[th.Tensor, ...]:
            """
            Samples the input APDs for all graphs in the batch.
            Args:
            ----
                apds (th.Tensor) : APDs for a batch of graphs.
                batch_size (int) : Batch size.
            Returns:
            -------
                add_idc (th.Tensor) : Nonzero elements in `f_add`.
                conn_idc (th.Tensor) : Nonzero elements in `f_conn`.
                term_idc (th.Tensor) : Nonzero elements in `f_term`.
                nlls (th.Tensor) : Contains NLLs for samples actions.
            """
            action_probability_distribution = th.distributions.Multinomial(1, probs=apds)
            apd_one_hot = action_probability_distribution.sample()
            f_add, f_conn, f_term = _reshape_apd(apd_one_hot, batch_size)

            nlls = apds[apd_one_hot == 1]

            add_idc = th.nonzero(f_add, as_tuple=True)
            conn_idc = th.nonzero(f_conn, as_tuple=True)
            term_idc = th.nonzero(f_term).view(-1)

            return add_idc, conn_idc, term_idc, nlls

        # sample the APD for all graphs in the batch for action indices
        f_add_idc, f_conn_idc, f_term_idc, nlls = _sample_apd(apds, self.batch_size)

        # get indices for the "add" action
        f_add_from = self.n_nodes[f_add_idc[0]]
        f_add_idc = (*f_add_idc, f_add_from)

        # get indices for the "connect" action
        f_conn_from = self.n_nodes[f_conn_idc[0]] - 1
        f_conn_idc = (*f_conn_idc, f_conn_from)

        # get indices for the invalid add and connect actions
        invalid_idc, max_node_idc = self.get_invalid_actions(f_add_idc, f_conn_idc)

        # change "connect to" index for graphs trying to add more than max num nodes
        f_add_idc[5][max_node_idc] = 0

        return f_add_idc, f_conn_idc, f_term_idc, invalid_idc, nlls


    def get_invalid_actions(self,
                            f_add_idc : Tuple[th.Tensor, ...],
                            f_conn_idc : Tuple[th.Tensor, ...]) \
                            -> Tuple[th.Tensor, th.Tensor]:
        """
        Gets the indices corresponding to any invalid sampled actions.
        Args:
        ----
            f_add_idc (th.Tensor) : Indices for "add" actions for batch of graphs.
            f_conn_idc (th.Tensor) : Indices for the "connect" actions for batch
              of graphs.
        Returns:
        -------
            invalid_action_idc (th.Tensor) : Indices corresponding to all invalid
              actions (include the indices below).
            invalid_action_idc_needing_reset (th.Tensor) : Indices corresponding to
              add actions attempting to add more than the maximum number of nodes.
              These must be treated separately because the "connect to" index needs
              to be reset.
        """
        n_max_nodes = self.params["dim_nodes"][0]

        # empty graphs for which "add" action sampled
        f_add_empty_graphs = th.nonzero(self.n_nodes[f_add_idc[0]] == 0)

        # get invalid indices for when adding a new node to a non-empty graph
        invalid_add_idx_tmp = th.nonzero(f_add_idc[1] >= self.n_nodes[f_add_idc[0]])
        combined = th.cat((invalid_add_idx_tmp, f_add_empty_graphs)).squeeze(1)
        uniques, counts = combined.unique(return_counts=True)
        invalid_add_idc = uniques[counts == 1].unsqueeze(dim=1)  # set difference

        # get invalid indices for when adding a new node to an empty graph
        invalid_add_empty_idc = th.nonzero(f_add_idc[1] != self.n_nodes[f_add_idc[0]])
        combined = th.cat((invalid_add_empty_idc, f_add_empty_graphs)).squeeze(1)
        uniques, counts = combined.unique(return_counts=True)
        invalid_add_empty_idc = uniques[counts > 1].unsqueeze(dim=1)  # set intersection

        # get invalid indices for when adding more nodes than possible
        invalid_madd_idc = th.nonzero(f_add_idc[5] >= n_max_nodes)

        # get invalid indices for when connecting a node to nonexisting node
        invalid_conn_idc = th.nonzero(f_conn_idc[1] >= self.n_nodes[f_conn_idc[0]])

        # get invalid indices for when "connecting" a node in a graph with zero nodes
        invalid_conn_nonex_idc = th.nonzero(self.n_nodes[f_conn_idc[0]] == 0)

        # get invalid indices for when creating self-loops
        invalid_sconn_idc = th.nonzero(f_conn_idc[1] == f_conn_idc[3])

        # get invalid indices for when attemting to add multiple edges
        invalid_dconn_idc = th.nonzero(
            th.sum(self.edges, dim=-1)[f_conn_idc[0].long(),
                                          f_conn_idc[1].long(),
                                          f_conn_idc[-1].long()] == 1
        )

        # only need one invalid index per graph
        invalid_action_idc =th.unique(
            th.cat(
                (f_add_idc[0][invalid_add_idc],
                 f_add_idc[0][invalid_add_empty_idc],
                 f_conn_idc[0][invalid_conn_idc],
                 f_conn_idc[0][invalid_conn_nonex_idc],
                 f_conn_idc[0][invalid_sconn_idc],
                 f_conn_idc[0][invalid_dconn_idc],
                 f_add_idc[0][invalid_madd_idc])
            )
        )

        # keep track of invalid indices which require reseting during the final "apply_action()"
        invalid_action_idc_needing_reset = th.unique(
            th.cat(
                (invalid_madd_idc, f_add_empty_graphs)
            )
        )

        return invalid_action_idc, invalid_action_idc_needing_reset

    def graph_to_graph(self, idx : int) -> ChemGG_GenerationGraph:
        """
        Converts a molecular graph representation from the individual node and edge feature
        tensors into `GenerationGraph` objects.
        Args:
        ----
            idx (int) : Index for the molecular graph to convert.
        Returns:
        -------
            graph (GenerationGraph) : Generated graph.
        """
        def _features_to_atom(node_idx : int, node_features : th.Tensor) -> rdkit.Chem.Atom:
            """
            Converts the node feature vector corresponding to the specified node into
            an atom object.
            Args:
            ----
                node_idx (int) : Index denoting the specific node on the graph to convert.
                node_features (th.Tensor) : Node features tensor for one graph.
            Returns:
            -------
                new_atom (rdkit.Atom) : Atom object corresponding to specified node features.
            """
            # get all the nonzero indices in the specified node feature vector
            nonzero_idc = th.nonzero(node_features[node_idx])

            # determine atom symbol
            atom_idx = nonzero_idc[0]
            atom_type = self.params["atom_types"][atom_idx]

            # initialize atom
            new_atom = rdkit.Chem.Atom(atom_type)

            # determine formal charge
            fc_idx = nonzero_idc[1] - self.params["n_atom_types"]
            formal_charge = self.params["formal_charge"][fc_idx]

            new_atom.SetFormalCharge(formal_charge)  # set property

            # determine number of implicit Hs (if used)
            if not self.params["use_explicit_H"] and not self.params["ignore_H"]:
                total_num_h_idx = (nonzero_idc[2] -
                                   self.params["n_atom_types"] -
                                   self.params["n_formal_charge"])
                total_num_h = self.params["imp_H"][total_num_h_idx]

                new_atom.SetUnsignedProp("_TotalNumHs", total_num_h)  # set property
            elif self.params["ignore_H"]:
                # Hs will be set with structure is "sanitized" (corrected) later in `mol_to_graph()`
                pass

            # determine chirality (if used)
            if self.params["use_chirality"]:
                cip_code_idx = (
                    nonzero_idc[-1]
                    - self.params["n_atom_types"]
                    - self.params["n_formal_charge"]
                    - (not self.params["use_explicit_H"] and not self.params["ignore_H"]) * self.params["n_imp_H"]
                )
                cip_code = self.params["chirality"][cip_code_idx]
                new_atom.SetProp("_CIPCode", cip_code)  # set property

            return new_atom

        def _graph_to_mol(node_features : th.Tensor, edge_features : th.Tensor,
                         n_nodes : int) -> rdkit.Chem.Mol:
            """
            Converts input graph represenetation (node and edge features) into an
            `rdkit.Mol` object.
            Args:
            ----
                node_features (th.Tensor) : Node features tensor.
                edge_features (th.Tensor) : Edge features tensor.
                n_nodes (int) : Number of nodes in the graph representation.
            Returns:
            -------
                molecule (rdkit.Chem.Mol) : Molecule object.
            """

            # create empty editable `rdkit.Chem.Mol` object
            molecule = rdkit.Chem.RWMol()
            node_to_idx = {}

            # add atoms to editable mol object
            for node_idx in range(n_nodes):
                atom_to_add = _features_to_atom(node_idx, node_features)
                molecule_idx = molecule.AddAtom(atom_to_add)
                node_to_idx[node_idx] = molecule_idx

            # add bonds to atoms in editable mol object; to not add the same bond twice
            # (which leads to an error), mask half of the edge features beyond diagonal
            n_max_nodes = self.params["dim_nodes"][0]
            edge_mask = th.triu(
                th.ones((n_max_nodes, n_max_nodes)), diagonal=1
            )
            edge_mask = edge_mask.view(n_max_nodes, n_max_nodes, 1)
            edges_idc = th.nonzero(edge_features * edge_mask)

            for node_idx1, node_idx2, bond_idx in edges_idc:
                molecule.AddBond(
                    node_to_idx[node_idx1.item()],
                    node_to_idx[node_idx2.item()],
                    self.params["int_to_bondtype"][bond_idx.item()],
                )

            try:  # convert editable mol object to non-editable mol object
                molecule.GetMol()
            except AttributeError:  # will throw an error if molecule is `None`
                pass

            if self.params["ignore_H"] and molecule:
                try:  # correct for ignored Hs
                    rdkit.Chem.SanitizeMol(molecule)
                except ValueError:  # throws exception if molecule is too ugly to correct
                    pass

            return molecule

        try:
            # first get the `rdkit.Mol` object corresponding to the selected graph
            mol = _graph_to_mol(self.generated_nodes[idx],
                                self.generated_edges[idx],
                                self.generated_n_nodes[idx])
        except (IndexError, AttributeError):  # raised when graph is empty
            mol = None

        # use the `rdkit.Mol` object, and node and edge features tensors, to get
        # the `GenerationGraph` object
        graph = ChemGG_GenerationGraph(params=self.params,
                                       molecule=mol,
                                       node_features=self.generated_nodes[idx],
                                       edge_features=self.generated_edges[idx])
        return graph