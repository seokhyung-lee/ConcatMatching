import pickle
import random
import sys
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Sequence, Self

import igraph as ig
import networkx as nx
import numpy as np
import pymatching
import scipy as sp
import scipy.sparse as spsp
# import mip
from numpy.typing import NDArray
from networkx.algorithms import bipartite


# try:
#     import parmap
#
#     PARMAP = True
# except ImportError:
#     PARMAP = False


def compress_identical_cols(sparse_matrix: spsp.csc_matrix,
                            *,
                            p: Sequence[float] | None = None) \
        -> Tuple[spsp.csc_matrix, np.ndarray, np.ndarray]:
    # Create a list of row indices representing the non-zero structure of each column
    columns_as_tuples = np.empty(sparse_matrix.shape[1], dtype=object)
    columns_as_tuples[:] = [
        tuple(sparse_matrix.indices[
              sparse_matrix.indptr[i]:sparse_matrix.indptr[i + 1]])
        for i in range(sparse_matrix.shape[1])
    ]

    # Use np.unique to find identical columns based on their row indices
    _, unique_indices, identical_col_groups \
        = np.unique(columns_as_tuples,
                    return_index=True,
                    return_inverse=True)

    # Sort inverse_indices to group identical columns together
    # sorted_indices = np.argsort(inverse_indices)
    # identical_cols = np.split(sorted_indices,
    #                           np.where(np.diff(
    #                               inverse_indices[sorted_indices]))[0] + 1)

    # Compress the weights using np.bincount to sum weights for each unique group
    row_indices = np.arange(unique_indices.shape[0])
    mask = identical_col_groups.reshape(1, -1) == row_indices.reshape(-1, 1)
    if p is None:
        p = np.full(sparse_matrix.shape[1], 1e-3)
    p = np.asanyarray(p, dtype='float64')
    p_masked = p.reshape(1, -1) * mask
    new_p = (1 - (1 - p_masked).prod(axis=1)).ravel()
    # compressed_ps = np.bincount(identical_col_groups, weights=new_p)

    # Use unique_indices directly to select the representative columns without a for loop
    compressed_matrix = sparse_matrix[:, unique_indices]

    return compressed_matrix, new_p, identical_col_groups


def _check_graphlike(H: spsp.csc_matrix) -> bool:
    non_zero_per_col = np.diff(H.indptr)
    return np.all(non_zero_per_col <= 2)


# def greedy_check_filtering(H: spsp.csc_matrix,
#                            seed: int | None = None) -> np.ndarray:
#     np.random.seed(seed)
#
#     num_checks = H.shape[0]
#     checks = np.arange(num_checks)
#     surv_check_filter = np.full(num_checks, False)
#     seen = np.full(num_checks, False)
#     degrees = H.getnnz(axis=0)
#     fault_costs = np.clip(degrees - 2, 0, None)**2
#     check_costs = fault_costs @ H.T
#
#     # select an initial check to exclude
#     init_check_cands \
#         = np.nonzero(check_costs == check_costs.max())[0]
#     init_check = np.random.choice(init_check_cands)
#     current_checks = [init_check]
#     seen[init_check] = True
#
#     fault_degrees = None
#     while not np.all(seen):
#         adj_check_filter \
#             = (H[current_checks, :].getnnz(axis=0) @ H.T > 0) & ~seen
#         adj_checks = np.nonzero(adj_check_filter)[0]
#         # include adj checks not conflicting with checks included previously
#         if fault_degrees is not None:
#             mask = H[adj_checks, :][:, fault_degrees == 2].getnnz(axis=1) == 0
#             included_adj_checks = adj_checks[mask]
#             surv_check_filter[included_adj_checks] = True
#         else:
#             surv_check_filter[adj_checks] = True
#         # exclude adj checks conflicting with each other
#         while True:
#             fault_degrees = H[surv_check_filter, :].getnnz(axis=0)
#             if fault_degrees.max() < 3:
#                 break
#             surv_adj_checks \
#                 = np.nonzero(adj_check_filter & surv_check_filter)[0]
#             adj_check_costs \
#                 = (fault_degrees - 2)**2 @ H[surv_adj_checks, :].T
#             max_cost_adj_checks \
#                 = np.nonzero(adj_check_costs == adj_check_costs.max())[0]
#             excluded_check \
#                 = surv_adj_checks[np.random.choice(max_cost_adj_checks)]
#             surv_check_filter[excluded_check] = False
#         seen[adj_checks] = True
#         current_checks = adj_checks
#
#     return surv_check_filter

def find_largest_indices(nums: Sequence[int]) -> Tuple[int, int]:
    if len(nums) < 2:
        raise ValueError("List must contain at least two elements.")

    # Find the max value and its indices
    max_val = max(nums)
    max_indices = [i for i, x in enumerate(nums) if x == max_val]

    if len(max_indices) == 2:
        return tuple(max_indices)
    elif len(max_indices) > 2:
        return tuple(random.sample(max_indices, 2))
    else:
        second_max_val = max([x for x in nums if x != max_val])
        second_max_indices = [i for i, x in enumerate(nums)
                              if x == second_max_val]
        second_largest_idx = random.choice(second_max_indices)

        return max_indices[0], second_largest_idx


def indirect_random_sort(data: Sequence) -> np.ndarray:
    # Get the array of random values for tie-breaking
    random_vals = np.random.random(len(data))

    # Lexicographically sort by value and then by random values for ties
    sorted_indices = np.lexsort((random_vals, data))

    return sorted_indices


def check_filtering_by_degree(H: spsp.csc_matrix,
                              seed: int | None = None,
                              _debug: bool = False) -> np.ndarray:
    np.random.seed(seed)

    g: ig.Graph = ig.Graph.Biadjacency(H.toarray().tolist())
    fault_costs = np.clip(H.getnnz(axis=0) - 2, 0, None)**2
    check_costs = fault_costs @ H.T
    all_checks = g.vs.select(type=False)
    all_faults = g.vs.select(type=True)
    all_checks['cost'] = check_costs
    all_faults['cost'] = fault_costs
    g.vs['filter'] = None
    g.vs['inc_degree'] = 0
    g.vs['exc_degree'] = 0
    g.vs['unused_degree'] = g.degree()
    g.vs['seen'] = False
    g.vs.select(type=False, _degree=0)['filter'] = False

    def filter_checks(checks: ig.Vertex | ig.VertexSeq, include: bool):
        if isinstance(checks, ig.Vertex):
            assert checks['filter'] is None
        else:
            assert all(value is None for value in checks['filter'])
        checks['filter'] = include
        all_adj_fault_vids = g.neighborhood(checks, mindist=1)
        if isinstance(checks, ig.Vertex):
            all_adj_fault_vids = [all_adj_fault_vids]
        attr = 'inc_degree' if include else 'exc_degree'
        for adj_fault_vids in all_adj_fault_vids:
            adj_faults = g.vs[adj_fault_vids]
            adj_faults[attr] = [deg + 1 for deg in adj_faults[attr]]
            adj_faults['unused_degree'] \
                = [deg - 1 for deg in adj_faults['unused_degree']]

    def all_neighbors(vs: ig.VertexSeq) -> ig.VertexSeq:
        return g.vs[set().union(*g.neighborhood(vs, mindist=1))]

    while True:
        # select a fault with maximal cost randomly
        # and filter surrounding checks
        unseen_faults = all_faults.select(seen=False)
        if not len(unseen_faults):
            break
        max_fault_cost = max(unseen_faults['cost'])
        init_fault: ig.Vertex \
            = np.random.choice(unseen_faults.select(cost=max_fault_cost))
        assert init_fault['inc_degree'] == 0
        assert init_fault['exc_degree'] == 0
        adj_checks = g.vs[g.neighbors(init_fault)]
        if len(adj_checks) < 3:
            filter_checks(adj_checks, True)
        else:
            check_inds_sorted = indirect_random_sort(adj_checks['cost'])
            filter_checks(adj_checks[check_inds_sorted[-2:]], True)
            filter_checks(adj_checks[check_inds_sorted[:-2]], False)
        init_fault['seen'] = adj_checks['seen'] = True

        checks_now = adj_checks
        while True:
            faults_now = all_neighbors(checks_now).select(seen=False)
            if not len(faults_now):
                break
            checks_now = all_neighbors(faults_now).select(seen=False)
            faults_now['seen'] = True
            checks_now['seen'] = True

            faults_now = faults_now.select(unused_degree_ne=0)
            if not len(faults_now):
                continue

            while True:
                # exclude checks with two adjacent included checks
                faults_inc_2 = all_faults.select(inc_degree=2)
                if len(faults_inc_2):
                    excluded_checks = (all_neighbors(faults_inc_2)
                                       .select(filter=None))
                    filter_checks(excluded_checks, False)

                    faults_now = faults_now.select(unused_degree_ne=0)
                    if not len(faults_now):
                        break

                # select a fault for filtering nearby checks
                degrees_sum = (np.array(faults_now['unused_degree'])
                               + np.array(faults_now['inc_degree']))
                inds_min_degree_sum = \
                    np.where(degrees_sum == degrees_sum.min())[0]
                if len(inds_min_degree_sum) == 1:
                    ind = inds_min_degree_sum[0]
                else:
                    degrees = faults_now[inds_min_degree_sum].degree()
                    degrees = np.array(degrees)
                    ind = inds_min_degree_sum[
                        np.random.choice(np.where(degrees == degrees.max())[0])
                    ]
                fault = faults_now[ind]

                # print(f"faults_now: {len(faults_now)}, fault: {ind}, degree = {fault.degree()}, inc_degree = {fault['inc_degree']}, exc_degree = {fault['exc_degree']}, "
                # f"unused_degree = {fault['unused_degree']}", end='')

                # filter checks
                inc_degree = fault['inc_degree']
                assert inc_degree < 2
                adj_checks = g.vs[g.neighbors(fault)].select(filter=None)
                num_adj_checks = len(adj_checks)
                assert num_adj_checks == fault['unused_degree']
                # if len(adj_checks) != fault['unused_degree']:
                #     print(fault['unused_degree'])
                #     for v in g.vs[g.neighbors(fault)]:
                #         print(v)
                #     raise AssertionError
                if num_adj_checks == 1:
                    filter_checks(adj_checks, True)
                else:
                    # check_inds_sorted = indirect_random_sort(
                    #     adj_checks['cost'])
                    # inc_check_inds = check_inds_sorted[-1]
                    # exc_check_inds = check_inds_sorted[:-1]
                    max_cost = max(adj_checks['cost'])
                    inc_check \
                        = np.random.choice(adj_checks.select(cost=max_cost))
                    filter_checks(inc_check, True)

                # print(f"-> {fault['unused_degree']}")
                faults_now = faults_now.select(unused_degree_ne=0)
                if not len(faults_now):
                    break

    # return g
    check_filter = all_checks['filter']
    assert all(value is not None for value in check_filter)
    check_filter = np.array(check_filter, dtype='bool')

    return check_filter


def checks_coloring(H: spsp.csc_matrix,
                    strategy: str | None = None,
                    interchange: str | None = True) \
        -> Tuple[Dict[int, List[int]], str]:
    graph = bipartite.from_biadjacency_matrix(H)
    graph = bipartite.projected_graph(graph, range(H.shape[0]))
    if strategy is None:
        strategies = ['largest_first',
                      'smallest_last',
                      'independent_set',
                      'connected_sequential_bfs',
                      'connected_sequential_dfs',
                      'saturation_largest_first']
    else:
        strategies = [strategy]
    colors = None
    num_colors = None
    opt_strat = None
    for strat in strategies:
        interchange_curr = interchange
        if strat in ['saturation_largest_first', 'independent_set']:
            interchange_curr = False
        colors_curr: dict \
            = nx.coloring.greedy_color(graph,
                                       strategy=strat,
                                       interchange=interchange_curr)
        num_colors_curr = max(list(colors_curr.values())) + 1
        if colors is None or num_colors_curr < num_colors:
            colors = colors_curr
            num_colors = num_colors_curr
            opt_strat = strat

    color_groups = {}
    for check, color in colors.items():
        try:
            color_groups[color].append(check)
        except KeyError:
            color_groups[color] = [check]

    return color_groups, opt_strat


@dataclass
class GraphDecomp:
    H_left: spsp.csc_matrix
    checks_left: np.ndarray
    last_check_id: int
    decomp_Hs: List[spsp.csc_matrix]
    decomp_checks: List[np.ndarray]
    decomp_ps: List[np.ndarray | None]
    init_fault_ids: List[int]
    complete: bool = False


class Decoder:
    H: spsp.csc_matrix
    p: np.ndarray | None
    filtering_strategy: str
    filtering_options: Dict[str, Any]
    graph_decomps: List[GraphDecomp]
    verbose: bool

    def __init__(self,
                 H: np.ndarray | spsp.spmatrix | List[List[bool | int]],
                 *,
                 p: Sequence[float] | None = None,
                 filtering_strategy: str = 'greedy_coloring',
                 filtering_options: Dict[str, Any] | None = None,
                 comparison: bool = False,
                 verbose: bool = False):
        if p is None:
            self.p = None
        else:
            assert len(p) == H.shape[1]
            p = np.array(p, dtype='float64')
            self.p = p

        if not isinstance(H, spsp.csc_matrix):
            H = spsp.csc_matrix(H)
        H = H.astype('bool')

        self.H = H
        self.filtering_strategy = filtering_strategy
        if filtering_options is None:
            filtering_options = {}
        self.filtering_options = filtering_options
        self.comparison = comparison
        self.graph_decomps = []
        self.verbose = verbose

        if verbose:
            print(f"{H.shape[0]} checks, {H.shape[1]} faults")
            print(f"p given: {p is not None}")
            print(f"filtering_strategy = {filtering_strategy}")
            print(f"filtering_options = {filtering_options}")
            print("Start decomposition.")

        self._decomp_full()

    @property
    def num_checks(self) -> int:
        return self.H.shape[0]

    @property
    def num_faults(self) -> int:
        return self.H.shape[1]

    # def _preprocess_H(self, strategy):
    #     H = self.H
    #     if strategy == 'distant_checks_first':
    #         for fault in range(H.shape[1]):
    #             degree = H.indptr[fault + 1] - H.indptr[fault]
    #             if degree > 3:
    #

    def _filter_checks_for_reduced_H(self,
                                     H: spsp.csc_matrix) \
            -> np.ndarray | List[np.ndarray]:
        verbose = self.verbose
        strategy = self.filtering_strategy
        options = self.filtering_options
        comparison = self.comparison

        if strategy == 'intprog_simple':
            model = mip.Model(sense=mip.MAXIMIZE)

            model.verbose = 0
            x = [model.add_var(var_type=mip.BINARY) for _ in range(H.shape[0])]
            if verbose:
                print("    Initialising LIP...")
            for j in range(H.shape[1]):
                mip_sum = mip.xsum(x[i] for i in np.nonzero(H[:, j])[0])
                model += mip_sum <= 2

            model.objective = mip.xsum(x)
            try:
                model.emphasis = options['emphasis']
            except KeyError:
                model.emphasis = 0  # balanced emphasis
            opt_options = options.copy()
            opt_options.pop('emphasis', None)
            if verbose:
                print("    Running LIP... ", end='')
            status = model.optimize(**opt_options, )
            if status == mip.OptimizationStatus.OPTIMAL \
                    or status == mip.OptimizationStatus.FEASIBLE:
                if verbose:
                    print("Success!")
                check_filter = [v.x for v in model.vars]
                check_filter = np.round(np.array(check_filter)).astype(bool)
                if status == mip.OptimizationStatus.FEASIBLE:
                    raise Warning("Decomposition is not optimal.")
            else:
                raise RuntimeError("Decomposition fails.")

        elif strategy == 'intprog_advanced':
            H_col_sum = H.getnnz(axis=0)
            max_check_degree = np.max(H_col_sum)
            opt_options = options.copy()
            opt_options.pop('emphasis', None)

            for l in range(max_check_degree - 2, max_check_degree + 1):
                if verbose:
                    print(f"l = {l}")
                    print("Initializing LIP...")

                model = mip.Model(sense=mip.MAXIMIZE)

                model.verbose = False
                x = [model.add_var(var_type=mip.BINARY) for _ in
                     range(H.shape[0])]
                const_lb = H_col_sum - l
                for j in range(H.shape[1]):
                    mip_sum = mip.xsum(x[i] for i in np.nonzero(H[:, j])[0])
                    model += mip_sum <= 2
                    if const_lb[j] >= 1:
                        model += mip_sum >= const_lb[j]

                model.objective = mip.xsum(x)
                try:
                    model.emphasis = options['emphasis']
                except KeyError:
                    model.emphasis = 0  # emphasis on optimality

                if verbose:
                    print("Running LIP... ", end='')
                status = model.optimize(**opt_options)
                if status == mip.OptimizationStatus.OPTIMAL \
                        or status == mip.OptimizationStatus.FEASIBLE:
                    if verbose:
                        print("Success!")
                    check_filter = [v.x for v in model.vars]
                    check_filter \
                        = np.round(np.array(check_filter)).astype(bool)
                    if status == mip.OptimizationStatus.FEASIBLE:
                        raise Warning("Decomposition is not optimal.")
                    break
                if verbose:
                    print("No solution found.")
            else:
                raise RuntimeError("Fail to find a decomposition.")

        elif strategy == 'greedy_by_degree':
            check_filter = check_filtering_by_degree(H,
                                                     seed=options.get('seed'))

        elif strategy == 'greedy_coloring':
            check_groups, _ = checks_coloring(H, **options)
            num_colors = len(check_groups)
            if num_colors < 3:
                check_filter = np.full(H.shape[0], True)
            else:
                if comparison:
                    if len(check_groups) != 3:
                        raise NotImplementedError(
                            'Currently supports only 3-colorable graphs')
                    check_filter = []
                    for check_group in check_groups.values():
                        check_filter_sng = np.full(H.shape[0], True)
                        check_filter_sng[check_group] = False
                        check_filter.append(check_filter_sng)
                else:
                    num_checks_each_color \
                        = [len(check_groups[c]) for c in range(num_colors)]
                    colors_largest_group \
                        = find_largest_indices(num_checks_each_color)
                    check_filter = np.full(H.shape[0], False)
                    for c in colors_largest_group:
                        check_filter[check_groups[c]] = True

        else:
            raise ValueError("Invalid strategy.")

        return check_filter

    def _decomp_sng_round(self,
                          H: spsp.csc_matrix,
                          check_filter: np.ndarray) \
            -> Tuple[spsp.csc_matrix, spsp.csc_matrix, np.ndarray]:
        ps = self.p

        # H matrix & weights for current round
        H_reduced = H[check_filter, :]
        H_reduced, ps_reduced, col_groups \
            = compress_identical_cols(H_reduced, p=ps)
        try:
            isolated_col = np.nonzero(H_reduced.getnnz(axis=0) == 0)[0][0]
        except IndexError:
            isolated_col = None

        # Leftover for next rounds
        H_left = H[~check_filter, :]
        nrows_add = H_reduced.shape[1]
        if isolated_col is None:
            row_indices = np.arange(nrows_add)
        else:
            row_indices \
                = np.concatenate([np.arange(isolated_col),
                                  np.arange(isolated_col + 1, nrows_add)])
            ps_reduced = np.delete(ps_reduced, isolated_col)
            if isolated_col == 0:
                H_reduced = H_reduced[:, 1:]
            elif isolated_col == nrows_add - 1:
                H_reduced = H_reduced[:, :-1]
            else:
                H_reduced = spsp.hstack([H_reduced[:, isolated_col],
                                         H_reduced[:, isolated_col + 1:]],
                                        format='csc')
            assert H_reduced.shape[1] == len(row_indices)

        H_left_add_rows \
            = col_groups.reshape(1, -1) == row_indices.reshape(-1, 1)
        H_left = spsp.vstack([H_left, H_left_add_rows],
                             format='csc')

        return H_reduced, H_left, ps_reduced

    def _decomp_full(self):
        verbose = self.verbose

        checks_left = np.arange(self.H.shape[0])
        decomp = GraphDecomp(H_left=self.H,
                             checks_left=checks_left,
                             last_check_id=checks_left[-1],
                             decomp_Hs=[],
                             decomp_checks=[],
                             decomp_ps=[],
                             init_fault_ids=[])
        self.graph_decomps = decomps = [decomp]
        i_round = 0
        while True:
            if all(decomp.complete for decomp in decomps):
                break

            for decomp_id in range(len(decomps)):
                decomp = decomps[decomp_id]

                if decomp.complete:
                    continue

                if verbose:
                    print()
                    if len(decomps) == 1:
                        print(f"ROUND {i_round}:")
                    else:
                        print(f"ROUND {i_round} (DECOMP {decomp_id}):")

                if _check_graphlike(decomp.H_left):
                    if verbose:
                        print(f"{decomp.H_left.shape[0]} checks, "
                              f"{decomp.H_left.shape[1]} edges")
                    decomp.decomp_Hs.append(decomp.H_left)
                    decomp.decomp_checks.append(decomp.checks_left)
                    decomp.decomp_ps.append(self.p)
                    decomp.complete = True
                    continue

                check_filters \
                    = self._filter_checks_for_reduced_H(decomp.H_left)
                if not self.comparison:
                    check_filters = [check_filters]
                if len(check_filters) > 1:
                    child_decomps = [deepcopy(decomp)
                                     for _ in range(len(check_filters) - 1)]
                    child_decomp_ids \
                        = list(range(len(decomps),
                                     len(decomps) + len(child_decomps)))
                    decomps.extend(child_decomps)
                    child_decomps = [decomp] + child_decomps
                    child_decomp_ids = [decomp_id] + child_decomp_ids
                else:
                    child_decomps = [decomp]
                    child_decomp_ids = [decomp_id]

                for child_decomp, check_filter, child_decomp_id \
                        in zip(child_decomps, check_filters, child_decomp_ids):
                    H_reduced, H_left, ps_reduced \
                        = self._decomp_sng_round(child_decomp.H_left,
                                                 check_filter)
                    child_decomp.H_left = H_left
                    child_decomp.decomp_Hs.append(H_reduced)
                    child_decomp.decomp_checks.append(
                        child_decomp.checks_left[check_filter])
                    child_decomp.decomp_ps.append(ps_reduced)
                    # left_Hs.append(H_left)

                    child_decomp.checks_left \
                        = child_decomp.checks_left[~check_filter]
                    faults_merged \
                        = np.arange(child_decomp.last_check_id + 1,
                                    child_decomp.last_check_id
                                    + H_reduced.shape[1] + 1)
                    child_decomp.checks_left \
                        = np.concatenate([child_decomp.checks_left,
                                          faults_merged])
                    child_decomp.init_fault_ids.append(
                        child_decomp.last_check_id + 1)
                    child_decomp.last_check_id = faults_merged[-1]

                    if verbose:
                        max_fault_degree = H_left.getnnz(axis=0).max()
                        print(f"CHILD DECOMP {child_decomp_id}")
                        print(f"{H_reduced.shape[0]} checks, "
                              f"{H_reduced.shape[1]} edges, "
                              f"max degree = {max_fault_degree}")

            i_round += 1

    def decode_sng_decomp(self,
                          syndrome: Sequence[bool | int],
                          decomp: GraphDecomp,
                          *,
                          return_weight: bool = False,
                          verbose: bool = False) \
            -> np.ndarray | Tuple[np.ndarray, int]:
        num_stages = len(decomp.decomp_Hs)
        syndrome_full = np.asanyarray(syndrome, dtype='bool')

        assert syndrome_full.ndim == 1

        if verbose:
            print(f"num_stages = {num_stages}")
            print("Start decoding.")

        i_stage = 0
        while i_stage < num_stages:
            if verbose:
                print(f"Stage {i_stage}... ", end="")
            H = decomp.decomp_Hs[i_stage]
            ps = decomp.decomp_ps[i_stage]
            if ps is None:
                weights = None
            else:
                weights = np.log((1 - ps) / ps)
            checks = decomp.decomp_checks[i_stage]
            matching = pymatching.Matching(H, weights=weights)
            syndrome_now = syndrome_full[..., checks]

            preds, sol_weight \
                = matching.decode(syndrome_now, return_weight=True)

            preds = preds.astype('bool')

            if i_stage < num_stages - 1:
                syndrome_full = np.concatenate([syndrome_full, preds], axis=-1)

            if verbose:
                print(f"Success!")

            i_stage += 1

        if return_weight:
            return preds, sol_weight
        else:
            return preds

    def decode(self,
               syndrome: Sequence[bool | int],
               *,
               return_data: bool = False,
               check_validity: bool = False,
               verbose: bool = False) \
            -> np.ndarray | Tuple[np.ndarray, Dict[str, Any]]:
        preds = []
        weights = []
        validities = []
        for decomp_id, decomp in enumerate(self.graph_decomps):
            if verbose:
                print(f">> Decoding DECOMP {decomp_id}...")
            preds_sng, weight = self.decode_sng_decomp(syndrome,
                                                       decomp,
                                                       return_weight=True,
                                                       verbose=verbose)
            preds.append(preds_sng)
            weights.append(weight)
            if check_validity:
                validity = self.check_validity(syndrome, preds_sng)
                validities.append(validity)
                if verbose:
                    print("Valid:", validity)

            if verbose:
                print("Weight =", weight)
                print()

        best_decomp = weights.index(min(weights))
        best_preds = preds[best_decomp]

        if verbose:
            print("Best DECOMP:", best_decomp)
            print("Min weight =", weights[best_decomp])

        if return_data:
            data = {
                'weight': weights[best_decomp],
                'selected_decomp': best_decomp,
                'weight_list': weights,
                'preds_list': preds
            }
            if check_validity:
                data['validity'] = validities

            return best_preds, data
        else:
            return best_preds

    # def decode_mp(self,
    #               syndrome: np.ndarray,
    #               *,
    #               num_procs: int | None = None,
    #               return_weights: bool = False,
    #               verbose: bool = False) \
    #         -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    #     if not PARMAP:
    #         raise ImportError("parmap is not installed.")
    #     num_procs = os.cpu_count() if num_procs is None else num_procs
    #
    #     if verbose:
    #         print(f"num_trials = {syndrome.shape[0]}")
    #         print(f"num_procs = {num_procs}")
    #         print("Start decoding.")
    #
    #     syndrome_splitted = np.array_split(syndrome, num_procs)
    #     results = parmap.map(self.decode,
    #                          syndrome_splitted,
    #                          return_weights=return_weights,
    #                          verbose=False,
    #                          pm_pbar=verbose)
    #     if return_weights:
    #         preds, sol_weights = zip(*results)
    #         preds = np.concatenate(list(preds), axis=0)
    #         sol_weights = np.concatenate(list(sol_weights), axis=0)
    #         return preds, sol_weights
    #     else:
    #         preds = np.concatenate(results, axis=0)
    #         return preds

    def check_validity(self,
                       syndrome: Sequence[bool | int],
                       preds: Sequence[bool | int]) -> bool:
        syndrome = np.asanyarray(syndrome, dtype='bool')
        preds = np.asanyarray(preds, dtype='uint32')
        syndrome_pred = (preds @ self.H.T) % 2
        return np.all(syndrome_pred.astype('bool') == syndrome, axis=-1)

    def save(self, fname: str):
        with open(fname, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, fname: str) -> Self:
        with open(fname, "rb") as f:
            return pickle.load(f)
