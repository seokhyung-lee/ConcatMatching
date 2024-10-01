import numpy as np
import numpy.typing as npt
import scipy as sp
import scipy.optimize as spop
from typing import Union, Optional


# def H_to_tanner_graph(H: Union[npt.ArrayLike, sp.spmatrix],
#                       *,
#                       p: npt.ArrayLike = None,
#                       weight: npt.ArrayLike = None) -> ig.Graph:
#     g: ig.Graph = ig.Graph.Biadjacency(H)
#     checks = g.vs.select(type=0)
#     faults = g.vs.select(type=1)
#     if p is not None:
#         p = np.asanyarray(p, dtype='float64')
#         faults['weight'] = np.log((1 - p) / p)
#     elif weight is not None:
#         faults['weight'] = weight
#     else:
#         faults['weight'] = 1
#
#     faults['name'] = ['f' + str(i) for i in range(len(faults))]
#     checks['name'] = ['c' + str(i) for i in range(len(checks))]
#
#     return g


def compress_identical_cols(sparse_matrix: sp.sparse.csc_matrix,
                            weights: npt.ArrayLike):
    """
    Find sets of column indices that have identical column vectors in a boolean sparse matrix, compress weights, and return a compressed matrix.

    Parameters:
        sparse_matrix (csc_matrix): A boolean sparse matrix in CSC format.
        weights (array_like): A 1D array of weights with the same number of elements as the number of columns in sparse_matrix.

    Returns:
        - A compressed sparse matrix with one representative column per group of identical columns.
        - A 1D array of compressed weights corresponding to the sum of weights for each unique column group.
        - A list of 1D arrays, where each inner list contains the column indices that have identical or unique column vectors.
    """
    # Create a list of row indices representing the non-zero structure of each column
    columns_as_tuples = [
        tuple(sparse_matrix.indices[
              sparse_matrix.indptr[i]:sparse_matrix.indptr[i + 1]])
        for i in range(sparse_matrix.shape[1])
    ]

    # Use np.unique to find identical columns based on their row indices
    _, unique_indices, inverse_indices = np.unique(columns_as_tuples,
                                                   return_index=True,
                                                   return_inverse=True)

    # Sort inverse_indices to group identical columns together
    sorted_indices = np.argsort(inverse_indices)
    identical_cols = np.split(sorted_indices,
                              np.where(np.diff(
                                  inverse_indices[sorted_indices]))[0] + 1)

    # Compress the weights using np.bincount to sum weights for each unique group
    compressed_weights = np.bincount(inverse_indices, weights=weights)

    # Use unique_indices directly to select the representative columns without a for loop
    compressed_matrix = sparse_matrix[:, unique_indices]

    return compressed_matrix, compressed_weights, identical_cols


class Decoder:
    def __init__(self,
                 H: Union[npt.ArrayLike, sp.sparse.spmatrix],
                 *,
                 p: npt.ArrayLike = None,
                 filtering_strategy: str = 'intprog'):
        if p is None:
            self.weights = 1
        else:
            p = np.array(p, dtype='float64')
            self.weights = -np.log(p / (1 - p))

        self.strategy = filtering_strategy

    def _filter_matgraph_checks(self,
                                H: sp.sparse.csc_matrix,
                                *,
                                options: Optional[dict] = None) -> np.ndarray:
        strategy = self.strategy
        if strategy == 'intprog':
            c = np.full(H.shape[0], -1)
            bounds = spop.Bounds(lb=0, ub=1)
            const = spop.LinearConstraint(H.T, lb=0, ub=2)
            res = spop.milp(c,
                            integrality=1,
                            bounds=bounds,
                            constraints=const,
                            options=options)
            check_filter = np.round(res['x']).astype(bool)
        else:
            raise ValueError

        return check_filter

    def _decomp_sng_round(self,
                          H: sp.sparse.csc_matrix,
                          weights: npt.ArrayLike,
                          *,
                          filtering_options: Optional[dict] = None):
        weights = np.asanyarray(weights, dtype='float64')

        check_filter = self._filter_matgraph_checks(H,
                                                    options=filtering_options)

        # H matrix & weights for current round
        H_matgraph = H[check_filter, :]
        H_matgraph, weights_matgraph, merged_faults \
            = compress_identical_cols(H_matgraph, weights)

        # Leftover for next rounds
        H_left = H[~check_filter, :]
