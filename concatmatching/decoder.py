import numpy as np
import scipy as sp
import scipy.optimize as spop
import pymatching
from typing import Union, Optional, Tuple, List


def compress_identical_cols(sparse_matrix: sp.sparse.csc_matrix,
                            *,
                            weights: Optional[
                                Union[np.ndarray, List[float]]] = None):
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
    columns_as_tuples = np.asanyarray(columns_as_tuples, dtype=object)

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
    compressed_weights = np.bincount(identical_col_groups, weights=weights)

    # Use unique_indices directly to select the representative columns without a for loop
    compressed_matrix = sparse_matrix[:, unique_indices]

    return compressed_matrix, compressed_weights, identical_col_groups


def _check_graphlike(H: sp.sparse.csc_matrix):
    non_zero_per_col = np.diff(H.indptr)
    return np.all(non_zero_per_col <= 2)


class Decoder:
    def __init__(self,
                 H: Union[np.ndarray, sp.sparse.spmatrix],
                 *,
                 p: Optional[Union[np.ndarray, List[float]]] = None,
                 filtering_strategy: str = 'intprog_advanced',
                 filtering_options: Optional[dict] = None,
                 verbose: bool = True):
        if p is None:
            self.weights = None
        else:
            assert len(p) == H.shape[1]
            p = np.array(p, dtype='float64')
            self.weights = -np.log(p / (1 - p))

        if not isinstance(H, sp.sparse.csc_matrix):
            H = sp.sparse.csc_matrix(H)

        self.H = H
        self.filtering_strategy = filtering_strategy
        self.filtering_options = filtering_options

        if verbose:
            print(f"{H.shape[0]} checks, {H.shape[1]} faults")
            print(f"p given: {p is not None}")
            print(f"filtering_strategy = {filtering_strategy}")
            print("Start decomposition:")

        self._decomp_full(verbose=verbose)

    @property
    def num_checks(self) -> int:
        return self.H.shape[0]

    @property
    def num_faults(self) -> int:
        return self.H.shape[1]

    def _filter_checks_for_reduced_H(self,
                                     H: sp.sparse.csc_matrix) \
            -> np.ndarray:
        strategy = self.filtering_strategy
        if strategy == 'intprog_simple':
            c = np.full(self.num_checks, -1)
            bounds = spop.Bounds(lb=0, ub=1)
            const = spop.LinearConstraint(H.T, lb=0, ub=2)
            res = spop.milp(c,
                            integrality=1,
                            bounds=bounds,
                            constraints=const,
                            options=self.filtering_options)
            check_filter = np.round(res['x']).astype(bool)
        elif strategy == 'intprog_advanced':
            c = np.full(H.shape[0], -1)
            bounds = spop.Bounds(lb=0, ub=1)
            H_col_sum = H.sum(axis=0).A1
            max_check_degree = np.max(H_col_sum)
            for l in range(1, max_check_degree + 1):
                const_lb = np.maximum(0, H_col_sum - l)
                const_ub = np.minimum(2, H_col_sum)
                const = spop.LinearConstraint(H.T, lb=const_lb, ub=const_ub)
                res = spop.milp(c,
                                integrality=1,
                                bounds=bounds,
                                constraints=const,
                                options=self.filtering_options)
                if res['success']:
                    check_filter = np.round(res['x']).astype(bool)
                    break
            else:
                raise RuntimeError("Fail to find a decomposition.")

        else:
            raise ValueError("Invalid strategy.")

        return check_filter

    def _decomp_sng_round(self,
                          H: sp.sparse.csc_matrix) \
            -> Tuple[sp.sparse.csc_matrix, sp.sparse.csc_matrix,
            np.ndarray, np.ndarray]:
        weights = self.weights

        check_filter \
            = self._filter_checks_for_reduced_H(H)

        # H matrix & weights for current round
        H_reduced = H[check_filter, :]
        H_reduced, weights_reduced, col_groups \
            = compress_identical_cols(H_reduced, weights=weights)

        # Leftover for next rounds
        H_left = H[~check_filter, :]
        nrows_add = H_reduced.shape[1]
        H_left_add_rows \
            = col_groups.reshape(1, -1) == np.arange(nrows_add).reshape(-1, 1)
        H_left = sp.sparse.vstack([H_left, H_left_add_rows],
                                  format='csc')

        return H_reduced, H_left, weights_reduced, check_filter

    def _decomp_full(self,
                     verbose: bool = True):
        H_left = self.H
        checks_left = np.arange(H_left.shape[0])
        last_check_id = checks_left[-1]

        self.decomp_Hs = decomp_Hs = []
        self.decomp_checks = decomp_checks = []
        self.decomp_weights = decomp_weights = []
        # self.left_Hs = left_Hs = []
        self.init_fault_ids = init_fault_ids = []

        while True:
            if _check_graphlike(H_left):
                if verbose:
                    print(f"    round {len(decomp_Hs)} "
                          f"({H_left.shape[0]} checks, "
                          f"{H_left.shape[1]} edges)")
                decomp_Hs.append(H_left)
                decomp_checks.append(checks_left)
                decomp_weights.append(self.weights)
                break

            if verbose:
                print(f"    round {len(decomp_Hs)} ", end='')

            H_reduced, H_left, weights_reduced, check_filter \
                = self._decomp_sng_round(H_left, checks_left)
            decomp_Hs.append(H_reduced)
            decomp_checks.append(checks_left[check_filter])
            decomp_weights.append(weights_reduced)
            # left_Hs.append(H_left)

            checks_left = checks_left[~check_filter]
            faults_merged = np.arange(last_check_id + 1,
                                      last_check_id + H_reduced.shape[1] + 1)
            checks_left = np.concatenate([checks_left, faults_merged])
            init_fault_ids.append(last_check_id + 1)
            last_check_id = faults_merged[-1]

            if verbose:
                print(f"({H_reduced.shape[0]} checks, "
                      f"{H_reduced.shape[1]} edges)")

    def decode(self,
               syndrome: Union[np.ndarray, List[bool]],
               *,
               return_weights: bool = False,
               return_faults: bool = False) \
            -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        num_stages = len(self.decomp_Hs)
        if isinstance(syndrome, np.ndarray):
            syndrome_full = syndrome.copy().astype('bool')
        else:
            syndrome_full = np.array(syndrome, dtype='bool')

        if syndrome_full.ndim not in {1, 2}:
            raise ValueError("Syndrome should be a 1D or 2D array.")

        batch = syndrome_full.ndim == 2

        for i_stage in range(num_stages):
            H = self.decomp_Hs[i_stage]
            weights = self.decomp_weights[i_stage]
            checks = self.decomp_checks[i_stage]

            matching = pymatching.Matching(H, weights=weights)
            syndrome_now = syndrome_full[..., checks]

            if batch:
                preds, sol_weights \
                    = matching.decode_batch(syndrome_now, return_weights=True)
            else:
                preds, sol_weights \
                    = matching.decode(syndrome_now, return_weight=True)
            preds = preds.astype('bool')

            if i_stage < num_stages - 1:
                syndrome_full = np.concatenate([syndrome_full, preds], axis=-1)

        if return_weights:
            return preds, sol_weights
        else:
            return preds

    def check_validity(self,
                       syndrome: Union[np.ndarray, List[bool]],
                       preds: Union[np.ndarray, List[bool]]) -> bool:
        syndrome = np.asanyarray(syndrome, dtype='bool')
        preds = np.asanyarray(preds, dtype='bool')
        syndrome_pred = (preds @ self.H.T) % 2
        return np.all(syndrome_pred == syndrome, axis=1)
