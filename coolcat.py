from __future__ import annotations

import math
import random
from collections import Counter
from copy import deepcopy
from typing import Any, List, Sequence, Union, Optional

try:
    import pandas as pd
except ImportError:
    pd = None
try:
    import numpy as np
    from numpy.typing import NDArray
except ImportError:
    np = None
    NDArray = Any


class Coolcat:
    r"""
    COOLCAT – entropy-based clustering for categorical data.

    The algorithm minimises the *global* entropy of attribute distributions
    inside each cluster (Barbara *et al.*, PKDD 2002).

    Parameters
    ----------
    n_clusters : int
        Target number of clusters *k*.
    random_state : int | None, default=None
        Seed for the built-in RNG.  Set for reproducible assignments.
    max_iter : int, default=1
        Maximum passes over the data.  One sweep is usually sufficient.
    """

    def __init__(self,
                 n_clusters: int,
                 *,
                 random_state: Optional[int] = None,
                 max_iter: int = 1) -> None:

        self.k: int = n_clusters
        self.rng: random.Random = random.Random(random_state)
        self.max_iter: int = max_iter

        self.labels_: List[int] = []
        self._counts: List[List[Counter[Any]]] = []
        self._sizes: List[int] = []
        self.n_items: int = 0
        self.n_cols: int = 0


    def fit(self, X: Union[pd.DataFrame, np.ndarray, Sequence[Sequence[Any]]]) -> "Coolcat":
        """
        Cluster the *categorical* matrix `X`.

        Parameters
        ----------
        X : pandas.DataFrame | numpy.ndarray | Sequence[Sequence]
            Dataset with shape ``(n_samples, n_features)``.  Every column
            must be **categorical** / hashable.

        Returns
        -------
        self : Coolcat
            Fitted estimator.
        """
        # ---- canonicalise input to a list[list[Any]] ------------------
        X_list: List[List[Any]] = self._to_list_of_lists(X)
        self.n_items, self.n_cols = len(X_list), len(X_list[0])

        # 1) -------- seed selection -----------------------------------
        seeds_idx: List[int] = self._select_seeds(X_list)

        # 2) -------- initialise structures ----------------------------
        self.labels_ = [-1] * self.n_items
        self._counts = [[Counter() for _ in range(self.n_cols)] for _ in range(self.k)]
        self._sizes = [0] * self.k

        for cid, obj_idx in enumerate(seeds_idx):
            self._add_object(cid, X_list[obj_idx])
            self.labels_[obj_idx] = cid

        # 3) -------- assignment passes --------------------------------
        for _ in range(self.max_iter):
            moved = False
            for i, row in enumerate(X_list):
                if self.labels_[i] != -1 and _ == 0:          # skip seeds in 1st pass
                    continue
                best_cid, _best_dH = self._best_cluster_for(row)
                if self.labels_[i] != best_cid:
                    moved = True
                    if self.labels_[i] != -1:
                        self._remove_object(self.labels_[i], row)
                    self._add_object(best_cid, row)
                    self.labels_[i] = best_cid
            if not moved:
                break
        return self

    def fit_predict(self, X: Union[pd.DataFrame, np.ndarray, Sequence[Sequence[Any]]]) -> List[int]:
        """
        Convenience wrapper ``labels_ = fit(X).labels_``.

        Returns
        -------
        labels : list[int]
            Cluster index for every object in *X*.
        """
        return self.fit(X).labels_

    def predict(self,X: Union[pd.DataFrame, np.ndarray, Sequence[Sequence[Any]]]) -> List[int]:
        """
        Assign **new** objects to the learned clusters.

        Parameters
        ----------
        X : same accepted types as :py:meth:`fit`.

        Returns
        -------
        labels : list[int]
            Predicted cluster for each row.
        """
        X_list = self._to_list_of_lists(X)
        return [self._best_cluster_for(row)[0] for row in X_list]


    @staticmethod
    def _to_list_of_lists(X: Union[pd.DataFrame, np.ndarray, Sequence[Sequence[Any]]]) -> List[List[Any]]:
        """Convert the diverse input types to mutable lists of lists."""
        if pd is not None and isinstance(X, pd.DataFrame):
            return X.values.tolist()            # keeps object dtype
        if np is not None and isinstance(X, np.ndarray):
            if X.ndim != 2:
                raise ValueError("NumPy array must be 2-D")
            return X.tolist()

        return [list(row) for row in X]


    @staticmethod
    def _entropy_from_counts(val_counts: Counter[Any], cluster_size: int) -> float:
        """Shannon entropy (log₂) of *one* attribute inside a cluster."""
        if cluster_size == 0:
            return 0.0
        return -sum(
            (c / cluster_size) * math.log2(c / cluster_size)
            for c in val_counts.values()
        )

    def _cluster_entropy(self, counts: List[Counter[Any]], size: int) -> float:
        """Average entropy across the m attributes of a cluster."""
        if size == 0:
            return 0.0
        return sum(self._entropy_from_counts(c, size) for c in counts) / self.n_cols

    def _total_entropy(self, *, except_id: Optional[int] = None) -> float:
        """
        Weighted global entropy **excluding** `except_id` if provided.

        Weights each cluster by its relative size |Cᵢ| / n.
        """
        H: float = 0.0
        for cid, (cnts, sz) in enumerate(zip(self._counts, self._sizes)):
            if cid == except_id or sz == 0:
                continue
            H += (sz / self.n_items) * self._cluster_entropy(cnts, sz)
        return H

    # ---------- seed selection ---------------------------------------- #
    def _select_seeds(self, X: List[List[Any]]) -> List[int]:
        """
        Greedy max-min Hamming distance initialisation (fast surrogate
        for the HU score in the original paper).
        """
        seeds: List[int] = [self.rng.randrange(self.n_items)]
        while len(seeds) < self.k:
            best_idx, best_dist = -1, -1
            for idx, row in enumerate(X):
                if idx in seeds:
                    continue
                d_min = min(
                    sum(a != b for a, b in zip(row, X[s])) for s in seeds
                )
                if d_min > best_dist:
                    best_idx, best_dist = idx, d_min
            seeds.append(best_idx)
        return seeds

    # ---------- bookkeeping ------------------------------------------- #
    def _add_object(self, cid: int, row: Sequence[Any]) -> None:
        for j, val in enumerate(row):
            self._counts[cid][j][val] += 1
        self._sizes[cid] += 1

    def _remove_object(self, cid: int, row: Sequence[Any]) -> None:
        for j, val in enumerate(row):
            c = self._counts[cid][j]
            c[val] -= 1
            if c[val] == 0:
                del c[val]
        self._sizes[cid] -= 1

    # ---------- assignment -------------------------------------------- #
    def _best_cluster_for(self, row: Sequence[Any]) -> tuple[int, float]:
        """
        Return the cluster id that yields the **smallest** ΔH
        if `row` were added to it, along with that ΔH.

        A deepcopy is used for clarity; performance-critical users can
        micro-optimise this section with in-place maths or NumPy.
        """
        H_before: float = self._total_entropy()
        best_cid, best_dH = -1, float("inf")

        for cid in range(self.k):
            tmp_counts = deepcopy(self._counts[cid])
            for j, val in enumerate(row):
                tmp_counts[j][val] += 1
            tmp_size = self._sizes[cid] + 1

            H_after = (
                self._total_entropy(except_id=cid)
                + (tmp_size / self.n_items) * self._cluster_entropy(tmp_counts, tmp_size)
            )
            delta = H_after - H_before
            if delta < best_dH:
                best_cid, best_dH = cid, delta
        return best_cid, best_dH
