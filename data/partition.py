"""Dirichlet-distribution non-IID partitioning (Hsu et al., 2019).

Simulates hospitals with different patient populations by splitting the
pooled dataset's indices across clients according to a per-class Dirichlet
draw, controlled by concentration parameter alpha: low alpha -> highly
skewed clients (some see almost only one class), high alpha -> near-IID.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def dirichlet_partition(
    y: pd.Series | np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int = 42,
) -> list[np.ndarray]:
    """Return a list of `num_clients` arrays of row indices into `y`.

    For each class, indices are shuffled then split across clients using
    proportions drawn from Dir(alpha, ..., alpha). Concatenating the
    per-class splits for a given client gives that client's non-IID shard.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    classes = np.unique(y)
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for c in classes:
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        proportions = rng.dirichlet(alpha=[alpha] * num_clients)
        split_points = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        for client_id, split in enumerate(np.split(idx_c, split_points)):
            client_indices[client_id].extend(split.tolist())

    result = []
    for indices in client_indices:
        arr = np.array(indices)
        rng.shuffle(arr)
        result.append(arr)
    return result


def client_label_distribution(
    y: pd.Series | np.ndarray, client_indices: list[np.ndarray]
) -> pd.DataFrame:
    """Return a (num_clients x num_classes) table of label counts per client."""
    y = pd.Series(np.asarray(y))
    rows = {
        f"client_{i}": y.iloc[idx].value_counts().sort_index()
        for i, idx in enumerate(client_indices)
    }
    return pd.DataFrame(rows).T.fillna(0).astype(int)


def build_experiment_splits(
    y: pd.Series,
    num_clients: int,
    alpha: float,
    global_test_size: float = 0.15,
    client_val_size: float = 0.15,
    seed: int = 42,
) -> dict:
    """Build the full experiment split used by every FL condition (local,
    FedAvg, FedProx, +/- DP) and later reused as-is for the MIA audit:

    - `global_test`: a held-out set never used in training by anyone --
      this doubles as the MIA "definitely non-member" pool in Step 8.
    - `clients`: for each of `num_clients` Dirichlet-partitioned shards, a
      local train/val split -- "members" for that client's MIA attack are
      exactly its own `train` indices.

    All indices are positional (0-based, into whatever X/y they were called
    with). Purely a function of (num_clients, alpha, seed) -- deterministic,
    so callers regenerate rather than needing to persist this to disk.
    """
    n = len(y)
    all_idx = np.arange(n)
    train_pool_idx, global_test_idx = train_test_split(
        all_idx, test_size=global_test_size, stratify=y, random_state=seed
    )

    y_pool = y.iloc[train_pool_idx].reset_index(drop=True)
    client_shards = dirichlet_partition(y_pool, num_clients, alpha, seed=seed)

    clients = []
    for shard in client_shards:
        original_idx = train_pool_idx[shard]
        train_idx, val_idx = train_test_split(
            original_idx, test_size=client_val_size, random_state=seed
        )
        clients.append({"train": train_idx, "val": val_idx})

    return {"global_test": global_test_idx, "clients": clients}
