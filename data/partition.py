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
    min_client_size: int = 1,
    max_attempts: int = 50,
) -> list[np.ndarray]:
    """Return a list of `num_clients` arrays of row indices into `y`.

    For each class, indices are shuffled then split across clients using
    proportions drawn from Dir(alpha, ..., alpha). Concatenating the
    per-class splits for a given client gives that client's non-IID shard.

    On a small dataset, a low alpha can draw a per-client proportion close
    enough to zero that some client ends up with zero total rows -- never
    happens on the 253k-row diabetes dataset, but happens routinely on a
    ~250-row dataset (e.g. heart disease) split 5 ways. Rather than
    silently returning an empty client (which crashes downstream in
    build_experiment_splits' train_test_split), this re-draws the Dirichlet
    proportions -- consuming further draws from the same seeded rng, so
    still fully deterministic given (seed, alpha, num_clients) -- up to
    `max_attempts` times until every client has >= `min_client_size` rows.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    classes = np.unique(y)

    for _ in range(max_attempts):
        client_indices: list[list[int]] = [[] for _ in range(num_clients)]
        for c in classes:
            idx_c = np.where(y == c)[0]
            rng.shuffle(idx_c)
            proportions = rng.dirichlet(alpha=[alpha] * num_clients)
            split_points = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
            for client_id, split in enumerate(np.split(idx_c, split_points)):
                client_indices[client_id].extend(split.tolist())
        if min(len(idx) for idx in client_indices) >= min_client_size:
            break
    else:
        raise ValueError(
            f"Could not draw a Dirichlet(alpha={alpha}) partition into {num_clients} "
            f"clients where every client has >= {min_client_size} rows, after "
            f"{max_attempts} attempts -- alpha is likely too low for this dataset size."
        )

    result = []
    for indices in client_indices:
        # dtype=int matters for empty shards specifically: np.array([]) defaults
        # to float64, which breaks integer indexing downstream (build_experiment_splits'
        # train_pool_idx[shard]). Never hit on the 253k-row diabetes dataset, but a
        # small dataset (e.g. heart disease, ~250 rows in the train pool) can genuinely
        # draw a near-zero Dirichlet proportion for some client, landing on an empty shard.
        arr = np.array(indices, dtype=int)
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
    min_client_size: int = 10,
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
    # min_client_size scaled to guarantee a non-empty train AND val split below
    # (client_val_size fraction of a too-small shard would otherwise round to 0).
    min_shard = max(min_client_size, int(np.ceil(1 / client_val_size)) + 1)
    client_shards = dirichlet_partition(y_pool, num_clients, alpha, seed=seed, min_client_size=min_shard)

    clients = []
    for shard in client_shards:
        original_idx = train_pool_idx[shard]
        train_idx, val_idx = train_test_split(
            original_idx, test_size=client_val_size, random_state=seed
        )
        clients.append({"train": train_idx, "val": val_idx})

    return {"global_test": global_test_idx, "clients": clients}
