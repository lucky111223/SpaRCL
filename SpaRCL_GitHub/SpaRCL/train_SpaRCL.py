"""SpaRCL training used for the Biology revision."""

from __future__ import annotations

import random
import time
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from tqdm import tqdm

from .SpaRCL import SpaRCL
from .label_schema import LABEL_COLUMNS


VALID_NEGATIVE_STRATEGIES = ("semi-hard", "random")


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return np.asarray(tensor.detach().cpu().numpy(), dtype=np.float32).copy()


def _set_seed(seed: int, deterministic_algorithms: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(deterministic_algorithms)


def _exact_knn_matches(
    query: np.ndarray,
    reference: np.ndarray,
    query_names: np.ndarray,
    reference_names: np.ndarray,
    k: int,
) -> set[tuple[str, str]]:
    neighbors = (
        NearestNeighbors(n_neighbors=min(k, reference.shape[0]), metric="euclidean", n_jobs=1)
        .fit(reference)
        .kneighbors(query, return_distance=False)
    )
    return {
        (str(query_names[row]), str(reference_names[column]))
        for row, columns in enumerate(neighbors)
        for column in columns
    }


def _deterministic_mnn_dictionary(adata, use_rep: str, batch_name: str, k: int, iter_comb=None):
    """Build the original MNN definition with exact KNN and deterministic edge order."""

    batches = list(adata.obs[batch_name].unique())
    combinations_to_run = list(combinations(range(len(batches)), 2)) if iter_comb is None else list(iter_comb)
    result: dict[str, dict[str, list[str]]] = {}
    for first, second in combinations_to_run:
        first_mask = adata.obs[batch_name].to_numpy() == batches[first]
        second_mask = adata.obs[batch_name].to_numpy() == batches[second]
        first_vectors = np.asarray(adata.obsm[use_rep][first_mask], dtype=np.float32)
        second_vectors = np.asarray(adata.obsm[use_rep][second_mask], dtype=np.float32)
        first_names = adata.obs_names[first_mask].to_numpy()
        second_names = adata.obs_names[second_mask].to_numpy()

        forward = _exact_knn_matches(second_vectors, first_vectors, second_names, first_names, k)
        reverse = _exact_knn_matches(first_vectors, second_vectors, first_names, second_names, k)
        mutual = sorted(forward & {(right, left) for left, right in reverse})

        adjacency: dict[str, list[str]] = {}
        for left, right in mutual:
            adjacency.setdefault(left, []).append(right)
            adjacency.setdefault(right, []).append(left)
        pair_key = f"{batches[first]}_{batches[second]}"
        result[pair_key] = {
            anchor: sorted(neighbors) for anchor, neighbors in sorted(adjacency.items())
        }
    return result


def _dense_float_tensor(x: Any) -> torch.Tensor:
    if sp.issparse(x):
        x = x.toarray()
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


def _spatial_neighbor_sets(edge_list: tuple[np.ndarray, np.ndarray], n_nodes: int) -> list[set[int]]:
    neighbors = [set() for _ in range(n_nodes)]
    for u, v in zip(edge_list[0], edge_list[1]):
        u = int(u)
        v = int(v)
        if u == v:
            continue
        neighbors[u].add(v)
        neighbors[v].add(u)
    return neighbors


def _same_batch_knn(
    embedding: np.ndarray,
    batches: np.ndarray,
    n_neighbors: int,
) -> list[np.ndarray | None]:
    result: list[np.ndarray | None] = [None] * embedding.shape[0]
    for batch in np.unique(batches):
        batch_idx = np.flatnonzero(batches == batch)
        if len(batch_idx) <= 1:
            continue
        n_eff = min(int(n_neighbors) + 1, len(batch_idx))
        model = NearestNeighbors(n_neighbors=n_eff, metric="euclidean", algorithm="auto", n_jobs=1)
        local_neighbors = model.fit(embedding[batch_idx]).kneighbors(return_distance=False)
        for local_row, global_idx in enumerate(batch_idx):
            candidates = batch_idx[local_neighbors[local_row]]
            result[int(global_idx)] = candidates[candidates != global_idx][:n_neighbors]
    return result


def _filter_candidates(
    candidates: np.ndarray,
    anchor_idx: int,
    spatial_neighbors: list[set[int]],
    labels: np.ndarray | None,
) -> np.ndarray:
    filtered = np.asarray(
        [int(c) for c in candidates if int(c) != anchor_idx and int(c) not in spatial_neighbors[anchor_idx]],
        dtype=int,
    )
    if labels is None or filtered.size == 0:
        return filtered

    anchor_label = labels[anchor_idx]
    if anchor_label is None or str(anchor_label).lower() in {"unknown", "nan", "none"}:
        return filtered
    return np.asarray([c for c in filtered if labels[c] != anchor_label], dtype=int)


def train_sparcl(
    adata,
    *,
    hidden_dims: tuple[int, int] = (512, 30),
    n_epochs: int = 1000,
    pretrain_epochs: int = 500,
    lr: float = 0.001,
    key_added: str = "SpaRCL",
    gradient_clipping: float = 5.0,
    weight_decay: float = 0.0001,
    margin: float = 1.0,
    random_seed: int = 0,
    iter_comb=None,
    knn_neigh: int = 100,
    positive_top_k: int = 3,
    negative_k: int = 50,
    negative_strategy: str = "semi-hard",
    lambda_weight: float = 1.0,
    use_label_filter: bool = False,
    label_key: str = "Ground Truth",
    update_interval: int = 100,
    gradient_checkpointing: bool = False,
    record_negative_indices: bool = False,
    checkpoint_dir: str | Path | None = None,
    deterministic_algorithms: bool = True,
    verbose: bool = False,
    device: torch.device | None = None,
):
    """Train SpaRCL and store update diagnostics in ``adata.uns``.

    Positive centroids are detached targets rebuilt every ``update_interval``
    epochs, matching the revised Methods. Semi-hard negatives strictly
    satisfy ``d_pos < d_neg < d_pos + margin``. Anchors without a valid negative
    are skipped.
    """

    if negative_strategy not in VALID_NEGATIVE_STRATEGIES:
        raise ValueError(f"negative_strategy must be one of {VALID_NEGATIVE_STRATEGIES}")
    if positive_top_k < 1 or negative_k < 1 or knn_neigh < 1:
        raise ValueError("positive_top_k, negative_k, and knn_neigh must be positive")
    if not 0 <= pretrain_epochs < n_epochs:
        raise ValueError("pretrain_epochs must be in [0, n_epochs)")

    present_labels = [column for column in LABEL_COLUMNS if column in adata.obs.columns]
    if use_label_filter:
        if label_key not in adata.obs.columns:
            raise ValueError(f"Label filtering requires observation column {label_key!r}")
    elif present_labels:
        raise ValueError(
            "Training data contain annotation columns while use_label_filter=False: "
            + ", ".join(present_labels)
        )

    _set_seed(random_seed, deterministic_algorithms=deterministic_algorithms)
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    start_time = time.time()

    edge_list = adata.uns["edgeList"]
    data = Data(
        edge_index=torch.as_tensor(np.asarray([edge_list[0], edge_list[1]]), dtype=torch.long),
        x=_dense_float_tensor(adata.X),
    ).to(device)
    if gradient_checkpointing:
        data.x.requires_grad_(True)
    model = SpaRCL(hidden_dims=[data.x.shape[1], hidden_dims[0], hidden_dims[1]]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    iterator = tqdm(range(pretrain_epochs), disable=not verbose, desc="SpaRCL pretrain")
    for _ in iterator:
        model.train()
        optimizer.zero_grad()
        if gradient_checkpointing:
            z, reconstruction = checkpoint(model, data.x, data.edge_index, use_reentrant=True)
        else:
            z, reconstruction = model(data.x, data.edge_index)
        loss = F.mse_loss(data.x, reconstruction)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()

    batches = adata.obs["batch_name"].to_numpy()
    labels = adata.obs[label_key].to_numpy() if use_label_filter else None
    section_ids = np.asarray(adata.obs["batch_name"].unique())
    spatial_neighbors = _spatial_neighbor_sets(edge_list, adata.n_obs)
    name_to_idx = {name: idx for idx, name in enumerate(adata.obs_names)}
    batch_indices = {batch: np.flatnonzero(batches == batch) for batch in section_ids}

    mined: dict[str, torch.Tensor] | None = None
    previous_positive_targets: dict[tuple[str, str], np.ndarray] = {}
    diagnostics: list[dict[str, Any]] = []
    negative_pair_records: list[dict[str, Any]] = []
    checkpoint_path = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if checkpoint_path is not None:
        checkpoint_path.mkdir(parents=True, exist_ok=True)

    iterator = tqdm(range(pretrain_epochs, n_epochs), disable=not verbose, desc="SpaRCL align")
    for epoch in iterator:
        if epoch == pretrain_epochs or (epoch - pretrain_epochs) % update_interval == 0:
            model.eval()
            with torch.no_grad():
                z_update, _ = model(data.x, data.edge_index)
            z_np = _as_numpy(z_update)
            adata.obsm["_SpaRCL_refresh"] = z_np
            mnn_dict = _deterministic_mnn_dictionary(
                adata,
                use_rep="_SpaRCL_refresh",
                batch_name="batch_name",
                k=knn_neigh,
                iter_comb=iter_comb,
            )
            same_batch_knn = (
                _same_batch_knn(z_np, batches, negative_k)
                if negative_strategy == "semi-hard"
                else None
            )

            anchor_indices: list[int] = []
            positive_vectors: list[np.ndarray] = []
            negative_indices: list[int] = []
            current_positive_targets: dict[tuple[str, str], np.ndarray] = {}
            anchors_with_positive = 0
            skipped_no_candidates = 0
            skipped_no_semihard = 0

            for pair_key, pair_matches in mnn_dict.items():
                for anchor_name, positive_names in pair_matches.items():
                    anchor_idx = name_to_idx.get(anchor_name)
                    if anchor_idx is None:
                        continue
                    positive_idx = [name_to_idx[name] for name in positive_names if name in name_to_idx]
                    positive_idx = [idx for idx in positive_idx if batches[idx] != batches[anchor_idx]]
                    if not positive_idx:
                        continue

                    anchor_vec = z_np[anchor_idx]
                    positive_distances = np.linalg.norm(z_np[positive_idx] - anchor_vec[None, :], axis=1)
                    selected = np.argsort(positive_distances)[: min(positive_top_k, len(positive_idx))]
                    centroid = z_np[[positive_idx[i] for i in selected]].mean(axis=0).astype(np.float32)
                    target_key = (str(pair_key), str(anchor_name))
                    current_positive_targets[target_key] = centroid
                    anchors_with_positive += 1

                    if negative_strategy == "random":
                        candidates = _filter_candidates(
                            batch_indices[batches[anchor_idx]],
                            anchor_idx,
                            spatial_neighbors,
                            labels,
                        )
                        if candidates.size == 0:
                            skipped_no_candidates += 1
                            continue
                        negative_idx = int(np.random.choice(candidates))
                    else:
                        neighbors = same_batch_knn[anchor_idx] if same_batch_knn is not None else None
                        if neighbors is None or len(neighbors) == 0:
                            skipped_no_candidates += 1
                            continue
                        candidates = _filter_candidates(
                            np.asarray(neighbors, dtype=int),
                            anchor_idx,
                            spatial_neighbors,
                            labels,
                        )
                        if candidates.size == 0:
                            skipped_no_candidates += 1
                            continue
                        negative_distances = np.linalg.norm(z_np[candidates] - anchor_vec[None, :], axis=1)
                        positive_distance = float(np.linalg.norm(anchor_vec - centroid))
                        valid = (negative_distances > positive_distance) & (
                            negative_distances < positive_distance + margin
                        )
                        if not np.any(valid):
                            skipped_no_semihard += 1
                            continue
                        valid_candidates = candidates[valid]
                        valid_distances = negative_distances[valid]
                        negative_idx = int(valid_candidates[np.argmin(valid_distances)])

                    anchor_indices.append(anchor_idx)
                    positive_vectors.append(centroid)
                    negative_indices.append(negative_idx)

            shared_targets = previous_positive_targets.keys() & current_positive_targets.keys()
            drift_values = [
                float(np.linalg.norm(current_positive_targets[key] - previous_positive_targets[key]))
                for key in shared_targets
            ]
            diagnostics.append(
                {
                    "update_epoch": int(epoch),
                    "positive_top_k": int(positive_top_k),
                    "negative_strategy": negative_strategy,
                    "label_filter": bool(use_label_filter),
                    "positive_target_count": int(anchors_with_positive),
                    "drift_overlap_count": int(len(drift_values)),
                    "positive_target_drift_mean": float(np.mean(drift_values)) if drift_values else None,
                    "positive_target_drift_std": float(np.std(drift_values, ddof=0)) if drift_values else None,
                    "effective_triplet_count": int(len(anchor_indices)),
                    "skipped_no_candidates": int(skipped_no_candidates),
                    "skipped_no_semihard": int(skipped_no_semihard),
                }
            )
            previous_positive_targets = current_positive_targets

            if record_negative_indices:
                negative_pair_records.append(
                    {
                        "update_epoch": int(epoch),
                        "anchor_idx": np.asarray(anchor_indices, dtype=np.int32),
                        "negative_idx": np.asarray(negative_indices, dtype=np.int32),
                    }
                )

            if anchor_indices:
                mined = {
                    "anchor_idx": torch.as_tensor(anchor_indices, dtype=torch.long, device=device),
                    "positive_vec": torch.as_tensor(
                        np.stack(positive_vectors), dtype=torch.float32, device=device
                    ),
                    "negative_idx": torch.as_tensor(negative_indices, dtype=torch.long, device=device),
                }
            else:
                mined = None

            if checkpoint_path is not None:
                np.savez_compressed(
                    checkpoint_path / f"embedding_refresh_epoch_{epoch}.npz",
                    embedding=z_np,
                    obs_names=adata.obs_names.astype(str).to_numpy(),
                )
                (checkpoint_path / "training_diagnostics_so_far.json").write_text(
                    json.dumps(diagnostics, indent=2), encoding="utf-8"
                )
                if record_negative_indices:
                    np.savez_compressed(
                        checkpoint_path / f"negative_pairs_epoch_{epoch}.npz",
                        anchor_idx=np.asarray(anchor_indices, dtype=np.int32),
                        negative_idx=np.asarray(negative_indices, dtype=np.int32),
                    )
                torch.save(
                    {
                        "next_epoch": int(epoch),
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "random_seed": int(random_seed),
                    },
                    checkpoint_path / f"checkpoint_refresh_epoch_{epoch}.pt",
                )

        model.train()
        optimizer.zero_grad()
        if gradient_checkpointing:
            z, reconstruction = checkpoint(model, data.x, data.edge_index, use_reentrant=True)
        else:
            z, reconstruction = model(data.x, data.edge_index)
        mse_loss = F.mse_loss(data.x, reconstruction)
        loss = mse_loss
        if mined is not None:
            anchor_vec = z.index_select(0, mined["anchor_idx"])
            negative_vec = z.index_select(0, mined["negative_idx"])
            triplet_loss = F.triplet_margin_loss(
                anchor_vec,
                mined["positive_vec"],
                negative_vec,
                margin=margin,
                p=2,
                reduction="mean",
            )
            loss = mse_loss + lambda_weight * triplet_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        embedding, _ = model(data.x, data.edge_index)
    adata.obsm[key_added] = _as_numpy(embedding)
    adata.uns["sparcl_training_diagnostics"] = diagnostics
    if record_negative_indices:
        adata.uns["sparcl_negative_pair_indices"] = negative_pair_records
    adata.uns["sparcl_training_parameters"] = {
        "hidden_dims": list(hidden_dims),
        "n_epochs": int(n_epochs),
        "pretrain_epochs": int(pretrain_epochs),
        "knn_neigh": int(knn_neigh),
        "margin": float(margin),
        "positive_top_k": int(positive_top_k),
        "negative_k": int(negative_k),
        "lambda_weight": float(lambda_weight),
        "negative_strategy": negative_strategy,
        "use_label_filter": bool(use_label_filter),
        "random_seed": int(random_seed),
        "mnn_search": "exact",
        "mnn_threads": 1,
        "detached_centroid_target": True,
        "update_interval": int(update_interval),
        "gradient_checkpointing": bool(gradient_checkpointing),
        "record_negative_indices": bool(record_negative_indices),
        "checkpoint_dir": str(checkpoint_path) if checkpoint_path is not None else None,
        "deterministic_algorithms": bool(deterministic_algorithms),
        "runtime_seconds": float(time.time() - start_time),
    }
    return adata


def train_SpaRCL(adata, **kwargs):
    """Backward-compatible public name for :func:`train_sparcl`."""

    return train_sparcl(adata, **kwargs)
