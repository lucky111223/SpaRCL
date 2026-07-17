"""Run the mouse embryo preprocessing and SpaRCL integration workflow."""

from __future__ import annotations

import argparse
from functools import reduce
import json
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from scipy.spatial import cKDTree
import torch

from SpaRCL import train_SpaRCL
from SpaRCL.memory import install_memory_bounded_aggregation


SECTIONS = ("E9.5_E1S1", "E10.5_E2S1", "E11.5_E1S1", "E12.5_E1S1")


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def radius_adjacency(spatial: np.ndarray, radius: float) -> sp.csr_matrix:
    coordinates = np.asarray(spatial)
    neighborhoods = cKDTree(coordinates).query_ball_point(coordinates, r=radius)
    rows = []
    columns = []
    max_degree = 0
    for row, neighbors in enumerate(neighborhoods):
        neighbors = np.asarray(neighbors, dtype=np.int64)
        distances = np.linalg.norm(coordinates[neighbors] - coordinates[row], axis=1)
        neighbors = np.sort(neighbors[(neighbors != row) & (distances < radius)])
        max_degree = max(max_degree, len(neighbors))
        if len(neighbors):
            rows.append(np.full(len(neighbors), row, dtype=np.int64))
            columns.append(neighbors)
    if max_degree > 50:
        raise ValueError("The radius graph exceeds the verified top-50 neighborhood size")
    row_index = np.concatenate(rows) if rows else np.empty(0, dtype=np.int64)
    column_index = np.concatenate(columns) if columns else np.empty(0, dtype=np.int64)
    graph = sp.coo_matrix(
        (np.ones(len(row_index), dtype=np.float32), (row_index, column_index)),
        shape=(len(coordinates), len(coordinates)),
    )
    return (graph + sp.eye(len(coordinates), dtype=np.float32)).tocsr()


def load_mouse_embryo(data_dir: Path, config: dict) -> ad.AnnData:
    batches = []
    adjacency = []
    for section in SECTIONS:
        adata = sc.read_h5ad(data_dir / f"{section}.MOSTA.h5ad")
        if "count" not in adata.layers:
            raise KeyError(f"{section}.MOSTA.h5ad does not contain layers['count']")
        adata.X = sp.csr_matrix(adata.layers["count"])
        adata.layers.clear()
        adata.obs = adata.obs.iloc[:, 0:0].copy()
        adata.var_names_make_unique(join="++")
        adata.obs_names = [f"{name}_{section}" for name in adata.obs_names]

        adjacency.append(radius_adjacency(adata.obsm["spatial"], float(config["radius_cutoff"])))
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(
            adata,
            flavor="seurat_v3",
            n_top_genes=int(config["n_top_genes"]),
        )
        batches.append(adata[:, adata.var["highly_variable"]].copy())

    common_genes = reduce(np.intersect1d, [adata.var_names.to_numpy() for adata in batches])
    expected_genes = int(config["common_genes_in_revision_run"])
    if len(common_genes) != expected_genes:
        raise ValueError(f"Expected {expected_genes} common genes, found {len(common_genes)}")
    batches = [adata[:, common_genes].copy() for adata in batches]
    combined = ad.concat(batches, label="slice_name", keys=SECTIONS, merge="same")
    combined.obs["batch_name"] = combined.obs["slice_name"].astype("category")
    graph = sp.block_diag(adjacency, format="csr")
    combined.uns["edgeList"] = graph.nonzero()
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/mouse_embryo.json"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    adata = load_mouse_embryo(args.data_dir, config)
    training_seed = int(config["training_seed"] if args.seed is None else args.seed)
    install_memory_bounded_aggregation()
    device = torch.device("cuda:0" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    adata = train_SpaRCL(
        adata,
        hidden_dims=tuple(config["hidden_dims"]),
        n_epochs=int(config["n_epochs"]),
        pretrain_epochs=int(config["pretrain_epochs"]),
        update_interval=int(config["update_interval"]),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        gradient_clipping=float(config["gradient_clipping"]),
        positive_top_k=int(config["positive_top_k"]),
        margin=float(config["margin"]),
        lambda_weight=float(config["lambda_weight"]),
        negative_k=int(config["negative_k"]),
        knn_neigh=int(config["knn_neigh"]),
        negative_strategy=config["negative_strategy"],
        iter_comb=[tuple(pair) for pair in config["mnn_iter_comb"]],
        random_seed=training_seed,
        use_label_filter=False,
        deterministic_algorithms=bool(config["deterministic_algorithms"]),
        key_added="SpaRCL",
        device=device,
        verbose=True,
    )

    clustering_seed = int(config["clustering_seed"])
    sc.pp.neighbors(adata, use_rep="SpaRCL", random_state=clustering_seed)
    sc.tl.louvain(
        adata,
        key_added="louvain",
        resolution=float(config["louvain_resolution"]),
        random_state=clustering_seed,
    )
    observed_cluster_count = int(adata.obs["louvain"].nunique())
    expected_cluster_count = int(config["expected_cluster_count"])
    if observed_cluster_count != expected_cluster_count:
        print(
            f"Louvain produced {observed_cluster_count} clusters; "
            f"the paper workflow produced {expected_cluster_count}."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = adata.uns.pop("sparcl_training_diagnostics")
    parameters = adata.uns.pop("sparcl_training_parameters")
    adata.uns.pop("edgeList", None)
    (args.output_dir / "training_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    (args.output_dir / "training_parameters.json").write_text(
        json.dumps(parameters, indent=2), encoding="utf-8"
    )
    (args.output_dir / "clustering_parameters.json").write_text(
        json.dumps(
            {
                "algorithm": "Louvain",
                "embedding": "SpaRCL",
                "resolution": float(config["louvain_resolution"]),
                "random_state": clustering_seed,
                "observed_cluster_count": observed_cluster_count,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    adata.write_h5ad(args.output_dir / "mouse_embryo_sparcl.h5ad", compression="lzf")


if __name__ == "__main__":
    main()
