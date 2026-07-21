"""Reproducible four-section DLPFC SpaRCL example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch

from SpaRCL import Cal_Spatial_Net, mclust_R, train_SpaRCL


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_sections(data_dir: Path, config: dict):
    sections = config["sections"]
    batches = []
    annotations = []
    edge_rows = []
    edge_cols = []
    offset = 0

    for section in sections:
        section_dir = data_dir / section
        adata = sc.read_visium(
            path=section_dir,
            count_file=f"{section}_filtered_feature_bc_matrix.h5",
            load_images=False,
        )
        adata.var_names_make_unique(join="++")

        truth_path = section_dir / f"{section}_truth.txt"
        truth = None
        if truth_path.exists():
            truth = pd.read_csv(truth_path, sep="\t", header=None, index_col=0).iloc[:, 0]
            truth = truth.reindex(adata.obs_names).fillna("unknown")

        # Seurat v3 HVG selection is applied to counts before normalization.
        sc.pp.highly_variable_genes(
            adata,
            flavor="seurat_v3",
            n_top_genes=int(config["n_top_genes"]),
        )
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata = adata[:, adata.var["highly_variable"]].copy()
        Cal_Spatial_Net(adata, rad_cutoff=float(config["radius_cutoff"]), model="Radius")

        old_names = adata.obs_names.astype(str)
        new_names = pd.Index([f"{name}_{section}" for name in old_names])
        adata.obs_names = new_names
        if truth is not None:
            truth.index = new_names
            annotations.append(truth.rename("Ground Truth"))

        local_rows, local_cols = adata.uns["adj"].nonzero()
        edge_rows.append(np.asarray(local_rows, dtype=np.int64) + offset)
        edge_cols.append(np.asarray(local_cols, dtype=np.int64) + offset)
        offset += adata.n_obs
        batches.append(adata)

    combined = ad.concat(batches, label="slice_name", keys=sections, join="inner")
    combined.obs["batch_name"] = combined.obs["slice_name"].astype("category")
    combined.uns["edgeList"] = (np.concatenate(edge_rows), np.concatenate(edge_cols))
    annotation = pd.concat(annotations).reindex(combined.obs_names) if annotations else None
    return combined, annotation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/dlpfc4.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-mclust", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    adata, annotation = load_sections(args.data_dir, config)

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
        random_seed=args.seed,
        key_added="SpaRCL",
        device=device,
        verbose=True,
    )

    # Attach annotations after training when they are used only for evaluation.
    if annotation is not None:
        adata.obs["Ground Truth"] = annotation.astype(str).to_numpy()
    if args.run_mclust:
        mclust_R(
            adata,
            num_cluster=int(config["cluster_count"]),
            modelNames=config["mclust_model"],
            used_obsm="SpaRCL",
            random_seed=int(config["mclust_seed"]),
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
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {**config, "seed": args.seed},
            indent=2,
        ),
        encoding="utf-8",
    )
    adata.write_h5ad(args.output_dir / "sparcl_result.h5ad")
    print(f"Saved results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
