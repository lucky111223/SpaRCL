"""Small SpaRCL smoke test with synthetic data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import torch

from SpaRCL import train_SpaRCL


def make_toy_data(seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    spots_per_section = 24
    genes = 12
    first = rng.normal(0.0, 0.4, size=(spots_per_section, genes))
    second = first + rng.normal(0.0, 0.15, size=(spots_per_section, genes))
    x = np.maximum(np.vstack([first, second]) + 1.0, 0.0).astype(np.float32)

    obs_names = [f"A_{i:02d}" for i in range(spots_per_section)] + [
        f"B_{i:02d}" for i in range(spots_per_section)
    ]
    adata = ad.AnnData(X=x)
    adata.obs_names = obs_names
    adata.var_names = [f"gene_{i:02d}" for i in range(genes)]
    adata.obs["batch_name"] = ["section_A"] * spots_per_section + ["section_B"] * spots_per_section

    rows: list[int] = []
    cols: list[int] = []
    for offset in (0, spots_per_section):
        for i in range(spots_per_section):
            rows.append(offset + i)
            cols.append(offset + i)
            if i + 1 < spots_per_section:
                rows.extend([offset + i, offset + i + 1])
                cols.extend([offset + i + 1, offset + i])
    adata.uns["edgeList"] = (np.asarray(rows), np.asarray(cols))
    return adata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/toy"))
    args = parser.parse_args()

    device = torch.device("cuda:0" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    adata = make_toy_data()
    adata = train_SpaRCL(
        adata,
        hidden_dims=(32, 8),
        n_epochs=8,
        pretrain_epochs=3,
        update_interval=2,
        knn_neigh=3,
        positive_top_k=2,
        negative_k=8,
        margin=5.0,
        lambda_weight=0.2,
        random_seed=0,
        use_label_filter=False,
        device=device,
        verbose=True,
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
    adata.write_h5ad(args.output_dir / "toy_result.h5ad")
    print(f"SpaRCL embedding: {adata.obsm['SpaRCL'].shape}")
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
