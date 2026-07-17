# Reproducibility

## Required input fields

The training function expects a concatenated `AnnData` object containing:

- the preprocessed expression matrix in `adata.X`;
- section identifiers in `adata.obs['batch_name']`;
- a block-diagonal spatial edge list in `adata.uns['edgeList']`.

The final embedding is written to `adata.obsm['SpaRCL']`.

## Preprocessing

- DLPFC4: select 5,000 Seurat v3 HVGs from raw counts in each section, then
  normalize to 10,000 counts per spot and apply `log1p`.
- DLPFC12: use the same order with 10,000 HVGs per section.
- Mouse embryo: normalize each section to 10,000 counts per spot, apply
  `log1p`, select 5,000 Seurat v3 HVGs, and retain the 693 genes shared by all
  four sections.

## Dataset settings

| Dataset | Radius | Hidden / latent | K | Margin | Lambda | Negative KNN | MNN KNN | Epochs / pretrain / refresh |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DLPFC4 | 150 | 512 / 30 | 3 | 1.0 | 1.0 | 50 | 100 | 1000 / 500 / 100 |
| DLPFC12 | 150 | 512 / 30 | 3 | 1.0 | 1.0 | 50 | 100 | 1000 / 500 / 100 |
| Mouse embryo | 1.3 | 512 / 30 | 3 | 2.5 | 1.0 | 50 | 100 | 1000 / 500 / 100 |

All experiments use Adam with learning rate `0.001`, weight decay `0.0001`,
and gradient clipping at `5.0`. DLPFC clustering uses mclust model `EEE`, seed
`666`, and seven clusters. Mouse embryo clustering uses Louvain on a neighbor
graph built from the SpaRCL embedding, with resolution `0.5` and seed `666`;
the paper workflow produced 15 clusters.

Machine-readable settings are provided in `configs/`.
