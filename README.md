# SpaRCL

Implementation of **SpaRCL: Robust Spatially-Aware Graph Contrastive Learning for Multi-Slice Spatial Transcriptomics Integration**.

## Installation

Python 3.9 or 3.10 is recommended. Install PyTorch and PyTorch Geometric for
your CUDA version, then run:

```bash
git clone https://github.com/lucky111223/SpaRCL.git
cd SpaRCL
pip install -r requirements.txt
pip install -e .
```

The optional `mclust_R` function requires R, the R package `mclust`, and
`rpy2`. The additional legacy alignment utilities use the packages listed in
`requirements-optional.txt`.

## Quick test

```bash
python examples/toy_demo.py --device cpu
```

This example creates two small synthetic sections and writes the integrated
embedding and training settings to `outputs/toy/`.

## DLPFC example

Place the four SpatialLIBD/10x Visium sections `151673`, `151674`, `151675`,
and `151676` under one data directory, then run:

```bash
python examples/run_dlpfc4.py \
  --data-dir /path/to/dlpfc \
  --config configs/dlpfc4.json \
  --seed 0 \
  --output-dir outputs/dlpfc4_seed0
```

Add `--run-mclust` when R and `mclust` are available.

## Mouse embryo example

Place the four MOSTA files below in one directory:

```text
E9.5_E1S1.MOSTA.h5ad
E10.5_E2S1.MOSTA.h5ad
E11.5_E1S1.MOSTA.h5ad
E12.5_E1S1.MOSTA.h5ad
```

Run the full preprocessing and integration workflow with:

```bash
python examples/run_mouse_embryo.py \
  --data-dir /path/to/mouse_embryo \
  --config configs/mouse_embryo.json \
  --seed 0 \
  --output-dir outputs/mouse_embryo_seed0
```

The mouse embryo example contains 95,896 spots and uses memory-bounded graph
aggregation for an 8 GB GPU. It then applies Louvain clustering to the SpaRCL
embedding with resolution 0.5 and random state 666.

## Main settings

The reported experiments use a hidden width of 512, a latent dimension of 30,
1000 training epochs, 500 reconstruction-pretraining epochs, and a triplet
refresh interval of 100 epochs. Dataset-specific graph and sampling parameters
are provided in `configs/`.

The training output is stored in `adata.obsm['SpaRCL']`. Effective parameters
and refresh diagnostics are stored in `adata.uns['sparcl_training_parameters']`
and `adata.uns['sparcl_training_diagnostics']`, respectively.

See `docs/REPRODUCIBILITY.md` for input fields, preprocessing details, and the
dataset parameter table.

## Compared methods

The manuscript compares SpaRCL with Harmony, PASTE, STAMP, SpaCross, and
STAligner. These methods are available from their respective repositories.
