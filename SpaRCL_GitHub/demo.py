import os
import torch
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
import scipy.linalg
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score as ari_score

import SpaRCL
from SpaRCL.ST_utils import Cal_Spatial_Net, mclust_R, match_cluster_labels
from SpaRCL.train_SpaRCL import train_SpaRCL

used_device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

section_ids = ['151673', '151674', '151675', '151676']
print("Processing sections:", section_ids)

Batch_list = []
adj_list = []

for section_id in section_ids:
    print(f"Processing: {section_id}")

    input_dir = os.path.join('/home/yjj/STAligner/继承相邻DLPFC切片', section_id)
    adata = sc.read_visium(path=input_dir, count_file=section_id + '_filtered_feature_bc_matrix.h5', load_images=True)
    adata.var_names_make_unique(join="++")

    Ann_df = pd.read_csv(os.path.join(input_dir, section_id + '_truth.txt'), sep='\t', header=None, index_col=0)
    Ann_df.columns = ['Ground Truth']
    Ann_df[Ann_df.isna()] = "unknown"
    adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth'].astype('category')

    adata.obs_names = [x + '_' + section_id for x in adata.obs_names]

    Cal_Spatial_Net(adata, rad_cutoff=150)

    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable']]

    adj_list.append(adata.uns['adj'])
    Batch_list.append(adata)

adata_concat = ad.concat(Batch_list, label="slice_name", keys=section_ids)
adata_concat.obs['Ground Truth'] = adata_concat.obs['Ground Truth'].astype('category')
adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')
print('adata_concat.shape: ', adata_concat.shape)

adj_concat = np.asarray(adj_list[0].todense())
for batch_id in range(1, len(section_ids)):
    adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))
adata_concat.uns['edgeList'] = np.nonzero(adj_concat)

print("Start training SpaRCL...")
adata_concat = train_SpaRCL(adata_concat, n_epochs=1000, verbose=True, knn_neigh=50, device=used_device)

mclust_R(adata_concat, num_cluster=7, used_obsm='SpaRCL')
adata_concat = adata_concat[adata_concat.obs['Ground Truth'] != 'unknown']

print('mclust, Global ARI = %01.3f' % ari_score(adata_concat.obs['Ground Truth'], adata_concat.obs['mclust']))

sc.pp.neighbors(adata_concat, use_rep='SpaRCL', random_state=666)
sc.tl.umap(adata_concat, random_state=666)

section_color = ['#f8766d', '#7cae00', '#00bfc4', '#c77cff']
section_color_dict = dict(zip(section_ids, section_color))
adata_concat.uns['batch_name_colors'] = [section_color_dict[x] for x in adata_concat.obs.batch_name.cat.categories]

adata_concat.obs['mclust'] = pd.Series(
    match_cluster_labels(adata_concat.obs['Ground Truth'], adata_concat.obs['mclust'].values),
    index=adata_concat.obs.index, dtype='category')

plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams["figure.figsize"] = (3, 3)
plt.rcParams['font.size'] = 12

sc.pl.umap(adata_concat, color=['batch_name', 'Ground Truth', 'mclust'], ncols=3, wspace=0.5, show=True)

Batch_list = [adata_concat[adata_concat.obs['batch_name'] == section_id] for section_id in section_ids]
ARI_list = [round(ari_score(batch.obs['Ground Truth'], batch.obs['mclust']), 2) for batch in Batch_list]

spot_size = 200
title_size = 12

fig, ax = plt.subplots(1, 4, figsize=(10, 5), gridspec_kw={'wspace': 0.05, 'hspace': 0.1})
for i in range(4):
    _sc = sc.pl.spatial(Batch_list[i], img_key=None, color=['mclust'], title=[''],
                        legend_loc=None if i < 3 else 'right margin', legend_fontsize=12,
                        show=False, ax=ax[i], frameon=False, spot_size=spot_size)
    ax[i].set_title(f"ARI={ARI_list[i]}", size=title_size)

plt.show()