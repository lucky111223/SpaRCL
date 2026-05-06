import time
import psutil
import gc
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

from .mnn_utils import create_dictionary_mnn
from .SpaRCL import SpaRCL

import torch
import torch.backends.cudnn as cudnn

cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def pretrain_SpaRCL(adata, hidden_dims=[512, 30], n_epochs=1000, lr=0.001, key_added='SpaRCL_pre',
                    gradient_clipping=5., weight_decay=0.0001, verbose=True,
                    random_seed=0, save_reconstrction=False,
                    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    # seed_everything()
    seed = random_seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    adata.X = sp.csr_matrix(adata.X)

    if 'highly_variable' in adata.var.columns:
        adata_Vars = adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata

    if verbose:
        print('Size of Input: ', adata_Vars.shape)
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")

    # 注意：这里假设 Transfer_pytorch_Data 在你的其他 utils 中已经定义并导入
    data = Transfer_pytorch_Data(adata_Vars)

    model = SpaRCL(hidden_dims=[data.x.shape[1]] + hidden_dims).to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_list = []
    for epoch in tqdm(range(1, n_epochs + 1)):
        model.train()
        optimizer.zero_grad()
        z, out = model(data.x, data.edge_index)
        loss = F.mse_loss(data.x, out)
        loss_list.append(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()

    model.eval()
    z, out = model(data.x, data.edge_index)

    pretrain_rep = z.to('cpu').detach().numpy()
    adata.obsm[key_added] = pretrain_rep

    # 如果 save_loss 未定义，这里可能会报错，建议像原版一样加上 try-except 或确保定义
    try:
        if save_loss:
            adata.uns['SpaRCL_pre_loss'] = loss
    except NameError:
        pass

    if save_reconstrction:
        ReX = out.to('cpu').detach().numpy()
        ReX[ReX < 0] = 0
        adata.layers['SpaRCL_pre_ReX'] = ReX

    return adata


def train_SpaRCL(adata, hidden_dims=[512, 30], n_epochs=1000, lr=0.001, key_added='SpaRCL',
                 gradient_clipping=5., weight_decay=0.0001, margin=1.0, verbose=False,
                 random_seed=666, iter_comb=None, knn_neigh=100, positive_top_k=3, negative_k=50,
                 negative_strategy='semi-hard',  # 负采样策略
                 lambda_weight=1.0,  # 控制三元组损失的权重
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    # ---------------------------
    # 1. 辅助监控函数
    # ---------------------------
    def get_gpu_memory():
        if torch.cuda.is_available() and device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(device) / 1024 ** 3
            cached = torch.cuda.memory_reserved(device) / 1024 ** 3
            return f"GPU内存: {allocated:.2f}GB/{cached:.2f}GB"
        return "GPU内存: 不可用"

    def get_system_info():
        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        return f"CPU: {cpu}%, RAM: {memory.percent}%"

    # ---------------------------
    # 2. 优化的KNN构建函数
    # ---------------------------
    def build_knn_optimized(z_np, batch_arr, negative_k, verbose=False):
        knn_idx = [None] * z_np.shape[0]
        batch_sizes = {}
        for b in np.unique(batch_arr):
            batch_sizes[b] = np.sum(batch_arr == b)

        for i, b in enumerate(np.unique(batch_arr)):
            idx = np.where(batch_arr == b)[0]
            batch_size = len(idx)
            if batch_size == 0: continue

            X = z_np[idx].copy()
            n_neighbors = min(negative_k + 1, batch_size)

            if batch_size > 15000:  # 超大批次
                chunk_size = 5000
                n_chunks = (batch_size + chunk_size - 1) // chunk_size
                for chunk_i in range(n_chunks):
                    start_idx = chunk_i * chunk_size
                    end_idx = min((chunk_i + 1) * chunk_size, batch_size)
                    chunk_indices = np.arange(start_idx, end_idx)
                    X_chunk = X[chunk_indices]
                    nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean', algorithm='ball_tree', n_jobs=1)
                    nn.fit(X)
                    neigh = nn.kneighbors(X_chunk, return_distance=False)
                    neigh = neigh[:, 1:]
                    for local_i, global_i in enumerate(chunk_indices):
                        knn_idx[idx[global_i]] = idx[neigh[local_i]]
                    del nn, neigh, X_chunk
                    gc.collect()
            elif batch_size > 5000:  # 大批次
                nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean', algorithm='ball_tree', leaf_size=50,
                                      n_jobs=2)
                nn.fit(X)
                neigh = nn.kneighbors(X, return_distance=False)
                neigh = neigh[:, 1:]
                for row_i, gi in enumerate(idx):
                    knn_idx[gi] = idx[neigh[row_i]]
                del nn, neigh
            else:  # 中小批次
                nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean', algorithm='auto')
                nn.fit(X)
                neigh = nn.kneighbors(X, return_distance=False)
                neigh = neigh[:, 1:]
                for row_i, gi in enumerate(idx):
                    knn_idx[gi] = idx[neigh[row_i]]
                del nn, neigh
            del X
            gc.collect()
        return knn_idx

    # ---------------------------
    # 3. 初始化
    # ---------------------------
    start_time = time.time()
    n_cells = adata.shape[0]

    print(f"开始训练 - 数据规模: {n_cells:,} 个细胞")
    print(f"参数设置: 负采样={negative_strategy}, Lambda权重={lambda_weight}")  # 打印 lambda 信息
    if verbose:
        print(f"设备: {device}, 初始状态 - {get_system_info()}, {get_gpu_memory()}")

    exclude_spatial_neg = True
    label_key = next((c for c in ['Ground Truth', 'label', 'labels', 'cell_type', 'type']
                      if c in adata.obs.columns), None)

    seed = random_seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    section_ids = np.array(adata.obs['batch_name'].unique())
    edgeList = adata.uns['edgeList']
    data = Data(edge_index=torch.LongTensor(np.array([edgeList[0], edgeList[1]])),
                prune_edge_index=torch.LongTensor(np.array([])),
                x=torch.FloatTensor(adata.X.todense())).to(device)

    model = SpaRCL(hidden_dims=[data.x.shape[1], hidden_dims[0], hidden_dims[1]]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ---------------------------
    # 4. 预训练 (Pretrain) - 仅使用 MSE
    # ---------------------------
    print('Pretrain backbone (MSE only)...')
    pretrain_start = time.time()
    for epoch in tqdm(range(0, 500)):
        model.train()
        optimizer.zero_grad()
        z, out = model(data.x, data.edge_index)
        loss = F.mse_loss(data.x, out)  # 预训练阶段不受 lambda 影响
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()

    print(f"预训练完成，耗时: {(time.time() - pretrain_start) / 60:.2f}分钟")
    with torch.no_grad():
        z, _ = model(data.x, data.edge_index)
    adata.obsm['SpaRCL_pre'] = z.cpu().detach().numpy()

    # ---------------------------
    # 5. 准备工作
    # ---------------------------
    def _build_spatial_neighbor_sets(edge_list, n_nodes: int):
        rows, cols = edge_list
        neigh = [set() for _ in range(n_nodes)]
        for u, v in zip(rows, cols):
            if u == v: continue
            neigh[u].add(v)
        return neigh

    spatial_neigh = _build_spatial_neighbor_sets(edgeList, n_nodes=adata.shape[0]) if exclude_spatial_neg else None
    batch_arr = adata.obs['batch_name'].values
    labels_arr = adata.obs[label_key].values if label_key is not None else None
    mined = None

    # ---------------------------
    # 6. 主训练循环
    # ---------------------------
    print(f'Train with SpaRCL (Lambda={lambda_weight})...')
    main_train_start = time.time()

    for epoch in tqdm(range(500, n_epochs)):
        # === 周期性更新三元组 ===
        if epoch % 100 == 0 or epoch == 500:
            if verbose:
                print(f'\n=== Epoch {epoch}: 更新三元组 ({negative_strategy}) ===')

            with torch.no_grad():
                z, _ = model(data.x, data.edge_index)
            z_np = z.detach().cpu().numpy()

            # 使用专属名字存储中间特征
            adata.obsm['SpaRCL_pre'] = z_np

            # 这里使用统一的 'SpaRCL_pre' 进行 MNN 寻找
            mnn_dict = create_dictionary_mnn(adata, use_rep='SpaRCL_pre', batch_name='batch_name',
                                             k=knn_neigh, iter_comb=iter_comb, verbose=0)

            knn_idx = None
            if negative_strategy == 'semi-hard':
                if verbose: print('   构建同批次KNN...')
                knn_idx = build_knn_optimized(z_np, batch_arr, negative_k, verbose=verbose)

            name2idx = dict(zip(list(adata.obs_names), range(adata.shape[0])))
            anchor_idx_list = []
            pos_vec_list = []
            neg_idx_list = []

            for pair_key in mnn_dict.keys():
                for anchor_name, pos_names in mnn_dict[pair_key].items():
                    a = name2idx.get(anchor_name, None)
                    if a is None: continue

                    pos_idx_all = [name2idx[pn] for pn in pos_names if pn in name2idx]
                    pos_idx_all = [p for p in pos_idx_all if batch_arr[p] != batch_arr[a]]
                    if len(pos_idx_all) == 0: continue

                    a_vec = z_np[a]
                    dists_p = np.linalg.norm(z_np[pos_idx_all] - a_vec[None, :], axis=1)
                    order = np.argsort(dists_p)
                    k_eff = int(min(positive_top_k, len(order)))
                    sel = [pos_idx_all[i] for i in order[:k_eff]]
                    pos_centroid = z_np[sel].mean(axis=0)

                    neg_idx = None
                    if negative_strategy == 'random':
                        same_batch_indices = np.where(batch_arr == batch_arr[a])[0]
                        candidates = same_batch_indices[same_batch_indices != a]
                        if exclude_spatial_neg and spatial_neigh is not None:
                            for _ in range(5):
                                cand = np.random.choice(candidates)
                                if cand not in spatial_neigh[a]:
                                    neg_idx = cand
                                    break
                        else:
                            if len(candidates) > 0:
                                neg_idx = np.random.choice(candidates)

                    elif negative_strategy == 'semi-hard':
                        neighs = knn_idx[a]
                        if neighs is None or len(neighs) == 0: continue
                        cand = np.array(neighs, dtype=int)
                        if spatial_neigh is not None:
                            cand = np.array([c for c in cand if c not in spatial_neigh[a]], dtype=int)
                        if label_key is not None:
                            cand = np.array([c for c in cand if labels_arr[c] != labels_arr[a]], dtype=int)

                        if cand.size > 0:
                            dists_n = np.linalg.norm(z_np[cand] - a_vec[None, :], axis=1)
                            pos_centroid_dist = np.linalg.norm(a_vec - pos_centroid)
                            mask = dists_n > pos_centroid_dist
                            if np.any(mask):
                                neg_idx = cand[mask][np.argmin(dists_n[mask])]
                            else:
                                neg_idx = cand[np.argmin(dists_n)]

                    if neg_idx is not None:
                        anchor_idx_list.append(a)
                        pos_vec_list.append(pos_centroid.astype(np.float32))
                        neg_idx_list.append(int(neg_idx))

            if len(anchor_idx_list) > 0:
                mined = {
                    'anchor_idx': torch.LongTensor(anchor_idx_list).to(device),
                    'pos_vec': torch.from_numpy(np.stack(pos_vec_list)).float().to(device),
                    'neg_idx': torch.LongTensor(neg_idx_list).to(device),
                }
            if verbose:
                print(f'   生成 {len(anchor_idx_list)} 个三元组')

        # === 梯度下降 ===
        if mined is not None and len(mined['anchor_idx']) > 0:
            model.train()
            optimizer.zero_grad()
            z, out = model(data.x, data.edge_index)
            mse_loss = F.mse_loss(data.x, out)

            anchor_arr = z.index_select(0, mined['anchor_idx'])
            positive_arr = mined['pos_vec']
            negative_arr = z.index_select(0, mined['neg_idx'])

            triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
            tri_output = triplet_loss(anchor_arr, positive_arr, negative_arr)

            # --- 修改处：加入 lambda_weight ---
            loss = mse_loss + lambda_weight * tri_output

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        z, _ = model(data.x, data.edge_index)
    adata.obsm[key_added] = z.cpu().detach().numpy()

    total_time = (time.time() - start_time) / 60
    print(f"\n=== 训练完成 (Strategy: {negative_strategy}, Lambda: {lambda_weight}) ===")
    print(f"总耗时: {total_time:.2f}分钟")
    return adata