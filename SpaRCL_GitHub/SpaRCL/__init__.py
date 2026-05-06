#!/usr/bin/env python
"""
# Author: SpaRCL Team (Jinjie Yang, Xiang Zhou, et al.)
# File Name: __init__.py
# Description: SpaRCL: Robust Spatially-Aware Graph Contrastive Learning for Multi-Slice Spatial Transcriptomics Integration
"""

__author__ = "Jinjie Yang, Xiang Zhou"


# 导入工具函数
from .ST_utils import match_cluster_labels, Cal_Spatial_Net, Stats_Spatial_Net, mclust_R, ICP_align
from .mnn_utils import create_dictionary_mnn

# 导入核心训练函数 (注意这里换成了你改名后的 SpaRCL)
from .train_SpaRCL import train_SpaRCL, pretrain_SpaRCL