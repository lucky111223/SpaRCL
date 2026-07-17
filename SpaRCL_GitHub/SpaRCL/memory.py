"""Optional memory-bounded aggregation used for the mouse embryo workflow."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch_geometric.typing import OptTensor
from torch_scatter import scatter

from .gat_conv import GATConv


class MemoryBoundedGATConv(GATConv):
    """Aggregate wide messages in feature chunks to reduce peak GPU memory."""

    def aggregate(
        self,
        inputs: Tensor,
        index: Tensor,
        ptr: OptTensor = None,
        dim_size: Optional[int] = None,
    ) -> Tensor:
        if inputs.is_cuda and inputs.size(-1) > 32:
            outputs = [
                scatter(
                    chunk,
                    index,
                    dim=self.node_dim,
                    dim_size=dim_size,
                    reduce="sum",
                )
                for chunk in inputs.split(32, dim=-1)
            ]
            return torch.cat(outputs, dim=-1)
        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce="sum")


def install_memory_bounded_aggregation() -> None:
    """Use memory-bounded aggregation for models created in this process."""

    import importlib

    model_module = importlib.import_module("SpaRCL.SpaRCL")
    model_module.GATConv = MemoryBoundedGATConv
