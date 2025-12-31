import numpy as np, scipy.sparse as sp
from torch_geometric.utils import convert

import torch
from torch_sparse import SparseTensor

# --------------------- DropEdge util ---------------------------------------
def dropedge_sparse_mx(adj: sp.spmatrix, drop_rate: float = 1.0) -> sp.coo_matrix:
    if drop_rate <= 0.0:
        return adj.tocoo()
    adj = adj.tocoo()
    keep = np.random.rand(adj.nnz) > drop_rate
    return sp.coo_matrix((adj.data[keep], (adj.row[keep], adj.col[keep])),
                         shape=adj.shape)

# --------------------- adj 병합 + 정규화 -----------------------------------a
def make_final_adj_scipy(adj_orig, overlay_adj,
                   drop_rate_overlay: float = 0.5,
                   add_self_loop: bool = False,
                   ret_scipy_sparse: bool = False):
    ov_drop = dropedge_sparse_mx(overlay_adj, drop_rate_overlay)
    merged = (adj_orig.tocoo() + ov_drop).tocoo()
    if add_self_loop:
        merged = merged + sp.eye(merged.shape[0], format='coo')
    merged.sum_duplicates()               # 중복 edge weight 합산
    merged.data[:] = 1.0
    if ret_scipy_sparse: # scipy sparse
        return merged
    else:
        return convert.from_scipy_sparse_matrix(merged)[0]

def make_final_adj(adj_orig, overlay_adj,
                   drop_rate_overlay: float = 0.5,
                   add_self_loop: bool = False,
                   ret_scipy_sparse: bool = False,
                   device: torch.device = torch.device("cuda:0"),
                   keep_weight_dict: dict | None = None):
    """
    • adj_orig, overlay_adj : SciPy CSR/COO  **or**  torch_sparse.SparseTensor
    • keep_weight_dict      : {(row, col): keep_weight}  (stochastic filter용)
    """
    # -- SciPy → SparseTensor (1회 변환) ---------------------------------
    to_st = (lambda m:
             SparseTensor.from_scipy(m).coalesce().to(device)
             if isinstance(m, sp.spmatrix) else m.to(device))

    A_orig = to_st(adj_orig)
    A_ov   = to_st(overlay_adj)

    # -- DropEdge -------------------------------------------------------
    if drop_rate_overlay > 0:
        row, col, val = A_ov.coo()
        if keep_weight_dict:
            kw = torch.tensor(
                [keep_weight_dict.get((int(r), int(c)), 0.0) for r, c
                 in zip(row.cpu(), col.cpu())],
                device=val.device, dtype=val.dtype)
            prob = drop_rate_overlay * (1 - kw)
        else:
            prob = torch.full_like(val, drop_rate_overlay)
        keep_mask = torch.rand_like(val) >= prob
        A_ov = SparseTensor(row=row[keep_mask], col=col[keep_mask],
                            value=val[keep_mask], sparse_sizes=A_ov.sizes())

    # -- Merge & self-loop ---------------------------------------------
    A_final = A_orig + A_ov
    if add_self_loop:
        A_final = A_final.set_diag()

    # -- Return ---------------------------------------------------------
    if ret_scipy_sparse:
        return A_final.to_scipy(layout="coo")
    else:
        return A_final.coo()[:2]      # edge_index (row, col)

def filter_overlay_by_feat(overlay_adj: sp.spmatrix,
                           feat: np.ndarray,
                           tau: float = 0.5) -> sp.coo_matrix:
    """cos(feature_i, feature_j) > tau 인 edge만 유지"""
    overlay_adj = overlay_adj.tocoo()
    # cosine 계산을 위해 feature 정규화
    feat_norm = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-9)
    cos_vals = np.einsum('nd,nd->n', feat_norm[overlay_adj.row],
                                      feat_norm[overlay_adj.col])
    keep = cos_vals > tau
    return sp.coo_matrix((overlay_adj.data[keep],
                         (overlay_adj.row[keep], overlay_adj.col[keep])),
                         shape=overlay_adj.shape)

