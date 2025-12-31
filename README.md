# Path Cover-based Augmentation for Graph Neural Network

## Short Intro.
This repository includes a powerful tool for augmenting input data for graph neural networks.
The tool is based on k-path cover, which is one of famous graph theoretical concepts.

## Usage
This is an example of using this tool as a python library.
```
from pathcover import cover
from pathcover.utils import make_final_adj_scipy as make_final_adj, filter_overlay_by_feat

def build_overlay(method, nodelist=None):
    """
    return: overlay_adj, keep_weight_dict
    """
    d = cover.adj_overlay(adj, args.k, add_local_edge=False,
                          method=method, ret_mode=args.ret,
                          nodelist=nodelist)
    if args.use_stochastic_filter:
        coo = d.tocoo(); m = coo.row < coo.col
        keep_d = _edge_keep_weights(coo.row[m], coo.col[m], args.tau)
    else:
        d = filter_overlay_by_feat(d, feats_arr, args.tau)
        keep_d = {}
    d.data = d.data * args.alpha
    return d, keep_d
```

method can be `random`, `pagerank`, or `gradient`.
