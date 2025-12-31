# ============================================================
# train_lagcn_overlay.py   (2025-07-01 업데이트)
# ============================================================
from __future__ import division, print_function
import argparse, os, copy, random
import numpy as np, scipy.sparse as sp, torch, torch.nn.functional as F
import torch.optim as optim
from tqdm import trange

from utils import (load_data, accuracy, normalize_adj, normalize_features,
                   sparse_mx_to_torch_sparse_tensor)
from gcn.models_ours import LAGCN
import cvae_pretrain

# ------------------------- 전역 시드 -------------------------
def set_global_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

# ------------------------- CLI -------------------------------
parser = argparse.ArgumentParser(
    description="Deterministic LAGCN with k-path overlay & online refresh")
# (기존 인자)
parser.add_argument("--samples", type=int, default=4)
parser.add_argument("--concat", type=int, default=4)
parser.add_argument("--runs", type=int, default=10)

parser.add_argument("--dataset", default="cora")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--hidden", type=int, default=8)
parser.add_argument("--dropout", type=float, default=0.5)

parser.add_argument("--tem", type=float, default=0.5)
parser.add_argument("--lam", type=float, default=1.0)

parser.add_argument("--ret", type=str, default="acc")
parser.add_argument("--method", type=str, default="pagerank")
parser.add_argument("--dr", type=float, default=0.3)
parser.add_argument("--tau", type=float, default=0.5)
parser.add_argument("--alpha", type=float, default=0.1)
parser.add_argument("--k", type=int, default=3)

parser.add_argument("--use_mlp", action="store_true")
parser.add_argument("--use_original_for_last", action="store_true")
parser.add_argument("--use_original_for_eval", action="store_true")
parser.add_argument("--use_warmup", action="store_true")

parser.add_argument("--warmup_epochs", type=int, default=1000)
parser.add_argument("--refresh_period", type=int, default=0)

parser.add_argument("--use_test_aug", action="store_true")
parser.add_argument("--test_aug_runs", type=int, default=10)

# ★★ 새 옵션
parser.add_argument("--use_global_dropedge", action="store_true")
parser.add_argument("--use_stochastic_filter", action="store_true")

args = parser.parse_args()

# ------------------------- 시드·디바이스 ---------------------
set_global_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng = np.random.default_rng(args.seed)

# ------------------------- Consistency -----------------------
def consis_loss(logps, temp=args.tem):
    ps = [torch.exp(p) for p in logps]
    avg = sum(ps) / len(ps)
    sharp = (avg.pow(1/temp) /
             torch.sum(avg.pow(1/temp), 1, keepdim=True)).detach()
    return args.lam * sum([(p-sharp).pow(2).sum(1).mean() for p in ps]) / len(ps)

# ------------------------- 데이터 ----------------------------
adj, feats, idx_tr, idx_va, idx_te, lbl_np = load_data(args.dataset)
feats_arr = feats.toarray()
adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
feats_norm_arr = normalize_features(feats_arr)

labels = torch.LongTensor(lbl_np).max(1)[1].to(device)
feats_norm = torch.FloatTensor(feats_norm_arr).to(device)
adj_normalized = sparse_mx_to_torch_sparse_tensor(adj_norm).to(device)
idx_train = torch.LongTensor(idx_tr).to(device)
idx_val   = torch.LongTensor(idx_va).to(device)
idx_test  = torch.LongTensor(idx_te).to(device)

# ------------------------- CVAE 증강 ------------------------
proj_root = os.path.dirname(os.path.abspath(__file__))
cvae = torch.load(f"{proj_root}/model/{args.dataset}.pkl")
def get_aug_feats(n_sets: int):
    base = torch.tensor(feats_arr, dtype=torch.float32, device=device)
    outs = []
    for _ in range(n_sets):
        z = torch.randn([base.size(0), cvae.latent_size], device=device)
        aug = cvae_pretrain.feature_tensor_normalize(
            cvae.inference(z, base)).detach()
        outs.append(aug)
    return outs

# ------------------ 노드 리스트 -----------------------------
def gradient_nodelist_from_model(model):
    model.eval(); model.zero_grad()
    feat_grad = feats_norm.clone().detach().requires_grad_(True)
    logit = model(get_aug_feats(args.concat)+[feat_grad], adj_normalized)
    F.nll_loss(torch.log_softmax(logit,1)[idx_train],
               labels[idx_train]).backward()
    score = torch.norm(feat_grad.grad.cpu(), 2, 1)
    return torch.argsort(score, descending=True).tolist()

def gradient_nodelist_warmup(ep):
    warm = LAGCN(concat=args.concat+1, nfeat=feats_norm.size(1),
                 nhid=args.hidden, nclass=labels.max().item()+1,
                 dropout=args.dropout, use_mlp=args.use_mlp).to(device)
    opt = optim.Adam(warm.parameters(), lr=args.lr,
                     weight_decay=args.weight_decay)
    for _ in range(ep):
        warm.train(); opt.zero_grad()
        out = warm(get_aug_feats(args.concat)+[feats_norm], adj_normalized)
        F.nll_loss(torch.log_softmax(out,1)[idx_train],
                   labels[idx_train]).backward()
        opt.step()
    return gradient_nodelist_from_model(warm), warm

def random_nodelist():
    lst = list(range(adj.shape[0])); rng.shuffle(lst); return lst

# ---------- keep-weight 계산 util ---------------------------
def _edge_keep_weights(rows, cols, tau):
    """cosine(sim)/τ ∈ [0,1]"""
    f_r, f_c = feats_arr[rows], feats_arr[cols]
    sim = np.einsum("ij,ij->i", f_r, f_c) / (
        np.linalg.norm(f_r,2,1)*np.linalg.norm(f_c,2,1)+1e-9)
    keep = np.clip(sim / tau, 0.0, 1.0)
    d = {}
    for r, c, w in zip(rows, cols, keep):
        if r == c: continue
        d[(r, c)] = d[(c, r)] = float(w)
    return d

# ----------- 원본 그래프 keep-weight (global-on 전용) ---------
orig_keep = {}
if args.use_stochastic_filter and args.use_global_dropedge:
    coo = adj.tocoo(); mask = coo.row < coo.col
    orig_keep = _edge_keep_weights(coo.row[mask], coo.col[mask], args.tau)

# -------------------------- overlay -------------------------
from pathcover import cover
from pathcover.utils import make_final_adj_scipy as make_final_adj, filter_overlay_by_feat

def build_overlay(method, nodelist=None):
    """
    return: overlay_adj, keep_weight_dict
    - use_stochastic_filter=True  → edge 삭제 없이 keep_weight만 계산
    - False                       → τ 임계 이하 edge 삭제
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

# ------------- weighted / uniform drop-edge -----------------
def _weighted_drop(adj_csr, drop_rate, keep_dict):
    if drop_rate <= 0 or adj_csr.nnz == 0: return adj_csr
    coo = adj_csr.tocoo(); rows, cols, data = coo.row, coo.col, coo.data
    keep_rows, keep_cols, keep_data = [], [], []
    up = np.where(rows < cols)[0]
    for i in up:
        r, c = int(rows[i]), int(cols[i])
        kw = keep_dict.get((r, c), 0.0)
        if rng.random() >= drop_rate * (1 - kw):   # keep
            keep_rows.append(r); keep_cols.append(c); keep_data.append(data[i])
    # 대칭 및 self-loop
    keep_rows += keep_cols + list(rows[rows==cols])
    keep_cols += keep_rows[:len(keep_cols)] + list(cols[rows==cols])
    keep_data += keep_data + list(data[rows==cols])
    new = sp.coo_matrix((keep_data, (keep_rows, keep_cols)),
                        shape=adj_csr.shape)
    new.eliminate_zeros(); return new.tocsr()

def _uniform_drop(adj_csr, drop_rate):
    if drop_rate <= 0: return adj_csr
    coo = adj_csr.tocoo(); up = np.where(coo.row < coo.col)[0]
    keep = rng.choice(up, int(len(up)*(1-drop_rate)), replace=False)
    mask = np.zeros(coo.nnz, bool); mask[keep] = True
    pair = {(coo.row[i], coo.col[i]) for i in keep}
    for i in range(coo.nnz):
        if (coo.col[i], coo.row[i]) in pair or coo.row[i]==coo.col[i]:
            mask[i] = True
    new = sp.coo_matrix((coo.data[mask], (coo.row[mask], coo.col[mask])),
                        shape=adj_csr.shape)
    new.eliminate_zeros(); return new.tocsr()

# -------- final adj 생성 ------------------------------------
def make_final(orig_adj, orig_keep, ov_adj, ov_keep,
               drop_rate, global_drop=False):
    if not args.use_stochastic_filter:
        merged = make_final_adj(orig_adj, ov_adj,
                                drop_rate if not global_drop else 0,
                                add_self_loop=True, ret_scipy_sparse=True)
        if global_drop and drop_rate:
            merged = _uniform_drop(merged, drop_rate)
        return merged

    # stochastic mode
    if global_drop:
        merged = make_final_adj(orig_adj, ov_adj, 0,
                                add_self_loop=True, ret_scipy_sparse=True)
        merged = _weighted_drop(merged, drop_rate,
                                {**orig_keep, **ov_keep})
    else:
        ov_dropped = _weighted_drop(ov_adj, drop_rate, ov_keep)
        merged = make_final_adj(orig_adj, ov_dropped, 0,
                                add_self_loop=True, ret_scipy_sparse=True)
    return merged

def to_tensor(sp_adj):
    return sparse_mx_to_torch_sparse_tensor(normalize_adj(sp_adj))

# --------------------- 학습 루프 -----------------------------
val_scores, test_scores = [], []
set_global_seed(args.seed)
for _ in trange(args.runs, desc="runs", leave=False):

    # ------------------ 초기 overlay -----------------------------
    method = args.method.lower()
    if method == "gradient":
        print(f"⇨ warm-up {args.warmup_epochs} epochs for gradient nodelist …")
        nl, warm = gradient_nodelist_warmup(args.warmup_epochs)
        ov_adj, ov_keep = build_overlay("pagerank", nl)
    elif method == "random":
        nl = random_nodelist()
        ov_adj, ov_keep = build_overlay("pagerank", nl)
    else:
        nl = None
        ov_adj, ov_keep = build_overlay(method)

    print(f"overlay edges after filter: {ov_adj.nnz}")

    final_adj_norm = (adj_normalized if args.use_original_for_eval else
                      to_tensor(make_final(adj, orig_keep, ov_adj, ov_keep, 0.0,
                                           global_drop=args.use_global_dropedge))).to(device)

    if args.use_warmup:
        model = warm
    else:
        model = LAGCN(concat=args.concat+1, nfeat=feats_norm.size(1),
                      nhid=args.hidden, nclass=labels.max().item()+1,
                      dropout=args.dropout, use_mlp=args.use_mlp).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr,
                     weight_decay=args.weight_decay)

    best_val, best_state, best_X = 1e9, None, None
    for ep in range(args.epochs):

        # -------- overlay refresh --------
        if args.refresh_period and ep and ep % args.refresh_period == 0:
            if method == "gradient":
                print(f"[ep {ep}] refresh overlay (gradient)")
                ov_adj, ov_keep = build_overlay("pagerank",
                            gradient_nodelist_from_model(model))
            elif method == "random":
                print(f"[ep {ep}] refresh overlay (random)")
                ov_adj, ov_keep = build_overlay("pagerank", random_nodelist())
            final_adj_norm = (adj_normalized if args.use_original_for_eval else
                              to_tensor(make_final(adj, orig_keep,
                                                   ov_adj, ov_keep, 0.0,
                                                   global_drop=args.use_global_dropedge))).to(device)

        # -------- train --------
        model.train(); opt.zero_grad()
        outs = []
        for _ in range(args.samples):
            tgt = make_final(adj, orig_keep, ov_adj, ov_keep,
                             args.dr, global_drop=args.use_global_dropedge)
            tgt_t = to_tensor(tgt).to(device)
            Xs = get_aug_feats(args.concat)
            if args.use_original_for_last:
                logit = model(Xs + [feats_norm], tgt_t, adj_normalized)
            else:
                logit = model(Xs + [feats_norm], tgt_t)
            outs.append(torch.log_softmax(logit,1))
        loss_sup = sum(F.nll_loss(o[idx_train], labels[idx_train])
                       for o in outs) / len(outs)
        (loss_sup + consis_loss(outs)).backward(); opt.step()

        # -------- validation --------
        model.eval()
        val_Xs = get_aug_feats(args.concat)
        val_log = model(val_Xs + [feats_norm], final_adj_norm, adj_normalized) \
                  if args.use_original_for_last else \
                  model(val_Xs + [feats_norm], final_adj_norm)
        val_loss = F.nll_loss(torch.log_softmax(val_log,1)[idx_val],
                              labels[idx_val])

        if (ep+1) % 50 == 0:
            print(f"Ep{ep+1:03d}  train {loss_sup.item():.4f}  val {val_loss:.4f}")

        if val_loss < best_val:
            best_val, best_state, best_X = val_loss, copy.deepcopy(model), val_Xs

    # -------- test --------
    best_state.eval()
    if args.use_test_aug:
        probs = []
        for _ in range(args.test_aug_runs):
            aug_adj, aug_keep = build_overlay("pagerank", random_nodelist())
            fn = (adj_normalized if args.use_original_for_eval else
                  to_tensor(make_final(adj, orig_keep, aug_adj, aug_keep, 0.0,
                                       global_drop=args.use_global_dropedge))).to(device)
            tmp_log = best_state(best_X + [feats_norm], fn, adj_normalized) \
                      if args.use_original_for_last else \
                      best_state(best_X + [feats_norm], fn)
            probs.append(torch.softmax(tmp_log,1))
        avg_log = torch.log(torch.stack(probs).mean(0) + 1e-12)
        val_scores.append(accuracy(avg_log[idx_val], labels[idx_val]).item())
        test_scores.append(accuracy(avg_log[idx_test], labels[idx_test]).item())
    else:
        tst_log = best_state(best_X + [feats_norm], final_adj_norm, adj_normalized) \
                  if args.use_original_for_last else \
                  best_state(best_X + [feats_norm], final_adj_norm)
        val_scores.append(accuracy(torch.log_softmax(tst_log,1)[idx_val],
                                   labels[idx_val]).item())
        test_scores.append(accuracy(torch.log_softmax(tst_log,1)[idx_test],
                                    labels[idx_test]).item())

# ------------------------- 결과 -----------------------------
print("Val  : {:.2f}±{:.2f}".format(np.mean(val_scores)*100,
                                     np.std(val_scores)*100))
print("Test : {:.2f}±{:.2f}".format(np.mean(test_scores)*100,
                                     np.std(test_scores)*100))
# ============================================================

