#!/usr/bin/env python3
"""Train ONLY the contrastive retrieval head on frozen encoder features.

The finetune's joint contrastive loss (weight 0.1, 2 epochs, from random init)
left the head at ~62% hit@6 — not usable. This trains the head properly:

  1. extract mean-pooled encoder features for every query + all 30 tool texts
     (frozen finetuned encoder, one forward pass each)
  2. train the 512->128(relu)->128(l2norm) head with softmax over the 30 tool
     embeddings (InfoNCE where the negative set = all tools, which is exactly
     the deployment condition)
  3. write the trained head back into the checkpoint

Decoder and encoder weights are untouched — zero risk to generation quality.
"""
import json
import pickle
import sys

import numpy as np

sys.path.insert(0, "/agent/workspace/needle")

import jax
import jax.numpy as jnp
from needle import SimpleAttentionNetwork, load_checkpoint, get_tokenizer
from needle.model.architecture import make_padding_mask

CKPT_IN = sys.argv[1]
CKPT_OUT = sys.argv[2]

params, config = load_checkpoint(CKPT_IN)
model = SimpleAttentionNetwork(config)
tok = get_tokenizer()

# ── collect texts ────────────────────────────────────────────────────────────
termux = json.load(open("/agent/workspace/ftdata/termux_tools.json"))
generic = json.load(open("/agent/workspace/ftdata/generic_tools.json"))
tools = termux + generic
tool_names = [t["name"] for t in tools]
tool_texts = [json.dumps(t, separators=(",", ":")) for t in tools]
name_to_idx = {n: i for i, n in enumerate(tool_names)}

queries, labels = [], []
for line in open("/agent/workspace/ftdata/data.jsonl"):
    ex = json.loads(line)
    calls = json.loads(ex["answers"])
    names = [c["name"] for c in calls if isinstance(c, dict) and c.get("name") in name_to_idx]
    if names:
        queries.append(ex["query"])
        labels.append(name_to_idx[names[0]])  # primary tool
labels = np.array(labels, dtype=np.int64)
print(f"{len(queries)} labeled queries, {len(tools)} tools")

# ── frozen pooled encoder features ───────────────────────────────────────────
def pooled_features(texts, batch_size=64, max_len=256):
    pad = tok.pad_token_id
    out = []
    for s in range(0, len(texts), batch_size):
        chunk = texts[s:s + batch_size]
        idlists = [tok.encode(t)[:max_len] for t in chunk]
        maxlen = max(len(x) for x in idlists)
        arr = np.full((len(chunk), maxlen), pad, dtype=np.int32)
        for i, ids in enumerate(idlists):
            arr[i, :len(ids)] = ids
        x = jnp.array(arr)
        mask = make_padding_mask(x, pad)
        enc_out, enc_mask = model.apply({"params": params}, x, src_mask=mask, method="encode_text")
        m2 = np.array(enc_mask[:, 0, 0, :], dtype=np.float32)          # (B, T)
        eo = np.array(enc_out, dtype=np.float32)                        # (B, T, d)
        pooled = (eo * m2[:, :, None]).sum(1) / np.maximum(m2.sum(1, keepdims=True), 1.0)
        out.append(pooled)
        print(".", end="", flush=True)
    print()
    return np.concatenate(out, 0)

print("extracting query features...")
QF = pooled_features(queries)          # (N, 512)
print("extracting tool features...")
TF = pooled_features(tool_texts)       # (30, 512)

# ── train the head (numpy Adam) ──────────────────────────────────────────────
rng = np.random.RandomState(0)
d, h, cdim = 512, 128, 128
W1 = rng.normal(0, 0.05, (d, h)).astype(np.float32)
b1 = np.zeros(h, dtype=np.float32)
W2 = rng.normal(0, 0.05, (h, cdim)).astype(np.float32)
log_tau = np.array(np.log(0.07), dtype=np.float32)

def head_fwd(F, W1, b1, W2):
    H = np.maximum(F @ W1 + b1, 0.0)
    Z = H @ W2
    Zn = Z / np.sqrt((Z ** 2).sum(-1, keepdims=True) + 1e-12)
    return H, Z, Zn

# split for honest early stopping
n = len(QF)
perm = rng.permutation(n)
val_idx, tr_idx = perm[:400], perm[400:]

mom = {k: 0 for k in "W1 b1 W2 t".split()}
vel = {k: 0 for k in "W1 b1 W2 t".split()}
lr, beta1, beta2, eps = 3e-3, 0.9, 0.999, 1e-8
best_val, best = -1.0, None
BS = 512
step = 0

for epoch in range(60):
    ep_perm = rng.permutation(len(tr_idx))
    for s in range(0, len(tr_idx), BS):
        idx = tr_idx[ep_perm[s:s + BS]]
        F, y = QF[idx], labels[idx]
        step += 1

        Hq, Zq, Q = head_fwd(F, W1, b1, W2)          # (B, c)
        Ht, Zt, T = head_fwd(TF, W1, b1, W2)         # (30, c)
        tau = np.exp(log_tau)
        S = Q @ T.T / tau                             # (B, 30)
        S -= S.max(1, keepdims=True)
        P = np.exp(S); P /= P.sum(1, keepdims=True)

        B = len(F)
        dS = P.copy(); dS[np.arange(B), y] -= 1.0; dS /= B     # (B, 30)
        # grads through normalized embeddings
        dQ = dS @ T / tau                                       # (B, c)
        dT = dS.T @ Q / tau                                     # (30, c)
        dlog_tau = -(dS * (Q @ T.T)).sum() / tau * np.exp(log_tau) / np.exp(log_tau)
        dlog_tau = float((-(dS * (Q @ T.T) / tau).sum()))       # d/dlog_tau of S = -S

        def back_norm(Z, Zn, dZn):
            nrm = np.sqrt((Z ** 2).sum(-1, keepdims=True) + 1e-12)
            return (dZn - Zn * (dZn * Zn).sum(-1, keepdims=True)) / nrm

        dZq = back_norm(Zq, Q, dQ)
        dZt = back_norm(Zt, T, dT)
        dW2 = Hq.T @ dZq + Ht.T @ dZt
        dHq = dZq @ W2.T; dHt = dZt @ W2.T
        dHq[Hq <= 0] = 0; dHt[Ht <= 0] = 0
        dW1 = F.T @ dHq + TF.T @ dHt
        db1 = dHq.sum(0) + dHt.sum(0)

        for k, g in (("W1", dW1), ("b1", db1), ("W2", dW2), ("t", dlog_tau)):
            mom[k] = beta1 * mom[k] + (1 - beta1) * g
            vel[k] = beta2 * vel[k] + (1 - beta2) * (g * g if not np.isscalar(g) else g ** 2)
            upd = lr * mom[k] / (1 - beta1 ** step) / (np.sqrt(vel[k] / (1 - beta2 ** step)) + eps)
            if k == "W1": W1 -= upd
            elif k == "b1": b1 -= upd
            elif k == "W2": W2 -= upd
            else: log_tau -= upd

    # validation hit@k
    _, _, Qv = head_fwd(QF[val_idx], W1, b1, W2)
    _, _, Tv = head_fwd(TF, W1, b1, W2)
    Sv = Qv @ Tv.T
    order = np.argsort(-Sv, axis=1)
    yv = labels[val_idx]
    h1 = (order[:, 0] == yv).mean()
    h3 = np.array([yv[i] in order[i, :3] for i in range(len(yv))]).mean()
    h6 = np.array([yv[i] in order[i, :6] for i in range(len(yv))]).mean()
    if h6 > best_val:
        best_val, best = h6, (W1.copy(), b1.copy(), W2.copy())
    if epoch % 10 == 0 or epoch == 59:
        print(f"epoch {epoch:3d}  val hit@1={h1:.1%} hit@3={h3:.1%} hit@6={h6:.1%}  tau={np.exp(log_tau):.3f}")

W1, b1, W2 = best
print(f"best val hit@6: {best_val:.1%}")

# ── write back into the checkpoint ───────────────────────────────────────────
with open(CKPT_IN, "rb") as f:
    d_out = pickle.load(f)
d_out["params"]["contrastive_hidden"]["kernel"] = W1.astype(np.float16)
d_out["params"]["contrastive_hidden"]["bias"] = b1.astype(np.float16)
d_out["params"]["contrastive_proj"]["kernel"] = W2.astype(np.float16)
with open(CKPT_OUT, "wb") as f:
    pickle.dump(d_out, f)
print(f"wrote {CKPT_OUT}")
