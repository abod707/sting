# Finetuning recipe

How the shipped model was made (runs on a PC/Mac — NOT on the phone):

```bash
# 1. clone needle and set up its env (needs python >= 3.11, CPU is fine)
git clone https://github.com/cactus-compute/needle && cd needle && source ./setup

# 2. generate the dataset (stdlib only, deterministic)
python3 finetune/gen_data.py            # -> data.jsonl + schema packs

# 3. IMPORTANT: the released checkpoint ships an all-zero contrastive
#    (retrieval) head, and zero+ReLU is a gradient fixed point — randomize it
#    first or retrieval can never train:
python3 - <<'PY'
import pickle, numpy as np
rng = np.random.RandomState(42)
d = pickle.load(open("checkpoints/needle.pkl", "rb"))
d["params"]["contrastive_hidden"]["kernel"] = rng.normal(0, .02, (512,128)).astype(np.float16)
d["params"]["contrastive_hidden"]["bias"]   = np.zeros(128, np.float16)
d["params"]["contrastive_proj"]["kernel"]   = rng.normal(0, .02, (128,128)).astype(np.float16)
pickle.dump(d, open("checkpoints/needle_init.pkl", "wb"))
PY

# 4. finetune (driver adds per-N-step eval for best-checkpoint selection;
#    ~4GB RAM at batch 8; a couple of hours on CPU, minutes on a GPU)
python3 finetune/run_finetune.py data.jsonl \
  --epochs 2 --batch-size 8 --eval-every 25 \
  --base-checkpoint checkpoints/needle_init.pkl

# 5. convert the best checkpoint for sting (f16 safetensors + config)
python3 finetune/convert_to_safetensors.py \
  checkpoints/needle_finetuned_<id>_best.pkl ../model
python3 finetune/export_tokenizer_spec.py   # -> tokenizer_spec.json
```

Adding your own tools: append ≥120 examples per tool to `data.jsonl`
(query / tools / answers — see the needle README for the schema) or extend the
template pools in `gen_data.py`. Vary phrasings; include examples where your
tool is present but NOT the right choice.

## Step 6 — train the retrieval head (required!)

Two gotchas make this step non-optional:
1. the released checkpoint's contrastive head is all zeros (zero+ReLU = no
   gradients, joint finetuning cannot revive it), and
2. even from random init, the joint loss (weight 0.1) leaves it at ~62% hit@6.

Train it properly on frozen encoder features (seconds of compute):

```bash
python3 finetune/train_head.py \
  checkpoints/needle_finetuned_<id>_best.pkl checkpoints/needle_sting_final.pkl
python3 finetune/retrieval_hitrate.py checkpoints/needle_sting_final.pkl 400
# expect hit@6 ~100% — then convert needle_sting_final.pkl for sting
```
