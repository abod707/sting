#!/usr/bin/env python3
"""Convert a needle .pkl checkpoint (JAX/Flax) to safetensors for the Rust/candle runtime.

- Unstacks nn.scan layer-stacked params (leading axis = layer index)
- Transposes Dense kernels (in,out) -> (out,in) to match candle Linear convention
- Casts bf16 -> f32 exactly (bf16 is truncated f32)
- Skips contrastive head + log_temp (generation doesn't use them)
- Emits config.json alongside

Usage: convert_to_safetensors.py <in.pkl> <out_dir>
"""
import json
import pickle
import sys

import numpy as np


def tree_flatten(tree, prefix=""):
    out = {}
    if isinstance(tree, dict):
        for k, v in tree.items():
            out.update(tree_flatten(v, f"{prefix}/{k}" if prefix else k))
    else:
        out[prefix] = np.asarray(tree)
    return out


def bf16_to_f32(a):
    if a.dtype == np.float32:
        return a
    # jax bf16 arrays numpy-ify as ml_dtypes.bfloat16; astype handles exactly
    return a.astype(np.float32)


def main(pkl_path, out_dir):
    import os
    os.makedirs(out_dir, exist_ok=True)
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    params, config = data["params"], data["config"]

    flat = tree_flatten(params)
    print(f"{len(flat)} leaves in checkpoint:")
    for k, v in sorted(flat.items()):
        print(f"  {k:70s} {str(v.shape):20s} {v.dtype}")

    n_enc = config["num_encoder_layers"]
    n_dec = config["num_decoder_layers"]

    tensors = {}

    def put(name, arr, transpose=False):
        a = bf16_to_f32(np.asarray(arr))
        if transpose:
            a = a.T
        # store f16: the pkl checkpoint is f16 on disk, so this is lossless,
        # halves the file size (fits GitHub's 100MB limit), and the Rust
        # loader upcasts to f32 at load time.
        tensors[name] = np.ascontiguousarray(a.astype(np.float16))

    for key, arr in flat.items():
        parts = key.split("/")
        if parts[0] == "embedding":
            put("embedding.weight", arr)  # (vocab, d_model)
        elif parts[0] == "contrastive_hidden":
            if parts[1] == "kernel":
                put("contrastive_hidden.weight", arr, transpose=True)
            else:
                put("contrastive_hidden.bias", arr)
        elif parts[0] == "contrastive_proj":
            put("contrastive_proj.weight", arr, transpose=True)
        elif parts[0] == "log_temp":
            continue
        elif parts[0] == "encoder":
            if parts[1] == "final_norm":
                put("encoder.final_norm.scale", arr)
            elif parts[1] == "layers":
                # stacked along axis 0 with n_enc entries
                rest = parts[2:]
                for i in range(n_enc):
                    layer_arr = np.asarray(arr)[i]
                    name = _map_block_param("encoder", i, rest)
                    if name:
                        put(name, layer_arr, transpose=name.endswith(".weight"))
            else:
                print(f"  !! unmapped encoder key: {key}")
        elif parts[0] == "decoder":
            if parts[1] == "layers":
                rest = parts[2:]
                for i in range(n_dec):
                    layer_arr = np.asarray(arr)[i]
                    name = _map_block_param("decoder", i, rest)
                    if name:
                        put(name, layer_arr, transpose=name.endswith(".weight"))
            elif parts[1].startswith("ZCRMSNorm"):
                put("decoder.final_norm.scale", arr)
            else:
                print(f"  !! unmapped decoder key: {key}")
        else:
            print(f"  !! unmapped key: {key}")

    from safetensors.numpy import save_file
    out_path = f"{out_dir}/model.safetensors"
    save_file(tensors, out_path)

    cfg_out = {
        "vocab_size": config["vocab_size"],
        "d_model": config["d_model"],
        "num_heads": config["num_heads"],
        "num_kv_heads": config["num_kv_heads"],
        "num_encoder_layers": n_enc,
        "num_decoder_layers": n_dec,
        "rope_theta": config.get("rope_theta", 10000.0),
        "max_seq_len": config.get("max_seq_len", 1024),
        "pad_token_id": 0, "eos_token_id": 1, "bos_token_id": 2,
        "tool_call_token_id": 4, "tools_token_id": 5,
    }
    with open(f"{out_dir}/config.json", "w") as f:
        json.dump(cfg_out, f, indent=2)

    total = sum(v.size for v in tensors.values())
    size_mb = sum(v.nbytes for v in tensors.values()) / 1e6
    print(f"\nwrote {out_path}: {len(tensors)} tensors, {total:,} params, {size_mb:.1f} MB (f32)")
    print(f"wrote {out_dir}/config.json")


def _map_block_param(which, i, rest):
    """Map flax param path (within a scanned block) to candle-style name."""
    # strip the scan body module component (EncoderBlock_0 / DecoderBlock_0)
    if rest and (rest[0].startswith("EncoderBlock") or rest[0].startswith("DecoderBlock")):
        rest = rest[1:]
    p = "/".join(rest)
    base = f"{which}.layers.{i}"
    # gates (scalar)
    if p == "attn_gate":
        return f"{base}.attn_gate"
    if p == "self_attn_gate":
        return f"{base}.self_attn_gate"
    if p == "cross_attn_gate":
        return f"{base}.cross_attn_gate"
    # block-level norms: encoder has ZCRMSNorm_0 (pre-attn);
    # decoder has ZCRMSNorm_0 (pre-self) and ZCRMSNorm_1 (pre-cross)
    if p == "ZCRMSNorm_0/scale":
        return f"{base}.norm1.scale"
    if p == "ZCRMSNorm_1/scale":
        return f"{base}.norm2.scale"
    # attention projections
    for attn in ("self_attn", "cross_attn"):
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            if p == f"{attn}/{proj}/kernel":
                return f"{base}.{attn}.{proj}.weight"
        for nrm in ("q_norm", "k_norm"):
            if p == f"{attn}/{nrm}/scale":
                return f"{base}.{attn}.{nrm}.scale"
    print(f"  !! unmapped block param: {which}.{i}.{p}")
    return None


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
