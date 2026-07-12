#!/usr/bin/env python3
"""Driver for needle finetuning with per-N-step eval cadence (best-ckpt selection).

Reuses needle's own finetune pipeline; patches:
  1. _resolve_checkpoint  -> use the local base checkpoint (no forced HF re-download)
  2. train() args         -> eval_every / max_eval_samples for real best-ckpt tracking
"""
import argparse
import sys

sys.path.insert(0, "/agent/workspace/needle")

import importlib

from needle.training import finetune as ft

tr = importlib.import_module("needle.training.train")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl_path")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--max-eval-samples", type=int, default=40)
    ap.add_argument("--max-enc-len", type=int, default=512)
    ap.add_argument("--max-dec-len", type=int, default=192)
    ap.add_argument("--base-checkpoint", type=str, default="checkpoints/needle.pkl")
    ap.add_argument("--skip-base-eval", action="store_true",
                    help="skip the pre-training test-set eval (already measured)")
    args = ap.parse_args()

    ft._resolve_checkpoint = lambda p: args.base_checkpoint

    if args.skip_base_eval:
        # First _quick_tool_eval call is the base eval -> return {} (skipped);
        # later calls (finetuned eval) run for real.
        _orig_eval = ft._quick_tool_eval
        _calls = {"n": 0}

        def eval_skipping_first(*a, **kw):
            _calls["n"] += 1
            if _calls["n"] == 1:
                print("(base eval skipped — using previously measured numbers)")
                return {}
            return _orig_eval(*a, **kw)

        ft._quick_tool_eval = eval_skipping_first

    _orig_train = tr.train

    def train_with_cadence(targs):
        targs.eval_every = args.eval_every
        targs.max_eval_samples = args.max_eval_samples
        return _orig_train(targs)

    tr.train = train_with_cadence

    ft_args = argparse.Namespace(
        jsonl_path=args.jsonl_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint=None,
        checkpoint_dir="checkpoints",
        cache_dir=None,
        max_enc_len=args.max_enc_len,
        max_dec_len=args.max_dec_len,
    )
    ft.finetune_local(ft_args)


if __name__ == "__main__":
    main()
