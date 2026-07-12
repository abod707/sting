#!/usr/bin/env python3
"""Export needle's SentencePiece model to a plain-JSON spec for the pure-Rust tokenizer.

Emits: pieces [(text, score, type)], normalizer flags, special ids.
Types: 1=NORMAL 2=UNKNOWN 3=CONTROL 4=USER_DEFINED 5=UNUSED 6=BYTE
"""
import json
import sys

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as model_pb2

model_path = sys.argv[1] if len(sys.argv) > 1 else "/agent/workspace/needle/needle/tokenizer/needle.model"
out_path = sys.argv[2] if len(sys.argv) > 2 else "/agent/workspace/ftdata/tokenizer_spec.json"

sp = spm.SentencePieceProcessor()
sp.Load(model_path)

proto = model_pb2.ModelProto()
proto.ParseFromString(sp.serialized_model_proto())

ns = proto.normalizer_spec
ts = proto.trainer_spec
print("normalizer:", ns.name)
print("  add_dummy_prefix:", ns.add_dummy_prefix)
print("  remove_extra_whitespaces:", ns.remove_extra_whitespaces)
print("  escape_whitespaces:", ns.escape_whitespaces)
print("  precompiled_charsmap bytes:", len(ns.precompiled_charsmap))
print("trainer: model_type:", ts.model_type, "(1=unigram 2=bpe)", " byte_fallback:", ts.byte_fallback)
print("  vocab_size:", len(proto.pieces))

pieces = [[p.piece, p.score, p.type] for p in proto.pieces]
spec = {
    "model_type": "bpe",
    "add_dummy_prefix": ns.add_dummy_prefix,
    "remove_extra_whitespaces": ns.remove_extra_whitespaces,
    "escape_whitespaces": ns.escape_whitespaces,
    "byte_fallback": ts.byte_fallback,
    "unk_id": 3,
    "pieces": pieces,
}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(spec, f, ensure_ascii=False)
print(f"wrote {out_path} ({len(pieces)} pieces)")

# sanity: show the special + first few + some byte pieces
for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 20, 100]:
    p = proto.pieces[i]
    print(f"  id={i} piece={p.piece!r} score={p.score} type={p.type}")
