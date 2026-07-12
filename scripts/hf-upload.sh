#!/usr/bin/env bash
# Publish the finetuned model to Hugging Face.
#
# Prereqs (any machine — works in Termux too):
#   pip install -U huggingface_hub
#   hf auth login          # paste a WRITE token from hf.co/settings/tokens
# Then, from the sting repo root:
#   ./scripts/hf-upload.sh [repo_id]     # default: abod707/needle-termux-sting
set -e
REPO_ID="${1:-abod707/needle-termux-sting}"
say() { printf "\033[1;35m[hf]\033[0m %s\n" "$*"; }

reasm() { # $1=output path  $2=expected sha256
  if [ ! -f "$1" ]; then
    say "reassembling $(basename "$1") from base64 parts"
    cat "$1".b64.part* | base64 -d > "$1"
  fi
  GOT=$(sha256sum "$1" | cut -d' ' -f1)
  if [ "$GOT" != "$2" ]; then
    say "checksum mismatch for $1 (got $GOT) — aborting"
    rm -f "$1"; exit 1
  fi
  say "$(basename "$1") checksum OK"
}

reasm model/model.safetensors      36443d27a663f2839e6f53c9b19bfa0b06c2e158d1c30819a4f6b6b87f887f24
reasm model/needle_sting_final.pkl c78e0077d72351d8aedaabe1abd75759d7f3f04818664f2dcac6c14cad1e368a

say "staging hf_upload/"
rm -rf hf_upload && mkdir hf_upload
cp hf/README.md hf_upload/README.md
cp model/model.safetensors model/config.json model/tokenizer_spec.json hf_upload/
cp model/needle_sting_final.pkl hf_upload/

say "uploading to $REPO_ID (this pushes ~105MB)"
python3 - "$REPO_ID" <<'PY'
import sys
from huggingface_hub import HfApi
repo = sys.argv[1]
api = HfApi()
try:
    api.create_repo(repo, repo_type="model", exist_ok=True)
except Exception as e:
    # Fine-grained tokens scoped to a single existing repo can't call
    # create_repo at all — that's fine if you created the repo in the web UI.
    print(f"create_repo skipped ({type(e).__name__}) — assuming {repo} already exists")
api.upload_folder(folder_path="hf_upload", repo_id=repo)
print(f"\ndone: https://huggingface.co/{repo}")
PY
