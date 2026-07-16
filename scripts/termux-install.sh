#!/data/data/com.termux/files/usr/bin/bash
# sting installer for Termux
# Builds the release binary, installs it to $PREFIX/bin, and sets up ~/.sting
set -e

say() { printf "\033[1;36m[sting]\033[0m %s\n" "$*"; }

# ── sanity checks ────────────────────────────────────────────────────────────
if [ ! -f Cargo.toml ] || [ ! -d src ]; then
  echo "run this from the sting repo root: ./scripts/termux-install.sh"
  exit 1
fi

command -v cargo >/dev/null 2>&1 || {
  say "rust not found — installing (pkg install rust binutils)"
  pkg install -y rust binutils
}

# ── build ────────────────────────────────────────────────────────────────────
say "building release binary (grab a coffee — 5-10 min on a phone)"
# target-cpu=native lets the tensor kernels use this phone's exact CPU features
# (dotprod, fp16, etc. on top of the aarch64 NEON baseline). Safe here because
# we build and run on the same device. Override by exporting RUSTFLAGS yourself.
: "${RUSTFLAGS:=-C target-cpu=native}"
export RUSTFLAGS
say "RUSTFLAGS=$RUSTFLAGS"
cargo build --release

# ── install ──────────────────────────────────────────────────────────────────

# ── reassemble model weights (stored as base64 parts for GitHub API limits) ──
if [ ! -f model/model.safetensors ] && ls model/model.safetensors.b64.part* >/dev/null 2>&1; then
  say "reassembling model weights from base64 parts"
  cat model/model.safetensors.b64.part* | base64 -d > model/model.safetensors
  GOT=$(sha256sum model/model.safetensors | cut -d' ' -f1)
  WANT="36443d27a663f2839e6f53c9b19bfa0b06c2e158d1c30819a4f6b6b87f887f24"
  if [ "$GOT" != "$WANT" ]; then
    say "checksum mismatch after reassembly — aborting (got $GOT)"
    rm -f model/model.safetensors
    exit 1
  fi
  say "model checksum OK"
fi

STING_HOME="${STING_HOME:-$HOME/.sting}"
say "installing binary to \$PREFIX/bin/sting"
install -m 755 target/release/sting "$PREFIX/bin/sting"

say "setting up $STING_HOME"
mkdir -p "$STING_HOME/model"
cp -f model/model.safetensors model/config.json model/tokenizer_spec.json "$STING_HOME/model/"
[ -f "$STING_HOME/tools.json" ] || cp tools.json "$STING_HOME/tools.json"

# make STING_HOME available in future shells
PROFILE="$HOME/.profile"
if ! grep -q "STING_HOME" "$PROFILE" 2>/dev/null; then
  echo "export STING_HOME=\"$STING_HOME\"" >> "$PROFILE"
  say "added STING_HOME to ~/.profile (restart the session or: source ~/.profile)"
fi
export STING_HOME

# ── termux-api check ─────────────────────────────────────────────────────────
if ! command -v termux-battery-status >/dev/null 2>&1; then
  say "NOTE: termux-api package not found. Install it with:"
  say "  pkg install termux-api"
  say "and install the Termux:API *app* from F-Droid (must match your Termux source)."
fi

# ── smoke test ───────────────────────────────────────────────────────────────
say "smoke test (dry run, no command executed):"
sting --dry-run --time "how much battery do I have left" || {
  say "smoke test failed — check the output above"
  exit 1
}

say "done. try:  sting \"turn on the flashlight\""
say "       or:  sting --repl"
