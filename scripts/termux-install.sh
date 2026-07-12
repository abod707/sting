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
cargo build --release

# ── install ──────────────────────────────────────────────────────────────────
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
