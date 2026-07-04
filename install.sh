#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== openvast installer ==="
echo ""

# --- prerequisites ---
for cmd in python3 ssh; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "Missing: $cmd — install it first."; exit 1; }
done

echo "python3  : $(python3 --version 2>&1)"
echo "ssh      : $(ssh -V 2>&1 | awk '{print $1,$2}')"

# --- vastai CLI ---
if command -v vastai >/dev/null 2>&1; then
  echo "vastai   : $(vastai --version 2>/dev/null || echo 'installed')"
else
  echo "vastai   : not found — installing..."
  python3 -m pip install --quiet vastai
  echo "         : installed (authenticate with: vastai set api-key <KEY>)"
fi

# --- opencode (optional; required for editing/auto-wiring) ---
if command -v opencode >/dev/null 2>&1; then
  echo "opencode : $(opencode --version 2>/dev/null || echo 'installed')"
else
  echo "opencode : not found — monitoring still works, but launching/pausing"
  echo "           and auto-wiring are disabled until it is installed."
  echo "           See: https://github.com/opencode-ai/opencode"
fi

# --- SSH key ---
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa}"
if [[ -f "$SSH_KEY" && -f "${SSH_KEY}.pub" ]]; then
  echo "SSH key  : $SSH_KEY ✓"
else
  echo "SSH key  : not found at $SSH_KEY"
  echo "           Generate one with: ssh-keygen -t ed25519 -f $SSH_KEY"
fi
echo ""

# --- install the package ---
echo "Installing openvast..."
python3 -m pip install --quiet "$SCRIPT_DIR"
echo ""

# --- verify ---
if command -v openvast >/dev/null 2>&1; then
  echo "Done! Run: openvast"
else
  echo "Install succeeded but 'openvast' is not on PATH."
  echo "Try: python3 -m openvast"
fi
