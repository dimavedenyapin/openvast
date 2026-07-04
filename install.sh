#!/bin/sh
# openvast bootstrap installer — works on a bare laptop (needs only curl + sh).
#
#   curl -fsSL https://raw.githubusercontent.com/dimavedenyapin/openvast/main/install.sh | sh
#
# Installs uv (which also manages Python), then vastai + openvast as isolated
# tools, and generates an SSH key if you don't have one. No pre-existing Python
# required.
set -eu

REPO="dimavedenyapin/openvast"
GIT_URL="git+https://github.com/${REPO}"

c_info='\033[36m'; c_ok='\033[32m'; c_warn='\033[33m'; c_dim='\033[90m'; c_off='\033[0m'
info() { printf "${c_info}▸${c_off} %s\n" "$1"; }
ok()   { printf "${c_ok}✓${c_off} %s\n" "$1"; }
warn() { printf "${c_warn}!${c_off} %s\n" "$1"; }

printf "\n${c_info}openvast installer${c_off}\n\n"

# --- uv (bundles Python management; the only thing we bootstrap) -------------
if ! command -v uv >/dev/null 2>&1; then
  info "Installing uv (Python toolchain manager)…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# make uv + its tool bin available for the rest of this script
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  warn "uv is installed but not on PATH yet."
  warn "Open a new terminal (or run: export PATH=\"\$HOME/.local/bin:\$PATH\") and re-run."
  exit 1
fi
ok "uv $(uv --version 2>/dev/null | awk '{print $2}')"

# --- Python (uv fetches one if the system has none) -------------------------
info "Ensuring a Python 3.12 runtime…"
uv python install 3.12 >/dev/null 2>&1 || true

# --- vastai CLI + openvast, each in its own isolated tool env ---------------
info "Installing vastai CLI…"
uv tool install --quiet --force vastai

info "Installing openvast…"
uv tool install --quiet --force "$GIT_URL"

uv tool update-shell >/dev/null 2>&1 || true
ok "openvast installed"

# --- SSH key (needed to attach to instances + logs/download-% viewer) -------
if [ ! -f "$HOME/.ssh/id_rsa" ]; then
  info "Generating an SSH key at ~/.ssh/id_rsa…"
  mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh"
  ssh-keygen -t rsa -b 4096 -N "" -f "$HOME/.ssh/id_rsa" -q && ok "SSH key created"
else
  ok "SSH key present (~/.ssh/id_rsa)"
fi

# --- next steps -------------------------------------------------------------
printf "\n${c_ok}Done!${c_off}  Next steps:\n\n"
printf "  1. Authenticate vast.ai:   ${c_info}vastai set api-key <YOUR_KEY>${c_off}\n"
printf "     ${c_dim}(get a key at https://cloud.vast.ai/account/)${c_off}\n"
printf "  2. (optional) opencode for auto-wiring / launching:\n"
printf "     ${c_info}curl -fsSL https://opencode.ai/install | bash${c_off}\n"
printf "  3. Run:                    ${c_info}openvast${c_off}\n\n"
if ! command -v openvast >/dev/null 2>&1; then
  printf "  ${c_warn}!${c_off} 'openvast' isn't on PATH in this shell yet — open a new terminal,\n"
  printf "    or run: ${c_info}export PATH=\"\$HOME/.local/bin:\$PATH\"${c_off}\n\n"
fi
