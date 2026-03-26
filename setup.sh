#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────
#  Whisper Real-Time Transcription — Installer
# ─────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
die()     { error "$*"; exit 1; }
step()    { echo -e "\n${BOLD}── $* ──${NC}"; }

# ── Check we are in the repo root ──────────────────────────────────────────────
[[ -f "Cargo.toml" ]] || die "Run this script from the repository root."

REPO_ROOT="$(pwd)"

# ── Detect distro ──────────────────────────────────────────────────────────────
detect_distro() {
    if command -v pacman &>/dev/null; then echo "arch"
    elif command -v apt &>/dev/null;  then echo "debian"
    elif command -v dnf &>/dev/null;  then echo "fedora"
    else echo "unknown"
    fi
}

DISTRO=$(detect_distro)

# ── Install system dependencies ───────────────────────────────────────────────
step "Installing system dependencies"

case "$DISTRO" in
    arch)
        info "Detected Arch Linux / Manjaro"
        sudo pacman -Sy --needed --noconfirm \
            alsa-lib portaudio cmake clang \
            vulkan-icd-loader vulkan-headers vulkan-tools \
            wget curl
        ;;
    debian)
        info "Detected Ubuntu / Debian"
        sudo apt-get update -qq
        sudo apt-get install -y \
            libasound2-dev portaudio19-dev cmake clang \
            libvulkan-dev vulkan-tools \
            wget curl
        ;;
    fedora)
        info "Detected Fedora"
        sudo dnf install -y \
            alsa-lib-devel portaudio-devel cmake clang \
            vulkan-loader-devel vulkan-headers vulkan-tools \
            wget curl
        ;;
    *)
        warn "Unknown distro — skipping automatic dependency install."
        warn "Please install manually: alsa-lib, portaudio, cmake, clang, vulkan headers + loader."
        ;;
esac

success "System dependencies ready"

# ── Rust toolchain ─────────────────────────────────────────────────────────────
step "Checking Rust toolchain"

if ! command -v cargo &>/dev/null; then
    info "Rust not found — installing via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
    # shellcheck source=/dev/null
    source "$HOME/.cargo/env"
else
    info "Rust found: $(rustc --version)"
fi

RUST_VER=$(rustc --version | grep -oP '\d+\.\d+' | head -1)
RUST_MAJOR=$(echo "$RUST_VER" | cut -d. -f1)
RUST_MINOR=$(echo "$RUST_VER" | cut -d. -f2)

if [[ "$RUST_MAJOR" -lt 1 ]] || { [[ "$RUST_MAJOR" -eq 1 ]] && [[ "$RUST_MINOR" -lt 85 ]]; }; then
    warn "Rust 1.85+ is required (edition 2024). Updating..."
    rustup update stable
fi

success "Rust toolchain: $(rustc --version)"

# ── Check Vulkan ───────────────────────────────────────────────────────────────
step "Checking Vulkan support"

if command -v vulkaninfo &>/dev/null && vulkaninfo --summary &>/dev/null 2>&1; then
    success "Vulkan is available"
else
    warn "Vulkan not detected or vulkaninfo failed."
    warn "GPU acceleration will not be available — CPU fallback will be used."
    warn "Make sure your GPU drivers include Vulkan support:"
    warn "  NVIDIA: nvidia drivers"
    warn "  AMD:    mesa + vulkan-radeon"
    warn "  Intel:  mesa + vulkan-intel"
fi

# ── Download models ────────────────────────────────────────────────────────────
step "Downloading models"

mkdir -p models/vad models/whisper-fast models/whisper-accurate

# Silero VAD
VAD_TARGET="models/vad/silero_vad.onnx"
VAD_URL="https://github.com/snakers4/silero-vad/raw/v4.0/files/silero_vad.onnx"

if [[ -f "$VAD_TARGET" ]]; then
    success "Silero VAD model already present — skipping"
else
    info "Downloading Silero VAD model (~1.8 MB)..."
    wget -q --show-progress -O "$VAD_TARGET" "$VAD_URL" \
        || die "Failed to download Silero VAD model. Check your internet connection."
    success "Silero VAD model downloaded"
fi

# Whisper fast model
FAST_DIR="models/whisper-fast"
FAST_MODEL="ggml-tiny-q8_0.bin"
FAST_TARGET="$FAST_DIR/$FAST_MODEL"
HF_BASE="https://huggingface.co/ggerganov/whisper.cpp/resolve/main"

if ls "$FAST_DIR"/*.bin &>/dev/null 2>&1; then
    success "Fast Whisper model already present — skipping"
else
    info "Downloading fast Whisper model: $FAST_MODEL (~42 MB)..."
    wget -q --show-progress -O "$FAST_TARGET" "$HF_BASE/$FAST_MODEL" \
        || die "Failed to download fast model. Visit https://huggingface.co/ggerganov/whisper.cpp to download manually and place in $FAST_DIR/"
    success "Fast model downloaded: $FAST_MODEL"
fi

# Whisper accurate model
ACC_DIR="models/whisper-accurate"
ACC_MODEL="ggml-medium-q8_0.bin"
ACC_TARGET="$ACC_DIR/$ACC_MODEL"

if ls "$ACC_DIR"/*.bin &>/dev/null 2>&1; then
    success "Accurate Whisper model already present — skipping"
else
    info "Downloading accurate Whisper model: $ACC_MODEL (~470 MB)..."
    info "This may take a while depending on your connection..."
    wget -q --show-progress -O "$ACC_TARGET" "$HF_BASE/$ACC_MODEL" \
        || {
            warn "Failed to download accurate model automatically."
            warn "Please download $ACC_MODEL manually from:"
            warn "  https://huggingface.co/ggerganov/whisper.cpp/tree/main"
            warn "and place it in: $ACC_DIR/"
        }
    [[ -f "$ACC_TARGET" ]] && success "Accurate model downloaded: $ACC_MODEL"
fi

# ── Verify model layout ────────────────────────────────────────────────────────
step "Verifying model layout"

ALL_OK=true

check_model() {
    local dir="$1" ext="$2" label="$3"
    if ls "$dir"/*."$ext" &>/dev/null 2>&1; then
        success "$label: $(ls "$dir"/*."$ext" | head -1 | xargs basename)"
    else
        error "Missing $label in $dir/"
        ALL_OK=false
    fi
}

check_model "models/vad"              "onnx" "Silero VAD"
check_model "models/whisper-fast"     "bin"  "Fast Whisper model"
check_model "models/whisper-accurate" "bin"  "Accurate Whisper model"

$ALL_OK || die "One or more models are missing. Please download them manually (see README)."

# ── Build ──────────────────────────────────────────────────────────────────────
step "Building (release)"

info "Running: cargo build --release"
info "This will take a few minutes on first build..."
cargo build --release 2>&1

success "Build complete: target/release/translator"

# ── Done ───────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}╔══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}${BOLD}║   Installation complete! Ready to run.   ║${NC}"
echo -e "${GREEN}${BOLD}╚══════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Run:  ${BOLD}./target/release/translator${NC}"
echo ""
echo -e "  ${YELLOW}Audio tip:${NC} To transcribe system audio (e.g. browser, media player),"
echo -e "  open ${BOLD}pavucontrol${NC} → Recording tab → change the app's source to a monitor."
echo ""