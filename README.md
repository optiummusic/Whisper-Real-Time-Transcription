# Whisper Real-Time Transcription
# ONLY USE SILERO VAD VERSION 4!
A fully offline, real-time speech transcription tool written in Rust. Built around a two-pass Whisper pipeline with Silero VAD for voice detection — designed to help understand foreign-language audio with low latency.

https://github.com/user-attachments/assets/fe91eb4a-0f2a-47c8-8536-2a978c4a61de

## How it works

```
Microphone / System Audio
         │
         ▼
   [Audio Capture]  ──  cpal + rubato resampler → 16 kHz mono
         │
         ▼
   [Silero VAD]  ──  ONNX runtime, detects speech segments
         │
         ├─────────────────────────┐
         ▼                         ▼
  [Pass 1 — Fast Model]    [Pass 2 — Accurate Model]
   ggml-tiny-q8_0.bin       ggml-medium-q8_0.bin
   Low latency preview       Final, high-quality text
         │                         │
         └──────────┬──────────────┘
                    ▼
             [egui/eframe UI]
         Live transcription window
```

**Pass 1** runs on a small fast model and immediately emits partial results as you speak. **Pass 2** runs the accurate model on the full completed phrase and replaces the partial with a final result. Short phrases (below the fast-track threshold) skip Pass 2 entirely.

## Features

- **Fully offline** — no internet connection, no cloud, no data leaves your machine
- **Two-pass pipeline** — low-latency preview from a fast model, replaced by accurate final text
- **Silero VAD** — ONNX-based voice activity detection with configurable sensitivity
- **GPU acceleration** — Vulkan backend via `whisper-rs`; per-model GPU selection (fast/accurate can use different devices)
- **Live egui UI** — transcription window with real-time word-by-word rendering and inline RTF/duration stats
- **Translation layer** — word-level translation with a dictionary override system (`dictionary/rules.toml`)
- **Audio resampling** — any input sample rate is resampled to 16 kHz via high-quality sinc interpolation
- **Phrase stitching** — short adjacent phrases are merged to reduce fragmented output
- **Configurable** — all VAD and engine parameters are tunable at runtime via the settings panel and persisted in `config.toml`
- **Transcription saving** — optional disk save with timestamped file names

## Prerequisites

### Rust toolchain

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

> **Edition 2024** is required — make sure you have a recent stable Rust (1.85+).

### System libraries

**Arch Linux / Manjaro:**
```bash
sudo pacman -S alsa-lib portaudio cmake clang vulkan-icd-loader vulkan-headers
```

**Ubuntu / Debian:**
```bash
sudo apt install libasound2-dev portaudio19-dev cmake clang libvulkan-dev vulkan-tools
```

**Fedora:**
```bash
sudo dnf install alsa-lib-devel portaudio-devel cmake clang vulkan-loader-devel vulkan-headers
```

### GPU drivers (Vulkan)

| GPU | Driver |
|-----|--------|
| NVIDIA | `nvidia` (proprietary) or `nouveau` with Vulkan support |
| AMD | `mesa` + `vulkan-radeon` |
| Intel | `mesa` + `vulkan-intel` |

Verify Vulkan is working: `vulkaninfo --summary`

If you don't have a GPU or Vulkan isn't available, disable GPU in the setup screen — the app will fall back to CPU inference.

## Installation

The easiest way is to use the provided install script:

```bash
git clone https://github.com/optiummusic/Whisper-Real-Time-Transcription.git
cd Whisper-Real-Time-Transcription
chmod +x setup.sh
./setup.sh
```

The script will download the models and build the binary for you.

### Manual installation

**1. Clone the repository**
```bash
git clone https://github.com/optiummusic/Whisper-Real-Time-Transcription.git
cd Whisper-Real-Time-Transcription
```

**2. Download the Silero VAD model**
```bash
wget -P models/vad/ \
  https://github.com/snakers4/silero-vad/raw/v4.0/files/silero_vad.onnx
```

**3. Download Whisper models**

Go to [https://huggingface.co/ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp/tree/main) and download:

| File | Destination | Role |
|------|-------------|------|
| `ggml-tiny-q8_0.bin` | `models/whisper-fast/` | Pass 1 — fast preview |
| `ggml-medium-q8_0.bin` | `models/whisper-accurate/` | Pass 2 — accurate final |

```
models/
├── vad/
│   └── silero_vad.onnx
├── whisper-fast/
│   └── ggml-tiny-q8_0.bin
└── whisper-accurate/
    └── ggml-medium-q8_0.bin
```

You can use other `ggml-*.bin` models as long as they match the naming convention. The app auto-detects the first `.bin` file in each directory.

**4. Build and run**
```bash
cargo build --release
./target/release/translator
```

## Audio setup

The app listens on any input device selectable from the setup screen. For transcribing **system audio** (e.g. a video call or media player), you need to route audio through a virtual sink:

**PipeWire (recommended):**
```bash
# Use pavucontrol → Recording tab → point the app to a monitor source
pavucontrol
```

**PulseAudio:**
```bash
pactl load-module module-loopback
# Then select the loopback source in the app
```

## Configuration

Settings are saved to `config.toml` automatically on exit. Most parameters can be adjusted live in the **Settings panel** on the right side of the main window.

```toml
language = "en"           # Target language for Whisper (en, uk, ru, auto)
device = "pipewire"       # Last selected audio device

use_gpu_fast = true       # GPU acceleration for Pass 1
use_gpu_acc  = true       # GPU acceleration for Pass 2
gpu_device_fast = 0       # GPU index for Pass 1
gpu_device_acc  = 0       # GPU index for Pass 2

speech_probability  = 0.5   # VAD sensitivity (0.1 = very sensitive, 0.9 = strict)
max_silence_chunks  = 14    # Chunks of silence before phrase ends
min_window_secs     = 4.0   # Min audio context sent to Whisper
max_window_secs     = 10.0  # Max audio context sent to Whisper
min_phrase_secs     = 1.0   # Phrases shorter than this are discarded
max_phrase_secs     = 12.0  # Phrases longer than this are force-ended
dump_audio          = false # Dump captured audio to WAV files for debugging
```

### Word replacement dictionary

Edit `dictionary/rules.toml` to override specific words in transcription output:

```toml
[rules]
"apple" = "груша"
"new york" = "нью-йорк"
```

Rules are case-insensitive and applied after transcription.

## Runtime logging

Control log verbosity with `RUST_LOG`:

```bash
RUST_LOG=debug ./target/release/translator     # Full debug output
RUST_LOG=info ./target/release/translator      # Normal (default)
RUST_LOG=warn ./target/release/translator      # Errors and warnings only
```

To enable [tokio-console](https://github.com/tokio-rs/console) for async task inspection:

```bash
RUSTFLAGS="--cfg tokio_unstable" cargo build --release
```

## Dependencies

| Crate | Purpose |
|-------|---------|
| `whisper-rs` | Rust bindings to whisper.cpp |
| `ort` | ONNX Runtime for Silero VAD |
| `cpal` | Cross-platform audio capture |
| `rubato` | High-quality audio resampling |
| `eframe` / `egui` | Immediate-mode GUI |
| `tokio` | Async runtime |
| `ndarray` | Tensor operations for VAD |
| `mimalloc` | High-performance memory allocator |
| `parking_lot` | Fast synchronization primitives |
| `wgpu` | GPU enumeration via Vulkan |
