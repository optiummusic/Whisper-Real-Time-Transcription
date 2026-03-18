**Real-time transcription on Silero VAD and Whisper basis.**
— Two-pass system: accurate model and fast model, aimed to reduce latency.
— Visible output on egui/eframe window.
— Async tokio logic.
— Voice Activation Detection (VAD) based on ONNX Silero runtime.
— Fully offline.

**How to install?**
1) Clone the repository (git clone https://github.com/optiummusic/Whisper-Real-Time-Transcription.git)
2) Download Silero VAD, and Whisper.cpp models.
2.1) (I used) =
   Silero: https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/data/silero_vad.onnx
   Whisper: https://huggingface.co/ggerganov/whisper.cpp/tree/main
   Whisper versions i used: ggml-tiny-q8_0.bin (fast)
                            ggml-medium-q8_0.bin (accurate)
3) Move the models into their respective folders in the "models" folder.
4) Use cargo to run and build the binary.

### In order for the system to listen to your system sounds, you may need to tweak the "Recording" tab to change the channel in pavucontrol.

### 🛠 Prerequisites
Before running `cargo build`, ensure you have the following installed:
**Arch Linux:**
```bash
sudo pacman -S alsa-lib portaudio cmake clang
```
