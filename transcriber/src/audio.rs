use crate::types::AudioPacket;
use audioadapter_buffers::direct::InterleavedSlice;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rubato::{
    Async, FixedAsync, Indexing, Resampler, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};
use std::process::Command;
#[cfg(target_os = "linux")]
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::TrySendError;
use tracing::{info, warn};

use crate::config::{TARGET_SAMPLE_RATE, VAD_CHUNK_SIZE};

static PEAK_LEVEL: AtomicU32 = AtomicU32::new(0);

#[cfg(target_os = "linux")]
static LINUX_VIRTUAL_DEVICE_ID: AtomicU32 = AtomicU32::new(0);

#[cfg(target_os = "linux")]
static LINUX_LOOPBACK_ID: AtomicU32 = AtomicU32::new(0);

#[cfg(target_os = "linux")]
static PREVIOUS_DEFAULT_SOURCE: Mutex<Option<String>> = Mutex::new(None);

#[cfg(target_os = "linux")]
fn pa_get_default_source() -> Option<String> {
    let out = Command::new("pactl")
        .args(["get-default-source"])
        .output()
        .ok()?;
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() { None } else { Some(s) }
}

#[cfg(target_os = "linux")]
fn pa_get_default_sink() -> Option<String> {
    let out = Command::new("pactl")
        .args(["get-default-sink"])
        .output()
        .ok()?;
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() { None } else { Some(s) }
}

#[cfg(target_os = "linux")]
fn pa_set_default_source(name: &str) {
    let status = Command::new("pactl")
        .args(["set-default-source", name])
        .status();
    match status {
        Ok(s) if s.success() => info!("PA default source → {}", name),
        Ok(s) => warn!("set-default-source failed: exit={}", s),
        Err(e) => warn!("set-default-source error: {}", e),
    }
}

#[allow(deprecated)]
fn find_device(device_name: &str) -> Option<cpal::Device> {
    let host = cpal::default_host();

    #[cfg(target_os = "linux")]
    if device_name.starts_with("[Monitor]") {
        let pa_name = device_name
            .trim_start_matches("[Monitor] ")
            .replace(' ', "_")
            + ".monitor";

        pa_set_default_source(&pa_name);
        std::thread::sleep(std::time::Duration::from_millis(150));

        return host.default_input_device();
    }

    #[cfg(target_os = "linux")]
    {
        let current = pa_get_default_source().unwrap_or_default();
        if current.ends_with(".monitor")
            && let Ok(prev) = PREVIOUS_DEFAULT_SOURCE.lock()
                && let Some(name) = prev.as_ref() {
                    info!("Switching back from monitor to: {}", name);
                    pa_set_default_source(name);
                    std::thread::sleep(std::time::Duration::from_millis(150));
                }
    }

    #[cfg(target_os = "windows")]
    if device_name.ends_with(" [Loopback]") {
        let real_name = device_name.trim_end_matches(" [Loopback]");
        return host
            .output_devices()
            .ok()?
            .find(|d| d.name().map(|n| n == real_name).unwrap_or(false));
    }

    host.input_devices().ok()?.find(|d| {
        d.name()
            .map(|name| {
                let n = name.to_lowercase();
                let target = device_name.to_lowercase();
                n == target || n.contains(&target)
            })
            .unwrap_or(false)
    })
}

#[allow(deprecated)]
pub fn get_input_devices() -> Vec<String> {
    let host = cpal::default_host();
    let mut available_devices = Vec::new();

    if let Ok(devices) = host.input_devices() {
        for device in devices {
            if let Ok(name) = device.name() {
                let n = name.to_lowercase();

                let trash_keywords = [
                    "null",
                    "oss",
                    "lavrate",
                    "upmix",
                    "vdownmix",
                    "usbstream",
                    "hw:",
                    "plughw:",
                    "speex",
                    "jack",
                    "dmix",
                    "dsnoop",
                    "front",
                    "surround",
                    "iec958",
                    "default:",
                    "samplerate",
                ];

                let is_trash = trash_keywords.iter().any(|&k| n.contains(k));

                if !is_trash {
                    available_devices.push(name);
                }
            }
        }
    }

    #[cfg(target_os = "linux")]
    {
        if let Ok(out) = Command::new("pactl")
            .args(["list", "sources", "short"])
            .output()
        {
            let text = String::from_utf8_lossy(&out.stdout);
            for line in text.lines() {
                let cols: Vec<&str> = line.split_whitespace().collect();
                if cols.len() >= 2 && cols[1].ends_with(".monitor") {
                    let raw = cols[1];

                    if raw.contains("Whisper") || !raw.contains("alsa_output") {
                        let display = raw.trim_end_matches(".monitor").replace('_', " ");
                        available_devices.push(format!("[Monitor] {display}"));
                    }
                }
            }
        }
    }

    #[cfg(target_os = "windows")]
    if let Ok(devices) = host.output_devices() {
        for device in devices {
            if device.default_output_config().is_ok() {
                if let Ok(name) = device.name() {
                    available_devices.push(format!("{} [Loopback]", name));
                }
            }
        }
    }

    available_devices.sort();
    available_devices.dedup();
    available_devices
}

fn make_resampler(source_rate: f64, chunk_size: usize) -> Async<f32> {
    let params = SincInterpolationParameters {
        sinc_len: 64,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 64,
        window: WindowFunction::BlackmanHarris2,
    };
    Async::<f32>::new_sinc(
        TARGET_SAMPLE_RATE as f64 / source_rate,
        2.0,
        &params,
        chunk_size,
        1,
        FixedAsync::Input,
    )
    .expect("Failed to create rubato resampler")
}

#[allow(deprecated)] // !!!DEVICE.NAME DEPRECATED!!!
pub fn start_preview(device_name: &str) -> Option<cpal::Stream> {
    let device = find_device(device_name)?;

    let config = device
        .default_input_config()
        .or_else(|_| device.default_output_config())
        .ok()?;

    device
        .build_input_stream(
            &config.into(),
            |data: &[f32], _| {
                let max = data.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                update_peak(max);
            },
            |err| tracing::error!("Preview error: {err}"),
            None,
        )
        .ok()
        .and_then(|s| {
            use cpal::traits::StreamTrait;
            s.play().ok().map(|_| s)
        })
}

pub fn get_ui_level() -> f32 {
    PEAK_LEVEL.load(Ordering::Relaxed) as f32 / 1000.0
}

fn update_peak(raw_peak: f32) {
    let normalized = raw_peak.sqrt().min(1.0);

    let current = PEAK_LEVEL.load(Ordering::Relaxed) as f32 / 1000.0;

    let display_level = if normalized > current {
        normalized
    } else {
        current * 0.85
    };

    PEAK_LEVEL.store((display_level * 1000.0) as u32, Ordering::Relaxed);
}

fn create_audio_stream(
    device: &cpal::Device,
    tx: mpsc::Sender<AudioPacket>,
) -> Result<cpal::Stream, Box<dyn std::error::Error>> {
    let config = device
        .default_input_config()
        .or_else(|_| device.default_output_config())?;
    let source_rate = config.sample_rate() as f64;
    let channels = config.channels() as usize;
    let needs_resample = (source_rate - TARGET_SAMPLE_RATE as f64).abs() > 1.0;

    let resampler_input_size = if needs_resample {
        ((VAD_CHUNK_SIZE as f64 * source_rate / TARGET_SAMPLE_RATE as f64).ceil() as usize).max(64)
    } else {
        VAD_CHUNK_SIZE
    };

    info!(
        "Audio stream config: rate={}, channels={}, resample={}",
        source_rate, channels, needs_resample
    );
    info!("Resampler input size goal: {}", resampler_input_size);

    let mut resampler = if needs_resample {
        Some(make_resampler(source_rate, resampler_input_size))
    } else {
        None
    };

    let mut raw_buf: Vec<f32> = Vec::with_capacity(resampler_input_size * 2);
    let mut accumulator: Vec<f32> = Vec::with_capacity(VAD_CHUNK_SIZE * 4);

    let mut output_buf = vec![0.0f32; resampler_input_size * 4];
    let mut indexing = Indexing {
        input_offset: 0,
        output_offset: 0,
        active_channels_mask: None,
        partial_len: None,
    };

    let mut sample_metrics = Vec::with_capacity(1000);

    let stream = device.build_input_stream(
        &config.into(),
        move |data: &[f32], _| {
            let current_gain = crate::config::audio_gain();
            let local_max = data.iter()
                .fold(0.0f32, |m, &v| m.max(v.abs())) * current_gain;

            update_peak(local_max);

            let mono_iter = data.chunks_exact(channels)
                .map(|ch| {
                    let sum: f32 = ch.iter().sum();
                    let amplified = (sum / channels as f32) * current_gain;
                    amplified.clamp(-1.0, 1.0)
                });

            if let Some(ref mut rs) = resampler {
                raw_buf.extend(mono_iter);

                while raw_buf.len() >= resampler_input_size {
                    let t0 = std::time::Instant::now();

                    let input = InterleavedSlice::new(&raw_buf[..resampler_input_size], 1, resampler_input_size).unwrap();
                    let len = output_buf.len();
                    let mut output = InterleavedSlice::new_mut(&mut output_buf, 1, len).unwrap();

                    indexing.input_offset = 0;
                    indexing.output_offset = 0;

                    match rs.process_into_buffer(&input, &mut output, Some(&indexing)) {
                        Ok((_, frames_out)) => {
                            let produced = frames_out;
                            accumulator.extend_from_slice(&output_buf[..produced]);
                        }
                        Err(e) => warn!("Resampler error: {}", e),
                    }
                    raw_buf.drain(..resampler_input_size);
                    sample_metrics.push(t0.elapsed().as_micros());
                    if sample_metrics.len() >= 100 {
                        let avg = (sample_metrics.iter().sum::<u128>() as f32 / 100.0) / 1000.0; // в мс
                        crate::utility::utils::performance(avg, "audio_resample_avg_100".to_string());
                        sample_metrics.clear();
                    }
                }
            } else {
                accumulator.extend(mono_iter);
            }

            while accumulator.len() >= VAD_CHUNK_SIZE {
                let chunk: Vec<f32> = accumulator.drain(..VAD_CHUNK_SIZE).collect();
                let t_send = std::time::Instant::now();
                match tx.try_send(chunk) {
                    Ok(()) => {
                        let dur = t_send.elapsed().as_micros();
                        if dur > 1000 {
                            crate::utility::utils::performance(dur as f32 / 1000.0, "audio_tx_slowdown_warn".to_string());
                        }
                    }
                    Err(e) => match e {
                        TrySendError::Full(_) => {
                            warn!(
                                target: "audio_source",
                                "Audio buffer OVERFLOW! VAD task is too slow. Dropping audio data..."
                            );
                            crate::utility::stats::get().audio_overflow.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                        TrySendError::Closed(_) => {
                            warn!("Audio receiver (VAD) closed. Stream will continue but data is lost.");
                        }
                    },
                }
            }
        },
        |err| eprintln!("Audio stream error: {}", err),
        None,
    )?;
    Ok(stream)
}

#[allow(deprecated)] // !!!DEVICE.NAME DEPRECATED!!!
pub fn start_listening(
    device_name: &str,
    tx: mpsc::Sender<AudioPacket>,
) -> Result<cpal::Stream, Box<dyn std::error::Error>> {
    let device = find_device(device_name)
        .or_else(|| cpal::default_host().default_input_device())
        .expect("No input device found");

    if let Ok(name) = device.name() {
        info!("Using audio device: {}", name);
    }

    let stream = create_audio_stream(&device, tx)?;
    stream.play()?;
    Ok(stream)
}

#[cfg(target_os = "linux")]
pub fn setup_linux_virtual_sink(description: &str) -> bool {
    let sink_name = "Whisper_Virtual_Sink";
    if let Ok(mut prev) = PREVIOUS_DEFAULT_SOURCE.lock()
        && prev.is_none() {
            *prev = pa_get_default_source();
            info!("Saved previous default source: {:?}", *prev);
        }
    let out = Command::new("pactl")
        .args([
            "load-module",
            "module-null-sink",
            &format!("sink_name={}", sink_name),
            &format!("sink_properties=device.description=\"{}\"", description),
        ])
        .output();

    let sink_id = match out {
        Ok(o) if o.status.success() => {
            match String::from_utf8_lossy(&o.stdout).trim().parse::<u32>() {
                Ok(id) => id,
                Err(e) => {
                    warn!("Failed to parse sink module ID: {}", e);
                    return false;
                }
            }
        }
        Ok(o) => {
            warn!(
                "module-null-sink failed: {}",
                String::from_utf8_lossy(&o.stderr)
            );
            return false;
        }
        Err(e) => {
            warn!("pactl error: {}", e);
            return false;
        }
    };
    LINUX_VIRTUAL_DEVICE_ID.store(sink_id, Ordering::Relaxed);
    info!(
        "Created virtual sink '{}' (module ID: {})",
        sink_name, sink_id
    );

    let Some(real_sink) = pa_get_default_sink() else {
        warn!("Could not get default sink — loopback not created");
        return true;
    };
    let monitor_source = format!("{}.monitor", real_sink);

    let lb_out = Command::new("pactl")
        .args([
            "load-module",
            "module-loopback",
            &format!("source={}", monitor_source),
            &format!("sink={}", sink_name),
            "latency_msec=0",
            "adjust_time=0",
        ])
        .output();

    match lb_out {
        Ok(o) if o.status.success() => {
            match String::from_utf8_lossy(&o.stdout).trim().parse::<u32>() {
                Ok(id) => {
                    LINUX_LOOPBACK_ID.store(id, Ordering::Relaxed);
                    info!(
                        "Loopback: {} → {} (module ID: {})",
                        monitor_source, sink_name, id
                    );
                }
                Err(e) => warn!("Failed to parse loopback module ID: {}", e),
            }
            true
        }
        Ok(o) => {
            warn!(
                "module-loopback failed: {}",
                String::from_utf8_lossy(&o.stderr)
            );
            false
        }
        Err(e) => {
            warn!("pactl error: {}", e);
            false
        }
    }
}

#[cfg(target_os = "linux")]
pub fn cleanup_linux_virtual_sink() {
    if let Ok(mut prev) = PREVIOUS_DEFAULT_SOURCE.lock()
        && let Some(name) = prev.take() {
            info!("Restoring default source: {}", name);
            pa_set_default_source(&name);
        }

    let loopback_id = LINUX_LOOPBACK_ID.swap(0, Ordering::Relaxed);
    if loopback_id != 0 {
        let _ = Command::new("pactl")
            .args(["unload-module", &loopback_id.to_string()])
            .status();
        info!("Unloaded loopback module ID: {}", loopback_id);
    }

    let sink_id = LINUX_VIRTUAL_DEVICE_ID.swap(0, Ordering::Relaxed);
    if sink_id != 0 {
        let _ = Command::new("pactl")
            .args(["unload-module", &sink_id.to_string()])
            .status();
        info!("Unloaded virtual sink module ID: {}", sink_id);
    }
}

pub struct LinuxCleanupGuard;
impl Drop for LinuxCleanupGuard {
    fn drop(&mut self) {
        #[cfg(target_os = "linux")]
        cleanup_linux_virtual_sink();
    }
}
