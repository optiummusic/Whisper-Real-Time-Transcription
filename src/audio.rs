use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::TrySendError;
use crate::types::AudioPacket;
use tracing::{warn, info };
use rubato::{
    Async,
    FixedAsync, 
    SincInterpolationParameters, 
    SincInterpolationType, 
    WindowFunction,
    Resampler,
    Indexing
};
use audioadapter_buffers::direct::InterleavedSlice;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::config::{
    TARGET_SAMPLE_RATE, VAD_CHUNK_SIZE
};
static PEAK_LEVEL: AtomicU32 = AtomicU32::new(0);

#[allow(deprecated)] // !!!DEVICE.NAME DEPRECATED!!!
pub fn get_input_devices() -> Vec<String> {
    let host = cpal::default_host();
    host.input_devices()
        .into_iter()
        .flatten()
        .filter(|device| {
            if device.default_input_config().is_err() {
                return false;
            }

            if let Ok(name) = device.name() {
                let n = name.to_lowercase();
                let is_trash = n.contains("null") 
                    || n.contains("oss") 
                    || n.contains("lavrate") 
                    || n.contains("upmix")
                    || n.contains("vdownmix");
                
                return !is_trash;
            }
            false
        })
        .filter_map(|d| d.name().ok())
        .collect()
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
    let host = cpal::default_host();
    let device = host.input_devices().ok()?.into_iter()
        .find(|d| d.name().map(|n| n == device_name).unwrap_or(false))?;

    let config = device.default_input_config().ok()?;
    
    device.build_input_stream(
        &config.into(),
        |data: &[f32], _| {
            let max = data.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            update_peak(max);
        },
        |err| tracing::error!("Preview error: {err}"),
        None
    ).ok().and_then(|s| {
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
    let config      = device.default_input_config()?;
    let source_rate = config.sample_rate() as f64;
    let channels    = config.channels() as usize;
    let needs_resample = (source_rate - TARGET_SAMPLE_RATE as f64).abs() > 1.0;

 
    let resampler_input_size = if needs_resample {
        ((VAD_CHUNK_SIZE as f64 * source_rate / TARGET_SAMPLE_RATE as f64).ceil() as usize)
            .max(64)
    } else {
        VAD_CHUNK_SIZE
    };

    info!("Audio stream config: rate={}, channels={}, resample={}", source_rate, channels, needs_resample);
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

    let mut sample_metrics = Vec::with_capacity(100);
 
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
                    let block: Vec<f32> = raw_buf.drain(..resampler_input_size).collect();
                    let t0 = std::time::Instant::now();

                    let input_frames = block.len();

                    let input = InterleavedSlice::new(&block, 1, input_frames).unwrap();
                    let len = output_buf.len();
                    let mut output =
                        InterleavedSlice::new_mut(&mut output_buf, 1, len).unwrap();

                    indexing.input_offset = 0;
                    indexing.output_offset = 0;

                    match rs.process_into_buffer(&input, &mut output, Some(&indexing)) {
                        Ok((_, frames_out)) => {
                            let produced = frames_out;
                            accumulator.extend_from_slice(&output_buf[..produced]);
                        }
                        Err(e) => warn!("Resampler error: {}", e),
                    }
                    sample_metrics.push(t0.elapsed().as_micros());
                    if sample_metrics.len() >= 100 {
                        let avg = (sample_metrics.iter().sum::<u128>() as f32 / 100.0) / 1000.0; // в мс
                        crate::utils::performance(avg, "audio_resample_avg_100".to_string());
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
                            crate::utils::performance(dur as f32 / 1000.0, "audio_tx_slowdown_warn".to_string());
                        }
                    }
                    Err(e) => match e {
                        TrySendError::Full(_) => {
                            warn!(
                                target: "audio_source",
                                "Audio buffer OVERFLOW! VAD task is too slow. Dropping audio data..."
                            );
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
    tx: mpsc::Sender<AudioPacket>) -> Result<cpal::Stream, Box<dyn std::error::Error>> {
    let host = cpal::default_host();
    let device = host.input_devices()
        .into_iter()
        .flatten()
        .find(|d| d.name().ok().as_deref() == Some(device_name))
        .or_else(|| host.default_input_device())
        .expect("No input device found");

    if let Ok(name) = device.name() {
        info!("Using audio device: {}", name);
    }

    let stream = create_audio_stream(&device, tx)?;
    stream.play()?;
    Ok(stream)
}