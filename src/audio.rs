use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use tokio::sync::mpsc;
use crate::types::AudioPacket;
use crate::TARGET_SAMPLE_RATE;
use crate::VAD_CHUNK_SIZE;
use tracing::{warn, info, trace};

fn find_device(host: &cpal::Host) -> Option<cpal::Device> {
    let devices: Vec<_> = host.input_devices().ok()?.collect();
    
    let get_desc_name = |d: &cpal::Device| -> String {
        d.name().ok()
            .or_else(|| {
                None 
            })
            .unwrap_or_default()
            .to_lowercase()
    };

    for d in &devices {
        if let Ok(name) = d.name() {
            trace!("Available input device: {}", name);
        }
    }

    let monitor = devices.iter().find(|d| {
        let name = get_desc_name(d);
        name.contains("monitor") || name.contains("loopback")
    });

    if let Some(d) = monitor {
        if let Ok(n) = d.name() { info!("Selected system monitor: {}", n); }
        return Some(d.clone());
    }

    let sound_server = devices.iter().find(|d| {
        let name = get_desc_name(d);
        name.contains("pipewire") || name.contains("pulse")
    });

    if let Some(d) = sound_server {
        if let Ok(n) = d.name() { info!("Selected sound server device: {}", n); }
        return Some(d.clone());
    }

    host.default_input_device().inspect(|d| {
        if let Ok(n) = d.name() {
            info!("Falling back to default device: {}", n);
        }
    })
}

fn resample_mono(input: &[f32], ratio: f64, out: &mut Vec<f32>) {
    if input.is_empty() { return; }
    let last = input.len() - 1;
    let mut pos = 0.0f64;
    loop {
        let idx = pos as usize;
        if idx >= last { break; }
        let frac = (pos - idx as f64) as f32;
        out.push(input[idx] * (1.0 - frac) + input[idx + 1] * frac);
        pos += 1.0 / ratio;
    }
}

fn create_audio_stream(
    device: &cpal::Device,
    tx: mpsc::Sender<AudioPacket>,
) -> Result<cpal::Stream, Box<dyn std::error::Error>> {
    let config      = device.default_input_config()?;
    let source_rate = config.sample_rate() as f64;
    let channels    = config.channels() as usize;
    let ratio       = TARGET_SAMPLE_RATE as f64 / source_rate;

    let mut accumulator   = Vec::<f32>::with_capacity(VAD_CHUNK_SIZE * 2);
    let mut resampled_buf = Vec::<f32>::with_capacity(VAD_CHUNK_SIZE * 2);

    let stream = device.build_input_stream(
        &config.into(),
        move |data: &[f32], _| {
            let mono_iter = data.chunks_exact(channels)
                .map(|ch| ch.iter().sum::<f32>() / channels as f32);
            if (ratio - 1.0).abs() < 1e-6 {
                let mono: Vec<f32> = mono_iter.collect();
                accumulator.extend_from_slice(&mono);
            } else {
                let raw: Vec<f32> = mono_iter.collect();
                resampled_buf.clear();
                resample_mono(&raw, ratio, &mut resampled_buf);
                accumulator.append(&mut resampled_buf); // move, no clone
            };
            while accumulator.len() >= VAD_CHUNK_SIZE {
                let chunk: Vec<f32> = accumulator.drain(..VAD_CHUNK_SIZE).collect();
                match tx.try_send(chunk) {
                    Ok(()) => {}
                    Err(e) => warn!(%e, "audio chunk dropped"),
                }
            }
        },
        |err| eprintln!("Audio stream error: {}", err),
        None,
    )?;
    Ok(stream)
}

pub fn start_listening(tx: mpsc::Sender<AudioPacket>) -> Result<cpal::Stream, Box<dyn std::error::Error>> {
    let host = cpal::default_host();
    let device = find_device(&host).expect("No input device found");
    if let Ok(desc) = device.description() {
        info!("Using audio device: {:?}", desc.name());
    }

    let stream = create_audio_stream(&device, tx)?;
    stream.play()?;
    Ok(stream)
}