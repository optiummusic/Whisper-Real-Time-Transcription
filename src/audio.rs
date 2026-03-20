use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::TrySendError;
use crate::types::AudioPacket;
use tracing::{warn, info, trace};
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

use crate::config::{
    TARGET_SAMPLE_RATE, VAD_CHUNK_SIZE, STREAM_CHUNK_SAMPLES, STITCH_MIN_SAMPLES
};

#[allow(deprecated)] // !!!DEVICE.NAME DEPRECATED!!!
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
        name.contains("monitor") || name.contains("loopback") || name.contains("stereo mix") || name.contains("wave out mix") 
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
 
    let stream = device.build_input_stream(
        &config.into(),
        move |data: &[f32], _| {
            let mono_iter = data.chunks_exact(channels)
                .map(|ch| ch.iter().sum::<f32>() / channels as f32);
 
            if let Some(ref mut rs) = resampler {
                raw_buf.extend(mono_iter);

                while raw_buf.len() >= resampler_input_size {
                    let block: Vec<f32> = raw_buf.drain(..resampler_input_size).collect();

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
                }
            } else {
                accumulator.extend(mono_iter);
            }
 
            while accumulator.len() >= VAD_CHUNK_SIZE {
                let chunk: Vec<f32> = accumulator.drain(..VAD_CHUNK_SIZE).collect();
                match tx.try_send(chunk) {
                    Ok(()) => {
                        // trace!("Audio: chunk sent to VAD");
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