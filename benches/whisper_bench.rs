use criterion::{criterion_group, criterion_main, Criterion};
use whisper_rs::{WhisperContext, WhisperContextParameters};
use translator::whisper::engine::{run_whisper, WhisperConfig};
use translator::utility::utils::find_first_file_in_dir;

fn bench_whisper_passes(c: &mut Criterion) {
    let audio_bytes = include_bytes!("test_audio.raw");
    let audio_data: Vec<f32> = audio_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    
    let duration_s = audio_data.len() as f32 / 16000.0;

    let fast_path = find_first_file_in_dir("models/whisper-fast", "bin")
        .expect("No Whisper model found in models/whisper-fast");
    let acc_path = find_first_file_in_dir("models/whisper-accurate", "bin")
        .expect("No Whisper model found in models/whisper-accurate");

    let ctx_fast = WhisperContext::new_with_params(&fast_path, WhisperContextParameters::default())
        .expect("Failed to load fast model");
    let mut state_fast = ctx_fast.create_state().unwrap();

    let ctx_acc = WhisperContext::new_with_params(&acc_path, WhisperContextParameters::default())
        .expect("Failed to load acc model");
    let mut state_acc = ctx_acc.create_state().unwrap();

    let mut group = c.benchmark_group("Whisper_Inference");
    
    group.throughput(criterion::Throughput::Elements(audio_data.len() as u64));
    group.sample_size(10); 

    group.bench_function(format!("pass1_fast_{:.1}s", duration_s), |b| {
        b.iter_custom(|iters| {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                run_whisper(&mut state_fast, &audio_data, &WhisperConfig::fast());
            }
            let elapsed = start.elapsed();
            let rtf = elapsed.as_secs_f32() / (iters as f32 * duration_s);
            eprintln!("RTF fast: {:.3}x realtime", rtf);
            elapsed
        })
    });

    group.bench_function(format!("pass2_accurate_{:.1}s", duration_s), |b| {
        b.iter_custom(|iters| {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                run_whisper(&mut state_acc, &audio_data, &WhisperConfig::accurate("Testing context performance."));
            }
            let elapsed = start.elapsed();
            let rtf = elapsed.as_secs_f32() / (iters as f32 * duration_s);
            eprintln!("RTF acc: {:.3}x realtime", rtf);
            elapsed
        })
    });

    group.finish();
}

criterion_group!(benches, bench_whisper_passes);
criterion_main!(benches);