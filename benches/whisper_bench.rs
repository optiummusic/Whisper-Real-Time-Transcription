use criterion::{black_box, criterion_group, criterion_main, Criterion};
use whisper_rs::{WhisperContext, WhisperContextParameters};
use translator::whisper::engine::{run_whisper, WhisperConfig};
use translator::utils::find_first_file_in_dir; // используем имя библиотеки

fn bench_whisper_passes(c: &mut Criterion) {
    // 1. Подгружаем реальное аудио
    let audio_bytes = include_bytes!("test_audio.raw");
    let audio_data: Vec<f32> = audio_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    
    let duration_s = audio_data.len() as f32 / 16000.0;

    // 2. Находим пути к моделям
    let fast_path = find_first_file_in_dir("models/whisper-fast", "bin")
        .expect("No Whisper model found in models/whisper-fast");
    let acc_path = find_first_file_in_dir("models/whisper-accurate", "bin")
        .expect("No Whisper model found in models/whisper-accurate");

    // 3. Инициализация контекстов (GPU настройки подхватятся из твоих констант, если нужно)
    let ctx_fast = WhisperContext::new_with_params(&fast_path, WhisperContextParameters::default())
        .expect("Failed to load fast model");
    let mut state_fast = ctx_fast.create_state().unwrap();

    let ctx_acc = WhisperContext::new_with_params(&acc_path, WhisperContextParameters::default())
        .expect("Failed to load acc model");
    let mut state_acc = ctx_acc.create_state().unwrap();

    let mut group = c.benchmark_group("Whisper_Inference");
    
    // Полезно для статистики: сколько семплов в секунду обрабатываем
    group.throughput(criterion::Throughput::Elements(audio_data.len() as u64));
    // Увеличим время замера, так как Whisper — штука тяжелая
    group.sample_size(10); 

    group.bench_function(format!("pass1_fast_{:.1}s", duration_s), |b| {
        b.iter(|| {
            run_whisper(
                &mut state_fast, 
                black_box(&audio_data), // исправлено
                black_box(&WhisperConfig::fast())
            )
        })
    });

    group.bench_function(format!("pass2_accurate_{:.1}s", duration_s), |b| {
        b.iter(|| {
            run_whisper(
                &mut state_acc, 
                black_box(&audio_data), // исправлено
                black_box(&WhisperConfig::accurate("Testing context performance."))
            )
        })
    });

    group.finish();
}

criterion_group!(benches, bench_whisper_passes);
criterion_main!(benches);