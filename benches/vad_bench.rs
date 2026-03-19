use criterion::{black_box, criterion_group, criterion_main, Criterion};
use translator::vad::VadEngine;
use std::path::Path;
use translator::config::VAD_CHUNK_SIZE;

fn bench_vad_process(c: &mut Criterion) {
    let vad_path = Path::new("models/vad/silero_vad.onnx");
    let mut engine = VadEngine::new(&vad_path);
    let mut results = Vec::with_capacity(4);
    
    // ВАЖНО: убедись, что размер совпадает с VAD_CHUNK_SIZE (обычно 512 или 1536)
    let fake_audio = vec![0.0f32; VAD_CHUNK_SIZE]; 

    c.bench_function("vad_process", |b| {
        // Используем iter_with_setup, чтобы избежать аллокации клона в самом цикле
        b.iter_batched(
            || fake_audio.clone(),
            |data| {
                results.clear();
                engine.process(data, &mut results);
            },
            criterion::BatchSize::SmallInput
        );
    });
    c.bench_function("vad_only", |b| {
        b.iter(|| {
            black_box(engine.run_vad(&fake_audio));
        })
    });
}

criterion_group!(benches, bench_vad_process);
criterion_main!(benches);