use criterion::{black_box, criterion_group, criterion_main, Criterion};
use translator::vad::VadEngine;
use std::path::Path;

fn bench_vad_process(c: &mut Criterion) {
    let vad_path = Path::new("models/vad/silero_vad.onnx");
    let mut engine = VadEngine::new(&vad_path);
    
    let fake_audio = vec![0.0f32; 480];

    c.bench_function("vad_process_chunk", |b| {
        b.iter(|| {
            engine.process(black_box(fake_audio.clone()))
        })
    });
}

criterion_group!(benches, bench_vad_process);
criterion_main!(benches);