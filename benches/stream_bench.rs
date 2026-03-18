use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tokio::sync::mpsc;
use std::sync::Arc;

// Импортируем всё необходимое напрямую из твоей библиотеки
// (Предполагается, что в Cargo.toml проект называется "translator")
use translator::{StreamInfo, types::PhraseChunk};

fn bench_process_incoming(c: &mut Criterion) {
    let mut stream = StreamInfo::new();
    
    // Создаем канал. Важно: используем mpsc из tokio, как и в основном коде.
    let (_tx, mut rx) = mpsc::channel(100);

    // Подготавливаем тестовые данные (10 мс аудио)
    let chunk = PhraseChunk {
        phrase_id: 1,
        chunk_id: 1,
        data: Arc::new(vec![0.0; 160]),
        is_last: false,
        short: false,
    };

    c.bench_function("stream_process_10ms_chunk", |b| {
        b.iter(|| {
            // black_box предотвращает оптимизацию компилятором, 
            // которая могла бы просто "выкинуть" вызов функции
            stream.process_incoming(black_box(chunk.clone()), &mut rx)
        })
    });
}

criterion_group!(benches, bench_process_incoming);
criterion_main!(benches);