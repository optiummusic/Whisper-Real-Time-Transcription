// use criterion::{Criterion, black_box, criterion_group, criterion_main};
// use std::sync::Arc;
// use tokio::sync::mpsc;

// use translator::types::PhraseChunk;
// use translator::whisper::stream_info::StreamInfo;
// fn bench_process_incoming(c: &mut Criterion) {
//     let mut stream = StreamInfo::new();

//     let (_tx, mut rx) = mpsc::channel(100);

//     let chunk = PhraseChunk {
//         phrase_id: 1,
//         chunk_id: 1,
//         data: Arc::new(vec![0.0; 160]),
//         is_last: false,
//         short: false,
//     };

//     c.bench_function("stream_process_10ms_chunk", |b| {
//         b.iter(|| stream.process_incoming(black_box(chunk.clone()), &mut rx))
//     });
// }

// criterion_group!(benches, bench_process_incoming);
// criterion_main!(benches);
