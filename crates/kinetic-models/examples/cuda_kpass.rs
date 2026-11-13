//! Times a pass of k drafted tokens against a one-token pass, after a
//! prompt, on the GPU: what speculative decoding pays per round. Reports
//! greedy passes, whose logits stay on the device, and passes that return
//! every row's logits, as sampling needs.
//!
//! cargo run --release -p kinetic-models --features cuda --example cuda_kpass -- model.gguf prompt.txt [devices] [tensor]
//!
//! `devices` is a list such as `0,1`; `tensor` splits every layer over them
//! instead of handing each a run of layers.

use std::time::Instant;

use kinetic_gguf::GgufFile;
use kinetic_models::cuda::{CudaModel, Placement};
use kinetic_tokenizer::Tokenizer;

const RUNS: usize = 8;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: cuda_kpass <model.gguf> <prompt.txt> [devices] [tensor]";
    let path = args.next().ok_or(usage)?;
    let prompt = std::fs::read_to_string(args.next().ok_or(usage)?)?;
    let devices: Vec<usize> = match args.next() {
        Some(d) => d.split(',').map(str::parse).collect::<Result<_, _>>()?,
        None => Vec::new(),
    };
    let tensor = args.next().as_deref() == Some("tensor");

    let file = GgufFile::open(&path)?;
    let tokenizer = Tokenizer::from_gguf(file.header())?;
    let model = kinetic_models::load(&file)?;
    let ids = tokenizer.encode(&prompt, true);
    let placement = Placement {
        devices,
        tensor,
        ..Placement::default()
    };
    let gpu = CudaModel::load_with(&model, ids.len() + 64, placement)?;
    let mut state = gpu.new_state()?;
    let (_, hidden) = gpu.run_greedy(&mut state, &ids, false)?;
    let draft: Vec<u32> = ids.iter().rev().take(16).copied().collect();
    println!("prompt {} tokens", ids.len());
    println!(" k   greedy ms     x   logits ms     x");
    let (mut greedy_one, mut logits_one) = (0.0, 0.0);
    for k in 1..=16 {
        let mut time = |logits: bool| -> Result<f64, Box<dyn std::error::Error>> {
            let mut best = f64::INFINITY;
            // The first run records the pass; the rest replay it.
            for _ in 0..RUNS {
                gpu.checkpoint(&mut state)?;
                let start = Instant::now();
                if logits {
                    gpu.run(&mut state, &draft[..k], true)?;
                } else {
                    gpu.run_greedy(&mut state, &draft[..k], true)?;
                }
                best = best.min(start.elapsed().as_secs_f64());
                gpu.rollback(&mut state, 0)?;
            }
            Ok(best * 1e3)
        };
        let (g, l) = (time(false)?, time(true)?);
        if k == 1 {
            (greedy_one, logits_one) = (g, l);
        }
        println!(
            "{k:>2}  {g:>9.2}  {:>5.2}  {l:>9.2}  {:>5.2}",
            g / greedy_one,
            l / logits_one
        );
    }
    if let Some(mut mtp) = gpu.new_mtp_state()? {
        let d = gpu.n_embd();
        let last = &hidden[hidden.len() - d..];
        gpu.mtp_forward(&mut mtp, last, &draft[..1])?;
        let (mut forward, mut drafted) = (f64::INFINITY, f64::INFINITY);
        for _ in 0..RUNS {
            gpu.mtp_checkpoint(&mut mtp)?;
            let start = Instant::now();
            let h = gpu.mtp_forward(&mut mtp, last, &draft[1..2])?;
            forward = forward.min(start.elapsed().as_secs_f64());
            let start = Instant::now();
            gpu.draft_token(&mut mtp, &h, 32768)?;
            drafted = drafted.min(start.elapsed().as_secs_f64());
            gpu.mtp_rollback(&mut mtp, 1, 0)?;
        }
        println!(
            "prediction block: a row {:.2} ms, a drafted token {:.2} ms",
            forward * 1e3,
            drafted * 1e3
        );
    }
    Ok(())
}
