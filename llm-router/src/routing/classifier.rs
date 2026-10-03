//! Request-classifier evaluation harness.
//!
//! ```sh
//! EVAL_SET=/tmp/classifier-eval.json OUT=/tmp/classifier-out.jsonl \
//!   cargo run --release -p nasiko-llm-router --example classifier_eval
//! ```
//!
//! * `EVAL_SET` — JSON file with an `examples` array (a bare array also works). Each case:
//!   `{id?, query, context?, request_type?, complexity?}`. Labels are optional; with them a
//!   summary (accuracy, complexity error, ECE, latency) is printed to stderr.
//! * `OUT` — optional. One JSONL line per case:
//!   `{"id","request_type","complexity","confidence","latency_us","fallback"}`.
//!   Outputs only; scoring happens elsewhere.
//! * Backend: `CLASSIFIER_BACKEND` (`regex` default | `http`), `CLASSIFIER_MODEL`,
//!   `CLASSIFIER_ENDPOINT`, `CLASSIFIER_API_KEY`, `CLASSIFIER_TIMEOUT_MS` (5000),
//!   `CLASSIFIER_MIN_CONFIDENCE` (0.5). Same variables as the router binary.
//!
//! Every case goes through [`classify_with_fallback`] — the function the router calls — so
//! this exercises the production path, including regex fallback on error/timeout/low
//! confidence. Per-call `latency_us` covers only the classify call (backend construction,
//! i.e. one-time load, happens before the loop). A warm-up call is made first and not
//! recorded.

use std::io::Write;
use std::time::Instant;

use nasiko_llm_router::routing::classifier::{
    Classification, ClassifierSettings, ClassifierStats, ClassifyInput, RegexRequestClassifier,
    RequestClassifier, RequestType, build_classifier, classify_with_fallback,
};
use serde_json::{Value, json};

struct Case {
    id: String,
    query: String,
    context: Option<String>,
    request_type: Option<RequestType>,
    complexity: Option<u8>,
}

struct Row {
    c: Classification,
    latency_us: u128,
    fallback: bool,
}

fn env(k: &str) -> Option<String> {
    std::env::var(k).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty())
}

fn settings_from_env() -> ClassifierSettings {
    let d = ClassifierSettings::default();
    ClassifierSettings {
        backend: env("CLASSIFIER_BACKEND").unwrap_or(d.backend),
        model: env("CLASSIFIER_MODEL").unwrap_or(d.model),
        endpoint: env("CLASSIFIER_ENDPOINT").unwrap_or(d.endpoint),
        api_key: env("CLASSIFIER_API_KEY"),
        timeout_ms: env("CLASSIFIER_TIMEOUT_MS").and_then(|v| v.parse().ok()).unwrap_or(d.timeout_ms),
        min_confidence: env("CLASSIFIER_MIN_CONFIDENCE")
            .and_then(|v| v.parse().ok())
            .unwrap_or(d.min_confidence),
    }
}

fn load_cases(path: &str) -> Vec<Case> {
    let raw = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("cannot read EVAL_SET {path}: {e}"));
    let v: Value = serde_json::from_str(&raw).unwrap_or_else(|e| panic!("EVAL_SET is not valid JSON: {e}"));
    let items = v
        .get("examples")
        .and_then(|e| e.as_array())
        .or_else(|| v.as_array())
        .unwrap_or_else(|| panic!("EVAL_SET needs an `examples` array"));
    items
        .iter()
        .enumerate()
        .map(|(i, it)| Case {
            id: it.get("id").and_then(|x| x.as_str()).map(str::to_string).unwrap_or_else(|| format!("case-{:03}", i + 1)),
            query: it.get("query").and_then(|x| x.as_str()).unwrap_or_default().to_string(),
            context: it.get("context").and_then(|x| x.as_str()).map(str::to_string),
            request_type: it.get("request_type").and_then(|x| x.as_str()).and_then(RequestType::from_wire),
            complexity: it.get("complexity").and_then(|x| x.as_u64()).map(|c| c as u8),
        })
        .collect()
}

async fn run(backend: &dyn RequestClassifier, cases: &[Case]) -> (Vec<Row>, ClassifierStats) {
    let stats = ClassifierStats::default();
    // Warm-up (connection setup etc.) — not recorded and not counted.
    if let Some(first) = cases.first() {
        let _ = backend
            .classify(&ClassifyInput { query: &first.query, context: first.context.as_deref() })
            .await;
    }
    let mut rows = Vec::with_capacity(cases.len());
    for case in cases {
        let input = ClassifyInput { query: &case.query, context: case.context.as_deref() };
        let start = Instant::now();
        let (c, fallback) = classify_with_fallback(backend, &input, Some(&stats)).await;
        rows.push(Row { c, latency_us: start.elapsed().as_micros(), fallback });
    }
    (rows, stats)
}

fn percentile(sorted: &[u128], p: f64) -> u128 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx]
}

/// Expected calibration error over 10 equal-width confidence bins:
/// `Σ (bin_size / N) · |accuracy(bin) − mean_confidence(bin)|`.
fn ece(pairs: &[(f32, bool)]) -> f64 {
    let n = pairs.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let mut bins = [(0usize, 0.0f64, 0usize); 10]; // (count, conf_sum, correct)
    for &(conf, ok) in pairs {
        let b = ((conf as f64 * 10.0) as usize).min(9);
        bins[b].0 += 1;
        bins[b].1 += conf as f64;
        bins[b].2 += ok as usize;
    }
    bins.iter()
        .filter(|b| b.0 > 0)
        .map(|&(cnt, csum, ok)| {
            let c = cnt as f64;
            (c / n) * ((ok as f64 / c) - (csum / c)).abs()
        })
        .sum()
}

fn summarize(name: &str, cases: &[Case], rows: &[Row], stats: &ClassifierStats) {
    let labelled: Vec<_> = cases.iter().zip(rows).filter(|(c, _)| c.request_type.is_some()).collect();
    let mut lat: Vec<u128> = rows.iter().map(|r| r.latency_us).collect();
    lat.sort_unstable();
    eprintln!("Backend: {name}");
    if labelled.is_empty() {
        eprintln!("  (no labels in EVAL_SET: accuracy/ECE not computed)");
    } else {
        let correct: Vec<bool> = labelled.iter().map(|(c, r)| c.request_type == Some(r.c.request_type)).collect();
        let acc = correct.iter().filter(|x| **x).count() as f64 / correct.len() as f64;
        let cx: Vec<(u8, u8)> = labelled
            .iter()
            .filter_map(|(c, r)| c.complexity.map(|g| (g, r.c.complexity)))
            .collect();
        let pairs: Vec<(f32, bool)> = labelled.iter().zip(&correct).map(|((_, r), ok)| (r.c.confidence, *ok)).collect();
        eprintln!("  Request-type accuracy: {:.1}% ({}/{})", acc * 100.0, correct.iter().filter(|x| **x).count(), correct.len());
        if !cx.is_empty() {
            let mae = cx.iter().map(|(g, p)| (*g as f64 - *p as f64).abs()).sum::<f64>() / cx.len() as f64;
            let exact = cx.iter().filter(|(g, p)| g == p).count() as f64 / cx.len() as f64;
            eprintln!("  Complexity: MAE {mae:.2}, exact {:.1}%", exact * 100.0);
        }
        eprintln!("  ECE (10 bins): {:.3}", ece(&pairs));
    }
    eprintln!("  Latency p50/p95: {}us / {}us", percentile(&lat, 0.50), percentile(&lat, 0.95));
    eprintln!("  Fallbacks: {}/{}", stats.fallbacks(), stats.calls());
}

#[tokio::main]
async fn main() {
    let path = env("EVAL_SET").expect("set EVAL_SET to the evaluation JSON file");
    let cases = load_cases(&path);
    let settings = settings_from_env();
    let backend = build_classifier(&settings);

    let (rows, stats) = run(backend.as_ref(), &cases).await;

    if let Some(out) = env("OUT") {
        let mut f = std::io::BufWriter::new(std::fs::File::create(&out).unwrap_or_else(|e| panic!("cannot create OUT {out}: {e}")));
        for (case, row) in cases.iter().zip(&rows) {
            let line = json!({
                "id": case.id,
                "request_type": row.c.request_type.as_str(),
                "complexity": row.c.complexity,
                "confidence": row.c.confidence,
                "latency_us": row.latency_us as u64,
                "fallback": row.fallback,
            });
            writeln!(f, "{line}").expect("write OUT");
        }
        f.flush().expect("flush OUT");
    }

    // Human-readable comparison on stderr (OUT carries only the configured backend).
    summarize(backend.name(), &cases, &rows, &stats);
    if backend.name() != "regex" {
        let regex = RegexRequestClassifier;
        let (r_rows, r_stats) = run(&regex, &cases).await;
        summarize("regex (baseline)", &cases, &r_rows, &r_stats);
    }
}
