//! Request-classifier evaluation harness.
//!
//! Usage:
//!
//! ```text
//! EVAL_SET=/tmp/classifier-eval.json OUT=/tmp/classifier-out.jsonl \
//! cargo run --release -p nasiko-llm-router --example classifier_eval
//! ```
//!
//! EVAL_SET may contain either:
//!
//! {
//!   "examples": [
//!     {
//!       "id": "1",
//!       "query": "Write a Python function to sort a list",
//!       "request_type": "code_generation",
//!       "complexity": 3
//!     }
//!   ]
//! }
//!
//! or a bare JSON array with the same objects.
//!
//! OUT is optional. When provided, one JSON object is written per line.

use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

use nasiko_llm_router::routing::{
    Classification,
    ClassifyInput,
    RequestClassifier,
    RequestType,
    RegexRequestClassifier,
};

use serde_json::{Value, json};

struct Case {
    id: String,
    query: String,
    context: Option<String>,
    expected_type: Option<RequestType>,
    expected_complexity: Option<u8>,
}

struct Row {
    classification: Classification,
    latency_us: u128,
}

fn env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

/// Convert the public wire label from the evaluation JSON into RequestType.
fn parse_request_type(value: Option<&str>) -> Option<RequestType> {
    match value? {
        "code_generation" => Some(RequestType::CodeGeneration),
        "code_understanding" => Some(RequestType::CodeUnderstanding),
        "technical_design" => Some(RequestType::TechnicalDesign),
        "analytical_reasoning" => Some(RequestType::AnalyticalReasoning),
        "writing" => Some(RequestType::Writing),
        "factual_lookup" => Some(RequestType::FactualLookup),
        "general" => Some(RequestType::General),
        _ => None,
    }
}

/// Load evaluation cases from EVAL_SET.
fn load_cases(path: &str) -> Vec<Case> {
    let raw = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("cannot read EVAL_SET {path}: {error}"));

    let value: Value = serde_json::from_str(&raw)
        .unwrap_or_else(|error| panic!("EVAL_SET is not valid JSON: {error}"));

    let items = value
        .get("examples")
        .and_then(Value::as_array)
        .or_else(|| value.as_array())
        .unwrap_or_else(|| {
            panic!("EVAL_SET must contain an `examples` array or be a JSON array")
        });

    items
        .iter()
        .enumerate()
        .map(|(index, item)| {
            let query = item
                .get("query")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();

            let id = item
                .get("id")
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| format!("case-{:03}", index + 1));

            let context = item
                .get("context")
                .and_then(Value::as_str)
                .map(str::to_string);

            let expected_type = item
                .get("request_type")
                .and_then(Value::as_str)
                .and_then(parse_request_type);

            let expected_complexity = item
                .get("complexity")
                .and_then(Value::as_u64)
                .map(|value| value as u8);

            Case {
                id,
                query,
                context,
                expected_type,
                expected_complexity,
            }
        })
        .collect()
}

/// Run one classifier backend over the complete evaluation set.
async fn run_classifier(
    classifier: &dyn RequestClassifier,
    cases: &[Case],
) -> Vec<Row> {
    let mut rows = Vec::with_capacity(cases.len());

    // Warm-up call.
    if let Some(first) = cases.first() {
        let _ = classifier
            .classify(ClassifyInput {
                query: &first.query,
                context: first.context.as_deref(),
            })
            .await;
    }

    for case in cases {
        let input = ClassifyInput {
            query: &case.query,
            context: case.context.as_deref(),
        };

        let start = Instant::now();

        let classification = classifier
            .classify(input)
            .await
            .unwrap_or_else(|error| {
                panic!(
                    "classifier failed for {}: {}",
                    case.id, error
                )
            });

        let latency_us = start.elapsed().as_micros();

        rows.push(Row {
            classification,
            latency_us,
        });
    }

    rows
}

/// Calculate percentile latency.
fn percentile(values: &[u128], percentile: f64) -> u128 {
    if values.is_empty() {
        return 0;
    }

    let mut sorted = values.to_vec();
    sorted.sort_unstable();

    let position =
        ((sorted.len() - 1) as f64 * percentile).round() as usize;

    sorted[position]
}

/// Expected Calibration Error.
///
/// Ten equal-width confidence bins are used.
fn ece(pairs: &[(f32, bool)]) -> f64 {
    if pairs.is_empty() {
        return 0.0;
    }

    let total = pairs.len() as f64;

    let mut bins = [(0usize, 0.0f64, 0usize); 10];

    for &(confidence, correct) in pairs {
        let confidence = confidence.clamp(0.0, 1.0) as f64;

        let bin = ((confidence * 10.0) as usize).min(9);

        bins[bin].0 += 1;
        bins[bin].1 += confidence;
        bins[bin].2 += usize::from(correct);
    }

    bins.iter()
        .filter(|(count, _, _)| *count > 0)
        .map(|(count, confidence_sum, correct)| {
            let count_f = *count as f64;

            let accuracy = *correct as f64 / count_f;
            let confidence = *confidence_sum / count_f;

            (count_f / total) * (accuracy - confidence).abs()
        })
        .sum()
}

/// Print evaluation metrics.
fn summarize(
    name: &str,
    cases: &[Case],
    rows: &[Row],
) {
    println!();
    println!("========================================");
    println!("Backend: {name}");
    println!("========================================");

    let latencies: Vec<u128> =
        rows.iter().map(|row| row.latency_us).collect();

    println!(
        "Latency p50/p95: {}us / {}us",
        percentile(&latencies, 0.50),
        percentile(&latencies, 0.95)
    );

    let mut type_correct = Vec::new();

    let mut complexity_pairs = Vec::new();

    let mut calibration_pairs = Vec::new();

    for (case, row) in cases.iter().zip(rows.iter()) {
        if let Some(expected) = case.expected_type {
            let correct =
                expected == row.classification.request_type;

            type_correct.push(correct);

            calibration_pairs.push((
                row.classification.confidence,
                correct,
            ));
        }

        if let Some(expected_complexity) =
            case.expected_complexity
        {
            complexity_pairs.push((
                expected_complexity,
                row.classification.complexity,
            ));
        }
    }

    if !type_correct.is_empty() {
        let correct_count =
            type_correct.iter().filter(|value| **value).count();

        let accuracy =
            correct_count as f64 / type_correct.len() as f64;

        println!(
            "Request-type accuracy: {:.1}% ({}/{})",
            accuracy * 100.0,
            correct_count,
            type_correct.len()
        );

        println!(
            "ECE (10 bins): {:.3}",
            ece(&calibration_pairs)
        );
    } else {
        println!(
            "Request-type accuracy: not available (no labels)"
        );
        println!("ECE: not available (no labels)");
    }

    if !complexity_pairs.is_empty() {
        let mae = complexity_pairs
            .iter()
            .map(|(expected, predicted)| {
                (*expected as f64 - *predicted as f64).abs()
            })
            .sum::<f64>()
            / complexity_pairs.len() as f64;

        let exact = complexity_pairs
            .iter()
            .filter(|(expected, predicted)| expected == predicted)
            .count() as f64
            / complexity_pairs.len() as f64;

        println!("Complexity MAE: {:.2}", mae);
        println!(
            "Complexity exact accuracy: {:.1}%",
            exact * 100.0
        );
    } else {
        println!(
            "Complexity metrics: not available (no labels)"
        );
    }
}

/// Write evaluation results as JSONL.
fn write_output(
    path: &str,
    cases: &[Case],
    rows: &[Row],
) {
    let file = std::fs::File::create(path)
        .unwrap_or_else(|error| {
            panic!("cannot create OUT {path}: {error}")
        });

    let mut writer = std::io::BufWriter::new(file);

    for (case, row) in cases.iter().zip(rows.iter()) {
        let output = json!({
            "id": case.id,
            "request_type":
                row.classification.request_type.as_str(),
            "complexity":
                row.classification.complexity,
            "confidence":
                row.classification.confidence,
            "latency_us":
                row.latency_us as u64,
        });

        writeln!(writer, "{output}")
            .expect("failed to write OUT");

    }

    writer.flush().expect("failed to flush OUT");
}

#[tokio::main]
async fn main() {
    let eval_path = env("EVAL_SET")
        .expect("set EVAL_SET to the evaluation JSON file");

    let cases = load_cases(&eval_path);

    if cases.is_empty() {
        panic!("EVAL_SET contains no evaluation cases");
    }

    println!(
        "Loaded {} evaluation cases",
        cases.len()
    );

    // Current default backend.
    //
    // This is intentionally the existing deterministic regex
    // implementation. Additional model-backed backends will
    // implement the same RequestClassifier trait.
    let regex_classifier =
        Arc::new(RegexRequestClassifier);

    let rows = run_classifier(
        regex_classifier.as_ref(),
        &cases,
    )
    .await;

    summarize(
        "regex",
        &cases,
        &rows,
    );

    if let Some(output_path) = env("OUT") {
        write_output(
            &output_path,
            &cases,
            &rows,
        );

        println!();
        println!("Results written to: {output_path}");
    }
}
