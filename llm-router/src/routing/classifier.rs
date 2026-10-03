//! Query classifier — maps an incoming query to a model [`Tier`] for the destination
//! provider.
//!
//! The classifier answers "how much model does this query need?" as a coarse tier; the
//! [tier registry](super::registry) then maps `(provider, tier)` to a concrete model.
//! Provider selection and request translation happen elsewhere (the resolver / inbound
//! spokes) — the classifier only chooses the *strength* of the model, never the provider.
//!
//! ## How the tier is chosen
//!
//! Two steps, both faithful ports of the litellm-rust **Adaptive Router** reference
//! (`classifier/{categories,signals}.rs`, `scoring.rs`) — see `THIRD_PARTY_LICENSES.md`
//! (crate root) for the upstream MIT attribution this requires:
//!
//! 1. **Request type** — a regex vote-count classifier buckets the query into one of a
//!    handful of [`RequestType`]s (code generation, factual lookup, …), defaulting to
//!    `General`.
//! 2. **Tier** — the three tiers are treated as bandit *arms*. [`pick_model_thompson`]
//!    Thompson-samples a quality estimate per tier from a Beta posterior — seeded by a
//!    cold-start prior (stronger/on-strength tiers start higher) and updated by learned
//!    [`Cell`]s — then blends it with a normalized cost term and takes the argmax.
//!
//! The learned [`Cell`]s come from real feedback: the router credits a tier's quality from
//! the user's next-turn reaction ([`signal`]), persisted per provider by the
//! [cell store](super::cells). With no learning yet the priors + cost blend decide; as
//! feedback accumulates the posterior tightens and selection converges. Thompson's
//! stochasticity is the exploration that makes that learning possible, so production feeds
//! it an entropy RNG; tests inject a seeded one.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use rand::Rng;
use rand_distr::{Beta, Distribution};

use super::patterns::{CATEGORY_PATTERNS, NEGATIVE_SIGNALS, POSITIVE_SIGNALS};

/// Coarse model strength tier. Tier 1 = most capable (complex queries), Tier 3 = smallest
/// (very simple queries), Tier 2 = in between.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tier {
    /// Complex queries — the strongest model in the provider's registry.
    Tier1,
    /// Mid-complexity queries.
    Tier2,
    /// Very simple queries — the smallest/cheapest model.
    Tier3,
}

/// The coarse kind of work a query represents. Learning is keyed on this, so the router can
/// discover (e.g.) that the cheap tier is good enough for `FactualLookup` but not
/// `CodeGeneration`. Order is irrelevant; `General` is the catch-all default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RequestType {
    CodeGeneration,
    CodeUnderstanding,
    TechnicalDesign,
    AnalyticalReasoning,
    Writing,
    FactualLookup,
    General,
}

impl RequestType {
    /// Stable string form used as the persisted cell key (`router_quality_cells.request_type`).
    pub fn as_str(self) -> &'static str {
        match self {
            RequestType::CodeGeneration => "code_generation",
            RequestType::CodeUnderstanding => "code_understanding",
            RequestType::TechnicalDesign => "technical_design",
            RequestType::AnalyticalReasoning => "analytical_reasoning",
            RequestType::Writing => "writing",
            RequestType::FactualLookup => "factual_lookup",
            RequestType::General => "general",
        }
    }

    /// Inverse of [`RequestType::as_str`]; `None` for unknown values (a row written by an
    /// older/newer schema is skipped rather than trusted). Named `from_wire` rather than
    /// `from_str` to avoid shadowing the `std::str::FromStr` trait method.
    pub fn from_wire(s: &str) -> Option<RequestType> {
        Some(match s {
            "code_generation" => RequestType::CodeGeneration,
            "code_understanding" => RequestType::CodeUnderstanding,
            "technical_design" => RequestType::TechnicalDesign,
            "analytical_reasoning" => RequestType::AnalyticalReasoning,
            "writing" => RequestType::Writing,
            "factual_lookup" => RequestType::FactualLookup,
            "general" => RequestType::General,
            _ => return None,
        })
    }
}

/// One learned quality estimate: a running mean of observed reward for a `(tier,
/// request_type)` under some provider, plus how many observations back it. This is the unit
/// the [cell store](super::cells) persists; it is a direct port of the reference
/// `scoring.rs::Cell`.
#[derive(Debug, Clone, Copy)]
pub struct Cell {
    pub quality_mean: f64,
    pub samples: i64,
}

/// Learned cells for a single provider, keyed by `(tier, request_type)`. The provider is
/// the scope of the whole map, so it is not part of the key.
pub type CellMap = HashMap<(Tier, RequestType), Cell>;

/// Sample cap for the running mean — past this the mean stops chasing new observations, so
/// a cell's estimate is stable once well-sampled. Port of the reference `MAX_SAMPLES`.
pub const MAX_SAMPLES: i64 = 200;

/// Strength of the cold-start prior, in Beta pseudo-observations. Port of the reference
/// `PRIOR_PSEUDO_COUNT`.
const PRIOR_PSEUDO_COUNT: f64 = 4.0;

/// Quality/cost blend weights (`w_quality`, `w_cost`). The reference default: quality leads,
/// cost trims. Tunable — learning corrects any cold-start bias over time.
pub const DEFAULT_W_QUALITY: f64 = 0.7;
pub const DEFAULT_W_COST: f64 = 0.3;

/// A tier as a bandit arm: its nominal quality tier (for the cold-start prior), a relative
/// cost, and the request types it is expected to be good at (a prior bonus). Costs are a
/// generic gradient — only their *relative* ordering matters after normalization, so this is
/// provider-independent for now.
struct TierArm {
    tier: Tier,
    quality_tier: i32,
    cost: f64,
    strengths: &'static [RequestType],
}

/// The three tiers as bandit arms. Tier1 = strongest+priciest, Tier3 = weakest+cheapest.
const TIER_ARMS: [TierArm; 3] = [
    TierArm {
        tier: Tier::Tier1,
        quality_tier: 3,
        cost: 15.0,
        strengths: &[
            RequestType::CodeGeneration,
            RequestType::AnalyticalReasoning,
            RequestType::TechnicalDesign,
        ],
    },
    TierArm {
        tier: Tier::Tier2,
        quality_tier: 2,
        cost: 3.0,
        strengths: &[RequestType::CodeUnderstanding, RequestType::Writing],
    },
    TierArm {
        tier: Tier::Tier3,
        quality_tier: 1,
        cost: 0.8,
        strengths: &[RequestType::FactualLookup, RequestType::General],
    },
];

// --------------------------------------------------------------------------
// 1. Request-type classifier — port of classifier/categories.rs
//    (order matters: on a tie the earlier category wins; patterns in `super::patterns`)
// --------------------------------------------------------------------------

/// Bucket a query into a [`RequestType`] by vote count — the category matching the most
/// patterns wins, ties broken by declaration order, defaulting to `General`. Port of
/// `categories.rs::classify`.
pub fn classify_request_type(text: &str) -> RequestType {
    let mut best = RequestType::General;
    let mut best_score = 0usize;
    for (rt, pats) in CATEGORY_PATTERNS.iter() {
        let score = pats.iter().filter(|p| p.is_match(text)).count();
        if score > best_score {
            best_score = score;
            best = *rt;
        }
    }
    best
}

// --------------------------------------------------------------------------
// 1b. Pluggable request classifier — `RequestClassifier` trait + backends
//
// The regex vote-count classifier above stays the out-of-the-box default. A hosted or
// local model can be plugged in behind the same trait through binary configuration; any
// failure (transport, timeout, malformed answer, low confidence) falls back to regex so
// routing never breaks and never depends on the network unless the operator opts in.
// --------------------------------------------------------------------------

/// What the classifier sees. `context` is optional extra text (e.g. surrounding files or
/// the previous turn); the router currently passes `None`.
#[derive(Debug, Clone, Copy)]
pub struct ClassifyInput<'a> {
    pub query: &'a str,
    pub context: Option<&'a str>,
}

/// A classifier verdict.
///
/// * `request_type` — one of the seven public [`RequestType`] labels.
/// * `complexity` — 1 (trivial, one-liner) ..= 5 (multi-step, expert-level work).
/// * `confidence` — 0.0 ..= 1.0, the backend's own probability that `request_type` is
///   right. The regex baseline reports a fixed `0.5` ("a keyword vote, not a probability").
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Classification {
    pub request_type: RequestType,
    pub complexity: u8,
    pub confidence: f32,
}

/// Why a classifier backend could not produce a usable [`Classification`]. Every variant
/// is handled the same way by the router: regex fallback + fallback counter.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ClassifyError {
    /// The backend did not answer within the configured timeout.
    #[error("classifier timeout")]
    Timeout,
    /// Connection, HTTP status, or model-inference failure.
    #[error("classifier backend failure: {0}")]
    Backend(String),
    /// The backend answered, but with an unknown label or out-of-range number.
    #[error("classifier returned an invalid response: {0}")]
    InvalidResponse(String),
    /// The backend answered validly but below the configured minimum confidence. Treated
    /// as "don't trust it": the router uses the regex result instead.
    #[error("classifier confidence {0} below threshold")]
    LowConfidence(f32),
}

/// Maps a query (plus optional context) to a [`Classification`].
///
/// Implementations must be deterministic for identical input and backend state. The router
/// holds an `Arc<dyn RequestClassifier>` and only calls it at safe routing boundaries
/// (`cold_start` / `switch`); tool-loop `continue` steps read the sticky cached decision.
#[async_trait::async_trait]
pub trait RequestClassifier: Send + Sync {
    /// Short stable backend name (`"regex"`, `"http"`), used in logs and eval output.
    fn name(&self) -> &str;

    async fn classify(&self, input: &ClassifyInput<'_>) -> Result<Classification, ClassifyError>;
}

/// Confidence the regex baseline reports for every query: a keyword vote is weak evidence,
/// so it sits at the midpoint rather than pretending to be calibrated.
pub const REGEX_CONFIDENCE: f32 = 0.5;

/// Fixed complexity (1–5) the regex baseline assigns per request type.
///
/// These are **constant estimates, not learned values**: the regex only sees keywords, so
/// it cannot tell "fix a typo" from "build a compiler" — both are `code_generation` → 4.
/// That blind spot is exactly what a model-based backend is meant to improve on.
pub fn regex_complexity(request_type: RequestType) -> u8 {
    match request_type {
        RequestType::CodeGeneration => 4,
        RequestType::TechnicalDesign => 4,
        RequestType::AnalyticalReasoning => 4,
        RequestType::CodeUnderstanding => 3,
        RequestType::Writing => 3,
        RequestType::FactualLookup => 2,
        RequestType::General => 1,
    }
}

/// The default backend: wraps [`classify_request_type`] and reports the fixed
/// [`regex_complexity`] / [`REGEX_CONFIDENCE`]. Never fails, never touches the network,
/// ignores `context`. Behaviour is identical to the router before the trait existed.
#[derive(Debug, Clone, Copy, Default)]
pub struct RegexRequestClassifier;

#[async_trait::async_trait]
impl RequestClassifier for RegexRequestClassifier {
    fn name(&self) -> &str {
        "regex"
    }

    async fn classify(&self, input: &ClassifyInput<'_>) -> Result<Classification, ClassifyError> {
        let request_type = classify_request_type(input.query);
        Ok(Classification {
            request_type,
            complexity: regex_complexity(request_type),
            confidence: REGEX_CONFIDENCE,
        })
    }
}

/// Model-agnostic HTTP backend: POSTs `{"model", "query", "context"}` as JSON to
/// `endpoint` and expects `{"request_type", "complexity", "confidence"}` back. Works
/// against a self-hosted model server or a thin adapter in front of a hosted API.
///
/// Determinism: the router sends no sampling parameters; the endpoint is expected to run
/// greedily (temperature 0 / fixed seed) so the same input yields the same label.
pub struct HttpRequestClassifier {
    client: reqwest::Client,
    endpoint: String,
    model: String,
    api_key: Option<String>,
    timeout: Duration,
    min_confidence: f32,
}

impl HttpRequestClassifier {
    pub fn new(
        endpoint: String,
        model: String,
        api_key: Option<String>,
        timeout: Duration,
        min_confidence: f32,
    ) -> Self {
        Self {
            client: reqwest::Client::new(),
            endpoint,
            model,
            api_key,
            timeout,
            min_confidence,
        }
    }

    /// Validate a decoded response body into a [`Classification`].
    fn validate(body: &serde_json::Value) -> Result<Classification, ClassifyError> {
        let invalid = |m: &str| ClassifyError::InvalidResponse(m.to_string());
        let label = body
            .get("request_type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| invalid("missing request_type"))?;
        let request_type = RequestType::from_wire(label)
            .ok_or_else(|| ClassifyError::InvalidResponse(format!("unknown request_type {label:?}")))?;
        let complexity = body
            .get("complexity")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| invalid("complexity must be an integer"))?;
        if !(1..=5).contains(&complexity) {
            return Err(ClassifyError::InvalidResponse(format!(
                "complexity {complexity} outside 1..=5"
            )));
        }
        let confidence = body
            .get("confidence")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| invalid("confidence must be a number"))?;
        if !(0.0..=1.0).contains(&confidence) {
            return Err(ClassifyError::InvalidResponse(format!(
                "confidence {confidence} outside 0.0..=1.0"
            )));
        }
        Ok(Classification {
            request_type,
            complexity: complexity as u8,
            confidence: confidence as f32,
        })
    }

    async fn request(&self, input: &ClassifyInput<'_>) -> Result<serde_json::Value, ClassifyError> {
        let mut req = self.client.post(&self.endpoint).json(&serde_json::json!({
            "model": self.model,
            "query": input.query,
            "context": input.context,
        }));
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| ClassifyError::Backend(e.to_string()))?;
        if !resp.status().is_success() {
            return Err(ClassifyError::Backend(format!("HTTP {}", resp.status())));
        }
        resp.json()
            .await
            .map_err(|e| ClassifyError::InvalidResponse(e.to_string()))
    }
}

#[async_trait::async_trait]
impl RequestClassifier for HttpRequestClassifier {
    fn name(&self) -> &str {
        "http"
    }

    async fn classify(&self, input: &ClassifyInput<'_>) -> Result<Classification, ClassifyError> {
        // The timeout covers connect + response headers + body, so a stalled backend can
        // never hold up a routing decision longer than the configured budget.
        let body = tokio::time::timeout(self.timeout, self.request(input))
            .await
            .map_err(|_| ClassifyError::Timeout)??;
        let c = Self::validate(&body)?;
        // Safe low-confidence behaviour: an unsure model is not trusted — the router uses
        // the regex verdict (and counts a fallback) rather than acting on a weak guess.
        if c.confidence < self.min_confidence {
            return Err(ClassifyError::LowConfidence(c.confidence));
        }
        Ok(c)
    }
}

/// Backend selection, supplied by the host binary (the library never reads the env).
#[derive(Debug, Clone)]
pub struct ClassifierSettings {
    /// `"regex"` (default), or `"http"` / `"hosted"`. Anything else ⇒ regex.
    pub backend: String,
    pub model: String,
    pub endpoint: String,
    pub api_key: Option<String>,
    pub timeout_ms: u64,
    /// HTTP verdicts below this confidence fall back to regex. `0.0` disables the check.
    pub min_confidence: f32,
}

impl Default for ClassifierSettings {
    fn default() -> Self {
        Self {
            backend: "regex".to_string(),
            model: String::new(),
            endpoint: String::new(),
            api_key: None,
            timeout_ms: 5000,
            min_confidence: 0.5,
        }
    }
}

/// Build the configured classifier. Unknown backend names, or an `http` backend without an
/// endpoint, safely resolve to [`RegexRequestClassifier`] — the default behaviour.
pub fn build_classifier(settings: &ClassifierSettings) -> Arc<dyn RequestClassifier> {
    match settings.backend.trim().to_ascii_lowercase().as_str() {
        "http" | "hosted" if !settings.endpoint.trim().is_empty() => {
            Arc::new(HttpRequestClassifier::new(
                settings.endpoint.trim().to_string(),
                settings.model.clone(),
                settings.api_key.clone(),
                Duration::from_millis(settings.timeout_ms),
                settings.min_confidence,
            ))
        }
        "http" | "hosted" => {
            tracing::warn!(
                target: "nasiko::llm_router::classifier",
                "CLASSIFIER_BACKEND=http but no endpoint configured; using regex"
            );
            Arc::new(RegexRequestClassifier)
        }
        _ => Arc::new(RegexRequestClassifier),
    }
}

/// Counters for classifier health. `fallbacks` is the number of times the configured
/// backend failed (error, timeout, invalid answer, low confidence) and regex was used.
#[derive(Debug, Default)]
pub struct ClassifierStats {
    calls: AtomicU64,
    fallbacks: AtomicU64,
}

impl ClassifierStats {
    pub fn calls(&self) -> u64 {
        self.calls.load(Ordering::Relaxed)
    }
    pub fn fallbacks(&self) -> u64 {
        self.fallbacks.load(Ordering::Relaxed)
    }
}

/// Run `classifier`; on any error use the regex classifier and count a fallback. Always
/// returns a usable [`Classification`] — a classifier failure can never fail routing. The
/// router and `classifier_eval` both go through this function, so the eval exercises the
/// production path. The bool is `true` when the fallback was used.
pub async fn classify_with_fallback(
    classifier: &dyn RequestClassifier,
    input: &ClassifyInput<'_>,
    stats: Option<&ClassifierStats>,
) -> (Classification, bool) {
    if let Some(s) = stats {
        s.calls.fetch_add(1, Ordering::Relaxed);
    }
    match classifier.classify(input).await {
        Ok(c) => (c, false),
        Err(e) => {
            if let Some(s) = stats {
                s.fallbacks.fetch_add(1, Ordering::Relaxed);
            }
            tracing::warn!(
                target: "nasiko::llm_router::classifier",
                backend = classifier.name(),
                error = %e,
                "classifier backend failed; falling back to regex"
            );
            let request_type = classify_request_type(input.query);
            (
                Classification {
                    request_type,
                    complexity: regex_complexity(request_type),
                    confidence: REGEX_CONFIDENCE,
                },
                true,
            )
        }
    }
}

// --------------------------------------------------------------------------
// 2. Feedback signal — port of classifier/signals.rs (patterns in `super::patterns`)
// --------------------------------------------------------------------------

/// Extract a reward from a follow-up message: `0.0` on a complaint, `1.0` on approval,
/// `None` when the text carries no clear verdict. Negative is checked first so a mixed
/// message ("thanks but that's wrong") counts as negative. The regexes are deliberately
/// conservative, so an ordinary new question yields `None` and earns no false credit. Port
/// of `signals.rs::signal`.
pub fn signal(text: &str) -> Option<f64> {
    if NEGATIVE_SIGNALS.iter().any(|p| p.is_match(text)) {
        return Some(0.0);
    }
    if POSITIVE_SIGNALS.iter().any(|p| p.is_match(text)) {
        return Some(1.0);
    }
    None
}

// --------------------------------------------------------------------------
// 3. Scoring — port of scoring.rs
// --------------------------------------------------------------------------

/// Initial quality estimate for a tier before any feedback: a base that grows with the
/// quality tier plus a bonus when the request type is one of the tier's strengths, clamped
/// away from the extremes. Port of `scoring.rs::cold_start_prior`.
fn cold_start_prior(quality_tier: i32, strengths: &[RequestType], rt: RequestType) -> f64 {
    let tier_base = 0.5 + 0.15 * (quality_tier - 1).max(0) as f64;
    let bonus = if strengths.contains(&rt) { 0.15 } else { 0.0 };
    (tier_base + bonus).clamp(0.05, 0.95)
}

/// Fold one observation into a cell's running mean, capping the effective sample count so a
/// well-sampled estimate stays stable. Port of `scoring.rs::update_cell`.
pub fn update_cell(cell: Cell, observation: f64) -> Cell {
    let n_eff = cell.samples.min(MAX_SAMPLES);
    let new_mean = cell.quality_mean + (observation - cell.quality_mean) / (n_eff as f64 + 1.0);
    Cell {
        quality_mean: new_mean,
        samples: (cell.samples + 1).min(MAX_SAMPLES),
    }
}

/// The cold-start prior for a given tier and request type, used to seed both the Beta
/// posterior in [`pick_model_thompson`] and a fresh cell in the store.
pub fn tier_prior(tier: Tier, rt: RequestType) -> f64 {
    let arm = TIER_ARMS
        .iter()
        .find(|a| a.tier == tier)
        .expect("every Tier has a TierArm");
    cold_start_prior(arm.quality_tier, arm.strengths, rt)
}

/// Sample a `Beta(alpha, beta)` variate, guarding degenerate parameters. Falls back to the
/// distribution mean if the parameters can't form a valid Beta.
fn beta_sample<R: Rng + ?Sized>(alpha: f64, beta: f64, rng: &mut R) -> f64 {
    let a = alpha.max(1e-6);
    let b = beta.max(1e-6);
    match Beta::new(a, b) {
        Ok(dist) => dist.sample(rng),
        Err(_) => a / (a + b),
    }
}

/// Thompson-sample a [`Tier`] for `request_type`: draw a quality per tier from its Beta
/// posterior (cold-start prior as pseudo-observations + learned [`Cell`] as real ones),
/// blend with a normalized cost term, and take the argmax (ties → earlier/stronger tier).
/// Port of the reference `pick_model_thompson`, with the three tiers as the candidate arms.
pub fn pick_model_thompson<R: Rng + ?Sized>(
    cells: &CellMap,
    request_type: RequestType,
    w_quality: f64,
    w_cost: f64,
    rng: &mut R,
) -> Tier {
    let lo = TIER_ARMS
        .iter()
        .map(|a| a.cost)
        .fold(f64::INFINITY, f64::min);
    let hi = TIER_ARMS
        .iter()
        .map(|a| a.cost)
        .fold(f64::NEG_INFINITY, f64::max);
    let span = hi - lo;

    let mut best = TIER_ARMS[0].tier;
    let mut best_score = f64::NEG_INFINITY;
    for arm in TIER_ARMS.iter() {
        let prior = cold_start_prior(arm.quality_tier, arm.strengths, request_type);
        let (successes, failures) = match cells.get(&(arm.tier, request_type)) {
            Some(cell) => {
                let s = cell.quality_mean * cell.samples as f64;
                (s, cell.samples as f64 - s)
            }
            None => (0.0, 0.0),
        };
        let alpha = prior * PRIOR_PSEUDO_COUNT + successes;
        let beta = (1.0 - prior) * PRIOR_PSEUDO_COUNT + failures;
        let q = beta_sample(alpha, beta, rng);
        let norm_cost = if span > 0.0 {
            (arm.cost - lo) / span
        } else {
            0.0
        };
        let score = w_quality * q + w_cost * (1.0 - norm_cost);
        if score > best_score {
            best_score = score;
            best = arm.tier;
        }
    }
    best
}

// --------------------------------------------------------------------------
// 4. Public entry point
// --------------------------------------------------------------------------

/// Classify a `query` into a model [`Tier`] (and the [`RequestType`] it was bucketed as) for
/// the destination `provider`, using the regex request-type classifier.
///
/// `provider` is the **destination** provider the request will be routed to (already
/// resolved), not the agent's client SDK — the tier is later looked up in *that* provider's
/// registry, and the returned `RequestType` is what feedback is later credited to.
///
/// `cells` are the provider's learned quality estimates (empty ⇒ pure cold-start priors);
/// `rng` drives Thompson exploration (entropy in production, seeded in tests).
pub fn classify<R: Rng + ?Sized>(
    query: &str,
    provider: &str,
    cells: &CellMap,
    rng: &mut R,
) -> (Tier, RequestType) {
    let request_type = classify_request_type(query);
    let tier = select_tier(query, provider, request_type, cells, rng);
    (tier, request_type)
}

/// Pick the [`Tier`] for an already-classified `request_type` (Thompson sampling over the
/// provider's learned cells). Split out of [`classify`] so any [`RequestClassifier`] backend
/// feeds the same, unchanged tier selection: the classifier decides *what kind of request*
/// this is; provider/tier mapping and cost trade-offs stay exactly as they were.
pub fn select_tier<R: Rng + ?Sized>(
    query: &str,
    provider: &str,
    request_type: RequestType,
    cells: &CellMap,
    rng: &mut R,
) -> Tier {
    let tier = pick_model_thompson(cells, request_type, DEFAULT_W_QUALITY, DEFAULT_W_COST, rng);
    let preview: String = query.chars().take(120).collect();
    tracing::info!(
        target: "nasiko::llm_router::classifier",
        provider = %provider,
        query_chars = query.chars().count(),
        query_preview = %preview,
        request_type = %request_type.as_str(),
        learned_cells = cells.len(),
        classified_tier = ?tier,
        "classifier: classified query into request type and Thompson-sampled a model tier"
    );
    tier
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    // --- request-type classifier (ports of the reference self-test) ---

    #[test]
    fn request_type_matches_reference_examples() {
        use RequestType::*;
        assert_eq!(
            classify_request_type("build me a python script that parses CSV"),
            CodeGeneration
        );
        assert_eq!(
            classify_request_type("write me a Python sort function"),
            CodeGeneration
        );
        assert_eq!(
            classify_request_type("explain what this function does"),
            CodeUnderstanding
        );
        assert_eq!(
            classify_request_type("how should I design this API?"),
            TechnicalDesign
        );
        assert_eq!(
            classify_request_type("calculate the probability that it rains tomorrow"),
            AnalyticalReasoning
        );
        assert_eq!(
            classify_request_type("draft an email to my team about the outage"),
            Writing
        );
        assert_eq!(
            classify_request_type("what is the capital of France?"),
            FactualLookup
        );
        assert_eq!(classify_request_type("hello there"), General);
    }

    #[test]
    fn request_type_round_trips_through_string() {
        for rt in [
            RequestType::CodeGeneration,
            RequestType::CodeUnderstanding,
            RequestType::TechnicalDesign,
            RequestType::AnalyticalReasoning,
            RequestType::Writing,
            RequestType::FactualLookup,
            RequestType::General,
        ] {
            assert_eq!(RequestType::from_wire(rt.as_str()), Some(rt));
        }
        assert_eq!(RequestType::from_wire("nonsense"), None);
    }

    // --- feedback signal ---

    #[test]
    fn signal_matches_reference() {
        assert_eq!(signal("perfect, that worked. thanks!"), Some(1.0));
        assert_eq!(signal("that's wrong, try again"), Some(0.0));
        assert_eq!(signal("now add error handling for missing files"), None);
        // negative wins a mixed message
        assert_eq!(signal("thanks but that's wrong"), Some(0.0));
    }

    // --- scoring primitives ---

    #[test]
    fn cold_start_prior_matches_reference() {
        assert_eq!(
            cold_start_prior(
                3,
                &[RequestType::AnalyticalReasoning],
                RequestType::AnalyticalReasoning
            ),
            0.95
        );
        assert_eq!(
            cold_start_prior(1, &[], RequestType::AnalyticalReasoning),
            0.5
        );
    }

    #[test]
    fn update_cell_matches_reference() {
        let c = update_cell(
            Cell {
                quality_mean: 0.5,
                samples: 0,
            },
            1.0,
        );
        assert_eq!(c.quality_mean, 1.0);
        assert_eq!(c.samples, 1);
        let c = update_cell(c, 0.0);
        assert!((c.quality_mean - 0.5).abs() < 1e-9);
        assert_eq!(c.samples, 2);
        let c = update_cell(
            Cell {
                quality_mean: 0.9,
                samples: MAX_SAMPLES,
            },
            0.9,
        );
        assert_eq!(c.samples, MAX_SAMPLES);
    }

    #[test]
    fn beta_sample_stays_in_unit_interval() {
        let mut rng = StdRng::seed_from_u64(1);
        for _ in 0..1000 {
            let x = beta_sample(2.0, 5.0, &mut rng);
            assert!((0.0..=1.0).contains(&x), "sample out of range: {x}");
        }
        // degenerate params fall back to the mean, not NaN
        assert!(beta_sample(0.0, 0.0, &mut rng).is_finite());
    }

    // --- Thompson tier selection ---

    #[test]
    fn thompson_converges_to_the_learned_best_tier() {
        // All three tiers are well-sampled for code generation: Tier1 excellent, the others
        // poor. Once every arm's posterior is tight (no wide unexplored arm left to gamble
        // on), all-quality Thompson picks the learned best on every draw.
        let mut cells = CellMap::new();
        cells.insert(
            (Tier::Tier1, RequestType::CodeGeneration),
            Cell {
                quality_mean: 0.99,
                samples: MAX_SAMPLES,
            },
        );
        for tier in [Tier::Tier2, Tier::Tier3] {
            cells.insert(
                (tier, RequestType::CodeGeneration),
                Cell {
                    quality_mean: 0.05,
                    samples: MAX_SAMPLES,
                },
            );
        }
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..200 {
            let tier = pick_model_thompson(&cells, RequestType::CodeGeneration, 1.0, 0.0, &mut rng);
            assert_eq!(tier, Tier::Tier1);
        }
    }

    #[test]
    fn thompson_explores_a_wide_unlearned_arm() {
        // The flip side of convergence: with the best arm only *mildly* learned and a rival
        // arm still unexplored (wide posterior), exploration must sometimes pick the rival —
        // this is what generates the feedback that eventually tightens it.
        let mut cells = CellMap::new();
        cells.insert(
            (Tier::Tier1, RequestType::CodeGeneration),
            Cell {
                quality_mean: 0.7,
                samples: 8,
            },
        );
        let mut rng = StdRng::seed_from_u64(1);
        let mut distinct = std::collections::HashSet::new();
        for _ in 0..200 {
            distinct.insert(pick_model_thompson(
                &cells,
                RequestType::CodeGeneration,
                1.0,
                0.0,
                &mut rng,
            ));
        }
        assert!(
            distinct.len() > 1,
            "expected exploration across arms, got {distinct:?}"
        );
    }

    #[test]
    fn thompson_all_cost_prefers_the_cheapest_tier() {
        // No learning; pure cost weight ⇒ the cheapest tier (Tier3) always wins.
        let cells = CellMap::new();
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..200 {
            let tier = pick_model_thompson(&cells, RequestType::General, 0.0, 1.0, &mut rng);
            assert_eq!(tier, Tier::Tier3);
        }
    }

    #[test]
    fn classify_returns_valid_tier_and_request_type() {
        let cells = CellMap::new();
        let mut rng = StdRng::seed_from_u64(3);
        let (tier, rt) = classify(
            "write a python function that sorts a list",
            "anthropic",
            &cells,
            &mut rng,
        );
        assert_eq!(rt, RequestType::CodeGeneration);
        assert!(matches!(tier, Tier::Tier1 | Tier::Tier2 | Tier::Tier3));
    }

    // --- RequestClassifier trait: regex backend ---

    fn input(q: &str) -> ClassifyInput<'_> {
        ClassifyInput {
            query: q,
            context: None,
        }
    }

    #[tokio::test]
    async fn regex_backend_wraps_classify_request_type() {
        let c = RegexRequestClassifier;
        assert_eq!(c.name(), "regex");
        for q in [
            "write me a Python sort function",
            "what is the capital of France?",
            "hello",
        ] {
            let got = c.classify(&input(q)).await.unwrap();
            assert_eq!(got.request_type, classify_request_type(q), "{q}");
        }
    }

    #[test]
    fn regex_complexity_is_the_documented_fixed_mapping() {
        use RequestType::*;
        for (rt, want) in [
            (CodeGeneration, 4),
            (TechnicalDesign, 4),
            (AnalyticalReasoning, 4),
            (CodeUnderstanding, 3),
            (Writing, 3),
            (FactualLookup, 2),
            (General, 1),
        ] {
            assert_eq!(regex_complexity(rt), want, "{rt:?}");
        }
    }

    #[tokio::test]
    async fn regex_backend_is_deterministic_with_fixed_confidence() {
        let c = RegexRequestClassifier;
        let first = c.classify(&input("design a scalable payment system")).await.unwrap();
        assert_eq!(first.confidence, REGEX_CONFIDENCE);
        for _ in 0..50 {
            let again = c.classify(&input("design a scalable payment system")).await.unwrap();
            assert_eq!(again, first);
        }
    }

    // --- backend selection ---

    #[test]
    fn factory_selects_regex_by_default_and_for_unknown_backends() {
        assert_eq!(build_classifier(&ClassifierSettings::default()).name(), "regex");
        for backend in ["regex", "", "bogus", "local"] {
            let s = ClassifierSettings {
                backend: backend.into(),
                endpoint: "http://127.0.0.1:1/x".into(),
                ..Default::default()
            };
            assert_eq!(build_classifier(&s).name(), "regex", "backend={backend:?}");
        }
    }

    #[test]
    fn factory_selects_http_only_with_an_endpoint() {
        let mut s = ClassifierSettings {
            backend: "http".into(),
            endpoint: "http://127.0.0.1:1/x".into(),
            ..Default::default()
        };
        assert_eq!(build_classifier(&s).name(), "http");
        s.backend = "hosted".into();
        assert_eq!(build_classifier(&s).name(), "http");
        s.endpoint = String::new();
        assert_eq!(build_classifier(&s).name(), "regex", "no endpoint ⇒ regex");
    }

    // --- HTTP backend ---

    fn http_to(url: String, timeout_ms: u64, min_conf: f32) -> HttpRequestClassifier {
        HttpRequestClassifier::new(url, "m".into(), None, Duration::from_millis(timeout_ms), min_conf)
    }

    #[tokio::test]
    async fn http_backend_parses_a_valid_response_and_sends_the_documented_body() {
        let mut server = mockito::Server::new_async().await;
        let m = server
            .mock("POST", "/c")
            .match_body(mockito::Matcher::PartialJson(serde_json::json!({
                "model": "m", "query": "fix this bug", "context": "main.rs"
            })))
            .with_body(r#"{"request_type":"code_generation","complexity":4,"confidence":0.91}"#)
            .create_async()
            .await;
        let c = http_to(format!("{}/c", server.url()), 2000, 0.5);
        let got = c
            .classify(&ClassifyInput { query: "fix this bug", context: Some("main.rs") })
            .await
            .unwrap();
        assert_eq!(got.request_type, RequestType::CodeGeneration);
        assert_eq!(got.complexity, 4);
        assert!((got.confidence - 0.91).abs() < 1e-6);
        m.assert_async().await;
    }

    #[tokio::test]
    async fn http_backend_rejects_invalid_responses() {
        let bad = [
            r#"{"request_type":"poetry","complexity":3,"confidence":0.9}"#, // unknown label
            r#"{"request_type":"writing","complexity":0,"confidence":0.9}"#, // complexity < 1
            r#"{"request_type":"writing","complexity":6,"confidence":0.9}"#, // complexity > 5
            r#"{"request_type":"writing","complexity":2.5,"confidence":0.9}"#, // non-integer
            r#"{"request_type":"writing","complexity":3,"confidence":1.2}"#, // confidence > 1
            r#"{"request_type":"writing","complexity":3,"confidence":-0.1}"#, // confidence < 0
            r#"{"complexity":3,"confidence":0.9}"#,                       // missing label
            r#"not json"#,
        ];
        for body in bad {
            let mut server = mockito::Server::new_async().await;
            let _m = server.mock("POST", "/c").with_body(body).create_async().await;
            let c = http_to(format!("{}/c", server.url()), 2000, 0.0);
            let err = c.classify(&input("q")).await.unwrap_err();
            assert!(matches!(err, ClassifyError::InvalidResponse(_)), "{body} -> {err:?}");
        }
    }

    #[tokio::test]
    async fn http_backend_errors_on_non_2xx() {
        let mut server = mockito::Server::new_async().await;
        let _m = server.mock("POST", "/c").with_status(500).create_async().await;
        let c = http_to(format!("{}/c", server.url()), 2000, 0.0);
        assert!(matches!(c.classify(&input("q")).await, Err(ClassifyError::Backend(_))));
    }

    #[tokio::test]
    async fn http_backend_low_confidence_is_an_error() {
        let mut server = mockito::Server::new_async().await;
        let _m = server
            .mock("POST", "/c")
            .with_body(r#"{"request_type":"writing","complexity":3,"confidence":0.3}"#)
            .create_async()
            .await;
        let c = http_to(format!("{}/c", server.url()), 2000, 0.5);
        assert_eq!(
            c.classify(&input("q")).await,
            Err(ClassifyError::LowConfidence(0.3))
        );
    }

    #[tokio::test]
    async fn http_backend_times_out_on_a_stalled_server() {
        // Accepts the connection and never answers.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let mut held = vec![];
            while let Ok((sock, _)) = listener.accept().await {
                held.push(sock);
            }
        });
        let c = http_to(format!("http://{addr}/c"), 150, 0.0);
        let started = std::time::Instant::now();
        assert_eq!(c.classify(&input("q")).await, Err(ClassifyError::Timeout));
        assert!(started.elapsed() < Duration::from_secs(3));
    }

    // --- fallback ---

    struct FailingClassifier;
    #[async_trait::async_trait]
    impl RequestClassifier for FailingClassifier {
        fn name(&self) -> &str {
            "failing"
        }
        async fn classify(&self, _: &ClassifyInput<'_>) -> Result<Classification, ClassifyError> {
            Err(ClassifyError::Backend("backend failure".into()))
        }
    }

    #[tokio::test]
    async fn fallback_uses_regex_and_counts() {
        let stats = ClassifierStats::default();
        let q = "write me a Python sort function";
        let (c, fell_back) = classify_with_fallback(&FailingClassifier, &input(q), Some(&stats)).await;
        assert!(fell_back);
        assert_eq!(c.request_type, classify_request_type(q));
        assert_eq!(c.confidence, REGEX_CONFIDENCE);
        assert_eq!((stats.calls(), stats.fallbacks()), (1, 1));

        let (_, fell_back) = classify_with_fallback(&RegexRequestClassifier, &input(q), Some(&stats)).await;
        assert!(!fell_back);
        assert_eq!((stats.calls(), stats.fallbacks()), (2, 1));
    }
}
