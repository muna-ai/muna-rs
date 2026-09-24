/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Anthropic `ModelInfo` objects, derived from predictor signatures. The
//! limits and capabilities come from the signature's chat denotations, so
//! what is advertised is exactly what the predictor accepts.

use serde::{Deserialize, Serialize};

use crate::beta::utils::{rfc3339_to_unix, split_tag};
use crate::client::{MunaError, Result};
use crate::services::PredictorService;
use crate::types::{EnumerationValue, Parameter, Signature};

/// Denotation of the chat conversation input carrying the context length.
const CHAT_MESSAGES: &str = "openai.chat.completions.messages";
/// Denotation of the output token cap.
const CHAT_MAX_OUTPUT_TOKENS: &str = "openai.chat.completions.max_output_tokens";
/// Denotation of the reasoning effort selector.
const CHAT_REASONING_EFFORT: &str = "openai.chat.completions.reasoning_effort";
/// Denotation of the decoded image content parts input.
const CHAT_IMAGES: &str = "openai.chat.completions.images";
/// Denotation of the response format selector.
const CHAT_RESPONSE_FORMAT: &str = "openai.chat.completions.response_format";

/// Whether a model supports a capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilitySupport {
    /// Whether this capability is supported by the model.
    pub supported: bool,
}

impl From<bool> for CapabilitySupport {

    fn from(supported: bool) -> Self {
        Self { supported }
    }
}

/// Reasoning effort levels a model accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffortCapability {
    /// Whether the model supports an effort selector at all.
    pub supported: bool,
    /// Whether the model supports the `low` effort level.
    pub low: CapabilitySupport,
    /// Whether the model supports the `medium` effort level.
    pub medium: CapabilitySupport,
    /// Whether the model supports the `high` effort level.
    pub high: CapabilitySupport,
    /// Whether the model supports the `xhigh` effort level.
    pub xhigh: CapabilitySupport,
    /// Whether the model supports the `max` effort level.
    pub max: CapabilitySupport,
}

/// Thinking type configurations a model accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThinkingTypes {
    /// Whether the model supports thinking with type `adaptive`.
    pub adaptive: CapabilitySupport,
    /// Whether the model supports thinking with type `enabled`.
    pub enabled: CapabilitySupport,
}

/// Thinking capability of a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThinkingCapability {
    /// Whether the model can think before answering.
    pub supported: bool,
    /// Supported thinking type configurations.
    pub types: ThinkingTypes,
}

/// Model capabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCapabilities {
    /// Whether the model supports the Batch API.
    pub batch: CapabilitySupport,
    /// Whether the model supports citation generation.
    pub citations: CapabilitySupport,
    /// Whether the model supports code execution tools.
    pub code_execution: CapabilitySupport,
    /// Whether the model supports server-side context management.
    pub context_management: CapabilitySupport,
    /// Reasoning effort support and available levels.
    pub effort: EffortCapability,
    /// Whether the model accepts image content blocks.
    pub image_input: CapabilitySupport,
    /// Whether the model accepts PDF content blocks.
    pub pdf_input: CapabilitySupport,
    /// Whether the model supports structured outputs.
    pub structured_outputs: CapabilitySupport,
    /// Thinking capability and supported type configurations.
    pub thinking: ThinkingCapability,
}

/// Anthropic `ModelInfo` object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model identifier (the predictor tag).
    pub id: String,
    /// Object type, always `model`.
    #[serde(rename = "type")]
    pub kind: String,
    /// Human-readable name, the tag's `name`.
    pub display_name: String,
    /// Release time as an RFC 3339 timestamp; the epoch when unknown.
    pub created_at: String,
    /// Maximum input context window in tokens.
    pub max_input_tokens: Option<u32>,
    /// Maximum value of the `max_tokens` parameter.
    pub max_tokens: Option<u32>,
    /// Model capabilities.
    pub capabilities: ModelCapabilities,
}

impl ModelInfo {

    /// Build the `ModelInfo` object for a predictor from its signature.
    ///
    /// `created` is the Unix time the model became available; Anthropic
    /// allows an epoch value when unknown, so `None` renders as
    /// `1970-01-01T00:00:00Z`.
    pub fn from_signature(
        tag: &str,
        signature: &Signature,
        created: Option<u64>
    ) -> Self {
        let (_, name) = split_tag(tag);
        let inputs = &signature.inputs;
        let thinking = chat_effort_accepts(inputs, "low").is_some();
        let accepts = |level: &str| CapabilitySupport::from(
            chat_effort_accepts(inputs, level).unwrap_or(false)
        );
        Self {
            id: tag.to_string(),
            kind: "model".into(),
            display_name: name.to_string(),
            created_at: unix_to_rfc3339(created.unwrap_or_default()),
            max_input_tokens: chat_context_length(inputs),
            max_tokens: chat_max_output_tokens(inputs),
            capabilities: ModelCapabilities {
                batch: false.into(),
                citations: false.into(),
                code_execution: false.into(),
                context_management: false.into(),
                effort: EffortCapability {
                    supported: thinking,
                    low: accepts("low"),
                    medium: accepts("medium"),
                    high: accepts("high"),
                    xhigh: accepts("xhigh"),
                    max: accepts("max"),
                },
                image_input: find_denoted(inputs, CHAT_IMAGES).is_some().into(),
                pdf_input: false.into(),
                structured_outputs: find_denoted(inputs, CHAT_RESPONSE_FORMAT).is_some().into(),
                thinking: ThinkingCapability {
                    supported: thinking,
                    types: ThinkingTypes {
                        adaptive: false.into(),
                        enabled: thinking.into(),
                    },
                },
            },
        }
    }
}

/// Describe models.
#[derive(Clone)]
pub struct ModelService {
    predictors: PredictorService,
}

impl ModelService {

    pub fn new(predictors: PredictorService) -> Self {
        Self { predictors }
    }

    /// Retrieve a model.
    ///
    /// # Arguments
    /// * `model` - Predictor tag.
    pub async fn retrieve(&self, model: &str) -> Result<ModelInfo> {
        let predictor = self.predictors.retrieve(model).await?.ok_or_else(|| {
            MunaError::Prediction(format!(
                "{model} cannot be retrieved with the Anthropic models API because \
                the predictor could not be found. Check that your access key \
                is valid and that you have access to the predictor."
            ))
        })?;
        Ok(ModelInfo::from_signature(
            &predictor.tag,
            &predictor.signature,
            rfc3339_to_unix(&predictor.created)
        ))
    }
}

/// The first parameter carrying `denotation`, if any.
fn find_denoted<'a>(
    parameters: &'a [Parameter],
    denotation: &str
) -> Option<&'a Parameter> {
    parameters.iter().find(|p| p.denotation.as_deref() == Some(denotation))
}

/// The model's total context window in tokens, from the chat messages input.
fn chat_context_length(inputs: &[Parameter]) -> Option<u32> {
    find_denoted(inputs, CHAT_MESSAGES).and_then(|p| p.context_length)
}

/// The ceiling on output tokens, from the `max_output_tokens` input's range.
fn chat_max_output_tokens(inputs: &[Parameter]) -> Option<u32> {
    find_denoted(inputs, CHAT_MAX_OUTPUT_TOKENS)
        .and_then(|p| p.max)
        .filter(|max| *max >= 1.0)
        .map(|max| max.min(u32::MAX as f64) as u32)
}

/// Whether the reasoning effort selector accepts `level`. `None` when the
/// model has no selector; `Some(true)` for every level when the selector
/// is free text (no enumeration).
fn chat_effort_accepts(
    inputs: &[Parameter],
    level: &str
) -> Option<bool> {
    let parameter = find_denoted(inputs, CHAT_REASONING_EFFORT)?;
    Some(match &parameter.enumeration {
        Some(members) => members.iter().any(|m| matches!(
            &m.value,
            EnumerationValue::String(value) if value == level
        )),
        None => true,
    })
}

/// Unix seconds to `YYYY-MM-DDTHH:MM:SSZ` (Howard Hinnant's civil-from-days).
fn unix_to_rfc3339(unix: u64) -> String {
    let days = (unix / 86_400) as i64;
    let secs = unix % 86_400;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        secs / 3_600,
        (secs % 3_600) / 60,
        secs % 60
    )
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::beta::openai::Model;

    fn chat_signature() -> Signature {
        serde_json::from_value(json!({
            "inputs": [
                {
                    "name": "messages",
                    "dtype": "list",
                    "denotation": "openai.chat.completions.messages",
                    "contextLength": 262144,
                    "batch": { "mode": "continuous" }
                },
                {
                    "name": "tools",
                    "dtype": "list",
                    "denotation": "openai.chat.completions.tools",
                    "optional": true
                },
                {
                    "name": "reasoning_effort",
                    "dtype": "string",
                    "denotation": "openai.chat.completions.reasoning_effort",
                    "optional": true,
                    "enumeration": [
                        { "name": "none", "value": "none" },
                        { "name": "low", "value": "low" },
                        { "name": "medium", "value": "medium" },
                        { "name": "high", "value": "high" },
                        { "name": "xhigh", "value": "xhigh" }
                    ]
                },
                {
                    "name": "max_output_tokens",
                    "dtype": "int32",
                    "denotation": "openai.chat.completions.max_output_tokens",
                    "optional": true,
                    "min": 1,
                    "max": 65536
                },
                {
                    "name": "temperature",
                    "dtype": "float32",
                    "denotation": "openai.chat.completions.temperature",
                    "optional": true
                }
            ],
            "outputs": []
        })).unwrap()
    }

    #[test]
    fn rfc3339_round_trips() {
        for (unix, text) in [
            (0u64, "1970-01-01T00:00:00Z"),
            (951_782_400, "2000-02-29T00:00:00Z"),
            (1_709_164_799, "2024-02-28T23:59:59Z"),
            (1_789_936_000, "2026-09-20T20:26:40Z"),
        ] {
            assert_eq!(unix_to_rfc3339(unix), text);
            assert_eq!(rfc3339_to_unix(text), Some(unix));
        }
        // The API emits millisecond precision.
        assert_eq!(rfc3339_to_unix("2026-09-20T20:26:40.000Z"), Some(1_789_936_000));
        assert_eq!(rfc3339_to_unix("1789936000"), None);
        assert_eq!(rfc3339_to_unix("2026-09-20T20:26:40+02:00"), None);
    }

    #[test]
    fn chat_model_reports_limits_and_capabilities() {
        let info = ModelInfo::from_signature(
            "@qwen/qwen-3.8-27b",
            &chat_signature(),
            Some(1_789_936_000)
        );
        assert_eq!(info.kind, "model");
        assert_eq!(info.display_name, "qwen-3.8-27b");
        assert_eq!(info.created_at, "2026-09-20T20:26:40Z");
        assert_eq!(info.max_input_tokens, Some(262_144));
        assert_eq!(info.max_tokens, Some(65_536));
        let caps = info.capabilities;
        assert!(caps.thinking.supported);
        assert!(caps.thinking.types.enabled.supported);
        assert!(!caps.thinking.types.adaptive.supported);
        assert!(caps.effort.supported);
        assert!(caps.effort.low.supported && caps.effort.xhigh.supported);
        assert!(!caps.effort.max.supported);
        assert!(!caps.image_input.supported);
        assert!(!caps.structured_outputs.supported);
        let json = serde_json::to_value(&info).unwrap();
        assert_eq!(json["type"], "model");
        assert_eq!(json["capabilities"]["effort"]["high"]["supported"], true);
    }

    #[test]
    fn signature_without_chat_denotations_is_a_bare_model() {
        let signature: Signature = serde_json::from_value(json!({
            "inputs": [{ "name": "image", "dtype": "image" }],
            "outputs": [{ "name": "label", "dtype": "string" }]
        })).unwrap();
        let info = ModelInfo::from_signature("@acme/classifier", &signature, None);
        assert_eq!(info.created_at, "1970-01-01T00:00:00Z");
        assert!(!info.capabilities.thinking.supported);
        assert!(!info.capabilities.effort.supported);
        // Limits are nullable and always present.
        let json = serde_json::to_value(&info).unwrap();
        assert!(json["max_input_tokens"].is_null());
        assert!(json["max_tokens"].is_null());
    }

    #[test]
    fn free_text_effort_advertises_every_level() {
        let signature: Signature = serde_json::from_value(json!({
            "inputs": [
                {
                    "name": "messages",
                    "dtype": "list"
                },
                {
                    "name": "effort",
                    "dtype": "string",
                    "denotation": "openai.chat.completions.reasoning_effort"
                },
                {
                    "name": "images",
                    "dtype": "list",
                    "denotation": "openai.chat.completions.images"
                }
            ],
            "outputs": []
        })).unwrap();
        let info = ModelInfo::from_signature("@a/b", &signature, None);
        assert!(info.capabilities.effort.max.supported);
        assert!(info.capabilities.image_input.supported);
    }

    /// The two dialects read the same signature independently; a server
    /// flattens both into one row, so they must agree wherever they overlap.
    #[test]
    fn projections_agree_with_the_openai_model() {
        let signature = chat_signature();
        let created = Some(1_789_936_000);
        let info = ModelInfo::from_signature("@qwen/qwen-3.8-27b", &signature, created);
        let model = Model::from_signature("@qwen/qwen-3.8-27b", &signature, created);
        assert_eq!(info.id, model.id);
        assert_eq!(rfc3339_to_unix(&info.created_at), Some(model.created));
        assert_eq!(format!("@{}/{}", model.owned_by, info.display_name), model.id);
        // Flattening the two must not collide on any key but `id`.
        let a: serde_json::Map<_, _> = serde_json::from_value(serde_json::to_value(&model).unwrap()).unwrap();
        let b: serde_json::Map<_, _> = serde_json::from_value(serde_json::to_value(&info).unwrap()).unwrap();
        let shared: Vec<&String> = a.keys().filter(|k| b.contains_key(*k)).collect();
        assert_eq!(shared, vec!["id"]);
    }
}
