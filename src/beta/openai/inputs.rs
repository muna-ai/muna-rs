/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Chat predictor inputs: what a chat predictor receives from a request,
//! and how those inputs bind to a predictor's parameter names.
//!
//! Two steps, deliberately kept apart:
//!
//! 1. Request -> [`ChatInputs`]. Dialect-specific (OpenAI vs Anthropic
//!    disagree on where the system prompt lives and how tools are spelled)
//!    and signature-agnostic. Owned by `ChatCompletionCreateParams::chat_inputs`
//!    and `MessageCreateParams::chat_inputs`.
//! 2. [`ChatInputs`] -> predictor input map, via [`bind_chat_inputs`].
//!    Dialect-agnostic and signature-specific: every field binds to the
//!    parameter carrying its denotation, so nothing here presupposes the
//!    names the model author gave their parameters.
//!
//! Servers bind against the base predictor's signature; the control
//! plane's router-hash path binds the same `ChatInputs` against the router
//! sidecar (which mirrors the base signature by construction), so both see
//! the same bytes.

use std::collections::HashMap;

use crate::client::Result;
use crate::MunaError;
use crate::beta::utils::get_parameter;
use crate::types::{Dtype, Parameter, Signature, Value};

/// What a chat predictor receives from a request, before parameter-name
/// binding and media decoding: the typed mirror of the chat denotation
/// vocabulary (`openai.chat.completions.*`, `anthropic.messages.*`).
///
/// Every input a chat predictor can receive without I/O lives here; media
/// stay out (decoding is the client's job, and the control plane must
/// never fetch URLs). Fields a surface cannot express stay `None`
/// (e.g. `top_k` on OpenAI requests, `reasoning_effort` on Anthropic).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ChatInputs {
    /// Conversation for the predictor's messages input.
    pub messages: Vec<serde_json::Value>,
    /// Tools for the predictor's tools input; `None` when the request
    /// carries no tools (absent or empty).
    pub tools: Option<Vec<serde_json::Value>>,
    /// Response format (`openai.chat.completions.response_format`).
    pub response_format: Option<serde_json::Map<String, serde_json::Value>>,
    /// Reasoning effort (`openai.chat.completions.reasoning_effort`).
    pub reasoning_effort: Option<String>,
    /// Maximum output tokens (`openai.chat.completions.max_output_tokens`).
    pub max_output_tokens: Option<i32>,
    /// Sampling temperature (`openai.chat.completions.temperature`).
    pub temperature: Option<f32>,
    /// Nucleus sampling coefficient (`openai.chat.completions.top_p`).
    pub top_p: Option<f32>,
    /// Token frequency penalty (`openai.chat.completions.frequency_penalty`).
    pub frequency_penalty: Option<f32>,
    /// Token presence penalty (`openai.chat.completions.presence_penalty`).
    pub presence_penalty: Option<f32>,
    /// Stop sequences (`anthropic.messages.stop_sequences`).
    pub stop_sequences: Option<Vec<String>>,
    /// Top-k sampling (`anthropic.messages.top_k`).
    pub top_k: Option<i32>,
}

const FLOAT_DTYPES: &[Dtype] = &[Dtype::Float32, Dtype::Float64];
const INT_DTYPES: &[Dtype] = &[
    Dtype::Int8,
    Dtype::Int16,
    Dtype::Int32,
    Dtype::Int64,
    Dtype::Uint8,
    Dtype::Uint16,
    Dtype::Uint32,
    Dtype::Uint64,
];

/// Bind chat inputs to a predictor's parameter names.
///
/// The predictor's sole required input, which must be a `list`, receives
/// `messages`. Every other field binds to the input carrying its
/// denotation. Tools present with no denoted tools parameter is a caller
/// error (`InvalidInput`): the request asked for something the model cannot
/// do. Any other knob without a matching parameter is dropped: sampling
/// knobs are advisory and the predictor simply does not expose them.
pub fn bind_chat_inputs(
    inputs: ChatInputs,
    signature: &Signature
) -> Result<HashMap<String, Value>> {
    let required: Vec<&Parameter> = signature
        .inputs
        .iter()
        .filter(|p| !p.optional.unwrap_or(false))
        .collect();
    let [messages_param] = required.as_slice() else {
        return Err(MunaError::Prediction(format!(
            "This predictor cannot be used as a chat predictor because it \
            declares {} required input parameters; chat predictors declare \
            exactly one (the messages input).",
            required.len()
        )));
    };
    if messages_param.dtype != Some(Dtype::List) {
        return Err(MunaError::Prediction(
            "This predictor cannot be used as a chat predictor because its \
            required input parameter is not a `list` of chat messages."
                .into()
        ));
    }
    let mut map = HashMap::new();
    map.insert(messages_param.name.clone(), Value::List(inputs.messages));
    if let Some(tools) = inputs.tools {
        let Some(param) = get_parameter(
            &signature.inputs,
            &[Dtype::List],
            Some("openai.chat.completions.tools")
        ).1 else {
            return Err(MunaError::InvalidInput(
                "This predictor does not support tool calling because it does \
                not declare a tools input parameter."
                    .into()
            ));
        };
        map.insert(param.name.clone(), Value::List(tools));
    }
    let mut bind = |value: Option<Value>, dtypes: &[Dtype], denotation: &str| {
        let Some(value) = value else { return };
        if let Some(param) = get_parameter(&signature.inputs, dtypes, Some(denotation)).1 {
            map.insert(param.name.clone(), value);
        }
    };
    bind(
        inputs.response_format.map(Value::Dict),
        &[Dtype::Dict],
        "openai.chat.completions.response_format"
    );
    bind(
        inputs.reasoning_effort.map(Value::String),
        &[Dtype::String],
        "openai.chat.completions.reasoning_effort"
    );
    bind(
        inputs.max_output_tokens.map(Value::Int),
        INT_DTYPES,
        "openai.chat.completions.max_output_tokens"
    );
    bind(
        inputs.temperature.map(Value::Float),
        FLOAT_DTYPES,
        "openai.chat.completions.temperature"
    );
    bind(
        inputs.top_p.map(Value::Float),
        FLOAT_DTYPES,
        "openai.chat.completions.top_p"
    );
    bind(
        inputs.frequency_penalty.map(Value::Float),
        FLOAT_DTYPES,
        "openai.chat.completions.frequency_penalty"
    );
    bind(
        inputs.presence_penalty.map(Value::Float),
        FLOAT_DTYPES,
        "openai.chat.completions.presence_penalty"
    );
    bind(
        inputs.stop_sequences.map(|sequences| {
            Value::List(sequences.into_iter().map(serde_json::Value::String).collect())
        }),
        &[Dtype::List],
        "anthropic.messages.stop_sequences"
    );
    bind(
        inputs.top_k.map(Value::Int),
        INT_DTYPES,
        "anthropic.messages.top_k"
    );
    Ok(map)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn signature(inputs: serde_json::Value) -> Signature {
        serde_json::from_value(json!({ "inputs": inputs, "outputs": [] })).unwrap()
    }

    #[test]
    fn binds_by_role_not_by_name() {
        // Parameter names are the model author's; binding must not
        // presuppose `messages` / `tools`.
        let signature = signature(json!([
            { "name": "conversation", "dtype": "list" },
            { "name": "functions", "dtype": "list", "optional": true,
              "denotation": "openai.chat.completions.tools" },
            { "name": "heat", "dtype": "float32", "optional": true,
              "denotation": "openai.chat.completions.temperature" },
            { "name": "budget", "dtype": "int32", "optional": true,
              "denotation": "openai.chat.completions.max_output_tokens" },
            { "name": "effort", "dtype": "string", "optional": true,
              "denotation": "openai.chat.completions.reasoning_effort" },
            { "name": "format", "dtype": "dict", "optional": true,
              "denotation": "openai.chat.completions.response_format" },
            { "name": "stops", "dtype": "list", "optional": true,
              "denotation": "anthropic.messages.stop_sequences" },
            { "name": "k", "dtype": "int32", "optional": true,
              "denotation": "anthropic.messages.top_k" }
        ]));
        let inputs = ChatInputs {
            messages: vec![json!({ "role": "user", "content": "hi" })],
            tools: Some(vec![json!({ "type": "function" })]),
            response_format: Some(json!({ "type": "json_object" }).as_object().cloned().unwrap()),
            reasoning_effort: Some("high".into()),
            max_output_tokens: Some(64),
            temperature: Some(0.5),
            top_p: Some(0.9),
            frequency_penalty: None,
            presence_penalty: None,
            stop_sequences: Some(vec!["END".into()]),
            top_k: Some(40),
        };
        let map = bind_chat_inputs(inputs, &signature).unwrap();
        assert!(matches!(map.get("conversation"), Some(Value::List(m)) if m.len() == 1));
        assert!(matches!(map.get("functions"), Some(Value::List(t)) if t.len() == 1));
        assert!(matches!(map.get("heat"), Some(Value::Float(v)) if *v == 0.5));
        assert!(matches!(map.get("budget"), Some(Value::Int(64))));
        assert!(matches!(map.get("effort"), Some(Value::String(s)) if s == "high"));
        assert!(matches!(map.get("format"), Some(Value::Dict(d)) if d["type"] == "json_object"));
        assert!(matches!(
            map.get("stops"),
            Some(Value::List(s)) if s == &vec![json!("END")]
        ));
        assert!(matches!(map.get("k"), Some(Value::Int(40))));
        // `top_p` set but undeclared: silently dropped, nothing else bound.
        assert_eq!(map.len(), 8);
    }

    #[test]
    fn undeclared_knobs_are_dropped_without_error() {
        let signature = signature(json!([{ "name": "messages", "dtype": "list" }]));
        let inputs = ChatInputs {
            messages: vec![],
            temperature: Some(0.2),
            max_output_tokens: Some(8),
            ..Default::default()
        };
        let map = bind_chat_inputs(inputs, &signature).unwrap();
        assert_eq!(map.len(), 1);
        assert!(matches!(map.get("messages"), Some(Value::List(_))));
    }

    #[test]
    fn tools_without_a_tools_parameter_is_a_caller_error() {
        let signature = signature(json!([{ "name": "messages", "dtype": "list" }]));
        let inputs = ChatInputs {
            messages: vec![],
            tools: Some(vec![json!({ "type": "function" })]),
            ..Default::default()
        };
        assert!(matches!(
            bind_chat_inputs(inputs, &signature),
            Err(MunaError::InvalidInput(_))
        ));
        // No tools requested: the missing parameter is irrelevant.
        let inputs = ChatInputs { messages: vec![], ..Default::default() };
        assert!(bind_chat_inputs(inputs, &signature).is_ok());
    }

    #[test]
    fn rejects_signatures_without_a_single_required_list_input() {
        let none = signature(json!([{ "name": "x", "dtype": "list", "optional": true }]));
        let two = signature(json!([
            { "name": "a", "dtype": "list" },
            { "name": "b", "dtype": "list" }
        ]));
        let not_list = signature(json!([{ "name": "prompt", "dtype": "string" }]));
        for signature in [none, two, not_list] {
            assert!(matches!(
                bind_chat_inputs(ChatInputs::default(), &signature),
                Err(MunaError::Prediction(_))
            ));
        }
    }
}
