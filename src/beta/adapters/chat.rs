/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use crate::types;

/// What a chat predictor receives from a request, before parameter-name
/// binding: the typed mirror of the chat denotation vocabulary
/// (`openai.chat.completions.*`, `anthropic.messages.*`).
///
/// Every input a chat predictor can receive without I/O lives here.
/// Images are decoded from inline payloads only (`data:` URLs, Anthropic
/// `base64` sources); callers fetch remote images and inline them first,
/// so building these inputs never touches the network. Audio is not here
/// yet: the OpenAI client still decodes it at prediction time. Fields a
/// surface cannot express stay `None` (e.g. `top_k` on OpenAI requests,
/// `reasoning_effort` on Anthropic).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ChatInputs {
    /// Conversation for the predictor's messages input.
    pub messages: Vec<serde_json::Value>,
    /// Decoded images for the predictor's images input
    /// (`openai.chat.completions.images`), in order of appearance across
    /// the conversation: the nth `{"type": "image"}` placeholder in
    /// `messages` is the nth entry. `None` when the request has no images.
    pub images: Option<Vec<types::Image>>,
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
    /// Sampling seed (`openai.chat.completions.seed`).
    pub seed: Option<i32>,
    /// Token frequency penalty (`openai.chat.completions.frequency_penalty`).
    pub frequency_penalty: Option<f32>,
    /// Token presence penalty (`openai.chat.completions.presence_penalty`).
    pub presence_penalty: Option<f32>,
    /// Stop sequences (`openai.chat.completions.stop`).
    pub stop_sequences: Option<Vec<String>>,
    /// Top-k sampling (`anthropic.messages.top_k`).
    pub top_k: Option<i32>,
}

use super::utils::get_parameter;
use crate::client::{MunaError, Result};
use crate::types::{Dtype, Parameter, Signature, Value};
use std::collections::HashMap;

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
/// denotation. Tools or images present with no denoted parameter to take
/// them is a caller error (`InvalidInput`): the request asked for something
/// the model cannot do. Any other knob without a matching parameter is dropped: sampling
/// knobs are advisory and the predictor simply does not expose them.
pub fn bind_chat_inputs(
    inputs: ChatInputs,
    signature: &Signature,
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
                .into(),
        ));
    }
    let mut map = HashMap::new();
    map.insert(messages_param.name.clone(), Value::List(inputs.messages));
    if let Some(tools) = inputs.tools {
        let Some(param) = get_parameter(
            &signature.inputs,
            &[Dtype::List],
            Some("openai.chat.completions.tools"),
        )?
        .map(|(_, parameter)| parameter) else {
            return Err(MunaError::InvalidInput(
                "This predictor does not support tool calling because it does \
                not declare a tools input parameter."
                    .into(),
            ));
        };
        map.insert(param.name.clone(), Value::List(tools));
    }
    if let Some(images) = inputs.images {
        let Some(param) = get_parameter(
            &signature.inputs,
            &[Dtype::ImageList],
            Some("openai.chat.completions.images"),
        )?
        .map(|(_, parameter)| parameter) else {
            return Err(MunaError::InvalidInput(
                "Image content is not supported by this model because it does \
                not declare an images input parameter."
                    .into(),
            ));
        };
        map.insert(param.name.clone(), Value::ImageList(images));
    }
    let mut bind = |value: Option<Value>, dtypes: &[Dtype], denotation: &str| -> Result<()> {
        let Some(value) = value else { return Ok(()) };
        if let Some(param) = get_parameter(&signature.inputs, dtypes, Some(denotation))?
            .map(|(_, parameter)| parameter)
        {
            map.insert(param.name.clone(), value);
        }
        Ok(())
    };
    bind(
        inputs.response_format.map(Value::Dict),
        &[Dtype::Dict],
        "openai.chat.completions.response_format",
    )?;
    bind(
        inputs.reasoning_effort.map(Value::String),
        &[Dtype::String],
        "openai.chat.completions.reasoning_effort",
    )?;
    bind(
        inputs.max_output_tokens.map(Value::Int),
        INT_DTYPES,
        "openai.chat.completions.max_output_tokens",
    )?;
    bind(
        inputs.temperature.map(Value::Float),
        FLOAT_DTYPES,
        "openai.chat.completions.temperature",
    )?;
    bind(
        inputs.top_p.map(Value::Float),
        FLOAT_DTYPES,
        "openai.chat.completions.top_p",
    )?;
    bind(
        inputs.seed.map(Value::Int),
        INT_DTYPES,
        "openai.chat.completions.seed",
    )?;
    bind(
        inputs.frequency_penalty.map(Value::Float),
        FLOAT_DTYPES,
        "openai.chat.completions.frequency_penalty",
    )?;
    bind(
        inputs.presence_penalty.map(Value::Float),
        FLOAT_DTYPES,
        "openai.chat.completions.presence_penalty",
    )?;
    bind(
        inputs.stop_sequences.map(|sequences| {
            Value::List(
                sequences
                    .into_iter()
                    .map(serde_json::Value::String)
                    .collect(),
            )
        }),
        &[Dtype::List],
        "openai.chat.completions.stop",
    )?;
    bind(
        inputs.top_k.map(Value::Int),
        INT_DTYPES,
        "anthropic.messages.top_k",
    )?;
    Ok(map)
}

#[cfg(test)]
#[path = "tests/chat.rs"]
mod tests;
