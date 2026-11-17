/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::collections::{BTreeMap, HashMap};
use std::pin::Pin;

use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use futures_core::Stream;
use futures_util::StreamExt;
use serde_json::json;

use crate::c;
use crate::client::{MunaError, Result};
use crate::types::{self, Dtype, Parameter, Signature, Value};

use super::chat::ChatInputs;
use super::{
    utils::{
        decode, output, parameter, require, required, result, validate_inputs, FLOATS, INTEGERS,
    },
    Adapter, PredictionStream,
};

mod schema;
pub use schema::*;
pub type ChatCompletionStream = Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk>> + Send>>;

#[derive(Debug, Clone, Default)]
pub struct OpenAICompletionsOptions {
    pub include_usage: bool,
}

#[derive(Debug, Clone)]
pub struct OpenAICompletionsAdapter {
    pub messages: Parameter,
    pub output: usize,
    pub images: Option<Parameter>,
    pub audios: Option<Parameter>,
    pub temperature: Option<Parameter>,
    pub top_p: Option<Parameter>,
    pub seed: Option<Parameter>,
    pub max_completion_tokens: Option<Parameter>,
    pub frequency_penalty: Option<Parameter>,
    pub presence_penalty: Option<Parameter>,
    pub response_format: Option<Parameter>,
    pub reasoning_effort: Option<Parameter>,
    pub tools: Option<Parameter>,
    pub stop: Option<Parameter>,
}

impl OpenAICompletionsAdapter {
    pub fn from_signature(signature: &Signature) -> Result<Self> {
        // Required conversation input.
        let messages = require(
            parameter(&required(&signature.inputs), &[Dtype::List], None)?,
            "Chat requires one required messages list input",
        )?;
        // Images parameter.
        let images = parameter(
            &signature.inputs,
            &[Dtype::ImageList],
            Some("openai.chat.completions.images"),
        )?;
        // Audios parameter.
        let audios = parameter(
            &signature.inputs,
            &[Dtype::ArrayList],
            Some("openai.chat.completions.audios"),
        )?;
        // Temperature parameter.
        let temperature = parameter(
            &signature.inputs,
            FLOATS,
            Some("openai.chat.completions.temperature"),
        )?;
        // Top p parameter.
        let top_p = parameter(
            &signature.inputs,
            FLOATS,
            Some("openai.chat.completions.top_p"),
        )?;
        // Seed parameter.
        let seed = parameter(
            &signature.inputs,
            INTEGERS,
            Some("openai.chat.completions.seed"),
        )?;
        // Max completion tokens parameter.
        let max_completion_tokens = parameter(
            &signature.inputs,
            INTEGERS,
            Some("openai.chat.completions.max_output_tokens"),
        )?;
        // Frequency penalty parameter.
        let frequency_penalty = parameter(
            &signature.inputs,
            FLOATS,
            Some("openai.chat.completions.frequency_penalty"),
        )?;
        // Presence penalty parameter.
        let presence_penalty = parameter(
            &signature.inputs,
            FLOATS,
            Some("openai.chat.completions.presence_penalty"),
        )?;
        // Response format parameter.
        let response_format = parameter(
            &signature.inputs,
            &[Dtype::Dict],
            Some("openai.chat.completions.response_format"),
        )?;
        // Reasoning effort parameter.
        let reasoning_effort = parameter(
            &signature.inputs,
            &[Dtype::String],
            Some("openai.chat.completions.reasoning_effort"),
        )?;
        // Tools parameter.
        let tools = parameter(
            &signature.inputs,
            &[Dtype::List],
            Some("openai.chat.completions.tools"),
        )?;
        // Stop parameter.
        let stop = parameter(
            &signature.inputs,
            &[Dtype::List],
            Some("openai.chat.completions.stop"),
        )?;
        validate_inputs(
            signature,
            &[
                Some(&messages),
                images.as_ref(),
                audios.as_ref(),
                temperature.as_ref(),
                top_p.as_ref(),
                seed.as_ref(),
                max_completion_tokens.as_ref(),
                frequency_penalty.as_ref(),
                presence_penalty.as_ref(),
                response_format.as_ref(),
                reasoning_effort.as_ref(),
                tools.as_ref(),
                stop.as_ref(),
            ],
        )?;
        if audios
            .as_ref()
            .is_some_and(|parameter| parameter.sample_rate.is_none_or(|rate| rate == 0))
        {
            return Err(MunaError::InvalidInput(
                "Chat audio input must declare a positive sample rate".into(),
            ));
        }
        // Predictors yield the same chunk schema for both API dialects.
        let output = require(
            output(signature, &[Dtype::Dict], "ChatCompletionChunk")?,
            "Chat requires one ChatCompletionChunk dictionary output",
        )?;
        Ok(Self {
            messages,
            output,
            images,
            audios,
            temperature,
            top_p,
            seed,
            max_completion_tokens,
            frequency_penalty,
            presence_penalty,
            response_format,
            reasoning_effort,
            tools,
            stop,
        })
    }

    pub fn bind_inputs(&self, input: ChatInputs) -> Result<HashMap<String, Value>> {
        let mut inputs = HashMap::from([(self.messages.name.clone(), Value::List(input.messages))]);
        if let Some(images) = input.images {
            let parameter = require(
                self.images.as_ref(),
                "Image content is not supported by this model",
            )?;
            inputs.insert(parameter.name.clone(), Value::ImageList(images));
        }
        if let Some(tools) = input.tools {
            let parameter = require(
                self.tools.as_ref(),
                "This predictor does not support tool calling",
            )?;
            inputs.insert(parameter.name.clone(), Value::List(tools));
        }
        if let (Some(parameter), Some(value)) = (&self.response_format, input.response_format) {
            inputs.insert(parameter.name.clone(), Value::Dict(value));
        }
        if let (Some(parameter), Some(value)) = (&self.reasoning_effort, input.reasoning_effort) {
            inputs.insert(parameter.name.clone(), Value::String(value));
        }
        if let (Some(parameter), Some(value)) =
            (&self.max_completion_tokens, input.max_output_tokens)
        {
            inputs.insert(parameter.name.clone(), Value::Int(value));
        }
        if let (Some(parameter), Some(value)) = (&self.temperature, input.temperature) {
            inputs.insert(
                parameter.name.clone(),
                if parameter.dtype == Some(Dtype::Float64) {
                    Value::Double(value as f64)
                } else {
                    Value::Float(value)
                },
            );
        }
        if let (Some(parameter), Some(value)) = (&self.top_p, input.top_p) {
            inputs.insert(
                parameter.name.clone(),
                if parameter.dtype == Some(Dtype::Float64) {
                    Value::Double(value as f64)
                } else {
                    Value::Float(value)
                },
            );
        }
        if let (Some(parameter), Some(value)) = (&self.seed, input.seed) {
            inputs.insert(parameter.name.clone(), Value::Int(value));
        }
        if let (Some(parameter), Some(value)) = (&self.frequency_penalty, input.frequency_penalty) {
            inputs.insert(
                parameter.name.clone(),
                if parameter.dtype == Some(Dtype::Float64) {
                    Value::Double(value as f64)
                } else {
                    Value::Float(value)
                },
            );
        }
        if let (Some(parameter), Some(value)) = (&self.presence_penalty, input.presence_penalty) {
            inputs.insert(
                parameter.name.clone(),
                if parameter.dtype == Some(Dtype::Float64) {
                    Value::Double(value as f64)
                } else {
                    Value::Float(value)
                },
            );
        }
        if let (Some(parameter), Some(value)) = (&self.stop, input.stop_sequences) {
            inputs.insert(
                parameter.name.clone(),
                Value::List(value.into_iter().map(serde_json::Value::String).collect()),
            );
        }
        Ok(inputs)
    }

    pub fn inputs(&self, params: &ChatCompletionCreateParams) -> Result<HashMap<String, Value>> {
        let audios = decode_audios(&params.messages, self)?;
        // Retain image URLs as wire values until the execution boundary.
        let images = media_parts(&params.messages)
            .into_iter()
            .filter_map(|part| match part {
                ChatCompletionContentPart::ImageUrl { image_url } => Some(
                    super::edits::ImageInput::Remote(crate::types::RemoteValue {
                        dtype: Dtype::Image,
                        data: Some(image_url.url.clone().into()),
                    }),
                ),
                _ => None,
            })
            .collect::<Vec<_>>();
        // Normalization preserves media placeholders; only chat_inputs' eager
        // image decoding is bypassed here.
        let input = ChatInputs {
            messages: normalize_messages(&params.messages)?,
            tools: params
                .tools
                .as_ref()
                .filter(|tools| !tools.is_empty())
                .map(|tools| tools.iter().map(serde_json::to_value).collect())
                .transpose()?,
            response_format: params.response_format.clone(),
            reasoning_effort: params
                .reasoning_effort
                .map(|effort| effort.as_str().to_string()),
            max_output_tokens: params.max_completion_tokens,
            temperature: params.temperature,
            top_p: params.top_p,
            seed: params.seed,
            frequency_penalty: params.frequency_penalty,
            presence_penalty: params.presence_penalty,
            ..Default::default()
        };
        let mut inputs = self.bind_inputs(input)?;
        if !images.is_empty() {
            let parameter = require(
                self.images.as_ref(),
                "Image content is not supported by this model",
            )?;
            inputs.insert(parameter.name.clone(), super::edits::image_list(images)?);
        }
        if !audios.is_empty() {
            inputs.insert(
                self.audios.as_ref().unwrap().name.clone(),
                Value::ArrayList(audios),
            );
        }
        Ok(inputs)
    }
}

#[async_trait]
impl Adapter for OpenAICompletionsAdapter {
    type Source = PredictionStream;
    type Options = OpenAICompletionsOptions;
    type Response = ChatCompletionStream;

    fn kind(&self) -> &'static str {
        "openai.chat.completions.create"
    }

    async fn adapt(
        &self,
        mut predictions: PredictionStream,
        options: Option<Self::Options>,
    ) -> Result<ChatCompletionStream> {
        let options = options.unwrap_or_default();
        let output = self.output;
        Ok(Box::pin(async_stream::try_stream! {
            let mut last = None;
            let mut usage: Option<ChatCompletionUsage> = None;
            while let Some(prediction) = predictions.next().await {
                let prediction = prediction?;
                let Value::Dict(value) = decode(result(&prediction, output)?, Dtype::Dict).await? else {
                    Err(MunaError::Prediction("Chat output must be a dictionary".into()))?;
                    unreachable!()
                };
                let mut chunk = parse_chat_completion_chunk(value)?;
                if let Some(delta) = chunk.usage.take() {
                    let total = usage.get_or_insert(ChatCompletionUsage { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0, prompt_tokens_details: None, completion_tokens_details: None });
                    total.prompt_tokens += delta.prompt_tokens;
                    total.completion_tokens += delta.completion_tokens;
                    total.total_tokens += delta.total_tokens;
                    if delta.prompt_tokens_details.is_some() { total.prompt_tokens_details = delta.prompt_tokens_details; }
                    if delta.completion_tokens_details.is_some() { total.completion_tokens_details = delta.completion_tokens_details; }
                }
                last = Some(chunk.clone());
                if !chunk.choices.is_empty() { yield chunk; }
            }
            let mut last = last.ok_or_else(|| MunaError::Prediction("Chat predictor did not return any chunks".into()))?;
            if options.include_usage && usage.is_some() {
                last.choices.clear();
                last.usage = usage;
                yield last;
            }
        }))
    }
}
impl ChatCompletionCreateParams {
    /// Inputs the chat predictor receives for this request. `tool_choice` is
    /// the predictor's to interpret (drop, prefill, constrain), so `tools`
    /// is forwarded verbatim whenever present. Anthropic-only knobs stay
    /// `None`.
    ///
    /// `image_url` parts must be `data:` URLs: they are decoded here, in
    /// order of appearance, without any network access. Callers that accept
    /// remote image URLs fetch them and inline them as `data:` URLs first.
    /// Decoding is CPU-bound (tens of milliseconds per large image), so
    /// async callers should run this off the executor.
    pub fn chat_inputs(&self) -> Result<ChatInputs> {
        let tools = match &self.tools {
            Some(tools) if !tools.is_empty() => Some(
                tools
                    .iter()
                    .map(serde_json::to_value)
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(|e| MunaError::Prediction(e.to_string()))?,
            ),
            _ => None,
        };
        let images = media_parts(&self.messages)
            .into_iter()
            .filter_map(|part| match part {
                ChatCompletionContentPart::ImageUrl { image_url } => Some(&image_url.url),
                _ => None,
            })
            .map(|url| decode_image_url(url))
            .collect::<Result<Vec<_>>>()?;
        Ok(ChatInputs {
            messages: normalize_messages(&self.messages)?,
            images: (!images.is_empty()).then_some(images),
            tools,
            response_format: self.response_format.clone(),
            reasoning_effort: self.reasoning_effort.map(|e| e.as_str().to_string()),
            max_output_tokens: self.max_completion_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            seed: self.seed,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            stop_sequences: None,
            top_k: None,
        })
    }
}

/// Messages as the chat predictor receives them: wire shape preserved,
/// except media parts are swapped for payload-free `{"type": "image"}` /
/// `{"type": "audio"}` placeholders (the nth placeholder across the
/// conversation indexes the nth entry of the parallel media input). Text
/// and refusal parts ride through untouched: the compiled chat template
/// flattens and canonicalizes content parts itself.
fn normalize_messages(messages: &[ChatCompletionMessage]) -> Result<Vec<serde_json::Value>> {
    if messages.is_empty() {
        return Err(MunaError::InvalidInput(
            "`messages` must contain at least one message.".into(),
        ));
    }
    messages
        .iter()
        .enumerate()
        .map(|(index, message)| {
            validate_message(index, message)?;
            let mut value =
                serde_json::to_value(message).map_err(|e| MunaError::Prediction(e.to_string()))?;
            // An assistant turn with nothing to say is rendered as an empty
            // turn by the chat templates; hand them `""` rather than `null`
            // so templates that index into content need no `is none` guard.
            if message.role == "assistant" && message.content.is_none() {
                value["content"] = json!("");
            }
            if let Some(parts) = value.get_mut("content").and_then(|c| c.as_array_mut()) {
                for part in parts.iter_mut() {
                    match part.get("type").and_then(|t| t.as_str()) {
                        Some("image_url") => *part = json!({ "type": "image" }),
                        Some("input_audio") => *part = json!({ "type": "audio" }),
                        _ => {}
                    }
                }
            }
            Ok(value)
        })
        .collect()
}

/// Reject a message that carries nothing for the model to read.
///
/// Only messages the model must *answer* are checked. Chat templates trim
/// text content (Gemma 4's `content | trim`, for one), so a whitespace-only
/// user turn renders as an EMPTY turn and the model answers a question it was
/// never asked. An empty assistant turn is different: it is history the model
/// wrote, and it renders harmlessly as "the assistant said nothing". Clients
/// legitimately replay such turns -- a reasoning-only completion, a stream
/// cut off after the role chunk, an agent keeping strict role alternation --
/// and refusing them wedges the conversation on our own output. Tool results
/// may be empty by design.
fn validate_message(index: usize, message: &ChatCompletionMessage) -> Result<()> {
    if message.role == "tool" || message.role == "assistant" {
        return Ok(());
    }
    let has_payload = match &message.content {
        None => false,
        Some(ChatCompletionContent::Text(text)) => !text.trim().is_empty(),
        Some(ChatCompletionContent::Parts(parts)) => parts.iter().any(|part| match part {
            ChatCompletionContentPart::Text { text } => !text.trim().is_empty(),
            ChatCompletionContentPart::Refusal { refusal } => !refusal.trim().is_empty(),
            _ => true,
        }),
    };
    if has_payload {
        return Ok(());
    }
    Err(MunaError::InvalidInput(format!(
        "`messages[{index}]` ({} message) must have non-empty `content`. Text \
        content must contain non-whitespace characters.",
        message.role
    )))
}

/// Media content parts in order of appearance across the conversation;
/// the nth image / audio part here is the nth placeholder `normalize_messages`
/// emits and the nth entry of the corresponding parallel media input.
fn media_parts(messages: &[ChatCompletionMessage]) -> Vec<&ChatCompletionContentPart> {
    messages
        .iter()
        .filter_map(|message| match &message.content {
            Some(ChatCompletionContent::Parts(parts)) => Some(parts.iter()),
            _ => None,
        })
        .flatten()
        .filter(|part| {
            !matches!(
                part,
                ChatCompletionContentPart::Text { .. } | ChatCompletionContentPart::Refusal { .. }
            )
        })
        .collect()
}

/// Decode the conversation's audio parts in order of appearance, rejecting
/// audio the model does not declare and file parts. Images are decoded by
/// `chat_inputs` and bound (or rejected) by `bind_chat_inputs`.
fn decode_audios(
    messages: &[ChatCompletionMessage],
    info: &OpenAICompletionsAdapter,
) -> Result<Vec<types::Tensor>> {
    let mut audios = Vec::new();
    for part in media_parts(messages) {
        match part {
            ChatCompletionContentPart::InputAudio { input_audio } if info.audios.is_some() => {
                audios.push(decode_audio(
                    input_audio,
                    info.audios
                        .as_ref()
                        .and_then(|parameter| parameter.sample_rate),
                )?);
            }
            ChatCompletionContentPart::InputAudio { .. } => {
                return Err(MunaError::InvalidInput(
                    "`input_audio` content is not supported by this model.".into(),
                ));
            }
            ChatCompletionContentPart::File { .. } => {
                return Err(MunaError::InvalidInput(
                    "File content parts are not yet supported.".into(),
                ));
            }
            ChatCompletionContentPart::ImageUrl { .. }
            | ChatCompletionContentPart::Text { .. }
            | ChatCompletionContentPart::Refusal { .. } => {}
        }
    }
    Ok(audios)
}

/// Decode an `image_url` content part's `data:` URL into a pixel buffer.
/// Remote URLs are rejected: fetching is the caller's job.
fn decode_image_url(url: &str) -> Result<types::Image> {
    let Some(rest) = url.strip_prefix("data:") else {
        return Err(MunaError::InvalidInput(
            "`image_url` content must be a base64 `data:` URL. Fetch remote \
            images and pass them inline as data URLs."
                .into(),
        ));
    };
    let (meta, payload) = rest.split_once(',').ok_or_else(|| {
        MunaError::InvalidInput("Malformed data URL in `image_url` content part.".into())
    })?;
    let mime = meta
        .split(';')
        .next()
        .filter(|m| !m.is_empty())
        .unwrap_or("image/*");
    decode_base64_image(payload, mime)
}

/// Decode a base64 image payload of the given MIME type (e.g. `image/png`)
/// into a pixel buffer through the Function C library.
pub(crate) fn decode_base64_image(payload: &str, mime: &str) -> Result<types::Image> {
    let data = BASE64
        .decode(payload)
        .map_err(|e| MunaError::InvalidInput(format!("Failed to decode base64 image data: {e}")))?;
    let invalid = || {
        MunaError::InvalidInput(format!(
            "Failed to decode image content of type `{mime}` into an image."
        ))
    };
    let value = c::Value::from_bytes(&data, mime).map_err(|_| invalid())?;
    match value.to_object().map_err(|_| invalid())? {
        types::Value::Image(image) => Ok(image),
        _ => Err(invalid()),
    }
}

/// Decode an audio content part into linear PCM samples at the model's
/// declared sample rate via the Function C library.
fn decode_audio(
    input_audio: &ChatCompletionContentPartInputAudio,
    sample_rate: Option<u32>,
) -> Result<types::Tensor> {
    let sample_rate = sample_rate.ok_or_else(|| {
        MunaError::Prediction("Model does not declare a sample rate for its audio input.".into())
    })?;
    let data = BASE64
        .decode(&input_audio.data)
        .map_err(|e| MunaError::InvalidInput(format!("Failed to decode audio data: {e}")))?;
    let mime = format!("audio/{};rate={sample_rate}", input_audio.format.as_str());
    let value = c::Value::from_bytes(&data, &mime)?;
    match value.to_object()? {
        types::Value::Tensor(tensor) => Ok(tensor),
        _ => Err(MunaError::InvalidInput(
            "Failed to decode `input_audio` content part into PCM samples.".into(),
        )),
    }
}

fn parse_chat_completion_chunk(
    output: serde_json::Map<String, serde_json::Value>,
) -> Result<ChatCompletionChunk> {
    match object_kind(&output) {
        Some("chat.completion.chunk") => from_object(output),
        _ => Err(MunaError::Prediction(
            "Failed to parse chat completion chunk from model output. \
            Chat predictors must yield `ChatCompletionChunk` outputs."
                .into(),
        )),
    }
}

pub(crate) fn merge_chunks(chunks: Vec<ChatCompletionChunk>) -> Result<ChatCompletion> {
    let first = chunks.first().ok_or_else(|| {
        MunaError::Prediction(
            "Failed to parse chat completion because model did not return any outputs".into(),
        )
    })?;
    let mut choices_map = BTreeMap::<usize, Vec<ChatCompletionChunkChoice>>::new();
    for chunk in &chunks {
        for choice in &chunk.choices {
            choices_map
                .entry(choice.index)
                .or_default()
                .push(choice.clone());
        }
    }
    let choices = choices_map
        .into_iter()
        .map(|(index, choices)| create_completion_choice(index, choices))
        .collect();
    let usage = ChatCompletionUsage {
        prompt_tokens: chunks
            .iter()
            .filter_map(|c| c.usage.as_ref())
            .map(|u| u.prompt_tokens)
            .sum(),
        completion_tokens: chunks
            .iter()
            .filter_map(|c| c.usage.as_ref())
            .map(|u| u.completion_tokens)
            .sum(),
        total_tokens: chunks
            .iter()
            .filter_map(|c| c.usage.as_ref())
            .map(|u| u.total_tokens)
            .sum(),
        // Engines report token details (e.g. cached / reasoning tokens) on the
        // final usage-bearing chunk; sums would double-count, so take the last.
        prompt_tokens_details: chunks
            .iter()
            .filter_map(|c| c.usage.as_ref())
            .filter_map(|u| u.prompt_tokens_details.clone())
            .last(),
        completion_tokens_details: chunks
            .iter()
            .filter_map(|c| c.usage.as_ref())
            .filter_map(|u| u.completion_tokens_details.clone())
            .last(),
    };
    Ok(ChatCompletion {
        object: "chat.completion".to_string(),
        id: first.id.clone(),
        model: first.model.clone(),
        created: first.created,
        choices,
        usage: Some(usage),
    })
}

fn create_completion_choice(
    index: usize,
    choices: Vec<ChatCompletionChunkChoice>,
) -> ChatCompletionChoice {
    let role = choices
        .iter()
        .filter_map(|choice| choice.delta.as_ref())
        .filter_map(|delta| delta.role.clone())
        .next()
        .unwrap_or_else(|| "assistant".to_string());
    let content = choices
        .iter()
        .filter_map(|choice| choice.delta.as_ref())
        .filter_map(|delta| delta.content.as_deref())
        .collect::<String>();
    let reasoning_content = choices
        .iter()
        .filter_map(|choice| choice.delta.as_ref())
        .filter_map(|delta| delta.reasoning_content.as_deref())
        .collect::<String>();
    let finish_reason = choices
        .iter()
        .filter_map(|choice| choice.finish_reason.clone())
        .next();
    ChatCompletionChoice {
        index,
        message: ChatCompletionMessage {
            role,
            content: Some(ChatCompletionContent::Text(content)),
            reasoning_content: (!reasoning_content.is_empty()).then_some(reasoning_content),
            tool_calls: merge_tool_calls(&choices),
            tool_call_id: None,
        },
        finish_reason,
        logprobs: None,
    }
}

/// Accumulate streamed tool call fragments into completed tool calls,
/// keyed by fragment index: the first fragment carries the id and
/// function name; subsequent fragments append argument text.
fn merge_tool_calls(
    choices: &[ChatCompletionChunkChoice],
) -> Option<Vec<ChatCompletionMessageFunctionToolCall>> {
    let mut calls: BTreeMap<usize, ChatCompletionMessageFunctionToolCall> = BTreeMap::new();
    for choice in choices {
        let Some(fragments) = choice.delta.as_ref().and_then(|d| d.tool_calls.as_ref()) else {
            continue;
        };
        for fragment in fragments {
            let call = calls.entry(fragment.index).or_insert_with(|| {
                ChatCompletionMessageFunctionToolCall {
                    id: String::new(),
                    r#type: "function".to_string(),
                    function: ChatCompletionToolCallFunction {
                        name: String::new(),
                        arguments: String::new(),
                    },
                }
            });
            if let Some(id) = &fragment.id {
                call.id = id.clone();
            }
            if let Some(function) = &fragment.function {
                if let Some(name) = &function.name {
                    call.function.name = name.clone();
                }
                if let Some(arguments) = &function.arguments {
                    call.function.arguments.push_str(arguments);
                }
            }
        }
    }
    (!calls.is_empty()).then(|| calls.into_values().collect())
}

fn object_kind(output: &serde_json::Map<String, serde_json::Value>) -> Option<&str> {
    output.get("object").and_then(|v| v.as_str())
}

fn from_object<T>(output: serde_json::Map<String, serde_json::Value>) -> Result<T>
where
    T: serde::de::DeserializeOwned,
{
    serde_json::from_value(serde_json::Value::Object(output))
        .map_err(|e| MunaError::Prediction(e.to_string()))
}

#[cfg(test)]
mod tests;
