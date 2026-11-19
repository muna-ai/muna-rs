/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::collections::HashMap;
use std::pin::Pin;

use async_trait::async_trait;
use futures_core::Stream;
use futures_util::StreamExt;

use crate::client::{MunaError, Result};
use crate::types::{self, Dtype, Parameter, Prediction, Signature, Value};

use super::chat::ChatInputs;
use super::completions::{decode_base64_image, ChatCompletionChunk, ChatCompletionReasoningEffort};
use super::{
    utils::{
        decode, output, parameter, require, required, result, validate_inputs, FLOATS, INTEGERS,
    },
    Adapter, PredictionStream,
};

mod schema;
pub use schema::*;
pub type MessageEventStream = Pin<Box<dyn Stream<Item = Result<RawMessageStreamEvent>> + Send>>;
type OutputStream =
    Pin<Box<dyn Stream<Item = Result<serde_json::Map<String, serde_json::Value>>> + Send>>;
type ChunkStream = Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk>> + Send>>;

#[derive(Debug, Clone, Default)]
pub struct AnthropicMessagesOptions;

#[derive(Debug, Clone)]
pub struct AnthropicMessagesAdapter {
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
    pub top_k: Option<Parameter>,
}

impl AnthropicMessagesAdapter {
    /// Returns `None` when the signature is incompatible or ambiguous.
    pub fn from_signature(signature: &Signature) -> Option<Self> {
        // Required conversation input.
        let messages = parameter(
            &required(&signature.inputs),
            &[Dtype::List],
            Some("openai.chat.completions.messages"),
        )
        .ok()??;
        // Images parameter.
        let images = parameter(
            &signature.inputs,
            &[Dtype::ImageList],
            Some("openai.chat.completions.images"),
        )
        .ok()?;
        // Audios parameter.
        let audios = parameter(
            &signature.inputs,
            &[Dtype::ArrayList],
            Some("openai.chat.completions.audios"),
        )
        .ok()?;
        // Temperature parameter.
        let temperature = parameter(
            &signature.inputs,
            FLOATS,
            Some("openai.chat.completions.temperature"),
        )
        .ok()?;
        // Top p parameter.
        let top_p = parameter(
            &signature.inputs,
            FLOATS,
            Some("openai.chat.completions.top_p"),
        )
        .ok()?;
        // Seed parameter.
        let seed = parameter(
            &signature.inputs,
            INTEGERS,
            Some("openai.chat.completions.seed"),
        )
        .ok()?;
        // Max completion tokens parameter.
        let max_completion_tokens = parameter(
            &signature.inputs,
            INTEGERS,
            Some("openai.chat.completions.max_output_tokens"),
        )
        .ok()?;
        // Frequency penalty parameter.
        let frequency_penalty = parameter(
            &signature.inputs,
            FLOATS,
            Some("openai.chat.completions.frequency_penalty"),
        )
        .ok()?;
        // Presence penalty parameter.
        let presence_penalty = parameter(
            &signature.inputs,
            FLOATS,
            Some("openai.chat.completions.presence_penalty"),
        )
        .ok()?;
        // Response format parameter.
        let response_format = parameter(
            &signature.inputs,
            &[Dtype::Dict],
            Some("openai.chat.completions.response_format"),
        )
        .ok()?;
        // Reasoning effort parameter.
        let reasoning_effort = parameter(
            &signature.inputs,
            &[Dtype::String],
            Some("openai.chat.completions.reasoning_effort"),
        )
        .ok()?;
        // Tools parameter.
        let tools = parameter(
            &signature.inputs,
            &[Dtype::List],
            Some("openai.chat.completions.tools"),
        )
        .ok()?;
        // Stop parameter.
        let stop = parameter(
            &signature.inputs,
            &[Dtype::List],
            Some("openai.chat.completions.stop"),
        )
        .ok()?;
        // Top k parameter.
        let top_k = parameter(
            &signature.inputs,
            INTEGERS,
            Some("anthropic.messages.top_k"),
        )
        .ok()?;
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
                top_k.as_ref(),
            ],
        )
        .ok()?;
        if audios
            .as_ref()
            .is_some_and(|parameter| parameter.sample_rate.is_none_or(|rate| rate == 0))
        {
            return None;
        }
        // Predictors yield the same chunk schema for both API dialects.
        let output = output(signature, &[Dtype::Dict], "ChatCompletionChunk").ok()??;
        Some(Self {
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
            top_k,
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
        if let (Some(parameter), Some(value)) = (&self.top_k, input.top_k) {
            inputs.insert(parameter.name.clone(), Value::Int(value));
        }
        Ok(inputs)
    }

    pub fn inputs(&self, params: &MessageCreateParams) -> Result<HashMap<String, Value>> {
        if params.messages.is_empty() {
            return Err(MunaError::InvalidInput(
                "messages must contain at least one message".into(),
            ));
        }
        let mut conversation = Vec::new();
        let mut images = Vec::new();
        if let Some(system) = &params.system {
            if matches!(system, MessageContent::Blocks(blocks) if blocks.iter().any(|block| !matches!(block, ContentBlockParam::Text { .. })))
            {
                return Err(MunaError::InvalidInput(
                    "System content must contain only text blocks".into(),
                ));
            }
            if let Some(system) = strip_billing_header(system) {
                conversation.push(serde_json::json!({"role":"system", "content":system.flatten()}));
            }
        }
        for (index, message) in params.messages.iter().enumerate() {
            if !matches!(message.role.as_str(), "user" | "assistant") {
                return Err(MunaError::InvalidInput(
                    "Anthropic messages require user or assistant roles".into(),
                ));
            }
            validate_message(index, message)?;
            let MessageContent::Blocks(blocks) = &message.content else {
                conversation.push(
                    serde_json::json!({"role":message.role, "content":message.content.flatten()}),
                );
                continue;
            };
            let mut parts = Vec::new();
            let mut calls = Vec::new();
            let mut thinking = String::new();
            let mut results = Vec::new();
            // Tool results precede the user's text, so their images must also
            // precede the images referenced by the following user message.
            let mut blocks = blocks.iter().collect::<Vec<_>>();
            blocks.sort_by_key(|block| !matches!(block, ContentBlockParam::ToolResult { .. }));
            for block in blocks {
                match block {
                    ContentBlockParam::Text { text } => {
                        parts.push(serde_json::json!({"type":"text", "text":text}))
                    }
                    ContentBlockParam::Image { source } => {
                        parts.push(self.image_part(source, &mut images)?)
                    }
                    ContentBlockParam::Thinking {
                        thinking: fragment, ..
                    } if message.role == "assistant" => thinking.push_str(fragment),
                    ContentBlockParam::ToolUse { id, name, input }
                        if message.role == "assistant" =>
                    {
                        calls.push(serde_json::json!({"id":id, "type":"function", "function":{"name":name, "arguments":input.to_string()}}));
                    }
                    ContentBlockParam::ToolResult {
                        tool_use_id,
                        content,
                        is_error,
                    } if message.role == "user" => {
                        let mut content = match content.as_deref() {
                            None => serde_json::json!(""),
                            Some(MessageContent::Text(text)) => serde_json::json!(text),
                            Some(MessageContent::Blocks(blocks)) => {
                                let mut parts = Vec::new();
                                for block in blocks {
                                    parts.push(match block {
                                        ContentBlockParam::Text { text } => {
                                            serde_json::json!({"type":"text", "text":text})
                                        }
                                        ContentBlockParam::Image { source } => {
                                            self.image_part(source, &mut images)?
                                        }
                                        _ => {
                                            return Err(MunaError::InvalidInput(
                                                "Unsupported tool-result content".into(),
                                            ))
                                        }
                                    });
                                }
                                serde_json::Value::Array(parts)
                            }
                        };
                        if *is_error == Some(true) {
                            match &mut content {
                                serde_json::Value::String(text) => *text = format!("Error: {text}"),
                                serde_json::Value::Array(parts) => parts
                                    .insert(0, serde_json::json!({"type":"text", "text":"Error:"})),
                                _ => {}
                            }
                        }
                        results.push(serde_json::json!({"role":"tool", "tool_call_id":tool_use_id, "content":content}));
                    }
                    _ => {
                        return Err(MunaError::InvalidInput(
                            "Unsupported Anthropic content block for this role".into(),
                        ))
                    }
                }
            }
            let only_results =
                !results.is_empty() && parts.is_empty() && calls.is_empty() && thinking.is_empty();
            conversation.extend(results);
            if !only_results {
                let content = if parts.iter().all(|part| part["type"] == "text") {
                    serde_json::Value::String(
                        parts
                            .iter()
                            .filter_map(|part| part["text"].as_str())
                            .collect::<Vec<_>>()
                            .join("\n"),
                    )
                } else {
                    serde_json::Value::Array(parts)
                };
                let mut wire = serde_json::json!({"role":message.role, "content":content});
                if !calls.is_empty() {
                    wire["tool_calls"] = serde_json::Value::Array(calls);
                }
                if !thinking.is_empty() {
                    wire["reasoning_content"] = serde_json::Value::String(thinking);
                }
                conversation.push(wire);
            }
        }
        let mut inputs = self.bind_inputs(ChatInputs {
            messages: conversation,
            tools: params
                .tools
                .as_deref()
                .filter(|tools| !tools.is_empty())
                .map(to_openai_tools),
            reasoning_effort: params
                .reasoning_effort()
                .map(|effort| effort.as_str().to_string()),
            max_output_tokens: Some(params.max_tokens),
            temperature: params.temperature,
            top_p: params.top_p,
            stop_sequences: params.stop_sequences.clone(),
            top_k: params.top_k,
            ..Default::default()
        })?;
        if !images.is_empty() {
            let parameter = require(
                self.images.as_ref(),
                "This predictor does not support image content",
            )?;
            inputs.insert(
                parameter.name.clone(),
                Value::Remote(crate::types::RemoteValue {
                    dtype: Dtype::ImageList,
                    data: Some(crate::types::RemoteData::Sequence(images)),
                }),
            );
        }
        Ok(inputs)
    }

    fn image_part(
        &self,
        source: &ImageSource,
        images: &mut Vec<crate::types::RemoteValue>,
    ) -> Result<serde_json::Value> {
        require(
            self.images.as_ref(),
            "This predictor does not support image content",
        )?;
        let url = match source {
            ImageSource::Base64 { media_type, data } => format!("data:{media_type};base64,{data}"),
            ImageSource::Url { url } => url.clone(),
            ImageSource::Unsupported => {
                return Err(MunaError::InvalidInput(
                    "Unsupported Anthropic image source".into(),
                ))
            }
        };
        images.push(crate::types::RemoteValue {
            dtype: Dtype::Image,
            data: Some(url.into()),
        });
        Ok(serde_json::json!({"type":"image"}))
    }
}

#[async_trait]
impl Adapter for AnthropicMessagesAdapter {
    type Source = PredictionStream;
    type Options = AnthropicMessagesOptions;
    type Response = MessageEventStream;

    fn kind(&self) -> &'static str {
        "anthropic.messages.create"
    }

    async fn adapt(
        &self,
        predictions: PredictionStream,
        _options: Option<Self::Options>,
    ) -> Result<MessageEventStream> {
        Ok(events_from_chunks(parse_completion_chunks(
            gather_prediction_outputs(predictions, self.output),
        )))
    }
}
impl MessageCreateParams {
    /// Reasoning effort the predictor receives, derived from Anthropic's
    /// `thinking` and `output_config.effort`:
    /// - `thinking.type == "disabled"`            -> `none`
    /// - effort given (thinking absent/enabled)   -> that effort, `max` -> `xhigh`
    /// - thinking enabled/adaptive, no effort     -> `medium`
    /// - neither                                  -> `None` (predictor default)
    pub fn reasoning_effort(&self) -> Option<ChatCompletionReasoningEffort> {
        use ChatCompletionReasoningEffort as E;
        if matches!(self.thinking, Some(ThinkingConfig::Disabled)) {
            return Some(E::None);
        }
        let effort = self
            .output_config
            .as_ref()
            .and_then(|config| config.effort)
            .map(|effort| match effort {
                OutputEffort::Low => E::Low,
                OutputEffort::Medium => E::Medium,
                OutputEffort::High => E::High,
                OutputEffort::XHigh => E::XHigh,
                OutputEffort::Max => E::XHigh,
            });
        match (effort, &self.thinking) {
            (Some(effort), _) => Some(effort),
            (None, Some(_)) => Some(E::Medium),
            (None, None) => None,
        }
    }

    /// Inputs the chat predictor receives for this request: the system
    /// prompt folded into OpenAI-shaped messages, tools translated to
    /// OpenAI function tools (`None` when absent or empty), and sampling
    /// knobs under their shared denotations (`max_tokens` is the
    /// `max_output_tokens` knob). Anthropic-specific knobs (stop sequences,
    /// top-k) keep their own denotations: denotations describe input meaning
    /// and are orthogonal to the output contract, so an OpenAI-shaped
    /// predictor can declare them for this surface. Chat predictors speak
    /// the OpenAI contract, so this is the whole Anthropic adaptation of the
    /// request; servers and the control plane's router-hash path both call it.
    ///
    /// Image blocks must have `base64` sources: they are decoded here into
    /// `images`, in block order, without any network access. Decoding is
    /// CPU-bound (tens of milliseconds per large image), so async callers
    /// should run this off the executor.
    ///
    /// Fails with `InvalidInput` when a message carries nothing for the model
    /// to read (see `validate_message`) or content the chat surface cannot
    /// translate (see `validate_blocks`).
    pub fn chat_inputs(&self) -> Result<ChatInputs> {
        if self.messages.is_empty() {
            return Err(MunaError::InvalidInput(
                "`messages` must contain at least one message.".into(),
            ));
        }
        if let Some(MessageContent::Blocks(blocks)) = &self.system {
            if blocks
                .iter()
                .any(|block| matches!(block, ContentBlockParam::Image { .. }))
            {
                return Err(MunaError::InvalidInput(
                    "`system` content cannot contain `image` blocks.".into(),
                ));
            }
        }
        for (index, message) in self.messages.iter().enumerate() {
            validate_blocks(index, message)?;
            validate_message(index, message)?;
        }
        let images = decode_images(&self.messages)?;
        Ok(ChatInputs {
            messages: to_openai_messages(self.system.as_ref(), &self.messages),
            images: (!images.is_empty()).then_some(images),
            tools: self
                .tools
                .as_deref()
                .filter(|tools| !tools.is_empty())
                .map(to_openai_tools),
            response_format: None,
            reasoning_effort: self.reasoning_effort().map(|e| e.as_str().to_string()),
            max_output_tokens: Some(self.max_tokens),
            temperature: self.temperature,
            top_p: self.top_p,
            seed: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop_sequences: self.stop_sequences.clone(),
            top_k: self.top_k,
        })
    }
}

/// Reject content blocks the chat surface cannot translate. Dropping them
/// silently would have the model answer without content the user sent.
///
/// Images are accepted only where `translate_message_for_openai` emits an
/// image placeholder for them: top-level blocks of user and assistant
/// turns. System turns and tool results are flattened to text, so an image
/// there would decode without a placeholder to bind to.
fn validate_blocks(index: usize, message: &MessageParam) -> Result<()> {
    let MessageContent::Blocks(blocks) = &message.content else {
        return Ok(());
    };
    for (block_index, block) in blocks.iter().enumerate() {
        let location = format!("`messages[{index}].content[{block_index}]`");
        let kind = match block {
            ContentBlockParam::Image { .. } if message.role == "system" => {
                return Err(MunaError::InvalidInput(format!(
                    "{location} is an `image` block in a `system` message, which \
                    is not supported."
                )));
            }
            ContentBlockParam::Image { source } => {
                validate_image_source(&location, source)?;
                continue;
            }
            ContentBlockParam::ToolResult {
                content: Some(content),
                ..
            } => {
                if let MessageContent::Blocks(inner) = content.as_ref() {
                    if inner
                        .iter()
                        .any(|block| matches!(block, ContentBlockParam::Image { .. }))
                    {
                        return Err(MunaError::InvalidInput(format!(
                            "{location} is a `tool_result` block with `image` \
                            content, which is not supported."
                        )));
                    }
                }
                continue;
            }
            ContentBlockParam::Document => "a `document` block",
            ContentBlockParam::Unsupported => "a block of an unrecognized type",
            _ => continue,
        };
        return Err(MunaError::InvalidInput(format!(
            "{location} is {kind}, which is not supported. Supported block types \
            are `text`, `image`, `tool_use`, `tool_result`, `thinking`, and \
            `redacted_thinking`."
        )));
    }
    Ok(())
}

/// Accept only inline `base64` image sources: the client never fetches.
fn validate_image_source(location: &str, source: &ImageSource) -> Result<()> {
    let kind = match source {
        ImageSource::Base64 { .. } => return Ok(()),
        ImageSource::Url { .. } => "a `url` source",
        ImageSource::Unsupported => "an unsupported source",
    };
    Err(MunaError::InvalidInput(format!(
        "{location} is an `image` block with {kind}. Only `base64` image sources \
        are supported; fetch remote images and pass them inline as `base64`."
    )))
}

/// Decode the conversation's image blocks in the order
/// `translate_message_for_openai` emits their placeholders. Assumes
/// `validate_blocks` passed, so every image is a top-level `base64` block
/// of a non-system turn.
fn decode_images(messages: &[MessageParam]) -> Result<Vec<types::Image>> {
    messages
        .iter()
        .filter_map(|message| match &message.content {
            MessageContent::Blocks(blocks) => Some(blocks.iter()),
            MessageContent::Text(_) => None,
        })
        .flatten()
        .filter_map(|block| match block {
            ContentBlockParam::Image {
                source: ImageSource::Base64 { media_type, data },
            } => Some(decode_base64_image(data, media_type)),
            _ => None,
        })
        .collect()
}

/// Reject a message that carries nothing for the model to read.
///
/// Only user turns are checked: a whitespace-only user turn renders as an
/// empty turn once the chat template trims it, and the model answers a
/// question it was never asked. An empty assistant turn is history the model
/// wrote (a stream cut off early, a reasoning-only reply) and renders
/// harmlessly; refusing it would wedge a conversation on our own output.
fn validate_message(index: usize, message: &MessageParam) -> Result<()> {
    if message.role == "assistant" {
        return Ok(());
    }
    // An inline system turn that is nothing but the billing header is
    // dropped by `to_openai_messages`, so it is not an empty message.
    let stripped;
    let content = if message.role == "system" {
        match strip_billing_header(&message.content) {
            Some(content) => {
                stripped = content;
                &stripped
            }
            None => return Ok(()),
        }
    } else {
        &message.content
    };
    let has_payload = match content {
        MessageContent::Text(text) => !text.trim().is_empty(),
        MessageContent::Blocks(blocks) => blocks.iter().any(|block| match block {
            ContentBlockParam::Text { text } => !text.trim().is_empty(),
            ContentBlockParam::ToolUse { .. } | ContentBlockParam::ToolResult { .. } => true,
            ContentBlockParam::Thinking { .. } | ContentBlockParam::RedactedThinking { .. } => {
                false
            }
            ContentBlockParam::Image { .. }
            | ContentBlockParam::Document
            | ContentBlockParam::Unsupported => true,
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

/// Prefix of the text block Claude Code prepends to its system prompt. It
/// carries a per-request hash, so left in place it changes the prompt on
/// every turn and defeats prefix caching (and, for us, KV page-hash
/// routing). Mirrors upstream vLLM
/// `vllm/entrypoints/anthropic/serving.py:236-272`.
const BILLING_HEADER_PREFIX: &str = "x-anthropic-billing-header";

/// Drop billing-header text blocks from system content. Returns `None`
/// when nothing readable remains, so the caller emits no message at all.
fn strip_billing_header(content: &MessageContent) -> Option<MessageContent> {
    match content {
        MessageContent::Text(text) => {
            (!text.starts_with(BILLING_HEADER_PREFIX)).then(|| content.clone())
        }
        MessageContent::Blocks(blocks) => {
            let kept: Vec<ContentBlockParam> = blocks
                .iter()
                .filter(|block| {
                    !matches!(block, ContentBlockParam::Text { text } if text.starts_with(BILLING_HEADER_PREFIX))
                })
                .cloned()
                .collect();
            (!kept.is_empty()).then_some(MessageContent::Blocks(kept))
        }
    }
}

/// Anthropic request -> OpenAI-shaped messages, exactly as the chat
/// predictor receives them. The system prompt is folded in as a leading
/// `system` message (predictors that need it separately can filter it out).
///
/// Billing-header blocks are stripped from the system prompt and from
/// inline `system` turns (see `strip_billing_header`). Inline `system`
/// turns otherwise keep their role and position: whether a template
/// accepts a non-leading system message is the compiled predictor's call
/// (its `apply_chat_template` probes the template and folds only where
/// needed), and deciding here would make the plane's router-hash render
/// and the node's disagree.
fn to_openai_messages(
    system: Option<&MessageContent>,
    messages: &[MessageParam],
) -> Vec<serde_json::Value> {
    let mut result = Vec::with_capacity(messages.len() + usize::from(system.is_some()));
    if let Some(system) = system.and_then(strip_billing_header) {
        result.push(serde_json::json!({
            "role": "system",
            "content": system.flatten(),
        }));
    }
    for message in messages {
        if message.role == "system" {
            if let Some(content) = strip_billing_header(&message.content) {
                result.push(serde_json::json!({
                    "role": "system",
                    "content": content.flatten(),
                }));
            }
            continue;
        }
        translate_message_for_openai(message, &mut result);
    }
    result
}

/// Anthropic tool definitions -> OpenAI function tools, exactly as the chat
/// predictor's tools input receives them.
fn to_openai_tools(tools: &[Tool]) -> Vec<serde_json::Value> {
    tools.iter().map(openai_tool).collect()
}

/// Convert an Anthropic tool definition into the OpenAI function tool
/// shape expected by the delegate's tools input parameter.
fn openai_tool(tool: &Tool) -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        }
    })
}

/// Content block that stays on the message's own role.
enum RunPart<'a> {
    Text(&'a str),
    Image,
}

/// Translate one Anthropic input message into OpenAI-shaped messages.
/// Text blocks stay on the original role (flattened, joined by newline);
/// image blocks stay on it too, as payload-free `{"type": "image"}` parts
/// (the nth placeholder across the conversation indexes the nth decoded
/// image), which keeps a run with images as a content-part list. Replayed
/// `tool_use` blocks become assistant `tool_calls`; `tool_result` blocks
/// become `tool` role messages bound by `tool_call_id`. Block order is
/// preserved by flushing the pending run before each tool block.
fn translate_message_for_openai(message: &MessageParam, messages: &mut Vec<serde_json::Value>) {
    let MessageContent::Blocks(blocks) = &message.content else {
        messages.push(serde_json::json!({
            "role": message.role,
            "content": message.content.flatten(),
        }));
        return;
    };
    let mut run: Vec<RunPart> = Vec::new();
    let flush = |run: &mut Vec<RunPart>, messages: &mut Vec<serde_json::Value>| {
        if run.is_empty() {
            return;
        }
        let content = if run.iter().any(|part| matches!(part, RunPart::Image)) {
            serde_json::Value::Array(
                run.iter()
                    .map(|part| match part {
                        RunPart::Text(text) => serde_json::json!({ "type": "text", "text": text }),
                        RunPart::Image => serde_json::json!({ "type": "image" }),
                    })
                    .collect(),
            )
        } else {
            let texts: Vec<&str> = run
                .iter()
                .filter_map(|part| match part {
                    RunPart::Text(text) => Some(*text),
                    RunPart::Image => None,
                })
                .collect();
            serde_json::Value::String(texts.join("\n"))
        };
        messages.push(serde_json::json!({
            "role": message.role,
            "content": content,
        }));
        run.clear();
    };
    for block in blocks {
        match block {
            ContentBlockParam::Text { text } => {
                run.push(RunPart::Text(text.as_str()));
            }
            ContentBlockParam::Image { .. } => {
                run.push(RunPart::Image);
            }
            ContentBlockParam::ToolUse { id, name, input } => {
                flush(&mut run, messages);
                messages.push(serde_json::json!({
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": id,
                        "type": "function",
                        "function": { "name": name, "arguments": input.to_string() },
                    }],
                }));
            }
            ContentBlockParam::ToolResult {
                tool_use_id,
                content,
                ..
            } => {
                flush(&mut run, messages);
                messages.push(serde_json::json!({
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "content": content.as_ref().map(|c| c.flatten()),
                }));
            }
            ContentBlockParam::Thinking { .. }
            | ContentBlockParam::RedactedThinking { .. }
            | ContentBlockParam::Document
            | ContentBlockParam::Unsupported => {}
        }
    }
    flush(&mut run, messages);
}

/// Stream the dict output at `output_param_idx` from each prediction.
fn gather_prediction_outputs(
    mut predictions: Pin<Box<dyn Stream<Item = Result<Prediction>> + Send>>,
    output_param_idx: usize,
) -> OutputStream {
    Box::pin(async_stream::try_stream! {
        while let Some(prediction) = predictions.next().await {
            let prediction = prediction?;
            match decode(result(&prediction, output_param_idx)?, Dtype::Dict).await? {
                Value::Dict(output) => yield output,
                _ => Err(MunaError::Prediction("Message output must be a dictionary".into()))?,
            }
        }
    })
}

/// Parse each dict output as a chat completion chunk,
/// converting full chat completions into single chunks.
fn parse_completion_chunks(mut outputs: OutputStream) -> ChunkStream {
    Box::pin(async_stream::try_stream! {
        while let Some(output) = outputs.next().await {
            yield parse_completion_chunk(output?)?;
        }
    })
}

fn parse_completion_chunk(
    output: serde_json::Map<String, serde_json::Value>,
) -> Result<ChatCompletionChunk> {
    match output.get("object").and_then(|v| v.as_str()) {
        Some("chat.completion.chunk") => from_object(output),
        _ => Err(MunaError::Prediction(
            "Failed to parse chat completion chunk from model output. \
            Chat predictors must yield `ChatCompletionChunk` outputs."
                .into(),
        )),
    }
}

#[derive(Default)]
struct ToolCallState {
    block: Option<usize>,
    id: String,
    name: String,
    arguments: Vec<String>,
}

/// Translate predictor deltas without collecting the generated text. Tool
/// headers can arrive after their arguments, and tool calls may interleave.
fn events_from_chunks(mut chunks: ChunkStream) -> MessageEventStream {
    Box::pin(async_stream::try_stream! {
        let mut started = false;
        let mut next_block = 0usize;
        let mut text_block: Option<(usize, bool)> = None;
        let mut tools = std::collections::BTreeMap::<usize, ToolCallState>::new();
        let mut stop_reason = StopReason::EndTurn;
        let mut input_tokens = 0u64;
        let mut usage = Usage::default();
        while let Some(chunk) = chunks.next().await {
            let chunk = chunk?;
            if chunk.choices.len() > 1 || chunk.choices.iter().any(|choice| choice.index != 0) {
                Err(MunaError::Prediction("Anthropic messages require a single completion choice".into()))?;
            }
            if !started {
                yield RawMessageStreamEvent::MessageStart {
                    message: Message {
                        id: chunk.id.clone(), r#type: "message".into(), role: "assistant".into(),
                        content: Vec::new(), model: chunk.model.clone(), stop_reason: None,
                        stop_sequence: None, usage: Usage { input_tokens: Some(0), ..Default::default() }
                    }
                };
                started = true;
            }
            if let Some(delta) = chunk.usage {
                input_tokens += delta.prompt_tokens;
                usage.output_tokens += delta.completion_tokens;
                if let Some(details) = delta.prompt_tokens_details {
                    if details.cached_tokens.is_some() { usage.cache_read_input_tokens = details.cached_tokens; }
                    if details.cache_write_tokens.is_some() { usage.cache_creation_input_tokens = details.cache_write_tokens; }
                }
            }
            let Some(choice) = chunk.choices.into_iter().next() else { continue };
            if let Some(reason) = choice.finish_reason { stop_reason = map_finish_reason(&reason); }
            let Some(delta) = choice.delta else { continue };
            for (thinking, fragment) in [(true, delta.reasoning_content), (false, delta.content)] {
                let Some(fragment) = fragment.filter(|text| !text.is_empty()) else { continue };
                if text_block.is_none_or(|(_, current)| current != thinking) {
                    if let Some((index, _)) = text_block.take() {
                        yield RawMessageStreamEvent::ContentBlockStop { index };
                    }
                    let index = next_block;
                    next_block += 1;
                    text_block = Some((index, thinking));
                    yield RawMessageStreamEvent::ContentBlockStart {
                        index,
                        content_block: if thinking {
                            ContentBlock::Thinking { thinking: String::new(), signature: String::new() }
                        } else { ContentBlock::Text { text: String::new() } }
                    };
                }
                let index = text_block.unwrap().0;
                yield RawMessageStreamEvent::ContentBlockDelta {
                    index,
                    delta: if thinking { ContentBlockDelta::ThinkingDelta { thinking: fragment } }
                        else { ContentBlockDelta::TextDelta { text: fragment } }
                };
            }
            for call in delta.tool_calls.unwrap_or_default() {
                if let Some((index, _)) = text_block.take() {
                    yield RawMessageStreamEvent::ContentBlockStop { index };
                }
                let state = tools.entry(call.index).or_default();
                if let Some(id) = call.id { state.id.push_str(&id); }
                if let Some(function) = call.function {
                    if let Some(name) = function.name { state.name.push_str(&name); }
                    if let Some(arguments) = function.arguments { state.arguments.push(arguments); }
                }
                if state.block.is_none() && !state.id.is_empty() && !state.name.is_empty() {
                    let index = next_block;
                    next_block += 1;
                    state.block = Some(index);
                    yield RawMessageStreamEvent::ContentBlockStart {
                        index,
                        content_block: ContentBlock::ToolUse {
                            id: state.id.clone(), name: state.name.clone(), input: serde_json::json!({})
                        }
                    };
                }
                if let Some(index) = state.block {
                    for partial_json in std::mem::take(&mut state.arguments) {
                        if !partial_json.is_empty() {
                            yield RawMessageStreamEvent::ContentBlockDelta {
                                index, delta: ContentBlockDelta::InputJsonDelta { partial_json }
                            };
                        }
                    }
                }
            }
        }
        if !started { Err(MunaError::Prediction("Message predictor did not return any chunks".into()))?; }
        if let Some((index, _)) = text_block {
            yield RawMessageStreamEvent::ContentBlockStop { index };
        }
        for state in tools.into_values() {
            let index = state.block.ok_or_else(|| MunaError::Prediction("Tool call did not provide an id and name".into()))?;
            yield RawMessageStreamEvent::ContentBlockStop { index };
        }
        // Token counts are deltas; cache details are final snapshots.
        usage.input_tokens = Some(input_tokens.saturating_sub(
            usage.cache_read_input_tokens.unwrap_or(0) + usage.cache_creation_input_tokens.unwrap_or(0)
        ));
        yield RawMessageStreamEvent::MessageDelta {
            delta: MessageDelta { stop_reason: Some(stop_reason), stop_sequence: None }, usage
        };
        yield RawMessageStreamEvent::MessageStop;
    })
}

/// Accumulate a raw message stream event into the final message.
pub(crate) fn apply_event(message: &mut Option<Message>, event: &RawMessageStreamEvent) {
    match event {
        RawMessageStreamEvent::MessageStart { message: started } => {
            *message = Some(started.clone());
        }
        RawMessageStreamEvent::ContentBlockStart { content_block, .. } => {
            if let Some(message) = message {
                message.content.push(content_block.clone());
            }
        }
        RawMessageStreamEvent::ContentBlockDelta { index, delta } => {
            if let Some(message) = message {
                if let Some(block) = message.content.get_mut(*index) {
                    match (block, delta) {
                        (
                            ContentBlock::Text { text },
                            ContentBlockDelta::TextDelta { text: fragment },
                        ) => text.push_str(fragment),
                        (
                            ContentBlock::Thinking { thinking, .. },
                            ContentBlockDelta::ThinkingDelta { thinking: fragment },
                        ) => thinking.push_str(fragment),
                        // Input JSON fragments buffer as a string on the
                        // block's `input`; the buffer is parsed into the
                        // final input object when the block stops.
                        (
                            ContentBlock::ToolUse { input, .. },
                            ContentBlockDelta::InputJsonDelta { partial_json },
                        ) => match input {
                            serde_json::Value::String(buffer) => buffer.push_str(partial_json),
                            _ => *input = serde_json::Value::String(partial_json.clone()),
                        },
                        _ => {}
                    }
                }
            }
        }
        RawMessageStreamEvent::ContentBlockStop { index } => {
            if let Some(message) = message {
                if let Some(ContentBlock::ToolUse { input, .. }) = message.content.get_mut(*index) {
                    if let serde_json::Value::String(buffer) = input {
                        *input =
                            serde_json::from_str(buffer).unwrap_or_else(|_| serde_json::json!({}));
                    }
                }
            }
        }
        RawMessageStreamEvent::MessageDelta { delta, usage } => {
            if let Some(message) = message {
                message.stop_reason = delta.stop_reason;
                message.stop_sequence = delta.stop_sequence.clone();
                if usage.input_tokens.is_some() {
                    message.usage.input_tokens = usage.input_tokens;
                }
                message.usage.output_tokens = usage.output_tokens;
                message.usage.cache_creation_input_tokens = usage.cache_creation_input_tokens;
                message.usage.cache_read_input_tokens = usage.cache_read_input_tokens;
            }
        }
        _ => {}
    }
}

fn map_finish_reason(finish_reason: &str) -> StopReason {
    match finish_reason {
        "length" => StopReason::MaxTokens,
        "content_filter" => StopReason::Refusal,
        "tool_calls" => StopReason::ToolUse,
        _ => StopReason::EndTurn,
    }
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
