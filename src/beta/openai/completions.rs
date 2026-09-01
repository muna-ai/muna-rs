/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::collections::{BTreeMap, HashMap};
use std::pin::Pin;
use std::sync::Arc;
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use futures_core::Stream;
use futures_util::StreamExt;
use serde_json::json;
use tokio::sync::RwLock;

use crate::beta::utils::get_parameter;
use crate::c;
use crate::client::{Client, Result};
use crate::MunaError;
use crate::services::{PredictionService, PredictorService};
use crate::types::{self, Acceleration, Dtype, Parameter, Prediction, Value};

use super::schema::{
    ChatCompletion, ChatCompletionChoice, ChatCompletionChunk,
    ChatCompletionChunkChoice, ChatCompletionContent,
    ChatCompletionContentPart, ChatCompletionContentPartInputAudio,
    ChatCompletionCreateParams, ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall, ChatCompletionToolCallFunction,
    ChatCompletionToolChoice, ChatCompletionUsage,
};

/// Stream of chat completion chunks.
pub type ChatCompletionStream = Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk>> + Send>>;

/// Cached predictor metadata for fast chat completion creation.
#[derive(Clone)]
struct DelegateInfo {
    input_param_name: String,
    response_format_param_name: Option<String>,
    reasoning_effort_param_name: Option<String>,
    max_output_tokens_param_name: Option<String>,
    temperature_param_name: Option<String>,
    top_p_param_name: Option<String>,
    frequency_penalty_param_name: Option<String>,
    presence_penalty_param_name: Option<String>,
    images_param_name: Option<String>,
    audios_param_name: Option<String>,
    /// Declared PCM sample rate of the audios parameter; all decoded
    /// audio content parts are resampled to it.
    audio_sample_rate: Option<u32>,
    tools_param_name: Option<String>,
    completion_param_idx: usize,
}

/// Create chat completions.
#[derive(Clone)]
pub struct ChatCompletionService {
    client: Arc<dyn Client>,
    predictors: PredictorService,
    predictions: PredictionService,
    cache: Arc<RwLock<HashMap<String, DelegateInfo>>>,
}

impl ChatCompletionService {

    pub fn new(
        client: Arc<dyn Client>,
        predictors: PredictorService,
        predictions: PredictionService
    ) -> Self {
        Self {
            client,
            predictors,
            predictions,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a chat completion.
    pub async fn create(&self, params: ChatCompletionCreateParams) -> Result<ChatCompletion> {
        let model = params.model.clone();
        let (
            input_map,
            completion_param_idx,
            acceleration
        ) = self.prepare_prediction(params).await?;
        let mut prediction_stream = self
            .predictions
            .stream(&model, input_map, Some(acceleration))
            .await?;
        let mut chunks = Vec::new();
        while let Some(prediction) = prediction_stream.next().await {
            let output = gather_completion_output(prediction?, completion_param_idx, &model)?;
            chunks.push(parse_chat_completion_chunk(output)?);
        }
        merge_chunks(chunks)
    }

    /// Stream a chat completion.
    pub async fn stream(&self, params: ChatCompletionCreateParams) -> Result<ChatCompletionStream> {
        let model = params.model.clone();
        let (
            input_map,
            completion_param_idx,
            acceleration
        ) = self.prepare_prediction(params).await?;
        let mut prediction_stream = self
            .predictions
            .stream(&model, input_map, Some(acceleration))
            .await?;
        let stream_model = model.clone();
        let stream = async_stream::try_stream! {
            while let Some(prediction) = prediction_stream.next().await {
                let output = gather_completion_output(
                    prediction?,
                    completion_param_idx,
                    &stream_model,
                )?;
                yield parse_chat_completion_chunk(output)?;
            }
        };
        Ok(Box::pin(stream))
    }

    async fn prepare_prediction(
        &self,
        params: ChatCompletionCreateParams,
    ) -> Result<(HashMap<String, Value>, usize, Acceleration)> {
        self.ensure_delegate_info(&params.model).await?;
        let info = {
            let cache = self.cache.read().await;
            cache.get(&params.model).cloned().ok_or_else(|| {
                MunaError::Prediction(format!(
                    "{} cannot be used with OpenAI chat completions API because \
                    the predictor metadata could not be cached.",
                    params.model
                ))
            })?
        };
        let mut input_map = HashMap::new();
        let conversation = normalize_conversation(
            self.client.as_ref(),
            &params.messages,
            &info
        ).await?;
        input_map.insert(info.input_param_name, Value::List(conversation.messages));
        if let (false, Some(name)) = (conversation.images.is_empty(), info.images_param_name) {
            input_map.insert(name, Value::ImageList(conversation.images));
        }
        if let (false, Some(name)) = (conversation.audios.is_empty(), info.audios_param_name) {
            input_map.insert(name, Value::ArrayList(conversation.audios));
        }
        if let Some(tools) = &params.tools {
            if !tools.is_empty() && params.tool_choice != Some(ChatCompletionToolChoice::None) {
                let Some(name) = info.tools_param_name else {
                    return Err(MunaError::InvalidInput(format!(
                        "{} does not support tool calling because it does not \
                        declare a tools input parameter.",
                        params.model
                    )));
                };
                let tools = tools
                    .iter()
                    .map(serde_json::to_value)
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(|e| MunaError::Prediction(e.to_string()))?;
                input_map.insert(name, Value::List(tools));
            }
        }
        if let (Some(value), Some(name)) = (params.response_format, info.response_format_param_name)
        {
            input_map.insert(name, Value::Dict(value));
        }
        if let (Some(value), Some(name)) =
            (params.reasoning_effort, info.reasoning_effort_param_name)
        {
            input_map.insert(name, Value::String(value.as_str().to_string()));
        }
        if let (Some(value), Some(name)) = (
            params.max_completion_tokens,
            info.max_output_tokens_param_name,
        ) {
            input_map.insert(name, Value::Int(value));
        }
        if let (Some(value), Some(name)) = (params.temperature, info.temperature_param_name) {
            input_map.insert(name, Value::Float(value));
        }
        if let (Some(value), Some(name)) = (params.top_p, info.top_p_param_name) {
            input_map.insert(name, Value::Float(value));
        }
        if let (Some(value), Some(name)) =
            (params.frequency_penalty, info.frequency_penalty_param_name)
        {
            input_map.insert(name, Value::Float(value));
        }
        if let (Some(value), Some(name)) =
            (params.presence_penalty, info.presence_penalty_param_name)
        {
            input_map.insert(name, Value::Float(value));
        }
        let acceleration = params.acceleration.unwrap_or(Acceleration::LocalAuto);
        Ok((input_map, info.completion_param_idx, acceleration))
    }

    async fn ensure_delegate_info(&self, tag: &str) -> Result<()> {
        {
            let cache = self.cache.read().await;
            if cache.contains_key(tag) {
                return Ok(());
            }
        }
        let info = self.create_delegate_info(tag).await?;
        self.cache
            .write()
            .await
            .entry(tag.to_string())
            .or_insert(info);
        Ok(())
    }

    async fn create_delegate_info(&self, tag: &str) -> Result<DelegateInfo> {
        let predictor = self.predictors.retrieve(tag).await?.ok_or_else(|| {
            MunaError::Prediction(format!(
                "{tag} cannot be used with OpenAI chat completions API because \
                the predictor could not be found. Check that your access key \
                is valid and that you have access to the predictor."
            ))
        })?;
        let signature = &predictor.signature;
        let required_inputs: Vec<&Parameter> = signature
            .inputs
            .iter()
            .filter(|p| !p.optional.unwrap_or(false))
            .collect();
        if required_inputs.len() != 1 {
            return Err(MunaError::Prediction(format!(
                "{tag} cannot be used with OpenAI chat completions API because \
                it has more than one required input parameter."
            )));
        }
        let input_param = required_inputs[0];
        if input_param.dtype != Some(Dtype::List) {
            return Err(MunaError::Prediction(format!(
                "{tag} cannot be used with OpenAI chat completions API because \
                it does not have a valid chat messages input parameter."
            )));
        }
        let float_dtypes = [Dtype::Float32, Dtype::Float64];
        let int_dtypes = [
            Dtype::Int8,
            Dtype::Int16,
            Dtype::Int32,
            Dtype::Int64,
            Dtype::Uint8,
            Dtype::Uint16,
            Dtype::Uint32,
            Dtype::Uint64,
        ];
        let response_format_param_name = get_parameter(
            &signature.inputs,
            &[Dtype::Dict],
            Some("openai.chat.completions.response_format"),
        )
        .1
        .map(|p| p.name.clone());
        let reasoning_effort_param_name = get_parameter(
            &signature.inputs,
            &[Dtype::String],
            Some("openai.chat.completions.reasoning_effort"),
        )
        .1
        .map(|p| p.name.clone());
        let max_output_tokens_param_name = get_parameter(
            &signature.inputs,
            &int_dtypes,
            Some("openai.chat.completions.max_output_tokens"),
        )
        .1
        .map(|p| p.name.clone());
        let temperature_param_name = get_parameter(
            &signature.inputs,
            &float_dtypes,
            Some("openai.chat.completions.temperature"),
        )
        .1
        .map(|p| p.name.clone());
        let top_p_param_name = get_parameter(
            &signature.inputs,
            &float_dtypes,
            Some("openai.chat.completions.top_p"),
        )
        .1
        .map(|p| p.name.clone());
        let frequency_penalty_param_name = get_parameter(
            &signature.inputs,
            &float_dtypes,
            Some("openai.chat.completions.frequency_penalty"),
        )
        .1
        .map(|p| p.name.clone());
        let presence_penalty_param_name = get_parameter(
            &signature.inputs,
            &float_dtypes,
            Some("openai.chat.completions.presence_penalty"),
        )
        .1
        .map(|p| p.name.clone());
        let images_param_name = get_parameter(
            &signature.inputs,
            &[Dtype::ImageList],
            Some("openai.chat.completions.images"),
        )
        .1
        .map(|p| p.name.clone());
        let audios_param = get_parameter(
            &signature.inputs,
            &[Dtype::ArrayList],
            Some("openai.chat.completions.audios"),
        )
        .1;
        let audios_param_name = audios_param.map(|p| p.name.clone());
        let audio_sample_rate = audios_param.and_then(|p| p.sample_rate);
        let tools_param_name = get_parameter(
            &signature.inputs,
            &[Dtype::List],
            Some("openai.chat.completions.tools"),
        )
        .1
        .map(|p| p.name.clone());
        let completion_param_idx = signature
            .outputs
            .iter()
            .position(|param| {
                param.dtype == Some(Dtype::Dict)
                    && param
                        .schema
                        .as_ref()
                        .and_then(|s| s.get("title"))
                        .and_then(|v| v.as_str())
                        .is_some_and(|title| title == "ChatCompletionChunk")
            })
            .ok_or_else(|| {
                MunaError::Prediction(format!(
                    "{tag} cannot be used with OpenAI chat completions API because \
                it does not have a valid chat completion chunk output parameter. \
                Chat predictors must yield `ChatCompletionChunk` outputs."
                ))
            })?;
        Ok(DelegateInfo {
            input_param_name: input_param.name.clone(),
            response_format_param_name,
            reasoning_effort_param_name,
            max_output_tokens_param_name,
            temperature_param_name,
            top_p_param_name,
            frequency_penalty_param_name,
            presence_penalty_param_name,
            images_param_name,
            audios_param_name,
            audio_sample_rate,
            tools_param_name,
            completion_param_idx,
        })
    }
}

/// HF-canonical media placeholder emitted in place of a consumed media
/// part: the nth `image` / `audio` placeholder across the conversation
/// corresponds to the nth entry of the parallel media input.
fn media_placeholder(kind: &str) -> serde_json::Value {
    json!({ "type": kind })
}

/// Conversation after content-part normalization: messages ready for the
/// predictor's messages input, plus the parallel media lists their
/// placeholders index into.
#[derive(Debug)]
struct NormalizedConversation {
    messages: Vec<serde_json::Value>,
    images: Vec<types::Image>,
    audios: Vec<types::Tensor>,
}

/// Normalize content parts across all messages: flatten textual content
/// to plain strings, decode media parts into parallel lists correlated by
/// order of appearance, and reject modalities the model does not declare.
async fn normalize_conversation(
    client: &dyn Client,
    messages: &[ChatCompletionMessage],
    info: &DelegateInfo
) -> Result<NormalizedConversation> {
    let mut normalized = Vec::with_capacity(messages.len());
    let mut images = Vec::new();
    let mut audios = Vec::new();
    for message in messages {
        let content: Option<serde_json::Value> = match &message.content {
            None => None,
            Some(content @ ChatCompletionContent::Text(_)) => Some(content.flatten().into()),
            Some(content @ ChatCompletionContent::Parts(_)) if content.is_text() => {
                Some(content.flatten().into())
            }
            Some(ChatCompletionContent::Parts(source_parts)) => {
                let mut parts = Vec::with_capacity(source_parts.len());
                for part in source_parts {
                    match part {
                        ChatCompletionContentPart::Text { .. } |
                        ChatCompletionContentPart::Refusal { .. } => {
                            parts.push(serde_json::to_value(part)?);
                        }
                        ChatCompletionContentPart::ImageUrl { image_url }
                            if info.images_param_name.is_some() =>
                        {
                            images.push(decode_image(client, &image_url.url).await?);
                            parts.push(media_placeholder("image"));
                        }
                        ChatCompletionContentPart::InputAudio { input_audio }
                            if info.audios_param_name.is_some() =>
                        {
                            audios.push(decode_audio(input_audio, info.audio_sample_rate)?);
                            parts.push(media_placeholder("audio"));
                        }
                        ChatCompletionContentPart::ImageUrl { .. } => {
                            return Err(MunaError::InvalidInput(
                                "`image_url` content is not supported by this model.".into(),
                            ));
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
                    }
                }
                Some(serde_json::Value::Array(parts))
            }
        };
        let mut value = json!({ "role": message.role, "content": content });
        if let Some(reasoning) = &message.reasoning_content {
            value["reasoning_content"] = reasoning.clone().into();
        }
        if let Some(tool_calls) = &message.tool_calls {
            value["tool_calls"] = serde_json::to_value(tool_calls)
                .map_err(|e| MunaError::Prediction(e.to_string()))?;
        }
        if let Some(tool_call_id) = &message.tool_call_id {
            value["tool_call_id"] = tool_call_id.clone().into();
        }
        normalized.push(value);
    }
    Ok(NormalizedConversation {
        messages: normalized,
        images,
        audios,
    })
}

/// Maximum size of a remotely-fetched image.
const MAX_IMAGE_FETCH_BYTES: usize = 20 * 1024 * 1024;

/// Decode an image content part URL (base64 data URL or remote URL) into
/// a decoded pixel buffer.
async fn decode_image(
    client: &dyn Client,
    url: &str
) -> Result<types::Image> {
    let (data, mime) = if let Some(rest) = url.strip_prefix("data:") {
        let (meta, payload) = rest.split_once(',').ok_or_else(|| {
            MunaError::InvalidInput("Malformed data URL in `image_url` content part.".into())
        })?;
        let mime = meta
            .split(';')
            .next()
            .filter(|m| !m.is_empty())
            .unwrap_or("image/*")
            .to_string();
        let data = BASE64.decode(payload).map_err(|e| {
            MunaError::InvalidInput(format!("Failed to decode image data URL: {e}"))
        })?;
        (data, mime)
    } else {
        let data = client.fetch(url).await.map_err(|e| {
            MunaError::InvalidInput(format!("Failed to fetch image at {url}: {e}"))
        })?;
        if data.len() > MAX_IMAGE_FETCH_BYTES {
            return Err(MunaError::InvalidInput(format!(
                "Image at {url} exceeds the maximum size of {MAX_IMAGE_FETCH_BYTES} bytes."
            )));
        }
        (data, "image/*".to_string())
    };
    let value = c::Value::from_bytes(&data, &mime)?;
    match value.to_object()? {
        types::Value::Image(image) => Ok(image),
        _ => Err(MunaError::InvalidInput(
            "Failed to decode `image_url` content part into an image.".into(),
        )),
    }
}

/// Decode an audio content part into linear PCM samples at the model's
/// declared sample rate via the Function C library.
fn decode_audio(
    input_audio: &ChatCompletionContentPartInputAudio,
    sample_rate: Option<u32>
) -> Result<types::Tensor> {
    let sample_rate = sample_rate.ok_or_else(|| {
        MunaError::Prediction(
            "Model does not declare a sample rate for its audio input.".into(),
        )
    })?;
    let data = BASE64.decode(&input_audio.data).map_err(|e| {
        MunaError::InvalidInput(format!("Failed to decode audio data: {e}"))
    })?;
    let mime = format!("audio/{};rate={sample_rate}", input_audio.format.as_str());
    let value = c::Value::from_bytes(&data, &mime)?;
    match value.to_object()? {
        types::Value::Tensor(tensor) => Ok(tensor),
        _ => Err(MunaError::InvalidInput(
            "Failed to decode `input_audio` content part into PCM samples.".into(),
        )),
    }
}

fn gather_completion_output(
    prediction: Prediction,
    completion_param_idx: usize,
    model: &str,
) -> Result<serde_json::Map<String, serde_json::Value>> {
    if let Some(error) = prediction.error {
        return Err(MunaError::Prediction(error));
    }
    let results = prediction
        .results
        .ok_or_else(|| MunaError::Prediction(format!("{model} returned no results")))?;
    let output = results.get(completion_param_idx).ok_or_else(|| {
        MunaError::Prediction(format!("{model} returned fewer results than expected"))
    })?;
    match output {
        Value::Dict(map) => Ok(map.clone()),
        _ => Err(MunaError::Prediction(format!(
            "{model} returned non-dict chat completion output"
        ))),
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

fn merge_chunks(chunks: Vec<ChatCompletionChunk>) -> Result<ChatCompletion> {
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
    choices: &[ChatCompletionChunkChoice]
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
mod tests {
    use super::*;
    use crate::beta::openai::{ChatCompletionDelta, CompletionTokensDetails};

    fn chunk(
        delta: ChatCompletionDelta,
        usage: Option<ChatCompletionUsage>
    ) -> ChatCompletionChunk {
        ChatCompletionChunk {
            object: "chat.completion.chunk".to_string(),
            id: "chatcmpl-test".to_string(),
            model: "test-model".to_string(),
            choices: vec![ChatCompletionChunkChoice {
                index: 0,
                delta: Some(delta),
                finish_reason: None,
                logprobs: None,
            }],
            created: 0,
            usage,
        }
    }

    #[test]
    fn merge_accumulates_reasoning_and_content() {
        let chunks = vec![
            chunk(
                ChatCompletionDelta {
                    role: Some("assistant".to_string()),
                    content: None,
                    reasoning_content: Some("Let me ".to_string()),
                    tool_calls: None,
                },
                None,
            ),
            chunk(
                ChatCompletionDelta {
                    role: None,
                    content: None,
                    reasoning_content: Some("think.".to_string()),
                    tool_calls: None,
                },
                None,
            ),
            chunk(
                ChatCompletionDelta {
                    role: None,
                    content: Some("Paris.".to_string()),
                    reasoning_content: None,
                    tool_calls: None,
                },
                Some(ChatCompletionUsage {
                    prompt_tokens: 10,
                    completion_tokens: 5,
                    total_tokens: 15,
                    prompt_tokens_details: None,
                    completion_tokens_details: Some(CompletionTokensDetails {
                        reasoning_tokens: Some(3),
                    }),
                }),
            ),
        ];
        let completion = merge_chunks(chunks).unwrap();
        let message = &completion.choices[0].message;
        assert_eq!(message.content.as_ref().map(|c| c.flatten()), Some("Paris.".to_string()));
        assert_eq!(message.reasoning_content.as_deref(), Some("Let me think."));
        let usage = completion.usage.unwrap();
        let details = usage.completion_tokens_details.unwrap();
        assert_eq!(details.reasoning_tokens, Some(3));
    }

    /// Real client for normalization tests; never touches the network
    /// because the tests only use data URLs.
    fn test_client() -> crate::client::MunaClient {
        crate::client::MunaClient::new(None, None)
    }

    fn delegate_info(images: bool) -> DelegateInfo {
        DelegateInfo {
            input_param_name: "messages".to_string(),
            response_format_param_name: None,
            reasoning_effort_param_name: None,
            max_output_tokens_param_name: None,
            temperature_param_name: None,
            top_p_param_name: None,
            frequency_penalty_param_name: None,
            presence_penalty_param_name: None,
            images_param_name: images.then(|| "images".to_string()),
            audios_param_name: None,
            audio_sample_rate: None,
            tools_param_name: None,
            completion_param_idx: 0,
        }
    }

    /// 2x2 solid-red RGBA PNG.
    const RED_PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAFUlEQVR4nGP8z8Dwn4GBgYEJRIAwAB8XAgICR7MUAAAAAElFTkSuQmCC";

    #[test]
    fn content_deserializes_from_string_and_parts() {
        let message: ChatCompletionMessage = serde_json::from_value(json!({
            "role": "user",
            "content": "What is the capital of France?",
        })).unwrap();
        assert!(matches!(message.content, Some(ChatCompletionContent::Text(_))));
        // OpenCode-shaped body: content as an array of text parts.
        let message: ChatCompletionMessage = serde_json::from_value(json!({
            "role": "user",
            "content": [{ "type": "text", "text": "What is the capital of France?" }],
        })).unwrap();
        let Some(ChatCompletionContent::Parts(parts)) = &message.content else {
            panic!("expected parts content");
        };
        assert!(matches!(&parts[0], ChatCompletionContentPart::Text { text } if text == "What is the capital of France?"));
        // Text serializes back as a bare string (untagged).
        let text = ChatCompletionContent::Text("Paris.".to_string());
        assert_eq!(serde_json::to_value(&text).unwrap(), json!("Paris."));
    }

    #[test]
    fn content_flatten_joins_text_parts() {
        let content: ChatCompletionContent = serde_json::from_value(json!([
            { "type": "text", "text": "line one" },
            { "type": "text", "text": "line two" },
            { "type": "refusal", "refusal": "I cannot help with that." },
        ])).unwrap();
        assert!(content.is_text());
        assert_eq!(content.flatten(), "line one\nline two\nI cannot help with that.");
        let mixed: ChatCompletionContent = serde_json::from_value(json!([
            { "type": "text", "text": "describe this" },
            { "type": "image_url", "image_url": { "url": "https://example.com/cat.png" } },
        ])).unwrap();
        assert!(!mixed.is_text());
    }

    #[tokio::test]
    async fn normalize_flattens_text_and_passes_strings_through() {
        let messages: Vec<ChatCompletionMessage> = serde_json::from_value(json!([
            { "role": "system", "content": "You are a helpful assistant." },
            { "role": "user", "content": [
                { "type": "text", "text": "line one" },
                { "type": "text", "text": "line two" },
            ] },
        ])).unwrap();
        let conversation = normalize_conversation(
            &test_client(),
            &messages,
            &delegate_info(false)
        ).await.unwrap();
        assert_eq!(conversation.messages[0]["content"], json!("You are a helpful assistant."));
        assert_eq!(conversation.messages[1]["content"], json!("line one\nline two"));
        assert!(conversation.images.is_empty());
        assert!(conversation.audios.is_empty());
    }

    #[tokio::test]
    async fn normalize_decodes_image_parts_into_placeholders() {
        let messages: Vec<ChatCompletionMessage> = serde_json::from_value(json!([
            { "role": "user", "content": [
                { "type": "text", "text": "describe this" },
                { "type": "image_url", "image_url": {
                    "url": format!("data:image/png;base64,{RED_PNG_B64}"),
                } },
            ] },
        ])).unwrap();
        let conversation = normalize_conversation(
            &test_client(),
            &messages,
            &delegate_info(true)
        ).await.unwrap();
        assert_eq!(conversation.images.len(), 1);
        assert!(conversation.audios.is_empty());
        assert_eq!(
            conversation.messages[0]["content"],
            json!([
                { "type": "text", "text": "describe this" },
                { "type": "image" },
            ])
        );
    }

    #[tokio::test]
    async fn normalize_rejects_undeclared_and_file_modalities() {
        let messages: Vec<ChatCompletionMessage> = serde_json::from_value(json!([
            { "role": "user", "content": [
                { "type": "image_url", "image_url": { "url": "data:image/png;base64,AAAA" } },
            ] },
        ])).unwrap();
        let error = normalize_conversation(
            &test_client(),
            &messages,
            &delegate_info(false)
        ).await.unwrap_err();
        assert!(matches!(&error, MunaError::InvalidInput(m) if m.contains("image_url")));
        let messages: Vec<ChatCompletionMessage> = serde_json::from_value(json!([
            { "role": "user", "content": [
                { "type": "file", "file": { "file_data": "AAAA", "filename": "doc.pdf" } },
            ] },
        ])).unwrap();
        let error = normalize_conversation(
            &test_client(),
            &messages,
            &delegate_info(true)
        ).await.unwrap_err();
        assert!(matches!(&error, MunaError::InvalidInput(m) if m.contains("File content parts")));
    }

    #[test]
    fn merge_accumulates_tool_call_fragments() {
        // Fragments arrive as: id + name, then two argument fragments.
        let deltas: Vec<ChatCompletionDelta> = vec![
            serde_json::from_value(json!({
                "role": "assistant",
                "tool_calls": [{
                    "index": 0,
                    "id": "call_test_0",
                    "type": "function",
                    "function": { "name": "get_weather", "arguments": "" },
                }],
            })).unwrap(),
            serde_json::from_value(json!({
                "tool_calls": [{
                    "index": 0,
                    "function": { "arguments": "{\"location\": " },
                }],
            })).unwrap(),
            serde_json::from_value(json!({
                "tool_calls": [{
                    "index": 0,
                    "function": { "arguments": "\"Paris\"}" },
                }],
            })).unwrap(),
        ];
        let chunks: Vec<ChatCompletionChunk> = deltas
            .into_iter()
            .map(|delta| chunk(delta, None))
            .collect();
        let completion = merge_chunks(chunks).unwrap();
        let calls = completion.choices[0].message.tool_calls.as_ref().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_test_0");
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, "{\"location\": \"Paris\"}");
    }

    #[tokio::test]
    async fn normalize_passes_tool_turns_through() {
        let messages: Vec<ChatCompletionMessage> = serde_json::from_value(json!([
            { "role": "assistant", "content": null, "tool_calls": [{
                "id": "call_123",
                "type": "function",
                "function": { "name": "get_weather", "arguments": "{\"city\": \"Paris\"}" },
            }] },
            { "role": "tool", "content": "18C and sunny", "tool_call_id": "call_123" },
        ])).unwrap();
        let conversation = normalize_conversation(
            &test_client(),
            &messages,
            &delegate_info(false)
        ).await.unwrap();
        assert_eq!(
            conversation.messages[0]["tool_calls"][0]["function"]["name"],
            json!("get_weather")
        );
        assert_eq!(conversation.messages[1]["role"], json!("tool"));
        assert_eq!(conversation.messages[1]["tool_call_id"], json!("call_123"));
    }

    #[test]
    fn merge_without_reasoning_leaves_field_absent() {
        let chunks = vec![chunk(
            ChatCompletionDelta {
                role: Some("assistant".to_string()),
                content: Some("Hello.".to_string()),
                reasoning_content: None,
                tool_calls: None,
            },
            None,
        )];
        let completion = merge_chunks(chunks).unwrap();
        let message = &completion.choices[0].message;
        assert_eq!(message.reasoning_content, None);
        // Absent reasoning must not appear on the wire (skip_serializing_if).
        let json = serde_json::to_value(message).unwrap();
        assert!(json.get("reasoning_content").is_none());
    }
}
