/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use futures_core::Stream;
use futures_util::StreamExt;
use tokio::sync::RwLock;

use crate::beta::openai::ChatCompletionChunk;
use crate::beta::utils::get_parameter;
use crate::client::Result;
use crate::MunaError;
use crate::services::{PredictionService, PredictorService};
use crate::types::{Acceleration, Dtype, Parameter, Prediction, Value};

use super::schema::{
    ContentBlock, ContentBlockDelta, Message, MessageCreateParams,
    MessageDelta, MessageParam, RawMessageStreamEvent, StopReason,
    Usage,
};

/// Stream of raw message stream events.
pub type MessageEventStream = Pin<Box<dyn Stream<Item = Result<RawMessageStreamEvent>> + Send>>;

/// Stream of dict outputs gathered from predictions.
type OutputStream = Pin<Box<dyn Stream<Item = Result<serde_json::Map<String, serde_json::Value>>> + Send>>;

/// Stream of chat completion chunks.
type ChunkStream = Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk>> + Send>>;

/// Whether the underlying predictor is written against the OpenAI chat
/// completions API or the Anthropic messages API.
#[derive(Clone, Copy, PartialEq, Eq)]
enum DelegateKind {
    OpenAI,
    Native,
}

/// Cached predictor metadata for fast message creation.
#[derive(Clone)]
struct DelegateInfo {
    kind: DelegateKind,
    input_param_name: String,
    max_tokens_param_name: Option<String>,
    stop_sequences_param_name: Option<String>,
    temperature_param_name: Option<String>,
    top_k_param_name: Option<String>,
    top_p_param_name: Option<String>,
    output_param_idx: usize,
}

/// Create messages.
#[derive(Clone)]
pub struct MessageService {
    predictors: PredictorService,
    predictions: PredictionService,
    cache: Arc<RwLock<HashMap<String, DelegateInfo>>>,
}

impl MessageService {

    pub fn new(
        predictors: PredictorService,
        predictions: PredictionService
    ) -> Self {
        Self {
            predictors,
            predictions,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a message.
    pub async fn create(&self, params: MessageCreateParams) -> Result<Message> {
        let mut events = self.stream(params).await?;
        let mut message: Option<Message> = None;
        while let Some(event) = events.next().await {
            apply_event(&mut message, &event?);
        }
        message.ok_or_else(|| {
            MunaError::Prediction(
                "Failed to create message because the model did not return any outputs".into(),
            )
        })
    }

    /// Stream a message.
    pub async fn stream(&self, params: MessageCreateParams) -> Result<MessageEventStream> {
        let model = params.model.clone();
        let (
            input_map,
            kind,
            output_param_idx,
            acceleration
        ) = self.prepare_prediction(params).await?;
        let prediction_stream = self
            .predictions
            .stream(&model, input_map, Some(acceleration))
            .await?;
        let outputs = gather_prediction_outputs(prediction_stream, output_param_idx, model);
        let events = match kind {
            DelegateKind::OpenAI => events_from_chunks(parse_completion_chunks(outputs)),
            DelegateKind::Native => events_from_native_outputs(outputs),
        };
        Ok(events)
    }

    async fn prepare_prediction(
        &self,
        params: MessageCreateParams,
    ) -> Result<(HashMap<String, Value>, DelegateKind, usize, Acceleration)> {
        self.ensure_delegate_info(&params.model).await?;
        let info = {
            let cache = self.cache.read().await;
            cache.get(&params.model).cloned().ok_or_else(|| {
                MunaError::Prediction(format!(
                    "{} cannot be used with Anthropic messages API because \
                    the predictor metadata could not be cached.",
                    params.model
                ))
            })?
        };
        // Build the message list, folding the system prompt into the messages.
        // Predictors that need the system prompt separately can filter it out.
        let messages = match info.kind {
            DelegateKind::OpenAI => {
                let mut messages = Vec::new();
                if let Some(system) = &params.system {
                    messages.push(serde_json::json!({
                        "role": "system",
                        "content": system.flatten(),
                    }));
                }
                for message in &params.messages {
                    messages.push(serde_json::json!({
                        "role": message.role,
                        "content": message.content.flatten(),
                    }));
                }
                messages
            }
            DelegateKind::Native => {
                let mut messages = Vec::new();
                if let Some(system) = &params.system {
                    messages.push(serde_json::to_value(MessageParam {
                        role: "system".to_string(),
                        content: system.clone(),
                    })?);
                }
                for message in &params.messages {
                    messages.push(serde_json::to_value(message)?);
                }
                messages
            }
        };
        let mut input_map = HashMap::new();
        input_map.insert(info.input_param_name, Value::List(messages));
        if let Some(name) = info.max_tokens_param_name {
            input_map.insert(name, Value::Int(params.max_tokens));
        }
        if let (Some(value), Some(name)) = (params.stop_sequences, info.stop_sequences_param_name)
        {
            let sequences = value.into_iter().map(serde_json::Value::String).collect();
            input_map.insert(name, Value::List(sequences));
        }
        if let (Some(value), Some(name)) = (params.temperature, info.temperature_param_name) {
            input_map.insert(name, Value::Float(value));
        }
        if let (Some(value), Some(name)) = (params.top_k, info.top_k_param_name) {
            input_map.insert(name, Value::Int(value));
        }
        if let (Some(value), Some(name)) = (params.top_p, info.top_p_param_name) {
            input_map.insert(name, Value::Float(value));
        }
        let acceleration = params.acceleration.unwrap_or(Acceleration::LocalAuto);
        Ok((input_map, info.kind, info.output_param_idx, acceleration))
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
                "{tag} cannot be used with Anthropic messages API because \
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
                "{tag} cannot be used with Anthropic messages API because \
                it has more than one required input parameter."
            )));
        }
        let input_param = required_inputs[0];
        if input_param.dtype != Some(Dtype::List) {
            return Err(MunaError::Prediction(format!(
                "{tag} cannot be used with Anthropic messages API because \
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
        let max_tokens_param_name = get_parameter(
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
        // Check whether the predictor is written against the OpenAI chat completions API.
        // If not, assume it is written against the Anthropic messages API.
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
            });
        if let Some(output_param_idx) = completion_param_idx {
            return Ok(DelegateInfo {
                kind: DelegateKind::OpenAI,
                input_param_name: input_param.name.clone(),
                max_tokens_param_name,
                stop_sequences_param_name: None,
                temperature_param_name,
                top_k_param_name: None,
                top_p_param_name,
                output_param_idx,
            });
        }
        let stop_sequences_param_name = get_parameter(
            &signature.inputs,
            &[Dtype::List],
            Some("anthropic.messages.stop_sequences"),
        )
        .1
        .map(|p| p.name.clone());
        let top_k_param_name = get_parameter(
            &signature.inputs,
            &int_dtypes,
            Some("anthropic.messages.top_k"),
        )
        .1
        .map(|p| p.name.clone());
        let output_param_idx = signature
            .outputs
            .iter()
            .position(|param| param.dtype == Some(Dtype::Dict))
            .ok_or_else(|| {
                MunaError::Prediction(format!(
                    "{tag} cannot be used with Anthropic messages API because \
                it does not have a valid message output parameter."
                ))
            })?;
        Ok(DelegateInfo {
            kind: DelegateKind::Native,
            input_param_name: input_param.name.clone(),
            max_tokens_param_name,
            stop_sequences_param_name,
            temperature_param_name,
            top_k_param_name,
            top_p_param_name,
            output_param_idx,
        })
    }
}

/// Kind of content block being streamed.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BlockKind {
    Thinking,
    Text,
}

impl BlockKind {

    fn empty_block(self) -> ContentBlock {
        match self {
            Self::Thinking => ContentBlock::Thinking {
                thinking: String::new(),
                signature: String::new(),
            },
            Self::Text => ContentBlock::Text {
                text: String::new(),
            },
        }
    }

    fn delta(self, fragment: &str) -> ContentBlockDelta {
        match self {
            Self::Thinking => ContentBlockDelta::ThinkingDelta {
                thinking: fragment.to_string(),
            },
            Self::Text => ContentBlockDelta::TextDelta {
                text: fragment.to_string(),
            },
        }
    }
}

/// Stream the dict output at `output_param_idx` from each prediction.
fn gather_prediction_outputs(
    mut predictions: Pin<Box<dyn Stream<Item = Result<Prediction>> + Send>>,
    output_param_idx: usize,
    model: String,
) -> OutputStream {
    Box::pin(async_stream::try_stream! {
        while let Some(prediction) = predictions.next().await {
            let prediction = prediction?;
            if let Some(error) = prediction.error {
                Err(MunaError::Prediction(error))?;
            }
            let results = prediction
                .results
                .ok_or_else(|| MunaError::Prediction(format!("{model} returned no results")))?;
            let output = results.get(output_param_idx).ok_or_else(|| {
                MunaError::Prediction(format!("{model} returned fewer results than expected"))
            })?;
            match output {
                Value::Dict(map) => yield map.clone(),
                _ => Err(MunaError::Prediction(format!(
                    "{model} returned non-dict message output"
                )))?,
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

/// Convert a stream of chat completion chunks into raw message stream events.
fn events_from_chunks(mut chunks: ChunkStream) -> MessageEventStream {
    Box::pin(async_stream::try_stream! {
        let mut started = false;
        let mut block_idx = 0usize;
        let mut block_kind: Option<BlockKind> = None;
        let mut stop_reason = StopReason::EndTurn;
        let mut usage = Usage::default();
        while let Some(chunk) = chunks.next().await {
            let chunk = chunk?;
            if !started {
                yield RawMessageStreamEvent::MessageStart {
                    message: Message {
                        id: chunk.id.clone(),
                        r#type: "message".to_string(),
                        role: "assistant".to_string(),
                        content: Vec::new(),
                        model: chunk.model.clone(),
                        stop_reason: None,
                        stop_sequence: None,
                        usage: Usage {
                            input_tokens: Some(0),
                            ..Default::default()
                        },
                    },
                };
                started = true;
            }
            if let Some(chunk_usage) = &chunk.usage {
                let details = chunk_usage.prompt_tokens_details.as_ref();
                let cache_read = details.and_then(|d| d.cached_tokens);
                let cache_write = details.and_then(|d| d.cache_write_tokens);
                usage.input_tokens = Some(
                    chunk_usage
                        .prompt_tokens
                        .saturating_sub(cache_read.unwrap_or(0) + cache_write.unwrap_or(0)),
                );
                usage.output_tokens += chunk_usage.completion_tokens;
                usage.cache_read_input_tokens = cache_read;
                usage.cache_creation_input_tokens = cache_write;
            }
            let Some(choice) = chunk.choices.first() else {
                continue;
            };
            if let Some(reason) = &choice.finish_reason {
                stop_reason = map_finish_reason(reason);
            }
            let Some(delta) = &choice.delta else {
                continue;
            };
            let fragments = [
                (BlockKind::Thinking, delta.reasoning_content.clone()),
                (BlockKind::Text, delta.content.clone()),
            ];
            for (kind, fragment) in fragments {
                let Some(fragment) = fragment else {
                    continue;
                };
                if fragment.is_empty() {
                    continue;
                }
                if block_kind != Some(kind) {
                    if block_kind.is_some() {
                        yield RawMessageStreamEvent::ContentBlockStop { index: block_idx };
                        block_idx += 1;
                    }
                    block_kind = Some(kind);
                    yield RawMessageStreamEvent::ContentBlockStart {
                        index: block_idx,
                        content_block: kind.empty_block(),
                    };
                }
                yield RawMessageStreamEvent::ContentBlockDelta {
                    index: block_idx,
                    delta: kind.delta(&fragment),
                };
            }
        }
        if block_kind.is_some() {
            yield RawMessageStreamEvent::ContentBlockStop { index: block_idx };
        }
        yield RawMessageStreamEvent::MessageDelta {
            delta: MessageDelta {
                stop_reason: Some(stop_reason),
                stop_sequence: None,
            },
            usage,
        };
        yield RawMessageStreamEvent::MessageStop;
    })
}

/// Parse each dict output as a raw message stream event.
fn events_from_native_outputs(mut outputs: OutputStream) -> MessageEventStream {
    Box::pin(async_stream::try_stream! {
        while let Some(output) = outputs.next().await {
            let output = output?;
            let event = serde_json::from_value::<RawMessageStreamEvent>(
                serde_json::Value::Object(output)
            )
            .map_err(|_| MunaError::Prediction(
                "Failed to parse message stream event from model output. \
                Message predictors must yield `RawMessageStreamEvent` outputs."
                    .into(),
            ))?;
            yield event;
        }
    })
}

/// Accumulate a raw message stream event into the final message.
fn apply_event(message: &mut Option<Message>, event: &RawMessageStreamEvent) {
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
                        _ => {}
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
where T: serde::de::DeserializeOwned,
{
    serde_json::from_value(serde_json::Value::Object(output))
        .map_err(|e| MunaError::Prediction(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beta::openai::{
        ChatCompletionChunkChoice, ChatCompletionDelta, ChatCompletionUsage,
        PromptTokensDetails,
    };
    use futures_util::stream;

    fn chunk(
        delta: ChatCompletionDelta,
        finish_reason: Option<&str>,
        usage: Option<ChatCompletionUsage>,
    ) -> ChatCompletionChunk {
        ChatCompletionChunk {
            object: "chat.completion.chunk".to_string(),
            id: "msg-test".to_string(),
            model: "test-model".to_string(),
            choices: vec![ChatCompletionChunkChoice {
                index: 0,
                delta: Some(delta),
                finish_reason: finish_reason.map(str::to_string),
                logprobs: None,
            }],
            created: 0,
            usage,
        }
    }

    async fn collect_events(chunks: Vec<ChatCompletionChunk>) -> Vec<RawMessageStreamEvent> {
        let chunk_stream = Box::pin(stream::iter(chunks.into_iter().map(Ok)));
        let mut events = events_from_chunks(chunk_stream);
        let mut collected = Vec::new();
        while let Some(event) = events.next().await {
            collected.push(event.unwrap());
        }
        collected
    }

    fn accumulate(events: &[RawMessageStreamEvent]) -> Message {
        let mut message = None;
        for event in events {
            apply_event(&mut message, event);
        }
        message.unwrap()
    }

    #[tokio::test]
    async fn chunks_become_ordered_events() {
        let events = collect_events(vec![
            chunk(
                ChatCompletionDelta {
                    role: Some("assistant".to_string()),
                    content: None,
                    reasoning_content: Some("Let me think.".to_string()),
                },
                None,
                None,
            ),
            chunk(
                ChatCompletionDelta {
                    role: None,
                    content: Some("Paris.".to_string()),
                    reasoning_content: None,
                },
                Some("stop"),
                Some(ChatCompletionUsage {
                    prompt_tokens: 10,
                    completion_tokens: 5,
                    total_tokens: 15,
                    prompt_tokens_details: Some(PromptTokensDetails {
                        audio_tokens: None,
                        cache_write_tokens: Some(2),
                        cached_tokens: Some(3),
                    }),
                    completion_tokens_details: None,
                }),
            ),
        ])
        .await;
        let types: Vec<&str> = events.iter().map(|e| e.event_type()).collect();
        assert_eq!(
            types,
            vec![
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        let message = accumulate(&events);
        assert!(matches!(
            &message.content[0],
            ContentBlock::Thinking { thinking, .. } if thinking == "Let me think."
        ));
        assert!(matches!(
            &message.content[1],
            ContentBlock::Text { text } if text == "Paris."
        ));
        assert_eq!(message.stop_reason, Some(StopReason::EndTurn));
        // Anthropic convention: input tokens exclude cached tokens.
        assert_eq!(message.usage.input_tokens, Some(5));
        assert_eq!(message.usage.output_tokens, 5);
        assert_eq!(message.usage.cache_read_input_tokens, Some(3));
        assert_eq!(message.usage.cache_creation_input_tokens, Some(2));
    }

    #[tokio::test]
    async fn length_finish_reason_maps_to_max_tokens() {
        let events = collect_events(vec![chunk(
            ChatCompletionDelta {
                role: Some("assistant".to_string()),
                content: Some("Hello".to_string()),
                reasoning_content: None,
            },
            Some("length"),
            None,
        )])
        .await;
        let message = accumulate(&events);
        assert_eq!(message.stop_reason, Some(StopReason::MaxTokens));
    }

    #[test]
    fn events_accumulate_to_final_message() {
        let events = vec![
            RawMessageStreamEvent::MessageStart {
                message: Message {
                    id: "msg-1".to_string(),
                    r#type: "message".to_string(),
                    role: "assistant".to_string(),
                    content: Vec::new(),
                    model: "test-model".to_string(),
                    stop_reason: None,
                    stop_sequence: None,
                    usage: Usage {
                        input_tokens: Some(10),
                        ..Default::default()
                    },
                },
            },
            RawMessageStreamEvent::ContentBlockStart {
                index: 0,
                content_block: ContentBlock::Text {
                    text: String::new(),
                },
            },
            RawMessageStreamEvent::ContentBlockDelta {
                index: 0,
                delta: ContentBlockDelta::TextDelta {
                    text: "Par".to_string(),
                },
            },
            RawMessageStreamEvent::ContentBlockDelta {
                index: 0,
                delta: ContentBlockDelta::TextDelta {
                    text: "is.".to_string(),
                },
            },
            RawMessageStreamEvent::ContentBlockStop { index: 0 },
            RawMessageStreamEvent::MessageDelta {
                delta: MessageDelta {
                    stop_reason: Some(StopReason::EndTurn),
                    stop_sequence: None,
                },
                usage: Usage {
                    input_tokens: Some(10),
                    output_tokens: 5,
                    ..Default::default()
                },
            },
            RawMessageStreamEvent::MessageStop,
        ];
        let message = accumulate(&events);
        assert!(matches!(
            &message.content[0],
            ContentBlock::Text { text } if text == "Paris."
        ));
        assert_eq!(message.stop_reason, Some(StopReason::EndTurn));
        assert_eq!(message.usage.input_tokens, Some(10));
        assert_eq!(message.usage.output_tokens, 5);
    }

    #[test]
    fn events_serialize_with_wire_tags() {
        let event = RawMessageStreamEvent::ContentBlockDelta {
            index: 0,
            delta: ContentBlockDelta::TextDelta {
                text: "Hi".to_string(),
            },
        };
        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(json["type"], "content_block_delta");
        assert_eq!(json["delta"]["type"], "text_delta");
        assert_eq!(json["delta"]["text"], "Hi");
        let round_trip: RawMessageStreamEvent = serde_json::from_value(json).unwrap();
        assert_eq!(round_trip.event_type(), "content_block_delta");
    }

    #[test]
    fn native_output_parses_events_only() {
        let event_value = serde_json::json!({
            "type": "message_stop"
        });
        let event: RawMessageStreamEvent = serde_json::from_value(event_value).unwrap();
        assert_eq!(event.event_type(), "message_stop");
        // Full messages are not valid stream events; predictors must yield events.
        let message_value = serde_json::json!({
            "id": "msg-1",
            "type": "message",
            "role": "assistant",
            "content": [{ "type": "text", "text": "Paris." }],
            "model": "test-model",
            "stop_reason": "end_turn",
            "usage": { "input_tokens": 10, "output_tokens": 5 }
        });
        assert!(serde_json::from_value::<RawMessageStreamEvent>(message_value).is_err());
    }
}
