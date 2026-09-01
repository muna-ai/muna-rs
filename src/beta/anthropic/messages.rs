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
    ContentBlock, ContentBlockDelta, ContentBlockParam, Message,
    MessageContent, MessageCreateParams, MessageDelta, MessageParam,
    RawMessageStreamEvent, StopReason, Tool, Usage,
};

/// Stream of raw message stream events.
pub type MessageEventStream = Pin<Box<dyn Stream<Item = Result<RawMessageStreamEvent>> + Send>>;

/// Stream of dict outputs gathered from predictions.
type OutputStream = Pin<Box<dyn Stream<Item = Result<serde_json::Map<String, serde_json::Value>>> + Send>>;

/// Stream of chat completion chunks.
type ChunkStream = Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk>> + Send>>;

/// Cached predictor metadata for fast message creation. Chat predictors
/// always speak the OpenAI chat completions contract; this surface is a
/// pure adapter that translates Anthropic requests and responses to and
/// from it.
#[derive(Clone)]
struct DelegateInfo {
    input_param_name: String,
    max_tokens_param_name: Option<String>,
    stop_sequences_param_name: Option<String>,
    temperature_param_name: Option<String>,
    top_k_param_name: Option<String>,
    top_p_param_name: Option<String>,
    tools_param_name: Option<String>,
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
            output_param_idx,
            acceleration
        ) = self.prepare_prediction(params).await?;
        let prediction_stream = self
            .predictions
            .stream(&model, input_map, Some(acceleration))
            .await?;
        let outputs = gather_prediction_outputs(
            prediction_stream,
            output_param_idx,
            model
        );
        Ok(events_from_chunks(parse_completion_chunks(outputs)))
    }

    async fn prepare_prediction(
        &self,
        params: MessageCreateParams,
    ) -> Result<(HashMap<String, Value>, usize, Acceleration)> {
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
        // Build the OpenAI-shaped message list, folding the system prompt
        // into the messages. Predictors that need the system prompt
        // separately can filter it out.
        let mut messages = Vec::new();
        if let Some(system) = &params.system {
            messages.push(serde_json::json!({
                "role": "system",
                "content": system.flatten(),
            }));
        }
        for message in &params.messages {
            translate_message_for_openai(message, &mut messages);
        }
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
        if let Some(tools) = &params.tools {
            if !tools.is_empty() {
                let Some(name) = info.tools_param_name else {
                    return Err(MunaError::InvalidInput(format!(
                        "{} does not support tool calling because it does not \
                        declare a tools input parameter.",
                        params.model
                    )));
                };
                let tools = tools.iter().map(openai_tool).collect();
                input_map.insert(name, Value::List(tools));
            }
        }
        let acceleration = params.acceleration.unwrap_or(Acceleration::LocalAuto);
        Ok((input_map, info.output_param_idx, acceleration))
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
        // Anthropic-specific knobs (stop sequences, top-k) keep their own
        // denotations: parameter denotations describe input meaning and are
        // orthogonal to the output contract, so an OpenAI-shaped predictor
        // can declare them for this surface.
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
        let tools_param_name = get_parameter(
            &signature.inputs,
            &[Dtype::List],
            Some("openai.chat.completions.tools"),
        )
        .1
        .map(|p| p.name.clone());
        let output_param_idx = signature
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
                    "{tag} cannot be used with Anthropic messages API because \
                it does not have a valid chat completion chunk output parameter. \
                Chat predictors must yield `ChatCompletionChunk` outputs."
                ))
            })?;
        Ok(DelegateInfo {
            input_param_name: input_param.name.clone(),
            max_tokens_param_name,
            stop_sequences_param_name,
            temperature_param_name,
            top_k_param_name,
            top_p_param_name,
            tools_param_name,
            output_param_idx,
        })
    }
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

/// Translate one Anthropic input message into OpenAI-shaped messages.
/// Text blocks stay on the original role (flattened, joined by newline);
/// replayed `tool_use` blocks become assistant `tool_calls`; `tool_result`
/// blocks become `tool` role messages bound by `tool_call_id`. Block order
/// is preserved by flushing the pending text run before each tool block.
fn translate_message_for_openai(
    message: &MessageParam,
    messages: &mut Vec<serde_json::Value>
) {
    let MessageContent::Blocks(blocks) = &message.content else {
        messages.push(serde_json::json!({
            "role": message.role,
            "content": message.content.flatten(),
        }));
        return;
    };
    let mut text_run: Vec<&str> = Vec::new();
    let flush = |text_run: &mut Vec<&str>, messages: &mut Vec<serde_json::Value>| {
        if !text_run.is_empty() {
            messages.push(serde_json::json!({
                "role": message.role,
                "content": text_run.join("\n"),
            }));
            text_run.clear();
        }
    };
    for block in blocks {
        match block {
            ContentBlockParam::Text { text } => {
                text_run.push(text.as_str());
            }
            ContentBlockParam::ToolUse { id, name, input } => {
                flush(&mut text_run, messages);
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
            ContentBlockParam::ToolResult { tool_use_id, content, .. } => {
                flush(&mut text_run, messages);
                messages.push(serde_json::json!({
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "content": content.as_ref().map(|c| c.flatten()),
                }));
            }
        }
    }
    flush(&mut text_run, messages);
}

/// Kind of content block being streamed.
#[derive(Clone, PartialEq, Eq)]
enum BlockKind {
    Thinking,
    Text,
    /// Tool-use block; id and name arrive on the first tool call fragment.
    ToolUse {
        id: String,
        name: String,
    },
}

impl BlockKind {

    fn empty_block(&self) -> ContentBlock {
        match self {
            Self::Thinking => ContentBlock::Thinking {
                thinking: String::new(),
                signature: String::new(),
            },
            Self::Text => ContentBlock::Text {
                text: String::new(),
            },
            Self::ToolUse { id, name } => ContentBlock::ToolUse {
                id: id.clone(),
                name: name.clone(),
                input: serde_json::json!({}),
            },
        }
    }

    fn delta(&self, fragment: &str) -> ContentBlockDelta {
        match self {
            Self::Thinking => ContentBlockDelta::ThinkingDelta {
                thinking: fragment.to_string(),
            },
            Self::Text => ContentBlockDelta::TextDelta {
                text: fragment.to_string(),
            },
            Self::ToolUse { .. } => ContentBlockDelta::InputJsonDelta {
                partial_json: fragment.to_string(),
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
                if block_kind.as_ref() != Some(&kind) {
                    if block_kind.is_some() {
                        yield RawMessageStreamEvent::ContentBlockStop { index: block_idx };
                        block_idx += 1;
                    }
                    yield RawMessageStreamEvent::ContentBlockStart {
                        index: block_idx,
                        content_block: kind.empty_block(),
                    };
                    block_kind = Some(kind.clone());
                }
                yield RawMessageStreamEvent::ContentBlockDelta {
                    index: block_idx,
                    delta: kind.delta(&fragment),
                };
            }
            // Tool call fragments are a third channel beside reasoning and
            // content: a fragment carrying an id starts a new tool-use block;
            // argument fragments stream as input JSON deltas.
            let Some(tool_fragments) = &delta.tool_calls else {
                continue;
            };
            for fragment in tool_fragments {
                if let Some(id) = &fragment.id {
                    if block_kind.is_some() {
                        yield RawMessageStreamEvent::ContentBlockStop { index: block_idx };
                        block_idx += 1;
                    }
                    let name = fragment
                        .function
                        .as_ref()
                        .and_then(|f| f.name.clone())
                        .unwrap_or_default();
                    let kind = BlockKind::ToolUse { id: id.clone(), name };
                    yield RawMessageStreamEvent::ContentBlockStart {
                        index: block_idx,
                        content_block: kind.empty_block(),
                    };
                    block_kind = Some(kind);
                }
                let arguments = fragment
                    .function
                    .as_ref()
                    .and_then(|f| f.arguments.as_ref());
                if let Some(arguments) = arguments {
                    if !arguments.is_empty() &&
                        matches!(block_kind, Some(BlockKind::ToolUse { .. }))
                    {
                        yield RawMessageStreamEvent::ContentBlockDelta {
                            index: block_idx,
                            delta: ContentBlockDelta::InputJsonDelta {
                                partial_json: arguments.clone(),
                            },
                        };
                    }
                }
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
                        *input = serde_json::from_str(buffer)
                            .unwrap_or_else(|_| serde_json::json!({}));
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
                    tool_calls: None,
                },
                None,
                None,
            ),
            chunk(
                ChatCompletionDelta {
                    role: None,
                    content: Some("Paris.".to_string()),
                    reasoning_content: None,
                    tool_calls: None,
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
                tool_calls: None,
            },
            Some("length"),
            None,
        )])
        .await;
        let message = accumulate(&events);
        assert_eq!(message.stop_reason, Some(StopReason::MaxTokens));
    }

    #[tokio::test]
    async fn tool_call_deltas_become_tool_use_blocks() {
        // Fragments: id + name first, then two argument fragments, ending
        // with a `tool_calls` finish reason.
        let deltas: Vec<ChatCompletionDelta> = vec![
            serde_json::from_value(serde_json::json!({
                "role": "assistant",
                "tool_calls": [{
                    "index": 0,
                    "id": "call_test_0",
                    "type": "function",
                    "function": { "name": "get_weather", "arguments": "" },
                }],
            })).unwrap(),
            serde_json::from_value(serde_json::json!({
                "tool_calls": [{
                    "index": 0,
                    "function": { "arguments": "{\"location\": " },
                }],
            })).unwrap(),
            serde_json::from_value(serde_json::json!({
                "tool_calls": [{
                    "index": 0,
                    "function": { "arguments": "\"Paris\"}" },
                }],
            })).unwrap(),
        ];
        let mut chunks: Vec<ChatCompletionChunk> = deltas
            .into_iter()
            .map(|delta| chunk(delta, None, None))
            .collect();
        chunks.push(chunk(ChatCompletionDelta::default(), Some("tool_calls"), None));
        let events = collect_events(chunks).await;
        let message = accumulate(&events);
        assert!(matches!(
            &message.content[0],
            ContentBlock::ToolUse { id, name, input }
                if id == "call_test_0" &&
                    name == "get_weather" &&
                    *input == serde_json::json!({ "location": "Paris" })
        ));
        assert_eq!(message.stop_reason, Some(StopReason::ToolUse));
        // Wire tags: tool_use block start and input_json_delta fragments.
        let start = events.iter().find(|e| e.event_type() == "content_block_start").unwrap();
        let json = serde_json::to_value(start).unwrap();
        assert_eq!(json["content_block"]["type"], "tool_use");
        let delta = events.iter().find(|e| e.event_type() == "content_block_delta").unwrap();
        let json = serde_json::to_value(delta).unwrap();
        assert_eq!(json["delta"]["type"], "input_json_delta");
    }

    #[test]
    fn tool_blocks_translate_to_openai_messages() {
        // Assistant turn replaying a tool call, then a user turn with the result.
        let assistant: MessageParam = serde_json::from_value(serde_json::json!({
            "role": "assistant",
            "content": [
                { "type": "text", "text": "Checking the weather." },
                { "type": "tool_use", "id": "call_123", "name": "get_weather",
                  "input": { "city": "Paris" } },
            ],
        })).unwrap();
        let user: MessageParam = serde_json::from_value(serde_json::json!({
            "role": "user",
            "content": [
                { "type": "tool_result", "tool_use_id": "call_123",
                  "content": "18C and sunny" },
            ],
        })).unwrap();
        let mut messages = Vec::new();
        translate_message_for_openai(&assistant, &mut messages);
        translate_message_for_openai(&user, &mut messages);
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0]["role"], "assistant");
        assert_eq!(messages[0]["content"], "Checking the weather.");
        assert_eq!(messages[1]["tool_calls"][0]["id"], "call_123");
        assert_eq!(
            messages[1]["tool_calls"][0]["function"]["arguments"],
            "{\"city\":\"Paris\"}"
        );
        assert_eq!(messages[2]["role"], "tool");
        assert_eq!(messages[2]["tool_call_id"], "call_123");
        assert_eq!(messages[2]["content"], "18C and sunny");
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
}
