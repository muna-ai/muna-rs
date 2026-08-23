/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::collections::{BTreeMap, HashMap};
use std::pin::Pin;
use std::sync::Arc;

use futures_core::Stream;
use futures_util::StreamExt;
use tokio::sync::RwLock;

use crate::beta::utils::get_parameter;
use crate::client::Result;
use crate::MunaError;
use crate::services::{PredictionService, PredictorService};
use crate::types::{Acceleration, Dtype, Parameter, Prediction, Value};

use super::schema::{
    ChatCompletion, ChatCompletionChoice, ChatCompletionChunk,
    ChatCompletionChunkChoice, ChatCompletionCreateParams,
    ChatCompletionMessage, ChatCompletionUsage,
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
    completion_param_idx: usize,
}

/// Create chat completions.
#[derive(Clone)]
pub struct ChatCompletionService {
    predictors: PredictorService,
    predictions: PredictionService,
    cache: Arc<RwLock<HashMap<String, DelegateInfo>>>,
}

impl ChatCompletionService {

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
        let messages = params
            .messages
            .iter()
            .map(serde_json::to_value)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        input_map.insert(info.input_param_name, Value::List(messages));
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
            completion_param_idx,
        })
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
            content: Some(content),
            reasoning_content: (!reasoning_content.is_empty()).then_some(reasoning_content),
        },
        finish_reason,
        logprobs: None,
    }
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
                },
                None,
            ),
            chunk(
                ChatCompletionDelta {
                    role: None,
                    content: None,
                    reasoning_content: Some("think.".to_string()),
                },
                None,
            ),
            chunk(
                ChatCompletionDelta {
                    role: None,
                    content: Some("Paris.".to_string()),
                    reasoning_content: None,
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
        assert_eq!(message.content.as_deref(), Some("Paris."));
        assert_eq!(message.reasoning_content.as_deref(), Some("Let me think."));
        let usage = completion.usage.unwrap();
        let details = usage.completion_tokens_details.unwrap();
        assert_eq!(details.reasoning_tokens, Some(3));
    }

    #[test]
    fn merge_without_reasoning_leaves_field_absent() {
        let chunks = vec![chunk(
            ChatCompletionDelta {
                role: Some("assistant".to_string()),
                content: Some("Hello.".to_string()),
                reasoning_content: None,
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
