/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::collections::HashMap;
use std::sync::Arc;
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use tokio::sync::RwLock;
use crate::client::Result;
use crate::MunaError;
use crate::services::{PredictionService, PredictorService};
use crate::types::{Acceleration, Dtype, Parameter, TensorData, Value};
use crate::beta::remote::RemotePredictionService;
use super::schema::{Embedding, EmbeddingCreateResponse, EmbeddingData, EmbeddingUsage};
use super::utils::get_parameter;

/// Cached predictor metadata for fast embedding creation.
struct DelegateInfo {
    input_param_name: String,
    matryoshka_param_name: Option<String>,
    embedding_param_idx: usize,
    usage_param_idx: Option<usize>,
}

/// Encoding format for embedding vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodingFormat {
    Float,
    Base64,
}

/// Create embedding vectors representing input text.
#[derive(Clone)]
pub struct EmbeddingService {
    predictors: PredictorService,
    predictions: PredictionService,
    remote_predictions: RemotePredictionService,
    cache: Arc<RwLock<HashMap<String, DelegateInfo>>>,
}

impl EmbeddingService {

    pub fn new(
        predictors: PredictorService,
        predictions: PredictionService,
        remote_predictions: RemotePredictionService,
    ) -> Self {
        Self {
            predictors,
            predictions,
            remote_predictions,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create embedding vectors representing the input text.
    ///
    /// # Arguments
    /// * `input` - Input text to embed. The input must not exceed the max input tokens for the model.
    /// * `model` - Embedding model tag.
    /// * `dimensions` - The number of dimensions for Matryoshka embedding models.
    /// * `encoding_format` - The format to return the embeddings in.
    /// * `acceleration` - Prediction acceleration.
    pub async fn create(
        &self,
        input: Vec<String>,
        model: &str,
        dimensions: Option<i32>,
        encoding_format: Option<EncodingFormat>,
        acceleration: Option<Acceleration>,
    ) -> Result<EmbeddingCreateResponse> {
        let input_texts = input;
        let encoding_format = encoding_format.unwrap_or(EncodingFormat::Float);
        let acceleration = acceleration.unwrap_or(Acceleration::LocalAuto);
        {
            let needs_create = !self.cache.read().await.contains_key(model);
            if needs_create {
                let info = self.create_delegate_info(model).await?;
                self.cache.write().await.entry(model.to_string()).or_insert(info);
            }
        }
        let cache = self.cache.read().await;
        let info = &cache[model];
        let mut input_map = HashMap::new();
        let input_value = Value::List(
            input_texts.iter().map(|s| serde_json::Value::String(s.clone())).collect()
        );
        input_map.insert(info.input_param_name.clone(), input_value);
        if let (Some(dims), Some(dim_name)) = (dimensions, &info.matryoshka_param_name) {
            input_map.insert(dim_name.clone(), Value::Int(dims));
        }
        let embedding_param_idx = info.embedding_param_idx;
        let usage_param_idx = info.usage_param_idx;
        drop(cache);
        let prediction = if is_remote(&acceleration) {
            self.remote_predictions.create(model, &input_map, Some(acceleration)).await?
        } else {
            self.predictions.create(model, Some(input_map), Some(acceleration), None, None).await?
        };
        if let Some(ref error) = prediction.error {
            return Err(MunaError::Prediction(error.clone()));
        }
        let results = prediction.results.ok_or_else(|| {
            MunaError::Prediction(format!("{model} returned no results"))
        })?;
        let embedding_value = results.get(embedding_param_idx).ok_or_else(|| {
            MunaError::Prediction(format!("{model} returned fewer results than expected"))
        })?;
        let (flat_data, shape) = match embedding_value {
            Value::Tensor(tensor) => match &tensor.data {
                TensorData::Float32(data) => (data, &tensor.shape),
                _ => return Err(MunaError::Prediction(format!(
                    "{model} returned embedding matrix with invalid data type"
                ))),
            },
            _ => return Err(MunaError::Prediction(format!(
                "{model} returned non-tensor embedding value"
            ))),
        };
        if shape.len() != 2 {
            return Err(MunaError::Prediction(format!(
                "{model} returned embedding matrix with invalid shape: {shape:?}"
            )));
        }
        let n = shape[0] as usize;
        let d = shape[1] as usize;
        let embeddings: Vec<Embedding> = (0..n)
            .map(|i| {
                let start = i * d;
                let end = start + d;
                let embedding_vec = &flat_data[start..end];
                parse_embedding(embedding_vec, i, encoding_format)
            })
            .collect();
        let usage = match usage_param_idx {
            Some(idx) => match results.get(idx) {
                Some(Value::Dict(map)) => {
                    serde_json::from_value::<EmbeddingUsage>(
                        serde_json::Value::Object(map.clone())
                    ).unwrap_or(EmbeddingUsage { prompt_tokens: 0, total_tokens: 0 })
                }
                _ => EmbeddingUsage { prompt_tokens: 0, total_tokens: 0 },
            },
            None => EmbeddingUsage { prompt_tokens: 0, total_tokens: 0 },
        };
        Ok(EmbeddingCreateResponse {
            object: "list".to_string(),
            model: model.to_string(),
            data: embeddings,
            usage,
        })
    }

    async fn create_delegate_info(&self, tag: &str) -> Result<DelegateInfo> {
        let predictor = self.predictors.retrieve(tag).await?.ok_or_else(|| {
            MunaError::Prediction(format!(
                "{tag} cannot be used with OpenAI embedding API because \
                the predictor could not be found. Check that your access key \
                is valid and that you have access to the predictor."
            ))
        })?;
        let signature = &predictor.signature;
        let required_inputs: Vec<&Parameter> = signature.inputs.iter()
            .filter(|p| !p.optional.unwrap_or(false))
            .collect();
        if required_inputs.len() != 1 {
            return Err(MunaError::Prediction(format!(
                "{tag} cannot be used with OpenAI embedding API because \
                it has more than one required input parameter."
            )));
        }
        let input_param = required_inputs[0];
        if input_param.dtype != Some(Dtype::List) {
            return Err(MunaError::Prediction(format!(
                "{tag} cannot be used with OpenAI embedding API because \
                it does not have a valid text embedding input parameter."
            )));
        }
        let input_param_name = input_param.name.clone();
        let int_dtypes = [
            Dtype::Int8, Dtype::Int16, Dtype::Int32, Dtype::Int64,
            Dtype::Uint8, Dtype::Uint16, Dtype::Uint32, Dtype::Uint64,
        ];
        let matryoshka_param_name = get_parameter(
            &signature.inputs,
            &int_dtypes,
            Some("openai.embeddings.dims"),
        ).1.map(|p| p.name.clone());
        let embedding_param_idx = get_parameter(
            &signature.outputs,
            &[Dtype::Float32],
            Some("embedding"),
        ).0.ok_or_else(|| {
            MunaError::Prediction(format!(
                "{tag} cannot be used with OpenAI embedding API because \
                it has no outputs with an `embedding` denotation."
            ))
        })?;
        let usage_param_idx = signature.outputs.iter().position(|param| {
            param.schema.as_ref()
                .and_then(|s| s.get("title"))
                .and_then(|v| v.as_str())
                == Some("Usage")
        });
        Ok(DelegateInfo {
            input_param_name,
            matryoshka_param_name,
            embedding_param_idx,
            usage_param_idx,
        })
    }
}

fn parse_embedding(
    embedding_vec: &[f32],
    index: usize,
    encoding_format: EncodingFormat,
) -> Embedding {
    let data = match encoding_format {
        EncodingFormat::Base64 => {
            let bytes: Vec<u8> = embedding_vec.iter()
                .flat_map(|f| f.to_ne_bytes())
                .collect();
            EmbeddingData::Base64(BASE64.encode(&bytes))
        }
        EncodingFormat::Float => {
            EmbeddingData::Float(embedding_vec.to_vec())
        }
    };
    Embedding {
        object: "embedding".to_string(),
        embedding: data,
        index,
    }
}

fn is_remote(acceleration: &Acceleration) -> bool {
    matches!(
        acceleration,
        Acceleration::RemoteAuto |
        Acceleration::RemoteCpu |
        Acceleration::RemoteA10 |
        Acceleration::RemoteA40 |
        Acceleration::RemoteA100 |
        Acceleration::RemoteH200 |
        Acceleration::RemoteB200
    ) || matches!(
        acceleration,
        Acceleration::Adaptive(s) if s.starts_with("remote_")
    )
}
