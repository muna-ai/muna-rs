/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::collections::HashMap;

use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use serde::{Deserialize, Serialize};

use crate::client::{MunaError, Result};
use crate::types::{Dtype, Parameter, Prediction, Signature, Value};

use super::{
    utils::{
        decode, get_parameter, output, parameter, required, result, validate_inputs, INTEGERS,
    },
    Adapter,
};

/// Embedding data, either as float values or base64-encoded bytes.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingData {
    Float(Vec<f32>),
    Base64(String),
}

/// Embedding vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    /// Object type, always `embedding`.
    pub object: String,
    /// Embedding vector as float values or a base64-encoded string.
    pub embedding: EmbeddingData,
    /// Index of the embedding in the response list.
    pub index: usize,
}

/// Usage information for an embedding request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingUsage {
    /// Number of tokens in the input prompt.
    pub prompt_tokens: u64,
    /// Total number of tokens used in the request.
    pub total_tokens: u64,
}

/// Response from creating embeddings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingCreateResponse {
    pub object: String,
    pub model: String,
    pub data: Vec<Embedding>,
    pub usage: EmbeddingUsage,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum EncodingFormat {
    #[default]
    Float,
    Base64,
}

#[derive(Debug, Clone, Default)]
pub struct OpenAIEmbeddingsOptions {
    pub encoding_format: EncodingFormat,
}

/// Text embedding inputs and a float32 embedding matrix with optional token usage.
#[derive(Debug, Clone)]
pub struct OpenAIEmbeddingsAdapter {
    pub input: Parameter,
    pub output: usize,
    pub dimensions: Option<Parameter>,
    pub usage: Option<usize>,
}

impl OpenAIEmbeddingsAdapter {
    /// Returns `None` when the signature is incompatible or ambiguous.
    pub fn from_signature(signature: &Signature) -> Option<Self> {
        // Text input.
        let input = parameter(&required(&signature.inputs), &[Dtype::List], None).ok()??;
        // Matryoshka dimensions.
        let dimensions =
            parameter(&signature.inputs, INTEGERS, Some("openai.embeddings.dims")).ok()?;
        validate_inputs(signature, &[Some(&input), dimensions.as_ref()]).ok()?;
        // Embedding matrix and optional token usage.
        let (output, _) =
            get_parameter(&signature.outputs, &[Dtype::Float32], Some("embedding")).ok()??;
        let usage = output_usage(signature).ok()?;
        Some(Self {
            input,
            output,
            dimensions,
            usage,
        })
    }

    pub fn inputs(&self, input: Vec<String>, dimensions: Option<i32>) -> HashMap<String, Value> {
        let mut inputs = HashMap::from([(
            self.input.name.clone(),
            Value::List(input.into_iter().map(serde_json::Value::String).collect()),
        )]);
        if let (Some(parameter), Some(dimensions)) = (&self.dimensions, dimensions) {
            inputs.insert(parameter.name.clone(), Value::Int(dimensions));
        }
        inputs
    }
}

fn output_usage(signature: &Signature) -> Result<Option<usize>> {
    output(signature, &[Dtype::Dict], "Usage")
}

#[async_trait]
impl Adapter for OpenAIEmbeddingsAdapter {
    type Source = Prediction;
    type Options = OpenAIEmbeddingsOptions;
    type Response = EmbeddingCreateResponse;

    fn kind(&self) -> &'static str {
        "openai.embeddings.create"
    }

    async fn adapt(
        &self,
        prediction: Prediction,
        options: Option<Self::Options>,
    ) -> Result<Self::Response> {
        let options = options.unwrap_or_default();
        let Value::Tensor(matrix) =
            decode(result(&prediction, self.output)?, Dtype::Float32).await?
        else {
            return Err(MunaError::Prediction(
                "Embeddings output must be a float32 matrix".into(),
            ));
        };
        let crate::types::TensorData::Float32(values) = matrix.data else {
            return Err(MunaError::Prediction(
                "Embeddings output must be a float32 matrix".into(),
            ));
        };
        let [rows, columns] = matrix.shape.as_slice() else {
            return Err(MunaError::Prediction(
                "Embeddings output must have two dimensions".into(),
            ));
        };
        let rows = usize::try_from(*rows)
            .map_err(|_| MunaError::Prediction("Invalid embedding shape".into()))?;
        let columns = usize::try_from(*columns)
            .map_err(|_| MunaError::Prediction("Invalid embedding shape".into()))?;
        if rows.checked_mul(columns) != Some(values.len()) {
            return Err(MunaError::Prediction(
                "Embedding shape does not match its data".into(),
            ));
        }
        let usage = match self.usage {
            Some(index) => {
                let Value::Dict(usage) = decode(result(&prediction, index)?, Dtype::Dict).await?
                else {
                    return Err(MunaError::Prediction("Invalid embedding usage".into()));
                };
                serde_json::from_value(serde_json::Value::Object(usage))?
            }
            None => EmbeddingUsage {
                prompt_tokens: 0,
                total_tokens: 0,
            },
        };
        let data = (0..rows)
            .map(|index| {
                let vector = &values[index * columns..(index + 1) * columns];
                let embedding = match options.encoding_format {
                    EncodingFormat::Float => EmbeddingData::Float(vector.to_vec()),
                    // OpenAI expects raw little-endian float32 bytes, not an NPZ archive.
                    EncodingFormat::Base64 => EmbeddingData::Base64(
                        BASE64.encode(
                            vector
                                .iter()
                                .flat_map(|value| value.to_le_bytes())
                                .collect::<Vec<_>>(),
                        ),
                    ),
                };
                Embedding {
                    object: "embedding".into(),
                    embedding,
                    index,
                }
            })
            .collect();
        Ok(EmbeddingCreateResponse {
            object: "list".into(),
            model: prediction.tag,
            data,
            usage,
        })
    }
}
