/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use serde::{Deserialize, Serialize};

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
    pub object: String,
    pub embedding: EmbeddingData,
    pub index: usize,
}

/// Usage information for an embedding request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u64,
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
