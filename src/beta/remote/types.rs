/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use serde::{Deserialize, Serialize};
use crate::types::Dtype;

/// Serialized remote value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteValue {
    /// Data URL or base64 data URI.
    pub data: Option<String>,
    /// Value data type.
    pub dtype: Dtype,
}

/// Remote prediction result.
#[derive(Debug, Clone, Deserialize)]
pub struct RemotePrediction {
    pub id: String,
    pub tag: String,
    pub created: String,
    #[serde(default)]
    pub results: Option<Vec<RemoteValue>>,
    #[serde(default)]
    pub latency: Option<f64>,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub logs: Option<String>,
}

/// Remote prediction SSE event.
#[derive(Debug, Deserialize)]
pub struct RemotePredictionEvent {
    pub event: String,
    pub data: RemotePrediction,
}
