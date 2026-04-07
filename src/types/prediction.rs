/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use serde::{Deserialize, Serialize};

use super::RemoteValue;
use super::Value;

/// Prediction acceleration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Acceleration {
    /// Run locally with automatic hardware selection.
    #[serde(rename = "local_auto")]
    LocalAuto,
    /// Run locally on the CPU.
    #[serde(rename = "local_cpu")] 
    LocalCpu,
    /// Run locally on the GPU.
    #[serde(rename = "local_gpu")] 
    LocalGpu,
    /// Run locally on the NPU.
    #[serde(rename = "local_npu")] 
    LocalNpu,
    /// Run remotely with automatic hardware selection.
    #[serde(rename = "remote_auto")]
    RemoteAuto,
    /// Run remotely on the CPU.
    #[serde(rename = "remote_cpu")]
    RemoteCpu,
    /// Run remotely on an NVIDIA A10 GPU.
    #[serde(rename = "remote_a10")]
    RemoteA10,
    /// Run remotely on an NVIDIA L40S GPU.
    #[serde(rename = "remote_l40s")]
    RemoteL40S,
    /// Run remotely on an NVIDIA A100 GPU.
    #[serde(rename = "remote_a100")]
    RemoteA100,
    /// Run remotely on an NVIDIA H200 GPU.
    #[serde(rename = "remote_h200")]
    RemoteH200,
    /// Run remotely on an NVIDIA B200 GPU.
    #[serde(rename = "remote_b200")]
    RemoteB200,
    /// Run remotely on an AMD MI350X GPU.
    #[serde(rename = "remote_mi350x")]
    RemoteMI350X,
    /// Run remotely on an AMD MI355X GPU.
    #[serde(rename = "remote_mi355x")]
    RemoteMI355X,
    /// Adaptive acceleration from an opaque string identifier.
    #[serde(untagged)]
    Adaptive(String),
}

/// Prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    /// Prediction ID.
    pub id: String,
    /// Predictor tag.
    pub tag: String,
    /// Date created.
    pub created: String,
    /// Prediction configuration token.
    #[serde(default)]
    pub configuration: Option<String>,
    /// Predictor resources.
    #[serde(default)]
    pub resources: Option<Vec<PredictionResource>>,
    /// Prediction results.
    #[serde(skip)]
    pub results: Option<Vec<Value>>,
    /// Prediction latency in milliseconds.
    #[serde(default)]
    pub latency: Option<f64>,
    /// Prediction error.
    #[serde(default)]
    pub error: Option<String>,
    /// Prediction logs.
    #[serde(default)]
    pub logs: Option<String>,
}

/// Prediction resource.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResource {
    /// Resource type.
    #[serde(rename = "type")]
    pub kind: String,
    /// Resource URL.
    pub url: String,
    /// Resource name.
    #[serde(default)]
    pub name: Option<String>,
}

/// Remote prediction result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemotePrediction {
    /// Prediction ID.
    pub id: String,
    /// Predictor tag.
    pub tag: String,
    /// Date created.
    pub created: String,
    /// Prediction results.
    #[serde(default)]
    pub results: Option<Vec<RemoteValue>>,
    /// Prediction latency in milliseconds.
    #[serde(default)]
    pub latency: Option<f64>,
    /// Prediction error.
    #[serde(default)]
    pub error: Option<String>,
    /// Prediction logs.
    #[serde(default)]
    pub logs: Option<String>,
}

/// Remote prediction SSE event.
#[derive(Debug, Serialize, Deserialize)]
pub struct RemotePredictionEvent {
    pub event: String,
    pub data: RemotePrediction,
}
