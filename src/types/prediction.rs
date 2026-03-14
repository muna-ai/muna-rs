/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use serde::{Deserialize, Serialize};

use super::Value;

/// Prediction acceleration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Acceleration {
    #[serde(rename = "local_auto")]   LocalAuto,
    #[serde(rename = "local_cpu")]    LocalCpu,
    #[serde(rename = "local_gpu")]    LocalGpu,
    #[serde(rename = "local_npu")]    LocalNpu,
    #[serde(rename = "remote_auto")]  RemoteAuto,
    #[serde(rename = "remote_cpu")]   RemoteCpu,
    #[serde(rename = "remote_a10")]   RemoteA10,
    #[serde(rename = "remote_a40")]   RemoteA40,
    #[serde(rename = "remote_a100")]  RemoteA100,
    #[serde(rename = "remote_h200")]  RemoteH200,
    #[serde(rename = "remote_b200")]  RemoteB200,
    #[serde(untagged)]                Adaptive(String),
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
