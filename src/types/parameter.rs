/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use serde::{Deserialize, Serialize};

use super::Dtype;

/// Prediction parameter.
/// Describes a value that is consumed or produced by a predictor.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Parameter {
    /// Parameter name.
    pub name: String,
    /// Parameter data type.
    #[serde(default)]
    pub dtype: Option<Dtype>,
    /// Parameter description.
    #[serde(default)]
    pub description: Option<String>,
    /// Parameter denotation for specialized data types.
    #[serde(default)]
    pub denotation: Option<String>,
    /// Parameter is optional.
    #[serde(default)]
    pub optional: Option<bool>,
    /// Parameter value choices for enumeration parameters.
    #[serde(default)]
    pub enumeration: Option<Vec<EnumerationMember>>,
    /// Parameter JSON schema (only populated for `list` and `dict` parameters).
    #[serde(default)]
    pub schema: Option<serde_json::Map<String, serde_json::Value>>,
    /// Parameter minimum value.
    #[serde(default)]
    pub min: Option<f64>,
    /// Parameter maximum value.
    #[serde(default)]
    pub max: Option<f64>,
    /// Audio sample rate in Hertz.
    #[serde(default)]
    pub sample_rate: Option<u32>,
}

/// Enumeration member value.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EnumerationValue {
    String(String),
    Int(i64),
}

/// Prediction parameter enumeration member.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnumerationMember {
    /// Enumeration member name.
    pub name: String,
    /// Enumeration member value.
    pub value: EnumerationValue,
}
