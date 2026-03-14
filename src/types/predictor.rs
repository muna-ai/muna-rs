/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::fmt;
use serde::{Deserialize, Serialize};
use super::{Parameter, User};

/// Predictor access mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PredictorAccess {
    Private,
    Public,
    Unlisted,
}

/// Predictor status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PredictorStatus {
    Compiling,
    Active,
    Archived,
}

/// Prediction function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Predictor {
    /// Predictor tag.
    pub tag: String,
    /// Predictor owner.
    pub owner: User,
    /// Predictor name.
    pub name: String,
    /// Predictor status.
    pub status: PredictorStatus,
    /// Predictor access mode.
    pub access: PredictorAccess,
    /// Predictor signature.
    pub signature: Signature,
    /// Date created.
    pub created: String,
    /// Predictor description.
    #[serde(default)]
    pub description: Option<String>,
    /// Predictor card.
    #[serde(default)]
    pub card: Option<String>,
    /// Predictor media URL.
    #[serde(default)]
    pub media: Option<String>,
    /// Predictor license URL.
    #[serde(default)]
    pub license: Option<String>,
}

/// Prediction signature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signature {
    /// Prediction inputs.
    pub inputs: Vec<Parameter>,
    /// Prediction outputs.
    pub outputs: Vec<Parameter>,
}

impl fmt::Display for Predictor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match serde_json::to_string_pretty(self) {
            Ok(json) => f.write_str(&json),
            Err(_) => write!(f, "{:?}", self),
        }
    }
}
