/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value as Json;

use crate::types::Acceleration;

/// Request to evaluate a state against typed questions.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct SystemOneCreateParams {
    /// Decision model tag.
    pub model: String,
    /// Content the questions refer to: a string, object, or array.
    pub state: Json,
    /// Questions keyed by caller-chosen ids. Ids are never sent to the model.
    pub questions: IndexMap<String, Question>,
    /// Prediction acceleration. Not a wire field.
    #[serde(skip)]
    pub acceleration: Option<Acceleration>,
}

/// Typed question.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
    /// Yes/no question, answered with the probability of yes.
    Noul {
        instructions: Json,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        criteria: Option<NoulCriteria>,
    },
    /// Pick one of the named options.
    Choice {
        instructions: Json,
        criteria: IndexMap<String, Json>,
    },
    /// Rate on an ordered scale of levels, low to high.
    Score {
        instructions: Json,
        criteria: Vec<Json>,
    },
}

/// Descriptions of what a yes and a no mean.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoulCriteria {
    #[serde(default, rename = "true", skip_serializing_if = "Option::is_none")]
    pub yes: Option<Json>,
    #[serde(default, rename = "false", skip_serializing_if = "Option::is_none")]
    pub no: Option<Json>,
}

/// Typed answer, carrying the same `type` as its question.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Noul(NoulAnswer),
    Choice(ChoiceAnswer),
    Score(ScoreAnswer),
}

/// Answer to a noul question.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoulAnswer {
    /// Probability of yes, from 0 to 1.
    pub noul: f64,
}

/// Answer to a choice question.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChoiceAnswer {
    /// Option with the highest probability.
    pub choice: String,
    /// Confidence in the selection, from 0 to 1.
    pub confidence: f64,
    /// Every option and its probability.
    pub probabilities: IndexMap<String, f64>,
}

/// Answer to a score question.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreAnswer {
    /// Probability-weighted average of the levels.
    pub score: f64,
    /// Confidence in the score, from 0 to 1.
    pub confidence: f64,
    /// Level index (string on the wire) to the question's level description.
    pub legend: IndexMap<String, Json>,
    /// Level index (string on the wire) to its probability.
    pub probabilities: IndexMap<String, f64>,
}

/// Token usage for a System One request.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SystemOneUsage {
    /// Input token count.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    /// Output token count.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
}

/// One answer per question, keyed by the question ids.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemOneResponse {
    /// Model that answered the request.
    pub model: String,
    /// Answers keyed by question id, in question order.
    pub answers: IndexMap<String, Answer>,
    /// Token usage.
    #[serde(default)]
    pub usage: SystemOneUsage,
}

/// Output dict as returned by the predictor. The client adds `model`.
#[derive(Deserialize)]
pub(crate) struct PredictorOutput {
    pub answers: IndexMap<String, Answer>,
    #[serde(default)]
    pub usage: SystemOneUsage,
}
