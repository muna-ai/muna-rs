/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::collections::HashMap;

use async_trait::async_trait;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value as Json;

use crate::client::{MunaError, Result};
use crate::types::{Acceleration, Device, Dtype, Parameter, Prediction, Signature, Value};

use super::{
    utils::{decode, output, parameter, result, validate_inputs},
    Adapter,
};

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
    /// Local compute devices. Not a wire field.
    #[serde(skip)]
    pub devices: Option<Vec<Device>>,
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

#[derive(Debug, Clone, Default)]
pub struct TypeSafeSystemOneOptions {
    pub questions: Option<IndexMap<String, Question>>,
}

#[derive(Debug, Clone)]
pub struct TypeSafeSystemOneAdapter {
    pub state: Parameter,
    pub questions: Parameter,
    pub output: usize,
}

impl TypeSafeSystemOneAdapter {
    /// Returns `None` when the signature is incompatible or ambiguous.
    pub fn from_signature(signature: &Signature) -> Option<Self> {
        // State input.
        let state = parameter(
            &signature.inputs,
            &[Dtype::List],
            Some("typesafe.systemone.state"),
        )
        .ok()??;
        // Questions input.
        let questions = parameter(
            &signature.inputs,
            &[Dtype::Dict],
            Some("typesafe.systemone.questions"),
        )
        .ok()??;
        validate_inputs(signature, &[Some(&state), Some(&questions)]).ok()?;
        // Typed response output.
        let output = output(signature, &[Dtype::Dict], "SystemOneResponse").ok()??;
        Some(Self {
            state,
            questions,
            output,
        })
    }

    pub fn inputs(
        &self,
        state: Json,
        questions: &IndexMap<String, Question>,
    ) -> Result<HashMap<String, Value>> {
        validate_questions(questions)?;
        bind_inputs(state, questions, self)
    }
}

#[async_trait]
impl Adapter for TypeSafeSystemOneAdapter {
    type Source = Prediction;
    type Options = TypeSafeSystemOneOptions;
    type Response = SystemOneResponse;

    fn kind(&self) -> &'static str {
        "typesafe.system_one"
    }

    async fn adapt(
        &self,
        prediction: Prediction,
        options: Option<Self::Options>,
    ) -> Result<Self::Response> {
        let Value::Dict(value) = decode(result(&prediction, self.output)?, Dtype::Dict).await?
        else {
            return Err(MunaError::Prediction(
                "System One response must be a dictionary".into(),
            ));
        };
        let output: PredictorOutput = serde_json::from_value(Json::Object(value))?;
        let answers = match options.unwrap_or_default().questions {
            Some(questions) => order_answers(&questions, output.answers),
            None => output.answers,
        };
        Ok(SystemOneResponse {
            model: prediction.tag,
            answers,
            usage: output.usage,
        })
    }
}

fn bind_inputs(
    state: Json,
    questions: &IndexMap<String, Question>,
    info: &TypeSafeSystemOneAdapter,
) -> Result<HashMap<String, Value>> {
    // `list` is the most general state shape: a string or object is a
    // one-item list, an array passes through unchanged.
    let state = match state {
        Json::Array(items) => items,
        Json::String(_) | Json::Object(_) => vec![state],
        _ => {
            return Err(MunaError::InvalidInput(
                "`state` must be a string, object, or array.".into(),
            ))
        }
    };
    let Json::Object(questions) = serde_json::to_value(questions)? else {
        return Err(MunaError::InvalidInput(
            "`questions` must be an object.".into(),
        ));
    };
    Ok(HashMap::from([
        (info.state.name.clone(), Value::List(state)),
        (info.questions.name.clone(), Value::Dict(questions)),
    ]))
}

fn order_answers(
    questions: &IndexMap<String, Question>,
    mut answers: IndexMap<String, Answer>,
) -> IndexMap<String, Answer> {
    // Predictor dicts cross the FFI as `serde_json::Map`, which sorts keys,
    // so restore the caller's question order. Unknown ids keep their place
    // after the known ones.
    let mut ordered: IndexMap<String, Answer> = questions
        .keys()
        .filter_map(|id| answers.shift_remove_entry(id))
        .collect();
    ordered.extend(answers);
    ordered
}

fn validate_questions(questions: &IndexMap<String, Question>) -> Result<()> {
    if questions.is_empty() {
        return Err(MunaError::InvalidInput(
            "`questions` must contain at least one question.".into(),
        ));
    }
    for (id, question) in questions {
        match question {
            Question::Choice { criteria, .. } if !(1..=255).contains(&criteria.len()) => {
                return Err(MunaError::InvalidInput(format!(
                    "`questions.{id}.criteria` must have between 1 and 255 options."
                )));
            }
            Question::Score { criteria, .. } if !(2..=10).contains(&criteria.len()) => {
                return Err(MunaError::InvalidInput(format!(
                    "`questions.{id}.criteria` must have between 2 and 10 levels."
                )));
            }
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "tests/system_one.rs"]
mod tests;
