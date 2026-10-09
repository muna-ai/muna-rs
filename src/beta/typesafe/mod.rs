/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

mod schema;

pub use schema::*;

use std::collections::HashMap;
use std::sync::Arc;

use indexmap::IndexMap;
use serde_json::Value as Json;
use tokio::sync::RwLock;

use crate::beta::utils::get_parameter;
use crate::client::Result;
use crate::MunaError;
use crate::services::{PredictionService, PredictorService};
use crate::types::{Acceleration, Dtype, Signature, Value};

/// Cached predictor metadata for fast System One evaluation.
struct DelegateInfo {
    state_param_name: String,
    questions_param_name: String,
    response_param_idx: usize,
}

/// Experimental TypeSafe-compatible System One client.
#[derive(Clone)]
pub struct TypeSafeClient {
    predictors: PredictorService,
    predictions: PredictionService,
    cache: Arc<RwLock<HashMap<String, DelegateInfo>>>,
}

impl TypeSafeClient {

    pub fn new(
        predictors: PredictorService,
        predictions: PredictionService
    ) -> Self {
        Self {
            predictors,
            predictions,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Evaluate a state against a map of typed questions.
    ///
    /// # Arguments
    /// * `params` - Decision model tag, state, questions, and acceleration.
    pub async fn system_one(
        &self,
        params: SystemOneCreateParams
    ) -> Result<SystemOneResponse> {
        validate_questions(&params.questions)?;
        let model = params.model.as_str();
        let acceleration = params.acceleration.unwrap_or(Acceleration::LocalAuto);
        {
            let needs_create = !self.cache.read().await.contains_key(model);
            if needs_create {
                let info = self.create_delegate_info(model).await?;
                self.cache
                    .write()
                    .await
                    .entry(model.to_string())
                    .or_insert(info);
            }
        }
        let cache = self.cache.read().await;
        let info = &cache[model];
        let input_map = bind_inputs(params.state, &params.questions, info)?;
        let response_param_idx = info.response_param_idx;
        drop(cache);
        let prediction = self
            .predictions
            .create(model, Some(input_map), Some(acceleration), None, None)
            .await?;
        if let Some(ref error) = prediction.error {
            return Err(MunaError::from_prediction_error(error.clone()));
        }
        let results = prediction
            .results
            .ok_or_else(|| MunaError::Prediction(format!("{model} returned no results")))?;
        let Some(Value::Dict(output)) = results.get(response_param_idx) else {
            return Err(MunaError::Prediction(format!(
                "{model} returned a non-dict System One response"
            )));
        };
        let output: PredictorOutput = serde_json::from_value(Json::Object(output.clone()))
            .map_err(|e| MunaError::Prediction(format!(
                "{model} returned an invalid System One response: {e}"
            )))?;
        Ok(SystemOneResponse {
            model: model.to_string(),
            answers: output.answers,
            usage: output.usage,
        })
    }

    async fn create_delegate_info(&self, tag: &str) -> Result<DelegateInfo> {
        let predictor = self.predictors.retrieve(tag).await?.ok_or_else(|| {
            MunaError::Prediction(format!(
                "{tag} cannot be used with TypeSafe System One API because \
                the predictor could not be found. Check that your access key \
                is valid and that you have access to the predictor."
            ))
        })?;
        resolve_delegate_info(tag, &predictor.signature)
    }
}

fn resolve_delegate_info(
    tag: &str,
    signature: &Signature
) -> Result<DelegateInfo> {
    let required_count = signature
        .inputs
        .iter()
        .filter(|p| !p.optional.unwrap_or(false))
        .count();
    if required_count != 2 {
        return Err(MunaError::Prediction(format!(
            "{tag} cannot be used with TypeSafe System One API because \
            it does not have exactly two required input parameters."
        )));
    }
    let find_input = |dtype: Dtype, dtype_name: &str, denotation: &str| {
        get_parameter(&signature.inputs, &[dtype], Some(denotation))
            .1
            .map(|p| p.name.clone())
            .ok_or_else(|| MunaError::Prediction(format!(
                "{tag} cannot be used with TypeSafe System One API because \
                it has no `{dtype_name}` input with a `{denotation}` denotation."
            )))
    };
    let state_param_name = find_input(
        Dtype::List,
        "list",
        "typesafe.systemone.state"
    )?;
    let questions_param_name = find_input(
        Dtype::Dict,
        "dict",
        "typesafe.systemone.questions"
    )?;
    let response_param_idx = signature
        .outputs
        .iter()
        .position(|param| {
            param.dtype == Some(Dtype::Dict) &&
            param.schema
                .as_ref()
                .and_then(|s| s.get("title"))
                .and_then(|v| v.as_str()) == Some("SystemOneResponse")
        })
        .ok_or_else(|| MunaError::Prediction(format!(
            "{tag} cannot be used with TypeSafe System One API because \
            it has no `SystemOneResponse` output."
        )))?;
    Ok(DelegateInfo { state_param_name, questions_param_name, response_param_idx })
}

fn bind_inputs(
    state: Json,
    questions: &IndexMap<String, Question>,
    info: &DelegateInfo
) -> Result<HashMap<String, Value>> {
    // `list` is the most general state shape: a string or object is a
    // one-item list, an array passes through unchanged.
    let state = match state {
        Json::Array(items) => items,
        Json::String(_) | Json::Object(_) => vec![state],
        _ => return Err(MunaError::InvalidInput(
            "`state` must be a string, object, or array.".into()
        )),
    };
    let Json::Object(questions) = serde_json::to_value(questions)? else {
        return Err(MunaError::InvalidInput("`questions` must be an object.".into()));
    };
    Ok(HashMap::from([
        (info.state_param_name.clone(), Value::List(state)),
        (info.questions_param_name.clone(), Value::Dict(questions)),
    ]))
}

fn validate_questions(questions: &IndexMap<String, Question>) -> Result<()> {
    if questions.is_empty() {
        return Err(MunaError::InvalidInput(
            "`questions` must contain at least one question.".into()
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
mod tests {
    use serde_json::json;

    use super::*;

    fn signature(
        inputs: Json,
        outputs: Json
    ) -> Signature {
        serde_json::from_value(json!({ "inputs": inputs, "outputs": outputs })).unwrap()
    }

    fn decision_signature() -> Signature {
        signature(
            json!([
                { "name": "context", "dtype": "list", "denotation": "typesafe.systemone.state" },
                { "name": "schema", "dtype": "dict", "denotation": "typesafe.systemone.questions" }
            ]),
            json!([
                { "name": "result", "dtype": "dict", "schema": { "title": "SystemOneResponse" } }
            ])
        )
    }

    fn questions(value: Json) -> IndexMap<String, Question> {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn deserializes_wire_request_and_ignores_router_fields() {
        let params: SystemOneCreateParams = serde_json::from_value(json!({
            "model": "@bespokelabs/nimble-9b",
            "state": { "ticket": "I was charged twice." },
            "questions": {
                "route": { "type": "choice", "instructions": "Where should this go?",
                           "criteria": { "billing": "money", "bug": null } },
                "urgency": { "type": "score", "instructions": "How urgent?",
                             "criteria": ["routine", "today", "urgent"] },
                "escalate": { "type": "noul", "instructions": "Escalate?",
                              "criteria": { "true": "Needs a human", "false": "Bot can handle it" } }
            },
            "provider": { "allow_fallbacks": true },
            "session_id": "session-1234"
        })).unwrap();
        assert_eq!(
            params.questions.keys().collect::<Vec<_>>(),
            ["route", "urgency", "escalate"]
        );
        assert!(validate_questions(&params.questions).is_ok());
        let round_trip = serde_json::to_value(&params.questions["escalate"]).unwrap();
        assert_eq!(round_trip["type"], "noul");
        assert_eq!(round_trip["criteria"]["true"], "Needs a human");
    }

    #[test]
    fn parses_mixed_response_in_order() {
        let output: PredictorOutput = serde_json::from_value(json!({
            "answers": {
                "route": { "type": "choice", "choice": "billing", "confidence": 0.99,
                           "probabilities": { "billing": 0.99, "bug": 0.0, "account": 0.01 } },
                "urgency": { "type": "score", "score": 2.1, "confidence": 0.7,
                             "legend": { "0": "routine", "1": "today" },
                             "probabilities": { "0": 0.4, "1": 0.6 } },
                "escalate": { "type": "noul", "noul": 0.94 }
            },
            "usage": { "input_tokens": 412, "output_tokens": 3 }
        })).unwrap();
        assert_eq!(
            output.answers.keys().collect::<Vec<_>>(),
            ["route", "urgency", "escalate"]
        );
        assert!(matches!(&output.answers["escalate"], Answer::Noul(a) if a.noul == 0.94));
        assert!(matches!(&output.answers["route"], Answer::Choice(a) if a.choice == "billing"));
        assert!(matches!(&output.answers["urgency"], Answer::Score(a) if a.probabilities["1"] == 0.6));
        assert_eq!(output.usage.input_tokens, Some(412));
    }

    #[test]
    fn rejects_out_of_range_criteria() {
        let empty = IndexMap::new();
        assert!(matches!(validate_questions(&empty), Err(MunaError::InvalidInput(_))));
        let one_level = questions(json!({
            "q": { "type": "score", "instructions": "Rate.", "criteria": ["only"] }
        }));
        assert!(matches!(validate_questions(&one_level), Err(MunaError::InvalidInput(_))));
        let eleven_levels = questions(json!({
            "q": { "type": "score", "instructions": "Rate.", "criteria": (0..11).collect::<Vec<_>>() }
        }));
        assert!(matches!(validate_questions(&eleven_levels), Err(MunaError::InvalidInput(_))));
        let no_options = questions(json!({
            "q": { "type": "choice", "instructions": "Pick.", "criteria": {} }
        }));
        assert!(matches!(validate_questions(&no_options), Err(MunaError::InvalidInput(_))));
    }

    #[test]
    fn rejects_unknown_noul_criteria_keys() {
        let result = serde_json::from_value::<IndexMap<String, Question>>(json!({
            "q": { "type": "noul", "instructions": "Yes?", "criteria": { "maybe": "unsure" } }
        }));
        assert!(result.is_err());
    }

    #[test]
    fn binds_by_role_not_by_name() {
        let info = resolve_delegate_info("@test/decide", &decision_signature()).unwrap();
        let questions = questions(json!({
            "escalate": { "type": "noul", "instructions": "Escalate?" }
        }));
        let map = bind_inputs(json!("Charged twice."), &questions, &info).unwrap();
        assert!(matches!(
            map.get("context"),
            Some(Value::List(s)) if s == &vec![json!("Charged twice.")]
        ));
        assert!(matches!(
            map.get("schema"),
            Some(Value::Dict(q)) if q["escalate"]["type"] == "noul"
        ));
        assert_eq!(info.response_param_idx, 0);
    }

    #[test]
    fn state_becomes_a_list() {
        let info = resolve_delegate_info("@test/decide", &decision_signature()).unwrap();
        let questions = questions(json!({ "q": { "type": "noul", "instructions": "Yes?" } }));
        let state_of = |state: Json| match bind_inputs(state, &questions, &info).unwrap().remove("context") {
            Some(Value::List(items)) => items,
            _ => panic!("state is not a list"),
        };
        assert_eq!(state_of(json!({ "a": 1 })), vec![json!({ "a": 1 })]);
        assert_eq!(state_of(json!(["x", { "b": 2 }])), vec![json!("x"), json!({ "b": 2 })]);
        assert!(matches!(
            bind_inputs(json!(42), &questions, &info),
            Err(MunaError::InvalidInput(_))
        ));
    }

    #[test]
    fn rejects_predictors_without_the_contract() {
        let missing_questions = signature(
            json!([
                { "name": "context", "dtype": "list", "denotation": "typesafe.systemone.state" },
                { "name": "other", "dtype": "dict" }
            ]),
            json!([{ "name": "result", "dtype": "dict", "schema": { "title": "SystemOneResponse" } }])
        );
        assert!(resolve_delegate_info("@test/a", &missing_questions).is_err());
        let missing_output = signature(
            json!([
                { "name": "context", "dtype": "list", "denotation": "typesafe.systemone.state" },
                { "name": "schema", "dtype": "dict", "denotation": "typesafe.systemone.questions" }
            ]),
            json!([{ "name": "result", "dtype": "dict", "schema": { "title": "Other" } }])
        );
        assert!(resolve_delegate_info("@test/b", &missing_output).is_err());
    }
}
