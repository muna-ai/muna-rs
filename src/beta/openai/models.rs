/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! OpenAI `Model` objects, derived from predictor signatures.

use serde::{Deserialize, Serialize};

use crate::beta::utils::{rfc3339_to_unix, split_tag};
use crate::client::{MunaError, Result};
use crate::services::PredictorService;
use crate::types::Signature;

/// OpenAI `Model` object: exactly the four fields OpenAI documents.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Model {
    /// Model identifier (the predictor tag).
    pub id: String,
    /// Object type, always `model`.
    pub object: String,
    /// Creation time in Unix seconds.
    pub created: u64,
    /// Model owner, the tag's `@owner`.
    pub owned_by: String,
}

impl Model {

    /// Build the `Model` object for a predictor.
    ///
    /// The signature is taken for parity with the Anthropic projection
    /// (which reads limits from it); OpenAI's object carries none, so it is
    /// unused today. `created` is the Unix time the model became available.
    pub fn from_signature(
        tag: &str,
        _signature: &Signature,
        created: Option<u64>
    ) -> Self {
        let (owner, _) = split_tag(tag);
        Self {
            id: tag.to_string(),
            object: "model".into(),
            created: created.unwrap_or_default(),
            owned_by: owner.to_string(),
        }
    }
}

/// Describe models.
#[derive(Clone)]
pub struct ModelService {
    predictors: PredictorService,
}

impl ModelService {

    pub fn new(predictors: PredictorService) -> Self {
        Self { predictors }
    }

    /// Retrieve a model.
    ///
    /// # Arguments
    /// * `model` - Predictor tag.
    pub async fn retrieve(&self, model: &str) -> Result<Model> {
        let predictor = self.predictors.retrieve(model).await?.ok_or_else(|| {
            MunaError::Prediction(format!(
                "{model} cannot be retrieved with the OpenAI models API because \
                the predictor could not be found. Check that your access key \
                is valid and that you have access to the predictor."
            ))
        })?;
        Ok(Model::from_signature(
            &predictor.tag,
            &predictor.signature,
            rfc3339_to_unix(&predictor.created)
        ))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn model_is_exactly_the_openai_object() {
        let signature: Signature = serde_json::from_value(json!({
            "inputs": [{ "name": "messages", "dtype": "list", "contextLength": 262144 }],
            "outputs": []
        })).unwrap();
        let model = Model::from_signature("@qwen/qwen-3.8-27b", &signature, Some(1_789_936_000));
        let json = serde_json::to_value(&model).unwrap();
        assert_eq!(json, json!({
            "id": "@qwen/qwen-3.8-27b",
            "object": "model",
            "created": 1_789_936_000,
            "owned_by": "qwen"
        }));
        let unknown = Model::from_signature("@a/b", &signature, None);
        assert_eq!(unknown.created, 0);
    }
}
