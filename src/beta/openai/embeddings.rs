/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;

use crate::beta::adapters::{
    Adapter, EmbeddingCreateResponse, OpenAIEmbeddingsAdapter,
    OpenAIEmbeddingsOptions,
};
use crate::client::{MunaError, Result};
use crate::services::{PredictionService, PredictorService};
use crate::types::{Acceleration, Device};

/// Execute predictions through a cached signature adapter.
#[derive(Clone)]
pub struct EmbeddingService {
    predictors: PredictorService,
    predictions: PredictionService,
    cache: Arc<RwLock<HashMap<String, OpenAIEmbeddingsAdapter>>>,
}

impl EmbeddingService {
    pub fn new(predictors: PredictorService, predictions: PredictionService) -> Self {
        Self {
            predictors,
            predictions,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn create(
        &self,
        input: Vec<String>,
        model: &str,
        dimensions: Option<i32>,
        encoding_format: Option<EncodingFormat>,
        acceleration: Option<Acceleration>,
        devices: Option<&[Device]>,
    ) -> Result<EmbeddingCreateResponse> {
        let adapter = self.adapter(model).await?;
        let inputs = adapter.inputs(input, dimensions);
        let prediction = self
            .predictions
            .create(
                model,
                Some(inputs),
                acceleration,
                devices,
                None,
                None,
                None,
                Some(true),
            )
            .await?;
        adapter
            .adapt(
                prediction,
                Some(OpenAIEmbeddingsOptions {
                    encoding_format: encoding_format.unwrap_or_default(),
                }),
            )
            .await
    }

    async fn adapter(&self, model: &str) -> Result<OpenAIEmbeddingsAdapter> {
        if let Some(adapter) = self.cache.read().await.get(model).cloned() {
            return Ok(adapter);
        }
        let predictor = self.predictors.retrieve(model).await?.ok_or_else(|| {
            MunaError::InvalidInput(format!("Predictor {model} could not be found"))
        })?;
        let adapter = OpenAIEmbeddingsAdapter::from_signature(&predictor.signature)?;
        self.cache
            .write()
            .await
            .insert(model.into(), adapter.clone());
        Ok(adapter)
    }
}

pub use crate::beta::adapters::EncodingFormat;
