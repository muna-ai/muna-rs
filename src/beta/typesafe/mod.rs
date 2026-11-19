/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;

use crate::beta::adapters::{Adapter, TypeSafeSystemOneAdapter, TypeSafeSystemOneOptions};
use crate::client::{MunaError, Result};
use crate::services::{PredictionService, PredictorService};

mod schema;
pub use schema::*;

/// Execute predictions through a cached signature adapter.
#[derive(Clone)]
pub struct TypeSafeClient {
    predictors: PredictorService,
    predictions: PredictionService,
    cache: Arc<RwLock<HashMap<String, TypeSafeSystemOneAdapter>>>,
}

impl TypeSafeClient {
    pub fn new(predictors: PredictorService, predictions: PredictionService) -> Self {
        Self {
            predictors,
            predictions,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn system_one(&self, params: SystemOneCreateParams) -> Result<SystemOneResponse> {
        let adapter = self.adapter(&params.model).await?;
        let inputs = adapter.inputs(params.state, &params.questions)?;
        let prediction = self
            .predictions
            .create(
                &params.model,
                Some(inputs),
                params.acceleration,
                params.devices.as_deref(),
                None,
                None,
                None,
                Some(true),
            )
            .await?;
        adapter
            .adapt(
                prediction,
                Some(TypeSafeSystemOneOptions {
                    questions: Some(params.questions),
                }),
            )
            .await
    }

    async fn adapter(&self, model: &str) -> Result<TypeSafeSystemOneAdapter> {
        if let Some(adapter) = self.cache.read().await.get(model).cloned() {
            return Ok(adapter);
        }
        let predictor = self.predictors.retrieve(model).await?.ok_or_else(|| {
            MunaError::InvalidInput(format!("Predictor {model} could not be found"))
        })?;
        let adapter =
            TypeSafeSystemOneAdapter::from_signature(&predictor.signature).ok_or_else(|| {
                MunaError::InvalidInput(format!(
                    "Predictor {model} does not support TypeSafe System One"
                ))
            })?;
        self.cache
            .write()
            .await
            .insert(model.into(), adapter.clone());
        Ok(adapter)
    }
}
