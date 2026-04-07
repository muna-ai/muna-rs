/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use futures_core::Stream;

use crate::client::{MunaClient, Result};
use crate::types::{Acceleration, Prediction, Value};
use super::local::LocalPredictionService;
use super::remote::RemotePredictionService;

/// Make predictions.
#[derive(Clone)]
pub struct PredictionService {
    local: Arc<LocalPredictionService>,
    remote: Arc<RemotePredictionService>,
}

impl PredictionService {

    pub fn new(client: Arc<MunaClient>) -> Self {
        Self {
            local: Arc::new(LocalPredictionService::new(client.clone())),
            remote: Arc::new(RemotePredictionService::new(client)),
        }
    }

    /// Create a prediction.
    pub async fn create(
        &self,
        tag: &str,
        inputs: Option<HashMap<String, Value>>,
        acceleration: Option<Acceleration>,
        client_id: Option<String>,
        configuration_id: Option<String>,
    ) -> Result<Prediction> {
        let is_local = inputs.is_none() || match acceleration.as_ref() {
            Some(Acceleration::LocalAuto | Acceleration::LocalCpu | Acceleration::LocalGpu | Acceleration::LocalNpu) => true,
            Some(Acceleration::Adaptive(s)) => s.starts_with("local_"),
            None => true,
            _ => false,
        };
        if is_local {
            self.local.create(tag, inputs, acceleration, client_id, configuration_id).await
        } else {
            self.remote.create(tag, &inputs.unwrap(), acceleration).await
        }
    }

    /// Stream a prediction.
    pub async fn stream(
        &self,
        tag: &str,
        inputs: HashMap<String, Value>,
        acceleration: Option<Acceleration>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Prediction>> + Send>>> {
        let is_local = match acceleration.as_ref() {
            Some(Acceleration::LocalAuto | Acceleration::LocalCpu | Acceleration::LocalGpu | Acceleration::LocalNpu) => true,
            Some(Acceleration::Adaptive(s)) => s.starts_with("local_"),
            None => true,
            _ => false,
        };
        if is_local {
            self.local.stream(tag, inputs, acceleration).await
        } else {
            self.remote.stream(tag, &inputs, acceleration).await
        }
    }

    /// Delete a predictor that is loaded in memory.
    pub async fn delete(&self, tag: &str) -> Result<bool> {
        self.local.delete(tag).await
    }
}
