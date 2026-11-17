/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::collections::HashMap;
use std::sync::Arc;

use futures_util::StreamExt;
use tokio::sync::RwLock;

use crate::beta::adapters::{Adapter, AnthropicMessagesAdapter, Message, MessageCreateParams};
use crate::client::{MunaError, Result};
use crate::services::{PredictionService, PredictorService};

/// Execute predictions through a cached signature adapter.
#[derive(Clone)]
pub struct MessageService {
    predictors: PredictorService,
    predictions: PredictionService,
    cache: Arc<RwLock<HashMap<String, AnthropicMessagesAdapter>>>,
}

impl MessageService {
    pub fn new(predictors: PredictorService, predictions: PredictionService) -> Self {
        Self {
            predictors,
            predictions,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn create(&self, params: MessageCreateParams) -> Result<Message> {
        let mut events = self.stream(params).await?;
        let mut message = None;
        while let Some(event) = events.next().await {
            crate::beta::adapters::messages::apply_event(&mut message, &event?);
        }
        message
            .ok_or_else(|| MunaError::Prediction("Message predictor did not return outputs".into()))
    }

    pub async fn stream(&self, params: MessageCreateParams) -> Result<MessageEventStream> {
        let adapter = self.adapter(&params.model).await?;
        let input_adapter = adapter.clone();
        let request = params.clone();
        let inputs = tokio::task::spawn_blocking(move || input_adapter.inputs(&request))
            .await
            .map_err(|error| {
                MunaError::Prediction(format!("Failed to prepare message inputs: {error}"))
            })??;
        let predictions = self
            .predictions
            .stream(
                &params.model,
                inputs,
                params.acceleration,
                params.devices.as_deref(),
                Some(true),
            )
            .await?;
        adapter.adapt(predictions, None).await
    }

    async fn adapter(&self, model: &str) -> Result<AnthropicMessagesAdapter> {
        if let Some(adapter) = self.cache.read().await.get(model).cloned() {
            return Ok(adapter);
        }
        let predictor = self.predictors.retrieve(model).await?.ok_or_else(|| {
            MunaError::InvalidInput(format!("Predictor {model} could not be found"))
        })?;
        let adapter = AnthropicMessagesAdapter::from_signature(&predictor.signature)?;
        self.cache
            .write()
            .await
            .insert(model.into(), adapter.clone());
        Ok(adapter)
    }
}

pub use crate::beta::adapters::MessageEventStream;
