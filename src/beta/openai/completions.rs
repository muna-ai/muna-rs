/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::collections::HashMap;
use std::sync::Arc;

use futures_util::StreamExt;
use tokio::sync::RwLock;

use crate::beta::adapters::{
    Adapter, ChatCompletion, ChatCompletionCreateParams, OpenAICompletionsAdapter,
    OpenAICompletionsOptions,
};
use crate::client::{MunaError, Result};
use crate::services::{PredictionService, PredictorService};

/// Execute predictions through a cached signature adapter.
#[derive(Clone)]
pub struct ChatCompletionService {
    predictors: PredictorService,
    predictions: PredictionService,
    cache: Arc<RwLock<HashMap<String, OpenAICompletionsAdapter>>>,
}

impl ChatCompletionService {
    pub fn new(predictors: PredictorService, predictions: PredictionService) -> Self {
        Self {
            predictors,
            predictions,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn create(&self, params: ChatCompletionCreateParams) -> Result<ChatCompletion> {
        let mut chunks = self
            .stream_with_options(
                params,
                OpenAICompletionsOptions {
                    include_usage: true,
                },
            )
            .await?;
        let mut result = Vec::new();
        while let Some(chunk) = chunks.next().await {
            result.push(chunk?);
        }
        crate::beta::adapters::completions::merge_chunks(result)
    }

    pub async fn stream(&self, params: ChatCompletionCreateParams) -> Result<ChatCompletionStream> {
        // This Rust API has always included usage on streams; explicit adapter
        // callers can choose a response without the final usage-only chunk.
        self.stream_with_options(
            params,
            OpenAICompletionsOptions {
                include_usage: true,
            },
        )
        .await
    }

    async fn stream_with_options(
        &self,
        params: ChatCompletionCreateParams,
        options: OpenAICompletionsOptions,
    ) -> Result<ChatCompletionStream> {
        let adapter = self.adapter(&params.model).await?;
        let input_adapter = adapter.clone();
        let request = params.clone();
        let inputs = tokio::task::spawn_blocking(move || input_adapter.inputs(&request))
            .await
            .map_err(|error| {
                MunaError::Prediction(format!("Failed to prepare chat inputs: {error}"))
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
        adapter.adapt(predictions, Some(options)).await
    }

    async fn adapter(&self, model: &str) -> Result<OpenAICompletionsAdapter> {
        if let Some(adapter) = self.cache.read().await.get(model).cloned() {
            return Ok(adapter);
        }
        let predictor = self.predictors.retrieve(model).await?.ok_or_else(|| {
            MunaError::InvalidInput(format!("Predictor {model} could not be found"))
        })?;
        let adapter =
            OpenAICompletionsAdapter::from_signature(&predictor.signature).ok_or_else(|| {
                MunaError::InvalidInput(format!(
                    "Predictor {model} does not support chat completions"
                ))
            })?;
        self.cache
            .write()
            .await
            .insert(model.into(), adapter.clone());
        Ok(adapter)
    }
}

pub use crate::beta::adapters::ChatCompletionStream;
