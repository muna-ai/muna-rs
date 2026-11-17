/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;

use crate::beta::adapters::{
    Adapter, AudioInput, OpenAITranscriptionsAdapter,
    OpenAITranscriptionsOptions, Transcription,
};
use crate::client::{MunaError, Result};
use crate::services::{PredictionService, PredictorService};
use crate::types::{Acceleration, Device};

/// Parameters for non-streaming transcription. Encoded files are decoded/resampled locally.
#[derive(Debug, Clone, Default)]
pub struct TranscriptionCreateParams {
    pub file: AudioInput,
    pub model: String,
    pub language: Option<String>,
    pub prompt: Option<String>,
    /// Streaming transcription is not currently supported.
    pub stream: bool,
    pub temperature: Option<f64>,
    pub acceleration: Option<Acceleration>,
    pub devices: Option<Vec<Device>>,
}

/// Execute predictions through a cached signature adapter.
#[derive(Clone)]
pub struct TranscriptionService {
    predictors: PredictorService,
    predictions: PredictionService,
    cache: Arc<RwLock<HashMap<String, OpenAITranscriptionsAdapter>>>,
}

impl TranscriptionService {
    pub fn new(predictors: PredictorService, predictions: PredictionService) -> Self {
        Self {
            predictors,
            predictions,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn create(&self, params: TranscriptionCreateParams) -> Result<Transcription> {
        if params.stream {
            return Err(MunaError::InvalidInput(
                "Streaming transcription is not supported".into(),
            ));
        }
        if params
            .temperature
            .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
        {
            return Err(MunaError::InvalidInput(
                "Transcription temperature must be between 0 and 1.".into(),
            ));
        }
        let adapter = self.adapter(&params.model).await?;
        let inputs = adapter.inputs(
            params.file,
            params.language,
            params.prompt,
            params.temperature,
        )?;
        let duration = adapter.duration(&inputs)?;
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
                Some(OpenAITranscriptionsOptions {
                    duration: Some(duration),
                }),
            )
            .await
    }

    async fn adapter(&self, model: &str) -> Result<OpenAITranscriptionsAdapter> {
        if let Some(adapter) = self.cache.read().await.get(model).cloned() {
            return Ok(adapter);
        }
        let predictor = self.predictors.retrieve(model).await?.ok_or_else(|| {
            MunaError::InvalidInput(format!("Predictor {model} could not be found"))
        })?;
        let adapter = OpenAITranscriptionsAdapter::from_signature(&predictor.signature)?;
        self.cache
            .write()
            .await
            .insert(model.into(), adapter.clone());
        Ok(adapter)
    }
}
