/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;

use crate::beta::adapters::{
    Adapter, OpenAISpeechAdapter, OpenAISpeechOptions, SpeechCreateResponse,
    SpeechResponseFormat, SpeechStreamFormat,
};
use crate::client::{MunaError, Result};
use crate::services::{PredictionService, PredictorService};
use crate::types::{Acceleration, Device};

/// Parameters for speech generation. Binary audio is currently the only supported stream format.
#[derive(Debug, Clone, Default)]
pub struct SpeechCreateParams {
    pub input: String,
    pub model: String,
    pub voice: String,
    pub response_format: Option<SpeechResponseFormat>,
    pub speed: Option<f64>,
    pub stream_format: Option<SpeechStreamFormat>,
    pub acceleration: Option<Acceleration>,
    pub devices: Option<Vec<Device>>,
}

/// Execute predictions through a cached signature adapter.
#[derive(Clone)]
pub struct SpeechService {
    predictors: PredictorService,
    predictions: PredictionService,
    cache: Arc<RwLock<HashMap<String, OpenAISpeechAdapter>>>,
}

impl SpeechService {
    pub fn new(predictors: PredictorService, predictions: PredictionService) -> Self {
        Self {
            predictors,
            predictions,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn create(&self, params: SpeechCreateParams) -> Result<SpeechCreateResponse> {
        if params.stream_format == Some(SpeechStreamFormat::Sse) {
            return Err(MunaError::InvalidInput(
                "Only audio speech output is supported".into(),
            ));
        }
        if params
            .speed
            .is_some_and(|speed| !speed.is_finite() || speed <= 0.0 || speed > f32::MAX as f64)
        {
            return Err(MunaError::InvalidInput(
                "Speech speed must be finite and positive.".into(),
            ));
        }
        let adapter = self.adapter(&params.model).await?;
        let inputs = adapter.inputs(params.input, params.voice, params.speed);
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
                Some(OpenAISpeechOptions {
                    response_format: params.response_format.unwrap_or_default(),
                    stream_format: params.stream_format.unwrap_or_default(),
                }),
            )
            .await
    }

    async fn adapter(&self, model: &str) -> Result<OpenAISpeechAdapter> {
        if let Some(adapter) = self.cache.read().await.get(model).cloned() {
            return Ok(adapter);
        }
        let predictor = self.predictors.retrieve(model).await?.ok_or_else(|| {
            MunaError::InvalidInput(format!("Predictor {model} could not be found"))
        })?;
        let adapter = OpenAISpeechAdapter::from_signature(&predictor.signature)?;
        self.cache
            .write()
            .await
            .insert(model.into(), adapter.clone());
        Ok(adapter)
    }
}
