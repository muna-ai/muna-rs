/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use super::{SpeechService, TranscriptionService};

pub use crate::beta::adapters::audio::{Audio, AudioInput};
use crate::services::{PredictionService, PredictorService};

/// OpenAI-compatible audio adapters.
#[derive(Clone)]
pub struct AudioService {
    pub speech: SpeechService,
    pub transcriptions: TranscriptionService,
}

impl AudioService {
    pub fn new(predictors: PredictorService, predictions: PredictionService) -> Self {
        Self {
            speech: SpeechService::new(predictors.clone(), predictions.clone()),
            transcriptions: TranscriptionService::new(predictors, predictions),
        }
    }
}
