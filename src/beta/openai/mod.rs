/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

mod embeddings;
mod schema;
mod utils;

pub use embeddings::*;
pub use schema::*;

use crate::services::{PredictionService, PredictorService};

/// Experimental OpenAI client.
#[derive(Clone)]
pub struct OpenAIClient {
    /// Embeddings service.
    pub embeddings: EmbeddingService,
}

impl OpenAIClient {

    pub fn new(
        predictors: PredictorService,
        predictions: PredictionService
    ) -> Self {
        Self {
            embeddings: EmbeddingService::new(predictors, predictions),
        }
    }
}
