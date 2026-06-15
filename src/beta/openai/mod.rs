/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

mod chat;
mod completions;
mod embeddings;
mod schema;
mod utils;

pub use chat::*;
pub use completions::*;
pub use embeddings::*;
pub use schema::*;

use crate::services::{PredictionService, PredictorService};

/// Experimental OpenAI client.
#[derive(Clone)]
pub struct OpenAIClient {
    /// Chat service.
    pub chat: ChatService,
    /// Embeddings service.
    pub embeddings: EmbeddingService,
}

impl OpenAIClient {

    pub fn new(
        predictors: PredictorService,
        predictions: PredictionService
    ) -> Self {
        let chat = ChatService::new(predictors.clone(), predictions.clone());
        let embeddings = EmbeddingService::new(predictors, predictions);
        Self { chat, embeddings }
    }
}
