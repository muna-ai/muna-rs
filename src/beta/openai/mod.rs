/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

mod chat;
mod completions;
mod embeddings;
mod images;
mod inputs;
mod schema;

pub use chat::*;
pub use completions::*;
pub use embeddings::*;
pub use images::*;
pub use inputs::*;
pub use schema::*;

use std::sync::Arc;

use crate::client::Client;
use crate::services::{PredictionService, PredictorService};

/// Experimental OpenAI client.
#[derive(Clone)]
pub struct OpenAIClient {
    /// Chat service.
    pub chat: ChatService,
    /// Embeddings service.
    pub embeddings: EmbeddingService,
    /// Image generation service.
    pub images: ImageService,
}

impl OpenAIClient {

    pub fn new(
        client: Arc<dyn Client>,
        predictors: PredictorService,
        predictions: PredictionService
    ) -> Self {
        let chat = ChatService::new(client, predictors.clone(), predictions.clone());
        let embeddings = EmbeddingService::new(predictors.clone(), predictions.clone());
        let images = ImageService::new(predictors, predictions);
        Self { chat, embeddings, images }
    }
}
