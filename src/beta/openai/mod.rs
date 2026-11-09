/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

mod audio;
mod chat;
mod completions;
mod embeddings;
mod images;
mod inputs;
mod models;
mod schema;
mod speech;
mod transcriptions;

pub use audio::*;
pub use chat::*;
pub use completions::*;
pub use embeddings::*;
pub use images::*;
pub use inputs::*;
pub use models::*;
pub use schema::*;
pub use speech::*;
pub use transcriptions::*;

use crate::services::{PredictionService, PredictorService};

/// Experimental OpenAI client.
#[derive(Clone)]
pub struct OpenAIClient {
    /// Audio transcription and speech generation services.
    pub audio: AudioService,
    /// Chat service.
    pub chat: ChatService,
    /// Embeddings service.
    pub embeddings: EmbeddingService,
    /// Image generation service.
    pub images: ImageService,
    /// Models service.
    pub models: ModelService,
}

impl OpenAIClient {

    pub fn new(
        predictors: PredictorService,
        predictions: PredictionService
    ) -> Self {
        let audio = AudioService::new(predictors.clone(), predictions.clone());
        let chat = ChatService::new(predictors.clone(), predictions.clone());
        let embeddings = EmbeddingService::new(predictors.clone(), predictions.clone());
        let images = ImageService::new(predictors.clone(), predictions);
        let models = ModelService::new(predictors);
        Self { audio, chat, embeddings, images, models }
    }
}
