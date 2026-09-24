/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

mod messages;
mod models;
mod schema;

pub use messages::*;
pub use models::*;
pub use schema::*;

use crate::services::{PredictionService, PredictorService};

/// Experimental Anthropic client.
#[derive(Clone)]
pub struct AnthropicClient {
    /// Messages service.
    pub messages: MessageService,
    /// Models service.
    pub models: ModelService,
}

impl AnthropicClient {

    pub fn new(
        predictors: PredictorService,
        predictions: PredictionService
    ) -> Self {
        let messages = MessageService::new(predictors.clone(), predictions);
        let models = ModelService::new(predictors);
        Self { messages, models }
    }
}
