/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

mod messages;
mod schema;

pub use messages::*;
pub use schema::*;

use crate::services::{PredictionService, PredictorService};

/// Experimental Anthropic client.
#[derive(Clone)]
pub struct AnthropicClient {
    /// Messages service.
    pub messages: MessageService,
}

impl AnthropicClient {

    pub fn new(
        predictors: PredictorService,
        predictions: PredictionService
    ) -> Self {
        let messages = MessageService::new(predictors, predictions);
        Self { messages }
    }
}
