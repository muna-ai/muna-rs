/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::sync::Arc;

use crate::client::Client;
use crate::services::{PredictionService, PredictorService};

use super::anthropic::AnthropicClient;
use super::openai::OpenAIClient;

/// Client for incubating features.
#[derive(Clone)]
pub struct BetaClient {
    /// Anthropic-compatible client.
    pub anthropic: AnthropicClient,
    /// OpenAI-compatible client.
    pub openai: OpenAIClient,
}

impl BetaClient {

    pub fn new(
        client: Arc<dyn Client>,
        predictors: PredictorService,
        predictions: PredictionService,
    ) -> Self {
        let anthropic = AnthropicClient::new(predictors.clone(), predictions.clone());
        let openai = OpenAIClient::new(client, predictors, predictions);
        Self { anthropic, openai }
    }
}
