/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use crate::services::{PredictionService, PredictorService};

use super::anthropic::AnthropicClient;
use super::openai::OpenAIClient;
use super::typesafe::TypeSafeClient;

/// Client for incubating features.
#[derive(Clone)]
pub struct BetaClient {
    /// Anthropic-compatible client.
    pub anthropic: AnthropicClient,
    /// OpenAI-compatible client.
    pub openai: OpenAIClient,
    /// TypeSafe-compatible client.
    pub typesafe: TypeSafeClient,
}

impl BetaClient {

    pub fn new(
        predictors: PredictorService,
        predictions: PredictionService,
    ) -> Self {
        let anthropic = AnthropicClient::new(predictors.clone(), predictions.clone());
        let openai = OpenAIClient::new(predictors.clone(), predictions.clone());
        let typesafe = TypeSafeClient::new(predictors, predictions);
        Self { anthropic, openai, typesafe }
    }
}
