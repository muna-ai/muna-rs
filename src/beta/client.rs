/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use super::openai::OpenAIClient;
use crate::client::MunaClient;
use crate::services::{PredictionService, PredictorService};
use std::sync::Arc;

/// Client for incubating features.
#[derive(Clone)]
pub struct BetaClient {
    /// OpenAI-compatible client.
    pub openai: OpenAIClient,
}

impl BetaClient {

    pub fn new(
        _: Arc<MunaClient>,
        predictors: PredictorService,
        predictions: PredictionService,
    ) -> Self {
        let openai = OpenAIClient::new(predictors, predictions);
        Self { openai }
    }
}
