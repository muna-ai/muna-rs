/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::sync::Arc;

use crate::client::MunaClient;
use crate::services::{PredictionService, PredictorService};

use super::openai::OpenAIClient;
use super::remote::BetaPredictionService;

/// Client for incubating features.
#[derive(Clone)]
pub struct BetaClient {
    /// Make remote predictions.
    pub predictions: BetaPredictionService,
    /// OpenAI-compatible client.
    pub openai: OpenAIClient,
}

impl BetaClient {

    pub fn new(
        client: Arc<MunaClient>,
        predictors: PredictorService,
        predictions: PredictionService,
    ) -> Self {
        let beta_predictions = BetaPredictionService::new(client);
        let openai = OpenAIClient::new(
            predictors,
            predictions,
            beta_predictions.remote.clone(),
        );
        Self {
            predictions: beta_predictions,
            openai,
        }
    }
}
