/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::sync::Arc;
use crate::client::MunaClient;
use crate::services::{PredictionService, PredictorService};
use super::openai::OpenAIClient;
use super::remote::{BetaPredictionService, RemotePredictionService};

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
        let beta_predictions = BetaPredictionService::new(client.clone());
        let remote_predictions = RemotePredictionService::new(client);
        let openai = OpenAIClient::new(predictors, predictions, remote_predictions);
        Self { predictions: beta_predictions, openai }
    }
}
