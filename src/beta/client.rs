/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::sync::Arc;
use crate::client::MunaClient;
use crate::services::{PredictionService, PredictorService};
use super::remote::BetaPredictionService;

/// Client for incubating features.
#[derive(Clone)]
pub struct BetaClient {
    /// Make remote predictions.
    pub predictions: BetaPredictionService
}

impl BetaClient {

    pub fn new(
        client: Arc<MunaClient>,
        _predictors: PredictorService,
        _predictions: PredictionService,
    ) -> Self {
        let beta_predictions = BetaPredictionService::new(client);
        Self { predictions: beta_predictions }
    }
}
