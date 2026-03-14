/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::sync::Arc;

use crate::client::MunaClient;

use super::remote::RemotePredictionService;

/// Beta prediction service exposing remote predictions.
#[derive(Clone)]
pub struct BetaPredictionService {
    /// Make remote predictions.
    pub remote: RemotePredictionService,
}

impl BetaPredictionService {

    pub fn new(client: Arc<MunaClient>) -> Self {
        Self {
            remote: RemotePredictionService::new(client),
        }
    }
}
