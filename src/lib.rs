/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

pub mod beta;
pub mod c;
pub mod client;
pub mod services;
pub mod types;

pub use client::{MunaClient, MunaError};
pub use types::*;

use std::sync::Arc;

/// Muna client.
pub struct Muna {
    /// Muna API client.
    pub client: Arc<MunaClient>,
    /// Manage users.
    pub users: services::UserService,
    /// Manage predictors.
    pub predictors: services::PredictorService,
    /// Make predictions.
    pub predictions: services::PredictionService,
    /// Beta client for incubating features.
    pub beta: beta::BetaClient,
}

impl Muna {

    /// Create a Muna client.
    ///
    /// # Arguments
    /// * `access_key` - Muna access key. Falls back to `MUNA_ACCESS_KEY` or `FXN_ACCESS_KEY` env var.
    /// * `url` - Muna API URL. Falls back to `MUNA_API_URL` or `FXN_API_URL` env var.
    pub fn new(access_key: Option<&str>, url: Option<&str>) -> Self {
        let client = Arc::new(MunaClient::new(access_key, url));
        let users = services::UserService::new(client.clone());
        let predictors = services::PredictorService::new(client.clone());
        let predictions = services::PredictionService::new(client.clone());
        let beta = beta::BetaClient::new(
            client.clone(),
            predictors.clone(),
            predictions.clone(),
        );
        Self {
            client,
            users,
            predictors,
            predictions,
            beta,
        }
    }
}

impl Default for Muna {
    fn default() -> Self {
        Self::new(None, None)
    }
}
