/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::sync::Arc;

use crate::client::{MunaClient, MunaError, RequestInput, Result};
use crate::types::Predictor;

/// Manage predictors.
#[derive(Clone)]
pub struct PredictorService {
    client: Arc<MunaClient>,
}

impl PredictorService {

    pub fn new(client: Arc<MunaClient>) -> Self {
        Self { client }
    }

    /// Retrieve a predictor.
    pub async fn retrieve(&self, tag: &str) -> Result<Option<Predictor>> {
        match self.client.request(RequestInput::get(format!("/predictors/{tag}"))).await {
            Ok(predictor) => Ok(Some(predictor)),
            Err(MunaError::Api { status: 404, .. }) => Ok(None),
            Err(e) => Err(e),
        }
    }
}
