/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::sync::Arc;

use crate::client::{Client, ClientExt, MunaError, RequestInput, Result};
use crate::types::Predictor;

/// Manage predictors.
#[derive(Clone)]
pub struct PredictorService {
    client: Arc<dyn Client>,
}

impl PredictorService {

    pub fn new(client: Arc<dyn Client>) -> Self {
        Self { client }
    }

    /// Retrieve a predictor.
    pub async fn retrieve(&self, tag: &str) -> Result<Option<Predictor>> {
        match self
            .client
            .request_as(RequestInput::get(format!("/predictors/{tag}")))
            .await
        {
            Ok(predictor) => Ok(Some(predictor)),
            Err(MunaError::Api { status: 404, .. }) => Ok(None),
            Err(e) => Err(e),
        }
    }
}
