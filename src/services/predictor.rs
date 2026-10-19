/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::sync::Arc;

use crate::client::{Client, ClientExt, MunaError, RequestInput, Result};
use crate::hf::{load_hf_manifest, parse_hf_tag};
use crate::types::{Predictor, PredictorAccess, PredictorStatus, User};

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
        if let Some(hf_tag) = parse_hf_tag(tag) {
            let hf_manifest = load_hf_manifest(&hf_tag).await?;
            let manifest_tag = hf_manifest.prediction.tag;
            let (username, name) = manifest_tag
                .trim_start_matches('@')
                .split_once('/')
                .ok_or_else(|| MunaError::Prediction(format!(
                    "Failed to retrieve {tag} because its manifest has an invalid tag: {manifest_tag}"
                )))?;
            return Ok(Some(Predictor {
                owner: User {
                    username: username.to_string(),
                    created: None,
                    name: None,
                    avatar: None,
                    bio: None,
                    website: None,
                    github: None,
                },
                name: name.to_string(),
                status: PredictorStatus::Active,
                access: PredictorAccess::Unlisted,
                signature: hf_manifest.signature,
                created: String::new(),
                description: None,
                tag: manifest_tag,
            }));
        }
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
