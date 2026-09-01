/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::sync::Arc;

use crate::client::Client;
use crate::services::{PredictionService, PredictorService};

use super::ChatCompletionService;

/// Create chat conversations.
#[derive(Clone)]
pub struct ChatService {
    /// Create completions.
    pub completions: ChatCompletionService,
}

impl ChatService {

    pub fn new(
        client: Arc<dyn Client>,
        predictors: PredictorService,
        predictions: PredictionService
    ) -> Self {
        Self {
            completions: ChatCompletionService::new(client, predictors, predictions),
        }
    }
}
