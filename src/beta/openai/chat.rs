/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

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
        predictors: PredictorService,
        predictions: PredictionService
    ) -> Self {
        Self {
            completions: ChatCompletionService::new(predictors, predictions),
        }
    }
}
