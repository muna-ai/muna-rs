/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use async_trait::async_trait;

use crate::client::Result;

/// Convert raw predictions to an API response without executing a predictor.
/// Stream adapters return a lazy stream; dropping it drops the prediction source.
#[async_trait]
pub trait Adapter: Send + Sync {
    type Source: Send;
    type Options: Default + Send;
    type Response: Send;

    fn kind(&self) -> &'static str;

    async fn adapt(
        &self,
        prediction: Self::Source,
        options: Option<Self::Options>,
    ) -> Result<Self::Response>;
}
