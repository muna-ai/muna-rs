/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Signature-based bindings and response schemas shared by SDK clients and servers.
//!
//! Resolve an adapter once, prepare its inputs, execute a raw prediction with
//! `passthrough`, then adapt the result. Adapters never load or invoke predictors.

mod adapter;
pub mod audio;
pub mod chat;
pub mod completions;
pub mod depth;
pub mod edits;
pub mod embeddings;
pub mod images;
pub mod messages;
pub mod speech;
pub mod system_one;
pub mod transcriptions;
mod utils;

pub use adapter::*;
pub use audio::*;
pub use completions::*;
pub use depth::*;
pub use edits::*;
pub use embeddings::*;
pub use images::*;
pub use messages::*;
pub use speech::*;
pub use system_one::*;
pub use transcriptions::*;

/// Raw prediction stream consumed by streaming adapters.
pub type PredictionStream = std::pin::Pin<
    Box<dyn futures_core::Stream<Item = crate::client::Result<crate::types::Prediction>> + Send>,
>;
