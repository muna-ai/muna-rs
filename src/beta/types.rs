/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

/// Audio buffer.
#[derive(Debug, Clone)]
pub struct Audio {
    /// Linear PCM audio samples with shape `(F,C)`.
    pub samples: Vec<f32>,
    /// Audio sample rate in Hertz.
    pub sample_rate: u32,
    /// Audio channel count.
    pub channel_count: u32,
}