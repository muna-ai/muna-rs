/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use crate::c;
use crate::client::{MunaError, Result};
use crate::types::{Tensor, TensorData, Value};

/// Decoded float32 PCM samples. Frames are the first axis; channels are optional.
#[derive(Debug, Clone)]
pub struct Audio {
    pub samples: Tensor,
    pub sample_rate: u32,
}

/// An encoded audio file or already decoded PCM with an explicit sample rate.
#[derive(Debug, Clone)]
pub enum AudioInput {
    Encoded(Vec<u8>),
    Samples(Audio),
}

impl Default for AudioInput {
    fn default() -> Self {
        Self::Encoded(Vec::new())
    }
}

impl From<Vec<u8>> for AudioInput {
    fn from(data: Vec<u8>) -> Self {
        Self::Encoded(data)
    }
}

impl From<Audio> for AudioInput {
    fn from(audio: Audio) -> Self {
        Self::Samples(audio)
    }
}

pub(crate) fn decode_audio(input: AudioInput, sample_rate: u32) -> Result<Tensor> {
    if sample_rate == 0 {
        return Err(MunaError::InvalidInput(
            "Audio sample rate must be positive.".into(),
        ));
    }
    let samples = match input {
        AudioInput::Samples(audio) => {
            if audio.sample_rate != sample_rate {
                return Err(MunaError::InvalidInput(format!(
                    "Audio sample rate {}Hz does not match the required sample rate of {sample_rate}Hz.",
                    audio.sample_rate
                )));
            }
            audio.samples
        }
        AudioInput::Encoded(bytes) => {
            if bytes.is_empty() {
                return Err(MunaError::InvalidInput("Audio file is empty.".into()));
            }
            let decoded = c::Value::from_bytes(&bytes, &format!("audio/*;rate={sample_rate}"))
                .and_then(|value| value.to_object())
                .map_err(|error| {
                    MunaError::InvalidInput(format!("Cannot decode audio: {error}"))
                })?;
            match decoded {
                Value::Tensor(samples) => samples,
                _ => {
                    return Err(MunaError::InvalidInput(
                        "Audio did not decode to PCM samples.".into(),
                    ))
                }
            }
        }
    };
    validate_samples(&samples)?;
    Ok(samples)
}

pub(crate) fn validate_samples(samples: &Tensor) -> Result<usize> {
    let invalid = || {
        MunaError::InvalidInput(
            "Audio must be a nonempty float32 tensor shaped [frames] or [frames, channels].".into(),
        )
    };
    if !(1..=2).contains(&samples.shape.len()) || samples.shape.iter().any(|dim| *dim <= 0) {
        return Err(invalid());
    }
    let expected = samples
        .shape
        .iter()
        .try_fold(1usize, |n, dim| n.checked_mul(*dim as usize))
        .ok_or_else(invalid)?;
    match &samples.data {
        TensorData::Float32(data)
            if data.len() == expected && data.iter().all(|v| v.is_finite()) =>
        {
            Ok(samples.shape[0] as usize)
        }
        _ => Err(invalid()),
    }
}

#[cfg(test)]
#[path = "tests/audio.rs"]
mod tests;
