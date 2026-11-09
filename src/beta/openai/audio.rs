/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use crate::c;
use crate::client::{MunaError, Result};
use crate::services::{PredictionService, PredictorService};
use crate::types::{Tensor, TensorData, Value};

use super::{SpeechService, TranscriptionService};

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

/// OpenAI-compatible audio adapters.
#[derive(Clone)]
pub struct AudioService {
    pub speech: SpeechService,
    pub transcriptions: TranscriptionService,
}

impl AudioService {

    pub fn new(
        predictors: PredictorService,
        predictions: PredictionService
    ) -> Self {
        Self {
            speech: SpeechService::new(predictors.clone(), predictions.clone()),
            transcriptions: TranscriptionService::new(predictors, predictions),
        }
    }
}

pub(super) fn decode_audio(input: AudioInput, sample_rate: u32) -> Result<Tensor> {
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

pub(super) fn validate_samples(samples: &Tensor) -> Result<usize> {
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
mod tests {
    use super::*;

    #[test]
    fn stereo_counts_frames_and_rejects_invalid_pcm() {
        let stereo = Tensor {
            data: TensorData::Float32(vec![0.0; 20]),
            shape: vec![10, 2],
        };
        assert_eq!(validate_samples(&stereo).unwrap(), 10);
        for samples in [
            Tensor {
                data: TensorData::Float32(vec![]),
                shape: vec![0],
            },
            Tensor {
                data: TensorData::Float32(vec![0.0]),
                shape: vec![1, 1, 1],
            },
            Tensor {
                data: TensorData::Float32(vec![0.0]),
                shape: vec![2],
            },
            Tensor {
                data: TensorData::Float32(vec![f32::NAN]),
                shape: vec![1],
            },
            Tensor {
                data: TensorData::Int32(vec![0]),
                shape: vec![1],
            },
        ] {
            assert!(validate_samples(&samples).is_err());
        }
    }

    #[test]
    fn decoded_pcm_requires_matching_sample_rate() {
        let audio = Audio {
            samples: Tensor {
                data: TensorData::Float32(vec![0.0; 16]),
                shape: vec![16],
            },
            sample_rate: 16_000,
        };
        assert!(decode_audio(audio.clone().into(), 8_000).is_err());
        assert!(decode_audio(audio.clone().into(), 0).is_err());
        assert_eq!(decode_audio(audio.into(), 16_000).unwrap().shape, vec![16]);
        assert!(decode_audio(Vec::new().into(), 16_000).is_err());
    }

    #[test]
    fn encoded_wav_resampling_preserves_content_duration() {
        let samples = Value::Tensor(Tensor {
            data: TensorData::Float32(vec![0.0; 8_000]),
            shape: vec![8_000],
        });
        let wav = c::Value::from_object(&samples)
            .unwrap()
            .serialize(Some("audio/wav;rate=8000"))
            .unwrap();
        let decoded = decode_audio(wav.into(), 16_000).unwrap();
        assert_eq!(validate_samples(&decoded).unwrap(), 16_000);
    }
    #[tokio::test]
    async fn unsupported_streams_and_invalid_parameters_fail_before_prediction() {
        let muna = crate::Muna::new(None, Some("http://127.0.0.1:1"));
        let transcriptions = &muna.beta.openai.audio.transcriptions;
        assert!(matches!(
            transcriptions
                .create(super::super::TranscriptionCreateParams {
                    stream: true,
                    ..Default::default()
                })
                .await,
            Err(MunaError::InvalidInput(_))
        ));
        for temperature in [f64::NAN, -0.1, 1.1] {
            assert!(matches!(
                transcriptions
                    .create(super::super::TranscriptionCreateParams {
                        temperature: Some(temperature),
                        ..Default::default()
                    })
                    .await,
                Err(MunaError::InvalidInput(_))
            ));
        }
        let speech = &muna.beta.openai.audio.speech;
        assert!(matches!(
            speech
                .create(super::super::SpeechCreateParams {
                    stream_format: Some(super::super::SpeechStreamFormat::Sse),
                    ..Default::default()
                })
                .await,
            Err(MunaError::InvalidInput(_))
        ));
        for speed in [f64::NAN, 0.0, -1.0] {
            assert!(matches!(
                speech
                    .create(super::super::SpeechCreateParams {
                        speed: Some(speed),
                        ..Default::default()
                    })
                    .await,
                Err(MunaError::InvalidInput(_))
            ));
        }
    }
}
