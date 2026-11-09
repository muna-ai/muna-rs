/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;

use crate::beta::utils::get_parameter;
use crate::c;
use crate::client::{MunaError, Result};
use crate::services::{PredictionService, PredictorService};
use crate::types::{Acceleration, Device, Dtype, Signature, Value};

use super::audio::validate_samples;
use super::{SpeechCreateResponse, SpeechResponseFormat, SpeechStreamFormat};

/// Parameters for speech generation. Binary audio is currently the only supported stream format.
#[derive(Debug, Clone, Default)]
pub struct SpeechCreateParams {
    pub input: String,
    pub model: String,
    pub voice: String,
    pub response_format: Option<SpeechResponseFormat>,
    pub speed: Option<f64>,
    pub stream_format: Option<SpeechStreamFormat>,
    pub acceleration: Option<Acceleration>,
    pub devices: Option<Vec<Device>>,
}

#[derive(Clone)]
struct SpeechConfig {
    input_name: String,
    voice_name: String,
    speed: Option<(String, Dtype)>,
    output_index: usize,
    sample_rate: u32,
}

impl SpeechConfig {
    fn from_signature(signature: &Signature) -> Result<Self> {
        let required: Vec<_> = signature
            .inputs
            .iter()
            .filter(|p| p.optional != Some(true))
            .collect();
        if required.len() != 2 {
            return Err(MunaError::InvalidInput(
                "Speech requires exactly two required text/voice inputs.".into(),
            ));
        }
        let voice = required
            .iter()
            .find(|p| {
                p.dtype == Some(Dtype::String)
                    && p.denotation.as_deref() == Some("openai.audio.speech.voice")
            })
            .ok_or_else(|| MunaError::InvalidInput("Speech requires a voice input.".into()))?;
        // Select a distinct text input even when voice is first in the signature.
        let input = required
            .iter()
            .find(|p| p.dtype == Some(Dtype::String) && p.name != voice.name)
            .ok_or_else(|| MunaError::InvalidInput("Speech requires a text input.".into()))?;
        let speed = get_parameter(
            &signature.inputs,
            &[Dtype::Float32, Dtype::Float64],
            Some("openai.audio.speech.speed"),
        )
        .1
        .map(|p| (p.name.clone(), p.dtype.unwrap()));
        let (index, audio) = get_parameter(&signature.outputs, &[Dtype::Float32], Some("audio"));
        let audio = audio.ok_or_else(|| {
            MunaError::InvalidInput("Speech requires a float32 audio output.".into())
        })?;
        let sample_rate = audio.sample_rate.filter(|rate| *rate > 0).ok_or_else(|| {
            MunaError::InvalidInput(
                "Speech audio output must declare a positive sample rate.".into(),
            )
        })?;
        Ok(Self {
            input_name: input.name.clone(),
            voice_name: voice.name.clone(),
            speed,
            output_index: index.unwrap(),
            sample_rate,
        })
    }

    fn inputs(&self, params: &SpeechCreateParams) -> HashMap<String, Value> {
        let mut inputs = HashMap::from([
            (self.input_name.clone(), Value::String(params.input.clone())),
            (self.voice_name.clone(), Value::String(params.voice.clone())),
        ]);
        if let Some((name, dtype)) = &self.speed {
            let speed = params.speed.unwrap_or(1.0);
            inputs.insert(
                name.clone(),
                if *dtype == Dtype::Float64 {
                    Value::Double(speed)
                } else {
                    Value::Float(speed as f32)
                },
            );
        }
        inputs
    }

    fn response(
        &self,
        results: &[Value],
        format: SpeechResponseFormat,
    ) -> Result<SpeechCreateResponse> {
        let samples = match results.get(self.output_index) {
            Some(Value::Tensor(samples)) => samples,
            _ => {
                return Err(MunaError::Prediction(
                    "Speech predictor did not return an audio tensor.".into(),
                ))
            }
        };
        validate_samples(samples).map_err(|error| MunaError::Prediction(error.to_string()))?;
        let (content, content_type) = if format == SpeechResponseFormat::Pcm {
            let channels = samples.shape.get(1).copied().unwrap_or(1);
            let content_type = format!(
                "audio/pcm;rate={};channels={channels};encoding=float;bits=32",
                self.sample_rate
            );
            let crate::types::TensorData::Float32(data) = &samples.data else {
                unreachable!()
            };
            (
                data.iter().flat_map(|value| value.to_ne_bytes()).collect(),
                content_type,
            )
        } else {
            let content_type = format!("audio/{};rate={}", format.as_str(), self.sample_rate);
            let content = c::Value::from_object(&Value::Tensor(samples.clone()))?
                .serialize(Some(&content_type))?;
            (content, content_type)
        };
        let response = http::Response::builder()
            .status(200)
            .header("Content-Type", content_type)
            .header("Content-Length", content.len().to_string())
            .body(content.clone())
            .map_err(|error| MunaError::Prediction(error.to_string()))?;
        Ok(SpeechCreateResponse {
            content,
            response: response.into(),
        })
    }
}

/// Generate encoded speech bytes from text using predictor signature annotations.
#[derive(Clone)]
pub struct SpeechService {
    predictors: PredictorService,
    predictions: PredictionService,
    cache: Arc<RwLock<HashMap<String, SpeechConfig>>>,
}

impl SpeechService {

    pub fn new(
        predictors: PredictorService,
        predictions: PredictionService
    ) -> Self {
        Self {
            predictors,
            predictions,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn create(&self, params: SpeechCreateParams) -> Result<SpeechCreateResponse> {
        if params.stream_format == Some(SpeechStreamFormat::Sse) {
            return Err(MunaError::InvalidInput(
                "Only the audio speech stream format is supported.".into(),
            ));
        }
        if params
            .speed
            .is_some_and(|speed| !speed.is_finite() || speed <= 0.0 || speed > f32::MAX as f64)
        {
            return Err(MunaError::InvalidInput(
                "Speech speed must be finite and positive.".into(),
            ));
        }
        let cached = self.cache.read().await.get(&params.model).cloned();
        let config = match cached {
            Some(config) => config,
            None => {
                let predictor =
                    self.predictors
                        .retrieve(&params.model)
                        .await?
                        .ok_or_else(|| {
                            MunaError::InvalidInput(format!(
                                "Speech predictor {} could not be found.",
                                params.model
                            ))
                        })?;
                let config = SpeechConfig::from_signature(&predictor.signature)?;
                self.cache
                    .write()
                    .await
                    .insert(params.model.clone(), config.clone());
                config
            }
        };
        let inputs = config.inputs(&params);
        let prediction = self
            .predictions
            .create(
                &params.model,
                Some(inputs),
                params.acceleration,
                params.devices.as_deref(),
                None,
                None,
                None,
            )
            .await?;
        if let Some(error) = prediction.error {
            return Err(MunaError::from_prediction_error(error));
        }
        config.response(
            &prediction.results.unwrap_or_default(),
            params.response_format.unwrap_or_default(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Tensor, TensorData};
    use serde_json::json;

    fn config() -> SpeechConfig {
        SpeechConfig::from_signature(&serde_json::from_value(json!({
            "inputs": [
                {"name":"speaker", "dtype":"string", "denotation":"openai.audio.speech.voice"},
                {"name":"text", "dtype":"string"},
                {"name":"rate", "dtype":"float64", "optional":true,
                 "denotation":"openai.audio.speech.speed"}
            ],
            "outputs": [{"name":"audio", "dtype":"float32", "denotation":"audio", "sampleRate":16000}]
        })).unwrap()).unwrap()
    }

    #[test]
    fn speech_binds_text_and_voice_independent_of_signature_order() {
        let config = config();
        let inputs = config.inputs(&SpeechCreateParams {
            input: "Hello".into(),
            voice: "alloy".into(),
            ..Default::default()
        });
        assert!(matches!(&inputs["text"], Value::String(s) if s == "Hello"));
        assert!(matches!(&inputs["speaker"], Value::String(s) if s == "alloy"));
        assert!(matches!(&inputs["rate"], Value::Double(1.0)));
    }

    #[tokio::test]
    async fn speech_pcm_contains_samples_and_correct_metadata() {
        let config = config();
        let samples = vec![0.0f32, 0.5, -0.5, 1.0];
        let expected: Vec<u8> = samples.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let result = config
            .response(
                &[Value::Tensor(Tensor {
                    data: TensorData::Float32(samples),
                    shape: vec![2, 2],
                })],
                SpeechResponseFormat::Pcm,
            )
            .unwrap();
        assert_eq!(result.content, expected);
        assert_eq!(result.response.status(), 200);
        assert_eq!(
            result.response.headers()["content-type"],
            "audio/pcm;rate=16000;channels=2;encoding=float;bits=32"
        );
        assert_eq!(result.response.headers()["content-length"], "16");
        assert_eq!(result.response.bytes().await.unwrap().as_ref(), expected);
        assert!(config.response(&[], SpeechResponseFormat::Pcm).is_err());
        assert!(config
            .response(&[Value::String("oops".into())], SpeechResponseFormat::Pcm)
            .is_err());
    }

    #[test]
    fn speech_wav_is_decodable_audio() {
        let result = config()
            .response(
                &[Value::Tensor(Tensor {
                    data: TensorData::Float32(vec![0.0; 1600]),
                    shape: vec![1600],
                })],
                SpeechResponseFormat::Wav,
            )
            .unwrap();
        assert_eq!(&result.content[..4], b"RIFF");
        let decoded = super::super::audio::decode_audio(result.content.into(), 16_000).unwrap();
        assert_eq!(validate_samples(&decoded).unwrap(), 1600);
    }
}
