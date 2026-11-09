/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;

use crate::beta::utils::get_parameter;
use crate::client::{MunaError, Result};
use crate::services::{PredictionService, PredictorService};
use crate::types::{Acceleration, Device, Dtype, Signature, Value};

use super::audio::{decode_audio, validate_samples};
use super::{AudioInput, Transcription, TranscriptionUsage};

/// Parameters for non-streaming transcription. Encoded files are decoded/resampled locally.
#[derive(Debug, Clone, Default)]
pub struct TranscriptionCreateParams {
    pub file: AudioInput,
    pub model: String,
    pub language: Option<String>,
    pub prompt: Option<String>,
    /// Streaming transcription is not currently supported.
    pub stream: bool,
    pub temperature: Option<f64>,
    pub acceleration: Option<Acceleration>,
    pub devices: Option<Vec<Device>>,
}

#[derive(Clone)]
struct TranscriptionConfig {
    audio_name: String,
    array_list: bool,
    sample_rate: u32,
    language_name: Option<String>,
    prompt_name: Option<String>,
    temperature: Option<(String, Dtype)>,
    output_index: usize,
}

impl TranscriptionConfig {

    fn from_signature(signature: &Signature) -> Result<Self> {
        let required: Vec<_> = signature
            .inputs
            .iter()
            .filter(|p| p.optional != Some(true))
            .collect();
        if required.len() != 1 {
            return Err(MunaError::InvalidInput(
                "Transcription requires exactly one required audio input.".into(),
            ));
        }
        let audio = required[0];
        if audio.denotation.as_deref() != Some("audio")
            || !matches!(audio.dtype, Some(Dtype::Float32 | Dtype::ArrayList))
        {
            return Err(MunaError::InvalidInput(
                "Transcription requires a float32 or array_list audio input.".into(),
            ));
        }
        let sample_rate = audio.sample_rate.filter(|rate| *rate > 0).ok_or_else(|| {
            MunaError::InvalidInput(
                "Transcription audio input must declare a positive sample rate.".into(),
            )
        })?;
        let optional_name = |denotation| {
            get_parameter(&signature.inputs, &[Dtype::String], Some(denotation))
                .1
                .map(|p| p.name.clone())
        };
        let temperature = get_parameter(
            &signature.inputs,
            &[Dtype::Float32, Dtype::Float64],
            Some("openai.chat.completions.temperature"),
        )
        .1
        .map(|p| (p.name.clone(), p.dtype.unwrap()));
        let output_index = get_parameter(&signature.outputs, &[Dtype::String, Dtype::List], None)
            .0
            .ok_or_else(|| {
                MunaError::InvalidInput("Transcription requires a string or list output.".into())
            })?;
        Ok(Self {
            audio_name: audio.name.clone(),
            array_list: audio.dtype == Some(Dtype::ArrayList),
            sample_rate,
            language_name: optional_name("openai.audio.transcriptions.language"),
            prompt_name: optional_name("openai.audio.transcriptions.prompt"),
            temperature,
            output_index,
        })
    }

    fn prepare(&self, params: TranscriptionCreateParams) -> Result<(HashMap<String, Value>, f64)> {
        let samples = decode_audio(params.file, self.sample_rate)?;
        // Like muna-py's transcription adapter: frames, not interleaved sample count or execution time.
        let seconds = validate_samples(&samples)? as f64 / self.sample_rate as f64;
        let audio = if self.array_list {
            Value::ArrayList(vec![samples])
        } else {
            Value::Tensor(samples)
        };
        let mut inputs = HashMap::from([(self.audio_name.clone(), audio)]);
        if let (Some(name), Some(language)) = (&self.language_name, params.language) {
            inputs.insert(name.clone(), Value::String(language));
        }
        if let (Some(name), Some(prompt)) = (&self.prompt_name, params.prompt) {
            inputs.insert(name.clone(), Value::String(prompt));
        }
        if let Some((name, dtype)) = &self.temperature {
            let temperature = params.temperature.unwrap_or(0.0);
            inputs.insert(
                name.clone(),
                if *dtype == Dtype::Float64 {
                    Value::Double(temperature)
                } else {
                    Value::Float(temperature as f32)
                },
            );
        }
        Ok((inputs, seconds))
    }

    fn response(&self, results: &[Value], seconds: f64) -> Result<Transcription> {
        let text = match results.get(self.output_index) {
            Some(Value::String(text)) => Some(text.clone()),
            Some(Value::List(texts)) => texts.first().and_then(|v| v.as_str()).map(str::to_owned),
            _ => None,
        }
        .ok_or_else(|| {
            MunaError::Prediction("Transcription predictor did not return a string.".into())
        })?;
        Ok(Transcription {
            text,
            usage: TranscriptionUsage::Duration { seconds },
        })
    }
}

/// Transcribe audio and report content duration, independent of model latency.
#[derive(Clone)]
pub struct TranscriptionService {
    predictors: PredictorService,
    predictions: PredictionService,
    cache: Arc<RwLock<HashMap<String, TranscriptionConfig>>>,
}

impl TranscriptionService {

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

    pub async fn create(&self, params: TranscriptionCreateParams) -> Result<Transcription> {
        if params.stream {
            return Err(MunaError::InvalidInput(
                "Streaming transcription is not supported.".into(),
            ));
        }
        if params
            .temperature
            .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
        {
            return Err(MunaError::InvalidInput(
                "Transcription temperature must be between 0 and 1.".into(),
            ));
        }
        let model = params.model.clone();
        let cached = self.cache.read().await.get(&model).cloned();
        let config = match cached {
            Some(config) => config,
            None => {
                let predictor = self.predictors.retrieve(&model).await?.ok_or_else(|| {
                    MunaError::InvalidInput(format!(
                        "Transcription predictor {model} could not be found."
                    ))
                })?;
                let config = TranscriptionConfig::from_signature(&predictor.signature)?;
                self.cache
                    .write()
                    .await
                    .insert(model.clone(), config.clone());
                config
            }
        };
        let acceleration = params.acceleration.clone();
        let devices = params.devices.clone();
        let (inputs, seconds) = config.prepare(params)?;
        let prediction = self
            .predictions
            .create(
                &model,
                Some(inputs),
                acceleration,
                devices.as_deref(),
                None,
                None,
                None,
            )
            .await?;
        if let Some(error) = prediction.error {
            return Err(MunaError::from_prediction_error(error));
        }
        config.response(&prediction.results.unwrap_or_default(), seconds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beta::openai::Audio;
    use crate::types::{Tensor, TensorData};
    use serde_json::json;

    fn signature(dtype: &str) -> Signature {
        serde_json::from_value(json!({
            "inputs": [
                {"name":"samples", "dtype":dtype, "denotation":"audio", "sampleRate":16000},
                {"name":"lang", "dtype":"string", "optional":true,
                 "denotation":"openai.audio.transcriptions.language"},
                {"name":"hint", "dtype":"string", "optional":true,
                 "denotation":"openai.audio.transcriptions.prompt"},
                {"name":"temp", "dtype":"float64", "optional":true,
                 "denotation":"openai.chat.completions.temperature"}
            ],
            "outputs": [{"name":"ignored", "dtype":"float32"}, {"name":"text", "dtype":"list"}]
        }))
        .unwrap()
    }

    #[test]
    fn transcription_reports_audio_seconds_for_mono_and_stereo() {
        for dtype in ["float32", "array_list"] {
            for channels in [1, 2] {
                let config = TranscriptionConfig::from_signature(&signature(dtype)).unwrap();
                let audio = Audio {
                    samples: Tensor {
                        data: TensorData::Float32(vec![0.0; 24_000 * channels]),
                        shape: if channels == 1 {
                            vec![24_000]
                        } else {
                            vec![24_000, 2]
                        },
                    },
                    sample_rate: 16_000,
                };
                let (inputs, seconds) = config
                    .prepare(TranscriptionCreateParams {
                        file: audio.into(),
                        language: Some("en".into()),
                        prompt: Some("Hello".into()),
                        ..Default::default()
                    })
                    .unwrap();
                assert_eq!(seconds, 1.5);
                assert!(matches!(inputs.get("lang"), Some(Value::String(s)) if s == "en"));
                assert!(matches!(inputs.get("hint"), Some(Value::String(s)) if s == "Hello"));
                assert!(matches!(inputs.get("temp"), Some(Value::Double(0.0))));
                match &inputs["samples"] {
                    Value::ArrayList(tensors) => assert_eq!(tensors.len(), 1),
                    Value::Tensor(_) => assert_eq!(dtype, "float32"),
                    other => panic!("Unexpected audio input: {other:?}"),
                }
                for text in [
                    Value::String("Hello".into()),
                    Value::List(vec![json!("Hello")]),
                ] {
                    let response = config
                        .response(&[Value::Float(123.0), text], seconds)
                        .unwrap();
                    assert_eq!(
                        serde_json::to_value(&response).unwrap(),
                        json!({
                            "text":"Hello", "usage":{"type":"duration", "seconds":1.5}
                        })
                    );
                }
            }
        }
    }

    #[test]
    fn transcription_rejects_invalid_signature_and_results() {
        let mut signature = signature("float32");
        let config = TranscriptionConfig::from_signature(&signature).unwrap();
        assert!(config.response(&[], 1.0).is_err());
        assert!(config
            .response(&[Value::Float(0.0), Value::List(vec![])], 1.0)
            .is_err());
        assert!(config
            .response(&[Value::Float(0.0), Value::List(vec![json!(42)])], 1.0)
            .is_err());
        signature.inputs[0].sample_rate = None;
        assert!(TranscriptionConfig::from_signature(&signature).is_err());
        signature.inputs[0].sample_rate = Some(0);
        assert!(TranscriptionConfig::from_signature(&signature).is_err());
        assert!(TranscriptionConfig::from_signature(&self::signature("string")).is_err());
    }

    #[test]
    fn audio_usage_variants_round_trip() {
        for value in [
            json!({"type":"duration", "seconds":2.125}),
            json!({"type":"tokens", "input_tokens":12, "output_tokens":3, "total_tokens":15,
                "input_token_details":{"audio_tokens":10,"text_tokens":2}}),
        ] {
            let usage: TranscriptionUsage = serde_json::from_value(value.clone()).unwrap();
            assert_eq!(serde_json::to_value(usage).unwrap(), value);
        }
    }
}
