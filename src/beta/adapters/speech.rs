/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::collections::HashMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::c;
use crate::client::{MunaError, Result};
use crate::types::{Dtype, Parameter, Prediction, Signature, Value};

use super::audio::validate_samples;
use super::{
    utils::{decode, get_parameter, parameter, require, required, result, validate_inputs, FLOATS},
    Adapter,
};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SpeechResponseFormat {
    #[default]
    Mp3,
    Opus,
    Aac,
    Flac,
    Wav,
    Pcm,
}

impl SpeechResponseFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mp3 => "mp3",
            Self::Opus => "opus",
            Self::Aac => "aac",
            Self::Flac => "flac",
            Self::Wav => "wav",
            Self::Pcm => "pcm",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SpeechStreamFormat {
    #[default]
    Audio,
    Sse,
}

/// Generated audio bytes and HTTP-style metadata, matching the Python adapter.
#[derive(Debug)]
pub struct SpeechCreateResponse {
    pub content: Vec<u8>,
    pub response: reqwest::Response,
}

#[derive(Debug, Clone, Default)]
pub struct OpenAISpeechOptions {
    pub response_format: SpeechResponseFormat,
    pub stream_format: SpeechStreamFormat,
}

#[derive(Debug, Clone)]
pub struct OpenAISpeechAdapter {
    pub input: Parameter,
    pub voice: Parameter,
    pub output: usize,
    pub sample_rate: u32,
    pub speed: Option<Parameter>,
}

impl OpenAISpeechAdapter {
    pub fn from_signature(signature: &Signature) -> Result<Self> {
        // Voice input.
        let required = required(&signature.inputs);
        let voice = require(
            parameter(
                &required,
                &[Dtype::String],
                Some("openai.audio.speech.voice"),
            )?,
            "Requires one required string input annotated as speech voice",
        )?;
        // Voice is also a string; exclude it from the text lookup.
        let remaining = required
            .into_iter()
            .filter(|parameter| parameter.name != voice.name)
            .collect::<Vec<_>>();
        let input = require(
            parameter(&remaining, &[Dtype::String], None)?,
            "Requires one required speech text input",
        )?;
        let speed = parameter(&signature.inputs, FLOATS, Some("openai.audio.speech.speed"))?;
        validate_inputs(signature, &[Some(&input), Some(&voice), speed.as_ref()])?;
        // Audio output.
        let (output, audio) = require(
            get_parameter(&signature.outputs, &[Dtype::Float32], Some("audio"))?,
            "Requires one float32 output annotated as audio",
        )?;
        let sample_rate = require(
            audio.sample_rate.filter(|rate| *rate > 0),
            "Speech audio output must declare a positive sample rate",
        )?;
        Ok(Self {
            input,
            voice,
            output,
            sample_rate,
            speed,
        })
    }

    pub fn inputs(
        &self,
        input: String,
        voice: String,
        speed: Option<f64>,
    ) -> HashMap<String, Value> {
        let mut inputs = HashMap::from([
            (self.input.name.clone(), Value::String(input)),
            (self.voice.name.clone(), Value::String(voice)),
        ]);
        if let Some(parameter) = &self.speed {
            let speed = speed.unwrap_or(1.0);
            inputs.insert(
                parameter.name.clone(),
                if parameter.dtype == Some(Dtype::Float64) {
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
        audio: &Value,
        format: SpeechResponseFormat,
    ) -> Result<SpeechCreateResponse> {
        let samples = match audio {
            Value::Tensor(samples) => samples,
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

#[async_trait]
impl Adapter for OpenAISpeechAdapter {
    type Source = Prediction;
    type Options = OpenAISpeechOptions;
    type Response = SpeechCreateResponse;

    fn kind(&self) -> &'static str {
        "openai.audio.speech.create"
    }

    async fn adapt(
        &self,
        prediction: Prediction,
        options: Option<Self::Options>,
    ) -> Result<Self::Response> {
        let options = options.unwrap_or_default();
        if options.stream_format != SpeechStreamFormat::Audio {
            return Err(MunaError::InvalidInput(
                "Only the audio speech stream format is supported".into(),
            ));
        }
        let audio = decode(result(&prediction, self.output)?, Dtype::Float32).await?;
        self.response(&audio, options.response_format)
    }
}
