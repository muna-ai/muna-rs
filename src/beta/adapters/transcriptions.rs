/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::collections::HashMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::client::{MunaError, Result};
use crate::types::{Dtype, Parameter, Prediction, Signature, Value};

use super::audio::{decode_audio, validate_samples, AudioInput};
use super::{
    utils::{decode, get_parameter, parameter, require, required, result, validate_inputs, FLOATS},
    Adapter,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscriptionInputTokenDetails {
    pub audio_tokens: u64,
    pub text_tokens: u64,
}

/// Audio content usage; duration is independent of prediction latency.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum TranscriptionUsage {
    Tokens {
        input_tokens: u64,
        output_tokens: u64,
        total_tokens: u64,
        input_token_details: TranscriptionInputTokenDetails,
    },
    Duration {
        seconds: f64,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Transcription {
    pub text: String,
    pub usage: TranscriptionUsage,
}

#[derive(Debug, Clone, Default)]
pub struct OpenAITranscriptionsOptions {
    pub duration: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct OpenAITranscriptionsAdapter {
    pub audio: Parameter,
    pub output: usize,
    pub usage: Option<usize>,
    pub language: Option<Parameter>,
    pub prompt: Option<Parameter>,
    pub temperature: Option<Parameter>,
}

impl OpenAITranscriptionsAdapter {
    /// Returns `None` when the signature is incompatible or ambiguous.
    pub fn from_signature(signature: &Signature) -> Option<Self> {
        // Audio input.
        let audio = parameter(
            &required(&signature.inputs),
            &[Dtype::Float32, Dtype::ArrayList],
            Some("audio"),
        )
        .ok()??;
        audio.sample_rate.filter(|rate| *rate > 0)?;
        // Optional language, prompt, and temperature.
        let language = parameter(
            &signature.inputs,
            &[Dtype::String],
            Some("openai.audio.transcriptions.language"),
        )
        .ok()?;
        let prompt = parameter(
            &signature.inputs,
            &[Dtype::String],
            Some("openai.audio.transcriptions.prompt"),
        )
        .ok()?;
        let temperature = parameter(
            &signature.inputs,
            FLOATS,
            Some("openai.chat.completions.temperature"),
        )
        .ok()?;
        validate_inputs(
            signature,
            &[
                Some(&audio),
                language.as_ref(),
                prompt.as_ref(),
                temperature.as_ref(),
            ],
        )
        .ok()?;
        // Text output and optional reported usage.
        let (output, _) =
            get_parameter(&signature.outputs, &[Dtype::String, Dtype::List], None).ok()??;
        let usage = super::utils::output(signature, &[Dtype::Dict], "Usage").ok()?;
        Some(Self {
            audio,
            output,
            usage,
            language,
            prompt,
            temperature,
        })
    }

    pub fn inputs(
        &self,
        file: AudioInput,
        language: Option<String>,
        prompt: Option<String>,
        temperature: Option<f64>,
    ) -> Result<HashMap<String, Value>> {
        let samples = decode_audio(file, self.audio.sample_rate.unwrap())?;
        let audio = if self.audio.dtype == Some(Dtype::ArrayList) {
            Value::ArrayList(vec![samples])
        } else {
            Value::Tensor(samples)
        };
        let mut inputs = HashMap::from([(self.audio.name.clone(), audio)]);
        if let (Some(parameter), Some(language)) = (&self.language, language) {
            inputs.insert(parameter.name.clone(), Value::String(language));
        }
        if let (Some(parameter), Some(prompt)) = (&self.prompt, prompt) {
            inputs.insert(parameter.name.clone(), Value::String(prompt));
        }
        if let Some(parameter) = &self.temperature {
            let temperature = temperature.unwrap_or(0.0);
            inputs.insert(
                parameter.name.clone(),
                if parameter.dtype == Some(Dtype::Float64) {
                    Value::Double(temperature)
                } else {
                    Value::Float(temperature as f32)
                },
            );
        }
        Ok(inputs)
    }

    pub fn duration(&self, inputs: &HashMap<String, Value>) -> Result<f64> {
        let audio = match inputs.get(&self.audio.name) {
            Some(Value::Tensor(audio)) => audio,
            Some(Value::ArrayList(audio)) if audio.len() == 1 => &audio[0],
            _ => {
                return Err(MunaError::InvalidInput(
                    "Missing transcription audio input".into(),
                ))
            }
        };
        Ok(validate_samples(audio)? as f64 / self.audio.sample_rate.unwrap() as f64)
    }
}

#[async_trait]
impl Adapter for OpenAITranscriptionsAdapter {
    type Source = Prediction;
    type Options = OpenAITranscriptionsOptions;
    type Response = Transcription;

    fn kind(&self) -> &'static str {
        "openai.audio.transcriptions.create"
    }

    async fn adapt(
        &self,
        prediction: Prediction,
        options: Option<Self::Options>,
    ) -> Result<Self::Response> {
        let value = result(&prediction, self.output)?;
        let dtype = match value {
            Value::Remote(remote) => remote.dtype,
            Value::List(_) => Dtype::List,
            _ => Dtype::String,
        };
        let text = match decode(value, dtype).await? {
            Value::String(text) => Some(text),
            Value::List(texts) => texts
                .first()
                .and_then(|text| text.as_str())
                .map(str::to_owned),
            _ => None,
        }
        .ok_or_else(|| {
            MunaError::Prediction("Transcription predictor did not return text".into())
        })?;
        let usage = if let Some(index) = self.usage {
            let Value::Dict(usage) = decode(result(&prediction, index)?, Dtype::Dict).await? else {
                return Err(MunaError::Prediction(
                    "Transcription usage must be a dictionary".into(),
                ));
            };
            serde_json::from_value(serde_json::Value::Object(usage))?
        } else {
            let seconds = require(
                options.unwrap_or_default().duration,
                "Transcription requires input duration when usage is absent",
            )?;
            if !seconds.is_finite() || seconds < 0.0 {
                return Err(MunaError::InvalidInput(
                    "Transcription duration must be finite and nonnegative".into(),
                ));
            }
            TranscriptionUsage::Duration { seconds }
        };
        Ok(Transcription { text, usage })
    }
}
