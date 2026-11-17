/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use muna::beta::adapters::*;
use muna::{Dtype, Tensor, TensorData, Value};
use serde_json::json;

use super::common::*;

fn transcription_signature(dtype: &str, usage: bool) -> muna::Signature {
    let mut outputs = vec![json!({"name":"text", "dtype":"string"})];
    if usage {
        outputs.push(json!({"name":"usage", "dtype":"dict", "schema":{"title":"Usage"}}));
    }
    signature(
        json!([
            {"name":"samples", "dtype":dtype, "denotation":"audio", "sampleRate":16000},
            {"name":"hint", "dtype":"string", "optional":true, "denotation":"openai.audio.transcriptions.prompt"}
        ]),
        json!(outputs),
    )
}

#[test]
fn transcription_duration_counts_frames_for_both_input_shapes() {
    for dtype in ["float32", "array_list"] {
        let adapter =
            OpenAITranscriptionsAdapter::from_signature(&transcription_signature(dtype, false))
                .unwrap();
        let inputs = adapter
            .inputs(
                Audio {
                    samples: Tensor {
                        data: TensorData::Float32(vec![0.; 48000]),
                        shape: vec![24000, 2],
                    },
                    sample_rate: 16000,
                }
                .into(),
                None,
                Some("hint".into()),
                None,
            )
            .unwrap();
        assert_eq!(adapter.duration(&inputs).unwrap(), 1.5);
        assert!(matches!(&inputs["hint"], Value::String(value) if value == "hint"));
        if dtype == "array_list" {
            assert!(matches!(&inputs["samples"], Value::ArrayList(values) if values.len() == 1));
        }
    }
}

#[tokio::test]
async fn transcription_prefers_returned_usage_over_duration_or_latency() {
    let adapter =
        OpenAITranscriptionsAdapter::from_signature(&transcription_signature("float32", true))
            .unwrap();
    for usage in [
        json!({"type":"duration", "seconds":12.5}),
        json!({
            "type":"tokens", "input_tokens":20, "output_tokens":2, "total_tokens":22,
            "input_token_details":{"audio_tokens":18, "text_tokens":2}
        }),
    ] {
        let mut prediction = prediction(vec![
            Value::String("Hello".into()),
            remote(Dtype::Dict, usage.clone()),
        ]);
        prediction.latency = Some(9999.);
        let response = adapter
            .adapt(
                prediction,
                Some(OpenAITranscriptionsOptions { duration: Some(1.) }),
            )
            .await
            .unwrap();
        assert_eq!(serde_json::to_value(response.usage).unwrap(), usage);
    }
}

#[tokio::test]
async fn transcription_without_usage_requires_valid_input_duration() {
    let adapter =
        OpenAITranscriptionsAdapter::from_signature(&transcription_signature("array_list", false))
            .unwrap();
    let response = adapter
        .adapt(
            prediction(vec![Value::String("Hello".into())]),
            Some(OpenAITranscriptionsOptions {
                duration: Some(2.5),
            }),
        )
        .await
        .unwrap();
    assert_eq!(
        response.usage,
        TranscriptionUsage::Duration { seconds: 2.5 }
    );
    assert!(adapter
        .adapt(prediction(vec![Value::String("Hello".into())]), None)
        .await
        .is_err());
    for duration in [f64::NAN, -1., f64::INFINITY] {
        assert!(adapter
            .adapt(
                prediction(vec![Value::String("Hello".into())]),
                Some(OpenAITranscriptionsOptions {
                    duration: Some(duration)
                })
            )
            .await
            .is_err());
    }
}

#[tokio::test]
async fn speech_binds_voice_first_and_encodes_pcm_and_wav() {
    let signature = signature(
        json!([
            {"name":"speaker", "dtype":"string", "denotation":"openai.audio.speech.voice"},
            {"name":"text", "dtype":"string"},
            {"name":"speed", "dtype":"float64", "optional":true, "denotation":"openai.audio.speech.speed"}
        ]),
        json!([
            {"name":"metadata", "dtype":"dict"},
            {"name":"audio", "dtype":"float32", "denotation":"audio", "sampleRate":16000}
        ]),
    );
    let adapter = OpenAISpeechAdapter::from_signature(&signature).unwrap();
    let inputs = adapter.inputs("Hello".into(), "alloy".into(), None);
    assert!(matches!(&inputs["text"], Value::String(text) if text == "Hello"));
    assert!(matches!(&inputs["speaker"], Value::String(text) if text == "alloy"));
    assert!(matches!(&inputs["speed"], Value::Double(1.)));
    for response_format in [SpeechResponseFormat::Pcm, SpeechResponseFormat::Wav] {
        let result = Value::Tensor(Tensor {
            shape: vec![2, 2],
            data: TensorData::Float32(vec![0., 0.5, -0.5, 1.]),
        });
        let response = adapter
            .adapt(
                prediction(vec![Value::Null, result]),
                Some(OpenAISpeechOptions {
                    response_format,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
        assert_eq!(response.response.status(), 200);
        if response_format == SpeechResponseFormat::Pcm {
            assert_eq!(response.content.len(), 16);
            assert_eq!(
                response.response.headers()["content-type"],
                "audio/pcm;rate=16000;channels=2;encoding=float;bits=32"
            );
        } else {
            assert!(response.content.starts_with(b"RIFF"));
        }
    }
}

#[test]
fn audio_signatures_reject_ambiguous_and_missing_sample_rates() {
    let mut signature = transcription_signature("float32", false);
    signature.inputs[0].sample_rate = None;
    assert!(OpenAITranscriptionsAdapter::from_signature(&signature).is_err());
    signature.inputs[0].sample_rate = Some(16000);
    signature.outputs.push(signature.outputs[0].clone());
    assert!(OpenAITranscriptionsAdapter::from_signature(&signature).is_err());
}
