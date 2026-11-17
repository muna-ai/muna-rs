/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

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
            .create(crate::beta::openai::TranscriptionCreateParams {
                stream: true,
                ..Default::default()
            })
            .await,
        Err(MunaError::InvalidInput(_))
    ));
    for temperature in [f64::NAN, -0.1, 1.1] {
        assert!(matches!(
            transcriptions
                .create(crate::beta::openai::TranscriptionCreateParams {
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
            .create(crate::beta::openai::SpeechCreateParams {
                stream_format: Some(crate::beta::openai::SpeechStreamFormat::Sse),
                ..Default::default()
            })
            .await,
        Err(MunaError::InvalidInput(_))
    ));
    for speed in [f64::NAN, 0.0, -1.0] {
        assert!(matches!(
            speech
                .create(crate::beta::openai::SpeechCreateParams {
                    speed: Some(speed),
                    ..Default::default()
                })
                .await,
            Err(MunaError::InvalidInput(_))
        ));
    }
}
