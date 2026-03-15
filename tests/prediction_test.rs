/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::collections::HashMap;
use futures_util::StreamExt;
use muna::{Muna, MunaError, Value};

#[tokio::test]
async fn test_create_raw_prediction() {
    let _ = dotenvy::dotenv();
    let muna = Muna::default();
    let prediction = muna.predictions.create(
        "@fxn/greeting",
        None,
        None,
        None,
        None,
    ).await.unwrap();
    assert!(prediction.configuration.is_some());
    assert!(prediction.resources.is_some());
}

#[tokio::test]
async fn test_create_prediction() {
    let _ = dotenvy::dotenv();
    let muna = Muna::default();
    let mut inputs = HashMap::new();
    inputs.insert("radius".to_string(), 4.0f32.into());
    let prediction = muna.predictions.create(
        "@yusuf/area",
        Some(inputs),
        None,
        None,
        None,
    ).await.unwrap();
    assert!(prediction.results.is_some());
    let results = prediction.results.unwrap();
    assert!(matches!(results[0], Value::Float(_) | Value::Double(_)));
}

#[tokio::test]
async fn test_stream_prediction() {
    let _ = dotenvy::dotenv();
    let muna = Muna::default();
    let mut inputs = HashMap::new();
    inputs.insert("sentence".to_string(), "The fat cat sat on the mat.".into());
    let mut stream = muna.predictions.stream(
        "@yusuf/generator",
        inputs,
        None,
    ).await.unwrap();
    let mut count = 0;
    while let Some(prediction) = stream.next().await {
        let prediction = prediction.unwrap();
        assert!(prediction.results.is_some());
        let results = prediction.results.unwrap();
        assert!(matches!(results[0], Value::String(_)));
        count += 1;
    }
    assert!(count > 0);
}

#[tokio::test]
async fn test_create_remote_prediction() {
    let _ = dotenvy::dotenv();
    let muna = Muna::default();
    let mut inputs = HashMap::new();
    inputs.insert("sentence".to_string(), "The fat cat sat on the mat.".into());
    let prediction = muna.beta.predictions.remote.create(
        "@yusuf/generator",
        &inputs,
        None,
    ).await.unwrap();
    assert!(prediction.results.is_some());
    let results = prediction.results.unwrap();
    assert!(matches!(results[0], Value::String(_)));
}

#[tokio::test]
async fn test_stream_remote_prediction() {
    let _ = dotenvy::dotenv();
    let muna = Muna::default();
    let mut inputs = HashMap::new();
    inputs.insert("sentence".to_string(), "The fat cat sat on the mat.".into());
    let mut stream = muna.beta.predictions.remote.stream(
        "@yusuf/generator",
        &inputs,
        None,
    ).await.unwrap();
    let mut count = 0;
    while let Some(prediction) = stream.next().await {
        let prediction = prediction.unwrap();
        assert!(prediction.results.is_some());
        let results = prediction.results.unwrap();
        assert!(matches!(results[0], Value::String(_)));
        count += 1;
    }
    assert!(count > 0);
}

#[tokio::test]
async fn test_create_invalid_prediction() {
    let _ = dotenvy::dotenv();
    let muna = Muna::default();
    let result = muna.predictions.create(
        "@yusu/invalid-predictor",
        None,
        None,
        None,
        None,
    ).await;
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, MunaError::Api { .. }));
}
