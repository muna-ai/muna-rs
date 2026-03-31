/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use futures_core::Stream;
use crate::c;
use crate::client::{MunaClient, MunaError, RequestInput, Result, SseEvent};
use crate::types::{self, Acceleration, Dtype, Prediction, Value};
use super::types::{RemotePrediction, RemoteValue};

/// Make remote predictions.
#[derive(Clone)]
pub struct RemotePredictionService {
    client: Arc<MunaClient>,
}

impl RemotePredictionService {

    pub fn new(client: Arc<MunaClient>) -> Self {
        Self { client }
    }

    /// Create a remote prediction.
    pub async fn create(
        &self,
        tag: &str,
        inputs: &HashMap<String, Value>,
        acceleration: Option<Acceleration>,
    ) -> Result<Prediction> {
        let acceleration = acceleration.unwrap_or(Acceleration::RemoteAuto);
        let input_map = serialize_inputs(inputs)?;
        let client_id = c::Configuration::get_client_id().unwrap_or_else(|_| "rust".to_string());
        let body = serde_json::json!({
            "tag": tag,
            "inputs": input_map,
            "acceleration": acceleration,
            "clientId": client_id,
        });
        let remote: RemotePrediction = self.client.request(
            RequestInput::post("/predictions/remote").body(body)
        ).await?;
        parse_remote_prediction(remote).await
    }

    /// Stream a remote prediction.
    pub async fn stream(
        &self,
        tag: &str,
        inputs: &HashMap<String, Value>,
        acceleration: Option<Acceleration>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Prediction>> + Send>>> {
        let acceleration = acceleration.unwrap_or(Acceleration::RemoteAuto);
        let input_map = serialize_inputs(inputs)?;
        let client_id = c::Configuration::get_client_id().unwrap_or_else(|_| "rust".to_string());
        let body = serde_json::json!({
            "tag": tag,
            "inputs": input_map,
            "acceleration": acceleration,
            "clientId": client_id,
            "stream": true,
        });
        let event_stream = self.client.stream::<RemotePrediction>(
            RequestInput::post("/predictions/remote").body(body)
        ).await?;
        let stream = async_stream::try_stream! {
            for await event in event_stream {
                let event: SseEvent<RemotePrediction> = event?;
                let prediction = parse_remote_prediction(event.data).await?;
                yield prediction;
            }
        };
        Ok(Box::pin(stream))
    }
}

fn serialize_inputs(inputs: &HashMap<String, Value>) -> Result<serde_json::Value> {
    let mut map = serde_json::Map::new();
    for (name, value) in inputs {
        let remote = create_remote_value(value)?;
        map.insert(name.clone(), serde_json::to_value(&remote)?);
    }
    Ok(serde_json::Value::Object(map))
}

fn create_remote_value(value: &Value) -> Result<RemoteValue> {
    match value {
        Value::Null => Ok(RemoteValue { data: None, dtype: Dtype::Null }),
        Value::Float(v) => {
            let tensor = types::Tensor {
                data: types::TensorData::Float32(vec![*v]),
                shape: vec![],
            };
            create_remote_value(&Value::Tensor(tensor))
        }
        Value::Double(v) => {
            let tensor = types::Tensor {
                data: types::TensorData::Float32(vec![*v as f32]),
                shape: vec![],
            };
            create_remote_value(&Value::Tensor(tensor))
        }
        Value::Int(v) => {
            let tensor = types::Tensor {
                data: types::TensorData::Int32(vec![*v]),
                shape: vec![],
            };
            create_remote_value(&Value::Tensor(tensor))
        }
        Value::Long(v) => {
            let tensor = types::Tensor {
                data: types::TensorData::Int64(vec![*v]),
                shape: vec![],
            };
            create_remote_value(&Value::Tensor(tensor))
        }
        Value::Bool(v) => {
            let tensor = types::Tensor {
                data: types::TensorData::Bool(vec![*v]),
                shape: vec![],
            };
            create_remote_value(&Value::Tensor(tensor))
        }
        Value::Tensor(tensor) => {
            let fxn_value = c::Value::from_object(value)?;
            let buffer = fxn_value.serialize(None)?;
            let data = upload_value_data(&buffer, "application/octet-stream");
            let dtype = tensor.data.dtype();
            Ok(RemoteValue { data: Some(data), dtype })
        }
        Value::String(s) => {
            let data = upload_value_data(s.as_bytes(), "text/plain");
            Ok(RemoteValue { data: Some(data), dtype: Dtype::String })
        }
        Value::List(v) => {
            let json = serde_json::to_string(v)?;
            let data = upload_value_data(json.as_bytes(), "application/json");
            Ok(RemoteValue { data: Some(data), dtype: Dtype::List })
        }
        Value::Dict(v) => {
            let json = serde_json::to_string(v)?;
            let data = upload_value_data(json.as_bytes(), "application/json");
            Ok(RemoteValue { data: Some(data), dtype: Dtype::Dict })
        }
        Value::Image(_) => {
            let fxn_value = c::Value::from_object(value)?;
            let buffer = fxn_value.serialize(None)?;
            let data = upload_value_data(&buffer, "image/png");
            Ok(RemoteValue { data: Some(data), dtype: Dtype::Image })
        }
        Value::ArrayList(_) => {
            let fxn_value = c::Value::from_object(value)?;
            let buffer = fxn_value.serialize(None)?;
            let data = upload_value_data(&buffer, "application/x-npz");
            Ok(RemoteValue { data: Some(data), dtype: Dtype::ArrayList })
        }
        Value::ImageList(_) => {
            let fxn_value = c::Value::from_object(value)?;
            let buffer = fxn_value.serialize(None)?;
            let data = upload_value_data(&buffer, "image/avif");
            Ok(RemoteValue { data: Some(data), dtype: Dtype::ImageList })
        }
        Value::Binary(bytes) => {
            let data = upload_value_data(bytes, "application/octet-stream");
            Ok(RemoteValue { data: Some(data), dtype: Dtype::Binary })
        }
    }
}

fn upload_value_data(buffer: &[u8], mime: &str) -> String {
    let encoded = BASE64.encode(buffer);
    format!("data:{mime};base64,{encoded}")
}

async fn download_value_data(url: &str) -> Result<Vec<u8>> {
    if let Some(data_part) = url.strip_prefix("data:") {
        if let Some((_mime, encoded)) = data_part.split_once(";base64,") {
            let bytes = BASE64.decode(encoded)
                .map_err(|e| MunaError::Prediction(format!("Base64 decode error: {e}")))?;
            return Ok(bytes);
        }
    }
    let response = reqwest::get(url).await?;
    let bytes = response.bytes().await?;
    Ok(bytes.to_vec())
}

async fn parse_remote_value(rv: &RemoteValue) -> Result<Value> {
    if rv.dtype == Dtype::Null {
        return Ok(Value::Null);
    }
    let url = rv.data.as_deref().ok_or_else(|| {
        MunaError::Prediction("Remote value has no data URL".into())
    })?;
    let buffer = download_value_data(url).await?;
    match rv.dtype {
        Dtype::Null => Ok(Value::Null),
        dtype if c::is_tensor_dtype(dtype) => {
            let fxn_value = c::Value::from_bytes(&buffer, "application/vnd.muna.tensor")?;
            fxn_value.to_object()
        }
        Dtype::String => {
            let s = String::from_utf8(buffer)
                .map_err(|e| MunaError::Prediction(format!("UTF-8 decode error: {e}")))?;
            Ok(Value::String(s))
        }
        Dtype::List => {
            let s = String::from_utf8(buffer)
                .map_err(|e| MunaError::Prediction(format!("UTF-8 decode error: {e}")))?;
            let v: Vec<serde_json::Value> = serde_json::from_str(&s)?;
            Ok(Value::List(v))
        }
        Dtype::Dict => {
            let s = String::from_utf8(buffer)
                .map_err(|e| MunaError::Prediction(format!("UTF-8 decode error: {e}")))?;
            let v: serde_json::Map<String, serde_json::Value> = serde_json::from_str(&s)?;
            Ok(Value::Dict(v))
        }
        Dtype::Image => {
            let fxn_value = c::Value::from_bytes(&buffer, "image/*")?;
            fxn_value.to_object()
        }
        Dtype::ArrayList => {
            let fxn_value = c::Value::from_bytes(&buffer, "application/x-npz")?;
            fxn_value.to_object()
        }
        Dtype::ImageList => {
            let fxn_value = c::Value::from_bytes(&buffer, "image/avif")?;
            fxn_value.to_object()
        }
        Dtype::Binary => Ok(Value::Binary(buffer)),
        dtype => Err(MunaError::Prediction(format!(
            "Cannot deserialize remote value with type `{dtype:?}`"
        ))),
    }
}

async fn parse_remote_prediction(prediction: RemotePrediction) -> Result<Prediction> {
    let results = match prediction.results {
        Some(remote_values) => {
            let mut values = Vec::with_capacity(remote_values.len());
            for rv in &remote_values {
                values.push(parse_remote_value(rv).await?);
            }
            Some(values)
        }
        None => None,
    };
    Ok(Prediction {
        id: prediction.id,
        tag: prediction.tag,
        created: prediction.created,
        configuration: None,
        resources: None,
        results,
        latency: prediction.latency,
        error: prediction.error,
        logs: prediction.logs,
    })
}
