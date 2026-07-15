/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use base64::engine::general_purpose::{STANDARD as BASE64, URL_SAFE_NO_PAD};
use base64::Engine;
use futures_core::Stream;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use crate::c;
use crate::client::{MunaClient, MunaError, RequestInput, Result, SseEvent};
use crate::types::{
    self, Acceleration, Dtype, Prediction, PredictionResource,
    RemotePrediction, RemoteValue, Value,
};

/// Make predictions.
#[derive(Clone)]
pub struct PredictionService {
    client: Arc<MunaClient>,
    cache: Arc<tokio::sync::RwLock<HashMap<String, Arc<c::Predictor>>>>,
    cache_dir: PathBuf,
}

impl PredictionService {

    pub fn new(client: Arc<MunaClient>) -> Self {
        Self {
            client,
            cache: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            cache_dir: get_cache_dir(),
        }
    }

    /// Create a prediction.
    pub async fn create(
        &self,
        tag: &str,
        inputs: Option<HashMap<String, Value>>,
        acceleration: Option<Acceleration>,
        client_id: Option<String>,
        configuration_id: Option<String>,
    ) -> Result<Prediction> {
        let is_download_only = inputs.as_ref().is_some_and(HashMap::is_empty);
        let is_local = inputs.is_none() || is_download_only || is_local_acceleration(acceleration.as_ref());
        if is_local {
            self.create_local(tag, inputs, acceleration, client_id, configuration_id)
                .await
        } else {
            self.create_remote(tag, &inputs.unwrap(), acceleration)
                .await
        }
    }

    /// Stream a prediction.
    pub async fn stream(
        &self,
        tag: &str,
        inputs: HashMap<String, Value>,
        acceleration: Option<Acceleration>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Prediction>> + Send>>> {
        if is_local_acceleration(acceleration.as_ref()) {
            self.stream_local(tag, inputs, acceleration).await
        } else {
            self.stream_remote(tag, &inputs, acceleration).await
        }
    }

    /// Delete a predictor that is loaded in memory.
    pub async fn delete(&self, tag: &str) -> Result<bool> {
        let mut cache = self.cache.write().await;
        Ok(cache.remove(tag).is_some())
    }

    async fn create_local(
        &self,
        tag: &str,
        inputs: Option<HashMap<String, Value>>,
        acceleration: Option<Acceleration>,
        client_id: Option<String>,
        configuration_id: Option<String>,
    ) -> Result<Prediction> {
        let inputs = match inputs {
            Some(inputs) if inputs.is_empty() => {
                let prediction = self
                    .create_raw_prediction(tag, client_id, configuration_id)
                    .await?;
                self.create_cached_prediction(&prediction).await?;
                return Ok(prediction);
            }
            Some(inputs) => inputs,
            None => {
                return self
                    .create_raw_prediction(tag, client_id, configuration_id)
                    .await
            }
        };
        self.load_predictor(tag, &acceleration, client_id, configuration_id)
            .await?;
        let predictor = {
            let cache = self.cache.read().await;
            cache[tag].clone()
        };
        let input_map = c::ValueMap::from_dict(&inputs)?;
        let prediction = predictor.create_prediction(&input_map)?;
        Ok(to_prediction(tag, &prediction))
    }

    async fn stream_local(
        &self,
        tag: &str,
        inputs: HashMap<String, Value>,
        acceleration: Option<Acceleration>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Prediction>> + Send>>> {
        self.load_predictor(tag, &acceleration, None, None).await?;
        let tag = tag.to_string();
        let predictor = {
            let cache = self.cache.read().await;
            cache[tag.as_str()].clone()
        };
        let input_map = c::ValueMap::from_dict(&inputs)?;
        let stream_handle = c::PredictionStream::create(predictor.raw_ptr(), &input_map)?;
        let stream = async_stream::try_stream! {
            for prediction in stream_handle {
                let prediction = prediction?;
                yield to_prediction(&tag, &prediction);
            }
        };
        Ok(Box::pin(stream))
    }

    async fn create_remote(
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
        let remote: RemotePrediction = self
            .client
            .request(RequestInput::post("/predictions/remote").body(body))
            .await?;
        parse_remote_prediction(&self.client, remote).await
    }

    async fn stream_remote(
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
        let event_stream = self
            .client
            .stream::<RemotePrediction>(RequestInput::post("/predictions/remote").body(body))
            .await?;
        let client = self.client.clone();
        let stream = async_stream::try_stream! {
            for await event in event_stream {
                let event: SseEvent<RemotePrediction> = event?;
                let prediction = parse_remote_prediction(&client, event.data).await?;
                yield prediction;
            }
        };
        Ok(Box::pin(stream))
    }

    async fn create_raw_prediction(
        &self,
        tag: &str,
        client_id: Option<String>,
        configuration_id: Option<String>,
    ) -> Result<Prediction> {
        let client_id = client_id
            .or_else(|| c::Configuration::get_client_id().ok())
            .unwrap_or_else(|| "rust".to_string());
        let configuration_id = configuration_id.or_else(|| c::Configuration::get_unique_id().ok());
        let mut body = serde_json::json!({
            "tag": tag,
            "clientId": client_id,
        });
        if let Some(config_id) = configuration_id {
            body["configurationId"] = serde_json::Value::String(config_id);
        }
        self.client
            .request(RequestInput::post("/predictions").body(body))
            .await
    }

    async fn load_predictor(
        &self,
        tag: &str,
        acceleration: &Option<Acceleration>,
        client_id: Option<String>,
        configuration_id: Option<String>,
    ) -> Result<()> {
        {
            let cache = self.cache.read().await;
            if cache.contains_key(tag) {
                return Ok(());
            }
        }
        let acceleration = acceleration.clone().unwrap_or(Acceleration::LocalAuto);
        let prediction = self
            .create_raw_prediction(tag, client_id, configuration_id)
            .await?;
        let prediction = self.create_cached_prediction(&prediction).await?;
        let config_token = prediction.configuration.clone().ok_or_else(|| {
            MunaError::Prediction(format!(
                "Failed to create {tag} prediction because configuration token is missing"
            ))
        })?;
        let mut configuration = c::Configuration::new()?;
        configuration.set_tag(tag)?;
        configuration.set_token(&config_token)?;
        configuration.set_acceleration(c::acceleration_to_c(&acceleration))?;
        if let Some(resources) = &prediction.resources {
            for resource in resources {
                configuration.add_resource(&resource.kind, &resource.url)?;
            }
        }
        for entry in parse_preload_claim(&config_token) {
            // Box::pin breaks the `create -> load_predictor -> create` async
            // recursion that would otherwise make the future type infinite.
            let inputs = HashMap::from([("_".to_string(), Value::Null)]);
            let prediction = Box::pin(self.create(
                &entry.tag,
                Some(inputs),
                Some(entry.acceleration.clone()),
                None,
                None,
            ))
            .await?;
            let value = preload_output(&prediction, &entry.tag)?;
            configuration.set_metadata(&entry.metadata, value)?;
        }
        let predictor = c::Predictor::new(&configuration)?;
        let mut cache = self.cache.write().await;
        cache.entry(tag.to_string()).or_insert(Arc::new(predictor));
        Ok(())
    }

    fn get_resource_path(&self, resource: &PredictionResource) -> PathBuf {
        let url = url::Url::parse(&resource.url).ok();
        let stem = url
            .as_ref()
            .and_then(|u| u.path_segments())
            .and_then(|mut s| s.next_back())
            .unwrap_or("resource");
        let mut path = self.cache_dir.join(stem);
        if let Some(name) = &resource.name {
            path = path.join(name);
        }
        path
    }

    /// Download a prediction's resources and return a new prediction whose
    /// resource URLs point to the downloaded local paths.
    async fn create_cached_prediction(&self, prediction: &Prediction) -> Result<Prediction> {
        let resources = match &prediction.resources {
            Some(resources) => {
                let mut materialized = Vec::with_capacity(resources.len());
                for resource in resources {
                    materialized.push(self.download_resource(resource).await?);
                }
                Some(materialized)
            }
            None => None,
        };
        Ok(Prediction {
            resources,
            ..prediction.clone()
        })
    }

    /// Download a single resource and return it with its URL set to the local
    /// downloaded path.
    async fn download_resource(&self, resource: &PredictionResource) -> Result<PredictionResource> {
        let path = self.get_resource_path(resource);
        if !path.exists() {
            self.client.download(&resource.url, &path).await?;
        }
        Ok(PredictionResource {
            url: path.to_string_lossy().into_owned(),
            ..resource.clone()
        })
    }
}

fn parse_preload_claim(config_token: &str) -> Vec<PreloadEntry> {
    let Some(payload) = config_token.split('.').nth(1) else {
        return Vec::new();
    };
    let Ok(payload) = URL_SAFE_NO_PAD.decode(payload) else {
        return Vec::new();
    };
    serde_json::from_slice::<ConfigurationClaims>(&payload)
        .map(|claims| claims.preload)
        .unwrap_or_default()
}

fn is_local_acceleration(acceleration: Option<&Acceleration>) -> bool {
    match acceleration {
        Some(
            Acceleration::LocalAuto
            | Acceleration::LocalCpu
            | Acceleration::LocalGpu
            | Acceleration::LocalNpu,
        ) => true,
        Some(Acceleration::Adaptive(value)) => value.starts_with("local_"),
        None => true,
        _ => false,
    }
}

fn to_prediction(tag: &str, prediction: &c::Prediction) -> Prediction {
    let results = prediction.results().ok().map(|map| {
        let size = map.len();
        (0..size)
            .filter_map(|i| {
                let key = map.key(i).ok()?;
                let value = map.get(&key).ok()?;
                value.to_object().ok()
            })
            .collect()
    });
    Prediction {
        id: prediction.id().unwrap_or_default(),
        tag: tag.to_string(),
        created: chrono_now(),
        configuration: None,
        resources: None,
        results,
        latency: prediction.latency().ok(),
        error: prediction.error().ok().flatten(),
        logs: prediction.logs().ok().flatten(),
    }
}

fn preload_output<'a>(prediction: &'a Prediction, tag: &str) -> Result<&'a str> {
    if let Some(error) = prediction.error.as_deref() {
        return Err(MunaError::Prediction(format!(
            "Failed to preload {tag}: {error}"
        )));
    }
    match prediction
        .results
        .as_ref()
        .and_then(|results| results.first())
    {
        Some(Value::String(value)) => Ok(value),
        Some(_) => Err(MunaError::Prediction(format!(
            "Failed to preload {tag} because its first result is not a string"
        ))),
        None => Err(MunaError::Prediction(format!(
            "Failed to preload {tag} because it returned no results"
        ))),
    }
}

fn get_cache_dir() -> PathBuf {
    let dir = get_muna_home().join("cache");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

fn get_muna_home() -> PathBuf {
    let candidates = std::env::var("MUNA_HOME")
        .ok()
        .map(PathBuf::from)
        .into_iter()
        .chain(home::home_dir().map(|h| h.join(".fxn")))
        .chain(std::iter::once(std::env::temp_dir().join(".fxn")));
    for dir in candidates {
        if std::fs::create_dir_all(&dir).is_ok() {
            let test = dir.join(".muna_write_test");
            if std::fs::write(&test, "muna").is_ok() {
                let _ = std::fs::remove_file(&test);
                return dir;
            }
        }
    }
    std::env::temp_dir().join(".fxn")
}

fn chrono_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("{secs}")
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
        Value::Null => Ok(RemoteValue {
            data: None,
            dtype: Dtype::Null,
        }),
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
            Ok(RemoteValue {
                data: Some(data),
                dtype,
            })
        }
        Value::String(s) => {
            let data = upload_value_data(s.as_bytes(), "text/plain");
            Ok(RemoteValue {
                data: Some(data),
                dtype: Dtype::String,
            })
        }
        Value::List(v) => {
            let json = serde_json::to_string(v)?;
            let data = upload_value_data(json.as_bytes(), "application/json");
            Ok(RemoteValue {
                data: Some(data),
                dtype: Dtype::List,
            })
        }
        Value::Dict(v) => {
            let json = serde_json::to_string(v)?;
            let data = upload_value_data(json.as_bytes(), "application/json");
            Ok(RemoteValue {
                data: Some(data),
                dtype: Dtype::Dict,
            })
        }
        Value::Image(_) => {
            let fxn_value = c::Value::from_object(value)?;
            let buffer = fxn_value.serialize(None)?;
            let data = upload_value_data(&buffer, "image/png");
            Ok(RemoteValue {
                data: Some(data),
                dtype: Dtype::Image,
            })
        }
        Value::ArrayList(_) => {
            let fxn_value = c::Value::from_object(value)?;
            let buffer = fxn_value.serialize(None)?;
            let data = upload_value_data(&buffer, "application/x-npz");
            Ok(RemoteValue {
                data: Some(data),
                dtype: Dtype::ArrayList,
            })
        }
        Value::ImageList(_) => {
            let fxn_value = c::Value::from_object(value)?;
            let buffer = fxn_value.serialize(None)?;
            let data = upload_value_data(&buffer, "image/avif");
            Ok(RemoteValue {
                data: Some(data),
                dtype: Dtype::ImageList,
            })
        }
        Value::Binary(bytes) => {
            let data = upload_value_data(bytes, "application/octet-stream");
            Ok(RemoteValue {
                data: Some(data),
                dtype: Dtype::Binary,
            })
        }
    }
}

fn upload_value_data(buffer: &[u8], mime: &str) -> String {
    let encoded = BASE64.encode(buffer);
    format!("data:{mime};base64,{encoded}")
}

async fn download_value_data(client: &MunaClient, url: &str) -> Result<Vec<u8>> {
    if let Some(data_part) = url.strip_prefix("data:") {
        if let Some((_mime, encoded)) = data_part.split_once(";base64,") {
            let bytes = BASE64
                .decode(encoded)
                .map_err(|e| MunaError::Prediction(format!("Base64 decode error: {e}")))?;
            return Ok(bytes);
        }
    }
    let response = client.http().get(url).send().await?;
    let status = response.status();
    if !status.is_success() {
        return Err(MunaError::Api {
            message: format!("Failed to download resource: {status}"),
            status: status.as_u16(),
        });
    }
    Ok(response.bytes().await?.to_vec())
}

async fn parse_remote_value(client: &MunaClient, rv: &RemoteValue) -> Result<Value> {
    if rv.dtype == Dtype::Null {
        return Ok(Value::Null);
    }
    let url = rv
        .data
        .as_deref()
        .ok_or_else(|| MunaError::Prediction("Remote value has no data URL".into()))?;
    let buffer = download_value_data(client, url).await?;
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

async fn parse_remote_prediction(
    client: &MunaClient,
    prediction: RemotePrediction,
) -> Result<Prediction> {
    let results = match prediction.results {
        Some(remote_values) => {
            let mut values = Vec::with_capacity(remote_values.len());
            for rv in &remote_values {
                values.push(parse_remote_value(client, rv).await?);
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

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct PreloadEntry {
    tag: String,
    acceleration: Acceleration,
    metadata: String,
}

#[derive(Deserialize)]
struct ConfigurationClaims {
    #[serde(default)]
    preload: Vec<PreloadEntry>,
}

#[cfg(test)]
mod tests {
    use super::{parse_preload_claim, preload_output, PreloadEntry};
    use crate::types::{Acceleration, Prediction, Value};
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use base64::Engine;

    fn token(payload: serde_json::Value) -> String {
        format!(
            "header.{}.signature",
            URL_SAFE_NO_PAD.encode(serde_json::to_vec(&payload).unwrap())
        )
    }

    #[test]
    fn parses_preload_claim() {
        let entries = parse_preload_claim(&token(serde_json::json!({
            "tag": "@user/model",
            "preload": [
                {
                    "tag": "@user/model:decode",
                    "acceleration": "remote_b200",
                    "metadata": "tent_endpoint"
                },
                {
                    "tag": "@user/model:drafter",
                    "acceleration": "remote_h200",
                    "metadata": "draft_endpoint"
                }
            ]
        })));

        assert_eq!(
            entries,
            vec![
                PreloadEntry {
                    tag: "@user/model:decode".to_string(),
                    acceleration: Acceleration::RemoteB200,
                    metadata: "tent_endpoint".to_string(),
                },
                PreloadEntry {
                    tag: "@user/model:drafter".to_string(),
                    acceleration: Acceleration::RemoteH200,
                    metadata: "draft_endpoint".to_string(),
                }
            ]
        );
    }

    #[test]
    fn ignores_unknown_preload_fields() {
        let entries = parse_preload_claim(&token(serde_json::json!({
            "preload": [{
                "tag": "@user/model:decode",
                "acceleration": "remote_future_accelerator",
                "metadata": "tent_endpoint",
                "future": {"value": true}
            }]
        })));

        assert_eq!(
            entries,
            vec![PreloadEntry {
                tag: "@user/model:decode".to_string(),
                acceleration: Acceleration::Adaptive("remote_future_accelerator".to_string()),
                metadata: "tent_endpoint".to_string(),
            }]
        );
    }

    #[test]
    fn absent_or_malformed_preload_claim_is_empty() {
        assert!(parse_preload_claim(&token(serde_json::json!({
            "tag": "@user/model"
        })))
        .is_empty());
        assert!(parse_preload_claim("not-a-jwt").is_empty());
        assert!(parse_preload_claim("header.%%%.signature").is_empty());
        assert!(parse_preload_claim(&token(serde_json::json!({
            "preload": [{"tag": "@user/model:decode"}]
        })))
        .is_empty());
    }

    #[test]
    fn preload_output_requires_first_string_result() {
        let prediction = Prediction {
            id: "pred".to_string(),
            tag: "@user/model:decode".to_string(),
            created: "0".to_string(),
            configuration: None,
            resources: None,
            results: Some(vec![Value::String("endpoint".to_string())]),
            latency: None,
            error: None,
            logs: None,
        };
        assert_eq!(
            preload_output(&prediction, "@user/model:decode").unwrap(),
            "endpoint"
        );

        let mut missing = prediction.clone();
        missing.results = None;
        assert!(preload_output(&missing, "@user/model:decode").is_err());

        let mut wrong_type = prediction.clone();
        wrong_type.results = Some(vec![Value::Bool(true)]);
        assert!(preload_output(&wrong_type, "@user/model:decode").is_err());

        let mut failed = prediction;
        failed.error = Some("sidecar failed".to_string());
        assert!(preload_output(&failed, "@user/model:decode").is_err());
    }
}
