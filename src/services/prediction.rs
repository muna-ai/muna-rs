/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use base64::engine::general_purpose::{STANDARD as BASE64, URL_SAFE_NO_PAD};
use base64::Engine;
use futures_core::Stream;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::{Mutex, RwLock};

use crate::c;
use crate::client::{Client, ClientExt, MunaError, RequestInput, Result, SseEvent};
use crate::types::{
    self, Acceleration, Dtype, Prediction, PredictionResource,
    RemotePrediction, RemoteValue, Value,
};

/// Make predictions.
#[derive(Clone)]
pub struct PredictionService {
    client: Arc<dyn Client>,
    cache: Arc<RwLock<HashMap<PredictionCacheKey, LoadedPredictor>>>,
    operation_locks: Arc<Mutex<HashMap<PredictionCacheKey, Arc<Mutex<()>>>>>,
    cache_dir: PathBuf,
}

#[derive(Clone)]
struct LoadedPredictor {
    predictor: Arc<c::Predictor>,
    next_refresh: Arc<Mutex<Option<i64>>>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
struct PredictionCacheKey {
    tag: String,
    target: String,
    configuration_id: Option<String>,
}

#[derive(Deserialize, Serialize)]
struct DiskCachedPrediction {
    version: u8,
    key: PredictionCacheKey,
    cached_at: i64,
    prediction: Prediction,
}

struct PredictionResolution {
    prediction: Prediction,
    next_refresh: Option<i64>,
}

static CACHE_WRITE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

impl PredictionService {
    const CACHE_VERSION: u8 = 1;
    const REFRESH_RETRY_SECONDS: i64 = 60 * 60;

    pub fn new(client: Arc<dyn Client>) -> Self {
        // The client owns the cache location (muna-unity parity); snapshot it
        // here since resource paths never move within a process lifetime.
        let cache_dir = client.cache_path().to_path_buf();
        Self {
            client,
            cache: Arc::new(RwLock::new(HashMap::new())),
            operation_locks: Arc::new(Mutex::new(HashMap::new())),
            cache_dir,
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
        let is_local = inputs.is_none() ||
            is_download_only ||
            is_local_acceleration(acceleration.as_ref());
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
            self.stream_local(tag, inputs, acceleration, None, None).await
        } else {
            self.stream_remote(tag, &inputs, acceleration).await
        }
    }

    /// Delete a predictor that is loaded in memory.
    pub async fn delete(&self, tag: &str) -> Result<bool> {
        let mut cache = self.cache.write().await;
        let previous_len = cache.len();
        cache.retain(|key, _| key.tag != tag);
        Ok(cache.len() != previous_len)
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
        let predictor = self
            .load_predictor(tag, &acceleration, client_id, configuration_id)
            .await?;
        let input_map = c::ValueMap::from_dict(&inputs)?;
        let prediction = predictor.create_prediction(&input_map)?;
        Ok(to_prediction(tag, &prediction))
    }

    async fn stream_local(
        &self,
        tag: &str,
        inputs: HashMap<String, Value>,
        acceleration: Option<Acceleration>,
        client_id: Option<String>,
        configuration_id: Option<String>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Prediction>> + Send>>> {
        let predictor = self
            .load_predictor(tag, &acceleration, client_id, configuration_id)
            .await?;
        let tag = tag.to_string();
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
            .request_as(RequestInput::post("/predictions/remote").body(body))
            .await?;
        parse_remote_prediction(&*self.client, remote).await
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
            .stream_as::<RemotePrediction>(RequestInput::post("/predictions/remote").body(body))
            .await?;
        let client = self.client.clone();
        let stream = async_stream::try_stream! {
            for await event in event_stream {
                let event: SseEvent<RemotePrediction> = event?;
                let prediction = parse_remote_prediction(&*client, event.data).await?;
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
        let key = prediction_cache_key(tag, client_id, configuration_id);
        self.request_raw_prediction(&key, None).await
    }

    async fn request_raw_prediction(
        &self,
        key: &PredictionCacheKey,
        prediction_id: Option<&str>,
    ) -> Result<Prediction> {
        let mut body = serde_json::json!({
            "tag": key.tag,
            "clientId": key.target,
        });
        if let Some(configuration_id) = &key.configuration_id {
            body["configurationId"] = serde_json::Value::String(configuration_id.clone());
        }
        if let Some(prediction_id) = prediction_id {
            body["predictionId"] = serde_json::Value::String(prediction_id.to_string());
        }
        self.client
            .request_as(RequestInput::post("/predictions").body(body))
            .await
    }

    async fn load_predictor(
        &self,
        tag: &str,
        acceleration: &Option<Acceleration>,
        client_id: Option<String>,
        configuration_id: Option<String>,
    ) -> Result<Arc<c::Predictor>> {
        let key = prediction_cache_key(tag, client_id, configuration_id);
        if let Some(loaded) = self.cache.read().await.get(&key).cloned() {
            self.refresh_loaded_prediction(&key, &loaded).await;
            return Ok(loaded.predictor);
        }
        let operation_lock = self.operation_lock(&key).await;
        let _guard = operation_lock.lock().await;
        if let Some(loaded) = self.cache.read().await.get(&key).cloned() {
            return Ok(loaded.predictor);
        }

        let acceleration = acceleration.clone().unwrap_or(Acceleration::LocalAuto);
        let mut resolution = self.get_or_refresh_prediction(&key, false).await?;
        let configuration = self.create_native_configuration(
            tag,
            &acceleration,
            &resolution.prediction
        ).await?;
        let predictor = match c::Predictor::new(&configuration) {
            Ok(predictor) => predictor,
            Err(_) => {
                // The native API does not expose enough detail to distinguish a
                // bad cached token/resource from other creation failures. Clear
                // the complete cache entry and perform one unpinned refetch.
                self.invalidate_prediction_cache(&key).await;
                resolution = self.get_or_refresh_prediction(&key, true).await?;
                let configuration = self.create_native_configuration(
                    tag,
                    &acceleration,
                    &resolution.prediction
                ).await?;
                c::Predictor::new(&configuration)?
            }
        };
        let predictor = Arc::new(predictor);
        let loaded = LoadedPredictor {
            predictor: predictor.clone(),
            next_refresh: Arc::new(Mutex::new(resolution.next_refresh)),
        };
        self.cache.write().await.insert(key, loaded);
        Ok(predictor)
    }

    async fn create_native_configuration(
        &self,
        tag: &str,
        acceleration: &Acceleration,
        prediction: &Prediction,
    ) -> Result<c::Configuration> {
        let config_token = prediction.configuration.clone().ok_or_else(|| {
            MunaError::Prediction(format!(
                "Failed to create {tag} prediction because configuration token is missing"
            ))
        })?;
        let mut configuration = c::Configuration::new()?;
        configuration.set_tag(tag)?;
        configuration.set_token(&config_token)?;
        configuration.set_acceleration(c::acceleration_to_c(acceleration))?;
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
        Ok(configuration)
    }

    async fn get_or_refresh_prediction(
        &self,
        key: &PredictionCacheKey,
        force_refresh: bool,
    ) -> Result<PredictionResolution> {
        let cached = self.read_prediction_cache(key).await;
        if !force_refresh {
            if let Some(cached) = cached.as_ref() {
                let next_refresh = token_refresh_at(&cached.prediction, cached.cached_at);
                if next_refresh.is_none_or(|refresh_at| unix_now() < refresh_at) {
                    return Ok(PredictionResolution {
                        prediction: cached.prediction.clone(),
                        next_refresh,
                    });
                }
                match self
                    .fetch_and_cache_prediction(key, Some(&cached.prediction.id))
                    .await
                {
                    Ok(fresh) => return Ok(fresh),
                    Err(_) => {
                        return Ok(PredictionResolution {
                            prediction: cached.prediction.clone(),
                            next_refresh: Some(unix_now() + Self::REFRESH_RETRY_SECONDS),
                        });
                    }
                }
            }
        }
        self.fetch_and_cache_prediction(key, None).await
    }

    async fn fetch_and_cache_prediction(
        &self,
        key: &PredictionCacheKey,
        prediction_id: Option<&str>,
    ) -> Result<PredictionResolution> {
        let prediction = self.request_raw_prediction(key, prediction_id).await?;
        let prediction = self.create_cached_prediction(&prediction).await?;
        let cached_at = unix_now();
        let next_refresh = token_refresh_at(&prediction, cached_at);
        let cached = DiskCachedPrediction {
            version: Self::CACHE_VERSION,
            key: key.clone(),
            cached_at,
            prediction: prediction.clone(),
        };
        // A cache permission or disk-space problem must not prevent an online
        // prediction from loading successfully.
        let _ = self.write_prediction_cache(key, &cached).await;
        Ok(PredictionResolution {
            prediction,
            next_refresh,
        })
    }

    async fn refresh_loaded_prediction(&self, key: &PredictionCacheKey, loaded: &LoadedPredictor) {
        let now = unix_now();
        if loaded
            .next_refresh
            .lock()
            .await
            .is_none_or(|refresh_at| now < refresh_at)
        {
            return;
        }
        let operation_lock = self.operation_lock(key).await;
        let _guard = operation_lock.lock().await;
        let mut next_refresh = loaded.next_refresh.lock().await;
        if next_refresh.is_none_or(|refresh_at| unix_now() < refresh_at) {
            return;
        }
        let cached = self.read_prediction_cache(key).await;
        let prediction_id = cached.as_ref().map(|cached| cached.prediction.id.as_str());
        *next_refresh = match self.fetch_and_cache_prediction(key, prediction_id).await {
            Ok(fresh) => fresh.next_refresh,
            Err(_) => Some(unix_now() + Self::REFRESH_RETRY_SECONDS),
        };
    }

    async fn operation_lock(&self, key: &PredictionCacheKey) -> Arc<Mutex<()>> {
        let mut locks = self.operation_locks.lock().await;
        locks
            .entry(key.clone())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    }

    fn prediction_cache_path(&self, key: &PredictionCacheKey) -> PathBuf {
        self.cache_dir
            .join("predictions")
            .join(format!("{:016x}.json", prediction_cache_hash(key)))
    }

    async fn read_prediction_cache(
        &self,
        key: &PredictionCacheKey,
    ) -> Option<DiskCachedPrediction> {
        let data = tokio::fs::read(self.prediction_cache_path(key))
            .await
            .ok()?;
        let cached: DiskCachedPrediction = serde_json::from_slice(&data).ok()?;
        (cached.version == Self::CACHE_VERSION && cached.key == *key).then_some(cached)
    }

    async fn write_prediction_cache(
        &self,
        key: &PredictionCacheKey,
        cached: &DiskCachedPrediction,
    ) -> Result<()> {
        let path = self.prediction_cache_path(key);
        let parent = path.parent().ok_or_else(|| {
            MunaError::Prediction("Prediction cache path has no parent".to_string())
        })?;
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| MunaError::Prediction(format!("Failed to create cache: {e}")))?;
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let sequence = CACHE_WRITE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("prediction.json");
        let temporary = parent.join(format!(
            ".{file_name}.{}.{}.{}.tmp",
            std::process::id(),
            nonce,
            sequence
        ));
        let data = serde_json::to_vec(cached)?;
        if let Err(error) = tokio::fs::write(&temporary, data).await {
            return Err(MunaError::Prediction(format!(
                "Failed to write prediction cache: {error}"
            )));
        }
        if let Err(first_error) = tokio::fs::rename(&temporary, &path).await {
            // Windows does not replace an existing destination with rename.
            // Move the old complete entry aside, install the new complete
            // entry, then remove the backup. Cross-process races can choose
            // either valid writer, but never expose partial JSON.
            let backup = parent.join(format!(
                ".{file_name}.{}.{}.{}.backup",
                std::process::id(),
                nonce,
                sequence
            ));
            let replace_result = if tokio::fs::metadata(&path).await.is_ok()
                && tokio::fs::rename(&path, &backup).await.is_ok()
            {
                match tokio::fs::rename(&temporary, &path).await {
                    Ok(()) => {
                        let _ = tokio::fs::remove_file(&backup).await;
                        Ok(())
                    }
                    Err(error) => {
                        if tokio::fs::metadata(&path).await.is_err() {
                            let _ = tokio::fs::rename(&backup, &path).await;
                        } else {
                            let _ = tokio::fs::remove_file(&backup).await;
                        }
                        Err(error)
                    }
                }
            } else {
                Err(first_error)
            };
            if let Err(error) = replace_result {
                let _ = tokio::fs::remove_file(&temporary).await;
                return Err(MunaError::Prediction(format!(
                    "Failed to commit prediction cache: {error}"
                )));
            }
        }
        Ok(())
    }

    async fn invalidate_prediction_cache(&self, key: &PredictionCacheKey) {
        let _ = tokio::fs::remove_file(self.prediction_cache_path(key)).await;
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
            self.client.download(&resource.url, &path, None).await?;
        }
        Ok(PredictionResource {
            url: path.to_string_lossy().into_owned(),
            ..resource.clone()
        })
    }
}

fn prediction_cache_key(
    tag: &str,
    client_id: Option<String>,
    configuration_id: Option<String>,
) -> PredictionCacheKey {
    let target = client_id
        .or_else(|| c::Configuration::get_client_id().ok())
        .unwrap_or_else(|| "rust".to_string());
    let configuration_id = configuration_id.or_else(|| c::Configuration::get_unique_id().ok());
    PredictionCacheKey {
        tag: tag.to_string(),
        target,
        configuration_id,
    }
}

fn parse_configuration_claims<T: serde::de::DeserializeOwned>(config_token: &str) -> Option<T> {
    let payload = config_token.split('.').nth(1)?;
    let payload = URL_SAFE_NO_PAD.decode(payload).ok()?;
    serde_json::from_slice(&payload).ok()
}

fn parse_preload_claim(config_token: &str) -> Vec<PreloadEntry> {
    parse_configuration_claims::<ConfigurationClaims>(config_token)
        .map(|claims| claims.preload)
        .unwrap_or_default()
}

fn token_refresh_at(
    prediction: &Prediction,
    cached_at: i64,
) -> Option<i64> {
    let claims = parse_configuration_claims::<TokenTimingClaims>(prediction.configuration.as_deref()?)?;
    let exp = claims.exp?;
    let issued_at = claims.iat.unwrap_or(cached_at);
    let lifetime = exp.saturating_sub(issued_at);
    if lifetime <= 0 {
        return Some(exp);
    }
    Some(issued_at.saturating_add(lifetime / 2))
}

fn prediction_cache_hash(key: &PredictionCacheKey) -> u64 {
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    hasher.finish()
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

fn chrono_now() -> String {
    unix_now().to_string()
}

fn unix_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .try_into()
        .unwrap_or(i64::MAX)
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

async fn download_value_data(client: &dyn Client, url: &str) -> Result<Vec<u8>> {
    if let Some(data_part) = url.strip_prefix("data:") {
        if let Some((_mime, encoded)) = data_part.split_once(";base64,") {
            let bytes = BASE64
                .decode(encoded)
                .map_err(|e| MunaError::Prediction(format!("Base64 decode error: {e}")))?;
            return Ok(bytes);
        }
    }
    client.fetch(url).await
}

async fn parse_remote_value(client: &dyn Client, rv: &RemoteValue) -> Result<Value> {
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
    client: &dyn Client,
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

#[derive(Deserialize)]
struct TokenTimingClaims {
    #[serde(default)]
    exp: Option<i64>,
    #[serde(default)]
    iat: Option<i64>,
}

#[cfg(test)]
mod tests {
    use super::{
        parse_preload_claim, preload_output, token_refresh_at, unix_now, DiskCachedPrediction,
        PredictionCacheKey, PredictionService, PreloadEntry,
    };
    use crate::client::{Client, MunaClient, MunaError, RequestInput, Result, SseStream};
    use crate::types::{Acceleration, Prediction, PredictionResource, Value};
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use base64::Engine;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    fn token(payload: serde_json::Value) -> String {
        format!(
            "header.{}.signature",
            URL_SAFE_NO_PAD.encode(serde_json::to_vec(&payload).unwrap())
        )
    }

    fn cache_key(target: &str, configuration_id: &str) -> PredictionCacheKey {
        PredictionCacheKey {
            tag: "@user/model".to_string(),
            target: target.to_string(),
            configuration_id: Some(configuration_id.to_string()),
        }
    }

    fn prediction(id: &str, configuration: String) -> Prediction {
        Prediction {
            id: id.to_string(),
            tag: "@user/model".to_string(),
            created: "0".to_string(),
            configuration: Some(configuration),
            resources: None,
            results: None,
            latency: None,
            error: None,
            logs: None,
        }
    }

    fn temp_cache_dir() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "muna-prediction-cache-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    fn offline_service(cache_dir: PathBuf) -> PredictionService {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        drop(listener);
        let mut service = PredictionService::new(Arc::new(MunaClient::new(None, Some(&url))));
        service.cache_dir = cache_dir;
        service
    }

    /// Fully fake `Client` (no network, no `MunaClient`): records download
    /// calls and writes a marker file where the real client would.
    struct FakeClient {
        cache_dir: PathBuf,
        downloads: std::sync::Mutex<Vec<String>>,
    }

    #[async_trait::async_trait]
    impl Client for FakeClient {
        fn url(&self) -> &str {
            "http://fake.invalid"
        }

        fn cache_path(&self) -> &Path {
            &self.cache_dir
        }

        async fn request(&self, _input: RequestInput) -> Result<serde_json::Value> {
            Err(MunaError::Prediction("fake client has no API".into()))
        }

        async fn stream(&self, _input: RequestInput) -> Result<SseStream<serde_json::Value>> {
            Err(MunaError::Prediction("fake client has no API".into()))
        }

        async fn fetch(&self, _url: &str) -> Result<Vec<u8>> {
            Ok(Vec::new())
        }

        async fn download(
            &self,
            url: &str,
            path: &Path,
            _progress: Option<crate::client::DownloadProgressFn>,
        ) -> Result<()> {
            self.downloads.lock().unwrap().push(url.to_string());
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(path, b"fake").unwrap();
            Ok(())
        }

        async fn upload(&self, _path: &Path) -> Result<String> {
            Err(MunaError::Prediction("fake client has no API".into()))
        }
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
    fn parses_exp_and_refreshes_at_half_life() {
        let expiring_prediction = prediction(
            "pred",
            token(serde_json::json!({ "iat": 1_000, "exp": 2_000 })),
        );
        assert_eq!(token_refresh_at(&expiring_prediction, 1_000), Some(1_500));

        let no_exp = prediction("legacy", token(serde_json::json!({ "iat": 1_000 })));
        assert_eq!(token_refresh_at(&no_exp, 1_000), None);

        let malformed_preload = prediction(
            "malformed-preload",
            token(serde_json::json!({
                "iat": 1_000,
                "exp": 2_000,
                "preload": [{"unexpected": true}]
            })),
        );
        assert_eq!(token_refresh_at(&malformed_preload, 1_000), Some(1_500));
    }

    #[test]
    fn cache_key_includes_target_and_configuration_identity() {
        assert_ne!(
            cache_key("linux-x86_64", "device-a"),
            cache_key("linux-aarch64", "device-a")
        );
        assert_ne!(
            cache_key("linux-x86_64", "device-a"),
            cache_key("linux-x86_64", "device-b")
        );
    }

    #[tokio::test]
    async fn injected_client_receives_resource_downloads() {
        let cache_dir = temp_cache_dir();
        let fake = Arc::new(FakeClient {
            cache_dir: cache_dir.clone(),
            downloads: std::sync::Mutex::new(Vec::new()),
        });
        let service = PredictionService::new(fake.clone());
        let source = Prediction {
            resources: Some(vec![PredictionResource {
                kind: "dso".into(),
                url: "https://cdn.example/resources/libfake.so".into(),
                name: Some("libfake.so".into()),
            }]),
            ..prediction("pred", token(serde_json::json!({})))
        };
        let cached = service.create_cached_prediction(&source).await.unwrap();
        // The download went through the injected client...
        assert_eq!(
            fake.downloads.lock().unwrap().as_slice(),
            ["https://cdn.example/resources/libfake.so"]
        );
        // ...and the returned resource URL points at the local file it wrote.
        let local = &cached.resources.unwrap()[0].url;
        assert!(std::path::Path::new(local).exists());
        assert!(local.starts_with(cache_dir.to_str().unwrap()));
    }

    #[tokio::test]
    async fn expired_refresh_failure_uses_persistent_cached_token() {
        let cache_dir = temp_cache_dir();
        let service = offline_service(cache_dir.clone());
        let key = cache_key("linux-x86_64", "device-a");
        let cached = DiskCachedPrediction {
            version: PredictionService::CACHE_VERSION,
            key: key.clone(),
            cached_at: 1,
            prediction: prediction(
                "cached-prediction",
                token(serde_json::json!({ "iat": 0, "exp": 1 })),
            ),
        };
        service.write_prediction_cache(&key, &cached).await.unwrap();

        let resolved = service
            .get_or_refresh_prediction(&key, false)
            .await
            .unwrap();
        assert_eq!(resolved.prediction.id, "cached-prediction");
        assert!(resolved.next_refresh.unwrap() > unix_now());
        let _ = std::fs::remove_dir_all(cache_dir);
    }

    #[tokio::test]
    async fn first_load_without_cache_must_contact_api() {
        let cache_dir = temp_cache_dir();
        let service = offline_service(cache_dir.clone());
        let key = cache_key("linux-x86_64", "device-a");

        assert!(service
            .get_or_refresh_prediction(&key, false)
            .await
            .is_err());
        let _ = std::fs::remove_dir_all(cache_dir);
    }

    #[tokio::test]
    async fn concurrent_cache_writes_leave_complete_entry() {
        let cache_dir = temp_cache_dir();
        let service = offline_service(cache_dir.clone());
        let key = cache_key("linux-x86_64", "device-a");
        let first = DiskCachedPrediction {
            version: PredictionService::CACHE_VERSION,
            key: key.clone(),
            cached_at: 1,
            prediction: prediction("first", token(serde_json::json!({ "exp": 10 }))),
        };
        let second = DiskCachedPrediction {
            version: PredictionService::CACHE_VERSION,
            key: key.clone(),
            cached_at: 2,
            prediction: prediction("second", token(serde_json::json!({ "exp": 20 }))),
        };

        let (a, b) = tokio::join!(
            service.write_prediction_cache(&key, &first),
            service.write_prediction_cache(&key, &second)
        );
        a.unwrap();
        b.unwrap();
        let cached = service.read_prediction_cache(&key).await.unwrap();
        assert!(matches!(cached.prediction.id.as_str(), "first" | "second"));
        let _ = std::fs::remove_dir_all(cache_dir);
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
