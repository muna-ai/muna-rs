/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use futures_core::Stream;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use tokio::io::AsyncWriteExt;

use crate::c;
use crate::client::{MunaClient, MunaError, RequestInput, Result};
use crate::types::{Acceleration, Prediction, PredictionResource, Value};

/// Make local predictions.
#[derive(Clone)]
pub struct LocalPredictionService {
    client: Arc<MunaClient>,
    cache: Arc<tokio::sync::RwLock<HashMap<String, Arc<c::Predictor>>>>,
    cache_dir: PathBuf,
}

impl LocalPredictionService {

    pub fn new(client: Arc<MunaClient>) -> Self {
        let cache_dir = get_cache_dir();
        Self {
            client,
            cache: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
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
        let inputs = match inputs {
            Some(inputs) => inputs,
            None => {
                return self
                    .create_raw_prediction(tag, client_id, configuration_id)
                    .await
            }
        };
        if inputs.is_empty() {
            let prediction = self
                .create_raw_prediction(tag, client_id, configuration_id)
                .await?;
            self.download_prediction_resources(&prediction).await?;
            return Ok(prediction);
        }
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

    /// Stream a prediction.
    pub async fn stream(
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

    /// Delete a predictor that is loaded in memory.
    pub async fn delete(&self, tag: &str) -> Result<bool> {
        let mut cache = self.cache.write().await;
        Ok(cache.remove(tag).is_some())
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
        let config_token = prediction.configuration.as_deref().ok_or_else(|| {
            MunaError::Prediction(format!(
                "Failed to create {tag} prediction because configuration token is missing"
            ))
        })?;
        let mut configuration = c::Configuration::new()?;
        configuration.set_tag(tag)?;
        configuration.set_token(&config_token)?;
        configuration.set_acceleration(c::acceleration_to_c(&acceleration))?;
        for (kind, path) in self.download_prediction_resources(&prediction).await? {
            configuration.add_resource(&kind, path.to_string_lossy().as_ref())?;
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
            .and_then(|s| s.last())
            .unwrap_or("resource");
        let mut path = self.cache_dir.join(stem);
        if let Some(name) = &resource.name {
            path = path.join(name);
        }
        path
    }

    async fn download_prediction_resources(
        &self,
        prediction: &Prediction,
    ) -> Result<Vec<(String, PathBuf)>> {
        let mut paths = Vec::new();
        if let Some(resources) = &prediction.resources {
            for resource in resources {
                let path = self.get_resource_path(resource);
                if !path.exists() {
                    self.download_resource(&resource.url, &path).await?;
                }
                paths.push((resource.kind.clone(), path));
            }
        }
        Ok(paths)
    }

    async fn download_resource(&self, url: &str, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                MunaError::Prediction(format!("Failed to create cache directory: {e}"))
            })?;
        }
        let tmp_path = std::env::temp_dir().join(format!("muna-{}", uuid_v4()));
        let mut file = tokio::fs::File::create(&tmp_path)
            .await
            .map_err(|e| MunaError::Prediction(format!("Failed to create temp file: {e}")))?;
        let mut response = self.client.download(url).await?;
        while let Some(chunk) = response.chunk().await? {
            file.write_all(&chunk)
                .await
                .map_err(|e| MunaError::Prediction(format!("Failed to write chunk: {e}")))?;
        }
        file.flush()
            .await
            .map_err(|e| MunaError::Prediction(format!("Failed to flush file: {e}")))?;
        drop(file);
        tokio::fs::rename(&tmp_path, path).await.map_err(|e| {
            MunaError::Prediction(format!(
                "Failed to move resource to {}: {e}",
                path.display()
            ))
        })?;
        Ok(())
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

fn uuid_v4() -> String {
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let r: u64 = (t ^ (t >> 32)) as u64;
    format!("{:016x}", r)
}
