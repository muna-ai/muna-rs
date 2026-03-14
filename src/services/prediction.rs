/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::collections::HashMap;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use futures_core::Stream;

use crate::c;
use crate::client::{MunaClient, MunaError, RequestInput, Result};
use crate::types::{Acceleration, Prediction, PredictionResource, Value};

/// Make predictions.
#[derive(Clone)]
pub struct PredictionService {
    client: Arc<MunaClient>,
    cache: Arc<tokio::sync::Mutex<HashMap<String, c::Predictor>>>,
    cache_dir: PathBuf,
}

impl PredictionService {

    pub fn new(client: Arc<MunaClient>) -> Self {
        let cache_dir = get_cache_dir();
        Self {
            client,
            cache: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
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
            None => return self.create_raw_prediction(tag, client_id, configuration_id).await,
        };
        self.load_predictor(tag, &acceleration, client_id, configuration_id).await?;
        let cache = self.cache.lock().await;
        let predictor = &cache[tag];
        let input_map = c::ValueMap::from_dict(&inputs)?;
        let prediction = predictor.create_prediction(&input_map)?;
        Ok(Self::to_prediction(tag, &prediction))
    }

    /// Create a streaming prediction.
    pub async fn stream(
        &self,
        tag: &str,
        inputs: HashMap<String, Value>,
        acceleration: Option<Acceleration>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Prediction>> + Send>>> {
        self.load_predictor(tag, &acceleration, None, None).await?;
        let tag = tag.to_string();
        let cache = self.cache.lock().await;
        let predictor_ptr = cache[tag.as_str()].raw_ptr();
        let input_map = c::ValueMap::from_dict(&inputs)?;
        let stream_handle = c::PredictionStream::create(predictor_ptr, &input_map)?;
        drop(cache);
        let stream = async_stream::try_stream! {
            for prediction in stream_handle {
                let prediction = prediction?;
                yield Self::to_prediction(&tag, &prediction);
            }
        };
        Ok(Box::pin(stream))
    }

    /// Delete a predictor that is loaded in memory.
    pub async fn delete(&self, tag: &str) -> Result<bool> {
        let mut cache = self.cache.lock().await;
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
        let configuration_id = configuration_id
            .or_else(|| c::Configuration::get_unique_id().ok());
        let mut body = serde_json::json!({
            "tag": tag,
            "clientId": client_id,
        });
        if let Some(config_id) = configuration_id {
            body["configurationId"] = serde_json::Value::String(config_id);
        }
        self.client.request(RequestInput::post("/predictions").body(body)).await
    }

    async fn load_predictor(
        &self,
        tag: &str,
        acceleration: &Option<Acceleration>,
        client_id: Option<String>,
        configuration_id: Option<String>,
    ) -> Result<()> {
        {
            let cache = self.cache.lock().await;
            if cache.contains_key(tag) {
                return Ok(());
            }
        }
        let acceleration = acceleration.clone().unwrap_or(Acceleration::LocalAuto);
        let prediction = self.create_raw_prediction(tag, client_id, configuration_id).await?;
        let config_token = prediction.configuration.ok_or_else(|| {
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
                let path = self.get_resource_path(resource);
                if !path.exists() {
                    self.client.download(&resource.url, &path).await?;
                }
                configuration.add_resource(&resource.kind, &path.to_string_lossy())?;
            }
        }
        let predictor = c::Predictor::new(&configuration)?;
        let mut cache = self.cache.lock().await;
        cache.entry(tag.to_string()).or_insert(predictor);
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
}

fn get_cache_dir() -> PathBuf {
    let home = dirs_or_tmp();
    let dir = home.join(".fxn").join("cache");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

fn dirs_or_tmp() -> PathBuf {
    if let Some(home) = home::home_dir() {
        let test = home.join(".fxntest");
        if std::fs::write(&test, "fxn").is_ok() {
            let _ = std::fs::remove_file(&test);
            return home;
        }
    }
    std::env::temp_dir()
}

fn chrono_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("{secs}")
}
