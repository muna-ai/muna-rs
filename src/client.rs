/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::collections::HashMap;
use std::path::Path;
use std::pin::Pin;

use futures_core::Stream;
use reqwest::Method;
use serde::de::DeserializeOwned;
use serde::Deserialize;

/// Muna error.
#[derive(Debug, thiserror::Error)]
pub enum MunaError {
    /// API error with HTTP status.
    #[error("{message}")]
    Api {
        message: String,
        status: u16,
    },
    /// HTTP transport error.
    #[error(transparent)]
    Http(#[from] reqwest::Error),
    /// Prediction error.
    #[error("{0}")]
    Prediction(String),
    /// JSON serialization error.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// Native library error.
    #[error("{0}")]
    Native(String),
}

impl MunaError {
    pub fn api_status(&self) -> Option<u16> {
        match self {
            Self::Api { status, .. } => Some(*status),
            _ => None,
        }
    }
}

pub type Result<T> = std::result::Result<T, MunaError>;

/// Server-sent event.
#[derive(Debug, Deserialize)]
pub struct SseEvent<T> {
    pub event: String,
    pub data: T,
}

/// HTTP request input.
pub struct RequestInput {
    pub path: String,
    pub method: Method,
    pub headers: Option<HashMap<String, String>>,
    pub body: Option<serde_json::Value>,
}

impl RequestInput {

    pub fn get(path: impl Into<String>) -> Self {
        Self { path: path.into(), method: Method::GET, headers: None, body: None }
    }

    pub fn post(path: impl Into<String>) -> Self {
        Self { path: path.into(), method: Method::POST, headers: None, body: None }
    }

    pub fn delete(path: impl Into<String>) -> Self {
        Self { path: path.into(), method: Method::DELETE, headers: None, body: None }
    }

    pub fn body(mut self, body: serde_json::Value) -> Self {
        self.body = Some(body);
        self
    }

    pub fn header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.get_or_insert_with(HashMap::new).insert(key.into(), value.into());
        self
    }
}

/// Muna API client.
pub struct MunaClient {
    /// Muna API URL.
    pub url: String,
    auth: String,
    http: reqwest::Client,
}

impl MunaClient {
    const DEFAULT_URL: &'static str = "https://api.muna.ai/v1";

    /// Create a Muna API client.
    pub fn new(access_key: Option<&str>, url: Option<&str>) -> Self {
        let url = url
            .unwrap_or(Self::DEFAULT_URL)
            .to_string();
        let auth = access_key
            .map(|key| format!("Bearer {key}"))
            .unwrap_or_default();
        let http = reqwest::Client::builder()
            .user_agent("muna-rs")
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self {
            url,
            auth,
            http,
        }
    }

    /// Make a request to a REST endpoint.
    pub async fn request<T: DeserializeOwned>(&self, input: RequestInput) -> Result<T> {
        let url = format!("{}{}", self.url, input.path);
        let mut builder = self.http.request(input.method, &url)
            .header("Authorization", &self.auth);
        if let Some(headers) = input.headers {
            for (k, v) in headers {
                builder = builder.header(k, v);
            }
        }
        if let Some(body) = input.body {
            builder = builder
                .header("Content-Type", "application/json")
                .body(serde_json::to_string(&body)?);
        }
        let response = builder.send().await?;
        let status = response.status();
        if !status.is_success() {
            let payload: serde_json::Value = response.json().await.unwrap_or_default();
            let message = payload["errors"][0]["message"]
                .as_str()
                .unwrap_or("An unknown error occurred")
                .to_string();
            return Err(MunaError::Api { message, status: status.as_u16() });
        }
        let result = response.json().await?;
        Ok(result)
    }

    /// Download a resource to a local path.
    pub async fn download(&self, url: &str, path: &Path) -> Result<()> {
        use tokio::io::AsyncWriteExt;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| MunaError::Prediction(format!(
                "Failed to create cache directory: {e}"
            )))?;
        }
        let response = self.http.get(url)
            .send()
            .await?;
        let status = response.status();
        if !status.is_success() {
            return Err(MunaError::Api {
                message: format!("Failed to download resource: {status}"),
                status: status.as_u16(),
            });
        }
        let tmp_path = std::env::temp_dir().join(format!("muna-{}", uuid_v4()));
        let mut file = tokio::fs::File::create(&tmp_path).await.map_err(|e| {
            MunaError::Prediction(format!("Failed to create temp file: {e}"))
        })?;
        let mut response = response;
        while let Some(chunk) = response.chunk().await? {
            file.write_all(&chunk).await.map_err(|e| {
                MunaError::Prediction(format!("Failed to write chunk: {e}"))
            })?;
        }
        file.flush().await.map_err(|e| {
            MunaError::Prediction(format!("Failed to flush file: {e}"))
        })?;
        drop(file);
        tokio::fs::rename(&tmp_path, path).await.map_err(|e| {
            MunaError::Prediction(format!(
                "Failed to move resource to {}: {e}", path.display()
            ))
        })?;
        Ok(())
    }

    /// Make a request and consume the response as a server-sent events stream.
    pub async fn stream<T: DeserializeOwned + Send + 'static>(
        &self,
        input: RequestInput,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<SseEvent<T>>> + Send>>> {
        let url = format!("{}{}", self.url, input.path);
        let mut builder = self.http.request(input.method, &url)
            .header("Authorization", &self.auth);
        if let Some(headers) = input.headers {
            for (k, v) in headers {
                builder = builder.header(k, v);
            }
        }
        if let Some(body) = input.body {
            builder = builder
                .header("Content-Type", "application/json")
                .body(serde_json::to_string(&body)?);
        }
        let response = builder.send().await?;
        let status = response.status();
        if !status.is_success() {
            let payload: serde_json::Value = response.json().await.unwrap_or_default();
            let message = payload["errors"][0]["message"]
                .as_str()
                .unwrap_or("An unknown error occurred")
                .to_string();
            return Err(MunaError::Api { message, status: status.as_u16() });
        }
        let stream = async_stream::try_stream! {
            let mut buffer = String::new();
            for await chunk in response.bytes_stream() {
                let chunk = chunk?;
                buffer.push_str(&String::from_utf8_lossy(&chunk));
                while let Some(boundary) = buffer.find("\n\n") {
                    let event_block = buffer[..boundary].to_string();
                    buffer = buffer[boundary + 2..].to_string();
                    let mut event_name = String::new();
                    let mut data = String::new();
                    for line in event_block.lines() {
                        if let Some(v) = line.strip_prefix("event:") {
                            event_name = v.trim().to_string();
                        } else if let Some(v) = line.strip_prefix("data:") {
                            data = v.trim().to_string();
                        }
                    }
                    if !data.is_empty() {
                        let parsed: T = serde_json::from_str(&data)?;
                        yield SseEvent { event: event_name, data: parsed };
                    }
                }
            }
        };
        Ok(Box::pin(stream))
    }
}

fn uuid_v4() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let t = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
    let r: u64 = (t ^ (t >> 32)) as u64;
    format!("{:016x}", r)
}
