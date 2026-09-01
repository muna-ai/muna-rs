/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures_core::Stream;
use reqwest::Method;
use serde::de::DeserializeOwned;
use serde::Deserialize;
use tokio::io::AsyncWriteExt;

/// Muna error.
#[derive(Debug, thiserror::Error)]
pub enum MunaError {
    /// API error with HTTP status.
    #[error("{message}")]
    Api { message: String, status: u16 },
    /// HTTP transport error.
    #[error(transparent)]
    Http(#[from] reqwest::Error),
    /// Prediction error.
    #[error("{0}")]
    Prediction(String),
    /// Invalid input provided by the caller (e.g. an unsupported or
    /// malformed content part). Servers should surface this as a client
    /// error rather than an internal error.
    #[error("{0}")]
    InvalidInput(String),
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

#[derive(Debug, Deserialize)]
struct CreateResourceResponse {
    url: String,
}

#[derive(Debug, Deserialize)]
struct CreateResourceMultipartResponse {
    #[serde(rename = "uploadId")]
    upload_id: String,
    urls: Vec<String>,
}

/// HTTP request input.
pub struct RequestInput {
    /// Request path, relative to the client's API URL (e.g. `/predictions`).
    pub path: String,
    /// HTTP method.
    pub method: Method,
    /// Additional request headers. `Authorization` and `Content-Type` are
    /// set by the client and need not be provided.
    pub headers: Option<HashMap<String, String>>,
    /// JSON request body.
    pub body: Option<serde_json::Value>,
}

impl RequestInput {
    pub fn get(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            method: Method::GET,
            headers: None,
            body: None,
        }
    }

    pub fn post(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            method: Method::POST,
            headers: None,
            body: None,
        }
    }

    pub fn delete(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            method: Method::DELETE,
            headers: None,
            body: None,
        }
    }

    pub fn body(mut self, body: serde_json::Value) -> Self {
        self.body = Some(body);
        self
    }

    pub fn header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers
            .get_or_insert_with(HashMap::new)
            .insert(key.into(), value.into());
        self
    }
}

/// Boxed server-sent event stream returned by `Client::stream`.
pub type SseStream<T> = Pin<Box<dyn Stream<Item = Result<SseEvent<T>>> + Send>>;

/// Download progress callback: invoked with the byte increment of each
/// received chunk and the file's total size when known (from the range
/// probe; `None` when the server does not support range requests).
///
/// Mirrors muna-py's `progress` callable on `client.download` / `upload`.
/// Chunks arrive concurrently on the parallel-range path, so callbacks must
/// be cheap and thread-safe (e.g. an `indicatif` bar or an atomic counter).
pub type DownloadProgressFn = Arc<dyn Fn(u64, Option<u64>) + Send + Sync>;

/// Muna API client interface.
///
/// Mirrors the muna-unity `MunaClient` abstract base: `request` / `stream` /
/// `download` / `upload` primitives with client-owned `url` and `cache_path`.
/// All methods are required; custom clients wrap the default [`MunaClient`]
/// and forward the methods they do not change.
///
/// `request` and `stream` are JSON-in/JSON-out so the trait stays object
/// safe; [`ClientExt`] restores the typed `request_as::<T>` / `stream_as::<T>`
/// ergonomics on top.
#[async_trait]
pub trait Client: Send + Sync {
    /// Muna API URL.
    fn url(&self) -> &str;

    /// Muna cache path.
    fn cache_path(&self) -> &Path;

    /// Make a request to a REST endpoint.
    async fn request(&self, input: RequestInput) -> Result<serde_json::Value>;

    /// Make a request and consume the response as a server-sent events stream.
    async fn stream(&self, input: RequestInput) -> Result<SseStream<serde_json::Value>>;

    /// Fetch a URL's bytes into memory (value data; distinct from the file
    /// `download` below, which supports parallel range requests).
    async fn fetch(&self, url: &str) -> Result<Vec<u8>>;

    /// Download a file, optionally reporting progress through a callback
    /// (see [`DownloadProgressFn`]). Pass `None` to let the client decide
    /// its own presentation (the default client is silent; custom clients
    /// may attach bars or logs).
    async fn download(
        &self,
        url: &str,
        path: &Path,
        progress: Option<DownloadProgressFn>,
    ) -> Result<()>;

    /// Upload a resource, returning its URL.
    async fn upload(&self, path: &Path) -> Result<String>;
}

/// Typed conveniences over [`Client`], blanket-implemented for every client
/// (including `dyn Client`). Kept separate because generic methods cannot
/// live on an object-safe trait.
#[async_trait]
pub trait ClientExt: Client {

    /// Make a request to a REST endpoint, decoding the response as `T`.
    async fn request_as<T: DeserializeOwned>(&self, input: RequestInput) -> Result<T> {
        let value = self.request(input).await?;
        Ok(serde_json::from_value(value)?)
    }

    /// Make a request and consume the response as a typed server-sent
    /// events stream.
    async fn stream_as<T: DeserializeOwned + Send + 'static>(
        &self,
        input: RequestInput,
    ) -> Result<SseStream<T>> {
        use futures_util::StreamExt;
        let stream = Client::stream(self, input).await?;
        let typed = stream.map(|event| {
            let event = event?;
            let data: T = serde_json::from_value(event.data)?;
            Ok(SseEvent { event: event.event, data })
        });
        Ok(Box::pin(typed))
    }
}

impl<C: Client + ?Sized> ClientExt for C {}

/// Muna API client.
pub struct MunaClient {
    /// Muna API URL.
    pub url: String,
    auth: String,
    http: reqwest::Client,
    cache_dir: PathBuf,
}

impl MunaClient {
    const DEFAULT_URL: &'static str = "https://api.muna.ai/v1";
    const RESOURCE_URL_BASE: &'static str = "https://cdn.fxn.ai/resources";
    const DOWNLOAD_CHUNK_SIZE: u64 = 50 * 1024 * 1024; // 50 MB per range request
    const DOWNLOAD_MAX_FILES: usize = 16; // maximum parallel connections
    const MULTIPART_THRESHOLD: u64 = 100 * 1024 * 1024; // 100 MB
    const MULTIPART_CHUNK_SIZE: u64 = 50 * 1024 * 1024; // 50 MB per part
    const UPLOAD_MAX_PARALLEL: usize = 8; // maximum parallel part uploads
    const UPLOAD_MAX_RETRIES: u32 = 5;
    const RETRYABLE_STATUS_CODES: [u16; 7] = [400, 408, 429, 500, 502, 503, 504];

    /// Create a Muna API client.
    pub fn new(access_key: Option<&str>, url: Option<&str>) -> Self {
        let url = url.unwrap_or(Self::DEFAULT_URL).to_string();
        let auth = access_key
            .map(|key| format!("Bearer {key}"))
            .unwrap_or_default();
        let http = reqwest::Client::builder()
            .user_agent("muna-rs")
            .build()
            .expect("failed to build reqwest client");
        let cache_dir = get_cache_dir();
        Self { url, auth, http, cache_dir }
    }

    /// Muna cache path.
    pub fn cache_path(&self) -> &Path {
        &self.cache_dir
    }

    /// Fetch a URL's bytes into memory.
    pub async fn fetch(&self, url: &str) -> Result<Vec<u8>> {
        let response = self.http.get(url).send().await?;
        let status = response.status();
        if !status.is_success() {
            return Err(MunaError::Api {
                message: format!("Failed to fetch resource: {status}"),
                status: status.as_u16(),
            });
        }
        Ok(response.bytes().await?.to_vec())
    }

    /// Make a request to a REST endpoint.
    pub async fn request<T: DeserializeOwned>(&self, input: RequestInput) -> Result<T> {
        let url = format!("{}{}", self.url, input.path);
        let mut builder = self
            .http
            .request(input.method, &url)
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
            return Err(MunaError::Api {
                message,
                status: status.as_u16(),
            });
        }
        let result = response.json().await?;
        Ok(result)
    }

    /// Make a request and consume the response as a server-sent events stream.
    pub async fn stream<T: DeserializeOwned + Send + 'static>(
        &self,
        input: RequestInput,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<SseEvent<T>>> + Send>>> {
        let url = format!("{}{}", self.url, input.path);
        let mut builder = self
            .http
            .request(input.method, &url)
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
            return Err(MunaError::Api {
                message,
                status: status.as_u16(),
            });
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

    /// Download a resource to a file, optionally reporting progress through
    /// a callback (see [`DownloadProgressFn`]).
    ///
    /// Range-capable resources are downloaded with parallel chunked range
    /// requests to saturate available bandwidth; resources whose server does
    /// not support range requests fall back to a single-connection stream.
    /// The download is atomic: data is written to a temporary file in the
    /// destination directory and renamed into place only on success.
    pub async fn download(
        &self,
        url: &str,
        path: &Path,
        progress: Option<DownloadProgressFn>,
    ) -> Result<()> {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| MunaError::Prediction(format!("Failed to create directory: {e}")))?;
        }
        let tmp_path = download_temp_path(path);
        let size = self.probe_download(url).await;
        // Bind the probed total so the chunk loops report plain increments.
        let progress: ChunkProgressFn = progress.map(|callback| {
            Arc::new(move |increment: u64| callback(increment, size))
                as Arc<dyn Fn(u64) + Send + Sync>
        });
        let result = match size {
            Some(size) => self.download_ranges(url, &tmp_path, size, &progress).await,
            None => self.download_stream(url, &tmp_path, &progress).await,
        };
        match result {
            Ok(()) => tokio::fs::rename(&tmp_path, path).await.map_err(|e| {
                MunaError::Prediction(format!(
                    "Failed to move resource to {}: {e}",
                    path.display()
                ))
            }),
            Err(e) => {
                let _ = tokio::fs::remove_file(&tmp_path).await;
                Err(e)
            }
        }
    }

    /// Probe a resource URL for its size and HTTP range support.
    ///
    /// Uses a single-byte range request rather than a `HEAD` so that the
    /// probe works with method-scoped presigned URLs. Returns the total size
    /// only when the server responds with `206 Partial Content`.
    async fn probe_download(&self, url: &str) -> Option<u64> {
        let response = self
            .http
            .get(url)
            .header(reqwest::header::RANGE, "bytes=0-0")
            .send()
            .await
            .ok()?;
        if response.status() != reqwest::StatusCode::PARTIAL_CONTENT {
            return None;
        }
        let content_range = response
            .headers()
            .get(reqwest::header::CONTENT_RANGE)?
            .to_str()
            .ok()?;
        content_range.rsplit('/').next()?.parse::<u64>().ok()
    }

    /// Download a resource using concurrent range requests. A single range
    /// (small file) streams straight to the destination; otherwise each chunk
    /// goes to its own part file which are then assembled in order.
    async fn download_ranges(
        &self,
        url: &str,
        path: &Path,
        size: u64,
        progress: &ChunkProgressFn,
    ) -> Result<()> {
        use futures_util::stream::{StreamExt, TryStreamExt};
        // Build the byte ranges that cover the file.
        let mut ranges: Vec<(usize, u64, u64)> = Vec::new();
        let mut start = 0u64;
        let mut index = 0usize;
        while start < size {
            let end = (start + Self::DOWNLOAD_CHUNK_SIZE).min(size) - 1;
            ranges.push((index, start, end));
            start = end + 1;
            index += 1;
        }
        let part_count = ranges.len();
        // Small file: stream the single range straight to the destination,
        // avoiding the extra part-file assembly pass.
        if part_count <= 1 {
            return download_range(&self.http, url, 0, size.saturating_sub(1), path, progress).await;
        }
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("resource");
        // Destination names are unique, so the file name alone is a safe,
        // collision-free prefix for the part files.
        let part_path = |i: usize| parent.join(format!(".{file_name}.part{i}"));
        // Download each range concurrently, capping the number of open connections.
        let download_result = futures_util::stream::iter(ranges)
            .map(|(i, start, end)| {
                let http = self.http.clone();
                let url = url.to_string();
                let part = part_path(i);
                let progress = progress.clone();
                async move { download_range(&http, &url, start, end, &part, &progress).await }
            })
            .buffer_unordered(Self::DOWNLOAD_MAX_FILES)
            .try_collect::<Vec<()>>()
            .await;
        // Assemble the part files into the destination on success; always clean up.
        let result = match download_result {
            Ok(_) => assemble_parts(path, &part_path, part_count).await,
            Err(e) => Err(e),
        };
        for i in 0..part_count {
            let _ = tokio::fs::remove_file(part_path(i)).await;
        }
        result
    }

    /// Upload a resource and return the resource URL.
    ///
    /// Resources already known to the API (matched by SHA-256) are not
    /// re-uploaded. Files at or above the multipart threshold are uploaded
    /// as multiple parts over parallel connections to saturate available
    /// bandwidth; smaller files go up in a single `PUT`.
    pub async fn upload(&self, path: &Path) -> Result<String> {
        let metadata = tokio::fs::metadata(path)
            .await
            .map_err(|e| MunaError::Native(format!("Failed to stat resource: {e}")))?;
        if !metadata.is_file() {
            return Err(MunaError::Native(format!(
                "Cannot upload resource at path {} because it is not a file",
                path.display()
            )));
        }
        let file_size = metadata.len();
        let resource_hash = sha256_file(path).await?;
        if self.resource_exists(&resource_hash).await? {
            return Ok(format!("{}/{resource_hash}", Self::RESOURCE_URL_BASE));
        }
        if file_size >= Self::MULTIPART_THRESHOLD {
            self.upload_resource_multipart(path, file_size, &resource_hash)
                .await?;
        } else {
            self.upload_resource_single(path, &resource_hash).await?;
        }
        Ok(format!("{}/{resource_hash}", Self::RESOURCE_URL_BASE))
    }

    /// Check whether a resource with the given hash already exists.
    async fn resource_exists(&self, resource_hash: &str) -> Result<bool> {
        let url = format!("{}/resources/{resource_hash}", self.url);
        let response = self
            .http
            .head(&url)
            .header("Authorization", &self.auth)
            .send()
            .await?;
        let status = response.status();
        if status.is_success() {
            return Ok(true);
        }
        if status == reqwest::StatusCode::NOT_FOUND {
            return Ok(false);
        }
        Err(MunaError::Api {
            message: format!("Failed to check resource: {status}"),
            status: status.as_u16(),
        })
    }

    /// Upload a resource using a single `PUT`.
    async fn upload_resource_single(&self, path: &Path, resource_hash: &str) -> Result<()> {
        let resource: CreateResourceResponse = self
            .request(RequestInput::post(format!("/resources/{resource_hash}")))
            .await?;
        let data = tokio::fs::read(path)
            .await
            .map_err(|e| MunaError::Native(format!("Failed to read resource: {e}")))?;
        upload_part(&self.http, &resource.url, data, Self::UPLOAD_MAX_RETRIES).await?;
        Ok(())
    }

    /// Upload a resource using multipart upload. Parts are uploaded over
    /// parallel connections; part order is preserved for the completion call.
    async fn upload_resource_multipart(
        &self,
        path: &Path,
        file_size: u64,
        resource_hash: &str,
    ) -> Result<()> {
        let num_parts = file_size.div_ceil(Self::MULTIPART_CHUNK_SIZE);
        let resource: CreateResourceMultipartResponse = self
            .request(
                RequestInput::post(format!("/resources/{resource_hash}/multipart"))
                    .body(serde_json::json!({ "parts": num_parts })),
            )
            .await?;
        match self.upload_parts(path, &resource.urls).await {
            Ok(etags) => {
                let parts: Vec<serde_json::Value> = etags
                    .iter()
                    .enumerate()
                    .map(|(i, etag)| serde_json::json!({ "partNumber": i + 1, "etag": etag }))
                    .collect();
                self.request_no_content(
                    RequestInput::post(format!(
                        "/resources/{resource_hash}/multipart/{}",
                        resource.upload_id
                    ))
                    .body(serde_json::json!({ "parts": parts })),
                )
                .await
            }
            Err(e) => {
                let _ = self
                    .request_no_content(RequestInput::delete(format!(
                        "/resources/{resource_hash}/multipart/{}",
                        resource.upload_id
                    )))
                    .await;
                Err(e)
            }
        }
    }

    /// Upload parts over parallel connections and return ETags in part order.
    /// Memory is bounded by one in-flight chunk per connection because each
    /// part is read from disk inside its own future.
    async fn upload_parts(&self, path: &Path, urls: &[String]) -> Result<Vec<String>> {
        use futures_util::stream::{StreamExt, TryStreamExt};
        // Owned URLs sidestep rustc's higher-ranked closure inference bug
        // (rust-lang/rust#89976), triggered once `upload` is boxed via
        // async_trait.
        futures_util::stream::iter(urls.to_vec().into_iter().enumerate())
            .map(|(index, url)| {
                let http = self.http.clone();
                let path = path.to_path_buf();
                async move {
                    let chunk =
                        read_part(&path, index as u64 * Self::MULTIPART_CHUNK_SIZE).await?;
                    upload_part(&http, &url, chunk, Self::UPLOAD_MAX_RETRIES).await
                }
            })
            .buffered(Self::UPLOAD_MAX_PARALLEL)
            .try_collect::<Vec<String>>()
            .await
    }

    /// Make a request to a REST endpoint, discarding any response body.
    async fn request_no_content(&self, input: RequestInput) -> Result<()> {
        let url = format!("{}{}", self.url, input.path);
        let mut builder = self
            .http
            .request(input.method, &url)
            .header("Authorization", &self.auth);
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
            return Err(MunaError::Api {
                message,
                status: status.as_u16(),
            });
        }
        Ok(())
    }

    /// Download a resource to a file over a single connection.
    async fn download_stream(
        &self,
        url: &str,
        path: &Path,
        progress: &ChunkProgressFn,
    ) -> Result<()> {
        let mut response = self.http.get(url).send().await?;
        let status = response.status();
        if !status.is_success() {
            return Err(MunaError::Api {
                message: format!("Failed to download resource: {status}"),
                status: status.as_u16(),
            });
        }
        let mut file = tokio::fs::File::create(path)
            .await
            .map_err(|e| MunaError::Prediction(format!("Failed to create file: {e}")))?;
        while let Some(chunk) = response.chunk().await? {
            file.write_all(&chunk)
                .await
                .map_err(|e| MunaError::Prediction(format!("Failed to write chunk: {e}")))?;
            if let Some(report) = progress {
                report(chunk.len() as u64);
            }
        }
        file.flush()
            .await
            .map_err(|e| MunaError::Prediction(format!("Failed to flush file: {e}")))?;
        Ok(())
    }
}

#[async_trait]
impl Client for MunaClient {

    fn url(&self) -> &str {
        &self.url
    }

    fn cache_path(&self) -> &Path {
        MunaClient::cache_path(self)
    }

    async fn request(&self, input: RequestInput) -> Result<serde_json::Value> {
        MunaClient::request::<serde_json::Value>(self, input).await
    }

    async fn stream(&self, input: RequestInput) -> Result<SseStream<serde_json::Value>> {
        MunaClient::stream::<serde_json::Value>(self, input).await
    }

    async fn fetch(&self, url: &str) -> Result<Vec<u8>> {
        MunaClient::fetch(self, url).await
    }

    async fn download(
        &self,
        url: &str,
        path: &Path,
        progress: Option<DownloadProgressFn>,
    ) -> Result<()> {
        MunaClient::download(self, url, path, progress).await
    }

    async fn upload(&self, path: &Path) -> Result<String> {
        MunaClient::upload(self, path).await
    }
}

/// Build a temporary download path in the destination's directory so the
/// final rename stays on the same filesystem (atomic, no cross-device move).
fn download_temp_path(path: &Path) -> PathBuf {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("resource");
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    parent.join(format!(".{file_name}.{nonce}.part"))
}

/// Internal chunk-increment callback: the probed total is already bound, so
/// the download loops report plain byte increments.
type ChunkProgressFn = Option<Arc<dyn Fn(u64) + Send + Sync>>;

/// Download a single byte range to a file.
async fn download_range(
    http: &reqwest::Client,
    url: &str,
    start: u64,
    end: u64,
    path: &Path,
    progress: &ChunkProgressFn,
) -> Result<()> {
    let mut response = http
        .get(url)
        .header(reqwest::header::RANGE, format!("bytes={start}-{end}"))
        .send()
        .await?;
    let status = response.status();
    if !status.is_success() {
        return Err(MunaError::Api {
            message: format!("Failed to download resource chunk: {status}"),
            status: status.as_u16(),
        });
    }
    let mut file = tokio::fs::File::create(path)
        .await
        .map_err(|e| MunaError::Prediction(format!("Failed to create file: {e}")))?;
    while let Some(chunk) = response.chunk().await? {
        file.write_all(&chunk)
            .await
            .map_err(|e| MunaError::Prediction(format!("Failed to write chunk: {e}")))?;
        if let Some(report) = progress {
            report(chunk.len() as u64);
        }
    }
    file.flush()
        .await
        .map_err(|e| MunaError::Prediction(format!("Failed to flush file: {e}")))?;
    Ok(())
}

/// Directory for downloaded predictor resources and cached predictions.
fn get_cache_dir() -> PathBuf {
    let dir = get_muna_home().join("cache");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

/// Muna home directory.
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

/// Compute the SHA-256 hex digest of a file without loading it into memory.
async fn sha256_file(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    use tokio::io::AsyncReadExt;
    let mut file = tokio::fs::File::open(path)
        .await
        .map_err(|e| MunaError::Native(format!("Failed to open resource: {e}")))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 4 * 1024 * 1024];
    loop {
        let n = file
            .read(&mut buffer)
            .await
            .map_err(|e| MunaError::Native(format!("Failed to read resource: {e}")))?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Read one multipart chunk from a file at the given byte offset.
async fn read_part(path: &Path, offset: u64) -> Result<Vec<u8>> {
    use tokio::io::{AsyncReadExt, AsyncSeekExt};
    let mut file = tokio::fs::File::open(path)
        .await
        .map_err(|e| MunaError::Native(format!("Failed to open resource: {e}")))?;
    file.seek(std::io::SeekFrom::Start(offset))
        .await
        .map_err(|e| MunaError::Native(format!("Failed to seek resource: {e}")))?;
    let mut chunk = Vec::with_capacity(MunaClient::MULTIPART_CHUNK_SIZE as usize);
    file.take(MunaClient::MULTIPART_CHUNK_SIZE)
        .read_to_end(&mut chunk)
        .await
        .map_err(|e| MunaError::Native(format!("Failed to read resource: {e}")))?;
    Ok(chunk)
}

/// `PUT` a single part with exponential-backoff retries and return its ETag.
async fn upload_part(
    http: &reqwest::Client,
    url: &str,
    chunk: Vec<u8>,
    max_retries: u32,
) -> Result<String> {
    let mut attempt = 0u32;
    loop {
        let result = http.put(url).body(chunk.clone()).send().await;
        let error = match result {
            Ok(response) => {
                let status = response.status();
                if status.is_success() {
                    let etag = response
                        .headers()
                        .get(reqwest::header::ETAG)
                        .and_then(|v| v.to_str().ok())
                        .unwrap_or_default()
                        .to_string();
                    return Ok(etag);
                }
                if !MunaClient::RETRYABLE_STATUS_CODES.contains(&status.as_u16()) {
                    return Err(MunaError::Api {
                        message: format!("Failed to upload resource part: {status}"),
                        status: status.as_u16(),
                    });
                }
                MunaError::Api {
                    message: format!("Failed to upload resource part: {status}"),
                    status: status.as_u16(),
                }
            }
            Err(e) => MunaError::Http(e),
        };
        if attempt >= max_retries - 1 {
            return Err(error);
        }
        tokio::time::sleep(std::time::Duration::from_secs(1 << attempt)).await;
        attempt += 1;
    }
}

/// Assemble downloaded part files into the destination in order.
async fn assemble_parts(
    path: &Path,
    part_path: &impl Fn(usize) -> PathBuf,
    part_count: usize,
) -> Result<()> {
    let mut file = tokio::fs::File::create(path)
        .await
        .map_err(|e| MunaError::Prediction(format!("Failed to create file: {e}")))?;
    for i in 0..part_count {
        let bytes = tokio::fs::read(part_path(i))
            .await
            .map_err(|e| MunaError::Prediction(format!("Failed to read part file: {e}")))?;
        file.write_all(&bytes)
            .await
            .map_err(|e| MunaError::Prediction(format!("Failed to write chunk: {e}")))?;
    }
    file.flush()
        .await
        .map_err(|e| MunaError::Prediction(format!("Failed to flush file: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::Arc;
    use std::thread;

    /// Start a minimal HTTP server that serves `data`, optionally honoring
    /// HTTP range requests, and return its base URL. When `support_ranges`
    /// is false the server ignores `Range` headers and always responds
    /// `200 OK`, which exercises the single-connection fallback path.
    fn start_server(data: Arc<Vec<u8>>, support_ranges: bool) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let data = data.clone();
                thread::spawn(move || {
                    let mut buf = Vec::new();
                    let mut tmp = [0u8; 1024];
                    loop {
                        match stream.read(&mut tmp) {
                            Ok(0) => break,
                            Ok(n) => {
                                buf.extend_from_slice(&tmp[..n]);
                                if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                                    break;
                                }
                            }
                            Err(_) => return,
                        }
                    }
                    let request = String::from_utf8_lossy(&buf);
                    let range = request.lines().find_map(|line| {
                        line.strip_prefix("Range:")
                            .or_else(|| line.strip_prefix("range:"))
                            .map(|value| value.trim().to_string())
                    });
                    let total = data.len();
                    let (status, body, content_range) = match (support_ranges, range) {
                        (true, Some(range)) => {
                            let spec = range.trim_start_matches("bytes=");
                            let mut parts = spec.split('-');
                            let start: usize = parts.next().unwrap_or("0").parse().unwrap_or(0);
                            let end: usize = parts
                                .next()
                                .and_then(|end| end.parse().ok())
                                .unwrap_or(total - 1)
                                .min(total - 1);
                            (
                                "206 Partial Content",
                                data[start..=end].to_vec(),
                                Some(format!("bytes {start}-{end}/{total}")),
                            )
                        }
                        _ => ("200 OK", data.as_ref().clone(), None),
                    };
                    let mut header = format!(
                        "HTTP/1.1 {status}\r\nContent-Length: {}\r\nAccept-Ranges: bytes\r\nConnection: close\r\n",
                        body.len()
                    );
                    if let Some(content_range) = content_range {
                        header.push_str(&format!("Content-Range: {content_range}\r\n"));
                    }
                    header.push_str("\r\n");
                    let _ = stream.write_all(header.as_bytes());
                    let _ = stream.write_all(&body);
                    let _ = stream.flush();
                });
            }
        });
        format!("http://{addr}")
    }

    fn test_payload(size: usize) -> Arc<Vec<u8>> {
        Arc::new((0..size).map(|i| (i % 251) as u8).collect())
    }

    async fn download_to_temp(base: &str, data: &Arc<Vec<u8>>) -> Vec<u8> {
        let client = MunaClient::new(None, None);
        let dir = std::env::temp_dir().join(format!(
            "muna-dl-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("resource.bin");
        client
            .download(&format!("{base}/resource"), &path, None)
            .await
            .unwrap();
        let downloaded = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(downloaded.len(), data.len());
        downloaded
    }

    #[tokio::test]
    async fn test_download_to_file_parallel() {
        // 64 MiB exceeds the 50 MiB chunk size, exercising the parallel
        // multi-range path.
        let data = test_payload(64 * 1024 * 1024);
        let base = start_server(data.clone(), true);
        assert!(download_to_temp(&base, &data).await == *data);
    }

    #[tokio::test]
    async fn test_download_to_file_single_part() {
        // A small range-capable file takes the single-part fast path.
        let data = test_payload(1024 * 1024);
        let base = start_server(data.clone(), true);
        assert!(download_to_temp(&base, &data).await == *data);
    }

    #[tokio::test]
    async fn test_download_to_file_fallback() {
        // A server that ignores Range headers downloads via the
        // single-connection fallback.
        let data = test_payload(2 * 1024 * 1024);
        let base = start_server(data.clone(), false);
        assert!(download_to_temp(&base, &data).await == *data);
    }

    #[tokio::test]
    async fn test_fetch_bytes() {
        // The in-memory path (used by remote value parsing) fetches via
        // `fetch`.
        let data = test_payload(512 * 1024);
        let base = start_server(data.clone(), true);
        let client = MunaClient::new(None, None);
        let bytes = client.fetch(&format!("{base}/resource")).await.unwrap();
        assert!(bytes == *data);
    }
}
