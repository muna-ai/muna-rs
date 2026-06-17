/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::pin::Pin;

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

/// Muna API client.
pub struct MunaClient {
    /// Muna API URL.
    pub url: String,
    auth: String,
    http: reqwest::Client,
}

impl MunaClient {
    const DEFAULT_URL: &'static str = "https://api.muna.ai/v1";
    const DOWNLOAD_CHUNK_SIZE: u64 = 50 * 1024 * 1024; // 50 MB per range request
    const DOWNLOAD_MAX_FILES: usize = 16; // maximum parallel connections

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
        Self { url, auth, http }
    }

    /// Access the underlying HTTP client.
    pub(crate) fn http(&self) -> &reqwest::Client {
        &self.http
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

    /// Download a resource to a file.
    ///
    /// Range-capable resources are downloaded with parallel chunked range
    /// requests to saturate available bandwidth; resources whose server does
    /// not support range requests fall back to a single-connection stream.
    /// The download is atomic: data is written to a temporary file in the
    /// destination directory and renamed into place only on success.
    pub async fn download(&self, url: &str, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                MunaError::Prediction(format!("Failed to create directory: {e}"))
            })?;
        }
        let tmp_path = download_temp_path(path);
        let result = match self.probe_download(url).await {
            Some(size) => self.download_ranges(url, &tmp_path, size).await,
            None => self.download_stream(url, &tmp_path).await,
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
    async fn download_ranges(&self, url: &str, path: &Path, size: u64) -> Result<()> {
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
            return download_range(&self.http, url, 0, size.saturating_sub(1), path).await;
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
                async move { download_range(&http, &url, start, end, &part).await }
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

    /// Download a resource to a file over a single connection.
    async fn download_stream(&self, url: &str, path: &Path) -> Result<()> {
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
        }
        file.flush()
            .await
            .map_err(|e| MunaError::Prediction(format!("Failed to flush file: {e}")))?;
        Ok(())
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

/// Download a single byte range to a file.
async fn download_range(
    http: &reqwest::Client,
    url: &str,
    start: u64,
    end: u64,
    path: &Path,
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
    }
    file.flush()
        .await
        .map_err(|e| MunaError::Prediction(format!("Failed to flush file: {e}")))?;
    Ok(())
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
            .download(&format!("{base}/resource"), &path)
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
    async fn test_http_accessor_fetches_bytes() {
        // The in-memory path (used by remote.rs) fetches via the shared HTTP
        // client exposed by `http()`.
        let data = test_payload(512 * 1024);
        let base = start_server(data.clone(), true);
        let client = MunaClient::new(None, None);
        let response = client
            .http()
            .get(format!("{base}/resource"))
            .send()
            .await
            .unwrap();
        let bytes = response.bytes().await.unwrap().to_vec();
        assert!(bytes == *data);
    }
}
