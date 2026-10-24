/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::client::{download, DownloadDurability, MunaError, Result};
use crate::types::{Prediction, Signature};

/// Predictor manifest file in a Hugging Face repository.
const HF_MANIFEST: &str = "config.json";

/// Hugging Face predictor tag.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HfTag {
    /// Hugging Face repository identifier.
    pub repo_id: String,
    /// Repository revision.
    pub revision: Option<String>,
}

/// Predictor manifest stored in a Hugging Face repository.
///
/// This is just a raw prediction but with an added `signature` from the predictor
/// so that beta clients like `muna.beta.openai` and `muna.beta.anthropic` still work.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfManifest {
    /// Raw prediction.
    #[serde(flatten)]
    pub prediction: Prediction,
    /// Predictor signature.
    pub signature: Signature,
}

/// Parse a Hugging Face predictor tag, like `hf:owner/repo[@revision]`.
///
/// Returns `None` if the tag is not a valid Hugging Face tag.
pub fn parse_hf_tag(tag: &str) -> Option<HfTag> {
    const HF_TAG_PREFIX: &str = "hf:";
    let rest = tag.strip_prefix(HF_TAG_PREFIX)?;
    let (repo_id, revision) = match rest.split_once('@') {
        Some((repo_id, revision)) => (repo_id, revision),
        None => (rest, ""),
    };
    let (owner, name) = repo_id.split_once('/')?;
    if owner.is_empty() || name.is_empty() || name.contains('/') {
        return None;
    }
    Some(HfTag {
        repo_id: repo_id.to_string(),
        revision: (!revision.is_empty()).then(|| revision.to_string()),
    })
}

/// Load a predictor manifest from Hugging Face.
///
/// The manifest is `config.json`, one of the files the Hub counts downloads by.
pub async fn load_hf_manifest(hf_tag: &HfTag) -> Result<HfManifest> {
    let revision = hf_tag.revision.as_deref();
    let manifest_path = download_hf_file(&hf_tag.repo_id, HF_MANIFEST, revision).await?;
    let data = tokio::fs::read(&manifest_path)
        .await
        .map_err(|e| MunaError::Prediction(format!("Failed to read Hugging Face manifest: {e}")))?;
    Ok(serde_json::from_slice(&data)?)
}

/// Download a file from Hugging Face into the Hugging Face cache, returning its path.
///
/// The cache is shared with `huggingface_hub` and other Hugging Face clients, so files
/// they have already downloaded are reused. When the Hub cannot be reached, the file
/// is resolved from the cache.
pub async fn download_hf_file(
    repo_id: &str,
    filename: &str,
    revision: Option<&str>,
) -> Result<PathBuf> {
    let cache = HfCache::new(repo_id);
    let revision = revision.unwrap_or("main");
    if is_commit_hash(revision) {
        let snapshot = cache.snapshot_path(revision, filename);
        if snapshot.exists() {
            return Ok(snapshot);
        }
    }
    match download_hf_file_to_cache(&cache, repo_id, filename, revision).await {
        Err(error) if is_transient(&error) => cache
            .resolve_snapshot(revision, filename)
            .ok_or(error),
        result => result,
    }
}

async fn download_hf_file_to_cache(
    cache: &HfCache,
    repo_id: &str,
    filename: &str,
    revision: &str,
) -> Result<PathBuf> {
    let url = hf_file_url(repo_id, filename, revision)?;
    let response = head_with_relative_redirects(&url).await?;
    let status = response.status();
    let headers = response.headers();
    let commit_hash = header_str(headers, "x-repo-commit").map(str::to_string);
    if status == reqwest::StatusCode::NOT_FOUND {
        if let Some(commit_hash) = &commit_hash {
            cache.mark_no_exist(revision, commit_hash, filename);
        }
    }
    if !status.is_success() && !status.is_redirection() {
        let hint = match status {
            reqwest::StatusCode::UNAUTHORIZED => " Set `HF_TOKEN` or run `hf auth login` \
                to access private and gated repositories.",
            reqwest::StatusCode::FORBIDDEN => " Check that your Hugging Face account has \
                accepted the repository's terms, and that your token can read gated repositories.",
            _ => "",
        };
        return Err(MunaError::Api {
            message: format!(
                "Failed to download {filename} from Hugging Face repository {repo_id}: {status}.{hint}"
            ),
            status: status.as_u16(),
        });
    }
    let etag = header_str(headers, "x-linked-etag")
        .or_else(|| header_str(headers, "etag"))
        .map(|etag| etag.trim_start_matches("W/").trim_matches('"').to_string())
        .ok_or_else(|| MunaError::Prediction(format!(
            "Failed to download {filename} from Hugging Face because the response has no ETag"
        )))?;
    let commit_hash = commit_hash.ok_or_else(|| MunaError::Prediction(format!(
        "Failed to download {filename} from Hugging Face because the response has no commit hash"
    )))?;
    let blob = cache.blob_path(&etag);
    if !blob.exists() {
        let _lock = cache.lock(&etag).await?;
        if !blob.exists() {
            download(hf_http(), &url, &blob, None, DownloadDurability::Durable).await?;
        }
    }
    cache.finalize(revision, &commit_hash, filename, &etag)
}

/// Hugging Face cache for one model repository, laid out like `huggingface_hub`:
/// `blobs/<etag>` holds file contents, `snapshots/<commit>/<filename>` points at a
/// blob, and `refs/<revision>` holds the commit that a branch or tag resolved to.
struct HfCache {
    cache_dir: PathBuf,
    repo_folder: String,
}

impl HfCache {

    fn new(repo_id: &str) -> Self {
        Self {
            cache_dir: hf_cache_dir(),
            repo_folder: format!("models--{}", repo_id.replace('/', "--")),
        }
    }

    fn repo_path(&self) -> PathBuf {
        self.cache_dir.join(&self.repo_folder)
    }

    fn blob_path(&self, etag: &str) -> PathBuf {
        self.repo_path().join("blobs").join(etag)
    }

    fn snapshot_path(&self, commit_hash: &str, filename: &str) -> PathBuf {
        self.repo_path().join("snapshots").join(commit_hash).join(filename)
    }

    fn ref_path(&self, revision: &str) -> PathBuf {
        self.repo_path().join("refs").join(revision)
    }

    fn resolve_snapshot(&self, revision: &str, filename: &str) -> Option<PathBuf> {
        let commit_hash = if is_commit_hash(revision) {
            revision.to_string()
        } else {
            std::fs::read_to_string(self.ref_path(revision)).ok()?.trim().to_string()
        };
        Some(self.snapshot_path(&commit_hash, filename)).filter(|path| path.exists())
    }

    fn write_ref(&self, revision: &str, commit_hash: &str) -> Result<()> {
        if is_commit_hash(revision) {
            return Ok(());
        }
        let path = self.ref_path(revision);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(cache_error)?;
        }
        std::fs::write(&path, commit_hash).map_err(cache_error)
    }

    fn mark_no_exist(&self, revision: &str, commit_hash: &str, filename: &str) {
        let path = self.repo_path().join(".no_exist").join(commit_hash).join(filename);
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(&path, b"");
        let _ = self.write_ref(revision, commit_hash);
    }

    /// Acquire the same lock file that `huggingface_hub` takes while downloading a blob.
    /// Waits as long as another process holds it, since that process may be downloading
    /// a very large file.
    async fn lock(&self, etag: &str) -> Result<std::fs::File> {
        let path = self.cache_dir
            .join(".locks")
            .join(&self.repo_folder)
            .join(format!("{etag}.lock"));
        tokio::task::spawn_blocking(move || {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let file = std::fs::File::create(&path)?;
            file.lock()?;
            Ok::<_, std::io::Error>(file)
        })
        .await
        .map_err(|e| MunaError::Prediction(format!("Failed to lock Hugging Face cache: {e}")))?
        .map_err(cache_error)
    }

    fn finalize(&self, revision: &str, commit_hash: &str, filename: &str, etag: &str) -> Result<PathBuf> {
        self.write_ref(revision, commit_hash)?;
        let snapshot = self.snapshot_path(commit_hash, filename);
        let parent = snapshot.parent().ok_or_else(|| {
            MunaError::Prediction("Hugging Face snapshot path has no parent".to_string())
        })?;
        std::fs::create_dir_all(parent).map_err(cache_error)?;
        let _ = std::fs::remove_file(&snapshot);
        let depth = Path::new(filename).components().count() + 1;
        let relative_blob = std::iter::repeat_n(Path::new(".."), depth)
            .collect::<PathBuf>()
            .join("blobs")
            .join(etag);
        link_snapshot(&relative_blob, &self.blob_path(etag), &snapshot).map_err(cache_error)?;
        Ok(snapshot)
    }
}

#[cfg(unix)]
fn link_snapshot(relative_blob: &Path, _blob: &Path, snapshot: &Path) -> std::io::Result<()> {
    match std::os::unix::fs::symlink(relative_blob, snapshot) {
        Err(error) if error.kind() != std::io::ErrorKind::AlreadyExists => Err(error),
        _ => Ok(()),
    }
}

/// Symlinks on Windows require Developer Mode, so fall back to a hard link, which
/// does not duplicate the blob on disk, and to a copy as a last resort.
#[cfg(windows)]
fn link_snapshot(relative_blob: &Path, blob: &Path, snapshot: &Path) -> std::io::Result<()> {
    std::os::windows::fs::symlink_file(relative_blob, snapshot)
        .or_else(|_| std::fs::hard_link(blob, snapshot))
        .or_else(|_| std::fs::copy(blob, snapshot).map(|_| ()))
}

fn cache_error(error: std::io::Error) -> MunaError {
    MunaError::Prediction(format!("Failed to update Hugging Face cache: {error}"))
}

fn is_commit_hash(revision: &str) -> bool {
    revision.len() == 40 && revision.chars().all(|c| c.is_ascii_hexdigit())
}

fn is_transient(error: &MunaError) -> bool {
    match error {
        MunaError::Http(_) => true,
        MunaError::Api { status, .. } => *status == 429 || *status >= 500,
        _ => false,
    }
}

fn header_str<'a>(headers: &'a reqwest::header::HeaderMap, name: &str) -> Option<&'a str> {
    headers.get(name).and_then(|value| value.to_str().ok())
}

/// HEAD a resolve URL, following only relative redirects. The Hub redirects to the
/// canonical repo path when the repo casing differs, and that response carries no
/// file metadata; redirects to file storage already carry it.
async fn head_with_relative_redirects(url: &str) -> Result<reqwest::Response> {
    const MAX_RELATIVE_REDIRECTS: usize = 10;
    let mut url = url::Url::parse(url)
        .map_err(|e| MunaError::InvalidInput(format!("Invalid Hugging Face URL: {e}")))?;
    for _ in 0..=MAX_RELATIVE_REDIRECTS {
        let response = hf_head_http().head(url.clone()).send().await?;
        let location = response
            .status()
            .is_redirection()
            .then(|| header_str(response.headers(), "location"))
            .flatten()
            .filter(|location| url::Url::parse(location).is_err());
        match location {
            Some(location) => {
                url = url.join(location).map_err(|e| {
                    MunaError::Prediction(format!("Invalid Hugging Face redirect: {e}"))
                })?;
            }
            None => return Ok(response),
        }
    }
    Err(MunaError::Prediction("Too many Hugging Face redirects".to_string()))
}

fn hf_file_url(repo_id: &str, filename: &str, revision: &str) -> Result<String> {
    let endpoint = std::env::var("HF_ENDPOINT")
        .unwrap_or_else(|_| "https://huggingface.co".to_string());
    let mut url = url::Url::parse(&endpoint)
        .map_err(|e| MunaError::InvalidInput(format!("Invalid Hugging Face endpoint: {e}")))?;
    url.path_segments_mut()
        .map_err(|_| MunaError::InvalidInput("Invalid Hugging Face endpoint".to_string()))?
        .pop_if_empty()
        .extend(repo_id.split('/'))
        .push("resolve")
        .push(revision)
        .extend(filename.split('/'));
    Ok(url.into())
}

/// HTTP client for Hugging Face downloads. The token is sent to the Hugging Face
/// endpoint only; redirects to file storage drop the `Authorization` header.
fn hf_http() -> &'static reqwest::Client {
    static HTTP: OnceLock<reqwest::Client> = OnceLock::new();
    HTTP.get_or_init(|| build_hf_http(reqwest::redirect::Policy::default()))
}

/// HTTP client for Hugging Face metadata requests, which must not follow redirects.
fn hf_head_http() -> &'static reqwest::Client {
    static HTTP: OnceLock<reqwest::Client> = OnceLock::new();
    HTTP.get_or_init(|| build_hf_http(reqwest::redirect::Policy::none()))
}

fn build_hf_http(redirect: reqwest::redirect::Policy) -> reqwest::Client {
    let mut headers = reqwest::header::HeaderMap::new();
    if let Some(token) = hf_token() {
        if let Ok(value) = reqwest::header::HeaderValue::from_str(&format!("Bearer {token}")) {
            headers.insert(reqwest::header::AUTHORIZATION, value);
        }
    }
    reqwest::Client::builder()
        .user_agent(concat!("muna/", env!("CARGO_PKG_VERSION")))
        .default_headers(headers)
        .redirect(redirect)
        .build()
        .expect("failed to build reqwest client")
}

/// Hugging Face token, from `HF_TOKEN` or the token saved by `hf auth login`.
fn hf_token() -> Option<String> {
    if let Some(token) = std::env::var("HF_TOKEN").ok().filter(|token| !token.is_empty()) {
        return Some(token);
    }
    let token_path = std::env::var("HF_TOKEN_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| hf_home().join("token"));
    let token = std::fs::read_to_string(token_path).ok()?;
    Some(token.trim().to_string()).filter(|token| !token.is_empty())
}

fn hf_home() -> PathBuf {
    if let Ok(path) = std::env::var("HF_HOME") {
        return PathBuf::from(path);
    }
    if let Ok(path) = std::env::var("XDG_CACHE_HOME") {
        return PathBuf::from(path).join("huggingface");
    }
    home::home_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join(".cache")
        .join("huggingface")
}

fn hf_cache_dir() -> PathBuf {
    std::env::var("HF_HUB_CACHE")
        .or_else(|_| std::env::var("HUGGINGFACE_HUB_CACHE"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| hf_home().join("hub"))
}

#[cfg(test)]
mod tests {
    use super::{hf_file_url, parse_hf_tag, HfCache, HfManifest, HfTag};

    #[test]
    fn parses_hf_tags() {
        assert_eq!(
            parse_hf_tag("hf:owner/repo"),
            Some(HfTag { repo_id: "owner/repo".to_string(), revision: None })
        );
        assert_eq!(
            parse_hf_tag("hf:owner/repo@v1"),
            Some(HfTag { repo_id: "owner/repo".to_string(), revision: Some("v1".to_string()) })
        );
        assert_eq!(
            parse_hf_tag("hf:owner/repo@"),
            Some(HfTag { repo_id: "owner/repo".to_string(), revision: None })
        );
        for tag in ["@owner/repo", "hf:owner", "hf:/repo", "hf:owner/", "hf:a/b/c", "hf:"] {
            assert_eq!(parse_hf_tag(tag), None, "{tag}");
        }
    }

    #[test]
    fn builds_resolve_urls() {
        assert_eq!(
            hf_file_url("owner/repo", "config.json", "main").unwrap(),
            "https://huggingface.co/owner/repo/resolve/main/config.json"
        );
        assert_eq!(
            hf_file_url("owner/repo", "dir/a b.bin", "refs/pr/1").unwrap(),
            "https://huggingface.co/owner/repo/resolve/refs%2Fpr%2F1/dir/a%20b.bin"
        );
    }

    #[test]
    fn finalizes_snapshot_as_relative_link_to_blob() {
        let cache_dir = std::env::temp_dir().join(format!("muna-hf-cache-test-{}", std::process::id()));
        let cache = HfCache { cache_dir: cache_dir.clone(), repo_folder: "models--owner--repo".to_string() };
        let blob = cache.blob_path("abc");
        std::fs::create_dir_all(blob.parent().unwrap()).unwrap();
        std::fs::write(&blob, b"weights").unwrap();
        let commit = "0123456789abcdef0123456789abcdef01234567";
        let snapshot = cache.finalize("main", commit, "dir/model.bin", "abc").unwrap();
        assert_eq!(std::fs::read(&snapshot).unwrap(), b"weights");
        assert_eq!(cache.resolve_snapshot("main", "dir/model.bin"), Some(snapshot.clone()));
        #[cfg(unix)]
        assert_eq!(
            std::fs::read_link(&snapshot).unwrap(),
            std::path::Path::new("../../../blobs/abc")
        );
        let _ = std::fs::remove_dir_all(cache_dir);
    }

    #[test]
    fn parses_manifest() {
        let manifest: HfManifest = serde_json::from_value(serde_json::json!({
            "id": "pred_123",
            "tag": "@owner/area",
            "created": "2026-10-19T00:00:00Z",
            "configuration": null,
            "resources": [{ "type": "dso", "url": "abc123", "name": "libarea.so" }],
            "results": null,
            "latency": null,
            "error": null,
            "logs": null,
            "signature": { "inputs": [], "outputs": [] }
        }))
        .unwrap();
        assert_eq!(manifest.prediction.tag, "@owner/area");
        assert_eq!(manifest.prediction.resources.unwrap()[0].url, "abc123");
    }
}
