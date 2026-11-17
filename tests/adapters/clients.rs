/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures_util::StreamExt;
use muna::beta::adapters::{self, Adapter};
use muna::beta::openai::{ChatCompletionCreateParams, ImageEditParams};
use muna::client::{Client, DownloadProgressFn, RequestInput, Result, SseEvent, SseStream};
use muna::{Acceleration, Muna};
use serde_json::{json, Value as Json};

use super::common::*;

struct Transport {
    signature: muna::Signature,
    prediction: Json,
    requests: Mutex<Vec<RequestInput>>,
}

impl Transport {
    fn new(signature: muna::Signature, values: Vec<muna::Value>) -> Arc<Self> {
        Arc::new(Self {
            signature,
            prediction: serde_json::to_value(prediction(values)).unwrap(),
            requests: Mutex::default(),
        })
    }
}

#[async_trait]
impl Client for Transport {
    fn url(&self) -> &str {
        "https://metadata.invalid/v1"
    }
    fn inference_url(&self) -> &str {
        "https://inference.invalid/v1"
    }
    fn cache_path(&self) -> &Path {
        Path::new("/tmp/muna-adapter-client-test")
    }
    async fn request(&self, input: RequestInput) -> Result<Json> {
        let metadata = input.path.starts_with("/predictors/");
        self.requests.lock().unwrap().push(input);
        Ok(if metadata {
            json!({
                "tag":"@test/model", "name":"model", "owner":{"username":"test"},
                "created":"2026-11-17T00:00:00Z", "status":"active", "access":"public", "signature":self.signature
            })
        } else {
            self.prediction.clone()
        })
    }
    async fn stream(&self, input: RequestInput) -> Result<SseStream<Json>> {
        self.requests.lock().unwrap().push(input);
        Ok(Box::pin(futures_util::stream::iter(vec![Ok(SseEvent {
            event: "prediction".into(),
            data: self.prediction.clone(),
        })])))
    }
    async fn fetch(&self, _: &str) -> Result<Vec<u8>> {
        panic!("unexpected fetch")
    }
    async fn download(&self, _: &str, _: &Path, _: Option<DownloadProgressFn>) -> Result<()> {
        panic!("unexpected download")
    }
    async fn upload(&self, _: &Path) -> Result<String> {
        panic!("unexpected upload")
    }
}

#[tokio::test]
async fn chat_client_uses_cached_adapter_and_handles_missing_initial_role() {
    let signature = signature(
        json!([{"name":"conversation","dtype":"list"}]),
        json!([
            {"name":"chunk","dtype":"dict","schema":{"title":"ChatCompletionChunk"}}
        ]),
    );
    let client = Transport::new(
        signature,
        vec![chunk(
            json!({"content":"Hello"}),
            json!({"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}),
        )],
    );
    let muna = Muna::with_client(client.clone());
    let params = ChatCompletionCreateParams {
        model: "@test/model".into(),
        messages: serde_json::from_value(json!([{"role":"user","content":"Hi"}])).unwrap(),
        acceleration: Some(Acceleration::RemoteAuto),
        ..Default::default()
    };
    let completion = muna
        .beta
        .openai
        .chat
        .completions
        .create(params.clone())
        .await
        .unwrap();
    assert_eq!(completion.choices[0].message.role, "assistant");
    assert_eq!(completion.usage.unwrap().total_tokens, 2);
    let chunks = muna
        .beta
        .openai
        .chat
        .completions
        .stream(params)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert_eq!(chunks.len(), 2);
    let requests = client.requests.lock().unwrap();
    assert_eq!(
        requests
            .iter()
            .filter(|request| request.path.starts_with("/predictors/"))
            .count(),
        1
    );
    assert_eq!(
        requests[1].base_url.as_deref(),
        Some("https://inference.invalid/v1")
    );
    assert!(requests[1].body.as_ref().unwrap()["inputs"]
        .get("conversation")
        .is_some());
}

#[tokio::test]
async fn edit_client_uses_strict_adapter_and_preserves_encoded_result() {
    let signature = signature(
        json!([
            {"name":"caption","dtype":"string","denotation":"openai.images.prompt"},
            {"name":"pictures","dtype":"image_list","denotation":"openai.chat.completions.images"}
        ]),
        json!([{"name":"output","dtype":"image_list"}]),
    );
    let client =
        Transport::new(
            signature,
            vec![muna::Value::Remote(serde_json::from_value(json!({
        "dtype":"image_list","data":[{"dtype":"image","data":"data:image/png;base64,YWJj"}]
    })).unwrap())],
        );
    let muna = Muna::with_client(client.clone());
    let response = muna
        .beta
        .openai
        .images
        .edit(ImageEditParams {
            model: "@test/model".into(),
            prompt: Some("edit".into()),
            image: vec![adapters::ImageInput::Encoded(vec![1, 2, 3])],
            acceleration: Some(Acceleration::RemoteAuto),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(response.data.unwrap()[0].b64_json.as_deref(), Some("YWJj"));
    let requests = client.requests.lock().unwrap();
    assert_eq!(
        requests[1].body.as_ref().unwrap()["inputs"]["pictures"]["dtype"],
        "image_list"
    );
}

#[test]
fn response_schemas_are_reexported_without_duplicate_definitions() {
    fn identity(value: muna::beta::openai::ChatCompletionChunk) -> adapters::ChatCompletionChunk {
        value
    }
    fn message(value: muna::beta::anthropic::Message) -> adapters::Message {
        value
    }
    fn transcription(value: muna::beta::openai::Transcription) -> adapters::Transcription {
        value
    }
    fn system_one(value: muna::beta::typesafe::SystemOneResponse) -> adapters::SystemOneResponse {
        value
    }
    let _ = (identity, message, transcription, system_one);
    let signature = signature(
        json!([{"name":"texts","dtype":"list"}]),
        json!([
            {"name":"vectors","dtype":"float32","denotation":"embedding"}
        ]),
    );
    assert_eq!(
        adapters::OpenAIEmbeddingsAdapter::from_signature(&signature)
            .unwrap()
            .kind(),
        "openai.embeddings.create"
    );
}
