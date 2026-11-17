/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures_util::StreamExt;
use muna::client::{Client, DownloadProgressFn, RequestInput, Result, SseEvent, SseStream};
use muna::{Acceleration, Dtype, Muna, MunaError, Prediction, RemoteValue, Value};
use serde_json::{json, Value as Json};

struct Transport {
    requests: Mutex<Vec<RequestInput>>,
    fetched: Mutex<Vec<String>>,
    response: Json,
}

impl Transport {
    fn new(results: Json) -> Arc<Self> {
        Arc::new(Self {
            requests: Mutex::default(),
            fetched: Mutex::default(),
            response: json!({
                "id":"pred_test", "tag":"@test/model", "created":"2026-11-17T00:00:00.123456789Z",
                "configuration":"token", "resources":[], "requirements":{"gpuCount":1},
                "latency":1.25, "logs":"log", "error":null, "results":results,
            }),
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
        Path::new("/tmp/muna-passthrough-test")
    }
    async fn request(&self, input: RequestInput) -> Result<Json> {
        self.requests.lock().unwrap().push(input);
        Ok(self.response.clone())
    }
    async fn stream(&self, input: RequestInput) -> Result<SseStream<Json>> {
        self.requests.lock().unwrap().push(input);
        Ok(Box::pin(futures_util::stream::iter(vec![
            Ok(SseEvent {
                event: "prediction".into(),
                data: self.response.clone(),
            }),
            Ok(SseEvent {
                event: "prediction".into(),
                data: self.response.clone(),
            }),
        ])))
    }
    async fn fetch(&self, url: &str) -> Result<Vec<u8>> {
        self.fetched.lock().unwrap().push(url.into());
        Ok(b"fetched text".to_vec())
    }
    async fn download(&self, _: &str, _: &Path, _: Option<DownloadProgressFn>) -> Result<()> {
        panic!("unexpected download")
    }
    async fn upload(&self, _: &Path) -> Result<String> {
        panic!("unexpected upload")
    }
}

fn inputs() -> HashMap<String, Value> {
    HashMap::from([("prompt".into(), Value::String("hello".into()))])
}

#[tokio::test]
async fn passthrough_preserves_all_wire_values_and_metadata_without_fetching() {
    // Invalid codec data is intentional: passthrough must never inspect it.
    let results = json!([
        {"dtype":"image", "data":"data:image/png;base64,not-base64"},
        {"dtype":"float32", "data":"https://cdn.invalid/tensor"},
        {"dtype":"null", "data":null},
        {"dtype":"image_list", "data":[{"dtype":"image", "data":"https://cdn.invalid/image"}]}
    ]);
    let client = Transport::new(results.clone());
    let muna = Muna::with_client(client.clone());
    let prediction = muna
        .predictions
        .create(
            "@test/model",
            Some(inputs()),
            Some(Acceleration::RemoteAuto),
            None,
            None,
            None,
            None,
            Some(true),
        )
        .await
        .unwrap();
    assert_eq!(prediction.created, "2026-11-17T00:00:00.123456789Z");
    assert_eq!(prediction.latency, Some(1.25));
    assert_eq!(prediction.logs.as_deref(), Some("log"));
    assert_eq!(prediction.configuration.as_deref(), Some("token"));
    assert_eq!(prediction.requirements.as_ref().unwrap().gpu_count, Some(1));
    assert!(prediction
        .results
        .as_ref()
        .unwrap()
        .iter()
        .all(|v| matches!(v, Value::Remote(_))));
    assert_eq!(
        serde_json::to_value(&prediction).unwrap()["results"],
        results
    );
    assert!(client.fetched.lock().unwrap().is_empty());
    let requests = client.requests.lock().unwrap();
    assert_eq!(
        requests[0].base_url.as_deref(),
        Some(client.inference_url())
    );
    assert!(requests[0]
        .body
        .as_ref()
        .unwrap()
        .get("passthrough")
        .is_none());
}

#[tokio::test]
async fn default_and_explicit_false_decode_while_remote_inputs_are_forwarded() {
    let client = Transport::new(json!([{"dtype":"string", "data":"https://cdn.invalid/text"}]));
    let muna = Muna::with_client(client.clone());
    let encoded: RemoteValue = serde_json::from_value(json!({"dtype":"image_list", "data":[{"dtype":"image", "data":"https://cdn.invalid/input"}]})).unwrap();
    for passthrough in [None, Some(false)] {
        let mut inputs = inputs();
        inputs.insert("images".into(), Value::Remote(encoded.clone()));
        let prediction = muna
            .predictions
            .create(
                "@test/model",
                Some(inputs),
                Some(Acceleration::RemoteAuto),
                None,
                None,
                None,
                None,
                passthrough,
            )
            .await
            .unwrap();
        assert!(
            matches!(&prediction.results.unwrap()[0], Value::String(text) if text == "fetched text")
        );
    }
    assert_eq!(
        *client.fetched.lock().unwrap(),
        ["https://cdn.invalid/text", "https://cdn.invalid/text"]
    );
    let requests = client.requests.lock().unwrap();
    for request in requests.iter() {
        assert_eq!(
            request.body.as_ref().unwrap()["inputs"]["images"],
            serde_json::to_value(&encoded).unwrap()
        );
    }
}

#[tokio::test]
async fn streams_apply_passthrough_to_every_event_and_default_to_decoding() {
    let client = Transport::new(json!([{"dtype":"string", "data":"https://cdn.invalid/text"}]));
    let muna = Muna::with_client(client.clone());
    for passthrough in [Some(true), Some(false), None] {
        let mut stream = muna
            .predictions
            .stream(
                "@test/model",
                inputs(),
                Some(Acceleration::RemoteAuto),
                None,
                passthrough,
            )
            .await
            .unwrap();
        let mut count = 0;
        while let Some(prediction) = stream.next().await {
            let result = prediction.unwrap().results.unwrap().remove(0);
            match passthrough {
                Some(true) => assert!(matches!(result, Value::Remote(_))),
                _ => assert!(matches!(result, Value::String(_))),
            }
            count += 1;
        }
        assert_eq!(count, 2);
    }
    assert_eq!(client.fetched.lock().unwrap().len(), 4);
}

#[tokio::test]
async fn passthrough_preserves_errors_absent_results_and_empty_results() {
    for results in [Json::Null, json!([])] {
        let mut client = Transport::new(results.clone());
        Arc::get_mut(&mut client).unwrap().response["error"] = json!("execution failed");
        let muna = Muna::with_client(client);
        let prediction = muna
            .predictions
            .create(
                "@test/model",
                Some(inputs()),
                Some(Acceleration::RemoteAuto),
                None,
                None,
                None,
                None,
                Some(true),
            )
            .await
            .unwrap();
        assert_eq!(prediction.error.as_deref(), Some("execution failed"));
        assert_eq!(
            serde_json::to_value(prediction).unwrap()["results"],
            results
        );
    }
}

#[test]
fn native_ffi_rejects_unresolved_remote_values() {
    let remote = Value::Remote(RemoteValue {
        dtype: Dtype::Null,
        data: None,
    });
    assert!(matches!(
        muna::c::Value::from_object(&remote),
        Err(MunaError::InvalidInput(_))
    ));
    // Native outputs remain native; serialization can still construct their wire form.
    let prediction: Prediction =
        serde_json::from_value(json!({"id":"a", "tag":"b", "created":"c"})).unwrap();
    assert!(prediction.results.is_none());
}
