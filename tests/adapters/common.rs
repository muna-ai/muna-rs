/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use muna::{Dtype, Prediction, RemoteValue, Signature, Value};
use serde_json::{json, Value as Json};

pub fn signature(inputs: Json, outputs: Json) -> Signature {
    serde_json::from_value(json!({"inputs":inputs, "outputs":outputs})).unwrap()
}

pub fn prediction(values: Vec<Value>) -> Prediction {
    let mut prediction: Prediction = serde_json::from_value(json!({
        "id":"pred_test", "tag":"@test/model", "created":"2026-11-17T00:00:00.123456789Z"
    }))
    .unwrap();
    prediction.results = Some(values);
    prediction
}

pub fn remote(dtype: Dtype, value: Json) -> Value {
    Value::Remote(RemoteValue {
        dtype,
        data: Some(
            format!(
                "data:application/json;base64,{}",
                BASE64.encode(serde_json::to_vec(&value).unwrap())
            )
            .into(),
        ),
    })
}

pub fn chunk(delta: Json, usage: Json) -> Value {
    remote(
        Dtype::Dict,
        json!({
            "object":"chat.completion.chunk", "id":"chat_test", "model":"@test/model", "created":1,
            "choices":[{"index":0, "delta":delta, "finish_reason":null}], "usage":usage
        }),
    )
}
