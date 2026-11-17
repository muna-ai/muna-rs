/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use muna::beta::adapters::*;
use muna::{Dtype, Value};
use serde_json::json;

use super::common::*;

#[tokio::test]
async fn system_one_adapts_remote_response_and_preserves_question_order() {
    let signature = signature(
        json!([
            {"name":"context","dtype":"list","denotation":"typesafe.systemone.state"},
            {"name":"schema","dtype":"dict","denotation":"typesafe.systemone.questions"}
        ]),
        json!([{"name":"result","dtype":"dict","schema":{"title":"SystemOneResponse"}}]),
    );
    let adapter = TypeSafeSystemOneAdapter::from_signature(&signature).unwrap();
    let questions = serde_json::from_str(
        r#"{
        "z":{"type":"noul","instructions":"Last?"},
        "a":{"type":"noul","instructions":"First?"}
    }"#,
    )
    .unwrap();
    let inputs = adapter.inputs(json!("Context"), &questions).unwrap();
    assert!(matches!(&inputs["context"], Value::List(values) if values == &vec![json!("Context")]));
    let response = adapter
        .adapt(
            prediction(vec![remote(
                Dtype::Dict,
                json!({
                    "answers":{"a":{"type":"noul","noul":0.1},"z":{"type":"noul","noul":0.9}},
                    "usage":{"input_tokens":10,"output_tokens":2}
                }),
            )]),
            Some(TypeSafeSystemOneOptions {
                questions: Some(questions),
            }),
        )
        .await
        .unwrap();
    assert_eq!(response.model, "@test/model");
    assert_eq!(
        response
            .answers
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        vec!["z", "a"]
    );
    assert_eq!(response.usage.input_tokens, Some(10));
    assert_eq!(adapter.kind(), "typesafe.system_one");
    let mut failed = prediction(vec![]);
    failed.error = Some("ValueError: bad input".into());
    assert!(matches!(
        adapter.adapt(failed, None).await,
        Err(muna::MunaError::InvalidInput(_))
    ));
}
