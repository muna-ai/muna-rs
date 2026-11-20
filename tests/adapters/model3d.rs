/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use muna::beta::adapters::*;
use muna::{Dtype, RemoteValue, Value};
use serde_json::json;

use super::common::*;

const PLY: &[u8] = b"ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nend_header\n\x00\x00\x00\x40";

fn model_signature() -> muna::Signature {
    signature(
        json!([
            {
                "name": "references",
                "dtype": "image_list",
                "denotation": "openai.chat.completions.images"
            },
            {
                "name": "caption",
                "dtype": "string",
                "denotation": "openai.images.prompt",
                "optional": true
            },
            {
                "name": "seed",
                "dtype": "int32",
                "optional": true
            }
        ]),
        json!([
            {
                "name": "asset",
                "dtype": "binary",
                "denotation": "3d",
                "description": "PLY Gaussian splats."
            }
        ]),
    )
}

#[test]
fn requires_annotations_binary_output_and_unambiguous_signature() {
    let original = model_signature();
    assert!(OpenAIModel3DAdapter::from_signature(&original).is_some());
    for index in [0, 1] {
        let mut invalid = original.clone();
        invalid.inputs[index].denotation = None;
        assert!(OpenAIModel3DAdapter::from_signature(&invalid).is_none());
    }
    let mut invalid = original.clone();
    invalid.outputs[0].dtype = Some(Dtype::String);
    assert!(OpenAIModel3DAdapter::from_signature(&invalid).is_none());
    let mut invalid = original.clone();
    invalid.outputs[0].denotation = None;
    assert!(OpenAIModel3DAdapter::from_signature(&invalid).is_none());
    let mut invalid = original.clone();
    invalid.outputs.push(invalid.outputs[0].clone());
    assert!(OpenAIModel3DAdapter::from_signature(&invalid).is_none());
    let mut undocumented = original;
    undocumented.outputs[0].description =
        Some("An asset with a human-readable description.".into());
    assert!(OpenAIModel3DAdapter::from_signature(&undocumented).is_some());
}

#[tokio::test]
async fn preserves_binary_and_inline_ply_and_rejects_other_formats() {
    let adapter = OpenAIModel3DAdapter::from_signature(&model_signature()).unwrap();
    assert_eq!(adapter.kind(), "openai.images.edit.model3d");
    let payload = BASE64.encode(PLY);
    for value in [
        Value::Binary(PLY.to_vec()),
        Value::Remote(RemoteValue {
            dtype: Dtype::Binary,
            data: Some(format!("data:application/octet-stream;base64,{payload}").into()),
        }),
    ] {
        let response = adapter.adapt(prediction(vec![value]), None).await.unwrap();
        assert_eq!(
            response.data.unwrap()[0].b64_json.as_deref(),
            Some(payload.as_str())
        );
        assert_eq!(response.created, 1_794_873_600);
    }
    for value in [
        Value::Binary(vec![]),
        Value::Binary(b"glTF".to_vec()),
        Value::String("wrong".into()),
    ] {
        assert!(adapter.adapt(prediction(vec![value]), None).await.is_err());
    }
    assert!(adapter
        .adapt(
            prediction(vec![Value::Binary(PLY.to_vec())]),
            Some(OpenAIModel3DOptions {
                format: "glb".into()
            })
        )
        .await
        .is_err());
}
