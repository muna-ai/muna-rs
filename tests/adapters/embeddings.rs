/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use muna::beta::adapters::*;
use muna::{Dtype, Tensor, TensorData, Value};
use serde_json::json;

use super::common::*;

fn embedding_signature() -> muna::Signature {
    signature(
        json!([
            {"name":"texts", "dtype":"list"},
            {"name":"dims", "dtype":"int32", "optional":true, "denotation":"openai.embeddings.dims"}
        ]),
        json!([
            {"name":"matrix", "dtype":"float32", "denotation":"embedding"},
            {"name":"usage", "dtype":"dict", "schema":{"title":"Usage"}}
        ]),
    )
}

#[tokio::test]
async fn embeddings_convert_native_and_remote_matrices_and_usage() {
    let adapter = OpenAIEmbeddingsAdapter::from_signature(&embedding_signature()).unwrap();
    let matrix = Value::Tensor(Tensor {
        shape: vec![2, 2],
        data: TensorData::Float32(vec![1., 2., 3., 4.]),
    });
    let bytes = muna::c::Value::from_object(&matrix)
        .unwrap()
        .serialize(Some("application/vnd.muna.tensor"))
        .unwrap();
    let encoded = muna::RemoteValue {
        dtype: Dtype::Float32,
        data: Some(
            format!(
                "data:application/octet-stream;base64,{}",
                BASE64.encode(bytes)
            )
            .into(),
        ),
    };
    let inputs = adapter.inputs(vec!["hi".into()], Some(2));
    assert!(matches!(inputs["dims"], Value::Int(2)));
    for value in [matrix, Value::Remote(encoded)] {
        for encoding_format in [EncodingFormat::Float, EncodingFormat::Base64] {
            let response = adapter
                .adapt(
                    prediction(vec![
                        value.clone(),
                        remote(Dtype::Dict, json!({"prompt_tokens":10,"total_tokens":10})),
                    ]),
                    Some(OpenAIEmbeddingsOptions { encoding_format }),
                )
                .await
                .unwrap();
            assert_eq!(response.model, "@test/model");
            assert_eq!(response.usage.total_tokens, 10);
            match &response.data[0].embedding {
                EmbeddingData::Float(values) => assert_eq!(values, &vec![1., 2.]),
                EmbeddingData::Base64(value) => assert_eq!(
                    BASE64.decode(value).unwrap(),
                    [1f32, 2.]
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect::<Vec<_>>()
                ),
            }
        }
    }
}

#[tokio::test]
async fn invalid_embedding_shapes_and_usage_return_errors() {
    let adapter = OpenAIEmbeddingsAdapter::from_signature(&embedding_signature()).unwrap();
    for shape in [vec![4], vec![3, 2], vec![-1, 4]] {
        let matrix = Value::Tensor(Tensor {
            shape,
            data: TensorData::Float32(vec![1.; 4]),
        });
        assert!(adapter
            .adapt(
                prediction(vec![matrix, remote(Dtype::Dict, json!({"wrong":1}))]),
                None
            )
            .await
            .is_err());
    }
    let matrix = Value::Tensor(Tensor {
        shape: vec![2, 2],
        data: TensorData::Float32(vec![1.; 4]),
    });
    assert!(adapter
        .adapt(
            prediction(vec![matrix, remote(Dtype::Dict, json!({"wrong":1}))]),
            None
        )
        .await
        .is_err());
}

#[test]
fn embedding_signatures_reject_duplicate_optional_bindings() {
    let mut signature = embedding_signature();
    signature.inputs.push(signature.inputs[1].clone());
    assert!(OpenAIEmbeddingsAdapter::from_signature(&signature).is_err());
}
