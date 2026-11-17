/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use muna::beta::adapters::*;
use muna::{Dtype, RemoteData, RemoteValue, Tensor, TensorData, Value};
use serde_json::json;

use super::common::*;

fn image_signature() -> muna::Signature {
    signature(
        json!([
            {"name":"caption", "dtype":"string", "denotation":"openai.images.prompt"},
            {"name":"references", "dtype":"image_list", "denotation":"openai.chat.completions.images"},
            {"name":"w", "dtype":"int32", "optional":true, "denotation":"openai.images.width"}
        ]),
        json!([{"name":"pictures", "dtype":"image_list"}]),
    )
}

#[test]
fn edit_requires_annotations_and_a_list() {
    let original = image_signature();
    assert!(OpenAIImageEditAdapter::from_signature(&original).is_ok());
    for index in [0, 1] {
        let mut invalid = original.clone();
        invalid.inputs[index].denotation = None;
        assert!(OpenAIImageEditAdapter::from_signature(&invalid).is_err());
    }
    let mut invalid = original.clone();
    invalid.inputs[1].dtype = Some(Dtype::Image);
    assert!(OpenAIImageEditAdapter::from_signature(&invalid).is_err());
    let mut ambiguous = original;
    ambiguous.inputs.push(ambiguous.inputs[1].clone());
    assert!(OpenAIImageEditAdapter::from_signature(&ambiguous).is_err());
}

#[test]
fn edits_preserve_encoded_inputs_and_leave_dimensions_to_predictor() {
    let adapter = OpenAIImageEditAdapter::from_signature(&image_signature()).unwrap();
    assert_eq!(adapter.kind(), "openai.images.edit");
    let inputs = adapter
        .inputs(
            vec![ImageInput::Encoded(vec![1, 2, 3])],
            Some("edit".into()),
            None,
            Some(-1),
            None,
        )
        .unwrap();
    assert!(matches!(inputs["w"], Value::Int(-1)));
    assert!(
        matches!(&inputs["references"], Value::Remote(value) if value.dtype == Dtype::ImageList)
    );
    assert!(adapter
        .inputs(vec![], Some("edit".into()), None, None, None)
        .is_err());
    assert!(adapter
        .inputs(vec![ImageInput::Encoded(vec![1])], None, None, None, None)
        .is_err());
}

#[test]
fn generation_requires_prompt_and_rejects_extra_required_inputs() {
    let mut signature = image_signature();
    assert!(OpenAIImageGenerationAdapter::from_signature(&signature).is_err());
    signature.inputs.remove(1);
    let adapter = OpenAIImageGenerationAdapter::from_signature(&signature).unwrap();
    let inputs = adapter.inputs("draw".into(), None, Some(100), None);
    assert!(matches!(&inputs["caption"], Value::String(prompt) if prompt == "draw"));
    assert!(matches!(inputs["w"], Value::Int(100)));
    signature.inputs[0].denotation = None;
    assert!(OpenAIImageGenerationAdapter::from_signature(&signature).is_err());
}

#[tokio::test]
async fn matching_image_encoding_bypasses_native_decoder() {
    let adapter = OpenAIImageEditAdapter::from_signature(&image_signature()).unwrap();
    // Deliberately not an actual PNG: passthrough must not decode the payload.
    let response = adapter
        .adapt(
            prediction(vec![Value::Remote(RemoteValue {
                dtype: Dtype::ImageList,
                data: Some(RemoteData::Sequence(vec![RemoteValue {
                    dtype: Dtype::Image,
                    data: Some(RemoteData::Encoded(
                        "data:image/png;base64,cGFzc3Rocm91Z2g=".into(),
                    )),
                }])),
            })]),
            None,
        )
        .await
        .unwrap();
    assert_eq!(response.created, 1_794_873_600);
    assert_eq!(
        response.data.unwrap()[0].b64_json.as_deref(),
        Some("cGFzc3Rocm91Z2g=")
    );
}

#[tokio::test]
async fn images_encode_native_and_reject_empty_or_wrong_results() {
    let adapter = OpenAIImageEditAdapter::from_signature(&image_signature()).unwrap();
    let image = muna::Image {
        width: 2,
        height: 2,
        channels: 3,
        data: vec![128; 12],
    };
    let response = adapter
        .adapt(
            prediction(vec![Value::ImageList(vec![image.clone()])]),
            None,
        )
        .await
        .unwrap();
    let bytes = BASE64
        .decode(response.data.unwrap()[0].b64_json.as_ref().unwrap())
        .unwrap();
    assert!(bytes.starts_with(b"\x89PNG"));
    let response = adapter
        .adapt(
            prediction(vec![Value::ImageList(vec![image.clone()])]),
            Some(OpenAIImageOptions {
                output_format: "raw".into(),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    assert_eq!(response.data.unwrap()[0].image.as_ref(), Some(&image));
    for value in [Value::ImageList(vec![]), Value::String("wrong".into())] {
        assert!(adapter.adapt(prediction(vec![value]), None).await.is_err());
    }
}

#[tokio::test]
async fn depth_produces_lossless_npy_and_npz() {
    let mut signature = image_signature();
    signature.outputs[0].dtype = Some(Dtype::ArrayList);
    signature.outputs[0].denotation = Some("depth_map".into());
    let adapter = OpenAIDepthAdapter::from_signature(&signature).unwrap();
    for format in [DepthFormat::Npy, DepthFormat::Npz] {
        let maps = Value::ArrayList(vec![Tensor {
            data: TensorData::Float32(vec![1., 2., 3., 4.]),
            shape: vec![2, 2],
        }]);
        let response = adapter
            .adapt(prediction(vec![maps]), Some(OpenAIDepthOptions { format }))
            .await
            .unwrap();
        let bytes = BASE64
            .decode(response.data.unwrap()[0].b64_json.as_ref().unwrap())
            .unwrap();
        if format == DepthFormat::Npy {
            assert!(bytes.starts_with(b"\x93NUMPY"));
            assert_eq!(
                &bytes[bytes.len() - 16..],
                [1f32, 2., 3., 4.]
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>()
            );
        } else {
            let decoded = muna::c::Value::from_bytes(&bytes, "application/vnd.muna.tensor")
                .unwrap()
                .to_object()
                .unwrap();
            assert!(matches!(decoded, Value::Tensor(tensor) if tensor.shape == vec![2,2]));
        }
    }
    signature.inputs[1].denotation = None;
    assert!(OpenAIDepthAdapter::from_signature(&signature).is_err());
}
