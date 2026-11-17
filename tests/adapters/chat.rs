/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use futures_util::StreamExt;
use muna::beta::adapters::*;
use muna::{MunaError, Value};
use serde_json::json;

use super::common::*;

fn chat_signature() -> muna::Signature {
    signature(
        json!([
            {"name":"conversation", "dtype":"list"},
            {"name":"pictures", "dtype":"image_list", "optional":true, "denotation":"openai.chat.completions.images"},
            {"name":"heat", "dtype":"float64", "optional":true, "denotation":"openai.chat.completions.temperature"},
            {"name":"budget", "dtype":"int32", "optional":true, "denotation":"openai.chat.completions.max_output_tokens"}
        ]),
        json!([{"name":"chunk", "dtype":"dict", "schema":{"title":"ChatCompletionChunk"}}]),
    )
}

#[test]
fn chat_retains_remote_images_and_binds_explicit_parameters() {
    let adapter = OpenAICompletionsAdapter::from_signature(&chat_signature()).unwrap();
    let params = serde_json::from_value(json!({"model":"@test/model", "temperature":0.5, "messages":[{
        "role":"user", "content":[{"type":"image_url", "image_url":{"url":"https://cdn.invalid/image.png"}}]
    }]})).unwrap();
    let inputs = adapter.inputs(&params).unwrap();
    assert!(matches!(&inputs["pictures"], Value::Remote(_)));
    assert!(matches!(&inputs["heat"],Value::Double(value) if *value == 0.5));
    assert!(
        matches!(&inputs["conversation"],Value::List(messages) if messages[0]["content"][0]["type"] == "image")
    );
    assert_eq!(adapter.messages.name, "conversation");
}

#[test]
fn chat_rejects_ambiguous_outputs_and_extra_required_inputs() {
    let mut signature = chat_signature();
    signature.outputs.push(signature.outputs[0].clone());
    assert!(OpenAICompletionsAdapter::from_signature(&signature).is_err());
    signature.outputs.pop();
    signature.inputs[1].optional = Some(false);
    // A required images field is supported; an unrelated required field is not.
    signature.inputs[1].denotation = None;
    assert!(OpenAICompletionsAdapter::from_signature(&signature).is_err());
    assert!(AnthropicMessagesAdapter::from_signature(&signature).is_err());
}

#[tokio::test]
async fn chat_usage_is_reduced_into_one_optional_final_chunk() {
    let adapter = OpenAICompletionsAdapter::from_signature(&chat_signature()).unwrap();
    for include_usage in [false, true] {
        let predictions = Box::pin(futures_util::stream::iter(vec![
            Ok(prediction(vec![chunk(
                json!({"content":"Hi"}),
                json!({"prompt_tokens":10,"completion_tokens":1,"total_tokens":11,"prompt_tokens_details":{"cached_tokens":4}}),
            )])),
            Ok(prediction(vec![chunk(
                json!({"content":" there"}),
                json!({"prompt_tokens":0,"completion_tokens":2,"total_tokens":2}),
            )])),
        ]));
        let chunks = adapter
            .adapt(
                predictions,
                Some(OpenAICompletionsOptions { include_usage }),
            )
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<muna::client::Result<Vec<_>>>()
            .unwrap();
        assert_eq!(chunks.len(), if include_usage { 3 } else { 2 });
        assert!(chunks[0].usage.is_none());
        if include_usage {
            let last = chunks.last().unwrap();
            assert!(last.choices.is_empty());
            let usage = last.usage.as_ref().unwrap();
            assert_eq!(usage.prompt_tokens, 10);
            assert_eq!(usage.completion_tokens, 3);
            assert_eq!(usage.total_tokens, 13);
            assert_eq!(
                usage.prompt_tokens_details.as_ref().unwrap().cached_tokens,
                Some(4)
            );
        }
    }
}

#[tokio::test]
async fn streaming_errors_propagate_and_empty_streams_fail() {
    let adapter = OpenAICompletionsAdapter::from_signature(&chat_signature()).unwrap();
    let mut stream = adapter
        .adapt(Box::pin(futures_util::stream::empty()), None)
        .await
        .unwrap();
    assert!(stream.next().await.unwrap().is_err());
    let mut failed = prediction(vec![]);
    failed.error = Some("ValueError: invalid request".into());
    let mut stream = adapter
        .adapt(Box::pin(futures_util::stream::iter(vec![Ok(failed)])), None)
        .await
        .unwrap();
    assert!(matches!(
        stream.next().await.unwrap(),
        Err(MunaError::InvalidInput(_))
    ));
    let adapter = AnthropicMessagesAdapter::from_signature(&chat_signature()).unwrap();
    let mut stream = adapter
        .adapt(Box::pin(futures_util::stream::empty()), None)
        .await
        .unwrap();
    assert!(stream.next().await.unwrap().is_err());
}

#[test]
fn anthropic_images_and_tool_results_retain_order_without_decoding() {
    let adapter = AnthropicMessagesAdapter::from_signature(&chat_signature()).unwrap();
    let params = serde_json::from_value(json!({"model":"@test/model","max_tokens":10,"messages":[{
        "role":"user","content":[
            {"type":"image","source":{"type":"url","url":"https://cdn.invalid/second"}},
            {"type":"tool_result","tool_use_id":"call1","is_error":true,"content":[
                {"type":"image","source":{"type":"base64","media_type":"image/png","data":"first"}}
            ]}
        ]
    }]}))
    .unwrap();
    let inputs = adapter.inputs(&params).unwrap();
    let Value::Remote(images) = &inputs["pictures"] else {
        panic!("Expected remote images")
    };
    let Some(muna::RemoteData::Sequence(images)) = &images.data else {
        panic!("Expected image sequence")
    };
    assert!(images[0]
        .data
        .as_ref()
        .unwrap()
        .as_str()
        .unwrap()
        .ends_with(",first"));
    assert_eq!(
        images[1].data.as_ref().unwrap().as_str(),
        Some("https://cdn.invalid/second")
    );
    let Value::List(messages) = &inputs["conversation"] else {
        panic!("Expected conversation")
    };
    assert_eq!(messages[0]["role"], "tool");
    assert_eq!(messages[0]["content"][0]["text"], "Error:");
    assert_eq!(messages[1]["role"], "user");
}

#[tokio::test]
async fn anthropic_interleaved_tools_buffer_arguments_until_headers_arrive() {
    let adapter = AnthropicMessagesAdapter::from_signature(&chat_signature()).unwrap();
    let deltas = vec![
        json!({"tool_calls":[{"index":0,"function":{"arguments":"{\"x\":"}}]}),
        json!({"tool_calls":[{"index":1,"id":"call2","function":{"name":"second","arguments":"{}"}}]}),
        json!({"tool_calls":[{"index":0,"id":"call1","function":{"name":"first","arguments":"1}"}}]}),
    ];
    let source = Box::pin(futures_util::stream::iter(
        deltas
            .into_iter()
            .map(|delta| Ok(prediction(vec![chunk(delta, json!(null))]))),
    ));
    let events = adapter
        .adapt(source, None)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<muna::client::Result<Vec<_>>>()
        .unwrap();
    let starts = events
        .iter()
        .filter_map(|event| match event {
            RawMessageStreamEvent::ContentBlockStart {
                index,
                content_block: ContentBlock::ToolUse { id, name, .. },
            } => Some((*index, id.as_str(), name.as_str())),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(starts, vec![(0, "call2", "second"), (1, "call1", "first")]);
    let arguments = events
        .iter()
        .filter_map(|event| match event {
            RawMessageStreamEvent::ContentBlockDelta {
                index: 1,
                delta: ContentBlockDelta::InputJsonDelta { partial_json },
            } => Some(partial_json.as_str()),
            _ => None,
        })
        .collect::<String>();
    assert_eq!(arguments, "{\"x\":1}");
}

#[tokio::test]
async fn dropping_adapter_stream_drops_its_source() {
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };
    struct Guard(Arc<AtomicBool>);
    impl Drop for Guard {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }
    let dropped = Arc::new(AtomicBool::new(false));
    let guard = Guard(dropped.clone());
    let source = Box::pin(async_stream::try_stream! {
        let _guard = guard;
        yield prediction(vec![chunk(json!({"content":"Hi"}),json!(null))]);
        std::future::pending::<()>().await;
    });
    let adapter = OpenAICompletionsAdapter::from_signature(&chat_signature()).unwrap();
    let mut stream = adapter.adapt(source, None).await.unwrap();
    assert!(stream.next().await.unwrap().is_ok());
    drop(stream);
    assert!(dropped.load(Ordering::SeqCst));
}
