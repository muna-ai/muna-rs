/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use crate::beta::openai::{
    ChatCompletionDelta, ChatCompletionReasoningEffort, CompletionTokensDetails,
};

use super::*;

fn chunk(delta: ChatCompletionDelta, usage: Option<ChatCompletionUsage>) -> ChatCompletionChunk {
    ChatCompletionChunk {
        object: "chat.completion.chunk".to_string(),
        id: "chatcmpl-test".to_string(),
        model: "test-model".to_string(),
        choices: vec![ChatCompletionChunkChoice {
            index: 0,
            delta: Some(delta),
            finish_reason: None,
            logprobs: None,
        }],
        created: 0,
        usage,
    }
}

#[test]
fn merge_accumulates_reasoning_and_content() {
    let chunks = vec![
        chunk(
            ChatCompletionDelta {
                role: Some("assistant".to_string()),
                content: None,
                reasoning_content: Some("Let me ".to_string()),
                tool_calls: None,
            },
            None,
        ),
        chunk(
            ChatCompletionDelta {
                role: None,
                content: None,
                reasoning_content: Some("think.".to_string()),
                tool_calls: None,
            },
            None,
        ),
        chunk(
            ChatCompletionDelta {
                role: None,
                content: Some("Paris.".to_string()),
                reasoning_content: None,
                tool_calls: None,
            },
            Some(ChatCompletionUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
                prompt_tokens_details: None,
                completion_tokens_details: Some(CompletionTokensDetails {
                    reasoning_tokens: Some(3),
                }),
            }),
        ),
    ];
    let completion = merge_chunks(chunks).unwrap();
    let message = &completion.choices[0].message;
    assert_eq!(
        message.content.as_ref().map(|c| c.flatten()),
        Some("Paris.".to_string())
    );
    assert_eq!(message.reasoning_content.as_deref(), Some("Let me think."));
    let usage = completion.usage.unwrap();
    let details = usage.completion_tokens_details.unwrap();
    assert_eq!(details.reasoning_tokens, Some(3));
}

fn delegate_info() -> OpenAICompletionsAdapter {
    OpenAICompletionsAdapter::from_signature(
        &serde_json::from_value(json!({
            "inputs": [{"name":"messages", "dtype":"list"}],
            "outputs": [{"name":"chunk", "dtype":"dict", "schema":{"title":"ChatCompletionChunk"}}]
        }))
        .unwrap(),
    )
    .unwrap()
}

/// 2x2 solid-red RGBA PNG.
const RED_PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAFUlEQVR4nGP8z8Dwn4GBgYEJRIAwAB8XAgICR7MUAAAAAElFTkSuQmCC";

#[test]
fn content_deserializes_from_string_and_parts() {
    let message: ChatCompletionMessage = serde_json::from_value(json!({
        "role": "user",
        "content": "What is the capital of France?",
    }))
    .unwrap();
    assert!(matches!(
        message.content,
        Some(ChatCompletionContent::Text(_))
    ));
    // OpenCode-shaped body: content as an array of text parts.
    let message: ChatCompletionMessage = serde_json::from_value(json!({
        "role": "user",
        "content": [{ "type": "text", "text": "What is the capital of France?" }],
    }))
    .unwrap();
    let Some(ChatCompletionContent::Parts(parts)) = &message.content else {
        panic!("expected parts content");
    };
    assert!(
        matches!(&parts[0], ChatCompletionContentPart::Text { text } if text == "What is the capital of France?")
    );
    // Text serializes back as a bare string (untagged).
    let text = ChatCompletionContent::Text("Paris.".to_string());
    assert_eq!(serde_json::to_value(&text).unwrap(), json!("Paris."));
}

#[test]
fn content_flatten_joins_text_parts() {
    let content: ChatCompletionContent = serde_json::from_value(json!([
        { "type": "text", "text": "line one" },
        { "type": "text", "text": "line two" },
        { "type": "refusal", "refusal": "I cannot help with that." },
    ]))
    .unwrap();
    assert!(content.is_text());
    assert_eq!(
        content.flatten(),
        "line one\nline two\nI cannot help with that."
    );
    let mixed: ChatCompletionContent = serde_json::from_value(json!([
        { "type": "text", "text": "describe this" },
        { "type": "image_url", "image_url": { "url": "https://example.com/cat.png" } },
    ]))
    .unwrap();
    assert!(!mixed.is_text());
}

#[test]
fn normalize_messages_keeps_text_parts_verbatim() {
    // The compiled chat template flattens text parts; the client must
    // not, so the plane's hash path and the predictor see the same bytes.
    let messages: Vec<ChatCompletionMessage> = serde_json::from_value(json!([
        { "role": "system", "content": "You are a helpful assistant." },
        { "role": "user", "content": [
            { "type": "text", "text": "line one" },
            { "type": "text", "text": "line two" },
            { "type": "refusal", "refusal": "I cannot help with that." },
        ] },
    ]))
    .unwrap();
    let normalized = normalize_messages(&messages).unwrap();
    assert_eq!(
        normalized[0],
        json!({ "role": "system", "content": "You are a helpful assistant." })
    );
    assert_eq!(
        normalized[1]["content"],
        json!([
            { "type": "text", "text": "line one" },
            { "type": "text", "text": "line two" },
            { "type": "refusal", "refusal": "I cannot help with that." },
        ])
    );
    assert!(media_parts(&messages).is_empty());
}

fn image_params(urls: &[String]) -> ChatCompletionCreateParams {
    let mut parts = vec![json!({ "type": "text", "text": "describe these" })];
    parts.extend(
        urls.iter()
            .map(|url| json!({ "type": "image_url", "image_url": { "url": url } })),
    );
    serde_json::from_value(json!({
        "model": "@a/x",
        "messages": [{ "role": "user", "content": parts }],
    }))
    .unwrap()
}

#[test]
fn data_url_images_decode_into_chat_inputs_in_order() {
    let url = format!("data:image/png;base64,{RED_PNG_B64}");
    let inputs = image_params(&[url.clone(), url]).chat_inputs().unwrap();
    let images = inputs.images.unwrap();
    assert_eq!(images.len(), 2);
    assert_eq!((images[0].width, images[0].height), (2, 2));
    assert_eq!(images[0], images[1]);
    assert_eq!(
        inputs.messages[0]["content"],
        json!([
            { "type": "text", "text": "describe these" },
            { "type": "image" },
            { "type": "image" },
        ])
    );
    // No image parts: no images input at all.
    assert_eq!(image_params(&[]).chat_inputs().unwrap().images, None);
}

#[test]
fn remote_and_malformed_image_urls_are_rejected() {
    for url in [
        "https://example.com/cat.png",
        "data:image/png;base64",
        "data:image/png;base64,!!!",
        "data:image/png;base64,AA==",
    ] {
        let error = image_params(&[url.to_string()]).chat_inputs().unwrap_err();
        assert!(
            matches!(error, MunaError::InvalidInput(_)),
            "{url}: {error}"
        );
    }
    let error = image_params(&["https://example.com/cat.png".into()])
        .chat_inputs()
        .unwrap_err();
    assert!(error.to_string().contains("data:"), "{error}");
}

#[test]
fn decode_audios_rejects_undeclared_audio_and_file_parts() {
    let messages: Vec<ChatCompletionMessage> = serde_json::from_value(json!([
        { "role": "user", "content": [
            { "type": "image_url", "image_url": { "url": "https://example.com/cat.png" } },
        ] },
    ]))
    .unwrap();
    assert!(decode_audios(&messages, &delegate_info())
        .unwrap()
        .is_empty());
    let messages: Vec<ChatCompletionMessage> = serde_json::from_value(json!([
        { "role": "user", "content": [
            { "type": "input_audio", "input_audio": { "data": "AAAA", "format": "wav" } },
        ] },
    ]))
    .unwrap();
    let error = decode_audios(&messages, &delegate_info()).unwrap_err();
    assert!(matches!(&error, MunaError::InvalidInput(m) if m.contains("input_audio")));
    let messages: Vec<ChatCompletionMessage> = serde_json::from_value(json!([
        { "role": "user", "content": [
            { "type": "file", "file": { "file_data": "AAAA", "filename": "doc.pdf" } },
        ] },
    ]))
    .unwrap();
    let error = decode_audios(&messages, &delegate_info()).unwrap_err();
    assert!(matches!(&error, MunaError::InvalidInput(m) if m.contains("File content parts")));
}

#[test]
fn merge_accumulates_tool_call_fragments() {
    // Fragments arrive as: id + name, then two argument fragments.
    let deltas: Vec<ChatCompletionDelta> = vec![
        serde_json::from_value(json!({
            "role": "assistant",
            "tool_calls": [{
                "index": 0,
                "id": "call_test_0",
                "type": "function",
                "function": { "name": "get_weather", "arguments": "" },
            }],
        }))
        .unwrap(),
        serde_json::from_value(json!({
            "tool_calls": [{
                "index": 0,
                "function": { "arguments": "{\"location\": " },
            }],
        }))
        .unwrap(),
        serde_json::from_value(json!({
            "tool_calls": [{
                "index": 0,
                "function": { "arguments": "\"Paris\"}" },
            }],
        }))
        .unwrap(),
    ];
    let chunks: Vec<ChatCompletionChunk> =
        deltas.into_iter().map(|delta| chunk(delta, None)).collect();
    let completion = merge_chunks(chunks).unwrap();
    let calls = completion.choices[0].message.tool_calls.as_ref().unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id, "call_test_0");
    assert_eq!(calls[0].function.name, "get_weather");
    assert_eq!(calls[0].function.arguments, "{\"location\": \"Paris\"}");
}

#[test]
fn chat_inputs_deserialize_from_wire_body_and_drop_empty_tools() {
    // Wire body with fields the predictor never sees (`stream`) and the
    // deprecated `max_tokens` spelling.
    let params: ChatCompletionCreateParams = serde_json::from_value(json!({
        "model": "@a/x",
        "stream": true,
        "max_tokens": 32,
        "messages": [{ "role": "user", "content": [{ "type": "text", "text": "hi" }] }],
        "tools": [{
            "type": "function",
            "function": { "name": "get_weather", "parameters": { "type": "object" } },
        }],
        "tool_choice": "none",
        "reasoning_effort": "high",
        "response_format": { "type": "json_object" },
        "temperature": 0.5,
        "seed": 7,
    }))
    .unwrap();
    assert_eq!(params.max_completion_tokens, Some(32));
    assert!(params.acceleration.is_none());
    let inputs = params.chat_inputs().unwrap();
    assert_eq!(
        inputs.messages[0]["content"],
        json!([{ "type": "text", "text": "hi" }])
    );
    // Knobs ride through under their denotation names; Anthropic-only
    // knobs stay unset.
    assert_eq!(inputs.max_output_tokens, Some(32));
    assert_eq!(inputs.reasoning_effort.as_deref(), Some("high"));
    assert_eq!(
        inputs.response_format.as_ref().unwrap()["type"],
        json!("json_object")
    );
    assert_eq!(inputs.temperature, Some(0.5));
    assert_eq!(inputs.top_p, None);
    assert_eq!(inputs.seed, Some(7));
    assert_eq!(inputs.stop_sequences, None);
    assert_eq!(inputs.top_k, None);
    // `tool_choice: none` does not drop tools: that is the predictor's call.
    assert_eq!(
        inputs.tools,
        Some(vec![json!({
            "type": "function",
            "function": { "name": "get_weather", "parameters": { "type": "object" } },
        })])
    );
    let params: ChatCompletionCreateParams = serde_json::from_value(json!({
        "model": "@a/x",
        "messages": [{ "role": "user", "content": "hi" }],
        "tools": [],
    }))
    .unwrap();
    assert_eq!(params.chat_inputs().unwrap().tools, None);
}

#[test]
fn reasoning_effort_none_round_trips() {
    // OpenAI's way of turning reasoning off; predictors map it onto
    // their template's thinking switch.
    let params: ChatCompletionCreateParams = serde_json::from_value(json!({
        "model": "@a/x",
        "messages": [{ "role": "user", "content": "hi" }],
        "reasoning_effort": "none",
    }))
    .unwrap();
    assert_eq!(
        params.reasoning_effort,
        Some(ChatCompletionReasoningEffort::None)
    );
    let inputs = params.chat_inputs().unwrap();
    assert_eq!(inputs.reasoning_effort.as_deref(), Some("none"));
}
#[test]
fn chat_inputs_reject_messages_without_payload() {
    fn inputs(messages: serde_json::Value) -> Result<ChatInputs> {
        let params: ChatCompletionCreateParams = serde_json::from_value(json!({
            "model": "@a/x",
            "messages": messages,
        }))
        .unwrap();
        params.chat_inputs()
    }
    // Whitespace-only text renders as an empty turn once the chat
    // template trims it (Gemma 4 then describes an image it never got);
    // reject it like OpenAI does.
    for content in [
        json!("   "),
        json!(""),
        json!(null),
        json!([]),
        json!([{ "type": "text", "text": " \n" }]),
    ] {
        let error = inputs(json!([{ "role": "user", "content": content }]))
            .err()
            .unwrap_or_else(|| panic!("{content} was accepted"));
        assert!(matches!(error, MunaError::InvalidInput(_)), "{error}");
        assert!(error.to_string().contains("messages[0]"), "{error}");
    }
    // Empty assistant turns are replayed history (a reasoning-only
    // completion, a stream cut after the role chunk) and must pass;
    // `null` content is handed to the predictor as `""`.
    for content in [json!(""), json!(null), json!("  ")] {
        let inputs = inputs(json!([
            { "role": "user", "content": "hi" },
            { "role": "assistant", "content": content },
            { "role": "user", "content": "still there?" },
        ]))
        .unwrap_or_else(|e| panic!("assistant {content} was rejected: {e}"));
        assert_eq!(inputs.messages[1]["role"], json!("assistant"));
        assert!(
            inputs.messages[1]["content"].is_string(),
            "{:?}",
            inputs.messages[1]
        );
    }
    assert!(matches!(inputs(json!([])), Err(MunaError::InvalidInput(_))));
    // Media parts, assistant tool calls, and (possibly empty) tool
    // results are payload on their own.
    inputs(json!([{ "role": "user", "content": [
        { "type": "image_url", "image_url": {
            "url": format!("data:image/png;base64,{RED_PNG_B64}"),
        } },
    ] }]))
    .unwrap();
    inputs(json!([
        { "role": "user", "content": "weather?" },
        { "role": "assistant", "content": null, "tool_calls": [{
            "id": "t1", "type": "function",
            "function": { "name": "get_weather", "arguments": "{}" },
        }] },
        { "role": "tool", "tool_call_id": "t1", "content": "" },
    ]))
    .unwrap();
    inputs(json!([{ "role": "user", "content": " hi " }])).unwrap();
}

#[test]
fn normalize_messages_passes_tool_turns_through() {
    let messages: Vec<ChatCompletionMessage> = serde_json::from_value(json!([
        { "role": "assistant", "content": null, "tool_calls": [{
            "id": "call_123",
            "type": "function",
            "function": { "name": "get_weather", "arguments": "{\"city\": \"Paris\"}" },
        }] },
        { "role": "tool", "content": "18C and sunny", "tool_call_id": "call_123" },
    ]))
    .unwrap();
    let normalized = normalize_messages(&messages).unwrap();
    assert_eq!(
        normalized[0]["tool_calls"][0]["function"]["name"],
        json!("get_weather")
    );
    assert_eq!(normalized[1]["role"], json!("tool"));
    assert_eq!(normalized[1]["tool_call_id"], json!("call_123"));
}

#[test]
fn merge_without_reasoning_leaves_field_absent() {
    let chunks = vec![chunk(
        ChatCompletionDelta {
            role: Some("assistant".to_string()),
            content: Some("Hello.".to_string()),
            reasoning_content: None,
            tool_calls: None,
        },
        None,
    )];
    let completion = merge_chunks(chunks).unwrap();
    let message = &completion.choices[0].message;
    assert_eq!(message.reasoning_content, None);
    // Absent reasoning must not appear on the wire (skip_serializing_if).
    let json = serde_json::to_value(message).unwrap();
    assert!(json.get("reasoning_content").is_none());
}
