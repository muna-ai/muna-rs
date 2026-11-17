/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use futures_util::stream;

use crate::beta::openai::{
    ChatCompletionChunkChoice, ChatCompletionDelta, ChatCompletionUsage, PromptTokensDetails,
};

use super::*;

fn chunk(
    delta: ChatCompletionDelta,
    finish_reason: Option<&str>,
    usage: Option<ChatCompletionUsage>,
) -> ChatCompletionChunk {
    ChatCompletionChunk {
        object: "chat.completion.chunk".to_string(),
        id: "msg-test".to_string(),
        model: "test-model".to_string(),
        choices: vec![ChatCompletionChunkChoice {
            index: 0,
            delta: Some(delta),
            finish_reason: finish_reason.map(str::to_string),
            logprobs: None,
        }],
        created: 0,
        usage,
    }
}

async fn collect_events(chunks: Vec<ChatCompletionChunk>) -> Vec<RawMessageStreamEvent> {
    let chunk_stream = Box::pin(stream::iter(chunks.into_iter().map(Ok)));
    let mut events = events_from_chunks(chunk_stream);
    let mut collected = Vec::new();
    while let Some(event) = events.next().await {
        collected.push(event.unwrap());
    }
    collected
}

fn accumulate(events: &[RawMessageStreamEvent]) -> Message {
    let mut message = None;
    for event in events {
        apply_event(&mut message, event);
    }
    message.unwrap()
}

#[tokio::test]
async fn chunks_become_ordered_events() {
    let events = collect_events(vec![
        chunk(
            ChatCompletionDelta {
                role: Some("assistant".to_string()),
                content: None,
                reasoning_content: Some("Let me think.".to_string()),
                tool_calls: None,
            },
            None,
            None,
        ),
        chunk(
            ChatCompletionDelta {
                role: None,
                content: Some("Paris.".to_string()),
                reasoning_content: None,
                tool_calls: None,
            },
            Some("stop"),
            Some(ChatCompletionUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
                prompt_tokens_details: Some(PromptTokensDetails {
                    audio_tokens: None,
                    cache_write_tokens: Some(2),
                    cached_tokens: Some(3),
                }),
                completion_tokens_details: None,
            }),
        ),
    ])
    .await;
    let types: Vec<&str> = events.iter().map(|e| e.event_type()).collect();
    assert_eq!(
        types,
        vec![
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
    );
    let message = accumulate(&events);
    assert!(matches!(
        &message.content[0],
        ContentBlock::Thinking { thinking, .. } if thinking == "Let me think."
    ));
    assert!(matches!(
        &message.content[1],
        ContentBlock::Text { text } if text == "Paris."
    ));
    assert_eq!(message.stop_reason, Some(StopReason::EndTurn));
    // Anthropic convention: input tokens exclude cached tokens.
    assert_eq!(message.usage.input_tokens, Some(5));
    assert_eq!(message.usage.output_tokens, 5);
    assert_eq!(message.usage.cache_read_input_tokens, Some(3));
    assert_eq!(message.usage.cache_creation_input_tokens, Some(2));
}

#[tokio::test]
async fn length_finish_reason_maps_to_max_tokens() {
    let events = collect_events(vec![chunk(
        ChatCompletionDelta {
            role: Some("assistant".to_string()),
            content: Some("Hello".to_string()),
            reasoning_content: None,
            tool_calls: None,
        },
        Some("length"),
        None,
    )])
    .await;
    let message = accumulate(&events);
    assert_eq!(message.stop_reason, Some(StopReason::MaxTokens));
}

#[tokio::test]
async fn tool_call_deltas_become_tool_use_blocks() {
    // Fragments: id + name first, then two argument fragments, ending
    // with a `tool_calls` finish reason.
    let deltas: Vec<ChatCompletionDelta> = vec![
        serde_json::from_value(serde_json::json!({
            "role": "assistant",
            "tool_calls": [{
                "index": 0,
                "id": "call_test_0",
                "type": "function",
                "function": { "name": "get_weather", "arguments": "" },
            }],
        }))
        .unwrap(),
        serde_json::from_value(serde_json::json!({
            "tool_calls": [{
                "index": 0,
                "function": { "arguments": "{\"location\": " },
            }],
        }))
        .unwrap(),
        serde_json::from_value(serde_json::json!({
            "tool_calls": [{
                "index": 0,
                "function": { "arguments": "\"Paris\"}" },
            }],
        }))
        .unwrap(),
    ];
    let mut chunks: Vec<ChatCompletionChunk> = deltas
        .into_iter()
        .map(|delta| chunk(delta, None, None))
        .collect();
    chunks.push(chunk(
        ChatCompletionDelta::default(),
        Some("tool_calls"),
        None,
    ));
    let events = collect_events(chunks).await;
    let message = accumulate(&events);
    assert!(matches!(
        &message.content[0],
        ContentBlock::ToolUse { id, name, input }
            if id == "call_test_0" &&
                name == "get_weather" &&
                *input == serde_json::json!({ "location": "Paris" })
    ));
    assert_eq!(message.stop_reason, Some(StopReason::ToolUse));
    // Wire tags: tool_use block start and input_json_delta fragments.
    let start = events
        .iter()
        .find(|e| e.event_type() == "content_block_start")
        .unwrap();
    let json = serde_json::to_value(start).unwrap();
    assert_eq!(json["content_block"]["type"], "tool_use");
    let delta = events
        .iter()
        .find(|e| e.event_type() == "content_block_delta")
        .unwrap();
    let json = serde_json::to_value(delta).unwrap();
    assert_eq!(json["delta"]["type"], "input_json_delta");
}

#[test]
fn tool_blocks_translate_to_openai_messages() {
    // Assistant turn replaying a tool call, then a user turn with the result.
    let assistant: MessageParam = serde_json::from_value(serde_json::json!({
        "role": "assistant",
        "content": [
            { "type": "text", "text": "Checking the weather." },
            { "type": "tool_use", "id": "call_123", "name": "get_weather",
              "input": { "city": "Paris" } },
        ],
    }))
    .unwrap();
    let user: MessageParam = serde_json::from_value(serde_json::json!({
        "role": "user",
        "content": [
            { "type": "tool_result", "tool_use_id": "call_123",
              "content": "18C and sunny" },
        ],
    }))
    .unwrap();
    let system = MessageContent::Text("You are a weather bot.".to_string());
    let messages = to_openai_messages(Some(&system), &[assistant, user]);
    assert_eq!(messages.len(), 4);
    assert_eq!(
        messages[0],
        serde_json::json!({ "role": "system", "content": "You are a weather bot." })
    );
    let messages = &messages[1..];
    assert_eq!(messages[0]["role"], "assistant");
    assert_eq!(messages[0]["content"], "Checking the weather.");
    assert_eq!(messages[1]["tool_calls"][0]["id"], "call_123");
    assert_eq!(
        messages[1]["tool_calls"][0]["function"]["arguments"],
        "{\"city\":\"Paris\"}"
    );
    assert_eq!(messages[2]["role"], "tool");
    assert_eq!(messages[2]["tool_call_id"], "call_123");
    assert_eq!(messages[2]["content"], "18C and sunny");
}

#[test]
fn chat_inputs_fold_system_and_translate_tools() {
    let params: MessageCreateParams = serde_json::from_value(serde_json::json!({
        "model": "@a/x",
        "max_tokens": 64,
        "stream": true,
        "system": [{ "type": "text", "text": "be brief" }],
        "messages": [{ "role": "user", "content": "weather in Paris?" }],
        "tools": [{
            "name": "get_weather",
            "description": "Weather lookup",
            "input_schema": { "type": "object" },
        }],
        "stop_sequences": ["END"],
        "top_k": 40,
        "temperature": 0.5,
    }))
    .unwrap();
    let inputs = params.chat_inputs().unwrap();
    // `max_tokens` is the shared `max_output_tokens` knob; Anthropic-only
    // knobs ride through; OpenAI-only knobs stay unset.
    assert_eq!(inputs.max_output_tokens, Some(64));
    assert_eq!(inputs.stop_sequences, Some(vec!["END".to_string()]));
    assert_eq!(inputs.top_k, Some(40));
    assert_eq!(inputs.temperature, Some(0.5));
    assert_eq!(inputs.top_p, None);
    assert_eq!(inputs.reasoning_effort, None);
    assert_eq!(inputs.response_format, None);
    assert_eq!(
        inputs.messages,
        vec![
            serde_json::json!({ "role": "system", "content": "be brief" }),
            serde_json::json!({ "role": "user", "content": "weather in Paris?" }),
        ]
    );
    assert_eq!(
        inputs.tools,
        Some(vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Weather lookup",
                "parameters": { "type": "object" },
            },
        })])
    );
    let params: MessageCreateParams = serde_json::from_value(serde_json::json!({
        "model": "@a/x",
        "max_tokens": 64,
        "messages": [{ "role": "user", "content": "hi" }],
        "tools": [],
    }))
    .unwrap();
    assert_eq!(params.chat_inputs().unwrap().tools, None);
}

#[test]
fn chat_inputs_map_thinking_to_reasoning_effort() {
    fn effort(extra: serde_json::Value) -> Option<String> {
        let mut body = serde_json::json!({
            "model": "@a/x",
            "max_tokens": 4096,
            "messages": [{ "role": "user", "content": "hi" }],
        });
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        let params: MessageCreateParams = serde_json::from_value(body).unwrap();
        params.chat_inputs().unwrap().reasoning_effort
    }
    // Explicitly disabled thinking turns reasoning off.
    assert_eq!(
        effort(serde_json::json!({ "thinking": { "type": "disabled" } })),
        Some("none".to_string())
    );
    // Enabled with a budget (Claude Code's shape): budget is ignored,
    // reasoning is on at the default level.
    assert_eq!(
        effort(serde_json::json!({
            "thinking": { "type": "enabled", "budget_tokens": 4096 },
        })),
        Some("medium".to_string())
    );
    // Adaptive thinking with an explicit effort; `max` clamps to `xhigh`.
    assert_eq!(
        effort(serde_json::json!({
            "thinking": { "type": "adaptive", "display": "omitted" },
            "output_config": { "effort": "max" },
        })),
        Some("xhigh".to_string())
    );
    // Effort alone implies thinking at that level.
    assert_eq!(
        effort(serde_json::json!({ "output_config": { "effort": "low" } })),
        Some("low".to_string())
    );
    // Neither: leave the predictor's default alone.
    assert_eq!(effort(serde_json::json!({})), None);
    // Unknown future thinking types degrade to "enabled" instead of a 400.
    assert_eq!(
        effort(serde_json::json!({ "thinking": { "type": "telepathic" } })),
        Some("medium".to_string())
    );
}

#[test]
fn chat_inputs_reject_messages_without_payload() {
    fn inputs(messages: serde_json::Value) -> Result<ChatInputs> {
        let params: MessageCreateParams = serde_json::from_value(serde_json::json!({
            "model": "@a/x",
            "max_tokens": 64,
            "messages": messages,
        }))
        .unwrap();
        params.chat_inputs()
    }
    // Whitespace-only text renders as an empty turn once the chat
    // template trims it; reject it like Anthropic does.
    for content in [
        serde_json::json!("   "),
        serde_json::json!(""),
        serde_json::json!([]),
        serde_json::json!([{ "type": "text", "text": " \n" }]),
    ] {
        let error = inputs(serde_json::json!([{ "role": "user", "content": content }]))
            .err()
            .unwrap_or_else(|| panic!("{content} was accepted"));
        assert!(matches!(error, MunaError::InvalidInput(_)), "{error}");
        assert!(error.to_string().contains("messages[0]"), "{error}");
    }
    assert!(matches!(
        inputs(serde_json::json!([])),
        Err(MunaError::InvalidInput(_))
    ));
    // Tool blocks are payload on their own, including an empty result.
    inputs(serde_json::json!([
        { "role": "user", "content": "weather?" },
        { "role": "assistant", "content": [{
            "type": "tool_use", "id": "t1", "name": "get_weather", "input": {}
        }] },
        { "role": "user", "content": [{ "type": "tool_result", "tool_use_id": "t1" }] },
    ]))
    .unwrap();
    inputs(serde_json::json!([{ "role": "user", "content": " hi " }])).unwrap();
    // Empty assistant turns are replayed history and must pass.
    let inputs = inputs(serde_json::json!([
        { "role": "user", "content": "hi" },
        { "role": "assistant", "content": "" },
        { "role": "user", "content": "still there?" },
    ]))
    .unwrap();
    assert_eq!(inputs.messages[1]["role"], serde_json::json!("assistant"));
    assert_eq!(inputs.messages[1]["content"], serde_json::json!(""));
}

#[test]
fn replayed_response_content_is_accepted() {
    // Anthropic SDKs append the prior response's content verbatim, so
    // everything we emit must parse as input on the next turn.
    let reply = serde_json::to_value(vec![
        ContentBlock::Thinking {
            thinking: "The user said hello.".into(),
            signature: String::new(),
        },
        ContentBlock::Text {
            text: "Hey! What can I help with?".into(),
        },
    ])
    .unwrap();
    let params: MessageCreateParams = serde_json::from_value(serde_json::json!({
        "model": "@a/x",
        "max_tokens": 64,
        "messages": [
            { "role": "user", "content": "hello" },
            { "role": "assistant", "content": reply },
            { "role": "assistant", "content": [
                { "type": "redacted_thinking", "data": "opaque" }
            ] },
            { "role": "user", "content": "Is this folder please" },
        ],
    }))
    .unwrap();
    let inputs = params.chat_inputs().unwrap();
    // Thinking is dropped; the text stays on the assistant turn, and the
    // redacted-only turn emits nothing.
    assert_eq!(
        inputs.messages,
        vec![
            serde_json::json!({ "role": "user", "content": "hello" }),
            serde_json::json!({ "role": "assistant", "content": "Hey! What can I help with?" }),
            serde_json::json!({ "role": "user", "content": "Is this folder please" }),
        ]
    );
}

/// 2x2 solid-red RGBA PNG.
const RED_PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAFUlEQVR4nGP8z8Dwn4GBgYEJRIAwAB8XAgICR7MUAAAAAElFTkSuQmCC";

fn image_block() -> serde_json::Value {
    serde_json::json!({
        "type": "image",
        "source": { "type": "base64", "media_type": "image/png", "data": RED_PNG_B64 }
    })
}

#[test]
fn base64_images_become_ordered_placeholders() {
    let params: MessageCreateParams = serde_json::from_value(serde_json::json!({
        "model": "@a/x",
        "max_tokens": 64,
        "messages": [
            { "role": "user", "content": [
                image_block(),
                { "type": "text", "text": "what is this?" }
            ] },
            { "role": "assistant", "content": "A red square." },
            { "role": "user", "content": [
                { "type": "text", "text": "and this?" },
                image_block(),
                { "type": "text", "text": "same?" }
            ] },
        ],
    }))
    .unwrap();
    let inputs = params.chat_inputs().unwrap();
    let images = inputs.images.unwrap();
    assert_eq!(images.len(), 2);
    assert_eq!((images[0].width, images[0].height), (2, 2));
    assert_eq!(
        inputs.messages,
        vec![
            serde_json::json!({ "role": "user", "content": [
            { "type": "image" },
            { "type": "text", "text": "what is this?" },
        ] }),
            serde_json::json!({ "role": "assistant", "content": "A red square." }),
            serde_json::json!({ "role": "user", "content": [
            { "type": "text", "text": "and this?" },
            { "type": "image" },
            { "type": "text", "text": "same?" },
        ] }),
        ]
    );
}

#[test]
fn images_outside_user_turns_are_rejected() {
    let system: MessageCreateParams = serde_json::from_value(serde_json::json!({
        "model": "@a/x",
        "max_tokens": 64,
        "system": [image_block()],
        "messages": [{ "role": "user", "content": "hi" }],
    }))
    .unwrap();
    assert!(
        matches!(system.chat_inputs(), Err(MunaError::InvalidInput(m)) if m.contains("`system`"))
    );
    let inline_system: MessageCreateParams = serde_json::from_value(serde_json::json!({
        "model": "@a/x",
        "max_tokens": 64,
        "messages": [
            { "role": "system", "content": [image_block()] },
            { "role": "user", "content": "hi" },
        ],
    }))
    .unwrap();
    assert!(
        matches!(inline_system.chat_inputs(), Err(MunaError::InvalidInput(m)) if m.contains("`system`"))
    );
}

#[test]
fn unsupported_blocks_are_rejected_by_name() {
    fn error(block: serde_json::Value) -> String {
        let params: MessageCreateParams = serde_json::from_value(serde_json::json!({
            "model": "@a/x",
            "max_tokens": 64,
            "messages": [{ "role": "user", "content": [
                { "type": "text", "text": "what is this?" },
                block
            ] }],
        }))
        .unwrap();
        let error = params.chat_inputs().unwrap_err();
        assert!(matches!(error, MunaError::InvalidInput(_)), "{error}");
        error.to_string()
    }
    let url = error(serde_json::json!({
        "type": "image",
        "source": { "type": "url", "url": "https://example.com/cat.png" }
    }));
    assert!(
        url.contains("messages[0].content[1]") && url.contains("`url` source"),
        "{url}"
    );
    let file = error(serde_json::json!({
        "type": "image",
        "source": { "type": "file", "file_id": "file_123" }
    }));
    assert!(file.contains("unsupported source"), "{file}");
    let tool_result_image = error(serde_json::json!({
        "type": "tool_result",
        "tool_use_id": "t1",
        "content": [{
            "type": "image",
            "source": { "type": "base64", "media_type": "image/png", "data": RED_PNG_B64 }
        }]
    }));
    assert!(
        tool_result_image.contains("`tool_result`"),
        "{tool_result_image}"
    );
    let document = error(serde_json::json!({
        "type": "document",
        "source": { "type": "text", "media_type": "text/plain", "data": "hi" }
    }));
    assert!(document.contains("`document`"), "{document}");
    let unknown = error(serde_json::json!({ "type": "search_result", "title": "t" }));
    assert!(unknown.contains("unrecognized type"), "{unknown}");
}

#[test]
fn billing_header_is_stripped_and_inline_system_kept_in_place() {
    let params: MessageCreateParams = serde_json::from_value(serde_json::json!({
        "model": "@a/x",
        "max_tokens": 64,
        "system": [
            { "type": "text", "text": "x-anthropic-billing-header: cc_version=2.1 hash=abc" },
            { "type": "text", "text": "You are Claude Code." }
        ],
        "messages": [
            { "role": "user", "content": "hi" },
            // Header-only inline system turn: emits nothing.
            { "role": "system", "content": "x-anthropic-billing-header: hash=def" },
            // Real inline system turn: role and position preserved; the
            // compiled predictor decides whether to fold it.
            { "role": "system", "content": [
                { "type": "text", "text": "x-anthropic-billing-header: hash=ghi" },
                { "type": "text", "text": "Conversation was compacted." }
            ] },
            { "role": "user", "content": "again" },
        ],
    }))
    .unwrap();
    let inputs = params.chat_inputs().unwrap();
    assert_eq!(
        inputs.messages,
        vec![
            serde_json::json!({ "role": "system", "content": "You are Claude Code." }),
            serde_json::json!({ "role": "user", "content": "hi" }),
            serde_json::json!({ "role": "system", "content": "Conversation was compacted." }),
            serde_json::json!({ "role": "user", "content": "again" }),
        ]
    );
    // A header-only system prompt yields no leading system message.
    let params: MessageCreateParams = serde_json::from_value(serde_json::json!({
        "model": "@a/x",
        "max_tokens": 64,
        "system": "x-anthropic-billing-header: hash=abc",
        "messages": [{ "role": "user", "content": "hi" }],
    }))
    .unwrap();
    let inputs = params.chat_inputs().unwrap();
    assert_eq!(
        inputs.messages,
        vec![serde_json::json!({ "role": "user", "content": "hi" })]
    );
    // An empty inline system turn that is NOT a billing header is still
    // rejected like any other empty non-assistant message.
    assert!(matches!(
        inputs_for(serde_json::json!([
            { "role": "user", "content": "hi" },
            { "role": "system", "content": "   " },
        ])),
        Err(MunaError::InvalidInput(_))
    ));
}

fn inputs_for(messages: serde_json::Value) -> Result<ChatInputs> {
    let params: MessageCreateParams = serde_json::from_value(serde_json::json!({
        "model": "@a/x",
        "max_tokens": 64,
        "messages": messages,
    }))
    .unwrap();
    params.chat_inputs()
}

#[test]
fn events_accumulate_to_final_message() {
    let events = vec![
        RawMessageStreamEvent::MessageStart {
            message: Message {
                id: "msg-1".to_string(),
                r#type: "message".to_string(),
                role: "assistant".to_string(),
                content: Vec::new(),
                model: "test-model".to_string(),
                stop_reason: None,
                stop_sequence: None,
                usage: Usage {
                    input_tokens: Some(10),
                    ..Default::default()
                },
            },
        },
        RawMessageStreamEvent::ContentBlockStart {
            index: 0,
            content_block: ContentBlock::Text {
                text: String::new(),
            },
        },
        RawMessageStreamEvent::ContentBlockDelta {
            index: 0,
            delta: ContentBlockDelta::TextDelta {
                text: "Par".to_string(),
            },
        },
        RawMessageStreamEvent::ContentBlockDelta {
            index: 0,
            delta: ContentBlockDelta::TextDelta {
                text: "is.".to_string(),
            },
        },
        RawMessageStreamEvent::ContentBlockStop { index: 0 },
        RawMessageStreamEvent::MessageDelta {
            delta: MessageDelta {
                stop_reason: Some(StopReason::EndTurn),
                stop_sequence: None,
            },
            usage: Usage {
                input_tokens: Some(10),
                output_tokens: 5,
                ..Default::default()
            },
        },
        RawMessageStreamEvent::MessageStop,
    ];
    let message = accumulate(&events);
    assert!(matches!(
        &message.content[0],
        ContentBlock::Text { text } if text == "Paris."
    ));
    assert_eq!(message.stop_reason, Some(StopReason::EndTurn));
    assert_eq!(message.usage.input_tokens, Some(10));
    assert_eq!(message.usage.output_tokens, 5);
}

#[test]
fn events_serialize_with_wire_tags() {
    let event = RawMessageStreamEvent::ContentBlockDelta {
        index: 0,
        delta: ContentBlockDelta::TextDelta {
            text: "Hi".to_string(),
        },
    };
    let json = serde_json::to_value(&event).unwrap();
    assert_eq!(json["type"], "content_block_delta");
    assert_eq!(json["delta"]["type"], "text_delta");
    assert_eq!(json["delta"]["text"], "Hi");
    let round_trip: RawMessageStreamEvent = serde_json::from_value(json).unwrap();
    assert_eq!(round_trip.event_type(), "content_block_delta");
}
