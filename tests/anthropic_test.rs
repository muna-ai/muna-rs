/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use futures_util::StreamExt;
use muna::beta::anthropic::{
    ContentBlock, MessageContent, MessageCreateParams,
    MessageParam, RawMessageStreamEvent,
};
use muna::Muna;

#[tokio::test]
async fn test_create_message() {
    let _ = dotenvy::dotenv();
    let muna = Muna::default();
    let message = muna
        .beta
        .anthropic
        .messages
        .create(message_params())
        .await
        .unwrap();
    assert_eq!(message.r#type, "message");
    assert_eq!(message.role, "assistant");
    assert!(message.stop_reason.is_some());
    assert!(message
        .content
        .iter()
        .any(|block| matches!(block, ContentBlock::Text { text } if !text.is_empty())));
    assert!(message.usage.output_tokens > 0);
}

#[tokio::test]
async fn test_stream_message() {
    let _ = dotenvy::dotenv();
    let muna = Muna::default();
    let mut events = muna
        .beta
        .anthropic
        .messages
        .stream(message_params())
        .await
        .unwrap();
    let mut event_types = Vec::new();
    let mut text = String::new();
    while let Some(event) = events.next().await {
        let event = event.unwrap();
        event_types.push(event.event_type());
        if let RawMessageStreamEvent::ContentBlockDelta { delta, .. } = &event {
            if let muna::beta::anthropic::ContentBlockDelta::TextDelta { text: fragment } = delta {
                text.push_str(fragment);
            }
        }
    }
    assert_eq!(event_types.first(), Some(&"message_start"));
    assert_eq!(event_types.last(), Some(&"message_stop"));
    assert!(event_types.contains(&"content_block_start"));
    assert!(event_types.contains(&"message_delta"));
    assert!(!text.is_empty());
}

fn message_params() -> MessageCreateParams {
    MessageCreateParams {
        model: "@huggingface/smollm2-135m".to_string(),
        max_tokens: 64,
        messages: vec![MessageParam {
            role: "user".to_string(),
            content: MessageContent::Text("Say hello in one sentence.".to_string()),
        }],
        system: Some(MessageContent::Text(
            "You are a friendly assistant.".to_string(),
        )),
        ..Default::default()
    }
}
