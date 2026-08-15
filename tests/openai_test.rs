/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use futures_util::StreamExt;
use muna::beta::openai::{
    ChatCompletionCreateParams, ChatCompletionMessage,
    EmbeddingData, EncodingFormat, ImageCreateParams,
};
use muna::Muna;

#[tokio::test]
async fn test_create_embedding() {
    let _ = dotenvy::dotenv();
    let muna = Muna::default();
    let response = muna
        .beta
        .openai
        .embeddings
        .create(
            vec!["Hello world".to_string()],
            "@google/embedding-gemma",
            None,
            None,
            None,
        )
        .await
        .unwrap();
    assert_eq!(response.object, "list");
    assert!(!response.data.is_empty());
    assert_eq!(response.data[0].object, "embedding");
    assert!(matches!(
        response.data[0].embedding,
        EmbeddingData::Float(_)
    ));
}

#[tokio::test]
async fn test_create_embedding_base64() {
    let _ = dotenvy::dotenv();
    let muna = Muna::default();
    let response = muna
        .beta
        .openai
        .embeddings
        .create(
            vec!["Hello world".to_string()],
            "@google/embedding-gemma",
            None,
            Some(EncodingFormat::Base64),
            None,
        )
        .await
        .unwrap();
    assert_eq!(response.object, "list");
    assert!(!response.data.is_empty());
    assert_eq!(response.data[0].object, "embedding");
    assert!(matches!(
        response.data[0].embedding,
        EmbeddingData::Base64(_)
    ));
}

#[tokio::test]
async fn test_chat_completion_api_shape() {
    let muna = Muna::default();
    let _ = &muna.beta.openai.chat.completions;
}

#[tokio::test]
#[ignore = "requires MUNA_CHAT_MODEL to reference a chat predictor"]
async fn test_create_chat_completion() {
    let _ = dotenvy::dotenv();
    let muna = Muna::default();
    let response = muna
        .beta
        .openai
        .chat
        .completions
        .create(chat_params())
        .await
        .unwrap();
    assert_eq!(response.object, "chat.completion");
    assert!(!response.choices.is_empty());
    assert_eq!(response.choices[0].message.role, "assistant");
}

#[tokio::test]
async fn test_stream_chat_completion() {
    let _ = dotenvy::dotenv();
    let muna = Muna::default();
    let mut chunks = muna
        .beta
        .openai
        .chat
        .completions
        .stream(chat_params())
        .await
        .unwrap();
    let mut count = 0;
    while let Some(chunk) = chunks.next().await {
        let chunk = chunk.unwrap();
        assert_eq!(chunk.object, "chat.completion.chunk");
        count += 1;
    }
    assert!(count > 0);
}

#[tokio::test]
async fn test_generate_image() {
    let _ = dotenvy::dotenv();
    let muna = Muna::default();
    let response = muna
        .beta
        .openai
        .images
        .generate(ImageCreateParams {
            prompt: "a cat".to_string(),
            model: "@yusuf/image-model".to_string(),
            background: None,
            n: None,
            output_format: None,
            output_compression: None,
            size: None,
            acceleration: None,
        })
        .await
        .unwrap();
    let data = response.data.expect("no image data returned");
    assert!(!data.is_empty());
    let b64 = data[0].b64_json.as_ref().expect("no b64_json in image data");
    use base64::Engine;
    let bytes = base64::engine::general_purpose::STANDARD.decode(b64).unwrap();
    assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n", "expected PNG magic");
    std::fs::write("/tmp/muna_cat.png", &bytes).unwrap();
}

fn chat_params() -> ChatCompletionCreateParams {
    ChatCompletionCreateParams {
        model: "@huggingface/smollm2-360m".to_string(),
        messages: vec![ChatCompletionMessage {
            role: "user".to_string(),
            content: Some("Say hello in one sentence.".to_string()),
        }],
        ..Default::default()
    }
}
