/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use muna::Muna;
use muna::beta::openai::{EmbeddingData, EncodingFormat};

#[tokio::test]
async fn test_create_embedding() {
    let _ = dotenvy::dotenv();
    let muna = Muna::default();
    let response = muna.beta.openai.embeddings.create(
        vec!["Hello world".to_string()],
        "@google/embedding-gemma",
        None,
        None,
        None,
    ).await.unwrap();
    assert_eq!(response.object, "list");
    assert!(!response.data.is_empty());
    assert_eq!(response.data[0].object, "embedding");
    assert!(matches!(response.data[0].embedding, EmbeddingData::Float(_)));
}

#[tokio::test]
async fn test_create_embedding_base64() {
    let _ = dotenvy::dotenv();
    let muna = Muna::default();
    let response = muna.beta.openai.embeddings.create(
        vec!["Hello world".to_string()],
        "@google/embedding-gemma",
        None,
        Some(EncodingFormat::Base64),
        None,
    ).await.unwrap();
    assert_eq!(response.object, "list");
    assert!(!response.data.is_empty());
    assert_eq!(response.data[0].object, "embedding");
    assert!(matches!(response.data[0].embedding, EmbeddingData::Base64(_)));
}
