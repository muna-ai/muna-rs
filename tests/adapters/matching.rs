/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use muna::beta::adapters::*;
use muna::Signature;
use serde_json::json;

use super::common::signature;

#[test]
fn signature_matching_is_optional_and_strict_for_every_adapter() {
    type Matches = fn(&Signature) -> bool;
    let cases: Vec<(Matches, Signature)> = vec![
        (
            |s| OpenAICompletionsAdapter::from_signature(s).is_some(),
            signature(
                json!([{"name": "messages", "dtype": "list", "denotation": "openai.chat.completions.messages"}]),
                json!([{"name": "chunk", "dtype": "dict", "schema": {"title": "ChatCompletionChunk"}}]),
            ),
        ),
        (
            |s| AnthropicMessagesAdapter::from_signature(s).is_some(),
            signature(
                json!([{"name": "messages", "dtype": "list", "denotation": "openai.chat.completions.messages"}]),
                json!([{"name": "chunk", "dtype": "dict", "schema": {"title": "ChatCompletionChunk"}}]),
            ),
        ),
        (
            |s| OpenAIEmbeddingsAdapter::from_signature(s).is_some(),
            signature(
                json!([{"name": "texts", "dtype": "list"}]),
                json!([{"name": "vectors", "dtype": "float32", "denotation": "embedding"}]),
            ),
        ),
        (
            |s| OpenAIImageGenerationAdapter::from_signature(s).is_some(),
            signature(
                json!([{"name": "prompt", "dtype": "string", "denotation": "openai.images.prompt"}]),
                json!([{"name": "images", "dtype": "image_list"}]),
            ),
        ),
        (
            |s| OpenAIImageEditAdapter::from_signature(s).is_some(),
            signature(
                json!([{"name": "prompt", "dtype": "string", "denotation": "openai.images.prompt"}, {"name": "images", "dtype": "image_list", "denotation": "openai.chat.completions.images"}]),
                json!([{"name": "images", "dtype": "image_list"}]),
            ),
        ),
        (
            |s| OpenAIDepthAdapter::from_signature(s).is_some(),
            signature(
                json!([{"name": "prompt", "dtype": "string", "denotation": "openai.images.prompt"}, {"name": "images", "dtype": "image_list", "denotation": "openai.chat.completions.images"}]),
                json!([{"name": "depth", "dtype": "array_list", "denotation": "depth_map"}]),
            ),
        ),
        (
            |s| OpenAISpeechAdapter::from_signature(s).is_some(),
            signature(
                json!([{"name": "text", "dtype": "string"}, {"name": "voice", "dtype": "string", "denotation": "openai.audio.speech.voice"}]),
                json!([{"name": "audio", "dtype": "float32", "denotation": "audio", "sampleRate": 24000}]),
            ),
        ),
        (
            |s| OpenAITranscriptionsAdapter::from_signature(s).is_some(),
            signature(
                json!([{"name": "audio", "dtype": "float32", "denotation": "audio", "sampleRate": 16000}]),
                json!([{"name": "text", "dtype": "string"}]),
            ),
        ),
        (
            |s| TypeSafeSystemOneAdapter::from_signature(s).is_some(),
            signature(
                json!([{"name": "state", "dtype": "list", "denotation": "typesafe.systemone.state"}, {"name": "questions", "dtype": "dict", "denotation": "typesafe.systemone.questions"}]),
                json!([{"name": "response", "dtype": "dict", "schema": {"title": "SystemOneResponse"}}]),
            ),
        ),
    ];
    for (matches, original) in cases {
        assert!(matches(&original));
        assert!(!matches(&signature(json!([]), json!([]))));

        let mut missing_output = original.clone();
        missing_output.outputs.clear();
        assert!(!matches(&missing_output));

        let mut wrong_type = original.clone();
        wrong_type.inputs[0].dtype = Some(muna::Dtype::Bool);
        assert!(!matches(&wrong_type));

        let mut unsupported = original.clone();
        unsupported
            .inputs
            .push(serde_json::from_value(json!({"name":"unsupported","dtype":"bool"})).unwrap());
        assert!(!matches(&unsupported));

        let mut duplicate_name = original.clone();
        duplicate_name.inputs.push(
            serde_json::from_value(
                json!({"name":original.inputs[0].name,"dtype":"bool","optional":true}),
            )
            .unwrap(),
        );
        assert!(!matches(&duplicate_name));

        let mut ambiguous = original.clone();
        let mut duplicate = original.inputs[0].clone();
        duplicate.name = "duplicate".into();
        ambiguous.inputs.push(duplicate);
        assert!(!matches(&ambiguous));

        let mut ambiguous_output = original.clone();
        let mut duplicate = original.outputs[0].clone();
        duplicate.name = "duplicate".into();
        ambiguous_output.outputs.push(duplicate);
        assert!(!matches(&ambiguous_output));

        let mut optional = original.clone();
        optional.inputs.push(
            serde_json::from_value(json!({"name":"unused","dtype":"bool","optional":true}))
                .unwrap(),
        );
        assert!(matches(&optional));
    }
}

#[test]
fn chat_requires_messages_denotation() {
    let mut signature = signature(
        json!([{
            "name":"messages",
            "dtype":"list",
            "schema":{"type":"array","items":{"type":"object","title":"Message"}}
        }]),
        json!([{"name":"chunk","dtype":"dict","schema":{"title":"ChatCompletionChunk"}}]),
    );
    for denotation in [
        None,
        Some("openai.chat.completions.tools"),
        Some("openai.chat.completions.messages"),
    ] {
        signature.inputs[0].denotation = denotation.map(str::to_string);
        let compatible = denotation == Some("openai.chat.completions.messages");
        assert_eq!(
            OpenAICompletionsAdapter::from_signature(&signature).is_some(),
            compatible
        );
        assert_eq!(
            AnthropicMessagesAdapter::from_signature(&signature).is_some(),
            compatible
        );
    }

    signature.inputs[0].name = "conversation".into();
    signature.inputs.push(
        serde_json::from_value(json!({"name":"unrelated","dtype":"list","optional":true})).unwrap(),
    );
    assert_eq!(
        OpenAICompletionsAdapter::from_signature(&signature)
            .unwrap()
            .messages
            .name,
        "conversation"
    );
    assert_eq!(
        AnthropicMessagesAdapter::from_signature(&signature)
            .unwrap()
            .messages
            .name,
        "conversation"
    );
}

#[test]
fn ambiguous_optional_parameter_is_not_treated_as_absent() {
    let signature = signature(
        json!([
            {"name":"messages","dtype":"list","denotation":"openai.chat.completions.messages"},
            {"name":"first","dtype":"float32","optional":true,"denotation":"openai.chat.completions.temperature"},
            {"name":"second","dtype":"float32","optional":true,"denotation":"openai.chat.completions.temperature"}
        ]),
        json!([{"name":"chunk","dtype":"dict","schema":{"title":"ChatCompletionChunk"}}]),
    );
    assert!(OpenAICompletionsAdapter::from_signature(&signature).is_none());
    assert!(AnthropicMessagesAdapter::from_signature(&signature).is_none());
}
