/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use serde_json::json;

use crate::types;

use super::*;

fn signature(inputs: serde_json::Value) -> Signature {
    serde_json::from_value(json!({ "inputs": inputs, "outputs": [] })).unwrap()
}

#[test]
fn binds_by_role_not_by_name() {
    // Parameter names are the model author's; binding must not
    // presuppose `messages` / `tools`.
    let signature = signature(json!([
        { "name": "conversation", "dtype": "list" },
        { "name": "functions", "dtype": "list", "optional": true,
          "denotation": "openai.chat.completions.tools" },
        { "name": "heat", "dtype": "float32", "optional": true,
          "denotation": "openai.chat.completions.temperature" },
        { "name": "budget", "dtype": "int32", "optional": true,
          "denotation": "openai.chat.completions.max_output_tokens" },
        { "name": "rng", "dtype": "int32", "optional": true,
          "denotation": "openai.chat.completions.seed" },
        { "name": "effort", "dtype": "string", "optional": true,
          "denotation": "openai.chat.completions.reasoning_effort" },
        { "name": "format", "dtype": "dict", "optional": true,
          "denotation": "openai.chat.completions.response_format" },
        { "name": "stops", "dtype": "list", "optional": true,
          "denotation": "openai.chat.completions.stop" },
        { "name": "k", "dtype": "int32", "optional": true,
          "denotation": "anthropic.messages.top_k" }
    ]));
    let inputs = ChatInputs {
        messages: vec![json!({ "role": "user", "content": "hi" })],
        images: None,
        tools: Some(vec![json!({ "type": "function" })]),
        response_format: Some(
            json!({ "type": "json_object" })
                .as_object()
                .cloned()
                .unwrap(),
        ),
        reasoning_effort: Some("high".into()),
        max_output_tokens: Some(64),
        temperature: Some(0.5),
        top_p: Some(0.9),
        seed: Some(1234),
        frequency_penalty: None,
        presence_penalty: None,
        stop_sequences: Some(vec!["END".into()]),
        top_k: Some(40),
    };
    let map = bind_chat_inputs(inputs, &signature).unwrap();
    assert!(matches!(map.get("conversation"), Some(Value::List(m)) if m.len() == 1));
    assert!(matches!(map.get("functions"), Some(Value::List(t)) if t.len() == 1));
    assert!(matches!(map.get("heat"), Some(Value::Float(v)) if *v == 0.5));
    assert!(matches!(map.get("budget"), Some(Value::Int(64))));
    assert!(matches!(map.get("effort"), Some(Value::String(s)) if s == "high"));
    assert!(matches!(map.get("format"), Some(Value::Dict(d)) if d["type"] == "json_object"));
    assert!(matches!(
        map.get("stops"),
        Some(Value::List(s)) if s == &vec![json!("END")]
    ));
    assert!(matches!(map.get("k"), Some(Value::Int(40))));
    assert!(matches!(map.get("rng"), Some(Value::Int(1234))));
    // `top_p` set but undeclared: silently dropped, nothing else bound.
    assert_eq!(map.len(), 9);
}

#[test]
fn undeclared_knobs_are_dropped_without_error() {
    let signature = signature(json!([{ "name": "messages", "dtype": "list" }]));
    let inputs = ChatInputs {
        messages: vec![],
        temperature: Some(0.2),
        max_output_tokens: Some(8),
        ..Default::default()
    };
    let map = bind_chat_inputs(inputs, &signature).unwrap();
    assert_eq!(map.len(), 1);
    assert!(matches!(map.get("messages"), Some(Value::List(_))));
}

#[test]
fn tools_without_a_tools_parameter_is_a_caller_error() {
    let signature = signature(json!([{ "name": "messages", "dtype": "list" }]));
    let inputs = ChatInputs {
        messages: vec![],
        tools: Some(vec![json!({ "type": "function" })]),
        ..Default::default()
    };
    assert!(matches!(
        bind_chat_inputs(inputs, &signature),
        Err(MunaError::InvalidInput(_))
    ));
    // No tools requested: the missing parameter is irrelevant.
    let inputs = ChatInputs {
        messages: vec![],
        ..Default::default()
    };
    assert!(bind_chat_inputs(inputs, &signature).is_ok());
}

#[test]
fn images_bind_to_the_images_parameter_or_fail() {
    let image = types::Image {
        data: vec![0; 4],
        width: 1,
        height: 1,
        channels: 4,
    };
    let inputs = || ChatInputs {
        messages: vec![],
        images: Some(vec![image.clone()]),
        ..Default::default()
    };
    let with_images = signature(json!([
        { "name": "conversation", "dtype": "list" },
        { "name": "pictures", "dtype": "image_list", "optional": true,
          "denotation": "openai.chat.completions.images" }
    ]));
    let map = bind_chat_inputs(inputs(), &with_images).unwrap();
    assert!(matches!(map.get("pictures"), Some(Value::ImageList(i)) if i == &vec![image.clone()]));
    let without = signature(json!([{ "name": "messages", "dtype": "list" }]));
    assert!(matches!(
        bind_chat_inputs(inputs(), &without),
        Err(MunaError::InvalidInput(_))
    ));
    // No images in the request: the images parameter stays unbound.
    let map = bind_chat_inputs(ChatInputs::default(), &with_images).unwrap();
    assert!(!map.contains_key("pictures"));
}

#[test]
fn rejects_signatures_without_a_single_required_list_input() {
    let none = signature(json!([{ "name": "x", "dtype": "list", "optional": true }]));
    let two = signature(json!([
        { "name": "a", "dtype": "list" },
        { "name": "b", "dtype": "list" }
    ]));
    let not_list = signature(json!([{ "name": "prompt", "dtype": "string" }]));
    for signature in [none, two, not_list] {
        assert!(matches!(
            bind_chat_inputs(ChatInputs::default(), &signature),
            Err(MunaError::Prediction(_))
        ));
    }
}
