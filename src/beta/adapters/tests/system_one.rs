/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use serde_json::json;

use super::*;

fn signature(inputs: Json, outputs: Json) -> Signature {
    serde_json::from_value(json!({ "inputs": inputs, "outputs": outputs })).unwrap()
}

fn decision_signature() -> Signature {
    signature(
        json!([
            {
                "name": "context",
                "dtype": "list",
                "denotation": "typesafe.systemone.state"
            },
            {
                "name": "schema",
                "dtype": "dict",
                "denotation": "typesafe.systemone.questions"
            }
        ]),
        json!([
            {
                "name": "result",
                "dtype": "dict",
                "schema": { "title": "SystemOneResponse" }
            }
        ]),
    )
}

fn questions(value: Json) -> IndexMap<String, Question> {
    serde_json::from_value(value).unwrap()
}

#[test]
fn deserializes_wire_request_and_ignores_router_fields() {
    // Parsed from text, as a server receives it: `json!` would build a
    // key-sorted `serde_json::Map` and lose the question order.
    let params: SystemOneCreateParams = serde_json::from_str(
        r#"{
        "model": "@bespokelabs/nimble-9b",
        "state": { "ticket": "I was charged twice." },
        "questions": {
            "route": { "type": "choice", "instructions": "Where should this go?",
                       "criteria": { "billing": "money", "bug": null } },
            "urgency": { "type": "score", "instructions": "How urgent?",
                         "criteria": ["routine", "today", "urgent"] },
            "escalate": { "type": "noul", "instructions": "Escalate?",
                          "criteria": { "true": "Needs a human", "false": "Bot can handle it" } }
        },
        "provider": { "allow_fallbacks": true },
        "session_id": "session-1234"
    }"#,
    )
    .unwrap();
    assert_eq!(
        params.questions.keys().collect::<Vec<_>>(),
        ["route", "urgency", "escalate"]
    );
    assert!(validate_questions(&params.questions).is_ok());
    let round_trip = serde_json::to_value(&params.questions["escalate"]).unwrap();
    assert_eq!(round_trip["type"], "noul");
    assert_eq!(round_trip["criteria"]["true"], "Needs a human");
}

#[test]
fn parses_mixed_response_in_order() {
    let output: PredictorOutput = serde_json::from_value(json!({
        "answers": {
            "route": {
                "type": "choice",
                "choice": "billing",
                "confidence": 0.99,
                "probabilities": { "billing": 0.99, "bug": 0.0, "account": 0.01 }
            },
            "urgency": {
                "type": "score",
                "score": 2.1,
                "confidence": 0.7,
                "legend": { "0": "routine", "1": "today" },
                "probabilities": { "0": 0.4, "1": 0.6 }
            },
            "escalate": { "type": "noul", "noul": 0.94 }
        },
        "usage": { "input_tokens": 412, "output_tokens": 3 }
    }))
    .unwrap();
    let questions: IndexMap<String, Question> = serde_json::from_str(r#"{
        "route": { "type": "choice", "instructions": "Where?", "criteria": { "billing": null } },
        "urgency": { "type": "score", "instructions": "How urgent?", "criteria": ["routine", "today"] },
        "escalate": { "type": "noul", "instructions": "Escalate?" }
    }"#).unwrap();
    let answers = order_answers(&questions, output.answers);
    assert_eq!(
        answers.keys().collect::<Vec<_>>(),
        ["route", "urgency", "escalate"]
    );
    assert!(matches!(&answers["escalate"], Answer::Noul(a) if a.noul == 0.94));
    assert!(matches!(&answers["route"], Answer::Choice(a) if a.choice == "billing"));
    assert!(matches!(&answers["urgency"], Answer::Score(a) if a.probabilities["1"] == 0.6));
    assert_eq!(output.usage.input_tokens, Some(412));
}

#[test]
fn rejects_out_of_range_criteria() {
    let empty = IndexMap::new();
    assert!(matches!(
        validate_questions(&empty),
        Err(MunaError::InvalidInput(_))
    ));
    let one_level = questions(json!({
        "q": { "type": "score", "instructions": "Rate.", "criteria": ["only"] }
    }));
    assert!(matches!(
        validate_questions(&one_level),
        Err(MunaError::InvalidInput(_))
    ));
    let eleven_levels = questions(json!({
        "q": { "type": "score", "instructions": "Rate.", "criteria": (0..11).collect::<Vec<_>>() }
    }));
    assert!(matches!(
        validate_questions(&eleven_levels),
        Err(MunaError::InvalidInput(_))
    ));
    let no_options = questions(json!({
        "q": { "type": "choice", "instructions": "Pick.", "criteria": {} }
    }));
    assert!(matches!(
        validate_questions(&no_options),
        Err(MunaError::InvalidInput(_))
    ));
}

#[test]
fn rejects_unknown_noul_criteria_keys() {
    let result = serde_json::from_value::<IndexMap<String, Question>>(json!({
        "q": {
            "type": "noul",
            "instructions": "Yes?",
            "criteria": { "maybe": "unsure" }
        }
    }));
    assert!(result.is_err());
}

#[test]
fn binds_by_role_not_by_name() {
    let info = TypeSafeSystemOneAdapter::from_signature(&decision_signature()).unwrap();
    let questions = questions(json!({
        "escalate": { "type": "noul", "instructions": "Escalate?" }
    }));
    let map = bind_inputs(json!("Charged twice."), &questions, &info).unwrap();
    assert!(matches!(
        map.get("context"),
        Some(Value::List(s)) if s == &vec![json!("Charged twice.")]
    ));
    assert!(matches!(
        map.get("schema"),
        Some(Value::Dict(q)) if q["escalate"]["type"] == "noul"
    ));
    assert_eq!(info.output, 0);
}

#[test]
fn state_becomes_a_list() {
    let info = TypeSafeSystemOneAdapter::from_signature(&decision_signature()).unwrap();
    let questions = questions(json!({ "q": { "type": "noul", "instructions": "Yes?" } }));
    let state_of = |state: Json| match bind_inputs(state, &questions, &info)
        .unwrap()
        .remove("context")
    {
        Some(Value::List(items)) => items,
        _ => panic!("state is not a list"),
    };
    assert_eq!(state_of(json!({ "a": 1 })), vec![json!({ "a": 1 })]);
    assert_eq!(
        state_of(json!(["x", { "b": 2 }])),
        vec![json!("x"), json!({ "b": 2 })]
    );
    assert!(matches!(
        bind_inputs(json!(42), &questions, &info),
        Err(MunaError::InvalidInput(_))
    ));
}

#[test]
fn rejects_predictors_without_the_contract() {
    let missing_questions = signature(
        json!([
            {
                "name": "context",
                "dtype": "list",
                "denotation": "typesafe.systemone.state"
            },
            {
                "name": "other",
                "dtype": "dict"
            }
        ]),
        json!([
            {
                "name": "result",
                "dtype": "dict",
                "schema": { "title": "SystemOneResponse" }
            }
        ]),
    );
    assert!(TypeSafeSystemOneAdapter::from_signature(&missing_questions).is_err());
    let missing_output = signature(
        json!([
            {
                "name": "context",
                "dtype": "list",
                "denotation": "typesafe.systemone.state"
            },
            {
                "name": "schema",
                "dtype": "dict",
                "denotation": "typesafe.systemone.questions"
            }
        ]),
        json!([{ "name": "result", "dtype": "dict", "schema": { "title": "Other" } }]),
    );
    assert!(TypeSafeSystemOneAdapter::from_signature(&missing_output).is_err());
}
