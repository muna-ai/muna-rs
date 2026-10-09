/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use muna::beta::typesafe::{Answer, SystemOneCreateParams};
use muna::Muna;
use serde_json::json;

#[tokio::test]
#[ignore = "requires a published System One decision predictor"]
async fn test_system_one() {
    let _ = dotenvy::dotenv();
    let muna = Muna::default();
    let response = muna
        .beta
        .typesafe
        .system_one(SystemOneCreateParams {
            model: "@bespokelabs/nimble-9b".into(),
            state: json!("Customer: I was charged twice and nobody has replied for 3 days."),
            questions: serde_json::from_value(json!({
                "route": {
                    "type": "choice",
                    "instructions": "Where should this go?",
                    "criteria": { "billing": "money", "bug": "broken", "account": "login" }
                },
                "urgency": {
                    "type": "score",
                    "instructions": "How urgent is this?",
                    "criteria": ["routine", "today", "urgent", "critical"]
                },
                "escalate": { "type": "noul", "instructions": "Escalate to a human now?" }
            }))
            .unwrap(),
            acceleration: None,
        })
        .await
        .unwrap();
    assert_eq!(response.answers.len(), 3);
    assert!(matches!(response.answers["route"], Answer::Choice(_)));
    assert!(matches!(response.answers["urgency"], Answer::Score(_)));
    assert!(matches!(response.answers["escalate"], Answer::Noul(_)));
}
