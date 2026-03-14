/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use muna::Muna;

#[tokio::test]
async fn test_retrieve_predictor() {
    let _ = dotenvy::dotenv();
    let muna = Muna::default();
    let predictor = muna.predictors.retrieve("@fxn/greeting").await.unwrap();
    assert!(predictor.is_some());
}
