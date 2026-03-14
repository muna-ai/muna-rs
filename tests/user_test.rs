/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use muna::Muna;

#[tokio::test]
async fn test_retrieve_user() {
    let _ = dotenvy::dotenv();
    let muna = Muna::default();
    let user = muna.users.retrieve().await.unwrap();
    assert!(user.is_some());
}
