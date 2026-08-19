/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::sync::Arc;

use crate::client::{Client, ClientExt, MunaError, RequestInput, Result};
use crate::types::User;

/// Manage users.
#[derive(Clone)]
pub struct UserService {
    client: Arc<dyn Client>,
}

impl UserService {

    pub fn new(client: Arc<dyn Client>) -> Self {
        Self { client }
    }

    /// Retrieve the currently authenticated user.
    pub async fn retrieve(&self) -> Result<Option<User>> {
        match self.client.request_as(RequestInput::get("/users")).await {
            Ok(user) => Ok(Some(user)),
            Err(MunaError::Api { status: 401, .. }) => Ok(None),
            Err(e) => Err(e),
        }
    }
}
