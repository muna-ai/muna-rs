/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use serde::{Deserialize, Serialize};

/// Muna user.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    /// Muna username.
    pub username: String,
    /// Date created.
    pub created: Option<String>,
    /// Full name.
    pub name: Option<String>,
    /// User avatar URL.
    pub avatar: Option<String>,
    /// User bio.
    pub bio: Option<String>,
    /// User website.
    pub website: Option<String>,
    /// User GitHub.
    pub github: Option<String>,
}
