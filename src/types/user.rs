/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::fmt;
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

impl fmt::Display for User {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match serde_json::to_string_pretty(self) {
            Ok(json) => f.write_str(&json),
            Err(_) => write!(f, "{:?}", self),
        }
    }
}
