/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::ffi::{c_void, CString};

use crate::client::Result;

use super::check_status;

/// Configuration.
pub struct Configuration {
    handle: *mut c_void,
}

// SAFETY: The native handle is safe to send across threads.
unsafe impl Send for Configuration {}

impl Configuration {

    /// Create a new configuration.
    pub fn new() -> Result<Self> {
        let mut handle = std::ptr::null_mut();
        let status = unsafe { super::FXNConfigurationCreate(&mut handle) };
        check_status(status, "Failed to create configuration")?;
        Ok(Self { handle })
    }

    /// Get the configuration tag.
    pub fn tag(&self) -> Result<Option<String>> {
        let mut buffer = vec![0u8; 2048];
        let status = unsafe {
            super::FXNConfigurationGetTag(self.handle, buffer.as_mut_ptr() as *mut _, buffer.len() as i32)
        };
        check_status(status, "Failed to get configuration tag")?;
        let tag = unsafe { std::ffi::CStr::from_ptr(buffer.as_ptr() as *const _) }
            .to_string_lossy()
            .into_owned();
        Ok(if tag.is_empty() { None } else { Some(tag) })
    }

    /// Set the configuration tag.
    pub fn set_tag(&mut self, tag: &str) -> Result<()> {
        let tag = CString::new(tag).map_err(|e| crate::client::MunaError::Native(e.to_string()))?;
        let status = unsafe { super::FXNConfigurationSetTag(self.handle, tag.as_ptr()) };
        check_status(status, "Failed to set configuration tag")
    }

    /// Get the configuration token.
    pub fn token(&self) -> Result<Option<String>> {
        let mut buffer = vec![0u8; 2048];
        let status = unsafe {
            super::FXNConfigurationGetToken(self.handle, buffer.as_mut_ptr() as *mut _, buffer.len() as i32)
        };
        check_status(status, "Failed to get configuration token")?;
        let token = unsafe { std::ffi::CStr::from_ptr(buffer.as_ptr() as *const _) }
            .to_string_lossy()
            .into_owned();
        Ok(if token.is_empty() { None } else { Some(token) })
    }

    /// Set the configuration token.
    pub fn set_token(&mut self, token: &str) -> Result<()> {
        let token = CString::new(token).map_err(|e| crate::client::MunaError::Native(e.to_string()))?;
        let status = unsafe { super::FXNConfigurationSetToken(self.handle, token.as_ptr()) };
        check_status(status, "Failed to set configuration token")
    }

    /// Set the prediction acceleration.
    pub fn set_acceleration(&mut self, acceleration: i32) -> Result<()> {
        let status = unsafe { super::FXNConfigurationSetAcceleration(self.handle, acceleration) };
        check_status(status, "Failed to set configuration acceleration")
    }

    /// Add a resource to the configuration.
    pub fn add_resource(&mut self, resource_type: &str, path: &str) -> Result<()> {
        let rtype = CString::new(resource_type).map_err(|e| crate::client::MunaError::Native(e.to_string()))?;
        let rpath = CString::new(path).map_err(|e| crate::client::MunaError::Native(e.to_string()))?;
        let status = unsafe { super::FXNConfigurationAddResource(self.handle, rtype.as_ptr(), rpath.as_ptr()) };
        check_status(status, "Failed to add configuration resource")
    }

    /// Get the unique configuration identifier.
    pub fn get_unique_id() -> Result<String> {
        let mut buffer = vec![0u8; 2048];
        let status = unsafe {
            super::FXNConfigurationGetUniqueID(buffer.as_mut_ptr() as *mut _, buffer.len() as i32)
        };
        check_status(status, "Failed to get unique ID")?;
        let id = unsafe { std::ffi::CStr::from_ptr(buffer.as_ptr() as *const _) }
            .to_string_lossy()
            .into_owned();
        Ok(id)
    }

    /// Get the client identifier.
    pub fn get_client_id() -> Result<String> {
        let mut buffer = vec![0u8; 64];
        let status = unsafe {
            super::FXNConfigurationGetClientID(buffer.as_mut_ptr() as *mut _, buffer.len() as i32)
        };
        check_status(status, "Failed to get client ID")?;
        let id = unsafe { std::ffi::CStr::from_ptr(buffer.as_ptr() as *const _) }
            .to_string_lossy()
            .into_owned();
        Ok(id)
    }

    pub(crate) fn handle(&self) -> *mut c_void {
        self.handle
    }
}

impl Drop for Configuration {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { super::FXNConfigurationRelease(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}
