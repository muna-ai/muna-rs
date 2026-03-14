/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::ffi::c_void;

use crate::client::Result;

use super::{check_status, ValueMap};

/// Prediction.
pub struct Prediction {
    handle: *mut c_void,
}

// SAFETY: The native handle is safe to send across threads.
unsafe impl Send for Prediction {}

impl Prediction {

    pub(crate) fn from_raw(handle: *mut c_void) -> Self {
        Self { handle }
    }

    /// Get the prediction ID.
    pub fn id(&self) -> Result<String> {
        let mut buffer = vec![0u8; 256];
        let status = unsafe {
            super::FXNPredictionGetID(self.handle, buffer.as_mut_ptr() as *mut _, buffer.len() as i32)
        };
        check_status(status, "Failed to get prediction ID")?;
        let id = unsafe { std::ffi::CStr::from_ptr(buffer.as_ptr() as *const _) }
            .to_string_lossy()
            .into_owned();
        Ok(id)
    }

    /// Get the prediction latency in milliseconds.
    pub fn latency(&self) -> Result<f64> {
        let mut latency: f64 = 0.0;
        let status = unsafe { super::FXNPredictionGetLatency(self.handle, &mut latency) };
        check_status(status, "Failed to get prediction latency")?;
        Ok(latency)
    }

    /// Get the prediction results as a value map.
    pub fn results(&self) -> Result<ValueMap> {
        let mut handle = std::ptr::null_mut();
        let status = unsafe { super::FXNPredictionGetResults(self.handle, &mut handle) };
        check_status(status, "Failed to get prediction results")?;
        Ok(ValueMap::from_raw(handle, false))
    }

    /// Get the prediction error, if any.
    pub fn error(&self) -> Result<Option<String>> {
        let mut buffer = vec![0u8; 2048];
        let _status = unsafe {
            super::FXNPredictionGetError(self.handle, buffer.as_mut_ptr() as *mut _, buffer.len() as i32)
        };
        let error = unsafe { std::ffi::CStr::from_ptr(buffer.as_ptr() as *const _) }
            .to_string_lossy()
            .into_owned();
        Ok(if error.is_empty() { None } else { Some(error) })
    }

    /// Get the prediction logs.
    pub fn logs(&self) -> Result<Option<String>> {
        let mut log_length: i32 = 0;
        let status = unsafe { super::FXNPredictionGetLogLength(self.handle, &mut log_length) };
        check_status(status, "Failed to get prediction log length")?;
        let mut buffer = vec![0u8; (log_length + 1) as usize];
        let status = unsafe {
            super::FXNPredictionGetLogs(self.handle, buffer.as_mut_ptr() as *mut _, buffer.len() as i32)
        };
        check_status(status, "Failed to get prediction logs")?;
        let logs = unsafe { std::ffi::CStr::from_ptr(buffer.as_ptr() as *const _) }
            .to_string_lossy()
            .into_owned();
        Ok(if logs.is_empty() { None } else { Some(logs) })
    }
}

impl Drop for Prediction {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { super::FXNPredictionRelease(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}
