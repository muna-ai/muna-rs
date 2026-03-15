/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::ffi::c_void;

use crate::client::Result;

use super::{check_status, Configuration, Prediction, ValueMap};

/// Predictor.
pub struct Predictor {
    handle: *mut c_void,
}

// SAFETY: The native handle is safe to send across threads.
// Callers are responsible for serializing calls per-model when the backend requires it.
unsafe impl Send for Predictor {}
unsafe impl Sync for Predictor {}

impl Predictor {

    /// Create a predictor from a configuration.
    pub fn new(configuration: &Configuration) -> Result<Self> {
        let mut handle = std::ptr::null_mut();
        let status = unsafe {
            super::FXNPredictorCreate(configuration.handle(), &mut handle)
        };
        check_status(status, "Failed to create predictor")?;
        Ok(Self { handle })
    }

    /// Create a prediction from input values.
    pub fn create_prediction(&self, inputs: &ValueMap) -> Result<Prediction> {
        let mut prediction = std::ptr::null_mut();
        let status = unsafe {
            super::FXNPredictorCreatePrediction(self.handle, inputs.handle(), &mut prediction)
        };
        check_status(status, "Failed to create prediction")?;
        Ok(Prediction::from_raw(prediction))
    }

    /// Get the raw handle pointer (for stream creation).
    pub(crate) fn raw_ptr(&self) -> *mut c_void {
        self.handle
    }
}

impl Drop for Predictor {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { super::FXNPredictorRelease(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}
