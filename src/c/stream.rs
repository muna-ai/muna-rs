/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::ffi::c_void;

use crate::client::Result;

use super::{FXNStatus, Prediction, ValueMap};

/// Prediction stream.
pub struct PredictionStream {
    handle: *mut c_void,
}

// SAFETY: The native handle is safe to send across threads.
unsafe impl Send for PredictionStream {}
unsafe impl Sync for PredictionStream {}

impl PredictionStream {

    /// Create a prediction stream from a predictor and input map.
    pub fn create(predictor: *mut c_void, inputs: &ValueMap) -> Result<Self> {
        let mut handle = std::ptr::null_mut();
        let status = unsafe {
            super::FXNPredictorStreamPrediction(predictor, inputs.handle(), &mut handle)
        };
        super::check_status(status, "Failed to create prediction stream")?;
        Ok(Self { handle })
    }

    /// Read the next prediction from the stream.
    pub fn read_next(&mut self) -> Result<Option<Prediction>> {
        let mut prediction = std::ptr::null_mut();
        let status = unsafe {
            super::FXNPredictionStreamReadNext(self.handle, &mut prediction)
        };
        if status == FXNStatus::ErrorInvalidOperation as i32 {
            return Ok(None);
        }
        super::check_status(status, "Failed to read next prediction")?;
        Ok(Some(Prediction::from_raw(prediction)))
    }
}

impl Iterator for PredictionStream {
    type Item = Result<Prediction>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.read_next() {
            Ok(Some(prediction)) => Some(Ok(prediction)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

impl Drop for PredictionStream {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { super::FXNPredictionStreamRelease(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}
