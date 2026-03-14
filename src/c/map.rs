/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::collections::HashMap;
use std::ffi::{c_void, CString};

use crate::client::Result;
use crate::types;

use super::{check_status, Value};

/// Value map.
pub struct ValueMap {
    handle: *mut c_void,
    owned: bool,
}

// SAFETY: The native handle is safe to send across threads.
unsafe impl Send for ValueMap {}

impl ValueMap {

    /// Create an empty value map.
    pub fn new() -> Result<Self> {
        let mut handle = std::ptr::null_mut();
        let status = unsafe { super::FXNValueMapCreate(&mut handle) };
        check_status(status, "Failed to create value map")?;
        Ok(Self { handle, owned: true })
    }

    pub(crate) fn from_raw(handle: *mut c_void, owned: bool) -> Self {
        Self { handle, owned }
    }

    /// Get the number of entries.
    pub fn len(&self) -> usize {
        let mut count: i32 = 0;
        let status = unsafe { super::FXNValueMapGetSize(self.handle, &mut count) };
        if status != 0 { 0 } else { count as usize }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the key at an index.
    pub fn key(&self, index: usize) -> Result<String> {
        let mut buffer = vec![0u8; 256];
        let status = unsafe {
            super::FXNValueMapGetKey(self.handle, index as i32, buffer.as_mut_ptr() as *mut _, buffer.len() as i32)
        };
        check_status(status, &format!("Failed to get key at index {index}"))?;
        let key = unsafe { std::ffi::CStr::from_ptr(buffer.as_ptr() as *const _) }
            .to_string_lossy()
            .into_owned();
        Ok(key)
    }

    /// Get a value by key (non-owning).
    pub fn get(&self, key: &str) -> Result<Value> {
        let key_c = CString::new(key)
            .map_err(|e| crate::client::MunaError::Native(e.to_string()))?;
        let mut handle = std::ptr::null_mut();
        let status = unsafe { super::FXNValueMapGetValue(self.handle, key_c.as_ptr(), &mut handle) };
        check_status(status, &format!("Failed to get value for key '{key}'"))?;
        Ok(Value::from_raw(handle, false))
    }

    /// Set a value by key.
    pub fn set(&mut self, key: &str, value: Value) -> Result<()> {
        let key_c = CString::new(key)
            .map_err(|e| crate::client::MunaError::Native(e.to_string()))?;
        let status = unsafe {
            super::FXNValueMapSetValue(self.handle, key_c.as_ptr(), value.raw_handle())
        };
        check_status(status, &format!("Failed to set value for key '{key}'"))?;
        std::mem::forget(value);
        Ok(())
    }

    pub(crate) fn handle(&self) -> *mut c_void {
        self.handle
    }

    /// Create a value map from a dictionary
    pub fn from_dict(inputs: &HashMap<String, types::Value>) -> Result<Self> {
        let mut map = Self::new()?;
        for (name, obj) in inputs {
            let value = Value::from_object(obj)?;
            map.set(name, value)?;
        }
        Ok(map)
    }
}

impl Drop for ValueMap {
    fn drop(&mut self) {
        if !self.handle.is_null() && self.owned {
            unsafe { super::FXNValueMapRelease(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}
