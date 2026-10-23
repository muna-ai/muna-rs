/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::ffi::{c_void, CString};

use crate::client::Result;
use crate::types::Device;

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
            super::FXNConfigurationGetTag(
                self.handle,
                buffer.as_mut_ptr() as *mut _,
                buffer.len() as i32,
            )
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
            super::FXNConfigurationGetToken(
                self.handle,
                buffer.as_mut_ptr() as *mut _,
                buffer.len() as i32,
            )
        };
        check_status(status, "Failed to get configuration token")?;
        let token = unsafe { std::ffi::CStr::from_ptr(buffer.as_ptr() as *const _) }
            .to_string_lossy()
            .into_owned();
        Ok(if token.is_empty() { None } else { Some(token) })
    }

    /// Set the configuration token.
    pub fn set_token(&mut self, token: &str) -> Result<()> {
        let token =
            CString::new(token).map_err(|e| crate::client::MunaError::Native(e.to_string()))?;
        let status = unsafe { super::FXNConfigurationSetToken(self.handle, token.as_ptr()) };
        check_status(status, "Failed to set configuration token")
    }

    /// Set the prediction acceleration.
    pub fn set_acceleration(&mut self, acceleration: i32) -> Result<()> {
        let status = unsafe { super::FXNConfigurationSetAcceleration(self.handle, acceleration) };
        check_status(status, "Failed to set configuration acceleration")
    }

    /// Get the compute devices.
    pub fn devices(&self) -> Result<Vec<*mut c_void>> {
        let mut count = 0i32;
        let status = unsafe {
            super::FXNConfigurationGetDevices(self.handle, std::ptr::null_mut(), &mut count)
        };
        check_status(status, "Failed to get configuration device count")?;
        let mut devices = vec![std::ptr::null_mut(); count as usize];
        if count == 0 {
            return Ok(devices);
        }
        let status = unsafe {
            super::FXNConfigurationGetDevices(self.handle, devices.as_mut_ptr(), &mut count)
        };
        check_status(status, "Failed to get configuration devices")?;
        devices.truncate(count as usize);
        Ok(devices)
    }

    /// Set the compute devices.
    ///
    /// The native configuration stores the device pointers
    /// so the devices must outlive predictor creation.
    pub fn set_devices(&mut self, devices: &[Device]) -> Result<()> {
        let count = i32::try_from(devices.len()).map_err(|_| {
            crate::client::MunaError::Native(
                "Cannot set more than i32::MAX configuration devices".to_string(),
            )
        })?;
        let pointers = devices.iter().map(Device::as_ptr).collect::<Vec<_>>();
        let ptr = if pointers.is_empty() {
            std::ptr::null()
        } else {
            pointers.as_ptr()
        };
        let status = unsafe { super::FXNConfigurationSetDevices(self.handle, ptr, count) };
        check_status(status, "Failed to set configuration devices")
    }

    /// Add a resource to the configuration.
    pub fn add_resource(&mut self, resource_type: &str, path: &str) -> Result<()> {
        let rtype = CString::new(resource_type)
            .map_err(|e| crate::client::MunaError::Native(e.to_string()))?;
        let rpath =
            CString::new(path).map_err(|e| crate::client::MunaError::Native(e.to_string()))?;
        let status = unsafe {
            super::FXNConfigurationAddResource(self.handle, rtype.as_ptr(), rpath.as_ptr())
        };
        check_status(status, "Failed to add configuration resource")
    }

    /// Get a metadata value.
    pub fn metadata(&self, key: &str) -> Result<Option<String>> {
        let key = CString::new(key).map_err(|e| crate::client::MunaError::Native(e.to_string()))?;
        let mut buffer = vec![0u8; 2048];
        let status = unsafe {
            super::FXNConfigurationGetMetadata(
                self.handle,
                key.as_ptr(),
                buffer.as_mut_ptr() as *mut _,
                buffer.len() as i32,
            )
        };
        if status == super::FXNStatus::ErrorInvalidArgument as i32 {
            return Ok(None);
        }
        check_status(status, "Failed to get configuration metadata")?;
        let value = unsafe { std::ffi::CStr::from_ptr(buffer.as_ptr() as *const _) }
            .to_string_lossy()
            .into_owned();
        Ok(Some(value))
    }

    /// Set a metadata value.
    pub fn set_metadata(&mut self, key: &str, value: &str) -> Result<()> {
        let key = CString::new(key).map_err(|e| crate::client::MunaError::Native(e.to_string()))?;
        let value =
            CString::new(value).map_err(|e| crate::client::MunaError::Native(e.to_string()))?;
        let status = unsafe {
            super::FXNConfigurationSetMetadata(self.handle, key.as_ptr(), value.as_ptr())
        };
        check_status(status, "Failed to set configuration metadata")
    }

    /// Remove a metadata value.
    pub fn remove_metadata(&mut self, key: &str) -> Result<()> {
        let key = CString::new(key).map_err(|e| crate::client::MunaError::Native(e.to_string()))?;
        let status = unsafe {
            super::FXNConfigurationSetMetadata(self.handle, key.as_ptr(), std::ptr::null())
        };
        check_status(status, "Failed to remove configuration metadata")
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

    /// Get the current target triple.
    pub fn get_target() -> Result<String> {
        let mut buffer = vec![0u8; 64];
        let status = unsafe {
            super::FXNConfigurationGetTarget(buffer.as_mut_ptr() as *mut _, buffer.len() as i32)
        };
        check_status(status, "Failed to get target")?;
        let target = unsafe { std::ffi::CStr::from_ptr(buffer.as_ptr() as *const _) }
            .to_string_lossy()
            .into_owned();
        Ok(target)
    }

    /// Get the current GPU compute architecture, or `None` without a supported GPU.
    pub fn get_compute_architecture() -> Result<Option<String>> {
        let mut buffer = vec![0u8; 16];
        let status = unsafe {
            super::FXNConfigurationGetComputeArchitecture(
                buffer.as_mut_ptr() as *mut _,
                buffer.len() as i32
            )
        };
        check_status(status, "Failed to get compute architecture")?;
        let architecture = unsafe { std::ffi::CStr::from_ptr(buffer.as_ptr() as *const _) }
            .to_string_lossy()
            .into_owned();
        Ok((!architecture.is_empty()).then_some(architecture))
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

#[cfg(test)]
mod tests {
    use super::Configuration;
    use crate::types::Device;
    use std::ffi::c_void;

    #[test]
    fn metadata_roundtrip_and_remove() {
        let mut configuration = Configuration::new().unwrap();
        assert_eq!(configuration.metadata("owner").unwrap(), None);

        configuration.set_metadata("owner", "muna").unwrap();
        assert_eq!(
            configuration.metadata("owner").unwrap().as_deref(),
            Some("muna")
        );

        configuration.remove_metadata("owner").unwrap();
        assert_eq!(configuration.metadata("owner").unwrap(), None);
    }

    #[test]
    fn devices_roundtrip_and_clear() {
        let mut configuration = Configuration::new().unwrap();
        assert!(configuration.devices().unwrap().is_empty());

        let handle = 0u8;
        let devices = [
            Device::cuda(4),
            Device::from_pointer(std::ptr::from_ref(&handle).cast::<c_void>()),
            Device::cuda(5),
        ];
        configuration.set_devices(&devices).unwrap();
        let pointers = configuration.devices().unwrap();
        assert_eq!(
            pointers,
            devices.iter().map(|device| device.as_ptr().cast_mut()).collect::<Vec<_>>()
        );
        assert_eq!(unsafe { *pointers[0].cast::<i32>() }, 4);
        assert_eq!(unsafe { *pointers[2].cast::<i32>() }, 5);

        configuration.set_devices(&[]).unwrap();
        assert!(configuration.devices().unwrap().is_empty());
    }
}
