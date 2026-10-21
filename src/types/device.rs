/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::ffi::c_void;
use std::str::FromStr;
use std::sync::Arc;

use crate::client::{MunaError, Result};

/// Compute device used for local prediction acceleration.
///
/// The device is an opaque, platform-dependent pointer.
/// See https://docs.muna.ai/sdk/predictions for its type on each platform.
#[derive(Debug, Clone)]
pub struct Device {
    /// Held as an address so `Device` stays `Send + Sync`.
    address: usize,
    /// Backing storage when the pointer refers to an ordinal.
    #[allow(dead_code)]
    buffer: Option<Arc<i32>>,
}

impl Device {
    /// Create a device from a platform device pointer.
    pub fn from_pointer(pointer: *const c_void) -> Self {
        Self { address: pointer as usize, buffer: None }
    }

    /// Create a CUDA or HIP device from its ordinal.
    pub fn cuda(index: i32) -> Self {
        let buffer = Arc::new(index);
        let address = Arc::as_ptr(&buffer) as usize;
        Self { address, buffer: Some(buffer) }
    }

    /// Device pointer.
    pub fn as_ptr(&self) -> *const c_void {
        self.address as *const c_void
    }
}

impl FromStr for Device {
    type Err = MunaError;

    fn from_str(s: &str) -> Result<Self> {
        match s.split_once(':') {
            Some(("cuda", index)) => index
                .parse::<i32>()
                .ok()
                .filter(|index| *index >= 0)
                .map(Self::cuda)
                .ok_or_else(|| MunaError::Prediction(format!("Failed to parse device `{s}`"))),
            _ => Err(MunaError::Prediction(format!(
                "Failed to parse device `{s}`; expected `cuda:<index>`"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Device;

    fn ordinal(device: &Device) -> i32 {
        unsafe { *device.as_ptr().cast::<i32>() }
    }

    #[test]
    fn cuda_device_points_at_ordinal() {
        assert_eq!(ordinal(&Device::cuda(4)), 4);
    }

    #[test]
    fn cuda_device_clones_share_pointer() {
        let device = Device::cuda(3);
        let clone = device.clone();
        drop(device);
        assert_eq!(ordinal(&clone), 3);
        assert_eq!(clone.as_ptr(), clone.clone().as_ptr());
    }

    #[test]
    fn cuda_devices_have_distinct_pointers() {
        assert_ne!(Device::cuda(1).as_ptr(), Device::cuda(1).as_ptr());
    }

    #[test]
    fn device_from_pointer_round_trips() {
        let value = 0u8;
        let pointer = std::ptr::from_ref(&value).cast();
        assert_eq!(Device::from_pointer(pointer).as_ptr(), pointer);
    }

    #[test]
    fn parse_device() {
        let device: Device = "cuda:7".parse().unwrap();
        assert_eq!(ordinal(&device), 7);
        for value in ["cuda", "cuda:", "cuda:-1", "cuda:x", "gpu:0", "metal:0"] {
            assert!(value.parse::<Device>().is_err(), "{value} should not parse");
        }
    }

    #[test]
    fn device_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Device>();
    }
}
