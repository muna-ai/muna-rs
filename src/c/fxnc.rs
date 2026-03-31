/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::ffi::{c_char, c_double, c_int, c_void};

use crate::client::MunaError;

/// Status codes.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FXNStatus {
    Ok = 0,
    ErrorInvalidArgument = 1,
    ErrorInvalidOperation = 2,
    ErrorNotImplemented = 3,
}

impl FXNStatus {

    pub fn is_ok(self) -> bool {
        self == Self::Ok
    }

    pub fn message(self) -> &'static str {
        match self {
            Self::Ok => "",
            Self::ErrorInvalidArgument => "invalid argument",
            Self::ErrorInvalidOperation => "invalid operation",
            Self::ErrorNotImplemented => "not implemented",
        }
    }
}

/// Value creation flags.
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum ValueFlags {
    None = 0,
    CopyData = 1,
}

extern "C" {
    // Configuration
    pub(crate) fn FXNConfigurationCreate(configuration: *mut *mut c_void) -> c_int;
    pub(crate) fn FXNConfigurationGetTag(configuration: *mut c_void, tag: *mut c_char, size: c_int) -> c_int;
    pub(crate) fn FXNConfigurationSetTag(configuration: *mut c_void, tag: *const c_char) -> c_int;
    pub(crate) fn FXNConfigurationGetToken(configuration: *mut c_void, token: *mut c_char, size: c_int) -> c_int;
    pub(crate) fn FXNConfigurationSetToken(configuration: *mut c_void, token: *const c_char) -> c_int;
    pub(crate) fn FXNConfigurationSetAcceleration(configuration: *mut c_void, acceleration: c_int) -> c_int;
    pub(crate) fn FXNConfigurationAddResource(configuration: *mut c_void, resource_type: *const c_char, path: *const c_char) -> c_int;
    pub(crate) fn FXNConfigurationGetUniqueID(buffer: *mut c_char, size: c_int) -> c_int;
    pub(crate) fn FXNConfigurationGetClientID(buffer: *mut c_char, size: c_int) -> c_int;
    pub(crate) fn FXNConfigurationRelease(configuration: *mut c_void) -> c_int;
    // Value
    pub(crate) fn FXNValueCreateNull(value: *mut *mut c_void) -> c_int;
    pub(crate) fn FXNValueCreateArray(data: *const c_void, shape: *const i32, dims: c_int, dtype: c_int, flags: c_int, value: *mut *mut c_void) -> c_int;
    pub(crate) fn FXNValueCreateString(data: *const c_char, value: *mut *mut c_void) -> c_int;
    pub(crate) fn FXNValueCreateList(json: *const c_char, value: *mut *mut c_void) -> c_int;
    pub(crate) fn FXNValueCreateDict(json: *const c_char, value: *mut *mut c_void) -> c_int;
    pub(crate) fn FXNValueCreateImage(data: *const c_void, width: c_int, height: c_int, channels: c_int, flags: c_int, value: *mut *mut c_void) -> c_int;
    pub(crate) fn FXNValueCreateBinary(data: *const c_char, size: c_int, flags: c_int, value: *mut *mut c_void) -> c_int;
    pub(crate) fn FXNValueCreateArrayList(data: *const *const c_void, shapes: *const *const i32, dims: *const i32, dtypes: *const i32, count: c_int, flags: c_int, value: *mut *mut c_void) -> c_int;
    pub(crate) fn FXNValueCreateImageList(pixel_buffers: *const *const c_void, widths: *const i32, heights: *const i32, channels: *const i32, count: c_int, flags: c_int, value: *mut *mut c_void) -> c_int;
    pub(crate) fn FXNValueCreateSerializedValue(value: *mut c_void, mime: *const c_char, serialized: *mut *mut c_void) -> c_int;
    pub(crate) fn FXNValueCreateFromSerializedValue(data: *mut c_void, mime: *const c_char, value: *mut *mut c_void) -> c_int;
    pub(crate) fn FXNValueGetData(value: *mut c_void, data: *mut *mut c_void) -> c_int;
    pub(crate) fn FXNValueGetType(value: *mut c_void, dtype: *mut c_int) -> c_int;
    pub(crate) fn FXNValueGetDimensions(value: *mut c_void, dimensions: *mut i32) -> c_int;
    pub(crate) fn FXNValueGetShape(value: *mut c_void, shape: *mut i32, dims: i32) -> c_int;
    pub(crate) fn FXNValueRelease(value: *mut c_void) -> c_int;
    // ValueMap
    pub(crate) fn FXNValueMapCreate(map: *mut *mut c_void) -> c_int;
    pub(crate) fn FXNValueMapGetSize(map: *mut c_void, size: *mut i32) -> c_int;
    pub(crate) fn FXNValueMapGetKey(map: *mut c_void, index: c_int, key: *mut c_char, size: c_int) -> c_int;
    pub(crate) fn FXNValueMapGetValue(map: *mut c_void, key: *const c_char, value: *mut *mut c_void) -> c_int;
    pub(crate) fn FXNValueMapSetValue(map: *mut c_void, key: *const c_char, value: *mut c_void) -> c_int;
    pub(crate) fn FXNValueMapRelease(map: *mut c_void) -> c_int;
    // Prediction
    pub(crate) fn FXNPredictionGetID(prediction: *mut c_void, id: *mut c_char, size: c_int) -> c_int;
    pub(crate) fn FXNPredictionGetLatency(prediction: *mut c_void, latency: *mut c_double) -> c_int;
    pub(crate) fn FXNPredictionGetResults(prediction: *mut c_void, results: *mut *mut c_void) -> c_int;
    pub(crate) fn FXNPredictionGetError(prediction: *mut c_void, error: *mut c_char, size: c_int) -> c_int;
    pub(crate) fn FXNPredictionGetLogLength(prediction: *mut c_void, length: *mut i32) -> c_int;
    pub(crate) fn FXNPredictionGetLogs(prediction: *mut c_void, logs: *mut c_char, size: c_int) -> c_int;
    pub(crate) fn FXNPredictionRelease(prediction: *mut c_void) -> c_int;
    // PredictionStream
    pub(crate) fn FXNPredictionStreamReadNext(stream: *mut c_void, prediction: *mut *mut c_void) -> c_int;
    pub(crate) fn FXNPredictionStreamRelease(stream: *mut c_void) -> c_int;
    // Predictor
    pub(crate) fn FXNPredictorCreate(configuration: *mut c_void, predictor: *mut *mut c_void) -> c_int;
    pub(crate) fn FXNPredictorCreatePrediction(predictor: *mut c_void, inputs: *mut c_void, prediction: *mut *mut c_void) -> c_int;
    pub(crate) fn FXNPredictorStreamPrediction(predictor: *mut c_void, inputs: *mut c_void, stream: *mut *mut c_void) -> c_int;
    pub(crate) fn FXNPredictorRelease(predictor: *mut c_void) -> c_int;
}

pub(crate) fn dtype_to_c(dtype: crate::types::Dtype) -> i32 {
    use crate::types::Dtype;
    match dtype {
        Dtype::Null         => 0,
        Dtype::Float16      => 1,
        Dtype::Float32      => 2,
        Dtype::Float64      => 3,
        Dtype::Int8         => 4,
        Dtype::Int16        => 5,
        Dtype::Int32        => 6,
        Dtype::Int64        => 7,
        Dtype::Uint8        => 8,
        Dtype::Uint16       => 9,
        Dtype::Uint32       => 10,
        Dtype::Uint64       => 11,
        Dtype::Bool         => 12,
        Dtype::String       => 13,
        Dtype::List         => 14,
        Dtype::Dict         => 15,
        Dtype::Image        => 16,
        Dtype::Binary       => 17,
        Dtype::BFloat16     => 18,
        Dtype::ImageList    => 19,
        Dtype::ArrayList    => 20,
        Dtype::Complex64    => 21,
        Dtype::Complex128   => 22,
    }
}

pub(crate) fn dtype_from_c(value: i32) -> Option<crate::types::Dtype> {
    use crate::types::Dtype;
    match value {
        0  => Some(Dtype::Null),
        1  => Some(Dtype::Float16),
        2  => Some(Dtype::Float32),
        3  => Some(Dtype::Float64),
        4  => Some(Dtype::Int8),
        5  => Some(Dtype::Int16),
        6  => Some(Dtype::Int32),
        7  => Some(Dtype::Int64),
        8  => Some(Dtype::Uint8),
        9  => Some(Dtype::Uint16),
        10 => Some(Dtype::Uint32),
        11 => Some(Dtype::Uint64),
        12 => Some(Dtype::Bool),
        13 => Some(Dtype::String),
        14 => Some(Dtype::List),
        15 => Some(Dtype::Dict),
        16 => Some(Dtype::Image),
        17 => Some(Dtype::Binary),
        18 => Some(Dtype::BFloat16),
        19 => Some(Dtype::ImageList),
        20 => Some(Dtype::ArrayList),
        21 => Some(Dtype::Complex64),
        22 => Some(Dtype::Complex128),
        _  => None,
    }
}

pub(crate) fn acceleration_to_c(acceleration: &crate::types::Acceleration) -> i32 {
    use crate::types::Acceleration;
    match acceleration {
        Acceleration::LocalAuto  => 0,
        Acceleration::LocalCpu   => 1,
        Acceleration::LocalGpu   => 2,
        Acceleration::LocalNpu   => 4,
        _                        => 0,
    }
}

pub(crate) fn is_tensor_dtype(dtype: crate::types::Dtype) -> bool {
    use crate::types::Dtype;
    matches!(
        dtype,
        Dtype::BFloat16 | Dtype::Float16 | Dtype::Float32 | Dtype::Float64 |
        Dtype::Int8 | Dtype::Int16 | Dtype::Int32 | Dtype::Int64 |
        Dtype::Uint8 | Dtype::Uint16 | Dtype::Uint32 | Dtype::Uint64 |
        Dtype::Complex64 | Dtype::Complex128 | Dtype::Bool
    )
}

pub(crate) fn check_status(status: i32, context: &str) -> crate::client::Result<()> {
    let status = match status {
        0 => return Ok(()),
        1 => FXNStatus::ErrorInvalidArgument,
        2 => FXNStatus::ErrorInvalidOperation,
        3 => FXNStatus::ErrorNotImplemented,
        _ => return Err(MunaError::Native(format!("{context}: unknown status {status}"))),
    };
    Err(MunaError::Native(format!("{context}: {}", status.message())))
}
