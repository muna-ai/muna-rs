/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use super::Dtype;

/// Tensor data buffer.
#[derive(Debug, Clone)]
pub enum TensorData {
    Float32(Vec<f32>),
    Float64(Vec<f64>),
    Int8(Vec<i8>),
    Int16(Vec<i16>),
    Int32(Vec<i32>),
    Int64(Vec<i64>),
    Uint8(Vec<u8>),
    Uint16(Vec<u16>),
    Uint32(Vec<u32>),
    Uint64(Vec<u64>),
    Bool(Vec<bool>),
}

impl TensorData {
    pub fn dtype(&self) -> Dtype {
        match self {
            Self::Float32(_) => Dtype::Float32,
            Self::Float64(_) => Dtype::Float64,
            Self::Int8(_)    => Dtype::Int8,
            Self::Int16(_)   => Dtype::Int16,
            Self::Int32(_)   => Dtype::Int32,
            Self::Int64(_)   => Dtype::Int64,
            Self::Uint8(_)   => Dtype::Uint8,
            Self::Uint16(_)  => Dtype::Uint16,
            Self::Uint32(_)  => Dtype::Uint32,
            Self::Uint64(_)  => Dtype::Uint64,
            Self::Bool(_)    => Dtype::Bool,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Float32(v) => v.len(),
            Self::Float64(v) => v.len(),
            Self::Int8(v)    => v.len(),
            Self::Int16(v)   => v.len(),
            Self::Int32(v)   => v.len(),
            Self::Int64(v)   => v.len(),
            Self::Uint8(v)   => v.len(),
            Self::Uint16(v)  => v.len(),
            Self::Uint32(v)  => v.len(),
            Self::Uint64(v)  => v.len(),
            Self::Bool(v)    => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_ptr(&self) -> *const u8 {
        match self {
            Self::Float32(v) => v.as_ptr() as *const u8,
            Self::Float64(v) => v.as_ptr() as *const u8,
            Self::Int8(v)    => v.as_ptr() as *const u8,
            Self::Int16(v)   => v.as_ptr() as *const u8,
            Self::Int32(v)   => v.as_ptr() as *const u8,
            Self::Int64(v)   => v.as_ptr() as *const u8,
            Self::Uint8(v)   => v.as_ptr() as *const u8,
            Self::Uint16(v)  => v.as_ptr() as *const u8,
            Self::Uint32(v)  => v.as_ptr() as *const u8,
            Self::Uint64(v)  => v.as_ptr() as *const u8,
            Self::Bool(v)    => v.as_ptr() as *const u8,
        }
    }
}

/// Tensor value.
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Tensor data.
    pub data: TensorData,
    /// Tensor shape.
    pub shape: Vec<i32>,
}

/// Image value.
#[derive(Debug, Clone)]
pub struct Image {
    /// Pixel buffer (RGBA or RGB).
    pub data: Vec<u8>,
    /// Image width.
    pub width: u32,
    /// Image height.
    pub height: u32,
    /// Image channels.
    pub channels: u32,
}

/// Prediction value.
#[derive(Debug, Clone)]
pub enum Value {
    Null,
    Float(f32),
    Double(f64),
    Int(i32),
    Long(i64),
    Bool(bool),
    String(String),
    List(Vec<serde_json::Value>),
    Dict(serde_json::Map<String, serde_json::Value>),
    Tensor(Tensor),
    Image(Image),
    Binary(Vec<u8>),
}

impl From<f32> for Value {
    fn from(v: f32) -> Self { Value::Float(v) }
}
impl From<f64> for Value {
    fn from(v: f64) -> Self { Value::Double(v) }
}
impl From<i32> for Value {
    fn from(v: i32) -> Self { Value::Int(v) }
}
impl From<i64> for Value {
    fn from(v: i64) -> Self { Value::Long(v) }
}
impl From<bool> for Value {
    fn from(v: bool) -> Self { Value::Bool(v) }
}
impl From<String> for Value {
    fn from(v: String) -> Self { Value::String(v) }
}
impl From<&str> for Value {
    fn from(v: &str) -> Self { Value::String(v.to_string()) }
}
