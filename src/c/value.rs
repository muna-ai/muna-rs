/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::ffi::{c_void, CString};
use std::slice;

use crate::client::Result;
use crate::types::{self, Dtype};

use super::{check_status, ValueFlags};

/// Prediction value.
pub struct Value {
    handle: *mut c_void,
    owned: bool,
}

// SAFETY: The native handle is safe to send across threads.
unsafe impl Send for Value {}
unsafe impl Sync for Value {}

impl Value {

    pub(crate) fn from_raw(handle: *mut c_void, owned: bool) -> Self {
        Self { handle, owned }
    }

    pub(crate) fn raw_handle(&self) -> *mut c_void {
        self.handle
    }

    /// Get the value data type.
    pub fn dtype(&self) -> Result<Dtype> {
        let mut dtype: i32 = 0;
        let status = unsafe { super::FXNValueGetType(self.handle, &mut dtype) };
        check_status(status, "Failed to get value type")?;
        super::dtype_from_c(dtype)
            .ok_or_else(|| crate::client::MunaError::Native(format!("Unknown dtype: {dtype}")))
    }

    /// Get the value shape (for tensors and images).
    pub fn shape(&self) -> Result<Option<Vec<i32>>> {
        let dtype = self.dtype()?;
        if !super::is_tensor_dtype(dtype) &&
            dtype != Dtype::Image &&
            dtype != Dtype::Binary &&
            dtype != Dtype::ArrayList &&
            dtype != Dtype::ImageList {
            return Ok(None);
        }
        let mut dims: i32 = 0;
        let status = unsafe { super::FXNValueGetDimensions(self.handle, &mut dims) };
        check_status(status, "Failed to get value dimensions")?;
        let mut shape = vec![0i32; dims as usize];
        let status = unsafe { super::FXNValueGetShape(self.handle, shape.as_mut_ptr(), dims) };
        check_status(status, "Failed to get value shape")?;
        Ok(Some(shape))
    }

    /// Get the raw data pointer.
    pub fn data_ptr(&self) -> Result<*mut c_void> {
        let mut data = std::ptr::null_mut();
        let status = unsafe { super::FXNValueGetData(self.handle, &mut data) };
        check_status(status, "Failed to get value data")?;
        Ok(data)
    }

    /// Serialize the value to bytes.
    pub fn serialize(&self, mime: Option<&str>) -> Result<Vec<u8>> {
        let mime_c = mime
            .map(|m| CString::new(m).map_err(|e| crate::client::MunaError::Native(e.to_string())))
            .transpose()?;
        let mime_ptr = mime_c.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
        let mut serialized = std::ptr::null_mut();
        let status = unsafe { super::FXNValueCreateSerializedValue(self.handle, mime_ptr, &mut serialized) };
        check_status(status, "Failed to serialize value")?;
        let serialized_value = Value::from_raw(serialized, true);
        let data_ptr = serialized_value.data_ptr()?;
        let shape = serialized_value.shape()?;
        let byte_len = shape
            .and_then(|s| s.first().copied())
            .unwrap_or(0) as usize;
        let bytes = if byte_len > 0 && !data_ptr.is_null() {
            unsafe { slice::from_raw_parts(data_ptr as *const u8, byte_len) }.to_vec()
        } else {
            Vec::new()
        };
        Ok(bytes)
    }

    /// Convert the native value to a Rust `Value`.
    pub fn to_object(&self) -> Result<types::Value> {
        let dtype = self.dtype()?;
        let data_ptr = self.data_ptr()?;
        match dtype {
            Dtype::Null => Ok(types::Value::Null),
            Dtype::Float32 => self.read_tensor::<f32>(data_ptr, |v| types::TensorData::Float32(v)),
            Dtype::Float64 => self.read_tensor::<f64>(data_ptr, |v| types::TensorData::Float64(v)),
            Dtype::Int8    => self.read_tensor::<i8>(data_ptr, |v| types::TensorData::Int8(v)),
            Dtype::Int16   => self.read_tensor::<i16>(data_ptr, |v| types::TensorData::Int16(v)),
            Dtype::Int32   => self.read_tensor::<i32>(data_ptr, |v| types::TensorData::Int32(v)),
            Dtype::Int64   => self.read_tensor::<i64>(data_ptr, |v| types::TensorData::Int64(v)),
            Dtype::Uint8   => self.read_tensor::<u8>(data_ptr, |v| types::TensorData::Uint8(v)),
            Dtype::Uint16  => self.read_tensor::<u16>(data_ptr, |v| types::TensorData::Uint16(v)),
            Dtype::Uint32  => self.read_tensor::<u32>(data_ptr, |v| types::TensorData::Uint32(v)),
            Dtype::Uint64  => self.read_tensor::<u64>(data_ptr, |v| types::TensorData::Uint64(v)),
            Dtype::Bool       => self.read_bool_tensor(data_ptr),
            Dtype::Complex64  => self.read_tensor::<[f32; 2]>(data_ptr, |v| types::TensorData::Complex64(v)),
            Dtype::Complex128 => self.read_tensor::<[f64; 2]>(data_ptr, |v| types::TensorData::Complex128(v)),
            Dtype::String => {
                let s = unsafe { std::ffi::CStr::from_ptr(data_ptr as *const _) }
                    .to_string_lossy()
                    .into_owned();
                Ok(types::Value::String(s))
            }
            Dtype::List => {
                let s = unsafe { std::ffi::CStr::from_ptr(data_ptr as *const _) }
                    .to_string_lossy()
                    .into_owned();
                let v: Vec<serde_json::Value> = serde_json::from_str(&s)
                    .map_err(|e| crate::client::MunaError::Native(e.to_string()))?;
                Ok(types::Value::List(v))
            }
            Dtype::Dict => {
                let s = unsafe { std::ffi::CStr::from_ptr(data_ptr as *const _) }
                    .to_string_lossy()
                    .into_owned();
                let v: serde_json::Map<String, serde_json::Value> = serde_json::from_str(&s)
                    .map_err(|e| crate::client::MunaError::Native(e.to_string()))?;
                Ok(types::Value::Dict(v))
            }
            Dtype::Image => {
                let shape = self.shape()?.unwrap_or_default();
                if shape.len() < 2 {
                    return Err(crate::client::MunaError::Native("Invalid image shape".into()));
                }
                let height = shape[0] as u32;
                let width = shape[1] as u32;
                let channels = if shape.len() > 2 { shape[2] as u32 } else { 1 };
                let byte_len = (height * width * channels) as usize;
                let data = unsafe { slice::from_raw_parts(data_ptr as *const u8, byte_len) }.to_vec();
                Ok(types::Value::Image(types::Image { data, width, height, channels }))
            }
            Dtype::Binary => {
                let shape = self.shape()?.unwrap_or_default();
                let byte_len = shape.first().copied().unwrap_or(0) as usize;
                let data = unsafe { slice::from_raw_parts(data_ptr as *const u8, byte_len) }.to_vec();
                Ok(types::Value::Binary(data))
            }
            Dtype::ImageList | Dtype::ArrayList => {
                let shape = self.shape()?.unwrap_or_default();
                let count = shape.first().copied().unwrap_or(0) as usize;
                let elements = unsafe { slice::from_raw_parts(data_ptr as *const *mut c_void, count) };
                let mut values = Vec::with_capacity(count);
                for &element_ptr in elements {
                    let element = Value::from_raw(element_ptr, false);
                    values.push(element.to_object()?);
                }
                if dtype == Dtype::ImageList {
                    let images: std::result::Result<Vec<_>, _> = values.into_iter().map(|v| match v {
                        types::Value::Image(img) => Ok(img),
                        _ => Err(crate::client::MunaError::Native("Expected image in image list".into())),
                    }).collect();
                    Ok(types::Value::ImageList(images?))
                } else {
                    let tensors: std::result::Result<Vec<_>, _> = values.into_iter().map(|v| match v {
                        types::Value::Tensor(t) => Ok(t),
                        _ => Err(crate::client::MunaError::Native("Expected tensor in array list".into())),
                    }).collect();
                    Ok(types::Value::ArrayList(tensors?))
                }
            }
            _ => Err(crate::client::MunaError::Native(format!(
                "Cannot convert value with type `{dtype:?}` to object"
            ))),
        }
    }

    /// Create a native value from a Rust `Value`.
    pub fn from_object(obj: &types::Value) -> Result<Self> {
        match obj {
            types::Value::Null => Self::create_null(),
            types::Value::Int(v) => {
                let data = [*v];
                Self::create_array(
                    data.as_ptr() as *const c_void,
                    &[],
                    0,
                    super::dtype_to_c(Dtype::Int32),
                    ValueFlags::CopyData as i32,
                )
            }
            types::Value::Float(v) => {
                let data = [*v];
                Self::create_array(
                    data.as_ptr() as *const c_void,
                    &[],
                    0,
                    super::dtype_to_c(Dtype::Float32),
                    ValueFlags::CopyData as i32,
                )
            }
            types::Value::Double(v) => {
                let data = [*v as f32];
                Self::create_array(
                    data.as_ptr() as *const c_void,
                    &[],
                    0,
                    super::dtype_to_c(Dtype::Float32),
                    ValueFlags::CopyData as i32,
                )
            }
            types::Value::Long(v) => {
                let data = [*v];
                Self::create_array(
                    data.as_ptr() as *const c_void,
                    &[],
                    0,
                    super::dtype_to_c(Dtype::Int64),
                    ValueFlags::CopyData as i32,
                )
            }
            types::Value::Bool(v) => {
                let data: [u8; 1] = [*v as u8];
                Self::create_array(
                    data.as_ptr() as *const c_void,
                    &[],
                    0,
                    super::dtype_to_c(Dtype::Bool),
                    ValueFlags::CopyData as i32,
                )
            }
            types::Value::String(v) => Self::create_string(v),
            types::Value::List(v) => {
                let json = serde_json::to_string(v)
                    .map_err(|e| crate::client::MunaError::Native(e.to_string()))?;
                Self::create_list(&json)
            }
            types::Value::Dict(v) => {
                let json = serde_json::to_string(v)
                    .map_err(|e| crate::client::MunaError::Native(e.to_string()))?;
                Self::create_dict(&json)
            }
            types::Value::Tensor(tensor) => {
                let dtype = super::dtype_to_c(tensor.data.dtype());
                let shape: Vec<i32> = tensor.shape.clone();
                let dims = shape.len() as i32;
                Self::create_array(
                    tensor.data.as_ptr() as *const c_void,
                    &shape,
                    dims,
                    dtype,
                    ValueFlags::None as i32,
                )
            }
            types::Value::Image(image) => Self::create_image(image),
            types::Value::Binary(data) => Self::create_binary(data),
            types::Value::ImageList(images) => Self::create_image_list(images),
            types::Value::ArrayList(tensors) => Self::create_array_list(tensors),
        }
    }

    /// Create a value from serialized bytes.
    pub fn from_bytes(data: &[u8], mime: &str) -> Result<Self> {
        let binary = Self::create_binary(data)?;
        let mime_c = CString::new(mime).map_err(|e| crate::client::MunaError::Native(e.to_string()))?;
        let mut handle = std::ptr::null_mut();
        let status = unsafe {
            super::FXNValueCreateFromSerializedValue(binary.handle, mime_c.as_ptr(), &mut handle)
        };
        check_status(status, "Failed to deserialize value")?;
        Ok(Self::from_raw(handle, true))
    }

    fn create_null() -> Result<Self> {
        let mut handle = std::ptr::null_mut();
        let status = unsafe { super::FXNValueCreateNull(&mut handle) };
        check_status(status, "Failed to create null value")?;
        Ok(Self::from_raw(handle, true))
    }

    fn create_array(
        data: *const c_void,
        shape: &[i32],
        dims: i32,
        dtype: i32,
        flags: i32,
    ) -> Result<Self> {
        let shape_ptr = if shape.is_empty() { std::ptr::null() } else { shape.as_ptr() };
        let mut handle = std::ptr::null_mut();
        let status = unsafe {
            super::FXNValueCreateArray(data, shape_ptr, dims, dtype, flags, &mut handle)
        };
        check_status(status, "Failed to create array value")?;
        Ok(Self::from_raw(handle, true))
    }

    fn create_string(data: &str) -> Result<Self> {
        let cstr = CString::new(data).map_err(|e| crate::client::MunaError::Native(e.to_string()))?;
        let mut handle = std::ptr::null_mut();
        let status = unsafe { super::FXNValueCreateString(cstr.as_ptr(), &mut handle) };
        check_status(status, "Failed to create string value")?;
        Ok(Self::from_raw(handle, true))
    }

    fn create_list(json: &str) -> Result<Self> {
        let cstr = CString::new(json).map_err(|e| crate::client::MunaError::Native(e.to_string()))?;
        let mut handle = std::ptr::null_mut();
        let status = unsafe { super::FXNValueCreateList(cstr.as_ptr(), &mut handle) };
        check_status(status, "Failed to create list value")?;
        Ok(Self::from_raw(handle, true))
    }

    fn create_dict(json: &str) -> Result<Self> {
        let cstr = CString::new(json).map_err(|e| crate::client::MunaError::Native(e.to_string()))?;
        let mut handle = std::ptr::null_mut();
        let status = unsafe { super::FXNValueCreateDict(cstr.as_ptr(), &mut handle) };
        check_status(status, "Failed to create dict value")?;
        Ok(Self::from_raw(handle, true))
    }

    fn create_image(image: &types::Image) -> Result<Self> {
        let mut handle = std::ptr::null_mut();
        let status = unsafe {
            super::FXNValueCreateImage(
                image.data.as_ptr() as *const c_void,
                image.width as i32,
                image.height as i32,
                image.channels as i32,
                ValueFlags::CopyData as i32,
                &mut handle,
            )
        };
        check_status(status, "Failed to create image value")?;
        Ok(Self::from_raw(handle, true))
    }

    fn create_image_list(images: &[types::Image]) -> Result<Self> {
        let pixel_buffers: Vec<*const c_void> = images.iter().map(|img| img.data.as_ptr() as *const c_void).collect();
        let widths: Vec<i32> = images.iter().map(|img| img.width as i32).collect();
        let heights: Vec<i32> = images.iter().map(|img| img.height as i32).collect();
        let channels: Vec<i32> = images.iter().map(|img| img.channels as i32).collect();
        let mut handle = std::ptr::null_mut();
        let status = unsafe {
            super::FXNValueCreateImageList(
                pixel_buffers.as_ptr(),
                widths.as_ptr(),
                heights.as_ptr(),
                channels.as_ptr(),
                images.len() as i32,
                ValueFlags::CopyData as i32,
                &mut handle,
            )
        };
        check_status(status, "Failed to create image list value")?;
        Ok(Self::from_raw(handle, true))
    }

    fn create_array_list(tensors: &[types::Tensor]) -> Result<Self> {
        let data_ptrs: Vec<*const c_void> = tensors.iter().map(|t| t.data.as_ptr() as *const c_void).collect();
        let shapes: Vec<Vec<i32>> = tensors.iter().map(|t| t.shape.clone()).collect();
        let shape_ptrs: Vec<*const i32> = shapes.iter().map(|s| s.as_ptr()).collect();
        let dims: Vec<i32> = tensors.iter().map(|t| t.shape.len() as i32).collect();
        let dtypes: Vec<i32> = tensors.iter().map(|t| super::dtype_to_c(t.data.dtype())).collect();
        let mut handle = std::ptr::null_mut();
        let status = unsafe {
            super::FXNValueCreateArrayList(
                data_ptrs.as_ptr(),
                shape_ptrs.as_ptr(),
                dims.as_ptr(),
                dtypes.as_ptr(),
                tensors.len() as i32,
                ValueFlags::CopyData as i32,
                &mut handle,
            )
        };
        check_status(status, "Failed to create array list value")?;
        Ok(Self::from_raw(handle, true))
    }

    fn create_binary(data: &[u8]) -> Result<Self> {
        let mut handle = std::ptr::null_mut();
        let status = unsafe {
            super::FXNValueCreateBinary(
                data.as_ptr() as *const _,
                data.len() as i32,
                ValueFlags::CopyData as i32,
                &mut handle,
            )
        };
        check_status(status, "Failed to create binary value")?;
        Ok(Self::from_raw(handle, true))
    }

    fn read_tensor<T: Clone>(
        &self,
        data_ptr: *mut c_void,
        wrap: impl FnOnce(Vec<T>) -> types::TensorData,
    ) -> Result<types::Value> {
        let shape = self.shape()?.unwrap_or_default();
        let elem_count: usize = if shape.is_empty() {
            1
        } else {
            shape.iter().map(|&s| s as usize).product()
        };
        let data = unsafe { slice::from_raw_parts(data_ptr as *const T, elem_count) }.to_vec();
        if shape.is_empty() {
            return Ok(scalar_from_tensor_data(&wrap(data)));
        }
        Ok(types::Value::Tensor(types::Tensor { data: wrap(data), shape }))
    }

    fn read_bool_tensor(&self, data_ptr: *mut c_void) -> Result<types::Value> {
        let shape = self.shape()?.unwrap_or_default();
        let elem_count: usize = if shape.is_empty() {
            1
        } else {
            shape.iter().map(|&s| s as usize).product()
        };
        let raw = unsafe { slice::from_raw_parts(data_ptr as *const u8, elem_count) };
        let data: Vec<bool> = raw.iter().map(|&v| v != 0).collect();
        if shape.is_empty() {
            return Ok(types::Value::Bool(data[0]));
        }
        Ok(types::Value::Tensor(types::Tensor {
            data: types::TensorData::Bool(data),
            shape,
        }))
    }
}

impl Drop for Value {
    fn drop(&mut self) {
        if !self.handle.is_null() && self.owned {
            unsafe { super::FXNValueRelease(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

fn scalar_from_tensor_data(data: &types::TensorData) -> types::Value {
    match data {
        types::TensorData::Float32(v) => types::Value::Float(v[0]),
        types::TensorData::Float64(v) => types::Value::Double(v[0]),
        types::TensorData::Int32(v)   => types::Value::Int(v[0]),
        types::TensorData::Int64(v)   => types::Value::Long(v[0]),
        types::TensorData::Bool(v)    => types::Value::Bool(v[0]),
        _ => types::Value::Tensor(types::Tensor {
            data: data.clone(),
            shape: vec![],
        }),
    }
}
