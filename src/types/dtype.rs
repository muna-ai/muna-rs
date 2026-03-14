/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use serde::{Deserialize, Serialize};

/// Value data type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Dtype {
    #[serde(rename = "null")]       Null,
    #[serde(rename = "bfloat16")]   BFloat16,
    #[serde(rename = "float16")]    Float16,
    #[serde(rename = "float32")]    Float32,
    #[serde(rename = "float64")]    Float64,
    #[serde(rename = "int8")]       Int8,
    #[serde(rename = "int16")]      Int16,
    #[serde(rename = "int32")]      Int32,
    #[serde(rename = "int64")]      Int64,
    #[serde(rename = "uint8")]      Uint8,
    #[serde(rename = "uint16")]     Uint16,
    #[serde(rename = "uint32")]     Uint32,
    #[serde(rename = "uint64")]     Uint64,
    #[serde(rename = "bool")]       Bool,
    #[serde(rename = "string")]     String,
    #[serde(rename = "list")]       List,
    #[serde(rename = "dict")]       Dict,
    #[serde(rename = "image")]      Image,
    #[serde(rename = "image_list")] ImageList,
    #[serde(rename = "binary")]     Binary,
}
