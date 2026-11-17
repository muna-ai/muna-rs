/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::collections::HashMap;

use async_trait::async_trait;

use crate::client::{MunaError, Result};
use crate::types::{Dtype, Parameter, Prediction, Signature, Value};

use super::edits::image_list;
pub use super::edits::ImageInput;
use super::images::ImageResponse;
use super::{
    utils::{
        created, decode, get_parameter, inline_base64, parameter,
        require, result, validate_inputs, INTEGERS,
    },
    Adapter,
};

/// Signature-resolved depth inputs and response conversion.
#[derive(Debug, Clone)]
pub struct OpenAIDepthAdapter {
    pub prompt: Parameter,
    pub images: Parameter,
    pub output: usize,
    pub count: Option<Parameter>,
    pub width: Option<Parameter>,
    pub height: Option<Parameter>,
}

impl OpenAIDepthAdapter {
    pub fn from_signature(signature: &Signature) -> Result<Self> {
        // Prompt parameter.
        let prompt = require(
            parameter(
                &signature.inputs,
                &[Dtype::String],
                Some("openai.images.prompt"),
            )?,
            "Requires one scalar string input annotated with ImagePrompt",
        )?;
        // Images parameter. An unannotated image input is not a match.
        let images = require(
            parameter(
                &signature.inputs,
                &[Dtype::ImageList],
                Some("openai.chat.completions.images"),
            )?,
            "Requires one image_list input annotated with ChatImages",
        )?;
        // Optional generation width.
        let width = parameter(&signature.inputs, INTEGERS, Some("openai.images.width"))?;
        if width
            .as_ref()
            .is_some_and(|parameter| parameter.optional != Some(true))
        {
            return Err(MunaError::InvalidInput(
                "Image `width` input must be optional".into(),
            ));
        }
        // Optional generation height.
        let height = parameter(&signature.inputs, INTEGERS, Some("openai.images.height"))?;
        if height
            .as_ref()
            .is_some_and(|parameter| parameter.optional != Some(true))
        {
            return Err(MunaError::InvalidInput(
                "Image `height` input must be optional".into(),
            ));
        }
        // Optional generation count.
        let count = parameter(&signature.inputs, INTEGERS, Some("openai.images.count"))?;
        if count
            .as_ref()
            .is_some_and(|parameter| parameter.optional != Some(true))
        {
            return Err(MunaError::InvalidInput(
                "Image `count` input must be optional".into(),
            ));
        }
        validate_inputs(
            signature,
            &[
                Some(&prompt),
                Some(&images),
                width.as_ref(),
                height.as_ref(),
                count.as_ref(),
            ],
        )?;
        // Depth map output.
        let (output, _) = require(
            get_parameter(&signature.outputs, &[Dtype::ArrayList], Some("depth_map"))?,
            "Requires one unambiguous depth_map array_list output",
        )?;
        Ok(Self {
            prompt,
            images,
            output,
            count,
            width,
            height,
        })
    }

    pub fn inputs(
        &self,
        images: Vec<ImageInput>,
        prompt: Option<String>,
        n: Option<i32>,
        width: Option<i32>,
        height: Option<i32>,
    ) -> Result<HashMap<String, Value>> {
        let mut inputs = HashMap::new();
        inputs.insert(self.images.name.clone(), image_list(images)?);
        if let Some(prompt) = prompt {
            inputs.insert(self.prompt.name.clone(), Value::String(prompt));
        } else if self.prompt.optional != Some(true) {
            return Err(MunaError::InvalidInput(
                "Prompt is required by this predictor".into(),
            ));
        }
        if let (Some(parameter), Some(value)) = (&self.count, n) {
            inputs.insert(parameter.name.clone(), Value::Int(value));
        }
        if let (Some(parameter), Some(value)) = (&self.width, width) {
            inputs.insert(parameter.name.clone(), Value::Int(value));
        }
        if let (Some(parameter), Some(value)) = (&self.height, height) {
            inputs.insert(parameter.name.clone(), Value::Int(value));
        }
        Ok(inputs)
    }
}

#[async_trait]
impl Adapter for OpenAIDepthAdapter {
    type Source = Prediction;
    type Options = OpenAIDepthOptions;
    type Response = ImageResponse;

    fn kind(&self) -> &'static str {
        "openai.images.edit.depth"
    }

    async fn adapt(
        &self,
        prediction: Prediction,
        options: Option<Self::Options>,
    ) -> Result<ImageResponse> {
        depth_response(&prediction, self.output, options.unwrap_or_default()).await
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DepthFormat {
    Npy,
    #[default]
    Npz,
}

#[derive(Debug, Clone, Default)]
pub struct OpenAIDepthOptions {
    pub format: DepthFormat,
}

async fn depth_response(
    prediction: &Prediction,
    output: usize,
    options: OpenAIDepthOptions,
) -> Result<ImageResponse> {
    use super::images::ImageData;
    use crate::types::{RemoteData, TensorData};
    use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
    let value = result(prediction, output)?;
    let maps = match value {
        Value::Remote(remote) if remote.dtype == Dtype::ArrayList => match &remote.data {
            Some(RemoteData::Sequence(maps)) => maps.iter().cloned().map(Value::Remote).collect(),
            _ => match decode(value, Dtype::ArrayList).await? {
                Value::ArrayList(maps) => maps.into_iter().map(Value::Tensor).collect(),
                _ => {
                    return Err(MunaError::Prediction(
                        "Expected array_list depth output".into(),
                    ))
                }
            },
        },
        Value::ArrayList(maps) => maps.iter().cloned().map(Value::Tensor).collect::<Vec<_>>(),
        _ => {
            return Err(MunaError::Prediction(
                "Expected array_list depth output".into(),
            ))
        }
    };
    if maps.is_empty() {
        return Err(MunaError::Prediction(
            "Depth output must contain maps".into(),
        ));
    }
    let mut data = Vec::with_capacity(maps.len());
    for map in maps {
        let dtype = match &map {
            Value::Remote(remote) => remote.dtype,
            Value::Tensor(map) => map.data.dtype(),
            _ => return Err(MunaError::Prediction("Expected a depth tensor".into())),
        };
        if !matches!(dtype, Dtype::Float32 | Dtype::Float64) {
            return Err(MunaError::Prediction(
                "Depth maps must be float32 or float64".into(),
            ));
        }
        if let Value::Remote(remote) = &map {
            if options.format == DepthFormat::Npz {
                if let Some(payload) = inline_base64(
                    remote,
                    &["application/octet-stream", "application/vnd.muna.tensor"],
                ) {
                    data.push(ImageData {
                        b64_json: Some(payload.into()),
                        image: None,
                    });
                    continue;
                }
            }
        }
        let Value::Tensor(map) = decode(&map, dtype).await? else {
            return Err(MunaError::Prediction("Expected a depth tensor".into()));
        };
        let [height, width] = map.shape.as_slice() else {
            return Err(MunaError::Prediction(
                "Depth map must have two dimensions".into(),
            ));
        };
        if *height <= 0
            || *width <= 0
            || (*height as usize).checked_mul(*width as usize) != Some(map.data.len())
        {
            return Err(MunaError::Prediction("Depth map shape is invalid".into()));
        }
        let bytes = match options.format {
            DepthFormat::Npz => crate::c::Value::from_object(&Value::Tensor(map))?
                .serialize(Some("application/vnd.muna.tensor"))?,
            DepthFormat::Npy => {
                // NPY v1: a little-endian, C-order array with a padded ASCII header.
                let (descriptor, values) = match &map.data {
                    TensorData::Float32(values) => (
                        "<f4",
                        values
                            .iter()
                            .flat_map(|value| value.to_le_bytes())
                            .collect::<Vec<_>>(),
                    ),
                    TensorData::Float64(values) => (
                        "<f8",
                        values
                            .iter()
                            .flat_map(|value| value.to_le_bytes())
                            .collect::<Vec<_>>(),
                    ),
                    _ => {
                        return Err(MunaError::Prediction(
                            "Depth maps must be float32 or float64".into(),
                        ))
                    }
                };
                let mut header = format!("{{'descr': '{descriptor}', 'fortran_order': False, 'shape': ({height}, {width}), }}");
                header.push_str(&" ".repeat((64 - (10 + header.len() + 1) % 64) % 64));
                header.push('\n');
                let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
                bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
                bytes.extend_from_slice(header.as_bytes());
                bytes.extend_from_slice(&values);
                bytes
            }
        };
        data.push(ImageData {
            b64_json: Some(BASE64.encode(bytes)),
            image: None,
        });
    }
    Ok(ImageResponse {
        data: Some(data),
        background: None,
        created: created(prediction)?,
        usage: None,
    })
}
