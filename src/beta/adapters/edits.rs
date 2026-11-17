/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::collections::HashMap;

use async_trait::async_trait;

use crate::client::{MunaError, Result};
use crate::types::{Dtype, Parameter, Prediction, Signature, Value};

use super::images::{image_response, ImageResponse, OpenAIImageOptions};
use super::{
    utils::{get_parameter, parameter, require, validate_inputs, INTEGERS},
    Adapter,
};

/// Signature-resolved image inputs and response conversion.
#[derive(Debug, Clone)]
pub struct OpenAIImageEditAdapter {
    pub prompt: Parameter,
    pub images: Parameter,
    pub output: usize,
    pub count: Option<Parameter>,
    pub width: Option<Parameter>,
    pub height: Option<Parameter>,
}

impl OpenAIImageEditAdapter {
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
        // Image output.
        let (output, _) = require(
            get_parameter(&signature.outputs, &[Dtype::ImageList], None)?,
            "Requires one unambiguous image_list output",
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
impl Adapter for OpenAIImageEditAdapter {
    type Source = Prediction;
    type Options = OpenAIImageOptions;
    type Response = ImageResponse;

    fn kind(&self) -> &'static str {
        "openai.images.edit"
    }

    async fn adapt(
        &self,
        prediction: Prediction,
        options: Option<Self::Options>,
    ) -> Result<ImageResponse> {
        image_response(&prediction, self.output, options.unwrap_or_default()).await
    }
}

/// An encoded file, decoded pixels, or a wire value that local execution will decode.
#[derive(Debug, Clone)]
pub enum ImageInput {
    Encoded(Vec<u8>),
    Pixels(crate::types::Image),
    Remote(crate::types::RemoteValue),
}

impl From<Vec<u8>> for ImageInput {
    fn from(bytes: Vec<u8>) -> Self {
        Self::Encoded(bytes)
    }
}
impl From<crate::types::Image> for ImageInput {
    fn from(image: crate::types::Image) -> Self {
        Self::Pixels(image)
    }
}
impl From<crate::types::RemoteValue> for ImageInput {
    fn from(image: crate::types::RemoteValue) -> Self {
        Self::Remote(image)
    }
}

pub fn remote_image(image: ImageInput) -> Result<crate::types::RemoteValue> {
    use crate::types::{RemoteData, RemoteValue};
    use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
    let (bytes, mime) = match image {
        ImageInput::Remote(remote) => {
            if remote.dtype != Dtype::Image {
                return Err(MunaError::InvalidInput(
                    "Each image must have dtype image".into(),
                ));
            }
            return Ok(remote);
        }
        ImageInput::Encoded(bytes) if !bytes.is_empty() => (bytes, "image/*"),
        ImageInput::Encoded(_) => {
            return Err(MunaError::InvalidInput(
                "Encoded image must not be empty".into(),
            ))
        }
        ImageInput::Pixels(image) => (
            crate::c::Value::from_object(&Value::Image(image))?.serialize(Some("image/png"))?,
            "image/png",
        ),
    };
    Ok(RemoteValue {
        dtype: Dtype::Image,
        data: Some(RemoteData::Encoded(format!(
            "data:{mime};base64,{}",
            BASE64.encode(bytes)
        ))),
    })
}

pub fn image_list(images: Vec<ImageInput>) -> Result<Value> {
    if images.is_empty() {
        return Err(MunaError::InvalidInput(
            "images must be a nonempty list".into(),
        ));
    }
    if images
        .iter()
        .all(|image| matches!(image, ImageInput::Pixels(_)))
    {
        return Ok(Value::ImageList(
            images
                .into_iter()
                .filter_map(|image| match image {
                    ImageInput::Pixels(image) => Some(image),
                    _ => None,
                })
                .collect(),
        ));
    }
    let images = images
        .into_iter()
        .map(remote_image)
        .collect::<Result<Vec<_>>>()?;
    Ok(Value::Remote(crate::types::RemoteValue {
        dtype: Dtype::ImageList,
        data: Some(crate::types::RemoteData::Sequence(images)),
    }))
}
