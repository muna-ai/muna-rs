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
        created, decode, get_parameter, inline_base64, parameter, result, validate_inputs, INTEGERS,
    },
    Adapter,
};

/// Signature-resolved 3D inputs and response conversion.
#[derive(Debug, Clone)]
pub struct OpenAIModel3DAdapter {
    pub prompt: Parameter,
    pub images: Parameter,
    pub output: usize,
    pub count: Option<Parameter>,
    pub width: Option<Parameter>,
    pub height: Option<Parameter>,
}

impl OpenAIModel3DAdapter {
    /// Returns `None` when the signature is incompatible or ambiguous.
    pub fn from_signature(signature: &Signature) -> Option<Self> {
        // Prompt parameter.
        let prompt = parameter(
            &signature.inputs,
            &[Dtype::String],
            Some("openai.images.prompt"),
        )
        .ok()??;
        // Images parameter. An unannotated image input is not a match.
        let images = parameter(
            &signature.inputs,
            &[Dtype::ImageList],
            Some("openai.chat.completions.images"),
        )
        .ok()??;
        // Optional generation width.
        let width = parameter(&signature.inputs, INTEGERS, Some("openai.images.width")).ok()?;
        if width
            .as_ref()
            .is_some_and(|parameter| parameter.optional != Some(true))
        {
            return None;
        }
        // Optional generation height.
        let height = parameter(&signature.inputs, INTEGERS, Some("openai.images.height")).ok()?;
        if height
            .as_ref()
            .is_some_and(|parameter| parameter.optional != Some(true))
        {
            return None;
        }
        // Optional generation count.
        let count = parameter(&signature.inputs, INTEGERS, Some("openai.images.count")).ok()?;
        if count
            .as_ref()
            .is_some_and(|parameter| parameter.optional != Some(true))
        {
            return None;
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
        )
        .ok()?;
        // 3D asset output.
        let (output, _) =
            get_parameter(&signature.outputs, &[Dtype::Binary], Some("3d")).ok()??;
        Some(Self {
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
impl Adapter for OpenAIModel3DAdapter {
    type Source = Prediction;
    type Options = OpenAIModel3DOptions;
    type Response = ImageResponse;

    fn kind(&self) -> &'static str {
        "openai.images.edit.model3d"
    }

    async fn adapt(
        &self,
        prediction: Prediction,
        options: Option<Self::Options>,
    ) -> Result<ImageResponse> {
        model3d_response(&prediction, self.output, options.unwrap_or_default()).await
    }
}

/// Response format only; the predictor documents how its 3D asset was encoded.
#[derive(Debug, Clone)]
pub struct OpenAIModel3DOptions {
    pub format: String,
}

impl Default for OpenAIModel3DOptions {
    fn default() -> Self {
        Self {
            format: "ply".into(),
        }
    }
}

impl OpenAIModel3DOptions {
    pub fn validate(&self) -> Result<()> {
        if self.format != "ply" {
            return Err(MunaError::InvalidInput(
                "3D output requires output_format=ply.".into(),
            ));
        }
        Ok(())
    }
}

async fn model3d_response(
    prediction: &Prediction,
    output: usize,
    options: OpenAIModel3DOptions,
) -> Result<ImageResponse> {
    use super::images::ImageData;
    use base64::{engine::general_purpose::STANDARD as BASE64, Engine};

    options.validate()?;
    let value = result(prediction, output)?;
    let inline = if let Value::Remote(remote) = value {
        if remote.dtype != Dtype::Binary {
            return Err(MunaError::Prediction("Expected a binary 3D output".into()));
        }
        inline_base64(
            remote,
            &["application/octet-stream", "application/ply", "model/ply"],
        )
    } else {
        None
    };
    let (prefix, payload) = if let Some(payload) = inline {
        // Inspect the magic without decoding or re-encoding the large file body.
        let prefix = BASE64
            .decode(&payload.as_bytes()[..payload.len().min(8)])
            .map_err(|_| MunaError::Prediction("Expected a PLY output".into()))?;
        (prefix, payload.to_owned())
    } else {
        let Value::Binary(bytes) = decode(value, Dtype::Binary).await? else {
            return Err(MunaError::Prediction("Expected a binary 3D output".into()));
        };
        (bytes[..bytes.len().min(6)].to_vec(), BASE64.encode(bytes))
    };
    if !prefix.starts_with(b"ply\n") && !prefix.starts_with(b"ply\r\n") {
        return Err(MunaError::Prediction(
            "Expected a PLY output; the adapter does not convert 3D formats".into(),
        ));
    }
    Ok(ImageResponse {
        data: Some(vec![ImageData {
            b64_json: Some(payload),
            image: None,
        }]),
        background: None,
        created: created(prediction)?,
        usage: None,
    })
}
