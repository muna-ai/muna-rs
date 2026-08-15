/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::collections::HashMap;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;

use crate::c;
use crate::client::{MunaError, Result};
use crate::services::{PredictionService, PredictorService};
use crate::types::{Acceleration, Value};

use super::schema::{ImageData, ImageResponse};

/// Image size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageSize {
    Auto,
    Size256x256,
    Size512x512,
    Size1024x1024,
    Size1536x1024,
    Size1024x1536,
    Size1792x1024,
    Size1024x1792,
}

impl ImageSize {
    /// Requested `(width, height)`; `Auto` requests neither.
    pub fn dimensions(&self) -> (Option<i32>, Option<i32>) {
        match self {
            Self::Auto => (None, None),
            Self::Size256x256 => (Some(256), Some(256)),
            Self::Size512x512 => (Some(512), Some(512)),
            Self::Size1024x1024 => (Some(1024), Some(1024)),
            Self::Size1536x1024 => (Some(1536), Some(1024)),
            Self::Size1024x1536 => (Some(1024), Some(1536)),
            Self::Size1792x1024 => (Some(1792), Some(1024)),
            Self::Size1024x1792 => (Some(1024), Some(1792)),
        }
    }
}

/// Image creation parameters.
pub struct ImageCreateParams {
    /// Text prompt.
    pub prompt: String,
    /// Image predictor tag.
    pub model: String,
    /// Background transparency.
    pub background: Option<String>,
    /// Number of images to generate.
    pub n: Option<i32>,
    /// Output format (`png`, `jpeg`, `webp`, `raw`).
    pub output_format: Option<String>,
    /// Compression level (0-100).
    pub output_compression: Option<i32>,
    /// Image size.
    pub size: Option<ImageSize>,
    /// Prediction acceleration.
    pub acceleration: Option<Acceleration>,
}

struct ImageConfig {
    prompt_param_name: String,
    width_param: Option<String>,
    height_param: Option<String>,
    count_param: Option<String>,
    image_param_idx: usize,
}

/// Image generation service.
#[derive(Clone)]
pub struct ImageService {
    predictors: PredictorService,
    predictions: PredictionService,
    cache: std::sync::Arc<tokio::sync::Mutex<HashMap<String, ImageConfig>>>,
}

impl ImageService {

    pub fn new(
        predictors: PredictorService,
        predictions: PredictionService,
    ) -> Self {
        Self {
            predictors,
            predictions,
            cache: std::sync::Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        }
    }

    /// Generate images from a text prompt.
    pub async fn generate(&self, params: ImageCreateParams) -> Result<ImageResponse> {
        let tag = params.model.clone();
        let output_format = params.output_format.as_deref().unwrap_or("png");
        let acceleration = params.acceleration.unwrap_or(Acceleration::LocalAuto);
        {
            let mut cache = self.cache.lock().await;
            if !cache.contains_key(&tag) {
                let config = self.create_config(&tag).await?;
                cache.insert(tag.clone(), config);
            }
        }
        let cache = self.cache.lock().await;
        let config = cache.get(&tag).unwrap();
        let (req_width, req_height) = params.size.unwrap_or(ImageSize::Auto).dimensions();
        let mut prediction_inputs: HashMap<String, Value> = HashMap::new();
        // The prompt parameter is list-typed so prompts from multiple requests
        // can be batched into one invocation; a single generation is just a
        // batch of one.
        prediction_inputs.insert(
            config.prompt_param_name.clone(),
            Value::List(vec![serde_json::Value::String(params.prompt)])
        );
        if let (Some(n), Some(name)) = (&params.n, &config.count_param) {
            prediction_inputs.insert(name.clone(), Value::Int(*n));
        }
        if let (Some(w), Some(name)) = (req_width, &config.width_param) {
            prediction_inputs.insert(name.clone(), Value::Int(w));
        }
        if let (Some(h), Some(name)) = (req_height, &config.height_param) {
            prediction_inputs.insert(name.clone(), Value::Int(h));
        }
        let image_idx = config.image_param_idx;
        drop(cache);

        let prediction = self.create_prediction(&tag, prediction_inputs, acceleration).await?;
        if let Some(error) = &prediction.error {
            return Err(MunaError::Prediction(error.clone()));
        }
        let results = prediction.results.as_ref().ok_or_else(|| {
            MunaError::Prediction("No results returned".into())
        })?;
        let images_value = results.get(image_idx).ok_or_else(|| {
            MunaError::Prediction(format!("{tag} did not return images"))
        })?;
        // `create_config` guarantees the output at `image_idx` is an image list.
        let images = match images_value {
            Value::ImageList(list) => {
                let mut image_data = Vec::new();
                for img in list {
                    image_data.push(Self::encode_image(img, output_format)?);
                }
                image_data
            }
            _ => return Err(MunaError::Prediction(format!(
                "{tag} returned unexpected type instead of images"
            ))),
        };
        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        Ok(ImageResponse {
            data: Some(images),
            background: Some("opaque".to_string()),
            created,
            usage: None,
        })
    }

    async fn create_config(&self, tag: &str) -> Result<ImageConfig> {
        let predictor = self.predictors.retrieve(tag).await?.ok_or_else(|| {
            MunaError::Prediction(format!("{tag} cannot be used with image API: predictor not found"))
        })?;
        let signature = &predictor.signature;
        let required: Vec<_> = signature.inputs.iter()
            .filter(|p| p.optional != Some(true))
            .collect();
        if required.len() != 1 {
            return Err(MunaError::Prediction(format!(
                "{tag} cannot be used with image API: expected 1 required input"
            )));
        }
        let prompt_param = required.iter()
            .find(|p| p.dtype == Some(crate::types::Dtype::List))
            .ok_or_else(|| MunaError::Prediction(format!("{tag}: no text prompt parameter")))?;
        let is_int = |d: crate::types::Dtype| matches!(d,
            crate::types::Dtype::Int8 | crate::types::Dtype::Int16 | crate::types::Dtype::Int32 | crate::types::Dtype::Int64 |
            crate::types::Dtype::Uint8 | crate::types::Dtype::Uint16 | crate::types::Dtype::Uint32 | crate::types::Dtype::Uint64
        );
        let width_param = signature.inputs.iter()
            .find(|p| p.dtype.map_or(false, is_int) && p.denotation.as_deref() == Some("openai.images.width"))
            .map(|p| p.name.clone());
        let height_param = signature.inputs.iter()
            .find(|p| p.dtype.map_or(false, is_int) && p.denotation.as_deref() == Some("openai.images.height"))
            .map(|p| p.name.clone());
        let count_param = signature.inputs.iter()
            .find(|p| p.dtype.map_or(false, is_int) && p.denotation.as_deref() == Some("openai.images.count"))
            .map(|p| p.name.clone());
        let image_idx = signature.outputs.iter().position(|p| {
            p.dtype == Some(crate::types::Dtype::ImageList)
        }).ok_or_else(|| MunaError::Prediction(format!("{tag}: no image_list output")))?;
        Ok(ImageConfig {
            prompt_param_name: prompt_param.name.clone(),
            width_param,
            height_param,
            count_param,
            image_param_idx: image_idx,
        })
    }

    fn encode_image(
        image: &crate::types::Image,
        output_format: &str,
    ) -> Result<ImageData> {
        if output_format == "raw" {
            return Ok(ImageData { b64_json: None, image: Some(image.clone()) });
        }
        let fxn_value = c::Value::from_object(&Value::Image(image.clone()))?;
        let mime = format!("image/{output_format}");
        let buffer = fxn_value.serialize(Some(&mime))?;
        Ok(ImageData { b64_json: Some(BASE64.encode(&buffer)), image: None })
    }

    async fn create_prediction(
        &self,
        tag: &str,
        inputs: HashMap<String, Value>,
        acceleration: Acceleration,
    ) -> Result<crate::types::Prediction> {
        // `predictions.create` routes remote accelerations internally.
        self.predictions
            .create(tag, Some(inputs), Some(acceleration), None, None)
            .await
    }
}
