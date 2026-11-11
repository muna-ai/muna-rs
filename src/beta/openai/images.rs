/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::collections::HashMap;

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;

use crate::c;
use crate::client::{ClientExt, MunaError, RequestInput, Result};
use crate::services::{create_remote_value, is_local_acceleration};
use crate::services::{PredictionService, PredictorService};
use crate::types::{Acceleration, Device, Dtype, RemoteData, RemoteValue, Signature, Value};

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
#[derive(Debug, Clone, Default)]
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
    /// Local compute devices.
    pub devices: Option<Vec<Device>>,
}

/// An encoded image file or an already decoded image. File I/O belongs to the caller.
#[derive(Debug, Clone)]
pub enum ImageInput {
    Encoded(Vec<u8>),
    Pixels(crate::types::Image),
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

/// Image editing parameters. Multiple images are references for one prompt.
#[derive(Debug, Clone, Default)]
pub struct ImageEditParams {
    pub image: Vec<ImageInput>,
    pub prompt: String,
    pub model: String,
    pub background: Option<String>,
    pub n: Option<i32>,
    pub output_format: Option<String>,
    pub output_compression: Option<i32>,
    pub size: Option<ImageSize>,
    pub acceleration: Option<Acceleration>,
    pub devices: Option<Vec<Device>>,
}

#[derive(Clone)]
struct ImageConfig {
    prompt_param_name: String,
    input_image: Option<(String, Dtype)>,
    width_param: Option<String>,
    height_param: Option<String>,
    count_param: Option<String>,
    image_param_idx: usize,
}

impl ImageConfig {
    fn from_signature(signature: &Signature, edit: bool) -> Result<Self> {
        let required: Vec<_> = signature
            .inputs
            .iter()
            .filter(|p| p.optional != Some(true))
            .collect();
        let prompts: Vec<_> = required
            .iter()
            .filter(|p| p.dtype == Some(Dtype::String))
            .collect();
        if prompts.len() != 1 {
            return Err(MunaError::InvalidInput(
                "Images API requires one required scalar string prompt.".into(),
            ));
        }
        let prompt = prompts[0];
        let image = if edit {
            let candidates: Vec<_> = signature
                .inputs
                .iter()
                .filter(|p| matches!(p.dtype, Some(Dtype::Image | Dtype::ImageList)))
                .collect();
            let annotated: Vec<_> = candidates
                .iter()
                .copied()
                .filter(|p| p.denotation.as_deref() == Some("openai.chat.completions.images"))
                .collect();
            let image = match annotated.as_slice() {
                [image] => *image,
                [] if candidates.len() == 1 => candidates[0],
                _ => {
                    return Err(MunaError::InvalidInput(
                        "Images edit requires one unambiguous image or image_list input.".into(),
                    ))
                }
            };
            Some(image)
        } else {
            None
        };
        if required
            .iter()
            .any(|p| p.name != prompt.name && image.map_or(true, |image| p.name != image.name))
        {
            return Err(MunaError::InvalidInput(
                "Predictor has unsupported required inputs for the Images API.".into(),
            ));
        }
        let numeric_name = |denotation| {
            signature
                .inputs
                .iter()
                .find(|p| {
                    matches!(
                        p.dtype,
                        Some(
                            Dtype::Int8
                                | Dtype::Int16
                                | Dtype::Int32
                                | Dtype::Int64
                                | Dtype::Uint8
                                | Dtype::Uint16
                                | Dtype::Uint32
                                | Dtype::Uint64
                        )
                    ) && p.denotation.as_deref() == Some(denotation)
                })
                .map(|p| p.name.clone())
        };
        let image_param_idx = signature
            .outputs
            .iter()
            .position(|p| p.dtype == Some(Dtype::ImageList))
            .ok_or_else(|| {
                MunaError::InvalidInput("Images API requires an image_list output.".into())
            })?;
        Ok(Self {
            prompt_param_name: prompt.name.clone(),
            input_image: image.map(|p| (p.name.clone(), p.dtype.unwrap())),
            width_param: numeric_name("openai.images.width"),
            height_param: numeric_name("openai.images.height"),
            count_param: numeric_name("openai.images.count"),
            image_param_idx,
        })
    }

    fn bind_remote(
        &self,
        params: &ImageCreateParams,
        image: Option<Vec<ImageInput>>,
    ) -> Result<serde_json::Value> {
        // Reuse scalar binding and validation without decoding image files.
        let mut scalars = self.clone();
        scalars.input_image = None;
        let mut inputs = scalars
            .bind(params, None)?
            .iter()
            .map(|(name, value)| Ok((name.clone(), create_remote_value(value)?)))
            .collect::<Result<HashMap<String, RemoteValue>>>()?;
        if let Some((name, dtype)) = &self.input_image {
            let images = image
                .ok_or_else(|| MunaError::InvalidInput("Input images are required.".into()))?;
            if images.is_empty()
                || images.len() > 1024
                || (*dtype == Dtype::Image && images.len() != 1)
            {
                return Err(MunaError::InvalidInput(
                    "Invalid number of input images.".into(),
                ));
            }
            let mut images = images
                .into_iter()
                .map(encoded_remote_image)
                .collect::<Result<Vec<_>>>()?;
            let value = if *dtype == Dtype::Image {
                images.remove(0)
            } else {
                RemoteValue {
                    dtype: Dtype::ImageList,
                    data: Some(RemoteData::Sequence(images)),
                }
            };
            inputs.insert(name.clone(), value);
        }
        Ok(serde_json::to_value(inputs)?)
    }

    fn bind(
        &self,
        params: &ImageCreateParams,
        image: Option<Vec<ImageInput>>,
    ) -> Result<HashMap<String, Value>> {
        if params.n.is_some_and(|n| n < 1) {
            return Err(MunaError::InvalidInput(
                "Image count must be positive.".into(),
            ));
        }
        let mut inputs = HashMap::from([(
            self.prompt_param_name.clone(),
            Value::String(params.prompt.clone()),
        )]);
        if let Some((name, dtype)) = &self.input_image {
            let images = image
                .ok_or_else(|| MunaError::InvalidInput("Input images are required.".into()))?;
            if images.is_empty() || (*dtype == Dtype::Image && images.len() != 1) {
                return Err(MunaError::InvalidInput(
                    "Images edit requires at least one image; scalar image predictors accept exactly one.".into()
                ));
            }
            let images: Vec<_> = images
                .into_iter()
                .map(decode_image)
                .collect::<Result<_>>()?;
            let value = if *dtype == Dtype::Image {
                Value::Image(images.into_iter().next().unwrap())
            } else {
                Value::ImageList(images)
            };
            inputs.insert(name.clone(), value);
        }
        let (width, height) = params.size.unwrap_or(ImageSize::Auto).dimensions();
        for (name, value) in [
            (&self.count_param, params.n),
            (&self.width_param, width),
            (&self.height_param, height),
        ] {
            if let (Some(name), Some(value)) = (name, value) {
                inputs.insert(name.clone(), Value::Int(value));
            }
        }
        Ok(inputs)
    }
}

// File bytes remain unchanged. Function sniffs the actual format at decode time.
fn encoded_remote_image(input: ImageInput) -> Result<RemoteValue> {
    let (bytes, mime) = match input {
        ImageInput::Encoded(bytes) if !bytes.is_empty() => (bytes, "image/*"),
        ImageInput::Encoded(_) => {
            return Err(MunaError::InvalidInput("Image file is empty.".into()))
        }
        ImageInput::Pixels(image) => (
            c::Value::from_object(&Value::Image(image))?.serialize(Some("image/png"))?,
            "image/png",
        ),
    };
    Ok(RemoteValue {
        dtype: Dtype::Image,
        data: Some(format!("data:{mime};base64,{}", BASE64.encode(bytes)).into()),
    })
}

fn image_response(images: Vec<ImageData>) -> ImageResponse {
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    ImageResponse {
        data: Some(images),
        background: Some("opaque".into()),
        created,
        usage: None,
    }
}

fn encoded_image_results(
    prediction: crate::types::RemotePrediction,
    image_idx: usize,
    format: &str,
    quality: Option<i32>,
) -> Result<ImageResponse> {
    if let Some(error) = prediction.error {
        return Err(MunaError::from_prediction_error(error));
    }
    let value = prediction
        .results
        .as_ref()
        .and_then(|values| values.get(image_idx))
        .ok_or_else(|| MunaError::Prediction("No image results returned.".into()))?;
    if value.dtype != Dtype::ImageList {
        return Err(MunaError::Prediction("Invalid image-list output.".into()));
    }
    let items = match &value.data {
        Some(RemoteData::Sequence(items)) if items.len() <= 1024 => items,
        // A plain SDK client has no server-specific encoding policy. Preserve
        // compatibility with legacy sequence responses in that case.
        Some(RemoteData::Encoded(url)) => {
            let encoded = url.strip_prefix("data:image/avif;base64,").ok_or_else(|| {
                MunaError::Prediction("Invalid legacy image-list encoding.".into())
            })?;
            let bytes = BASE64.decode(encoded).map_err(|error| {
                MunaError::Prediction(format!("Invalid image-list data: {error}"))
            })?;
            let decoded = c::Value::from_bytes(&bytes, "image/avif")?.to_object()?;
            let Value::ImageList(images) = decoded else {
                return Err(MunaError::Prediction("Invalid image-list output.".into()));
            };
            return Ok(image_response(
                images
                    .iter()
                    .map(|image| ImageService::encode_image(image, format, quality))
                    .collect::<Result<_>>()?,
            ));
        }
        _ => return Err(MunaError::Prediction("Invalid image-list output.".into())),
    };
    let prefix = format!("data:image/{format};base64,");
    let images = items
        .iter()
        .map(|item| {
            let data = item
                .data
                .as_ref()
                .and_then(RemoteData::as_str)
                .filter(|_| item.dtype == Dtype::Image)
                .and_then(|url| url.strip_prefix(&prefix))
                .ok_or_else(|| {
                    MunaError::Prediction("Node returned an unexpected image encoding.".into())
                })?;
            Ok(ImageData {
                b64_json: Some(data.to_owned()),
                image: None,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(image_response(images))
}

fn decode_image(input: ImageInput) -> Result<crate::types::Image> {
    match input {
        ImageInput::Pixels(image) => Ok(image),
        ImageInput::Encoded(bytes) => {
            if bytes.is_empty() {
                return Err(MunaError::InvalidInput("Image file is empty.".into()));
            }
            let value = c::Value::from_bytes(&bytes, "image/*")
                .and_then(|value| value.to_object())
                .map_err(|error| {
                    MunaError::InvalidInput(format!("Cannot decode image: {error}"))
                })?;
            match value {
                Value::Image(image) => Ok(image),
                _ => Err(MunaError::InvalidInput(
                    "File did not decode to an image.".into(),
                )),
            }
        }
    }
}

/// Image generation and editing service.
#[derive(Clone)]
pub struct ImageService {
    predictors: PredictorService,
    predictions: PredictionService,
    cache: std::sync::Arc<tokio::sync::Mutex<HashMap<(String, bool), ImageConfig>>>,
}

impl ImageService {
    pub fn new(predictors: PredictorService, predictions: PredictionService) -> Self {
        Self {
            predictors,
            predictions,
            cache: std::sync::Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        }
    }

    /// Generate images from a text prompt.
    pub async fn generate(&self, params: ImageCreateParams) -> Result<ImageResponse> {
        self.execute(params, None).await
    }

    /// Edit one or more reference images using a scalar text prompt.
    pub async fn edit(&self, params: ImageEditParams) -> Result<ImageResponse> {
        let options = ImageCreateParams {
            prompt: params.prompt,
            model: params.model,
            background: params.background,
            n: params.n,
            output_format: params.output_format,
            output_compression: params.output_compression,
            size: params.size,
            acceleration: params.acceleration,
            devices: params.devices,
        };
        self.execute(options, Some(params.image)).await
    }

    async fn execute(
        &self,
        params: ImageCreateParams,
        image: Option<Vec<ImageInput>>,
    ) -> Result<ImageResponse> {
        let tag = params.model.clone();
        let key = (tag.clone(), image.is_some());
        // Clone the config and release the cache before decoding or inference.
        let cached = self.cache.lock().await.get(&key).cloned();
        let config = match cached {
            Some(config) => config,
            None => {
                let predictor = self.predictors.retrieve(&tag).await?.ok_or_else(|| {
                    MunaError::InvalidInput(format!("Image predictor {tag} could not be found."))
                })?;
                let config = ImageConfig::from_signature(&predictor.signature, key.1)?;
                self.cache.lock().await.insert(key, config.clone());
                config
            }
        };
        let output_format = params.output_format.as_deref().unwrap_or("png");
        if !matches!(output_format, "png" | "jpeg" | "webp" | "raw") {
            return Err(MunaError::InvalidInput(
                "Unsupported image output format.".into(),
            ));
        }
        if params
            .output_compression
            .is_some_and(|quality| !(0..=100).contains(&quality))
        {
            return Err(MunaError::InvalidInput(
                "Image output compression must be between 0 and 100.".into(),
            ));
        }
        let acceleration = params
            .acceleration
            .clone()
            .unwrap_or(Acceleration::LocalAuto);
        let image_idx = config.image_param_idx;
        if !is_local_acceleration(Some(&acceleration)) && output_format != "raw" {
            let inputs = config.bind_remote(&params, image)?;
            // Keep remote results encoded. Application Client wrappers own
            // request policy (headers, routing, and prediction billing capture).
            let client = &self.predictions.client;
            let target = c::Configuration::get_target().unwrap_or_else(|_| "rust".to_string());
            let mut prediction: crate::types::RemotePrediction = client
                .request_as(
                    RequestInput::post("/predictions")
                        .base_url(client.inference_url())
                        .body(serde_json::json!({
                            "tag": tag,
                            "inputs": inputs,
                            "acceleration": acceleration,
                            "target": target,
                        })),
                )
                .await?;
            // Legacy backends may reference a resource instead of returning
            // inline AVIF. Fetch through the caller's existing Client contract.
            if let Some(value) = prediction
                .results
                .as_mut()
                .and_then(|values| values.get_mut(image_idx))
            {
                if let Some(RemoteData::Encoded(url)) = &value.data {
                    if !url.starts_with("data:") {
                        let bytes = client.fetch(url).await?;
                        value.data =
                            Some(format!("data:image/avif;base64,{}", BASE64.encode(bytes)).into());
                    }
                }
            }
            return encoded_image_results(
                prediction,
                image_idx,
                output_format,
                params.output_compression,
            );
        }
        let prediction_inputs = config.bind(&params, image)?;
        // Create prediction
        let prediction = self
            .create_prediction(
                &tag,
                prediction_inputs,
                acceleration,
                params.devices.as_deref(),
            )
            .await?;
        if let Some(error) = &prediction.error {
            return Err(MunaError::from_prediction_error(error.clone()));
        }
        let results = prediction
            .results
            .as_ref()
            .ok_or_else(|| MunaError::Prediction("No results returned".into()))?;
        let images_value = results
            .get(image_idx)
            .ok_or_else(|| MunaError::Prediction(format!("{tag} did not return images")))?;
        // Signature validation guarantees the output at `image_idx` is an image list.
        let images = match images_value {
            Value::ImageList(list) => {
                let mut image_data = Vec::new();
                for img in list {
                    image_data.push(Self::encode_image(
                        img,
                        output_format,
                        params.output_compression,
                    )?);
                }
                image_data
            }
            _ => {
                return Err(MunaError::Prediction(format!(
                    "{tag} returned unexpected type instead of images"
                )))
            }
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

    fn encode_image(
        image: &crate::types::Image,
        output_format: &str,
        quality: Option<i32>,
    ) -> Result<ImageData> {
        if output_format == "raw" {
            return Ok(ImageData {
                b64_json: None,
                image: Some(image.clone()),
            });
        }
        let fxn_value = c::Value::from_object(&Value::Image(image.clone()))?;
        let mut mime = format!("image/{output_format}");
        if let Some(quality) = quality {
            mime.push_str(&format!(";q={:.2}", quality as f32 / 100.0));
        }
        let buffer = fxn_value.serialize(Some(&mime))?;
        Ok(ImageData {
            b64_json: Some(BASE64.encode(&buffer)),
            image: None,
        })
    }

    async fn create_prediction(
        &self,
        tag: &str,
        inputs: HashMap<String, Value>,
        acceleration: Acceleration,
        devices: Option<&[Device]>,
    ) -> Result<crate::types::Prediction> {
        // `predictions.create` routes remote accelerations internally.
        self.predictions
            .create(
                tag,
                Some(inputs),
                Some(acceleration),
                devices,
                None,
                None,
                None,
            )
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Image;
    use serde_json::json;

    fn signature(image_dtype: Option<&str>) -> Signature {
        let mut inputs = vec![
            json!({"name":"prompt", "dtype":"string"}),
            json!({"name":"count", "dtype":"int32", "optional":true, "denotation":"openai.images.count"}),
            json!({"name":"width", "dtype":"int32", "optional":true, "denotation":"openai.images.width"}),
            json!({"name":"height", "dtype":"int32", "optional":true, "denotation":"openai.images.height"}),
        ];
        if let Some(dtype) = image_dtype {
            inputs
                .push(json!({"name":"source", "dtype":dtype, "denotation":"openai.chat.completions.images"}));
        }
        serde_json::from_value(json!({
            "inputs":inputs,
            "outputs":[{"name":"images", "dtype":"image_list"}]
        }))
        .unwrap()
    }

    fn image(red: u8) -> Image {
        Image {
            data: vec![red, 0, 0],
            width: 1,
            height: 1,
            channels: 3,
        }
    }

    #[test]
    fn scalar_prompt_binds_count_and_size_without_expanding_requests() {
        let config = ImageConfig::from_signature(&signature(None), false).unwrap();
        let inputs = config
            .bind(
                &ImageCreateParams {
                    prompt: "cat".into(),
                    n: Some(2),
                    size: Some(ImageSize::Size512x512),
                    ..Default::default()
                },
                None,
            )
            .unwrap();
        assert!(matches!(&inputs["prompt"], Value::String(s) if s == "cat"));
        assert!(matches!(&inputs["count"], Value::Int(2)));
        assert!(matches!(&inputs["width"], Value::Int(512)));
        assert!(matches!(&inputs["height"], Value::Int(512)));
    }

    #[test]
    fn edit_binds_scalar_image_and_multiple_references_in_order() {
        for dtype in ["image", "image_list"] {
            let config = ImageConfig::from_signature(&signature(Some(dtype)), true).unwrap();
            let inputs = config
                .bind(&ImageCreateParams::default(), Some(vec![image(10).into()]))
                .unwrap();
            assert!(matches!(&inputs["prompt"], Value::String(s) if s.is_empty()));
            match &inputs["source"] {
                Value::Image(value) => assert_eq!(*value, image(10)),
                Value::ImageList(values) => assert_eq!(values, &vec![image(10)]),
                other => panic!("Unexpected image input: {other:?}"),
            }
            let multiple = config.bind(
                &ImageCreateParams::default(),
                Some(vec![image(10).into(), image(20).into()]),
            );
            if dtype == "image" {
                assert!(multiple.is_err());
            } else {
                assert!(
                    matches!(&multiple.unwrap()["source"], Value::ImageList(v) if v == &vec![image(10), image(20)])
                );
            }
            assert!(config
                .bind(&ImageCreateParams::default(), Some(vec![]))
                .is_err());
        }
    }

    #[test]
    fn rejects_legacy_list_prompts_and_incompatible_endpoint_signatures() {
        let mut legacy = signature(None);
        legacy.inputs[0].dtype = Some(Dtype::List);
        assert!(ImageConfig::from_signature(&legacy, false).is_err());
        assert!(ImageConfig::from_signature(&signature(Some("image")), false).is_err());
        assert!(ImageConfig::from_signature(&signature(None), true).is_err());
        let config = ImageConfig::from_signature(&signature(None), false).unwrap();
        assert!(config
            .bind(
                &ImageCreateParams {
                    n: Some(0),
                    ..Default::default()
                },
                None
            )
            .is_err());
    }

    #[test]
    fn encoded_images_decode_and_invalid_files_fail() {
        let input = image(255);
        let encoded = c::Value::from_object(&Value::Image(input.clone()))
            .unwrap()
            .serialize(Some("image/png"))
            .unwrap();
        let decoded = decode_image(encoded.into()).unwrap();
        assert_eq!((decoded.width, decoded.height), (1, 1));
        assert_eq!(decoded.data[0], 255);
        assert!(decode_image(Vec::<u8>::new().into()).is_err());
        assert!(decode_image(b"invalid image".to_vec().into()).is_err());
    }
    #[test]
    fn remote_binding_preserves_encoded_files_without_decoding() {
        let config = ImageConfig::from_signature(&signature(Some("image_list")), true).unwrap();
        // Intentionally undecodable: binding must not touch any image codec.
        let original = b"original JPEG bytes".to_vec();
        let inputs = config
            .bind_remote(
                &ImageCreateParams::default(),
                Some(vec![
                    original.clone().into(),
                    b"second file".to_vec().into(),
                ]),
            )
            .unwrap();
        let url = inputs["source"]["data"][0]["data"].as_str().unwrap();
        assert_eq!(
            BASE64
                .decode(url.split_once(";base64,").unwrap().1)
                .unwrap(),
            original
        );
        assert_eq!(inputs["source"]["data"].as_array().unwrap().len(), 2);
        assert!(config
            .bind_remote(&ImageCreateParams::default(), Some(vec![]))
            .is_err());
        let scalar = ImageConfig::from_signature(&signature(Some("image")), true).unwrap();
        assert!(scalar
            .bind_remote(
                &ImageCreateParams::default(),
                Some(vec![b"a".to_vec().into(), b"b".to_vec().into()])
            )
            .is_err());
    }

    #[test]
    fn remote_results_pass_through_without_an_image_codec() {
        let prediction: crate::types::RemotePrediction = serde_json::from_value(json!({
            "id":"pred_test", "tag":"@a/albedo", "created":"2026-11-11T00:00:00Z", "latency":0.4,
            "results":[{"dtype":"image_list", "data":[
                {"dtype":"image", "data":"data:image/jpeg;base64,b3JpZ2luYWw="},
                {"dtype":"image", "data":"data:image/jpeg;base64,c2Vjb25k"}
            ]}]
        }))
        .unwrap();
        let result = encoded_image_results(prediction.clone(), 0, "jpeg", None).unwrap();
        let data = result.data.unwrap();
        assert_eq!(data[0].b64_json.as_deref(), Some("b3JpZ2luYWw="));
        assert_eq!(data[1].b64_json.as_deref(), Some("c2Vjb25k"));
        assert!(encoded_image_results(prediction, 0, "png", None).is_err());
    }
    #[test]
    fn plain_client_can_still_convert_legacy_avif_results() {
        let bytes = c::Value::from_object(&Value::ImageList(vec![image(80), image(160)]))
            .unwrap()
            .serialize(None)
            .unwrap();
        let prediction = serde_json::from_value(json!({
            "id":"pred_legacy", "tag":"@a/albedo", "created":"2026-11-11T00:00:00Z", "latency":0.4,
            "results":[{"dtype":"image_list", "data":format!("data:image/avif;base64,{}", BASE64.encode(bytes))}]
        })).unwrap();
        let result = encoded_image_results(prediction, 0, "jpeg", Some(90)).unwrap();
        let images = result.data.unwrap();
        assert_eq!(images.len(), 2);
        for image in images {
            let bytes = BASE64.decode(image.b64_json.unwrap()).unwrap();
            assert!(bytes.starts_with(&[0xff, 0xd8]));
            assert!(matches!(
                c::Value::from_bytes(&bytes, "image/jpeg")
                    .unwrap()
                    .to_object()
                    .unwrap(),
                Value::Image(_)
            ));
        }
    }
}
