/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;

use crate::beta::adapters::{
    Adapter, DepthFormat, ImageResponse, OpenAIDepthAdapter, OpenAIDepthOptions,
    OpenAIImageEditAdapter, OpenAIImageGenerationAdapter, OpenAIImageOptions,
    OpenAIModel3DAdapter, OpenAIModel3DOptions,
};
use crate::client::{MunaError, Result};
use crate::services::{PredictionService, PredictorService};
use crate::types::{Acceleration, Device};

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

/// Image editing parameters. Multiple images are references for one prompt.
#[derive(Debug, Clone, Default)]
pub struct ImageEditParams {
    pub image: Vec<ImageInput>,
    /// Editing prompt; omission uses the default of an optional annotated predictor input.
    pub prompt: Option<String>,
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
enum EditAdapter {
    Image(OpenAIImageEditAdapter),
    Depth(OpenAIDepthAdapter),
    Model3D(OpenAIModel3DAdapter),
}

#[derive(Clone)]
pub struct ImageService {
    predictors: PredictorService,
    predictions: PredictionService,
    generations: Arc<RwLock<HashMap<String, OpenAIImageGenerationAdapter>>>,
    edits: Arc<RwLock<HashMap<String, EditAdapter>>>,
}

impl ImageService {
    pub fn new(predictors: PredictorService, predictions: PredictionService) -> Self {
        Self {
            predictors,
            predictions,
            generations: Arc::default(),
            edits: Arc::default(),
        }
    }

    pub async fn generate(&self, params: ImageCreateParams) -> Result<ImageResponse> {
        let cached = self.generations.read().await.get(&params.model).cloned();
        let adapter = match cached {
            Some(adapter) => adapter,
            None => {
                let predictor =
                    self.predictors
                        .retrieve(&params.model)
                        .await?
                        .ok_or_else(|| {
                            MunaError::InvalidInput(format!(
                                "Predictor {} could not be found",
                                params.model
                            ))
                        })?;
                let adapter = OpenAIImageGenerationAdapter::from_signature(&predictor.signature)
                    .ok_or_else(|| {
                        MunaError::InvalidInput(format!(
                            "Predictor {} does not support image generation",
                            params.model
                        ))
                    })?;
                self.generations
                    .write()
                    .await
                    .insert(params.model.clone(), adapter.clone());
                adapter
            }
        };
        let (width, height) = params.size.unwrap_or(ImageSize::Auto).dimensions();
        let inputs = adapter.inputs(params.prompt, params.n, width, height);
        let prediction = self
            .predictions
            .create(
                &params.model,
                Some(inputs),
                params.acceleration,
                params.devices.as_deref(),
                None,
                None,
                None,
                Some(true),
            )
            .await?;
        adapter
            .adapt(
                prediction,
                Some(OpenAIImageOptions {
                    output_format: params.output_format.unwrap_or("png".into()),
                    output_compression: params.output_compression,
                }),
            )
            .await
    }

    pub async fn edit(&self, params: ImageEditParams) -> Result<ImageResponse> {
        let cached = self.edits.read().await.get(&params.model).cloned();
        let adapter = match cached {
            Some(adapter) => adapter,
            None => {
                let predictor =
                    self.predictors
                        .retrieve(&params.model)
                        .await?
                        .ok_or_else(|| {
                            MunaError::InvalidInput(format!(
                                "Predictor {} could not be found",
                                params.model
                            ))
                        })?;
                let signature = &predictor.signature;
                let adapter = if signature.outputs.iter().any(|p| p.denotation.as_deref() == Some("depth_map")) {
                    OpenAIDepthAdapter::from_signature(signature).map(EditAdapter::Depth)
                } else if signature.outputs.iter().any(|p| p.denotation.as_deref() == Some("3d")) {
                    OpenAIModel3DAdapter::from_signature(signature).map(EditAdapter::Model3D)
                } else {
                    OpenAIImageEditAdapter::from_signature(signature).map(EditAdapter::Image)
                }.ok_or_else(|| MunaError::InvalidInput(format!(
                    "Predictor {} does not support image editing, depth estimation, or 3D generation",
                    params.model,
                )))?;
                self.edits
                    .write()
                    .await
                    .insert(params.model.clone(), adapter.clone());
                adapter
            }
        };
        // Validate the response encoding before invoking an expensive predictor.
        let format = params.output_format.as_deref().unwrap_or(match &adapter {
            EditAdapter::Image(_) => "png",
            EditAdapter::Depth(_) => "npz",
            EditAdapter::Model3D(_) => "ply",
        });
        match &adapter {
            EditAdapter::Image(_) if !matches!(format, "png" | "jpeg" | "webp" | "raw") => {
                return Err(MunaError::InvalidInput(
                    "Image output requires png, jpeg, webp, or raw.".into(),
                ));
            }
            EditAdapter::Depth(_) if !matches!(format, "npy" | "npz") => {
                return Err(MunaError::InvalidInput(
                    "Depth output requires output_format=npy or npz.".into(),
                ));
            }
            EditAdapter::Model3D(_) => OpenAIModel3DOptions {
                format: format.into(),
            }
            .validate()?,
            _ => {}
        }
        if !matches!(&adapter, EditAdapter::Image(_))
            && (params.background.is_some() || params.output_compression.is_some())
        {
            return Err(MunaError::InvalidInput(
                "Depth and 3D output do not accept background or output_compression.".into(),
            ));
        }
        let (width, height) = params.size.unwrap_or(ImageSize::Auto).dimensions();
        let inputs = match &adapter {
            EditAdapter::Image(adapter) => {
                adapter.inputs(params.image, params.prompt, params.n, width, height)?
            }
            EditAdapter::Depth(adapter) => {
                adapter.inputs(params.image, params.prompt, params.n, width, height)?
            }
            EditAdapter::Model3D(adapter) => {
                adapter.inputs(params.image, params.prompt, params.n, width, height)?
            }
        };
        let prediction = self
            .predictions
            .create(
                &params.model,
                Some(inputs),
                params.acceleration,
                params.devices.as_deref(),
                None,
                None,
                None,
                Some(true),
            )
            .await?;
        match adapter {
            EditAdapter::Image(adapter) => {
                adapter
                    .adapt(
                        prediction,
                        Some(OpenAIImageOptions {
                            output_format: format.into(),
                            output_compression: params.output_compression,
                        }),
                    )
                    .await
            }
            EditAdapter::Depth(adapter) => {
                adapter
                    .adapt(
                        prediction,
                        Some(OpenAIDepthOptions {
                            format: if format == "npy" {
                                DepthFormat::Npy
                            } else {
                                DepthFormat::Npz
                            },
                        }),
                    )
                    .await
            }
            EditAdapter::Model3D(adapter) => {
                adapter
                    .adapt(
                        prediction,
                        Some(OpenAIModel3DOptions {
                            format: format.into(),
                        }),
                    )
                    .await
            }
        }
    }
}

pub use crate::beta::adapters::ImageInput;
