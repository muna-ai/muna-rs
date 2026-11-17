/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::collections::HashSet;
use std::sync::OnceLock;

use crate::client::{MunaClient, MunaError, Result};
use crate::types::{Dtype, Parameter, Prediction, Signature, Value};

pub const INTEGERS: &[Dtype] = &[
    Dtype::Int8,
    Dtype::Int16,
    Dtype::Int32,
    Dtype::Int64,
    Dtype::Uint8,
    Dtype::Uint16,
    Dtype::Uint32,
    Dtype::Uint64,
];
pub const FLOATS: &[Dtype] = &[
    Dtype::Float16,
    Dtype::BFloat16,
    Dtype::Float32,
    Dtype::Float64,
];

pub fn get_parameter(
    parameters: &[Parameter],
    dtypes: &[Dtype],
    denotation: Option<&str>,
) -> Result<Option<(usize, Parameter)>> {
    let mut matches = parameters.iter().enumerate().filter(|(_, parameter)| {
        parameter.dtype.is_some_and(|dtype| dtypes.contains(&dtype))
            && denotation
                .is_none_or(|denotation| parameter.denotation.as_deref() == Some(denotation))
    });
    let result = matches
        .next()
        .map(|(index, parameter)| (index, parameter.clone()));
    if matches.next().is_some() {
        return Err(MunaError::InvalidInput(format!(
            "{} must identify at most one parameter with dtype {dtypes:?}",
            denotation.unwrap_or("Parameter lookup")
        )));
    }
    Ok(result)
}

pub fn parameter(
    parameters: &[Parameter],
    dtypes: &[Dtype],
    denotation: Option<&str>,
) -> Result<Option<Parameter>> {
    Ok(get_parameter(parameters, dtypes, denotation)?.map(|(_, parameter)| parameter))
}

pub fn required(parameters: &[Parameter]) -> Vec<Parameter> {
    parameters
        .iter()
        .filter(|parameter| parameter.optional != Some(true))
        .cloned()
        .collect()
}

pub fn require<T>(value: Option<T>, message: &str) -> Result<T> {
    value.ok_or_else(|| MunaError::InvalidInput(message.into()))
}

pub fn validate_inputs(signature: &Signature, parameters: &[Option<&Parameter>]) -> Result<()> {
    let mut names = HashSet::new();
    for parameter in &signature.inputs {
        if !names.insert(&parameter.name) {
            return Err(MunaError::InvalidInput(
                "Signature input names must be unique".into(),
            ));
        }
        if parameter.optional != Some(true)
            && !parameters
                .iter()
                .flatten()
                .any(|supported| supported.name == parameter.name)
        {
            return Err(MunaError::InvalidInput(format!(
                "Unsupported required input: {}",
                parameter.name
            )));
        }
    }
    Ok(())
}

pub fn output(signature: &Signature, dtypes: &[Dtype], title: &str) -> Result<Option<usize>> {
    let mut matches = signature
        .outputs
        .iter()
        .enumerate()
        .filter(|(_, parameter)| {
            parameter.dtype.is_some_and(|dtype| dtypes.contains(&dtype))
                && parameter
                    .schema
                    .as_ref()
                    .and_then(|schema| schema.get("title"))
                    .and_then(|title| title.as_str())
                    == Some(title)
        });
    let matched = matches.next().map(|(index, _)| index);
    if matches.next().is_some() {
        return Err(MunaError::InvalidInput(format!(
            "Requires one unambiguous {title} output"
        )));
    }
    Ok(matched)
}

pub fn result(prediction: &Prediction, index: usize) -> Result<&Value> {
    if let Some(error) = &prediction.error {
        return Err(MunaError::from_prediction_error(error.clone()));
    }
    prediction
        .results
        .as_ref()
        .and_then(|results| results.get(index))
        .ok_or_else(|| {
            MunaError::Prediction("Prediction did not return the adapter's output".into())
        })
}

pub async fn decode(value: &Value, dtype: Dtype) -> Result<Value> {
    if let Value::Remote(remote) = value {
        if remote.dtype != dtype {
            return Err(MunaError::Prediction(format!(
                "Expected {dtype:?}, received {:?}",
                remote.dtype
            )));
        }
        // Resource URLs are already signed. No API key is attached to downloads.
        static CLIENT: OnceLock<MunaClient> = OnceLock::new();
        crate::services::parse_remote_value(
            CLIENT.get_or_init(|| MunaClient::new(None, None)),
            remote,
        )
        .await
    } else {
        Ok(value.clone())
    }
}

pub fn created(prediction: &Prediction) -> Result<i64> {
    chrono::DateTime::parse_from_rfc3339(&prediction.created)
        .map(|created| created.timestamp())
        .map_err(|error| MunaError::Prediction(format!("Invalid prediction timestamp: {error}")))
}

pub fn inline_base64<'a>(
    value: &'a crate::types::RemoteValue,
    mime_types: &[&str],
) -> Option<&'a str> {
    let url = value.data.as_ref()?.as_str()?;
    let (header, payload) = url.strip_prefix("data:")?.split_once(";base64,")?;
    mime_types
        .iter()
        .any(|mime| header.eq_ignore_ascii_case(mime))
        .then_some(payload)
}
