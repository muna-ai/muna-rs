/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use crate::types::{Dtype, Parameter};

pub fn get_parameter<'a>(
    parameters: &'a [Parameter],
    dtypes: &[Dtype],
    denotation: Option<&str>,
) -> (Option<usize>, Option<&'a Parameter>) {
    for (idx, param) in parameters.iter().enumerate() {
        if let Some(param_dtype) = param.dtype {
            if dtypes.contains(&param_dtype)
                && (denotation.is_none() || param.denotation.as_deref() == denotation)
            {
                return (Some(idx), Some(param));
            }
        }
    }
    (None, None)
}
