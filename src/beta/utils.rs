/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use crate::types::{Dtype, Parameter};

pub(crate) fn get_parameter<'a>(
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

/// Split `@owner/name` into `(owner, name)`. Tags without a slash are
/// their own name with an empty owner.
pub(crate) fn split_tag(tag: &str) -> (&str, &str) {
    match tag.split_once('/') {
        Some((owner, name)) => (owner.trim_start_matches('@'), name),
        None => ("", tag.trim_start_matches('@')),
    }
}

/// `YYYY-MM-DDTHH:MM:SS[.fff]Z` (the API's date format) to Unix seconds
/// (Howard Hinnant's days-from-civil). `None` for anything else, including
/// offsets other than `Z`.
pub(crate) fn rfc3339_to_unix(value: &str) -> Option<u64> {
    let value = value.strip_suffix('Z')?;
    let (date, time) = value.split_once('T')?;
    let time = time.split_once('.').map_or(time, |(whole, _)| whole);
    let mut date = date.splitn(3, '-').map(|part| part.parse::<i64>().ok());
    let (y, m, d) = (date.next()??, date.next()??, date.next()??);
    let mut time = time.splitn(3, ':').map(|part| part.parse::<i64>().ok());
    let (hh, mm, ss) = (time.next()??, time.next()??, time.next()??);
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) || hh > 23 || mm > 59 || ss > 60 {
        return None;
    }
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let mp = if m > 2 { m - 3 } else { m + 9 };
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    let secs = days * 86_400 + hh * 3_600 + mm * 60 + ss;
    u64::try_from(secs).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tags_split_into_owner_and_name() {
        assert_eq!(split_tag("@qwen/qwen-3.8-27b"), ("qwen", "qwen-3.8-27b"));
        assert_eq!(split_tag("bare"), ("", "bare"));
    }
}
