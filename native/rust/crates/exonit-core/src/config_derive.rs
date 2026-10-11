//! Transliterated from `Sources/ExoNitCore/AgentConfigDeriver.swift` in the
//! Swift reference worker. Operates on untyped JSON objects — same shape as
//! Swift's `JSONSerialization`-based `[String: Any]` dictionaries — using
//! `serde_json::Value`/`Map` here instead (Rust has no bespoke `JSONValue`
//! type to port: `serde_json::Value` already covers exactly what the Swift
//! `JSONValue` enum in `WorkerProtocolDTO.swift` exists to provide, and
//! `serde_json::Map` is BTreeMap-backed by default — i.e. it always iterates
//! and serializes in sorted-key order, matching Swift's
//! `.sortedKeys` JSON encoding option for free).

use serde_json::{Map, Value};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentConfigError {
    InvalidJsonObject(String),
    InvalidInstance { index: i64, count: i64 },
}

impl std::fmt::Display for AgentConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentConfigError::InvalidJsonObject(reason) => {
                write!(f, "Configuration must be a JSON object: {reason}")
            }
            AgentConfigError::InvalidInstance { index, count } => {
                write!(f, "Invalid instance {index} of {count}")
            }
        }
    }
}
impl std::error::Error for AgentConfigError {}

fn object_from(data: &[u8]) -> Result<Map<String, Value>, AgentConfigError> {
    let value: Value = serde_json::from_slice(data)
        .map_err(|e| AgentConfigError::InvalidJsonObject(e.to_string()))?;
    match value {
        Value::Object(map) => Ok(map),
        _ => Err(AgentConfigError::InvalidJsonObject(
            "top-level value is not an object".to_string(),
        )),
    }
}

fn encoded(object: &Map<String, Value>) -> Result<Vec<u8>, AgentConfigError> {
    // serde_json::Map is BTreeMap-backed (sorted) by default; pretty-printing
    // mirrors Swift's `.prettyPrinted` option (exact whitespace need not
    // match — nothing depends on byte-for-byte formatting, only on valid,
    // semantically-equal JSON, which every contract test checks by decoding
    // the result back).
    serde_json::to_vec_pretty(object).map_err(|e| AgentConfigError::InvalidJsonObject(e.to_string()))
}

fn integer(value: Option<&Value>) -> Option<i64> {
    value.and_then(|v| v.as_i64())
}

/// Mirrors `AgentConfigDeriver.applySingleInstanceOverrides`: a flat merge of
/// `overrides` on top of `base`, no derived fields.
pub fn apply_single_instance_overrides(
    base_config_data: &[u8],
    overrides_data: &[u8],
) -> Result<Vec<u8>, AgentConfigError> {
    let mut base = object_from(base_config_data)?;
    let overrides = object_from(overrides_data)?;
    for (key, value) in overrides {
        base.insert(key, value);
    }
    encoded(&base)
}

/// Mirrors `AgentConfigDeriver.derive`: builds a per-instance config from a
/// shared base + optional host override blob, computing a unique GUI port,
/// FITS cache directory, log file name and CPU/FPP thread share for this
/// instance out of the *base* config (not the post-override one — see the
/// inline notes below, which reproduce a deliberate Swift behaviour pinned
/// by `testMultiInstanceConfigDerivation`: "base GUI port wins over
/// override").
#[allow(clippy::too_many_arguments)]
pub fn derive(
    base_config_data: &[u8],
    overrides_data: Option<&[u8]>,
    worker_name: &str,
    instance_index: i64,
    instances: i64,
    worker_directory: &Path,
    cpu_count: i64,
) -> Result<Vec<u8>, AgentConfigError> {
    if !(instances > 0 && instance_index > 0 && instance_index <= instances) {
        return Err(AgentConfigError::InvalidInstance {
            index: instance_index,
            count: instances,
        });
    }
    let raw_base = object_from(base_config_data)?;
    let mut config = raw_base.clone();
    if let Some(overrides_data) = overrides_data {
        for (key, value) in object_from(overrides_data)? {
            config.insert(key, value);
        }
    }

    config.insert("worker_name".to_string(), Value::String(worker_name.to_string()));
    // Intentionally reads gui_port from `raw_base`, not `config`: any override
    // of gui_port is discarded, because the per-instance port must stay a
    // deterministic offset from the *shared* base value regardless of what a
    // host override blob says (see AgentConfigDeriver.swift comment + the
    // Swift test "base GUI port wins over override").
    let base_gui_port = integer(raw_base.get("gui_port")).unwrap_or(8401);
    config.insert(
        "gui_port".to_string(),
        Value::Number((base_gui_port + instance_index - 1).into()),
    );
    if instance_index > 1 {
        config.insert("gui_enabled".to_string(), Value::Bool(false));
    }

    let base_fits = raw_base
        .get("fits_cache_dir")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let cache_path: PathBuf = if base_fits.is_empty() {
        worker_directory.join(format!("fits_cache_{instance_index}"))
    } else {
        let resolved: PathBuf = if base_fits.starts_with('/') {
            PathBuf::from(base_fits)
        } else {
            worker_directory.join(base_fits)
        };
        let parent = resolved.parent().unwrap_or_else(|| Path::new(""));
        let file_name = resolved
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_default();
        parent.join(format!("{file_name}_{instance_index}"))
    };
    config.insert(
        "fits_cache_dir".to_string(),
        Value::String(cache_path.to_string_lossy().to_string()),
    );
    config.insert(
        "log_file".to_string(),
        Value::String(format!("worker-{worker_name}.log")),
    );

    let budget = ((0.75_f64 * (cpu_count.max(1) as f64)).floor()) as i64;
    let base_share = budget / instances;
    let remainder = budget % instances;
    let threads = 1.max(base_share + if instance_index <= remainder { 1 } else { 0 });
    config.insert("cpu_threads".to_string(), Value::Number(threads.into()));
    config.insert("fpp_cpu_cores".to_string(), Value::Number(threads.into()));

    encoded(&config)
}

/// Mirrors `AgentConfigDeriver.decodeOverrides(base64:)`: decodes and
/// validates (but does not merge) a `--config-overrides-b64` blob.
pub fn decode_overrides(base64_str: &str) -> Result<Vec<u8>, AgentConfigError> {
    let data = decode_base64_standard(base64_str)
        .ok_or_else(|| AgentConfigError::InvalidJsonObject("invalid base64".to_string()))?;
    let _ = object_from(&data)?;
    Ok(data)
}

/// Standard-alphabet, padded base64 decoder (strict — no unknown characters
/// tolerated), matching `Data(base64Encoded:)` called without
/// `.ignoreUnknownCharacters` in Swift. Hand-rolled because no `base64`
/// crate is in the F0 dependency allowlist (serde, serde_json, libc only) —
/// this one function does not justify pulling in a new dependency.
fn decode_base64_standard(input: &str) -> Option<Vec<u8>> {
    fn value(byte: u8) -> Option<u8> {
        match byte {
            b'A'..=b'Z' => Some(byte - b'A'),
            b'a'..=b'z' => Some(byte - b'a' + 26),
            b'0'..=b'9' => Some(byte - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let bytes = input.as_bytes();
    if bytes.is_empty() || bytes.len() % 4 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    let mut chunks = bytes.chunks_exact(4).peekable();
    while let Some(chunk) = chunks.next() {
        let is_last = chunks.peek().is_none();
        let pad = if is_last {
            chunk.iter().rev().take_while(|&&b| b == b'=').count()
        } else {
            0
        };
        if pad > 2 || chunk[..4 - pad].iter().any(|&b| b == b'=') {
            return None;
        }
        let mut sextets = [0u8; 4];
        for (i, &b) in chunk.iter().enumerate() {
            sextets[i] = if i >= 4 - pad { 0 } else { value(b)? };
        }
        let combined = (sextets[0] as u32) << 18
            | (sextets[1] as u32) << 12
            | (sextets[2] as u32) << 6
            | (sextets[3] as u32);
        out.push((combined >> 16) as u8);
        if pad < 2 {
            out.push((combined >> 8) as u8);
        }
        if pad < 1 {
            out.push(combined as u8);
        }
    }
    Some(out)
}
