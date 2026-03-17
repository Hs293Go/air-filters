// ── Minimal PX4 ULog parser ───────────────────────────────────────────────────
//
// Specification: https://docs.px4.io/main/en/dev_log/ulog_file_format.html
//
// Handles exactly the messages needed to extract one named topic:
//   'B'  FLAG_BITS        — incompatibility check
//   'F'  FORMAT           — field-type definitions (including nested types)
//   'A'  ADD_LOGGED_MSG   — topic subscription (msg_id assignment)
//   'D'  DATA             — sample payloads
//
// All other message types (INFO, PARAMETER, LOGGING, DROPOUT, …) are skipped
// in O(1) — the 3-byte message header gives the body size, so we consume it
// and move on without touching its contents.

use std::collections::HashMap;
use std::io::{BufReader, Read};

const MAGIC: &[u8] = b"\x55\x4c\x6f\x67\x01\x12\x35";

const MSG_FLAG_BITS: u8 = b'B';
const MSG_FORMAT: u8 = b'F';
const MSG_ADD_LOGGED: u8 = b'A';
const MSG_DATA: u8 = b'D';

// ── Type catalogue ────────────────────────────────────────────────────────

/// Byte width of a primitive ULog type; `None` for compound (nested) types.
fn primitive_size(t: &str) -> Option<usize> {
    match t {
        "int8_t" | "uint8_t" | "bool" | "char" => Some(1),
        "int16_t" | "uint16_t" => Some(2),
        "int32_t" | "uint32_t" | "float" => Some(4),
        "int64_t" | "uint64_t" | "double" => Some(8),
        _ => None,
    }
}

/// Deserialise one primitive ULog value as `f64`.
///
/// `u64` timestamps remain exact: even a 1-year log at 1 MHz produces
/// ~3.15 × 10¹³ µs, comfortably within f64's 53-bit mantissa (~9 × 10¹⁵).
fn to_f64(bytes: &[u8], t: &str) -> f64 {
    match t {
        "int8_t" => i8::from_le_bytes([bytes[0]]) as f64,
        "uint8_t" | "bool" | "char" => bytes[0] as f64,
        "int16_t" => i16::from_le_bytes(bytes[..2].try_into().unwrap()) as f64,
        "uint16_t" => u16::from_le_bytes(bytes[..2].try_into().unwrap()) as f64,
        "int32_t" => i32::from_le_bytes(bytes[..4].try_into().unwrap()) as f64,
        "uint32_t" => u32::from_le_bytes(bytes[..4].try_into().unwrap()) as f64,
        "float" => f32::from_le_bytes(bytes[..4].try_into().unwrap()) as f64,
        "int64_t" => i64::from_le_bytes(bytes[..8].try_into().unwrap()) as f64,
        "uint64_t" => u64::from_le_bytes(bytes[..8].try_into().unwrap()) as f64,
        "double" => f64::from_le_bytes(bytes[..8].try_into().unwrap()),
        _ => f64::NAN,
    }
}

// ── Format parsing ────────────────────────────────────────────────────────

/// A flattened primitive field after nested-type expansion.
#[derive(Clone)]
pub struct Field {
    pub name: String,
    pub type_name: String,
}

type ULogFormat = (String, usize, String);

/// Parse a FORMAT message body into `(topic_name, [(type_str, array_size, field_name)])`.
///
/// Format body: `"name:type1 field1;type2[N] field2;…"`
fn parse_format(body: &[u8]) -> Option<(String, Vec<ULogFormat>)> {
    let s = std::str::from_utf8(body).ok()?;
    let (name, rest) = s.split_once(':')?;
    let mut fields = Vec::new();
    for token in rest.split(';') {
        if token.is_empty() {
            continue;
        }
        let (type_str, field_name) = token.split_once(' ')?;
        let (type_name, array_size) = if let Some(b) = type_str.find('[') {
            let e = type_str.find(']')?;
            (type_str[..b].to_owned(), type_str[b + 1..e].parse().ok()?)
        } else {
            (type_str.to_owned(), 0usize)
        };
        fields.push((type_name, array_size, field_name.to_owned()));
    }
    Some((name.to_owned(), fields))
}

/// Recursively expand nested compound types into a flat list of primitive fields.
///
/// ULog allows FORMAT messages to embed other named formats (equivalent to
/// C structs-within-structs). `_parse_nested_type` in pyulog does the same.
fn flatten(topic: &str, formats: &HashMap<String, Vec<(String, usize, String)>>) -> Vec<Field> {
    let Some(spec) = formats.get(topic) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (type_name, array_size, field_name) in spec {
        if primitive_size(type_name).is_some() {
            if *array_size > 0 {
                for i in 0..*array_size {
                    out.push(Field {
                        name: format!("{field_name}[{i}]"),
                        type_name: type_name.clone(),
                    });
                }
            } else {
                out.push(Field {
                    name: field_name.clone(),
                    type_name: type_name.clone(),
                });
            }
        } else {
            // Compound type — recurse, prepend field_name as prefix
            let nested = flatten(type_name, formats);
            if *array_size > 0 {
                for i in 0..*array_size {
                    for f in &nested {
                        out.push(Field {
                            name: format!("{field_name}[{i}].{}", f.name),
                            type_name: f.type_name.clone(),
                        });
                    }
                }
            } else {
                for f in nested {
                    out.push(Field {
                        name: format!("{field_name}.{}", f.name),
                        type_name: f.type_name.clone(),
                    });
                }
            }
        }
    }
    out
}

// ── Public API ────────────────────────────────────────────────────────────

/// Parse one topic from a ULog file; returns `field_name → Vec<f64>`.
///
/// Only `multi_id = 0` is returned. All primitive types are widened to f64.
pub fn read_topic(
    path: &str,
    topic_name: &str,
) -> Result<HashMap<String, Vec<f64>>, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(path)?;
    let mut r = BufReader::new(file);

    // File header: 7-byte magic + 1 byte version + 8-byte timestamp
    let mut hdr = [0u8; 16];
    r.read_exact(&mut hdr)?;
    if &hdr[..7] != MAGIC {
        return Err("Not a valid ULog file".into());
    }

    let mut formats: HashMap<String, Vec<(String, usize, String)>> = HashMap::new();
    let mut sub_id: Option<u16> = None;
    let mut sub_fields: Vec<Field> = Vec::new();
    let mut sub_record_size: usize = 0;
    let mut buffers: HashMap<String, Vec<f64>> = HashMap::new();

    let mut mhdr = [0u8; 3];
    loop {
        if r.read_exact(&mut mhdr).is_err() {
            break; // clean EOF
        }
        let msg_size = u16::from_le_bytes([mhdr[0], mhdr[1]]) as usize;
        let msg_type = mhdr[2];

        let mut body = vec![0u8; msg_size];
        if r.read_exact(&mut body).is_err() {
            break; // truncated — return whatever was accumulated
        }

        match msg_type {
            MSG_FLAG_BITS => {
                // Bytes [8..16] are incompatibility flags.
                // Bit 0 of incompat[0] = "data appended" — we read to EOF anyway.
                // Any other bit means the format is not supported.
                if body.len() >= 16 && (body[8] & !0x1 != 0 || body[9..16].iter().any(|&b| b != 0))
                {
                    return Err(format!(
                        "Unsupported ULog incompatibility flags: {:?}",
                        &body[8..16]
                    )
                    .into());
                }
            }

            MSG_FORMAT => {
                if let Some((name, fields)) = parse_format(&body) {
                    formats.insert(name, fields);
                }
            }

            MSG_ADD_LOGGED => {
                // Layout: uint8 multi_id | uint16 msg_id | char[] message_name
                if body.len() < 3 || sub_id.is_some() {
                    continue;
                }
                let multi_id = body[0];
                let msg_id = u16::from_le_bytes([body[1], body[2]]);
                let name = std::str::from_utf8(&body[3..])
                    .unwrap_or("")
                    .trim_end_matches('\0');

                if name == topic_name && multi_id == 0 {
                    let mut fields = flatten(name, &formats);
                    // Strip trailing _padding fields added by the PX4 struct
                    // packer to satisfy alignment requirements.
                    while fields
                        .last()
                        .map_or_else(|| false, |f| f.name.starts_with("_padding"))
                    {
                        fields.pop();
                    }
                    sub_record_size = fields
                        .iter()
                        .filter_map(|f| primitive_size(&f.type_name))
                        .sum();
                    buffers = fields
                        .iter()
                        .map(|f| (f.name.clone(), Vec::new()))
                        .collect();
                    sub_id = Some(msg_id);
                    sub_fields = fields;
                }
            }

            MSG_DATA => {
                if body.len() < 2 {
                    continue;
                }
                let msg_id = u16::from_le_bytes([body[0], body[1]]);
                if Some(msg_id) != sub_id {
                    continue;
                }
                let payload = &body[2..];
                if payload.len() < sub_record_size {
                    continue; // corrupt or short sample
                }
                let mut offset = 0;
                for field in &sub_fields {
                    let sz = match primitive_size(&field.type_name) {
                        Some(s) => s,
                        None => continue,
                    };
                    if offset + sz <= payload.len() {
                        let val = to_f64(&payload[offset..], &field.type_name);
                        buffers.get_mut(&field.name).unwrap().push(val);
                    }
                    offset += sz;
                }
            }

            _ => {} // INFO, PARAMETER, LOGGING, DROPOUT, SYNC — skip
        }
    }

    if sub_id.is_none() {
        return Err(format!("Topic '{topic_name}' not found in '{path}'").into());
    }
    Ok(buffers)
}
