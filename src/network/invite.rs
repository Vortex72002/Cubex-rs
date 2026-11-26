use anyhow::{bail, Result};
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;

pub const INVITE_PREFIX: &str = "CBX-";
pub const DEFAULT_PORT: u16 = 8090;

pub fn encode_invite(host: &str, port: u16) -> Result<String> {
    let clean = host.trim();
    if clean.is_empty() || port == 0 {
        bail!("Invalid host or port");
    }
    let payload = format!("{clean}:{port}");
    let encoded = URL_SAFE_NO_PAD.encode(payload.as_bytes());
    Ok(format!("{INVITE_PREFIX}{encoded}"))
}

pub fn decode_invite(input: &str) -> Result<(String, u16)> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        bail!("Empty address or invite code");
    }

    let target = if trimmed.len() >= INVITE_PREFIX.len()
        && trimmed[..INVITE_PREFIX.len()].eq_ignore_ascii_case(INVITE_PREFIX)
    {
        let b64 = &trimmed[INVITE_PREFIX.len()..];
        if b64.is_empty() {
            bail!("Empty CBX invite payload");
        }
        let bytes = URL_SAFE_NO_PAD.decode(b64.trim_end_matches('='))?;
        String::from_utf8(bytes)?
    } else {
        trimmed.to_string()
    };

    if let Some((h, p_str)) = target.rsplit_once(':') {
        let host = h.trim().to_string();
        let port: u16 = p_str.trim().parse()?;
        if host.is_empty() || port == 0 {
            bail!("Invalid decoded host/port");
        }
        Ok((host, port))
    } else {
        Ok((target.trim().to_string(), DEFAULT_PORT))
    }
}
