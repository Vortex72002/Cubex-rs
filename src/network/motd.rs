use egui::Color32;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MotdSpan {
    pub text: String,
    pub color: [u8; 3],
    pub bold: bool,
    pub italic: bool,
    pub underline: bool,
    pub strikethrough: bool,
}

pub fn mc_color_rgb(code: char) -> Option<[u8; 3]> {
    Some(match code.to_ascii_lowercase() {
        '0' => [0x00, 0x00, 0x00],
        '1' => [0x00, 0x00, 0xAA],
        '2' => [0x00, 0xAA, 0x00],
        '3' => [0x00, 0xAA, 0xAA],
        '4' => [0xAA, 0x00, 0x00],
        '5' => [0xAA, 0x00, 0xAA],
        '6' => [0xFF, 0xAA, 0x00],
        '7' => [0xAA, 0xAA, 0xAA],
        '8' => [0x55, 0x55, 0x55],
        '9' => [0x55, 0x55, 0xFF],
        'a' => [0x55, 0xFF, 0x55],
        'b' => [0x55, 0xFF, 0xFF],
        'c' => [0xFF, 0x55, 0x55],
        'd' => [0xFF, 0x55, 0xFF],
        'e' => [0xFF, 0xFF, 0x55],
        'f' => [0xFF, 0xFF, 0xFF],
        _ => return None,
    })
}

fn parse_luanti_color_token(tok: &str) -> Option<[u8; 3]> {
    let t = tok.trim();
    if let Some(hex) = t.strip_prefix('#') {
        if hex.len() >= 6 {
            let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
            let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
            let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
            return Some([r, g, b]);
        } else if hex.len() >= 3 {
            let r = u8::from_str_radix(&hex[0..1], 16).ok()? * 17;
            let g = u8::from_str_radix(&hex[1..2], 16).ok()? * 17;
            let b = u8::from_str_radix(&hex[2..3], 16).ok()? * 17;
            return Some([r, g, b]);
        }
    }
    match t.to_ascii_lowercase().as_str() {
        "white" => Some([255, 255, 255]),
        "black" => Some([0, 0, 0]),
        "red" => Some([255, 85, 85]),
        "green" => Some([85, 255, 85]),
        "blue" => Some([85, 85, 255]),
        "yellow" => Some([255, 255, 85]),
        "aqua" | "cyan" => Some([85, 255, 255]),
        "gray" | "grey" => Some([170, 170, 170]),
        "darkgray" | "darkgrey" => Some([85, 85, 85]),
        "gold" | "orange" => Some([255, 170, 0]),
        "light_purple" | "magenta" => Some([255, 85, 255]),
        "dark_purple" | "purple" => Some([170, 0, 170]),
        "dark_green" => Some([0, 170, 0]),
        "dark_red" => Some([170, 0, 0]),
        "dark_aqua" => Some([0, 170, 170]),
        "dark_blue" => Some([0, 0, 170]),
        _ => None,
    }
}

pub fn parse_motd(raw: &str) -> Vec<MotdSpan> {
    parse_colored_text(raw, [0xAA, 0xAA, 0xAA])
}

pub fn parse_colored_text(raw: &str, default_col: [u8; 3]) -> Vec<MotdSpan> {
    let mut out = Vec::new();
    let mut color = default_col;
    let mut bold = false;
    let mut italic = false;
    let mut underline = false;
    let mut strikethrough = false;
    let mut buf = String::new();

    let chars: Vec<char> = raw.chars().collect();
    let mut i = 0usize;
    while i < chars.len() {
        let ch = chars[i];
        // Handle Luanti escape sequences: \x1b(c@#RRGGBB), \x1b(b@...), \x1b(T@...), \x1bT, \x1bF, \x1bE
        if ch == '\x1b' && i + 1 < chars.len() {
            let next = chars[i + 1];
            if next == '(' {
                if let Some(rel_end) = chars[i + 2..].iter().position(|&c| c == ')') {
                    let inner: String = chars[i + 2..i + 2 + rel_end].iter().collect();
                    if let Some(c_arg) = inner.strip_prefix("c@") {
                        if let Some(rgb) = parse_luanti_color_token(c_arg) {
                            if !buf.is_empty() {
                                out.push(MotdSpan {
                                    text: std::mem::take(&mut buf),
                                    color,
                                    bold,
                                    italic,
                                    underline,
                                    strikethrough,
                                });
                            }
                            color = rgb;
                        }
                    }
                    i += 2 + rel_end + 1;
                    continue;
                }
            } else if matches!(next, 'T' | 'F' | 'E') {
                i += 2;
                continue;
            }
        }

        if (ch == '§' || ch == '&') && i + 1 < chars.len() {
            let code = chars[i + 1].to_ascii_lowercase();
            let handled = if let Some(rgb) = mc_color_rgb(code) {
                if !buf.is_empty() {
                    out.push(MotdSpan {
                        text: std::mem::take(&mut buf),
                        color,
                        bold,
                        italic,
                        underline,
                        strikethrough,
                    });
                }
                color = rgb;
                bold = false;
                italic = false;
                underline = false;
                strikethrough = false;
                true
            } else if matches!(code, 'l' | 'o' | 'n' | 'm' | 'r' | 'k') {
                if !buf.is_empty() {
                    out.push(MotdSpan {
                        text: std::mem::take(&mut buf),
                        color,
                        bold,
                        italic,
                        underline,
                        strikethrough,
                    });
                }
                match code {
                    'l' => bold = true,
                    'o' => italic = true,
                    'n' => underline = true,
                    'm' => strikethrough = true,
                    'r' => {
                        color = default_col;
                        bold = false;
                        italic = false;
                        underline = false;
                        strikethrough = false;
                    }
                    _ => {}
                }
                true
            } else {
                false
            };
            if handled {
                i += 2;
                continue;
            }
        }
        buf.push(ch);
        i += 1;
    }

    if !buf.is_empty() {
        out.push(MotdSpan {
            text: buf,
            color,
            bold,
            italic,
            underline,
            strikethrough,
        });
    }
    out
}

pub fn strip_motd_codes(raw: &str) -> String {
    parse_motd(raw).into_iter().map(|s| s.text).collect()
}

pub fn motd_to_layout_job(raw: &str, font_size: f32) -> egui::text::LayoutJob {
    motd_to_layout_job_with_default(raw, font_size, [0xAA, 0xAA, 0xAA])
}

pub fn motd_to_layout_job_with_default(
    raw: &str,
    font_size: f32,
    default_col: [u8; 3],
) -> egui::text::LayoutJob {
    let mut job = egui::text::LayoutJob::default();
    for span in parse_colored_text(raw, default_col) {
        let mut fmt = egui::TextFormat {
            font_id: egui::FontId::monospace(font_size),
            color: Color32::from_rgb(span.color[0], span.color[1], span.color[2]),
            italics: span.italic,
            ..Default::default()
        };
        if span.underline {
            fmt.underline = egui::Stroke::new(1.0, fmt.color);
        }
        if span.strikethrough {
            fmt.strikethrough = egui::Stroke::new(1.0, fmt.color);
        }
        job.append(&span.text, 0.0, fmt);
    }
    job
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_mc_and_luanti_color_escapes() {
        let luanti = "\x1bT\x1b(c@#55FF55)Diamond Pickaxe\x1b(c@#FFFFFF)\n\x1b(c@#FF5555)Rare\x1bE";
        let stripped = strip_motd_codes(luanti);
        assert_eq!(stripped, "Diamond Pickaxe\nRare");
        let spans = parse_motd(luanti);
        assert_eq!(spans[0].color, [0x55, 0xFF, 0x55]);
        assert_eq!(spans[2].color, [0xFF, 0x55, 0x55]);
    }
}
