use egui::{Align2, Color32, ColorImage, FontId, Painter, Pos2, Rect, Response, Stroke, TextureHandle, TextureId, TextureOptions, Ui, Vec2};
use image::imageops::FilterType;
use image::{Rgba, RgbaImage};
use std::collections::HashMap;
use std::sync::Arc;

use crate::network::motd::motd_to_layout_job_with_default;
use crate::registry::EngineRegistries;
use crate::textures::TextureAtlas;

/// Strip Luanti translation/formatting escapes (`\x1b(T@...)`, `\x1b(F@...)`,
/// `\x1b(c@...)`, `\x1bE`, `\x1bF`, `\x1bT`, ANSI CSI) and Minecraft `§.` color/style codes.
pub fn strip_luanti_escapes(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0usize;
    while i < chars.len() {
        let ch = chars[i];
        if ch == '\x1b' {
            if i + 1 < chars.len() {
                let next = chars[i + 1];
                if next == '(' {
                    if let Some(rel_end) = chars[i + 2..].iter().position(|&c| c == ')') {
                        i += 2 + rel_end + 1;
                        continue;
                    }
                } else if next == '[' {
                    let mut j = i + 2;
                    while j < chars.len() && !('\x40'..='\x7e').contains(&chars[j]) {
                        j += 1;
                    }
                    i = if j < chars.len() { j + 1 } else { chars.len() };
                    continue;
                } else if next.is_ascii_alphabetic() {
                    i += 2;
                    continue;
                }
            }
            i += 1;
            continue;
        }
        if ch == '§' {
            if i + 1 < chars.len() {
                i += 2;
            } else {
                i += 1;
            }
            continue;
        }
        out.push(ch);
        i += 1;
    }
    out
}

/// Classic translucent #00000099 pause & overlay backdrop color.
pub const PAUSE_OVERLAY_COLOR: Color32 = Color32::from_rgba_premultiplied(0, 0, 0, 0x99);

pub type ClickSoundCallback = Arc<dyn Fn() + Send + Sync>;

pub struct McTheme {
    pub dirt_tile: TextureHandle,
    pub deepslate_tile: TextureHandle,
    pub stone_btn_norm: TextureHandle,
    pub stone_btn_hover: TextureHandle,
    pub stone_btn_green: TextureHandle,
    pub stone_btn_disabled: TextureHandle,
    pub stone_btn_pressed: TextureHandle,
    pub cubex_logo: TextureHandle,
    pub on_click: Option<ClickSoundCallback>,
}

impl McTheme {
    pub fn load(ctx: &egui::Context) -> Self {
        Self {
            dirt_tile: upload_nearest(ctx, "mc_dirt", gen_dirt_tile(false)),
            deepslate_tile: upload_nearest(ctx, "mc_deepslate", gen_dirt_tile(true)),
            stone_btn_norm: upload_nearest(ctx, "mc_btn_norm", gen_stone_button(0)),
            stone_btn_hover: upload_nearest(ctx, "mc_btn_hover", gen_stone_button(1)),
            stone_btn_green: upload_nearest(ctx, "mc_btn_green", gen_stone_button(2)),
            stone_btn_disabled: upload_nearest(ctx, "mc_btn_disabled", gen_stone_button(3)),
            stone_btn_pressed: upload_nearest(ctx, "mc_btn_pressed", gen_stone_button(4)),
            cubex_logo: upload_nearest(ctx, "mc_cubex_logo", gen_cubex_logo()),
            on_click: None,
        }
    }

    pub fn with_click_callback(mut self, cb: ClickSoundCallback) -> Self {
        self.on_click = Some(cb);
        self
    }

    pub fn set_click_callback(&mut self, cb: ClickSoundCallback) {
        self.on_click = Some(cb);
    }

    pub fn play_click(&self) {
        if let Some(ref cb) = self.on_click {
            cb();
        }
    }
}

fn upload_nearest(ctx: &egui::Context, name: &str, img: ColorImage) -> TextureHandle {
    ctx.load_texture(name, img, TextureOptions::NEAREST)
}

fn upload_nearest_rgba(ctx: &egui::Context, name: &str, img: &RgbaImage) -> TextureHandle {
    let ci = ColorImage::from_rgba_unmultiplied(
        [img.width().max(1) as usize, img.height().max(1) as usize],
        img.as_raw(),
    );
    ctx.load_texture(name, ci, TextureOptions::NEAREST)
}

/// CPU-side 9x9/16x16 HUD icon bitmaps (full, half, and background containers).
#[derive(Clone, Debug)]
pub struct HudIconImages {
    pub heart_full: RgbaImage,
    pub heart_half: RgbaImage,
    pub heart_bg: RgbaImage,
    pub hunger_full: RgbaImage,
    pub hunger_half: RgbaImage,
    pub hunger_bg: RgbaImage,
    pub armor_full: RgbaImage,
    pub armor_half: RgbaImage,
    pub armor_bg: RgbaImage,
}

/// GPU nearest-neighbor HUD textures cached on `ItemIconCache` / `egui::Context`.
#[derive(Clone)]
pub struct HudIconTextures {
    pub heart_full: TextureHandle,
    pub heart_half: TextureHandle,
    pub heart_bg: TextureHandle,
    pub hunger_full: TextureHandle,
    pub hunger_half: TextureHandle,
    pub hunger_bg: TextureHandle,
    pub armor_full: TextureHandle,
    pub armor_half: TextureHandle,
    pub armor_bg: TextureHandle,
    pub from_atlas: bool,
}

impl HudIconTextures {
    pub fn load(ctx: &egui::Context, atlas: Option<&TextureAtlas>) -> Self {
        let imgs = build_hud_icon_images(atlas);
        Self {
            heart_full: upload_nearest_rgba(ctx, "hud_heart_full", &imgs.heart_full),
            heart_half: upload_nearest_rgba(ctx, "hud_heart_half", &imgs.heart_half),
            heart_bg: upload_nearest_rgba(ctx, "hud_heart_bg", &imgs.heart_bg),
            hunger_full: upload_nearest_rgba(ctx, "hud_hunger_full", &imgs.hunger_full),
            hunger_half: upload_nearest_rgba(ctx, "hud_hunger_half", &imgs.hunger_half),
            hunger_bg: upload_nearest_rgba(ctx, "hud_hunger_bg", &imgs.hunger_bg),
            armor_full: upload_nearest_rgba(ctx, "hud_armor_full", &imgs.armor_full),
            armor_half: upload_nearest_rgba(ctx, "hud_armor_half", &imgs.armor_half),
            armor_bg: upload_nearest_rgba(ctx, "hud_armor_bg", &imgs.armor_bg),
            from_atlas: atlas.is_some(),
        }
    }
}

fn rasterize_9x9(rows: [&str; 9], palette: &[(u8, [u8; 4])]) -> RgbaImage {
    let mut img = RgbaImage::from_pixel(9, 9, Rgba([0, 0, 0, 0]));
    for (y, row) in rows.iter().enumerate() {
        for (x, ch) in row.bytes().take(9).enumerate() {
            if let Some((_, col)) = palette.iter().find(|(k, _)| *k == ch) {
                img.put_pixel(x as u32, y as u32, Rgba(*col));
            }
        }
    }
    img
}

fn fallback_heart_9x9() -> RgbaImage {
    rasterize_9x9(
        [
            ".........",
            ".OO...OO.",
            "ORRO.ORRO",
            "OWRROORRO",
            "ORRROORRO",
            "ORRRRRRRO",
            ".ORRRRRO.",
            "..ORRRO..",
            "...OOO...",
        ],
        &[
            (b'O', [120, 18, 18, 255]),
            (b'R', [229, 37, 37, 255]),
            (b'W', [255, 224, 224, 255]),
        ],
    )
}

fn fallback_hunger_9x9() -> RgbaImage {
    rasterize_9x9(
        [
            ".OOO.....",
            "OMHMO....",
            "OMHHMO...",
            "OMMMMO...",
            ".OMMMMO..",
            ".OOMMMO..",
            "...OOOB..",
            "......BB.",
            ".......BB",
        ],
        &[
            (b'O', [120, 72, 42, 255]),
            (b'M', [196, 134, 78, 255]),
            (b'H', [224, 182, 128, 255]),
            (b'B', [233, 224, 201, 255]),
        ],
    )
}

fn fallback_armor_9x9() -> RgbaImage {
    rasterize_9x9(
        [
            ".OOOOOOO.",
            "OMHWWWHMO",
            "OMMMMMMMO",
            "OMMMMMMMO",
            "OMMMMMMMO",
            "OMMMMMMMO",
            ".OMMMMMO.",
            "..OMMMO..",
            "...OOO...",
        ],
        &[
            (b'O', [84, 81, 79, 255]),
            (b'M', [175, 172, 165, 255]),
            (b'H', [205, 203, 198, 255]),
            (b'W', [232, 232, 228, 255]),
        ],
    )
}

/// Derive a crisp darkened charcoal container silhouette from a foreground icon.
pub fn derive_bg_silhouette(fg: &RgbaImage) -> RgbaImage {
    let w = fg.width().max(1);
    let h = fg.height().max(1);
    let mut out = RgbaImage::from_pixel(w, h, Rgba([0, 0, 0, 0]));
    let is_solid = |x: i32, y: i32| -> bool {
        if x < 0 || y < 0 || x >= w as i32 || y >= h as i32 {
            return false;
        }
        fg.get_pixel(x as u32, y as u32).0[3] >= 128
    };
    for y in 0..h {
        for x in 0..w {
            let ix = x as i32;
            let iy = y as i32;
            if !is_solid(ix, iy) {
                continue;
            }
            let edge = !is_solid(ix - 1, iy)
                || !is_solid(ix + 1, iy)
                || !is_solid(ix, iy - 1)
                || !is_solid(ix, iy + 1);
            let col = if edge {
                Rgba([20, 18, 22, 255])
            } else {
                Rgba([54, 40, 44, 255])
            };
            out.put_pixel(x, y, col);
        }
    }
    out
}

/// Returns true when at least one opaque/semi-opaque pixel has visible color (not pure black).
fn has_visible_color(img: &RgbaImage) -> bool {
    img.pixels()
        .any(|p| p.0[3] >= 16 && (p.0[0] >= 24 || p.0[1] >= 24 || p.0[2] >= 24))
}

/// Composite left (`keep_right=false`) or right (`keep_right=true`) half of `fg` onto `bg`.
pub fn synthesize_half_icon(fg: &RgbaImage, bg: &RgbaImage, keep_right: bool) -> RgbaImage {
    let w = fg.width().max(1);
    let h = fg.height().max(1);
    let mut out = if bg.dimensions() == (w, h) {
        bg.clone()
    } else {
        image::imageops::resize(bg, w, h, FilterType::Nearest)
    };
    for y in 0..h {
        for x in 0..w {
            let in_half = if keep_right { x >= w / 2 } else { x < w.div_ceil(2) };
            if !in_half {
                continue;
            }
            let fp = fg.get_pixel(x, y).0;
            if fp[3] == 0 {
                continue;
            }
            if fp[3] >= 250 {
                out.put_pixel(x, y, Rgba(fp));
            } else {
                let bp = out.get_pixel(x, y).0;
                let sa = fp[3] as u16;
                let inv = 255 - sa;
                out.put_pixel(
                    x,
                    y,
                    Rgba([
                        ((fp[0] as u16 * sa + bp[0] as u16 * inv) / 255) as u8,
                        ((fp[1] as u16 * sa + bp[1] as u16 * inv) / 255) as u8,
                        ((fp[2] as u16 * sa + bp[2] as u16 * inv) / 255) as u8,
                        bp[3].max(fp[3]),
                    ]),
                );
            }
        }
    }
    out
}

/// Build/load all 9 pixel-art HUD icon bitmaps from `TextureAtlas` (with crisp 9x9 fallbacks).
pub fn build_hud_icon_images(atlas: Option<&TextureAtlas>) -> HudIconImages {
    let get_first = |names: &[&str]| -> Option<RgbaImage> {
        let a = atlas?;
        for name in names {
            if let Some(img) = a.get_rgba(name) {
                return Some(img.clone());
            }
        }
        None
    };
    let get_first_colored = |names: &[&str]| -> Option<RgbaImage> {
        let a = atlas?;
        for name in names {
            if let Some(img) = a.get_rgba(name) {
                if has_visible_color(img) {
                    return Some(img.clone());
                }
            }
        }
        None
    };

    let heart_full =
        get_first_colored(&["heart.png", "hud_heart_fg.png"]).unwrap_or_else(fallback_heart_9x9);
    let heart_bg = get_first(&["heart_bg.png", "hud_heart_bg.png"])
        .unwrap_or_else(|| derive_bg_silhouette(&heart_full));
    let heart_half = get_first_colored(&["heart_half.png", "hud_heart_half.png"])
        .unwrap_or_else(|| synthesize_half_icon(&heart_full, &heart_bg, false));

    let hunger_full = get_first_colored(&["hbhunger_icon.png", "hud_hunger_fg.png"])
        .unwrap_or_else(fallback_hunger_9x9);
    let hunger_bg = get_first(&["hbhunger_bgicon.png", "hud_hunger_bg.png"])
        .unwrap_or_else(|| derive_bg_silhouette(&hunger_full));
    let hunger_half = get_first_colored(&["hbhunger_icon_half.png", "hud_hunger_half.png"])
        .unwrap_or_else(|| synthesize_half_icon(&hunger_full, &hunger_bg, true));

    let armor_full = get_first_colored(&["hbarmor_icon.png", "hud_armor_fg.png"])
        .unwrap_or_else(fallback_armor_9x9);
    let armor_bg = get_first(&["hbarmor_bgicon.png", "hud_armor_bg.png"])
        .unwrap_or_else(|| derive_bg_silhouette(&armor_full));
    let armor_half = get_first_colored(&["hbarmor_icon_half.png", "hud_armor_half.png"])
        .unwrap_or_else(|| synthesize_half_icon(&armor_full, &armor_bg, false));

    HudIconImages {
        heart_full,
        heart_half,
        heart_bg,
        hunger_full,
        hunger_half,
        hunger_bg,
        armor_full,
        armor_half,
        armor_bg,
    }
}

/// On-demand cache of nearest-neighbor egui item icons, player skins, server icons, and HUD icons.
#[derive(Default)]
pub struct ItemIconCache {
    pub icons: HashMap<String, Option<TextureHandle>>,
    pub skin_tex: Option<TextureHandle>,
    pub server_icons: HashMap<String, TextureHandle>,
    pub hud_icons: Option<HudIconTextures>,
}

impl ItemIconCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn ensure_hud_icons(
        &mut self,
        ctx: &egui::Context,
        atlas: Option<&TextureAtlas>,
    ) -> &HudIconTextures {
        let need_reload = match (&self.hud_icons, atlas.is_some()) {
            (None, _) => true,
            (Some(tex), true) if !tex.from_atlas => true,
            _ => false,
        };
        if need_reload {
            self.hud_icons = Some(HudIconTextures::load(ctx, atlas));
        }
        self.hud_icons.as_ref().expect("hud_icons loaded")
    }

    pub fn insert_rgba(
        &mut self,
        ctx: &egui::Context,
        key: &str,
        rgba: &[u8],
        width: u32,
        height: u32,
    ) -> TextureId {
        let img = ColorImage::from_rgba_unmultiplied([width as usize, height as usize], rgba);
        let handle = ctx.load_texture(format!("icon_{key}"), img, TextureOptions::NEAREST);
        let id = handle.id();
        self.icons.insert(key.to_string(), Some(handle));
        id
    }

    pub fn get_cached(&self, key: &str) -> Option<TextureId> {
        self.icons.get(key).and_then(|o| o.as_ref().map(|h| h.id()))
    }

    pub fn get_or_insert_server_icon(
        &mut self,
        ctx: &egui::Context,
        seed: &str,
        rgba16: &[u8],
    ) -> TextureId {
        if let Some(h) = self.server_icons.get(seed) {
            return h.id();
        }
        let img = ColorImage::from_rgba_unmultiplied([16, 16], rgba16);
        let h = ctx.load_texture(format!("srv_icon_{seed}"), img, TextureOptions::NEAREST);
        let id = h.id();
        self.server_icons.insert(seed.to_string(), h);
        id
    }

    pub fn set_player_skin_rgba(&mut self, ctx: &egui::Context, rgba: &[u8], width: u32, height: u32) {
        let img = ColorImage::from_rgba_unmultiplied([width as usize, height as usize], rgba);
        self.skin_tex = Some(ctx.load_texture("mc_player_skin", img, TextureOptions::NEAREST));
    }

    pub fn ensure_player_skin(&mut self, ctx: &egui::Context, atlas: &TextureAtlas) -> Option<&TextureHandle> {
        if self.skin_tex.is_none() {
            let img = atlas.get_player_skin_rgba();
            self.set_player_skin_rgba(ctx, img.as_raw(), img.width(), img.height());
        }
        self.skin_tex.as_ref()
    }

    pub fn get_or_load_item(
        &mut self,
        ctx: &egui::Context,
        item_name: &str,
        atlas: &TextureAtlas,
        reg: &EngineRegistries,
    ) -> Option<TextureId> {
        if item_name.is_empty() {
            return None;
        }
        if let Some(entry) = self.icons.get(item_name) {
            return entry.as_ref().map(|h| h.id());
        }
        let loaded = atlas.get_item_rgba(item_name, reg).map(|img| {
            let ci = ColorImage::from_rgba_unmultiplied(
                [img.width() as usize, img.height() as usize],
                img.as_raw(),
            );
            ctx.load_texture(format!("item_icon_{item_name}"), ci, TextureOptions::NEAREST)
        });
        let id = loaded.as_ref().map(|h| h.id());
        self.icons.insert(item_name.to_string(), loaded);
        id
    }

    pub fn clear(&mut self) {
        self.icons.clear();
        self.skin_tex = None;
        self.hud_icons = None;
    }
}

fn lcg(seed: &mut u32) -> u32 {
    *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
    *seed >> 16
}

fn gen_dirt_tile(deepslate: bool) -> ColorImage {
    let mut img = ColorImage::new([16, 16], Color32::BLACK);
    let mut seed = if deepslate { 99u32 } else { 42u32 };
    let pal: &[[u8; 3]] = if deepslate {
        &[[32, 32, 36], [26, 26, 30], [38, 38, 44], [22, 22, 26]]
    } else {
        &[[52, 36, 24], [46, 32, 21], [58, 40, 27], [40, 28, 18]]
    };
    for y in 0..16 {
        for x in 0..16 {
            let c = pal[(lcg(&mut seed) as usize) % pal.len()];
            img.pixels[y * 16 + x] = Color32::from_rgb(c[0], c[1], c[2]);
        }
    }
    img
}

fn gen_stone_button(variant: u8) -> ColorImage {
    let w = 200usize;
    let h = 20usize;
    let mut img = ColorImage::new([w, h], Color32::BLACK);
    let mut seed = 1337u32;

    // Authentic Minecraft stone button colors:
    // #727272 (114,114,114) face, #AAAAAA (170,170,170) top/left bevel, #373737 (55,55,55) bottom/right bevel
    let (base, hi, lo, border): ([u8; 3], [u8; 3], [u8; 3], [u8; 3]) = match variant {
        1 => ([126, 136, 191], [196, 204, 245], [55, 60, 96], [255, 255, 255]), // hover
        2 => ([60, 133, 39], [112, 196, 86], [32, 78, 20], [16, 24, 14]),     // green
        3 => ([52, 52, 52], [68, 68, 68], [34, 34, 34], [24, 24, 24]),          // disabled
        4 => ([92, 98, 134], [55, 58, 84], [146, 152, 192], [230, 236, 255]),   // pressed
        _ => ([114, 114, 114], [170, 170, 170], [55, 55, 55], [24, 24, 24]),    // normal
    };

    for y in 0..h {
        for x in 0..w {
            let c = if x == 0 || y == 0 || x == w - 1 || y == h - 1 {
                border
            } else if x == 1 || y == 1 {
                hi
            } else if y >= h - 3 || x == w - 2 {
                lo
            } else {
                let j = ((lcg(&mut seed) % 13) as i16) - 6;
                [
                    (base[0] as i16 + j).clamp(0, 255) as u8,
                    (base[1] as i16 + j).clamp(0, 255) as u8,
                    (base[2] as i16 + j).clamp(0, 255) as u8,
                ]
            };
            img.pixels[y * w + x] = Color32::from_rgb(c[0], c[1], c[2]);
        }
    }
    img
}

fn gen_cubex_logo() -> ColorImage {
    const GLYPHS: [&[&str]; 5] = [
        &[" 1111", "11  1", "11   ", "11   ", "11   ", "11  1", " 1111"], // C
        &["11  1", "11  1", "11  1", "11  1", "11  1", "11  1", " 111 "], // U
        &["1111 ", "11  1", "11  1", "1111 ", "11  1", "11  1", "1111 "], // B
        &["11111", "11   ", "11   ", "1111 ", "11   ", "11   ", "11111"], // E
        &["11  1", "11  1", " 111 ", "  1  ", " 111 ", "11  1", "11  1"], // X
    ];

    let ps = 8usize;
    let gw = 5usize;
    let gh = 7usize;
    let gap = 1usize;
    let cols = GLYPHS.len() * gw + (GLYPHS.len() - 1) * gap;
    let w = (cols + 4) * ps;
    let h = (gh + 4) * ps;
    let mut img = ColorImage::new([w, h], Color32::TRANSPARENT);

    let mut fill_block = |bx: usize, by: usize, col: Color32| {
        for py in 0..ps {
            for px in 0..ps {
                let x = bx + px;
                let y = by + py;
                if x < w && y < h {
                    img.pixels[y * w + x] = col;
                }
            }
        }
    };

    // 3D Stone Shadow Extrusion
    for (gi, g) in GLYPHS.iter().enumerate() {
        let ox = (1 + gi * (gw + gap)) * ps;
        for (r, row) in g.iter().enumerate() {
            for (c, ch) in row.bytes().enumerate() {
                if ch == b'1' {
                    fill_block(ox + c * ps + ps / 2, ps + r * ps + ps, Color32::from_rgb(38, 38, 44));
                }
            }
        }
    }

    // Front Stone + Emerald X face
    for (gi, g) in GLYPHS.iter().enumerate() {
        let ox = (1 + gi * (gw + gap)) * ps;
        let is_x = gi == 4;
        for (r, row) in g.iter().enumerate() {
            for (c, ch) in row.bytes().enumerate() {
                if ch != b'1' {
                    continue;
                }
                let x0 = ox + c * ps;
                let y0 = ps + r * ps;
                let base = if is_x {
                    Color32::from_rgb(74, 198 - (r as u8 * 10), 92)
                } else {
                    let v = 224u8.saturating_sub(r as u8 * 11);
                    Color32::from_rgb(v, v, v.saturating_add(4))
                };
                for py in 0..ps {
                    for px in 0..ps {
                        let col = if px == 0 || py == 0 || px == ps - 1 || py == ps - 1 {
                            Color32::from_rgb(24, 24, 28)
                        } else if px == 1 || py == 1 {
                            if is_x { Color32::from_rgb(156, 245, 168) } else { Color32::WHITE }
                        } else if px == ps - 2 || py == ps - 2 {
                            if is_x { Color32::from_rgb(28, 96, 42) } else { Color32::from_rgb(108, 108, 114) }
                        } else {
                            base
                        };
                        img.pixels[(y0 + py) * w + (x0 + px)] = col;
                    }
                }
            }
        }
    }
    img
}

pub fn mc_shadow_text(
    painter: &Painter,
    pos: Pos2,
    align: Align2,
    text: &str,
    size: f32,
    color: Color32,
) {
    let font = FontId::monospace(size);
    painter.text(pos + Vec2::new(2.0, 2.0), align, text, font.clone(), Color32::from_rgb(36, 36, 36));
    painter.text(pos, align, text, font, color);
}

pub fn paint_tiled_bg(painter: &Painter, rect: Rect, tile: &TextureHandle) {
    let uv = Rect::from_min_max(
        Pos2::ZERO,
        Pos2::new((rect.width() / 64.0).max(1.0), (rect.height() / 64.0).max(1.0)),
    );
    painter.image(tile.id(), rect, uv, Color32::WHITE);
}

/// Authentic Minecraft subscreen tiled dirt/deepslate background with dark header/footer bands and 4px shadow lines.
pub fn paint_dirt_screen_bg(
    painter: &Painter,
    screen: Rect,
    theme: &McTheme,
    deepslate: bool,
    header_h: f32,
    footer_h: f32,
) {
    let tile = if deepslate { &theme.deepslate_tile } else { &theme.dirt_tile };
    paint_tiled_bg(painter, screen, tile);
    painter.rect_filled(screen, 0.0, Color32::from_rgba_unmultiplied(0, 0, 0, 80));

    let hdr = Rect::from_min_size(screen.min, Vec2::new(screen.width(), header_h));
    painter.rect_filled(hdr, 0.0, Color32::from_rgba_unmultiplied(0, 0, 0, 120));
    painter.rect_filled(
        Rect::from_min_size(Pos2::new(screen.left(), hdr.bottom()), Vec2::new(screen.width(), 4.0)),
        0.0,
        Color32::from_rgba_unmultiplied(0, 0, 0, 160),
    );

    let ftr = Rect::from_min_size(
        Pos2::new(screen.left(), (screen.bottom() - footer_h).max(hdr.bottom())),
        Vec2::new(screen.width(), footer_h),
    );
    painter.rect_filled(ftr, 0.0, Color32::from_rgba_unmultiplied(0, 0, 0, 120));
    painter.rect_filled(
        Rect::from_min_size(Pos2::new(screen.left(), ftr.top() - 4.0), Vec2::new(screen.width(), 4.0)),
        0.0,
        Color32::from_rgba_unmultiplied(0, 0, 0, 160),
    );
}

/// #00000099 semi-transparent pause backdrop overlay.
pub fn paint_pause_overlay(painter: &Painter, rect: Rect) {
    painter.rect_filled(rect, 0.0, PAUSE_OVERLAY_COLOR);
}

/// Paint a 9-slice stone texture rect so bevel borders stay razor-sharp at any aspect ratio.
pub fn paint_nine_slice(painter: &Painter, rect: Rect, tex: &TextureHandle) {
    let sz = tex.size();
    let tw = sz[0].max(1) as f32;
    let th = sz[1].max(1) as f32;
    let cap_u = (4.0 / tw).min(0.25);
    let cap_v = (4.0 / th).min(0.25);
    let cap_x = (rect.width() * 0.25).min(8.0);
    let cap_y = (rect.height() * 0.25).min(8.0);

    let xs = [rect.left(), rect.left() + cap_x, rect.right() - cap_x, rect.right()];
    let ys = [rect.top(), rect.top() + cap_y, rect.bottom() - cap_y, rect.bottom()];
    let us = [0.0, cap_u, 1.0 - cap_u, 1.0];
    let vs = [0.0, cap_v, 1.0 - cap_v, 1.0];

    for ry in 0..3 {
        for rx in 0..3 {
            let dst = Rect::from_min_max(Pos2::new(xs[rx], ys[ry]), Pos2::new(xs[rx + 1], ys[ry + 1]));
            if dst.width() <= 0.0 || dst.height() <= 0.0 {
                continue;
            }
            let uv = Rect::from_min_max(Pos2::new(us[rx], vs[ry]), Pos2::new(us[rx + 1], vs[ry + 1]));
            painter.image(tex.id(), dst, uv, Color32::WHITE);
        }
    }
}

pub fn mc_button(ui: &mut Ui, theme: &McTheme, text: &str, width: f32, height: f32) -> Response {
    mc_button_styled(ui, theme, text, width, height, false)
}

pub fn mc_button_green(ui: &mut Ui, theme: &McTheme, text: &str, width: f32, height: f32) -> Response {
    mc_button_styled(ui, theme, text, width, height, true)
}

pub fn mc_button_styled(
    ui: &mut Ui,
    theme: &McTheme,
    text: &str,
    width: f32,
    height: f32,
    green: bool,
) -> Response {
    mc_button_with_sound(ui, theme, text, width, height, green, true, None)
}

#[allow(clippy::too_many_arguments)]
pub fn mc_button_with_sound(
    ui: &mut Ui,
    theme: &McTheme,
    text: &str,
    width: f32,
    height: f32,
    green: bool,
    enabled: bool,
    extra_sound: Option<&dyn Fn()>,
) -> Response {
    let sense = if enabled { egui::Sense::click() } else { egui::Sense::hover() };
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(width, height), sense);
    let hovered = enabled && resp.hovered();
    let pressed = enabled && resp.is_pointer_button_down_on();
    let tex = if !enabled {
        &theme.stone_btn_disabled
    } else if pressed {
        &theme.stone_btn_pressed
    } else if hovered {
        &theme.stone_btn_hover
    } else if green {
        &theme.stone_btn_green
    } else {
        &theme.stone_btn_norm
    };

    let painter = ui.painter();
    paint_nine_slice(painter, rect, tex);

    // Crisp outer bevel border & #FFFFFF hover border
    let border_col = if hovered {
        Color32::WHITE
    } else {
        Color32::from_rgb(30, 30, 30)
    };
    painter.rect_stroke(rect, 0.0, Stroke::new(if hovered { 1.5 } else { 1.0 }, border_col), egui::StrokeKind::Inside);

    let fg = if !enabled {
        Color32::from_rgb(160, 160, 160)
    } else if hovered {
        Color32::from_rgb(255, 255, 160)
    } else {
        Color32::WHITE
    };
    let text_pos = if pressed { rect.center() + Vec2::new(0.0, 1.0) } else { rect.center() };
    let font_sz = if height < 30.0 { 12.0 } else { 14.5 };
    mc_shadow_text(painter, text_pos, Align2::CENTER_CENTER, text, font_sz, fg);

    if enabled && resp.clicked() {
        theme.play_click();
        if let Some(cb) = extra_sound {
            cb();
        }
    }
    resp
}

pub fn mc_slider_f32(
    ui: &mut Ui,
    theme: &McTheme,
    label: &str,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    width: f32,
    height: f32,
    format_fn: impl Fn(f32) -> String,
) -> bool {
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(width, height), egui::Sense::click_and_drag());
    let mut changed = false;
    let min_v = *range.start();
    let max_v = *range.end();
    let span = (max_v - min_v).max(0.0001);

    if (resp.dragged() || resp.clicked()) && resp.interact_pointer_pos().is_some() {
        let px = resp.interact_pointer_pos().unwrap().x;
        let t = ((px - rect.left() - 6.0) / (rect.width() - 12.0).max(1.0)).clamp(0.0, 1.0);
        let new_v = min_v + t * span;
        if (new_v - *value).abs() > 1e-4 {
            *value = new_v;
            changed = true;
        }
    }
    if resp.clicked() {
        theme.play_click();
    }

    let painter = ui.painter();
    paint_nine_slice(painter, rect, &theme.stone_btn_disabled);
    painter.rect_stroke(rect, 0.0, Stroke::new(1.5, Color32::BLACK), egui::StrokeKind::Inside);

    let t = ((*value - min_v) / span).clamp(0.0, 1.0);
    let knob_w = 16.0;
    let knob_x = rect.left() + 2.0 + t * (rect.width() - knob_w - 4.0).max(0.0);
    let knob_rect = Rect::from_min_size(Pos2::new(knob_x, rect.top() + 2.0), Vec2::new(knob_w, rect.height() - 4.0));
    let knob_tex = if resp.hovered() || resp.dragged() {
        &theme.stone_btn_hover
    } else {
        &theme.stone_btn_norm
    };
    paint_nine_slice(painter, knob_rect, knob_tex);
    if resp.hovered() || resp.dragged() {
        painter.rect_stroke(rect, 0.0, Stroke::new(1.5, Color32::WHITE), egui::StrokeKind::Inside);
    }

    let txt = format!("{}: {}", label, format_fn(*value));
    let fg = if resp.hovered() { Color32::from_rgb(255, 255, 160) } else { Color32::WHITE };
    mc_shadow_text(painter, rect.center(), Align2::CENTER_CENTER, &txt, 14.0, fg);
    changed
}

/// Paint an authentic #C6C6C6 beveled Minecraft inventory/crafting/furnace window panel.
pub fn paint_mc_panel(painter: &Painter, rect: Rect) {
    painter.rect_filled(rect, 2.0, Color32::from_rgb(198, 198, 198));
    painter.rect_stroke(rect, 2.0, Stroke::new(2.0, Color32::BLACK), egui::StrokeKind::Inside);
    // Top/left white highlight bevel
    painter.line_segment(
        [rect.left_top() + Vec2::new(2.5, 2.5), rect.right_top() + Vec2::new(-3.0, 2.5)],
        Stroke::new(2.0, Color32::WHITE),
    );
    painter.line_segment(
        [rect.left_top() + Vec2::new(2.5, 2.5), rect.left_bottom() + Vec2::new(2.5, -3.0)],
        Stroke::new(2.0, Color32::WHITE),
    );
    // Bottom/right #555555 shadow bevel
    painter.line_segment(
        [rect.left_bottom() + Vec2::new(3.0, -2.5), rect.right_bottom() + Vec2::new(-2.5, -2.5)],
        Stroke::new(2.5, Color32::from_rgb(85, 85, 85)),
    );
    painter.line_segment(
        [rect.right_top() + Vec2::new(-2.5, 3.0), rect.right_bottom() + Vec2::new(-2.5, -2.5)],
        Stroke::new(2.5, Color32::from_rgb(85, 85, 85)),
    );
}

/// Paint a recessed Minecraft inventory slot (#8B8B8B fill, #373737 top/left, #FFFFFF bottom/right).
pub fn paint_mc_slot(painter: &Painter, rect: Rect, hovered: bool) {
    painter.rect_filled(rect, 0.0, Color32::from_rgb(139, 139, 139));
    painter.line_segment(
        [rect.left_top() + Vec2::new(0.5, 0.5), rect.right_top() + Vec2::new(-0.5, 0.5)],
        Stroke::new(1.5, Color32::from_rgb(55, 55, 55)),
    );
    painter.line_segment(
        [rect.left_top() + Vec2::new(0.5, 0.5), rect.left_bottom() + Vec2::new(0.5, -0.5)],
        Stroke::new(1.5, Color32::from_rgb(55, 55, 55)),
    );
    painter.line_segment(
        [rect.left_bottom() + Vec2::new(0.5, -0.5), rect.right_bottom() + Vec2::new(-0.5, -0.5)],
        Stroke::new(1.5, Color32::WHITE),
    );
    painter.line_segment(
        [rect.right_top() + Vec2::new(-0.5, 0.5), rect.right_bottom() + Vec2::new(-0.5, -0.5)],
        Stroke::new(1.5, Color32::WHITE),
    );
    if hovered {
        painter.rect_filled(rect.shrink(1.5), 0.0, Color32::from_rgba_unmultiplied(255, 255, 255, 95));
    }
}

/// Paint an item icon, stack count, and durability bar inside a slot rectangle.
pub fn paint_item_in_slot(
    painter: &Painter,
    rect: Rect,
    icon_tex: Option<TextureId>,
    item_name: &str,
    count: u16,
    wear: u16,
) {
    if item_name.is_empty() || count == 0 {
        return;
    }
    let icon_rect = rect.shrink(4.0);
    if let Some(tid) = icon_tex {
        painter.image(
            tid,
            icon_rect,
            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            Color32::WHITE,
        );
    } else {
        let short = item_name.split(':').nth(1).unwrap_or(item_name);
        let abbr = &short[..short.len().min(5)];
        mc_shadow_text(painter, rect.center(), Align2::CENTER_CENTER, abbr, 10.0, Color32::WHITE);
    }

    if count > 1 {
        mc_shadow_text(
            painter,
            rect.right_bottom() - Vec2::new(2.0, 1.0),
            Align2::RIGHT_BOTTOM,
            &count.to_string(),
            12.0,
            Color32::WHITE,
        );
    }

    if wear > 0 {
        let frac = (1.0 - (wear as f32 / 65535.0)).clamp(0.0, 1.0);
        let bar_w = (rect.width() - 8.0).max(6.0);
        let bar_bg = Rect::from_min_size(Pos2::new(rect.left() + 4.0, rect.bottom() - 6.0), Vec2::new(bar_w, 3.0));
        painter.rect_filled(bar_bg, 0.0, Color32::BLACK);
        let fill_w = (bar_w * frac).max(1.0);
        let col = if frac > 0.55 {
            Color32::from_rgb(85, 255, 85)
        } else if frac > 0.25 {
            Color32::from_rgb(255, 255, 85)
        } else {
            Color32::from_rgb(255, 65, 65)
        };
        painter.rect_filled(
            Rect::from_min_size(bar_bg.min, Vec2::new(fill_w, 2.0)),
            0.0,
            col,
        );
    }
}

/// Paint an authentic Minecraft dark purple-bordered §-colored tooltip near the pointer.
pub fn paint_mc_tooltip(painter: &Painter, ctx: &egui::Context, anchor: Pos2, text: &str) {
    if text.trim().is_empty() {
        return;
    }
    let job = motd_to_layout_job_with_default(text, 13.0, [255, 255, 255]);
    let galley = ctx.fonts(|f| f.layout_job(job));
    let sz = galley.size();
    let pad = Vec2::new(10.0, 7.0);
    let screen = ctx.screen_rect();
    let mut box_pos = anchor + Vec2::new(14.0, -sz.y * 0.5);
    if box_pos.x + sz.x + pad.x * 2.0 > screen.right() - 8.0 {
        box_pos.x = (anchor.x - sz.x - pad.x * 2.0 - 10.0).max(screen.left() + 8.0);
    }
    if box_pos.y + sz.y + pad.y * 2.0 > screen.bottom() - 8.0 {
        box_pos.y = (screen.bottom() - sz.y - pad.y * 2.0 - 8.0).max(screen.top() + 8.0);
    }
    if box_pos.y < screen.top() + 8.0 {
        box_pos.y = screen.top() + 8.0;
    }
    let rect = Rect::from_min_size(box_pos, sz + pad * 2.0);
    painter.rect_filled(rect, 2.0, Color32::from_rgba_unmultiplied(16, 0, 16, 240));
    painter.rect_stroke(
        rect.shrink(1.5),
        1.0,
        Stroke::new(2.0, Color32::from_rgba_unmultiplied(80, 0, 255, 175)),
        egui::StrokeKind::Inside,
    );
    painter.galley(rect.min + pad, galley, Color32::WHITE);
}

/// Paint a crisp nearest-neighbor player skin front avatar card on the title screen.
pub fn paint_player_skin_card(
    painter: &Painter,
    rect: Rect,
    skin: Option<&TextureHandle>,
    player_name: &str,
    time_sec: f32,
) {
    painter.rect_filled(rect, 0.0, Color32::from_rgba_unmultiplied(12, 14, 20, 195));
    painter.rect_stroke(rect, 0.0, Stroke::new(2.0, Color32::from_rgb(90, 90, 90)), egui::StrokeKind::Inside);
    mc_shadow_text(
        painter,
        Pos2::new(rect.center().x, rect.top() + 14.0),
        Align2::CENTER_CENTER,
        "PLAYER",
        11.0,
        Color32::from_rgb(255, 255, 85),
    );

    let cx = rect.center().x;
    let top_y = rect.top() + 30.0 + (time_sec * 2.2).sin() * 2.0;
    let px = 4.0f32; // pixel scale

    if let Some(tex) = skin {
        // Standard Minecraft skin layout supports 64x32 (and 2:1 HD 128x64) and 64x64 (1:1 HD 128x128)
        let sz = tex.size();
        let tw = 64.0f32;
        let th = if sz[0] > 0 && sz[1] >= sz[0] { 64.0 } else { 32.0 };
        let uv_box = |x0: f32, y0: f32, x1: f32, y1: f32| {
            Rect::from_min_max(Pos2::new(x0 / tw, y0 / th), Pos2::new(x1 / tw, y1 / th))
        };
        // Head 8x8
        let head_r = Rect::from_min_size(Pos2::new(cx - 4.0 * px, top_y), Vec2::splat(8.0 * px));
        painter.image(tex.id(), head_r, uv_box(8.0, 8.0, 16.0, 16.0), Color32::WHITE);
        // Hat overlay 8x8
        painter.image(tex.id(), head_r.expand(1.5), uv_box(40.0, 8.0, 48.0, 16.0), Color32::WHITE);
        // Body 8x12
        let body_r = Rect::from_min_size(Pos2::new(cx - 4.0 * px, head_r.bottom()), Vec2::new(8.0 * px, 12.0 * px));
        painter.image(tex.id(), body_r, uv_box(20.0, 20.0, 28.0, 32.0), Color32::WHITE);
        // Arms 4x12
        let arm_l = Rect::from_min_size(Pos2::new(body_r.left() - 4.0 * px, body_r.top()), Vec2::new(4.0 * px, 12.0 * px));
        let arm_r = Rect::from_min_size(Pos2::new(body_r.right(), body_r.top()), Vec2::new(4.0 * px, 12.0 * px));
        painter.image(tex.id(), arm_l, uv_box(44.0, 20.0, 48.0, 32.0), Color32::WHITE);
        painter.image(tex.id(), arm_r, uv_box(44.0, 20.0, 48.0, 32.0), Color32::WHITE);
        // Legs 4x12
        let leg_l = Rect::from_min_size(Pos2::new(cx - 4.0 * px, body_r.bottom()), Vec2::new(4.0 * px, 12.0 * px));
        let leg_r = Rect::from_min_size(Pos2::new(cx, body_r.bottom()), Vec2::new(4.0 * px, 12.0 * px));
        painter.image(tex.id(), leg_l, uv_box(4.0, 20.0, 8.0, 32.0), Color32::WHITE);
        painter.image(tex.id(), leg_r, uv_box(4.0, 20.0, 8.0, 32.0), Color32::WHITE);
    } else {
        // Crisp procedural Steve silhouette fallback
        let head = Rect::from_min_size(Pos2::new(cx - 16.0, top_y), Vec2::splat(32.0));
        painter.rect_filled(head, 0.0, Color32::from_rgb(198, 152, 114));
        painter.rect_filled(Rect::from_min_size(head.min, Vec2::new(32.0, 9.0)), 0.0, Color32::from_rgb(68, 44, 24));
        painter.rect_filled(Rect::from_min_size(Pos2::new(cx - 16.0, head.bottom()), Vec2::new(32.0, 46.0)), 0.0, Color32::from_rgb(0, 148, 168));
        painter.rect_filled(Rect::from_min_size(Pos2::new(cx - 16.0, head.bottom() + 46.0), Vec2::new(32.0, 44.0)), 0.0, Color32::from_rgb(52, 58, 148));
    }

    mc_shadow_text(
        painter,
        Pos2::new(rect.center().x, rect.bottom() - 16.0),
        Align2::CENTER_CENTER,
        player_name,
        13.0,
        Color32::WHITE,
    );
}
