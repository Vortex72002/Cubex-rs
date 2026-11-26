use image::imageops::FilterType;
use image::{Rgba, RgbaImage};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::lua_api::core_api::parse_color_str;
use crate::registry::{EngineRegistries, CONTENT_AIR, CONTENT_IGNORE};

pub const TILE_SIZE: u32 = 32;
pub const MAX_ATLAS_LAYERS: usize = 2048;

#[derive(Debug, Clone)]
pub struct ResourcePackInfo {
    pub id: String,
    pub name: String,
    pub description: String,
    pub path: String,
}

#[derive(Debug, Clone)]
pub struct ObjVertex {
    pub pos: [f32; 3],
    pub uv: [f32; 2],
    pub normal: [f32; 3],
}

#[derive(Debug, Clone)]
pub struct ObjTriangle {
    pub verts: [ObjVertex; 3],
    pub material_slot: usize,
}

#[derive(Debug, Clone, Default)]
pub struct ObjMesh {
    pub triangles: Vec<ObjTriangle>,
}

pub fn list_resource_packs(textures_dir: &Path) -> Vec<ResourcePackInfo> {
    let mut out = vec![ResourcePackInfo {
        id: "default".into(),
        name: "Default Mineclonia".into(),
        description: "Built-in Mineclonia textures".into(),
        path: String::new(),
    }];
    if let Ok(entries) = std::fs::read_dir(textures_dir) {
        let mut dirs: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect();
        dirs.sort();
        for d in dirs {
            let id = d.file_name().unwrap_or_default().to_string_lossy().to_string();
            if id.starts_with('.') {
                continue;
            }
            let desc = std::fs::read_to_string(d.join("description.txt"))
                .ok()
                .and_then(|s| s.lines().next().map(ToString::to_string))
                .unwrap_or_else(|| id.replace('_', " "));
            out.push(ResourcePackInfo {
                id: id.clone(),
                name: id.replace('_', " "),
                description: desc,
                path: d.to_string_lossy().to_string(),
            });
        }
    }
    out
}

/// Raw decoded texture images and `.obj` meshes scanned independently of `EngineRegistries`.
/// Can be loaded in parallel on a background thread while GPU initialization & Lua mod loading run.
pub struct PreloadedTextures {
    pub files: HashMap<String, PathBuf>,
    pub raw_cache: HashMap<String, RgbaImage>,
    pub obj_meshes: HashMap<String, ObjMesh>,
}

impl PreloadedTextures {
    pub fn load(
        mod_paths: &HashMap<String, PathBuf>,
        game_dir: &Path,
        resource_pack_dir: Option<&Path>,
    ) -> Self {
        let mut files = HashMap::new();
        let mut model_files = HashMap::new();

        // 1. Scan game base textures & models
        TextureAtlas::scan_dir_recursive(&game_dir.join("textures"), &mut files);
        TextureAtlas::scan_models_recursive(&game_dir.join("models"), &mut model_files);
        // 2. Scan all mod textures/ & models/ folders
        for mp in mod_paths.values() {
            TextureAtlas::scan_dir_recursive(&mp.join("textures"), &mut files);
            TextureAtlas::scan_models_recursive(&mp.join("models"), &mut model_files);
        }
        // 3. Scan active resource pack (overrides mod textures by filename!)
        if let Some(rp) = resource_pack_dir {
            if rp.is_dir() {
                TextureAtlas::scan_dir_recursive(rp, &mut files);
            }
        }

        // Parallel preload all discovered image files so &self lookups (get_rgba, get_item_rgba) work instantly
        let raw_entries: Vec<(String, RgbaImage)> = files
            .par_iter()
            .filter_map(|(name, path)| {
                let img = image::open(path).ok()?.to_rgba8();
                Some((name.clone(), img))
            })
            .collect();
        let raw_cache: HashMap<String, RgbaImage> = raw_entries.into_iter().collect();

        // Parallel load .obj meshes for DrawType::Mesh nodes
        let obj_entries: Vec<(String, ObjMesh)> = model_files
            .par_iter()
            .filter_map(|(name, path)| {
                let text = std::fs::read_to_string(path).ok()?;
                let mesh = parse_obj_mesh(&text);
                if mesh.triangles.is_empty() {
                    None
                } else {
                    Some((name.clone(), mesh))
                }
            })
            .collect();
        let obj_meshes: HashMap<String, ObjMesh> = obj_entries.into_iter().collect();

        Self {
            files,
            raw_cache,
            obj_meshes,
        }
    }
}

pub struct TextureAtlas {
    pub files: HashMap<String, PathBuf>,
    pub raw_cache: HashMap<String, RgbaImage>,
    pub resolved_cache: HashMap<String, u16>,
    pub layers: Vec<RgbaImage>,
    /// Per-node 6-face layer indices ordered [+X, -X, +Y, -Y, +Z, -Z].
    pub node_face_layers: HashMap<u16, [u16; 6]>,
    /// Per-node 6-face overlay layer indices ordered [+X, -X, +Y, -Y, +Z, -Z]; 0 = no overlay.
    pub overlay_face_layers: HashMap<u16, [u16; 6]>,
    pub crack_layers: [u16; 10],
    pub palettes: HashMap<String, Vec<[u8; 4]>>,
    pub player_skin: RgbaImage,
    pub obj_meshes: HashMap<String, ObjMesh>,
    pub fallback_layer: u16,
}

impl TextureAtlas {
    /// Minimal 2-layer placeholder atlas constructed in <0.2ms for instant GPU splash startup.
    pub fn placeholder() -> Self {
        let fallback = Self::make_fallback_tile();
        Self {
            files: HashMap::new(),
            raw_cache: HashMap::new(),
            resolved_cache: HashMap::new(),
            layers: vec![fallback.clone(), fallback],
            node_face_layers: HashMap::new(),
            overlay_face_layers: HashMap::new(),
            crack_layers: [0; 10],
            palettes: HashMap::new(),
            player_skin: RgbaImage::new(64, 32),
            obj_meshes: HashMap::new(),
            fallback_layer: 0,
        }
    }

    pub fn build(
        mod_paths: &HashMap<String, PathBuf>,
        game_dir: &Path,
        resource_pack_dir: Option<&Path>,
        registries: &EngineRegistries,
    ) -> Self {
        let preloaded = PreloadedTextures::load(mod_paths, game_dir, resource_pack_dir);
        Self::from_preloaded(preloaded, registries)
    }

    pub fn from_preloaded(
        preloaded: PreloadedTextures,
        registries: &EngineRegistries,
    ) -> Self {
        // Layer 0: crisp neutral fallback tile
        let fallback = Self::make_fallback_tile();
        let mut atlas = Self {
            files: preloaded.files,
            raw_cache: preloaded.raw_cache,
            resolved_cache: HashMap::new(),
            layers: vec![fallback],
            node_face_layers: HashMap::new(),
            overlay_face_layers: HashMap::new(),
            crack_layers: [0; 10],
            palettes: HashMap::new(),
            player_skin: RgbaImage::new(64, 32),
            obj_meshes: preloaded.obj_meshes,
            fallback_layer: 0,
        };

        // Pre-intern 10 block-break crack layers (layers 1..=10) so they never overflow MAX_ATLAS_LAYERS
        atlas.crack_layers = atlas.build_crack_layers();

        // Pre-cache all palette PNGs (e.g. mcl_core_palette_grass.png, mcl_core_palette_leaves.png, etc.)
        let palette_names: Vec<String> = atlas
            .raw_cache
            .keys()
            .filter(|k| k.contains("palette"))
            .cloned()
            .collect();
        for pname in palette_names {
            atlas.build_palette(&pname);
        }
        for def in registries.nodes_by_id.iter().flatten() {
            if let Some(ref pname) = def.palette {
                if !pname.is_empty() {
                    atlas.build_palette(pname);
                }
            }
        }

        // Load player skin or build procedural Steve-style pixel skin fallback
        atlas.player_skin = atlas
            .raw_cache
            .get("character.png")
            .or_else(|| atlas.raw_cache.get("mcl_skins_character_1.png"))
            .cloned()
            .unwrap_or_else(Self::make_procedural_steve_skin);

        // Resolve tile layers & overlay layers for all registered nodes
        for def in registries.nodes_by_id.iter().flatten() {
            if def.id == CONTENT_AIR || def.id == CONTENT_IGNORE {
                continue;
            }
            let mut face_layers = [0u16; 6];
            if !def.tiles.is_empty() {
                // Luanti tile specification order: [0]=+Y(top), [1]=-Y(bottom), [2]=+X, [3]=-X, [4]=+Z, [5]=-Z
                let get_tile = |luanti_idx: usize| -> &str {
                    let len = def.tiles.len();
                    let idx = luanti_idx.min(len - 1);
                    def.tiles[idx].name.as_str()
                };
                let top = atlas.intern_spec(get_tile(0));
                let bottom = atlas.intern_spec(get_tile(1));
                let right = atlas.intern_spec(get_tile(2));
                let left = atlas.intern_spec(get_tile(3));
                let back = atlas.intern_spec(get_tile(4));
                let front = atlas.intern_spec(get_tile(5));

                face_layers = [right, left, top, bottom, back, front];
            }
            atlas.node_face_layers.insert(def.id, face_layers);

            if !def.overlay_tiles.is_empty() {
                let ov_specs: Vec<String> = (0..6)
                    .map(|luanti_idx| {
                        let len = def.overlay_tiles.len();
                        let idx = luanti_idx.min(len - 1);
                        def.overlay_tiles[idx].name.trim().to_string()
                    })
                    .collect();
                let mut intern_ov = |i: usize| -> u16 {
                    if ov_specs[i].is_empty() {
                        0
                    } else {
                        atlas.intern_spec(&ov_specs[i])
                    }
                };
                let ov_top = intern_ov(0);
                let ov_bot = intern_ov(1);
                let ov_r = intern_ov(2);
                let ov_l = intern_ov(3);
                let ov_b = intern_ov(4);
                let ov_f = intern_ov(5);
                let ov_arr = [ov_r, ov_l, ov_top, ov_bot, ov_b, ov_f];
                if ov_arr.iter().any(|&l| l > 0) {
                    atlas.overlay_face_layers.insert(def.id, ov_arr);
                }
            }
        }

        atlas
    }

    fn make_fallback_tile() -> RgbaImage {
        let mut img = RgbaImage::new(TILE_SIZE, TILE_SIZE);
        for y in 0..TILE_SIZE {
            for x in 0..TILE_SIZE {
                let c = if ((x / 8) + (y / 8)) % 2 == 0 {
                    Rgba([140, 140, 140, 255])
                } else {
                    Rgba([95, 95, 95, 255])
                };
                img.put_pixel(x, y, c);
            }
        }
        img
    }

    fn build_crack_layers(&mut self) -> [u16; 10] {
        let mut out = [self.fallback_layer; 10];
        if let Some(sheet) = self.raw_cache.get("crack_anylength.png").cloned() {
            let w = sheet.width().max(1);
            let h = sheet.height().max(1);
            let frames = if h >= w && h % w == 0 {
                (h / w).max(1)
            } else {
                10
            };
            let frame_h = (h / frames).max(1);
            for stage in 0..10u32 {
                let idx = (stage * frames / 10).min(frames - 1);
                let sy = (idx * frame_h).min(h.saturating_sub(frame_h));
                let sub = image::imageops::crop_imm(&sheet, 0, sy, w, frame_h).to_image();
                let norm = if sub.width() == TILE_SIZE && sub.height() == TILE_SIZE {
                    sub
                } else {
                    image::imageops::resize(&sub, TILE_SIZE, TILE_SIZE, FilterType::Nearest)
                };
                if self.layers.len() < MAX_ATLAS_LAYERS {
                    let layer = self.layers.len() as u16;
                    self.layers.push(norm);
                    out[stage as usize] = layer;
                }
            }
            return out;
        }

        // Procedural pixel-art crack stages 0..=9 fallback when no crack sheet is loaded
        for stage in 0..10usize {
            let mut img = RgbaImage::from_pixel(TILE_SIZE, TILE_SIZE, Rgba([0, 0, 0, 0]));
            let mut seed = 0xCAFE_0001u32 + (stage as u32) * 7919;
            let branch_count = 3 + stage * 3;
            let cx = 16i32;
            let cy = 16i32;
            for b in 0..branch_count {
                let angle = ((b as f32) * 2.39996) + ((seed % 100) as f32 * 0.01);
                let len = 3 + (stage * 2);
                let mut x = cx as f32;
                let mut y = cy as f32;
                let dx = angle.cos();
                let dy = angle.sin();
                for step in 0..len {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    let jx = if (seed >> 16) % 3 == 0 { 1.0 } else if (seed >> 16) % 3 == 1 { -1.0 } else { 0.0 };
                    x += dx + jx * 0.4;
                    y += dy;
                    let px = (x.round() as i32).clamp(0, (TILE_SIZE - 1) as i32) as u32;
                    let py = (y.round() as i32).clamp(0, (TILE_SIZE - 1) as i32) as u32;
                    let alpha = (155 + (stage * 10).min(100)) as u8;
                    img.put_pixel(px, py, Rgba([18, 18, 18, alpha]));
                    if step > 1 && px + 1 < TILE_SIZE && stage >= 4 {
                        img.put_pixel(px + 1, py, Rgba([42, 42, 42, (alpha / 2).max(90)]));
                    }
                }
            }
            if self.layers.len() < MAX_ATLAS_LAYERS {
                let layer = self.layers.len() as u16;
                self.layers.push(img);
                out[stage] = layer;
            }
        }
        out
    }

    fn make_procedural_steve_skin() -> RgbaImage {
        let mut img = RgbaImage::from_pixel(64, 32, Rgba([0, 0, 0, 0]));
        let fill_rect = |im: &mut RgbaImage, x0: u32, y0: u32, w: u32, h: u32, base: [u8; 3], var: i16| {
            let mut s = x0 * 97 + y0 * 193 + 17;
            for y in y0..(y0 + h).min(32) {
                for x in x0..(x0 + w).min(64) {
                    s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                    let j = (((s >> 16) % ((var as u32 * 2 + 1).max(1))) as i16) - var;
                    im.put_pixel(
                        x,
                        y,
                        Rgba([
                            (base[0] as i16 + j).clamp(0, 255) as u8,
                            (base[1] as i16 + j).clamp(0, 255) as u8,
                            (base[2] as i16 + j).clamp(0, 255) as u8,
                            255,
                        ]),
                    );
                }
            }
        };
        // Head (0..32, 0..16)
        fill_rect(&mut img, 0, 0, 32, 16, [186, 138, 108], 7);
        fill_rect(&mut img, 0, 0, 32, 10, [68, 44, 24], 5); // hair top/back
        fill_rect(&mut img, 8, 8, 8, 8, [186, 138, 108], 6); // face
        fill_rect(&mut img, 8, 8, 8, 2, [68, 44, 24], 4);
        // Eyes
        img.put_pixel(9, 12, Rgba([255, 255, 255, 255]));
        img.put_pixel(10, 12, Rgba([58, 92, 178, 255]));
        img.put_pixel(13, 12, Rgba([58, 92, 178, 255]));
        img.put_pixel(14, 12, Rgba([255, 255, 255, 255]));
        // Torso (16..40, 16..32)
        fill_rect(&mut img, 16, 16, 24, 16, [14, 172, 174], 8);
        // Legs (0..16, 16..32)
        fill_rect(&mut img, 0, 16, 16, 16, [52, 66, 148], 8);
        // Right Arm (40..56, 16..32): cyan sleeve top (y=16..24), warm tan forearm/hand (y=24..32)
        fill_rect(&mut img, 40, 16, 16, 4, [14, 172, 174], 6); // top/bottom caps
        fill_rect(&mut img, 48, 16, 4, 4, [172, 124, 96], 5);  // bottom hand cap
        fill_rect(&mut img, 40, 20, 16, 4, [14, 172, 174], 8); // shirt sleeve sides
        fill_rect(&mut img, 40, 24, 16, 8, [182, 134, 104], 7); // exposed forearm/hand
        img
    }

    fn build_palette(&mut self, palette_name: &str) {
        let clean = palette_name.trim();
        if clean.is_empty() || self.palettes.contains_key(clean) {
            return;
        }
        let Some(img) = self.raw_cache.get(clean) else {
            return;
        };
        let w = img.width().max(1);
        let h = img.height().max(1);
        let area = w * h;
        let step = (256 / area).max(1);
        let mut table = Vec::with_capacity(256);

        let first_opaque = img
            .pixels()
            .find(|p| p.0[3] > 0)
            .map(|p| [p.0[0], p.0[1], p.0[2], 255])
            .unwrap_or([136, 184, 78, 255]);

        for i in 0..area {
            if table.len() >= 256 {
                break;
            }
            let px = img.get_pixel(i % w, i / w).0;
            let entry = if px[3] > 0 {
                [px[0], px[1], px[2], px[3]]
            } else {
                first_opaque
            };
            for _ in 0..step {
                if table.len() < 256 {
                    table.push(entry);
                }
            }
        }
        while table.len() < 256 {
            table.push(first_opaque);
        }
        self.palettes.insert(clean.to_string(), table);
    }

    fn scan_dir_recursive(dir: &Path, out: &mut HashMap<String, PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for ent in entries.flatten() {
            let p = ent.path();
            if p.is_dir() {
                Self::scan_dir_recursive(&p, out);
            } else if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
                if matches!(ext.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg") {
                    if let Some(fname) = p.file_name().and_then(|f| f.to_str()) {
                        out.insert(fname.to_string(), p);
                    }
                }
            }
        }
    }

    fn scan_models_recursive(dir: &Path, out: &mut HashMap<String, PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for ent in entries.flatten() {
            let p = ent.path();
            if p.is_dir() {
                Self::scan_models_recursive(&p, out);
            } else if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
                if ext.eq_ignore_ascii_case("obj") {
                    if let Some(fname) = p.file_name().and_then(|f| f.to_str()) {
                        out.insert(fname.to_string(), p);
                    }
                }
            }
        }
    }

    /// Look up a cached raw base image by filename (`&self`, zero mutation).
    pub fn get_rgba(&self, filename: &str) -> Option<&RgbaImage> {
        let clean = filename.trim();
        if clean.is_empty() {
            return None;
        }
        if let Some(img) = self.raw_cache.get(clean) {
            return Some(img);
        }
        if let Some(base) = Path::new(clean).file_name().and_then(|f| f.to_str()) {
            return self.raw_cache.get(base);
        }
        None
    }

    /// Resolve a Mineclonia item's 2D icon (`inventory_image`, `wield_image`, or fallback node top/side texture)
    /// applying texture modifiers (`^`, `[colorize:`, `[multiply:`, `[makealpha:`, `[combine:`, etc.) and node tint.
    pub fn get_item_rgba(&self, item_name: &str, reg: &EngineRegistries) -> Option<RgbaImage> {
        let base_name = item_name.split_whitespace().next().unwrap_or(item_name).trim();
        if base_name.is_empty() {
            return None;
        }
        let resolved = reg.resolve_alias(base_name);

        if let Some(idef) = reg.get_item_def(resolved) {
            if !idef.inventory_image.trim().is_empty() {
                if let Some(img) = self.resolve_spec_immut(&idef.inventory_image) {
                    return Some(img);
                }
            }
            if !idef.wield_image.trim().is_empty() {
                if let Some(img) = self.resolve_spec_immut(&idef.wield_image) {
                    return Some(img);
                }
            }
        }

        let cid = reg.get_content_id(resolved);
        if let Some(ndef) = reg.get_node_def(cid) {
            if !ndef.inventory_image.trim().is_empty() {
                if let Some(img) = self.resolve_spec_immut(&ndef.inventory_image) {
                    return Some(img);
                }
            }
            if !ndef.wield_image.trim().is_empty() {
                if let Some(img) = self.resolve_spec_immut(&ndef.wield_image) {
                    return Some(img);
                }
            }
            if !ndef.tiles.is_empty() {
                // Prefer front/side tile for multi-tile blocks (grass block, furnace, crafting table), else top tile[0]
                let tile_idx = if ndef.tiles.len() >= 6 {
                    5 // front (+Z / -Z)
                } else if ndef.tiles.len() >= 3 {
                    2 // side
                } else {
                    0 // top/uniform
                };
                let tile = &ndef.tiles[tile_idx.min(ndef.tiles.len() - 1)];
                let mut img = self.resolve_spec_immut(&tile.name)?;
                img = normalize_anim_frame(img, &tile.name);

                let tint = tile
                    .color
                    .or_else(|| {
                        ndef.palette
                            .as_deref()
                            .and_then(|p| self.sample_palette_color(p, 0))
                    })
                    .or(ndef.color);
                if let Some(c) = tint {
                    if c != [255, 255, 255, 255] {
                        tint_rgba(&mut img, c);
                    }
                }

                // Composite overlay tile if present on the chosen face (e.g. grass block side overlay)
                if !ndef.overlay_tiles.is_empty() {
                    let ov_tile = &ndef.overlay_tiles[tile_idx.min(ndef.overlay_tiles.len() - 1)];
                    if !ov_tile.name.trim().is_empty() {
                        if let Some(mut ov) = self.resolve_spec_immut(&ov_tile.name) {
                            ov = normalize_anim_frame(ov, &ov_tile.name);
                            let ov_tint = ov_tile
                                .color
                                .or_else(|| {
                                    ndef.palette
                                        .as_deref()
                                        .and_then(|p| self.sample_palette_color(p, 0))
                                })
                                .or(ndef.color);
                            if let Some(oc) = ov_tint {
                                tint_rgba(&mut ov, oc);
                            }
                            img = alpha_composite(img, ov);
                        }
                    }
                }
                return Some(img);
            }
        }
        None
    }

    /// Sample a 256-entry Mineclonia biome palette color for `param2`.
    pub fn sample_palette_color(&self, palette_name: &str, param2: u8) -> Option<[u8; 4]> {
        let clean = palette_name.trim();
        if clean.is_empty() {
            return None;
        }
        if let Some(table) = self.palettes.get(clean) {
            return table.get(param2 as usize).copied();
        }
        let img = self.get_rgba(clean)?;
        let w = img.width().max(1);
        let h = img.height().max(1);
        let area = w * h;
        let idx = ((param2 as u32) * area / 256).min(area - 1);
        let px = img.get_pixel(idx % w, idx / w).0;
        if px[3] == 0 {
            img.pixels()
                .find(|p| p.0[3] > 0)
                .map(|p| [p.0[0], p.0[1], p.0[2], 255])
        } else {
            Some(px)
        }
    }

    pub fn get_crack_layers(&self) -> [u16; 10] {
        self.crack_layers
    }

    pub fn get_player_skin_rgba(&self) -> RgbaImage {
        self.player_skin.clone()
    }

    pub fn get_obj_mesh(&self, name: &str) -> Option<&ObjMesh> {
        let clean = name.trim();
        self.obj_meshes.get(clean).or_else(|| {
            Path::new(clean)
                .file_name()
                .and_then(|f| f.to_str())
                .and_then(|b| self.obj_meshes.get(b))
        })
    }

    pub fn intern_spec(&mut self, spec: &str) -> u16 {
        let clean = spec.trim();
        if clean.is_empty() {
            return self.fallback_layer;
        }
        if let Some(&layer) = self.resolved_cache.get(clean) {
            return layer;
        }
        if self.layers.len() >= MAX_ATLAS_LAYERS {
            return self.fallback_layer;
        }
        let img = normalize_anim_frame(self.resolve_spec(clean), clean);
        let norm = if img.width() == TILE_SIZE && img.height() == TILE_SIZE {
            img
        } else {
            image::imageops::resize(&img, TILE_SIZE, TILE_SIZE, FilterType::Nearest)
        };
        let layer = self.layers.len() as u16;
        self.layers.push(norm);
        self.resolved_cache.insert(clean.to_string(), layer);
        layer
    }

    pub fn load_base(&mut self, filename: &str) -> Option<RgbaImage> {
        let clean = filename.trim();
        if clean.is_empty() {
            return None;
        }
        if let Some(cached) = self.raw_cache.get(clean) {
            return Some(cached.clone());
        }
        let path = self.files.get(clean)?;
        let img = image::open(path).ok()?.to_rgba8();
        self.raw_cache.insert(clean.to_string(), img.clone());
        Some(img)
    }

    pub fn resolve_spec(&mut self, spec: &str) -> RgbaImage {
        self.resolve_spec_immut(spec)
            .unwrap_or_else(Self::make_fallback_tile)
    }

    pub fn resolve_spec_immut(&self, spec: &str) -> Option<RgbaImage> {
        let parts = split_top_level_carets(spec);
        let mut current: Option<RgbaImage> = None;

        for raw_part in parts {
            let part = raw_part.trim();
            if part.is_empty() {
                continue;
            }
            if part.starts_with('(') && part.ends_with(')') && part.len() > 2 {
                if let Some(inner) = self.resolve_spec_immut(&part[1..part.len() - 1]) {
                    current = Some(match current {
                        Some(base) => alpha_composite(base, inner),
                        None => inner,
                    });
                }
                continue;
            }

            if let Some(rest) = part.strip_prefix('[') {
                let base = current.get_or_insert_with(|| RgbaImage::new(TILE_SIZE, TILE_SIZE));
                if let Some(c_rest) = rest.strip_prefix("combine:") {
                    *base = self.eval_combine_immut(c_rest);
                } else if let Some(col_rest) = rest.strip_prefix("colorize:") {
                    apply_colorize(base, col_rest);
                } else if let Some(mul_rest) = rest.strip_prefix("multiply:") {
                    apply_multiply(base, mul_rest);
                } else if let Some(tr_rest) = rest.strip_prefix("transform") {
                    *base = apply_transform(base, tr_rest);
                } else if let Some(mask_rest) = rest.strip_prefix("mask:") {
                    if let Some(mask) = self.get_rgba(mask_rest) {
                        apply_mask(base, mask);
                    }
                } else if let Some(ma_rest) = rest.strip_prefix("makealpha:") {
                    apply_makealpha(base, ma_rest);
                } else if let Some(lp_rest) = rest.strip_prefix("lowpart:") {
                    *base = self.apply_lowpart(base, lp_rest);
                } else if let Some(sheet_rest) = rest.strip_prefix("sheet:") {
                    *base = apply_sheet(base, sheet_rest);
                } else if let Some(vf_rest) = rest.strip_prefix("verticalframe:") {
                    *base = apply_verticalframe(base, vf_rest);
                } else if let Some(rs_rest) = rest.strip_prefix("resize:") {
                    *base = apply_resize(base, rs_rest);
                } else if let Some(op_rest) = rest.strip_prefix("opacity:") {
                    apply_opacity(base, op_rest);
                } else if let Some(inv_rest) = rest.strip_prefix("invert:") {
                    apply_invert(base, inv_rest);
                } else if rest.starts_with("brighten") {
                    apply_brighten(base);
                } else if rest.starts_with("noalpha") {
                    apply_noalpha(base);
                }
            } else if let Some(img) = self.get_rgba(part) {
                current = Some(match current {
                    Some(base) => alpha_composite(base, img.clone()),
                    None => img.clone(),
                });
            }
        }

        current
    }

    fn eval_combine_immut(&self, expr: &str) -> RgbaImage {
        let mut iter = expr.split(':');
        let size_str = iter.next().unwrap_or("16x16");
        let (w, h) = parse_wh(size_str).unwrap_or((16, 16));
        let mut canvas = RgbaImage::new(w.max(1).min(512), h.max(1).min(512));

        for seg in iter {
            let Some((coord_s, img_spec)) = seg.split_once('=') else {
                continue;
            };
            let (ox, oy) = coord_s
                .split_once(',')
                .and_then(|(x, y)| Some((x.trim().parse::<i32>().ok()?, y.trim().parse::<i32>().ok()?)))
                .unwrap_or((0, 0));
            let overlay = self
                .resolve_spec_immut(&img_spec.replace("\\^", "^"))
                .unwrap_or_else(Self::make_fallback_tile);
            blit_rgba(&mut canvas, &overlay, ox, oy);
        }
        canvas
    }

    fn apply_lowpart(&self, base: &RgbaImage, expr: &str) -> RgbaImage {
        let Some((pct_s, img_spec)) = expr.split_once(':') else {
            return base.clone();
        };
        let pct = pct_s.trim().parse::<f32>().unwrap_or(50.0).clamp(0.0, 100.0) / 100.0;
        let Some(over) = self.resolve_spec_immut(&img_spec.replace("\\^", "^")) else {
            return base.clone();
        };
        let mut out = base.clone();
        let over_resized = if over.dimensions() != out.dimensions() {
            image::imageops::resize(&over, out.width(), out.height(), FilterType::Nearest)
        } else {
            over
        };
        let start_y = ((1.0 - pct) * out.height() as f32).round() as u32;
        for y in start_y.min(out.height())..out.height() {
            for x in 0..out.width() {
                let sp = over_resized.get_pixel(x, y).0;
                if sp[3] > 0 {
                    out.put_pixel(x, y, Rgba(sp));
                }
            }
        }
        out
    }

    /// Pack all tile layers into a single tightly-packed RGBA8 byte array for wgpu D2Array upload.
    pub fn pack_rgba_bytes(&self) -> (u32, u32, u32, Vec<u8>) {
        let count = self.layers.len() as u32;
        let mut bytes = Vec::with_capacity((TILE_SIZE * TILE_SIZE * 4 * count) as usize);
        for layer in &self.layers {
            bytes.extend_from_slice(layer.as_raw());
        }
        (TILE_SIZE, TILE_SIZE, count, bytes)
    }
}

fn normalize_anim_frame(img: RgbaImage, spec: &str) -> RgbaImage {
    if spec.contains("[verticalframe:") || spec.contains("[sheet:") {
        return img;
    }
    let w = img.width();
    let h = img.height();
    if w >= 16 && h >= w * 3 && h % w == 0 {
        image::imageops::crop_imm(&img, 0, 0, w, w).to_image()
    } else {
        img
    }
}

fn tint_rgba(img: &mut RgbaImage, tint: [u8; 4]) {
    for px in img.pixels_mut() {
        px.0[0] = ((px.0[0] as u16 * tint[0] as u16) / 255) as u8;
        px.0[1] = ((px.0[1] as u16 * tint[1] as u16) / 255) as u8;
        px.0[2] = ((px.0[2] as u16 * tint[2] as u16) / 255) as u8;
    }
}

fn parse_obj_mesh(src: &str) -> ObjMesh {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut triangles: Vec<ObjTriangle> = Vec::new();
    let mut mat_slot = 0usize;
    let mut seen_mats: Vec<String> = Vec::new();

    for raw_line in src.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        let Some(tag) = parts.next() else {
            continue;
        };
        match tag {
            "v" => {
                let x = parts.next().and_then(|s| s.parse::<f32>().ok()).unwrap_or(0.0);
                let y = parts.next().and_then(|s| s.parse::<f32>().ok()).unwrap_or(0.0);
                let z = parts.next().and_then(|s| s.parse::<f32>().ok()).unwrap_or(0.0);
                positions.push([x, y, z]);
            }
            "vt" => {
                let u = parts.next().and_then(|s| s.parse::<f32>().ok()).unwrap_or(0.0);
                let v = parts.next().and_then(|s| s.parse::<f32>().ok()).unwrap_or(0.0);
                uvs.push([u, 1.0 - v]);
            }
            "vn" => {
                let nx = parts.next().and_then(|s| s.parse::<f32>().ok()).unwrap_or(0.0);
                let ny = parts.next().and_then(|s| s.parse::<f32>().ok()).unwrap_or(1.0);
                let nz = parts.next().and_then(|s| s.parse::<f32>().ok()).unwrap_or(0.0);
                normals.push([nx, ny, nz]);
            }
            "usemtl" => {
                if let Some(mname) = parts.next() {
                    if let Some(idx) = seen_mats.iter().position(|m| m == mname) {
                        mat_slot = idx;
                    } else {
                        mat_slot = seen_mats.len().min(5);
                        seen_mats.push(mname.to_string());
                    }
                }
            }
            "f" => {
                let mut poly: Vec<ObjVertex> = Vec::new();
                for vert_tok in parts {
                    let mut segs = vert_tok.split('/');
                    let vi = segs.next().and_then(|s| s.parse::<isize>().ok()).unwrap_or(1);
                    let ti = segs.next().and_then(|s| if s.is_empty() { None } else { s.parse::<isize>().ok() });
                    let ni = segs.next().and_then(|s| if s.is_empty() { None } else { s.parse::<isize>().ok() });

                    let resolve_idx = |idx: isize, len: usize| -> usize {
                        if idx > 0 {
                            ((idx - 1) as usize).min(len.saturating_sub(1))
                        } else if idx < 0 {
                            len.saturating_sub((-idx) as usize)
                        } else {
                            0
                        }
                    };
                    let pos = if positions.is_empty() {
                        [0.0, 0.0, 0.0]
                    } else {
                        positions[resolve_idx(vi, positions.len())]
                    };
                    let uv = match ti {
                        Some(t) if !uvs.is_empty() => uvs[resolve_idx(t, uvs.len())],
                        _ => [0.0, 0.0],
                    };
                    let normal = match ni {
                        Some(n) if !normals.is_empty() => normals[resolve_idx(n, normals.len())],
                        _ => [0.0, 1.0, 0.0],
                    };
                    poly.push(ObjVertex { pos, uv, normal });
                }
                if poly.len() >= 3 {
                    for i in 1..(poly.len() - 1) {
                        triangles.push(ObjTriangle {
                            verts: [poly[0].clone(), poly[i].clone(), poly[i + 1].clone()],
                            material_slot: mat_slot,
                        });
                    }
                }
            }
            _ => {}
        }
    }
    ObjMesh { triangles }
}

fn split_top_level_carets(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let bytes = s.as_bytes();
    let mut depth = 0i32;
    let mut start = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'\\' && i + 1 < bytes.len() {
            i += 2;
            continue;
        }
        if b == b'(' {
            depth += 1;
        } else if b == b')' && depth > 0 {
            depth -= 1;
        } else if b == b'^' && depth == 0 {
            parts.push(&s[start..i]);
            start = i + 1;
        }
        i += 1;
    }
    parts.push(&s[start..]);
    parts
}

fn parse_wh(s: &str) -> Option<(u32, u32)> {
    let (w, h) = s.split_once('x')?;
    Some((w.trim().parse().ok()?, h.trim().parse().ok()?))
}

fn alpha_composite(mut base: RgbaImage, over: RgbaImage) -> RgbaImage {
    let over_resized = if over.dimensions() != base.dimensions() {
        image::imageops::resize(&over, base.width(), base.height(), FilterType::Nearest)
    } else {
        over
    };
    blit_rgba(&mut base, &over_resized, 0, 0);
    base
}

fn blit_rgba(dst: &mut RgbaImage, src: &RgbaImage, ox: i32, oy: i32) {
    let (dw, dh) = dst.dimensions();
    for sy in 0..src.height() {
        let dy = oy + sy as i32;
        if dy < 0 || dy >= dh as i32 {
            continue;
        }
        for sx in 0..src.width() {
            let dx = ox + sx as i32;
            if dx < 0 || dx >= dw as i32 {
                continue;
            }
            let sp = src.get_pixel(sx, sy).0;
            let sa = sp[3] as f32 / 255.0;
            if sa <= 0.0 {
                continue;
            }
            if sa >= 1.0 {
                dst.put_pixel(dx as u32, dy as u32, Rgba(sp));
            } else {
                let dp = dst.get_pixel(dx as u32, dy as u32).0;
                let da = dp[3] as f32 / 255.0;
                let out_a = sa + da * (1.0 - sa);
                if out_a > 0.0 {
                    let blend = |sc: u8, dc: u8| -> u8 {
                        ((sc as f32 * sa + dc as f32 * da * (1.0 - sa)) / out_a)
                            .round()
                            .clamp(0.0, 255.0) as u8
                    };
                    dst.put_pixel(
                        dx as u32,
                        dy as u32,
                        Rgba([blend(sp[0], dp[0]), blend(sp[1], dp[1]), blend(sp[2], dp[2]), (out_a * 255.0) as u8]),
                    );
                }
            }
        }
    }
}

fn apply_colorize(img: &mut RgbaImage, expr: &str) {
    let (col_s, ratio_s) = expr.split_once(':').unwrap_or((expr, "alpha"));
    let Some([cr, cg, cb, ca]) = parse_color_str(col_s) else {
        return;
    };
    let ratio = if ratio_s == "alpha" {
        ca as f32 / 255.0
    } else if let Ok(v) = ratio_s.trim().parse::<f32>() {
        if v > 1.0 { v / 255.0 } else { v }.clamp(0.0, 1.0)
    } else {
        0.5
    };
    for px in img.pixels_mut() {
        if px.0[3] == 0 {
            continue;
        }
        px.0[0] = (px.0[0] as f32 * (1.0 - ratio) + cr as f32 * ratio) as u8;
        px.0[1] = (px.0[1] as f32 * (1.0 - ratio) + cg as f32 * ratio) as u8;
        px.0[2] = (px.0[2] as f32 * (1.0 - ratio) + cb as f32 * ratio) as u8;
    }
}

fn apply_multiply(img: &mut RgbaImage, expr: &str) {
    let Some([mr, mg, mb, _]) = parse_color_str(expr) else {
        return;
    };
    for px in img.pixels_mut() {
        px.0[0] = ((px.0[0] as u16 * mr as u16) / 255) as u8;
        px.0[1] = ((px.0[1] as u16 * mg as u16) / 255) as u8;
        px.0[2] = ((px.0[2] as u16 * mb as u16) / 255) as u8;
    }
}

fn apply_makealpha(img: &mut RgbaImage, expr: &str) {
    let nums: Vec<u8> = expr
        .split(',')
        .filter_map(|s| s.trim().parse::<u8>().ok())
        .collect();
    if nums.len() < 3 {
        return;
    }
    let (kr, kg, kb) = (nums[0], nums[1], nums[2]);
    for px in img.pixels_mut() {
        if px.0[0] == kr && px.0[1] == kg && px.0[2] == kb {
            px.0[3] = 0;
        }
    }
}

fn apply_transform(img: &RgbaImage, expr: &str) -> RgbaImage {
    let t = expr.trim().to_ascii_uppercase();
    let mut cur = img.clone();
    match t.as_str() {
        "I" | "0" => return cur,
        "R90" | "1" => return image::imageops::rotate90(&cur),
        "R180" | "2" => return image::imageops::rotate180(&cur),
        "R270" | "3" => return image::imageops::rotate270(&cur),
        "FX" | "4" => return image::imageops::flip_horizontal(&cur),
        "FXR90" | "5" => {
            cur = image::imageops::flip_horizontal(&cur);
            return image::imageops::rotate90(&cur);
        }
        "FY" | "6" => return image::imageops::flip_vertical(&cur),
        "FYR90" | "7" => {
            cur = image::imageops::flip_vertical(&cur);
            return image::imageops::rotate90(&cur);
        }
        _ => {}
    }
    // Parse chained transforms like FXR90 / R90FX
    if t.contains("FX") {
        cur = image::imageops::flip_horizontal(&cur);
    }
    if t.contains("FY") {
        cur = image::imageops::flip_vertical(&cur);
    }
    if t.contains("R270") {
        cur = image::imageops::rotate270(&cur);
    } else if t.contains("R180") {
        cur = image::imageops::rotate180(&cur);
    } else if t.contains("R90") {
        cur = image::imageops::rotate90(&cur);
    }
    cur
}

fn apply_mask(img: &mut RgbaImage, mask: &RgbaImage) {
    let resized = if mask.dimensions() != img.dimensions() {
        image::imageops::resize(mask, img.width(), img.height(), FilterType::Nearest)
    } else {
        mask.clone()
    };
    for (p, m) in img.pixels_mut().zip(resized.pixels()) {
        p.0[3] = ((p.0[3] as u16 * m.0[3] as u16) / 255) as u8;
    }
}

fn apply_sheet(img: &RgbaImage, expr: &str) -> RgbaImage {
    let Some((wh_s, xy_s)) = expr.split_once(':') else {
        return img.clone();
    };
    let Some((cols, rows)) = parse_wh(wh_s) else {
        return img.clone();
    };
    let Some((cx, cy)) = xy_s
        .split_once(',')
        .and_then(|(a, b)| Some((a.trim().parse::<u32>().ok()?, b.trim().parse::<u32>().ok()?)))
    else {
        return img.clone();
    };
    let tw = (img.width() / cols.max(1)).max(1);
    let th = (img.height() / rows.max(1)).max(1);
    let sx = (cx * tw).min(img.width().saturating_sub(tw));
    let sy = (cy * th).min(img.height().saturating_sub(th));
    image::imageops::crop_imm(img, sx, sy, tw, th).to_image()
}

fn apply_verticalframe(img: &RgbaImage, expr: &str) -> RgbaImage {
    let Some((count_s, idx_s)) = expr.split_once(':') else {
        return img.clone();
    };
    let count = count_s.trim().parse::<u32>().unwrap_or(1).max(1);
    let idx = idx_s.trim().parse::<u32>().unwrap_or(0) % count;
    let fh = (img.height() / count).max(1);
    let sy = (idx * fh).min(img.height().saturating_sub(fh));
    image::imageops::crop_imm(img, 0, sy, img.width(), fh).to_image()
}

fn apply_resize(img: &RgbaImage, expr: &str) -> RgbaImage {
    if let Some((w, h)) = parse_wh(expr) {
        return image::imageops::resize(img, w.max(1).min(512), h.max(1).min(512), FilterType::Nearest);
    }
    img.clone()
}

fn apply_opacity(img: &mut RgbaImage, expr: &str) {
    let ratio = expr.trim().parse::<u16>().unwrap_or(255).min(255);
    for px in img.pixels_mut() {
        px.0[3] = ((px.0[3] as u16 * ratio) / 255) as u8;
    }
}

fn apply_invert(img: &mut RgbaImage, expr: &str) {
    let inv_a = expr.contains('a');
    for px in img.pixels_mut() {
        if expr.contains('r') { px.0[0] = 255 - px.0[0]; }
        if expr.contains('g') { px.0[1] = 255 - px.0[1]; }
        if expr.contains('b') { px.0[2] = 255 - px.0[2]; }
        if inv_a { px.0[3] = 255 - px.0[3]; }
    }
}

fn apply_brighten(img: &mut RgbaImage) {
    for px in img.pixels_mut() {
        px.0[0] = px.0[0].saturating_add(40);
        px.0[1] = px.0[1].saturating_add(40);
        px.0[2] = px.0[2].saturating_add(40);
    }
}

fn apply_noalpha(img: &mut RgbaImage) {
    for px in img.pixels_mut() {
        px.0[3] = 255;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{ItemDef, NodeDef, TileSpec};

    #[test]
    fn atlas_helpers_crack_skin_palette_and_modifiers() {
        let mut reg = EngineRegistries::new();
        reg.register_node(NodeDef {
            name: "test:grass".into(),
            tiles: vec![TileSpec { name: "tile.png".into(), color: None }],
            color: Some([100, 200, 80, 255]),
            ..Default::default()
        });
        reg.register_item(ItemDef {
            name: "test:sword".into(),
            inventory_image: "sword.png^[multiply:#80ff80".into(),
            ..Default::default()
        });

        let tmp = std::env::temp_dir().join("cubex_tex_helper_test");
        let tex_dir = tmp.join("textures");
        let _ = std::fs::create_dir_all(&tex_dir);
        let tile = RgbaImage::from_pixel(16, 16, Rgba([200, 200, 200, 255]));
        tile.save(tex_dir.join("tile.png")).unwrap();
        let sword = RgbaImage::from_pixel(16, 16, Rgba([255, 255, 255, 255]));
        sword.save(tex_dir.join("sword.png")).unwrap();
        let mut pal = RgbaImage::new(16, 16);
        pal.put_pixel(0, 0, Rgba([120, 190, 60, 255]));
        pal.put_pixel(1, 0, Rgba([180, 170, 50, 255]));
        pal.save(tex_dir.join("test_palette.png")).unwrap();

        let atlas = TextureAtlas::build(&HashMap::new(), &tmp, None, &reg);
        assert_eq!(TILE_SIZE, 32);
        assert_eq!(MAX_ATLAS_LAYERS, 2048);
        assert!(atlas.get_rgba("tile.png").is_some());

        let cracks = atlas.get_crack_layers();
        assert!(cracks.iter().all(|&l| l > 0 && (l as usize) < MAX_ATLAS_LAYERS));
        assert_ne!(cracks[0], cracks[9]);

        let skin = atlas.get_player_skin_rgba();
        assert_eq!(skin.dimensions(), (64, 32));

        assert_eq!(atlas.sample_palette_color("test_palette.png", 0), Some([120, 190, 60, 255]));
        assert_eq!(atlas.sample_palette_color("test_palette.png", 1), Some([180, 170, 50, 255]));

        let item_img = atlas.get_item_rgba("test:sword", &reg).unwrap();
        assert_eq!(item_img.get_pixel(0, 0).0, [128, 255, 128, 255]);

        let node_img = atlas.get_item_rgba("test:grass", &reg).unwrap();
        assert!(node_img.get_pixel(0, 0).0[1] > node_img.get_pixel(0, 0).0[0]);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn placeholder_atlas_builds_instantly_and_preloaded_matches_build() {
        let t0 = std::time::Instant::now();
        let placeholder = TextureAtlas::placeholder();
        let elapsed = t0.elapsed();
        assert!(
            elapsed.as_millis() < 20,
            "placeholder atlas should build in <20ms, took {:?}",
            elapsed
        );
        let (w, h, layers, bytes) = placeholder.pack_rgba_bytes();
        assert_eq!((w, h), (TILE_SIZE, TILE_SIZE));
        assert!(layers >= 2);
        assert_eq!(bytes.len(), (w * h * 4 * layers) as usize);

        let mut reg = EngineRegistries::new();
        reg.register_node(NodeDef {
            name: "test:stone".into(),
            tiles: vec![TileSpec {
                name: "stone.png".into(),
                color: None,
            }],
            ..Default::default()
        });
        let tmp = std::env::temp_dir().join("cubex_preloaded_tex_test");
        let tex_dir = tmp.join("textures");
        let _ = std::fs::create_dir_all(&tex_dir);
        RgbaImage::from_pixel(16, 16, Rgba([120, 120, 125, 255]))
            .save(tex_dir.join("stone.png"))
            .unwrap();

        let built = TextureAtlas::build(&HashMap::new(), &tmp, None, &reg);
        let pre = PreloadedTextures::load(&HashMap::new(), &tmp, None);
        let from_pre = TextureAtlas::from_preloaded(pre, &reg);
        assert_eq!(built.layers.len(), from_pre.layers.len());
        assert_eq!(
            built.node_face_layers.get(&reg.get_content_id("test:stone")),
            from_pre.node_face_layers.get(&reg.get_content_id("test:stone"))
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
