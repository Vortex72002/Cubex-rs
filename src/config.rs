use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::world::{Difficulty, Gamerules, WorldGenType};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CoopMode {
    #[default]
    Off,
    SplitVertical,
    SplitHorizontal,
    DualMonitor,
    QuadSplit,
    DualScreenTwoEach,
    QuadMonitor,
}

impl CoopMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::SplitVertical => "split_vertical",
            Self::SplitHorizontal => "split_horizontal",
            Self::DualMonitor => "dual_monitor",
            Self::QuadSplit => "quad_split",
            Self::DualScreenTwoEach => "dual_screen_two_each",
            Self::QuadMonitor => "quad_monitor",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "Off (Single Player)",
            Self::SplitVertical => "2P Split-Screen (Left / Right)",
            Self::SplitHorizontal => "2P Split-Screen (Top / Bottom)",
            Self::DualMonitor => "2P Dual-Screen (2 Windows)",
            Self::QuadSplit => "4P Quad-Split (2x2 on 1 Screen)",
            Self::DualScreenTwoEach => "4P Dual-Screen (2 Splits x 2 Windows)",
            Self::QuadMonitor => "4P Multi-Screen (Up to 4 Windows)",
        }
    }

    pub fn from_str(s: &str) -> Self {
        Self::from_str_loose(s)
    }

    pub fn from_str_loose(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "split_vertical" | "split_lr" | "vertical" => Self::SplitVertical,
            "split_horizontal" | "split_tb" | "horizontal" => Self::SplitHorizontal,
            "dual_monitor" | "dual" | "2monitors" => Self::DualMonitor,
            "quad_split" | "4split" | "quad" => Self::QuadSplit,
            "dual_screen_two_each" | "dual_2x2" | "2x2_dual" | "2perscreen" => {
                Self::DualScreenTwoEach
            }
            "quad_monitor" | "4monitors" | "4screens" => Self::QuadMonitor,
            _ => Self::Off,
        }
    }

    pub fn cycle(self) -> Self {
        match self {
            Self::Off => Self::SplitVertical,
            Self::SplitVertical => Self::SplitHorizontal,
            Self::SplitHorizontal => Self::DualMonitor,
            Self::DualMonitor => Self::QuadSplit,
            Self::QuadSplit => Self::DualScreenTwoEach,
            Self::DualScreenTwoEach => Self::QuadMonitor,
            Self::QuadMonitor => Self::Off,
        }
    }

    pub fn is_active(self) -> bool {
        self != Self::Off
    }

    pub fn uses_secondary_windows(self) -> bool {
        matches!(
            self,
            Self::DualMonitor | Self::DualScreenTwoEach | Self::QuadMonitor
        )
    }

    pub fn min_players(self) -> usize {
        match self {
            Self::Off => 1,
            Self::SplitVertical | Self::SplitHorizontal | Self::DualMonitor => 2,
            Self::QuadSplit | Self::DualScreenTwoEach | Self::QuadMonitor => 2,
        }
    }

    pub fn max_players(self) -> usize {
        match self {
            Self::Off => 1,
            Self::SplitVertical | Self::SplitHorizontal | Self::DualMonitor => 2,
            Self::QuadSplit | Self::DualScreenTwoEach | Self::QuadMonitor => 4,
        }
    }

    pub fn active_player_count(self, configured_count: u8) -> usize {
        match self {
            Self::Off => 1,
            Self::SplitVertical | Self::SplitHorizontal | Self::DualMonitor => 2,
            Self::QuadSplit | Self::DualScreenTwoEach | Self::QuadMonitor => {
                (configured_count as usize).clamp(2, 4)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum PlayerInputDevice {
    #[default]
    KeyboardMouse,
    Gamepad(usize),
    Keyboard2,
}

impl PlayerInputDevice {
    pub fn as_str(self) -> String {
        match self {
            Self::KeyboardMouse => "kbm".into(),
            Self::Gamepad(idx) => format!("gamepad_{idx}"),
            Self::Keyboard2 => "keyboard2".into(),
        }
    }

    pub fn label(self) -> String {
        match self {
            Self::KeyboardMouse => "Keyboard + Mouse".into(),
            Self::Gamepad(idx) => format!("Controller #{}", idx + 1),
            Self::Keyboard2 => "Keyboard 2 (Arrows/IJKL)".into(),
        }
    }

    pub fn from_str(s: &str) -> Self {
        let lower = s.trim().to_ascii_lowercase();
        if let Some(rest) = lower.strip_prefix("gamepad_") {
            if let Ok(idx) = rest.parse::<usize>() {
                return Self::Gamepad(idx.min(3));
            }
        }
        if lower == "keyboard2" || lower == "kb2" {
            Self::Keyboard2
        } else {
            Self::KeyboardMouse
        }
    }

    pub fn cycle(self) -> Self {
        match self {
            Self::KeyboardMouse => Self::Gamepad(0),
            Self::Gamepad(0) => Self::Gamepad(1),
            Self::Gamepad(1) => Self::Gamepad(2),
            Self::Gamepad(2) => Self::Gamepad(3),
            Self::Gamepad(_) => Self::Keyboard2,
            Self::Keyboard2 => Self::KeyboardMouse,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum GraphicsPreset {
    #[default]
    Performance,
    Balanced,
    Cinematic,
    Custom,
}

impl GraphicsPreset {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Performance => "performance",
            Self::Balanced => "balanced",
            Self::Cinematic => "cinematic",
            Self::Custom => "custom",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Performance => "Performance",
            Self::Balanced => "Balanced",
            Self::Cinematic => "Cinematic",
            Self::Custom => "Custom",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::Performance => "Vanilla-speed potato mode: flat shading, classic drops, max FPS.",
            Self::Balanced => "Dynamic torch light, warm sun & fog, wind foliage, shatter physics.",
            Self::Cinematic => "All Balanced enhancements plus quarter-res HDR Bloom & ACES tone mapping.",
            Self::Custom => "Custom combination of Enhanced Graphics & Juice toggles.",
        }
    }

    pub fn from_str_loose(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "balanced" | "enhanced" | "fancy" => Self::Balanced,
            "cinematic" | "fabulous" | "ultra" => Self::Cinematic,
            "custom" => Self::Custom,
            _ => Self::Performance,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GraphicsQuality {
    Fast,
    Fancy,
    Fabulous,
}

impl GraphicsQuality {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Fast => "Fast",
            Self::Fancy => "Fancy",
            Self::Fabulous => "Fabulous!",
        }
    }

    pub fn cycle(self) -> Self {
        match self {
            Self::Fast => Self::Fancy,
            Self::Fancy => Self::Fabulous,
            Self::Fabulous => Self::Fast,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioSettings {
    pub master: f32,
    pub music: f32,
    pub blocks: f32,
    pub weather: f32,
    pub hostile: f32,
    pub friendly: f32,
    pub ambient: f32,
}

impl Default for AudioSettings {
    fn default() -> Self {
        Self {
            master: 0.80,
            music: 0.70,
            blocks: 1.0,
            weather: 0.80,
            hostile: 0.90,
            friendly: 0.90,
            ambient: 0.75,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EngineSettings {
    pub player_name: String,
    pub fov: f32,
    pub render_distance_chunks: u32,
    pub max_fps: u32,
    pub vsync: bool,
    pub graphics: GraphicsQuality,
    pub graphics_preset: GraphicsPreset,
    pub smooth_lighting: bool,
    pub dynamic_shadows: bool,
    pub procedural_sky_clouds: bool,
    pub handheld_torch_light: bool,
    pub warm_sun_and_atmosphere: bool,
    pub bloom_glow: bool,
    pub block_shatter_physics: bool,
    pub wind_swaying_foliage: bool,
    pub extruded_held_items: bool,
    pub liquids_pointable: bool,
    pub clouds_enabled: bool,
    pub waving_water: bool,
    pub mipmapping: bool,
    pub gui_scale: f32,
    pub brightness: f32,
    pub audio: AudioSettings,
    pub mouse_sensitivity: f32,
    pub invert_mouse: bool,
    pub auto_jump: bool,
    pub toggle_sprint: bool,
    pub zoom_key: String,
    pub f3_enabled: bool,
    pub coords_hud: bool,
    pub active_resource_pack: String,
    pub keybinds: BTreeMap<String, String>,
    pub world_gen_type: WorldGenType,
    pub difficulty: Difficulty,
    pub gamerules: Gamerules,
    pub coop_mode: CoopMode,
    pub coop_player_count: u8,
    pub p2_name: String,
    pub p3_name: String,
    pub p4_name: String,
    pub p1_input: PlayerInputDevice,
    pub p2_input: PlayerInputDevice,
    pub p3_input: PlayerInputDevice,
    pub p4_input: PlayerInputDevice,
    pub controller_sensitivity: f32,
    pub controller_deadzone: f32,
    pub controller_invert_y: bool,
    pub touch_controls: bool,
    pub touch_sensitivity: f32,
}

impl Default for EngineSettings {
    fn default() -> Self {
        let mut keybinds = BTreeMap::new();
        keybinds.insert("keymap_forward".into(), "W".into());
        keybinds.insert("keymap_backward".into(), "S".into());
        keybinds.insert("keymap_left".into(), "A".into());
        keybinds.insert("keymap_right".into(), "D".into());
        keybinds.insert("keymap_jump".into(), "Space".into());
        keybinds.insert("keymap_sneak".into(), "LShift".into());
        keybinds.insert("keymap_aux1".into(), "LControl".into());
        keybinds.insert("keymap_inventory".into(), "E".into());
        keybinds.insert("keymap_zoom".into(), "C".into());
        keybinds.insert("keymap_drop".into(), "Q".into());
        keybinds.insert("keymap_chat".into(), "T".into());
        keybinds.insert("keymap_freemove".into(), "K".into());
        keybinds.insert("keymap_noclip".into(), "H".into());

        Self {
            player_name: "Vortex".into(),
            fov: 80.0,
            render_distance_chunks: 10,
            max_fps: 0,
            vsync: false,
            graphics: GraphicsQuality::Fast,
            graphics_preset: GraphicsPreset::Performance,
            smooth_lighting: true,
            dynamic_shadows: false,
            procedural_sky_clouds: true,
            handheld_torch_light: false,
            warm_sun_and_atmosphere: false,
            bloom_glow: false,
            block_shatter_physics: false,
            wind_swaying_foliage: false,
            extruded_held_items: true,
            liquids_pointable: false,
            clouds_enabled: true,
            waving_water: true,
            mipmapping: false,
            gui_scale: 1.0,
            brightness: 1.0,
            audio: AudioSettings::default(),
            mouse_sensitivity: 0.25,
            invert_mouse: false,
            auto_jump: false,
            toggle_sprint: true,
            zoom_key: "C".into(),
            f3_enabled: false,
            coords_hud: true,
            active_resource_pack: String::new(),
            keybinds,
            world_gen_type: WorldGenType::Default,
            difficulty: Difficulty::Normal,
            gamerules: Gamerules::default(),
            coop_mode: CoopMode::Off,
            coop_player_count: 4,
            p2_name: "Player2".into(),
            p3_name: "Player3".into(),
            p4_name: "Player4".into(),
            p1_input: PlayerInputDevice::KeyboardMouse,
            p2_input: PlayerInputDevice::Gamepad(0),
            p3_input: PlayerInputDevice::Gamepad(1),
            p4_input: PlayerInputDevice::Gamepad(2),
            controller_sensitivity: 1.0,
            controller_deadzone: 0.14,
            controller_invert_y: false,
            touch_controls: false,
            touch_sensitivity: 1.0,
        }
    }
}

impl EngineSettings {
    pub fn player_name_for_slot(&self, slot: usize) -> &str {
        match slot {
            0 => &self.player_name,
            1 => &self.p2_name,
            2 => &self.p3_name,
            _ => &self.p4_name,
        }
    }

    pub fn player_name_for_slot_mut(&mut self, slot: usize) -> &mut String {
        match slot {
            0 => &mut self.player_name,
            1 => &mut self.p2_name,
            2 => &mut self.p3_name,
            _ => &mut self.p4_name,
        }
    }

    pub fn player_input_for_slot(&self, slot: usize) -> PlayerInputDevice {
        match slot {
            0 => self.p1_input,
            1 => self.p2_input,
            2 => self.p3_input,
            _ => self.p4_input,
        }
    }

    pub fn set_player_input_for_slot(&mut self, slot: usize, dev: PlayerInputDevice) {
        match slot {
            0 => self.p1_input = dev,
            1 => self.p2_input = dev,
            2 => self.p3_input = dev,
            _ => self.p4_input = dev,
        }
    }

    pub fn cloud_octaves(&self) -> u32 {
        if !self.procedural_sky_clouds {
            0
        } else if self.graphics_preset == GraphicsPreset::Performance
            || (!self.warm_sun_and_atmosphere && !self.bloom_glow)
        {
            2
        } else {
            4
        }
    }

    pub fn apply_graphics_preset(&mut self, preset: GraphicsPreset) {
        self.graphics_preset = preset;
        match preset {
            GraphicsPreset::Performance => {
                self.procedural_sky_clouds = true;
                self.clouds_enabled = true;
                self.dynamic_shadows = false;
                self.handheld_torch_light = false;
                self.warm_sun_and_atmosphere = false;
                self.bloom_glow = false;
                self.block_shatter_physics = false;
                self.wind_swaying_foliage = false;
                self.extruded_held_items = true;
                self.graphics = GraphicsQuality::Fast;
            }
            GraphicsPreset::Balanced => {
                self.procedural_sky_clouds = true;
                self.clouds_enabled = true;
                self.dynamic_shadows = true;
                self.handheld_torch_light = true;
                self.warm_sun_and_atmosphere = true;
                self.bloom_glow = true;
                self.block_shatter_physics = true;
                self.wind_swaying_foliage = true;
                self.extruded_held_items = true;
                self.graphics = GraphicsQuality::Fancy;
            }
            GraphicsPreset::Cinematic => {
                self.procedural_sky_clouds = true;
                self.clouds_enabled = true;
                self.dynamic_shadows = true;
                self.handheld_torch_light = true;
                self.warm_sun_and_atmosphere = true;
                self.bloom_glow = true;
                self.block_shatter_physics = true;
                self.wind_swaying_foliage = true;
                self.extruded_held_items = true;
                self.graphics = GraphicsQuality::Fabulous;
            }
            GraphicsPreset::Custom => {}
        }
    }

    pub fn sync_graphics_preset_from_toggles(&mut self) {
        self.clouds_enabled = self.procedural_sky_clouds;
        let sig = (
            self.procedural_sky_clouds,
            self.handheld_torch_light,
            self.warm_sun_and_atmosphere,
            self.bloom_glow,
            self.block_shatter_physics,
            self.wind_swaying_foliage,
            self.extruded_held_items,
        );
        self.graphics_preset = match sig {
            (true, false, false, false, false, false, true) if !self.dynamic_shadows => {
                self.graphics = GraphicsQuality::Fast;
                GraphicsPreset::Performance
            }
            (true, true, true, true, true, true, true) => {
                if self.graphics_preset == GraphicsPreset::Balanced {
                    self.graphics = GraphicsQuality::Fancy;
                    GraphicsPreset::Balanced
                } else {
                    self.graphics = GraphicsQuality::Fabulous;
                    GraphicsPreset::Cinematic
                }
            }
            _ => {
                self.graphics = if self.bloom_glow {
                    GraphicsQuality::Fabulous
                } else if self.warm_sun_and_atmosphere
                    || self.handheld_torch_light
                    || self.wind_swaying_foliage
                {
                    GraphicsQuality::Fancy
                } else {
                    GraphicsQuality::Fast
                };
                GraphicsPreset::Custom
            }
        };
    }

    pub fn save_toml(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let body = toml::to_string_pretty(self)?;
        std::fs::write(path, body)?;
        Ok(())
    }

    pub fn load_toml(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let parsed: Self = toml::from_str(&text)?;
        Ok(parsed)
    }

    pub fn overlay_graphics_from_toml(&mut self, path: &Path) {
        if !path.is_file() {
            return;
        }
        if let Ok(loaded) = Self::load_toml(path) {
            self.graphics_preset = loaded.graphics_preset;
            self.procedural_sky_clouds = loaded.procedural_sky_clouds;
            self.clouds_enabled = loaded.procedural_sky_clouds;
            self.handheld_torch_light = loaded.handheld_torch_light;
            self.warm_sun_and_atmosphere = loaded.warm_sun_and_atmosphere;
            self.bloom_glow = loaded.bloom_glow;
            self.block_shatter_physics = loaded.block_shatter_physics;
            self.wind_swaying_foliage = loaded.wind_swaying_foliage;
            self.extruded_held_items = loaded.extruded_held_items;
            self.graphics = loaded.graphics;
        }
    }
}

/// Deduplicated minetest.conf reader/writer synchronized with EngineSettings and Lua core.settings.
#[derive(Debug, Clone)]
pub struct MinetestConfig {
    pub path: PathBuf,
    pub values: BTreeMap<String, String>,
}

impl MinetestConfig {
    pub fn load(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref().to_path_buf();
        let mut values = Self::default_map();
        if let Ok(text) = std::fs::read_to_string(&path) {
            for line in text.lines() {
                let trimmed = line.trim();
                if trimmed.is_empty() || trimmed.starts_with('#') {
                    continue;
                }
                if let Some((k, v)) = trimmed.split_once('=') {
                    values.insert(k.trim().to_string(), v.trim().to_string());
                }
            }
        }
        Self { path, values }
    }

    fn default_map() -> BTreeMap<String, String> {
        let mut m = BTreeMap::new();
        m.insert("name".into(), "Vortex".into());
        m.insert("fov".into(), "80".into());
        m.insert("viewing_range".into(), "160".into());
        m.insert("fps_max".into(), "0".into());
        m.insert("vsync".into(), "false".into());
        m.insert("smooth_lighting".into(), "true".into());
        m.insert("enable_dynamic_shadows".into(), "false".into());
        m.insert("cubex_mod_dynamic_lights".into(), "false".into());
        m.insert("cubex_graphics_preset".into(), "performance".into());
        m.insert("cubex_warm_sun_atmosphere".into(), "false".into());
        m.insert("cubex_bloom_glow".into(), "false".into());
        m.insert("cubex_block_shatter_physics".into(), "false".into());
        m.insert("cubex_wind_swaying_foliage".into(), "false".into());
        m.insert("cubex_extruded_held_items".into(), "true".into());
        m.insert("liquids_pointable".into(), "false".into());
        m.insert("enable_shaders".into(), "true".into());
        m.insert("enable_waving_water".into(), "true".into());
        m.insert("enable_waving_leaves".into(), "false".into());
        m.insert("enable_3d_clouds".into(), "true".into());
        m.insert("enable_clouds".into(), "true".into());
        m.insert("display_gamma".into(), "1.0".into());
        m.insert("gui_scaling".into(), "1.0".into());
        m.insert("sound_volume".into(), "0.80".into());
        m.insert("mouse_sensitivity".into(), "0.25".into());
        m.insert("invert_mouse".into(), "false".into());
        m.insert("autojump".into(), "false".into());
        m.insert("creative_mode".into(), "false".into());
        m.insert("enable_damage".into(), "true".into());
        m.insert("mcl_enable_hunger".into(), "true".into());
        m.insert("mcl_keepInventory".into(), "true".into());
        m.insert("cubex_do_daylight_cycle".into(), "true".into());
        m.insert("enable_pvp".into(), "true".into());
        m.insert("cubex_fall_damage".into(), "true".into());
        m.insert("cubex_world_gen_type".into(), "default".into());
        m.insert("cubex_difficulty".into(), "normal".into());
        m.insert("mobs_spawn".into(), "true".into());
        m.insert("default_stack_max".into(), "64".into());
        m.insert("movement_gravity".into(), "9.81".into());
        m.insert("default_privs".into(), "interact, shout".into());
        m.insert("movement_speed_walk".into(), "4.0".into());
        m.insert("movement_speed_crouch".into(), "1.35".into());
        m.insert("movement_speed_fast".into(), "20.0".into());
        m.insert("movement_speed_climb".into(), "3.0".into());
        m.insert("movement_speed_jump".into(), "6.5".into());
        m.insert("mg_flags".into(), "caves,dungeons,light,decorations,biomes,ores".into());
        m.insert("keymap_zoom".into(), "C".into());
        m.insert("cubex_hud_f3".into(), "false".into());
        m.insert("cubex_hud_coords".into(), "true".into());
        m.insert("cubex_coop_mode".into(), "off".into());
        m.insert("cubex_coop_player_count".into(), "4".into());
        m.insert("cubex_p2_name".into(), "Player2".into());
        m.insert("cubex_p3_name".into(), "Player3".into());
        m.insert("cubex_p4_name".into(), "Player4".into());
        m.insert("cubex_p1_input".into(), "kbm".into());
        m.insert("cubex_p2_input".into(), "gamepad_0".into());
        m.insert("cubex_p3_input".into(), "gamepad_1".into());
        m.insert("cubex_p4_input".into(), "gamepad_2".into());
        m.insert("cubex_controller_sens".into(), "1.00".into());
        m.insert("cubex_controller_deadzone".into(), "0.14".into());
        m.insert("cubex_controller_invert".into(), "false".into());
        m.insert("cubex_touch_controls".into(), "false".into());
        m.insert("cubex_touch_sens".into(), "1.00".into());
        m
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(String::as_str)
    }

    pub fn get_bool(&self, key: &str, default: bool) -> bool {
        match self.values.get(key).map(|s| s.to_ascii_lowercase()) {
            Some(ref v) if v == "true" || v == "1" || v == "yes" || v == "on" => true,
            Some(ref v) if v == "false" || v == "0" || v == "no" || v == "off" => false,
            _ => default,
        }
    }

    pub fn get_f32(&self, key: &str, default: f32) -> f32 {
        self.values
            .get(key)
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(default)
    }

    pub fn get_u32(&self, key: &str, default: u32) -> u32 {
        self.values
            .get(key)
            .and_then(|s| s.parse::<f32>().ok())
            .map(|v| v.max(0.0).round() as u32)
            .unwrap_or(default)
    }

    pub fn set(&mut self, key: impl Into<String>, value: impl ToString) {
        self.values.insert(key.into(), value.to_string());
    }

    pub fn remove(&mut self, key: &str) {
        self.values.remove(key);
    }

    pub fn save(&self) -> anyhow::Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut out = String::from("# Cubex-RS / Mineclonia Configuration (auto-managed & deduplicated)\n\n");
        for (k, v) in &self.values {
            out.push_str(&format!("{k} = {v}\n"));
        }
        std::fs::write(&self.path, out)?;
        Ok(())
    }

    pub fn to_engine_settings(&self, default_pack: &str) -> EngineSettings {
        let mut es = EngineSettings::default();
        if let Some(n) = self.get("name") {
            if !n.is_empty() {
                es.player_name = n.to_string();
            }
        }
        es.fov = self.get_f32("fov", 80.0).clamp(30.0, 110.0);
        let vr = self.get_u32("viewing_range", 160);
        es.render_distance_chunks = (vr / 16).clamp(2, 32);
        es.max_fps = self.get_u32("fps_max", 0);
        es.vsync = self.get_bool("vsync", false);
        es.smooth_lighting = self.get_bool("smooth_lighting", true);
        es.dynamic_shadows = self.get_bool("enable_dynamic_shadows", false);
        let legacy_clouds = self.get_bool("enable_clouds", true);
        es.procedural_sky_clouds = self.get_bool("cubex_procedural_sky_clouds", legacy_clouds);
        es.handheld_torch_light = self.get_bool("cubex_mod_dynamic_lights", false);
        es.warm_sun_and_atmosphere = self.get_bool("cubex_warm_sun_atmosphere", false);
        es.bloom_glow = self.get_bool("cubex_bloom_glow", self.get_bool("enable_bloom", false));
        es.block_shatter_physics = self.get_bool("cubex_block_shatter_physics", false);
        es.wind_swaying_foliage =
            self.get_bool("cubex_wind_swaying_foliage", self.get_bool("enable_waving_leaves", false));
        es.extruded_held_items = self.get_bool("cubex_extruded_held_items", true);
        if let Some(p) = self.get("cubex_graphics_preset") {
            es.graphics_preset = GraphicsPreset::from_str_loose(p);
        } else {
            es.sync_graphics_preset_from_toggles();
        }
        es.liquids_pointable = self.get_bool("liquids_pointable", false);
        es.clouds_enabled = es.procedural_sky_clouds;
        es.waving_water = self.get_bool("enable_waving_water", true);
        es.mipmapping = self.get_bool("mip_map", false);
        es.gui_scale = self.get_f32("gui_scaling", 1.0).clamp(0.75, 2.0);
        es.brightness = self.get_f32("display_gamma", 1.0).clamp(0.5, 1.5);
        es.audio.master = self.get_f32("sound_volume", 0.80).clamp(0.0, 1.0);
        es.audio.music = self.get_f32("cubex_vol_music", 0.70).clamp(0.0, 1.0);
        es.audio.blocks = self.get_f32("cubex_vol_blocks", 1.0).clamp(0.0, 1.0);
        es.audio.weather = self.get_f32("cubex_vol_weather", 0.80).clamp(0.0, 1.0);
        es.audio.hostile = self.get_f32("cubex_vol_hostile", 0.90).clamp(0.0, 1.0);
        es.audio.friendly = self.get_f32("cubex_vol_friendly", 0.90).clamp(0.0, 1.0);
        es.audio.ambient = self.get_f32("cubex_vol_ambient", 0.75).clamp(0.0, 1.0);
        es.mouse_sensitivity = self.get_f32("mouse_sensitivity", 0.25).clamp(0.05, 1.5);
        es.invert_mouse = self.get_bool("invert_mouse", false);
        es.auto_jump = self.get_bool("autojump", false);
        es.toggle_sprint = self.get_bool("cubex_mod_toggle_sprint", true);
        es.f3_enabled = self.get_bool("cubex_hud_f3", false);
        es.coords_hud = self.get_bool("cubex_hud_coords", true);
        es.active_resource_pack = self
            .get("texture_path")
            .map(ToString::to_string)
            .unwrap_or_else(|| default_pack.to_string());
        if let Some(zk) = self.get("keymap_zoom") {
            es.zoom_key = zk.replace("KEY_KEY_", "").replace("KEY_", "");
        }
        // Load persisted keybinds
        let kb_keys: Vec<String> = es.keybinds.keys().cloned().collect();
        for k in kb_keys {
            if let Some(v) = self.get(&k) {
                es.keybinds.insert(k, v.replace("KEY_KEY_", "").replace("KEY_", ""));
            }
        }
        es.world_gen_type = self
            .get("cubex_world_gen_type")
            .map(WorldGenType::from_str)
            .unwrap_or(WorldGenType::Default);
        es.difficulty = self
            .get("cubex_difficulty")
            .map(Difficulty::from_str)
            .unwrap_or(Difficulty::Normal);
        es.coop_mode = self
            .get("cubex_coop_mode")
            .map(CoopMode::from_str)
            .unwrap_or(CoopMode::Off);
        es.coop_player_count = self.get_u32("cubex_coop_player_count", 4).clamp(2, 4) as u8;
        if let Some(p2n) = self.get("cubex_p2_name") {
            if !p2n.trim().is_empty() {
                es.p2_name = p2n.trim().to_string();
            }
        }
        if let Some(p3n) = self.get("cubex_p3_name") {
            if !p3n.trim().is_empty() {
                es.p3_name = p3n.trim().to_string();
            }
        }
        if let Some(p4n) = self.get("cubex_p4_name") {
            if !p4n.trim().is_empty() {
                es.p4_name = p4n.trim().to_string();
            }
        }
        es.p1_input = self
            .get("cubex_p1_input")
            .map(PlayerInputDevice::from_str)
            .unwrap_or(PlayerInputDevice::KeyboardMouse);
        es.p2_input = self
            .get("cubex_p2_input")
            .map(PlayerInputDevice::from_str)
            .unwrap_or(PlayerInputDevice::Gamepad(0));
        es.p3_input = self
            .get("cubex_p3_input")
            .map(PlayerInputDevice::from_str)
            .unwrap_or(PlayerInputDevice::Gamepad(1));
        es.p4_input = self
            .get("cubex_p4_input")
            .map(PlayerInputDevice::from_str)
            .unwrap_or(PlayerInputDevice::Gamepad(2));
        es.controller_sensitivity = self.get_f32("cubex_controller_sens", 1.0).clamp(0.25, 3.0);
        es.controller_deadzone = self.get_f32("cubex_controller_deadzone", 0.14).clamp(0.05, 0.35);
        es.controller_invert_y = self.get_bool("cubex_controller_invert", false);
        es.touch_controls = self.get_bool("cubex_touch_controls", false);
        es.touch_sensitivity = self.get_f32("cubex_touch_sens", 1.0).clamp(0.25, 3.0);
        es.gamerules = Gamerules {
            keep_inventory: self.get_bool("mcl_keepInventory", true),
            do_daylight_cycle: self.get_bool("cubex_do_daylight_cycle", true),
            pvp: self.get_bool("enable_pvp", true),
            fall_damage: self.get_bool("cubex_fall_damage", true),
        };
        es.graphics = if !self.get_bool("enable_shaders", true) {
            GraphicsQuality::Fast
        } else if self.get_bool("enable_post_processing", false) {
            GraphicsQuality::Fabulous
        } else {
            GraphicsQuality::Fancy
        };
        es
    }

    pub fn apply_engine_settings(&mut self, es: &EngineSettings) {
        self.set("name", &es.player_name);
        self.set("fov", es.fov.round() as i32);
        self.set("viewing_range", es.render_distance_chunks * 16);
        self.set("fps_max", es.max_fps);
        self.set("vsync", es.vsync);
        self.set("smooth_lighting", es.smooth_lighting);
        self.set("enable_dynamic_shadows", es.dynamic_shadows);
        self.set("cubex_graphics_preset", es.graphics_preset.as_str());
        self.set("cubex_procedural_sky_clouds", es.procedural_sky_clouds);
        self.set("cubex_mod_dynamic_lights", es.handheld_torch_light);
        self.set("cubex_warm_sun_atmosphere", es.warm_sun_and_atmosphere);
        self.set("cubex_bloom_glow", es.bloom_glow);
        self.set("cubex_block_shatter_physics", es.block_shatter_physics);
        self.set("cubex_wind_swaying_foliage", es.wind_swaying_foliage);
        self.set("cubex_extruded_held_items", es.extruded_held_items);
        self.set("liquids_pointable", es.liquids_pointable);
        self.set("enable_clouds", es.procedural_sky_clouds);
        self.set("enable_waving_water", es.waving_water);
        self.set("enable_waving_leaves", es.wind_swaying_foliage);
        self.set("mip_map", es.mipmapping);
        self.set("gui_scaling", format!("{:.2}", es.gui_scale));
        self.set("display_gamma", format!("{:.2}", es.brightness));
        self.set("sound_volume", format!("{:.2}", es.audio.master));
        self.set("cubex_vol_music", format!("{:.2}", es.audio.music));
        self.set("cubex_vol_blocks", format!("{:.2}", es.audio.blocks));
        self.set("cubex_vol_weather", format!("{:.2}", es.audio.weather));
        self.set("cubex_vol_hostile", format!("{:.2}", es.audio.hostile));
        self.set("cubex_vol_friendly", format!("{:.2}", es.audio.friendly));
        self.set("cubex_vol_ambient", format!("{:.2}", es.audio.ambient));
        self.set("mouse_sensitivity", format!("{:.2}", es.mouse_sensitivity));
        self.set("invert_mouse", es.invert_mouse);
        self.set("autojump", es.auto_jump);
        self.set("cubex_mod_toggle_sprint", es.toggle_sprint);
        self.set("cubex_hud_f3", es.f3_enabled);
        self.set("cubex_hud_coords", es.coords_hud);
        self.set("texture_path", &es.active_resource_pack);
        self.set("keymap_zoom", &es.zoom_key);
        for (k, v) in &es.keybinds {
            self.set(k, v);
        }
        self.set("cubex_world_gen_type", es.world_gen_type.as_str());
        self.set("cubex_difficulty", es.difficulty.as_str());
        self.set("cubex_coop_mode", es.coop_mode.as_str());
        self.set("cubex_coop_player_count", es.coop_player_count.clamp(2, 4));
        self.set("cubex_p2_name", &es.p2_name);
        self.set("cubex_p3_name", &es.p3_name);
        self.set("cubex_p4_name", &es.p4_name);
        self.set("cubex_p1_input", es.p1_input.as_str());
        self.set("cubex_p2_input", es.p2_input.as_str());
        self.set("cubex_p3_input", es.p3_input.as_str());
        self.set("cubex_p4_input", es.p4_input.as_str());
        self.set("cubex_controller_sens", format!("{:.2}", es.controller_sensitivity));
        self.set("cubex_controller_deadzone", format!("{:.2}", es.controller_deadzone));
        self.set("cubex_controller_invert", es.controller_invert_y);
        self.set("cubex_touch_controls", es.touch_controls);
        self.set("cubex_touch_sens", format!("{:.2}", es.touch_sensitivity));
        self.set("mcl_keepInventory", es.gamerules.keep_inventory);
        self.set("cubex_do_daylight_cycle", es.gamerules.do_daylight_cycle);
        self.set("enable_pvp", es.gamerules.pvp);
        self.set("cubex_fall_damage", es.gamerules.fall_damage);
        match es.graphics {
            GraphicsQuality::Fast => {
                self.set("enable_shaders", false);
                self.set("enable_post_processing", es.bloom_glow);
                self.set("enable_bloom", es.bloom_glow);
            }
            GraphicsQuality::Fancy => {
                self.set("enable_shaders", true);
                self.set("enable_post_processing", es.bloom_glow);
                self.set("enable_bloom", es.bloom_glow);
            }
            GraphicsQuality::Fabulous => {
                self.set("enable_shaders", true);
                self.set("enable_post_processing", true);
                self.set("enable_bloom", true);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_settings_new_fields_roundtrip_dedup() {
        let dir = std::env::temp_dir().join("cubex_rs_cfg_roundtrip");
        let _ = std::fs::create_dir_all(&dir);
        let conf_path = dir.join("minetest.conf");
        let mut cfg = MinetestConfig::load(&conf_path);
        let mut es = EngineSettings::default();
        es.handheld_torch_light = true;
        es.liquids_pointable = true;
        es.world_gen_type = WorldGenType::Amplified;
        es.difficulty = Difficulty::Hard;
        es.gamerules = Gamerules {
            keep_inventory: true,
            do_daylight_cycle: false,
            pvp: false,
            fall_damage: false,
        };
        cfg.apply_engine_settings(&es);
        cfg.apply_engine_settings(&es);
        cfg.save().unwrap();

        let text = std::fs::read_to_string(&conf_path).unwrap();
        assert_eq!(text.lines().filter(|l| l.starts_with("cubex_difficulty =")).count(), 1);

        let reloaded = MinetestConfig::load(&conf_path).to_engine_settings("");
        assert!(reloaded.handheld_torch_light);
        assert!(reloaded.liquids_pointable);
        assert_eq!(reloaded.world_gen_type, WorldGenType::Amplified);
        assert_eq!(reloaded.difficulty, Difficulty::Hard);
        assert!(reloaded.gamerules.keep_inventory);
        assert!(!reloaded.gamerules.do_daylight_cycle);
        assert!(!reloaded.gamerules.pvp);
        assert!(!reloaded.gamerules.fall_damage);
    }

    #[test]
    fn graphics_presets_and_settings_toml_roundtrip() {
        let def = EngineSettings::default();
        assert_eq!(def.graphics_preset, GraphicsPreset::Performance);
        assert!(def.procedural_sky_clouds);
        assert_eq!(def.cloud_octaves(), 2);
        assert!(!def.handheld_torch_light);
        assert!(!def.warm_sun_and_atmosphere);
        assert!(!def.bloom_glow);
        assert!(!def.block_shatter_physics);
        assert!(!def.wind_swaying_foliage);
        assert!(def.extruded_held_items);

        let mut es = def.clone();
        es.apply_graphics_preset(GraphicsPreset::Balanced);
        assert_eq!(es.graphics_preset, GraphicsPreset::Balanced);
        assert!(es.procedural_sky_clouds);
        assert_eq!(es.cloud_octaves(), 4);
        assert!(es.dynamic_shadows);
        assert!(es.handheld_torch_light);
        assert!(es.warm_sun_and_atmosphere);
        assert!(es.bloom_glow);
        assert!(es.block_shatter_physics);
        assert!(es.wind_swaying_foliage);
        assert!(es.extruded_held_items);

        es.apply_graphics_preset(GraphicsPreset::Cinematic);
        assert_eq!(es.graphics_preset, GraphicsPreset::Cinematic);
        assert!(es.procedural_sky_clouds);
        assert_eq!(es.cloud_octaves(), 4);
        assert!(es.dynamic_shadows);
        assert!(es.bloom_glow);

        es.bloom_glow = false;
        es.wind_swaying_foliage = false;
        es.procedural_sky_clouds = false;
        es.sync_graphics_preset_from_toggles();
        assert_eq!(es.graphics_preset, GraphicsPreset::Custom);
        assert_eq!(es.cloud_octaves(), 0);

        let dir = std::env::temp_dir().join("cubex_rs_toml_roundtrip");
        let _ = std::fs::remove_dir_all(&dir);
        let toml_path = dir.join("settings.toml");
        es.save_toml(&toml_path).unwrap();
        let loaded = EngineSettings::load_toml(&toml_path).unwrap();
        assert_eq!(loaded.graphics_preset, GraphicsPreset::Custom);
        assert!(!loaded.procedural_sky_clouds);
        assert!(loaded.handheld_torch_light);
        assert!(loaded.warm_sun_and_atmosphere);
        assert!(!loaded.bloom_glow);
        assert!(loaded.block_shatter_physics);
        assert!(!loaded.wind_swaying_foliage);
        assert!(loaded.extruded_held_items);

        // MinetestConfig roundtrip too
        let conf_path = dir.join("minetest.conf");
        let mut cfg = MinetestConfig::load(&conf_path);
        cfg.apply_engine_settings(&es);
        cfg.save().unwrap();
        let mc_loaded = MinetestConfig::load(&conf_path).to_engine_settings("");
        assert_eq!(mc_loaded.graphics_preset, GraphicsPreset::Custom);
        assert_eq!(mc_loaded.procedural_sky_clouds, es.procedural_sky_clouds);
        assert_eq!(mc_loaded.clouds_enabled, es.procedural_sky_clouds);
        assert_eq!(mc_loaded.handheld_torch_light, es.handheld_torch_light);
        assert_eq!(mc_loaded.warm_sun_and_atmosphere, es.warm_sun_and_atmosphere);
        assert_eq!(mc_loaded.bloom_glow, es.bloom_glow);
        assert_eq!(mc_loaded.block_shatter_physics, es.block_shatter_physics);
        assert_eq!(mc_loaded.wind_swaying_foliage, es.wind_swaying_foliage);
        assert_eq!(mc_loaded.extruded_held_items, es.extruded_held_items);

        // Partial/legacy TOML backward compatibility via #[serde(default)]
        let partial_path = dir.join("partial.toml");
        std::fs::write(&partial_path, "player_name = \"LegacyPlayer\"\nfov = 95.0\n").unwrap();
        let partial = EngineSettings::load_toml(&partial_path).unwrap();
        assert_eq!(partial.player_name, "LegacyPlayer");
        assert!((partial.fov - 95.0).abs() < 0.01);
        assert!(partial.procedural_sky_clouds);
        assert_eq!(partial.graphics_preset, GraphicsPreset::Performance);
        assert!(partial.extruded_held_items);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
