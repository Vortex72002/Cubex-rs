use egui::{Align2, Color32, Pos2, Rect, Stroke, Vec2};

use crate::config::{EngineSettings, GraphicsPreset, MinetestConfig};
use crate::network::tunnel::PlayitTunnelManager;
use crate::physics::GameMode;
use crate::textures::ResourcePackInfo;
use crate::ui::theme::{
    mc_button, mc_button_styled, mc_button_with_sound, mc_shadow_text, mc_slider_f32,
    paint_pause_overlay, McTheme,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PausePage {
    Main,
    Options,
    Video,
    Audio,
    Controls,
    Coop,
    WorldCheats,
    WanInvite,
    ResourcePacks,
    Advancements,
    Statistics,
}

pub enum PauseAction {
    None,
    Resume,
    SaveAndQuit,
    SetGameMode(GameMode),
    SetTime(f32),
    ReloadResourcePack(String),
    ApplySettings,
    TeleportP2ToP1,
    OpenToLan,
    StopLan,
    StartTunnel,
    StopTunnel,
}

pub struct PauseMenuState {
    pub page: PausePage,
    pub listening_keybind: Option<String>,
    pub weather: String,
    pub pending_weather: Option<String>,
    pub copied_notice: Option<String>,
    /// Live local LAN host address (`<ip>:8090`) when LAN server is running.
    pub lan_active_addr: Option<String>,
    /// Live list of detected controller descriptions (populated each frame by main loop).
    pub connected_controllers: Vec<String>,
}

impl Default for PauseMenuState {
    fn default() -> Self {
        Self {
            page: PausePage::Main,
            listening_keybind: None,
            weather: "Clear".into(),
            pending_weather: None,
            copied_notice: None,
            lan_active_addr: None,
            connected_controllers: Vec::new(),
        }
    }
}

impl PauseMenuState {
    pub fn take_weather_change(&mut self) -> Option<String> {
        self.pending_weather.take()
    }
}

#[allow(clippy::too_many_arguments)]
pub fn render_pause_menu(
    ctx: &egui::Context,
    theme: &McTheme,
    state: &mut PauseMenuState,
    settings: &mut EngineSettings,
    config: &mut MinetestConfig,
    gamemode: GameMode,
    tunnel: &PlayitTunnelManager,
    packs: &[ResourcePackInfo],
    chunks_loaded: usize,
) -> PauseAction {
    let mut action = PauseAction::None;
    let screen = ctx.screen_rect();

    // Handle live keybind capture if user clicked a keybind button
    if let Some(ref key_name) = state.listening_keybind.clone() {
        ctx.input(|i| {
            for ev in &i.events {
                if let egui::Event::Key { key, pressed: true, .. } = ev {
                    if *key == egui::Key::Escape {
                        state.listening_keybind = None;
                        break;
                    }
                    let label = format!("{key:?}");
                    settings.keybinds.insert(key_name.clone(), label.clone());
                    if key_name == "keymap_zoom" {
                        settings.zoom_key = label;
                    }
                    state.listening_keybind = None;
                    action = PauseAction::ApplySettings;
                    break;
                }
            }
        });
    }

    egui::Area::new(egui::Id::new("cubex_pause_overlay"))
        .fixed_pos(screen.min)
        .show(ctx, |ui| {
            let p_owned = ui.painter().clone();
            let p = &p_owned;
            paint_pause_overlay(p, screen);

            let cx = screen.center().x;
            let top = (screen.height() * 0.08).clamp(24.0, 68.0);

            match state.page {
                PausePage::Main => {
                    mc_shadow_text(p, Pos2::new(cx, top), Align2::CENTER_CENTER, "Game Menu", 22.0, Color32::WHITE);
                    let w = (screen.width() * 0.42).clamp(380.0, 460.0);
                    let gap = 12.0;
                    let half = (w - gap) * 0.5;
                    let mut y = top + 46.0;

                    if place_btn(ui, theme, cx - w * 0.5, y, w, 42.0, "Back to Game", true) {
                        action = PauseAction::Resume;
                    }
                    y += 54.0;

                    if place_btn(ui, theme, cx - w * 0.5, y, half, 42.0, "Advancements", false) {
                        state.page = PausePage::Advancements;
                    }
                    if place_btn(ui, theme, cx + gap * 0.5, y, half, 42.0, "Statistics", false) {
                        state.page = PausePage::Statistics;
                    }
                    y += 52.0;

                    let lan_lbl = match state.lan_active_addr {
                        Some(ref addr) => format!("LAN Open: {addr} (Click to Stop)"),
                        None => "Open to LAN (Port 8090)".into(),
                    };
                    if place_btn(ui, theme, cx - w * 0.5, y, w, 42.0, &lan_lbl, state.lan_active_addr.is_some()) {
                        action = if state.lan_active_addr.is_some() {
                            PauseAction::StopLan
                        } else {
                            PauseAction::OpenToLan
                        };
                    }
                    y += 52.0;

                    if place_btn(ui, theme, cx - w * 0.5, y, half, 42.0, "Open to WAN (playit.gg)", false) {
                        state.copied_notice = None;
                        state.page = PausePage::WanInvite;
                    }
                    if place_btn(ui, theme, cx + gap * 0.5, y, half, 42.0, "World & Cheats", false) {
                        state.page = PausePage::WorldCheats;
                    }
                    y += 52.0;

                    if place_btn(ui, theme, cx - w * 0.5, y, half, 42.0, "Options...", false) {
                        state.page = PausePage::Options;
                    }
                    if place_btn(ui, theme, cx + gap * 0.5, y, half, 42.0, "Resource Packs...", false) {
                        state.page = PausePage::ResourcePacks;
                    }
                    y += 52.0;

                    let coop_lbl = if settings.coop_mode.is_active() {
                        format!(
                            "Local Co-op: {} ({}P)",
                            settings.coop_mode.label(),
                            settings.coop_mode.active_player_count(settings.coop_player_count)
                        )
                    } else {
                        "Local Co-op & Controllers...".to_string()
                    };
                    if place_btn(ui, theme, cx - w * 0.5, y, w, 42.0, &coop_lbl, settings.coop_mode.is_active()) {
                        state.page = PausePage::Coop;
                    }
                    y += 54.0;

                    if place_btn(ui, theme, cx - w * 0.5, y, w, 44.0, "Save and Quit to Title", false) {
                        state.page = PausePage::Main;
                        action = PauseAction::SaveAndQuit;
                    }
                }

                PausePage::Options => {
                    mc_shadow_text(p, Pos2::new(cx, top), Align2::CENTER_CENTER, "Options", 22.0, Color32::WHITE);
                    let w = (screen.width() * 0.48).clamp(420.0, 520.0);
                    let gap = 12.0;
                    let half = (w - gap) * 0.5;
                    let mut y = top + 48.0;

                    let sr = Rect::from_min_size(Pos2::new(cx - w * 0.5, y), Vec2::new(w, 42.0));
                    let changed = ui
                        .allocate_new_ui(egui::UiBuilder::new().max_rect(sr), |ui| {
                            mc_slider_f32(ui, theme, "FOV", &mut settings.fov, 30.0..=110.0, w, 42.0, |v| {
                                let r = v.round() as i32;
                                if r >= 110 {
                                    "Quake Pro".into()
                                } else if r == 70 {
                                    "Normal".into()
                                } else {
                                    r.to_string()
                                }
                            })
                        })
                        .inner;
                    if changed {
                        action = PauseAction::ApplySettings;
                    }
                    y += 58.0;

                    if place_btn(ui, theme, cx - w * 0.5, y, half, 42.0, "Video Settings...", false) {
                        state.page = PausePage::Video;
                    }
                    if place_btn(ui, theme, cx + gap * 0.5, y, half, 42.0, "Music & Sounds...", false) {
                        state.page = PausePage::Audio;
                    }
                    y += 52.0;

                    if place_btn(ui, theme, cx - w * 0.5, y, half, 42.0, "Controls & Keybinds...", false) {
                        state.page = PausePage::Controls;
                    }
                    let f3_lbl = format!("F3 Debug HUD: {}", if settings.f3_enabled { "ON" } else { "OFF" });
                    if place_btn(ui, theme, cx + gap * 0.5, y, half, 42.0, &f3_lbl, settings.f3_enabled) {
                        settings.f3_enabled = !settings.f3_enabled;
                        action = PauseAction::ApplySettings;
                    }
                    y += 52.0;

                    let liq_lbl = format!("Liquids Pointable: {}", if settings.liquids_pointable { "ON" } else { "OFF" });
                    if place_btn(ui, theme, cx - w * 0.5, y, half, 42.0, &liq_lbl, settings.liquids_pointable) {
                        settings.liquids_pointable = !settings.liquids_pointable;
                        action = PauseAction::ApplySettings;
                    }
                    if place_btn(ui, theme, cx + gap * 0.5, y, half, 42.0, "Resource Packs...", false) {
                        state.page = PausePage::ResourcePacks;
                    }
                    y += 72.0;

                    if place_btn(ui, theme, cx - 120.0, y, 240.0, 42.0, "Done", true) {
                        state.page = PausePage::Main;
                    }
                }

                PausePage::Video => {
                    mc_shadow_text(p, Pos2::new(cx, top), Align2::CENTER_CENTER, "Video Settings", 22.0, Color32::WHITE);
                    let w = (screen.width() * 0.62).clamp(520.0, 640.0);
                    let col_w = (w - 16.0) * 0.5;
                    let panel_r = Rect::from_min_max(
                        Pos2::new(cx - w * 0.5 - 10.0, top + 34.0),
                        Pos2::new(cx + w * 0.5 + 10.0, screen.bottom() - 80.0),
                    );
                    ui.allocate_new_ui(egui::UiBuilder::new().max_rect(panel_r), |ui| {
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            ui.add_space(6.0);
                            if paint_enhanced_graphics_controls(ui, theme, settings, w - 4.0) {
                                action = PauseAction::ApplySettings;
                            }
                            ui.add_space(10.0);
                            egui::Grid::new("pause_video_grid").spacing(Vec2::new(16.0, 10.0)).show(ui, |ui| {
                                let mut rd = settings.render_distance_chunks as f32;
                                if mc_slider_f32(ui, theme, "Render Distance", &mut rd, 2.0..=32.0, col_w, 40.0, |v| {
                                    format!("{} Chunks", v.round() as u32)
                                }) {
                                    settings.render_distance_chunks = rd.round() as u32;
                                    action = PauseAction::ApplySettings;
                                }

                                let mut fps_val = if settings.max_fps == 0 { 260.0 } else { settings.max_fps as f32 };
                                if mc_slider_f32(ui, theme, "Max Framerate", &mut fps_val, 30.0..=260.0, col_w, 40.0, |v| {
                                    if v >= 255.0 { "Unlimited".into() } else { format!("{} FPS", (v / 5.0).round() as u32 * 5) }
                                }) {
                                    settings.max_fps = if fps_val >= 255.0 { 0 } else { (fps_val / 5.0).round() as u32 * 5 };
                                    action = PauseAction::ApplySettings;
                                }
                                ui.end_row();

                                let vs_lbl = format!("VSync: {}", if settings.vsync { "ON" } else { "OFF" });
                                if mc_button_styled(ui, theme, &vs_lbl, col_w, 40.0, settings.vsync).clicked() {
                                    settings.vsync = !settings.vsync;
                                    action = PauseAction::ApplySettings;
                                }
                                let mm_lbl = format!("Mipmapping: {}", if settings.mipmapping { "ON" } else { "OFF" });
                                if mc_button_styled(ui, theme, &mm_lbl, col_w, 40.0, settings.mipmapping).clicked() {
                                    settings.mipmapping = !settings.mipmapping;
                                    action = PauseAction::ApplySettings;
                                }
                                ui.end_row();

                                let sl_lbl = format!("Smooth Lighting / AO: {}", if settings.smooth_lighting { "ON" } else { "OFF" });
                                if mc_button_styled(ui, theme, &sl_lbl, col_w, 40.0, settings.smooth_lighting).clicked() {
                                    settings.smooth_lighting = !settings.smooth_lighting;
                                    action = PauseAction::ApplySettings;
                                }
                                let sh_lbl = format!("Dynamic Shadows: {}", if settings.dynamic_shadows { "ON" } else { "OFF" });
                                if mc_button_styled(ui, theme, &sh_lbl, col_w, 40.0, settings.dynamic_shadows).clicked() {
                                    settings.dynamic_shadows = !settings.dynamic_shadows;
                                    action = PauseAction::ApplySettings;
                                }
                                ui.end_row();

                                if mc_slider_f32(ui, theme, "Brightness", &mut settings.brightness, 0.5..=1.5, col_w, 40.0, |v| {
                                    format!("{}%", ((v - 0.5) * 100.0).round() as i32)
                                }) {
                                    action = PauseAction::ApplySettings;
                                }
                                if mc_slider_f32(ui, theme, "GUI Scale", &mut settings.gui_scale, 0.75..=2.0, col_w, 40.0, |v| {
                                    format!("{v:.2}x")
                                }) {
                                    action = PauseAction::ApplySettings;
                                }
                                ui.end_row();
                            });
                            ui.add_space(6.0);
                            let wv_lbl = format!("Waving Water: {}", if settings.waving_water { "ON" } else { "OFF" });
                            if mc_button_styled(ui, theme, &wv_lbl, w - 4.0, 40.0, settings.waving_water).clicked() {
                                settings.waving_water = !settings.waving_water;
                                action = PauseAction::ApplySettings;
                            }
                        });
                    });

                    if place_btn(ui, theme, cx - 120.0, screen.bottom() - 62.0, 240.0, 42.0, "Done", true) {
                        state.page = PausePage::Options;
                    }
                }

                PausePage::Audio => {
                    mc_shadow_text(p, Pos2::new(cx, top), Align2::CENTER_CENTER, "Music & Sounds", 22.0, Color32::WHITE);
                    let w = (screen.width() * 0.58).clamp(500.0, 600.0);
                    let col_w = (w - 16.0) * 0.5;
                    let x0 = cx - w * 0.5;
                    let x1 = cx + 8.0;
                    let y = top + 48.0;

                    let mut sliders: [(&str, &mut f32, f32, f32, f32); 7] = [
                        ("Master Volume", &mut settings.audio.master, x0, y, w),
                        ("Music", &mut settings.audio.music, x0, y + 52.0, col_w),
                        ("Blocks", &mut settings.audio.blocks, x1, y + 52.0, col_w),
                        ("Weather", &mut settings.audio.weather, x0, y + 104.0, col_w),
                        ("Hostile Mobs", &mut settings.audio.hostile, x1, y + 104.0, col_w),
                        ("Friendly Mobs", &mut settings.audio.friendly, x0, y + 156.0, col_w),
                        ("Ambient", &mut settings.audio.ambient, x1, y + 156.0, col_w),
                    ];
                    for (lbl, val, sx, sy, sw) in sliders.iter_mut() {
                        let sr = Rect::from_min_size(Pos2::new(*sx, *sy), Vec2::new(*sw, 42.0));
                        if ui
                            .allocate_new_ui(egui::UiBuilder::new().max_rect(sr), |ui| {
                                mc_slider_f32(ui, theme, lbl, val, 0.0..=1.0, *sw, 42.0, |v| {
                                    format!("{}%", (v * 100.0).round() as i32)
                                })
                            })
                            .inner
                        {
                            action = PauseAction::ApplySettings;
                        }
                    }
                    if place_btn(ui, theme, cx - 120.0, y + 224.0, 240.0, 42.0, "Done", true) {
                        state.page = PausePage::Options;
                    }
                }

                PausePage::Controls => {
                    mc_shadow_text(p, Pos2::new(cx, top), Align2::CENTER_CENTER, "Controls & Keybinds", 22.0, Color32::WHITE);
                    let w = (screen.width() * 0.62).clamp(520.0, 640.0);
                    let col_w = (w - 16.0) * 0.5;
                    let panel_r = Rect::from_min_max(
                        Pos2::new(cx - w * 0.5 - 10.0, top + 34.0),
                        Pos2::new(cx + w * 0.5 + 10.0, screen.bottom() - 80.0),
                    );
                    ui.allocate_new_ui(egui::UiBuilder::new().max_rect(panel_r), |ui| {
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            ui.add_space(6.0);
                            egui::Grid::new("pause_ctrl_grid").spacing(Vec2::new(16.0, 8.0)).show(ui, |ui| {
                                if mc_slider_f32(ui, theme, "Sensitivity", &mut settings.mouse_sensitivity, 0.05..=1.5, col_w, 40.0, |v| {
                                    format!("{}%", (v * 100.0).round() as i32)
                                }) {
                                    action = PauseAction::ApplySettings;
                                }
                                let inv_lbl = format!("Invert Mouse: {}", if settings.invert_mouse { "ON" } else { "OFF" });
                                if mc_button_styled(ui, theme, &inv_lbl, col_w, 40.0, settings.invert_mouse).clicked() {
                                    settings.invert_mouse = !settings.invert_mouse;
                                    action = PauseAction::ApplySettings;
                                }
                                ui.end_row();

                                let aj_lbl = format!("Auto-Jump: {}", if settings.auto_jump { "ON" } else { "OFF" });
                                if mc_button_styled(ui, theme, &aj_lbl, col_w, 40.0, settings.auto_jump).clicked() {
                                    settings.auto_jump = !settings.auto_jump;
                                    action = PauseAction::ApplySettings;
                                }
                                let sp_lbl = format!("Toggle Sprint: {}", if settings.toggle_sprint { "ON" } else { "OFF" });
                                if mc_button_styled(ui, theme, &sp_lbl, col_w, 40.0, settings.toggle_sprint).clicked() {
                                    settings.toggle_sprint = !settings.toggle_sprint;
                                    action = PauseAction::ApplySettings;
                                }
                                ui.end_row();

                                let touch_lbl = format!("Touch Controls: {}", on_off(settings.touch_controls));
                                if mc_button_styled(ui, theme, &touch_lbl, col_w, 40.0, settings.touch_controls).clicked() {
                                    settings.touch_controls = !settings.touch_controls;
                                    action = PauseAction::ApplySettings;
                                }
                                if mc_slider_f32(ui, theme, "Touch Sensitivity", &mut settings.touch_sensitivity, 0.3..=3.0, col_w, 40.0, |v| {
                                    format!("{}%", (v * 100.0).round() as i32)
                                }) {
                                    action = PauseAction::ApplySettings;
                                }
                                ui.end_row();
                            });

                            ui.add_space(10.0);
                            mc_shadow_text(ui.painter(), ui.next_widget_position() + Vec2::new(w * 0.5, 10.0), Align2::CENTER_CENTER, "Keybindings (Click then press any key)", 13.0, Color32::from_rgb(255, 255, 160));
                            ui.add_space(24.0);

                            let keys = [
                                ("Walk Forward", "keymap_forward"),
                                ("Walk Backward", "keymap_backward"),
                                ("Strafe Left", "keymap_left"),
                                ("Strafe Right", "keymap_right"),
                                ("Jump", "keymap_jump"),
                                ("Sneak", "keymap_sneak"),
                                ("Sprint (Aux1)", "keymap_aux1"),
                                ("Open Inventory", "keymap_inventory"),
                                ("Optical Zoom", "keymap_zoom"),
                                ("Drop Item", "keymap_drop"),
                                ("Open Chat", "keymap_chat"),
                                ("Fly (Creative)", "keymap_freemove"),
                            ];
                            egui::Grid::new("pause_keybinds_grid").spacing(Vec2::new(16.0, 8.0)).show(ui, |ui| {
                                for (idx, (lbl, kname)) in keys.iter().enumerate() {
                                    let cur = settings.keybinds.get(*kname).cloned().unwrap_or_else(|| "?".into());
                                    let is_listening = state.listening_keybind.as_deref() == Some(*kname);
                                    let btxt = if is_listening {
                                        format!("{lbl}: > Press Key <")
                                    } else {
                                        format!("{lbl}: {cur}")
                                    };
                                    if mc_button_styled(ui, theme, &btxt, col_w, 38.0, is_listening).clicked() {
                                        state.listening_keybind = Some((*kname).to_string());
                                    }
                                    if idx % 2 == 1 {
                                        ui.end_row();
                                    }
                                }
                            });
                        });
                    });

                    if place_btn(ui, theme, cx - 120.0, screen.bottom() - 62.0, 240.0, 42.0, "Done", true) {
                        state.listening_keybind = None;
                        state.page = PausePage::Options;
                    }
                }

                PausePage::Coop => {
                    mc_shadow_text(p, Pos2::new(cx, top), Align2::CENTER_CENTER, "Local Co-op & PS5 Controllers", 22.0, Color32::WHITE);
                    let w = (screen.width() * 0.64).clamp(540.0, 680.0);
                    let col_w = (w - 16.0) * 0.5;
                    let panel_r = Rect::from_min_max(
                        Pos2::new(cx - w * 0.5 - 10.0, top + 34.0),
                        Pos2::new(cx + w * 0.5 + 10.0, screen.bottom() - 80.0),
                    );
                    ui.allocate_new_ui(egui::UiBuilder::new().max_rect(panel_r), |ui| {
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            ui.add_space(6.0);
                            if paint_coop_and_controller_controls(
                                ui,
                                theme,
                                settings,
                                &state.connected_controllers,
                                col_w,
                            ) {
                                action = PauseAction::ApplySettings;
                            }
                            ui.add_space(10.0);
                            if mc_button_with_sound(
                                ui,
                                theme,
                                "Teleport Co-op Players to P1",
                                w - 4.0,
                                38.0,
                                false,
                                settings.coop_mode.is_active(),
                                None,
                            )
                            .clicked()
                            {
                                action = PauseAction::TeleportP2ToP1;
                            }
                        });
                    });

                    if place_btn(ui, theme, cx - 120.0, screen.bottom() - 62.0, 240.0, 42.0, "Done", true) {
                        state.page = PausePage::Main;
                    }
                }

                PausePage::WorldCheats => {
                    mc_shadow_text(p, Pos2::new(cx, top), Align2::CENTER_CENTER, "World & Cheats", 22.0, Color32::WHITE);
                    let w = (screen.width() * 0.56).clamp(480.0, 580.0);
                    let panel_r = Rect::from_min_max(
                        Pos2::new(cx - w * 0.5 - 10.0, top + 34.0),
                        Pos2::new(cx + w * 0.5 + 10.0, screen.bottom() - 80.0),
                    );
                    ui.allocate_new_ui(egui::UiBuilder::new().max_rect(panel_r), |ui| {
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            ui.add_space(6.0);
                            ui.label(egui::RichText::new(format!("Game Mode: {}", gamemode.label())).color(Color32::from_rgb(255, 255, 160)));
                            ui.horizontal(|ui| {
                                let bw = (w - 16.0) / 3.0;
                                if mc_button_styled(ui, theme, "Survival", bw, 40.0, gamemode == GameMode::Survival).clicked() {
                                    action = PauseAction::SetGameMode(GameMode::Survival);
                                }
                                if mc_button_styled(ui, theme, "Creative", bw, 40.0, gamemode == GameMode::Creative).clicked() {
                                    action = PauseAction::SetGameMode(GameMode::Creative);
                                }
                                if mc_button_styled(ui, theme, "Spectator", bw, 40.0, gamemode == GameMode::Spectator).clicked() {
                                    action = PauseAction::SetGameMode(GameMode::Spectator);
                                }
                            });

                            ui.add_space(10.0);
                            ui.label(egui::RichText::new("Time of Day").color(Color32::LIGHT_GRAY));
                            ui.horizontal(|ui| {
                                let bw = (w - 24.0) / 4.0;
                                if mc_button(ui, theme, "Sunrise", bw, 38.0).clicked() {
                                    action = PauseAction::SetTime(0.25);
                                }
                                if mc_button(ui, theme, "Day (Noon)", bw, 38.0).clicked() {
                                    action = PauseAction::SetTime(0.50);
                                }
                                if mc_button(ui, theme, "Sunset", bw, 38.0).clicked() {
                                    action = PauseAction::SetTime(0.75);
                                }
                                if mc_button(ui, theme, "Midnight", bw, 38.0).clicked() {
                                    action = PauseAction::SetTime(0.0);
                                }
                            });

                            ui.add_space(10.0);
                            ui.label(egui::RichText::new(format!("Weather: {}", state.weather)).color(Color32::LIGHT_GRAY));
                            ui.horizontal(|ui| {
                                let bw = (w - 16.0) / 3.0;
                                for wname in ["Clear", "Rain", "Thunder"] {
                                    let active = state.weather == wname;
                                    if mc_button_styled(ui, theme, wname, bw, 38.0, active).clicked() {
                                        state.weather = wname.to_string();
                                        state.pending_weather = Some(wname.to_string());
                                        config.set("cubex_weather", wname);
                                        action = PauseAction::ApplySettings;
                                    }
                                }
                            });

                            ui.add_space(10.0);
                            let diff_lbl = format!("Difficulty: {}", settings.difficulty.label());
                            if mc_button(ui, theme, &diff_lbl, w, 40.0).clicked() {
                                settings.difficulty = settings.difficulty.cycle();
                                config.set("cubex_difficulty", settings.difficulty.as_str());
                                action = PauseAction::ApplySettings;
                            }

                            ui.add_space(10.0);
                            ui.label(egui::RichText::new("Gamerules").color(Color32::from_rgb(160, 240, 160)));
                            let half = (w - 8.0) * 0.5;
                            egui::Grid::new("pause_gamerules_grid").spacing(Vec2::new(8.0, 8.0)).show(ui, |ui| {
                                let ki = settings.gamerules.keep_inventory;
                                if mc_button_styled(ui, theme, &format!("Keep Inventory on Death: {}", on_off(ki)), half, 38.0, ki).clicked() {
                                    settings.gamerules.keep_inventory = !ki;
                                    config.set("mcl_keepInventory", !ki);
                                    action = PauseAction::ApplySettings;
                                }
                                let dlc = settings.gamerules.do_daylight_cycle;
                                if mc_button_styled(ui, theme, &format!("doDaylightCycle: {}", on_off(dlc)), half, 38.0, dlc).clicked() {
                                    settings.gamerules.do_daylight_cycle = !dlc;
                                    config.set("cubex_do_daylight_cycle", !dlc);
                                    action = PauseAction::ApplySettings;
                                }
                                ui.end_row();

                                let pvp = settings.gamerules.pvp;
                                if mc_button_styled(ui, theme, &format!("pvp: {}", on_off(pvp)), half, 38.0, pvp).clicked() {
                                    settings.gamerules.pvp = !pvp;
                                    config.set("enable_pvp", !pvp);
                                    action = PauseAction::ApplySettings;
                                }
                                let fd = settings.gamerules.fall_damage;
                                if mc_button_styled(ui, theme, &format!("fallDamage: {}", on_off(fd)), half, 38.0, fd).clicked() {
                                    settings.gamerules.fall_damage = !fd;
                                    config.set("cubex_fall_damage", !fd);
                                    action = PauseAction::ApplySettings;
                                }
                                ui.end_row();
                            });
                        });
                    });

                    if place_btn(ui, theme, cx - 120.0, screen.bottom() - 62.0, 240.0, 42.0, "Done", true) {
                        state.page = PausePage::Main;
                    }
                }

                PausePage::WanInvite => {
                    mc_shadow_text(p, Pos2::new(cx, top), Align2::CENTER_CENTER, "Open to WAN (playit.gg)", 22.0, Color32::from_rgb(85, 255, 85));
                    let st = tunnel.status.lock().clone();
                    let w = (screen.width() * 0.62).clamp(500.0, 640.0);
                    let card = Rect::from_min_size(Pos2::new(cx - w * 0.5, top + 36.0), Vec2::new(w, 150.0));
                    p.rect_filled(card, 0.0, Color32::from_rgba_unmultiplied(0, 0, 0, 200));
                    p.rect_stroke(card, 0.0, Stroke::new(1.5, Color32::from_rgb(90, 90, 90)), egui::StrokeKind::Inside);

                    mc_shadow_text(p, Pos2::new(cx, card.top() + 24.0), Align2::CENTER_CENTER, &format!("Status: {}", st.status_text), 14.0, Color32::WHITE);
                    let addr_txt = if st.public_address.is_empty() {
                        "Public Address: (Not started)".into()
                    } else {
                        format!("Public Address: {}", st.public_address)
                    };
                    mc_shadow_text(p, Pos2::new(cx, card.top() + 56.0), Align2::CENTER_CENTER, &addr_txt, 14.0, Color32::from_rgb(255, 255, 120));
                    let code_txt = if st.invite_code.is_empty() {
                        "Invite Code: CBX-...".into()
                    } else {
                        format!("Invite Code: {}", st.invite_code)
                    };
                    mc_shadow_text(p, Pos2::new(cx, card.top() + 88.0), Align2::CENTER_CENTER, &code_txt, 14.0, Color32::from_rgb(85, 255, 255));
                    if !st.claim_url.is_empty() {
                        mc_shadow_text(p, Pos2::new(cx, card.top() + 120.0), Align2::CENTER_CENTER, &format!("Claim URL: {}", st.claim_url), 12.0, Color32::from_rgb(255, 180, 90));
                    } else if let Some(ref notice) = state.copied_notice {
                        mc_shadow_text(p, Pos2::new(cx, card.top() + 120.0), Align2::CENTER_CENTER, notice, 12.0, Color32::from_rgb(85, 255, 85));
                    }

                    let y = card.bottom() + 14.0;
                    let bw = (w - 16.0) / 3.0;
                    if st.running {
                        if place_btn(ui, theme, cx - w * 0.5, y, bw, 40.0, "Stop Tunnel", false) {
                            action = PauseAction::StopTunnel;
                        }
                    } else if place_btn(ui, theme, cx - w * 0.5, y, bw, 40.0, "Start Playit Tunnel", true) {
                        action = PauseAction::StartTunnel;
                    }
                    if place_btn_enabled(ui, theme, cx - bw * 0.5, y, bw, 40.0, "Copy Invite Code", false, !st.invite_code.is_empty()) {
                        ctx.copy_text(st.invite_code.clone());
                        state.copied_notice = Some("Copied CBX-... invite code to clipboard!".into());
                    }
                    if place_btn_enabled(ui, theme, cx + w * 0.5 - bw, y, bw, 40.0, "Copy Address", false, !st.public_address.is_empty()) {
                        ctx.copy_text(st.public_address.clone());
                        state.copied_notice = Some("Copied public address to clipboard!".into());
                    }

                    // Live tunnel logs
                    let log_r = Rect::from_min_max(
                        Pos2::new(cx - w * 0.5, y + 54.0),
                        Pos2::new(cx + w * 0.5, (screen.bottom() - 78.0).max(y + 110.0)),
                    );
                    p.rect_filled(log_r, 0.0, Color32::from_rgba_unmultiplied(8, 8, 8, 220));
                    p.rect_stroke(log_r, 0.0, Stroke::new(1.0, Color32::from_rgb(70, 70, 70)), egui::StrokeKind::Inside);
                    ui.allocate_new_ui(egui::UiBuilder::new().max_rect(log_r.shrink(6.0)), |ui| {
                        egui::ScrollArea::vertical().stick_to_bottom(true).show(ui, |ui| {
                            if st.logs.is_empty() {
                                ui.label(egui::RichText::new("Playit tunnel logs will appear here...").color(Color32::GRAY));
                            } else {
                                for line in &st.logs {
                                    ui.label(egui::RichText::new(line).monospace().size(11.5).color(Color32::from_rgb(85, 255, 85)));
                                }
                            }
                        });
                    });

                    if place_btn(ui, theme, cx - 120.0, screen.bottom() - 62.0, 240.0, 42.0, "Done", false) {
                        state.page = PausePage::Main;
                    }
                }

                PausePage::ResourcePacks => {
                    mc_shadow_text(p, Pos2::new(cx, top), Align2::CENTER_CENTER, "Resource Packs", 22.0, Color32::WHITE);
                    let w = (screen.width() * 0.56).clamp(440.0, 560.0);
                    let list_r = Rect::from_min_max(
                        Pos2::new(cx - w * 0.5, top + 38.0),
                        Pos2::new(cx + w * 0.5, screen.bottom() - 80.0),
                    );
                    ui.allocate_new_ui(egui::UiBuilder::new().max_rect(list_r), |ui| {
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            for pack in packs {
                                let active = settings.active_resource_pack == pack.path;
                                let lbl = if active {
                                    format!("{} [ACTIVE]", pack.name)
                                } else {
                                    pack.name.clone()
                                };
                                if mc_button_styled(ui, theme, &lbl, w - 16.0, 44.0, active).clicked() {
                                    settings.active_resource_pack = pack.path.clone();
                                    action = PauseAction::ReloadResourcePack(pack.path.clone());
                                }
                                ui.add_space(6.0);
                            }
                        });
                    });
                    if place_btn(ui, theme, cx - 120.0, screen.bottom() - 62.0, 240.0, 42.0, "Done", true) {
                        state.page = PausePage::Main;
                    }
                }

                PausePage::Advancements => {
                    mc_shadow_text(p, Pos2::new(cx, top), Align2::CENTER_CENTER, "Advancements", 22.0, Color32::from_rgb(255, 255, 85));
                    let w = (screen.width() * 0.56).clamp(440.0, 560.0);
                    let box_r = Rect::from_min_max(
                        Pos2::new(cx - w * 0.5, top + 40.0),
                        Pos2::new(cx + w * 0.5, screen.bottom() - 80.0),
                    );
                    p.rect_filled(box_r, 0.0, Color32::from_rgba_unmultiplied(12, 12, 16, 210));
                    p.rect_stroke(box_r, 0.0, Stroke::new(2.0, Color32::from_rgb(80, 80, 80)), egui::StrokeKind::Inside);
                    ui.allocate_new_ui(egui::UiBuilder::new().max_rect(box_r.shrink(14.0)), |ui| {
                        for (title, desc) in [
                            ("Mineclonia Native", "World loaded with 224+ mods in native Rust + LuaJIT"),
                            ("Getting Wood", "Punch a tree trunk to gather oak or birch wood"),
                            ("Benchmaking", "Craft a 3x3 Crafting Table from four wooden planks"),
                            ("Time to Mine!", "Use planks and sticks to craft a wooden pickaxe"),
                            ("Hot Topic", "Construct a Furnace out of eight cobblestone blocks"),
                        ] {
                            ui.label(egui::RichText::new(format!("[Unlocked] {title}")).strong().color(Color32::from_rgb(85, 255, 85)));
                            ui.label(egui::RichText::new(desc).color(Color32::LIGHT_GRAY));
                            ui.add_space(8.0);
                        }
                    });
                    if place_btn(ui, theme, cx - 120.0, screen.bottom() - 62.0, 240.0, 42.0, "Done", true) {
                        state.page = PausePage::Main;
                    }
                }

                PausePage::Statistics => {
                    mc_shadow_text(p, Pos2::new(cx, top), Align2::CENTER_CENTER, "Statistics", 22.0, Color32::from_rgb(85, 255, 255));
                    let w = (screen.width() * 0.56).clamp(440.0, 560.0);
                    let box_r = Rect::from_min_max(
                        Pos2::new(cx - w * 0.5, top + 40.0),
                        Pos2::new(cx + w * 0.5, screen.bottom() - 80.0),
                    );
                    p.rect_filled(box_r, 0.0, Color32::from_rgba_unmultiplied(12, 12, 16, 210));
                    p.rect_stroke(box_r, 0.0, Stroke::new(2.0, Color32::from_rgb(80, 80, 80)), egui::StrokeKind::Inside);
                    ui.allocate_new_ui(egui::UiBuilder::new().max_rect(box_r.shrink(14.0)), |ui| {
                        for row in [
                            format!("Player Name: {}", settings.player_name),
                            format!("Game Mode: {}", gamemode.label()),
                            format!("Difficulty: {}", settings.difficulty.label()),
                            format!("Chunks Loaded in Memory: {}", chunks_loaded),
                            format!("Render Distance: {} chunks", settings.render_distance_chunks),
                            format!("Graphics Preset: {} ({})", settings.graphics_preset.label(), settings.graphics.as_str()),
                            format!("Dynamic Torch Light: {}", on_off(settings.handheld_torch_light)),
                        ] {
                            ui.label(egui::RichText::new(row).size(15.0).color(Color32::WHITE));
                            ui.add_space(6.0);
                        }
                    });
                    if place_btn(ui, theme, cx - 120.0, screen.bottom() - 62.0, 240.0, 42.0, "Done", true) {
                        state.page = PausePage::Main;
                    }
                }
            }
        });

    action
}

fn on_off(v: bool) -> &'static str {
    if v { "ON" } else { "OFF" }
}

/// Shared Local Co-op & PS5 DualSense controller panel used by both Title Options and Pause Menu.
/// Returns `true` when any setting changed.
pub fn paint_coop_and_controller_controls(
    ui: &mut egui::Ui,
    theme: &McTheme,
    settings: &mut EngineSettings,
    connected_controllers: &[String],
    col_w: f32,
) -> bool {
    let mut changed = false;
    let full_w = col_w * 2.0 + 14.0;

    let max_players = settings.coop_mode.max_players();
    if mc_button_styled(
        ui,
        theme,
        &format!("Co-op Mode: {}", settings.coop_mode.label()),
        full_w,
        42.0,
        settings.coop_mode.is_active(),
    )
    .clicked()
    {
        settings.coop_mode = settings.coop_mode.cycle();
        let min_p = settings.coop_mode.min_players() as u8;
        let max_p = settings.coop_mode.max_players() as u8;
        if settings.coop_player_count < min_p || settings.coop_player_count > max_p {
            settings.coop_player_count = max_p;
        }
        changed = true;
    }
    if max_players > 2 {
        ui.add_space(6.0);
        let cur_p = settings.coop_mode.active_player_count(settings.coop_player_count);
        if mc_button_styled(
            ui,
            theme,
            &format!("Local Players: {cur_p} Players (Click to Cycle 2-4P)"),
            full_w,
            38.0,
            true,
        )
        .clicked()
        {
            let min_p = settings.coop_mode.min_players();
            let next_p = if cur_p >= max_players { min_p } else { cur_p + 1 };
            settings.coop_player_count = next_p as u8;
            changed = true;
        }
    }
    ui.add_space(8.0);

    let active_players = settings.coop_mode.active_player_count(settings.coop_player_count);
    egui::Grid::new("coop_ctrl_grid")
        .spacing(Vec2::new(14.0, 8.0))
        .show(ui, |ui| {
            if mc_button(
                ui,
                theme,
                &format!("P1 Input: {}", settings.p1_input.label()),
                col_w,
                40.0,
            )
            .clicked()
            {
                settings.p1_input = settings.p1_input.cycle();
                changed = true;
            }
            if mc_button(
                ui,
                theme,
                &format!("P2 Input: {}", settings.p2_input.label()),
                col_w,
                40.0,
            )
            .clicked()
            {
                settings.p2_input = settings.p2_input.cycle();
                changed = true;
            }
            ui.end_row();

            if active_players >= 3 {
                if mc_button(
                    ui,
                    theme,
                    &format!("P3 Input: {}", settings.p3_input.label()),
                    col_w,
                    40.0,
                )
                .clicked()
                {
                    settings.p3_input = settings.p3_input.cycle();
                    changed = true;
                }
                if active_players >= 4 {
                    if mc_button(
                        ui,
                        theme,
                        &format!("P4 Input: {}", settings.p4_input.label()),
                        col_w,
                        40.0,
                    )
                    .clicked()
                    {
                        settings.p4_input = settings.p4_input.cycle();
                        changed = true;
                    }
                } else {
                    ui.label(egui::RichText::new("(3P Mode Active)").color(Color32::GRAY));
                }
                ui.end_row();
            }

            if mc_slider_f32(
                ui,
                theme,
                "Controller Sensitivity",
                &mut settings.controller_sensitivity,
                0.25..=3.0,
                col_w,
                40.0,
                |v| format!("{}%", (v * 100.0).round() as i32),
            ) {
                changed = true;
            }
            if mc_slider_f32(
                ui,
                theme,
                "Stick Deadzone",
                &mut settings.controller_deadzone,
                0.05..=0.35,
                col_w,
                40.0,
                |v| format!("{}%", (v * 100.0).round() as i32),
            ) {
                changed = true;
            }
            ui.end_row();

            if mc_button_styled(
                ui,
                theme,
                &format!("Invert Controller Y: {}", on_off(settings.controller_invert_y)),
                col_w,
                40.0,
                settings.controller_invert_y,
            )
            .clicked()
            {
                settings.controller_invert_y = !settings.controller_invert_y;
                changed = true;
            }
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("P2 Name:").color(Color32::LIGHT_GRAY));
                if ui.text_edit_singleline(&mut settings.p2_name).changed() {
                    changed = true;
                }
            });
            ui.end_row();

            if active_players >= 3 {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("P3 Name:").color(Color32::LIGHT_GRAY));
                    if ui.text_edit_singleline(&mut settings.p3_name).changed() {
                        changed = true;
                    }
                });
                if active_players >= 4 {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("P4 Name:").color(Color32::LIGHT_GRAY));
                        if ui.text_edit_singleline(&mut settings.p4_name).changed() {
                            changed = true;
                        }
                    });
                } else {
                    ui.label("");
                }
                ui.end_row();
            }
        });

    ui.add_space(10.0);
    ui.label(
        egui::RichText::new(format!(
            "Detected Controllers ({})",
            connected_controllers.len()
        ))
        .strong()
        .color(Color32::from_rgb(120, 255, 140)),
    );
    if connected_controllers.is_empty() {
        ui.label(
            egui::RichText::new(
                "No USB/Bluetooth controller detected yet. Plug in a PS5 DualSense or press Options on any controller to join.",
            )
            .color(Color32::GRAY),
        );
    } else {
        for desc in connected_controllers {
            ui.label(egui::RichText::new(format!("• {desc}")).color(Color32::WHITE));
        }
    }

    ui.add_space(10.0);
    ui.label(
        egui::RichText::new("PS5 DualSense & Gamepad Mapping")
            .strong()
            .color(Color32::from_rgb(255, 255, 160)),
    );
    ui.label(
        egui::RichText::new(
            "L-Stick: Move  |  R-Stick: Look  |  L3: Sprint Toggle  |  R3: Zoom\nCross (X): Jump / Select  |  Circle (O): Sneak / Back  |  Square: Inventory / Craft\nTriangle: 3rd-Person (F5) / Shift-Move  |  R2: Break  |  L2: Place / Interact\nL1 / R1 or D-Pad L/R: Hotbar  |  D-Pad Down: Drop  |  Options: Join P2 / Pause  |  F8: Cycle Split/Dual-Monitor",
        )
        .size(12.0)
        .color(Color32::LIGHT_GRAY),
    );
    changed
}

/// Shared Enhanced Graphics & Juice preset selector + 6 individual toggle checkboxes.
pub fn paint_enhanced_graphics_controls(
    ui: &mut egui::Ui,
    theme: &McTheme,
    settings: &mut EngineSettings,
    total_w: f32,
) -> bool {
    let mut changed = false;
    ui.label(
        egui::RichText::new("Enhanced Graphics & Juice (Optional)")
            .strong()
            .size(14.0)
            .color(Color32::from_rgb(255, 224, 112)),
    );
    ui.add_space(4.0);

    let preset_w = ((total_w - 24.0) / 4.0).max(96.0);
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 8.0;
        for preset in [
            GraphicsPreset::Performance,
            GraphicsPreset::Balanced,
            GraphicsPreset::Cinematic,
            GraphicsPreset::Custom,
        ] {
            let active = settings.graphics_preset == preset;
            if mc_button_styled(ui, theme, preset.label(), preset_w, 34.0, active).clicked() {
                if preset == GraphicsPreset::Custom {
                    settings.graphics_preset = GraphicsPreset::Custom;
                } else {
                    settings.apply_graphics_preset(preset);
                }
                changed = true;
            }
        }
    });
    ui.add_space(4.0);
    ui.label(
        egui::RichText::new(settings.graphics_preset.description())
            .size(12.0)
            .color(Color32::from_rgb(190, 198, 212)),
    );
    ui.add_space(6.0);

    egui::Frame::NONE
        .fill(Color32::from_rgba_unmultiplied(12, 14, 18, 175))
        .stroke(Stroke::new(1.0, Color32::from_rgb(76, 82, 92)))
        .inner_margin(egui::Margin::symmetric(12, 8))
        .show(ui, |ui| {
            ui.set_min_width((total_w - 26.0).max(300.0));
            ui.spacing_mut().item_spacing.y = 5.0;
            let toggles: [(&str, &mut bool); 7] = [
                ("Procedural Clouds & Sky Atmosphere", &mut settings.procedural_sky_clouds),
                ("Dynamic Torch & Hand Lighting", &mut settings.handheld_torch_light),
                ("Warm Sun, Atmospheric Fog & Tone Mapping", &mut settings.warm_sun_and_atmosphere),
                ("Bloom Glow Post-Pass", &mut settings.bloom_glow),
                ("Block Shatter & Physical Drops", &mut settings.block_shatter_physics),
                ("Wind-Swaying Grass & Leaves", &mut settings.wind_swaying_foliage),
                ("3D Extruded Held Items & Arm Sway", &mut settings.extruded_held_items),
            ];
            let mut any_toggled = false;
            for (label, flag) in toggles {
                let txt = egui::RichText::new(label).size(13.5).color(Color32::WHITE);
                if ui.checkbox(flag, txt).changed() {
                    any_toggled = true;
                }
            }
            if any_toggled {
                settings.sync_graphics_preset_from_toggles();
                changed = true;
            }
        });
    changed
}

fn place_btn(ui: &mut egui::Ui, theme: &McTheme, x: f32, y: f32, w: f32, h: f32, text: &str, green: bool) -> bool {
    place_btn_enabled(ui, theme, x, y, w, h, text, green, true)
}

#[allow(clippy::too_many_arguments)]
fn place_btn_enabled(
    ui: &mut egui::Ui,
    theme: &McTheme,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    text: &str,
    green: bool,
    enabled: bool,
) -> bool {
    let rect = Rect::from_min_size(Pos2::new(x, y), Vec2::new(w, h));
    ui.allocate_new_ui(egui::UiBuilder::new().max_rect(rect), |ui| {
        mc_button_with_sound(ui, theme, text, w, h, green, enabled, None).clicked()
    })
    .inner
}
