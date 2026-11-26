use egui::{Align2, Color32, Pos2, Rect, Stroke, Vec2};
use std::collections::HashMap;
use std::time::Duration;

use crate::config::EngineSettings;
use crate::network::invite::decode_invite;
use crate::network::motd::motd_to_layout_job;
use crate::network::servers::{
    generate_server_icon_rgba16, load_servers, ping_server, save_custom_servers, scan_lan_servers,
    AsyncServerRefresh, ServerEntry,
};
use crate::network::tunnel::PlayitTunnelManager;
use crate::textures::{is_carterpcs_skin, ResourcePackInfo, SkinInfo, TextureAtlas};
use crate::ui::theme::{
    mc_button, mc_button_green, mc_button_styled, mc_button_with_sound, mc_shadow_text,
    mc_slider_f32, paint_dirt_screen_bg, paint_player_skin_card, ItemIconCache, McTheme,
};
use crate::world::{
    create_world_ex, delete_world, list_worlds, next_default_world_name, Difficulty, Gamerules,
    WorldGenType, WorldInfo,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TitlePage {
    Home,
    Singleplayer,
    CreateWorld,
    Multiplayer,
    HostWan,
    Options,
    Skins,
    ResourcePacks,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TitleOptionsTab {
    General,
    Video,
    Audio,
    Controls,
    Coop,
}

pub enum TitleAction {
    None,
    LaunchSingleplayer(WorldInfo),
    JoinServer(String, u16),
    ApplySettings,
    ReloadPack(String),
    StartTunnel,
    StopTunnel,
    QuitGame,
}

pub struct TitleScreenState {
    pub page: TitlePage,
    pub options_tab: TitleOptionsTab,
    pub splash: String,
    pub worlds: Vec<WorldInfo>,
    pub selected_world: usize,
    pub world_search: String,
    pub confirm_delete_world: Option<WorldInfo>,
    pub new_world_name: String,
    pub new_world_creative: bool,
    pub new_world_gamemode: String,
    pub new_world_seed: String,
    pub new_world_type: WorldGenType,
    pub new_world_difficulty: Difficulty,
    pub new_gamerules: Gamerules,
    pub servers: Vec<ServerEntry>,
    pub selected_server: usize,
    pub direct_addr: String,
    pub add_name: String,
    pub add_addr: String,
    pub show_add_modal: bool,
    pub status_msg: String,
    pub copied_notice: Option<String>,
    pub listening_keybind: Option<String>,
    pub icons: ItemIconCache,
    pub server_refresher: AsyncServerRefresh,
    pub connected_controllers: Vec<String>,
    pub available_skins: Vec<SkinInfo>,
    pub skin_edit_slot: usize,
    pub skin_preview_handles: HashMap<String, egui::TextureHandle>,
}

impl TitleScreenState {
    pub fn new(worlds_dir: &std::path::Path, servers_file: &std::path::Path) -> Self {
        Self {
            page: TitlePage::Home,
            options_tab: TitleOptionsTab::General,
            splash: "100% Native Rust + LuaJIT!".into(),
            worlds: list_worlds(worlds_dir),
            selected_world: 0,
            world_search: String::new(),
            confirm_delete_world: None,
            new_world_name: next_default_world_name(worlds_dir),
            new_world_creative: false,
            new_world_gamemode: "Survival".into(),
            new_world_seed: String::new(),
            new_world_type: WorldGenType::Default,
            new_world_difficulty: Difficulty::Normal,
            new_gamerules: Gamerules::default(),
            servers: load_servers(servers_file),
            selected_server: 0,
            direct_addr: String::new(),
            add_name: "My Server".into(),
            add_addr: "127.0.0.1:8090".into(),
            show_add_modal: false,
            status_msg: String::new(),
            copied_notice: None,
            listening_keybind: None,
            icons: ItemIconCache::new(),
            server_refresher: AsyncServerRefresh::default(),
            connected_controllers: Vec::new(),
            available_skins: Vec::new(),
            skin_edit_slot: 0,
            skin_preview_handles: HashMap::new(),
        }
    }

    pub fn refresh_worlds(&mut self, worlds_dir: &std::path::Path) {
        self.worlds = list_worlds(worlds_dir);
        if self.selected_world >= self.worlds.len() {
            self.selected_world = 0;
        }
    }

    pub fn refresh_servers_sync(&mut self, rt: &tokio::runtime::Runtime) {
        let mut entries = self.servers.clone();
        rt.block_on(async {
            for s in &mut entries {
                let (on, ms) = ping_server(&s.host, s.port, Duration::from_millis(350)).await;
                s.online = on;
                s.ping_ms = ms;
            }
            let lan = scan_lan_servers().await;
            entries.retain(|s| !s.is_lan);
            entries.extend(lan);
        });
        self.servers = entries;
    }

    pub fn trigger_async_refresh(&self, rt: &tokio::runtime::Runtime) {
        self.server_refresher.trigger(rt, self.servers.clone());
    }
}

#[allow(clippy::too_many_arguments)]
pub fn render_title_screen(
    ctx: &egui::Context,
    theme: &McTheme,
    state: &mut TitleScreenState,
    settings: &mut EngineSettings,
    worlds_dir: &std::path::Path,
    servers_file: &std::path::Path,
    packs: &[ResourcePackInfo],
    tunnel: &PlayitTunnelManager,
    tokio_rt: &tokio::runtime::Runtime,
    time_sec: f32,
) -> TitleAction {
    render_title_screen_with_atlas(
        ctx,
        theme,
        state,
        settings,
        worlds_dir,
        servers_file,
        packs,
        tunnel,
        tokio_rt,
        None,
        time_sec,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn render_title_screen_with_atlas(
    ctx: &egui::Context,
    theme: &McTheme,
    state: &mut TitleScreenState,
    settings: &mut EngineSettings,
    worlds_dir: &std::path::Path,
    servers_file: &std::path::Path,
    packs: &[ResourcePackInfo],
    tunnel: &PlayitTunnelManager,
    tokio_rt: &tokio::runtime::Runtime,
    atlas: Option<&TextureAtlas>,
    time_sec: f32,
) -> TitleAction {
    let mut action = TitleAction::None;
    let screen = ctx.screen_rect();

    // Poll non-blocking server ping & LAN discovery results
    if let Some(updated) = state.server_refresher.poll() {
        state.servers = updated;
        if state.selected_server >= state.servers.len() {
            state.selected_server = 0;
        }
    }

    if let Some(at) = atlas {
        state.icons.ensure_player_skin(ctx, at);
    }

    // Handle interactive keybind capture in Title -> Options -> Controls
    if state.page == TitlePage::Options {
        if let Some(ref key_name) = state.listening_keybind.clone() {
            ctx.input(|i| {
                for ev in &i.events {
                    if let egui::Event::Key { key, pressed: true, .. } = ev {
                        if *key != egui::Key::Escape {
                            let label = format!("{key:?}");
                            settings.keybinds.insert(key_name.clone(), label.clone());
                            if key_name == "keymap_zoom" {
                                settings.zoom_key = label;
                            }
                            action = TitleAction::ApplySettings;
                        }
                        state.listening_keybind = None;
                        break;
                    }
                }
            });
        }
    }

    egui::CentralPanel::default()
        .frame(egui::Frame::NONE)
        .show(ctx, |ui| {
            let p_owned = ui.painter().clone();
            let p = &p_owned;
            let cx = screen.center().x;

            if state.page != TitlePage::Home {
                let deepslate = matches!(state.page, TitlePage::Multiplayer | TitlePage::HostWan);
                paint_dirt_screen_bg(p, screen, theme, deepslate, 68.0, 96.0);
            } else {
                // Subtle vignette over WgpuRenderer's rotating 3D voxel panorama background
                p.rect_filled(screen, 0.0, Color32::from_rgba_unmultiplied(8, 12, 22, 42));
            }

            match state.page {
                TitlePage::Home => {
                    // 3D Embossed CUBEX Logo
                    let lw = (screen.width() * 0.40).clamp(320.0, 560.0);
                    let lh = lw * (theme.cubex_logo.size()[1] as f32 / theme.cubex_logo.size()[0] as f32);
                    let logo_rect = Rect::from_center_size(Pos2::new(cx, screen.height() * 0.21), Vec2::new(lw, lh));
                    p.image(
                        theme.cubex_logo.id(),
                        logo_rect,
                        Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                        Color32::WHITE,
                    );

                    // Bouncing diagonal yellow splash tagline
                    let pulse_sz = 15.5 + (time_sec * 5.2).sin() * 1.5;
                    let splash_pos = Pos2::new(logo_rect.right() - 150.0, logo_rect.bottom() - 12.0);
                    let sh_galley = ctx.fonts(|f| {
                        f.layout_no_wrap(state.splash.clone(), egui::FontId::monospace(pulse_sz), Color32::from_rgb(63, 63, 0))
                    });
                    let fg_galley = ctx.fonts(|f| {
                        f.layout_no_wrap(state.splash.clone(), egui::FontId::monospace(pulse_sz), Color32::YELLOW)
                    });
                    let mut sh_shape = egui::epaint::TextShape::new(splash_pos + Vec2::new(2.0, 2.0), sh_galley, Color32::from_rgb(63, 63, 0));
                    sh_shape.angle = -0.26;
                    p.add(sh_shape);
                    let mut fg_shape = egui::epaint::TextShape::new(splash_pos, fg_galley, Color32::YELLOW);
                    fg_shape.angle = -0.26;
                    p.add(fg_shape);

                    // Classic Java Edition center stack
                    let bw = (screen.width() * 0.34).clamp(380.0, 440.0);
                    let mut y = (screen.height() * 0.40).max(logo_rect.bottom() + 24.0);
                    if btn(ui, theme, cx - bw * 0.5, y, bw, 46.0, "Singleplayer", false) {
                        state.refresh_worlds(worlds_dir);
                        state.confirm_delete_world = None;
                        state.page = TitlePage::Singleplayer;
                    }
                    y += 54.0;
                    if btn(ui, theme, cx - bw * 0.5, y, bw, 46.0, "Multiplayer", false) {
                        state.page = TitlePage::Multiplayer;
                        state.trigger_async_refresh(tokio_rt);
                    }
                    y += 54.0;
                    if btn(ui, theme, cx - bw * 0.5, y, bw, 46.0, "Host WAN (playit.gg)", false) {
                        state.copied_notice = None;
                        state.page = TitlePage::HostWan;
                    }
                    y += 68.0;

                    let gap = 8.0;
                    let sub_w = (bw - gap * 2.0) / 3.0;
                    if btn(ui, theme, cx - bw * 0.5, y, sub_w, 42.0, "Options...", false) {
                        state.options_tab = TitleOptionsTab::General;
                        state.page = TitlePage::Options;
                    }
                    if btn(ui, theme, cx - sub_w * 0.5, y, sub_w, 42.0, "Resource Packs", false) {
                        state.page = TitlePage::ResourcePacks;
                    }
                    if btn(ui, theme, cx + bw * 0.5 - sub_w, y, sub_w, 42.0, "Quit Game", false) {
                        action = TitleAction::QuitGame;
                    }

                    // Player Skin Preview Card on right side (fits comfortably at 1280x720+)
                    let card_w = 180.0;
                    let card_h = 224.0;
                    let card_x = (cx + bw * 0.5 + 36.0).min(screen.right() - card_w - 20.0);
                    let active_p1_tex = state
                        .available_skins
                        .iter()
                        .find(|s| s.id.eq_ignore_ascii_case(&settings.player_skin))
                        .map(|sk| {
                            state
                                .skin_preview_handles
                                .entry(sk.id.clone())
                                .or_insert_with(|| {
                                    let ci = egui::ColorImage::from_rgba_unmultiplied(
                                        [sk.image.width() as usize, sk.image.height() as usize],
                                        sk.image.as_raw(),
                                    );
                                    ctx.load_texture(
                                        format!("skin_preview_{}", sk.id),
                                        ci,
                                        egui::TextureOptions::NEAREST,
                                    )
                                })
                                .clone()
                        })
                        .or_else(|| state.icons.skin_tex.clone());
                    if card_x > cx + bw * 0.5 + 8.0 {
                        let card_y = (screen.height() * 0.40).min(screen.bottom() - card_h - 84.0);
                        let card_r = Rect::from_min_size(Pos2::new(card_x, card_y), Vec2::new(card_w, card_h));
                        paint_player_skin_card(
                            p,
                            card_r,
                            active_p1_tex.as_ref(),
                            &settings.player_name,
                            time_sec,
                        );
                        // Player name editor directly below card
                        let name_r = Rect::from_min_size(Pos2::new(card_x + 10.0, card_r.bottom() + 6.0), Vec2::new(card_w - 20.0, 26.0));
                        ui.allocate_new_ui(egui::UiBuilder::new().max_rect(name_r), |ui| {
                            if ui.text_edit_singleline(&mut settings.player_name).changed() {
                                action = TitleAction::ApplySettings;
                            }
                        });
                        if btn(
                            ui,
                            theme,
                            card_x + 10.0,
                            card_r.bottom() + 38.0,
                            card_w - 20.0,
                            34.0,
                            "Skins...",
                            is_carterpcs_skin(&settings.player_skin),
                        ) {
                            state.page = TitlePage::Skins;
                        }
                    } else if btn(
                        ui,
                        theme,
                        cx - bw * 0.5,
                        y + 52.0,
                        bw,
                        40.0,
                        "Skins...",
                        is_carterpcs_skin(&settings.player_skin),
                    ) {
                        state.page = TitlePage::Skins;
                    }

                    // Bottom corner version & profile footer
                    mc_shadow_text(
                        p,
                        Pos2::new(14.0, screen.bottom() - 16.0),
                        Align2::LEFT_CENTER,
                        "Cubex-RS 0.1.0 (Mineclonia + wgpu Vulkan + LuaJIT)",
                        12.0,
                        Color32::WHITE,
                    );
                    mc_shadow_text(
                        p,
                        Pos2::new(screen.right() - 14.0, screen.bottom() - 16.0),
                        Align2::RIGHT_CENTER,
                        &format!("Player: {}  •  F11 Fullscreen", settings.player_name),
                        12.0,
                        Color32::from_rgb(200, 245, 200),
                    );
                }

                TitlePage::Singleplayer => {
                    mc_shadow_text(p, Pos2::new(cx, 26.0), Align2::CENTER_CENTER, "Select World", 22.0, Color32::WHITE);
                    let search_r = Rect::from_center_size(Pos2::new(cx, 52.0), Vec2::new(360.0, 26.0));
                    ui.allocate_new_ui(egui::UiBuilder::new().max_rect(search_r), |ui| {
                        ui.add(egui::TextEdit::singleline(&mut state.world_search).hint_text("Search worlds..."));
                    });

                    let list_w = (screen.width() * 0.68).clamp(540.0, 760.0);
                    let list_rect = Rect::from_min_max(
                        Pos2::new(cx - list_w * 0.5, 76.0),
                        Pos2::new(cx + list_w * 0.5, screen.bottom() - 108.0),
                    );
                    p.rect_filled(list_rect, 0.0, Color32::from_rgba_unmultiplied(0, 0, 0, 175));
                    p.rect_stroke(list_rect, 0.0, Stroke::new(2.0, Color32::from_rgb(80, 80, 80)), egui::StrokeKind::Inside);

                    let q = state.world_search.trim().to_ascii_lowercase();
                    ui.allocate_new_ui(egui::UiBuilder::new().max_rect(list_rect.shrink(6.0)), |ui| {
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            for (i, w) in state.worlds.iter().enumerate() {
                                if !q.is_empty() && !w.name.to_ascii_lowercase().contains(&q) {
                                    continue;
                                }
                                let sel = i == state.selected_world;
                                let (row_r, resp) = ui.allocate_exact_size(Vec2::new(list_rect.width() - 24.0, 64.0), egui::Sense::click());
                                if sel {
                                    ui.painter().rect_filled(row_r, 0.0, Color32::from_rgba_unmultiplied(40, 80, 40, 150));
                                    ui.painter().rect_stroke(row_r, 0.0, Stroke::new(1.5, Color32::WHITE), egui::StrokeKind::Inside);
                                }
                                let icon_rgba = generate_server_icon_rgba16(&w.name);
                                let tid = state.icons.get_or_insert_server_icon(ctx, &format!("world_{}", w.name), &icon_rgba);
                                ui.painter().image(
                                    tid,
                                    Rect::from_min_size(row_r.min + Vec2::new(8.0, 6.0), Vec2::splat(52.0)),
                                    Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                                    Color32::WHITE,
                                );
                                mc_shadow_text(ui.painter(), row_r.min + Vec2::new(72.0, 18.0), Align2::LEFT_CENTER, &w.name, 16.0, Color32::WHITE);
                                let meta_line = format!(
                                    "{} • {} • {} • Seed {}",
                                    w.gamemode,
                                    w.world_type.label(),
                                    w.difficulty.label(),
                                    w.seed
                                );
                                mc_shadow_text(ui.painter(), row_r.min + Vec2::new(72.0, 42.0), Align2::LEFT_CENTER, &meta_line, 12.0, Color32::GRAY);
                                if resp.clicked() {
                                    state.selected_world = i;
                                }
                                if resp.double_clicked() {
                                    action = TitleAction::LaunchSingleplayer(w.clone());
                                }
                            }
                        });
                    });

                    // Delete confirmation modal overlay
                    if let Some(target_w) = state.confirm_delete_world.clone() {
                        let modal = Rect::from_center_size(screen.center(), Vec2::new(500.0, 190.0));
                        p.rect_filled(screen, 0.0, Color32::from_rgba_unmultiplied(0, 0, 0, 170));
                        p.rect_filled(modal, 0.0, Color32::from_rgb(36, 36, 40));
                        p.rect_stroke(modal, 0.0, Stroke::new(2.0, Color32::WHITE), egui::StrokeKind::Inside);
                        mc_shadow_text(p, Pos2::new(cx, modal.top() + 34.0), Align2::CENTER_CENTER, &format!("Delete '{}'?", target_w.name), 18.0, Color32::from_rgb(255, 100, 100));
                        mc_shadow_text(p, Pos2::new(cx, modal.top() + 74.0), Align2::CENTER_CENTER, "It will be lost forever! (A long time!)", 13.0, Color32::LIGHT_GRAY);
                        if btn(ui, theme, modal.left() + 34.0, modal.bottom() - 56.0, 200.0, 40.0, "Delete", false) {
                            let _ = delete_world(&target_w.path);
                            state.confirm_delete_world = None;
                            state.refresh_worlds(worlds_dir);
                        }
                        if btn(ui, theme, modal.right() - 234.0, modal.bottom() - 56.0, 200.0, 40.0, "Cancel", true) {
                            state.confirm_delete_world = None;
                        }
                    } else {
                        let by = screen.bottom() - 92.0;
                        let half = (list_w - 12.0) * 0.5;
                        if btn(ui, theme, cx - list_w * 0.5, by, half, 40.0, "Play Selected World", true) {
                            if let Some(w) = state.worlds.get(state.selected_world).cloned() {
                                action = TitleAction::LaunchSingleplayer(w);
                            } else if let Ok(w) = create_world_ex(
                                worlds_dir,
                                &next_default_world_name(worlds_dir),
                                "Survival",
                                random_world_seed(),
                                WorldGenType::Default,
                                Difficulty::Normal,
                                Gamerules::default(),
                            ) {
                                state.refresh_worlds(worlds_dir);
                                action = TitleAction::LaunchSingleplayer(w);
                            }
                        }
                        if btn(ui, theme, cx + 6.0, by, half, 40.0, "Create New World", false) {
                            state.new_world_name = next_default_world_name(worlds_dir);
                            state.new_world_seed.clear();
                            state.page = TitlePage::CreateWorld;
                        }

                        let third = (list_w - 20.0) / 3.0;
                        let has_sel = state.worlds.get(state.selected_world).is_some();
                        if btn_en(ui, theme, cx - list_w * 0.5, by + 46.0, third, 36.0, "Delete", false, has_sel) {
                            if let Some(w) = state.worlds.get(state.selected_world).cloned() {
                                state.confirm_delete_world = Some(w);
                            }
                        }
                        if btn(ui, theme, cx - third * 0.5, by + 46.0, third, 36.0, "Refresh", false) {
                            state.refresh_worlds(worlds_dir);
                        }
                        if btn(ui, theme, cx + list_w * 0.5 - third, by + 46.0, third, 36.0, "Cancel", false) {
                            state.page = TitlePage::Home;
                        }
                    }
                }

                TitlePage::CreateWorld => {
                    mc_shadow_text(p, Pos2::new(cx, 34.0), Align2::CENTER_CENTER, "Create New World", 22.0, Color32::WHITE);
                    let w = (screen.width() * 0.58).clamp(480.0, 620.0);
                    let form_r = Rect::from_min_max(
                        Pos2::new(cx - w * 0.5, 76.0),
                        Pos2::new(cx + w * 0.5, screen.bottom() - 102.0),
                    );
                    ui.allocate_new_ui(egui::UiBuilder::new().max_rect(form_r), |ui| {
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            ui.label(egui::RichText::new("World Name").color(Color32::LIGHT_GRAY));
                            ui.text_edit_singleline(&mut state.new_world_name);
                            ui.add_space(8.0);
                            ui.label(egui::RichText::new("World Seed (numeric or text)").color(Color32::LIGHT_GRAY));
                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::TextEdit::singleline(&mut state.new_world_seed)
                                        .hint_text("Leave blank for random seed")
                                        .desired_width((w - 130.0).max(160.0)),
                                );
                                if mc_button(ui, theme, "Random", 110.0, 28.0).clicked() {
                                    state.new_world_seed = random_world_seed().to_string();
                                }
                            });
                            ui.add_space(12.0);

                            let col_w = (w - 20.0) * 0.5;
                            egui::Grid::new("create_world_grid").spacing(Vec2::new(12.0, 10.0)).show(ui, |ui| {
                                let gm_lbl = format!("Game Mode: {}", state.new_world_gamemode);
                                if mc_button(ui, theme, &gm_lbl, col_w, 42.0).clicked() {
                                    state.new_world_gamemode = match state.new_world_gamemode.as_str() {
                                        "Survival" => "Creative".into(),
                                        "Creative" => "Spectator".into(),
                                        _ => "Survival".into(),
                                    };
                                    state.new_world_creative = state.new_world_gamemode == "Creative";
                                }
                                let wt_lbl = format!("World Type: {}", state.new_world_type.label());
                                if mc_button(ui, theme, &wt_lbl, col_w, 42.0).clicked() {
                                    state.new_world_type = state.new_world_type.cycle();
                                }
                                ui.end_row();

                                let diff_lbl = format!("Difficulty: {}", state.new_world_difficulty.label());
                                if mc_button(ui, theme, &diff_lbl, col_w, 42.0).clicked() {
                                    state.new_world_difficulty = state.new_world_difficulty.cycle();
                                }
                                let ki = state.new_gamerules.keep_inventory;
                                if mc_button_styled(ui, theme, &format!("Keep Inventory on Death: {}", on_off(ki)), col_w, 42.0, ki).clicked() {
                                    state.new_gamerules.keep_inventory = !ki;
                                }
                                ui.end_row();

                                let dlc = state.new_gamerules.do_daylight_cycle;
                                if mc_button_styled(ui, theme, &format!("doDaylightCycle: {}", on_off(dlc)), col_w, 42.0, dlc).clicked() {
                                    state.new_gamerules.do_daylight_cycle = !dlc;
                                }
                                let pvp = state.new_gamerules.pvp;
                                if mc_button_styled(ui, theme, &format!("pvp: {}", on_off(pvp)), col_w, 42.0, pvp).clicked() {
                                    state.new_gamerules.pvp = !pvp;
                                }
                                ui.end_row();

                                let fd = state.new_gamerules.fall_damage;
                                if mc_button_styled(ui, theme, &format!("fallDamage: {}", on_off(fd)), col_w, 42.0, fd).clicked() {
                                    state.new_gamerules.fall_damage = !fd;
                                }
                                ui.end_row();
                            });
                        });
                    });

                    let by = screen.bottom() - 68.0;
                    if btn(ui, theme, cx - 220.0, by, 210.0, 42.0, "Create New World", true) {
                        let seed = parse_seed(&state.new_world_seed);
                        settings.world_gen_type = state.new_world_type;
                        settings.difficulty = state.new_world_difficulty;
                        settings.gamerules = state.new_gamerules.clone();
                        if let Ok(w) = create_world_ex(
                            worlds_dir,
                            &state.new_world_name,
                            &state.new_world_gamemode,
                            seed,
                            state.new_world_type,
                            state.new_world_difficulty,
                            state.new_gamerules.clone(),
                        ) {
                            state.refresh_worlds(worlds_dir);
                            state.new_world_name = next_default_world_name(worlds_dir);
                            state.new_world_seed.clear();
                            state.page = TitlePage::Singleplayer;
                            action = TitleAction::LaunchSingleplayer(w);
                        }
                    }
                    if btn(ui, theme, cx + 10.0, by, 210.0, 42.0, "Cancel", false) {
                        state.page = TitlePage::Singleplayer;
                    }
                }

                TitlePage::Multiplayer => {
                    mc_shadow_text(p, Pos2::new(cx, 34.0), Align2::CENTER_CENTER, "Play Multiplayer", 22.0, Color32::WHITE);
                    let list_w = (screen.width() * 0.72).clamp(620.0, 820.0);
                    let list_rect = Rect::from_min_max(
                        Pos2::new(cx - list_w * 0.5, 72.0),
                        Pos2::new(cx + list_w * 0.5, screen.bottom() - 144.0),
                    );
                    p.rect_filled(list_rect, 0.0, Color32::from_rgba_unmultiplied(0, 0, 0, 180));
                    p.rect_stroke(list_rect, 0.0, Stroke::new(2.0, Color32::from_rgb(70, 70, 70)), egui::StrokeKind::Inside);

                    ui.allocate_new_ui(egui::UiBuilder::new().max_rect(list_rect.shrink(6.0)), |ui| {
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            for (i, s) in state.servers.iter().enumerate() {
                                let sel = i == state.selected_server;
                                let (row_r, resp) = ui.allocate_exact_size(Vec2::new(list_rect.width() - 24.0, 74.0), egui::Sense::click());
                                if sel {
                                    ui.painter().rect_filled(row_r, 0.0, Color32::from_rgba_unmultiplied(40, 60, 100, 150));
                                    ui.painter().rect_stroke(row_r, 0.0, Stroke::new(1.5, Color32::WHITE), egui::StrokeKind::Inside);
                                }

                                // 64x64 pixel-crisp server icon
                                let icon_rgba = generate_server_icon_rgba16(&s.icon_seed());
                                let tid = state.icons.get_or_insert_server_icon(ctx, &s.icon_seed(), &icon_rgba);
                                ui.painter().image(
                                    tid,
                                    Rect::from_min_size(row_r.min + Vec2::new(6.0, 5.0), Vec2::splat(64.0)),
                                    Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                                    Color32::WHITE,
                                );

                                let title = if s.is_lan {
                                    format!("{} [LAN]", s.name)
                                } else if s.featured {
                                    format!("{} [{}]", s.name, s.category)
                                } else {
                                    s.name.clone()
                                };
                                mc_shadow_text(ui.painter(), row_r.min + Vec2::new(80.0, 14.0), Align2::LEFT_CENTER, &title, 15.0, Color32::WHITE);

                                let job = motd_to_layout_job(&s.motd, 12.0);
                                let galley = ui.fonts(|f| f.layout_job(job));
                                ui.painter().galley(row_r.min + Vec2::new(80.0, 28.0), galley, Color32::GRAY);

                                // Player count + ping ms + 5 signal bars
                                let players_txt = format!("{}/{}", s.players_online, s.players_max);
                                mc_shadow_text(ui.painter(), Pos2::new(row_r.right() - 88.0, row_r.top() + 14.0), Align2::RIGHT_CENTER, &players_txt, 11.5, Color32::LIGHT_GRAY);
                                let ping_txt = s.ping_ms.map(|m| format!("{m}ms")).unwrap_or_else(|| "---".into());
                                mc_shadow_text(ui.painter(), Pos2::new(row_r.right() - 42.0, row_r.top() + 14.0), Align2::RIGHT_CENTER, &ping_txt, 11.0, Color32::LIGHT_GRAY);

                                let bars = s.signal_bars();
                                for b in 0..5 {
                                    let bh = 4.0 + b as f32 * 3.0;
                                    let bx = row_r.right() - 34.0 + b as f32 * 5.0;
                                    let by = row_r.top() + 22.0 - bh;
                                    let col = if b < bars {
                                        if bars >= 4 { Color32::GREEN } else if bars >= 2 { Color32::YELLOW } else { Color32::RED }
                                    } else {
                                        Color32::DARK_GRAY
                                    };
                                    ui.painter().rect_filled(Rect::from_min_size(Pos2::new(bx, by), Vec2::new(3.0, bh)), 0.0, col);
                                }

                                if resp.clicked() {
                                    state.selected_server = i;
                                }
                                if resp.double_clicked() {
                                    action = TitleAction::JoinServer(s.host.clone(), s.port);
                                }
                            }
                        });
                    });

                    // Status + Direct connect bar
                    if !state.status_msg.is_empty() {
                        mc_shadow_text(p, Pos2::new(cx, screen.bottom() - 134.0), Align2::CENTER_CENTER, &state.status_msg, 12.0, Color32::from_rgb(255, 160, 80));
                    }
                    let d_r = Rect::from_min_size(Pos2::new(cx - list_w * 0.5, screen.bottom() - 122.0), Vec2::new(list_w - 200.0, 30.0));
                    ui.allocate_new_ui(egui::UiBuilder::new().max_rect(d_r), |ui| {
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("Direct / CBX-:").color(Color32::WHITE));
                            ui.add(egui::TextEdit::singleline(&mut state.direct_addr).hint_text("host:8090 or CBX-..."));
                        });
                    });
                    if btn(ui, theme, cx + list_w * 0.5 - 190.0, screen.bottom() - 124.0, 190.0, 32.0, "Direct Connect", true) {
                        if let Ok((h, pt)) = decode_invite(&state.direct_addr) {
                            state.status_msg.clear();
                            action = TitleAction::JoinServer(h, pt);
                        } else {
                            state.status_msg = "Invalid host:port or CBX-... invite code".into();
                        }
                    }

                    let by = screen.bottom() - 82.0;
                    let bw = (list_w - 32.0) / 5.0;
                    if btn(ui, theme, cx - list_w * 0.5, by, bw, 38.0, "Join Server", true) {
                        if let Some(s) = state.servers.get(state.selected_server) {
                            action = TitleAction::JoinServer(s.host.clone(), s.port);
                        }
                    }
                    if btn(ui, theme, cx - list_w * 0.5 + bw + 8.0, by, bw, 38.0, "Add Server", false) {
                        state.show_add_modal = !state.show_add_modal;
                    }
                    let can_del_srv = state
                        .servers
                        .get(state.selected_server)
                        .map(|s| !s.featured && !s.is_lan)
                        .unwrap_or(false);
                    if btn_en(ui, theme, cx - list_w * 0.5 + (bw + 8.0) * 2.0, by, bw, 38.0, "Delete", false, can_del_srv) {
                        if state.selected_server < state.servers.len() {
                            state.servers.remove(state.selected_server);
                            if state.selected_server >= state.servers.len() && !state.servers.is_empty() {
                                state.selected_server = state.servers.len() - 1;
                            }
                            save_custom_servers(servers_file, &state.servers);
                        }
                    }
                    let refreshing = *state.server_refresher.in_progress.lock();
                    let ref_lbl = if refreshing { "Scanning..." } else { "Refresh" };
                    if btn(ui, theme, cx - list_w * 0.5 + (bw + 8.0) * 3.0, by, bw, 38.0, ref_lbl, false) {
                        state.trigger_async_refresh(tokio_rt);
                    }
                    if btn(ui, theme, cx + list_w * 0.5 - bw, by, bw, 38.0, "Cancel", false) {
                        state.show_add_modal = false;
                        state.page = TitlePage::Home;
                    }

                    // Add Server modal
                    if state.show_add_modal {
                        let modal = Rect::from_center_size(screen.center(), Vec2::new(460.0, 220.0));
                        p.rect_filled(screen, 0.0, Color32::from_rgba_unmultiplied(0, 0, 0, 160));
                        p.rect_filled(modal, 0.0, Color32::from_rgb(36, 36, 40));
                        p.rect_stroke(modal, 0.0, Stroke::new(2.0, Color32::WHITE), egui::StrokeKind::Inside);
                        ui.allocate_new_ui(egui::UiBuilder::new().max_rect(modal.shrink(16.0)), |ui| {
                            ui.label(egui::RichText::new("Add Server (supports host:port or CBX-...)").strong().color(Color32::WHITE));
                            ui.add_space(6.0);
                            ui.label("Server Name:");
                            ui.text_edit_singleline(&mut state.add_name);
                            ui.label("Address / CBX Invite:");
                            ui.text_edit_singleline(&mut state.add_addr);
                            ui.add_space(10.0);
                            ui.horizontal(|ui| {
                                if mc_button_green(ui, theme, "Save", 180.0, 36.0).clicked() {
                                    if let Ok((h, pt)) = decode_invite(&state.add_addr) {
                                        state.servers.insert(
                                            0,
                                            ServerEntry {
                                                name: if state.add_name.trim().is_empty() { h.clone() } else { state.add_name.trim().to_string() },
                                                host: h.clone(),
                                                port: pt,
                                                motd: format!("§a{h}:{pt}\n§7Custom Cubex Server"),
                                                featured: false,
                                                category: "Custom".into(),
                                                players_online: 0,
                                                players_max: 20,
                                                ping_ms: None,
                                                online: false,
                                                is_lan: false,
                                            },
                                        );
                                        save_custom_servers(servers_file, &state.servers);
                                        state.show_add_modal = false;
                                        state.trigger_async_refresh(tokio_rt);
                                    } else {
                                        state.status_msg = "Invalid server address or CBX code".into();
                                    }
                                }
                                if mc_button(ui, theme, "Cancel", 180.0, 36.0).clicked() {
                                    state.show_add_modal = false;
                                }
                            });
                        });
                    }
                }

                TitlePage::HostWan => {
                    mc_shadow_text(p, Pos2::new(cx, 34.0), Align2::CENTER_CENTER, "Host WAN (playit.gg)", 22.0, Color32::from_rgb(85, 255, 85));
                    let st = tunnel.status.lock().clone();
                    let w = (screen.width() * 0.62).clamp(520.0, 640.0);
                    let card = Rect::from_min_size(Pos2::new(cx - w * 0.5, 82.0), Vec2::new(w, 150.0));
                    p.rect_filled(card, 0.0, Color32::from_rgba_unmultiplied(0, 0, 0, 195));
                    p.rect_stroke(card, 0.0, Stroke::new(1.5, Color32::from_rgb(80, 80, 80)), egui::StrokeKind::Inside);
                    mc_shadow_text(p, Pos2::new(cx, card.top() + 24.0), Align2::CENTER_CENTER, &format!("Status: {}", st.status_text), 15.0, Color32::WHITE);
                    mc_shadow_text(p, Pos2::new(cx, card.top() + 58.0), Align2::CENTER_CENTER, &format!("Public Address: {}", st.public_address), 14.0, Color32::YELLOW);
                    mc_shadow_text(p, Pos2::new(cx, card.top() + 92.0), Align2::CENTER_CENTER, &format!("Invite Code: {}", st.invite_code), 14.0, Color32::from_rgb(85, 255, 255));
                    if !st.claim_url.is_empty() {
                        mc_shadow_text(p, Pos2::new(cx, card.top() + 124.0), Align2::CENTER_CENTER, &format!("Claim URL: {}", st.claim_url), 12.0, Color32::from_rgb(255, 180, 85));
                    } else if let Some(ref notice) = state.copied_notice {
                        mc_shadow_text(p, Pos2::new(cx, card.top() + 124.0), Align2::CENTER_CENTER, notice, 12.0, Color32::from_rgb(85, 255, 85));
                    }

                    let y = card.bottom() + 14.0;
                    let bw = (w - 16.0) / 3.0;
                    if st.running {
                        if btn(ui, theme, cx - w * 0.5, y, bw, 42.0, "Stop Tunnel", false) {
                            action = TitleAction::StopTunnel;
                        }
                    } else if btn(ui, theme, cx - w * 0.5, y, bw, 42.0, "Start Playit Tunnel", true) {
                        action = TitleAction::StartTunnel;
                    }
                    if btn_en(ui, theme, cx - bw * 0.5, y, bw, 42.0, "Copy Invite Code", false, !st.invite_code.is_empty()) {
                        ctx.copy_text(st.invite_code.clone());
                        state.copied_notice = Some("Copied CBX-... invite code!".into());
                    }
                    if btn_en(ui, theme, cx + w * 0.5 - bw, y, bw, 42.0, "Copy Address", false, !st.public_address.is_empty()) {
                        ctx.copy_text(st.public_address.clone());
                        state.copied_notice = Some("Copied public address!".into());
                    }

                    let log_r = Rect::from_min_max(
                        Pos2::new(cx - w * 0.5, y + 54.0),
                        Pos2::new(cx + w * 0.5, (screen.bottom() - 104.0).max(y + 110.0)),
                    );
                    p.rect_filled(log_r, 0.0, Color32::from_rgba_unmultiplied(8, 8, 8, 220));
                    ui.allocate_new_ui(egui::UiBuilder::new().max_rect(log_r.shrink(6.0)), |ui| {
                        egui::ScrollArea::vertical().stick_to_bottom(true).show(ui, |ui| {
                            for line in &st.logs {
                                ui.label(egui::RichText::new(line).monospace().size(11.5).color(Color32::from_rgb(85, 255, 85)));
                            }
                        });
                    });

                    if btn(ui, theme, cx - 120.0, screen.bottom() - 68.0, 240.0, 42.0, "Done", false) {
                        state.page = TitlePage::Home;
                    }
                }

                TitlePage::Options => {
                    mc_shadow_text(p, Pos2::new(cx, 34.0), Align2::CENTER_CENTER, "Options", 22.0, Color32::WHITE);
                    let w = (screen.width() * 0.68).clamp(560.0, 720.0);
                    let tabs = [
                        ("General", TitleOptionsTab::General),
                        ("Video", TitleOptionsTab::Video),
                        ("Audio", TitleOptionsTab::Audio),
                        ("Controls", TitleOptionsTab::Controls),
                        ("Co-op & Gamepad", TitleOptionsTab::Coop),
                    ];
                    let tab_w = (w - 32.0) / tabs.len() as f32;
                    for (idx, (lbl, tab)) in tabs.iter().enumerate() {
                        let tx = cx - w * 0.5 + idx as f32 * (tab_w + 8.0);
                        if btn(ui, theme, tx, 76.0, tab_w, 34.0, lbl, state.options_tab == *tab) {
                            state.listening_keybind = None;
                            state.options_tab = *tab;
                        }
                    }

                    let body_r = Rect::from_min_max(
                        Pos2::new(cx - w * 0.5, 122.0),
                        Pos2::new(cx + w * 0.5, screen.bottom() - 104.0),
                    );
                    let col_w = (w - 20.0) * 0.5;
                    ui.allocate_new_ui(egui::UiBuilder::new().max_rect(body_r), |ui| {
                        egui::ScrollArea::vertical().show(ui, |ui| match state.options_tab {
                            TitleOptionsTab::General => {
                                if mc_slider_f32(ui, theme, "FOV", &mut settings.fov, 30.0..=110.0, w - 12.0, 42.0, |v| {
                                    format!("{}", v.round() as i32)
                                }) {
                                    action = TitleAction::ApplySettings;
                                }
                                ui.add_space(8.0);
                                let liq = settings.liquids_pointable;
                                if mc_button_styled(ui, theme, &format!("Liquids Pointable: {}", on_off(liq)), w - 12.0, 42.0, liq).clicked() {
                                    settings.liquids_pointable = !liq;
                                    action = TitleAction::ApplySettings;
                                }
                                ui.add_space(8.0);
                                let f3 = settings.f3_enabled;
                                if mc_button_styled(ui, theme, &format!("F3 Debug Overlay: {}", on_off(f3)), w - 12.0, 42.0, f3).clicked() {
                                    settings.f3_enabled = !f3;
                                    action = TitleAction::ApplySettings;
                                }
                            }
                            TitleOptionsTab::Video => {
                                if crate::ui::pause_menu::paint_enhanced_graphics_controls(
                                    ui,
                                    theme,
                                    settings,
                                    w - 12.0,
                                ) {
                                    action = TitleAction::ApplySettings;
                                }
                                ui.add_space(10.0);
                                egui::Grid::new("title_video_grid").spacing(Vec2::new(12.0, 10.0)).show(ui, |ui| {
                                    let mut rd = settings.render_distance_chunks as f32;
                                    if mc_slider_f32(ui, theme, "Render Distance", &mut rd, 2.0..=32.0, col_w, 40.0, |v| {
                                        format!("{} Chunks", v.round() as u32)
                                    }) {
                                        settings.render_distance_chunks = rd.round() as u32;
                                        action = TitleAction::ApplySettings;
                                    }
                                    let mut fps_val = if settings.max_fps == 0 { 260.0 } else { settings.max_fps as f32 };
                                    if mc_slider_f32(ui, theme, "Max Framerate", &mut fps_val, 30.0..=260.0, col_w, 40.0, |v| {
                                        if v >= 255.0 { "Unlimited".into() } else { format!("{} FPS", (v / 5.0).round() as u32 * 5) }
                                    }) {
                                        settings.max_fps = if fps_val >= 255.0 { 0 } else { (fps_val / 5.0).round() as u32 * 5 };
                                        action = TitleAction::ApplySettings;
                                    }
                                    ui.end_row();

                                    if mc_button_styled(ui, theme, &format!("VSync: {}", on_off(settings.vsync)), col_w, 40.0, settings.vsync).clicked() {
                                        settings.vsync = !settings.vsync;
                                        action = TitleAction::ApplySettings;
                                    }
                                    if mc_button_styled(
                                        ui,
                                        theme,
                                        &format!(
                                            "Anisotropic & Mipmap: {}",
                                            settings.anisotropy_quality.label()
                                        ),
                                        col_w,
                                        40.0,
                                        settings.anisotropy_quality.is_enabled(),
                                    )
                                    .clicked()
                                    {
                                        settings.anisotropy_quality =
                                            settings.anisotropy_quality.cycle();
                                        settings.mipmapping =
                                            settings.anisotropy_quality.is_enabled();
                                        action = TitleAction::ApplySettings;
                                    }
                                    ui.end_row();

                                    if mc_button_styled(ui, theme, &format!("Smooth Lighting: {}", on_off(settings.smooth_lighting)), col_w, 40.0, settings.smooth_lighting).clicked() {
                                        settings.smooth_lighting = !settings.smooth_lighting;
                                        action = TitleAction::ApplySettings;
                                    }
                                    if mc_button_styled(ui, theme, &format!("Dynamic Shadows: {}", on_off(settings.dynamic_shadows)), col_w, 40.0, settings.dynamic_shadows).clicked() {
                                        settings.dynamic_shadows = !settings.dynamic_shadows;
                                        action = TitleAction::ApplySettings;
                                    }
                                    ui.end_row();
                                });
                                ui.add_space(6.0);
                                if mc_button_styled(ui, theme, &format!("Waving Water: {}", on_off(settings.waving_water)), w - 12.0, 40.0, settings.waving_water).clicked() {
                                    settings.waving_water = !settings.waving_water;
                                    action = TitleAction::ApplySettings;
                                }
                            }
                            TitleOptionsTab::Audio => {
                                if mc_slider_f32(ui, theme, "Master Volume", &mut settings.audio.master, 0.0..=1.0, w - 12.0, 42.0, |v| {
                                    format!("{}%", (v * 100.0).round() as i32)
                                }) {
                                    action = TitleAction::ApplySettings;
                                }
                                ui.add_space(8.0);
                                egui::Grid::new("title_audio_grid").spacing(Vec2::new(12.0, 8.0)).show(ui, |ui| {
                                    let items = [
                                        ("Music", &mut settings.audio.music),
                                        ("Blocks", &mut settings.audio.blocks),
                                        ("Weather", &mut settings.audio.weather),
                                        ("Hostile Mobs", &mut settings.audio.hostile),
                                        ("Friendly Mobs", &mut settings.audio.friendly),
                                        ("Ambient", &mut settings.audio.ambient),
                                    ];
                                    for (idx, (lbl, val)) in items.into_iter().enumerate() {
                                        if mc_slider_f32(ui, theme, lbl, val, 0.0..=1.0, col_w, 40.0, |v| {
                                            format!("{}%", (v * 100.0).round() as i32)
                                        }) {
                                            action = TitleAction::ApplySettings;
                                        }
                                        if idx % 2 == 1 {
                                            ui.end_row();
                                        }
                                    }
                                });
                            }
                            TitleOptionsTab::Controls => {
                                egui::Grid::new("title_ctrl_grid").spacing(Vec2::new(12.0, 8.0)).show(ui, |ui| {
                                    if mc_slider_f32(ui, theme, "Sensitivity", &mut settings.mouse_sensitivity, 0.05..=1.5, col_w, 40.0, |v| {
                                        format!("{}%", (v * 100.0).round() as i32)
                                    }) {
                                        action = TitleAction::ApplySettings;
                                    }
                                    if mc_button_styled(ui, theme, &format!("Invert Mouse: {}", on_off(settings.invert_mouse)), col_w, 40.0, settings.invert_mouse).clicked() {
                                        settings.invert_mouse = !settings.invert_mouse;
                                        action = TitleAction::ApplySettings;
                                    }
                                    ui.end_row();
                                    for (idx, (lbl, kname)) in [
                                        ("Optical Zoom", "keymap_zoom"),
                                        ("Inventory", "keymap_inventory"),
                                        ("Sneak", "keymap_sneak"),
                                        ("Sprint", "keymap_aux1"),
                                    ].iter().enumerate() {
                                        let cur = settings.keybinds.get(*kname).cloned().unwrap_or_else(|| "?".into());
                                        let is_l = state.listening_keybind.as_deref() == Some(*kname);
                                        let btxt = if is_l { format!("{lbl}: > Press Key <") } else { format!("{lbl}: {cur}") };
                                        if mc_button_styled(ui, theme, &btxt, col_w, 38.0, is_l).clicked() {
                                            state.listening_keybind = Some((*kname).to_string());
                                        }
                                        if idx % 2 == 1 {
                                            ui.end_row();
                                        }
                                    }
                                });
                            }
                            TitleOptionsTab::Coop => {
                                if crate::ui::pause_menu::paint_coop_and_controller_controls(
                                    ui,
                                    theme,
                                    settings,
                                    &state.connected_controllers,
                                    (col_w - 8.0).max(200.0),
                                ) {
                                    action = TitleAction::ApplySettings;
                                }
                            }
                        });
                    });

                    if btn(ui, theme, cx - 120.0, screen.bottom() - 68.0, 240.0, 42.0, "Done", true) {
                        state.listening_keybind = None;
                        state.page = TitlePage::Home;
                    }
                }

                TitlePage::Skins => {
                    mc_shadow_text(p, Pos2::new(cx, 34.0), Align2::CENTER_CENTER, "Player Skins", 22.0, Color32::WHITE);
                    let w = (screen.width() * 0.68).clamp(520.0, 740.0);
                    let panel_r = Rect::from_min_max(
                        Pos2::new(cx - w * 0.5, 72.0),
                        Pos2::new(cx + w * 0.5, screen.bottom() - 88.0),
                    );
                    let skins = state.available_skins.clone();
                    let changed = ui
                        .allocate_new_ui(egui::UiBuilder::new().max_rect(panel_r), |ui| {
                            crate::ui::pause_menu::paint_skin_selector_panel(
                                ctx,
                                ui,
                                theme,
                                settings,
                                &skins,
                                &mut state.skin_edit_slot,
                                &mut state.skin_preview_handles,
                                panel_r,
                                time_sec,
                            )
                        })
                        .inner;
                    if changed {
                        action = TitleAction::ApplySettings;
                    }
                    if btn(ui, theme, cx - 120.0, screen.bottom() - 68.0, 240.0, 42.0, "Done", true) {
                        state.page = TitlePage::Home;
                    }
                }

                TitlePage::ResourcePacks => {
                    mc_shadow_text(p, Pos2::new(cx, 34.0), Align2::CENTER_CENTER, "Select Resource Packs", 22.0, Color32::WHITE);
                    let w = (screen.width() * 0.56).clamp(440.0, 560.0);
                    let list_r = Rect::from_min_max(
                        Pos2::new(cx - w * 0.5, 84.0),
                        Pos2::new(cx + w * 0.5, screen.bottom() - 104.0),
                    );
                    ui.allocate_new_ui(egui::UiBuilder::new().max_rect(list_r), |ui| {
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            for pack in packs {
                                let active = settings.active_resource_pack == pack.path;
                                let lbl = if active { format!("{} [SELECTED]", pack.name) } else { pack.name.clone() };
                                if mc_button_styled(ui, theme, &lbl, w - 16.0, 44.0, active).clicked() {
                                    settings.active_resource_pack = pack.path.clone();
                                    action = TitleAction::ReloadPack(pack.path.clone());
                                }
                                ui.add_space(6.0);
                            }
                        });
                    });
                    if btn(ui, theme, cx - 120.0, screen.bottom() - 68.0, 240.0, 42.0, "Done", true) {
                        state.page = TitlePage::Home;
                    }
                }
            }
        });

    action
}

fn on_off(v: bool) -> &'static str {
    if v { "ON" } else { "OFF" }
}

/// Generate a fresh randomized world seed in `1..=(1 << 52)` (exact in IEEE-754 Lua numbers & `i64`).
pub fn random_world_seed() -> u64 {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (rng.gen::<u64>() & ((1u64 << 52) - 1)).max(1)
}

pub fn parse_seed(raw: &str) -> u64 {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return random_world_seed();
    }
    if let Ok(n) = trimmed.parse::<u64>() {
        return n;
    }
    let mut h = 14695981039346656037u64;
    for b in trimmed.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

fn btn(ui: &mut egui::Ui, theme: &McTheme, x: f32, y: f32, w: f32, h: f32, text: &str, green: bool) -> bool {
    btn_en(ui, theme, x, y, w, h, text, green, true)
}

#[allow(clippy::too_many_arguments)]
fn btn_en(
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

#[cfg(test)]
mod title_screen_tests {
    use super::parse_seed;
    use std::collections::HashSet;

    #[test]
    fn empty_seed_produces_non_constant_random_seeds() {
        let mut seen = HashSet::new();
        for _ in 0..8 {
            seen.insert(parse_seed(""));
        }
        assert!(seen.len() > 1, "expected random seeds when input is blank");
        assert_eq!(parse_seed("42"), 42);
        assert_eq!(parse_seed("my_world"), parse_seed("my_world"));
    }
}
