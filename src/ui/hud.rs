use egui::{Align2, Color32, Pos2, Rect, Stroke, Vec2};
use parking_lot::RwLock;
use std::sync::Arc;

use crate::config::EngineSettings;
use crate::lua_api::userdata::ObjectState;
use crate::network::motd::motd_to_layout_job_with_default;
use crate::physics::{GameMode, PlayerController, RaycastHit};
use crate::registry::EngineRegistries;
use crate::textures::TextureAtlas;
use crate::ui::theme::{mc_shadow_text, paint_item_in_slot, ItemIconCache};
pub use crate::ui::theme::{strip_luanti_escapes, HudIconTextures};
use crate::world::WorldStore;

pub struct HudTelemetry<'a> {
    pub atlas: Option<&'a TextureAtlas>,
    pub icon_cache: Option<&'a mut ItemIconCache>,
    pub world: Option<&'a WorldStore>,
    pub dig_progress: f32,
    pub biome_name: Option<&'a str>,
    pub targeted_param2: Option<u8>,
    pub armor_points: i32,
    pub xp_level: i32,
    pub xp_progress: f32,
    /// Optional local co-op corner badge (e.g. `"P1"` or `"P2"`).
    pub player_badge: Option<&'a str>,
    /// Remaining red screen hurt flash timer in `0.0..=0.25` seconds.
    pub hurt_flash: f32,
}

impl<'a> Default for HudTelemetry<'a> {
    fn default() -> Self {
        Self {
            atlas: None,
            icon_cache: None,
            world: None,
            dig_progress: 0.0,
            biome_name: None,
            targeted_param2: None,
            armor_points: 0,
            xp_level: 0,
            xp_progress: 0.0,
            player_badge: None,
            hurt_flash: 0.0,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn render_hud(
    ctx: &egui::Context,
    player: &PlayerController,
    player_state: &Arc<RwLock<ObjectState>>,
    settings: &EngineSettings,
    registries: &EngineRegistries,
    targeted: Option<&RaycastHit>,
    fps: f32,
    frame_ms: f32,
    chunks_loaded: usize,
    visible_chunks: u32,
    draw_calls: u32,
    gpu_name: &str,
    chat_messages: &[String],
    chat_input: Option<&str>,
) {
    render_hud_full(
        ctx,
        player,
        player_state,
        settings,
        registries,
        targeted,
        fps,
        frame_ms,
        chunks_loaded,
        visible_chunks,
        draw_calls,
        gpu_name,
        chat_messages,
        chat_input,
        HudTelemetry::default(),
    );
}

#[allow(clippy::too_many_arguments)]
pub fn render_hud_full(
    ctx: &egui::Context,
    player: &PlayerController,
    player_state: &Arc<RwLock<ObjectState>>,
    settings: &EngineSettings,
    registries: &EngineRegistries,
    targeted: Option<&RaycastHit>,
    fps: f32,
    frame_ms: f32,
    chunks_loaded: usize,
    visible_chunks: u32,
    draw_calls: u32,
    gpu_name: &str,
    chat_messages: &[String],
    chat_input: Option<&str>,
    telemetry: HudTelemetry<'_>,
) {
    render_hud_in_rect(
        ctx,
        ctx.screen_rect(),
        player,
        player_state,
        settings,
        registries,
        targeted,
        fps,
        frame_ms,
        chunks_loaded,
        visible_chunks,
        draw_calls,
        gpu_name,
        chat_messages,
        chat_input,
        telemetry,
    );
}

/// Render a player's HUD strictly clipped and proportionally scaled inside `rect`
/// (supports full-window singleplayer, side-by-side / top-bottom split-screen, and Window 2).
#[allow(clippy::too_many_arguments)]
pub fn render_hud_in_rect(
    ctx: &egui::Context,
    rect: Rect,
    player: &PlayerController,
    player_state: &Arc<RwLock<ObjectState>>,
    settings: &EngineSettings,
    registries: &EngineRegistries,
    targeted: Option<&RaycastHit>,
    fps: f32,
    frame_ms: f32,
    chunks_loaded: usize,
    visible_chunks: u32,
    draw_calls: u32,
    gpu_name: &str,
    chat_messages: &[String],
    chat_input: Option<&str>,
    mut telemetry: HudTelemetry<'_>,
) {
    let painter = ctx
        .layer_painter(egui::LayerId::background())
        .with_clip_rect(rect);

    if telemetry.hurt_flash > 0.005 {
        let f = (telemetry.hurt_flash / 0.25).clamp(0.0, 1.0);
        let fill_a = (48.0 * f).round() as u8;
        let edge_a = (150.0 * f).round() as u8;
        painter.rect_filled(rect, 0.0, Color32::from_rgba_unmultiplied(215, 22, 22, fill_a));
        painter.rect_stroke(
            rect,
            0.0,
            Stroke::new(8.0, Color32::from_rgba_unmultiplied(230, 24, 24, edge_a)),
            egui::StrokeKind::Inside,
        );
    }
    let fit_factor = (rect.width() / 760.0)
        .min(rect.height() / 540.0)
        .clamp(0.42, 1.0);
    let scale = (settings.gui_scale * fit_factor).clamp(0.42, 2.0);

    // 1. Classic Minecraft Crosshair + Progressive Dig Ring
    let c = rect.center();
    let ch_sz = 9.0 * scale;
    let stroke = Stroke::new(2.0, Color32::from_rgba_unmultiplied(245, 245, 245, 220));
    painter.line_segment([c - Vec2::new(ch_sz, 0.0), c + Vec2::new(ch_sz, 0.0)], stroke);
    painter.line_segment([c - Vec2::new(0.0, ch_sz), c + Vec2::new(0.0, ch_sz)], stroke);

    if telemetry.dig_progress > 0.005 {
        let prog = telemetry.dig_progress.clamp(0.0, 1.0);
        let radius = 15.0 * scale;
        let segs = 28usize;
        let active_segs = ((segs as f32) * prog).ceil() as usize;
        for i in 0..segs {
            let a0 = -std::f32::consts::FRAC_PI_2 + (i as f32 / segs as f32) * std::f32::consts::TAU;
            let a1 = -std::f32::consts::FRAC_PI_2 + ((i + 1) as f32 / segs as f32) * std::f32::consts::TAU;
            let p0 = c + Vec2::new(a0.cos(), a0.sin()) * radius;
            let p1 = c + Vec2::new(a1.cos(), a1.sin()) * radius;
            let col = if i < active_segs {
                Color32::from_rgb(255, 220, 80)
            } else {
                Color32::from_rgba_unmultiplied(0, 0, 0, 120)
            };
            painter.line_segment([p0, p1], Stroke::new(2.5, col));
        }
        let bw = 34.0 * scale;
        let by = c.y + radius + 6.0;
        let bar_bg = Rect::from_min_size(Pos2::new(c.x - bw * 0.5, by), Vec2::new(bw, 4.0));
        painter.rect_filled(bar_bg, 0.0, Color32::from_rgba_unmultiplied(0, 0, 0, 175));
        painter.rect_filled(
            Rect::from_min_size(bar_bg.min, Vec2::new(bw * prog, 4.0)),
            0.0,
            Color32::from_rgb(128, 255, 64),
        );
    }

    // 2. 9-Slot Hotbar + Item Icons + Durability + Hearts / Armor / Hunger / XP
    let max_slot_for_width = ((rect.width() - 20.0) / 9.15).max(16.0);
    let slot_sz = (44.0 * scale).min(max_slot_for_width).clamp(16.0, 68.0);
    let bar_w = slot_sz * 9.0 + 4.0;
    let bar_x = (rect.center().x - bar_w * 0.5).round();
    let bar_y = (rect.bottom() - slot_sz - 12.0 * fit_factor).round();
    let hotbar_rect = Rect::from_min_size(Pos2::new(bar_x, bar_y), Vec2::new(bar_w, slot_sz + 4.0));

    painter.rect_filled(hotbar_rect, 0.0, Color32::from_rgba_unmultiplied(28, 28, 28, 215));
    painter.rect_stroke(hotbar_rect, 0.0, Stroke::new(2.0, Color32::from_rgb(74, 74, 74)), egui::StrokeKind::Inside);

    let ps = player_state.read();
    let mg = ps.meta.read();
    let main_inv = mg.inventories.get("main");
    let armor_inv = mg.inventories.get("armor");

    for slot in 0..9usize {
        let sx = bar_x + 2.0 + slot as f32 * slot_sz;
        let sy = bar_y + 2.0;
        let srect = Rect::from_min_size(Pos2::new(sx, sy), Vec2::splat(slot_sz));

        painter.rect_filled(srect.shrink(1.0), 0.0, Color32::from_rgba_unmultiplied(55, 55, 55, 130));
        painter.rect_stroke(
            srect.shrink(1.0),
            0.0,
            Stroke::new(1.5, Color32::from_rgb(95, 95, 95)),
            egui::StrokeKind::Inside,
        );

        if slot == player.selected_slot {
            painter.rect_stroke(
                srect.expand(1.5),
                0.0,
                Stroke::new(2.5, Color32::WHITE),
                egui::StrokeKind::Inside,
            );
        }

        if let Some(item) = main_inv.and_then(|v| v.get(slot)) {
            if !item.is_empty() {
                let tex_id = match (telemetry.icon_cache.as_deref_mut(), telemetry.atlas) {
                    (Some(cache), Some(atlas)) => cache.get_or_load_item(ctx, &item.name, atlas, registries),
                    (Some(cache), None) => cache.get_cached(&item.name),
                    _ => None,
                };
                paint_item_in_slot(&painter, srect, tex_id, &item.name, item.count, item.wear);
            }
        }
    }

    // Survival Status Bars geometry (XP bar, 9x9/16x16 pixel-art Hearts, Armor, Hunger)
    let icon_sz = (slot_sz * 0.41).round().clamp(11.0, 27.0);
    let icon_step = (icon_sz - (2.0 * scale).round()).max(9.0);
    let xp_y = (bar_y - 9.0 * fit_factor).round();
    let heart_y = (xp_y - icon_sz - 3.0 * fit_factor).round();
    let armor_y = (heart_y - icon_sz - 2.0 * fit_factor).round();

    let equipped_armor = armor_inv
        .map(|v| v.iter().filter(|s| !s.is_empty()).count() as i32 * 4)
        .unwrap_or(0);
    let armor_pts = telemetry.armor_points.max(equipped_armor).clamp(0, 20);

    // Active wield item title above hotbar / status bars (clean of Luanti & § escapes)
    let status_top_y = if player.gamemode == GameMode::Survival {
        if armor_pts > 0 { armor_y - 4.0 } else { heart_y - 4.0 }
    } else {
        bar_y - 10.0
    };
    if let Some(active) = main_inv.and_then(|v| v.get(player.selected_slot)) {
        if !active.is_empty() {
            let label = registries
                .get_item_def(&active.name)
                .map(|d| {
                    d.description
                        .lines()
                        .map(|l| strip_luanti_escapes(l).trim().to_string())
                        .find(|l| !l.is_empty())
                        .unwrap_or_else(|| strip_luanti_escapes(&d.name).trim().to_string())
                })
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| active.name.split(':').nth(1).unwrap_or(&active.name).replace('_', " "));
            mc_shadow_text(
                &painter,
                Pos2::new(rect.center().x, status_top_y - 3.0),
                Align2::CENTER_BOTTOM,
                &label,
                (13.0 * fit_factor.max(0.82)).clamp(10.5, 14.0),
                Color32::from_rgb(255, 255, 180),
            );
        }
    }

    // Survival Status Bars: XP Bar, Pixel-Art Hearts, Armor Icons, Hunger Drumsticks
    if player.gamemode == GameMode::Survival {
        let xp_rect = Rect::from_min_size(Pos2::new(bar_x, xp_y), Vec2::new(bar_w, 5.0));
        painter.rect_filled(xp_rect, 0.0, Color32::from_rgb(24, 24, 24));
        painter.rect_stroke(xp_rect, 0.0, Stroke::new(1.0, Color32::BLACK), egui::StrokeKind::Inside);
        let xp_fill = telemetry.xp_progress.clamp(0.0, 1.0);
        if xp_fill > 0.0 {
            painter.rect_filled(
                Rect::from_min_size(xp_rect.min + Vec2::splat(1.0), Vec2::new((bar_w - 2.0) * xp_fill, 3.0)),
                0.0,
                Color32::from_rgb(128, 255, 32),
            );
        }
        if telemetry.xp_level > 0 {
            mc_shadow_text(
                &painter,
                Pos2::new(rect.center().x, xp_y - 4.0),
                Align2::CENTER_BOTTOM,
                &telemetry.xp_level.to_string(),
                13.0,
                Color32::from_rgb(128, 255, 32),
            );
        }

        let fallback_hud;
        let hud_tex: &HudIconTextures = if let Some(cache) = telemetry.icon_cache.as_deref_mut() {
            cache.ensure_hud_icons(ctx, telemetry.atlas)
        } else {
            let id = egui::Id::new("cubex_default_hud_icons");
            let existing = ctx.data(|d| d.get_temp::<HudIconTextures>(id));
            let need_reload = match (&existing, telemetry.atlas.is_some()) {
                (None, _) => true,
                (Some(tex), true) if !tex.from_atlas => true,
                _ => false,
            };
            fallback_hud = if need_reload {
                // Must call load_texture outside ctx.data/ctx.data_mut; egui::Context's
                // internal RwLock is non-reentrant and deadlocks on recursive locking.
                let loaded = HudIconTextures::load(ctx, telemetry.atlas);
                ctx.data_mut(|d| d.insert_temp(id, loaded.clone()));
                loaded
            } else {
                existing.unwrap()
            };
            &fallback_hud
        };

        let full_uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
        let paint_status_icon = |r: Rect, slot: i32, bg_id, half_id, full_id| {
            painter.image(bg_id, r, full_uv, Color32::WHITE);
            match slot {
                2 => {
                    painter.image(full_id, r, full_uv, Color32::WHITE);
                }
                1 => {
                    painter.image(half_id, r, full_uv, Color32::WHITE);
                }
                _ => {}
            }
        };

        for i in 0..10 {
            let hx = (bar_x + i as f32 * icon_step).round();
            let r = Rect::from_min_size(Pos2::new(hx, heart_y), Vec2::splat(icon_sz));
            let hp_for_slot = (player.hp - i * 2).clamp(0, 2);
            paint_status_icon(
                r,
                hp_for_slot,
                hud_tex.heart_bg.id(),
                hud_tex.heart_half.id(),
                hud_tex.heart_full.id(),
            );
        }

        if armor_pts > 0 {
            for i in 0..10 {
                let ax = (bar_x + i as f32 * icon_step).round();
                let r = Rect::from_min_size(Pos2::new(ax, armor_y), Vec2::splat(icon_sz));
                let ap_slot = (armor_pts - i * 2).clamp(0, 2);
                paint_status_icon(
                    r,
                    ap_slot,
                    hud_tex.armor_bg.id(),
                    hud_tex.armor_half.id(),
                    hud_tex.armor_full.id(),
                );
            }
        }

        for i in 0..10 {
            let dx = (bar_x + bar_w - icon_sz - (9 - i) as f32 * icon_step).round();
            let r = Rect::from_min_size(Pos2::new(dx, heart_y), Vec2::splat(icon_sz));
            let h_slot = (player.hunger - (9 - i) * 2).clamp(0, 2);
            paint_status_icon(
                r,
                h_slot,
                hud_tex.hunger_bg.id(),
                hud_tex.hunger_half.id(),
                hud_tex.hunger_full.id(),
            );
        }
    }

    // Co-op corner player badge (P1 / P2) when active
    if let Some(badge) = telemetry.player_badge {
        let badge_w = 46.0;
        let badge_r = Rect::from_min_size(
            Pos2::new(rect.right() - badge_w - 10.0, rect.top() + 10.0),
            Vec2::new(badge_w, 24.0),
        );
        let accent = if badge == "P2" {
            Color32::from_rgb(90, 210, 255)
        } else {
            Color32::from_rgb(100, 255, 110)
        };
        painter.rect_filled(badge_r, 2.0, Color32::from_rgba_unmultiplied(15, 15, 18, 190));
        painter.rect_stroke(badge_r, 2.0, Stroke::new(1.5, accent), egui::StrokeKind::Inside);
        mc_shadow_text(&painter, badge_r.center(), Align2::CENTER_CENTER, badge, 13.0, accent);
    }

    // 3. Coordinates Mini-HUD (when F3 is off and coords_hud is on)
    if settings.coords_hud && !settings.f3_enabled {
        let coord_txt = format!(
            "XYZ: {} / {} / {}   [{}]",
            player.pos.x.floor() as i32,
            player.pos.y.floor() as i32,
            player.pos.z.floor() as i32,
            player.gamemode.label()
        );
        let box_w = (320.0 * fit_factor.max(0.78)).min((rect.width() - 68.0).max(160.0));
        let bg_r = Rect::from_min_size(Pos2::new(rect.left() + 10.0, rect.top() + 10.0), Vec2::new(box_w, 24.0));
        painter.rect_filled(bg_r, 0.0, Color32::from_rgba_unmultiplied(0, 0, 0, 140));
        mc_shadow_text(
            &painter,
            Pos2::new(bg_r.left() + 6.0, bg_r.center().y),
            Align2::LEFT_CENTER,
            &coord_txt,
            (12.5 * fit_factor.max(0.84)).clamp(10.5, 13.5),
            Color32::from_rgb(255, 255, 160),
        );
    }

    // 4. Complete Authentic F3 Debug Overlay
    if settings.f3_enabled {
        let cx = (player.pos.x / 16.0).floor() as i32;
        let cy = (player.pos.y / 16.0).floor() as i32;
        let cz = (player.pos.z / 16.0).floor() as i32;
        let culled_chunks = chunks_loaded.saturating_sub(visible_chunks as usize);

        let biome_lbl = telemetry
            .biome_name
            .map(ToString::to_string)
            .unwrap_or_else(|| resolve_biome_label(player.pos.y, registries));

        let target_line = if let Some(hit) = targeted {
            let tname = registries.get_name_from_content_id(hit.node_id);
            let (p1, p2) = if let Some(w) = telemetry.world {
                let (_, p1, p2) = w.get_node(hit.pos);
                (Some(p1), telemetry.targeted_param2.unwrap_or(p2))
            } else {
                (None, telemetry.targeted_param2.unwrap_or(0))
            };
            if let Some(light) = p1 {
                format!(
                    "Targeted Node: {} @ ({}, {}, {}) param2={} [sky:{} blk:{}]",
                    tname, hit.pos.x, hit.pos.y, hit.pos.z, p2, light & 0x0F, light >> 4
                )
            } else {
                format!("Targeted Node: {} @ ({}, {}, {}) param2={}", tname, hit.pos.x, hit.pos.y, hit.pos.z, p2)
            }
        } else {
            "Targeted Node: none".into()
        };

        let yaw_norm = (player.yaw.to_degrees() + 180.0).rem_euclid(360.0) - 180.0;
        let pitch_norm = player.pitch.to_degrees().clamp(-90.0, 90.0);
        let lines = [
            format!("Cubex-RS 0.1.0 (Mineclonia + wgpu Vulkan + LuaJIT)"),
            format!("{:.0} fps ({:.2} ms) • Render Dist: {} chunks • Graphics: {}", fps, frame_ms, settings.render_distance_chunks, settings.graphics.as_str()),
            format!("XYZ: {:.3} / {:.3} / {:.3}  (Yaw {:.1} Pitch {:.1})", player.pos.x, player.pos.y, player.pos.z, yaw_norm, pitch_norm),
            format!("Chunk: {} {} {}  •  Facing: {}", cx, cy, cz, player.facing_label()),
            format!("Biome: {}  •  Gamemode: {}  •  Difficulty: {}", biome_lbl, player.gamemode.label(), settings.difficulty.label()),
            format!("Chunks Loaded: {}  Visible: {}  Culled: {}  Draw Calls: {}", chunks_loaded, visible_chunks, culled_chunks, draw_calls),
            target_line,
            format!("GPU: {}", gpu_name),
        ];

        let line_step = (22.0 * fit_factor.max(0.8)).clamp(17.0, 22.0);
        let font_sz = (13.5 * fit_factor.max(0.8)).clamp(10.5, 13.5);
        for (idx, line) in lines.iter().enumerate() {
            let y = rect.top() + 10.0 + idx as f32 * line_step;
            let w = (line.len() as f32 * (font_sz * 0.62) + 14.0).min((rect.width() - 16.0).max(120.0));
            painter.rect_filled(
                Rect::from_min_size(Pos2::new(rect.left() + 8.0, y), Vec2::new(w, line_step - 2.0)),
                0.0,
                Color32::from_rgba_unmultiplied(20, 20, 20, 170),
            );
            mc_shadow_text(
                &painter,
                Pos2::new(rect.left() + 14.0, y + (line_step - 2.0) * 0.5),
                Align2::LEFT_CENTER,
                line,
                font_sz,
                Color32::WHITE,
            );
        }
    }

    // 5. Chat feed (bottom-left of this viewport) with § / Luanti color formatting
    let start = chat_messages.len().saturating_sub(if rect.height() < 460.0 { 4 } else { 8 });
    let vis_count = chat_messages.len().saturating_sub(start);
    let chat_base_y = (bar_y - 54.0 * fit_factor - vis_count as f32 * 19.0).max(rect.top() + 44.0);
    for (i, msg) in chat_messages[start..].iter().enumerate() {
        let y = chat_base_y + i as f32 * 19.0;
        let bg_w = (rect.width() * 0.58)
            .clamp(220.0, 560.0)
            .min((rect.width() - 20.0).max(120.0));
        painter.rect_filled(
            Rect::from_min_size(Pos2::new(rect.left() + 10.0, y), Vec2::new(bg_w, 18.0)),
            0.0,
            Color32::from_rgba_unmultiplied(0, 0, 0, 135),
        );
        let job = motd_to_layout_job_with_default(msg, 12.5, [255, 255, 255]);
        let galley = ctx.fonts(|f| f.layout_job(job));
        painter.galley(Pos2::new(rect.left() + 15.0, y + 2.0), galley, Color32::WHITE);
    }

    if let Some(cmd) = chat_input {
        let r = Rect::from_min_size(
            Pos2::new(rect.left() + 10.0, rect.bottom() - 42.0),
            Vec2::new((rect.width() - 20.0).max(120.0), 30.0),
        );
        painter.rect_filled(r, 0.0, Color32::from_rgba_unmultiplied(0, 0, 0, 205));
        painter.rect_stroke(r, 0.0, Stroke::new(1.5, Color32::from_rgb(160, 160, 160)), egui::StrokeKind::Inside);
        mc_shadow_text(
            &painter,
            Pos2::new(r.left() + 8.0, r.center().y),
            Align2::LEFT_CENTER,
            &format!("> {cmd}_"),
            13.5,
            Color32::WHITE,
        );
    }
}

fn resolve_biome_label(y: f32, registries: &EngineRegistries) -> String {
    let yi = y.floor() as i32;
    for b in &registries.biomes {
        if yi >= b.y_min && yi <= b.y_max {
            return b.name.clone();
        }
    }
    "Plains".into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lua_api::userdata::ItemStackData;
    use crate::registry::ItemDef;
    use crate::ui::theme::{build_hud_icon_images, derive_bg_silhouette, synthesize_half_icon};
    use glam::Vec3;
    use image::{Rgba, RgbaImage};
    use std::collections::HashMap;
    use std::path::Path;

    #[test]
    fn strip_luanti_escapes_handles_translation_color_and_section_codes() {
        assert_eq!(
            strip_luanti_escapes("\x1b(T@mcl_tools)Stone Axe\x1bE"),
            "Stone Axe"
        );
        assert_eq!(
            strip_luanti_escapes("\x1b(c@#55FF55)\x1b(T@mcl_tools)Diamond Pickaxe\x1bE\x1bF"),
            "Diamond Pickaxe"
        );
        assert_eq!(
            strip_luanti_escapes("\x1b(F@1)\x1bTEnchanted Book\x1bE"),
            "Enchanted Book"
        );
        assert_eq!(
            strip_luanti_escapes("§a§lEmerald §r§7- §eGem"),
            "Emerald - Gem"
        );
        assert_eq!(strip_luanti_escapes("Bow & Arrow"), "Bow & Arrow");
    }

    #[test]
    fn hud_icon_synthesis_and_atlas_lookup_produce_crisp_icons() {
        // 1. Minimal fallback (no atlas) produces 9x9 pixel-art fallbacks
        let fb = build_hud_icon_images(None);
        assert_eq!(fb.heart_full.dimensions(), (9, 9));
        assert_eq!(fb.heart_half.dimensions(), (9, 9));
        assert_eq!(fb.heart_bg.dimensions(), (9, 9));
        assert_eq!(fb.hunger_full.dimensions(), (9, 9));
        assert_eq!(fb.hunger_half.dimensions(), (9, 9));
        assert_eq!(fb.hunger_bg.dimensions(), (9, 9));
        assert_eq!(fb.armor_full.dimensions(), (9, 9));
        assert_eq!(fb.armor_half.dimensions(), (9, 9));
        assert_eq!(fb.armor_bg.dimensions(), (9, 9));
        // Left half of fallback heart matches full heart on left, bg on right
        assert_eq!(fb.heart_half.get_pixel(2, 4), fb.heart_full.get_pixel(2, 4));
        assert_eq!(fb.heart_half.get_pixel(6, 4), fb.heart_bg.get_pixel(6, 4));
        // Right half of fallback hunger matches full hunger on right, bg on left
        assert_eq!(fb.hunger_half.get_pixel(7, 7), fb.hunger_full.get_pixel(7, 7));
        assert_eq!(fb.hunger_half.get_pixel(2, 2), fb.hunger_bg.get_pixel(2, 2));

        // 2. Atlas with 16x16 heart.png (missing heart_bg.png & heart_half.png) derives matching 16x16 bg & half
        let reg = EngineRegistries::new();
        let mut atlas = TextureAtlas::build(&HashMap::new(), Path::new(""), None, &reg);
        let mut heart16 = RgbaImage::from_pixel(16, 16, Rgba([0, 0, 0, 0]));
        for y in 2..14 {
            for x in 2..14 {
                heart16.put_pixel(x, y, Rgba([220, 40, 40, 255]));
            }
        }
        atlas.raw_cache.insert("heart.png".into(), heart16);
        let imgs = build_hud_icon_images(Some(&atlas));
        assert_eq!(imgs.heart_full.dimensions(), (16, 16));
        assert_eq!(imgs.heart_bg.dimensions(), (16, 16));
        assert_eq!(imgs.heart_half.dimensions(), (16, 16));
        // Edge of silhouette is dark charcoal outline, center-right of half heart is dark interior bg
        assert!(imgs.heart_bg.get_pixel(2, 2).0[0] < 32);
        assert!(imgs.heart_half.get_pixel(10, 8).0[0] < 60);
        assert!(imgs.heart_half.get_pixel(5, 8).0[0] > 180);

        let bg_test = derive_bg_silhouette(&imgs.heart_full);
        let half_test = synthesize_half_icon(&imgs.heart_full, &bg_test, false);
        assert_eq!(half_test.dimensions(), (16, 16));
    }

    #[test]
    fn hud_renders_pixel_icons_and_clean_wield_label() {
        let mut reg = EngineRegistries::new();
        reg.register_item(ItemDef {
            name: "mcl_tools:axe_stone".into(),
            description: "\x1b(T@mcl_tools)Stone Axe\x1bE\n\x1b(c@#AAAAAA)Tool".into(),
            item_type: "tool".into(),
            ..Default::default()
        });
        let atlas = TextureAtlas::build(&HashMap::new(), Path::new(""), None, &reg);
        let mut cache = ItemIconCache::new();

        let mut player = PlayerController::default();
        player.hp = 15;
        player.hunger = 13;
        player.gamemode = GameMode::Survival;
        player.selected_slot = 0;

        let ps = Arc::new(RwLock::new(ObjectState::new_player("test")));
        {
            let p = ps.write();
            let mut m = p.meta.write();
            let mut inv = vec![ItemStackData::default(); 36];
            inv[0] = ItemStackData::from_spec("mcl_tools:axe_stone 1");
            m.inventories.insert("main".into(), inv);
        }

        let ctx = egui::Context::default();
        let mut raw = egui::RawInput::default();
        raw.screen_rect = Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 720.0)));
        let settings = EngineSettings::default();

        let output = ctx.run(raw, |c| {
            render_hud_full(
                c,
                &player,
                &ps,
                &settings,
                &reg,
                None,
                60.0,
                16.6,
                10,
                10,
                10,
                "TestGPU",
                &[],
                None,
                HudTelemetry {
                    atlas: Some(&atlas),
                    icon_cache: Some(&mut cache),
                    armor_points: 11,
                    xp_level: 3,
                    xp_progress: 0.5,
                    ..Default::default()
                },
            );
        });

        assert!(cache.hud_icons.is_some());
        let mut saw_clean_label = false;
        let mut saw_raw_escape = false;
        for clipped in &output.shapes {
            if let egui::epaint::Shape::Text(t) = &clipped.shape {
                let txt = t.galley.text();
                if txt == "Stone Axe" {
                    saw_clean_label = true;
                }
                if txt.contains('\x1b') || txt.contains("T@mcl_tools") {
                    saw_raw_escape = true;
                }
            }
        }
        assert!(saw_clean_label, "Expected clean 'Stone Axe' label in HUD");
        assert!(!saw_raw_escape, "HUD label must not contain raw Luanti escape codes");
    }

    #[test]
    fn hud_heart_fallback_is_bright_red_when_missing_or_black() {
        let missing = build_hud_icon_images(None);
        // Center pixel of fallback 9x9 heart must be #E52525 bright red
        assert_eq!(missing.heart_full.get_pixel(4, 5).0, [229, 37, 37, 255]);

        let reg = EngineRegistries::new();
        let mut atlas = TextureAtlas::build(&HashMap::new(), Path::new(""), None, &reg);
        let black_heart = RgbaImage::from_pixel(9, 9, Rgba([0, 0, 0, 255]));
        atlas.raw_cache.insert("heart.png".into(), black_heart);
        let recovered = build_hud_icon_images(Some(&atlas));
        assert_eq!(
            recovered.heart_full.get_pixel(4, 5).0,
            [229, 37, 37, 255],
            "all-black heart.png must fall back to bright red #E52525"
        );
    }

    #[test]
    fn split_screen_huds_fit_within_left_and_right_viewports() {
        let reg = EngineRegistries::new();
        let p1 = PlayerController::default();
        let mut p2 = PlayerController::default();
        p2.pos = Vec3::new(12.0, 64.0, 4.0);
        let ps1 = Arc::new(RwLock::new(ObjectState::new_player("P1")));
        let ps2 = Arc::new(RwLock::new(ObjectState::new_player("P2")));
        let settings = EngineSettings::default();
        let ctx = egui::Context::default();
        let raw = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 720.0))),
            ..Default::default()
        };
        let mut cache = ItemIconCache::new();
        let output = ctx.run(raw, |c| {
            render_hud_in_rect(
                c,
                Rect::from_min_size(Pos2::ZERO, Vec2::new(640.0, 720.0)),
                &p1,
                &ps1,
                &settings,
                &reg,
                None,
                120.0,
                8.3,
                100,
                50,
                20,
                "GPU",
                &[],
                None,
                HudTelemetry {
                    icon_cache: Some(&mut cache),
                    player_badge: Some("P1"),
                    hurt_flash: 0.20,
                    ..Default::default()
                },
            );
            render_hud_in_rect(
                c,
                Rect::from_min_size(Pos2::new(640.0, 0.0), Vec2::new(640.0, 720.0)),
                &p2,
                &ps2,
                &settings,
                &reg,
                None,
                120.0,
                8.3,
                100,
                50,
                20,
                "GPU",
                &[],
                None,
                HudTelemetry {
                    icon_cache: Some(&mut cache),
                    player_badge: Some("P2"),
                    ..Default::default()
                },
            );
        });
        assert!(output.shapes.iter().any(|s| matches!(&s.shape, egui::epaint::Shape::Text(t) if t.galley.text() == "P1")));
        assert!(output.shapes.iter().any(|s| matches!(&s.shape, egui::epaint::Shape::Text(t) if t.galley.text() == "P2")));
        // Both Survival viewports emit matching textured status icons (hearts + hunger) inside their rects
        let mut left_imgs = 0usize;
        let mut right_imgs = 0usize;
        for clipped in &output.shapes {
            if let egui::epaint::Shape::Mesh(m) = &clipped.shape {
                if !m.vertices.is_empty() {
                    if clipped.clip_rect.center().x < 640.0 {
                        left_imgs += 1;
                    } else {
                        right_imgs += 1;
                    }
                }
            }
        }
        assert!(left_imgs > 0 && right_imgs > 0, "Expected Survival HUD status icons in both P1 ({left_imgs}) and P2 ({right_imgs}) viewports");

        // Also verify 4-player QuadSplit (1280x720 -> four 640x360 quadrants) fits cleanly without panic or bleed
        let p3 = PlayerController::default();
        let p4 = PlayerController::default();
        let ps3 = Arc::new(RwLock::new(ObjectState::new_player("P3")));
        let ps4 = Arc::new(RwLock::new(ObjectState::new_player("P4")));
        let q_rects = [
            (Rect::from_min_size(Pos2::ZERO, Vec2::new(640.0, 360.0)), "P1", &p1, &ps1),
            (Rect::from_min_size(Pos2::new(640.0, 0.0), Vec2::new(640.0, 360.0)), "P2", &p2, &ps2),
            (Rect::from_min_size(Pos2::new(0.0, 360.0), Vec2::new(640.0, 360.0)), "P3", &p3, &ps3),
            (Rect::from_min_size(Pos2::new(640.0, 360.0), Vec2::new(640.0, 360.0)), "P4", &p4, &ps4),
        ];
        let q_out = ctx.run(egui::RawInput::default(), |c| {
            for &(rect, badge, pc, ps) in &q_rects {
                render_hud_in_rect(
                    c,
                    rect,
                    pc,
                    ps,
                    &settings,
                    &reg,
                    None,
                    120.0,
                    8.3,
                    100,
                    50,
                    20,
                    "GPU",
                    &[],
                    None,
                    HudTelemetry {
                        icon_cache: Some(&mut cache),
                        player_badge: Some(badge),
                        ..Default::default()
                    },
                );
            }
        });
        for badge in ["P1", "P2", "P3", "P4"] {
            assert!(
                q_out
                    .shapes
                    .iter()
                    .any(|s| matches!(&s.shape, egui::epaint::Shape::Text(t) if t.galley.text() == badge)),
                "Missing badge {badge} in QuadSplit HUD"
            );
        }
    }
}
