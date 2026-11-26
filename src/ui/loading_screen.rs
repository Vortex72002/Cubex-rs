use egui::{Align2, Color32, Pos2, Rect, Stroke, Vec2};

use crate::ui::theme::{mc_shadow_text, paint_dirt_screen_bg, McTheme};

/// Live snapshot rendered during asynchronous world creation, loading, and initial chunk meshing.
#[derive(Debug, Clone)]
pub struct LoadingScreenInfo {
    pub world_name: String,
    pub stage: String,
    pub detail: String,
    /// Normalized progress in `0.0..=1.0`.
    pub progress: f32,
    /// Radius around spawn/target chunk for the Minecraft-style column readiness grid (default `3` => `7x7`).
    pub grid_radius: i32,
    /// Row-major `((dz + r) * (2*r + 1) + (dx + r))` readiness flags for each chunk column.
    pub ready_columns: Vec<bool>,
}

impl Default for LoadingScreenInfo {
    fn default() -> Self {
        Self {
            world_name: "New World".to_string(),
            stage: "Preparing world...".to_string(),
            detail: String::new(),
            progress: 0.0,
            grid_radius: 3,
            ready_columns: vec![false; 49],
        }
    }
}

/// Render an authentic Minecraft dirt-tiled world loading screen with a spawn chunk grid and progress bar.
pub fn render_loading_screen(
    ctx: &egui::Context,
    theme: &McTheme,
    info: &LoadingScreenInfo,
    time_sec: f32,
) {
    let screen = ctx.screen_rect();
    let progress = info.progress.clamp(0.0, 1.0);

    egui::CentralPanel::default()
        .frame(egui::Frame::NONE)
        .show(ctx, |ui| {
            let p = ui.painter();
            paint_dirt_screen_bg(p, screen, theme, false, 0.0, 0.0);

            let cx = screen.center().x;
            let cy = screen.center().y;
            let scale = (screen.height() / 720.0).clamp(0.85, 1.65);

            // Spawn-chunk column grid (Minecraft-style square minimap of generated/meshed columns)
            let side_cells = (info.grid_radius.max(1) * 2 + 1) as usize;
            let cell_sz = (14.0 * scale).round().max(10.0);
            let gap = 2.0;
            let map_side = side_cells as f32 * cell_sz + (side_cells.saturating_sub(1)) as f32 * gap;
            let map_top = cy - map_side * 0.5 - 56.0 * scale;
            let map_left = cx - map_side * 0.5;
            let map_rect = Rect::from_min_size(
                Pos2::new(map_left - 6.0, map_top - 6.0),
                Vec2::new(map_side + 12.0, map_side + 12.0),
            );
            p.rect_filled(map_rect, 0.0, Color32::from_rgba_unmultiplied(12, 12, 12, 210));
            p.rect_stroke(
                map_rect,
                0.0,
                Stroke::new(2.0, Color32::from_rgb(72, 72, 72)),
                egui::StrokeKind::Inside,
            );

            let pulse = ((time_sec * 4.0).sin() * 0.5 + 0.5) * 14.0;
            let center_idx = info.grid_radius.max(1) as usize;
            for dz in 0..side_cells {
                for dx in 0..side_cells {
                    let idx = dz * side_cells + dx;
                    let ready = info.ready_columns.get(idx).copied().unwrap_or(false);
                    let x = map_left + dx as f32 * (cell_sz + gap);
                    let y = map_top + dz as f32 * (cell_sz + gap);
                    let cell = Rect::from_min_size(Pos2::new(x, y), Vec2::splat(cell_sz));
                    let is_center = dx == center_idx && dz == center_idx;
                    let fill = if ready {
                        if is_center {
                            Color32::from_rgb(130, 255, 130)
                        } else {
                            Color32::from_rgb(85, 224, 85)
                        }
                    } else if is_center {
                        let c = (74.0 + pulse) as u8;
                        Color32::from_rgb(c, c + 12, c)
                    } else {
                        let c = (40.0 + pulse * 0.4) as u8;
                        Color32::from_rgb(c, c, c + 4)
                    };
                    p.rect_filled(cell, 0.0, fill);
                }
            }

            // World title and current stage heading
            let title_y = map_rect.bottom() + 26.0 * scale;
            mc_shadow_text(
                p,
                Pos2::new(cx, title_y),
                Align2::CENTER_CENTER,
                &format!("Loading {}", info.world_name),
                (22.0 * scale).clamp(18.0, 30.0),
                Color32::WHITE,
            );
            mc_shadow_text(
                p,
                Pos2::new(cx, title_y + 30.0 * scale),
                Align2::CENTER_CENTER,
                &info.stage,
                (16.0 * scale).clamp(13.0, 22.0),
                Color32::from_rgb(220, 220, 220),
            );

            // Beveled Minecraft green progress bar (#55FF55)
            let bar_w = (screen.width() * 0.44).clamp(280.0, 520.0);
            let bar_h = (16.0 * scale).round().clamp(12.0, 22.0);
            let bar_rect = Rect::from_min_size(
                Pos2::new(cx - bar_w * 0.5, title_y + 52.0 * scale),
                Vec2::new(bar_w, bar_h),
            );
            p.rect_filled(bar_rect, 0.0, Color32::from_rgb(24, 24, 24));
            p.rect_stroke(
                bar_rect,
                0.0,
                Stroke::new(2.0, Color32::from_rgb(0, 0, 0)),
                egui::StrokeKind::Outside,
            );
            if progress > 0.0 {
                let fill_w = (bar_w - 4.0).max(1.0) * progress;
                let fill_rect = Rect::from_min_size(
                    Pos2::new(bar_rect.left() + 2.0, bar_rect.top() + 2.0),
                    Vec2::new(fill_w, (bar_h - 4.0).max(2.0)),
                );
                p.rect_filled(fill_rect, 0.0, Color32::from_rgb(85, 255, 85));
                let hi_rect = Rect::from_min_size(
                    fill_rect.min,
                    Vec2::new(fill_rect.width(), (fill_rect.height() * 0.28).max(1.0)),
                );
                p.rect_filled(hi_rect, 0.0, Color32::from_rgb(165, 255, 165));
            }

            let pct = (progress * 100.0).round() as u32;
            let footer = if info.detail.is_empty() {
                format!("{pct}%")
            } else {
                format!("{} ({pct}%)", info.detail)
            };
            mc_shadow_text(
                p,
                Pos2::new(cx, bar_rect.bottom() + 22.0 * scale),
                Align2::CENTER_CENTER,
                &footer,
                (14.0 * scale).clamp(12.0, 18.0),
                Color32::from_rgb(185, 185, 185),
            );
        });
}

#[cfg(test)]
mod loading_screen_tests {
    use super::*;

    #[test]
    fn loading_screen_renders_cleanly_at_all_standard_resolutions() {
        let ctx = egui::Context::default();
        let theme = McTheme::load(&ctx);
        let mut ready = vec![false; 49];
        for i in 0..25 {
            ready[i] = true;
        }
        let info = LoadingScreenInfo {
            world_name: "New World".to_string(),
            stage: "Generating terrain chunks...".to_string(),
            detail: "25 / 49 columns".to_string(),
            progress: 0.51,
            grid_radius: 3,
            ready_columns: ready,
        };
        for (w, h) in [(1280.0_f32, 720.0_f32), (1920.0, 1080.0), (2560.0, 1440.0)] {
            let input = egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(w, h))),
                ..Default::default()
            };
            let output = ctx.run(input, |ctx| {
                render_loading_screen(ctx, &theme, &info, 1.25);
            });
            assert!(!output.shapes.is_empty(), "expected shapes at {w}x{h}");
        }
    }
}
