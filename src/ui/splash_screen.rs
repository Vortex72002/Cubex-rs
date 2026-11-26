use egui::{Align2, Color32, FontId, Pos2, Rect, Stroke, Vec2};

/// Live progress snapshot displayed on the Vortex Studios launch splash screen.
#[derive(Debug, Clone)]
pub struct SplashScreenInfo {
    pub stage: String,
    pub detail: String,
    /// Normalized progress in `0.0..=1.0`.
    pub progress: f32,
}

impl Default for SplashScreenInfo {
    fn default() -> Self {
        Self {
            stage: "Starting Cubex Engine...".to_string(),
            detail: "Initializing Vortex Core".to_string(),
            progress: 0.04,
        }
    }
}

#[inline]
fn mul_alpha(c: Color32, alpha: f32) -> Color32 {
    let a = ((c.a() as f32) * alpha.clamp(0.0, 1.0)).round() as u8;
    Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), a)
}

#[inline]
fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    Color32::from_rgba_unmultiplied(
        (a.r() as f32 + (b.r() as f32 - a.r() as f32) * t).round() as u8,
        (a.g() as f32 + (b.g() as f32 - a.g() as f32) * t).round() as u8,
        (a.b() as f32 + (b.b() as f32 - a.b() as f32) * t).round() as u8,
        (a.a() as f32 + (b.a() as f32 - a.a() as f32) * t).round() as u8,
    )
}

fn draw_shadow_text(
    p: &egui::Painter,
    pos: Pos2,
    align: Align2,
    text: &str,
    size: f32,
    color: Color32,
    alpha: f32,
) {
    let font = FontId::proportional(size);
    let shadow_off = (size * 0.075).clamp(1.5, 3.5);
    p.text(
        pos + Vec2::splat(shadow_off),
        align,
        text,
        font.clone(),
        mul_alpha(Color32::from_rgba_unmultiplied(3, 6, 14, 220), alpha),
    );
    p.text(pos, align, text, font, mul_alpha(color, alpha));
}

/// Render the full-window animated **Vortex Studios** splash screen / cross-fade overlay.
/// `alpha` in `0.0..=1.0` allows smoothly dissolving out into the live 3D Title Screen panorama.
pub fn render_splash_screen(
    ctx: &egui::Context,
    info: &SplashScreenInfo,
    time_sec: f32,
    alpha: f32,
) {
    let a = alpha.clamp(0.0, 1.0);
    if a <= 0.005 {
        return;
    }

    let screen = ctx.screen_rect();
    let progress = info.progress.clamp(0.0, 1.0);
    let scale = (screen.height() / 720.0).min(screen.width() / 1120.0).clamp(0.65, 1.85);
    let cx = screen.center().x;
    let cy = screen.center().y;

    if a > 0.2 {
        egui::Area::new(egui::Id::new("cubex_vortex_splash_blocker"))
            .order(egui::Order::Middle)
            .fixed_pos(screen.min)
            .interactable(true)
            .show(ctx, |ui| {
                ui.allocate_exact_size(screen.size(), egui::Sense::click_and_drag());
            });
    }

    let p = &ctx
        .layer_painter(egui::LayerId::new(
            egui::Order::Foreground,
            egui::Id::new("cubex_vortex_splash"),
        ))
        .with_clip_rect(screen);
    {
            // 1. Deep obsidian backdrop + subtle atmospheric radial glow & architectural grid
            p.rect_filled(screen, 0.0, mul_alpha(Color32::from_rgb(9, 12, 21), a));

            let halo_center = Pos2::new(cx, cy - 82.0 * scale);
            for (r_mul, col) in [
                (260.0 * scale, Color32::from_rgba_unmultiplied(40, 72, 165, 22)),
                (170.0 * scale, Color32::from_rgba_unmultiplied(56, 200, 255, 28)),
                (96.0 * scale, Color32::from_rgba_unmultiplied(108, 99, 255, 36)),
            ] {
                p.circle_filled(halo_center, r_mul, mul_alpha(col, a));
            }

            let grid_step = (56.0 * scale).max(36.0);
            let grid_col = mul_alpha(Color32::from_rgba_unmultiplied(70, 96, 150, 18), a);
            let mut gx = (screen.left() / grid_step).floor() * grid_step;
            while gx <= screen.right() {
                p.line_segment(
                    [Pos2::new(gx, screen.top()), Pos2::new(gx, screen.bottom())],
                    Stroke::new(1.0, grid_col),
                );
                gx += grid_step;
            }
            let mut gy = (screen.top() / grid_step).floor() * grid_step;
            while gy <= screen.bottom() {
                p.line_segment(
                    [Pos2::new(screen.left(), gy), Pos2::new(screen.right(), gy)],
                    Stroke::new(1.0, grid_col),
                );
                gy += grid_step;
            }

            // 2. Animated Vortex Studios emblem: counter-rotating rings + spiraling vortex blades + voxel core
            let emblem_c = Pos2::new(cx, cy - 92.0 * scale);
            let outer_r = 68.0 * scale;
            let inner_r = 45.0 * scale;

            p.circle_stroke(
                emblem_c,
                outer_r + 10.0 * scale,
                Stroke::new(
                    1.5 * scale,
                    mul_alpha(Color32::from_rgba_unmultiplied(56, 232, 255, 55), a),
                ),
            );
            p.circle_stroke(
                emblem_c,
                inner_r,
                Stroke::new(
                    1.25 * scale,
                    mul_alpha(Color32::from_rgba_unmultiplied(108, 99, 255, 85), a),
                ),
            );

            let blade_count = 6usize;
            let base_rot = time_sec * 1.15;
            let cyan = Color32::from_rgb(56, 232, 255);
            let indigo = Color32::from_rgb(108, 99, 255);
            for i in 0..blade_count {
                let frac = i as f32 / blade_count as f32;
                let angle = base_rot + frac * std::f32::consts::TAU;
                let a_tip = angle + 0.38;
                let a_left = angle - 0.14;
                let a_right = angle + 0.16;
                let tip_r = outer_r * (0.96 + 0.05 * (time_sec * 3.0 + i as f32).sin());
                let p0 = emblem_c + Vec2::new(a_left.cos(), a_left.sin()) * (inner_r * 0.62);
                let p1 = emblem_c + Vec2::new(a_tip.cos(), a_tip.sin()) * tip_r;
                let p2 = emblem_c + Vec2::new(a_right.cos(), a_right.sin()) * (inner_r * 0.96);
                let blade_col = lerp_color(cyan, indigo, (i % 2) as f32 * 0.65);
                p.add(egui::Shape::convex_polygon(
                    vec![p0, p1, p2],
                    mul_alpha(
                        Color32::from_rgba_unmultiplied(blade_col.r(), blade_col.g(), blade_col.b(), 195),
                        a,
                    ),
                    Stroke::new(1.0, mul_alpha(Color32::from_rgb(215, 248, 255), a * 0.55)),
                ));
            }

            // Counter-rotating orbital nodes
            for j in 0..3usize {
                let orbit_a = -time_sec * 1.7 + (j as f32) * (std::f32::consts::TAU / 3.0);
                let orb_pos = emblem_c + Vec2::new(orbit_a.cos(), orbit_a.sin()) * (outer_r + 10.0 * scale);
                p.circle_filled(
                    orb_pos,
                    3.5 * scale,
                    mul_alpha(Color32::from_rgb(145, 245, 255), a),
                );
            }

            // Faceted isometric voxel cube at center of the Vortex
            let cube_r = 22.0 * scale;
            let top_pt = emblem_c + Vec2::new(0.0, -cube_r);
            let tr_pt = emblem_c + Vec2::new(cube_r * 0.866, -cube_r * 0.48);
            let br_pt = emblem_c + Vec2::new(cube_r * 0.866, cube_r * 0.48);
            let bot_pt = emblem_c + Vec2::new(0.0, cube_r);
            let bl_pt = emblem_c + Vec2::new(-cube_r * 0.866, cube_r * 0.48);
            let tl_pt = emblem_c + Vec2::new(-cube_r * 0.866, -cube_r * 0.48);
            let edge = Stroke::new(1.25 * scale, mul_alpha(Color32::from_rgb(230, 250, 255), a * 0.8));

            p.add(egui::Shape::convex_polygon(
                vec![top_pt, tr_pt, emblem_c, tl_pt],
                mul_alpha(Color32::from_rgb(92, 242, 255), a),
                edge,
            ));
            p.add(egui::Shape::convex_polygon(
                vec![tl_pt, emblem_c, bot_pt, bl_pt],
                mul_alpha(Color32::from_rgb(52, 165, 232), a),
                edge,
            ));
            p.add(egui::Shape::convex_polygon(
                vec![emblem_c, tr_pt, br_pt, bot_pt],
                mul_alpha(Color32::from_rgb(86, 82, 224), a),
                edge,
            ));

            // 3. Brand wordmark & subtitle
            let brand_y = emblem_c.y + outer_r + 42.0 * scale;
            draw_shadow_text(
                p,
                Pos2::new(cx, brand_y),
                Align2::CENTER_CENTER,
                "VORTEX STUDIOS",
                (44.0 * scale).clamp(28.0, 74.0),
                Color32::from_rgb(240, 248, 255),
                a,
            );
            draw_shadow_text(
                p,
                Pos2::new(cx, brand_y + 34.0 * scale),
                Align2::CENTER_CENTER,
                "CUBEX ENGINE  •  RUST + VULKAN",
                (14.5 * scale).clamp(11.0, 23.0),
                Color32::from_rgb(138, 220, 248),
                a * 0.92,
            );

            // 4. Sleek animated loading bar + status stage
            let bar_w = (screen.width() * 0.38).clamp(300.0, 560.0);
            let bar_h = (10.0 * scale).clamp(7.0, 16.0);
            let bar_y = (brand_y + 72.0 * scale).min(screen.bottom() - 88.0 * scale);
            let bar_rect = Rect::from_min_size(Pos2::new(cx - bar_w * 0.5, bar_y), Vec2::new(bar_w, bar_h));

            p.rect_filled(
                bar_rect,
                bar_h * 0.5,
                mul_alpha(Color32::from_rgb(20, 26, 44), a),
            );
            p.rect_stroke(
                bar_rect,
                bar_h * 0.5,
                Stroke::new(1.5, mul_alpha(Color32::from_rgb(64, 92, 148), a)),
                egui::StrokeKind::Outside,
            );

            if progress > 0.002 {
                let fill_w = (bar_w * progress).max(bar_h);
                let fill_rect = Rect::from_min_size(bar_rect.min, Vec2::new(fill_w.min(bar_w), bar_h));
                let bar_col = lerp_color(indigo, cyan, 0.35 + 0.65 * progress);
                p.rect_filled(fill_rect, bar_h * 0.5, mul_alpha(bar_col, a));
                let highlight = Rect::from_min_size(
                    fill_rect.min + Vec2::new(2.0, 1.5),
                    Vec2::new((fill_rect.width() - 4.0).max(1.0), (bar_h * 0.34).max(1.5)),
                );
                p.rect_filled(
                    highlight,
                    bar_h * 0.25,
                    mul_alpha(Color32::from_rgba_unmultiplied(235, 252, 255, 165), a),
                );
            }

            let pct = (progress * 100.0).round() as u32;
            let status_line = if info.detail.is_empty() {
                format!("{} ({pct}%)", info.stage)
            } else {
                format!("{} — {} ({pct}%)", info.stage, info.detail)
            };
            draw_shadow_text(
                p,
                Pos2::new(cx, bar_rect.bottom() + 22.0 * scale),
                Align2::CENTER_CENTER,
                &status_line,
                (13.5 * scale).clamp(11.0, 20.0),
                Color32::from_rgb(188, 204, 228),
                a,
            );
    }
}

#[cfg(test)]
mod splash_screen_tests {
    use super::*;

    #[test]
    fn vortex_splash_renders_cleanly_at_all_standard_resolutions() {
        let ctx = egui::Context::default();
        let info = SplashScreenInfo {
            stage: "Loading Mineclonia mods...".to_string(),
            detail: "120 / 224".to_string(),
            progress: 0.54,
        };
        for (w, h) in [(1280.0_f32, 720.0_f32), (1920.0, 1080.0), (2560.0, 1440.0)] {
            for alpha in [1.0_f32, 0.45_f32] {
                let input = egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(w, h))),
                    ..Default::default()
                };
                let output = ctx.run(input, |c| {
                    render_splash_screen(c, &info, 0.75, alpha);
                });
                assert!(!output.shapes.is_empty(), "expected shapes at {w}x{h} alpha={alpha}");
                let has_brand = output.shapes.iter().any(|s| {
                    matches!(&s.shape, egui::epaint::Shape::Text(t) if t.galley.text() == "VORTEX STUDIOS")
                });
                assert!(has_brand, "expected 'VORTEX STUDIOS' text at {w}x{h} alpha={alpha}");
            }
        }
    }
}
