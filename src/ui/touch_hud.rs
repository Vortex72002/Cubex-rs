use egui::{Align2, Color32, FontId, Pos2, Rect, Stroke};

use crate::input::{TouchButton, TouchControlsState, TouchHudLayout};

/// Render crisp semi-transparent Minecraft-styled touch controls over the player's viewport.
pub fn render_touch_hud(ctx: &egui::Context, viewport: Rect, state: &TouchControlsState) {
    let layout = TouchHudLayout::compute(viewport);
    let painter = ctx
        .layer_painter(egui::LayerId::new(
            egui::Order::Foreground,
            egui::Id::new("cubex_touch_overlay"),
        ))
        .with_clip_rect(viewport);

    // 1. Virtual Movement Thumbstick (left bottom)
    let ring_col = if state.joy_active {
        Color32::from_rgba_unmultiplied(235, 235, 235, 195)
    } else {
        Color32::from_rgba_unmultiplied(190, 190, 190, 130)
    };
    painter.circle_filled(
        layout.joy_center,
        layout.joy_radius,
        Color32::from_rgba_unmultiplied(20, 20, 20, 95),
    );
    painter.circle_stroke(layout.joy_center, layout.joy_radius, Stroke::new(2.0, ring_col));

    let knob_pos = layout.joy_center + state.joy_knob_offset;
    let knob_r = (layout.joy_radius * 0.42).clamp(14.0, 28.0);
    painter.circle_filled(
        knob_pos,
        knob_r,
        Color32::from_rgba_unmultiplied(220, 220, 220, if state.joy_active { 215 } else { 155 }),
    );
    painter.circle_stroke(knob_pos, knob_r, Stroke::new(1.5, Color32::from_rgb(55, 55, 55)));

    // 2. Beveled Minecraft-styled touch buttons
    let paint_btn = |r: Rect, label: &str, active: bool, accent: Color32| {
        let fill = if active {
            Color32::from_rgba_unmultiplied(88, 168, 84, 195)
        } else {
            Color32::from_rgba_unmultiplied(36, 36, 38, 155)
        };
        painter.rect_filled(r, 4.0, fill);
        painter.rect_stroke(
            r,
            4.0,
            Stroke::new(1.8, if active { accent } else { Color32::from_rgb(120, 120, 125) }),
            egui::StrokeKind::Inside,
        );
        let font_sz = (r.height() * 0.34).clamp(10.0, 16.0);
        painter.text(
            r.center() + egui::Vec2::new(1.0, 1.0),
            Align2::CENTER_CENTER,
            label,
            FontId::monospace(font_sz),
            Color32::from_rgba_unmultiplied(0, 0, 0, 190),
        );
        painter.text(
            r.center(),
            Align2::CENTER_CENTER,
            label,
            FontId::monospace(font_sz),
            Color32::WHITE,
        );
    };

    paint_btn(layout.btn_jump, "JMP", state.is_button_held(TouchButton::Jump), Color32::LIGHT_GREEN);
    paint_btn(layout.btn_sneak, "SNK", state.is_button_held(TouchButton::Sneak), Color32::YELLOW);
    paint_btn(layout.btn_attack, "ATK", state.is_button_held(TouchButton::Attack), Color32::LIGHT_RED);
    paint_btn(layout.btn_place, "USE", state.is_button_held(TouchButton::Place), Color32::LIGHT_BLUE);
    paint_btn(layout.btn_inv, "INV", state.is_button_held(TouchButton::Inventory), Color32::WHITE);
    paint_btn(layout.btn_pause, "||", state.is_button_held(TouchButton::Pause), Color32::WHITE);
    paint_btn(layout.btn_cam, "CAM", state.is_button_held(TouchButton::Camera), Color32::WHITE);

    // Subtle hint label below top-right controls when idle
    let _ = Pos2::ZERO;
}
