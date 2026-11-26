use egui::{Align2, Color32, FontId, Pos2, Rect};

use crate::ui::theme::{mc_button_styled, mc_shadow_text, McTheme};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeathScreenAction {
    None,
    Respawn,
    MainMenu,
}

/// Render a polished translucent crimson-black death screen overlay inside `viewport`,
/// supporting both singleplayer and independent per-player split-screen viewports.
#[allow(clippy::too_many_arguments)]
pub fn render_death_screen(
    ctx: &egui::Context,
    theme: &McTheme,
    viewport: Rect,
    id_salt: u64,
    player_label: &str,
    death_cause: &str,
    xp_level: i32,
    show_main_menu: bool,
    controller_confirm: bool,
) -> DeathScreenAction {
    let mut action = DeathScreenAction::None;
    let vw = viewport.width().max(200.0);
    let vh = viewport.height().max(180.0);
    let cx = viewport.center().x;
    let cy = viewport.center().y;

    let title_size = (vw * 0.065).clamp(28.0, 48.0);
    let sub_size = (vw * 0.024).clamp(14.0, 20.0);
    let score_size = (vw * 0.022).clamp(13.0, 18.0);
    let btn_w = (vw * 0.42).clamp(180.0, 320.0);
    let btn_h = (vh * 0.095).clamp(36.0, 46.0);

    egui::Area::new(egui::Id::new(("cubex_death_screen", id_salt)))
        .order(egui::Order::Foreground)
        .fixed_pos(viewport.min)
        .movable(false)
        .constrain_to(viewport)
        .show(ctx, |ui| {
            // Keep backdrop non-blocking for foreground buttons while painting the overlay
            let _ = ui.interact(
                viewport,
                ui.id().with(("death_backdrop", id_salt)),
                egui::Sense::hover(),
            );
            let p = ui.painter().with_clip_rect(viewport);

            // Translucent crimson-black backdrop rgba(45, 5, 5, 0.78)
            p.rect_filled(viewport, 0.0, Color32::from_rgba_unmultiplied(45, 5, 5, 199));
            // Subtle darker top and bottom vignette framing
            let band_h = (vh * 0.16).clamp(18.0, 96.0);
            p.rect_filled(
                Rect::from_min_max(viewport.min, Pos2::new(viewport.right(), viewport.top() + band_h)),
                0.0,
                Color32::from_rgba_unmultiplied(24, 2, 2, 110),
            );
            p.rect_filled(
                Rect::from_min_max(Pos2::new(viewport.left(), viewport.bottom() - band_h), viewport.max),
                0.0,
                Color32::from_rgba_unmultiplied(24, 2, 2, 110),
            );

            // Bold centered "YOU DIED" title with deep drop shadow
            let title_pos = Pos2::new(cx, (cy - vh * 0.19).max(viewport.top() + 36.0));
            let font_title = FontId::proportional(title_size);
            p.text(
                title_pos + egui::vec2(3.0, 3.0),
                Align2::CENTER_CENTER,
                "YOU DIED",
                font_title.clone(),
                Color32::from_rgba_unmultiplied(18, 2, 2, 235),
            );
            p.text(
                title_pos,
                Align2::CENTER_CENTER,
                "YOU DIED",
                font_title,
                Color32::from_rgb(255, 84, 84),
            );

            // Death cause + score subtitles
            let cause_clean = if death_cause.trim().is_empty() {
                "Died"
            } else {
                death_cause.trim()
            };
            let cause_line = if player_label.is_empty() || player_label == "You" {
                cause_clean.to_string()
            } else {
                format!("{player_label}: {cause_clean}")
            };
            let cause_y = title_pos.y + title_size * 0.78 + 8.0;
            mc_shadow_text(
                &p,
                Pos2::new(cx, cause_y),
                Align2::CENTER_CENTER,
                &cause_line,
                sub_size,
                Color32::from_rgb(245, 236, 236),
            );
            let score_val = (xp_level.max(0) as u64) * 100;
            mc_shadow_text(
                &p,
                Pos2::new(cx, cause_y + sub_size + 8.0),
                Align2::CENTER_CENTER,
                &format!("Score: {score_val}"),
                score_size,
                Color32::from_rgb(255, 224, 128),
            );

            let first_btn_y = (cause_y + sub_size + score_size + 28.0).min(viewport.bottom() - btn_h * 2.4 - 16.0);
            let respawn_rect = Rect::from_min_size(
                Pos2::new(cx - btn_w * 0.5, first_btn_y),
                egui::vec2(btn_w, btn_h),
            );
            let mut resp = false;
            ui.allocate_new_ui(egui::UiBuilder::new().max_rect(respawn_rect), |ui| {
                resp = mc_button_styled(ui, theme, "Respawn", btn_w, btn_h, true).clicked();
            });
            if resp || controller_confirm {
                action = DeathScreenAction::Respawn;
            }

            if show_main_menu {
                let menu_rect = Rect::from_min_size(
                    Pos2::new(cx - btn_w * 0.5, first_btn_y + btn_h + 12.0),
                    egui::vec2(btn_w, btn_h),
                );
                let mut quit = false;
                ui.allocate_new_ui(egui::UiBuilder::new().max_rect(menu_rect), |ui| {
                    quit = mc_button_styled(ui, theme, "Main Menu", btn_w, btn_h, false).clicked();
                });
                if quit {
                    action = DeathScreenAction::MainMenu;
                }
            } else {
                mc_shadow_text(
                    &p,
                    Pos2::new(cx, first_btn_y + btn_h + 22.0),
                    Align2::CENTER_CENTER,
                    "Press [A / X] or Click Respawn",
                    13.0,
                    Color32::from_rgb(220, 200, 200),
                );
            }
        });

    action
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn death_screen_renders_cleanly_singleplayer_and_splitscreen() {
        let ctx = egui::Context::default();
        let theme = McTheme::load(&ctx);

        for (w, h, show_menu, confirm, expected) in [
            (1280.0, 720.0, true, false, DeathScreenAction::None),
            (640.0, 720.0, false, true, DeathScreenAction::Respawn),
            (640.0, 360.0, false, false, DeathScreenAction::None),
        ] {
            let mut raw = egui::RawInput::default();
            raw.screen_rect = Some(Rect::from_min_size(Pos2::ZERO, egui::vec2(w, h)));
            let mut got = DeathScreenAction::None;
            let out = ctx.run(raw, |ctx| {
                got = render_death_screen(
                    ctx,
                    &theme,
                    Rect::from_min_size(Pos2::ZERO, egui::vec2(w, h)),
                    if show_menu { 0 } else { 1 },
                    if show_menu { "You" } else { "Player2" },
                    "Slain in combat",
                    7,
                    show_menu,
                    confirm,
                );
            });
            assert_eq!(got, expected);
            assert!(!out.shapes.is_empty());
        }

        // Verify clicking the Respawn button directly via pointer returns DeathScreenAction::Respawn
        let vp = Rect::from_min_size(Pos2::ZERO, egui::vec2(1280.0, 720.0));
        let title_size = (1280.0f32 * 0.065).clamp(28.0, 48.0);
        let sub_size = (1280.0f32 * 0.024).clamp(14.0, 20.0);
        let score_size = (1280.0f32 * 0.022).clamp(13.0, 18.0);
        let btn_h = (720.0f32 * 0.095).clamp(36.0, 46.0);
        let title_y = (vp.center().y - 720.0 * 0.19).max(36.0);
        let cause_y = title_y + title_size * 0.78 + 8.0;
        let btn_y = (cause_y + sub_size + score_size + 28.0).min(720.0 - btn_h * 2.4 - 16.0);
        let btn_center = Pos2::new(vp.center().x, btn_y + btn_h * 0.5);

        // Render two initial warmup frames (egui's first frame is a sizing pass with invisible=true,
        // so enabled widget hit rects are populated at the end of frame 2), then press + release.
        for t in [0.016, 0.032] {
            let mut warmup = egui::RawInput::default();
            warmup.time = Some(t);
            warmup.screen_rect = Some(vp);
            warmup.events.push(egui::Event::PointerMoved(btn_center));
            let _ = ctx.run(warmup, |ctx| {
                let _ = render_death_screen(ctx, &theme, vp, 99, "You", "Slain in combat", 3, true, false);
            });
        }

        let mut click_action = DeathScreenAction::None;
        for (pressed, t) in [(true, 0.048), (false, 0.064)] {
            let mut raw = egui::RawInput::default();
            raw.time = Some(t);
            raw.screen_rect = Some(vp);
            raw.events.push(egui::Event::PointerMoved(btn_center));
            raw.events.push(egui::Event::PointerButton {
                pos: btn_center,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::NONE,
            });
            let _ = ctx.run(raw, |ctx| {
                click_action = render_death_screen(
                    ctx,
                    &theme,
                    vp,
                    99,
                    "You",
                    "Slain in combat",
                    3,
                    true,
                    false,
                );
            });
        }
        assert_eq!(click_action, DeathScreenAction::Respawn);
    }
}
