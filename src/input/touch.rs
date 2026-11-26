use egui::{Pos2, Rect, Vec2};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TouchEventPhase {
    Started,
    Moved,
    Ended,
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TouchButton {
    Jump,
    Sneak,
    Attack,
    Place,
    Inventory,
    Pause,
    Camera,
}

#[derive(Debug, Clone)]
pub struct TouchHudLayout {
    pub viewport: Rect,
    pub joy_center: Pos2,
    pub joy_radius: f32,
    pub btn_jump: Rect,
    pub btn_sneak: Rect,
    pub btn_attack: Rect,
    pub btn_place: Rect,
    pub btn_inv: Rect,
    pub btn_pause: Rect,
    pub btn_cam: Rect,
    pub hotbar_slots: [Rect; 9],
}

impl TouchHudLayout {
    pub fn compute(viewport: Rect) -> Self {
        let fit = (viewport.width() / 760.0)
            .min(viewport.height() / 540.0)
            .clamp(0.42, 1.0);
        let joy_radius = (56.0 * fit).clamp(32.0, 70.0);
        let joy_margin = (26.0 * fit).clamp(12.0, 32.0);
        let joy_center = Pos2::new(
            viewport.left() + joy_margin + joy_radius,
            viewport.bottom() - joy_margin - joy_radius,
        );

        let bsz = (48.0 * fit).clamp(28.0, 60.0);
        let gap = (8.0 * fit).clamp(4.0, 12.0);
        let rx = viewport.right() - joy_margin - bsz;
        let by = viewport.bottom() - joy_margin - bsz;

        // Bottom-right ergonomic 2x2 diamond-like cluster
        let btn_jump = Rect::from_min_size(Pos2::new(rx, by - bsz - gap), Vec2::splat(bsz));
        let btn_sneak = Rect::from_min_size(Pos2::new(rx - bsz - gap, by), Vec2::splat(bsz));
        let btn_attack = Rect::from_min_size(Pos2::new(rx - bsz - gap, by - bsz - gap), Vec2::splat(bsz));
        let btn_place = Rect::from_min_size(Pos2::new(rx, by), Vec2::splat(bsz));

        // Top-right small utility cluster
        let usz = (34.0 * fit).clamp(22.0, 42.0);
        let uy = viewport.top() + (12.0 * fit).max(8.0);
        let btn_pause = Rect::from_min_size(Pos2::new(viewport.right() - 12.0 - usz, uy), Vec2::splat(usz));
        let btn_inv = Rect::from_min_size(Pos2::new(btn_pause.left() - gap - usz, uy), Vec2::splat(usz));
        let btn_cam = Rect::from_min_size(Pos2::new(btn_inv.left() - gap - usz, uy), Vec2::splat(usz));

        // Hotbar slots synchronized with hud.rs geometry
        let max_slot_w = ((viewport.width() - 20.0) / 9.15).max(16.0);
        let slot_sz = (44.0 * fit).min(max_slot_w).clamp(16.0, 68.0);
        let bar_w = slot_sz * 9.0 + 4.0;
        let bar_x = (viewport.center().x - bar_w * 0.5).round();
        let bar_y = (viewport.bottom() - slot_sz - 12.0 * fit).round();
        let mut hotbar_slots = [Rect::NOTHING; 9];
        for (i, r) in hotbar_slots.iter_mut().enumerate() {
            *r = Rect::from_min_size(
                Pos2::new(bar_x + 2.0 + i as f32 * slot_sz, bar_y + 2.0),
                Vec2::splat(slot_sz),
            );
        }

        Self {
            viewport,
            joy_center,
            joy_radius,
            btn_jump,
            btn_sneak,
            btn_attack,
            btn_place,
            btn_inv,
            btn_pause,
            btn_cam,
            hotbar_slots,
        }
    }

    pub fn button_at(&self, pos: Pos2) -> Option<TouchButton> {
        let checks = [
            (self.btn_jump, TouchButton::Jump),
            (self.btn_sneak, TouchButton::Sneak),
            (self.btn_attack, TouchButton::Attack),
            (self.btn_place, TouchButton::Place),
            (self.btn_inv, TouchButton::Inventory),
            (self.btn_pause, TouchButton::Pause),
            (self.btn_cam, TouchButton::Camera),
        ];
        checks
            .into_iter()
            .find(|(r, _)| r.expand(3.0).contains(pos))
            .map(|(_, b)| b)
    }

    pub fn hotbar_slot_at(&self, pos: Pos2) -> Option<usize> {
        self.hotbar_slots.iter().position(|r| r.contains(pos))
    }

    pub fn in_joystick(&self, pos: Pos2) -> bool {
        let hit_r = self.joy_radius * 1.45;
        pos.distance_sq(self.joy_center) <= hit_r * hit_r
    }
}

#[derive(Debug, Clone)]
enum TouchRole {
    Joystick { origin: Pos2, cur: Pos2 },
    Look { last: Pos2 },
    Button(TouchButton),
}

#[derive(Debug, Clone, Default)]
pub struct TouchFrameInput {
    pub move_x: f32,
    pub move_y: f32,
    pub sprint: bool,
    pub look_dx: f32,
    pub look_dy: f32,
    pub jump_held: bool,
    pub sneak_toggled: bool,
    pub attack_held: bool,
    pub attack_just_pressed: bool,
    pub place_just_pressed: bool,
    pub open_inv_just_pressed: bool,
    pub pause_just_pressed: bool,
    pub cycle_cam_just_pressed: bool,
    pub select_slot: Option<usize>,
}

#[derive(Debug, Default)]
pub struct TouchControlsState {
    active: HashMap<u64, TouchRole>,
    sneak_toggled: bool,
    pending_look_dx: f32,
    pending_look_dy: f32,
    pending_attack_just: bool,
    pending_place_just: bool,
    pending_inv_just: bool,
    pending_pause_just: bool,
    pending_cam_just: bool,
    pending_slot: Option<usize>,
    pub joy_knob_offset: Vec2,
    pub joy_active: bool,
}

impl TouchControlsState {
    pub fn on_touch_event(
        &mut self,
        finger_id: u64,
        phase: TouchEventPhase,
        pos: Pos2,
        viewport: Rect,
    ) {
        let layout = TouchHudLayout::compute(viewport);
        match phase {
            TouchEventPhase::Started => {
                if !viewport.contains(pos) {
                    return;
                }
                if let Some(slot) = layout.hotbar_slot_at(pos) {
                    self.pending_slot = Some(slot);
                    return;
                }
                if let Some(btn) = layout.button_at(pos) {
                    match btn {
                        TouchButton::Sneak => {
                            self.sneak_toggled = !self.sneak_toggled;
                        }
                        TouchButton::Attack => {
                            self.pending_attack_just = true;
                        }
                        TouchButton::Place => {
                            self.pending_place_just = true;
                        }
                        TouchButton::Inventory => {
                            self.pending_inv_just = true;
                        }
                        TouchButton::Pause => {
                            self.pending_pause_just = true;
                        }
                        TouchButton::Camera => {
                            self.pending_cam_just = true;
                        }
                        TouchButton::Jump => {}
                    }
                    self.active.insert(finger_id, TouchRole::Button(btn));
                } else if layout.in_joystick(pos) {
                    self.active.insert(
                        finger_id,
                        TouchRole::Joystick {
                            origin: layout.joy_center,
                            cur: pos,
                        },
                    );
                    self.recompute_joystick(&layout);
                } else {
                    self.active.insert(finger_id, TouchRole::Look { last: pos });
                }
            }
            TouchEventPhase::Moved => {
                let Some(role) = self.active.get_mut(&finger_id) else {
                    return;
                };
                match role {
                    TouchRole::Joystick { cur, .. } => {
                        *cur = pos;
                        self.recompute_joystick(&layout);
                    }
                    TouchRole::Look { last } => {
                        let d = pos - *last;
                        *last = pos;
                        self.pending_look_dx += d.x;
                        self.pending_look_dy += d.y;
                    }
                    TouchRole::Button(_) => {}
                }
            }
            TouchEventPhase::Ended | TouchEventPhase::Cancelled => {
                if let Some(TouchRole::Joystick { .. }) = self.active.remove(&finger_id) {
                    self.joy_knob_offset = Vec2::ZERO;
                    self.joy_active = false;
                }
            }
        }
    }

    fn recompute_joystick(&mut self, layout: &TouchHudLayout) {
        let mut off = Vec2::ZERO;
        let mut active = false;
        for role in self.active.values() {
            if let TouchRole::Joystick { origin, cur } = role {
                let d = *cur - *origin;
                let max_r = layout.joy_radius.max(1.0);
                let len = d.length();
                off = if len > max_r { d * (max_r / len) } else { d };
                active = true;
                break;
            }
        }
        self.joy_knob_offset = off;
        self.joy_active = active;
    }

    pub fn take_frame_input(&mut self, viewport: Rect) -> TouchFrameInput {
        let layout = TouchHudLayout::compute(viewport);
        let max_r = layout.joy_radius.max(1.0);
        let nx = (self.joy_knob_offset.x / max_r).clamp(-1.0, 1.0);
        // Screen Y down -> forward is negative dy => positive move_y
        let ny = (-self.joy_knob_offset.y / max_r).clamp(-1.0, 1.0);
        let mag = (nx * nx + ny * ny).sqrt();
        let (move_x, move_y) = if mag < 0.14 { (0.0, 0.0) } else { (nx, ny) };

        let jump_held = self
            .active
            .values()
            .any(|r| matches!(r, TouchRole::Button(TouchButton::Jump)));
        let attack_held = self
            .active
            .values()
            .any(|r| matches!(r, TouchRole::Button(TouchButton::Attack)));

        let out = TouchFrameInput {
            move_x,
            move_y,
            sprint: move_y > 0.85,
            look_dx: self.pending_look_dx,
            look_dy: self.pending_look_dy,
            jump_held,
            sneak_toggled: self.sneak_toggled,
            attack_held,
            attack_just_pressed: self.pending_attack_just,
            place_just_pressed: self.pending_place_just,
            open_inv_just_pressed: self.pending_inv_just,
            pause_just_pressed: self.pending_pause_just,
            cycle_cam_just_pressed: self.pending_cam_just,
            select_slot: self.pending_slot,
        };
        self.pending_look_dx = 0.0;
        self.pending_look_dy = 0.0;
        self.pending_attack_just = false;
        self.pending_place_just = false;
        self.pending_inv_just = false;
        self.pending_pause_just = false;
        self.pending_cam_just = false;
        self.pending_slot = None;
        out
    }

    pub fn is_button_held(&self, btn: TouchButton) -> bool {
        match btn {
            TouchButton::Sneak => self.sneak_toggled,
            _ => self
                .active
                .values()
                .any(|r| matches!(r, TouchRole::Button(b) if *b == btn)),
        }
    }

    pub fn reset_all(&mut self) {
        self.active.clear();
        self.joy_knob_offset = Vec2::ZERO;
        self.joy_active = false;
        self.pending_look_dx = 0.0;
        self.pending_look_dy = 0.0;
        self.pending_attack_just = false;
        self.pending_place_just = false;
        self.pending_inv_just = false;
        self.pending_pause_just = false;
        self.pending_cam_just = false;
        self.pending_slot = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn touch_joystick_drag_buttons_and_hotbar_mapping() {
        let vp = Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 720.0));
        let layout = TouchHudLayout::compute(vp);
        let mut state = TouchControlsState::default();

        // 1. Virtual joystick pushed straight up -> forward + sprint
        state.on_touch_event(1, TouchEventPhase::Started, layout.joy_center, vp);
        state.on_touch_event(
            1,
            TouchEventPhase::Moved,
            layout.joy_center + Vec2::new(0.0, -layout.joy_radius),
            vp,
        );
        // 2. Simultaneous right-side look finger drag + Jump press + hotbar tap on slot 3
        state.on_touch_event(2, TouchEventPhase::Started, Pos2::new(900.0, 300.0), vp);
        state.on_touch_event(2, TouchEventPhase::Moved, Pos2::new(945.0, 285.0), vp);
        state.on_touch_event(3, TouchEventPhase::Started, layout.btn_jump.center(), vp);
        state.on_touch_event(4, TouchEventPhase::Started, layout.btn_sneak.center(), vp);
        state.on_touch_event(5, TouchEventPhase::Started, layout.hotbar_slots[2].center(), vp);

        let f = state.take_frame_input(vp);
        assert!(f.move_y > 0.95 && f.sprint, "pushing joystick forward should sprint");
        assert!(f.move_x.abs() < 0.05);
        assert!((f.look_dx - 45.0).abs() < 0.01 && (f.look_dy + 15.0).abs() < 0.01);
        assert!(f.jump_held);
        assert!(f.sneak_toggled);
        assert_eq!(f.select_slot, Some(2));

        // Release finger 1 and 3 -> joystick returns to zero and jump stops
        state.on_touch_event(1, TouchEventPhase::Ended, layout.joy_center, vp);
        state.on_touch_event(3, TouchEventPhase::Ended, layout.btn_jump.center(), vp);
        let f2 = state.take_frame_input(vp);
        assert_eq!(f2.move_x, 0.0);
        assert_eq!(f2.move_y, 0.0);
        assert!(!f2.jump_held);
        assert!(f2.sneak_toggled, "sneak stays toggled until tapped again");
    }
}
