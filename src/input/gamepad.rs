use gilrs::{Axis, Button, EventType, GamepadId, Gilrs};

/// Information about a currently connected controller mapped to a local player slot.
#[derive(Debug, Clone)]
pub struct ConnectedGamepad {
    pub slot: usize,
    pub id: GamepadId,
    pub name: String,
    pub is_dualsense: bool,
}

/// Per-frame sampled & edge-triggered state for one controller slot.
#[derive(Debug, Clone, Default)]
pub struct GamepadFrameInput {
    pub slot: usize,
    pub name: String,
    pub is_dualsense: bool,
    /// Left stick horizontal deflection (`-1.0` left .. `+1.0` right).
    pub move_x: f32,
    /// Left stick vertical deflection (`+1.0` forward .. `-1.0` backward).
    pub move_y: f32,
    /// Right stick look rate (`-1.0..=1.0`) after deadzone & response curve.
    pub look_x: f32,
    pub look_y: f32,
    // Held actions
    pub jump: bool,
    pub sneak_held: bool,
    pub mine_held: bool,
    pub place_held: bool,
    pub zoom: bool,
    // Rising-edge gameplay actions
    pub jump_pressed: bool,
    pub sneak_toggled: bool,
    pub sprint_toggled: bool,
    pub inventory_pressed: bool,
    pub camera_toggle_pressed: bool,
    pub hotbar_prev: bool,
    pub hotbar_next: bool,
    pub drop_pressed: bool,
    pub f3_pressed: bool,
    pub pause_pressed: bool,
    pub place_pressed: bool,
    // Menu / Formspec UI navigation
    pub ui_up: bool,
    pub ui_down: bool,
    pub ui_left: bool,
    pub ui_right: bool,
    pub ui_confirm: bool,
    pub ui_secondary: bool,
    pub ui_shift: bool,
    pub ui_cancel: bool,
}

#[derive(Debug, Default, Clone)]
struct PrevSlotState {
    jump: bool,
    sneak: bool,
    sprint: bool,
    inventory: bool,
    camera: bool,
    hotbar_prev: bool,
    hotbar_next: bool,
    drop: bool,
    f3: bool,
    pause: bool,
    place: bool,
    ui_dir: (i32, i32),
    ui_repeat_timer: f32,
    ui_confirm: bool,
    ui_secondary: bool,
    ui_shift: bool,
    ui_cancel: bool,
}

/// Apply radial scaled deadzone + power response curve so sticks center without drift.
pub fn apply_radial_deadzone(x: f32, y: f32, deadzone: f32, curve: f32) -> (f32, f32) {
    let dz = deadzone.clamp(0.0, 0.9);
    let mag = (x * x + y * y).sqrt();
    if mag <= dz || mag < 1e-6 {
        return (0.0, 0.0);
    }
    let scaled = ((mag - dz) / (1.0 - dz)).clamp(0.0, 1.0);
    let curved = scaled.powf(curve.max(0.5));
    let nx = (x / mag) * curved;
    let ny = (y / mag) * curved;
    (nx.clamp(-1.0, 1.0), ny.clamp(-1.0, 1.0))
}

/// Detect PlayStation DualSense / DualShock controller product strings exposed by Linux `hid-playstation` / SDL.
pub fn is_dualsense_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.contains("dualsense")
        || lower.contains("ps5")
        || lower.contains("dualshock")
        || (lower.contains("sony") && lower.contains("wireless controller"))
        || lower.contains("054c")
}

/// Cross-platform / Linux PS5 DualSense & standard gamepad manager backed by `gilrs`.
pub struct GamepadManager {
    gilrs: Option<Gilrs>,
    slots: Vec<Option<GamepadId>>,
    prev: Vec<PrevSlotState>,
    frames: Vec<GamepadFrameInput>,
    connected_cache: Vec<ConnectedGamepad>,
}

impl Default for GamepadManager {
    fn default() -> Self {
        Self::new()
    }
}

impl GamepadManager {
    /// Inert placeholder for instant frame-0 startup before udev/gilrs initialization.
    pub fn empty() -> Self {
        Self {
            gilrs: None,
            slots: Vec::new(),
            prev: Vec::new(),
            frames: Vec::new(),
            connected_cache: Vec::new(),
        }
    }

    pub fn new() -> Self {
        let gilrs = match Gilrs::new() {
            Ok(g) => Some(g),
            Err(e) => {
                tracing::debug!("Gamepad backend unavailable (running without controller daemon): {e}");
                None
            }
        };
        let mut mgr = Self {
            gilrs,
            slots: Vec::new(),
            prev: Vec::new(),
            frames: Vec::new(),
            connected_cache: Vec::new(),
        };
        mgr.sync_connected_slots();
        mgr
    }

    fn assign_slot(&mut self, id: GamepadId) -> usize {
        if let Some(pos) = self.slots.iter().position(|s| *s == Some(id)) {
            return pos;
        }
        if let Some(empty_idx) = self.slots.iter().position(|s| s.is_none()) {
            self.slots[empty_idx] = Some(id);
            return empty_idx;
        }
        let idx = self.slots.len();
        self.slots.push(Some(id));
        self.prev.push(PrevSlotState::default());
        self.frames.push(GamepadFrameInput::default());
        idx
    }

    fn sync_connected_slots(&mut self) {
        let Some(ref gilrs) = self.gilrs else {
            return;
        };
        let mut live: Vec<(GamepadId, String)> = gilrs
            .gamepads()
            .filter(|(_, gp)| gp.is_connected())
            .map(|(id, gp)| (id, gp.name().to_string()))
            .collect();
        live.sort_by_key(|(id, _)| usize::from(*id));

        for (i, slot) in self.slots.iter_mut().enumerate() {
            if let Some(id) = *slot {
                if !live.iter().any(|(lid, _)| *lid == id) {
                    *slot = None;
                    self.prev[i] = PrevSlotState::default();
                    self.frames[i] = GamepadFrameInput::default();
                }
            }
        }
        for (id, _) in &live {
            self.assign_slot(*id);
        }

        self.connected_cache.clear();
        for (slot_idx, slot_opt) in self.slots.iter().enumerate() {
            if let Some(id) = *slot_opt {
                if let Some((_, name)) = live.iter().find(|(lid, _)| *lid == id) {
                    self.connected_cache.push(ConnectedGamepad {
                        slot: slot_idx,
                        id,
                        name: name.clone(),
                        is_dualsense: is_dualsense_name(name),
                    });
                }
            }
        }
    }

    /// Drain controller events and sample all connected controllers for the current frame.
    pub fn poll(&mut self, deadzone: f32, dt: f32) {
        let mut hotplug = false;
        if let Some(ref mut gilrs) = self.gilrs {
            while let Some(ev) = gilrs.next_event() {
                match ev.event {
                    EventType::Connected | EventType::Disconnected => {
                        hotplug = true;
                    }
                    _ => {}
                }
            }
        }
        if hotplug {
            self.sync_connected_slots();
        } else {
            // Ensure cache names/connections stay fresh even without explicit event
            let needs_resync = if let Some(ref gilrs) = self.gilrs {
                let connected_count = gilrs.gamepads().filter(|(_, g)| g.is_connected()).count();
                connected_count != self.connected_cache.len()
            } else {
                false
            };
            if needs_resync {
                self.sync_connected_slots();
            }
        }

        let Some(ref gilrs) = self.gilrs else {
            return;
        };

        for slot_idx in 0..self.slots.len() {
            let Some(id) = self.slots[slot_idx] else {
                self.frames[slot_idx] = GamepadFrameInput::default();
                continue;
            };
            let Some(gp) = gilrs.connected_gamepad(id) else {
                continue;
            };

            let name = gp.name().to_string();
            let is_ds = is_dualsense_name(&name);

            let (move_x, move_y) = apply_radial_deadzone(
                gp.value(Axis::LeftStickX),
                gp.value(Axis::LeftStickY),
                deadzone,
                1.0,
            );
            let (look_x, look_y) = apply_radial_deadzone(
                gp.value(Axis::RightStickX),
                gp.value(Axis::RightStickY),
                deadzone,
                1.8,
            );

            let l2_val = gp
                .button_data(Button::LeftTrigger2)
                .map(|d| d.value())
                .unwrap_or(0.0)
                .max(gp.value(Axis::LeftZ));
            let r2_val = gp
                .button_data(Button::RightTrigger2)
                .map(|d| d.value())
                .unwrap_or(0.0)
                .max(gp.value(Axis::RightZ));

            let place_held = gp.is_pressed(Button::LeftTrigger2) || l2_val > 0.35;
            let mine_held = gp.is_pressed(Button::RightTrigger2) || r2_val > 0.35;

            let dpad_x = gp.value(Axis::DPadX);
            let dpad_y = gp.value(Axis::DPadY);
            let d_left = gp.is_pressed(Button::DPadLeft) || dpad_x < -0.5;
            let d_right = gp.is_pressed(Button::DPadRight) || dpad_x > 0.5;
            let d_up = gp.is_pressed(Button::DPadUp) || dpad_y > 0.5;
            let d_down = gp.is_pressed(Button::DPadDown) || dpad_y < -0.5;

            let jump_now = gp.is_pressed(Button::South);
            let sneak_now = gp.is_pressed(Button::East);
            let sprint_now = gp.is_pressed(Button::LeftThumb);
            let zoom_now = gp.is_pressed(Button::RightThumb);
            let inv_now = gp.is_pressed(Button::West);
            let cam_now = gp.is_pressed(Button::North);
            let prev_now = gp.is_pressed(Button::LeftTrigger) || d_left;
            let next_now = gp.is_pressed(Button::RightTrigger) || d_right;
            let drop_now = d_down;
            let f3_now = gp.is_pressed(Button::Select) || d_up;
            let pause_now = gp.is_pressed(Button::Start);

            // UI discrete direction from either D-pad or left stick
            let ui_dx = if d_right || move_x > 0.55 {
                1
            } else if d_left || move_x < -0.55 {
                -1
            } else {
                0
            };
            let ui_dy = if d_up || move_y > 0.55 {
                -1
            } else if d_down || move_y < -0.55 {
                1
            } else {
                0
            };

            let prev = &mut self.prev[slot_idx];
            let jump_pressed = jump_now && !prev.jump;
            let sneak_toggled = sneak_now && !prev.sneak;
            let sprint_toggled = sprint_now && !prev.sprint;
            let inventory_pressed = inv_now && !prev.inventory;
            let camera_toggle_pressed = cam_now && !prev.camera;
            let hotbar_prev = prev_now && !prev.hotbar_prev;
            let hotbar_next = next_now && !prev.hotbar_next;
            let drop_pressed = drop_now && !prev.drop;
            let f3_pressed = f3_now && !prev.f3;
            let pause_pressed = pause_now && !prev.pause;
            let place_pressed = place_held && !prev.place;

            let (ui_up, ui_down, ui_left, ui_right) = if ui_dx == 0 && ui_dy == 0 {
                prev.ui_dir = (0, 0);
                prev.ui_repeat_timer = 0.0;
                (false, false, false, false)
            } else if prev.ui_dir != (ui_dx, ui_dy) {
                prev.ui_dir = (ui_dx, ui_dy);
                prev.ui_repeat_timer = 0.32;
                (ui_dy < 0, ui_dy > 0, ui_dx < 0, ui_dx > 0)
            } else {
                prev.ui_repeat_timer -= dt;
                if prev.ui_repeat_timer <= 0.0 {
                    prev.ui_repeat_timer = 0.13;
                    (ui_dy < 0, ui_dy > 0, ui_dx < 0, ui_dx > 0)
                } else {
                    (false, false, false, false)
                }
            };

            let ui_confirm = jump_now && !prev.ui_confirm;
            let ui_secondary = inv_now && !prev.ui_secondary;
            let ui_shift = cam_now && !prev.ui_shift;
            let ui_cancel = sneak_now && !prev.ui_cancel;

            prev.jump = jump_now;
            prev.sneak = sneak_now;
            prev.sprint = sprint_now;
            prev.inventory = inv_now;
            prev.camera = cam_now;
            prev.hotbar_prev = prev_now;
            prev.hotbar_next = next_now;
            prev.drop = drop_now;
            prev.f3 = f3_now;
            prev.pause = pause_now;
            prev.place = place_held;
            prev.ui_confirm = jump_now;
            prev.ui_secondary = inv_now;
            prev.ui_shift = cam_now;
            prev.ui_cancel = sneak_now;

            self.frames[slot_idx] = GamepadFrameInput {
                slot: slot_idx,
                name,
                is_dualsense: is_ds,
                move_x,
                move_y,
                look_x,
                look_y,
                jump: jump_now,
                sneak_held: sneak_now,
                mine_held,
                place_held,
                zoom: zoom_now,
                jump_pressed,
                sneak_toggled,
                sprint_toggled,
                inventory_pressed,
                camera_toggle_pressed,
                hotbar_prev,
                hotbar_next,
                drop_pressed,
                f3_pressed,
                pause_pressed,
                place_pressed,
                ui_up,
                ui_down,
                ui_left,
                ui_right,
                ui_confirm,
                ui_secondary,
                ui_shift,
                ui_cancel,
            };
        }
    }

    pub fn connected_gamepads(&self) -> &[ConnectedGamepad] {
        &self.connected_cache
    }

    pub fn frame_for_slot(&self, slot: usize) -> Option<&GamepadFrameInput> {
        self.slots
            .get(slot)
            .and_then(|s| s.as_ref())
            .and_then(|_| self.frames.get(slot))
    }

    /// Returns the slot index of any connected controller NOT in `assigned_slots` whose `Options` (`Start`) button was just pressed.
    pub fn any_unassigned_start_pressed(&self, assigned_slots: &[usize]) -> Option<usize> {
        for gp in &self.connected_cache {
            if !assigned_slots.contains(&gp.slot) {
                if let Some(f) = self.frames.get(gp.slot) {
                    if f.pause_pressed {
                        return Some(gp.slot);
                    }
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radial_deadzone_filters_drift_and_preserves_direction() {
        assert_eq!(apply_radial_deadzone(0.05, -0.08, 0.14, 1.0), (0.0, 0.0));
        let (dx, dy) = apply_radial_deadzone(1.0, 0.0, 0.14, 1.8);
        assert!((dx - 1.0).abs() < 1e-4);
        assert!(dy.abs() < 1e-6);

        let (mx, my) = apply_radial_deadzone(0.57, 0.0, 0.14, 1.0);
        assert!((mx - 0.5).abs() < 1e-4);
        assert!(my.abs() < 1e-6);
    }

    #[test]
    fn dualsense_product_names_detected_cleanly() {
        assert!(is_dualsense_name("DualSense Wireless Controller"));
        assert!(is_dualsense_name("Sony Interactive Entertainment Wireless Controller"));
        assert!(is_dualsense_name("PS5 Controller (054c:0ce6)"));
        assert!(!is_dualsense_name("Xbox Series X Controller"));
    }

    #[test]
    fn gamepad_manager_initializes_and_polls_safely_headless() {
        let mut mgr = GamepadManager::new();
        mgr.poll(0.14, 0.016);
        let _ = mgr.connected_gamepads();
        assert!(mgr.any_unassigned_start_pressed(&[]).is_none());
    }
}
