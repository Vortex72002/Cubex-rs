use egui::{Align2, Color32, Pos2, Rect, Stroke, Vec2};
use glam::IVec3;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

use crate::lua_api::core_api::resolve_craft_recipe;
use crate::lua_api::userdata::{ItemStackData, MetaDataState, ObjectState};
use crate::registry::EngineRegistries;
use crate::textures::TextureAtlas;
use crate::ui::theme::{
    mc_button, mc_shadow_text, paint_item_in_slot, paint_mc_panel, paint_mc_slot,
    paint_mc_tooltip, strip_luanti_escapes, ItemIconCache, McTheme,
};
use crate::world::WorldStore;

#[derive(Debug, Clone)]
pub enum FormElement {
    Label { x: f32, y: f32, text: String },
    VertLabel { x: f32, y: f32, text: String },
    Button { x: f32, y: f32, w: f32, h: f32, name: String, label: String, exit: bool },
    ImageButton { x: f32, y: f32, w: f32, h: f32, texture: String, name: String, label: String, exit: bool },
    ItemImageButton { x: f32, y: f32, w: f32, h: f32, item_name: String, name: String, label: String },
    Field { x: f32, y: f32, w: f32, h: f32, name: String, label: String, default: String },
    PwdField { x: f32, y: f32, w: f32, h: f32, name: String, label: String },
    TextArea { x: f32, y: f32, w: f32, h: f32, name: String, label: String, default: String },
    InvList { loc: String, list: String, x: f32, y: f32, w: usize, h: usize, start: usize },
    ListRing { loc: String, list: String },
    Box { x: f32, y: f32, w: f32, h: f32, color: Color32 },
    Image { x: f32, y: f32, w: f32, h: f32, texture: String },
    ItemImage { x: f32, y: f32, w: f32, h: f32, item_name: String },
    Checkbox { x: f32, y: f32, name: String, label: String, selected: bool },
    Dropdown {
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        name: String,
        items: Vec<String>,
        selected_idx: usize,
        index_event: bool,
    },
    TabHeader {
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        name: String,
        captions: Vec<String>,
        current_tab: usize,
    },
    Tooltip {
        target: Option<String>,
        rect: Option<[f32; 4]>,
        text: String,
    },
    BgColor {
        color: Color32,
        fullscreen: bool,
        fullscreen_color: Option<Color32>,
    },
}

#[derive(Debug, Clone)]
pub struct ParsedFormspec {
    pub version: u32,
    pub size: [f32; 2],
    pub elements: Vec<FormElement>,
}

impl Default for ParsedFormspec {
    fn default() -> Self {
        Self {
            version: 4,
            size: [11.75, 9.6],
            elements: Vec::new(),
        }
    }
}

/// Cursor holding an item stack currently picked up by the player inside an inventory UI.
#[derive(Debug, Clone, Default)]
pub struct InventoryCursor {
    pub held: ItemStackData,
}

impl InventoryCursor {
    pub fn is_empty(&self) -> bool {
        self.held.is_empty()
    }

    pub fn clear(&mut self) {
        self.held = ItemStackData::default();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotClickButton {
    Left,
    Right,
}

/// Handle a left or right mouse click on an inventory slot, updating `cursor` and `slots`.
/// Returns `true` if the inventory state changed (for `craftpreview`, returns `true` when output was taken).
pub fn handle_slot_click(
    cursor: &mut InventoryCursor,
    slots: &mut Vec<ItemStackData>,
    slot_idx: usize,
    button: SlotClickButton,
    output_only: bool,
    max_stack: u16,
) -> bool {
    if slot_idx >= 256 {
        return false;
    }
    if slot_idx >= slots.len() {
        slots.resize_with(slot_idx + 1, ItemStackData::default);
    }

    let max_stack = max_stack.max(1);
    let slot = &mut slots[slot_idx];

    if output_only {
        if slot.is_empty() {
            return false;
        }
        if cursor.held.is_empty() {
            cursor.held = slot.clone();
            *slot = ItemStackData::default();
            return true;
        }
        if cursor.held.name == slot.name
            && cursor.held.wear == 0
            && slot.wear == 0
            && cursor.held.count + slot.count <= max_stack
        {
            cursor.held.count += slot.count;
            *slot = ItemStackData::default();
            return true;
        }
        return false;
    }

    match button {
        SlotClickButton::Left => {
            if cursor.held.is_empty() && slot.is_empty() {
                return false;
            }
            if cursor.held.is_empty() {
                cursor.held = std::mem::take(slot);
                return true;
            }
            if slot.is_empty() {
                *slot = std::mem::take(&mut cursor.held);
                return true;
            }
            if slot.name == cursor.held.name && slot.wear == 0 && cursor.held.wear == 0 {
                let space = max_stack.saturating_sub(slot.count);
                if space > 0 {
                    let moved = space.min(cursor.held.count);
                    slot.count += moved;
                    cursor.held.count -= moved;
                    if cursor.held.count == 0 {
                        cursor.clear();
                    }
                    return true;
                }
                return false;
            }
            std::mem::swap(slot, &mut cursor.held);
            true
        }
        SlotClickButton::Right => {
            if cursor.held.is_empty() {
                if slot.is_empty() {
                    return false;
                }
                let take = (slot.count + 1) / 2;
                let mut picked = slot.clone();
                picked.count = take;
                slot.count -= take;
                if slot.count == 0 {
                    *slot = ItemStackData::default();
                }
                cursor.held = picked;
                return true;
            }
            if slot.is_empty() {
                let mut placed = cursor.held.clone();
                placed.count = 1;
                *slot = placed;
                cursor.held.count -= 1;
                if cursor.held.count == 0 {
                    cursor.clear();
                }
                return true;
            }
            if slot.name == cursor.held.name && slot.wear == 0 && cursor.held.wear == 0 {
                if slot.count < max_stack {
                    slot.count += 1;
                    cursor.held.count -= 1;
                    if cursor.held.count == 0 {
                        cursor.clear();
                    }
                    return true;
                }
                return false;
            }
            std::mem::swap(slot, &mut cursor.held);
            true
        }
    }
}

/// Shift-click quick transfer up to `max_move` items from `src_slots[src_idx]` into `dst_slots`.
/// Returns `(first_changed_dst_idx, moved_stack)` when at least 1 item moved.
pub fn shift_transfer_limited(
    src_slots: &mut [ItemStackData],
    src_idx: usize,
    dst_slots: &mut [ItemStackData],
    max_stack: u16,
    max_move: u16,
) -> Option<(usize, ItemStackData)> {
    if src_idx >= src_slots.len() || src_slots[src_idx].is_empty() || max_move == 0 {
        return None;
    }
    let max_stack = max_stack.max(1);
    let initial_take = src_slots[src_idx].count.min(max_move);
    let mut rem = src_slots[src_idx].clone();
    rem.count = initial_take;
    let mut first_dst: Option<usize> = None;

    // Pass 1: stack onto existing non-empty matching stacks
    if rem.wear == 0 {
        for (di, dst) in dst_slots.iter_mut().enumerate() {
            if rem.is_empty() {
                break;
            }
            if !dst.is_empty() && dst.name == rem.name && dst.wear == 0 && dst.count < max_stack {
                let take = (max_stack - dst.count).min(rem.count);
                if take > 0 {
                    dst.count += take;
                    rem.count -= take;
                    if first_dst.is_none() {
                        first_dst = Some(di);
                    }
                    if rem.count == 0 {
                        rem = ItemStackData::default();
                    }
                }
            }
        }
    }

    // Pass 2: place remainder into first empty slot(s)
    for (di, dst) in dst_slots.iter_mut().enumerate() {
        if rem.is_empty() {
            break;
        }
        if dst.is_empty() {
            let take = rem.count.min(max_stack);
            if take > 0 {
                let mut moved = rem.clone();
                moved.count = take;
                *dst = moved;
                rem.count -= take;
                if first_dst.is_none() {
                    first_dst = Some(di);
                }
                if rem.count == 0 {
                    rem = ItemStackData::default();
                }
            }
        }
    }

    let moved_count = initial_take.saturating_sub(rem.count);
    if moved_count == 0 {
        return None;
    }
    let mut moved_summary = src_slots[src_idx].clone();
    moved_summary.count = moved_count;
    src_slots[src_idx].count = src_slots[src_idx].count.saturating_sub(moved_count);
    if src_slots[src_idx].count == 0 {
        src_slots[src_idx] = ItemStackData::default();
    }
    Some((first_dst.unwrap_or(0), moved_summary))
}

/// Shift-click quick transfer from `src_slots[src_idx]` into `dst_slots`.
/// Pass 1 merges with existing identical stacks; pass 2 fills empty slots.
pub fn shift_transfer_slot(
    src_slots: &mut [ItemStackData],
    src_idx: usize,
    dst_slots: &mut [ItemStackData],
    max_stack: u16,
) -> bool {
    shift_transfer_limited(src_slots, src_idx, dst_slots, max_stack, u16::MAX).is_some()
}

/// Shift-click quick transfer within player's `"main"` inventory (hotbar 0..9 <-> storage 9..36).
pub fn shift_transfer_within_main(
    main_slots: &mut [ItemStackData],
    src_idx: usize,
    max_stack: u16,
) -> bool {
    if src_idx >= main_slots.len() || main_slots[src_idx].is_empty() || main_slots.len() <= 9 {
        return false;
    }
    let (hotbar, storage) = main_slots.split_at_mut(9);
    if src_idx < 9 {
        shift_transfer_slot(hotbar, src_idx, storage, max_stack)
    } else {
        shift_transfer_slot(storage, src_idx - 9, hotbar, max_stack)
    }
}

/// Resolve an inventory location string (`current_player`, `context`, `nodemeta:x,y,z`) to its live `MetaDataState`.
pub fn resolve_inv_meta(
    loc: &str,
    context_pos: Option<IVec3>,
    player_state: &Arc<RwLock<ObjectState>>,
    world: Option<&WorldStore>,
) -> Option<Arc<RwLock<MetaDataState>>> {
    let trimmed = loc.trim();
    if trimmed.is_empty() || trimmed == "current_player" || trimmed.starts_with("player:") {
        return Some(player_state.read().meta.clone());
    }
    if trimmed == "context" {
        if let (Some(pos), Some(w)) = (context_pos, world) {
            return Some(w.get_node_meta(pos));
        }
        return None;
    }
    if let Some(coords) = trimmed.strip_prefix("nodemeta:") {
        let mut it = coords.split(',').filter_map(|s| s.trim().parse::<i32>().ok());
        if let (Some(x), Some(y), Some(z), Some(w)) = (it.next(), it.next(), it.next(), world) {
            return Some(w.get_node_meta(IVec3::new(x, y, z)));
        }
    }
    None
}

/// Determine quick-transfer target `(loc, list)` pairs from `listring[]` declarations or defaults.
pub fn resolve_listring_targets(parsed: &ParsedFormspec, src_loc: &str, src_list: &str) -> Vec<(String, String)> {
    let rings: Vec<(String, String)> = parsed
        .elements
        .iter()
        .filter_map(|e| match e {
            FormElement::ListRing { loc, list } => Some((loc.clone(), list.clone())),
            _ => None,
        })
        .collect();

    let mut targets = Vec::new();
    for (idx, (l, lst)) in rings.iter().enumerate() {
        if l == src_loc && lst == src_list && rings.len() > 1 {
            let next = &rings[(idx + 1) % rings.len()];
            if !(next.0 == src_loc && next.1 == src_list) && !targets.contains(next) {
                targets.push(next.clone());
            }
        }
    }

    if targets.is_empty() {
        if src_loc == "current_player" && src_list == "main" {
            for e in &parsed.elements {
                if let FormElement::InvList { loc, list, .. } = e {
                    if !(loc == "current_player" && list == "main")
                        && list != "craftpreview"
                        && list != "craftresult"
                        && list != "dst"
                    {
                        targets.push((loc.clone(), list.clone()));
                    }
                }
            }
        } else {
            targets.push(("current_player".into(), "main".into()));
        }
    }
    targets
}

/// Resolve a node-inventory location (`"context"` or `"nodemeta:x,y,z"`) to its world coordinate.
pub fn resolve_inv_node_pos(loc: &str, context_pos: Option<IVec3>) -> Option<IVec3> {
    let trimmed = loc.trim();
    if trimmed == "context" {
        return context_pos;
    }
    if let Some(coords) = trimmed.strip_prefix("nodemeta:") {
        let mut it = coords.split(',').filter_map(|s| s.trim().parse::<i32>().ok());
        if let (Some(x), Some(y), Some(z)) = (it.next(), it.next(), it.next()) {
            return Some(IVec3::new(x, y, z));
        }
    }
    None
}

/// Returns true if the parsed formspec renders a `3x3` player crafting grid.
pub fn formspec_requires_3x3_craft(parsed: &ParsedFormspec) -> bool {
    parsed.elements.iter().any(|el| {
        matches!(
            el,
            FormElement::InvList { loc, list, w, h, start, .. }
                if list == "craft"
                    && (loc.is_empty() || loc == "current_player")
                    && (*w >= 3 || *h >= 3 || (*start + *w * *h) > 4)
        )
    })
}

/// Dynamically resize player's `"craft"` grid between `4` (`2x2`) and `9` (`3x3`),
/// safely returning any leftover items in slots `4..` to `"main"` when shrinking to `4`.
pub fn ensure_craft_grid_size(meta: &Arc<RwLock<MetaDataState>>, target_slots: usize) {
    let target = if target_slots >= 9 { 9 } else { 4 };
    let mut g = meta.write();
    let cur_len = g.inventories.get("craft").map(Vec::len).unwrap_or(0);
    if cur_len == target {
        return;
    }
    if cur_len > target {
        let mut craft = g.inventories.remove("craft").unwrap_or_default();
        if craft.len() > target {
            let main_slots = g
                .inventories
                .entry("main".into())
                .or_insert_with(|| vec![ItemStackData::default(); 36]);
            for idx in target..craft.len() {
                if !craft[idx].is_empty() {
                    let _ = shift_transfer_slot(&mut craft, idx, main_slots, 64);
                }
            }
            craft.truncate(target);
        }
        craft.resize_with(target, ItemStackData::default);
        g.inventories.insert("craft".into(), craft);
    } else {
        let craft = g.inventories.entry("craft".into()).or_default();
        craft.resize_with(target, ItemStackData::default);
    }
}

/// Evaluate crafting grid recipes via `resolve_craft_recipe` and populate `craftpreview`.
pub fn refresh_craft_preview(meta: &Arc<RwLock<MetaDataState>>, reg: &EngineRegistries) {
    let mut g = meta.write();
    let Some(craft) = g.inventories.get("craft").cloned() else {
        return;
    };
    let width = if craft.len() == 4 { 2 } else { 3 };
    let result = resolve_craft_recipe(reg, &craft, width, "normal").map(|out| out.item);
    let preview = g
        .inventories
        .entry("craftpreview".into())
        .or_insert_with(|| vec![ItemStackData::default()]);
    if preview.is_empty() {
        preview.push(ItemStackData::default());
    }
    preview[0] = result.unwrap_or_default();
}

/// Consume crafting ingredients via `resolve_craft_recipe`, apply replacements, and recompute `craftpreview`.
pub fn consume_craft_ingredients(meta: &Arc<RwLock<MetaDataState>>, reg: &EngineRegistries) {
    {
        let mut g = meta.write();
        if let Some(craft) = g.inventories.get("craft").cloned() {
            let width = if craft.len() == 4 { 2 } else { 3 };
            if let Some(out) = resolve_craft_recipe(reg, &craft, width, "normal") {
                g.inventories.insert("craft".into(), out.decrements);
                if !out.replacements.is_empty() {
                    let main_inv = g
                        .inventories
                        .entry("main".into())
                        .or_insert_with(|| vec![ItemStackData::default(); 36]);
                    let mut rep_buf = out.replacements;
                    for i in 0..rep_buf.len() {
                        let _ = shift_transfer_slot(&mut rep_buf, i, main_inv, 64);
                    }
                }
            }
        }
    }
    refresh_craft_preview(meta, reg);
}

fn unescape_fs(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(n) = chars.next() {
                out.push(n);
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn clean_fs_text(s: &str) -> String {
    strip_luanti_escapes(&unescape_fs(s))
}

fn split_args(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut escaped = false;
    for ch in s.chars() {
        if escaped {
            cur.push(ch);
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == ';' {
            out.push(std::mem::take(&mut cur));
        } else {
            cur.push(ch);
        }
    }
    out.push(cur);
    out
}

fn split_comma_escaped(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut escaped = false;
    for ch in s.chars() {
        if escaped {
            cur.push(ch);
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == ',' {
            out.push(std::mem::take(&mut cur));
        } else {
            cur.push(ch);
        }
    }
    out.push(cur);
    out
}

fn parse_xy(s: &str) -> (f32, f32) {
    let mut it = s.split(',');
    let x = it.next().and_then(|v| v.trim().parse::<f32>().ok()).unwrap_or(0.0);
    let y = it.next().and_then(|v| v.trim().parse::<f32>().ok()).unwrap_or(0.0);
    (x, y)
}

fn parse_hex_color(s: &str) -> Color32 {
    let hex = s.trim().trim_start_matches('#');
    if hex.len() >= 6 {
        let r = u8::from_str_radix(&hex[0..2], 16).unwrap_or(198);
        let g = u8::from_str_radix(&hex[2..4], 16).unwrap_or(198);
        let b = u8::from_str_radix(&hex[4..6], 16).unwrap_or(198);
        let a = if hex.len() >= 8 {
            u8::from_str_radix(&hex[6..8], 16).unwrap_or(230)
        } else {
            230
        };
        return Color32::from_rgba_unmultiplied(r, g, b, a);
    }
    Color32::from_rgba_unmultiplied(140, 140, 140, 200)
}

/// Parse Luanti/Mineclonia formspecs into structured UI elements for native Minecraft-styled rendering.
pub fn parse_formspec(raw: &str) -> ParsedFormspec {
    let mut parsed = ParsedFormspec {
        version: 1,
        size: [9.0, 8.5],
        elements: Vec::new(),
    };

    let bytes = raw.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        let cmd_start = i;
        while i < bytes.len() && bytes[i] != b'[' {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        let cmd = raw[cmd_start..i].trim();
        i += 1; // skip '['
        let arg_start = i;
        let mut esc = false;
        while i < bytes.len() {
            if esc {
                esc = false;
            } else if bytes[i] == b'\\' {
                esc = true;
            } else if bytes[i] == b']' {
                break;
            }
            i += 1;
        }
        let body = &raw[arg_start..i.min(bytes.len())];
        if i < bytes.len() {
            i += 1;
        }
        let args = split_args(body);

        match cmd {
            "formspec_version" => {
                if let Some(a0) = args.first() {
                    if let Ok(v) = a0.trim().parse::<u32>() {
                        parsed.version = v;
                    }
                }
            }
            "size" => {
                if let Some(a0) = args.first() {
                    let (w, h) = parse_xy(a0);
                    if w > 0.0 && h > 0.0 {
                        parsed.size = [w, h];
                    }
                }
            }
            "bgcolor" => {
                if let Some(a0) = args.first() {
                    let color = parse_hex_color(a0);
                    let fs = args.get(1).map(|s| s == "true" || s == "both").unwrap_or(false);
                    let fbg = args.get(2).filter(|s| !s.trim().is_empty()).map(|s| parse_hex_color(s));
                    parsed.elements.push(FormElement::BgColor {
                        color,
                        fullscreen: fs,
                        fullscreen_color: fbg,
                    });
                }
            }
            "label" => {
                if args.len() >= 2 {
                    let (x, y) = parse_xy(&args[0]);
                    parsed.elements.push(FormElement::Label {
                        x,
                        y,
                        text: clean_fs_text(&args[1]),
                    });
                }
            }
            "vertlabel" => {
                if args.len() >= 2 {
                    let (x, y) = parse_xy(&args[0]);
                    parsed.elements.push(FormElement::VertLabel {
                        x,
                        y,
                        text: clean_fs_text(&args[1]),
                    });
                }
            }
            "button" | "button_exit" => {
                if args.len() >= 4 {
                    let (x, y) = parse_xy(&args[0]);
                    let (w, h) = parse_xy(&args[1]);
                    parsed.elements.push(FormElement::Button {
                        x,
                        y,
                        w: w.max(0.8),
                        h: h.max(0.5),
                        name: args[2].clone(),
                        label: clean_fs_text(&args[3]),
                        exit: cmd == "button_exit",
                    });
                }
            }
            "image_button" | "image_button_exit" => {
                if args.len() >= 5 {
                    let (x, y) = parse_xy(&args[0]);
                    let (w, h) = parse_xy(&args[1]);
                    parsed.elements.push(FormElement::ImageButton {
                        x,
                        y,
                        w: w.max(0.6),
                        h: h.max(0.6),
                        texture: args[2].clone(),
                        name: args[3].clone(),
                        label: clean_fs_text(&args[4]),
                        exit: cmd == "image_button_exit",
                    });
                }
            }
            "item_image_button" => {
                if args.len() >= 5 {
                    let (x, y) = parse_xy(&args[0]);
                    let (w, h) = parse_xy(&args[1]);
                    parsed.elements.push(FormElement::ItemImageButton {
                        x,
                        y,
                        w: w.max(0.6),
                        h: h.max(0.6),
                        item_name: args[2].clone(),
                        name: args[3].clone(),
                        label: clean_fs_text(&args[4]),
                    });
                }
            }
            "field" => {
                if args.len() >= 4 {
                    let (x, y) = parse_xy(&args[0]);
                    let (w, h) = parse_xy(&args[1]);
                    parsed.elements.push(FormElement::Field {
                        x,
                        y,
                        w: w.max(1.2),
                        h: h.max(0.5),
                        name: args[2].clone(),
                        label: clean_fs_text(&args[3]),
                        default: args.get(4).map(|s| unescape_fs(s)).unwrap_or_default(),
                    });
                } else if args.len() == 3 {
                    parsed.elements.push(FormElement::Field {
                        x: 0.5,
                        y: 0.5,
                        w: 4.0,
                        h: 0.8,
                        name: args[0].clone(),
                        label: clean_fs_text(&args[1]),
                        default: unescape_fs(&args[2]),
                    });
                }
            }
            "pwdfield" => {
                if args.len() >= 4 {
                    let (x, y) = parse_xy(&args[0]);
                    let (w, h) = parse_xy(&args[1]);
                    parsed.elements.push(FormElement::PwdField {
                        x,
                        y,
                        w: w.max(1.2),
                        h: h.max(0.5),
                        name: args[2].clone(),
                        label: clean_fs_text(&args[3]),
                    });
                }
            }
            "textarea" => {
                if args.len() >= 4 {
                    let (x, y) = parse_xy(&args[0]);
                    let (w, h) = parse_xy(&args[1]);
                    parsed.elements.push(FormElement::TextArea {
                        x,
                        y,
                        w: w.max(1.5),
                        h: h.max(1.0),
                        name: args[2].clone(),
                        label: clean_fs_text(&args[3]),
                        default: args.get(4).map(|s| unescape_fs(s)).unwrap_or_default(),
                    });
                }
            }
            "list" => {
                if args.len() >= 4 {
                    let (x, y) = parse_xy(&args[2]);
                    let (w, h) = parse_xy(&args[3]);
                    let start = args.get(4).and_then(|s| s.trim().parse::<usize>().ok()).unwrap_or(0);
                    parsed.elements.push(FormElement::InvList {
                        loc: args[0].clone(),
                        list: args[1].clone(),
                        x,
                        y,
                        w: (w.round() as usize).max(1),
                        h: (h.round() as usize).max(1),
                        start,
                    });
                }
            }
            "listring" => {
                if args.len() >= 2 {
                    parsed.elements.push(FormElement::ListRing {
                        loc: args[0].clone(),
                        list: args[1].clone(),
                    });
                }
            }
            "box" => {
                if args.len() >= 3 {
                    let (x, y) = parse_xy(&args[0]);
                    let (w, h) = parse_xy(&args[1]);
                    parsed.elements.push(FormElement::Box {
                        x,
                        y,
                        w,
                        h,
                        color: parse_hex_color(&args[2]),
                    });
                }
            }
            "image" | "background" | "background9" => {
                if args.len() >= 3 && cmd == "image" {
                    let (x, y) = parse_xy(&args[0]);
                    let (w, h) = parse_xy(&args[1]);
                    parsed.elements.push(FormElement::Image {
                        x,
                        y,
                        w: w.max(0.2),
                        h: h.max(0.2),
                        texture: args[2].clone(),
                    });
                }
            }
            "item_image" => {
                if args.len() >= 3 {
                    let (x, y) = parse_xy(&args[0]);
                    let (w, h) = parse_xy(&args[1]);
                    parsed.elements.push(FormElement::ItemImage {
                        x,
                        y,
                        w: w.max(0.5),
                        h: h.max(0.5),
                        item_name: args[2].clone(),
                    });
                }
            }
            "checkbox" => {
                if args.len() >= 3 {
                    let (x, y) = parse_xy(&args[0]);
                    let sel = args.get(3).map(|s| s == "true").unwrap_or(false);
                    parsed.elements.push(FormElement::Checkbox {
                        x,
                        y,
                        name: args[1].clone(),
                        label: clean_fs_text(&args[2]),
                        selected: sel,
                    });
                }
            }
            "dropdown" => {
                if args.len() >= 5 {
                    let (x, y) = parse_xy(&args[0]);
                    let (w, h_raw) = parse_xy(&args[1]);
                    let h = if h_raw > 0.0 { h_raw } else { 0.8 };
                    let items: Vec<String> = split_comma_escaped(&args[3])
                        .into_iter()
                        .map(|s| clean_fs_text(&s))
                        .collect();
                    let selected_idx = args[4].trim().parse::<usize>().unwrap_or(1).max(1);
                    let index_event = args.get(5).map(|s| s == "true").unwrap_or(false);
                    parsed.elements.push(FormElement::Dropdown {
                        x,
                        y,
                        w: w.max(1.2),
                        h,
                        name: args[2].clone(),
                        items,
                        selected_idx,
                        index_event,
                    });
                }
            }
            "tabheader" => {
                if args.len() >= 4 {
                    let (x, y) = parse_xy(&args[0]);
                    let (w, h, name_idx) = if args.len() >= 5 && args[1].contains(',') {
                        let (tw, th) = parse_xy(&args[1]);
                        (tw.max(1.0), th.max(0.6), 2usize)
                    } else {
                        (parsed.size[0], 0.7, 1usize)
                    };
                    if name_idx + 2 < args.len() {
                        let name = args[name_idx].clone();
                        let captions: Vec<String> = split_comma_escaped(&args[name_idx + 1])
                            .into_iter()
                            .map(|s| clean_fs_text(&s))
                            .collect();
                        let current_tab = args[name_idx + 2].trim().parse::<usize>().unwrap_or(1).max(1);
                        parsed.elements.push(FormElement::TabHeader {
                            x,
                            y,
                            w,
                            h,
                            name,
                            captions,
                            current_tab,
                        });
                    }
                }
            }
            "tooltip" => {
                if args.len() >= 2 && !args[0].contains(',') {
                    parsed.elements.push(FormElement::Tooltip {
                        target: Some(args[0].clone()),
                        rect: None,
                        text: clean_fs_text(&args[1]),
                    });
                } else if args.len() >= 3 && args[0].contains(',') {
                    let (x, y) = parse_xy(&args[0]);
                    let (w, h) = parse_xy(&args[1]);
                    parsed.elements.push(FormElement::Tooltip {
                        target: None,
                        rect: Some([x, y, w, h]),
                        text: clean_fs_text(&args[2]),
                    });
                }
            }
            _ => {}
        }
    }

    parsed
}

/// Controller button & directional input passed into an open formspec UI.
#[derive(Debug, Clone, Copy, Default)]
pub struct ControllerFormInput {
    pub move_x: i32,
    pub move_y: i32,
    pub confirm: bool,
    pub secondary: bool,
    pub shift: bool,
    pub cancel: bool,
}

pub struct FormspecContext<'a> {
    pub world: Option<&'a WorldStore>,
    pub context_pos: Option<IVec3>,
    pub registries: Option<&'a EngineRegistries>,
    pub atlas: Option<&'a TextureAtlas>,
    pub icon_cache: Option<&'a mut ItemIconCache>,
    pub cursor: Option<&'a mut InventoryCursor>,
    pub lua_rt: Option<&'a crate::lua_api::LuaRuntime>,
    /// Optional sub-viewport bounds (used by Split-Screen co-op). Defaults to `ctx.screen_rect()`.
    pub viewport: Option<egui::Rect>,
    /// Unique egui Area salt (allows Player 1 & Player 2 formspecs simultaneously).
    pub id_salt: u64,
    /// Optional controller navigation input for this formspec.
    pub controller_input: Option<ControllerFormInput>,
    /// Persistent focused slot index when navigating with a controller.
    pub controller_slot: Option<&'a mut usize>,
}

impl<'a> Default for FormspecContext<'a> {
    fn default() -> Self {
        Self {
            world: None,
            context_pos: None,
            registries: None,
            atlas: None,
            icon_cache: None,
            cursor: None,
            lua_rt: None,
            viewport: None,
            id_salt: 0,
            controller_input: None,
            controller_slot: None,
        }
    }
}

/// Render a parsed Mineclonia/Luanti formspec inside an authentic `#C6C6C6` beveled grey Minecraft window.
pub fn render_formspec_overlay(
    ctx: &egui::Context,
    theme: &McTheme,
    formname: &str,
    parsed: &ParsedFormspec,
    player_state: &Arc<RwLock<ObjectState>>,
    field_bufs: &mut HashMap<String, String>,
) -> (Vec<(String, String)>, bool) {
    render_formspec_interactive(
        ctx,
        theme,
        formname,
        parsed,
        player_state,
        field_bufs,
        FormspecContext::default(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn render_formspec_interactive(
    ctx: &egui::Context,
    theme: &McTheme,
    formname: &str,
    parsed: &ParsedFormspec,
    player_state: &Arc<RwLock<ObjectState>>,
    field_bufs: &mut HashMap<String, String>,
    mut fctx: FormspecContext<'_>,
) -> (Vec<(String, String)>, bool) {
    let mut submitted_fields = Vec::new();
    let mut close_requested = false;
    let mut hover_tooltip: Option<String> = None;

    // Collect named tooltips + rect tooltips
    let mut elem_tooltips: HashMap<String, String> = HashMap::new();
    let mut rect_tooltips: Vec<([f32; 4], String)> = Vec::new();
    let mut panel_bg_override: Option<Color32> = None;
    let mut backdrop_col = Color32::from_rgba_unmultiplied(0, 0, 0, 150);

    for el in &parsed.elements {
        match el {
            FormElement::Tooltip { target: Some(t), text, .. } => {
                elem_tooltips.insert(t.clone(), text.clone());
            }
            FormElement::Tooltip { rect: Some(r), text, .. } => {
                rect_tooltips.push((*r, text.clone()));
            }
            FormElement::BgColor { color, fullscreen, fullscreen_color } => {
                panel_bg_override = Some(*color);
                if *fullscreen {
                    backdrop_col = fullscreen_color.unwrap_or(*color);
                } else if let Some(fc) = fullscreen_color {
                    backdrop_col = *fc;
                }
            }
            _ => {}
        }
    }

    let screen = fctx.viewport.unwrap_or_else(|| ctx.screen_rect());
    let raw_unit = 48.0f32;
    let max_w = (screen.width() - 24.0).max(220.0);
    let max_h = (screen.height() - 32.0).max(180.0);
    let unit = (max_w / parsed.size[0].max(1.0))
        .min(max_h / parsed.size[1].max(1.0))
        .min(raw_unit)
        .max(18.0);
    let win_w = (parsed.size[0] * unit).clamp(220.0, max_w);
    let win_h = (parsed.size[1] * unit).clamp(180.0, max_h);
    let win_rect = Rect::from_center_size(screen.center(), Vec2::new(win_w, win_h));

    let coord_scale = if parsed.version >= 2 { unit } else { unit * 0.92 };
    let origin = if parsed.version >= 2 {
        win_rect.min
    } else {
        win_rect.min + Vec2::new(6.0, 6.0)
    };

    // Precompute inventory slot centers for controller directional navigation
    let mut slot_centers: Vec<Pos2> = Vec::new();
    for el in &parsed.elements {
        if let FormElement::InvList { x, y, w, h, .. } = el {
            let (slot_sz, step) = if parsed.version >= 2 {
                (coord_scale, coord_scale * 1.25)
            } else {
                ((coord_scale * 0.90).clamp(20.0, 44.0), coord_scale)
            };
            for row in 0..*h {
                for col in 0..*w {
                    let srect = Rect::from_min_size(
                        origin + Vec2::new(*x * coord_scale + col as f32 * step, *y * coord_scale + row as f32 * step),
                        Vec2::splat(slot_sz),
                    );
                    slot_centers.push(srect.center());
                }
            }
        }
    }

    let ctrl_in = fctx.controller_input;
    let mut active_ctrl_slot: Option<usize> = None;
    if let Some(sel) = fctx.controller_slot.as_deref_mut() {
        if !slot_centers.is_empty() {
            *sel = (*sel).min(slot_centers.len() - 1);
            if let Some(ci) = ctrl_in {
                if ci.cancel {
                    close_requested = true;
                } else if ci.move_x != 0 || ci.move_y != 0 {
                    let cur = slot_centers[*sel];
                    let mut best_idx = None;
                    let mut best_score = f32::INFINITY;
                    for (i, &cand) in slot_centers.iter().enumerate() {
                        if i == *sel {
                            continue;
                        }
                        let dx = cand.x - cur.x;
                        let dy = cand.y - cur.y;
                        let in_dir = (ci.move_x < 0 && dx < -4.0 && dy.abs() <= 36.0)
                            || (ci.move_x > 0 && dx > 4.0 && dy.abs() <= 36.0)
                            || (ci.move_y < 0 && dy < -4.0)
                            || (ci.move_y > 0 && dy > 4.0);
                        if in_dir {
                            let score = if ci.move_x != 0 {
                                dx.abs() + dy.abs() * 3.0
                            } else {
                                dy.abs() + dx.abs() * 1.25
                            };
                            if score < best_score {
                                best_score = score;
                                best_idx = Some(i);
                            }
                        }
                    }
                    if let Some(next_i) = best_idx {
                        *sel = next_i;
                    } else if ci.move_x < 0 || ci.move_y < 0 {
                        *sel = if *sel == 0 { slot_centers.len() - 1 } else { *sel - 1 };
                    } else {
                        *sel = (*sel + 1) % slot_centers.len();
                    }
                }
            }
            if ctrl_in.is_some() {
                active_ctrl_slot = Some(*sel);
            }
        } else if let Some(ci) = ctrl_in {
            if ci.cancel {
                close_requested = true;
            }
        }
    }

    let shift_down = ctx.input(|i| i.modifiers.shift);
    let left_clicked = ctx.input(|i| i.pointer.button_pressed(egui::PointerButton::Primary));
    let right_clicked = ctx.input(|i| i.pointer.button_pressed(egui::PointerButton::Secondary));
    let pointer_pos = ctx
        .input(|i| i.pointer.interact_pos().or_else(|| i.pointer.latest_pos()))
        .filter(|p| screen.contains(*p));
    let mut slot_acted_this_frame = false;

    let has_player_craft = parsed.elements.iter().any(|el| {
        matches!(
            el,
            FormElement::InvList { loc, list, .. }
                if list == "craft" && (loc.is_empty() || loc == "current_player")
        )
    });
    if has_player_craft {
        let target_slots = if formspec_requires_3x3_craft(parsed) { 9 } else { 4 };
        ensure_craft_grid_size(&player_state.read().meta, target_slots);
    }

    if let Some(reg) = fctx.registries {
        for el in &parsed.elements {
            if let FormElement::InvList { loc, list, .. } = el {
                if list == "craftpreview" || list == "craftresult" {
                    if let Some(m) = resolve_inv_meta(loc, fctx.context_pos, player_state, fctx.world) {
                        if m.read().inventories.contains_key("craft") {
                            refresh_craft_preview(&m, reg);
                        }
                    }
                }
            }
        }
    }

    egui::Area::new(egui::Id::new(("cubex_formspec_backdrop", fctx.id_salt)))
        .fixed_pos(screen.min)
        .show(ctx, |ui| {
            ui.painter().rect_filled(screen, 0.0, backdrop_col);
        });

    let mut inv_slot_counter = 0usize;
    egui::Area::new(egui::Id::new(("cubex_formspec_window", fctx.id_salt)))
        .fixed_pos(win_rect.min)
        .show(ctx, |ui| {
            let p_owned = ui.painter().clone().with_clip_rect(screen);
            let p = &p_owned;

            if let Some(bg) = panel_bg_override {
                p.rect_filled(win_rect, 2.0, bg);
                p.rect_stroke(win_rect, 2.0, Stroke::new(2.0, Color32::BLACK), egui::StrokeKind::Inside);
            } else {
                paint_mc_panel(p, win_rect);
            }

            for (idx, el) in parsed.elements.iter().enumerate() {
                match el {
                    FormElement::BgColor { .. } | FormElement::ListRing { .. } | FormElement::Tooltip { .. } => {}
                    FormElement::Box { x, y, w, h, color } => {
                        let r = Rect::from_min_size(
                            origin + Vec2::new(*x * coord_scale, *y * coord_scale),
                            Vec2::new(*w * coord_scale, *h * coord_scale),
                        );
                        p.rect_filled(r, 0.0, *color);
                    }
                    FormElement::Label { x, y, text } => {
                        let pos = origin + Vec2::new(*x * coord_scale, *y * coord_scale);
                        p.text(pos, Align2::LEFT_TOP, text, egui::FontId::monospace(13.5), Color32::from_rgb(56, 56, 56));
                    }
                    FormElement::VertLabel { x, y, text } => {
                        let pos = origin + Vec2::new(*x * coord_scale, *y * coord_scale);
                        for (ci, ch) in text.chars().enumerate() {
                            let cpos = pos + Vec2::new(0.0, ci as f32 * 13.0);
                            p.text(cpos, Align2::LEFT_TOP, &ch.to_string(), egui::FontId::monospace(12.5), Color32::from_rgb(56, 56, 56));
                        }
                    }
                    FormElement::Image { x, y, w, h, texture } => {
                        let r = Rect::from_min_size(
                            origin + Vec2::new(*x * coord_scale, *y * coord_scale),
                            Vec2::new(*w * coord_scale, *h * coord_scale),
                        );
                        let tex_id = resolve_texture_id(ctx, texture, fctx.atlas, fctx.icon_cache.as_deref_mut());
                        if let Some(tid) = tex_id {
                            p.image(tid, r, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
                        } else {
                            p.rect_stroke(r, 0.0, Stroke::new(1.0, Color32::from_rgb(110, 110, 110)), egui::StrokeKind::Inside);
                        }
                    }
                    FormElement::ItemImage { x, y, w, h, item_name } => {
                        let r = Rect::from_min_size(
                            origin + Vec2::new(*x * coord_scale, *y * coord_scale),
                            Vec2::new(*w * coord_scale, *h * coord_scale),
                        );
                        let tid = resolve_item_tex_id(ctx, item_name, fctx.atlas, fctx.registries, fctx.icon_cache.as_deref_mut());
                        paint_item_in_slot(p, r, tid, item_name, 1, 0);
                    }
                    FormElement::Button { x, y, w, h, name, label, exit } => {
                        let r = Rect::from_min_size(
                            origin + Vec2::new(*x * coord_scale, *y * coord_scale),
                            Vec2::new(*w * coord_scale, *h * coord_scale),
                        );
                        let resp = ui
                            .allocate_new_ui(egui::UiBuilder::new().max_rect(r), |ui| {
                                mc_button(ui, theme, label, r.width(), r.height())
                            })
                            .inner;
                        if resp.hovered() {
                            if let Some(tt) = elem_tooltips.get(name) {
                                hover_tooltip = Some(tt.clone());
                            }
                        }
                        if resp.clicked() {
                            submitted_fields.push((name.clone(), label.clone()));
                            if *exit {
                                close_requested = true;
                            }
                        }
                    }
                    FormElement::ImageButton { x, y, w, h, texture, name, label, exit } => {
                        let r = Rect::from_min_size(
                            origin + Vec2::new(*x * coord_scale, *y * coord_scale),
                            Vec2::new(*w * coord_scale, *h * coord_scale),
                        );
                        let resp = ui.allocate_rect(r, egui::Sense::click());
                        paint_mc_slot(p, r, resp.hovered());
                        if let Some(tid) = resolve_texture_id(ctx, texture, fctx.atlas, fctx.icon_cache.as_deref_mut()) {
                            p.image(tid, r.shrink(4.0), Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
                        }
                        if !label.is_empty() {
                            mc_shadow_text(p, r.center(), Align2::CENTER_CENTER, label, 11.5, Color32::WHITE);
                        }
                        if resp.hovered() {
                            if let Some(tt) = elem_tooltips.get(name) {
                                hover_tooltip = Some(tt.clone());
                            }
                        }
                        if resp.clicked() {
                            theme.play_click();
                            submitted_fields.push((name.clone(), label.clone()));
                            if *exit {
                                close_requested = true;
                            }
                        }
                    }
                    FormElement::ItemImageButton { x, y, w, h, item_name, name, label } => {
                        let r = Rect::from_min_size(
                            origin + Vec2::new(*x * coord_scale, *y * coord_scale),
                            Vec2::new(*w * coord_scale, *h * coord_scale),
                        );
                        let resp = ui.allocate_rect(r, egui::Sense::click());
                        paint_mc_slot(p, r, resp.hovered());
                        let tid = resolve_item_tex_id(ctx, item_name, fctx.atlas, fctx.registries, fctx.icon_cache.as_deref_mut());
                        paint_item_in_slot(p, r, tid, item_name, 1, 0);
                        if !label.is_empty() {
                            mc_shadow_text(p, r.center(), Align2::CENTER_CENTER, label, 11.0, Color32::WHITE);
                        }
                        if resp.hovered() {
                            if let Some(tt) = elem_tooltips.get(name) {
                                hover_tooltip = Some(tt.clone());
                            }
                        }
                        if resp.clicked() {
                            theme.play_click();
                            submitted_fields.push((name.clone(), label.clone()));
                        }
                    }
                    FormElement::Field { x, y, w, h, name, label, default } => {
                        let key = format!("{formname}:{name}");
                        let buf = field_bufs.entry(key).or_insert_with(|| default.clone());
                        let r = Rect::from_min_size(
                            origin + Vec2::new(*x * coord_scale, *y * coord_scale),
                            Vec2::new(*w * coord_scale, *h * coord_scale),
                        );
                        ui.allocate_new_ui(egui::UiBuilder::new().max_rect(r), |ui| {
                            if !label.is_empty() {
                                ui.label(egui::RichText::new(label).color(Color32::from_rgb(50, 50, 50)));
                            }
                            if ui.text_edit_singleline(buf).lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                                submitted_fields.push((name.clone(), buf.clone()));
                            }
                        });
                    }
                    FormElement::PwdField { x, y, w, h, name, label } => {
                        let key = format!("{formname}:{name}");
                        let buf = field_bufs.entry(key).or_default();
                        let r = Rect::from_min_size(
                            origin + Vec2::new(*x * coord_scale, *y * coord_scale),
                            Vec2::new(*w * coord_scale, *h * coord_scale),
                        );
                        ui.allocate_new_ui(egui::UiBuilder::new().max_rect(r), |ui| {
                            if !label.is_empty() {
                                ui.label(egui::RichText::new(label).color(Color32::from_rgb(50, 50, 50)));
                            }
                            ui.add(egui::TextEdit::singleline(buf).password(true));
                        });
                    }
                    FormElement::TextArea { x, y, w, h, name, label, default } => {
                        let key = format!("{formname}:{name}");
                        let buf = field_bufs.entry(key).or_insert_with(|| default.clone());
                        let r = Rect::from_min_size(
                            origin + Vec2::new(*x * coord_scale, *y * coord_scale),
                            Vec2::new(*w * coord_scale, *h * coord_scale),
                        );
                        ui.allocate_new_ui(egui::UiBuilder::new().max_rect(r), |ui| {
                            if !label.is_empty() {
                                ui.label(egui::RichText::new(label).color(Color32::from_rgb(50, 50, 50)));
                            }
                            egui::ScrollArea::vertical().show(ui, |ui| {
                                ui.text_edit_multiline(buf);
                            });
                        });
                    }
                    FormElement::InvList { loc, list, x, y, w, h, start } => {
                        let meta_arc = resolve_inv_meta(loc, fctx.context_pos, player_state, fctx.world);
                        let (slot_sz, step) = if parsed.version >= 2 {
                            (coord_scale, coord_scale * 1.25)
                        } else {
                            ((coord_scale * 0.90).clamp(20.0, 44.0), coord_scale)
                        };

                        for row in 0..*h {
                            for col in 0..*w {
                                let this_slot_ord = inv_slot_counter;
                                inv_slot_counter += 1;
                                let slot_idx = start + row * *w + col;
                                let srect = Rect::from_min_size(
                                    origin + Vec2::new(*x * coord_scale + col as f32 * step, *y * coord_scale + row as f32 * step),
                                    Vec2::splat(slot_sz),
                                );
                                let resp = ui.allocate_rect(srect, egui::Sense::click());
                                let is_ctrl_focus = active_ctrl_slot == Some(this_slot_ord);
                                let hovered = resp.hovered() || is_ctrl_focus;
                                paint_mc_slot(p, srect, hovered);
                                if is_ctrl_focus {
                                    p.rect_stroke(
                                        srect.expand(1.5),
                                        0.0,
                                        Stroke::new(2.5, Color32::from_rgb(255, 220, 64)),
                                        egui::StrokeKind::Inside,
                                    );
                                }

                                let item_opt = meta_arc.as_ref().and_then(|m| {
                                    m.read().inventories.get(list).and_then(|v| v.get(slot_idx)).cloned()
                                });
                                if let Some(ref item) = item_opt {
                                    if !item.is_empty() {
                                        let tid = resolve_item_tex_id(
                                            ctx,
                                            &item.name,
                                            fctx.atlas,
                                            fctx.registries,
                                            fctx.icon_cache.as_deref_mut(),
                                        );
                                        paint_item_in_slot(p, srect, tid, &item.name, item.count, item.wear);
                                        if hovered {
                                            hover_tooltip = Some(build_item_stack_tooltip(item, fctx.registries));
                                        }
                                    }
                                }

                                // Interactive Slot Click & Shift-Click Handling (Mouse or Controller)
                                let ptr_hit = pointer_pos.map_or(resp.hovered(), |pos| srect.contains(pos));
                                let ctrl_left = is_ctrl_focus
                                    && ctrl_in.is_some_and(|ci| ci.confirm || ci.shift);
                                let ctrl_right = is_ctrl_focus
                                    && ctrl_in.is_some_and(|ci| ci.secondary);
                                let eff_left = (ptr_hit && left_clicked) || ctrl_left;
                                let eff_right = (ptr_hit && right_clicked) || ctrl_right;
                                let eff_shift = (ptr_hit && shift_down)
                                    || (is_ctrl_focus && ctrl_in.is_some_and(|ci| ci.shift));

                                if !slot_acted_this_frame && (eff_left || eff_right) {
                                    slot_acted_this_frame = true;
                                    if let Some(ref meta) = meta_arc {
                                        let is_craft_out = list == "craftpreview" || list == "craftresult";
                                        let is_output = is_craft_out || list == "dst";
                                        let mut handled_transfer = false;
                                        let mut transferred_dst_meta: Option<Arc<RwLock<MetaDataState>>> = None;

                                        if eff_shift && eff_left {
                                            if is_craft_out {
                                                // Shift-click craft output crafts up to one full output stack into player main inventory
                                                let initial_name = meta
                                                    .read()
                                                    .inventories
                                                    .get(list)
                                                    .and_then(|v| v.get(slot_idx))
                                                    .filter(|s| !s.is_empty())
                                                    .map(|s| s.name.clone());
                                                if let Some(target_name) = initial_name {
                                                    let mut crafted_total = 0u16;
                                                    for _ in 0..64 {
                                                        let crafted = meta
                                                            .read()
                                                            .inventories
                                                            .get(list)
                                                            .and_then(|v| v.get(slot_idx))
                                                            .cloned()
                                                            .unwrap_or_default();
                                                        if crafted.is_empty() || crafted.name != target_name {
                                                            break;
                                                        }
                                                        if crafted_total + crafted.count > 64 {
                                                            break;
                                                        }
                                                        let pmeta = player_state.read().meta.clone();
                                                        let mut tmp_src = vec![crafted.clone()];
                                                        let moved = {
                                                            let mut pm = pmeta.write();
                                                            let dst = pm.inventories.entry("main".into()).or_insert_with(|| vec![ItemStackData::default(); 36]);
                                                            shift_transfer_slot(&mut tmp_src, 0, dst, 64)
                                                        };
                                                        if !moved || !tmp_src[0].is_empty() {
                                                            break;
                                                        }
                                                        crafted_total += crafted.count;
                                                        if let Some(reg) = fctx.registries {
                                                            consume_craft_ingredients(meta, reg);
                                                        } else {
                                                            break;
                                                        }
                                                        handled_transfer = true;
                                                    }
                                                }
                                            } else {
                                                let targets = resolve_listring_targets(parsed, loc, list);
                                                let src_node_pos = resolve_inv_node_pos(loc, fctx.context_pos);
                                                let src_snap = meta
                                                    .read()
                                                    .inventories
                                                    .get(list)
                                                    .and_then(|v| v.get(slot_idx))
                                                    .cloned()
                                                    .unwrap_or_default();
                                                let mut moved = false;

                                                if !src_snap.is_empty() {
                                                    let allow_take_max = match (src_node_pos, fctx.lua_rt) {
                                                        (Some(spos), Some(rt)) => rt.allow_metadata_inventory_take(
                                                            spos,
                                                            list,
                                                            slot_idx + 1,
                                                            &src_snap,
                                                            player_state,
                                                        ),
                                                        _ => src_snap.count,
                                                    };

                                                    if allow_take_max > 0 {
                                                        for (tloc, tlist) in targets {
                                                            if let Some(dst_meta) = resolve_inv_meta(
                                                                &tloc,
                                                                fctx.context_pos,
                                                                player_state,
                                                                fctx.world,
                                                            ) {
                                                                if Arc::ptr_eq(meta, &dst_meta) && list == &tlist {
                                                                    continue;
                                                                }
                                                                let dst_node_pos = resolve_inv_node_pos(&tloc, fctx.context_pos);
                                                                let same_node = src_node_pos.is_some() && src_node_pos == dst_node_pos;

                                                                let mut max_move = allow_take_max;
                                                                if let (Some(dpos), Some(rt)) = (dst_node_pos, fctx.lua_rt) {
                                                                    let allowed_dst = if same_node {
                                                                        rt.allow_metadata_inventory_move(
                                                                            dpos,
                                                                            list,
                                                                            slot_idx + 1,
                                                                            &tlist,
                                                                            1,
                                                                            allow_take_max,
                                                                            player_state,
                                                                        )
                                                                    } else {
                                                                        let mut probe = src_snap.clone();
                                                                        probe.count = allow_take_max;
                                                                        rt.allow_metadata_inventory_put(
                                                                            dpos,
                                                                            &tlist,
                                                                            1,
                                                                            &probe,
                                                                            player_state,
                                                                        )
                                                                    };
                                                                    if allowed_dst == 0 {
                                                                        continue;
                                                                    }
                                                                    max_move = max_move.min(allowed_dst);
                                                                }

                                                                let transfer_res = if Arc::ptr_eq(meta, &dst_meta) {
                                                                    let mut g = meta.write();
                                                                    let mut src_vec = g.inventories.entry(list.clone()).or_default().clone();
                                                                    let dst_vec = g.inventories.entry(tlist.clone()).or_insert_with(|| vec![ItemStackData::default(); 36]);
                                                                    let res = shift_transfer_limited(&mut src_vec, slot_idx, dst_vec, 64, max_move);
                                                                    if res.is_some() {
                                                                        g.inventories.insert(list.clone(), src_vec);
                                                                    }
                                                                    res
                                                                } else {
                                                                    let mut src_g = meta.write();
                                                                    let mut dst_g = dst_meta.write();
                                                                    let mut src_vec = src_g.inventories.entry(list.clone()).or_default().clone();
                                                                    let dst_vec = dst_g.inventories.entry(tlist.clone()).or_insert_with(|| vec![ItemStackData::default(); 36]);
                                                                    let res = shift_transfer_limited(&mut src_vec, slot_idx, dst_vec, 64, max_move);
                                                                    if res.is_some() {
                                                                        src_g.inventories.insert(list.clone(), src_vec);
                                                                    }
                                                                    res
                                                                };

                                                                if let Some((dst_slot_idx, moved_stack)) = transfer_res {
                                                                    if let Some(rt) = fctx.lua_rt {
                                                                        if let Some(dpos) = dst_node_pos.filter(|_| same_node) {
                                                                            rt.on_metadata_inventory_move(
                                                                                dpos,
                                                                                list,
                                                                                slot_idx + 1,
                                                                                &tlist,
                                                                                dst_slot_idx + 1,
                                                                                moved_stack.count,
                                                                                player_state,
                                                                            );
                                                                        } else {
                                                                            if let Some(spos) = src_node_pos {
                                                                                rt.on_metadata_inventory_take(
                                                                                    spos,
                                                                                    list,
                                                                                    slot_idx + 1,
                                                                                    &moved_stack,
                                                                                    player_state,
                                                                                );
                                                                            }
                                                                            if let Some(dpos) = dst_node_pos {
                                                                                rt.on_metadata_inventory_put(
                                                                                    dpos,
                                                                                    &tlist,
                                                                                    dst_slot_idx + 1,
                                                                                    &moved_stack,
                                                                                    player_state,
                                                                                );
                                                                            }
                                                                        }
                                                                    }
                                                                    moved = true;
                                                                    transferred_dst_meta = Some(dst_meta);
                                                                    break;
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                                if !moved && loc == "current_player" && list == "main" {
                                                    let mut mg = meta.write();
                                                    if let Some(main_v) = mg.inventories.get_mut("main") {
                                                        moved = shift_transfer_within_main(main_v, slot_idx, 64);
                                                    }
                                                }
                                                if moved {
                                                    handled_transfer = true;
                                                }
                                            }
                                        } else if let Some(cursor) = fctx.cursor.as_deref_mut() {
                                            let btn = if eff_right { SlotClickButton::Right } else { SlotClickButton::Left };
                                            let node_pos = resolve_inv_node_pos(loc, fctx.context_pos);
                                            let old_slot = meta
                                                .read()
                                                .inventories
                                                .get(list)
                                                .and_then(|v| v.get(slot_idx))
                                                .cloned()
                                                .unwrap_or_default();

                                            let allow_click = match (node_pos, fctx.lua_rt) {
                                                (Some(npos), Some(rt)) => {
                                                    if is_output || (!old_slot.is_empty() && (cursor.is_empty() || btn == SlotClickButton::Left && old_slot.name != cursor.held.name)) {
                                                        if !old_slot.is_empty()
                                                            && rt.allow_metadata_inventory_take(npos, list, slot_idx + 1, &old_slot, player_state) == 0
                                                        {
                                                            false
                                                        } else if !is_output && !cursor.is_empty() {
                                                            let mut put_probe = cursor.held.clone();
                                                            if btn == SlotClickButton::Right {
                                                                put_probe.count = 1;
                                                            }
                                                            rt.allow_metadata_inventory_put(npos, list, slot_idx + 1, &put_probe, player_state) > 0
                                                        } else {
                                                            true
                                                        }
                                                    } else if !cursor.is_empty() {
                                                        let mut put_probe = cursor.held.clone();
                                                        if btn == SlotClickButton::Right {
                                                            put_probe.count = 1;
                                                        }
                                                        rt.allow_metadata_inventory_put(npos, list, slot_idx + 1, &put_probe, player_state) > 0
                                                    } else {
                                                        true
                                                    }
                                                }
                                                _ => true,
                                            };

                                            let (changed, new_slot) = if allow_click {
                                                let mut mg = meta.write();
                                                let slots = mg.inventories.entry(list.clone()).or_insert_with(|| vec![ItemStackData::default(); slot_idx + 1]);
                                                let ch = handle_slot_click(cursor, slots, slot_idx, btn, is_output, 64);
                                                let ns = slots.get(slot_idx).cloned().unwrap_or_default();
                                                (ch, ns)
                                            } else {
                                                (false, old_slot.clone())
                                            };

                                            if changed {
                                                if is_craft_out {
                                                    if let Some(reg) = fctx.registries {
                                                        consume_craft_ingredients(meta, reg);
                                                    }
                                                }
                                                if let (Some(npos), Some(rt)) = (node_pos, fctx.lua_rt) {
                                                    if new_slot.name == old_slot.name && new_slot.count > old_slot.count {
                                                        let mut added = new_slot.clone();
                                                        added.count = new_slot.count - old_slot.count;
                                                        rt.on_metadata_inventory_put(npos, list, slot_idx + 1, &added, player_state);
                                                    } else if !new_slot.is_empty() && new_slot.name != old_slot.name {
                                                        if !old_slot.is_empty() {
                                                            rt.on_metadata_inventory_take(npos, list, slot_idx + 1, &old_slot, player_state);
                                                        }
                                                        rt.on_metadata_inventory_put(npos, list, slot_idx + 1, &new_slot, player_state);
                                                    } else if old_slot.count > new_slot.count {
                                                        let mut taken = old_slot.clone();
                                                        taken.count = old_slot.count - new_slot.count;
                                                        rt.on_metadata_inventory_take(npos, list, slot_idx + 1, &taken, player_state);
                                                    }
                                                }
                                                handled_transfer = true;
                                            }
                                        }

                                        if handled_transfer {
                                            if let Some(reg) = fctx.registries {
                                                if meta.read().inventories.contains_key("craft") {
                                                    refresh_craft_preview(meta, reg);
                                                }
                                                if let Some(ref dmeta) = transferred_dst_meta {
                                                    if !Arc::ptr_eq(meta, dmeta)
                                                        && dmeta.read().inventories.contains_key("craft")
                                                    {
                                                        refresh_craft_preview(dmeta, reg);
                                                    }
                                                }
                                            }
                                            theme.play_click();
                                        }
                                    }
                                }
                            }
                        }
                    }
                    FormElement::Checkbox { x, y, name, label, selected } => {
                        let key = format!("{formname}:{name}");
                        let val_s = field_bufs.entry(key).or_insert_with(|| selected.to_string());
                        let mut val = val_s == "true";
                        let r = Rect::from_min_size(origin + Vec2::new(*x * coord_scale, *y * coord_scale), Vec2::new(240.0, 28.0));
                        ui.allocate_new_ui(egui::UiBuilder::new().max_rect(r), |ui| {
                            ui.push_id(idx, |ui| {
                                if ui.checkbox(&mut val, label).changed() {
                                    *val_s = val.to_string();
                                    theme.play_click();
                                    submitted_fields.push((name.clone(), val.to_string()));
                                }
                            });
                        });
                    }
                    FormElement::Dropdown { x, y, w, h, name, items, selected_idx, index_event } => {
                        let key = format!("{formname}:{name}");
                        let cur_idx_s = field_bufs
                            .entry(key)
                            .or_insert_with(|| selected_idx.to_string());
                        let mut cur_idx = cur_idx_s.parse::<usize>().unwrap_or(*selected_idx).clamp(1, items.len().max(1));
                        let r = Rect::from_min_size(
                            origin + Vec2::new(*x * coord_scale, *y * coord_scale),
                            Vec2::new(*w * coord_scale, (*h * coord_scale).max(26.0)),
                        );
                        ui.allocate_new_ui(egui::UiBuilder::new().max_rect(r), |ui| {
                            let cur_lbl = items.get(cur_idx.saturating_sub(1)).map(String::as_str).unwrap_or("");
                            egui::ComboBox::from_id_salt(("fs_dd", idx, name))
                                .selected_text(cur_lbl)
                                .width(r.width())
                                .show_ui(ui, |ui| {
                                    for (i, item_lbl) in items.iter().enumerate() {
                                        let one_based = i + 1;
                                        if ui.selectable_value(&mut cur_idx, one_based, item_lbl).clicked() {
                                            *cur_idx_s = cur_idx.to_string();
                                            theme.play_click();
                                            let emit = if *index_event { cur_idx.to_string() } else { item_lbl.clone() };
                                            submitted_fields.push((name.clone(), emit));
                                        }
                                    }
                                });
                        });
                    }
                    FormElement::TabHeader { x, y, w: _, h, name, captions, current_tab } => {
                        let key = format!("{formname}:{name}");
                        let cur_s = field_bufs.entry(key).or_insert_with(|| current_tab.to_string());
                        let cur_tab = cur_s.parse::<usize>().unwrap_or(*current_tab);
                        let tab_h = (*h * coord_scale).clamp(24.0, 34.0);
                        let mut tx = win_rect.left() + *x * coord_scale;
                        let ty = win_rect.top() + (*y * coord_scale).min(-tab_h + 2.0);
                        for (ti, cap) in captions.iter().enumerate() {
                            let one_based = ti + 1;
                            let tw = (cap.len() as f32 * 9.0 + 22.0).max(76.0);
                            let tr = Rect::from_min_size(Pos2::new(tx, ty), Vec2::new(tw, tab_h));
                            let resp = ui.allocate_rect(tr, egui::Sense::click());
                            let active = cur_tab == one_based;
                            p.rect_filled(tr, 2.0, if active { Color32::from_rgb(198, 198, 198) } else { Color32::from_rgb(130, 130, 130) });
                            p.rect_stroke(tr, 2.0, Stroke::new(1.5, Color32::BLACK), egui::StrokeKind::Inside);
                            p.text(tr.center(), Align2::CENTER_CENTER, cap, egui::FontId::monospace(12.5), Color32::from_rgb(40, 40, 40));
                            if resp.clicked() {
                                *cur_s = one_based.to_string();
                                theme.play_click();
                                submitted_fields.push((name.clone(), one_based.to_string()));
                            }
                            tx += tw + 4.0;
                        }
                    }
                }
            }

            // Rect-based tooltip hit testing
            if hover_tooltip.is_none() {
                if let Some(ptr) = ctx.pointer_hover_pos() {
                    for (tr, txt) in &rect_tooltips {
                        let rr = Rect::from_min_size(
                            origin + Vec2::new(tr[0] * coord_scale, tr[1] * coord_scale),
                            Vec2::new(tr[2] * coord_scale, tr[3] * coord_scale),
                        );
                        if rr.contains(ptr) {
                            hover_tooltip = Some(txt.clone());
                            break;
                        }
                    }
                }
            }

            // Close X button on top-right
            let close_r = Rect::from_min_size(Pos2::new(win_rect.right() - 32.0, win_rect.top() + 6.0), Vec2::splat(26.0));
            let c_resp = ui
                .allocate_new_ui(egui::UiBuilder::new().max_rect(close_r), |ui| mc_button(ui, theme, "X", 26.0, 26.0))
                .inner;
            if c_resp.clicked() {
                close_requested = true;
            }
        });

    if ctx.input(|i| i.key_pressed(egui::Key::Escape)) && fctx.controller_input.is_none() {
        close_requested = true;
    }

    // Floating cursor item & hover tooltip painted on top of everything
    let virtual_ptr = active_ctrl_slot
        .and_then(|idx| slot_centers.get(idx).copied())
        .map(|p| p + Vec2::new(10.0, 10.0));
    if let Some(ptr) = pointer_pos.or(virtual_ptr) {
        let overlay = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Tooltip,
            egui::Id::new(("cubex_fs_tip", fctx.id_salt)),
        ));
        if let Some(cursor) = fctx.cursor.as_deref() {
            if !cursor.is_empty() {
                let c_rect = Rect::from_center_size(ptr, Vec2::splat(34.0));
                let tid = resolve_item_tex_id(
                    ctx,
                    &cursor.held.name,
                    fctx.atlas,
                    fctx.registries,
                    fctx.icon_cache.as_deref_mut(),
                );
                paint_item_in_slot(&overlay, c_rect, tid, &cursor.held.name, cursor.held.count, cursor.held.wear);
            }
        }
        if let Some(ref tip) = hover_tooltip {
            if fctx.cursor.as_deref().map(|c| c.is_empty()).unwrap_or(true) {
                paint_mc_tooltip(&overlay, ctx, ptr, tip);
            }
        }
    }

    // If closing with an item still held on the cursor, safely return it to player's main inventory
    if close_requested {
        if let Some(cursor) = fctx.cursor {
            if !cursor.is_empty() {
                let pmeta = player_state.read().meta.clone();
                let mut pm = pmeta.write();
                let main = pm.inventories.entry("main".into()).or_insert_with(|| vec![ItemStackData::default(); 36]);
                let mut tmp = [std::mem::take(&mut cursor.held)];
                let _ = shift_transfer_slot(&mut tmp, 0, main, 64);
                cursor.clear();
            }
        }
        if has_player_craft {
            let pmeta = player_state.read().meta.clone();
            ensure_craft_grid_size(&pmeta, 4);
            if let Some(reg) = fctx.registries {
                refresh_craft_preview(&pmeta, reg);
            }
        }
        submitted_fields.push(("quit".into(), "true".into()));
    }

    (submitted_fields, close_requested)
}

fn resolve_item_tex_id(
    ctx: &egui::Context,
    item_name: &str,
    atlas: Option<&TextureAtlas>,
    registries: Option<&EngineRegistries>,
    cache: Option<&mut ItemIconCache>,
) -> Option<egui::TextureId> {
    match (cache, atlas, registries) {
        (Some(c), Some(a), Some(r)) => c.get_or_load_item(ctx, item_name, a, r),
        (Some(c), _, _) => c.get_cached(item_name),
        _ => None,
    }
}

fn resolve_texture_id(
    ctx: &egui::Context,
    tex_name: &str,
    atlas: Option<&TextureAtlas>,
    cache: Option<&mut ItemIconCache>,
) -> Option<egui::TextureId> {
    if tex_name.is_empty() {
        return None;
    }
    let key = format!("__tex:{tex_name}");
    match (cache, atlas) {
        (Some(c), Some(a)) => {
            if let Some(id) = c.get_cached(&key) {
                return Some(id);
            }
            let img = a.resolve_spec_immut(tex_name).or_else(|| {
                let base = tex_name.split('^').next().unwrap_or(tex_name).trim();
                a.get_rgba(base).cloned()
            })?;
            Some(c.insert_rgba(ctx, &key, img.as_raw(), img.width(), img.height()))
        }
        (Some(c), None) => c.get_cached(&key),
        _ => None,
    }
}

fn build_item_stack_tooltip(stack: &ItemStackData, registries: Option<&EngineRegistries>) -> String {
    let custom_desc = stack
        .meta
        .read()
        .fields
        .get("description")
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    if let Some(desc) = custom_desc {
        return desc;
    }
    build_item_tooltip(&stack.name, registries)
}

/// Build an interactive 27-slot chest formspec (`nodemeta:x,y,z;main`) paired with the player's
/// 36-slot inventory and `listring[]` Shift-click quick-transfer support.
pub fn build_loot_chest_formspec(pos: IVec3, title: &str) -> String {
    let clean_raw = if title.trim().is_empty() {
        "Loot Chest"
    } else {
        title.trim()
    };
    let clean_title: String = clean_raw
        .chars()
        .filter(|c| !matches!(c, '[' | ']' | ';'))
        .collect();
    format!(
        "formspec_version[4]size[11.75,10.65]\
         label[0.375,0.38;{clean_title}]\
         list[nodemeta:{},{},{};main;0.375,0.85;9,3;0]\
         label[0.375,4.7;Inventory]\
         list[current_player;main;0.375,5.1;9,3;9]\
         list[current_player;main;0.375,9.1;9,1;0]\
         listring[nodemeta:{},{},{};main]\
         listring[current_player;main]\
         button_exit[9.1,0.18;2.25,0.52;close;Close]",
        pos.x, pos.y, pos.z, pos.x, pos.y, pos.z
    )
}

fn build_item_tooltip(item_name: &str, registries: Option<&EngineRegistries>) -> String {
    if let Some(reg) = registries {
        if let Some(def) = reg.get_item_def(item_name) {
            if !def.description.is_empty() {
                return format!("{}\n§8{}", strip_luanti_escapes(&def.description), item_name);
            }
        }
        let id = reg.get_content_id(item_name);
        if let Some(ndef) = reg.get_node_def(id) {
            if !ndef.description.is_empty() {
                return format!("{}\n§8{}", strip_luanti_escapes(&ndef.description), item_name);
            }
        }
    }
    item_name.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{CraftRecipe, ItemDef};

    #[test]
    fn parses_all_required_formspec_elements() {
        let raw = concat!(
            "formspec_version[6]size[11.75,10.425]bgcolor[#222222;true;#101010]",
            "tabheader[0,0;tabs;Survival,Creative;1;false;false]",
            "box[0.2,0.2;2,2;#C6C6C6]label[0.4,0.3;Chest]vertlabel[0.1,1.0;SIDE]",
            "image[6.1,2.0;1.5,1.0;gui_crafting_arrow.png]",
            "item_image[1.0,1.0;1,1;mcl_core:stone]",
            "item_image_button[2.0,1.0;1,1;mcl_core:wood;craft_btn;Wood]",
            "image_button[3.0,1.0;1,1;book.png;guide;]",
            "image_button_exit[4.0,1.0;1,1;close.png;quit;X]",
            "field[0.5,3.0;3,0.8;q;Search;iron]pwdfield[0.5,4.0;3,0.8;pw;Pass]",
            "textarea[4.0,3.0;3,1.5;notes;Notes;hello]",
            "checkbox[0.5,5.0;cb;Auto;true]",
            "dropdown[4.0,5.0;3,0.8;mode;Normal,Fast,Fancy;2;false]",
            "list[context;main;0.375,0.75;9,3;]list[current_player;main;0.375,5.1;9,3;9]",
            "listring[context;main]listring[current_player;main]",
            "tooltip[guide;Recipe Book]",
            "button[1,8;2,0.8;apply;Apply]button_exit[4,8;2,0.8;ok;Done]"
        );
        let parsed = parse_formspec(raw);
        assert_eq!(parsed.version, 6);
        assert_eq!(parsed.size, [11.75, 10.425]);
        assert!(parsed.elements.iter().any(|e| matches!(e, FormElement::BgColor { fullscreen: true, .. })));
        assert!(parsed.elements.iter().any(|e| matches!(e, FormElement::TabHeader { name, captions, .. } if name == "tabs" && captions.len() == 2)));
        assert!(parsed.elements.iter().any(|e| matches!(e, FormElement::VertLabel { text, .. } if text == "SIDE")));
        assert!(parsed.elements.iter().any(|e| matches!(e, FormElement::Image { texture, .. } if texture == "gui_crafting_arrow.png")));
        assert!(parsed.elements.iter().any(|e| matches!(e, FormElement::ItemImage { item_name, .. } if item_name == "mcl_core:stone")));
        assert!(parsed.elements.iter().any(|e| matches!(e, FormElement::ItemImageButton { name, .. } if name == "craft_btn")));
        assert!(parsed.elements.iter().any(|e| matches!(e, FormElement::ImageButton { exit: true, .. })));
        assert!(parsed.elements.iter().any(|e| matches!(e, FormElement::PwdField { name, .. } if name == "pw")));
        assert!(parsed.elements.iter().any(|e| matches!(e, FormElement::TextArea { name, .. } if name == "notes")));
        assert!(parsed.elements.iter().any(|e| matches!(e, FormElement::Checkbox { selected: true, .. })));
        assert!(parsed.elements.iter().any(|e| matches!(e, FormElement::Dropdown { name, items, .. } if name == "mode" && items.len() == 3)));
        assert!(parsed.elements.iter().any(|e| matches!(e, FormElement::ListRing { loc, list } if loc == "context" && list == "main")));
        assert!(parsed.elements.iter().any(|e| matches!(e, FormElement::Tooltip { target: Some(t), .. } if t == "guide")));
    }

    #[test]
    fn slot_click_lmb_rmb_and_shift_transfer_work() {
        let mut slots = vec![ItemStackData::default(); 9];
        slots[0] = ItemStackData::from_spec("mcl_core:stone 10");
        let mut cursor = InventoryCursor::default();

        // RMB picks up half (5)
        assert!(handle_slot_click(&mut cursor, &mut slots, 0, SlotClickButton::Right, false, 64));
        assert_eq!(cursor.held.count, 5);
        assert_eq!(slots[0].count, 5);

        // RMB places 1 in empty slot 1
        assert!(handle_slot_click(&mut cursor, &mut slots, 1, SlotClickButton::Right, false, 64));
        assert_eq!(cursor.held.count, 4);
        assert_eq!(slots[1].count, 1);

        // LMB merges remaining 4 back into slot 0 -> 9
        assert!(handle_slot_click(&mut cursor, &mut slots, 0, SlotClickButton::Left, false, 64));
        assert!(cursor.is_empty());
        assert_eq!(slots[0].count, 9);

        // Output-only blocks placement
        cursor.held = ItemStackData::from_spec("mcl_core:dirt 3");
        assert!(!handle_slot_click(&mut cursor, &mut slots, 2, SlotClickButton::Left, true, 64));
        assert!(slots[2].is_empty());
        cursor.clear();

        // Shift-transfer moves slot 0 into dst inventory
        let mut dst = vec![ItemStackData::default(); 4];
        dst[0] = ItemStackData::from_spec("mcl_core:stone 60");
        assert!(shift_transfer_slot(&mut slots, 0, &mut dst, 64));
        assert!(slots[0].is_empty());
        assert_eq!(dst[0].count, 64);
        assert_eq!(dst[1].count, 5);
    }

    #[test]
    fn craft_preview_and_node_meta_live_sync() {
        let mut reg = EngineRegistries::new();
        reg.register_item(ItemDef {
            name: "mcl_core:tree".into(),
            item_type: "node".into(),
            description: "Oak Log".into(),
            inventory_image: String::new(),
            wield_image: String::new(),
            stack_max: 64,
            groups: HashMap::new(),
            is_node: true,
            tool_capabilities: None,
            range: None,
            liquids_pointable: false,
        });
        reg.crafts.push(CraftRecipe {
            recipe_type: "shapeless".into(),
            output: "mcl_core:wood 4".into(),
            recipe: vec![vec!["mcl_core:tree".into()]],
            cooktime: 0.0,
            burntime: 0.0,
            replacements: Vec::new(),
        });

        let player_state = Arc::new(RwLock::new(ObjectState::new_player("Tester")));
        let meta = player_state.read().meta.clone();
        meta.write().inventories.get_mut("craft").unwrap()[0] = ItemStackData::from_spec("mcl_core:tree 2");
        refresh_craft_preview(&meta, &reg);
        let preview = meta.read().inventories.get("craftpreview").unwrap()[0].clone();
        assert_eq!(preview.name, "mcl_core:wood");
        assert_eq!(preview.count, 4);

        consume_craft_ingredients(&meta, &reg);
        assert_eq!(meta.read().inventories.get("craft").unwrap()[0].count, 1);
        assert_eq!(meta.read().inventories.get("craftpreview").unwrap()[0].count, 4);

        // NodeMetaRef live resolution via WorldStore::get_node_meta
        let tmp = std::env::temp_dir().join("cubex_rs_fs_nodemeta");
        let world = WorldStore::new(tmp, 1);
        let nm = world.get_node_meta(IVec3::new(3, 4, 5));
        nm.write().inventories.insert("main".into(), vec![ItemStackData::from_spec("mcl_core:cobble 12")]);
        let resolved = resolve_inv_meta("nodemeta:3,4,5", None, &player_state, Some(&world)).unwrap();
        assert_eq!(resolved.read().inventories["main"][0].count, 12);
    }

    #[test]
    fn test_right_click_split_place_one_and_2x2_crafting_table_group_wood() {
        let mut reg = EngineRegistries::new();
        let mut oak_groups = HashMap::new();
        oak_groups.insert("wood".into(), 1);
        reg.register_item(ItemDef {
            name: "mcl_trees:wood_oak".into(),
            item_type: "node".into(),
            description: "Oak Planks".into(),
            stack_max: 64,
            groups: oak_groups.clone(),
            is_node: true,
            ..Default::default()
        });
        reg.register_item(ItemDef {
            name: "mcl_trees:wood_birch".into(),
            item_type: "node".into(),
            description: "Birch Planks".into(),
            stack_max: 64,
            groups: oak_groups,
            is_node: true,
            ..Default::default()
        });
        reg.crafts.push(CraftRecipe {
            recipe_type: "normal".into(),
            output: "mcl_crafting_table:crafting_table 1".into(),
            recipe: vec![
                vec!["group:wood".into(), "group:wood".into()],
                vec!["group:wood".into(), "group:wood".into()],
            ],
            cooktime: 0.0,
            burntime: 0.0,
            replacements: Vec::new(),
        });

        let player_state = Arc::new(RwLock::new(ObjectState::new_player("Crafter")));
        let meta = player_state.read().meta.clone();

        // Right-click odd stack (7 planks) -> cursor gets ceil(7/2)=4, slot keeps floor(7/2)=3
        let mut slots = vec![ItemStackData::from_spec("mcl_trees:wood_oak 7"), ItemStackData::default()];
        let mut cursor = InventoryCursor::default();
        assert!(handle_slot_click(&mut cursor, &mut slots, 0, SlotClickButton::Right, false, 64));
        assert_eq!(cursor.held.count, 4);
        assert_eq!(slots[0].count, 3);

        // Right-click even stack (6 planks) -> cursor gets 3, slot keeps 3
        let mut even_slots = vec![ItemStackData::from_spec("mcl_trees:wood_birch 6")];
        let mut even_cursor = InventoryCursor::default();
        assert!(handle_slot_click(&mut even_cursor, &mut even_slots, 0, SlotClickButton::Right, false, 64));
        assert_eq!(even_cursor.held.count, 3);
        assert_eq!(even_slots[0].count, 3);

        // Right-click with held stack places 1 item into empty or matching 2x2 craft slots [0, 1, 2, 3] (craft.len() == 4)
        let mut craft_cursor = InventoryCursor {
            held: ItemStackData::from_spec("mcl_trees:wood_oak 4"),
        };
        {
            let mut mg = meta.write();
            let craft = mg.inventories.get_mut("craft").unwrap();
            assert_eq!(craft.len(), 4);
            for &slot_idx in &[0usize, 1, 2, 3] {
                assert!(handle_slot_click(&mut craft_cursor, craft, slot_idx, SlotClickButton::Right, false, 64));
            }
            assert!(craft_cursor.is_empty());
            // Place one extra matching oak plank in slot 0 via another right-click -> count becomes 2
            craft_cursor.held = ItemStackData::from_spec("mcl_trees:wood_oak 1");
            assert!(handle_slot_click(&mut craft_cursor, craft, 0, SlotClickButton::Right, false, 64));
            assert_eq!(craft[0].count, 2);
            assert!(craft_cursor.is_empty());
        }

        refresh_craft_preview(&meta, &reg);
        let preview = meta.read().inventories["craftpreview"][0].clone();
        assert_eq!(preview.name, "mcl_crafting_table:crafting_table");
        assert_eq!(preview.count, 1);

        consume_craft_ingredients(&meta, &reg);
        {
            let mg = meta.read();
            let craft = &mg.inventories["craft"];
            assert_eq!(craft[0].count, 1, "slot 0 had 2 planks so 1 remains");
            assert!(craft[1].is_empty());
            assert!(craft[2].is_empty());
            assert!(craft[3].is_empty());
        }

        // Mixed group:wood planks in 2x2 [0, 1, 2, 3]
        {
            let mut mg = meta.write();
            let craft = mg.inventories.get_mut("craft").unwrap();
            craft[0] = ItemStackData::from_spec("mcl_trees:wood_oak 1");
            craft[1] = ItemStackData::from_spec("mcl_trees:wood_birch 1");
            craft[2] = ItemStackData::from_spec("mcl_trees:wood_birch 1");
            craft[3] = ItemStackData::from_spec("mcl_trees:wood_oak 1");
        }
        refresh_craft_preview(&meta, &reg);
        let mixed_preview = meta.read().inventories["craftpreview"][0].clone();
        assert_eq!(mixed_preview.name, "mcl_crafting_table:crafting_table");
        assert_eq!(mixed_preview.count, 1);

        // Expand to 3x3, place item in slot 7, then shrink back to 2x2 -> item safely returned to main
        ensure_craft_grid_size(&meta, 9);
        assert_eq!(meta.read().inventories["craft"].len(), 9);
        meta.write().inventories.get_mut("craft").unwrap()[7] =
            ItemStackData::from_spec("mcl_trees:wood_oak 3");
        ensure_craft_grid_size(&meta, 4);
        assert_eq!(meta.read().inventories["craft"].len(), 4);
        assert!(meta.read().inventories["main"]
            .iter()
            .any(|s| s.name == "mcl_trees:wood_oak" && s.count == 3));
    }

    #[test]
    fn test_parse_formspec_strips_luanti_escapes() {
        let raw = concat!(
            "formspec_version[4]size[11.75,10.425]",
            "label[2.25,0.375;\x1b(c@#313131)\x1b(T@mcl_crafting_table)Crafting\x1bE\x1bF]",
            "label[0.375,0.375;\x1b(c@#313131)\x1b(T@mcl_furnaces)Furnace\x1bE]",
            "label[0.375,4.7;\x1b(c@#313131)\x1b(T@mcl_inventory)Inventory\x1bE]",
            "list[current_player;craft;2.25,0.75;3,3;]",
            "tooltip[btn;\x1b(T@mcl_furnaces)Recipe book\x1bE]"
        );
        let parsed = parse_formspec(raw);
        assert!(formspec_requires_3x3_craft(&parsed));
        let labels: Vec<&str> = parsed
            .elements
            .iter()
            .filter_map(|e| match e {
                FormElement::Label { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(labels, vec!["Crafting", "Furnace", "Inventory"]);
        assert!(parsed.elements.iter().any(|e| matches!(
            e,
            FormElement::Tooltip { text, .. } if text == "Recipe book"
        )));
    }
}
