use egui::{Align2, Color32, Rect, Stroke, Vec2};
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::lua_api::userdata::{ItemStackData, ObjectState};
use crate::registry::{
    DrawType, EngineRegistries, CONTENT_AIR, CONTENT_IGNORE, CONTENT_UNKNOWN, RAM_MIC_ITEM_ID,
};
use crate::textures::TextureAtlas;
use crate::ui::formspec::{
    handle_slot_click, ControllerFormInput, InventoryCursor, SlotClickButton,
};
use crate::ui::theme::{
    mc_button, mc_shadow_text, paint_item_in_slot, paint_mc_panel, paint_mc_slot,
    paint_mc_tooltip, strip_luanti_escapes, ItemIconCache, McTheme,
};

pub const CREATIVE_INVENTORY_FORMNAME: &str = "cubex:creative_inventory";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CreativeTab {
    #[default]
    All,
    BuildingBlocks,
    NaturalFoliage,
    ToolsCombat,
    FoodMisc,
    CreatorProps,
}

impl CreativeTab {
    pub const ALL_TABS: [CreativeTab; 6] = [
        CreativeTab::All,
        CreativeTab::BuildingBlocks,
        CreativeTab::NaturalFoliage,
        CreativeTab::ToolsCombat,
        CreativeTab::FoodMisc,
        CreativeTab::CreatorProps,
    ];

    pub fn label(self) -> &'static str {
        match self {
            CreativeTab::All => "All",
            CreativeTab::BuildingBlocks => "Building Blocks",
            CreativeTab::NaturalFoliage => "Natural & Foliage",
            CreativeTab::ToolsCombat => "Tools & Combat",
            CreativeTab::FoodMisc => "Food & Misc",
            CreativeTab::CreatorProps => "Creator Props",
        }
    }

    pub fn compact_label(self) -> &'static str {
        match self {
            CreativeTab::All => "All",
            CreativeTab::BuildingBlocks => "Blocks",
            CreativeTab::NaturalFoliage => "Nature",
            CreativeTab::ToolsCombat => "Tools",
            CreativeTab::FoodMisc => "Food",
            CreativeTab::CreatorProps => "Props",
        }
    }
}

#[derive(Debug, Clone)]
pub struct CreativeCatalogEntry {
    pub id: String,
    pub display_name: String,
    pub search_haystack: String,
    pub tab: CreativeTab,
    pub stack_max: u16,
}

#[derive(Debug, Clone, Default)]
pub struct CreativeCatalog {
    pub entries: Vec<CreativeCatalogEntry>,
}

fn is_internal_hidden_id(id: &str) -> bool {
    let trimmed = id.trim();
    if trimmed.is_empty() {
        return true;
    }
    let short = trimmed.rsplit(':').next().unwrap_or(trimmed);
    matches!(trimmed, "air" | "ignore" | "unknown") || matches!(short, "air" | "ignore" | "unknown")
}

fn clean_display_name(raw_desc: &str, id: &str) -> String {
    for line in raw_desc.lines() {
        let cleaned = strip_luanti_escapes(line).trim().to_string();
        if !cleaned.is_empty() {
            return cleaned;
        }
    }
    let short = id.split(':').nth(1).unwrap_or(id);
    short
        .split('_')
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut chs = part.chars();
            match chs.next() {
                Some(first) => format!("{}{}", first.to_ascii_uppercase(), chs.as_str()),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn contains_any_substr(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| haystack.contains(n))
}

fn classify_creative_tab(
    id: &str,
    item_type: &str,
    groups: &HashMap<String, i32>,
    has_tool_caps: bool,
    is_node: bool,
    drawtype: Option<DrawType>,
) -> CreativeTab {
    let lower = id.to_ascii_lowercase();
    if id == RAM_MIC_ITEM_ID
        || lower.starts_with("cubex:")
        || groups.contains_key("cubex_prop")
        || groups.contains_key("creator_prop")
    {
        return CreativeTab::CreatorProps;
    }

    if has_tool_caps
        || item_type == "tool"
        || groups.contains_key("weapon")
        || groups.contains_key("tool")
        || groups.contains_key("sword")
        || groups.contains_key("pickaxe")
        || groups.contains_key("axe")
        || groups.contains_key("shovel")
        || groups.contains_key("hoe")
        || groups.contains_key("armor")
        || groups.contains_key("bow")
        || groups.contains_key("combat_item")
        || contains_any_substr(
            &lower,
            &[
                "_sword",
                "_pick",
                "_axe",
                "_shovel",
                "_hoe",
                "shears",
                "flint_and_steel",
                "shield",
                ":bow",
                ":arrow",
                "helmet_",
                "chestplate_",
                "leggings_",
                "boots_",
                "trident",
                "crossbow",
                ":tnt",
                "mcl_tnt:",
            ],
        )
    {
        return CreativeTab::ToolsCombat;
    }

    if groups.contains_key("food")
        || groups.contains_key("eatable")
        || groups.contains_key("spawn_egg")
        || groups.contains_key("brewitem")
        || groups.contains_key("dye")
        || groups.contains_key("book")
        || groups.contains_key("music_record")
        || lower.starts_with("mobs_mc:")
        || lower.contains("spawn_egg")
        || contains_any_substr(
            &lower,
            &[
                "bucket",
                "potion",
                "stew",
                "apple",
                "bread",
                "porkchop",
                "beef",
                "mutton",
                "chicken",
                "rabbit",
                "cod",
                "salmon",
                "carrot",
                "potato",
                "melon",
                "cookie",
                "cake",
                "pumpkin_pie",
                "berry",
                "honey",
            ],
        )
        || !is_node
    {
        return CreativeTab::FoodMisc;
    }

    if groups.contains_key("leaves")
        || groups.contains_key("tree")
        || groups.contains_key("sapling")
        || groups.contains_key("flower")
        || groups.contains_key("flora")
        || groups.contains_key("plant")
        || groups.contains_key("grass")
        || groups.contains_key("mushroom")
        || groups.contains_key("soil")
        || groups.contains_key("sand")
        || groups.contains_key("snowy")
        || groups.contains_key("water")
        || groups.contains_key("lava")
        || groups.contains_key("vine")
        || groups.contains_key("seed")
        || groups.contains_key("coral")
        || matches!(
            drawtype,
            Some(
                DrawType::Plantlike
                    | DrawType::Allfaces
                    | DrawType::AllfacesOptional
                    | DrawType::Liquid
                    | DrawType::FlowingLiquid
            )
        )
        || lower.starts_with("mcl_flowers:")
        || lower.starts_with("mcl_trees:leaves")
        || lower.starts_with("mcl_trees:tree")
        || lower.starts_with("mcl_trees:sapling")
        || lower.starts_with("mcl_farming:")
        || contains_any_substr(
            &lower,
            &[
                "dirt",
                "grass",
                "sand",
                "gravel",
                "snow",
                ":ice",
                "_ice",
                "clay",
                "moss",
                "mycelium",
                "podzol",
                "mud",
                "water",
                "lava",
                "kelp",
                "cactus",
                "reeds",
                "bamboo",
                "vine",
                "lily",
            ],
        )
    {
        return CreativeTab::NaturalFoliage;
    }

    CreativeTab::BuildingBlocks
}

impl CreativeCatalog {
    /// Collect every registered node and item from `reg`, excluding internal `air`/`ignore`/`unknown`.
    pub fn build(reg: &EngineRegistries) -> Self {
        let mut seen = HashSet::<String>::new();
        let mut ids = Vec::<String>::new();

        for name in reg.items.keys() {
            if !is_internal_hidden_id(name) && seen.insert(name.clone()) {
                ids.push(name.clone());
            }
        }
        for nd in reg.nodes_by_id.iter().flatten() {
            if matches!(nd.id, CONTENT_AIR | CONTENT_IGNORE | CONTENT_UNKNOWN)
                || is_internal_hidden_id(&nd.name)
            {
                continue;
            }
            if seen.insert(nd.name.clone()) {
                ids.push(nd.name.clone());
            }
        }

        let mut entries = Vec::with_capacity(ids.len());
        for id in ids {
            let item_def = reg.items.get(&id);
            let cid = reg.name_to_id.get(&id).copied();
            let node_def = cid.and_then(|c| reg.get_node_def(c));

            let mut groups: HashMap<String, i32> = node_def
                .map(|n| n.groups.clone())
                .unwrap_or_default();
            if let Some(it) = item_def {
                for (k, &v) in &it.groups {
                    groups.entry(k.clone()).or_insert(v);
                }
            }

            let item_type = item_def.map(|i| i.item_type.as_str()).unwrap_or_else(|| {
                if node_def.is_some() {
                    "node"
                } else {
                    "craft"
                }
            });
            let is_node = node_def.is_some() || item_def.is_some_and(|i| i.is_node);
            let has_tool_caps = item_def.is_some_and(|i| i.tool_capabilities.is_some());
            let drawtype = node_def.map(|n| n.drawtype);

            let raw_desc = item_def
                .map(|i| i.description.as_str())
                .filter(|s| !s.trim().is_empty())
                .or_else(|| node_def.map(|n| n.description.as_str()))
                .unwrap_or("");
            let display_name = clean_display_name(raw_desc, &id);

            let raw_stack_max = item_def
                .map(|i| i.stack_max)
                .filter(|&s| s > 0)
                .or_else(|| node_def.map(|n| n.stack_max).filter(|&s| s > 0))
                .unwrap_or(64);
            let stack_max = if item_type == "tool" {
                1
            } else {
                raw_stack_max.clamp(1, 64)
            };

            let tab = classify_creative_tab(
                &id,
                item_type,
                &groups,
                has_tool_caps,
                is_node,
                drawtype,
            );
            let search_haystack = format!(
                "{} {}",
                display_name.to_ascii_lowercase(),
                id.to_ascii_lowercase()
            );

            entries.push(CreativeCatalogEntry {
                id,
                display_name,
                search_haystack,
                tab,
                stack_max,
            });
        }

        entries.sort_by(|a, b| {
            let ap = if a.tab == CreativeTab::CreatorProps { 0u8 } else { 1u8 };
            let bp = if b.tab == CreativeTab::CreatorProps { 0u8 } else { 1u8 };
            ap.cmp(&bp)
                .then_with(|| a.display_name.to_ascii_lowercase().cmp(&b.display_name.to_ascii_lowercase()))
                .then_with(|| a.id.cmp(&b.id))
        });

        Self { entries }
    }

    pub fn filter(&self, tab: CreativeTab, query: &str) -> Vec<&CreativeCatalogEntry> {
        let tokens: Vec<String> = query
            .split_whitespace()
            .map(|t| t.to_ascii_lowercase())
            .filter(|t| !t.is_empty())
            .collect();

        self.entries
            .iter()
            .filter(|e| tab == CreativeTab::All || e.tab == tab)
            .filter(|e| {
                tokens
                    .iter()
                    .all(|tok| e.search_haystack.contains(tok.as_str()))
            })
            .collect()
    }
}

#[derive(Debug, Clone, Default)]
pub struct CreativeInventoryState {
    pub active_tab: CreativeTab,
    pub search: String,
    pub catalog: CreativeCatalog,
    catalog_sig: (usize, usize),
    pub controller_focus: usize,
}

impl CreativeInventoryState {
    pub fn ensure_synced(&mut self, reg: &EngineRegistries) {
        let sig = (reg.items.len(), reg.name_to_id.len());
        if self.catalog_sig != sig || self.catalog.entries.is_empty() {
            self.catalog = CreativeCatalog::build(reg);
            self.catalog_sig = sig;
        }
    }
}

/// Build a full creative stack (`64` for blocks/stackable items, `1` for unstackables/tools).
pub fn creative_full_stack(entry: &CreativeCatalogEntry) -> ItemStackData {
    let mut stack = ItemStackData::from_spec(&entry.id);
    stack.count = entry.stack_max.clamp(1, 64);
    stack
}

/// Place a full creative stack into the player's hotbar (`0..9`), or first empty slot (`9..36`),
/// falling back to overwriting `preferred_slot.min(8)`.
pub fn creative_grab_to_hotbar(
    player_state: &Arc<RwLock<ObjectState>>,
    entry: &CreativeCatalogEntry,
    preferred_slot: usize,
) -> bool {
    let full = creative_full_stack(entry);
    let ps = player_state.read();
    let mut mg = ps.meta.write();
    let main = mg
        .inventories
        .entry("main".into())
        .or_insert_with(|| vec![ItemStackData::default(); 36]);
    if main.len() < 36 {
        main.resize_with(36, ItemStackData::default);
    }

    // 1. Top up existing hotbar matching stack if stackable and non-full
    if full.count > 1 {
        for slot in main.iter_mut().take(9) {
            if !slot.is_empty() && slot.name == full.name && slot.wear == 0 && slot.count < full.count
            {
                slot.count = full.count;
                return true;
            }
        }
    }
    // 2. If unstackable / already present on hotbar at full count, keep cleanly without spam
    if full.count == 1 && main.iter().take(9).any(|s| !s.is_empty() && s.name == full.name) {
        return true;
    }
    // 3. First empty hotbar slot (0..9)
    for slot in main.iter_mut().take(9) {
        if slot.is_empty() {
            *slot = full;
            return true;
        }
    }
    // 4. First empty main slot (9..36)
    for slot in main.iter_mut().skip(9) {
        if slot.is_empty() {
            *slot = full;
            return true;
        }
    }
    // 5. Fallback: replace active hotbar slot
    let idx = preferred_slot.min(8);
    main[idx] = full;
    true
}

pub fn creative_catalog_left_click(
    cursor: &mut InventoryCursor,
    player_state: &Arc<RwLock<ObjectState>>,
    entry: &CreativeCatalogEntry,
    shift: bool,
    preferred_slot: usize,
) {
    if shift {
        let _ = creative_grab_to_hotbar(player_state, entry, preferred_slot);
    } else {
        cursor.held = creative_full_stack(entry);
    }
}

pub fn creative_catalog_right_click(
    cursor: &mut InventoryCursor,
    entry: &CreativeCatalogEntry,
) {
    let max_c = entry.stack_max.clamp(1, 64);
    if cursor.is_empty() || cursor.held.name != entry.id {
        let mut st = ItemStackData::from_spec(&entry.id);
        st.count = 1;
        cursor.held = st;
    } else if cursor.held.count < max_c {
        cursor.held.count += 1;
    }
}

#[allow(clippy::too_many_arguments)]
pub fn render_creative_inventory(
    ctx: &egui::Context,
    theme: &McTheme,
    state: &mut CreativeInventoryState,
    player_state: &Arc<RwLock<ObjectState>>,
    selected_hotbar_slot: usize,
    registries: &EngineRegistries,
    atlas: &TextureAtlas,
    icon_cache: &mut ItemIconCache,
    cursor: &mut InventoryCursor,
    viewport: Rect,
    id_salt: u64,
    controller_input: Option<ControllerFormInput>,
) -> bool {
    state.ensure_synced(registries);

    let mut close_requested = false;
    let shift_down = ctx.input(|i| i.modifiers.shift);
    let left_clicked = ctx.input(|i| i.pointer.button_pressed(egui::PointerButton::Primary));
    let right_clicked = ctx.input(|i| i.pointer.button_pressed(egui::PointerButton::Secondary));
    let pointer_pos = ctx
        .input(|i| i.pointer.interact_pos().or_else(|| i.pointer.latest_pos()))
        .filter(|p| viewport.contains(*p));

    // Backdrop strictly inside player's viewport
    egui::Area::new(egui::Id::new(("cubex_creative_backdrop", id_salt)))
        .fixed_pos(viewport.min)
        .show(ctx, |ui| {
            ui.painter()
                .rect_filled(viewport, 0.0, Color32::from_rgba_unmultiplied(0, 0, 0, 155));
        });

    let max_w = (viewport.width() - 16.0).max(260.0);
    let max_h = (viewport.height() - 18.0).max(220.0);
    let win_w = (viewport.width() * 0.72).clamp(320.0, 620.0).min(max_w);
    let win_h = (viewport.height() * 0.80).clamp(260.0, 560.0).min(max_h);
    let win_rect = Rect::from_center_size(viewport.center(), Vec2::new(win_w, win_h));
    let pad = 12.0f32;
    let inner_w = (win_rect.width() - pad * 2.0).max(200.0);
    let compact_tabs = inner_w < 490.0;

    let mut hover_tooltip: Option<String> = None;

    egui::Area::new(egui::Id::new(("cubex_creative_window", id_salt)))
        .fixed_pos(win_rect.min)
        .show(ctx, |ui| {
            ui.set_clip_rect(viewport);
            paint_mc_panel(ui.painter(), win_rect);

            let content_rect = win_rect.shrink(pad);
            let mut inner_ui = ui.new_child(egui::UiBuilder::new().max_rect(content_rect));
            inner_ui.set_clip_rect(content_rect);

            // Header bar: Title + Search + Done button
            let header_h = 30.0f32;
            let done_w = if compact_tabs { 58.0 } else { 76.0 };
            let clear_w = 24.0;
            let title_w = if compact_tabs { 112.0 } else { 172.0 };
            let search_w = (inner_w - title_w - done_w - clear_w - 18.0).max(70.0);

            inner_ui.horizontal(|ui| {
                ui.set_height(header_h);
                ui.allocate_ui(Vec2::new(title_w, header_h), |ui| {
                    ui.vertical_centered_justified(|ui| {
                        ui.label(
                            egui::RichText::new("Creative Catalog")
                                .color(Color32::from_rgb(38, 38, 38))
                                .strong()
                                .size(if compact_tabs { 13.0 } else { 15.0 }),
                        );
                    });
                });
                ui.add_sized(
                    [search_w, 26.0],
                    egui::TextEdit::singleline(&mut state.search)
                        .hint_text("Search items & blocks...")
                        .id_source(("cubex_creative_search", id_salt)),
                );
                if mc_button(ui, theme, "X", clear_w, 26.0).clicked() {
                    state.search.clear();
                }
                if mc_button(ui, theme, "Done", done_w, 26.0).clicked() {
                    close_requested = true;
                }
            });

            inner_ui.add_space(4.0);

            // Filter catalog after search input update
            let filtered = state.catalog.filter(state.active_tab, &state.search);
            let filtered_count = filtered.len();
            let total_count = state.catalog.entries.len();

            // Category Tabs Row
            let tab_count = CreativeTab::ALL_TABS.len() as f32;
            let tab_gap = 4.0;
            let tab_w = ((inner_w - tab_gap * (tab_count - 1.0)) / tab_count).max(40.0);
            inner_ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = tab_gap;
                for &tab in &CreativeTab::ALL_TABS {
                    let active = state.active_tab == tab;
                    let lbl = if compact_tabs {
                        tab.compact_label()
                    } else {
                        tab.label()
                    };
                    let (trect, tresp) =
                        ui.allocate_exact_size(Vec2::new(tab_w, 26.0), egui::Sense::click());
                    let bg = if active {
                        Color32::from_rgb(92, 134, 62)
                    } else if tresp.hovered() {
                        Color32::from_rgb(138, 138, 146)
                    } else {
                        Color32::from_rgb(110, 110, 114)
                    };
                    let p = ui.painter();
                    p.rect_filled(trect, 2.0, bg);
                    p.rect_stroke(
                        trect,
                        2.0,
                        Stroke::new(
                            if active { 1.5 } else { 1.0 },
                            if active {
                                Color32::WHITE
                            } else {
                                Color32::from_rgb(45, 45, 48)
                            },
                        ),
                        egui::StrokeKind::Inside,
                    );
                    mc_shadow_text(
                        p,
                        trect.center(),
                        Align2::CENTER_CENTER,
                        lbl,
                        if compact_tabs { 10.5 } else { 11.5 },
                        Color32::WHITE,
                    );
                    if tresp.clicked() {
                        theme.play_click();
                        state.active_tab = tab;
                        state.controller_focus = 0;
                    }
                }
            });

            // Handle controller actions on Creative screen (navigation, grab, tab cycle, close)
            if let Some(ci) = controller_input {
                if ci.cancel {
                    close_requested = true;
                } else if ci.shift {
                    let cur_idx = CreativeTab::ALL_TABS
                        .iter()
                        .position(|&t| t == state.active_tab)
                        .unwrap_or(0);
                    state.active_tab =
                        CreativeTab::ALL_TABS[(cur_idx + 1) % CreativeTab::ALL_TABS.len()];
                    state.controller_focus = 0;
                }
            }

            // Status count & usage hint bar
            inner_ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(format!(
                        "{} ({filtered_count}/{total_count}) • LMB: Grab 64 • Shift+Click: To Hotbar",
                        state.active_tab.label()
                    ))
                    .color(Color32::from_rgb(54, 54, 54))
                    .size(if compact_tabs { 10.5 } else { 11.5 }),
                );
            });
            inner_ui.add_space(4.0);

            // Compute grid & bottom hotbar dimensions
            let hotbar_gap = 4.0f32;
            let trash_w = if compact_tabs { 32.0 } else { 42.0 };
            let hotbar_slot_sz = ((inner_w - hotbar_gap * 9.0 - trash_w - 6.0) / 9.0)
                .clamp(24.0, 44.0)
                .floor();
            let hotbar_section_h = hotbar_slot_sz + 24.0;

            let slot_gap = 4.0f32;
            let cat_slot_sz = if compact_tabs { 36.0 } else { 44.0 };
            let scroll_bar_reserve = 14.0f32;
            let cols = (((inner_w - scroll_bar_reserve) + slot_gap) / (cat_slot_sz + slot_gap))
                .floor()
                .clamp(5.0, 11.0) as usize;
            let row_h = cat_slot_sz + slot_gap;
            let avail_scroll_h = (inner_ui.available_height() - hotbar_section_h - 6.0).max(76.0);

            if filtered_count > 0 {
                state.controller_focus = state.controller_focus.min(filtered_count - 1);
            } else {
                state.controller_focus = 0;
            }
            if let Some(ci) = controller_input {
                if filtered_count > 0 {
                    if ci.move_x > 0 {
                        state.controller_focus = (state.controller_focus + 1) % filtered_count;
                    } else if ci.move_x < 0 {
                        state.controller_focus = if state.controller_focus == 0 {
                            filtered_count - 1
                        } else {
                            state.controller_focus - 1
                        };
                    } else if ci.move_y > 0 {
                        state.controller_focus =
                            (state.controller_focus + cols).min(filtered_count - 1);
                    } else if ci.move_y < 0 {
                        state.controller_focus = state.controller_focus.saturating_sub(cols);
                    }
                    if ci.confirm {
                        let entry = filtered[state.controller_focus];
                        let _ = creative_grab_to_hotbar(player_state, entry, selected_hotbar_slot);
                    }
                }
            }

            // Virtualized ScrollArea rendering only visible rows so icon textures load lazily
            let total_rows = filtered_count.div_ceil(cols).max(1);
            let grid_bg_rect = Rect::from_min_size(
                inner_ui.cursor().min,
                Vec2::new(inner_w, avail_scroll_h),
            );
            inner_ui
                .painter()
                .rect_filled(grid_bg_rect, 1.0, Color32::from_rgba_unmultiplied(72, 72, 76, 85));

            egui::ScrollArea::vertical()
                .id_salt(("cubex_creative_scroll", id_salt))
                .max_height(avail_scroll_h)
                .min_scrolled_height(avail_scroll_h)
                .auto_shrink([false, false])
                .show_rows(&mut inner_ui, row_h, total_rows, |ui, row_range| {
                    if filtered_count == 0 {
                        let empty_r = Rect::from_min_size(
                            ui.cursor().min,
                            Vec2::new(inner_w - scroll_bar_reserve, row_h * 2.0),
                        );
                        ui.painter().text(
                            empty_r.center(),
                            Align2::CENTER_CENTER,
                            "No matching items or blocks",
                            egui::FontId::monospace(13.0),
                            Color32::from_rgb(235, 235, 235),
                        );
                        return;
                    }
                    let mut acted = false;
                    for row in row_range {
                        let start_idx = row * cols;
                        let end_idx = (start_idx + cols).min(filtered_count);
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = slot_gap;
                            ui.set_height(cat_slot_sz);
                            for idx in start_idx..end_idx {
                                let entry = filtered[idx];
                                let (srect, resp) = ui.allocate_exact_size(
                                    Vec2::splat(cat_slot_sz),
                                    egui::Sense::click(),
                                );
                                let is_ctrl_focus = controller_input.is_some()
                                    && idx == state.controller_focus;
                                let ptr_hit =
                                    pointer_pos.map_or(resp.hovered(), |pos| srect.contains(pos));
                                let hovered = resp.hovered() || is_ctrl_focus;
                                let p = ui.painter();
                                paint_mc_slot(p, srect, hovered);
                                if is_ctrl_focus {
                                    p.rect_stroke(
                                        srect.expand(1.5),
                                        0.0,
                                        Stroke::new(2.0, Color32::from_rgb(255, 220, 64)),
                                        egui::StrokeKind::Inside,
                                    );
                                }
                                let tid = icon_cache.get_or_load_item(
                                    ctx,
                                    &entry.id,
                                    atlas,
                                    registries,
                                );
                                paint_item_in_slot(p, srect, tid, &entry.id, 1, 0);

                                if hovered {
                                    hover_tooltip =
                                        Some(format!("{}\n§8{}", entry.display_name, entry.id));
                                }

                                if !acted && ptr_hit && left_clicked {
                                    acted = true;
                                    theme.play_click();
                                    creative_catalog_left_click(
                                        cursor,
                                        player_state,
                                        entry,
                                        shift_down,
                                        selected_hotbar_slot,
                                    );
                                } else if !acted && ptr_hit && right_clicked {
                                    acted = true;
                                    theme.play_click();
                                    creative_catalog_right_click(cursor, entry);
                                }
                            }
                        });
                        ui.add_space(slot_gap);
                    }
                });

            inner_ui.add_space(6.0);

            // Pinned Hotbar (Slots 1..9) + Trash/Clear Cursor button
            inner_ui.label(
                egui::RichText::new("Hotbar (1-9)")
                    .color(Color32::from_rgb(54, 54, 54))
                    .size(11.5),
            );
            inner_ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = hotbar_gap;
                let ps_r = player_state.read();
                let mut mg = ps_r.meta.write();
                let main_inv = mg
                    .inventories
                    .entry("main".into())
                    .or_insert_with(|| vec![ItemStackData::default(); 36]);
                if main_inv.len() < 36 {
                    main_inv.resize_with(36, ItemStackData::default);
                }

                let mut hotbar_acted = false;
                for slot_idx in 0..9usize {
                    let (srect, resp) =
                        ui.allocate_exact_size(Vec2::splat(hotbar_slot_sz), egui::Sense::click());
                    let ptr_hit = pointer_pos.map_or(resp.hovered(), |pos| srect.contains(pos));
                    let p = ui.painter();
                    paint_mc_slot(p, srect, resp.hovered());
                    if slot_idx == selected_hotbar_slot {
                        p.rect_stroke(
                            srect.expand(1.5),
                            0.0,
                            Stroke::new(2.0, Color32::WHITE),
                            egui::StrokeKind::Inside,
                        );
                    }

                    let item = &main_inv[slot_idx];
                    if !item.is_empty() {
                        let tid =
                            icon_cache.get_or_load_item(ctx, &item.name, atlas, registries);
                        paint_item_in_slot(p, srect, tid, &item.name, item.count, item.wear);
                        if resp.hovered() {
                            let label = registries
                                .get_item_def(&item.name)
                                .map(|d| clean_display_name(&d.description, &item.name))
                                .unwrap_or_else(|| clean_display_name("", &item.name));
                            hover_tooltip = Some(format!("{label}\n§8{}", item.name));
                        }
                    }

                    if !hotbar_acted && ptr_hit && (left_clicked || right_clicked) {
                        hotbar_acted = true;
                        theme.play_click();
                        if shift_down && left_clicked {
                            main_inv[slot_idx] = ItemStackData::default();
                        } else {
                            let max_c = registries
                                .get_item_def(&main_inv[slot_idx].name)
                                .or_else(|| {
                                    (!cursor.held.is_empty())
                                        .then(|| registries.get_item_def(&cursor.held.name))
                                        .flatten()
                                })
                                .map(|d| d.stack_max)
                                .unwrap_or(64)
                                .clamp(1, 64);
                            let btn = if right_clicked {
                                SlotClickButton::Right
                            } else {
                                SlotClickButton::Left
                            };
                            let _ = handle_slot_click(cursor, main_inv, slot_idx, btn, false, max_c);
                        }
                    }
                }

                // Red Trash / Clear Cursor slot (Shift+Click also clears active hotbar or all empty)
                let (trect, tresp) = ui.allocate_exact_size(
                    Vec2::new(trash_w, hotbar_slot_sz),
                    egui::Sense::click(),
                );
                let ptr_hit = pointer_pos.map_or(tresp.hovered(), |pos| trect.contains(pos));
                let tp = ui.painter();
                tp.rect_filled(
                    trect,
                    2.0,
                    if tresp.hovered() {
                        Color32::from_rgb(190, 56, 56)
                    } else {
                        Color32::from_rgb(138, 42, 42)
                    },
                );
                tp.rect_stroke(
                    trect,
                    2.0,
                    Stroke::new(1.5, Color32::from_rgb(48, 16, 16)),
                    egui::StrokeKind::Inside,
                );
                mc_shadow_text(
                    tp,
                    trect.center(),
                    Align2::CENTER_CENTER,
                    "X",
                    13.5,
                    Color32::WHITE,
                );
                if tresp.hovered() {
                    hover_tooltip =
                        Some("Clear Item\n§7Click to clear cursor (Shift: clear hotbar)".into());
                }
                if ptr_hit && (left_clicked || right_clicked) {
                    theme.play_click();
                    cursor.clear();
                    if shift_down {
                        for slot in main_inv.iter_mut().take(9) {
                            *slot = ItemStackData::default();
                        }
                    }
                }
            });
        });

    // Cursor stack & hover tooltip on Tooltip layer above Creative Catalog window
    if let Some(ptr) = pointer_pos {
        let overlay = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Tooltip,
            egui::Id::new(("cubex_creative_overlay", id_salt)),
        ));
        if !cursor.is_empty() {
            let crect = Rect::from_center_size(ptr, Vec2::splat(34.0));
            let tid = icon_cache.get_or_load_item(ctx, &cursor.held.name, atlas, registries);
            paint_item_in_slot(
                &overlay,
                crect,
                tid,
                &cursor.held.name,
                cursor.held.count,
                cursor.held.wear,
            );
        }
        if let Some(ref tip) = hover_tooltip {
            if cursor.is_empty() {
                paint_mc_tooltip(&overlay, ctx, ptr, tip);
            }
        }
    }

    if close_requested {
        cursor.clear();
    }
    close_requested
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{ItemDef, NodeDef, ToolCapabilities};
    use egui::Pos2;

    #[test]
    fn creative_catalog_exposes_all_registered_nodes_items_and_ram_mic() {
        let mut reg = EngineRegistries::new();
        reg.register_node(NodeDef {
            name: "mcl_core:stone".into(),
            description: "Stone".into(),
            drawtype: DrawType::Normal,
            ..Default::default()
        });
        reg.register_node(NodeDef {
            name: "mcl_flowers:dandelion".into(),
            description: "Dandelion".into(),
            drawtype: DrawType::Plantlike,
            ..Default::default()
        });
        reg.register_item(ItemDef {
            name: "mcl_tools:sword_diamond".into(),
            item_type: "tool".into(),
            description: "Diamond Sword".into(),
            stack_max: 1,
            tool_capabilities: Some(ToolCapabilities::default()),
            ..Default::default()
        });
        reg.register_item(ItemDef {
            name: "mcl_core:apple".into(),
            item_type: "craft".into(),
            description: "Apple".into(),
            stack_max: 64,
            groups: HashMap::from([("food".into(), 2)]),
            ..Default::default()
        });

        let catalog = CreativeCatalog::build(&reg);
        let ids: HashSet<&str> = catalog.entries.iter().map(|e| e.id.as_str()).collect();
        assert!(!ids.contains("air"));
        assert!(!ids.contains("ignore"));
        assert!(!ids.contains("unknown"));
        assert!(ids.contains("mcl_core:stone"));
        assert!(ids.contains("mcl_flowers:dandelion"));
        assert!(ids.contains("mcl_tools:sword_diamond"));
        assert!(ids.contains("mcl_core:apple"));
        assert!(ids.contains(RAM_MIC_ITEM_ID));

        let props = catalog.filter(CreativeTab::CreatorProps, "");
        assert!(props.iter().any(|e| e.id == RAM_MIC_ITEM_ID));
        assert!(catalog
            .filter(CreativeTab::BuildingBlocks, "")
            .iter()
            .any(|e| e.id == "mcl_core:stone"));
        assert!(catalog
            .filter(CreativeTab::NaturalFoliage, "")
            .iter()
            .any(|e| e.id == "mcl_flowers:dandelion"));
        assert!(catalog
            .filter(CreativeTab::ToolsCombat, "")
            .iter()
            .any(|e| e.id == "mcl_tools:sword_diamond"));
        assert!(catalog
            .filter(CreativeTab::FoodMisc, "")
            .iter()
            .any(|e| e.id == "mcl_core:apple"));
    }

    #[test]
    fn creative_catalog_search_and_grab_actions_work() {
        let mut reg = EngineRegistries::new();
        reg.register_node(NodeDef {
            name: "mcl_core:stone".into(),
            description: "Cobble Stone".into(),
            drawtype: DrawType::Normal,
            ..Default::default()
        });
        reg.register_node(NodeDef {
            name: "mcl_core:glass".into(),
            description: "Glass".into(),
            drawtype: DrawType::Glasslike,
            ..Default::default()
        });

        let mut p1_state = CreativeInventoryState::default();
        let mut p2_state = CreativeInventoryState::default();
        p1_state.ensure_synced(&reg);
        p2_state.ensure_synced(&reg);
        p1_state.search = "ram mic".into();
        p2_state.search = "stone".into();
        p2_state.active_tab = CreativeTab::BuildingBlocks;

        let p1_hits = p1_state.catalog.filter(p1_state.active_tab, &p1_state.search);
        assert_eq!(p1_hits.len(), 1);
        assert_eq!(p1_hits[0].id, RAM_MIC_ITEM_ID);

        let p2_hits = p2_state.catalog.filter(p2_state.active_tab, &p2_state.search);
        assert_eq!(p2_hits.len(), 1);
        assert_eq!(p2_hits[0].id, "mcl_core:stone");

        let p1 = Arc::new(RwLock::new(ObjectState::new_player("p1")));
        let mut cursor = InventoryCursor::default();
        // Left-click stone gives 64 on cursor
        creative_catalog_left_click(&mut cursor, &p1, p2_hits[0], false, 0);
        assert_eq!(cursor.held.name, "mcl_core:stone");
        assert_eq!(cursor.held.count, 64);

        // Right-click glass gives 1 on cursor
        let glass = p1_state
            .catalog
            .filter(CreativeTab::All, "mcl_core:glass")
            .into_iter()
            .next()
            .unwrap();
        creative_catalog_right_click(&mut cursor, glass);
        assert_eq!(cursor.held.name, "mcl_core:glass");
        assert_eq!(cursor.held.count, 1);

        // Shift+click stone places full stack 64 into first open hotbar slot
        creative_catalog_left_click(&mut cursor, &p1, p2_hits[0], true, 0);
        {
            let guard = p1.read();
            let mg = guard.meta.read();
            let main = mg.inventories.get("main").unwrap();
            assert_eq!(main[0].name, "mcl_core:stone");
            assert_eq!(main[0].count, 64);
        }

        // Shift+click RAM mic places unstackable 1 into next hotbar slot
        creative_catalog_left_click(&mut cursor, &p1, p1_hits[0], true, 0);
        let guard = p1.read();
        let mg = guard.meta.read();
        let main = mg.inventories.get("main").unwrap();
        assert_eq!(main[1].name, RAM_MIC_ITEM_ID);
        assert_eq!(main[1].count, 1);
    }

    #[test]
    fn creative_inventory_renders_cleanly_singleplayer_and_splitscreen() {
        let mut reg = EngineRegistries::new();
        for i in 0..40 {
            reg.register_node(NodeDef {
                name: format!("test:block_{i}"),
                description: format!("Block {i}"),
                drawtype: DrawType::Normal,
                ..Default::default()
            });
        }
        let atlas = TextureAtlas::placeholder();
        let ctx = egui::Context::default();
        let theme = McTheme::load(&ctx);
        let p1 = Arc::new(RwLock::new(ObjectState::new_player("p1")));
        let p2 = Arc::new(RwLock::new(ObjectState::new_player("p2")));
        let mut s1 = CreativeInventoryState::default();
        let mut s2 = CreativeInventoryState::default();
        s2.search = "block".into();
        let mut ic1 = ItemIconCache::new();
        let mut ic2 = ItemIconCache::new();
        let mut c1 = InventoryCursor::default();
        let mut c2 = InventoryCursor::default();

        // Singleplayer 1280x720
        let _ = ctx.run(
            egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 720.0))),
                ..Default::default()
            },
            |ctx| {
                let closed = render_creative_inventory(
                    ctx,
                    &theme,
                    &mut s1,
                    &p1,
                    0,
                    &reg,
                    &atlas,
                    &mut ic1,
                    &mut c1,
                    Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 720.0)),
                    1,
                    None,
                );
                assert!(!closed);
            },
        );

        // Split-screen side-by-side P1 + P2 (640x720 each)
        let _ = ctx.run(
            egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 720.0))),
                ..Default::default()
            },
            |ctx| {
                let _ = render_creative_inventory(
                    ctx,
                    &theme,
                    &mut s1,
                    &p1,
                    0,
                    &reg,
                    &atlas,
                    &mut ic1,
                    &mut c1,
                    Rect::from_min_size(Pos2::ZERO, Vec2::new(640.0, 720.0)),
                    10,
                    None,
                );
                let _ = render_creative_inventory(
                    ctx,
                    &theme,
                    &mut s2,
                    &p2,
                    2,
                    &reg,
                    &atlas,
                    &mut ic2,
                    &mut c2,
                    Rect::from_min_size(Pos2::new(640.0, 0.0), Vec2::new(640.0, 720.0)),
                    20,
                    None,
                );
            },
        );
    }
}
