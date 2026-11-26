use base64::Engine as _;
use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use glam::{IVec3, Vec3};
use mlua::{Lua, MultiValue, Result as LuaResult, Table, Value};
use parking_lot::RwLock;
use sha1::{Digest, Sha1};
use std::collections::{HashMap, HashSet};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use crate::config::MinetestConfig;
use crate::paths::EnginePaths;
use crate::registry::{
    compute_dig_params_with_wear, BiomeDef, CraftRecipe, DecorationDef, DrawType, DropItemEntry,
    EngineRegistries, ItemDef, NodeBoxSpec, NodeDef, NodeDrop, NodeSounds, OreDef, SoundSpec,
    TileSpec, ToolCapabilities, ToolGroupCap, CONTENT_AIR, CONTENT_IGNORE,
};
use crate::world::schematic::{
    PlaceFlags, Schematic, SchematicNode, SchematicRotation, MTSCHEM_PROB_ALWAYS,
};
use crate::world::WorldStore;

use super::noise::{NoiseParams, PerlinNoiseMapUD, PerlinNoiseUD};
use super::userdata::{
    cubex_to_lua_pos, lua_to_cubex_pos, object_ref_to_lua, InvRefUD, ItemStackData, ItemStackUD,
    MetaDataRefUD, MetaDataState, NodeTimerRefUD, ObjectState, SettingsUD, VoxelManipUD,
};

#[derive(Debug, Clone)]
pub struct FormspecRequest {
    pub player_name: String,
    pub formname: String,
    pub formspec: String,
}

#[derive(Debug, Clone)]
pub struct PendingSoundEvent {
    pub spec: SoundSpec,
    pub pos: Option<Vec3>,
    pub gain: f32,
}

#[derive(Debug, Clone)]
pub struct CraftOutput {
    pub item: ItemStackData,
    pub time: f32,
    pub decrements: Vec<ItemStackData>,
    pub replacements: Vec<ItemStackData>,
}

fn ingredient_matches(pattern: &str, item_name: &str, reg: &EngineRegistries) -> bool {
    let pat = pattern.trim();
    if pat.is_empty() {
        return item_name.is_empty();
    }
    if item_name.is_empty() {
        return false;
    }
    let resolved_item = reg.resolve_alias(item_name);
    if let Some(grp_spec) = pat.strip_prefix("group:") {
        let item_def = reg.get_item_def(item_name).or_else(|| reg.get_item_def(resolved_item));
        let node_id = reg.get_content_id(item_name);
        let node_def = (node_id != crate::registry::CONTENT_UNKNOWN)
            .then(|| reg.get_node_def(node_id))
            .flatten();
        if item_def.is_none() && node_def.is_none() {
            return false;
        }
        return grp_spec
            .split(',')
            .map(str::trim)
            .filter(|g| !g.is_empty())
            .all(|g| {
                item_def.and_then(|d| d.groups.get(g).copied()).unwrap_or(0) > 0
                    || node_def.and_then(|d| d.groups.get(g).copied()).unwrap_or(0) > 0
            });
    }
    let resolved_pat = reg.resolve_alias(pat);
    resolved_pat == resolved_item || pat == item_name
}

fn trim_shaped_recipe(rows: &[Vec<String>]) -> (usize, usize, Vec<Vec<&str>>) {
    let raw_h = rows.len();
    let raw_w = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    if raw_h == 0 || raw_w == 0 {
        return (0, 0, Vec::new());
    }
    let mut min_r = raw_h;
    let mut max_r = 0usize;
    let mut min_c = raw_w;
    let mut max_c = 0usize;
    for (r, row) in rows.iter().enumerate() {
        for (c, cell) in row.iter().enumerate() {
            if !cell.trim().is_empty() {
                min_r = min_r.min(r);
                max_r = max_r.max(r);
                min_c = min_c.min(c);
                max_c = max_c.max(c);
            }
        }
    }
    if min_r > max_r || min_c > max_c {
        return (0, 0, Vec::new());
    }
    let h = max_r - min_r + 1;
    let w = max_c - min_c + 1;
    let mut trimmed = Vec::with_capacity(h);
    for r in min_r..=max_r {
        let mut row_out = Vec::with_capacity(w);
        for c in min_c..=max_c {
            let s = rows[r].get(c).map(|v| v.trim()).unwrap_or("");
            row_out.push(s);
        }
        trimmed.push(row_out);
    }
    (w, h, trimmed)
}

fn match_shapeless_backtrack(
    reqs: &[&str],
    inputs: &[&str],
    used: &mut [bool],
    idx: usize,
    reg: &EngineRegistries,
) -> bool {
    if idx >= reqs.len() {
        return true;
    }
    let pat = reqs[idx];
    for (i, &item) in inputs.iter().enumerate() {
        if !used[i] && ingredient_matches(pat, item, reg) {
            used[i] = true;
            if match_shapeless_backtrack(reqs, inputs, used, idx + 1, reg) {
                return true;
            }
            used[i] = false;
        }
    }
    false
}

/// 1:1 Luanti recipe matcher supporting 2x2/3x3 shaped, shapeless, cooking, fuel, and toolrepair.
pub fn resolve_craft_recipe(
    reg: &EngineRegistries,
    grid: &[ItemStackData],
    width: usize,
    method: &str,
) -> Option<CraftOutput> {
    if method == "cooking" || method == "fuel" {
        let non_empty_idx = grid.iter().position(|s| !s.is_empty())?;
        let item_name = grid[non_empty_idx].name.as_str();
        for r in &reg.crafts {
            if r.recipe_type != method {
                continue;
            }
            let Some(pat) = r.recipe.first().and_then(|v| v.first()).map(String::as_str) else {
                continue;
            };
            if ingredient_matches(pat, item_name, reg) {
                let mut decrements = grid.to_vec();
                let mut replacements = Vec::new();
                if let Some(slot) = decrements.get_mut(non_empty_idx) {
                    let rep_opt = r
                        .replacements
                        .iter()
                        .find(|(from, _)| ingredient_matches(from, &slot.name, reg))
                        .map(|(_, rep)| ItemStackData::from_spec(rep));
                    slot.count = slot.count.saturating_sub(1);
                    if slot.count == 0 {
                        if let Some(ref rep_item) = rep_opt {
                            *slot = rep_item.clone();
                        } else {
                            *slot = ItemStackData::default();
                        }
                    }
                    if let Some(rep_item) = rep_opt {
                        replacements.push(rep_item);
                    }
                }
                let out_item = if method == "cooking" {
                    ItemStackData::from_spec(&r.output)
                } else {
                    ItemStackData::default()
                };
                let time = if method == "cooking" {
                    r.cooktime.max(0.1)
                } else {
                    r.burntime.max(0.1)
                };
                return Some(CraftOutput {
                    item: out_item,
                    time,
                    decrements,
                    replacements,
                });
            }
        }
        return None;
    }

    let w = width.max(1);
    let h = (grid.len() + w - 1) / w;
    let mut min_r = h;
    let mut max_r = 0usize;
    let mut min_c = w;
    let mut max_c = 0usize;
    let mut non_empty_indices = Vec::new();

    for (idx, stack) in grid.iter().enumerate() {
        if !stack.is_empty() {
            let r = idx / w;
            let c = idx % w;
            min_r = min_r.min(r);
            max_r = max_r.max(r);
            min_c = min_c.min(c);
            max_c = max_c.max(c);
            non_empty_indices.push(idx);
        }
    }
    if non_empty_indices.is_empty() {
        return None;
    }

    let sub_w = max_c - min_c + 1;
    let sub_h = max_r - min_r + 1;

    for r in &reg.crafts {
        let is_shaped = r.recipe_type == "shaped" || r.recipe_type == "normal" || r.recipe_type.is_empty();
        let matched = if is_shaped {
            let (rw, rh, pat_rows) = trim_shaped_recipe(&r.recipe);
            if rw == sub_w && rh == sub_h && rw > 0 && rh > 0 {
                let mut ok = true;
                'row_loop: for rr in 0..rh {
                    for cc in 0..rw {
                        let gi = (min_r + rr) * w + (min_c + cc);
                        let item_name = grid.get(gi).map(|s| s.name.as_str()).unwrap_or("");
                        if !ingredient_matches(pat_rows[rr][cc], item_name, reg) {
                            ok = false;
                            break 'row_loop;
                        }
                    }
                }
                ok
            } else {
                false
            }
        } else if r.recipe_type == "shapeless" {
            let reqs: Vec<&str> = r
                .recipe
                .iter()
                .flat_map(|row| row.iter().map(|s| s.trim()))
                .filter(|s| !s.is_empty())
                .collect();
            if reqs.len() == non_empty_indices.len() && !reqs.is_empty() {
                let inputs: Vec<&str> = non_empty_indices
                    .iter()
                    .map(|&i| grid[i].name.as_str())
                    .collect();
                let mut used = vec![false; inputs.len()];
                match_shapeless_backtrack(&reqs, &inputs, &mut used, 0, reg)
            } else {
                false
            }
        } else {
            false
        };

        if matched {
            let out_item = ItemStackData::from_spec(&r.output);
            if out_item.is_empty() {
                continue;
            }
            let mut decrements = grid.to_vec();
            let mut replacements = Vec::new();
            for &idx in &non_empty_indices {
                if let Some(slot) = decrements.get_mut(idx) {
                    if let Some((_, rep)) = r
                        .replacements
                        .iter()
                        .find(|(from, _)| ingredient_matches(from, &slot.name, reg))
                    {
                        replacements.push(ItemStackData::from_spec(rep));
                    }
                    slot.count = slot.count.saturating_sub(1);
                    if slot.count == 0 {
                        *slot = ItemStackData::default();
                    }
                }
            }
            return Some(CraftOutput {
                item: out_item,
                time: 0.0,
                decrements,
                replacements,
            });
        }
    }

    // Luanti built-in toolrepair (2 identical damaged tools -> combined durability + 5% bonus)
    if non_empty_indices.len() == 2 {
        let a = &grid[non_empty_indices[0]];
        let b = &grid[non_empty_indices[1]];
        if a.name == b.name && (a.wear > 0 || b.wear > 0) {
            let is_tool = reg
                .get_item_def(&a.name)
                .map(|d| d.item_type == "tool" || d.tool_capabilities.is_some())
                .unwrap_or(false);
            if is_tool {
                let uses_a = 65536i32 - a.wear as i32;
                let uses_b = 65536i32 - b.wear as i32;
                let repaired_uses = (uses_a + uses_b + 65536 / 20).clamp(0, 65536);
                let new_wear = (65536 - repaired_uses).clamp(0, 65535) as u16;
                let mut out_item = ItemStackData::from_spec(&a.name);
                out_item.wear = new_wear;
                let mut decrements = grid.to_vec();
                for &idx in &non_empty_indices {
                    decrements[idx] = ItemStackData::default();
                }
                return Some(CraftOutput {
                    item: out_item,
                    time: 0.0,
                    decrements,
                    replacements: Vec::new(),
                });
            }
        }
    }

    None
}

pub struct LuaEngineShared {
    pub paths: EnginePaths,
    pub registries: Arc<RwLock<EngineRegistries>>,
    pub world: Arc<WorldStore>,
    pub config: Arc<RwLock<MinetestConfig>>,
    pub mod_paths: RwLock<HashMap<String, PathBuf>>,
    pub mod_order: RwLock<Vec<String>>,
    pub current_modname: RwLock<Option<String>>,
    pub last_run_mod: RwLock<Option<String>>,
    pub mapgen_settings: RwLock<HashMap<String, String>>,
    pub mod_storages: RwLock<HashMap<String, Arc<RwLock<MetaDataState>>>>,
    pub detached_inventories: RwLock<HashMap<String, Arc<RwLock<MetaDataState>>>>,
    pub ipc_store: RwLock<HashMap<String, String>>,
    pub players: RwLock<HashMap<String, Arc<RwLock<ObjectState>>>>,
    pub entities: RwLock<HashMap<u64, Arc<RwLock<ObjectState>>>>,
    pub next_entity_id: RwLock<u64>,
    pub player_privs: RwLock<HashMap<String, HashMap<String, bool>>>,
    pub chat_log: RwLock<Vec<String>>,
    pub pending_formspecs: RwLock<Vec<FormspecRequest>>,
    pub pending_sounds: RwLock<Vec<PendingSoundEvent>>,
    pub time_of_day: RwLock<f32>,
    pub gametime: RwLock<f64>,
    pub abm_accum: RwLock<f64>,
    pub day_count: RwLock<u32>,
    pub start_instant: Instant,
}

impl LuaEngineShared {
    pub fn new(
        paths: EnginePaths,
        registries: Arc<RwLock<EngineRegistries>>,
        world: Arc<WorldStore>,
        config: Arc<RwLock<MinetestConfig>>,
    ) -> Arc<Self> {
        let mut mg = HashMap::new();
        mg.insert("mg_name".into(), "v7".into());
        mg.insert("seed".into(), world.seed.to_string());
        mg.insert("chunksize".into(), "5".into());
        mg.insert("water_level".into(), "1".into());
        mg.insert("mapgen_limit".into(), "31000".into());
        mg.insert("mg_flags".into(), "caves,dungeons,light,decorations,biomes,ores".into());
        mg.insert("mcl_singlenode_mapgen".into(), "false".into());

        Arc::new(Self {
            paths,
            registries,
            world,
            config,
            mod_paths: RwLock::new(HashMap::new()),
            mod_order: RwLock::new(Vec::new()),
            current_modname: RwLock::new(None),
            last_run_mod: RwLock::new(None),
            mapgen_settings: RwLock::new(mg),
            mod_storages: RwLock::new(HashMap::new()),
            detached_inventories: RwLock::new(HashMap::new()),
            ipc_store: RwLock::new(HashMap::new()),
            players: RwLock::new(HashMap::new()),
            entities: RwLock::new(HashMap::new()),
            next_entity_id: RwLock::new(1),
            player_privs: RwLock::new(HashMap::new()),
            chat_log: RwLock::new(Vec::new()),
            pending_formspecs: RwLock::new(Vec::new()),
            pending_sounds: RwLock::new(Vec::new()),
            time_of_day: RwLock::new(0.35),
            gametime: RwLock::new(0.0),
            abm_accum: RwLock::new(0.0),
            day_count: RwLock::new(1),
            start_instant: Instant::now(),
        })
    }
}

fn parse_ivec3(t: &Table) -> IVec3 {
    let x: f64 = t.get("x").unwrap_or(0.0);
    let y: f64 = t.get("y").unwrap_or(0.0);
    let z: f64 = t.get("z").unwrap_or(0.0);
    IVec3::new(x.round() as i32, y.round() as i32, z.round() as i32)
}

pub(crate) fn make_vec3(lua: &Lua, x: f64, y: f64, z: f64) -> LuaResult<Table> {
    let t = lua.create_table()?;
    t.set("x", x)?;
    t.set("y", y)?;
    t.set("z", z)?;
    if let Ok(vec_tbl) = lua.globals().get::<Table>("vector") {
        if let Ok(mt) = vec_tbl.get::<Table>("metatable") {
            let _ = t.set_metatable(Some(mt));
        }
    }
    Ok(t)
}

fn parse_color_value(val: Value) -> Option<[u8; 4]> {
    match val {
        Value::String(s) => {
            let str_v = s.to_str().ok()?;
            parse_color_str(&str_v)
        }
        Value::Table(t) => {
            let r: u8 = t.get("r").unwrap_or(255);
            let g: u8 = t.get("g").unwrap_or(255);
            let b: u8 = t.get("b").unwrap_or(255);
            let a: u8 = t.get("a").unwrap_or(255);
            Some([r, g, b, a])
        }
        _ => None,
    }
}

pub fn parse_color_str(s: &str) -> Option<[u8; 4]> {
    let clean = s.trim();
    if let Some(hex) = clean.strip_prefix('#') {
        if hex.len() == 6 {
            let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
            let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
            let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
            return Some([r, g, b, 255]);
        } else if hex.len() == 8 {
            let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
            let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
            let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
            let a = u8::from_str_radix(&hex[6..8], 16).ok()?;
            return Some([r, g, b, a]);
        } else if hex.len() == 3 {
            let r = u8::from_str_radix(&hex[0..1], 16).ok()? * 17;
            let g = u8::from_str_radix(&hex[1..2], 16).ok()? * 17;
            let b = u8::from_str_radix(&hex[2..3], 16).ok()? * 17;
            return Some([r, g, b, 255]);
        }
    }
    match clean.to_ascii_lowercase().as_str() {
        "white" => Some([255, 255, 255, 255]),
        "black" => Some([0, 0, 0, 255]),
        "red" => Some([255, 0, 0, 255]),
        "green" => Some([0, 255, 0, 255]),
        "blue" => Some([0, 0, 255, 255]),
        "yellow" => Some([255, 255, 0, 255]),
        _ => None,
    }
}

fn parse_tiles_list(val: Value) -> Vec<TileSpec> {
    let mut out = Vec::new();
    let Value::Table(tbl) = val else {
        return out;
    };
    if let Ok(len) = tbl.len() {
        for i in 1..=len {
            if let Ok(item) = tbl.get::<Value>(i) {
                match item {
                    Value::String(s) => {
                        if let Ok(st) = s.to_str() {
                            out.push(TileSpec {
                                name: st.to_string(),
                                color: None,
                            });
                        }
                    }
                    Value::Table(sub) => {
                        let name: String = sub
                            .get("name")
                            .or_else(|_| sub.get("image"))
                            .unwrap_or_default();
                        let color = sub.get::<Value>("color").ok().and_then(parse_color_value);
                        out.push(TileSpec { name, color });
                    }
                    _ => {}
                }
            }
        }
    }
    out
}

fn parse_string_list(val: Value) -> Vec<String> {
    match val {
        Value::String(s) => s.to_str().map(|st| vec![st.to_string()]).unwrap_or_default(),
        Value::Table(t) => {
            let mut out = Vec::new();
            if let Ok(len) = t.len() {
                for i in 1..=len {
                    if let Ok(s) = t.get::<String>(i) {
                        out.push(s);
                    }
                }
            }
            out
        }
        _ => Vec::new(),
    }
}

fn parse_single_box(val: Value) -> Option<[f32; 6]> {
    let Value::Table(t) = val else {
        return None;
    };
    let x1 = t.get::<f32>(1).ok()?;
    let y1 = t.get::<f32>(2).ok()?;
    let z1 = t.get::<f32>(3).ok()?;
    let x2 = t.get::<f32>(4).ok()?;
    let y2 = t.get::<f32>(5).ok()?;
    let z2 = t.get::<f32>(6).ok()?;
    Some([x1, y1, z1, x2, y2, z2])
}

fn parse_box_list(val: Value) -> Vec<[f32; 6]> {
    let mut out = Vec::new();
    let Value::Table(ft) = val else {
        return out;
    };
    let first: Value = ft.get(1).unwrap_or(Value::Nil);
    if matches!(first, Value::Table(_)) {
        let len = ft.len().unwrap_or(0);
        for i in 1..=len {
            if let Ok(v) = ft.get::<Value>(i) {
                if let Some(arr) = parse_single_box(v) {
                    out.push(arr);
                }
            }
        }
    } else if let Some(arr) = parse_single_box(Value::Table(ft)) {
        out.push(arr);
    }
    out
}

fn parse_node_box_spec(val: Value) -> Option<NodeBoxSpec> {
    let Value::Table(nb) = val else {
        return None;
    };
    let box_type: String = nb.get("type").unwrap_or_else(|_| "regular".into());
    let fixed = nb.get::<Value>("fixed").map(parse_box_list).unwrap_or_default();
    let wall_top = nb.get::<Value>("wall_top").ok().and_then(parse_single_box);
    let wall_bottom = nb.get::<Value>("wall_bottom").ok().and_then(parse_single_box);
    let wall_side = nb.get::<Value>("wall_side").ok().and_then(parse_single_box);
    let connect_top = nb.get::<Value>("connect_top").map(parse_box_list).unwrap_or_default();
    let connect_bottom = nb.get::<Value>("connect_bottom").map(parse_box_list).unwrap_or_default();
    let connect_front = nb.get::<Value>("connect_front").map(parse_box_list).unwrap_or_default();
    let connect_back = nb.get::<Value>("connect_back").map(parse_box_list).unwrap_or_default();
    let connect_left = nb.get::<Value>("connect_left").map(parse_box_list).unwrap_or_default();
    let connect_right = nb.get::<Value>("connect_right").map(parse_box_list).unwrap_or_default();
    let disconnected_top = nb.get::<Value>("disconnected_top").map(parse_box_list).unwrap_or_default();
    let disconnected_bottom = nb.get::<Value>("disconnected_bottom").map(parse_box_list).unwrap_or_default();
    let disconnected_front = nb.get::<Value>("disconnected_front").map(parse_box_list).unwrap_or_default();
    let disconnected_back = nb.get::<Value>("disconnected_back").map(parse_box_list).unwrap_or_default();
    let disconnected_left = nb.get::<Value>("disconnected_left").map(parse_box_list).unwrap_or_default();
    let disconnected_right = nb.get::<Value>("disconnected_right").map(parse_box_list).unwrap_or_default();
    let disconnected_sides = nb.get::<Value>("disconnected_sides").map(parse_box_list).unwrap_or_default();
    let mut disconnected_all = nb.get::<Value>("disconnected").map(parse_box_list).unwrap_or_default();
    if disconnected_all.is_empty() {
        disconnected_all = nb.get::<Value>("disconnected_all").map(parse_box_list).unwrap_or_default();
    }
    Some(NodeBoxSpec {
        box_type,
        fixed,
        wall_top,
        wall_bottom,
        wall_side,
        connect_top,
        connect_bottom,
        connect_front,
        connect_back,
        connect_left,
        connect_right,
        disconnected_top,
        disconnected_bottom,
        disconnected_front,
        disconnected_back,
        disconnected_left,
        disconnected_right,
        disconnected_sides,
        disconnected_all,
    })
}

fn parse_sound_spec(val: Value) -> Option<SoundSpec> {
    match val {
        Value::String(s) => {
            let name = s.to_str().ok()?.trim().to_string();
            if name.is_empty() {
                None
            } else {
                Some(SoundSpec {
                    name,
                    gain: 1.0,
                    pitch: 1.0,
                })
            }
        }
        Value::Table(t) => {
            let name: String = t.get("name").unwrap_or_default();
            if name.trim().is_empty() {
                return None;
            }
            let gain: f32 = t.get("gain").unwrap_or(1.0);
            let pitch: f32 = t.get("pitch").unwrap_or(1.0);
            Some(SoundSpec { name, gain, pitch })
        }
        _ => None,
    }
}

fn parse_node_sounds(val: Value) -> NodeSounds {
    let Value::Table(t) = val else {
        return NodeSounds::default();
    };
    NodeSounds {
        footstep: t.get::<Value>("footstep").ok().and_then(parse_sound_spec),
        dig: t.get::<Value>("dig").ok().and_then(parse_sound_spec),
        dug: t.get::<Value>("dug").ok().and_then(parse_sound_spec),
        place: t.get::<Value>("place").ok().and_then(parse_sound_spec),
        fall: t.get::<Value>("fall").ok().and_then(parse_sound_spec),
    }
}

fn parse_node_drop(val: Value) -> NodeDrop {
    match val {
        Value::Nil => NodeDrop::Default,
        Value::String(s) => {
            let st = s.to_str().map(|x| x.trim().to_string()).unwrap_or_default();
            if st.is_empty() {
                NodeDrop::Empty
            } else {
                NodeDrop::Single(st)
            }
        }
        Value::Table(t) => {
            let Ok(items_t) = t.get::<Table>("items") else {
                return NodeDrop::Empty;
            };
            let max_items = t.get::<usize>("max_items").ok();
            let mut items = Vec::new();
            let len = items_t.len().unwrap_or(0);
            for i in 1..=len {
                if let Ok(entry_t) = items_t.get::<Table>(i) {
                    let drop_items = entry_t
                        .get::<Value>("items")
                        .map(parse_string_list)
                        .unwrap_or_default();
                    let rarity: i32 = entry_t.get("rarity").unwrap_or(1);
                    let tools = entry_t
                        .get::<Value>("tools")
                        .map(parse_string_list)
                        .unwrap_or_default();
                    let mut tool_groups = Vec::new();
                    if let Ok(tg_val) = entry_t.get::<Value>("tool_groups") {
                        match tg_val {
                            Value::String(s) => {
                                if let Ok(st) = s.to_str() {
                                    tool_groups.push(vec![st.to_string()]);
                                }
                            }
                            Value::Table(gt) => {
                                let glen = gt.len().unwrap_or(0);
                                for j in 1..=glen {
                                    if let Ok(v) = gt.get::<Value>(j) {
                                        let list = parse_string_list(v);
                                        if !list.is_empty() {
                                            tool_groups.push(list);
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                    let inherit_color: bool = entry_t.get("inherit_color").unwrap_or(false);
                    items.push(DropItemEntry {
                        items: drop_items,
                        rarity,
                        tools,
                        tool_groups,
                        inherit_color,
                    });
                }
            }
            NodeDrop::Table { max_items, items }
        }
        _ => NodeDrop::Default,
    }
}

pub fn parse_tool_capabilities(val: Value) -> Option<ToolCapabilities> {
    let Value::Table(t) = val else {
        return None;
    };
    let full_punch_interval: f32 = t.get("full_punch_interval").unwrap_or(1.4);
    let max_drop_level: i32 = t.get("max_drop_level").unwrap_or(0);
    let punch_attack_uses: i32 = t.get("punch_attack_uses").unwrap_or(0);

    let mut groupcaps = HashMap::new();
    if let Ok(gc_t) = t.get::<Table>("groupcaps") {
        for pair in gc_t.pairs::<String, Table>() {
            if let Ok((gname, cap_t)) = pair {
                let uses: i32 = cap_t.get("uses").unwrap_or(0);
                let maxlevel: i32 = cap_t.get("maxlevel").unwrap_or(0);
                let mut times = HashMap::new();
                if let Ok(times_t) = cap_t.get::<Table>("times") {
                    for tp in times_t.pairs::<i32, f32>() {
                        if let Ok((lvl, sec)) = tp {
                            times.insert(lvl, sec);
                        }
                    }
                }
                groupcaps.insert(
                    gname,
                    ToolGroupCap {
                        times,
                        uses,
                        maxlevel,
                    },
                );
            }
        }
    }

    let mut damage_groups = HashMap::new();
    if let Ok(dg_t) = t.get::<Table>("damage_groups") {
        for pair in dg_t.pairs::<String, i32>() {
            if let Ok((k, v)) = pair {
                damage_groups.insert(k, v);
            }
        }
    }

    Some(ToolCapabilities {
        full_punch_interval,
        max_drop_level,
        groupcaps,
        damage_groups,
        punch_attack_uses,
    })
}

pub(crate) fn lua_value_to_json(val: Value) -> serde_json::Value {
    match val {
        Value::Nil => serde_json::Value::Null,
        Value::Boolean(b) => serde_json::Value::Bool(b),
        Value::Integer(i) => serde_json::Value::Number(i.into()),
        Value::Number(n) => serde_json::Number::from_f64(n)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        Value::String(s) => serde_json::Value::String(s.to_str().map(|x| x.to_string()).unwrap_or_default()),
        Value::Table(t) => {
            let len = t.len().unwrap_or(0);
            if len > 0 {
                let mut arr = Vec::with_capacity(len as usize);
                for i in 1..=len {
                    if let Ok(v) = t.get::<Value>(i) {
                        arr.push(lua_value_to_json(v));
                    }
                }
                serde_json::Value::Array(arr)
            } else {
                let mut map = serde_json::Map::new();
                for pair in t.pairs::<Value, Value>() {
                    if let Ok((k, v)) = pair {
                        let ks = match k {
                            Value::String(s) => s.to_str().map(|x| x.to_string()).unwrap_or_default(),
                            Value::Integer(i) => i.to_string(),
                            _ => continue,
                        };
                        map.insert(ks, lua_value_to_json(v));
                    }
                }
                serde_json::Value::Object(map)
            }
        }
        _ => serde_json::Value::Null,
    }
}

pub(crate) fn json_to_lua_value(lua: &Lua, jv: &serde_json::Value, nullval: &Value) -> LuaResult<Value> {
    match jv {
        serde_json::Value::Null => Ok(nullval.clone()),
        serde_json::Value::Bool(b) => Ok(Value::Boolean(*b)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(Value::Integer(i))
            } else {
                Ok(Value::Number(n.as_f64().unwrap_or(0.0)))
            }
        }
        serde_json::Value::String(s) => Ok(Value::String(lua.create_string(s)?)),
        serde_json::Value::Array(arr) => {
            let t = lua.create_table_with_capacity(arr.len(), 0)?;
            for (i, v) in arr.iter().enumerate() {
                t.set(i + 1, json_to_lua_value(lua, v, nullval)?)?;
            }
            Ok(Value::Table(t))
        }
        serde_json::Value::Object(map) => {
            let t = lua.create_table_with_capacity(0, map.len())?;
            for (k, v) in map {
                t.set(k.as_str(), json_to_lua_value(lua, v, nullval)?)?;
            }
            Ok(Value::Table(t))
        }
    }
}

fn schematic_from_lua_value(val: Value) -> Option<Schematic> {
    match val {
        Value::String(s) => {
            let p = s.to_str().ok()?.to_string();
            Schematic::load_from_file(std::path::Path::new(&p)).ok()
        }
        Value::Table(t) => {
            let sz_t = t.get::<Table>("size").ok()?;
            let sx: i32 = sz_t.get("x").unwrap_or(1);
            let sy: i32 = sz_t.get("y").unwrap_or(1);
            let sz: i32 = sz_t.get("z").unwrap_or(1);
            if sx <= 0 || sy <= 0 || sz <= 0 {
                return None;
            }
            let mut yslice_prob = vec![MTSCHEM_PROB_ALWAYS; sy as usize];
            if let Ok(ys_t) = t.get::<Table>("yslice_prob") {
                let len = ys_t.len().unwrap_or(0);
                for i in 1..=len {
                    if let Ok(row) = ys_t.get::<Table>(i) {
                        let ypos: i32 = row.get("ypos").unwrap_or(-1);
                        let prob: u16 = row.get("prob").unwrap_or(255);
                        if ypos >= 0 && (ypos as usize) < yslice_prob.len() {
                            yslice_prob[ypos as usize] = if prob >= 254 {
                                MTSCHEM_PROB_ALWAYS
                            } else {
                                (prob / 2).min(127) as u8
                            };
                        }
                    }
                }
            }
            let nodecount = (sx as usize) * (sy as usize) * (sz as usize);
            let mut data = vec![SchematicNode::default(); nodecount];
            let mut node_names = Vec::new();
            if let Ok(dt) = t.get::<Table>("data") {
                for i in 0..nodecount {
                    if let Ok(nt) = dt.get::<Table>(i + 1) {
                        let name: String = nt.get("name").unwrap_or_else(|_| "air".into());
                        let raw_prob: u16 = nt.get("prob").unwrap_or(255);
                        let prob = if raw_prob >= 254 {
                            MTSCHEM_PROB_ALWAYS
                        } else {
                            (raw_prob / 2).min(127) as u8
                        };
                        let force_place: bool = nt.get("force_place").unwrap_or(false);
                        let param2: u8 = nt.get("param2").unwrap_or(0);
                        if !node_names.contains(&name) {
                            node_names.push(name.clone());
                        }
                        data[i] = SchematicNode {
                            name,
                            prob,
                            force_place,
                            param2,
                        };
                    }
                }
            }
            Some(Schematic {
                size: IVec3::new(sx, sy, sz),
                yslice_prob,
                node_names,
                data,
            })
        }
        _ => None,
    }
}

fn parse_replacements_map(val: Option<Value>) -> HashMap<String, String> {
    let mut map = HashMap::new();
    if let Some(Value::Table(t)) = val {
        for pair in t.pairs::<String, String>() {
            if let Ok((k, v)) = pair {
                map.insert(k, v);
            }
        }
    }
    map
}

fn parse_place_flags(val: Option<Value>, force_placement: bool) -> PlaceFlags {
    let mut flags = match val {
        Some(Value::String(s)) => s
            .to_str()
            .map(|st| PlaceFlags::from_str(&st))
            .unwrap_or_default(),
        Some(Value::Table(t)) => PlaceFlags {
            place_center_x: t.get("place_center_x").unwrap_or(false),
            place_center_y: t.get("place_center_y").unwrap_or(false),
            place_center_z: t.get("place_center_z").unwrap_or(false),
            force_placement: t.get("force_placement").unwrap_or(false),
        },
        _ => PlaceFlags::default(),
    };
    if force_placement {
        flags.force_placement = true;
    }
    flags
}

fn parse_vec3f(t: &Table) -> Vec3 {
    let x: f32 = t.get("x").unwrap_or(0.0);
    let y: f32 = t.get("y").unwrap_or(0.0);
    let z: f32 = t.get("z").unwrap_or(0.0);
    Vec3::new(x, y, z)
}

fn resolve_query_node_ids(s: &LuaEngineShared, nodenames: Value) -> HashSet<u16> {
    let names = parse_string_list(nodenames);
    let mut ids = HashSet::new();
    let reg = s.registries.read();
    for n in names {
        if let Some(grp) = n.strip_prefix("group:") {
            for def in reg.nodes_by_id.iter().flatten() {
                if def.groups.get(grp).copied().unwrap_or(0) > 0 {
                    ids.insert(def.id);
                }
            }
        } else {
            ids.insert(reg.get_content_id(&n));
        }
    }
    ids
}

fn invoke_node_callback(
    lua: &Lua,
    node_name: &str,
    cb_name: &str,
    pos: IVec3,
    extra_node: Option<(&str, u8, u8)>,
) {
    if node_name == "air" || node_name == "ignore" || node_name == "unknown" {
        return;
    }
    let Ok(core) = lua.globals().get::<Table>("core") else {
        return;
    };
    let Ok(rn) = core.get::<Table>("registered_nodes") else {
        return;
    };
    let Ok(def) = rn.get::<Table>(node_name) else {
        return;
    };
    let Ok(cb) = def.get::<mlua::Function>(cb_name) else {
        return;
    };
    let Ok(pt) = make_vec3(lua, pos.x as f64, pos.y as f64, pos.z as f64) else {
        return;
    };
    if let Some((oname, op1, op2)) = extra_node {
        if let Ok(nt) = lua.create_table() {
            let _ = nt.set("name", oname);
            let _ = nt.set("param1", op1);
            let _ = nt.set("param2", op2);
            let _ = cb.call::<Value>((pt, nt));
        }
    } else {
        let _ = cb.call::<Value>(pt);
    }
}

fn set_node_with_callbacks(
    lua: &Lua,
    s: &LuaEngineShared,
    p: IVec3,
    new_name: &str,
    p1: u8,
    p2: u8,
) {
    let (old_id, old_p1, old_p2) = s.world.get_node(p);
    let (old_name, new_id) = {
        let reg = s.registries.read();
        (
            reg.get_name_from_content_id(old_id).to_string(),
            reg.get_content_id(new_name),
        )
    };
    if old_id != CONTENT_IGNORE && old_name != "air" && old_name != "ignore" {
        invoke_node_callback(lua, &old_name, "on_destruct", p, None);
    }
    if old_id != new_id {
        s.world.remove_node_meta(p);
    }
    s.world.set_node(p, new_id, p1, p2);
    if old_id != CONTENT_IGNORE && old_name != "air" && old_name != "ignore" {
        invoke_node_callback(
            lua,
            &old_name,
            "after_destruct",
            p,
            Some((&old_name, old_p1, old_p2)),
        );
    }
    if new_id != CONTENT_AIR && new_id != CONTENT_IGNORE {
        invoke_node_callback(lua, new_name, "on_construct", p, None);
    }
}

pub fn register_core_api(lua: &Lua, shared: Arc<LuaEngineShared>) -> LuaResult<()> {
    let g = lua.globals();
    let core = lua.create_table()?;
    let ents_tbl = lua.create_table()?;
    core.set("_cubex_luaentities", ents_tbl.clone())?;
    core.set("luaentities", ents_tbl)?;
    core.set("registered_entities", lua.create_table()?)?;
    core.set(
        "register_entity",
        lua.create_function(|lua, (mut name, prototype): (String, Table)| {
            if let Some(stripped) = name.strip_prefix(':') {
                name = stripped.to_string();
            }
            if let Ok(core_t) = lua.globals().get::<Table>("core") {
                if !name.contains(':') {
                    let cur: Option<String> = core_t
                        .get::<mlua::Function>("get_current_modname")
                        .ok()
                        .and_then(|f| f.call::<Option<String>>(()).ok().flatten());
                    if let Some(m) = cur.filter(|s| !s.is_empty()) {
                        name = format!("{m}:{name}");
                    }
                }
                prototype.set("name", name.as_str())?;
                prototype.set("__index", prototype.clone())?;
                if let Ok(reg_ents) = core_t.get::<Table>("registered_entities") {
                    reg_ents.set(name.as_str(), prototype)?;
                }
            }
            Ok(())
        })?,
    )?;

    // Required builtin globals
    g.set("INIT", "game")?;
    g.set("DIR_DELIM", "/")?;
    core.set("settings", SettingsUD(shared.config.clone()))?;

    // Paths & Mod Context
    {
        let s = shared.clone();
        core.set("get_builtin_path", lua.create_function(move |_, ()| {
            let mut p = s.paths.builtin_dir.to_string_lossy().to_string();
            if !p.ends_with('/') { p.push('/'); }
            Ok(p)
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("get_worldpath", lua.create_function(move |_, ()| {
            Ok(s.world.world_path.to_string_lossy().to_string())
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("get_current_modname", lua.create_function(move |_, ()| {
            Ok(s.current_modname.read().clone())
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("get_last_run_mod", lua.create_function(move |_, ()| {
            Ok(s.last_run_mod.read().clone().unwrap_or_else(|| "??".into()))
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("set_last_run_mod", lua.create_function(move |_, m: String| {
            *s.last_run_mod.write() = Some(m);
            Ok(())
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("get_modpath", lua.create_function(move |_, modname: String| {
            Ok(s.mod_paths.read().get(&modname).map(|p| p.to_string_lossy().to_string()))
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("get_modnames", lua.create_function(move |lua, ()| {
            let order = s.mod_order.read();
            let t = lua.create_table_with_capacity(order.len(), 0)?;
            for (i, m) in order.iter().enumerate() {
                t.set(i + 1, m.as_str())?;
            }
            Ok(t)
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("get_game_info", lua.create_function(move |lua, ()| {
            let t = lua.create_table()?;
            t.set("id", "mineclonia")?;
            t.set("title", "Mineclonia")?;
            t.set("path", s.paths.game_dir.to_string_lossy().to_string())?;
            Ok(t)
        })?)?;
    }

    // Logging & Version
    core.set("log", lua.create_function(|_, args: MultiValue| {
        let mut iter = args.into_iter();
        let first = iter.next();
        let second = iter.next();
        if let (Some(Value::String(lvl)), Some(Value::String(msg))) = (&first, &second) {
            if let (Ok(l), Ok(m)) = (lvl.to_str(), msg.to_str()) {
                if l == "error" {
                    tracing::error!("[Lua] {}", m);
                } else if l == "warning" {
                    tracing::warn!("[Lua] {}", m);
                } else {
                    tracing::debug!("[Lua:{}] {}", l, m);
                }
            }
        } else if let Some(Value::String(msg)) = first {
            if let Ok(m) = msg.to_str() {
                tracing::debug!("[Lua] {}", m);
            }
        }
        Ok(())
    })?)?;

    core.set("get_version", lua.create_function(|lua, ()| {
        let t = lua.create_table()?;
        t.set("project", "Cubex-RS")?;
        t.set("string", "5.10.0")?;
        t.set("hash", "cubex-rs")?;
        t.set("proto_min", 44)?;
        t.set("proto_max", 46)?;
        Ok(t)
    })?)?;
    core.set("protocol_versions", lua.create_function(|lua, ()| {
        let t = lua.create_table()?;
        t.set(44, "5.10.0")?;
        Ok(t)
    })?)?;

    // Raw Item / Node / Alias registration consumed by builtin/game/register.lua
    {
        let s = shared.clone();
        core.set("register_item_raw", lua.create_function(move |_, def: Table| {
            let name_val: Value = def.get("name").unwrap_or(Value::Nil);
            let name = match name_val {
                Value::String(st) => st.to_str()?.to_string(),
                _ => return Ok(()),
            };
            let item_type: String = def.get("type").unwrap_or_else(|_| "none".into());
            let description: String = def.get("description").unwrap_or_default();
            let inventory_image: String = def.get("inventory_image").unwrap_or_default();
            let wield_image: String = def.get("wield_image").unwrap_or_default();
            let stack_max: u16 = def.get("stack_max").unwrap_or(64);
            let range: Option<f32> = def.get("range").ok();
            let liquids_pointable: bool = def.get("liquids_pointable").unwrap_or(false);
            let tool_capabilities = def
                .get::<Value>("tool_capabilities")
                .ok()
                .and_then(parse_tool_capabilities);

            let mut groups = HashMap::new();
            if let Ok(gt) = def.get::<Table>("groups") {
                for pair in gt.pairs::<String, i32>() {
                    if let Ok((k, v)) = pair {
                        groups.insert(k, v);
                    }
                }
            }

            let mut reg = s.registries.write();
            reg.register_item(ItemDef {
                name: name.clone(),
                item_type: item_type.clone(),
                description: description.clone(),
                inventory_image: inventory_image.clone(),
                wield_image: wield_image.clone(),
                stack_max,
                groups: groups.clone(),
                is_node: item_type == "node",
                tool_capabilities,
                range,
                liquids_pointable,
            });

            if item_type == "node" && !name.is_empty() {
                let dt_str: String = def.get("drawtype").unwrap_or_else(|_| "normal".into());
                let drawtype = DrawType::from_str(&dt_str);
                let walkable: bool = def.get("walkable").unwrap_or(true);
                let pointable: bool = def.get("pointable").unwrap_or(true);
                let tiles = def
                    .get::<Value>("tiles")
                    .or_else(|_| def.get::<Value>("tile_images"))
                    .map(parse_tiles_list)
                    .unwrap_or_default();
                let overlay_tiles = def
                    .get::<Value>("overlay_tiles")
                    .map(parse_tiles_list)
                    .unwrap_or_default();

                let nb_spec = def
                    .get::<Value>("node_box")
                    .ok()
                    .and_then(parse_node_box_spec)
                    .unwrap_or_default();

                let selection_box = def
                    .get::<Value>("selection_box")
                    .ok()
                    .and_then(parse_node_box_spec)
                    .unwrap_or_else(|| {
                        if drawtype == DrawType::NodeBox {
                            nb_spec.clone()
                        } else {
                            NodeBoxSpec {
                                box_type: "regular".into(),
                                fixed: if pointable {
                                    vec![[-0.5, -0.5, -0.5, 0.5, 0.5, 0.5]]
                                } else {
                                    Vec::new()
                                },
                                ..Default::default()
                            }
                        }
                    });
                let mut selection_boxes = selection_box.fixed.clone();
                if selection_boxes.is_empty() {
                    if let Some(b) = selection_box
                        .wall_bottom
                        .or(selection_box.wall_side)
                        .or(selection_box.wall_top)
                    {
                        selection_boxes.push(b);
                    } else if drawtype == DrawType::NodeBox && !nb_spec.fixed.is_empty() {
                        selection_boxes = nb_spec.fixed.clone();
                    } else if pointable {
                        selection_boxes.push([-0.5, -0.5, -0.5, 0.5, 0.5, 0.5]);
                    }
                }

                let collision_box = def
                    .get::<Value>("collision_box")
                    .ok()
                    .and_then(parse_node_box_spec)
                    .unwrap_or_else(|| {
                        if drawtype == DrawType::NodeBox {
                            nb_spec.clone()
                        } else {
                            NodeBoxSpec {
                                box_type: "regular".into(),
                                fixed: if walkable {
                                    vec![[-0.5, -0.5, -0.5, 0.5, 0.5, 0.5]]
                                } else {
                                    Vec::new()
                                },
                                ..Default::default()
                            }
                        }
                    });
                let mut collision_boxes = collision_box.fixed.clone();
                if collision_boxes.is_empty() && walkable {
                    if let Some(b) = collision_box
                        .wall_bottom
                        .or(collision_box.wall_side)
                        .or(collision_box.wall_top)
                    {
                        collision_boxes.push(b);
                    } else if drawtype == DrawType::NodeBox && !nb_spec.fixed.is_empty() {
                        collision_boxes = nb_spec.fixed.clone();
                    } else {
                        collision_boxes.push([-0.5, -0.5, -0.5, 0.5, 0.5, 0.5]);
                    }
                }

                let connects_to = def
                    .get::<Value>("connects_to")
                    .map(parse_string_list)
                    .unwrap_or_default();
                let connect_sides = def
                    .get::<Value>("connect_sides")
                    .map(parse_string_list)
                    .unwrap_or_default();
                let sounds = def
                    .get::<Value>("sounds")
                    .map(parse_node_sounds)
                    .unwrap_or_default();
                let palette = def
                    .get::<String>("palette")
                    .ok()
                    .filter(|p| !p.trim().is_empty());
                let drop = def
                    .get::<Value>("drop")
                    .map(parse_node_drop)
                    .unwrap_or(NodeDrop::Default);
                let safe_terrain = crate::registry::is_safe_terrain_node(&name, &groups);

                let nd = NodeDef {
                    id: 0,
                    name,
                    description,
                    drawtype,
                    walkable,
                    pointable,
                    diggable: def.get("diggable").unwrap_or(true),
                    climbable: def.get("climbable").unwrap_or(false),
                    buildable_to: def.get("buildable_to").unwrap_or(false),
                    sunlight_propagates: def.get("sunlight_propagates").unwrap_or(false),
                    light_source: def.get("light_source").unwrap_or(0),
                    paramtype: def.get("paramtype").unwrap_or_else(|_| "none".into()),
                    paramtype2: def.get("paramtype2").unwrap_or_else(|_| "none".into()),
                    is_ground_content: def.get("is_ground_content").unwrap_or(true),
                    use_texture_alpha: match def.get::<Value>("use_texture_alpha") {
                        Ok(Value::String(st)) => st.to_str().map(|x| x.to_string()).unwrap_or_else(|_| "opaque".into()),
                        Ok(Value::Boolean(true)) => "blend".into(),
                        _ => "opaque".into(),
                    },
                    tiles,
                    overlay_tiles,
                    node_boxes: nb_spec.fixed,
                    node_box_type: nb_spec.box_type,
                    node_box_wall_top: nb_spec.wall_top,
                    node_box_wall_bottom: nb_spec.wall_bottom,
                    node_box_wall_side: nb_spec.wall_side,
                    connects_to,
                    connect_sides,
                    connect_top: nb_spec.connect_top,
                    connect_bottom: nb_spec.connect_bottom,
                    connect_front: nb_spec.connect_front,
                    connect_back: nb_spec.connect_back,
                    connect_left: nb_spec.connect_left,
                    connect_right: nb_spec.connect_right,
                    disconnected_top: nb_spec.disconnected_top,
                    disconnected_bottom: nb_spec.disconnected_bottom,
                    disconnected_front: nb_spec.disconnected_front,
                    disconnected_back: nb_spec.disconnected_back,
                    disconnected_left: nb_spec.disconnected_left,
                    disconnected_right: nb_spec.disconnected_right,
                    disconnected_sides: nb_spec.disconnected_sides,
                    disconnected_all: nb_spec.disconnected_all,
                    selection_boxes,
                    collision_boxes,
                    selection_box,
                    collision_box,
                    mesh: def.get("mesh").ok(),
                    groups,
                    color: def.get::<Value>("color").ok().and_then(parse_color_value),
                    palette,
                    sounds,
                    drop,
                    liquidtype: def.get("liquidtype").unwrap_or_else(|_| "none".into()),
                    liquid_viscosity: def.get("liquid_viscosity").unwrap_or(0),
                    liquid_alternative_flowing: def.get("liquid_alternative_flowing").unwrap_or_default(),
                    liquid_alternative_source: def.get("liquid_alternative_source").unwrap_or_default(),
                    floodable: def.get("floodable").unwrap_or(false),
                    damage_per_second: if safe_terrain {
                        0
                    } else {
                        def.get("damage_per_second").unwrap_or(0)
                    },
                    waving: def.get("waving").unwrap_or(0),
                    leveled: def.get("leveled").unwrap_or(0),
                    visual_scale: def.get("visual_scale").unwrap_or(1.0),
                    liquids_pointable,
                    inventory_image,
                    wield_image,
                    stack_max,
                };
                reg.register_node(nd);
            }
            Ok(())
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("unregister_item_raw", lua.create_function(move |_, name: String| {
            s.registries.write().unregister_item(&name);
            Ok(())
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("register_alias_raw", lua.create_function(move |_, (name, target): (String, String)| {
            s.registries.write().register_alias(name, target);
            Ok(())
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("get_content_id", lua.create_function(move |_, name: String| {
            Ok(s.registries.read().get_content_id(&name))
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("get_name_from_content_id", lua.create_function(move |_, id: u16| {
            Ok(s.registries.read().get_name_from_content_id(id).to_string())
        })?)?;
    }

    // Crafts
    {
        let s = shared.clone();
        core.set("register_craft", lua.create_function(move |_, spec: Table| {
            let recipe_type: String = spec.get("type").unwrap_or_else(|_| "shaped".into());
            let output: String = spec.get("output").unwrap_or_default();
            let cooktime: f32 = spec.get("cooktime").unwrap_or(3.0);
            let burntime: f32 = spec.get("burntime").unwrap_or(0.0);
            let mut rows: Vec<Vec<String>> = Vec::new();
            match spec.get::<Value>("recipe") {
                Ok(Value::Table(rt)) => {
                    let max_r = rt
                        .pairs::<Value, Value>()
                        .filter_map(|p| p.ok())
                        .filter_map(|(k, _)| match k {
                            Value::Integer(i) if (1..=9).contains(&i) => Some(i as usize),
                            _ => None,
                        })
                        .max()
                        .unwrap_or_else(|| rt.len().unwrap_or(0) as usize);
                    for i in 1..=max_r {
                        match rt.get::<Value>(i) {
                            Ok(Value::Table(sub)) => {
                                let max_c = sub
                                    .pairs::<Value, Value>()
                                    .filter_map(|p| p.ok())
                                    .filter_map(|(k, _)| match k {
                                        Value::Integer(ci) if (1..=9).contains(&ci) => Some(ci as usize),
                                        _ => None,
                                    })
                                    .max()
                                    .unwrap_or_else(|| sub.len().unwrap_or(0) as usize);
                                let mut row = Vec::with_capacity(max_c);
                                for c in 1..=max_c {
                                    let cell = match sub.get::<Value>(c) {
                                        Ok(Value::String(st)) => st.to_str().map(|x| x.to_string()).unwrap_or_default(),
                                        _ => String::new(),
                                    };
                                    row.push(cell);
                                }
                                rows.push(row);
                            }
                            Ok(Value::String(st)) => {
                                if let Ok(ss) = st.to_str() {
                                    if rows.is_empty() {
                                        rows.push(vec![ss.to_string()]);
                                    } else {
                                        rows[0].push(ss.to_string());
                                    }
                                }
                            }
                            _ => {
                                if !rows.is_empty() {
                                    rows[0].push(String::new());
                                }
                            }
                        }
                    }
                }
                Ok(Value::String(st)) => {
                    if let Ok(ss) = st.to_str() {
                        rows.push(vec![ss.to_string()]);
                    }
                }
                _ => {}
            }
            let mut replacements = Vec::new();
            if let Ok(rep_t) = spec.get::<Table>("replacements") {
                let len = rep_t.len().unwrap_or(0);
                for i in 1..=len {
                    if let Ok(pair_t) = rep_t.get::<Table>(i) {
                        let a: String = pair_t.get(1).unwrap_or_default();
                        let b: String = pair_t.get(2).unwrap_or_default();
                        if !a.is_empty() {
                            replacements.push((a, b));
                        }
                    }
                }
            }
            s.registries.write().crafts.push(CraftRecipe {
                recipe_type,
                output,
                recipe: rows,
                replacements,
                cooktime,
                burntime,
            });
            Ok(())
        })?)?;
    }
    core.set("clear_craft", lua.create_function(|_, _: Table| Ok(true))?)?;
    {
        let s = shared.clone();
        core.set("get_craft_result", lua.create_function(move |lua, input: Table| {
            let method: String = input.get("method").unwrap_or_else(|_| "normal".into());
            let width: usize = input.get("width").unwrap_or(3);
            let grid: Vec<ItemStackData> = input
                .get::<Table>("items")
                .map(|t| {
                    let max_idx = t
                        .pairs::<Value, Value>()
                        .filter_map(|p| p.ok())
                        .filter_map(|(k, _)| match k {
                            Value::Integer(i) if (1..=64).contains(&i) => Some(i as usize),
                            _ => None,
                        })
                        .max()
                        .unwrap_or_else(|| t.len().unwrap_or(0) as usize);
                    (1..=max_idx)
                        .map(|i| {
                            t.get::<Value>(i)
                                .ok()
                                .and_then(|v| ItemStackUD::from_lua_value(v).ok())
                                .map(|st| st.0.read().clone())
                                .unwrap_or_default()
                        })
                        .collect()
                })
                .unwrap_or_default();
            let reg = s.registries.read();
            let resolved = resolve_craft_recipe(&reg, &grid, width, &method);

            let out_t = lua.create_table()?;
            let dec_t = lua.create_table()?;
            let dec_items = lua.create_table()?;
            let rep_t = lua.create_table()?;

            if let Some(res) = resolved {
                out_t.set("item", ItemStackUD(Arc::new(RwLock::new(res.item))))?;
                out_t.set("time", res.time)?;
                for (i, rp) in res.replacements.into_iter().enumerate() {
                    rep_t.set(i + 1, ItemStackUD(Arc::new(RwLock::new(rp))))?;
                }
                for (i, d) in res.decrements.into_iter().enumerate() {
                    dec_items.set(i + 1, ItemStackUD(Arc::new(RwLock::new(d))))?;
                }
            } else {
                out_t.set("item", ItemStackUD(Arc::new(RwLock::new(ItemStackData::default()))))?;
                out_t.set("time", 0.0f32)?;
            }
            out_t.set("replacements", rep_t)?;
            dec_t.set("items", dec_items)?;
            Ok((out_t, dec_t))
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("get_craft_recipe", lua.create_function(move |lua, out_query: String| {
            let reg = s.registries.read();
            let target = out_query.split_whitespace().next().unwrap_or("");
            for r in &reg.crafts {
                let out_name = r.output.split_whitespace().next().unwrap_or("");
                if out_name == target {
                    let t = lua.create_table()?;
                    t.set("method", r.recipe_type.as_str())?;
                    let w = r.recipe.iter().map(|row| row.len()).max().unwrap_or(0);
                    t.set("width", w)?;
                    let items = lua.create_table()?;
                    let mut idx = 1;
                    for row in &r.recipe {
                        for it in row {
                            items.set(idx, it.as_str())?;
                            idx += 1;
                        }
                    }
                    t.set("items", items)?;
                    return Ok(t);
                }
            }
            Ok(lua.create_table()?)
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("get_all_craft_recipes", lua.create_function(move |lua, query: String| {
            let reg = s.registries.read();
            let t = lua.create_table()?;
            let mut idx = 1;
            for r in &reg.crafts {
                if r.output == query || r.output.starts_with(&(query.clone() + " ")) {
                    let row = lua.create_table()?;
                    row.set("method", r.recipe_type.as_str())?;
                    row.set("output", r.output.as_str())?;
                    let items = lua.create_table()?;
                    let mut ii = 1;
                    for rr in &r.recipe {
                        for it in rr {
                            items.set(ii, it.as_str())?;
                            ii += 1;
                        }
                    }
                    row.set("items", items)?;
                    t.set(idx, row)?;
                    idx += 1;
                }
            }
            if idx == 1 {
                return Ok(Value::Nil);
            }
            Ok(Value::Table(t))
        })?)?;
    }

    // Biomes / Ores / Decorations / Schematics
    {
        let s = shared.clone();
        core.set("register_biome", lua.create_function(move |_, def: Table| {
            let mut reg = s.registries.write();
            let id = reg.biomes.len() as u32 + 1;
            let name: String = def.get("name").unwrap_or_else(|_| format!("biome_{id}"));
            reg.biomes.push(BiomeDef {
                id,
                name,
                node_top: def.get("node_top").unwrap_or_else(|_| "mcl_core:dirt_with_grass".into()),
                depth_top: def.get("depth_top").unwrap_or(1),
                node_filler: def.get("node_filler").unwrap_or_else(|_| "mcl_core:dirt".into()),
                depth_filler: def.get("depth_filler").unwrap_or(3),
                node_stone: def.get("node_stone").unwrap_or_else(|_| "mcl_core:stone".into()),
                node_water_top: def.get("node_water_top").unwrap_or_else(|_| "mcl_core:water_source".into()),
                y_min: def.get("y_min").unwrap_or(-31000),
                y_max: def.get("y_max").unwrap_or(31000),
                heat_point: def.get("heat_point").unwrap_or(50.0),
                humidity_point: def.get("humidity_point").unwrap_or(50.0),
            });
            Ok(id)
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("unregister_biome", lua.create_function(move |_, name: String| {
            s.registries.write().biomes.retain(|b| b.name != name);
            Ok(())
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("clear_registered_biomes", lua.create_function(move |_, ()| {
            s.registries.write().biomes.clear();
            Ok(())
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("register_ore", lua.create_function(move |_, def: Table| {
            let mut reg = s.registries.write();
            reg.ores.push(OreDef {
                ore_type: def.get("ore_type").unwrap_or_else(|_| "scatter".into()),
                ore: def.get("ore").unwrap_or_default(),
                wherein: def.get::<Value>("wherein").map(parse_string_list).unwrap_or_default(),
                clust_scarcity: def.get("clust_scarcity").unwrap_or(512),
                clust_num_ores: def.get("clust_num_ores").unwrap_or(4),
                clust_size: def.get("clust_size").unwrap_or(2),
                y_min: def.get("y_min").unwrap_or(-64),
                y_max: def.get("y_max").unwrap_or(64),
            });
            Ok(reg.ores.len() as u32)
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("clear_registered_ores", lua.create_function(move |_, ()| {
            s.registries.write().ores.clear();
            Ok(())
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("register_decoration", lua.create_function(move |_, def: Table| {
            let mut reg = s.registries.write();
            let id = reg.decorations.len() as u32 + 1;
            reg.decorations.push(DecorationDef {
                deco_type: def.get("deco_type").unwrap_or_else(|_| "simple".into()),
                name: def.get("name").unwrap_or_else(|_| format!("deco_{id}")),
                place_on: def.get::<Value>("place_on").map(parse_string_list).unwrap_or_default(),
                fill_ratio: def.get("fill_ratio").unwrap_or(0.01),
                y_min: def.get("y_min").unwrap_or(1),
                y_max: def.get("y_max").unwrap_or(31000),
                decoration: def.get::<Value>("decoration").map(parse_string_list).unwrap_or_default(),
                schematic: def.get::<String>("schematic").ok(),
                flags: def.get("flags").unwrap_or_default(),
                rotation: def.get("rotation").unwrap_or_default(),
            });
            Ok(id)
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("clear_registered_decorations", lua.create_function(move |_, ()| {
            s.registries.write().decorations.clear();
            Ok(())
        })?)?;
    }
    core.set("register_schematic", lua.create_function(|_, _def: Value| Ok(1u32))?)?;
    core.set("clear_registered_schematics", lua.create_function(|_, ()| Ok(()))?)?;
    core.set("read_schematic", lua.create_function(|lua, (spec, opts): (Value, Option<Table>)| {
        if let Value::Table(t) = spec {
            return Ok(Value::Table(t));
        }
        let Some(schem) = schematic_from_lua_value(spec) else {
            return Ok(Value::Nil);
        };
        let write_yslice: String = opts
            .as_ref()
            .and_then(|o| o.get("write_yslice_prob").ok())
            .unwrap_or_else(|| "all".into());

        let out = lua.create_table()?;
        let sz_tbl = lua.create_table()?;
        sz_tbl.set("x", schem.size.x)?;
        sz_tbl.set("y", schem.size.y)?;
        sz_tbl.set("z", schem.size.z)?;
        out.set("size", sz_tbl)?;

        if write_yslice != "none" {
            let ys_tbl = lua.create_table_with_capacity(schem.yslice_prob.len(), 0)?;
            let mut idx = 1usize;
            for (i, &p) in schem.yslice_prob.iter().enumerate() {
                if p < MTSCHEM_PROB_ALWAYS || write_yslice != "low" {
                    let row = lua.create_table()?;
                    row.set("ypos", i)?;
                    row.set("prob", (p as u16) * 2)?;
                    ys_tbl.set(idx, row)?;
                    idx += 1;
                }
            }
            out.set("yslice_prob", ys_tbl)?;
        }

        let data_tbl = lua.create_table_with_capacity(schem.data.len(), 0)?;
        for (i, node) in schem.data.iter().enumerate() {
            let nt = lua.create_table()?;
            nt.set("name", node.name.as_str())?;
            nt.set("prob", (node.prob as u16) * 2)?;
            nt.set("param2", node.param2)?;
            if node.force_place {
                nt.set("force_place", true)?;
            }
            data_tbl.set(i + 1, nt)?;
        }
        out.set("data", data_tbl)?;
        Ok(Value::Table(out))
    })?)?;
    core.set("serialize_schematic", lua.create_function(|_, (spec, fmt, _opts): (Value, Option<String>, Option<Table>)| {
        let format = fmt.unwrap_or_else(|| "lua".into());
        let Some(schem) = schematic_from_lua_value(spec) else {
            return Ok(String::new());
        };
        if format == "mts" {
            let bytes = schem.to_mts_bytes().unwrap_or_default();
            return Ok(String::from_utf8_lossy(&bytes).to_string());
        }
        let mut out = format!(
            "schematic = {{\n\tsize = {{x={}, y={}, z={}}},\n\tyslice_prob = {{\n",
            schem.size.x, schem.size.y, schem.size.z
        );
        for (i, &p) in schem.yslice_prob.iter().enumerate() {
            out.push_str(&format!("\t\t{{ypos={i}, prob={}}},\n", (p as u16) * 2));
        }
        out.push_str("\t},\n\tdata = {\n");
        for node in &schem.data {
            let fp = if node.force_place { ", force_place=true" } else { "" };
            out.push_str(&format!(
                "\t\t{{name=\"{}\", prob={}, param2={}{fp}}},\n",
                node.name.replace('"', "\\\""),
                (node.prob as u16) * 2,
                node.param2
            ));
        }
        out.push_str("\t},\n}\n");
        Ok(out)
    })?)?;
    {
        let s = shared.clone();
        core.set("place_schematic", lua.create_function(move |_, args: MultiValue| {
            let mut it = args.into_iter();
            let Some(Value::Table(pos_t)) = it.next() else {
                return Ok(false);
            };
            let Some(schem_val) = it.next() else {
                return Ok(false);
            };
            let rot_str = match it.next() {
                Some(Value::String(st)) => st.to_str().map(|x| x.to_string()).unwrap_or_default(),
                _ => String::new(),
            };
            let rep_val = it.next();
            let force_placement = match it.next() {
                Some(Value::Boolean(b)) => b,
                _ => true,
            };
            let flags_val = it.next();

            let Some(schem) = schematic_from_lua_value(schem_val) else {
                return Ok(false);
            };
            let pos = parse_ivec3(&pos_t);
            let rot = SchematicRotation::from_str(&rot_str);
            let replacements = parse_replacements_map(rep_val);
            let flags = parse_place_flags(flags_val, force_placement);
            let reg = s.registries.read();
            let seed = s.world.seed ^ ((pos.x as i64 as u64) << 16) ^ (pos.z as i64 as u64);
            schem.place_on_world(
                &s.world,
                &reg,
                pos,
                rot,
                flags,
                if replacements.is_empty() { None } else { Some(&replacements) },
                seed,
            );
            Ok(true)
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("place_schematic_on_vmanip", lua.create_function(move |_, args: MultiValue| {
            let mut it = args.into_iter();
            let Some(Value::UserData(vm_ud)) = it.next() else {
                return Ok(false);
            };
            let Ok(vm) = vm_ud.borrow::<VoxelManipUD>() else {
                return Ok(false);
            };
            let Some(Value::Table(pos_t)) = it.next() else {
                return Ok(false);
            };
            let Some(schem_val) = it.next() else {
                return Ok(false);
            };
            let rot_str = match it.next() {
                Some(Value::String(st)) => st.to_str().map(|x| x.to_string()).unwrap_or_default(),
                _ => String::new(),
            };
            let rep_val = it.next();
            let force_placement = match it.next() {
                Some(Value::Boolean(b)) => b,
                _ => true,
            };
            let flags_val = it.next();

            let Some(schem) = schematic_from_lua_value(schem_val) else {
                return Ok(false);
            };
            let pos = parse_ivec3(&pos_t);
            let rot = SchematicRotation::from_str(&rot_str);
            let replacements = parse_replacements_map(rep_val);
            let flags = parse_place_flags(flags_val, force_placement);
            let reg = s.registries.read();
            let emin = *vm.emin.read();
            let emax = *vm.emax.read();
            let mut data = vm.data.write();
            let mut light = vm.light_data.write();
            let mut p2 = vm.param2_data.write();
            let seed = s.world.seed ^ ((pos.x as i64 as u64) << 16) ^ (pos.z as i64 as u64);
            let fits = schem.place_on_vmanip(
                emin,
                emax,
                &mut data,
                &mut light,
                &mut p2,
                &reg,
                pos,
                rot,
                flags,
                force_placement,
                if replacements.is_empty() { None } else { Some(&replacements) },
                seed,
            );
            *vm.modified.write() = true;
            Ok(fits)
        })?)?;
    }
    core.set("get_decoration_id", lua.create_function(|_, _name: String| Ok(1u32))?)?;
    {
        let s = shared.clone();
        core.set("get_biome_id", lua.create_function(move |_, name: String| {
            Ok(s.registries.read().biomes.iter().find(|b| b.name == name).map(|b| b.id).unwrap_or(0))
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("get_biome_name", lua.create_function(move |_, id: u32| {
            Ok(s.registries
                .read()
                .biomes
                .iter()
                .find(|b| b.id == id)
                .map(|b| b.name.clone())
                .unwrap_or_else(|| "Plains".into()))
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("get_biome_data", lua.create_function(move |lua, pos: Table| {
            let p = parse_ivec3(&pos);
            let bname = crate::world::mapgen::biome_name_at(p.x, p.z, s.world.seed);
            let canon = match bname {
                "Plains" => "Plains",
                "Forest" => "Forest",
                "Birch Forest" => "BirchForest",
                "Desert" => "Desert",
                "Taiga" => "Taiga",
                "Swamp" => "Swampland",
                "Snowy Tundra" => "IcePlains",
                _ => "Plains",
            };
            let reg = s.registries.read();
            let id = reg
                .biomes
                .iter()
                .find(|b| b.name.eq_ignore_ascii_case(canon))
                .or_else(|| reg.biomes.iter().find(|b| b.name.contains(canon)))
                .or_else(|| reg.biomes.first())
                .map(|b| b.id)
                .unwrap_or(1);
            let t = lua.create_table()?;
            t.set("biome", id)?;
            t.set("heat", 50.0f64)?;
            t.set("humidity", 50.0f64)?;
            Ok(t)
        })?)?;
    }
    core.set("get_heat", lua.create_function(|_, _pos: Table| Ok(50.0f64))?)?;
    core.set("get_humidity", lua.create_function(|_, _pos: Table| Ok(50.0f64))?)?;

    // Mapgen & IPC APIs
    {
        let s = shared.clone();
        core.set("get_mapgen_setting", lua.create_function(move |_, k: String| {
            Ok(s.mapgen_settings.read().get(&k).cloned())
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("set_mapgen_setting", lua.create_function(move |_, (k, v, ov): (String, String, Option<bool>)| {
            let mut g = s.mapgen_settings.write();
            if ov.unwrap_or(false) || !g.contains_key(&k) {
                g.insert(k, v);
            }
            Ok(())
        })?)?;
    }
    core.set("get_mapgen_setting_noiseparams", lua.create_function(|lua, _k: String| {
        let t = lua.create_table()?;
        t.set("offset", 0.0)?;
        t.set("scale", 1.0)?;
        t.set("spread", make_vec3(lua, 250.0, 250.0, 250.0)?)?;
        t.set("seed", 1337)?;
        t.set("octaves", 3)?;
        t.set("persist", 0.6)?;
        t.set("lacunarity", 2.0)?;
        Ok(t)
    })?)?;
    core.set("set_mapgen_setting_noiseparams", lua.create_function(|_, _: MultiValue| Ok(()))?)?;
    {
        let s = shared.clone();
        core.set("get_mapgen_params", lua.create_function(move |lua, ()| {
            let g = s.mapgen_settings.read();
            let t = lua.create_table()?;
            t.set("mgname", g.get("mg_name").map(String::as_str).unwrap_or("v7"))?;
            t.set("seed", g.get("seed").and_then(|v| v.parse::<u64>().ok()).unwrap_or(1337))?;
            t.set("water_level", g.get("water_level").and_then(|v| v.parse::<i32>().ok()).unwrap_or(1))?;
            t.set("chunksize", g.get("chunksize").and_then(|v| v.parse::<i32>().ok()).unwrap_or(5))?;
            t.set("flags", g.get("mg_flags").map(String::as_str).unwrap_or(""))?;
            Ok(t)
        })?)?;
    }
    core.set("get_mapgen_edges", lua.create_function(|lua, _limit: Option<i32>| {
        Ok((make_vec3(lua, -31000.0, -31000.0, -31000.0)?, make_vec3(lua, 31000.0, 31000.0, 31000.0)?))
    })?)?;
    core.set("get_mapgen_chunksize", lua.create_function(|lua, ()| make_vec3(lua, 5.0, 5.0, 5.0))?)?;
    core.set("set_gen_notify", lua.create_function(|_, _: MultiValue| Ok(()))?)?;
    core.set("get_gen_notify", lua.create_function(|lua, ()| Ok((lua.create_table()?, lua.create_table()?)))?)?;
    core.set("save_gen_notify", lua.create_function(|_, _: MultiValue| Ok(true))?)?;
    core.set("register_mapgen_script", lua.create_function(|_, _path: String| Ok(()))?)?;
    core.set("get_mapgen_object", lua.create_function(|_, _: String| Ok(Value::Nil))?)?;

    // IPC store
    {
        let s = shared.clone();
        core.set("ipc_get", lua.create_function(move |lua, k: String| {
            match s.ipc_store.read().get(&k) {
                Some(raw) => {
                    if let Ok(jv) = serde_json::from_str::<serde_json::Value>(raw) {
                        json_to_lua_value(lua, &jv, &Value::Nil)
                    } else {
                        Ok(Value::String(lua.create_string(raw)?))
                    }
                }
                None => Ok(Value::Nil),
            }
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("ipc_set", lua.create_function(move |_, (k, v): (String, Value)| {
            let jv = lua_value_to_json(v);
            if matches!(jv, serde_json::Value::Null) {
                s.ipc_store.write().remove(&k);
            } else {
                s.ipc_store.write().insert(k, jv.to_string());
            }
            Ok(())
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("ipc_cas", lua.create_function(move |_, (k, expected, new_v): (String, Value, Value)| {
            let exp_s = lua_value_to_json(expected);
            let new_s = lua_value_to_json(new_v);
            let mut g = s.ipc_store.write();
            let cur_matches = match g.get(&k) {
                None => matches!(exp_s, serde_json::Value::Null),
                Some(raw) => raw == &exp_s.to_string(),
            };
            if cur_matches {
                if matches!(new_s, serde_json::Value::Null) {
                    g.remove(&k);
                } else {
                    g.insert(k, new_s.to_string());
                }
                return Ok(true);
            }
            Ok(false)
        })?)?;
    }
    core.set("ipc_poll", lua.create_function(|_, _: MultiValue| Ok(false))?)?;

    // Noise helpers on core
    core.set("get_perlin", lua.create_function(|_, np: Value| Ok(PerlinNoiseUD(NoiseParams::from_lua(np))))?)?;
    core.set("get_value_noise", lua.create_function(|_, np: Value| Ok(PerlinNoiseUD(NoiseParams::from_lua(np))))?)?;
    let pmap_fn = lua.create_function(|_, (np, size): (Value, Table)| {
        let sx: usize = size.get("x").unwrap_or(16);
        let sy: usize = size.get("y").unwrap_or(16);
        let sz: usize = size.get("z").unwrap_or(1);
        Ok(PerlinNoiseMapUD::new(NoiseParams::from_lua(np), [sx, sy, sz]))
    })?;
    core.set("get_perlin_map", pmap_fn.clone())?;
    core.set("get_value_noise_map", pmap_fn)?;

    // Mod storage & detached inventories
    {
        let s = shared.clone();
        core.set("get_mod_storage", lua.create_function(move |_, ()| {
            let modname = s.current_modname.read().clone().unwrap_or_else(|| "global".into());
            let mut g = s.mod_storages.write();
            let st = g.entry(modname).or_insert_with(|| Arc::new(RwLock::new(MetaDataState::default()))).clone();
            Ok(MetaDataRefUD(st))
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("create_detached_inventory_raw", lua.create_function(move |_, (name, _cb, _player): (String, Value, Option<String>)| {
            let mut g = s.detached_inventories.write();
            let st = g.entry(name.clone()).or_insert_with(|| Arc::new(RwLock::new(MetaDataState::default()))).clone();
            Ok(InvRefUD { state: st, location: format!("detached:{name}") })
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("remove_detached_inventory_raw", lua.create_function(move |_, name: String| {
            Ok(s.detached_inventories.write().remove(&name).is_some())
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("get_inventory", lua.create_function(move |lua, loc: Table| {
            let kind: String = loc.get("type").unwrap_or_default();
            let name: String = loc.get("name").unwrap_or_default();
            if kind == "player" {
                if let Some(p) = s.players.read().get(&name) {
                    return Ok(Value::UserData(lua.create_userdata(InvRefUD {
                        state: p.read().meta.clone(),
                        location: "player".into(),
                    })?));
                }
            } else if kind == "detached" {
                let mut g = s.detached_inventories.write();
                let st = g.entry(name.clone()).or_insert_with(|| Arc::new(RwLock::new(MetaDataState::default()))).clone();
                return Ok(Value::UserData(lua.create_userdata(InvRefUD {
                    state: st,
                    location: format!("detached:{name}"),
                })?));
            }
            Ok(Value::Nil)
        })?)?;
    }

    // World node access
    {
        let s = shared.clone();
        core.set("get_node", lua.create_function(move |lua, pos: Table| {
            let p = parse_ivec3(&pos);
            let (id, p1, p2) = s.world.get_node(p);
            let name = s.registries.read().get_name_from_content_id(id).to_string();
            let t = lua.create_table()?;
            t.set("name", name)?;
            t.set("param1", p1)?;
            t.set("param2", p2)?;
            Ok(t)
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("get_node_or_nil", lua.create_function(move |lua, pos: Table| {
            let p = parse_ivec3(&pos);
            let (id, p1, p2) = s.world.get_node(p);
            if id == CONTENT_IGNORE {
                return Ok(Value::Nil);
            }
            let name = s.registries.read().get_name_from_content_id(id).to_string();
            let t = lua.create_table()?;
            t.set("name", name)?;
            t.set("param1", p1)?;
            t.set("param2", p2)?;
            Ok(Value::Table(t))
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("get_node_raw", lua.create_function(move |_, (x, y, z): (i32, i32, i32)| {
            let (id, p1, p2) = s.world.get_node(IVec3::new(x, y, z));
            Ok((id, p1, p2, id != CONTENT_IGNORE))
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("set_node", lua.create_function(move |lua, (pos, node): (Table, Table)| {
            let p = parse_ivec3(&pos);
            let name: String = node.get("name").unwrap_or_else(|_| "air".into());
            let p1: u8 = node.get("param1").unwrap_or(15);
            let p2: u8 = node.get("param2").unwrap_or(0);
            set_node_with_callbacks(lua, &s, p, &name, p1, p2);
            Ok(())
        })?)?;
    }
    core.set("add_node", core.get::<mlua::Function>("set_node")?)?;
    {
        let s = shared.clone();
        core.set("swap_node", lua.create_function(move |_, (pos, node): (Table, Table)| {
            let p = parse_ivec3(&pos);
            let name: String = node.get("name").unwrap_or_else(|_| "air".into());
            let p2: u8 = node.get("param2").unwrap_or(0);
            let (id, light_source) = {
                let reg = s.registries.read();
                let id = reg.get_content_id(&name);
                let ls = reg.get_node_def(id).map(|d| d.light_source).unwrap_or(0);
                (id, ls)
            };
            s.world.swap_node(p, id, p2);
            if light_source > 0 {
                let (_, old_p1, _) = s.world.get_node(p);
                let new_p1 = (old_p1 & 0x0F) | ((light_source.min(15)) << 4);
                s.world.set_node(p, id, new_p1, p2);
            }
            s.world.mark_priority_dirty_with_neighbors(p);
            Ok(())
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("remove_node", lua.create_function(move |lua, pos: Table| {
            let p = parse_ivec3(&pos);
            set_node_with_callbacks(lua, &s, p, "air", 15, 0);
            Ok(())
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("bulk_set_node", lua.create_function(move |_, (positions, node): (Table, Table)| {
            let name: String = node.get("name").unwrap_or_else(|_| "air".into());
            let p1: u8 = node.get("param1").unwrap_or(15);
            let p2: u8 = node.get("param2").unwrap_or(0);
            let id = s.registries.read().get_content_id(&name);
            let len = positions.len().unwrap_or(0);
            for i in 1..=len {
                if let Ok(pt) = positions.get::<Table>(i) {
                    s.world.set_node(parse_ivec3(&pt), id, p1, p2);
                }
            }
            Ok(())
        })?)?;
    }
    core.set("bulk_swap_node", core.get::<mlua::Function>("bulk_set_node")?)?;
    {
        let s = shared.clone();
        core.set("find_nodes_in_area", lua.create_function(move |lua, (p1, p2, nodenames, grouped): (Table, Table, Value, Option<bool>)| {
            let lo = parse_ivec3(&p1);
            let hi = parse_ivec3(&p2);
            let ids = resolve_query_node_ids(&s, nodenames);
            let found = s.world.find_nodes_in_area(lo, hi, &ids);
            if grouped.unwrap_or(false) {
                let out = lua.create_table()?;
                return Ok(Value::Table(out));
            }
            let out = lua.create_table_with_capacity(found.len(), 0)?;
            for (i, v) in found.iter().enumerate() {
                out.set(i + 1, make_vec3(lua, v.x as f64, v.y as f64, v.z as f64)?)?;
            }
            Ok(Value::Table(out))
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("find_nodes_in_area_under_air", lua.create_function(move |lua, (p1, p2, nodenames): (Table, Table, Value)| {
            let lo = parse_ivec3(&p1);
            let hi = parse_ivec3(&p2);
            let ids = resolve_query_node_ids(&s, nodenames);
            let found = s.world.find_nodes_in_area(lo, hi, &ids);
            let under_air: Vec<IVec3> = found
                .into_iter()
                .filter(|&v| s.world.get_node(v + IVec3::Y).0 == CONTENT_AIR)
                .collect();
            let out = lua.create_table_with_capacity(under_air.len(), 0)?;
            for (i, v) in under_air.iter().enumerate() {
                out.set(i + 1, make_vec3(lua, v.x as f64, v.y as f64, v.z as f64)?)?;
            }
            Ok(Value::Table(out))
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("find_node_near", lua.create_function(move |lua, (pos, radius, nodenames, search_center): (Table, f64, Value, Option<bool>)| {
            let c = parse_ivec3(&pos);
            let radius = radius.clamp(0.0, 32.0);
            let r = radius.ceil() as i32;
            let r2 = radius * radius;
            let names = parse_string_list(nodenames);
            let mut ids = HashSet::new();
            {
                let reg = s.registries.read();
                for n in names {
                    if let Some(grp) = n.strip_prefix("group:") {
                        for def in reg.nodes_by_id.iter().flatten() {
                            if def.groups.get(grp).copied().unwrap_or(0) > 0 {
                                ids.insert(def.id);
                            }
                        }
                    } else {
                        ids.insert(reg.get_content_id(&n));
                    }
                }
            }
            if search_center.unwrap_or(false) && ids.contains(&s.world.get_node(c).0) {
                return Ok(Value::Table(make_vec3(lua, c.x as f64, c.y as f64, c.z as f64)?));
            }
            let mut best: Option<(i32, IVec3)> = None;
            for dy in -r..=r {
                for dz in -r..=r {
                    for dx in -r..=r {
                        if dx == 0 && dy == 0 && dz == 0 {
                            continue;
                        }
                        let d2 = dx * dx + dy * dy + dz * dz;
                        if (d2 as f64) > r2 {
                            continue;
                        }
                        let p = c + IVec3::new(dx, dy, dz);
                        let (id, _, _) = s.world.get_node(p);
                        if ids.contains(&id) {
                            if best.map(|(bd, _)| d2 < bd).unwrap_or(true) {
                                best = Some((d2, p));
                            }
                        }
                    }
                }
            }
            if let Some((_, p)) = best {
                Ok(Value::Table(make_vec3(lua, p.x as f64, p.y as f64, p.z as f64)?))
            } else {
                Ok(Value::Nil)
            }
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("get_voxel_manip", lua.create_function(move |_, (p1, p2): (Option<Table>, Option<Table>)| {
            Ok(VoxelManipUD::new(
                s.world.clone(),
                s.registries.clone(),
                p1.as_ref().map(parse_ivec3),
                p2.as_ref().map(parse_ivec3),
            ))
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("get_meta", lua.create_function(move |_, pos: Table| {
            let p = parse_ivec3(&pos);
            Ok(MetaDataRefUD(s.world.get_node_meta(p)))
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("get_node_timer", lua.create_function(move |_, pos: Table| {
            Ok(NodeTimerRefUD {
                world: s.world.clone(),
                pos: parse_ivec3(&pos),
            })
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("get_node_light", lua.create_function(move |_, (pos, tod_opt): (Table, Option<f32>)| {
            let (_, p1, _) = s.world.get_node(parse_ivec3(&pos));
            let day = p1 & 0x0F;
            let block = (p1 >> 4) & 0x0F;
            let tod = tod_opt.unwrap_or_else(|| *s.time_of_day.read()).rem_euclid(1.0);
            let sun_factor = if (0.25..=0.75).contains(&tod) {
                1.0
            } else if (0.18..0.25).contains(&tod) {
                (tod - 0.18) / 0.07
            } else if (0.75..=0.82).contains(&tod) {
                (0.82 - tod) / 0.07
            } else {
                0.0
            };
            let eff_sky = ((day as f32) * sun_factor).round() as u8;
            Ok(Some(eff_sky.max(block)))
        })?)?;
    }
    core.set("get_natural_light", core.get::<mlua::Function>("get_node_light")?)?;
    {
        let s = shared.clone();
        core.set("get_node_level", lua.create_function(move |_, pos: Table| {
            let (id, _, p2) = s.world.get_node(parse_ivec3(&pos));
            let reg = s.registries.read();
            let lvl = reg.get_node_def(id).map(|d| {
                if d.paramtype2 == "leveled" {
                    p2 & 0x7F
                } else if d.paramtype2 == "flowingliquid" {
                    p2 & 0x07
                } else {
                    0
                }
            }).unwrap_or(0);
            Ok(lvl)
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("set_node_level", lua.create_function(move |_, (pos, lvl): (Table, Option<u8>)| {
            let p = parse_ivec3(&pos);
            let (id, _, p2) = s.world.get_node(p);
            let max_lvl = s.registries.read().get_node_def(id).map(|d| {
                if d.leveled > 0 { d.leveled } else if d.paramtype2 == "flowingliquid" { 7 } else { 63 }
            }).unwrap_or(63);
            let req = lvl.unwrap_or(1);
            let clamped = req.min(max_lvl);
            s.world.swap_node(p, id, (p2 & 0x80) | clamped);
            Ok(req.saturating_sub(clamped))
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("add_node_level", lua.create_function(move |_, (pos, lvl): (Table, Option<i32>)| {
            let p = parse_ivec3(&pos);
            let (id, _, p2) = s.world.get_node(p);
            let max_lvl = s.registries.read().get_node_def(id).map(|d| {
                if d.leveled > 0 { d.leveled as i32 } else if d.paramtype2 == "flowingliquid" { 7 } else { 63 }
            }).unwrap_or(63);
            let cur = (p2 & 0x7F) as i32;
            let target = cur + lvl.unwrap_or(1);
            let clamped = target.clamp(0, max_lvl);
            s.world.swap_node(p, id, (p2 & 0x80) | (clamped as u8));
            Ok((target - clamped).max(0) as u8)
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("get_node_max_level", lua.create_function(move |_, pos: Table| {
            let (id, _, _) = s.world.get_node(parse_ivec3(&pos));
            let max_lvl = s.registries.read().get_node_def(id).map(|d| {
                if d.leveled > 0 { d.leveled } else if d.paramtype2 == "flowingliquid" { 7 } else { 63 }
            }).unwrap_or(63);
            Ok(max_lvl)
        })?)?;
    }
    core.set("emerge_area", lua.create_function(|_, _: MultiValue| Ok(()))?)?;
    core.set("load_area", lua.create_function(|_, _: MultiValue| Ok(()))?)?;
    core.set("delete_area", lua.create_function(|_, _: MultiValue| Ok(()))?)?;
    core.set("fix_light", lua.create_function(|_, _: MultiValue| Ok(true))?)?;
    core.set("forceload_block", lua.create_function(|_, _: MultiValue| Ok(true))?)?;
    core.set("forceload_free_block", lua.create_function(|_, _: MultiValue| Ok(()))?)?;
    {
        let s = shared.clone();
        core.set("compare_block_status", lua.create_function(move |_, (pos, status): (Table, String)| {
            let p = parse_ivec3(&pos);
            let (cpos, _, _, _) = crate::world::world_to_chunk(p);
            if status == "active" {
                Ok(s.world.get_chunk(cpos).is_some())
            } else {
                Ok(true)
            }
        })?)?;
    }
    {
        let s = shared.clone();
        core.set(
            "get_node_boxes",
            lua.create_function(
                move |lua, (box_type, pos_t, node_opt): (String, Table, Option<Table>)| {
                    let p = parse_ivec3(&pos_t);
                    let reg = s.registries.read();
                    let (id, p2) = if let Some(ref nt) = node_opt {
                        let name: String = nt.get("name").unwrap_or_default();
                        let id = reg.get_content_id(&name);
                        let p2: u8 = nt.get("param2").unwrap_or(0);
                        (id, p2)
                    } else {
                        let (id, _, p2) = s.world.get_node(p);
                        (id, p2)
                    };
                    let out = lua.create_table()?;
                    if id == crate::registry::CONTENT_AIR || id == crate::registry::CONTENT_IGNORE {
                        return Ok(out);
                    }
                    let Some(def) = reg.get_node_def(id) else {
                        return Ok(out);
                    };
                    if !def.walkable && box_type == "collision_box" {
                        return Ok(out);
                    }
                    let boxes = if box_type == "selection_box" {
                        let nb = crate::physics::collision::compute_connected_neighbors(
                            p, def, &s.world, &reg,
                        );
                        crate::physics::collision::world_selection_boxes_with_neighbors(
                            p, def, p2, nb,
                        )
                    } else {
                        crate::physics::collision::world_collision_boxes(
                            p, id, p2, &s.world, &reg,
                        )
                    };
                    let origin = p.as_vec3() + Vec3::splat(0.5);
                    for (i, b) in boxes.into_iter().enumerate() {
                        let row = lua.create_table()?;
                        let mn = b.min - origin;
                        let mx = b.max - origin;
                        row.set(1, mn.x as f64)?;
                        row.set(2, mn.y as f64)?;
                        row.set(3, mn.z as f64)?;
                        row.set(4, mx.x as f64)?;
                        row.set(5, mx.y as f64)?;
                        row.set(6, mx.z as f64)?;
                        out.set(i + 1, row)?;
                    }
                    Ok(out)
                },
            )?,
        )?;
    }

    // Dig params (Luanti-compatible 1:1 tool group calculation)
    core.set("get_dig_params", lua.create_function(|lua, (groups_opt, toolcaps_opt, wear_opt): (Option<Table>, Option<Table>, Option<u32>)| {
        let mut groups = HashMap::new();
        if let Some(gt) = groups_opt {
            for pair in gt.pairs::<String, i32>() {
                if let Ok((k, v)) = pair {
                    groups.insert(k, v);
                }
            }
        }
        let tc = toolcaps_opt.and_then(|t| parse_tool_capabilities(Value::Table(t)));
        let initial_wear = wear_opt.unwrap_or(0).min(65535) as u16;
        let dp = compute_dig_params_with_wear(&groups, tc.as_ref(), false, initial_wear);
        let out = lua.create_table()?;
        out.set("diggable", dp.diggable)?;
        out.set("time", dp.time)?;
        out.set("wear", dp.wear)?;
        Ok(out)
    })?)?;

    // Time & timing
    {
        let s = shared.clone();
        core.set("get_us_time", lua.create_function(move |_, ()| {
            Ok(s.start_instant.elapsed().as_micros() as u64)
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("get_gametime", lua.create_function(move |_, ()| Ok(*s.gametime.read()))?)?;
    }
    {
        let s = shared.clone();
        core.set("get_timeofday", lua.create_function(move |_, ()| Ok(*s.time_of_day.read()))?)?;
    }
    {
        let s = shared.clone();
        core.set("set_timeofday", lua.create_function(move |_, v: f32| {
            *s.time_of_day.write() = v.rem_euclid(1.0);
            Ok(())
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("get_day_count", lua.create_function(move |_, ()| Ok(*s.day_count.read()))?)?;
    }
    {
        let s = shared.clone();
        core.set("time_to_day_night_ratio", lua.create_function(move |_, tod_opt: Option<f32>| {
            let tod = tod_opt.unwrap_or_else(|| *s.time_of_day.read()).rem_euclid(1.0);
            let ratio = if (0.25..=0.75).contains(&tod) {
                1.0
            } else if (0.18..0.25).contains(&tod) {
                (tod - 0.18) / 0.07
            } else if (0.75..=0.82).contains(&tod) {
                (0.82 - tod) / 0.07
            } else {
                0.0
            };
            Ok(ratio.clamp(0.0, 1.0) as f64)
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("line_of_sight", lua.create_function(move |lua, (pos1, pos2): (Table, Table)| {
            let a = parse_vec3f(&pos1);
            let b = parse_vec3f(&pos2);
            let delta = b - a;
            let dist = delta.length();
            if dist <= 1e-4 {
                return Ok((true, Value::Nil));
            }
            let steps = (dist / 0.25).ceil().clamp(1.0, 512.0) as usize;
            let reg = s.registries.read();
            for i in 1..steps {
                let t = i as f32 / steps as f32;
                let p = a + delta * t;
                let ip = IVec3::new(p.x.round() as i32, p.y.round() as i32, p.z.round() as i32);
                let (id, _, _) = s.world.get_node(ip);
                if id != CONTENT_AIR && id != CONTENT_IGNORE {
                    if let Some(def) = reg.get_node_def(id) {
                        if def.walkable {
                            let hit = make_vec3(lua, ip.x as f64, ip.y as f64, ip.z as f64)?;
                            return Ok((false, Value::Table(hit)));
                        }
                    }
                }
            }
            Ok((true, Value::Nil))
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("raycast", lua.create_function(move |lua, (pos1, pos2, objects_opt, liquids_opt, _pointabilities): (Table, Table, Option<bool>, Option<bool>, Option<Table>)| {
            let a = parse_vec3f(&pos1);
            let b = parse_vec3f(&pos2);
            let delta = b - a;
            let dist = delta.length();
            let check_objects = objects_opt.unwrap_or(true);
            let check_liquids = liquids_opt.unwrap_or(false);

            struct RawRayHit {
                dist_sq: f32,
                pt: Table,
            }
            let mut raw_hits: Vec<RawRayHit> = Vec::new();

            if dist > 1e-5 {
                // 1. Node hits along discrete voxel steps using exact box_line_collision
                let dir_n = delta / dist;
                let max_steps = (dist / 0.15).ceil().clamp(1.0, 512.0) as usize;
                let mut visited = HashSet::new();
                let reg = s.registries.read();
                let mut prev_ip = IVec3::new(a.x.round() as i32, a.y.round() as i32, a.z.round() as i32);
                for i in 0..=max_steps {
                    let t = i as f32 / max_steps as f32;
                    let p = a + delta * t;
                    let ip = IVec3::new(p.x.round() as i32, p.y.round() as i32, p.z.round() as i32);
                    if !visited.insert(ip) {
                        prev_ip = ip;
                        continue;
                    }
                    let (id, _, _) = s.world.get_node(ip);
                    if id != CONTENT_AIR && id != CONTENT_IGNORE {
                        if let Some(def) = reg.get_node_def(id) {
                            let is_liq = crate::physics::collision::is_liquid_node(def);
                            let hit_ok = if is_liq { check_liquids } else { def.pointable && def.walkable };
                            if hit_ok {
                                let bmin = ip.as_vec3() - Vec3::splat(0.5);
                                let bmax = ip.as_vec3() + Vec3::splat(0.5);
                                if let Some((hp, hn)) = crate::physics::raycast::box_line_collision(bmin, bmax, a, delta) {
                                    let norm_i = hn.round().as_ivec3();
                                    let above = if norm_i != IVec3::ZERO { ip + norm_i } else { prev_ip };
                                    let tbl = lua.create_table()?;
                                    tbl.set("type", "node")?;
                                    tbl.set("under", make_vec3(lua, ip.x as f64, ip.y as f64, ip.z as f64)?)?;
                                    tbl.set("above", make_vec3(lua, above.x as f64, above.y as f64, above.z as f64)?)?;
                                    tbl.set("intersection_point", make_vec3(lua, hp.x as f64, hp.y as f64, hp.z as f64)?)?;
                                    tbl.set("intersection_normal", make_vec3(lua, hn.x as f64, hn.y as f64, hn.z as f64)?)?;
                                    tbl.set("box_id", 1)?;
                                    raw_hits.push(RawRayHit {
                                        dist_sq: a.distance_squared(hp) + dir_n.length_squared() * 0.0,
                                        pt: tbl,
                                    });
                                }
                            }
                        }
                    }
                    prev_ip = ip;
                }
                drop(reg);

                // 2. Object hits (entities + players) — ObjectState stores Cubex coords
                if check_objects {
                    let obj_a = lua_to_cubex_pos(a);
                    let mut objs: Vec<Arc<RwLock<ObjectState>>> = s
                        .entities
                        .read()
                        .values()
                        .cloned()
                        .collect();
                    objs.extend(s.players.read().values().cloned());
                    for earc in objs {
                        let (pos, cbox, pointable, removed) = {
                            let g = earc.read();
                            (g.pos, g.collisionbox, g.pointable, g.removed)
                        };
                        if removed || !pointable {
                            continue;
                        }
                        let bmin = pos + Vec3::new(cbox[0], cbox[1], cbox[2]);
                        let bmax = pos + Vec3::new(cbox[3], cbox[4], cbox[5]);
                        if let Some((hp, hn)) = crate::physics::raycast::box_line_collision(bmin, bmax, obj_a, delta) {
                            let lua_hp = cubex_to_lua_pos(hp);
                            let obj_ud = object_ref_to_lua(lua, earc)?;
                            let tbl = lua.create_table()?;
                            tbl.set("type", "object")?;
                            tbl.set("ref", obj_ud)?;
                            tbl.set("intersection_point", make_vec3(lua, lua_hp.x as f64, lua_hp.y as f64, lua_hp.z as f64)?)?;
                            tbl.set("intersection_normal", make_vec3(lua, hn.x as f64, hn.y as f64, hn.z as f64)?)?;
                            tbl.set("box_id", 1)?;
                            raw_hits.push(RawRayHit {
                                dist_sq: obj_a.distance_squared(hp),
                                pt: tbl,
                            });
                        }
                    }
                }
            }

            raw_hits.sort_by(|x, y| x.dist_sq.total_cmp(&y.dist_sq));

            let rc = lua.create_table()?;
            let hits_tbl = lua.create_table_with_capacity(raw_hits.len(), 0)?;
            for (i, h) in raw_hits.into_iter().enumerate() {
                hits_tbl.set(i + 1, h.pt)?;
            }
            rc.set("_hits", hits_tbl)?;
            rc.set("_idx", 0usize)?;
            let next_fn = lua.create_function(|_, this: Table| {
                let idx: usize = this.get("_idx").unwrap_or(0) + 1;
                this.set("_idx", idx)?;
                let hits: Table = this.get("_hits")?;
                hits.get::<Value>(idx)
            })?;
            rc.set("next", next_fn.clone())?;
            let mt = lua.create_table()?;
            mt.set("__call", next_fn)?;
            let _ = rc.set_metatable(Some(mt));
            Ok(rc)
        })?)?;
    }

    // Players, Chat, Formspecs, Privs & Auth
    {
        let s = shared.clone();
        core.set("get_connected_players", lua.create_function(move |lua, ()| {
            let pl = s.players.read();
            let t = lua.create_table_with_capacity(pl.len(), 0)?;
            for (i, obj) in pl.values().enumerate() {
                t.set(i + 1, object_ref_to_lua(lua, obj.clone())?)?;
            }
            Ok(t)
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("get_player_by_name", lua.create_function(move |lua, name: String| {
            if let Some(obj) = s.players.read().get(&name) {
                Ok(Value::UserData(object_ref_to_lua(lua, obj.clone())?))
            } else {
                Ok(Value::Nil)
            }
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("get_player_privs", lua.create_function(move |lua, name: String| {
            let privs = s.player_privs.read();
            let t = lua.create_table()?;
            if let Some(m) = privs.get(&name) {
                for (k, &v) in m {
                    if v { t.set(k.as_str(), true)?; }
                }
            } else {
                t.set("interact", true)?;
                t.set("shout", true)?;
            }
            Ok(t)
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("set_player_privs", lua.create_function(move |_, (name, tbl): (String, Table)| {
            let mut m = HashMap::new();
            for pair in tbl.pairs::<String, bool>() {
                if let Ok((k, v)) = pair {
                    m.insert(k, v);
                }
            }
            s.player_privs.write().insert(name, m);
            Ok(())
        })?)?;
    }
    core.set("get_player_information", lua.create_function(|lua, _name: String| {
        let t = lua.create_table()?;
        t.set("address", "127.0.0.1")?;
        t.set("ip_version", 4)?;
        t.set("protocol_version", 46)?;
        t.set("formspec_version", 6)?;
        t.set("lang_code", "en")?;
        Ok(t)
    })?)?;
    core.set("get_player_window_information", lua.create_function(|lua, _name: String| {
        let t = lua.create_table()?;
        let sz = lua.create_table()?;
        sz.set("x", 1280)?; sz.set("y", 720)?;
        t.set("size", sz)?;
        t.set("real_gui_scaling", 1.0f32)?;
        t.set("real_hud_scaling", 1.0f32)?;
        Ok(t)
    })?)?;
    {
        let s = shared.clone();
        core.set("player_exists", lua.create_function(move |_, name: String| {
            Ok(s.players.read().contains_key(&name))
        })?)?;
    }
    core.set("is_singleplayer", lua.create_function(|_, ()| Ok(true))?)?;
    {
        let s = shared.clone();
        core.set("show_formspec", lua.create_function(move |_, (pname, fname, fs): (String, String, String)| {
            s.pending_formspecs.write().push(FormspecRequest {
                player_name: pname,
                formname: fname,
                formspec: fs,
            });
            Ok(())
        })?)?;
    }
    core.set("close_formspec", lua.create_function(|_, (_pname, _fname): (String, String)| Ok(()))?)?;
    {
        let s = shared.clone();
        core.set("chat_send_all", lua.create_function(move |_, msg: String| {
            s.chat_log.write().push(msg);
            Ok(())
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("chat_send_player", lua.create_function(move |_, (_p, msg): (String, String)| {
            s.chat_log.write().push(msg);
            Ok(())
        })?)?;
    }
    core.set("display_chat_message", lua.create_function(|_, _msg: String| Ok(()))?)?;

    // Entities, Particles, Sound & Objects
    {
        let s = shared.clone();
        core.set("add_entity", lua.create_function(move |lua, (pos, name, staticdata): (Table, String, Option<String>)| {
            let Ok(core_t) = lua.globals().get::<Table>("core") else {
                return Ok(Value::Nil);
            };
            let Ok(reg_ents) = core_t.get::<Table>("registered_entities") else {
                return Ok(Value::Nil);
            };
            let lookup_name = name.strip_prefix(':').unwrap_or(name.as_str());
            let proto = reg_ents
                .get::<Table>(lookup_name)
                .or_else(|_| reg_ents.get::<Table>(name.as_str()));
            let Ok(proto) = proto else {
                return Ok(Value::Nil);
            };

            let id = {
                let mut nid = s.next_entity_id.write();
                let cur = *nid;
                *nid += 1;
                cur
            };
            let pvec = lua_to_cubex_pos(parse_vec3f(&pos));
            let mut st = ObjectState::new_entity(id, lookup_name, pvec);
            st.apply_properties_table(&proto);
            if let Ok(ip) = proto.get::<Table>("initial_properties") {
                st.apply_properties_table(&ip);
            }
            st.hp = st.hp_max;
            let arc = Arc::new(RwLock::new(st));
            s.entities.write().insert(id, arc.clone());

            let ent_tbl = lua.create_table()?;
            let ud = object_ref_to_lua(lua, arc.clone())?;
            ent_tbl.set("object", ud.clone())?;
            ent_tbl.set("name", lookup_name)?;
            ent_tbl.set("_id", id)?;
            let _ = ent_tbl.set_metatable(Some(proto.clone()));

            if let Ok(ents_map) = core_t.get::<Table>("_cubex_luaentities") {
                ents_map.set(id, ent_tbl.clone())?;
            }
            if let Ok(on_act) = ent_tbl.get::<mlua::Function>("on_activate") {
                let sd = staticdata.unwrap_or_default();
                let _ = on_act.call::<Value>((ent_tbl.clone(), sd, 0.0f64));
            }
            if let Ok(lh) = ent_tbl.get::<f64>("health") {
                let mut g = arc.write();
                g.hp = (lh.ceil() as i32).clamp(0, g.hp_max);
            }
            if arc.read().removed {
                s.entities.write().remove(&id);
                if let Ok(ents_map) = core_t.get::<Table>("_cubex_luaentities") {
                    let _ = ents_map.set(id, Value::Nil);
                }
                return Ok(Value::Nil);
            }
            Ok(Value::UserData(ud))
        })?)?;
    }
    core.set("add_item", lua.create_function(|lua, (pos, item): (Table, Value)| {
        let Ok(core_t) = lua.globals().get::<Table>("core") else {
            return Ok(Value::Nil);
        };
        if let Ok(spawn_fn) = core_t.get::<mlua::Function>("spawn_item") {
            return spawn_fn.call::<Value>((pos, item));
        }
        Ok(Value::Nil)
    })?)?;
    {
        let s = shared.clone();
        core.set("get_objects_inside_radius", lua.create_function(move |lua, (pos, r): (Table, f64)| {
            let c = lua_to_cubex_pos(parse_vec3f(&pos));
            let r2 = (r as f32) * (r as f32);
            let out = lua.create_table()?;
            let mut idx = 1;
            for p in s.players.read().values() {
                if !p.read().removed && p.read().pos.distance_squared(c) <= r2 {
                    out.set(idx, object_ref_to_lua(lua, p.clone())?)?;
                    idx += 1;
                }
            }
            for e in s.entities.read().values() {
                if !e.read().removed && e.read().pos.distance_squared(c) <= r2 {
                    out.set(idx, object_ref_to_lua(lua, e.clone())?)?;
                    idx += 1;
                }
            }
            Ok(out)
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("get_objects_in_area", lua.create_function(move |lua, (p1, p2): (Table, Table)| {
            let a = lua_to_cubex_pos(parse_vec3f(&p1));
            let b = lua_to_cubex_pos(parse_vec3f(&p2));
            let lo = a.min(b);
            let hi = a.max(b);
            let out = lua.create_table()?;
            let mut idx = 1;
            for p in s.players.read().values() {
                let g = p.read();
                if !g.removed && g.pos.cmpge(lo).all() && g.pos.cmple(hi).all() {
                    out.set(idx, object_ref_to_lua(lua, p.clone())?)?;
                    idx += 1;
                }
            }
            for e in s.entities.read().values() {
                let g = e.read();
                if !g.removed && g.pos.cmpge(lo).all() && g.pos.cmple(hi).all() {
                    out.set(idx, object_ref_to_lua(lua, e.clone())?)?;
                    idx += 1;
                }
            }
            Ok(out)
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("clear_objects", lua.create_function(move |lua, _: Option<Table>| {
            for e in s.entities.write().drain().map(|(_, v)| v) {
                e.write().removed = true;
            }
            if let Ok(core_t) = lua.globals().get::<Table>("core") {
                let ents_tbl = lua.create_table()?;
                let _ = core_t.set("_cubex_luaentities", ents_tbl.clone());
                let _ = core_t.set("luaentities", ents_tbl);
            }
            Ok(())
        })?)?;
    }
    {
        let s = shared.clone();
        core.set("sound_play", lua.create_function(move |_, (spec_val, params, _ephemeral): (Value, Option<Table>, Option<bool>)| {
            if let Some(spec) = parse_sound_spec(spec_val) {
                let pos = params.as_ref().and_then(|p| p.get::<Table>("pos").ok()).map(|t| parse_vec3f(&t));
                let gain = params.as_ref().and_then(|p| p.get::<f32>("gain").ok()).unwrap_or(spec.gain);
                s.pending_sounds.write().push(PendingSoundEvent { spec, pos, gain });
            }
            Ok(1u32)
        })?)?;
    }
    core.set("sound_stop", lua.create_function(|_, _h: u32| Ok(()))?)?;
    core.set("sound_fade", lua.create_function(|_, (_h, _step, _gain): (u32, f32, f32)| Ok(()))?)?;
    core.set("add_particle", lua.create_function(|_, _: Table| Ok(()))?)?;
    core.set("add_particlespawner", lua.create_function(|_, _: Table| Ok(1u32))?)?;
    core.set("delete_particlespawner", lua.create_function(|_, _: MultiValue| Ok(()))?)?;
    core.set("dynamic_add_media", lua.create_function(|_, _: MultiValue| Ok(true))?)?;

    // Auth / Server stubs
    core.set("get_password_hash", lua.create_function(|_, (n, p): (String, String)| Ok(format!("{n}:{p}")))?)?;
    core.set("notify_authentication_modified", lua.create_function(|_, _: Option<String>| Ok(()))?)?;
    core.set("get_ban_list", lua.create_function(|_, ()| Ok(String::new()))?)?;
    core.set("get_ban_description", lua.create_function(|_, _ip: String| Ok(String::new()))?)?;
    core.set("ban_player", lua.create_function(|_, _n: String| Ok(true))?)?;
    core.set("unban_player_or_ip", lua.create_function(|_, _n: String| Ok(true))?)?;
    core.set("kick_player", lua.create_function(|_, _: MultiValue| Ok(true))?)?;
    core.set("remove_player", lua.create_function(|_, _n: String| Ok(0i32))?)?;
    core.set("disconnect_player", lua.create_function(|_, _: MultiValue| Ok(true))?)?;
    core.set("request_shutdown", lua.create_function(|_, _: MultiValue| Ok(()))?)?;
    core.set("get_server_status", lua.create_function(|_, ()| Ok("Cubex-RS Running".to_string()))?)?;
    core.set("rollback_get_node_actions", lua.create_function(|lua, _: MultiValue| Ok(lua.create_table()?))?)?;
    core.set("rollback_revert_actions_by", lua.create_function(|lua, _: MultiValue| Ok((true, lua.create_table()?)))?)?;

    // Async, HTTP hook, & Insecure environment shims
    core.set("set_http_api_lua", lua.create_function(|_, _f: mlua::Function| Ok(()))?)?;
    core.set("do_async_callback", lua.create_function(|_, _: MultiValue| Ok(1u64))?)?;
    core.set("register_async_dofile", lua.create_function(|_, _path: String| Ok(()))?)?;
    core.set("request_insecure_environment", lua.create_function(|_, ()| Ok(Value::Nil))?)?;

    // Filesystem helpers
    core.set("get_dir_list", lua.create_function(|lua, (path, is_dir): (String, Option<bool>)| {
        let out = lua.create_table()?;
        if let Ok(entries) = std::fs::read_dir(&path) {
            let mut names = Vec::new();
            for ent in entries.flatten() {
                let ft = ent.file_type();
                let keep = match (is_dir, ft) {
                    (Some(true), Ok(t)) => t.is_dir(),
                    (Some(false), Ok(t)) => t.is_file(),
                    (None, _) => true,
                    _ => false,
                };
                if keep {
                    names.push(ent.file_name().to_string_lossy().to_string());
                }
            }
            names.sort();
            for (i, n) in names.into_iter().enumerate() {
                out.set(i + 1, n)?;
            }
        }
        Ok(out)
    })?)?;
    core.set("mkdir", lua.create_function(|_, path: String| Ok(std::fs::create_dir_all(&path).is_ok()))?)?;
    core.set("rmdir", lua.create_function(|_, (path, rec): (String, Option<bool>)| {
        let ok = if rec.unwrap_or(false) {
            std::fs::remove_dir_all(&path).is_ok()
        } else {
            std::fs::remove_dir(&path).is_ok()
        };
        Ok(ok)
    })?)?;
    core.set("cpdir", lua.create_function(|_, (_src, _dst): (String, String)| Ok(true))?)?;
    core.set("mvdir", lua.create_function(|_, (src, dst): (String, String)| Ok(std::fs::rename(&src, &dst).is_ok()))?)?;
    core.set("safe_file_write", lua.create_function(|_, (path, content): (String, mlua::String)| {
        if let Some(p) = std::path::Path::new(&path).parent() {
            let _ = std::fs::create_dir_all(p);
        }
        Ok(std::fs::write(&path, content.as_bytes()).is_ok())
    })?)?;

    // Translation / JSON / Compression / Base64 / SHA1 / Colors / PNG
    core.set("gettext", lua.create_function(|_, s: String| Ok(s))?)?;
    core.set("translate", lua.create_function(|_, (_td, s): (String, String)| Ok(s))?)?;
    core.set("get_translated_string", lua.create_function(|_, (_lang, s): (String, String)| Ok(s))?)?;

    core.set("parse_json", lua.create_function(|lua, (raw, nullval): (String, Option<Value>)| {
        let nv = nullval.unwrap_or(Value::Nil);
        match serde_json::from_str::<serde_json::Value>(&raw) {
            Ok(jv) => json_to_lua_value(lua, &jv, &nv),
            Err(_) => Ok(Value::Nil),
        }
    })?)?;
    core.set("write_json", lua.create_function(|_, (val, styled): (Value, Option<bool>)| {
        let jv = lua_value_to_json(val);
        let s = if styled.unwrap_or(false) {
            serde_json::to_string_pretty(&jv).unwrap_or_default()
        } else {
            serde_json::to_string(&jv).unwrap_or_default()
        };
        Ok(s)
    })?)?;
    core.set("encode_base64", lua.create_function(|_, data: mlua::String| {
        Ok(base64::engine::general_purpose::STANDARD.encode(data.as_bytes()))
    })?)?;
    core.set("decode_base64", lua.create_function(|lua, data: String| {
        match base64::engine::general_purpose::STANDARD.decode(data.trim()) {
            Ok(bytes) => Ok(Value::String(lua.create_string(&bytes)?)),
            Err(_) => Ok(Value::Nil),
        }
    })?)?;
    core.set("sha1", lua.create_function(|lua, (data, raw): (mlua::String, Option<bool>)| {
        let mut hasher = Sha1::new();
        hasher.update(data.as_bytes());
        let res = hasher.finalize();
        if raw.unwrap_or(false) {
            Ok(Value::String(lua.create_string(res.as_slice())?))
        } else {
            let hex: String = res.iter().map(|b| format!("{b:02x}")).collect();
            Ok(Value::String(lua.create_string(&hex)?))
        }
    })?)?;
    core.set("compress", lua.create_function(|lua, (data, _method, level): (mlua::String, Option<String>, Option<u32>)| {
        let mut enc = ZlibEncoder::new(Vec::new(), Compression::new(level.unwrap_or(6).min(9)));
        let _ = enc.write_all(&data.as_bytes());
        let out = enc.finish().unwrap_or_default();
        Ok(lua.create_string(&out)?)
    })?)?;
    core.set("decompress", lua.create_function(|lua, (data, _method): (mlua::String, Option<String>)| {
        let raw = data.as_bytes();
        let mut dec = ZlibDecoder::new(&raw[..]);
        let mut out = Vec::new();
        let _ = dec.read_to_end(&mut out);
        Ok(lua.create_string(&out)?)
    })?)?;
    core.set("colorspec_to_bytes", lua.create_function(|lua, val: Value| {
        let rgba = parse_color_value(val).unwrap_or([255, 255, 255, 255]);
        Ok(lua.create_string(&rgba)?)
    })?)?;
    core.set("colorspec_to_colorstring", lua.create_function(|_, val: Value| {
        let rgba = parse_color_value(val).unwrap_or([255, 255, 255, 255]);
        if rgba[3] == 255 {
            Ok(format!("#{:02X}{:02X}{:02X}", rgba[0], rgba[1], rgba[2]))
        } else {
            Ok(format!("#{:02X}{:02X}{:02X}{:02X}", rgba[0], rgba[1], rgba[2], rgba[3]))
        }
    })?)?;
    core.set("encode_png", lua.create_function(|lua, (w, h, data, _comp): (u32, u32, mlua::String, Option<i32>)| {
        let mut out = Vec::new();
        let bytes = data.as_bytes();
        if let Some(img) = image::RgbaImage::from_raw(w, h, bytes.to_vec()) {
            let mut cursor = std::io::Cursor::new(&mut out);
            let _ = img.write_to(&mut cursor, image::ImageFormat::Png);
        }
        Ok(lua.create_string(&out)?)
    })?)?;
    core.set("get_color_escape_sequence", lua.create_function(|_, _c: String| Ok(String::new()))?)?;
    core.set("get_background_escape_sequence", lua.create_function(|_, _c: String| Ok(String::new()))?)?;

    // Fallback registration callback tables (overridden/populated by builtin/game/register.lua)
    for tbl_name in [
        "registered_on_authplayers",
        "registered_on_chat_messages",
        "registered_on_chatcommands",
        "registered_on_cheats",
        "registered_on_crafts",
        "registered_on_dieplayers",
        "registered_on_dignodes",
        "registered_on_generateds",
        "registered_on_item_eats",
        "registered_on_item_pickups",
        "registered_on_joinplayers",
        "registered_on_leaveplayers",
        "registered_on_liquid_transformed",
        "registered_on_mapblocks_changed",
        "registered_on_modchannel_message",
        "registered_on_mods_loaded",
        "registered_on_newplayers",
        "registered_on_placenodes",
        "registered_on_player_inventory_actions",
        "registered_on_player_receive_fields",
        "registered_on_prejoinplayers",
        "registered_on_priv_grant",
        "registered_on_priv_revoke",
        "registered_on_protection_violation",
        "registered_on_punchnodes",
        "registered_on_punchplayers",
        "registered_on_respawnplayers",
        "registered_on_rightclickplayers",
        "registered_on_shutdown",
        "registered_allow_player_inventory_actions",
        "registered_can_bypass_userlimit",
        "registered_craft_predicts",
        "registered_globalsteps",
        "registered_playerevents",
    ] {
        core.set(tbl_name, lua.create_table()?)?;
    }

    let hpchanges = lua.create_table()?;
    hpchanges.set("modifiers", lua.create_table()?)?;
    hpchanges.set("loggers", lua.create_table()?)?;
    core.set("registered_on_player_hpchanges", hpchanges)?;
    core.set(
        "register_on_player_hpchange",
        lua.create_function(|lua, (func, modifier): (mlua::Function, Option<bool>)| {
            if let Ok(core_t) = lua.globals().get::<Table>("core") {
                if let Ok(tbl) = core_t.get::<Table>("registered_on_player_hpchanges") {
                    let key = if modifier.unwrap_or(false) { "modifiers" } else { "loggers" };
                    if let Ok(arr) = tbl.get::<Table>(key) {
                        let len = arr.len().unwrap_or(0);
                        arr.set(len + 1, func)?;
                    }
                }
            }
            Ok(())
        })?,
    )?;
    for (reg_fn, tbl_name) in [
        ("register_on_punchplayer", "registered_on_punchplayers"),
        ("register_on_dieplayer", "registered_on_dieplayers"),
    ] {
        let tn = tbl_name.to_string();
        core.set(
            reg_fn,
            lua.create_function(move |lua, func: mlua::Function| {
                if let Ok(core_t) = lua.globals().get::<Table>("core") {
                    if let Ok(arr) = core_t.get::<Table>(tn.as_str()) {
                        let len = arr.len().unwrap_or(0);
                        arr.set(len + 1, func)?;
                    }
                }
                Ok(())
            })?,
        )?;
    }

    // Do NOT define core.get_http_api (builtin/init.lua line 53 asserts `assert(not core.get_http_api)`!)
    g.set("core", core.clone())?;
    g.set("minetest", core)?;
    Ok(())
}
