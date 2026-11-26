pub mod core_api;
pub mod mod_loader;
pub mod noise;
pub mod userdata;

use anyhow::{anyhow, Result};
use glam::{IVec3, Vec3};
use mlua::{Lua, ObjectLike, Table, Value};
use parking_lot::{Mutex, RwLock};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::config::MinetestConfig;
use crate::paths::EnginePaths;
use crate::registry::{DrawType, EngineRegistries, CONTENT_AIR, CONTENT_IGNORE};
pub use crate::registry::{compute_dig_params, compute_dig_params_with_wear, DigParams};
use crate::world::{Difficulty, WorldStore};

#[derive(Debug, Clone, Default)]
pub struct PunchEntityResult {
    pub hit: bool,
    pub damage: i32,
    pub killed: bool,
    pub drops: Vec<String>,
}

pub use self::core_api::{resolve_craft_recipe, CraftOutput, PendingSoundEvent};
use self::core_api::{register_core_api, FormspecRequest, LuaEngineShared};
use self::mod_loader::{
    finish_mod_loading, load_all_mods, load_single_mod, prepare_mod_environment, ModLoadReport,
};
pub use self::mod_loader::{discover_and_sort_mods, ModSpec};
use self::noise::register_noise_and_rng;
pub use self::userdata::{
    cubex_to_lua_pos, drain_player_inventory_for_death, format_death_cause, object_ref_to_lua,
    register_userdata_ctors, ItemStackData, ItemStackUD, ObjectState,
};

/// Time-sliced main-thread LuaJIT mod loader so splash frames animate smoothly at 60+ Hz.
pub struct LuaIncrementalBoot {
    lua: Lua,
    shared: Arc<LuaEngineShared>,
    mods: Vec<ModSpec>,
    next_idx: usize,
    mods_loaded: usize,
    errors: Vec<(String, String)>,
}

impl LuaIncrementalBoot {
    pub fn begin(
        paths: EnginePaths,
        registries: Arc<RwLock<EngineRegistries>>,
        world: Arc<WorldStore>,
        config: Arc<RwLock<MinetestConfig>>,
        mods: Vec<ModSpec>,
    ) -> Result<Self> {
        let lua = unsafe { Lua::unsafe_new() };
        let shared = LuaEngineShared::new(paths, registries.clone(), world.clone(), config);
        register_noise_and_rng(&lua).map_err(|e| anyhow!("{e}"))?;
        register_userdata_ctors(&lua, world, registries).map_err(|e| anyhow!("{e}"))?;
        register_core_api(&lua, shared.clone()).map_err(|e| anyhow!("{e}"))?;
        prepare_mod_environment(&lua, &shared, &mods).map_err(|e| anyhow!("{e}"))?;
        Ok(Self {
            lua,
            shared,
            mods,
            next_idx: 0,
            mods_loaded: 0,
            errors: Vec::new(),
        })
    }

    /// Execute mods up to `budget` (always running at least one if any remain).
    /// Returns `(processed_count, total_mods, last_mod_name)`.
    pub fn step_budget(&mut self, budget: std::time::Duration) -> (usize, usize, Option<String>) {
        let t0 = std::time::Instant::now();
        let total = self.mods.len();
        let mut last_name = None;
        while self.next_idx < total {
            let m = &self.mods[self.next_idx];
            last_name = Some(m.name.clone());
            match load_single_mod(&self.lua, &self.shared, m) {
                Ok(()) => self.mods_loaded += 1,
                Err(err) => self.errors.push((m.name.clone(), err)),
            }
            self.next_idx += 1;
            if t0.elapsed() >= budget {
                break;
            }
        }
        (self.next_idx, total, last_name)
    }

    pub fn is_done(&self) -> bool {
        self.next_idx >= self.mods.len()
    }

    pub fn finish(self) -> LuaRuntime {
        let load_report = finish_mod_loading(
            &self.lua,
            &self.shared,
            self.mods.len(),
            self.mods_loaded,
            self.errors,
        );
        let abm_by_node_id = LuaRuntime::build_abm_index(&self.lua, &self.shared.registries);
        LuaRuntime::install_cubex_lua_hooks(&self.lua);
        LuaRuntime {
            lua: self.lua,
            shared: self.shared,
            load_report,
            abm_by_node_id,
            abm_chunk_cursor: Mutex::new(0),
            mob_spawn_timer: Mutex::new(0.0),
            initial_mobs_spawned: Mutex::new(false),
            logged_step_errors: Mutex::new(HashSet::new()),
            entity_step_error_count: std::sync::atomic::AtomicU64::new(0),
            entity_step_errors: Mutex::new(Vec::new()),
        }
    }
}

pub struct LuaRuntime {
    pub lua: Lua,
    pub shared: Arc<LuaEngineShared>,
    pub load_report: ModLoadReport,
    abm_by_node_id: HashMap<u16, Vec<(usize, u32)>>,
    abm_chunk_cursor: Mutex<usize>,
    mob_spawn_timer: Mutex<f64>,
    initial_mobs_spawned: Mutex<bool>,
    logged_step_errors: Mutex<HashSet<String>>,
    pub entity_step_error_count: std::sync::atomic::AtomicU64,
    pub entity_step_errors: Mutex<Vec<String>>,
}

impl LuaRuntime {
    fn install_cubex_lua_hooks(lua: &Lua) {
        let _ = lua
            .load(
                r#"
                if type(mcl_mobs) == "table" then
                    function mcl_mobs.is_position_completely_generated(pos)
                        if type(pos) ~= "table" then return true end
                        local _, _, _, ok = core.get_node_raw(
                            math.floor((pos.x or 0) + 0.5),
                            math.floor((pos.y or 0) + 0.5),
                            math.floor((pos.z or 0) + 0.5)
                        )
                        return ok == true
                    end
                end
                "#,
            )
            .exec();
    }

    pub fn boot(
        paths: EnginePaths,
        registries: Arc<RwLock<EngineRegistries>>,
        world: Arc<WorldStore>,
        config: Arc<RwLock<MinetestConfig>>,
    ) -> Result<Self> {
        let lua = unsafe { Lua::unsafe_new() };
        let shared = LuaEngineShared::new(paths, registries.clone(), world.clone(), config);

        register_noise_and_rng(&lua).map_err(|e| anyhow!("{e}"))?;
        register_userdata_ctors(&lua, world, registries).map_err(|e| anyhow!("{e}"))?;
        register_core_api(&lua, shared.clone()).map_err(|e| anyhow!("{e}"))?;

        let load_report = load_all_mods(&lua, &shared).map_err(|e| anyhow!("{e}"))?;
        let abm_by_node_id = Self::build_abm_index(&lua, &shared.registries);
        Self::install_cubex_lua_hooks(&lua);
        Ok(Self {
            lua,
            shared,
            load_report,
            abm_by_node_id,
            abm_chunk_cursor: Mutex::new(0),
            mob_spawn_timer: Mutex::new(0.0),
            initial_mobs_spawned: Mutex::new(false),
            logged_step_errors: Mutex::new(HashSet::new()),
            entity_step_error_count: std::sync::atomic::AtomicU64::new(0),
            entity_step_errors: Mutex::new(Vec::new()),
        })
    }

    fn build_abm_index(
        lua: &Lua,
        registries: &Arc<RwLock<EngineRegistries>>,
    ) -> HashMap<u16, Vec<(usize, u32)>> {
        let mut by_node: HashMap<u16, Vec<(usize, u32)>> = HashMap::new();
        let Ok(core) = lua.globals().get::<Table>("core") else {
            return by_node;
        };
        let Ok(abms) = core.get::<Table>("registered_abms") else {
            return by_node;
        };
        let abm_len = abms.len().unwrap_or(0);
        if abm_len == 0 {
            return by_node;
        }
        let reg = registries.read();
        for i in 1..=abm_len {
            let Ok(abm) = abms.get::<Table>(i) else {
                continue;
            };
            if abm.get::<mlua::Function>("action").is_err() {
                continue;
            }
            let chance: u32 = abm.get::<u32>("chance").unwrap_or(1).max(1);
            let interval: f64 = abm.get::<f64>("interval").unwrap_or(1.0).max(1.0);
            let p_eff = (chance as f64 * interval).max(1.0) as u32;
            let mut target_ids: HashSet<u16> = HashSet::new();
            if let Ok(nn_val) = abm.get::<Value>("nodenames") {
                let mut names = Vec::new();
                match nn_val {
                    Value::String(st) => {
                        if let Ok(s) = st.to_str() {
                            names.push(s.to_string());
                        }
                    }
                    Value::Table(t) => {
                        let l = t.len().unwrap_or(0);
                        for k in 1..=l {
                            if let Ok(s) = t.get::<String>(k) {
                                names.push(s);
                            }
                        }
                    }
                    _ => {}
                }
                for n in names {
                    if let Some(grp) = n.strip_prefix("group:") {
                        for def in reg.nodes_by_id.iter().flatten() {
                            if def.groups.get(grp).copied().unwrap_or(0) > 0 {
                                target_ids.insert(def.id);
                            }
                        }
                    } else {
                        let id = reg.get_content_id(&n);
                        if id != CONTENT_IGNORE {
                            target_ids.insert(id);
                        }
                    }
                }
            }
            for nid in target_ids {
                by_node.entry(nid).or_default().push((i as usize, p_eff));
            }
        }
        by_node
    }

    pub fn join_player(&self, state: Arc<RwLock<ObjectState>>) -> Result<()> {
        let name = state.read().name.clone();
        self.shared.players.write().insert(name, state.clone());

        let core: Table = self.lua.globals().get("core").map_err(|e| anyhow!("{e}"))?;
        let obj = object_ref_to_lua(&self.lua, state.clone()).map_err(|e| anyhow!("{e}"))?;
        for tbl_name in ["registered_on_newplayers", "registered_on_joinplayers"] {
            if let Ok(cbs) = core.get::<Table>(tbl_name) {
                let len = cbs.len().unwrap_or(0);
                for i in 1..=len {
                    if let Ok(f) = cbs.get::<mlua::Function>(i) {
                        if let Err(e) = f.call::<()>(obj.clone()) {
                            tracing::debug!("{tbl_name}[{i}] error: {e}");
                        }
                    }
                }
            }
        }
        let hand_empty = state
            .read()
            .meta
            .read()
            .inventories
            .get("hand")
            .and_then(|v| v.first())
            .map(ItemStackData::is_empty)
            .unwrap_or(true);
        if hand_empty {
            if let Ok(mh) = self.lua.globals().get::<Table>("mcl_meshhand") {
                if let Ok(upd) = mh.get::<mlua::Function>("update_player") {
                    if let Err(e) = upd.call::<()>(obj) {
                        tracing::debug!("mcl_meshhand.update_player error: {e}");
                    }
                }
            }
        }
        Ok(())
    }

    fn make_lua_pos(&self, p: IVec3) -> mlua::Result<Table> {
        self::core_api::make_vec3(&self.lua, p.x as f64, p.y as f64, p.z as f64)
    }

    fn make_lua_node(&self, name: &str, p1: u8, p2: u8) -> mlua::Result<Table> {
        let t = self.lua.create_table()?;
        t.set("name", name)?;
        t.set("param1", p1)?;
        t.set("param2", p2)?;
        Ok(t)
    }

    /// Resolve node drops via Lua `minetest.get_node_drops(node_name, tool_name)` when available,
    /// falling back to `NodeDrop::resolve_drops` when offline/headless or absent from Lua.
    pub fn get_node_drops(
        &self,
        node_name: &str,
        tool_name: Option<&str>,
        player_state: Option<&Arc<RwLock<ObjectState>>>,
    ) -> Vec<String> {
        let effective_tool: Option<String> = match tool_name {
            Some(t) => Some(t.to_string()),
            None => player_state.and_then(|ps| {
                let (idx, meta) = {
                    let g = ps.read();
                    (g.wield_index.saturating_sub(1), g.meta.clone())
                };
                let mg = meta.read();
                mg.inventories
                    .get("main")
                    .and_then(|v| v.get(idx))
                    .filter(|s| !s.is_empty())
                    .map(|s| s.name.clone())
            }),
        };

        if let Ok(minetest) = self
            .lua
            .globals()
            .get::<Table>("minetest")
            .or_else(|_| self.lua.globals().get::<Table>("core"))
        {
            let in_lua = minetest
                .get::<Table>("registered_nodes")
                .ok()
                .and_then(|rn| rn.get::<Value>(node_name).ok())
                .map(|v| matches!(v, Value::Table(_)))
                .unwrap_or(false);

            if in_lua {
                if let Ok(f) = minetest.get::<mlua::Function>("get_node_drops") {
                    let arg_tool = match effective_tool.as_deref() {
                        Some(t) if !t.is_empty() => self
                            .lua
                            .create_string(t)
                            .map(Value::String)
                            .unwrap_or(Value::Nil),
                        _ => Value::Nil,
                    };
                    if let Ok(val) = f.call::<Value>((node_name, arg_tool)) {
                        return parse_drops_lua_value(val);
                    }
                }
            }
        }

        let reg = self.shared.registries.read();
        let id = reg.get_content_id(node_name);
        let Some(def) = reg.get_node_def(id) else {
            return Vec::new();
        };
        let tool_groups = effective_tool
            .as_deref()
            .filter(|s| !s.is_empty())
            .and_then(|tn| reg.get_item_def(tn))
            .map(|d| &d.groups);
        def.drop
            .resolve_drops(&def.name, effective_tool.as_deref(), tool_groups)
            .into_iter()
            .filter(|s| !s.trim().is_empty())
            .collect()
    }

    /// Returns true when Mineclonia's `mcl_item_entity` drop pipeline is active in Lua.
    pub fn has_mcl_item_entity(&self) -> bool {
        self.lua
            .globals()
            .get::<Value>("mcl_item_entity")
            .map(|v| matches!(v, Value::Table(_)))
            .unwrap_or(false)
    }

    /// Collect and remove `__builtin:item` drop entities spawned at or after `min_entity_id`.
    pub fn take_spawned_item_drops(&self, min_entity_id: u64) -> Vec<String> {
        let mut eids: Vec<u64> = {
            let ents = self.shared.entities.read();
            ents.iter()
                .filter_map(|(&eid, arc)| {
                    if eid >= min_entity_id {
                        let g = arc.read();
                        if !g.removed && g.entity_name == "__builtin:item" {
                            return Some(eid);
                        }
                    }
                    None
                })
                .collect()
        };
        if eids.is_empty() {
            return Vec::new();
        }
        eids.sort_unstable();

        let ents_tbl = self
            .lua
            .globals()
            .get::<Table>("core")
            .ok()
            .and_then(|c| c.get::<Table>("_cubex_luaentities").ok());
        let refs_tbl = self
            .lua
            .named_registry_value::<Value>("_cubex_object_refs")
            .ok()
            .and_then(|v| match v {
                Value::Table(t) => Some(t),
                _ => None,
            });

        let mut drops = Vec::new();
        let mut ents_w = self.shared.entities.write();
        for eid in eids {
            let mut spec = ents_tbl
                .as_ref()
                .and_then(|t| t.get::<Table>(eid).ok())
                .and_then(|et| et.get::<String>("itemstring").ok())
                .unwrap_or_default();
            if let Some(arc) = ents_w.remove(&eid) {
                let mut g = arc.write();
                if spec.trim().is_empty() && !g.wield_item.trim().is_empty() {
                    spec = g.wield_item.clone();
                }
                g.removed = true;
            }
            if let Some(ref t) = ents_tbl {
                let _ = t.set(eid, Value::Nil);
            }
            if let Some(ref rt) = refs_tbl {
                let _ = rt.set(format!("e:{eid}"), Value::Nil);
            }
            let trimmed = spec.trim();
            if !trimmed.is_empty() {
                drops.push(trimmed.to_string());
            }
        }
        drops
    }

    /// Give Lua-spawned `__builtin:item` drop entities playful outward/upward pop velocity
    /// and gravity physics instead of immediately consuming them (used when Block Shatter & Physical Drops is enabled).
    pub fn boost_spawned_item_drops(&self, min_entity_id: u64, block_pos: IVec3) -> Vec<String> {
        use rand::Rng as _;
        let mut eids: Vec<u64> = {
            let ents = self.shared.entities.read();
            ents.iter()
                .filter_map(|(&eid, arc)| {
                    if eid >= min_entity_id {
                        let g = arc.read();
                        if !g.removed && g.entity_name == "__builtin:item" {
                            return Some(eid);
                        }
                    }
                    None
                })
                .collect()
        };
        if eids.is_empty() {
            return Vec::new();
        }
        eids.sort_unstable();

        let ents_tbl = self
            .lua
            .globals()
            .get::<Table>("core")
            .ok()
            .and_then(|c| c.get::<Table>("_cubex_luaentities").ok());
        let mut rng = rand::thread_rng();
        let mut drops = Vec::new();
        let ents_r = self.shared.entities.read();
        for eid in eids {
            let spec_from_lua = ents_tbl
                .as_ref()
                .and_then(|t| t.get::<Table>(eid).ok())
                .and_then(|et| et.get::<String>("itemstring").ok())
                .unwrap_or_default();
            if let Some(arc) = ents_r.get(&eid) {
                let mut g = arc.write();
                if g.wield_item.trim().is_empty() && !spec_from_lua.trim().is_empty() {
                    g.wield_item = spec_from_lua.trim().to_string();
                }
                let item_name = if !spec_from_lua.trim().is_empty() {
                    spec_from_lua.trim().to_string()
                } else {
                    g.wield_item.trim().to_string()
                };
                if !item_name.is_empty() {
                    drops.push(item_name);
                }
                g.pos = block_pos.as_vec3() + Vec3::new(0.5, 0.45, 0.5);
                g.vel = Vec3::new(
                    rng.gen_range(-1.35..1.35),
                    rng.gen_range(2.2..3.5),
                    rng.gen_range(-1.35..1.35),
                );
                g.accel = Vec3::new(0.0, -9.81, 0.0);
                g.accel_set_by_lua = true;
                g.physical = true;
                g.collisionbox = [-0.18, -0.18, -0.18, 0.18, 0.18, 0.18];
                g.selectionbox = [-0.18, -0.18, -0.18, 0.18, 0.18, 0.18];
                g.visual = "wielditem".into();
                g.automatic_rotate = 1.8;
                g.age_sec = 0.0;
            }
        }
        drops
    }

    /// Spawn a physical dropped `__builtin:item` entity (preferring Lua `core.add_item`, with Rust fallback).
    pub fn spawn_physical_item_drop(&self, block_pos: IVec3, item_spec: &str) -> bool {
        use rand::Rng as _;
        let spec = item_spec.trim();
        if spec.is_empty() {
            return false;
        }
        let eid_before = *self.shared.next_entity_id.read();
        if let Ok(core) = self.lua.globals().get::<Table>("core") {
            if let Ok(add_item) = core.get::<mlua::Function>("add_item") {
                if let Ok(pt) = self.make_lua_pos(block_pos) {
                    let _ = add_item.call::<Value>((pt, spec));
                    let new_eids: Vec<u64> = self
                        .shared
                        .entities
                        .read()
                        .iter()
                        .filter_map(|(&eid, arc)| {
                            let g = arc.read();
                            (eid >= eid_before && !g.removed && g.entity_name == "__builtin:item")
                                .then_some(eid)
                        })
                        .collect();
                    if !new_eids.is_empty() {
                        // Ensure wield_item & Lua itemstring are populated even when Lua node wasn't registered
                        if let Ok(em) = core.get::<Table>("_cubex_luaentities") {
                            let ents_r = self.shared.entities.read();
                            for &eid in &new_eids {
                                if let Some(arc) = ents_r.get(&eid) {
                                    let mut g = arc.write();
                                    if g.wield_item.trim().is_empty() {
                                        g.wield_item = spec.to_string();
                                    }
                                }
                                if let Ok(et) = em.get::<Table>(eid) {
                                    let cur: String = et.get("itemstring").unwrap_or_default();
                                    if cur.trim().is_empty() {
                                        let _ = et.set("itemstring", spec);
                                    }
                                }
                            }
                        }
                        let _ = self.boost_spawned_item_drops(eid_before, block_pos);
                        return true;
                    }
                }
            }
        }

        // Pure-Rust physical drop entity fallback
        let eid = {
            let mut next = self.shared.next_entity_id.write();
            let id = *next;
            *next += 1;
            id
        };
        let mut st = ObjectState::new_entity(
            eid,
            "__builtin:item",
            block_pos.as_vec3() + Vec3::new(0.5, 0.45, 0.5),
        );
        let mut rng = rand::thread_rng();
        st.vel = Vec3::new(
            rng.gen_range(-1.35..1.35),
            rng.gen_range(2.2..3.5),
            rng.gen_range(-1.35..1.35),
        );
        st.accel = Vec3::new(0.0, -9.81, 0.0);
        st.accel_set_by_lua = true;
        st.physical = true;
        st.collisionbox = [-0.18, -0.18, -0.18, 0.18, 0.18, 0.18];
        st.selectionbox = [-0.18, -0.18, -0.18, 0.18, 0.18, 0.18];
        st.visual = "wielditem".into();
        st.wield_item = spec.to_string();
        st.automatic_rotate = 1.8;
        self.shared.entities.write().insert(eid, Arc::new(RwLock::new(st)));
        true
    }

    /// Execute Mineclonia's `on_dig` / `can_dig` / `core.node_dig` / `after_dig_node` / `on_destruct` pipeline.
    pub fn dig_node(&self, pos: IVec3, player_state: &Arc<RwLock<ObjectState>>) -> bool {
        let (old_id, p1, p2) = self.shared.world.get_node(pos);
        if old_id == CONTENT_AIR || old_id == CONTENT_IGNORE {
            return false;
        }
        let node_name = self
            .shared
            .registries
            .read()
            .get_name_from_content_id(old_id)
            .to_string();

        let Ok(core) = self.lua.globals().get::<Table>("core") else {
            return false;
        };
        let Ok(pt) = self.make_lua_pos(pos) else {
            return false;
        };
        let Ok(nt) = self.make_lua_node(&node_name, p1, p2) else {
            return false;
        };
        let Ok(digger) = object_ref_to_lua(&self.lua, player_state.clone()) else {
            return false;
        };

        let dig_fn = core
            .get::<Table>("registered_nodes")
            .ok()
            .and_then(|rn| rn.get::<Table>(node_name.as_str()).ok())
            .and_then(|def| def.get::<mlua::Function>("on_dig").ok())
            .or_else(|| core.get::<mlua::Function>("node_dig").ok());

        let mut dug = false;
        if let Some(f) = dig_fn {
            match f.call::<Value>((pt, nt, digger)) {
                Ok(Value::Boolean(b)) => {
                    dug = b || self.shared.world.get_node(pos).0 != old_id;
                }
                Ok(_) => {
                    dug = self.shared.world.get_node(pos).0 != old_id;
                }
                Err(e) => {
                    tracing::debug!("dig_node error on {node_name}: {e}");
                }
            }
        }
        if dug {
            self.shared.world.mark_priority_dirty_with_neighbors(pos);
        }
        dug
    }

    /// Invoke `registered_nodes[name].on_construct(pos)` for a newly placed node at `pos`.
    pub fn invoke_on_construct(&self, pos: IVec3) {
        let (id, _, _) = self.shared.world.get_node(pos);
        if id == CONTENT_AIR || id == CONTENT_IGNORE {
            return;
        }
        let name = self
            .shared
            .registries
            .read()
            .get_name_from_content_id(id)
            .to_string();
        let Ok(core) = self.lua.globals().get::<Table>("core") else {
            return;
        };
        let Ok(rn) = core.get::<Table>("registered_nodes") else {
            return;
        };
        let Ok(def) = rn.get::<Table>(name.as_str()) else {
            return;
        };
        let Ok(cb) = def.get::<mlua::Function>("on_construct") else {
            return;
        };
        if let Ok(pt) = self.make_lua_pos(pos) {
            if let Err(e) = cb.call::<Value>(pt) {
                tracing::debug!("on_construct error on {name} at {pos}: {e}");
            }
        }
    }

    fn get_node_callback_and_pos(
        &self,
        pos: IVec3,
        cb_name: &str,
    ) -> Option<(String, mlua::Function, Table)> {
        let (id, _, _) = self.shared.world.get_node(pos);
        if id == CONTENT_AIR || id == CONTENT_IGNORE {
            return None;
        }
        let name = self
            .shared
            .registries
            .read()
            .get_name_from_content_id(id)
            .to_string();
        let core = self.lua.globals().get::<Table>("core").ok()?;
        let rn = core.get::<Table>("registered_nodes").ok()?;
        let def = rn.get::<Table>(name.as_str()).ok()?;
        let cb = def.get::<mlua::Function>(cb_name).ok()?;
        let pt = self.make_lua_pos(pos).ok()?;
        Some((name, cb, pt))
    }

    pub fn allow_metadata_inventory_put(
        &self,
        pos: IVec3,
        list: &str,
        index_1based: usize,
        stack: &ItemStackData,
        player: &Arc<RwLock<ObjectState>>,
    ) -> u16 {
        let Some((name, cb, pt)) = self.get_node_callback_and_pos(pos, "allow_metadata_inventory_put")
        else {
            return stack.count;
        };
        let Ok(placer) = object_ref_to_lua(&self.lua, player.clone()) else {
            return stack.count;
        };
        let st_ud = ItemStackUD(Arc::new(RwLock::new(stack.clone())));
        match cb.call::<i64>((pt, list, index_1based, st_ud, placer)) {
            Ok(n) => n.max(0).min(u16::MAX as i64) as u16,
            Err(e) => {
                tracing::debug!("allow_metadata_inventory_put error on {name}: {e}");
                0
            }
        }
    }

    pub fn allow_metadata_inventory_take(
        &self,
        pos: IVec3,
        list: &str,
        index_1based: usize,
        stack: &ItemStackData,
        player: &Arc<RwLock<ObjectState>>,
    ) -> u16 {
        let Some((name, cb, pt)) = self.get_node_callback_and_pos(pos, "allow_metadata_inventory_take")
        else {
            return stack.count;
        };
        let Ok(placer) = object_ref_to_lua(&self.lua, player.clone()) else {
            return stack.count;
        };
        let st_ud = ItemStackUD(Arc::new(RwLock::new(stack.clone())));
        match cb.call::<i64>((pt, list, index_1based, st_ud, placer)) {
            Ok(n) => n.max(0).min(u16::MAX as i64) as u16,
            Err(e) => {
                tracing::debug!("allow_metadata_inventory_take error on {name}: {e}");
                0
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn allow_metadata_inventory_move(
        &self,
        pos: IVec3,
        from_list: &str,
        from_index_1based: usize,
        to_list: &str,
        to_index_1based: usize,
        count: u16,
        player: &Arc<RwLock<ObjectState>>,
    ) -> u16 {
        let Some((name, cb, pt)) = self.get_node_callback_and_pos(pos, "allow_metadata_inventory_move")
        else {
            return count;
        };
        let Ok(placer) = object_ref_to_lua(&self.lua, player.clone()) else {
            return count;
        };
        match cb.call::<i64>((
            pt,
            from_list,
            from_index_1based,
            to_list,
            to_index_1based,
            count,
            placer,
        )) {
            Ok(n) => n.max(0).min(u16::MAX as i64) as u16,
            Err(e) => {
                tracing::debug!("allow_metadata_inventory_move error on {name}: {e}");
                0
            }
        }
    }

    pub fn on_metadata_inventory_put(
        &self,
        pos: IVec3,
        list: &str,
        index_1based: usize,
        stack: &ItemStackData,
        player: &Arc<RwLock<ObjectState>>,
    ) {
        let Some((name, cb, pt)) = self.get_node_callback_and_pos(pos, "on_metadata_inventory_put")
        else {
            return;
        };
        let Ok(placer) = object_ref_to_lua(&self.lua, player.clone()) else {
            return;
        };
        let st_ud = ItemStackUD(Arc::new(RwLock::new(stack.clone())));
        if let Err(e) = cb.call::<Value>((pt, list, index_1based, st_ud, placer)) {
            tracing::debug!("on_metadata_inventory_put error on {name}: {e}");
        }
    }

    pub fn on_metadata_inventory_take(
        &self,
        pos: IVec3,
        list: &str,
        index_1based: usize,
        stack: &ItemStackData,
        player: &Arc<RwLock<ObjectState>>,
    ) {
        let Some((name, cb, pt)) = self.get_node_callback_and_pos(pos, "on_metadata_inventory_take")
        else {
            return;
        };
        let Ok(placer) = object_ref_to_lua(&self.lua, player.clone()) else {
            return;
        };
        let st_ud = ItemStackUD(Arc::new(RwLock::new(stack.clone())));
        if let Err(e) = cb.call::<Value>((pt, list, index_1based, st_ud, placer)) {
            tracing::debug!("on_metadata_inventory_take error on {name}: {e}");
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn on_metadata_inventory_move(
        &self,
        pos: IVec3,
        from_list: &str,
        from_index_1based: usize,
        to_list: &str,
        to_index_1based: usize,
        count: u16,
        player: &Arc<RwLock<ObjectState>>,
    ) {
        let Some((name, cb, pt)) = self.get_node_callback_and_pos(pos, "on_metadata_inventory_move")
        else {
            return;
        };
        let Ok(placer) = object_ref_to_lua(&self.lua, player.clone()) else {
            return;
        };
        if let Err(e) = cb.call::<Value>((
            pt,
            from_list,
            from_index_1based,
            to_list,
            to_index_1based,
            count,
            placer,
        )) {
            tracing::debug!("on_metadata_inventory_move error on {name}: {e}");
        }
    }

    /// Execute Mineclonia's `on_rightclick` (chests/furnaces/doors/crafting tables) or `meta["formspec"]` or `on_place` / `core.item_place_node`.
    pub fn rightclick_or_place_node(
        &self,
        under: IVec3,
        above: IVec3,
        player_state: &Arc<RwLock<ObjectState>>,
    ) -> bool {
        let sneak = player_state.read().sneak;
        self.rightclick_or_place_node_with_sneak(under, above, player_state, sneak)
    }

    pub fn rightclick_or_place_node_with_sneak(
        &self,
        under: IVec3,
        above: IVec3,
        player_state: &Arc<RwLock<ObjectState>>,
        sneak: bool,
    ) -> bool {
        let Ok(core) = self.lua.globals().get::<Table>("core") else {
            return false;
        };
        let (under_id, u_p1, u_p2) = self.shared.world.get_node(under);
        let old_above = self.shared.world.get_node(above);
        let under_name = self
            .shared
            .registries
            .read()
            .get_name_from_content_id(under_id)
            .to_string();

        let (wield_idx, wield_item, player_name) = {
            let ps = player_state.read();
            let idx = ps.wield_index.saturating_sub(1);
            let item = ps
                .meta
                .read()
                .inventories
                .get("main")
                .and_then(|v| v.get(idx))
                .cloned()
                .unwrap_or_default();
            (idx, item, ps.name.clone())
        };

        let Ok(under_t) = self.make_lua_pos(under) else {
            return false;
        };
        let Ok(above_t) = self.make_lua_pos(above) else {
            return false;
        };
        let Ok(pointed_thing) = self.lua.create_table() else {
            return false;
        };
        let _ = pointed_thing.set("type", "node");
        let _ = pointed_thing.set("under", under_t.clone());
        let _ = pointed_thing.set("above", above_t);

        let Ok(under_node_t) = self.make_lua_node(&under_name, u_p1, u_p2) else {
            return false;
        };
        let Ok(placer) = object_ref_to_lua(&self.lua, player_state.clone()) else {
            return false;
        };
        let stack_arc = Arc::new(RwLock::new(wield_item.clone()));
        let stack_ud = ItemStackUD(stack_arc.clone());

        let is_creative = core
            .get::<mlua::Function>("is_creative_enabled")
            .ok()
            .and_then(|f| f.call::<bool>(player_name.as_str()).ok())
            .unwrap_or(false);

        let mut handled = false;
        let mut updated_stack: Option<ItemStackData> = None;
        let form_before = self.shared.pending_formspecs.read().len();

        // 1. Prioritize `on_rightclick` or node-metadata `"formspec"` on the pointed node unless sneaking
        if !sneak {
            if let Some(rc_fn) = core
                .get::<Table>("registered_nodes")
                .ok()
                .and_then(|rn| rn.get::<Table>(under_name.as_str()).ok())
                .and_then(|def| def.get::<mlua::Function>("on_rightclick").ok())
            {
                match rc_fn.call::<Value>((
                    under_t,
                    under_node_t,
                    placer.clone(),
                    stack_ud.clone(),
                    pointed_thing.clone(),
                )) {
                    Ok(ret) => {
                        if let Ok(ret_stack) = ItemStackUD::from_lua_value(ret.clone()) {
                            if !ret_stack.0.read().is_empty() || stack_arc.read().is_empty() {
                                updated_stack = Some(ret_stack.0.read().clone());
                            }
                            handled = true;
                        } else if self.shared.pending_formspecs.read().len() > form_before
                            || self.shared.world.get_node(under) != (under_id, u_p1, u_p2)
                            || matches!(ret, Value::Boolean(_))
                        {
                            handled = true;
                        }
                    }
                    Err(e) => {
                        tracing::debug!("on_rightclick error on {under_name}: {e}");
                    }
                }
            }

            if !handled && self.shared.pending_formspecs.read().len() == form_before {
                let meta_fs = self
                    .shared
                    .world
                    .node_meta
                    .read()
                    .get(&under)
                    .and_then(|m| m.read().fields.get("formspec").cloned())
                    .unwrap_or_default();
                if !meta_fs.trim().is_empty() {
                    self.shared.pending_formspecs.write().push(FormspecRequest {
                        player_name: player_name.clone(),
                        formname: format!("nodemeta:{},{},{}", under.x, under.y, under.z),
                        formspec: meta_fs,
                    });
                    handled = true;
                }
            }
        }

        // 2. Fallback to held item's `on_place` / `core.item_place`
        if !handled && !wield_item.is_empty() {
            let place_fn = core
                .get::<Table>("registered_items")
                .ok()
                .and_then(|ri| ri.get::<Table>(wield_item.name.as_str()).ok())
                .and_then(|def| def.get::<mlua::Function>("on_place").ok())
                .or_else(|| core.get::<mlua::Function>("item_place").ok());
            if let Some(f) = place_fn {
                match f.call::<Value>((stack_ud.clone(), placer, pointed_thing)) {
                    Ok(ret) => {
                        if let Value::UserData(ud) = ret {
                            if let Ok(st) = ud.borrow::<ItemStackUD>() {
                                updated_stack = Some(st.0.read().clone());
                            }
                        } else if let Value::String(st) = ret {
                            if let Ok(s) = st.to_str() {
                                updated_stack = Some(ItemStackData::from_spec(&s));
                            }
                        }
                        let new_above = self.shared.world.get_node(above);
                        let new_under = self.shared.world.get_node(under);
                        if new_above != old_above || new_under != (under_id, u_p1, u_p2) {
                            handled = true;
                        }
                    }
                    Err(e) => {
                        tracing::debug!("on_place error on {}: {e}", wield_item.name);
                    }
                }
            }
        }

        if !is_creative {
            if let Some(new_st) = updated_stack {
                let ps = player_state.read();
                let mut mg = ps.meta.write();
                if let Some(inv) = mg.inventories.get_mut("main") {
                    if wield_idx < inv.len() {
                        inv[wield_idx] = new_st;
                    }
                }
            }
        }

        if handled {
            self.shared.world.mark_priority_dirty_with_neighbors(under);
            self.shared.world.mark_priority_dirty_with_neighbors(above);
        }
        handled
    }

    pub fn invoke_item_on_use(&self, player_state: &Arc<RwLock<ObjectState>>) -> bool {
        let Ok(core) = self.lua.globals().get::<Table>("core") else {
            return false;
        };
        let (wield_idx, wield_item) = {
            let ps = player_state.read();
            let idx = ps.wield_index.saturating_sub(1);
            let item = ps
                .meta
                .read()
                .inventories
                .get("main")
                .and_then(|v| v.get(idx))
                .cloned()
                .unwrap_or_default();
            (idx, item)
        };
        if wield_item.is_empty() {
            return false;
        }
        let Ok(ri) = core.get::<Table>("registered_items") else {
            return false;
        };
        let Ok(def) = ri.get::<Table>(wield_item.name.as_str()) else {
            return false;
        };
        let Ok(on_use) = def.get::<mlua::Function>("on_use") else {
            return false;
        };
        let Ok(user) = object_ref_to_lua(&self.lua, player_state.clone()) else {
            return false;
        };
        let Ok(pt) = self.lua.create_table() else {
            return false;
        };
        let _ = pt.set("type", "nothing");

        let stack_arc = Arc::new(RwLock::new(wield_item.clone()));
        let stack_ud = ItemStackUD(stack_arc.clone());
        let before_count = wield_item.count;
        let Ok(ret) = on_use.call::<Value>((stack_ud, user, pt)) else {
            return false;
        };
        if let Ok(ret_stack) = ItemStackUD::from_lua_value(ret) {
            let updated = ret_stack.0.read().clone();
            let changed = updated.name != wield_item.name || updated.count != before_count;
            let ps = player_state.read();
            let mut mg = ps.meta.write();
            if let Some(inv) = mg.inventories.get_mut("main") {
                if wield_idx < inv.len() {
                    inv[wield_idx] = updated;
                }
            }
            return changed;
        }
        false
    }

    pub fn step(&self, dtime: f64) {
        *self.shared.gametime.write() += dtime;
        let time_speed: f64 = self
            .shared
            .config
            .read()
            .get("time_speed")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(72.0);
        if self.shared.world.gamerules.read().do_daylight_cycle && time_speed > 0.0 {
            let delta_tod = (dtime * time_speed) / 86400.0;
            let mut tod = self.shared.time_of_day.write();
            let next_tod = (*tod as f64) + delta_tod;
            let whole_days = next_tod.floor() as u32;
            if whole_days > 0 {
                *self.shared.day_count.write() += whole_days;
            }
            *tod = (next_tod - whole_days as f64) as f32;
        }
        let Ok(core) = self.lua.globals().get::<Table>("core") else {
            return;
        };

        // 1. Run globalsteps (also ticks builtin/common/after.lua `core.after` jobs)
        if let Ok(cbs) = core.get::<Table>("registered_globalsteps") {
            let len = cbs.len().unwrap_or(0);
            for i in 1..=len {
                if let Ok(f) = cbs.get::<mlua::Function>(i) {
                    if let Err(e) = f.call::<()>(dtime) {
                        let key = format!("gs[{i}]:{}", e.to_string().lines().next().unwrap_or(""));
                        if self.logged_step_errors.lock().insert(key) {
                            let msg = format!("globalstep[{i}] error: {e}");
                            tracing::warn!("{msg}");
                            self.entity_step_errors.lock().push(msg);
                        }
                    }
                }
            }
        }

        // 2. Step LuaEntitySAO entities (mobs, falling nodes, dropped items + pickup)
        let dt_f = dtime as f32;
        let ent_ids: Vec<(u64, Arc<RwLock<ObjectState>>)> = self
            .shared
            .entities
            .read()
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect();

        let players: Vec<Arc<RwLock<ObjectState>>> =
            self.shared.players.read().values().cloned().collect();

        for (eid, earc) in ent_ids {
            let (removed, physical, entity_name) = {
                let mut g = earc.write();
                g.hurt_flash = (g.hurt_flash - dt_f).max(0.0);
                (g.removed, g.physical, g.entity_name.clone())
            };
            if removed {
                self.shared.entities.write().remove(&eid);
                continue;
            }

            let ents_map_opt = core.get::<Table>("_cubex_luaentities").ok();
            let ent_opt = ents_map_opt
                .as_ref()
                .and_then(|m| m.get::<Table>(eid).ok());
            let is_mcl_mob = ent_opt
                .as_ref()
                .and_then(|e| e.get::<bool>("is_mob").ok())
                .unwrap_or(false);

            let is_falling_node = entity_name == "__builtin:falling_node";
            let mut move_res = crate::physics::CollisionMoveResult::default();
            if physical {
                let (mut pos, mut vel, accel, accel_by_lua, cbox, stepheight) = {
                    let g = earc.read();
                    (
                        g.pos,
                        g.vel,
                        g.accel,
                        g.accel_set_by_lua,
                        g.collisionbox,
                        g.stepheight,
                    )
                };
                if is_falling_node {
                    vel.x = 0.0;
                    vel.z = 0.0;
                }
                let eff_accel = if is_falling_node {
                    Vec3::new(0.0, -18.0, 0.0)
                } else if accel_by_lua || accel.length_squared() > 1e-4 || is_mcl_mob {
                    accel
                } else {
                    Vec3::new(0.0, -9.81, 0.0)
                };
                let eff_step = if is_falling_node { 0.0 } else { stepheight };
                let restitution_scale = if is_falling_node { 0.0 } else { 1.0 };
                let reg = self.shared.registries.read();
                let aabb = crate::physics::Aabb::from_array(cbox);
                move_res = crate::physics::collision_move_simple_ex(
                    &self.shared.world,
                    &reg,
                    aabb,
                    eff_step,
                    dt_f,
                    &mut pos,
                    &mut vel,
                    eff_accel,
                    restitution_scale,
                );
                if is_falling_node {
                    vel.x = 0.0;
                    vel.z = 0.0;
                    let hit_y = move_res
                        .collisions
                        .iter()
                        .any(|c| c.axis == crate::physics::CollisionAxis::Y);
                    let bottom_y = pos.y + aabb.min.y;
                    let beneath_cell = IVec3::new(
                        pos.x.floor() as i32,
                        (bottom_y - 0.04).floor() as i32,
                        pos.z.floor() as i32,
                    );
                    let beneath_id = self.shared.world.get_node(beneath_cell).0;
                    let beneath_solid = beneath_id != CONTENT_AIR
                        && beneath_id != CONTENT_IGNORE
                        && reg
                            .get_node_def(beneath_id)
                            .is_some_and(|d| d.walkable && !d.buildable_to)
                        && (bottom_y - (beneath_cell.y + 1) as f32).abs() <= 0.04;

                    if move_res.touching_ground || hit_y || beneath_solid {
                        let lp = cubex_to_lua_pos(pos);
                        let mut snap = IVec3::new(
                            lp.x.round() as i32,
                            (lp.y + 0.1).floor() as i32,
                            lp.z.round() as i32,
                        );
                        for _ in 0..2 {
                            let existing_id = self.shared.world.get_node(snap).0;
                            let occupied = existing_id != CONTENT_AIR
                                && existing_id != CONTENT_IGNORE
                                && reg
                                    .get_node_def(existing_id)
                                    .is_some_and(|d| d.walkable && !d.buildable_to);
                            if occupied {
                                snap.y += 1;
                            } else {
                                break;
                            }
                        }

                        let (mut block_name, param2) = ent_opt
                            .as_ref()
                            .and_then(|e| e.get::<Table>("node").ok())
                            .map(|nt| {
                                (
                                    nt.get::<String>("name").unwrap_or_default(),
                                    nt.get::<u8>("param2").unwrap_or(0),
                                )
                            })
                            .unwrap_or_default();
                        if block_name.is_empty() {
                            let g = earc.read();
                            block_name = g
                                .textures
                                .first()
                                .cloned()
                                .unwrap_or_else(|| g.wield_item.clone());
                        }

                        let resolved_name = reg.resolve_alias(block_name.trim());
                        let node_id = reg.get_content_id(resolved_name);
                        if node_id != CONTENT_AIR
                            && node_id != CONTENT_IGNORE
                            && node_id != crate::registry::CONTENT_UNKNOWN
                        {
                            let (_, cur_p1, _) = self.shared.world.get_node(snap);
                            let light_src = reg
                                .get_node_def(node_id)
                                .map(|d| d.light_source.min(15))
                                .unwrap_or(0);
                            let sky = cur_p1 & 0x0F;
                            let param1 = (light_src << 4) | sky;
                            self.shared.world.set_node(snap, node_id, param1, param2);
                            crate::world::light::relight_around_pos(&self.shared.world, snap, &reg);
                            self.shared.world.mark_priority_dirty_with_neighbors(snap);
                            if let Some(place_snd) = reg
                                .get_node_def(node_id)
                                .and_then(|d| d.sounds.place.as_ref())
                            {
                                self.shared.pending_sounds.write().push(PendingSoundEvent {
                                    spec: place_snd.clone(),
                                    pos: Some(snap.as_vec3()),
                                    gain: place_snd.gain,
                                });
                            }
                        }
                        drop(reg);

                        {
                            let mut g = earc.write();
                            g.pos = pos;
                            g.vel = Vec3::ZERO;
                            g.accel = eff_accel;
                            g.is_on_ground = true;
                            g.removed = true;
                        }
                        self.shared.entities.write().remove(&eid);
                        if let Some(ref em) = ents_map_opt {
                            let _ = em.set(eid, Value::Nil);
                        }
                        if let Ok(Value::Table(refs)) =
                            self.lua.named_registry_value::<Value>("_cubex_object_refs")
                        {
                            let _ = refs.set(format!("e:{eid}"), Value::Nil);
                        }
                        continue;
                    }
                }
                drop(reg);

                let horiz_speed = (vel.x * vel.x + vel.z * vel.z).sqrt();
                let mut g = earc.write();
                g.pos = pos;
                g.vel = vel;
                if is_falling_node {
                    g.accel = Vec3::new(0.0, -18.0, 0.0);
                }
                g.is_on_ground = move_res.touching_ground;
                if horiz_speed > 0.15 {
                    let target_blend = (horiz_speed / 3.5).clamp(0.35, 1.15);
                    g.walk_blend += (target_blend - g.walk_blend) * (dt_f * 10.0).clamp(0.0, 1.0);
                    let freq = (horiz_speed * 2.1).clamp(3.0, 12.0);
                    g.walk_phase = (g.walk_phase + dt_f * freq) % std::f32::consts::TAU;
                } else {
                    g.walk_blend *= (1.0 - dt_f * 10.0).clamp(0.0, 1.0);
                }
            } else {
                let mut g = earc.write();
                let a = g.accel;
                g.vel += a * dt_f;
                let v = g.vel;
                g.pos += v * dt_f;
                if g.automatic_rotate.abs() > 1e-4 {
                    g.yaw += g.automatic_rotate * dt_f;
                }
                g.is_on_ground = false;
                g.walk_blend = 0.0;
            }

            if !is_falling_node {
                if let Some(ref ent) = ent_opt {
                    if let Ok(on_step) = ent.get::<mlua::Function>("on_step") {
                    if let Ok(mr) = self.lua.create_table() {
                        let _ = mr.set("touching_ground", move_res.touching_ground);
                        let _ = mr.set("collides", move_res.collides);
                        let _ = mr.set("standing_on_object", false);
                        if let Ok(cols) = self.lua.create_table() {
                            for (idx, col) in move_res.collisions.iter().enumerate() {
                                if let Ok(row) = self.lua.create_table() {
                                    let _ = row.set("type", "node");
                                    let axis_s = match col.axis {
                                        crate::physics::CollisionAxis::X => "x",
                                        crate::physics::CollisionAxis::Y => "y",
                                        crate::physics::CollisionAxis::Z => "z",
                                        crate::physics::CollisionAxis::None => "none",
                                    };
                                    let _ = row.set("axis", axis_s);
                                    if let Ok(npt) = self.make_lua_pos(col.node_pos) {
                                        let _ = row.set("node_pos", npt);
                                    }
                                    let mk_v = |v: Vec3| -> mlua::Result<Table> {
                                        let t = self.lua.create_table()?;
                                        t.set("x", v.x)?;
                                        t.set("y", v.y)?;
                                        t.set("z", v.z)?;
                                        Ok(t)
                                    };
                                    if let Ok(t) = mk_v(cubex_to_lua_pos(col.new_pos)) {
                                        let _ = row.set("new_pos", t);
                                    }
                                    if let Ok(t) = mk_v(col.old_speed) {
                                        let _ = row.set("old_velocity", t);
                                    }
                                    if let Ok(t) = mk_v(col.new_speed) {
                                        let _ = row.set("new_velocity", t);
                                    }
                                    let _ = cols.set(idx + 1, row);
                                }
                            }
                            let _ = mr.set("collisions", cols);
                        }
                        if let Err(e) = on_step.call::<Value>((ent.clone(), dtime, mr)) {
                            self.entity_step_error_count
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let key = format!("{entity_name}:{}", e.to_string().lines().next().unwrap_or(""));
                            if self.logged_step_errors.lock().insert(key) {
                                let msg = format!("entity[{eid}].on_step error ({entity_name}): {e}");
                                tracing::warn!("{msg}");
                                self.entity_step_errors.lock().push(msg);
                            }
                            if physical {
                                let mut g = earc.write();
                                if !g.is_on_ground {
                                    g.vel.y = (g.vel.y - 9.81 * dt_f).max(-20.0);
                                }
                            }
                        }
                    }
                }

                // Sync mcl_mobs custom `health` field -> engine ObjectState.hp when modified
                if is_mcl_mob && !earc.read().removed {
                    if let Ok(lh) = ent.get::<f64>("health") {
                        let mut g = earc.write();
                        g.hp = (lh.ceil() as i32).clamp(0, g.hp_max);
                    }
                }
            }
            }

            {
                let mut g = earc.write();
                g.age_sec += dt_f;
            }

            // Physical dropped item spin, friction, magnetic pull, and player pickup
            if entity_name == "__builtin:item" && !earc.read().removed {
                let lua_age: f64 = ent_opt
                    .as_ref()
                    .and_then(|e| e.get("age").ok())
                    .unwrap_or(0.0);
                let eff_age = {
                    let mut g = earc.write();
                    g.yaw = (g.yaw + 1.8 * dt_f) % std::f32::consts::TAU;
                    g.walk_phase = (g.walk_phase + dt_f * 2.5) % std::f32::consts::TAU;
                    if g.is_on_ground {
                        let damp = (1.0 - dt_f * 8.0).clamp(0.0, 1.0);
                        g.vel.x *= damp;
                        g.vel.z *= damp;
                    }
                    (g.age_sec as f64).max(lua_age)
                };

                if eff_age >= 0.30 {
                    let epos = earc.read().pos;
                    let mut best_pl: Option<(Arc<RwLock<ObjectState>>, Vec3, f32)> = None;
                    for pl in &players {
                        let pcenter = pl.read().pos + Vec3::new(0.0, 0.85, 0.0);
                        let d2 = epos.distance_squared(pcenter);
                        if d2 <= 2.6 * 2.6
                            && best_pl.as_ref().map(|b| d2 < b.2).unwrap_or(true)
                        {
                            best_pl = Some((pl.clone(), pcenter, d2));
                        }
                    }

                    if let Some((pl, pcenter, d2)) = best_pl {
                        let dist = d2.sqrt();
                        if dist > 1.20 {
                            let dir = (pcenter - epos).normalize_or_zero();
                            let pull = dir * 6.8;
                            let mut g = earc.write();
                            g.vel = g.vel.lerp(pull, (dt_f * 10.0).clamp(0.0, 1.0));
                        } else {
                            if let Some(ref ent) = ent_opt {
                                if let Ok(on_punch) = ent.get::<mlua::Function>("on_punch") {
                                    if let Ok(pl_ud) = object_ref_to_lua(&self.lua, pl.clone()) {
                                        let _ = on_punch.call::<Value>((ent.clone(), pl_ud));
                                    }
                                }
                            }
                            if !earc.read().removed {
                                let item_spec = ent_opt
                                    .as_ref()
                                    .and_then(|e| e.get::<String>("itemstring").ok())
                                    .filter(|s| !s.trim().is_empty())
                                    .unwrap_or_else(|| earc.read().wield_item.clone());
                                if !item_spec.trim().is_empty() {
                                    let reg = self.shared.registries.read();
                                    crate::physics::add_drops_to_inventory(
                                        &pl,
                                        &[item_spec],
                                        &reg,
                                    );
                                }
                                earc.write().removed = true;
                            }
                        }
                    }
                }
            }

            if earc.read().removed {
                self.shared.entities.write().remove(&eid);
                if let Some(ref em) = ents_map_opt {
                    let _ = em.set(eid, Value::Nil);
                }
                if let Ok(Value::Table(refs)) =
                    self.lua.named_registry_value::<Value>("_cubex_object_refs")
                {
                    let _ = refs.set(format!("e:{eid}"), Value::Nil);
                }
            }
        }

        // 3. Step NodeTimers (furnaces, saplings, crops)
        let expired_timers: Vec<(IVec3, f64)> = {
            let mut timers = self.shared.world.node_timers.write();
            let mut exp = Vec::new();
            for (&pos, st) in timers.iter_mut() {
                if st.started && st.timeout > 0.0 {
                    st.elapsed += dtime;
                    if st.elapsed >= st.timeout {
                        exp.push((pos, st.elapsed));
                    }
                }
            }
            exp
        };
        if !expired_timers.is_empty() {
            if let Ok(rn) = core.get::<Table>("registered_nodes") {
                for (pos, elapsed) in expired_timers {
                    let (id, _, _) = self.shared.world.get_node(pos);
                    let name = self
                        .shared
                        .registries
                        .read()
                        .get_name_from_content_id(id)
                        .to_string();
                    let mut restart = false;
                    if let Ok(def) = rn.get::<Table>(name.as_str()) {
                        if let Ok(on_timer) = def.get::<mlua::Function>("on_timer") {
                            if let Ok(pt) = self.make_lua_pos(pos) {
                                match on_timer.call::<bool>((pt, elapsed)) {
                                    Ok(r) => restart = r,
                                    Err(e) => {
                                        tracing::debug!("on_timer error on {name} at {pos}: {e}");
                                    }
                                }
                            }
                        }
                    }
                    let mut timers = self.shared.world.node_timers.write();
                    if restart {
                        if let Some(st) = timers.get_mut(&pos) {
                            st.elapsed = (st.elapsed - st.timeout).max(0.0);
                        }
                    } else if let Some(st) = timers.get(&pos) {
                        // Preserve timer if on_timer explicitly restarted it via :start()/:set()
                        if st.elapsed >= st.timeout {
                            timers.remove(&pos);
                        }
                    }
                }
            }
        }

        // 4. Incremental amortized ABMs (0.25s tick scanning 1 rotating chunk)
        let run_abms = {
            let mut acc = self.shared.abm_accum.write();
            *acc += dtime;
            if *acc >= 0.25 {
                *acc = (*acc - 0.25).min(0.25);
                true
            } else {
                false
            }
        };
        if run_abms && !players.is_empty() {
            self.step_abms(&core, &players[0]);
        }

        // 5. Biome-aware natural surface mob spawning & distant despawn
        self.tick_natural_mob_spawning(&core, &players, dtime);
    }

    fn is_hostile_mob_name(entity_name: &str) -> bool {
        let l = entity_name.to_ascii_lowercase();
        l.contains("zombie")
            || l.contains("skeleton")
            || l.contains("creeper")
            || l.contains("spider")
            || l.contains("husk")
            || l.contains("enderman")
            || l.contains("witch")
            || l.contains("pillager")
            || l.contains("slime")
            || l.contains("drowned")
    }

    fn find_surface_spawn_pos(
        &self,
        wx: i32,
        ref_y: i32,
        wz: i32,
    ) -> Option<(i32, u8, u8)> {
        let y_hi = (ref_y + 24).min(160);
        let y_lo = (ref_y - 24).max(2);
        let reg = self.shared.registries.read();
        let passable_space = |pos: IVec3| -> Option<(u8, u8)> {
            let (id, p1, _) = self.shared.world.get_node(pos);
            if id == CONTENT_IGNORE {
                return None;
            }
            if id == CONTENT_AIR {
                return Some((p1 & 0x0F, (p1 >> 4) & 0x0F));
            }
            let def = reg.get_node_def(id)?;
            if !def.walkable && !crate::physics::collision::is_liquid_node(def) {
                Some((p1 & 0x0F, (p1 >> 4) & 0x0F))
            } else {
                None
            }
        };

        for y in (y_lo..=y_hi).rev() {
            let (gid, _, _) = self.shared.world.get_node(IVec3::new(wx, y, wz));
            if gid == CONTENT_AIR || gid == CONTENT_IGNORE {
                continue;
            }
            let Some(def) = reg.get_node_def(gid) else {
                continue;
            };
            if !def.walkable
                || crate::physics::collision::is_liquid_node(def)
                || def.drawtype == DrawType::AllfacesOptional
                || def.name == "mcl_core:bedrock"
                || def.name.contains("leaves")
                || def.name.contains("tree")
                || def.name.contains("log")
                || def.groups.get("leaves").copied().unwrap_or(0) > 0
                || def.groups.get("tree").copied().unwrap_or(0) > 0
            {
                continue;
            }
            if let Some((sky, block)) = passable_space(IVec3::new(wx, y + 1, wz)) {
                if passable_space(IVec3::new(wx, y + 2, wz)).is_some() {
                    return Some((y + 1, sky, block));
                }
            }
        }
        None
    }

    fn tick_natural_mob_spawning(
        &self,
        core: &Table,
        players: &[Arc<RwLock<ObjectState>>],
        dtime: f64,
    ) {
        if players.is_empty() {
            return;
        }
        if self
            .shared
            .config
            .read()
            .get("mobs_spawn")
            .is_some_and(|v| v.eq_ignore_ascii_case("false") || v == "0")
        {
            return;
        }
        if self.shared.world.chunks.read().len() < 9 {
            return;
        }

        let Ok(reg_ents) = core.get::<Table>("registered_entities") else {
            return;
        };
        let Ok(add_entity_fn) = core.get::<mlua::Function>("add_entity") else {
            return;
        };

        let player_positions: Vec<Vec3> = players.iter().map(|p| p.read().pos).collect();
        let ents_map_opt = core.get::<Table>("_cubex_luaentities").ok();

        let mut passive_cnt = 0usize;
        let mut hostile_cnt = 0usize;
        let mut existing_positions = Vec::new();
        let mut to_despawn = Vec::new();

        {
            let ents = self.shared.entities.read();
            for (&eid, earc) in ents.iter() {
                let g = earc.read();
                if g.removed
                    || g.entity_name == "__builtin:item"
                    || g.entity_name == "__builtin:falling_node"
                {
                    continue;
                }
                let min_d2 = player_positions
                    .iter()
                    .map(|p| p.distance_squared(g.pos))
                    .fold(f32::INFINITY, f32::min);
                if min_d2 > 96.0 * 96.0 {
                    let keep = ents_map_opt
                        .as_ref()
                        .and_then(|m| m.get::<Table>(eid).ok())
                        .map(|t| {
                            matches!(t.get::<Value>("persistent"), Ok(Value::Boolean(true)))
                                || matches!(t.get::<Value>("tamed"), Ok(Value::Boolean(true)))
                                || matches!(t.get::<Value>("can_despawn"), Ok(Value::Boolean(false)))
                        })
                        .unwrap_or(false);
                    if !keep {
                        to_despawn.push(eid);
                        continue;
                    }
                }
                if Self::is_hostile_mob_name(&g.entity_name) {
                    hostile_cnt += 1;
                } else {
                    passive_cnt += 1;
                }
                existing_positions.push(g.pos);
            }
        }

        if !to_despawn.is_empty() {
            let mut ents = self.shared.entities.write();
            for eid in to_despawn {
                if let Some(arc) = ents.remove(&eid) {
                    arc.write().removed = true;
                }
                if let Some(ref m) = ents_map_opt {
                    let _ = m.set(eid, Value::Nil);
                }
            }
        }

        const PASSIVE_CAP: usize = 12;
        const HOSTILE_CAP: usize = 8;
        if passive_cnt >= PASSIVE_CAP && hostile_cnt >= HOSTILE_CAP {
            return;
        }

        let is_initial = !*self.initial_mobs_spawned.lock();
        let (max_probes, max_spawns, r_min, r_max) = if is_initial {
            *self.initial_mobs_spawned.lock() = true;
            (28usize, 8usize, 10.0f32, 26.0f32)
        } else {
            let mut t = self.mob_spawn_timer.lock();
            *t += dtime;
            if *t < 2.0 {
                return;
            }
            *t = (*t - 2.0).min(1.0);
            (8usize, 3usize, 12.0f32, 34.0f32)
        };

        let tod = *self.shared.time_of_day.read();
        let is_night = !(0.22..=0.78).contains(&tod);
        let allow_hostiles = *self.shared.world.difficulty.read() != Difficulty::Peaceful;

        let mut spawned = 0usize;
        for _ in 0..max_probes {
            if spawned >= max_spawns || (passive_cnt >= PASSIVE_CAP && hostile_cnt >= HOSTILE_CAP) {
                break;
            }
            let pidx = (rand::random::<usize>()) % player_positions.len();
            let ppos = player_positions[pidx];
            let angle = rand::random::<f32>() * std::f32::consts::TAU;
            let dist = r_min + rand::random::<f32>() * (r_max - r_min);
            let wx = (ppos.x + angle.cos() * dist).floor() as i32;
            let wz = (ppos.z + angle.sin() * dist).floor() as i32;

            let Some((sy, sky, block)) =
                self.find_surface_spawn_pos(wx, ppos.y.round() as i32, wz)
            else {
                continue;
            };
            let spawn_pos = Vec3::new(wx as f32 + 0.5, sy as f32 + 0.05, wz as f32 + 0.5);
            if player_positions
                .iter()
                .any(|p| p.distance_squared(spawn_pos) < 9.5 * 9.5)
            {
                continue;
            }
            if existing_positions
                .iter()
                .any(|e| e.distance_squared(spawn_pos) < 3.0 * 3.0)
            {
                continue;
            }

            let bname = crate::world::mapgen::biome_name_at(wx, wz, self.shared.world.seed);
            let want_hostile = allow_hostiles
                && hostile_cnt < HOSTILE_CAP
                && (is_night || (sky <= 4 && block <= 7))
                && (is_night || passive_cnt >= 4 || rand::random::<f32>() < 0.5);

            let candidates: &[&str] = if want_hostile {
                if bname == "Desert" {
                    &["mobs_mc:husk", "mobs_mc:skeleton", "mobs_mc:spider", "mobs_mc:creeper"]
                } else {
                    &["mobs_mc:zombie", "mobs_mc:skeleton", "mobs_mc:creeper", "mobs_mc:spider"]
                }
            } else if passive_cnt < PASSIVE_CAP {
                match bname {
                    "Plains" => &["mobs_mc:cow", "mobs_mc:sheep", "mobs_mc:pig", "mobs_mc:chicken"],
                    "Forest" | "Birch Forest" => {
                        &["mobs_mc:cow", "mobs_mc:pig", "mobs_mc:sheep", "mobs_mc:wolf", "mobs_mc:chicken"]
                    }
                    "Taiga" => &["mobs_mc:wolf", "mobs_mc:sheep", "mobs_mc:pig", "mobs_mc:rabbit"],
                    "Desert" => &["mobs_mc:rabbit", "mobs_mc:llama"],
                    "Swamp" => &["mobs_mc:pig", "mobs_mc:sheep", "mobs_mc:chicken"],
                    "Snowy Tundra" => &["mobs_mc:rabbit", "mobs_mc:wolf", "mobs_mc:sheep"],
                    _ => &["mobs_mc:cow", "mobs_mc:sheep", "mobs_mc:pig", "mobs_mc:chicken"],
                }
            } else {
                continue;
            };

            let start_i = rand::random::<usize>() % candidates.len();
            let chosen = (0..candidates.len())
                .map(|k| candidates[(start_i + k) % candidates.len()])
                .find(|name| matches!(reg_ents.get::<Value>(*name), Ok(Value::Table(_))));
            let Some(mob_name) = chosen else {
                continue;
            };

            if let Ok(ptbl) = self.lua.create_table() {
                let lua_spawn = cubex_to_lua_pos(spawn_pos);
                let _ = ptbl.set("x", lua_spawn.x as f64);
                let _ = ptbl.set("y", lua_spawn.y as f64);
                let _ = ptbl.set("z", lua_spawn.z as f64);
                if let Ok(Value::UserData(_)) = add_entity_fn.call::<Value>((ptbl, mob_name)) {
                    existing_positions.push(spawn_pos);
                    if want_hostile {
                        hostile_cnt += 1;
                    } else {
                        passive_cnt += 1;
                    }
                    spawned += 1;
                }
            }
        }
    }

    fn step_abms(&self, core: &Table, player: &Arc<RwLock<ObjectState>>) {
        const MAX_ACTIONS_PER_TICK: usize = 12;
        if self.abm_by_node_id.is_empty() {
            return;
        }
        let Ok(abms) = core.get::<Table>("registered_abms") else {
            return;
        };

        let center_pos = player.read().pos.floor().as_ivec3();
        let (center_chunk, _, _, _) = crate::world::world_to_chunk(center_pos);
        let cursor = {
            let mut c = self.abm_chunk_cursor.lock();
            let cur = *c % 18;
            *c = c.wrapping_add(1);
            cur
        };
        let dy_idx = (cursor / 9) as i32;
        let rem = cursor % 9;
        let dz = (rem / 3) as i32 - 1;
        let dx = (rem % 3) as i32 - 1;
        let dy = if center_pos.y.rem_euclid(16) < 8 {
            dy_idx - 1
        } else {
            dy_idx
        };
        let cpos = center_chunk + IVec3::new(dx, dy, dz);

        let Some(ch_arc) = self.shared.world.get_chunk(cpos) else {
            return;
        };

        let start_off = (rand::random::<usize>()) % crate::world::CHUNK_VOLUME;
        let mut to_fire: Vec<(IVec3, u16, u8, u8, usize)> = Vec::new();
        {
            let ch = ch_arc.read();
            for step_i in 0..crate::world::CHUNK_VOLUME {
                let li = (start_off + step_i) % crate::world::CHUNK_VOLUME;
                let id = ch.nodes[li];
                if id == CONTENT_IGNORE {
                    continue;
                }
                let Some(entries) = self.abm_by_node_id.get(&id) else {
                    continue;
                };
                for &(abm_idx, p_eff) in entries {
                    if p_eff <= 1 || (rand::random::<u32>() % p_eff) == 0 {
                        let lx = (li & 15) as i32;
                        let lz = ((li >> 4) & 15) as i32;
                        let ly = ((li >> 8) & 15) as i32;
                        let wpos = cpos * 16 + IVec3::new(lx, ly, lz);
                        to_fire.push((wpos, id, ch.param1[li], ch.param2[li], abm_idx));
                        if to_fire.len() >= MAX_ACTIONS_PER_TICK {
                            break;
                        }
                    }
                }
                if to_fire.len() >= MAX_ACTIONS_PER_TICK {
                    break;
                }
            }
        }

        for (pos, id, p1, p2, abm_idx) in to_fire {
            let Ok(abm) = abms.get::<Table>(abm_idx) else {
                continue;
            };
            let Ok(action) = abm.get::<mlua::Function>("action") else {
                continue;
            };
            let name = self
                .shared
                .registries
                .read()
                .get_name_from_content_id(id)
                .to_string();
            if let (Ok(pt), Ok(nt)) = (self.make_lua_pos(pos), self.make_lua_node(&name, p1, p2)) {
                let _ = action.call::<Value>((pt, nt, 0u32, 0u32));
            }
        }
    }

    pub fn pop_sounds(&self) -> Vec<PendingSoundEvent> {
        let mut g = self.shared.pending_sounds.write();
        std::mem::take(&mut *g)
    }

    pub fn active_entities(&self) -> Vec<Arc<RwLock<ObjectState>>> {
        self.shared
            .entities
            .read()
            .values()
            .filter(|e| !e.read().removed)
            .cloned()
            .collect()
    }

    pub fn punch_entity(
        &self,
        entity_id: u64,
        puncher: &Arc<RwLock<ObjectState>>,
        tflp: f32,
        dir: Vec3,
    ) -> PunchEntityResult {
        let Some(target_arc) = self.shared.entities.read().get(&entity_id).cloned() else {
            return PunchEntityResult::default();
        };
        if target_arc.read().removed {
            return PunchEntityResult::default();
        }

        let wield_item = {
            let g = puncher.read();
            let idx = g.wield_index.saturating_sub(1);
            let mg = g.meta.read();
            mg.inventories
                .get("main")
                .and_then(|v| v.get(idx))
                .filter(|s| !s.is_empty())
                .map(|s| s.name.clone())
                .unwrap_or_default()
        };

        let tc_tbl = self.build_tool_capabilities_table(&wield_item);
        let Ok(target_ud) = object_ref_to_lua(&self.lua, target_arc.clone()) else {
            return PunchEntityResult::default();
        };
        let Ok(puncher_ud) = object_ref_to_lua(&self.lua, puncher.clone()) else {
            return PunchEntityResult::default();
        };
        let dir_n = if dir.length_squared() > 1e-6 {
            dir.normalize()
        } else {
            Vec3::Z
        };
        let dir_tbl = self::core_api::make_vec3(&self.lua, dir_n.x as f64, dir_n.y as f64, dir_n.z as f64)
            .ok();

        let fallback_drops = self.collect_entity_fallback_drops(entity_id);
        let min_eid_before = *self.shared.next_entity_id.read();
        let old_hp = target_arc.read().hp;

        let damage = if let Ok(punch_fn) = target_ud.get::<mlua::Function>("punch") {
            let dir_val = dir_tbl.map(Value::Table).unwrap_or(Value::Nil);
            punch_fn
                .call::<i32>((
                    target_ud.clone(),
                    puncher_ud,
                    tflp.max(0.0) as f64,
                    tc_tbl,
                    dir_val,
                ))
                .unwrap_or(0)
        } else {
            0
        };

        let killed = {
            let g = target_arc.read();
            g.removed || g.hp <= 0
        };
        if killed {
            self.shared.entities.write().remove(&entity_id);
        }

        let mut drops = self.take_spawned_item_drops(min_eid_before);
        if killed && drops.is_empty() && !fallback_drops.is_empty() {
            drops = fallback_drops;
        }
        let effective_damage = damage.max((old_hp - target_arc.read().hp).max(0));
        PunchEntityResult {
            hit: true,
            damage: effective_damage,
            killed,
            drops,
        }
    }

    pub fn punch_player(
        &self,
        target_player: &Arc<RwLock<ObjectState>>,
        puncher: &Arc<RwLock<ObjectState>>,
        tflp: f32,
        dir: Vec3,
    ) -> PunchEntityResult {
        if !self.shared.world.gamerules.read().pvp {
            return PunchEntityResult::default();
        }
        {
            let tg = target_player.read();
            if tg.removed || tg.is_dead || tg.spawn_invuln_timer > 0.0 {
                return PunchEntityResult::default();
            }
        }
        let wield_item = {
            let g = puncher.read();
            let idx = g.wield_index.saturating_sub(1);
            let mg = g.meta.read();
            mg.inventories
                .get("main")
                .and_then(|v| v.get(idx))
                .filter(|s| !s.is_empty())
                .map(|s| s.name.clone())
                .unwrap_or_default()
        };
        let tc_tbl = self.build_tool_capabilities_table(&wield_item);
        let Ok(target_ud) = object_ref_to_lua(&self.lua, target_player.clone()) else {
            return PunchEntityResult::default();
        };
        let Ok(puncher_ud) = object_ref_to_lua(&self.lua, puncher.clone()) else {
            return PunchEntityResult::default();
        };
        let dir_n = if dir.length_squared() > 1e-6 {
            dir.normalize()
        } else {
            Vec3::Z
        };
        let dir_tbl = self::core_api::make_vec3(&self.lua, dir_n.x as f64, dir_n.y as f64, dir_n.z as f64)
            .ok();
        let old_hp = target_player.read().hp;

        let damage = if let Ok(punch_fn) = target_ud.get::<mlua::Function>("punch") {
            let dir_val = dir_tbl.map(Value::Table).unwrap_or(Value::Nil);
            punch_fn
                .call::<i32>((target_ud, puncher_ud, tflp.max(0.0) as f64, tc_tbl, dir_val))
                .unwrap_or(0)
        } else {
            0
        };
        let cur_hp = target_player.read().hp;
        PunchEntityResult {
            hit: true,
            damage: damage.max((old_hp - cur_hp).max(0)),
            killed: cur_hp <= 0,
            drops: Vec::new(),
        }
    }

    /// Trigger `core.registered_on_dieplayers` once for a player dying from Rust-side
    /// damage (such as fall damage or exact eye-position suffocation).
    pub fn notify_player_died(&self, player_state: &Arc<RwLock<ObjectState>>, reason_type: &str) {
        let Ok(player_ud) = object_ref_to_lua(&self.lua, player_state.clone()) else {
            return;
        };
        let Ok(core) = self.lua.globals().get::<Table>("core") else {
            return;
        };
        let reason_tbl = match self.lua.create_table() {
            Ok(t) => {
                let _ = t.set("type", reason_type);
                Value::Table(t)
            }
            Err(_) => Value::Nil,
        };
        if let Ok(die_cbs) = core.get::<Table>("registered_on_dieplayers") {
            let len = die_cbs.len().unwrap_or(0);
            for i in 1..=len {
                if let Ok(func) = die_cbs.get::<mlua::Function>(i) {
                    let _ = func.call::<Value>((player_ud.clone(), reason_tbl.clone()));
                }
            }
        }
    }

    /// Trigger `core.registered_on_respawnplayers` when the user clicks Respawn.
    pub fn notify_player_respawned(&self, player_state: &Arc<RwLock<ObjectState>>) {
        let Ok(player_ud) = object_ref_to_lua(&self.lua, player_state.clone()) else {
            return;
        };
        let Ok(core) = self.lua.globals().get::<Table>("core") else {
            return;
        };
        if let Ok(respawn_cbs) = core.get::<Table>("registered_on_respawnplayers") {
            let len = respawn_cbs.len().unwrap_or(0);
            for i in 1..=len {
                if let Ok(func) = respawn_cbs.get::<mlua::Function>(i) {
                    let _ = func.call::<Value>(player_ud.clone());
                }
            }
        }
    }

    fn build_tool_capabilities_table(&self, item_name: &str) -> Value {
        if !item_name.is_empty() {
            if let Ok(core) = self.lua.globals().get::<Table>("core") {
                if let Ok(items) = core.get::<Table>("registered_items") {
                    if let Ok(def) = items.get::<Table>(item_name) {
                        if let Ok(tc) = def.get::<Table>("tool_capabilities") {
                            let has_dg = tc
                                .get::<Table>("damage_groups")
                                .map(|dg| dg.pairs::<Value, Value>().next().is_some())
                                .unwrap_or(false);
                            if has_dg {
                                return Value::Table(tc);
                            }
                        }
                    }
                }
            }
            if let Some(tc_rust) = self
                .shared
                .registries
                .read()
                .get_item_def(item_name)
                .and_then(|d| d.tool_capabilities.clone())
            {
                if !tc_rust.damage_groups.is_empty() {
                    if let Ok(t) = self.lua.create_table() {
                        let _ = t.set("full_punch_interval", tc_rust.full_punch_interval);
                        if let Ok(dg) = self.lua.create_table() {
                            for (k, v) in &tc_rust.damage_groups {
                                let _ = dg.set(k.as_str(), *v);
                            }
                            let _ = t.set("damage_groups", dg);
                        }
                        return Value::Table(t);
                    }
                }
            }
        }

        let Ok(t) = self.lua.create_table() else {
            return Value::Nil;
        };
        let _ = t.set("full_punch_interval", 1.0f32);
        if let Ok(dg) = self.lua.create_table() {
            let _ = dg.set("fleshy", 1i32);
            let _ = t.set("damage_groups", dg);
        }
        Value::Table(t)
    }

    fn collect_entity_fallback_drops(&self, entity_id: u64) -> Vec<String> {
        let mut out = Vec::new();
        let Ok(core) = self.lua.globals().get::<Table>("core") else {
            return out;
        };
        let Ok(ents) = core.get::<Table>("_cubex_luaentities") else {
            return out;
        };
        let Ok(ent) = ents.get::<Table>(entity_id) else {
            return out;
        };
        let Ok(drops_tbl) = ent.get::<Table>("drops") else {
            return out;
        };
        let len = drops_tbl.len().unwrap_or(0);
        for i in 1..=len {
            match drops_tbl.get::<Value>(i) {
                Ok(Value::String(s)) => {
                    if let Ok(st) = s.to_str() {
                        if !st.trim().is_empty() {
                            out.push(st.to_string());
                        }
                    }
                }
                Ok(Value::Table(row)) => {
                    if let Ok(name) = row.get::<String>("name") {
                        let chance: u32 = row.get::<u32>("chance").unwrap_or(1).max(1);
                        let min_c: u32 = row.get::<u32>("min").unwrap_or(1);
                        let max_c: u32 = row.get::<u32>("max").unwrap_or(min_c).max(min_c);
                        if !name.trim().is_empty() && (chance <= 1 || rand::random::<u32>() % chance == 0) {
                            let c = if max_c > min_c {
                                min_c + (rand::random::<u32>() % (max_c - min_c + 1))
                            } else {
                                min_c
                            };
                            if c > 0 {
                                out.push(format!("{name} {c}"));
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        out
    }

    pub fn send_formspec_fields(
        &self,
        player: Arc<RwLock<ObjectState>>,
        formname: &str,
        fields: &[(String, String)],
    ) {
        self.send_formspec_fields_at(player, formname, fields, None);
    }

    pub fn send_formspec_fields_at(
        &self,
        player: Arc<RwLock<ObjectState>>,
        formname: &str,
        fields: &[(String, String)],
        context_pos: Option<IVec3>,
    ) {
        let Ok(core) = self.lua.globals().get::<Table>("core") else {
            return;
        };
        let Ok(ftbl) = self.lua.create_table() else {
            return;
        };
        for (k, v) in fields {
            let _ = ftbl.set(k.as_str(), v.as_str());
        }
        let Ok(obj) = object_ref_to_lua(&self.lua, player) else {
            return;
        };
        if let Some(pos) = context_pos {
            if let Some((name, cb, pt)) = self.get_node_callback_and_pos(pos, "on_receive_fields") {
                if let Err(e) = cb.call::<Value>((pt, formname, ftbl.clone(), obj.clone())) {
                    tracing::debug!("on_receive_fields error on {name} at {pos}: {e}");
                }
            }
        }
        let Ok(cbs) = core.get::<Table>("registered_on_player_receive_fields") else {
            return;
        };
        let len = cbs.len().unwrap_or(0);
        for i in (1..=len).rev() {
            if let Ok(f) = cbs.get::<mlua::Function>(i) {
                if let Ok(handled) = f.call::<bool>((obj.clone(), formname, ftbl.clone())) {
                    if handled {
                        break;
                    }
                }
            }
        }
    }

    pub fn pop_formspecs(&self) -> Vec<FormspecRequest> {
        let mut g = self.shared.pending_formspecs.write();
        std::mem::take(&mut *g)
    }

    pub fn set_gamemode(&self, player: Arc<RwLock<ObjectState>>, mode: &str) {
        let name = player.read().name.clone();
        if let Ok(core) = self.lua.globals().get::<Table>("core") {
            if let Ok(mcl_gm) = self.lua.globals().get::<Table>("mcl_gamemode") {
                if let Ok(set_gm) = mcl_gm.get::<mlua::Function>("set_gamemode") {
                    if let Ok(p_ud) = object_ref_to_lua(&self.lua, player.clone()) {
                        let _ = set_gm.call::<()>((p_ud, mode));
                    }
                }
            }
            if let Ok(settings) = core.get::<mlua::AnyUserData>("settings") {
                if let Ok(set_b) = settings.get::<mlua::Function>("set_bool") {
                    let _ = set_b.call::<()>((settings.clone(), "creative_mode", mode == "creative"));
                }
            }
            let _ = core;
        }
        let mut privs = self
            .shared
            .player_privs
            .read()
            .get(&name)
            .cloned()
            .unwrap_or_default();
        privs.insert("interact".into(), true);
        privs.insert("shout".into(), true);
        if mode == "creative" || mode == "spectator" {
            privs.insert("fly".into(), true);
        } else {
            privs.remove("fly");
        }
        self.shared.player_privs.write().insert(name, privs);
    }
}

fn parse_drops_lua_value(val: Value) -> Vec<String> {
    match val {
        Value::Table(t) => {
            let len = t.len().unwrap_or(0);
            let mut out = Vec::new();
            for i in 1..=len {
                if let Ok(v) = t.get::<Value>(i) {
                    match v {
                        Value::String(st) => {
                            if let Ok(s) = st.to_str() {
                                let trimmed = s.trim();
                                if !trimmed.is_empty() {
                                    out.push(trimmed.to_string());
                                }
                            }
                        }
                        Value::UserData(ud) => {
                            if let Ok(stk) = ud.borrow::<ItemStackUD>() {
                                let s = stk.0.read().to_itemstring();
                                if !s.is_empty() {
                                    out.push(s);
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            out
        }
        Value::String(st) => st
            .to_str()
            .ok()
            .and_then(|s| {
                let t = s.trim();
                (!t.is_empty()).then(|| vec![t.to_string()])
            })
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{CraftRecipe, ItemDef, ToolCapabilities, ToolGroupCap};
    use std::collections::HashMap;

    #[test]
    fn dig_params_hand_vs_tool_and_wear() {
        let mut stone_groups = HashMap::new();
        stone_groups.insert("cracky".into(), 2);
        stone_groups.insert("level".into(), 1);

        // No tool -> not diggable unless dig_immediate
        let dp_none = compute_dig_params(&stone_groups, None, false);
        assert!(!dp_none.diggable);

        // Creative -> instant 0s, 0 wear
        let dp_crea = compute_dig_params(&stone_groups, None, true);
        assert!(dp_crea.diggable);
        assert_eq!(dp_crea.time, 0.0);
        assert_eq!(dp_crea.wear, 0);

        // Diamond pickaxe with maxlevel = 2, uses = 30 -> leveldiff = 1, wear > 0
        let mut times = HashMap::new();
        times.insert(2, 1.2);
        let mut groupcaps = HashMap::new();
        groupcaps.insert(
            "cracky".into(),
            ToolGroupCap {
                times,
                uses: 30,
                maxlevel: 2,
            },
        );
        let pick = ToolCapabilities {
            full_punch_interval: 0.9,
            max_drop_level: 2,
            groupcaps,
            damage_groups: HashMap::new(),
            punch_attack_uses: 0,
        };
        let dp_pick = compute_dig_params(&stone_groups, Some(&pick), false);
        assert!(dp_pick.diggable);
        assert!((dp_pick.time - 1.2).abs() < 1e-5);
        // real_uses = 30 * 3^1 = 90 => 65536 / 90 = 728
        assert_eq!(dp_pick.wear, 728);

        // dig_immediate = 3 without explicit groupcap -> 0s, 0 wear
        let mut imm_groups = HashMap::new();
        imm_groups.insert("dig_immediate".into(), 3);
        let dp_imm = compute_dig_params(&imm_groups, Some(&pick), false);
        assert!(dp_imm.diggable);
        assert_eq!(dp_imm.time, 0.0);
        assert_eq!(dp_imm.wear, 0);
    }

    #[test]
    fn craft_recipe_matching_shaped_shapeless_cooking_and_fuel() {
        let mut reg = EngineRegistries::new();
        let mut wood_groups = HashMap::new();
        wood_groups.insert("wood".into(), 1);
        reg.register_item(ItemDef {
            name: "mcl_trees:wood_oak".into(),
            item_type: "node".into(),
            groups: wood_groups,
            is_node: true,
            ..Default::default()
        });
        reg.register_item(ItemDef {
            name: "mcl_core:stick".into(),
            item_type: "craft".into(),
            ..Default::default()
        });
        reg.register_item(ItemDef {
            name: "mcl_core:coal_lump".into(),
            item_type: "craft".into(),
            ..Default::default()
        });

        reg.crafts.push(CraftRecipe {
            recipe_type: "shaped".into(),
            output: "mcl_core:stick 4".into(),
            recipe: vec![
                vec!["group:wood".into()],
                vec!["group:wood".into()],
            ],
            replacements: Vec::new(),
            cooktime: 0.0,
            burntime: 0.0,
        });
        reg.crafts.push(CraftRecipe {
            recipe_type: "shapeless".into(),
            output: "mcl_torches:torch 4".into(),
            recipe: vec![vec!["mcl_core:coal_lump".into(), "mcl_core:stick".into()]],
            replacements: Vec::new(),
            cooktime: 0.0,
            burntime: 0.0,
        });
        reg.crafts.push(CraftRecipe {
            recipe_type: "cooking".into(),
            output: "mcl_core:stone".into(),
            recipe: vec![vec!["mcl_core:cobble".into()]],
            replacements: Vec::new(),
            cooktime: 10.0,
            burntime: 0.0,
        });
        reg.crafts.push(CraftRecipe {
            recipe_type: "fuel".into(),
            output: String::new(),
            recipe: vec![vec!["mcl_core:coal_lump".into()]],
            replacements: Vec::new(),
            cooktime: 0.0,
            burntime: 80.0,
        });

        // 2x2 inventory craft: two oak planks vertically in right column
        let mut grid2x2 = vec![ItemStackData::default(); 4];
        grid2x2[1] = ItemStackData::from_spec("mcl_trees:wood_oak 2");
        grid2x2[3] = ItemStackData::from_spec("mcl_trees:wood_oak 1");
        let res = resolve_craft_recipe(&reg, &grid2x2, 2, "normal").expect("shaped match");
        assert_eq!(res.item.name, "mcl_core:stick");
        assert_eq!(res.item.count, 4);
        assert_eq!(res.decrements[1].count, 1);
        assert!(res.decrements[3].is_empty());

        // 3x3 shapeless torch
        let mut grid3x3 = vec![ItemStackData::default(); 9];
        grid3x3[0] = ItemStackData::from_spec("mcl_core:stick 1");
        grid3x3[8] = ItemStackData::from_spec("mcl_core:coal_lump 1");
        let res_sl = resolve_craft_recipe(&reg, &grid3x3, 3, "normal").expect("shapeless match");
        assert_eq!(res_sl.item.name, "mcl_torches:torch");
        assert_eq!(res_sl.item.count, 4);

        // Furnace cooking & fuel
        let cook = resolve_craft_recipe(&reg, &[ItemStackData::from_spec("mcl_core:cobble 2")], 1, "cooking").unwrap();
        assert_eq!(cook.item.name, "mcl_core:stone");
        assert_eq!(cook.time, 10.0);
        let fuel = resolve_craft_recipe(&reg, &[ItemStackData::from_spec("mcl_core:coal_lump 1")], 1, "fuel").unwrap();
        assert_eq!(fuel.time, 80.0);
    }

    #[test]
    fn mineclonia_full_boot_and_gameplay_apis() {
        let paths = EnginePaths::resolve();
        assert!(
            paths.game_dir.join("mods").is_dir(),
            "Mineclonia game dir missing at {}",
            paths.game_dir.display()
        );

        let registries = Arc::new(RwLock::new(EngineRegistries::new()));
        let tmp_world = std::env::temp_dir().join(format!(
            "cubex_rs_core_it_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&tmp_world);
        let world = Arc::new(WorldStore::new(&tmp_world, 4242));
        let mut cfg = MinetestConfig::load(&paths.conf_file);
        cfg.set("creative_mode", false);
        cfg.set("enable_damage", true);
        let config = Arc::new(RwLock::new(cfg));

        let runtime = LuaRuntime::boot(paths.clone(), registries.clone(), world.clone(), config)
            .expect("LuaRuntime boot failed");

        assert!(
            runtime.load_report.mods_discovered >= 200,
            "Expected >= 200 Mineclonia mods, got {}",
            runtime.load_report.mods_discovered
        );
        assert!(
            runtime.load_report.errors.is_empty(),
            "Expected 0 Lua mod load errors, got {:?}",
            runtime.load_report.errors
        );

        // 1. Verify extended NodeDef & ItemDef fields populated from Mineclonia
        {
            let reg = registries.read();
            let fence = reg
                .get_node_def(reg.get_content_id("mcl_fences:oak_fence"))
                .expect("mcl_fences:oak_fence registered");
            assert_eq!(fence.name, "mcl_fences:oak_fence");
            assert_eq!(fence.node_box_type, "connected");
            assert!(!fence.connects_to.is_empty());
            assert!(!fence.selection_boxes.is_empty());
            assert!(!fence.collision_boxes.is_empty());

            let leaves = reg
                .get_node_def(reg.get_content_id("mcl_trees:leaves_oak"))
                .expect("mcl_trees:leaves_oak registered");
            assert_eq!(leaves.name, "mcl_trees:leaves_oak");
            assert!(leaves.palette.is_some());

            let stone = reg
                .get_node_def(reg.get_content_id("mcl_core:stone"))
                .expect("mcl_core:stone registered");
            assert_eq!(stone.name, "mcl_core:stone");
            assert!(stone.sounds.dig.is_some() || stone.sounds.dug.is_some());

            let pick = reg
                .get_item_def("mcl_tools:pick_diamond")
                .expect("mcl_tools:pick_diamond registered");
            let tc = pick
                .tool_capabilities
                .as_ref()
                .expect("diamond pick tool_capabilities");
            assert!(!tc.groupcaps.is_empty());

            let hand = reg.get_item_def("").expect("empty hand item registered");
            assert!(hand.tool_capabilities.is_some());
        }

        // 2. Join player and place + right-click a chest (on_construct + on_rightclick formspec + NodeMetaRef inv)
        let player = Arc::new(RwLock::new(ObjectState::new_player("CoreTester")));
        runtime.join_player(player.clone()).expect("join_player");
        runtime.set_gamemode(player.clone(), "survival");
        {
            let mut p = player.write();
            p.wield_index = 1;
            let mut m = p.meta.write();
            let inv = m.inventories.entry("main".into()).or_insert_with(|| vec![ItemStackData::default(); 36]);
            inv[0] = ItemStackData::from_spec("mcl_chests:chest 1");
            inv[1] = ItemStackData::from_spec("mcl_tools:pick_diamond 1");
        }
        let stone_id = registries.read().get_content_id("mcl_core:stone");
        world.set_node(IVec3::new(0, 10, 0), stone_id, 0, 0);
        let placed = runtime.rightclick_or_place_node_with_sneak(
            IVec3::new(0, 10, 0),
            IVec3::new(0, 11, 0),
            &player,
            false,
        );
        assert!(placed, "placing chest above stone should succeed");
        let chest_id = world.get_node(IVec3::new(0, 11, 0)).0;
        let chest_name = registries.read().get_name_from_content_id(chest_id).to_string();
        assert!(chest_name.contains("chest"), "expected chest node at (0,11,0), got {chest_name}");
        assert!(
            !world.take_priority_dirty_chunks().is_empty(),
            "priority dirty queue should record block placement"
        );
        let _ = runtime.pop_formspecs();
        let opened = runtime.rightclick_or_place_node_with_sneak(
            IVec3::new(0, 11, 0),
            IVec3::new(0, 12, 0),
            &player,
            false,
        );
        assert!(opened, "rightclicking chest should invoke on_rightclick");
        let forms = runtime.pop_formspecs();
        assert!(!forms.is_empty(), "chest rightclick should show formspec");
        assert!(
            world.peek_node_meta(IVec3::new(0, 11, 0)).is_some(),
            "chest should have initialized NodeMetaRef"
        );

        // 3. Dig stone with diamond pickaxe -> clears node to air
        player.write().wield_index = 2;
        let dug = runtime.dig_node(IVec3::new(0, 10, 0), &player);
        assert!(dug, "dig_node with diamond pickaxe should break stone");
        assert_eq!(world.get_node(IVec3::new(0, 10, 0)).0, CONTENT_AIR);
        let step3_drops = runtime.take_spawned_item_drops(0);
        assert_eq!(step3_drops, vec!["mcl_core:cobble".to_string()]);

        // 4. Test Lua core.get_craft_result, core.after + step(), and core.place_schematic
        runtime
            .lua
            .load(
                r#"
                _G.__cubex_after_fired = false
                core.after(0.05, function()
                    _G.__cubex_after_fired = true
                end)
                local res, dec = core.get_craft_result({
                    method = "normal",
                    width = 1,
                    items = { "mcl_trees:wood_oak", "mcl_trees:wood_oak" }
                })
                assert(res and res.item and res.item:get_name() == "mcl_core:stick",
                    "expected mcl_core:stick from two oak planks, got " .. tostring(res and res.item and res.item:get_name()))
                local schem_path = core.get_modpath("mcl_core") .. "/schematics/mcl_core_oak_classic.mts"
                assert(core.place_schematic({x = 20, y = 10, z = 20}, schem_path, "0", nil, true, "place_center_x, place_center_z"))
                "#,
            )
            .exec()
            .expect("Lua runtime gameplay verification script failed");

        runtime.step(0.1);
        let fired: bool = runtime
            .lua
            .globals()
            .get("__cubex_after_fired")
            .unwrap_or(false);
        assert!(fired, "core.after callback should fire during step(0.1)");

        let trunk_id = world.get_node(IVec3::new(20, 12, 20)).0;
        {
            let reg_guard = registries.read();
            let trunk_name = reg_guard.get_name_from_content_id(trunk_id);
            assert_eq!(trunk_name, "mcl_trees:tree_oak");
        }

        // 5. Verify Survival empty inventory, populate_creative_starter, get_node_drops,
        //    Survival DigController drop pickup, 2x2/3x3 crafting, and 3x3 crafting table right-click
        {
            let mut surv = ObjectState::new_player("SurvivalHero");
            assert_eq!(surv.meta.read().inventories["main"].len(), 36);
            assert!(
                surv.meta.read().inventories["main"].iter().all(|s| s.is_empty()),
                "New Survival player must have 36 empty main slots"
            );
            surv.populate_creative_starter();
            {
                let reg_g = registries.read();
                let crea_main = &surv.meta.read().inventories["main"];
                assert!(!crea_main[0].is_empty());
                for slot in crea_main.iter().filter(|s| !s.is_empty()) {
                    assert!(
                        reg_g.get_item_def(&slot.name).is_some(),
                        "Creative starter item '{}' should exist in Mineclonia",
                        slot.name
                    );
                }
            }
        }

        // get_node_drops verification on Mineclonia nodes
        assert_eq!(
            runtime.get_node_drops("mcl_core:stone", Some("mcl_tools:pick_diamond"), None),
            vec!["mcl_core:cobble".to_string()]
        );
        assert_eq!(
            runtime.get_node_drops("mcl_core:dirt_with_grass", None, None),
            vec!["mcl_core:dirt".to_string()]
        );
        assert_eq!(
            runtime.get_node_drops("mcl_trees:tree_oak", None, None),
            vec!["mcl_trees:tree_oak".to_string()]
        );
        assert!(
            runtime.get_node_drops("mcl_core:glass", None, None).is_empty(),
            "explicit drop = \"\" on glass must drop nothing"
        );
        for _ in 0..24 {
            for d in runtime.get_node_drops("mcl_trees:leaves_oak", None, None) {
                assert!(
                    matches!(
                        d.as_str(),
                        "mcl_core:stick 1"
                            | "mcl_core:stick 2"
                            | "mcl_core:stick"
                            | "mcl_trees:sapling_oak"
                            | "mcl_core:apple"
                    ),
                    "Unexpected oak leaves drop: {d}"
                );
            }
        }

        // Survival DigController direct pickup & zero duplicate entities
        let surv_arc = Arc::new(RwLock::new(ObjectState::new_player("MinerSurv")));
        runtime.join_player(surv_arc.clone()).expect("join survival miner");
        assert!(
            surv_arc.read().meta.read().inventories["main"]
                .iter()
                .all(|s| s.is_empty()),
            "join_player must not inject items into Survival main inventory"
        );
        assert!(
            !surv_arc.read().meta.read().inventories["hand"][0].is_empty(),
            "join_player should populate Survival hand inventory for bare-hand harvesting"
        );

        let mut pctrl = crate::physics::PlayerController::default();
        pctrl.gamemode = crate::physics::GameMode::Survival;
        pctrl.pos = Vec3::new(0.5, 10.0, 0.5);
        pctrl.selected_slot = 0;

        let mut digger = crate::physics::DigController::new();
        {
            let reg_g = registries.read();
            let tree_id = reg_g.get_content_id("mcl_trees:tree_oak");
            let grass_id = reg_g.get_content_id("mcl_core:dirt_with_grass");
            let glass_id = reg_g.get_content_id("mcl_core:glass");
            let stone_id2 = reg_g.get_content_id("mcl_core:stone");

            // Break oak log with bare hand
            world.set_node(IVec3::new(0, 10, 2), tree_id, 15, 0);
            let hit = crate::physics::raycast_voxel(
                Vec3::new(0.5, 10.5, 0.5),
                Vec3::Z,
                5.0,
                &world,
                &reg_g,
            )
            .unwrap();
            let mut comp_tree = None;
            for _ in 0..40 {
                let tick = digger.update(0.25, true, Some(&hit), &world, &reg_g, &pctrl, &surv_arc, Some(&runtime));
                if let Some(c) = tick.completed_dig {
                    comp_tree = Some(c);
                    break;
                }
            }
            let comp = comp_tree.expect("oak trunk broken in Survival");
            assert!(comp.picked_up_any);
            assert_eq!(comp.drops, vec!["mcl_trees:tree_oak".to_string()]);
            assert_eq!(surv_arc.read().meta.read().inventories["main"][0].name, "mcl_trees:tree_oak");
            assert_eq!(surv_arc.read().meta.read().inventories["main"][0].count, 1);
            assert!(
                runtime.active_entities().iter().all(|e| e.read().entity_name != "__builtin:item"),
                "no stray item entities should remain after direct pickup"
            );

            // Wait out nodig cooldown
            let _ = digger.update(0.5, false, None, &world, &reg_g, &pctrl, &surv_arc, Some(&runtime));

            // Break grass block with bare hand -> drops mcl_core:dirt
            world.set_node(IVec3::new(0, 10, 2), grass_id, 15, 0);
            let hit_grass = crate::physics::raycast_voxel(
                Vec3::new(0.5, 10.5, 0.5),
                Vec3::Z,
                5.0,
                &world,
                &reg_g,
            )
            .unwrap();
            let mut comp_grass = None;
            for _ in 0..20 {
                let tick = digger.update(0.25, true, Some(&hit_grass), &world, &reg_g, &pctrl, &surv_arc, Some(&runtime));
                if let Some(c) = tick.completed_dig {
                    comp_grass = Some(c);
                    break;
                }
            }
            let comp = comp_grass.expect("grass block broken in Survival");
            assert!(comp.picked_up_any);
            assert_eq!(surv_arc.read().meta.read().inventories["main"][1].name, "mcl_core:dirt");
            let _ = digger.update(0.5, false, None, &world, &reg_g, &pctrl, &surv_arc, Some(&runtime));

            // Break glass (drop = "") -> drops nothing, picked_up_any = false
            world.set_node(IVec3::new(0, 10, 2), glass_id, 15, 0);
            let hit_glass = crate::physics::raycast_voxel(
                Vec3::new(0.5, 10.5, 0.5),
                Vec3::Z,
                5.0,
                &world,
                &reg_g,
            )
            .unwrap();
            let mut comp_glass = None;
            for _ in 0..20 {
                let tick = digger.update(0.25, true, Some(&hit_glass), &world, &reg_g, &pctrl, &surv_arc, Some(&runtime));
                if let Some(c) = tick.completed_dig {
                    comp_glass = Some(c);
                    break;
                }
            }
            let comp = comp_glass.expect("glass broken in Survival");
            assert!(!comp.picked_up_any);
            assert!(comp.drops.is_empty());
            let _ = digger.update(0.5, false, None, &world, &reg_g, &pctrl, &surv_arc, Some(&runtime));

            // Equip stone pickaxe in slot 2 and break stone -> drops mcl_core:cobble + applies wear
            surv_arc.write().meta.write().inventories.get_mut("main").unwrap()[2] =
                ItemStackData::from_spec("mcl_tools:pick_stone 1");
            pctrl.selected_slot = 2;
            world.set_node(IVec3::new(0, 10, 2), stone_id2, 15, 0);
            let hit_stone = crate::physics::raycast_voxel(
                Vec3::new(0.5, 10.5, 0.5),
                Vec3::Z,
                5.0,
                &world,
                &reg_g,
            )
            .unwrap();
            let mut comp_stone = None;
            for _ in 0..20 {
                let tick = digger.update(0.25, true, Some(&hit_stone), &world, &reg_g, &pctrl, &surv_arc, Some(&runtime));
                if let Some(c) = tick.completed_dig {
                    comp_stone = Some(c);
                    break;
                }
            }
            let comp = comp_stone.expect("stone broken with pickaxe");
            assert!(comp.picked_up_any);
            assert_eq!(comp.drops, vec!["mcl_core:cobble".to_string()]);
            assert_eq!(surv_arc.read().meta.read().inventories["main"][3].name, "mcl_core:cobble");
            assert!(
                surv_arc.read().meta.read().inventories["main"][2].wear > 0,
                "stone pickaxe must receive wear in Survival"
            );

            // 2x2 recipes: tree_oak -> wood_oak 4, wood_oak -> stick 4, 2x2 wood_oak -> crafting_table
            let planks_out = resolve_craft_recipe(
                &reg_g,
                &[
                    ItemStackData::from_spec("mcl_trees:tree_oak 1"),
                    ItemStackData::default(),
                    ItemStackData::default(),
                    ItemStackData::default(),
                ],
                2,
                "normal",
            )
            .expect("tree_oak -> wood_oak 4");
            assert_eq!(planks_out.item.name, "mcl_trees:wood_oak");
            assert_eq!(planks_out.item.count, 4);

            let stick_out = resolve_craft_recipe(
                &reg_g,
                &[
                    ItemStackData::from_spec("mcl_trees:wood_oak 2"),
                    ItemStackData::default(),
                    ItemStackData::from_spec("mcl_trees:wood_oak 1"),
                    ItemStackData::default(),
                ],
                2,
                "normal",
            )
            .expect("wood_oak -> stick 4");
            assert_eq!(stick_out.item.name, "mcl_core:stick");
            assert_eq!(stick_out.item.count, 4);
            assert_eq!(stick_out.decrements[0].count, 1);
            assert!(stick_out.decrements[2].is_empty());

            let table_out = resolve_craft_recipe(
                &reg_g,
                &[
                    ItemStackData::from_spec("mcl_trees:wood_oak 1"),
                    ItemStackData::from_spec("mcl_trees:wood_oak 1"),
                    ItemStackData::from_spec("mcl_trees:wood_oak 1"),
                    ItemStackData::from_spec("mcl_trees:wood_oak 1"),
                ],
                2,
                "normal",
            )
            .expect("2x2 planks -> crafting_table");
            assert_eq!(table_out.item.name, "mcl_crafting_table:crafting_table");

            // 3x3 recipes: pick_wood & pick_stone
            let wood_pick_out = resolve_craft_recipe(
                &reg_g,
                &[
                    ItemStackData::from_spec("mcl_trees:wood_oak 1"),
                    ItemStackData::from_spec("mcl_trees:wood_oak 1"),
                    ItemStackData::from_spec("mcl_trees:wood_oak 1"),
                    ItemStackData::default(),
                    ItemStackData::from_spec("mcl_core:stick 1"),
                    ItemStackData::default(),
                    ItemStackData::default(),
                    ItemStackData::from_spec("mcl_core:stick 1"),
                    ItemStackData::default(),
                ],
                3,
                "normal",
            )
            .expect("3x3 pick_wood");
            assert_eq!(wood_pick_out.item.name, "mcl_tools:pick_wood");
            assert!(wood_pick_out.decrements.iter().all(|s| s.is_empty()));

            let stone_pick_out = resolve_craft_recipe(
                &reg_g,
                &[
                    ItemStackData::from_spec("mcl_core:cobble 2"),
                    ItemStackData::from_spec("mcl_core:cobble 1"),
                    ItemStackData::from_spec("mcl_core:cobble 1"),
                    ItemStackData::default(),
                    ItemStackData::from_spec("mcl_core:stick 2"),
                    ItemStackData::default(),
                    ItemStackData::default(),
                    ItemStackData::from_spec("mcl_core:stick 1"),
                    ItemStackData::default(),
                ],
                3,
                "normal",
            )
            .expect("3x3 pick_stone");
            assert_eq!(stone_pick_out.item.name, "mcl_tools:pick_stone");
            assert_eq!(stone_pick_out.decrements[0].count, 1);
            assert!(stone_pick_out.decrements[1].is_empty());
            assert_eq!(stone_pick_out.decrements[4].count, 1);
            assert!(stone_pick_out.decrements[7].is_empty());

            // Crafting table right-click opens 3x3 crafting formspec
            let ct_id = reg_g.get_content_id("mcl_crafting_table:crafting_table");
            world.set_node(IVec3::new(1, 10, 1), ct_id, 15, 0);
        }
        surv_arc.write().pos = Vec3::new(1.0, 10.0, 0.0);
        let _ = runtime.pop_formspecs();
        let ct_opened = runtime.rightclick_or_place_node_with_sneak(
            IVec3::new(1, 10, 1),
            IVec3::new(1, 11, 1),
            &surv_arc,
            false,
        );
        assert!(ct_opened, "rightclicking crafting table should succeed");
        let ct_forms = runtime.pop_formspecs();
        assert!(
            ct_forms.iter().any(|f| f.formspec.contains("list[current_player;craft;2.25,0.75;3,3;]")),
            "crafting table must open 3x3 crafting formspec, got {ct_forms:?}"
        );

        let _ = std::fs::remove_dir_all(&tmp_world);
    }

    #[test]
    fn group_matching_checks_nodedef_and_aliases() {
        let mut reg = EngineRegistries::new();
        reg.register_node(crate::registry::NodeDef {
            name: "test:oak_planks".into(),
            groups: HashMap::from([("wood".into(), 1)]),
            ..Default::default()
        });
        reg.register_node(crate::registry::NodeDef {
            name: "test:cobble_node".into(),
            groups: HashMap::from([("cobble".into(), 1)]),
            ..Default::default()
        });
        reg.register_alias("legacy:wood", "test:oak_planks");
        reg.crafts.push(CraftRecipe {
            recipe_type: "shaped".into(),
            output: "test:table 1".into(),
            recipe: vec![
                vec!["group:wood".into(), "group:wood".into()],
                vec!["group:wood".into(), "group:wood".into()],
            ],
            replacements: Vec::new(),
            cooktime: 0.0,
            burntime: 0.0,
        });
        reg.crafts.push(CraftRecipe {
            recipe_type: "shapeless".into(),
            output: "test:pebble 1".into(),
            recipe: vec![vec!["group:cobble".into()]],
            replacements: Vec::new(),
            cooktime: 0.0,
            burntime: 0.0,
        });

        let grid = [
            ItemStackData::from_spec("legacy:wood 1"),
            ItemStackData::from_spec("test:oak_planks 1"),
            ItemStackData::from_spec("test:oak_planks 1"),
            ItemStackData::from_spec("legacy:wood 1"),
        ];
        let res = resolve_craft_recipe(&reg, &grid, 2, "normal").expect("node group + alias match");
        assert_eq!(res.item.name, "test:table");

        let sl = resolve_craft_recipe(&reg, &[ItemStackData::from_spec("test:cobble_node 1")], 2, "normal")
            .expect("shapeless nodedef group match");
        assert_eq!(sl.item.name, "test:pebble");
    }

    #[test]
    fn incremental_lua_boot_matches_full_boot() {
        let paths = EnginePaths::resolve();
        if !paths.game_dir.join("mods").is_dir() {
            return;
        }
        let tmp_world = std::env::temp_dir().join("cubex_incremental_lua_boot_test");
        let _ = std::fs::remove_dir_all(&tmp_world);
        let _ = std::fs::create_dir_all(&tmp_world);

        let reg = Arc::new(RwLock::new(EngineRegistries::new()));
        let world = Arc::new(WorldStore::new(&tmp_world, 1337));
        let cfg = Arc::new(RwLock::new(MinetestConfig::load(&tmp_world.join("minetest.conf"))));
        let ov = if paths.lua_override_dir.is_dir() {
            Some(paths.lua_override_dir.as_path())
        } else {
            None
        };
        let mods = discover_and_sort_mods(&paths.game_dir, ov);
        let mut inc = LuaIncrementalBoot::begin(paths, reg.clone(), world, cfg, mods)
            .expect("incremental boot begin");

        let mut steps = 0usize;
        while !inc.is_done() {
            let _ = inc.step_budget(std::time::Duration::from_millis(12));
            steps += 1;
        }
        assert!(steps >= 2, "expected multi-slice incremental steps, got {steps}");
        let rt = inc.finish();
        assert!(rt.load_report.mods_loaded > 200);
        assert!(
            rt.load_report.errors.is_empty(),
            "unexpected incremental mod load errors: {:?}",
            rt.load_report.errors
        );
        assert_ne!(
            reg.read().get_content_id("mcl_core:dirt_with_grass"),
            CONTENT_AIR
        );
        let _ = std::fs::remove_dir_all(&tmp_world);
    }

    #[test]
    fn player_death_drops_once_clears_inventory_blocks_damage_and_respawns() {
        let tmp_world = std::env::temp_dir().join("cubex_death_dupe_lock_test");
        let _ = std::fs::remove_dir_all(&tmp_world);
        let _ = std::fs::create_dir_all(&tmp_world);

        let paths = EnginePaths::resolve();
        let reg = Arc::new(RwLock::new(EngineRegistries::new()));
        let world = Arc::new(WorldStore::new(&tmp_world, 2026));
        let cfg = Arc::new(RwLock::new(MinetestConfig::load(&tmp_world.join("minetest.conf"))));
        let rt = LuaIncrementalBoot::begin(paths, reg, world.clone(), cfg, Vec::new())
            .expect("minimal boot")
            .finish();

        let p_arc = Arc::new(RwLock::new(ObjectState::new_player("Victim")));
        {
            let p = p_arc.read();
            let mut m = p.meta.write();
            m.inventories.get_mut("main").unwrap()[0] =
                ItemStackData::from_spec("mcl_core:diamond 5");
            m.inventories.get_mut("armor").unwrap()[0] =
                ItemStackData::from_spec("mcl_armor:chestplate_diamond 1");
            m.inventories.get_mut("offhand").unwrap()[0] =
                ItemStackData::from_spec("mcl_torches:torch 12");
            m.inventories.get_mut("craft").unwrap()[0] =
                ItemStackData::from_spec("mcl_core:stone 3");
        }
        rt.join_player(p_arc.clone()).expect("join Victim");

        rt.lua
            .load(
                r#"
                _G._test_die_calls = 0
                core.register_on_dieplayer(function(player, reason)
                    _G._test_die_calls = _G._test_die_calls + 1
                end)
                "#,
            )
            .exec()
            .expect("register die hook");

        // Default keep_inventory = true preserves items on death, suppresses nested in_wall, and locks is_dead
        rt.lua
            .load(
                r#"
                local p = core.get_player_by_name("Victim")
                p:set_hp(10, { type = "set_hp", _mcl_reason = { type = "in_wall" } })
                assert(p:get_hp() == 20, "nested _mcl_reason.in_wall must be suppressed")
                p:set_hp(0, { type = "punch" })
                -- Second lethal call while dead must be ignored (no on_die, no dupe)
                p:set_hp(0, { type = "fall" })
                p:punch(p, 1.0, { damage_groups = { fleshy = 10 } }, {x=1,y=0,z=0})
                "#,
            )
            .exec()
            .expect("trigger lethal damage with keep_inventory");

        {
            let g = p_arc.read();
            assert!(g.is_dead, "player must enter is_dead after lethal damage");
            assert!(g.has_dropped_death_items, "death guard must be set");
            assert_eq!(g.hp, 0);
            assert_eq!(g.last_death_cause, "Slain in combat");
            let m = g.meta.read();
            assert_eq!(m.inventories["main"][0].name, "mcl_core:diamond");
            assert_eq!(m.inventories["main"][0].count, 5);
            assert_eq!(m.inventories["armor"][0].name, "mcl_armor:chestplate_diamond");
            assert_eq!(m.inventories["offhand"][0].name, "mcl_torches:torch");
            assert_eq!(m.inventories["craft"][0].name, "mcl_core:stone");
        }
        assert_eq!(
            rt.lua.globals().get::<i64>("_test_die_calls").unwrap(),
            1,
            "on_dieplayer must fire strictly once per death"
        );

        // Respawn resets state cleanly and keeps inventory intact
        {
            let mut g = p_arc.write();
            g.hp = 20;
            g.hunger = 20;
            g.is_dead = false;
            g.has_dropped_death_items = false;
            g.spawn_invuln_timer = 2.0;
            g.last_death_cause.clear();
        }
        rt.notify_player_respawned(&p_arc);

        // Damage during 2.0s spawn invuln must be ignored
        rt.lua
            .load(
                r#"
                local p = core.get_player_by_name("Victim")
                p:set_hp(5, { type = "punch" })
                "#,
            )
            .exec()
            .expect("invuln damage check");
        assert_eq!(
            p_arc.read().hp,
            20,
            "spawn invulnerability must protect freshly respawned player"
        );

        // Explicit mcl_keepInventory = false drops items once and clears inventory without dupes
        {
            let mut g = p_arc.write();
            g.spawn_invuln_timer = 0.0;
        }
        rt.lua
            .load(
                r#"
                core.settings:set_bool("mcl_keepInventory", false)
                local p = core.get_player_by_name("Victim")
                p:set_hp(0, { type = "punch" })
                p:set_hp(0, { type = "fall" })
                "#,
            )
            .exec()
            .expect("lethal damage with keepInventory=false");

        {
            let g = p_arc.read();
            assert!(g.is_dead);
            assert!(g.has_dropped_death_items);
            let m = g.meta.read();
            for list_name in ["main", "armor", "offhand", "craft"] {
                assert!(
                    m.inventories[list_name].iter().all(|s| s.is_empty()),
                    "inventory list {list_name} must be cleared when keepInventory=false"
                );
            }
        }
        assert_eq!(rt.lua.globals().get::<i64>("_test_die_calls").unwrap(), 2);

        // Second call to drain_player_inventory_for_death even with items re-inserted must yield empty
        {
            p_arc.read().meta.write().inventories.get_mut("main").unwrap()[0] =
                ItemStackData::from_spec("mcl_core:dirt 1");
        }
        let dupe_check = drain_player_inventory_for_death(&p_arc, false);
        assert!(
            dupe_check.is_empty(),
            "has_dropped_death_items flag must block duplicate drops"
        );

        // Verify InvRef:set_list("craft", {}) clears trailing slots cleanly
        {
            let m = p_arc.read().meta.clone();
            m.write().inventories.get_mut("craft").unwrap()[1] =
                ItemStackData::from_spec("mcl_core:dirt 4");
        }
        rt.lua
            .load(
                r#"
                local p = core.get_player_by_name("Victim")
                p:get_inventory():set_list("craft", {})
                "#,
            )
            .exec()
            .expect("set_list empty clears trailing items");
        assert!(
            p_arc.read().meta.read().inventories["craft"]
                .iter()
                .all(|s| s.is_empty()),
            "InvRef:set_list(list, {{}}) must clear all trailing items"
        );

        let _ = std::fs::remove_dir_all(&tmp_world);
    }

    #[test]
    fn falling_node_falls_without_bounce_snaps_to_grid_and_resolidifies() {
        let tmp_world = std::env::temp_dir().join("cubex_falling_node_resolidify_test");
        let _ = std::fs::remove_dir_all(&tmp_world);
        let _ = std::fs::create_dir_all(&tmp_world);

        let paths = EnginePaths::resolve();
        let reg = Arc::new(RwLock::new(EngineRegistries::new()));
        let (stone, sand) = {
            let mut r = reg.write();
            let s = r.register_node(crate::registry::NodeDef {
                name: "mcl_core:stone".into(),
                drawtype: DrawType::Normal,
                walkable: true,
                groups: HashMap::from([("bouncy".into(), 60)]),
                ..Default::default()
            });
            let sa = r.register_node(crate::registry::NodeDef {
                name: "mcl_core:sand".into(),
                drawtype: DrawType::Normal,
                walkable: true,
                ..Default::default()
            });
            (s, sa)
        };
        let world = Arc::new(WorldStore::new(&tmp_world, 777));
        // Solid stone floor at y = 10 across a 3x3 patch (even with bouncy group, sand must not bounce!)
        for dz in -1..=1 {
            for dx in -1..=1 {
                world.set_node(IVec3::new(dx, 10, dz), stone, 15, 0);
            }
        }
        let cfg = Arc::new(RwLock::new(MinetestConfig::load(&tmp_world.join("minetest.conf"))));
        let rt = LuaIncrementalBoot::begin(paths, reg.clone(), world.clone(), cfg, Vec::new())
            .expect("minimal boot")
            .finish();

        rt.lua
            .load(
                r#"
                core.register_entity(":__builtin:falling_node", {
                    initial_properties = {
                        visual = "wielditem",
                        textures = {},
                        physical = true,
                        collisionbox = {-0.5, -0.5, -0.5, 0.5, 0.5, 0.5},
                    },
                    node = { name = "mcl_core:sand", param2 = 0 },
                })
                local obj = core.add_entity({x = 0.0, y = 13.0, z = 0.0}, "__builtin:falling_node")
                assert(obj ~= nil, "falling_node entity must spawn")
                "#,
            )
            .exec()
            .expect("spawn falling sand");

        assert_eq!(rt.shared.entities.read().len(), 1);
        assert_eq!(world.get_node(IVec3::new(0, 11, 0)).0, CONTENT_AIR);

        // Step in 20ms increments up to 1.5s; verify downward vel never flips positive (no bounce) and sand places cleanly
        let mut landed = false;
        let mut max_vy = 0.0f32;
        for _ in 0..75 {
            rt.step(0.02);
            let ents = rt.shared.entities.read();
            if let Some(e) = ents.values().next() {
                max_vy = max_vy.max(e.read().vel.y);
            } else {
                landed = true;
                break;
            }
        }
        assert!(landed, "falling_node entity must despawn upon touching ground");
        assert!(max_vy <= 0.0, "falling_node must have zero bounce, saw vel.y = {max_vy}");
        assert_eq!(
            world.get_node(IVec3::new(0, 11, 0)).0,
            sand,
            "falling sand must re-solidify directly on top of stone floor"
        );
        assert!(
            !world.priority_dirty_queue.read().is_empty(),
            "chunk containing re-solidified sand must be queued for priority remesh"
        );

        let _ = std::fs::remove_dir_all(&tmp_world);
    }
}
