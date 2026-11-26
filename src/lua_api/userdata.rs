use glam::{IVec3, Vec3};
use mlua::{
    AnyUserData, Lua, MetaMethod, MultiValue, Result as LuaResult, Table, UserData,
    UserDataMethods, Value,
};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

use crate::config::MinetestConfig;
use crate::registry::{EngineRegistries, CONTENT_AIR, CONTENT_IGNORE};
use crate::world::WorldStore;
pub use crate::world::{ItemStackData, MetaDataState, MetaStore};

/// Shared settings userdata wrapping MinetestConfig.
#[derive(Clone)]
pub struct SettingsUD(pub Arc<RwLock<MinetestConfig>>);

impl UserData for SettingsUD {
    fn add_methods<M: UserDataMethods<Self>>(methods: &mut M) {
        methods.add_method("get", |_, this, key: String| {
            Ok(this.0.read().get(&key).map(ToString::to_string))
        });
        methods.add_method("get_bool", |_, this, (key, def): (String, Option<bool>)| {
            let guard = this.0.read();
            if guard.get(&key).is_none() {
                return Ok(def);
            }
            Ok(Some(guard.get_bool(&key, def.unwrap_or(false))))
        });
        methods.add_method("get_np_group", |lua, this, key: String| {
            let raw = this.0.read().get(&key).map(ToString::to_string);
            let Some(val) = raw else {
                return Ok(Value::Nil);
            };
            let tbl = lua.create_table()?;
            let parts: Vec<&str> = val.split(',').map(str::trim).collect();
            if parts.len() >= 2 {
                if let Ok(off) = parts[0].parse::<f64>() { tbl.set("offset", off)?; }
                if let Ok(sc) = parts[1].parse::<f64>() { tbl.set("scale", sc)?; }
            }
            if parts.len() >= 5 {
                let sp = lua.create_table()?;
                sp.set("x", parts[2].trim_start_matches('(').parse::<f64>().unwrap_or(250.0))?;
                sp.set("y", parts[3].parse::<f64>().unwrap_or(250.0))?;
                sp.set("z", parts[4].trim_end_matches(')').parse::<f64>().unwrap_or(250.0))?;
                tbl.set("spread", sp)?;
            }
            tbl.set("seed", parts.get(5).and_then(|s| s.parse::<i64>().ok()).unwrap_or(0))?;
            tbl.set("octaves", parts.get(6).and_then(|s| s.parse::<i64>().ok()).unwrap_or(3))?;
            tbl.set("persist", parts.get(7).and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.6))?;
            tbl.set("lacunarity", parts.get(8).and_then(|s| s.parse::<f64>().ok()).unwrap_or(2.0))?;
            Ok(Value::Table(tbl))
        });
        methods.add_method("get_flags", |lua, this, key: String| {
            let raw = this.0.read().get(&key).map(ToString::to_string);
            let Some(val) = raw else {
                return Ok(Value::Nil);
            };
            let tbl = lua.create_table()?;
            for tok in val.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                if let Some(rest) = tok.strip_prefix("no") {
                    tbl.set(rest, false)?;
                } else {
                    tbl.set(tok, true)?;
                }
            }
            Ok(Value::Table(tbl))
        });
        methods.add_method("set", |_, this, (key, val): (String, Value)| {
            let s = match val {
                Value::String(st) => st.to_str()?.to_string(),
                Value::Integer(i) => i.to_string(),
                Value::Number(n) => n.to_string(),
                Value::Boolean(b) => if b { "true" } else { "false" }.to_string(),
                _ => String::new(),
            };
            this.0.write().set(key, s);
            Ok(())
        });
        methods.add_method("set_bool", |_, this, (key, b): (String, bool)| {
            this.0.write().set(key, b);
            Ok(())
        });
        methods.add_method("set_np_group", |_, this, (key, val): (String, Table)| {
            let off: f64 = val.get("offset").unwrap_or(0.0);
            let sc: f64 = val.get("scale").unwrap_or(1.0);
            this.0.write().set(key, format!("{off},{sc},(250,250,250),0,3,0.6,2.0"));
            Ok(())
        });
        methods.add_method("remove", |_, this, key: String| {
            let existed = this.0.read().get(&key).is_some();
            this.0.write().remove(&key);
            Ok(existed)
        });
        methods.add_method("has", |_, this, key: String| {
            Ok(this.0.read().get(&key).is_some())
        });
        methods.add_method("get_names", |lua, this, ()| {
            let tbl = lua.create_table()?;
            for (i, k) in this.0.read().values.keys().enumerate() {
                tbl.set(i + 1, k.as_str())?;
            }
            Ok(tbl)
        });
        methods.add_method("to_table", |lua, this, ()| {
            let tbl = lua.create_table()?;
            for (k, v) in &this.0.read().values {
                tbl.set(k.as_str(), v.as_str())?;
            }
            Ok(tbl)
        });
        methods.add_method("write", |_, this, ()| {
            let _ = this.0.read().save();
            Ok(true)
        });
    }
}

#[derive(Clone)]
pub struct MetaDataRefUD(pub Arc<RwLock<MetaDataState>>);

impl UserData for MetaDataRefUD {
    fn add_methods<M: UserDataMethods<Self>>(methods: &mut M) {
        methods.add_method("contains", |_, this, key: String| {
            Ok(this.0.read().fields.contains_key(&key))
        });
        methods.add_method("get", |_, this, key: String| {
            Ok(this.0.read().fields.get(&key).cloned())
        });
        methods.add_method("get_string", |_, this, key: String| {
            Ok(this.0.read().fields.get(&key).cloned().unwrap_or_default())
        });
        methods.add_method("set_string", |_, this, (key, val): (String, Value)| {
            let s = match val {
                Value::Nil => String::new(),
                Value::String(st) => st.to_str()?.to_string(),
                Value::Integer(i) => i.to_string(),
                Value::Number(n) => n.to_string(),
                Value::Boolean(b) => if b { "true" } else { "false" }.to_string(),
                _ => String::new(),
            };
            if s.is_empty() {
                this.0.write().fields.remove(&key);
            } else {
                this.0.write().fields.insert(key, s);
            }
            Ok(())
        });
        methods.add_method("get_int", |_, this, key: String| {
            Ok(this
                .0
                .read()
                .fields
                .get(&key)
                .and_then(|s| s.parse::<i64>().ok())
                .unwrap_or(0))
        });
        methods.add_method("set_int", |_, this, (key, val): (String, i64)| {
            this.0.write().fields.insert(key, val.to_string());
            Ok(())
        });
        methods.add_method("get_float", |_, this, key: String| {
            Ok(this
                .0
                .read()
                .fields
                .get(&key)
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(0.0))
        });
        methods.add_method("set_float", |_, this, (key, val): (String, f64)| {
            this.0.write().fields.insert(key, val.to_string());
            Ok(())
        });
        methods.add_method("get_keys", |lua, this, ()| {
            let tbl = lua.create_table()?;
            for (i, k) in this.0.read().fields.keys().enumerate() {
                tbl.set(i + 1, k.as_str())?;
            }
            Ok(tbl)
        });
        methods.add_method("to_table", |lua, this, ()| {
            let g = this.0.read();
            let root = lua.create_table()?;
            let fields = lua.create_table()?;
            for (k, v) in &g.fields {
                fields.set(k.as_str(), v.as_str())?;
            }
            root.set("fields", fields)?;
            let inv = lua.create_table()?;
            for (lname, vec) in &g.inventories {
                let list_t = lua.create_table_with_capacity(vec.len(), 0)?;
                for (i, item) in vec.iter().enumerate() {
                    list_t.set(i + 1, ItemStackUD(Arc::new(RwLock::new(item.clone()))))?;
                }
                inv.set(lname.as_str(), list_t)?;
            }
            root.set("inventory", inv)?;
            Ok(root)
        });
        methods.add_method("from_table", |_, this, tbl: Option<Table>| {
            let mut g = this.0.write();
            g.fields.clear();
            g.inventories.clear();
            if let Some(t) = tbl {
                if let Ok(fields) = t.get::<Table>("fields") {
                    for pair in fields.pairs::<String, Value>() {
                        if let Ok((k, v)) = pair {
                            let s = match v {
                                Value::String(st) => st.to_str()?.to_string(),
                                Value::Integer(i) => i.to_string(),
                                Value::Number(n) => n.to_string(),
                                _ => continue,
                            };
                            g.fields.insert(k, s);
                        }
                    }
                }
                if let Ok(inv) = t.get::<Table>("inventory") {
                    for pair in inv.pairs::<String, Table>() {
                        if let Ok((lname, list_t)) = pair {
                            let len = list_t.len().unwrap_or(0) as usize;
                            let mut vec = Vec::with_capacity(len);
                            for i in 1..=len {
                                if let Ok(v) = list_t.get::<Value>(i) {
                                    let st = ItemStackUD::from_lua_value(v)?;
                                    vec.push(st.0.read().clone());
                                }
                            }
                            g.inventories.insert(lname, vec);
                        }
                    }
                }
            }
            Ok(true)
        });
        methods.add_method("get_inventory", |_, this, ()| {
            Ok(InvRefUD {
                state: this.0.clone(),
                location: "node".into(),
            })
        });
        methods.add_method("mark_as_private", |_, _, _: Value| Ok(()));
        methods.add_method("set_tool_capabilities", |_, this, val: Value| {
            if matches!(val, Value::Nil) {
                this.0.write().fields.remove("__tool_capabilities");
            } else {
                let jv = super::core_api::lua_value_to_json(val);
                this.0
                    .write()
                    .fields
                    .insert("__tool_capabilities".into(), jv.to_string());
            }
            Ok(())
        });
        methods.add_method("set_wear_bar_params", |_, _, _: Value| Ok(()));
        methods.add_method("equals", |_, this, other: AnyUserData| {
            if let Ok(o) = other.borrow::<MetaDataRefUD>() {
                Ok(this.0.read().fields == o.0.read().fields)
            } else {
                Ok(false)
            }
        });
    }
}

#[derive(Clone)]
pub struct ItemStackUD(pub Arc<RwLock<ItemStackData>>);

impl ItemStackUD {
    pub fn from_lua_value(val: Value) -> LuaResult<Self> {
        match val {
            Value::Nil => Ok(Self(Arc::new(RwLock::new(ItemStackData::default())))),
            Value::String(s) => {
                let st = s.to_str()?;
                Ok(Self(Arc::new(RwLock::new(ItemStackData::from_spec(&st)))))
            }
            Value::UserData(ud) => {
                if let Ok(stack) = ud.borrow::<ItemStackUD>() {
                    Ok(Self(Arc::new(RwLock::new(stack.0.read().clone()))))
                } else {
                    Ok(Self(Arc::new(RwLock::new(ItemStackData::default()))))
                }
            }
            Value::Table(t) => {
                let name: String = t.get("name").unwrap_or_default();
                let count: u16 = t.get("count").unwrap_or(if name.is_empty() { 0 } else { 1 });
                let wear: u16 = t.get("wear").unwrap_or(0);
                Ok(Self(Arc::new(RwLock::new(ItemStackData {
                    name,
                    count,
                    wear,
                    meta: Arc::new(RwLock::new(MetaDataState::default())),
                }))))
            }
            _ => Ok(Self(Arc::new(RwLock::new(ItemStackData::default())))),
        }
    }
}

impl UserData for ItemStackUD {
    fn add_methods<M: UserDataMethods<Self>>(methods: &mut M) {
        methods.add_method("is_empty", |_, this, ()| Ok(this.0.read().is_empty()));
        methods.add_method("get_name", |_, this, ()| Ok(this.0.read().name.clone()));
        methods.add_method("set_name", |_, this, name: Option<String>| {
            let mut g = this.0.write();
            if let Some(n) = name {
                g.name = n;
                if !g.name.is_empty() && g.count == 0 {
                    g.count = 1;
                }
            } else {
                g.name.clear();
                g.count = 0;
            }
            Ok(true)
        });
        methods.add_method("get_count", |_, this, ()| Ok(this.0.read().count));
        methods.add_method("set_count", |_, this, count: u16| {
            let mut g = this.0.write();
            g.count = count;
            if count == 0 {
                g.name.clear();
            }
            Ok(true)
        });
        methods.add_method("get_wear", |_, this, ()| Ok(this.0.read().wear));
        methods.add_method("set_wear", |_, this, wear: u16| {
            this.0.write().wear = wear;
            Ok(true)
        });
        methods.add_method("get_meta", |_, this, ()| Ok(MetaDataRefUD(this.0.read().meta.clone())));
        methods.add_method("get_metadata", |_, this, ()| {
            Ok(this.0.read().meta.read().fields.get("").cloned().unwrap_or_default())
        });
        methods.add_method("set_metadata", |_, this, s: String| {
            this.0.read().meta.write().fields.insert(String::new(), s);
            Ok(true)
        });
        methods.add_method("get_description", |lua, this, ()| {
            let name = this.0.read().name.clone();
            if let Ok(core) = lua.globals().get::<Table>("core") {
                if let Ok(items) = core.get::<Table>("registered_items") {
                    if let Ok(def) = items.get::<Table>(name.as_str()) {
                        if let Ok(desc) = def.get::<String>("description") {
                            return Ok(desc);
                        }
                    }
                }
            }
            Ok(name)
        });
        methods.add_method("get_short_description", |lua, this, ()| {
            let name = this.0.read().name.clone();
            if let Ok(core) = lua.globals().get::<Table>("core") {
                if let Ok(items) = core.get::<Table>("registered_items") {
                    if let Ok(def) = items.get::<Table>(name.as_str()) {
                        if let Ok(desc) = def.get::<String>("short_description") {
                            return Ok(desc);
                        }
                        if let Ok(desc) = def.get::<String>("description") {
                            return Ok(desc.lines().next().unwrap_or("").to_string());
                        }
                    }
                }
            }
            Ok(name)
        });
        methods.add_method("clear", |_, this, ()| {
            let mut g = this.0.write();
            g.name.clear();
            g.count = 0;
            g.wear = 0;
            Ok(())
        });
        methods.add_method("replace", |_, this, val: Value| {
            let other = ItemStackUD::from_lua_value(val)?;
            *this.0.write() = other.0.read().clone();
            Ok(())
        });
        methods.add_method("to_string", |_, this, ()| Ok(this.0.read().to_itemstring()));
        methods.add_method("to_table", |lua, this, ()| {
            let g = this.0.read();
            if g.is_empty() {
                return Ok(Value::Nil);
            }
            let tbl = lua.create_table()?;
            tbl.set("name", g.name.as_str())?;
            tbl.set("count", g.count)?;
            tbl.set("wear", g.wear)?;
            Ok(Value::Table(tbl))
        });
        methods.add_method("get_stack_max", |lua, this, ()| {
            let name = this.0.read().name.clone();
            if let Ok(core) = lua.globals().get::<Table>("core") {
                if let Ok(items) = core.get::<Table>("registered_items") {
                    if let Ok(def) = items.get::<Table>(name.as_str()) {
                        if let Ok(max) = def.get::<u16>("stack_max") {
                            return Ok(max);
                        }
                    }
                }
            }
            Ok(64u16)
        });
        methods.add_method("get_free_space", |lua, this, ()| {
            let (name, c) = {
                let g = this.0.read();
                (g.name.clone(), g.count)
            };
            let mut max_s = 64u16;
            if let Ok(core) = lua.globals().get::<Table>("core") {
                if let Ok(items) = core.get::<Table>("registered_items") {
                    if let Ok(def) = items.get::<Table>(name.as_str()) {
                        if let Ok(m) = def.get::<u16>("stack_max") {
                            max_s = m.max(1);
                        }
                    }
                }
            }
            Ok(max_s.saturating_sub(c))
        });
        methods.add_method("is_known", |lua, this, ()| {
            let name = this.0.read().name.clone();
            if let Ok(core) = lua.globals().get::<Table>("core") {
                if let Ok(items) = core.get::<Table>("registered_items") {
                    return Ok(items.get::<Value>(name.as_str()).map(|v| !matches!(v, Value::Nil)).unwrap_or(false));
                }
            }
            Ok(false)
        });
        methods.add_method("get_definition", |lua, this, ()| {
            let name = this.0.read().name.clone();
            if let Ok(core) = lua.globals().get::<Table>("core") {
                if let Ok(items) = core.get::<Table>("registered_items") {
                    if let Ok(v) = items.get::<Value>(name.as_str()) {
                        if !matches!(v, Value::Nil) {
                            return Ok(v);
                        }
                    }
                }
            }
            Ok(Value::Table(lua.create_table()?))
        });
        methods.add_method("get_tool_capabilities", |lua, this, ()| {
            let (name, meta_arc) = {
                let g = this.0.read();
                (g.name.clone(), g.meta.clone())
            };
            if let Some(raw_tc) = meta_arc.read().fields.get("__tool_capabilities").cloned() {
                if let Ok(jv) = serde_json::from_str::<serde_json::Value>(&raw_tc) {
                    if let Ok(Value::Table(tbl)) =
                        super::core_api::json_to_lua_value(lua, &jv, &Value::Nil)
                    {
                        return Ok(tbl);
                    }
                }
            }
            if let Ok(core) = lua.globals().get::<Table>("core") {
                if let Ok(items) = core.get::<Table>("registered_items") {
                    if let Ok(def) = items.get::<Table>(name.as_str()) {
                        if let Ok(tc) = def.get::<Table>("tool_capabilities") {
                            return Ok(tc);
                        }
                    }
                }
            }
            let tbl = lua.create_table()?;
            tbl.set("full_punch_interval", 1.4)?;
            tbl.set("max_drop_level", 0)?;
            tbl.set("groupcaps", lua.create_table()?)?;
            tbl.set("damage_groups", lua.create_table()?)?;
            Ok(tbl)
        });
        methods.add_method("add_wear", |_, this, amount: f64| {
            let amt = amount.round() as i32;
            let mut g = this.0.write();
            let nw = (g.wear as i32 + amt).clamp(0, 65536);
            if nw >= 65536 {
                g.name.clear();
                g.count = 0;
                g.wear = 0;
            } else {
                g.wear = nw as u16;
            }
            Ok(())
        });
        methods.add_method("add_wear_by_uses", |_, this, max_uses: u32| {
            if max_uses > 0 {
                let step = (65536u32 / max_uses).max(1) as i32;
                let mut g = this.0.write();
                let nw = g.wear as i32 + step;
                if nw >= 65536 {
                    g.name.clear();
                    g.count = 0;
                    g.wear = 0;
                } else {
                    g.wear = nw as u16;
                }
            }
            Ok(())
        });
        methods.add_method("add_item", |_, this, val: Value| {
            let add = ItemStackUD::from_lua_value(val)?;
            let mut g = this.0.write();
            let mut ag = add.0.write();
            if ag.is_empty() {
                return Ok(ItemStackUD(Arc::new(RwLock::new(ItemStackData::default()))));
            }
            if g.is_empty() {
                *g = ag.clone();
                ag.name.clear();
                ag.count = 0;
            } else if g.name == ag.name {
                let space = 64u16.saturating_sub(g.count);
                let move_n = space.min(ag.count);
                g.count += move_n;
                ag.count -= move_n;
                if ag.count == 0 {
                    ag.name.clear();
                }
            }
            let rem = ag.clone();
            drop(ag);
            Ok(ItemStackUD(Arc::new(RwLock::new(rem))))
        });
        methods.add_method("item_fits", |_, this, val: Value| {
            let add = ItemStackUD::from_lua_value(val)?;
            let g = this.0.read();
            let ag = add.0.read();
            let fits = g.is_empty() || (g.name == ag.name && g.count + ag.count <= 64);
            Ok((fits, ItemStackUD(Arc::new(RwLock::new(ItemStackData::default())))))
        });
        methods.add_method("take_item", |_, this, n: Option<u16>| {
            let req = n.unwrap_or(1);
            let mut g = this.0.write();
            let take = req.min(g.count);
            let taken = if take > 0 && !g.is_empty() {
                ItemStackData {
                    name: g.name.clone(),
                    count: take,
                    wear: g.wear,
                    meta: g.meta.clone(),
                }
            } else {
                ItemStackData::default()
            };
            g.count = g.count.saturating_sub(take);
            if g.count == 0 {
                g.name.clear();
            }
            Ok(ItemStackUD(Arc::new(RwLock::new(taken))))
        });
        methods.add_method("peek_item", |_, this, n: Option<u16>| {
            let req = n.unwrap_or(1);
            let g = this.0.read();
            let take = req.min(g.count);
            let peek = if take > 0 && !g.is_empty() {
                ItemStackData {
                    name: g.name.clone(),
                    count: take,
                    wear: g.wear,
                    meta: g.meta.clone(),
                }
            } else {
                ItemStackData::default()
            };
            Ok(ItemStackUD(Arc::new(RwLock::new(peek))))
        });
        methods.add_method("equals", |_, this, other: AnyUserData| {
            if let Ok(o) = other.borrow::<ItemStackUD>() {
                let a = this.0.read();
                let b = o.0.read();
                Ok(a.name == b.name && a.count == b.count && a.wear == b.wear)
            } else {
                Ok(false)
            }
        });
        methods.add_meta_method(MetaMethod::ToString, |_, this, ()| {
            Ok(this.0.read().to_itemstring())
        });
    }
}

#[derive(Clone)]
pub struct InvRefUD {
    pub state: Arc<RwLock<MetaDataState>>,
    pub location: String,
}

impl UserData for InvRefUD {
    fn add_methods<M: UserDataMethods<Self>>(methods: &mut M) {
        methods.add_method("is_empty", |_, this, list: String| {
            let g = this.state.read();
            Ok(g.inventories
                .get(&list)
                .map(|v| v.iter().all(ItemStackData::is_empty))
                .unwrap_or(true))
        });
        methods.add_method("get_size", |_, this, list: String| {
            Ok(this.state.read().inventories.get(&list).map(Vec::len).unwrap_or(0))
        });
        methods.add_method("set_size", |_, this, (list, sz): (String, usize)| {
            let mut g = this.state.write();
            let entry = g.inventories.entry(list).or_default();
            entry.resize_with(sz, ItemStackData::default);
            Ok(true)
        });
        methods.add_method("get_width", |_, this, list: String| {
            if list == "craft" {
                let len = this
                    .state
                    .read()
                    .inventories
                    .get(&list)
                    .map(Vec::len)
                    .unwrap_or(9);
                Ok(if len == 4 { 2usize } else { 3usize })
            } else {
                Ok(9usize)
            }
        });
        methods.add_method("set_width", |_, _, (_list, _w): (String, usize)| Ok(true));
        methods.add_method("get_stack", |_, this, (list, idx): (String, usize)| {
            let g = this.state.read();
            let item = if idx >= 1 {
                g.inventories
                    .get(&list)
                    .and_then(|v| v.get(idx - 1))
                    .cloned()
                    .unwrap_or_default()
            } else {
                ItemStackData::default()
            };
            Ok(ItemStackUD(Arc::new(RwLock::new(item))))
        });
        methods.add_method("set_stack", |_, this, (list, idx, val): (String, usize, Value)| {
            let stack = ItemStackUD::from_lua_value(val)?;
            let mut g = this.state.write();
            let entry = g.inventories.entry(list).or_default();
            if idx >= 1 && idx <= entry.len() {
                entry[idx - 1] = stack.0.read().clone();
            }
            Ok(true)
        });
        methods.add_method("get_list", |lua, this, list: String| {
            let g = this.state.read();
            let Some(vec) = g.inventories.get(&list) else {
                return Ok(Value::Nil);
            };
            let tbl = lua.create_table_with_capacity(vec.len(), 0)?;
            for (i, item) in vec.iter().enumerate() {
                tbl.set(i + 1, ItemStackUD(Arc::new(RwLock::new(item.clone()))))?;
            }
            Ok(Value::Table(tbl))
        });
        methods.add_method("set_list", |_, this, (list, tbl): (String, Table)| {
            let mut g = this.state.write();
            let entry = g.inventories.entry(list).or_default();
            let len = tbl.len()? as usize;
            if entry.len() < len {
                entry.resize_with(len, ItemStackData::default);
            }
            for i in 1..=len {
                let v: Value = tbl.get(i)?;
                let st = ItemStackUD::from_lua_value(v)?;
                entry[i - 1] = st.0.read().clone();
            }
            for slot in entry.iter_mut().skip(len) {
                *slot = ItemStackData::default();
            }
            Ok(())
        });
        methods.add_method("get_lists", |lua, this, ()| {
            let g = this.state.read();
            let out = lua.create_table()?;
            for (name, vec) in &g.inventories {
                let tbl = lua.create_table_with_capacity(vec.len(), 0)?;
                for (i, item) in vec.iter().enumerate() {
                    tbl.set(i + 1, ItemStackUD(Arc::new(RwLock::new(item.clone()))))?;
                }
                out.set(name.as_str(), tbl)?;
            }
            Ok(out)
        });
        methods.add_method("set_lists", |_, this, tbl: Table| {
            let mut g = this.state.write();
            for pair in tbl.pairs::<String, Table>() {
                if let Ok((name, list_t)) = pair {
                    let len = list_t.len()? as usize;
                    let mut vec = Vec::with_capacity(len);
                    for i in 1..=len {
                        let st = ItemStackUD::from_lua_value(list_t.get(i)?)?;
                        vec.push(st.0.read().clone());
                    }
                    g.inventories.insert(name, vec);
                }
            }
            Ok(())
        });
        methods.add_method("add_item", |lua, this, (list, val): (String, Value)| {
            let mut rem = ItemStackUD::from_lua_value(val)?.0.read().clone();
            if rem.is_empty() {
                return Ok(ItemStackUD(Arc::new(RwLock::new(ItemStackData::default()))));
            }
            let mut stack_max = 64u16;
            if let Ok(core) = lua.globals().get::<Table>("core") {
                if let Ok(items) = core.get::<Table>("registered_items") {
                    if let Ok(def) = items.get::<Table>(rem.name.as_str()) {
                        if let Ok(m) = def.get::<u16>("stack_max") {
                            stack_max = m.max(1);
                        }
                    }
                }
            }
            let mut g = this.state.write();
            if let Some(vec) = g.inventories.get_mut(&list) {
                if rem.wear == 0 {
                    for slot in vec.iter_mut() {
                        if !slot.is_empty()
                            && slot.name == rem.name
                            && slot.wear == 0
                            && slot.count < stack_max
                        {
                            let take = (stack_max - slot.count).min(rem.count);
                            slot.count += take;
                            rem.count -= take;
                            if rem.count == 0 {
                                rem.name.clear();
                                break;
                            }
                        }
                    }
                }
                if !rem.is_empty() {
                    let hotbar_end = vec.len().min(9);
                    let (hotbar, rest) = vec.split_at_mut(hotbar_end);
                    for slot in hotbar.iter_mut().chain(rest.iter_mut()) {
                        if slot.is_empty() {
                            let take = rem.count.min(stack_max);
                            let mut placed = rem.clone();
                            placed.count = take;
                            *slot = placed;
                            rem.count -= take;
                            if rem.count == 0 {
                                rem.name.clear();
                                rem.wear = 0;
                                break;
                            }
                        }
                    }
                }
            }
            Ok(ItemStackUD(Arc::new(RwLock::new(rem))))
        });
        methods.add_method("room_for_item", |lua, this, (list, val): (String, Value)| {
            let add = ItemStackUD::from_lua_value(val)?.0.read().clone();
            if add.is_empty() {
                return Ok(true);
            }
            let mut stack_max = 64u32;
            if let Ok(core) = lua.globals().get::<Table>("core") {
                if let Ok(items) = core.get::<Table>("registered_items") {
                    if let Ok(def) = items.get::<Table>(add.name.as_str()) {
                        if let Ok(m) = def.get::<u32>("stack_max") {
                            stack_max = m.max(1);
                        }
                    }
                }
            }
            let g = this.state.read();
            let mut free: u32 = 0;
            if let Some(vec) = g.inventories.get(&list) {
                for slot in vec {
                    if slot.is_empty() {
                        free += stack_max;
                    } else if slot.name == add.name && slot.wear == 0 && add.wear == 0 {
                        free += stack_max.saturating_sub(slot.count as u32);
                    }
                }
            }
            Ok(free >= add.count as u32)
        });
        methods.add_method("contains_item", |_, this, (list, val, _match_meta): (String, Value, Option<bool>)| {
            let req = ItemStackUD::from_lua_value(val)?.0.read().clone();
            let g = this.state.read();
            let mut total: u32 = 0;
            if let Some(vec) = g.inventories.get(&list) {
                for slot in vec {
                    if !slot.is_empty() && slot.name == req.name {
                        total += slot.count as u32;
                    }
                }
            }
            Ok(total >= req.count as u32)
        });
        methods.add_method("remove_item", |_, this, (list, val): (String, Value)| {
            let mut req = ItemStackUD::from_lua_value(val)?.0.read().clone();
            let mut removed = ItemStackData {
                name: req.name.clone(),
                count: 0,
                wear: 0,
                meta: Arc::new(RwLock::new(MetaDataState::default())),
            };
            let mut g = this.state.write();
            if let Some(vec) = g.inventories.get_mut(&list) {
                for slot in vec.iter_mut() {
                    if !slot.is_empty() && slot.name == req.name {
                        let take = slot.count.min(req.count);
                        slot.count -= take;
                        req.count -= take;
                        removed.count += take;
                        if slot.count == 0 {
                            slot.name.clear();
                        }
                        if req.count == 0 {
                            break;
                        }
                    }
                }
            }
            if removed.count == 0 {
                removed.name.clear();
            }
            Ok(ItemStackUD(Arc::new(RwLock::new(removed))))
        });
        methods.add_method("get_location", |lua, this, ()| {
            let tbl = lua.create_table()?;
            tbl.set("type", this.location.as_str())?;
            Ok(tbl)
        });
    }
}

#[derive(Clone)]
pub struct NodeTimerRefUD {
    pub world: Arc<WorldStore>,
    pub pos: IVec3,
}

impl UserData for NodeTimerRefUD {
    fn add_methods<M: UserDataMethods<Self>>(methods: &mut M) {
        methods.add_method("set", |_, this, (to, el): (f64, f64)| {
            let mut timers = this.world.node_timers.write();
            if to <= 0.0 {
                timers.remove(&this.pos);
            } else {
                timers.insert(
                    this.pos,
                    crate::world::NodeTimerState {
                        timeout: to,
                        elapsed: el,
                        started: true,
                    },
                );
            }
            Ok(())
        });
        methods.add_method("start", |_, this, to: f64| {
            let mut timers = this.world.node_timers.write();
            if to <= 0.0 {
                timers.remove(&this.pos);
            } else {
                timers.insert(
                    this.pos,
                    crate::world::NodeTimerState {
                        timeout: to,
                        elapsed: 0.0,
                        started: true,
                    },
                );
            }
            Ok(())
        });
        methods.add_method("stop", |_, this, ()| {
            this.world.node_timers.write().remove(&this.pos);
            Ok(())
        });
        methods.add_method("get_timeout", |_, this, ()| {
            Ok(this
                .world
                .node_timers
                .read()
                .get(&this.pos)
                .map(|t| t.timeout)
                .unwrap_or(0.0))
        });
        methods.add_method("get_elapsed", |_, this, ()| {
            Ok(this
                .world
                .node_timers
                .read()
                .get(&this.pos)
                .map(|t| t.elapsed)
                .unwrap_or(0.0))
        });
        methods.add_method("is_started", |_, this, ()| {
            Ok(this
                .world
                .node_timers
                .read()
                .get(&this.pos)
                .map(|t| t.started && t.timeout > 0.0)
                .unwrap_or(false))
        });
    }
}

/// VoxelManip userdata bridging LuaJIT mapgen scripts directly to WorldStore chunks.
#[derive(Clone)]
pub struct VoxelManipUD {
    pub world: Arc<WorldStore>,
    pub registries: Arc<RwLock<EngineRegistries>>,
    pub emin: Arc<RwLock<IVec3>>,
    pub emax: Arc<RwLock<IVec3>>,
    pub data: Arc<RwLock<Vec<u16>>>,
    pub light_data: Arc<RwLock<Vec<u8>>>,
    pub param2_data: Arc<RwLock<Vec<u8>>>,
    pub modified: Arc<RwLock<bool>>,
}

impl VoxelManipUD {
    pub fn new(world: Arc<WorldStore>, registries: Arc<RwLock<EngineRegistries>>, p1: Option<IVec3>, p2: Option<IVec3>) -> Self {
        let vm = Self {
            world,
            registries,
            emin: Arc::new(RwLock::new(IVec3::ZERO)),
            emax: Arc::new(RwLock::new(IVec3::new(15, 15, 15))),
            data: Arc::new(RwLock::new(vec![CONTENT_AIR; 4096])),
            light_data: Arc::new(RwLock::new(vec![0x0F; 4096])),
            param2_data: Arc::new(RwLock::new(vec![0; 4096])),
            modified: Arc::new(RwLock::new(false)),
        };
        if let (Some(a), Some(b)) = (p1, p2) {
            vm.read_from_map(a, b);
        }
        vm
    }

    pub fn read_from_map(&self, p1: IVec3, p2: IVec3) -> (IVec3, IVec3) {
        let lo = p1.min(p2);
        let hi = p1.max(p2);
        // Align to 16-voxel mapblock boundaries just like Luanti
        let emin = IVec3::new(
            lo.x.div_euclid(16) * 16,
            lo.y.div_euclid(16) * 16,
            lo.z.div_euclid(16) * 16,
        );
        let emax = IVec3::new(
            hi.x.div_euclid(16) * 16 + 15,
            hi.y.div_euclid(16) * 16 + 15,
            hi.z.div_euclid(16) * 16 + 15,
        );
        let sx = (emax.x - emin.x + 1).max(1) as usize;
        let sy = (emax.y - emin.y + 1).max(1) as usize;
        let sz = (emax.z - emin.z + 1).max(1) as usize;
        let vol = sx * sy * sz;

        let mut d = vec![CONTENT_IGNORE; vol];
        let mut ld = vec![0x0Fu8; vol];
        let mut p2d = vec![0u8; vol];

        for z in emin.z..=emax.z {
            for y in emin.y..=emax.y {
                for x in emin.x..=emax.x {
                    let (id, l, p2) = self.world.get_node(IVec3::new(x, y, z));
                    let idx = (z - emin.z) as usize * sy * sx + (y - emin.y) as usize * sx + (x - emin.x) as usize;
                    d[idx] = id;
                    ld[idx] = l;
                    p2d[idx] = p2;
                }
            }
        }

        *self.emin.write() = emin;
        *self.emax.write() = emax;
        *self.data.write() = d;
        *self.light_data.write() = ld;
        *self.param2_data.write() = p2d;
        (emin, emax)
    }

    pub fn write_to_map(&self) {
        let emin = *self.emin.read();
        let emax = *self.emax.read();
        let sx = (emax.x - emin.x + 1).max(1) as usize;
        let sy = (emax.y - emin.y + 1).max(1) as usize;
        let d = self.data.read();
        let ld = self.light_data.read();
        let p2d = self.param2_data.read();
        for z in emin.z..=emax.z {
            for y in emin.y..=emax.y {
                for x in emin.x..=emax.x {
                    let idx = (z - emin.z) as usize * sy * sx + (y - emin.y) as usize * sx + (x - emin.x) as usize;
                    if idx < d.len() && d[idx] != CONTENT_IGNORE {
                        self.world.set_node(IVec3::new(x, y, z), d[idx], ld[idx], p2d[idx]);
                    }
                }
            }
        }
        *self.modified.write() = false;
    }
}

fn lua_to_ivec3(t: &Table) -> IVec3 {
    let x: f64 = t.get("x").unwrap_or(0.0);
    let y: f64 = t.get("y").unwrap_or(0.0);
    let z: f64 = t.get("z").unwrap_or(0.0);
    IVec3::new(x.round() as i32, y.round() as i32, z.round() as i32)
}

fn ivec3_to_lua(lua: &Lua, v: IVec3) -> LuaResult<Table> {
    super::core_api::make_vec3(lua, v.x as f64, v.y as f64, v.z as f64)
}

impl UserData for VoxelManipUD {
    fn add_methods<M: UserDataMethods<Self>>(methods: &mut M) {
        methods.add_method("read_from_map", |lua, this, (p1, p2): (Table, Table)| {
            let (emin, emax) = this.read_from_map(lua_to_ivec3(&p1), lua_to_ivec3(&p2));
            Ok((ivec3_to_lua(lua, emin)?, ivec3_to_lua(lua, emax)?))
        });
        methods.add_method("write_to_map", |_, this, _light: Option<bool>| {
            this.write_to_map();
            Ok(())
        });
        methods.add_method("get_emerged_area", |lua, this, ()| {
            let emin = *this.emin.read();
            let emax = *this.emax.read();
            Ok((ivec3_to_lua(lua, emin)?, ivec3_to_lua(lua, emax)?))
        });
        methods.add_method("get_data", |lua, this, buf_opt: Option<Table>| {
            let d = this.data.read();
            let out = buf_opt.unwrap_or_else(|| lua.create_table_with_capacity(d.len(), 0).unwrap());
            for (i, &id) in d.iter().enumerate() {
                out.set(i + 1, id)?;
            }
            Ok(out)
        });
        methods.add_method("set_data", |_, this, tbl: Table| {
            let mut d = this.data.write();
            let len = (tbl.len()? as usize).min(d.len());
            for i in 1..=len {
                if let Ok(v) = tbl.get::<u16>(i) {
                    d[i - 1] = v;
                }
            }
            *this.modified.write() = true;
            Ok(())
        });
        methods.add_method("get_light_data", |lua, this, buf_opt: Option<Table>| {
            let ld = this.light_data.read();
            let out = buf_opt.unwrap_or_else(|| lua.create_table_with_capacity(ld.len(), 0).unwrap());
            for (i, &v) in ld.iter().enumerate() {
                out.set(i + 1, v)?;
            }
            Ok(out)
        });
        methods.add_method("set_light_data", |_, this, tbl: Table| {
            let mut ld = this.light_data.write();
            let len = (tbl.len()? as usize).min(ld.len());
            for i in 1..=len {
                if let Ok(v) = tbl.get::<u8>(i) {
                    ld[i - 1] = v;
                }
            }
            Ok(())
        });
        methods.add_method("get_param2_data", |lua, this, buf_opt: Option<Table>| {
            let pd = this.param2_data.read();
            let out = buf_opt.unwrap_or_else(|| lua.create_table_with_capacity(pd.len(), 0).unwrap());
            for (i, &v) in pd.iter().enumerate() {
                out.set(i + 1, v)?;
            }
            Ok(out)
        });
        methods.add_method("set_param2_data", |_, this, tbl: Table| {
            let mut pd = this.param2_data.write();
            let len = (tbl.len()? as usize).min(pd.len());
            for i in 1..=len {
                if let Ok(v) = tbl.get::<u8>(i) {
                    pd[i - 1] = v;
                }
            }
            Ok(())
        });
        methods.add_method("get_node_at", |lua, this, pos: Table| {
            let p = lua_to_ivec3(&pos);
            let emin = *this.emin.read();
            let emax = *this.emax.read();
            let sx = (emax.x - emin.x + 1).max(1) as usize;
            let sy = (emax.y - emin.y + 1).max(1) as usize;
            let mut name = "ignore".to_string();
            let mut p1 = 0u8;
            let mut p2 = 0u8;
            if p.x >= emin.x && p.x <= emax.x && p.y >= emin.y && p.y <= emax.y && p.z >= emin.z && p.z <= emax.z {
                let idx = (p.z - emin.z) as usize * sy * sx + (p.y - emin.y) as usize * sx + (p.x - emin.x) as usize;
                let id = this.data.read()[idx];
                p1 = this.light_data.read()[idx];
                p2 = this.param2_data.read()[idx];
                name = this.registries.read().get_name_from_content_id(id).to_string();
            }
            let tbl = lua.create_table()?;
            tbl.set("name", name)?;
            tbl.set("param1", p1)?;
            tbl.set("param2", p2)?;
            Ok(tbl)
        });
        methods.add_method("set_node_at", |_, this, (pos, node): (Table, Table)| {
            let p = lua_to_ivec3(&pos);
            let emin = *this.emin.read();
            let emax = *this.emax.read();
            let sx = (emax.x - emin.x + 1).max(1) as usize;
            let sy = (emax.y - emin.y + 1).max(1) as usize;
            if p.x >= emin.x && p.x <= emax.x && p.y >= emin.y && p.y <= emax.y && p.z >= emin.z && p.z <= emax.z {
                let idx = (p.z - emin.z) as usize * sy * sx + (p.y - emin.y) as usize * sx + (p.x - emin.x) as usize;
                let name: String = node.get("name").unwrap_or_else(|_| "air".into());
                let p1: u8 = node.get("param1").unwrap_or(15);
                let p2: u8 = node.get("param2").unwrap_or(0);
                let id = this.registries.read().get_content_id(&name);
                this.data.write()[idx] = id;
                this.light_data.write()[idx] = p1;
                this.param2_data.write()[idx] = p2;
                *this.modified.write() = true;
            }
            Ok(())
        });
        methods.add_method("calc_lighting", |_, _, _: MultiValue| Ok(()));
        methods.add_method("set_lighting", |_, _, _: MultiValue| Ok(()));
        methods.add_method("update_liquids", |_, _, ()| Ok(()));
        methods.add_method("update_map", |_, _, ()| Ok(()));
        methods.add_method("was_modified", |_, this, ()| Ok(*this.modified.read()));
    }
}

/// Live Player & Entity ObjectRef shared between Rust physics/UI and Mineclonia Lua mods.
#[derive(Debug, Clone)]
pub struct ObjectState {
    pub id: u64,
    pub is_player: bool,
    pub name: String,
    pub entity_name: String,
    pub pos: Vec3,
    pub vel: Vec3,
    pub accel: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub rotation: Vec3,
    pub removed: bool,
    pub physical: bool,
    pub collide_with_objects: bool,
    pub collisionbox: [f32; 6],
    pub selectionbox: [f32; 6],
    pub pointable: bool,
    pub stepheight: f32,
    pub visual: String,
    pub visual_size: Vec3,
    pub mesh: String,
    pub textures: Vec<String>,
    pub wield_item: String,
    pub automatic_rotate: f32,
    pub is_visible: bool,
    pub sneak: bool,
    pub hp: i32,
    pub hp_max: i32,
    pub hunger: i32,
    pub xp_level: i32,
    pub xp_progress: f32,
    pub breath: u16,
    pub breath_max: u16,
    pub speed_mult: f32,
    pub jump_mult: f32,
    pub gravity_mult: f32,
    pub fov: f32,
    pub wield_index: usize,
    pub formspec_prepend: String,
    pub inventory_formspec: String,
    pub meta: Arc<RwLock<MetaDataState>>,
    pub hud_elems: HashMap<u64, HashMap<String, String>>,
    pub next_hud_id: u64,
    pub hotbar_count: usize,
    pub properties: HashMap<String, String>,
    pub armor_groups: HashMap<String, i32>,
    pub anim_range: (f32, f32),
    pub anim_speed: f32,
    pub anim_blend: f32,
    pub anim_loop: bool,
    pub hurt_flash: f32,
    pub walk_phase: f32,
    pub walk_blend: f32,
    pub accel_set_by_lua: bool,
    pub is_on_ground: bool,
    pub pending_knockback: Vec3,
    pub hp_dirty_from_lua: bool,
    pub age_sec: f32,
    pub is_dead: bool,
    pub has_dropped_death_items: bool,
    pub spawn_invuln_timer: f32,
    pub last_death_cause: String,
}

/// Offset between Cubex world voxel bounds `[ix..ix+1]` and Luanti voxel centers `[ix-0.5..ix+0.5]`.
pub const LUA_POS_OFFSET: Vec3 = Vec3::new(0.5, 0.5, 0.5);

#[inline]
pub fn cubex_to_lua_pos(p: Vec3) -> Vec3 {
    p - LUA_POS_OFFSET
}

#[inline]
pub fn lua_to_cubex_pos(p: Vec3) -> Vec3 {
    p + LUA_POS_OFFSET
}

impl ObjectState {
    pub fn new_player(name: impl Into<String>) -> Self {
        let meta = Arc::new(RwLock::new(MetaDataState::default()));
        {
            let mut g = meta.write();
            g.inventories.insert("main".into(), vec![ItemStackData::default(); 36]);
            g.inventories.insert("craft".into(), vec![ItemStackData::default(); 4]);
            g.inventories.insert("craftpreview".into(), vec![ItemStackData::default(); 1]);
            g.inventories.insert("craftresult".into(), vec![ItemStackData::default(); 1]);
            g.inventories.insert("armor".into(), vec![ItemStackData::default(); 4]);
            g.inventories.insert("offhand".into(), vec![ItemStackData::default(); 1]);
            g.inventories.insert("hand".into(), vec![ItemStackData::default(); 1]);
        }
        Self {
            id: 0,
            is_player: true,
            name: name.into(),
            entity_name: String::new(),
            pos: Vec3::new(0.0, 24.0, 0.0),
            vel: Vec3::ZERO,
            accel: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            rotation: Vec3::ZERO,
            removed: false,
            physical: true,
            collide_with_objects: true,
            collisionbox: [-0.3, 0.0, -0.3, 0.3, 1.75, 0.3],
            selectionbox: [-0.3, 0.0, -0.3, 0.3, 1.75, 0.3],
            pointable: true,
            stepheight: 0.6,
            visual: "mesh".into(),
            visual_size: Vec3::ONE,
            mesh: String::new(),
            textures: Vec::new(),
            wield_item: String::new(),
            automatic_rotate: 0.0,
            is_visible: true,
            sneak: false,
            hp: 20,
            hp_max: 20,
            hunger: 20,
            xp_level: 0,
            xp_progress: 0.0,
            breath: 10,
            breath_max: 10,
            speed_mult: 1.0,
            jump_mult: 1.0,
            gravity_mult: 1.0,
            fov: 80.0,
            wield_index: 1,
            formspec_prepend: String::new(),
            inventory_formspec: String::new(),
            meta,
            hud_elems: HashMap::new(),
            next_hud_id: 1,
            hotbar_count: 9,
            properties: HashMap::new(),
            armor_groups: HashMap::from([("fleshy".into(), 100)]),
            anim_range: (0.0, 0.0),
            anim_speed: 15.0,
            anim_blend: 0.0,
            anim_loop: true,
            hurt_flash: 0.0,
            walk_phase: 0.0,
            walk_blend: 0.0,
            accel_set_by_lua: false,
            is_on_ground: false,
            pending_knockback: Vec3::ZERO,
            hp_dirty_from_lua: false,
            age_sec: 0.0,
            is_dead: false,
            has_dropped_death_items: false,
            spawn_invuln_timer: 0.0,
            last_death_cause: String::new(),
        }
    }

    /// Populate Creative-mode starter blocks & tools into hotbar slots `0..9`.
    /// Must ONLY be called when entering or creating a Creative world (never Survival).
    pub fn populate_creative_starter(&mut self) {
        let mut g = self.meta.write();
        let inv = g
            .inventories
            .entry("main".into())
            .or_insert_with(|| vec![ItemStackData::default(); 36]);
        if inv.len() < 36 {
            inv.resize_with(36, ItemStackData::default);
        }
        inv[0] = ItemStackData::from_spec("mcl_core:dirt_with_grass 64");
        inv[1] = ItemStackData::from_spec("mcl_core:stone 64");
        inv[2] = ItemStackData::from_spec("mcl_trees:wood_oak 64");
        inv[3] = ItemStackData::from_spec("mcl_core:cobble 64");
        inv[4] = ItemStackData::from_spec("mcl_torches:torch 64");
        inv[5] = ItemStackData::from_spec("mcl_core:glass 64");
        inv[6] = ItemStackData::from_spec("mcl_tools:pick_diamond 1");
        inv[7] = ItemStackData::from_spec("mcl_tools:axe_diamond 1");
        inv[8] = ItemStackData::from_spec("mcl_tools:sword_diamond 1");
    }

    pub fn new_entity(id: u64, entity_name: impl Into<String>, pos: Vec3) -> Self {
        Self {
            id,
            is_player: false,
            name: String::new(),
            entity_name: entity_name.into(),
            pos,
            vel: Vec3::ZERO,
            accel: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            rotation: Vec3::ZERO,
            removed: false,
            physical: false,
            collide_with_objects: false,
            collisionbox: [-0.3, -0.3, -0.3, 0.3, 0.3, 0.3],
            selectionbox: [-0.3, -0.3, -0.3, 0.3, 0.3, 0.3],
            pointable: true,
            stepheight: 1.05,
            visual: "cube".into(),
            visual_size: Vec3::ONE,
            mesh: String::new(),
            textures: Vec::new(),
            wield_item: String::new(),
            automatic_rotate: 0.0,
            is_visible: true,
            sneak: false,
            hp: 10,
            hp_max: 10,
            hunger: 20,
            xp_level: 0,
            xp_progress: 0.0,
            breath: 0,
            breath_max: 10,
            speed_mult: 1.0,
            jump_mult: 1.0,
            gravity_mult: 1.0,
            fov: 80.0,
            wield_index: 1,
            formspec_prepend: String::new(),
            inventory_formspec: String::new(),
            meta: Arc::new(RwLock::new(MetaDataState::default())),
            hud_elems: HashMap::new(),
            next_hud_id: 1,
            hotbar_count: 9,
            properties: HashMap::new(),
            armor_groups: HashMap::from([("fleshy".into(), 100)]),
            anim_range: (0.0, 0.0),
            anim_speed: 15.0,
            anim_blend: 0.0,
            anim_loop: true,
            hurt_flash: 0.0,
            walk_phase: 0.0,
            walk_blend: 0.0,
            accel_set_by_lua: false,
            is_on_ground: false,
            pending_knockback: Vec3::ZERO,
            hp_dirty_from_lua: false,
            age_sec: 0.0,
            is_dead: false,
            has_dropped_death_items: false,
            spawn_invuln_timer: 0.0,
            last_death_cause: String::new(),
        }
    }

    /// Atomically clone non-empty stacks from `main`, `craft`, `armor`, and `offhand`
    /// and immediately clear every slot so items can never be dropped twice on death.
    pub fn drain_inventory_for_death_locked(&mut self, keep_inventory: bool) -> Vec<ItemStackData> {
        if self.has_dropped_death_items {
            return Vec::new();
        }
        self.is_dead = true;
        self.has_dropped_death_items = true;
        if keep_inventory {
            return Vec::new();
        }
        let mut dropped = Vec::new();
        let mut meta = self.meta.write();
        for list_name in ["main", "craft", "armor", "offhand"] {
            if let Some(slots) = meta.inventories.get_mut(list_name) {
                for slot in slots.iter_mut() {
                    if !slot.is_empty() {
                        dropped.push(slot.clone());
                        *slot = ItemStackData::default();
                    }
                }
            }
        }
        dropped
    }

    /// Apply a Luanti `object_properties` table cleanly onto this `ObjectState`.
    pub fn apply_properties_table(&mut self, tbl: &Table) {
        let prev_cbox = self.collisionbox;
        if let Ok(hp_max) = tbl.get::<i32>("hp_max") {
            self.hp_max = hp_max.max(1);
            if self.hp > self.hp_max {
                self.hp = self.hp_max;
            }
        }
        if let Ok(bm) = tbl.get::<u16>("breath_max") {
            self.breath_max = bm.max(1);
            if self.breath > self.breath_max {
                self.breath = self.breath_max;
            }
        }
        if let Ok(eh) = tbl.get::<f32>("eye_height") {
            self.properties.insert("eye_height".into(), eh.to_string());
        }
        if let Ok(mesh) = tbl.get::<String>("mesh") {
            self.mesh = mesh.clone();
            self.properties.insert("mesh".into(), mesh);
        }
        if let Ok(sh) = tbl.get::<f32>("stepheight") {
            self.stepheight = sh.clamp(0.0, 2.5);
        }
        if let Ok(val) = tbl.get::<Value>("pointable") {
            match val {
                Value::Boolean(pt) => self.pointable = pt,
                Value::String(s) => {
                    if let Ok(st) = s.to_str() {
                        self.pointable = st != "blocking" && !st.eq_ignore_ascii_case("false");
                    }
                }
                _ => {}
            }
        }
        if let Ok(Value::Boolean(phys)) = tbl.get::<Value>("physical") {
            self.physical = phys;
        }
        if let Ok(Value::Boolean(cwo)) = tbl.get::<Value>("collide_with_objects") {
            self.collide_with_objects = cwo;
        }
        if let Ok(vis) = tbl.get::<String>("visual") {
            self.visual = vis;
        }
        if let Ok(wi) = tbl.get::<String>("wield_item") {
            self.wield_item = wi;
        }
        if let Ok(Value::Boolean(iv)) = tbl.get::<Value>("is_visible") {
            self.is_visible = iv;
        }
        if let Ok(ar) = tbl.get::<f32>("automatic_rotate") {
            self.automatic_rotate = ar;
        }
        if let Ok(vs) = tbl.get::<Table>("visual_size") {
            let x = vs.get("x").unwrap_or(1.0);
            let y = vs.get("y").unwrap_or(1.0);
            let z = vs.get("z").unwrap_or(x);
            self.visual_size = Vec3::new(x, y, z);
        } else if let Ok(v) = tbl.get::<f32>("visual_size") {
            self.visual_size = Vec3::splat(v);
        }
        if let Ok(cb) = tbl.get::<Table>("collisionbox") {
            for i in 0..6 {
                if let Ok(v) = cb.get::<f32>(i + 1) {
                    self.collisionbox[i] = v;
                }
            }
        }
        if let Ok(sb) = tbl.get::<Table>("selectionbox") {
            for i in 0..6 {
                if let Ok(v) = sb.get::<f32>(i + 1) {
                    self.selectionbox[i] = v;
                }
            }
        } else if self.selectionbox == prev_cbox {
            self.selectionbox = self.collisionbox;
        }
        if let Ok(tex) = tbl.get::<Table>("textures") {
            self.textures.clear();
            let len = tex.len().unwrap_or(0);
            for i in 1..=len {
                if let Ok(s) = tex.get::<String>(i) {
                    self.textures.push(s);
                }
            }
        }
    }
}

pub fn parse_tool_capabilities_from_lua(val: Option<&Value>) -> (f32, HashMap<String, i32>) {
    let mut full_punch_interval = 1.0f32;
    let mut damage_groups = HashMap::new();
    if let Some(Value::Table(t)) = val {
        if let Ok(fpi) = t.get::<f32>("full_punch_interval") {
            full_punch_interval = fpi.max(0.01);
        }
        if let Ok(dg) = t.get::<Table>("damage_groups") {
            for pair in dg.pairs::<String, f64>() {
                if let Ok((k, v)) = pair {
                    damage_groups.insert(k, v.round() as i32);
                }
            }
        }
    }
    if damage_groups.is_empty() {
        damage_groups.insert("fleshy".into(), 1);
    }
    (full_punch_interval, damage_groups)
}

/// Luanti damage formula (`src/content_cao.cpp` / `player.cpp`):
/// sum over `damage_groups` of `dmg * clamp(tflp / full_punch_interval, 0, 1) * (armor[group] / 100)`.
pub fn compute_punch_damage(
    tflp: f32,
    full_punch_interval: f32,
    damage_groups: &HashMap<String, i32>,
    armor_groups: &HashMap<String, i32>,
) -> i32 {
    let scale = (tflp / full_punch_interval.max(0.01)).clamp(0.0, 1.0) as f64;
    let mut total = 0.0f64;
    for (grp, &dmg) in damage_groups {
        let rating = armor_groups.get(grp).copied().unwrap_or(0) as f64;
        if rating != 0.0 {
            total += (dmg as f64) * scale * (rating / 100.0);
        }
    }
    total.round().max(0.0) as i32
}

fn parse_lua_vec3_opt(val: Option<&Value>) -> Option<Vec3> {
    let Some(Value::Table(t)) = val else {
        return None;
    };
    let x: f32 = t.get("x").ok()?;
    let y: f32 = t.get("y").ok()?;
    let z: f32 = t.get("z").ok()?;
    Some(Vec3::new(x, y, z))
}

fn despawn_lua_entity(lua: &Lua, state: &Arc<RwLock<ObjectState>>, puncher: Option<Value>) {
    let (id, already_removed) = {
        let g = state.read();
        (g.id, g.removed)
    };
    if id == 0 || already_removed {
        return;
    }
    if let Ok(core) = lua.globals().get::<Table>("core") {
        if let Ok(tbl) = core.get::<Table>("_cubex_luaentities") {
            if let Ok(ent) = tbl.get::<Table>(id) {
                if let Ok(on_death) = ent.get::<mlua::Function>("on_death") {
                    let killer = puncher.unwrap_or(Value::Nil);
                    let _ = on_death.call::<Value>((ent.clone(), killer));
                }
            }
            let _ = tbl.set(id, Value::Nil);
        }
    }
    state.write().removed = true;
    if let Ok(Value::Table(refs)) = lua.named_registry_value::<Value>("_cubex_object_refs") {
        let _ = refs.set(format!("e:{id}"), Value::Nil);
    }
}

pub fn drain_player_inventory_for_death(
    state: &Arc<RwLock<ObjectState>>,
    keep_inventory: bool,
) -> Vec<ItemStackData> {
    state.write().drain_inventory_for_death_locked(keep_inventory)
}

pub fn format_death_cause(reason_type: &str) -> &'static str {
    match reason_type.trim().to_ascii_lowercase().as_str() {
        "fall" => "Fell from a high place",
        "drown" => "Drowned",
        "in_wall" => "Suffocated in a wall",
        "punch" => "Slain in combat",
        "falling_node" => "Squashed by a falling block",
        "node_damage" => "Killed by hazardous terrain",
        "starve" => "Starved to death",
        _ => "Died",
    }
}

fn extract_effective_hp_reason(reason_tbl: &Table) -> String {
    for nested_key in ["_mcl_reason", "mcl_reason"] {
        if let Ok(nested) = reason_tbl.get::<Table>(nested_key) {
            if let Ok(t) = nested.get::<String>("type") {
                if !t.trim().is_empty() {
                    return t;
                }
            }
        }
    }
    reason_tbl
        .get::<String>("type")
        .unwrap_or_else(|_| "set_hp".into())
}

fn apply_player_hp_change_in_lua(
    lua: &Lua,
    state: &Arc<RwLock<ObjectState>>,
    target_hp: i32,
    reason_val: Option<Value>,
) -> LuaResult<()> {
    let (old_hp, hp_max, already_dead, invuln_active, death_pos) = {
        let g = state.read();
        (
            g.hp,
            g.hp_max,
            g.is_dead,
            g.spawn_invuln_timer > 0.0,
            cubex_to_lua_pos(g.pos),
        )
    };
    if already_dead {
        return Ok(());
    }

    let mut hp_change = target_hp - old_hp;
    let reason_tbl = match reason_val {
        Some(Value::Table(t)) => t,
        _ => {
            let t = lua.create_table()?;
            t.set("type", "set_hp")?;
            t
        }
    };
    let top_reason_type = reason_tbl
        .get::<String>("type")
        .unwrap_or_else(|_| "set_hp".into());
    let reason_type = extract_effective_hp_reason(&reason_tbl);
    if top_reason_type == "in_wall" || reason_type == "in_wall" {
        return Ok(());
    }
    if invuln_active && hp_change < 0 {
        return Ok(());
    }

    let player_ud = object_ref_to_lua(lua, state.clone())?;
    if let Ok(core) = lua.globals().get::<Table>("core") {
        if let Ok(hp_cbs) = core.get::<Table>("registered_on_player_hpchanges") {
            if let Ok(mods) = hp_cbs.get::<Table>("modifiers") {
                let len = mods.len().unwrap_or(0);
                for i in 1..=len {
                    if let Ok(func) = mods.get::<mlua::Function>(i) {
                        if let Ok(res) = func.call::<MultiValue>((
                            player_ud.clone(),
                            hp_change,
                            reason_tbl.clone(),
                        )) {
                            let mut it = res.into_iter();
                            if let Some(v) = it.next() {
                                match v {
                                    Value::Integer(n) => hp_change = n as i32,
                                    Value::Number(n) => hp_change = n.round() as i32,
                                    _ => {}
                                }
                            }
                            let stop = matches!(it.next(), Some(Value::Boolean(true)));
                            if stop || hp_change == 0 {
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    let new_hp = (old_hp + hp_change).clamp(0, hp_max);
    let actual_change = new_hp - old_hp;
    let dying_now = old_hp > 0 && new_hp == 0;
    {
        let mut g = state.write();
        if g.is_dead {
            return Ok(());
        }
        g.hp = new_hp;
        g.hp_dirty_from_lua = true;
        if new_hp < old_hp {
            g.hurt_flash = 0.25;
        }
        if dying_now {
            g.is_dead = true;
            g.last_death_cause = format_death_cause(&reason_type).to_string();
        }
    }

    if let Ok(core) = lua.globals().get::<Table>("core") {
        if let Ok(hp_cbs) = core.get::<Table>("registered_on_player_hpchanges") {
            if let Ok(loggers) = hp_cbs.get::<Table>("loggers") {
                let len = loggers.len().unwrap_or(0);
                for i in 1..=len {
                    if let Ok(func) = loggers.get::<mlua::Function>(i) {
                        let _ = func.call::<Value>((
                            player_ud.clone(),
                            actual_change,
                            reason_tbl.clone(),
                        ));
                    }
                }
            }
        }
        if dying_now {
            let keep_inv = core
                .get::<Value>("settings")
                .ok()
                .and_then(|sv| {
                    if let Value::UserData(ref ud) = sv {
                        if let Ok(settings_ud) = ud.borrow::<SettingsUD>() {
                            return Some(settings_ud.0.read().get_bool("mcl_keepInventory", true));
                        }
                    }
                    None
                })
                .unwrap_or(true);
            let dropped = drain_player_inventory_for_death(state, keep_inv);
            if !dropped.is_empty() {
                if let Ok(add_item) = core.get::<mlua::Function>("add_item") {
                    for (idx, st) in dropped.iter().enumerate() {
                        let ang = (idx as f32) * 1.1;
                        if let Ok(pt) = super::core_api::make_vec3(
                            lua,
                            (death_pos.x + ang.cos() * 0.35) as f64,
                            (death_pos.y + 0.35) as f64,
                            (death_pos.z + ang.sin() * 0.35) as f64,
                        ) {
                            let spec = if st.wear > 0 {
                                format!("{} {} {}", st.name, st.count.max(1), st.wear)
                            } else if st.count > 1 {
                                format!("{} {}", st.name, st.count)
                            } else {
                                st.name.clone()
                            };
                            let _ = add_item.call::<Value>((pt, spec));
                        }
                    }
                }
            }
            if let Ok(die_cbs) = core.get::<Table>("registered_on_dieplayers") {
                let len = die_cbs.len().unwrap_or(0);
                for i in 1..=len {
                    if let Ok(func) = die_cbs.get::<mlua::Function>(i) {
                        let _ = func.call::<Value>((player_ud.clone(), reason_tbl.clone()));
                    }
                }
            }
        }
    }
    Ok(())
}

#[derive(Clone)]
pub struct ObjectRefUD(pub Arc<RwLock<ObjectState>>);

/// Return a canonical Lua `AnyUserData` per live `ObjectState` so Lua mods
/// keying tables by `player` (`mcl_player.players[player]`, etc.) share identity.
pub fn object_ref_to_lua(lua: &Lua, state: Arc<RwLock<ObjectState>>) -> LuaResult<AnyUserData> {
    let key = {
        let g = state.read();
        if g.is_player {
            format!("p:{}", g.name)
        } else {
            format!("e:{}", g.id)
        }
    };
    let refs: Table = match lua.named_registry_value::<Value>("_cubex_object_refs")? {
        Value::Table(t) => t,
        _ => {
            let t = lua.create_table()?;
            lua.set_named_registry_value("_cubex_object_refs", t.clone())?;
            t
        }
    };
    if let Ok(Value::UserData(ud)) = refs.get::<Value>(key.as_str()) {
        if let Ok(existing) = ud.borrow::<ObjectRefUD>() {
            if Arc::ptr_eq(&existing.0, &state) {
                return Ok(ud);
            }
        }
    }
    let ud = lua.create_userdata(ObjectRefUD(state))?;
    refs.set(key, ud.clone())?;
    Ok(ud)
}

impl UserData for ObjectRefUD {
    fn add_methods<M: UserDataMethods<Self>>(methods: &mut M) {
        methods.add_method("is_player", |_, this, ()| Ok(this.0.read().is_player));
        methods.add_method("get_player_name", |_, this, ()| {
            let g = this.0.read();
            Ok(if g.is_player { g.name.clone() } else { String::new() })
        });
        methods.add_method("get_pos", |lua, this, ()| {
            let p = cubex_to_lua_pos(this.0.read().pos);
            super::core_api::make_vec3(lua, p.x as f64, p.y as f64, p.z as f64)
        });
        methods.add_method("set_pos", |_, this, pos: Table| {
            let x: f32 = pos.get("x").unwrap_or(0.0);
            let y: f32 = pos.get("y").unwrap_or(0.0);
            let z: f32 = pos.get("z").unwrap_or(0.0);
            this.0.write().pos = lua_to_cubex_pos(Vec3::new(x, y, z));
            Ok(())
        });
        methods.add_method("move_to", |_, this, (pos, _): (Table, Option<bool>)| {
            let x: f32 = pos.get("x").unwrap_or(0.0);
            let y: f32 = pos.get("y").unwrap_or(0.0);
            let z: f32 = pos.get("z").unwrap_or(0.0);
            this.0.write().pos = lua_to_cubex_pos(Vec3::new(x, y, z));
            Ok(())
        });
        methods.add_method("is_valid", |_, this, ()| Ok(!this.0.read().removed));
        methods.add_method("get_velocity", |lua, this, ()| {
            let v = this.0.read().vel;
            super::core_api::make_vec3(lua, v.x as f64, v.y as f64, v.z as f64)
        });
        methods.add_method("set_velocity", |_, this, v: Table| {
            let x: f32 = v.get("x").unwrap_or(0.0);
            let y: f32 = v.get("y").unwrap_or(0.0);
            let z: f32 = v.get("z").unwrap_or(0.0);
            this.0.write().vel = Vec3::new(x, y, z);
            Ok(())
        });
        methods.add_method("add_velocity", |_, this, v: Table| {
            let dx: f32 = v.get("x").unwrap_or(0.0);
            let dy: f32 = v.get("y").unwrap_or(0.0);
            let dz: f32 = v.get("z").unwrap_or(0.0);
            let delta = Vec3::new(dx, dy, dz);
            let mut g = this.0.write();
            g.vel += delta;
            if g.is_player {
                g.pending_knockback += delta;
            }
            Ok(())
        });
        methods.add_method("set_acceleration", |_, this, v: Table| {
            let x: f32 = v.get("x").unwrap_or(0.0);
            let y: f32 = v.get("y").unwrap_or(0.0);
            let z: f32 = v.get("z").unwrap_or(0.0);
            let mut g = this.0.write();
            g.accel = Vec3::new(x, y, z);
            g.accel_set_by_lua = true;
            Ok(())
        });
        methods.add_method("get_acceleration", |lua, this, ()| {
            let a = this.0.read().accel;
            super::core_api::make_vec3(lua, a.x as f64, a.y as f64, a.z as f64)
        });
        methods.add_method("set_yaw", |_, this, yaw: f32| {
            let mut g = this.0.write();
            g.yaw = yaw;
            g.rotation.y = yaw;
            Ok(())
        });
        methods.add_method("get_yaw", |_, this, ()| Ok(this.0.read().yaw));
        methods.add_method("set_rotation", |_, this, rot: Table| {
            let x: f32 = rot.get("x").unwrap_or(0.0);
            let y: f32 = rot.get("y").unwrap_or(0.0);
            let z: f32 = rot.get("z").unwrap_or(0.0);
            let mut g = this.0.write();
            g.rotation = Vec3::new(x, y, z);
            g.yaw = y;
            g.pitch = x;
            Ok(())
        });
        methods.add_method("get_rotation", |lua, this, ()| {
            let r = this.0.read().rotation;
            super::core_api::make_vec3(lua, r.x as f64, r.y as f64, r.z as f64)
        });
        methods.add_method("get_hp", |_, this, ()| Ok(this.0.read().hp));
        methods.add_method("set_hp", |lua, this, (hp, reason): (f64, Option<Value>)| {
            let (is_player, old_hp, hp_max, immortal) = {
                let g = this.0.read();
                (
                    g.is_player,
                    g.hp,
                    g.hp_max,
                    g.armor_groups.get("immortal").copied().unwrap_or(0) > 0,
                )
            };
            if is_player {
                return apply_player_hp_change_in_lua(
                    lua,
                    &this.0,
                    hp.round() as i32,
                    reason,
                );
            }
            let new_hp = (hp.round() as i32).clamp(0, hp_max);
            {
                let mut g = this.0.write();
                g.hp = new_hp;
                if new_hp < old_hp {
                    g.hurt_flash = 0.25;
                }
            }
            if old_hp > 0 && new_hp == 0 && !immortal {
                despawn_lua_entity(lua, &this.0, None);
            }
            Ok(())
        });
        methods.add_method("get_breath", |_, this, ()| Ok(this.0.read().breath));
        methods.add_method("set_breath", |_, this, b: u16| {
            this.0.write().breath = b;
            Ok(())
        });
        methods.add_method("get_armor_groups", |lua, this, ()| {
            let t = lua.create_table()?;
            for (k, &v) in &this.0.read().armor_groups {
                t.set(k.as_str(), v)?;
            }
            Ok(t)
        });
        methods.add_method("set_armor_groups", |_, this, tbl: Table| {
            let mut g = this.0.write();
            g.armor_groups.clear();
            for pair in tbl.pairs::<String, i32>() {
                if let Ok((k, v)) = pair {
                    g.armor_groups.insert(k, v);
                }
            }
            Ok(())
        });
        methods.add_method("get_properties", |lua, this, ()| {
            let g = this.0.read();
            let t = lua.create_table()?;
            t.set("hp_max", g.hp_max)?;
            t.set("breath_max", g.breath_max.max(1))?;
            t.set("eye_height", g.properties.get("eye_height").and_then(|s| s.parse::<f32>().ok()).unwrap_or(1.55))?;
            t.set("zoom_fov", 24.0f32)?;
            t.set("physical", g.physical)?;
            t.set("collide_with_objects", g.collide_with_objects)?;
            t.set("stepheight", g.stepheight)?;
            t.set("pointable", g.pointable)?;
            t.set("visual", g.visual.as_str())?;
            let mesh_str = if !g.mesh.is_empty() {
                g.mesh.as_str()
            } else {
                g.properties.get("mesh").map(String::as_str).unwrap_or("")
            };
            t.set("mesh", mesh_str)?;
            t.set("wield_item", g.wield_item.as_str())?;
            t.set("is_visible", g.is_visible)?;
            t.set("automatic_rotate", g.automatic_rotate)?;
            let nt = lua.create_table()?;
            nt.set("r", 255)?; nt.set("g", 255)?; nt.set("b", 255)?; nt.set("a", 255)?;
            t.set("nametag_color", nt)?;
            let vs = lua.create_table()?;
            vs.set("x", g.visual_size.x)?;
            vs.set("y", g.visual_size.y)?;
            vs.set("z", g.visual_size.z)?;
            t.set("visual_size", vs)?;
            let cb = lua.create_table()?;
            for (i, v) in g.collisionbox.iter().enumerate() {
                cb.set(i + 1, *v)?;
            }
            t.set("collisionbox", cb)?;
            let sb = lua.create_table()?;
            for (i, v) in g.selectionbox.iter().enumerate() {
                sb.set(i + 1, *v)?;
            }
            t.set("selectionbox", sb)?;
            let tex = lua.create_table()?;
            for (i, s) in g.textures.iter().enumerate() {
                tex.set(i + 1, s.as_str())?;
            }
            t.set("textures", tex)?;
            Ok(t)
        });
        methods.add_method("set_properties", |_, this, tbl: Table| {
            this.0.write().apply_properties_table(&tbl);
            Ok(())
        });
        methods.add_method("get_inventory", |lua, this, ()| {
            let g = this.0.read();
            if !g.is_player {
                return Ok(Value::Nil);
            }
            Ok(Value::UserData(lua.create_userdata(InvRefUD {
                state: g.meta.clone(),
                location: "player".into(),
            })?))
        });
        methods.add_method("get_wielded_item", |_, this, ()| {
            let g = this.0.read();
            let idx = g.wield_index.saturating_sub(1);
            let item = g
                .meta
                .read()
                .inventories
                .get("main")
                .and_then(|v| v.get(idx))
                .cloned()
                .unwrap_or_default();
            Ok(ItemStackUD(Arc::new(RwLock::new(item))))
        });
        methods.add_method("set_wielded_item", |_, this, val: Value| {
            let st = ItemStackUD::from_lua_value(val)?.0.read().clone();
            let g = this.0.read();
            let idx = g.wield_index.saturating_sub(1);
            let mut mg = g.meta.write();
            if let Some(vec) = mg.inventories.get_mut("main") {
                if idx < vec.len() {
                    vec[idx] = st;
                    return Ok(true);
                }
            }
            Ok(false)
        });
        methods.add_method("get_wield_list", |_, _, ()| Ok("main".to_string()));
        methods.add_method("get_wield_index", |_, this, ()| Ok(this.0.read().wield_index));
        methods.add_method("get_look_dir", |lua, this, ()| {
            let g = this.0.read();
            let cp = g.pitch.cos();
            let dir = Vec3::new(-g.yaw.sin() * cp, g.pitch.sin(), g.yaw.cos() * cp).normalize_or_zero();
            super::core_api::make_vec3(lua, dir.x as f64, dir.y as f64, dir.z as f64)
        });
        methods.add_method("get_look_horizontal", |_, this, ()| Ok(this.0.read().yaw));
        methods.add_method("get_look_vertical", |_, this, ()| Ok(this.0.read().pitch));
        methods.add_method("set_look_horizontal", |_, this, y: f32| {
            this.0.write().yaw = y;
            Ok(())
        });
        methods.add_method("set_look_vertical", |_, this, p: f32| {
            this.0.write().pitch = p;
            Ok(())
        });
        methods.add_method("get_player_control", |lua, this, ()| {
            let sneak = this.0.read().sneak;
            let t = lua.create_table()?;
            for k in ["up", "down", "left", "right", "jump", "aux1", "dig", "place", "zoom"] {
                t.set(k, false)?;
            }
            t.set("sneak", sneak)?;
            Ok(t)
        });
        methods.add_method("get_player_control_bits", |_, this, ()| {
            Ok(if this.0.read().sneak { 64u32 } else { 0u32 })
        });
        methods.add_method("get_meta", |_, this, ()| Ok(MetaDataRefUD(this.0.read().meta.clone())));
        methods.add_method("set_physics_override", |_, this, tbl: Table| {
            let mut g = this.0.write();
            if let Ok(s) = tbl.get::<f32>("speed") { g.speed_mult = s; }
            if let Ok(j) = tbl.get::<f32>("jump") { g.jump_mult = j; }
            if let Ok(gr) = tbl.get::<f32>("gravity") { g.gravity_mult = gr; }
            Ok(())
        });
        methods.add_method("get_physics_override", |lua, this, ()| {
            let g = this.0.read();
            let t = lua.create_table()?;
            t.set("speed", g.speed_mult)?;
            t.set("jump", g.jump_mult)?;
            t.set("gravity", g.gravity_mult)?;
            Ok(t)
        });
        methods.add_method("set_fov", |_, this, (fov, _, _): (f32, Option<bool>, Option<f32>)| {
            this.0.write().fov = fov;
            Ok(())
        });
        methods.add_method("get_fov", |_, this, ()| Ok((this.0.read().fov, false, 0.0f32)));
        methods.add_method("set_formspec_prepend", |_, this, s: String| {
            this.0.write().formspec_prepend = s;
            Ok(())
        });
        methods.add_method("get_formspec_prepend", |_, this, ()| Ok(this.0.read().formspec_prepend.clone()));
        methods.add_method("set_inventory_formspec", |_, this, s: String| {
            this.0.write().inventory_formspec = s;
            Ok(())
        });
        methods.add_method("get_inventory_formspec", |_, this, ()| Ok(this.0.read().inventory_formspec.clone()));
        methods.add_method("hud_add", |_, this, def: Table| {
            let mut g = this.0.write();
            let id = g.next_hud_id;
            g.next_hud_id += 1;
            let mut map = HashMap::new();
            if let Ok(t) = def.get::<String>("text") { map.insert("text".into(), t); }
            if let Ok(ty) = def.get::<String>("hud_elem_type") { map.insert("type".into(), ty); }
            g.hud_elems.insert(id, map);
            Ok(id)
        });
        methods.add_method("hud_remove", |_, this, id: u64| {
            this.0.write().hud_elems.remove(&id);
            Ok(())
        });
        methods.add_method("hud_change", |_, this, (id, stat, val): (u64, String, Value)| {
            let mut g = this.0.write();
            if let Some(m) = g.hud_elems.get_mut(&id) {
                if let Value::String(st) = val {
                    if let Ok(s) = st.to_str() {
                        m.insert(stat, s.to_string());
                    }
                }
            }
            Ok(())
        });
        methods.add_method("hud_get", |lua, this, id: u64| {
            let g = this.0.read();
            let t = lua.create_table()?;
            if let Some(m) = g.hud_elems.get(&id) {
                for (k, v) in m {
                    t.set(k.as_str(), v.as_str())?;
                }
            }
            Ok(t)
        });
        methods.add_method("hud_set_hotbar_itemcount", |_, this, c: usize| {
            this.0.write().hotbar_count = c.clamp(1, 32);
            Ok(())
        });
        methods.add_method("hud_get_hotbar_itemcount", |_, this, ()| Ok(this.0.read().hotbar_count));
        methods.add_method("hud_set_flags", |_, _, _: Table| Ok(()));
        methods.add_method("hud_get_flags", |lua, _, ()| {
            let t = lua.create_table()?;
            for k in ["hotbar", "healthbar", "crosshair", "wielditem", "breathbar", "minimap"] {
                t.set(k, true)?;
            }
            Ok(t)
        });
        methods.add_method("hud_set_hotbar_image", |_, _, _: String| Ok(()));
        methods.add_method("hud_set_hotbar_selected_image", |_, _, _: String| Ok(()));
        methods.add_method("set_sky", |_, _, _: MultiValue| Ok(()));
        methods.add_method("get_sky", |lua, _, _: Option<bool>| Ok(lua.create_table()?));
        methods.add_method("set_sun", |_, _, _: MultiValue| Ok(()));
        methods.add_method("set_moon", |_, _, _: MultiValue| Ok(()));
        methods.add_method("set_stars", |_, _, _: MultiValue| Ok(()));
        methods.add_method("set_clouds", |_, _, _: MultiValue| Ok(()));
        methods.add_method("set_lighting", |_, _, _: MultiValue| Ok(()));
        methods.add_method("set_eye_offset", |_, _, _: MultiValue| Ok(()));
        methods.add_method("get_eye_offset", |lua, _, ()| {
            let z = lua.create_table()?;
            z.set("x", 0)?; z.set("y", 0)?; z.set("z", 0)?;
            Ok((z.clone(), z))
        });
        methods.add_method("set_nametag_attributes", |_, _, _: Table| Ok(()));
        methods.add_method("get_nametag_attributes", |lua, _, ()| {
            let t = lua.create_table()?;
            t.set("text", "")?;
            Ok(t)
        });
        methods.add_method("set_texture_mod", |_, this, m: String| {
            let lower = m.to_ascii_lowercase();
            let mut g = this.0.write();
            if lower.contains("#d42222")
                || lower.contains("#ff0000")
                || lower.contains("colorize:#ff")
                || lower.contains("colorize:#d4")
                || lower.contains("colorize:red")
            {
                g.hurt_flash = 0.25;
            }
            g.properties.insert("texture_mod".into(), m);
            Ok(())
        });
        methods.add_method("get_texture_mod", |_, this, ()| {
            Ok(this.0.read().properties.get("texture_mod").cloned().unwrap_or_default())
        });
        methods.add_method("set_bone_position", |_, _, _: MultiValue| Ok(()));
        methods.add_method("get_bone_position", |lua, _, _bone: Option<String>| {
            let z = lua.create_table()?;
            z.set("x", 0.0f32)?; z.set("y", 0.0f32)?; z.set("z", 0.0f32)?;
            Ok((z.clone(), z))
        });
        methods.add_method("set_bone_position", |_, _, (_bone, _pos, _rot): (Option<String>, Option<Table>, Option<Table>)| {
            Ok(())
        });
        methods.add_method("set_bone_override", |_, _, _: MultiValue| Ok(()));
        methods.add_method("get_bone_override", |lua, _, _bone: Option<String>| {
            let out = lua.create_table()?;
            let mk = |x: f32, y: f32, z: f32| -> LuaResult<Table> {
                let entry = lua.create_table()?;
                let v = lua.create_table()?;
                v.set("x", x)?; v.set("y", y)?; v.set("z", z)?;
                entry.set("vec", v)?;
                entry.set("absolute", false)?;
                Ok(entry)
            };
            out.set("position", mk(0.0, 0.0, 0.0)?)?;
            out.set("rotation", mk(0.0, 0.0, 0.0)?)?;
            out.set("scale", mk(1.0, 1.0, 1.0)?)?;
            Ok(out)
        });
        methods.add_method(
            "set_animation",
            |_, this, (range, speed, blend, frame_loop): (Option<Table>, Option<f32>, Option<f32>, Option<bool>)| {
                let mut g = this.0.write();
                if let Some(rt) = range {
                    let x: f32 = rt.get("x").or_else(|_| rt.get(1)).unwrap_or(0.0);
                    let y: f32 = rt.get("y").or_else(|_| rt.get(2)).unwrap_or(x);
                    g.anim_range = (x, y);
                }
                if let Some(sp) = speed {
                    g.anim_speed = sp;
                }
                if let Some(bl) = blend {
                    g.anim_blend = bl;
                }
                if let Some(lp) = frame_loop {
                    g.anim_loop = lp;
                }
                Ok(())
            },
        );
        methods.add_method("get_animation", |lua, this, ()| {
            let g = this.0.read();
            let rt = lua.create_table()?;
            rt.set("x", g.anim_range.0)?;
            rt.set("y", g.anim_range.1)?;
            Ok((rt, g.anim_speed, g.anim_blend, g.anim_loop))
        });
        methods.add_method("set_animation_frame_speed", |_, this, speed: Option<f32>| {
            if let Some(sp) = speed {
                this.0.write().anim_speed = sp;
            }
            Ok(())
        });
        methods.add_method("set_local_animation", |_, _, _: MultiValue| Ok(()));
        methods.add_method("get_attach", |_, _, ()| Ok(Value::Nil));
        methods.add_method("set_attach", |_, _, _: MultiValue| Ok(()));
        methods.add_method("set_detach", |_, _, ()| Ok(()));
        methods.add_method("get_luaentity", |lua, this, ()| {
            let g = this.0.read();
            if g.is_player || g.id == 0 || g.removed {
                return Ok(Value::Nil);
            }
            if let Ok(core) = lua.globals().get::<Table>("core") {
                if let Ok(tbl) = core.get::<Table>("_cubex_luaentities") {
                    if let Ok(ent) = tbl.get::<Value>(g.id) {
                        return Ok(ent);
                    }
                }
            }
            Ok(Value::Nil)
        });
        methods.add_method("remove", |lua, this, ()| {
            let id = {
                let mut g = this.0.write();
                g.removed = true;
                g.id
            };
            if id != 0 {
                if let Ok(core) = lua.globals().get::<Table>("core") {
                    if let Ok(tbl) = core.get::<Table>("_cubex_luaentities") {
                        let _ = tbl.set(id, Value::Nil);
                    }
                }
                if let Ok(Value::Table(refs)) = lua.named_registry_value::<Value>("_cubex_object_refs") {
                    let _ = refs.set(format!("e:{id}"), Value::Nil);
                }
            }
            Ok(())
        });
        methods.add_method(
            "punch",
            |lua, this, (puncher, time_from_last_punch, tool_capabilities, dir): (Option<Value>, Option<f64>, Option<Value>, Option<Value>)| {
                let (is_player, id, removed, is_dead, invuln, target_pos, old_hp, armor_groups) = {
                    let g = this.0.read();
                    (
                        g.is_player,
                        g.id,
                        g.removed,
                        g.is_dead,
                        g.spawn_invuln_timer > 0.0,
                        g.pos,
                        g.hp,
                        g.armor_groups.clone(),
                    )
                };
                if removed || (is_player && (is_dead || invuln)) {
                    return Ok(0i32);
                }

                let tflp = time_from_last_punch.unwrap_or(1.0) as f32;
                let (full_interval, dmg_groups) =
                    parse_tool_capabilities_from_lua(tool_capabilities.as_ref());
                let raw_damage =
                    compute_punch_damage(tflp, full_interval, &dmg_groups, &armor_groups);
                let mut raw_armor_no_immortal = armor_groups.clone();
                if !raw_armor_no_immortal.contains_key("fleshy") {
                    raw_armor_no_immortal.insert("fleshy".into(), 100);
                }
                let callback_damage = compute_punch_damage(
                    tflp,
                    full_interval,
                    &dmg_groups,
                    &raw_armor_no_immortal,
                );

                let puncher_pos = puncher.as_ref().and_then(|v| match v {
                    Value::UserData(ud) => ud.borrow::<ObjectRefUD>().ok().map(|o| o.0.read().pos),
                    _ => None,
                });
                let dir_vec = parse_lua_vec3_opt(dir.as_ref())
                    .or_else(|| puncher_pos.map(|pp| (target_pos - pp).normalize_or_zero()))
                    .unwrap_or(Vec3::Z);
                let dir_tbl = match dir.as_ref() {
                    Some(Value::Table(t)) => Value::Table(t.clone()),
                    _ => Value::Table(super::core_api::make_vec3(
                        lua,
                        dir_vec.x as f64,
                        dir_vec.y as f64,
                        dir_vec.z as f64,
                    )?),
                };
                let tc_val = tool_capabilities.unwrap_or(Value::Nil);
                let puncher_val = puncher.unwrap_or(Value::Nil);

                if is_player {
                    let self_ud = object_ref_to_lua(lua, this.0.clone())?;
                    let mut handled_by_lua = false;
                    if let Ok(core) = lua.globals().get::<Table>("core") {
                        if let Ok(cbs) = core.get::<Table>("registered_on_punchplayers") {
                            let len = cbs.len().unwrap_or(0);
                            for i in 1..=len {
                                if let Ok(func) = cbs.get::<mlua::Function>(i) {
                                    if let Ok(ret) = func.call::<Value>((
                                        self_ud.clone(),
                                        puncher_val.clone(),
                                        tflp as f64,
                                        tc_val.clone(),
                                        dir_tbl.clone(),
                                        raw_damage,
                                    )) {
                                        if matches!(ret, Value::Boolean(true)) {
                                            handled_by_lua = true;
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if !handled_by_lua && raw_damage > 0 && armor_groups.get("immortal").copied().unwrap_or(0) == 0 {
                        let reason_t = lua.create_table()?;
                        reason_t.set("type", "punch")?;
                        reason_t.set("object", puncher_val)?;
                        apply_player_hp_change_in_lua(
                            lua,
                            &this.0,
                            (old_hp - raw_damage).max(0),
                            Some(Value::Table(reason_t)),
                        )?;
                    }

                    if raw_damage > 0 {
                        let horiz = Vec3::new(dir_vec.x, 0.0, dir_vec.z);
                        let hdir = if horiz.length_squared() > 1e-4 {
                            horiz.normalize()
                        } else {
                            Vec3::Z
                        };
                        let kb = Vec3::new(hdir.x * 5.5, 3.6, hdir.z * 5.5);
                        let mut g = this.0.write();
                        g.hurt_flash = 0.25;
                        g.vel += kb;
                        g.pending_knockback += kb;
                    }
                    return Ok(raw_damage);
                }

                if id != 0 {
                    let ent_opt = lua
                        .globals()
                        .get::<Table>("core")
                        .ok()
                        .and_then(|c| c.get::<Table>("_cubex_luaentities").ok())
                        .and_then(|tbl| tbl.get::<Table>(id).ok());

                    if let Some(ent) = ent_opt {
                        let old_lua_health: Option<f64> = ent.get("health").ok();
                        let old_vel = this.0.read().vel;
                        let mut prevent_default = false;
                        if let Ok(on_punch) = ent.get::<mlua::Function>("on_punch") {
                            let res = on_punch.call::<Value>((
                                ent.clone(),
                                puncher_val.clone(),
                                tflp as f64,
                                tc_val,
                                dir_tbl,
                                callback_damage,
                            ));
                            if let Ok(Value::Boolean(true)) = res {
                                prevent_default = true;
                            }
                        }

                        let new_lua_health: Option<f64> = ent.get("health").ok();
                        let cur_vel = {
                            let mut g = this.0.write();
                            if let Some(lh) = new_lua_health {
                                g.hp = (lh.ceil() as i32).clamp(0, g.hp_max);
                            }
                            let lua_damaged = matches!(
                                (old_lua_health, new_lua_health),
                                (Some(a), Some(b)) if b < a
                            );
                            if g.hp < old_hp || lua_damaged || (!prevent_default && raw_damage > 0) {
                                g.hurt_flash = 0.25;
                            }
                            g.vel
                        };

                        let is_immortal = armor_groups.get("immortal").copied().unwrap_or(0) > 0;
                        if !prevent_default && !is_immortal && raw_damage > 0 {
                            let mut g = this.0.write();
                            g.hp = (g.hp - raw_damage).max(0);
                            if (cur_vel - old_vel).length_squared() < 0.01 {
                                let horiz = Vec3::new(dir_vec.x, 0.0, dir_vec.z);
                                let hdir = if horiz.length_squared() > 1e-4 {
                                    horiz.normalize()
                                } else {
                                    Vec3::Z
                                };
                                g.vel += Vec3::new(hdir.x * 5.5, 3.2, hdir.z * 5.5);
                            }
                        }

                        if this.0.read().hp <= 0 && !this.0.read().removed {
                            despawn_lua_entity(lua, &this.0, Some(puncher_val));
                        }
                    }
                }
                Ok(raw_damage)
            },
        );
        methods.add_method("override_day_night_ratio", |_, _, _: Option<f32>| Ok(()));
        methods.add_method("set_minimap_modes", |_, _, _: MultiValue| Ok(()));
        methods.add_method("send_mapblock", |_, _, _: Table| Ok(true));
        methods.add_method("get_player_velocity", |lua, this, ()| {
            let v = this.0.read().vel;
            let t = lua.create_table()?;
            t.set("x", v.x)?; t.set("y", v.y)?; t.set("z", v.z)?;
            Ok(t)
        });
        methods.add_method("add_player_velocity", |_, this, v: Table| {
            let dx: f32 = v.get("x").unwrap_or(0.0);
            let dy: f32 = v.get("y").unwrap_or(0.0);
            let dz: f32 = v.get("z").unwrap_or(0.0);
            let delta = Vec3::new(dx, dy, dz);
            let mut g = this.0.write();
            g.vel += delta;
            if g.is_player {
                g.pending_knockback += delta;
            }
            Ok(())
        });
        methods.add_meta_method(MetaMethod::Eq, |_, this, other: AnyUserData| {
            if let Ok(o) = other.borrow::<ObjectRefUD>() {
                Ok(Arc::ptr_eq(&this.0, &o.0))
            } else {
                Ok(false)
            }
        });
    }
}

#[derive(Clone, Default)]
pub struct AreaStoreUD(pub Arc<RwLock<Vec<(u32, IVec3, IVec3, String)>>>);

impl UserData for AreaStoreUD {
    fn add_methods<M: UserDataMethods<Self>>(methods: &mut M) {
        methods.add_method("insert_area", |_, this, (e1, e2, data, id): (Table, Table, String, Option<u32>)| {
            let mut g = this.0.write();
            let nid = id.unwrap_or(g.len() as u32 + 1);
            g.push((nid, lua_to_ivec3(&e1), lua_to_ivec3(&e2), data));
            Ok(nid)
        });
        methods.add_method("remove_area", |_, this, id: u32| {
            let mut g = this.0.write();
            let before = g.len();
            g.retain(|x| x.0 != id);
            Ok(g.len() < before)
        });
        methods.add_method("get_areas_for_pos", |lua, this, (pos, _inc_e, _inc_d): (Table, Option<bool>, Option<bool>)| {
            let p = lua_to_ivec3(&pos);
            let out = lua.create_table()?;
            for (id, a, b, data) in this.0.read().iter() {
                let lo = a.min(*b);
                let hi = a.max(*b);
                if p.x >= lo.x && p.x <= hi.x && p.y >= lo.y && p.y <= hi.y && p.z >= lo.z && p.z <= hi.z {
                    let row = lua.create_table()?;
                    row.set("data", data.as_str())?;
                    out.set(*id, row)?;
                }
            }
            Ok(out)
        });
        methods.add_method("get_areas_in_area", |lua, this, (e1, e2, _acc, _inc_e, _inc_d): (Table, Table, Option<bool>, Option<bool>, Option<bool>)| {
            let p1 = lua_to_ivec3(&e1);
            let p2 = lua_to_ivec3(&e2);
            let qlo = p1.min(p2);
            let qhi = p1.max(p2);
            let out = lua.create_table()?;
            for (id, a, b, data) in this.0.read().iter() {
                let lo = a.min(*b);
                let hi = a.max(*b);
                if lo.x <= qhi.x && hi.x >= qlo.x && lo.y <= qhi.y && hi.y >= qlo.y && lo.z <= qhi.z && hi.z >= qlo.z {
                    let row = lua.create_table()?;
                    row.set("data", data.as_str())?;
                    out.set(*id, row)?;
                }
            }
            Ok(out)
        });
        methods.add_method("set_cache_params", |_, _, _: Table| Ok(()));
        methods.add_method("to_string", |_, _, ()| Ok(String::new()));
        methods.add_method("from_string", |_, _, _: String| Ok(true));
    }
}

#[derive(Clone)]
pub struct RaycastUD;

impl UserData for RaycastUD {
    fn add_methods<M: UserDataMethods<Self>>(methods: &mut M) {
        methods.add_method("next", |_, _, ()| Ok(Value::Nil));
        methods.add_meta_method(MetaMethod::Call, |_, _, ()| Ok(Value::Nil));
    }
}

pub fn register_userdata_ctors(
    lua: &Lua,
    world: Arc<WorldStore>,
    registries: Arc<RwLock<EngineRegistries>>,
) -> LuaResult<()> {
    let g = lua.globals();

    g.set("ItemStack", lua.create_function(|_, val: Value| ItemStackUD::from_lua_value(val))?)?;

    g.set("Settings", lua.create_function(|_, path: String| {
        Ok(SettingsUD(Arc::new(RwLock::new(MinetestConfig::load(path)))))
    })?)?;

    g.set("AreaStore", lua.create_function(|_, ()| Ok(AreaStoreUD::default()))?)?;

    g.set("Raycast", lua.create_function(|_, _: MultiValue| Ok(RaycastUD))?)?;

    let w = world.clone();
    let r = registries.clone();
    g.set("VoxelManip", lua.create_function(move |_, (p1, p2): (Option<Table>, Option<Table>)| {
        Ok(VoxelManipUD::new(
            w.clone(),
            r.clone(),
            p1.as_ref().map(lua_to_ivec3),
            p2.as_ref().map(lua_to_ivec3),
        ))
    })?)?;

    Ok(())
}
