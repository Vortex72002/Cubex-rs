use mlua::{Lua, Result as LuaResult, Table, Value};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::core_api::LuaEngineShared;

#[derive(Debug, Clone)]
pub struct ModSpec {
    pub name: String,
    pub path: PathBuf,
    pub depends: Vec<String>,
    pub optional_depends: Vec<String>,
}

fn parse_conf_kv(path: &Path) -> HashMap<String, String> {
    let mut out = HashMap::new();
    if let Ok(text) = std::fs::read_to_string(path) {
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            if let Some((k, v)) = trimmed.split_once('=') {
                out.insert(k.trim().to_string(), v.trim().to_string());
            }
        }
    }
    out
}

fn parse_dep_list(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
        .collect()
}

fn discover_mods_recursive(dir: &Path, found: &mut BTreeMap<String, ModSpec>) {
    let init_lua = dir.join("init.lua");
    if init_lua.is_file() {
        let mod_conf = parse_conf_kv(&dir.join("mod.conf"));
        let dir_name = dir
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        let name = mod_conf
            .get("name")
            .filter(|s| !s.is_empty())
            .cloned()
            .unwrap_or(dir_name);

        let mut depends = mod_conf
            .get("depends")
            .map(|s| parse_dep_list(s))
            .unwrap_or_default();
        let mut optional_depends = mod_conf
            .get("optional_depends")
            .map(|s| parse_dep_list(s))
            .unwrap_or_default();

        if let Ok(dtxt) = std::fs::read_to_string(dir.join("depends.txt")) {
            for line in dtxt.lines() {
                let t = line.trim();
                if t.is_empty() || t.starts_with('#') {
                    continue;
                }
                if let Some(opt) = t.strip_suffix('?') {
                    optional_depends.push(opt.trim().to_string());
                } else {
                    depends.push(t.to_string());
                }
            }
        }

        found.insert(
            name.clone(),
            ModSpec {
                name,
                path: dir.to_path_buf(),
                depends,
                optional_depends,
            },
        );
        return;
    }

    if let Ok(entries) = std::fs::read_dir(dir) {
        let mut subdirs: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect();
        subdirs.sort();
        for sub in subdirs {
            discover_mods_recursive(&sub, found);
        }
    }
}

pub fn discover_and_sort_mods(game_dir: &Path, override_dir: Option<&Path>) -> Vec<ModSpec> {
    let game_conf = parse_conf_kv(&game_dir.join("game.conf"));
    let first_mod = game_conf.get("first_mod").cloned();
    let last_mod = game_conf.get("last_mod").cloned();

    let mut mods: BTreeMap<String, ModSpec> = BTreeMap::new();
    let mods_root = game_dir.join("mods");
    if mods_root.is_dir() {
        discover_mods_recursive(&mods_root, &mut mods);
    }
    if let Some(ov) = override_dir {
        if ov.is_dir() && ov.join("init.lua").is_file() {
            discover_mods_recursive(ov, &mut mods);
        }
    }

    let names: Vec<String> = mods.keys().cloned().collect();
    let present: HashSet<String> = names.iter().cloned().collect();

    let mut edges: HashMap<String, Vec<String>> = HashMap::new();
    let mut indeg: HashMap<String, usize> = HashMap::new();
    for n in &names {
        edges.insert(n.clone(), Vec::new());
        indeg.insert(n.clone(), 0);
    }

    for (name, spec) in &mods {
        let mut all_deps = HashSet::new();
        for d in &spec.depends {
            if present.contains(d) && d != name {
                all_deps.insert(d.clone());
            }
        }
        for d in &spec.optional_depends {
            if present.contains(d) && d != name {
                all_deps.insert(d.clone());
            }
        }
        if let Some(ref fm) = first_mod {
            if name != fm && present.contains(fm) {
                all_deps.insert(fm.clone());
            }
        }
        if let Some(ref lm) = last_mod {
            if name == lm {
                for other in &names {
                    if other != lm {
                        all_deps.insert(other.clone());
                    }
                }
            }
        }
        for dep in all_deps {
            edges.get_mut(&dep).unwrap().push(name.clone());
            *indeg.get_mut(name).unwrap() += 1;
        }
    }

    let mut queue: VecDeque<String> = VecDeque::new();
    for n in &names {
        if indeg[n] == 0 {
            queue.push_back(n.clone());
        }
    }

    let mut sorted = Vec::with_capacity(names.len());
    let mut visited = HashSet::new();
    while let Some(u) = queue.pop_front() {
        if !visited.insert(u.clone()) {
            continue;
        }
        if let Some(spec) = mods.get(&u) {
            sorted.push(spec.clone());
        }
        if let Some(nexts) = edges.get(&u) {
            for v in nexts {
                let d = indeg.get_mut(v).unwrap();
                *d = d.saturating_sub(1);
                if *d == 0 {
                    queue.push_back(v.clone());
                }
            }
        }
    }

    // Append any remaining mods in case of dependency cycle
    for n in names {
        if !visited.contains(&n) {
            if let Some(spec) = mods.get(&n) {
                sorted.push(spec.clone());
            }
        }
    }

    sorted
}

#[derive(Debug, Clone)]
pub struct ModLoadReport {
    pub mods_discovered: usize,
    pub mods_loaded: usize,
    pub errors: Vec<(String, String)>,
}

pub fn prepare_mod_environment(
    lua: &Lua,
    shared: &Arc<LuaEngineShared>,
    mods: &[ModSpec],
) -> LuaResult<()> {
    // Populate mod_paths & mod_order BEFORE executing mcl_init (mcl_init inspects core.get_modnames()!)
    {
        let mut mp = shared.mod_paths.write();
        let mut mo = shared.mod_order.write();
        mp.clear();
        mo.clear();
        for m in mods {
            mp.insert(m.name.clone(), m.path.clone());
            mo.push(m.name.clone());
        }
    }

    // Bootstrap Luanti builtin/init.lua
    let builtin_init = shared.paths.builtin_dir.join("init.lua");
    let builtin_code = std::fs::read_to_string(&builtin_init).map_err(|e| {
        mlua::Error::RuntimeError(format!(
            "Failed to read builtin/init.lua at {}: {e}",
            builtin_init.display()
        ))
    })?;
    lua.load(&builtin_code)
        .set_name(builtin_init.to_string_lossy().as_ref())
        .exec()?;
    Ok(())
}

pub fn load_single_mod(lua: &Lua, shared: &Arc<LuaEngineShared>, m: &ModSpec) -> Result<(), String> {
    *shared.current_modname.write() = Some(m.name.clone());
    *shared.last_run_mod.write() = Some(m.name.clone());
    let init_path = m.path.join("init.lua");
    let code = std::fs::read_to_string(&init_path).map_err(|e| format!("read error: {e}"))?;
    lua.load(&code)
        .set_name(init_path.to_string_lossy().as_ref())
        .exec()
        .map_err(|err| {
            tracing::error!("Mod '{}' failed to load: {}", m.name, err);
            err.to_string()
        })
}

pub fn finish_mod_loading(
    lua: &Lua,
    shared: &Arc<LuaEngineShared>,
    mods_discovered: usize,
    mods_loaded: usize,
    mut errors: Vec<(String, String)>,
) -> ModLoadReport {
    *shared.current_modname.write() = None;

    // Run registered_on_mods_loaded callbacks
    if let Ok(core) = lua.globals().get::<Table>("core") {
        if let Ok(cbs) = core.get::<Table>("registered_on_mods_loaded") {
            let len = cbs.len().unwrap_or(0);
            for i in 1..=len {
                if let Ok(func) = cbs.get::<mlua::Function>(i) {
                    if let Err(e) = func.call::<()>(()) {
                        errors.push(("on_mods_loaded".into(), e.to_string()));
                    }
                }
            }
        }
        // Pump initial core.after(0) queue (e.g., builtin_shared.cache_content_ids)
        if let Ok(cbs) = core.get::<Table>("registered_globalsteps") {
            let len = cbs.len().unwrap_or(0);
            for i in 1..=len {
                if let Ok(func) = cbs.get::<mlua::Function>(i) {
                    let _ = func.call::<()>(0.05f64);
                }
            }
        }
    }

    // Sync any items/nodes registered in Lua tables directly (if mods mutated registered_nodes)
    if let Ok(core) = lua.globals().get::<Table>("core") {
        if let Ok(rn) = core.get::<Table>("registered_nodes") {
            let reg = shared.registries.read();
            for pair in rn.pairs::<String, Value>() {
                if let Ok((name, Value::Table(_))) = pair {
                    if !reg.name_to_id.contains_key(&name) {
                        tracing::debug!("Node {} in Lua table only", name);
                    }
                }
            }
        }
    }

    ModLoadReport {
        mods_discovered,
        mods_loaded,
        errors,
    }
}

pub fn load_all_mods(lua: &Lua, shared: &Arc<LuaEngineShared>) -> LuaResult<ModLoadReport> {
    let ov = if shared.paths.lua_override_dir.is_dir() {
        Some(shared.paths.lua_override_dir.as_path())
    } else {
        None
    };
    let mods = discover_and_sort_mods(&shared.paths.game_dir, ov);
    prepare_mod_environment(lua, shared, &mods)?;

    let mut errors = Vec::new();
    let mut loaded = 0usize;
    for m in &mods {
        match load_single_mod(lua, shared, m) {
            Ok(()) => loaded += 1,
            Err(err) => errors.push((m.name.clone(), err)),
        }
    }

    Ok(finish_mod_loading(lua, shared, mods.len(), loaded, errors))
}
