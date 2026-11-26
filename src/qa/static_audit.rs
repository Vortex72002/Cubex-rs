use std::collections::HashSet;

use glam::IVec3;
use serde::{Deserialize, Serialize};

use crate::lua_api::LuaRuntime;
use crate::physics::collision::{is_liquid_node, world_collision_boxes};
use crate::physics::raycast::is_chest_node_name;
use crate::registry::{
    is_safe_terrain_node, DrawType, EngineRegistries, ItemDef, NodeDef, NodeDrop, CONTENT_AIR,
    CONTENT_IGNORE, CONTENT_UNKNOWN, NETHER_PORTAL_NODE_ID,
};
use crate::textures::TextureAtlas;
use crate::world::mapgen::MapgenPalette;
use crate::world::nether::NetherPalette;
use crate::world::portal::is_nether_portal_node;
use crate::world::structures::{loot_pool_candidates, LootTableKind, StructurePalette};
use crate::world::WorldStore;

#[inline]
fn node_def_by_name<'a>(reg: &'a EngineRegistries, name: &str) -> Option<&'a NodeDef> {
    let id = reg.get_content_id(name);
    if id == CONTENT_UNKNOWN {
        None
    } else {
        reg.get_node_def(id)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractiveHandlerCheck {
    pub category: String,
    pub node_id: String,
    pub registered: bool,
    pub has_handler: bool,
    pub handler_kind: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StaticAuditReport {
    pub passed: bool,
    pub total_nodes: usize,
    pub total_items: usize,
    pub total_tools: usize,
    pub total_foods: usize,
    pub total_recipes: usize,
    pub loot_entries_checked: usize,
    pub missing_core_textures: Vec<String>,
    pub missing_optional_texture_count: usize,
    pub missing_optional_texture_samples: Vec<String>,
    pub empty_aabb_blocks: Vec<String>,
    pub damaging_safe_terrain_blocks: Vec<String>,
    pub interactive_checks: Vec<InteractiveHandlerCheck>,
    pub missing_loot_items: Vec<String>,
    pub reachable_item_count: usize,
    pub craft_graph_iterations: usize,
    pub unreachable_milestones: Vec<String>,
    pub failures: Vec<String>,
}

const CORE_AUDIT_ITEMS: &[&str] = &[
    "mcl_core:stone",
    "mcl_core:cobble",
    "mcl_core:dirt",
    "mcl_core:dirt_with_grass",
    "mcl_core:sand",
    "mcl_core:gravel",
    "mcl_core:bedrock",
    "mcl_core:obsidian",
    "mcl_core:water_source",
    "mcl_core:lava_source",
    "mcl_core:stick",
    "mcl_core:coal_lump",
    "mcl_core:iron_ingot",
    "mcl_tools:pick_wood",
    "mcl_tools:pick_stone",
    "mcl_tools:pick_iron",
    "mcl_tools:sword_stone",
    "mcl_crafting_table:crafting_table",
    "mcl_furnaces:furnace",
    "mcl_buckets:bucket_empty",
    "mcl_buckets:bucket_water",
    "mcl_buckets:bucket_lava",
    "mcl_fire:flint_and_steel",
    "mcl_core:flint",
    "mcl_chests:chest",
    "mcl_nether:netherrack",
    "mcl_nether:nether_brick",
    "mcl_nether:glowstone",
    "mcl_ocean:prismarine",
    "mcl_ocean:sea_lantern",
    NETHER_PORTAL_NODE_ID,
    "cubex:ram_mic",
];

/// Phase 1: Inspect every registered node, item, tool, food, craft recipe, texture,
/// solid collision box, interactive handler, structure loot table, and survival reachability graph.
pub fn run_static_audit(
    reg: &EngineRegistries,
    atlas: &TextureAtlas,
    lua_rt: Option<&LuaRuntime>,
) -> StaticAuditReport {
    let mut failures = Vec::new();
    let mut total_nodes = 0usize;
    let mut total_tools = 0usize;
    let mut total_foods = 0usize;
    let mut missing_core_textures = Vec::new();
    let mut missing_optional_texture_count = 0usize;
    let mut missing_optional_texture_samples = Vec::new();
    let mut empty_aabb_blocks = Vec::new();
    let mut damaging_safe_terrain_blocks = Vec::new();

    let tmp_world_dir = std::env::temp_dir().join("cubex_qa_static_audit_world");
    let _ = std::fs::remove_dir_all(&tmp_world_dir);
    let dummy_world = WorldStore::new(&tmp_world_dir, 1);

    // 1. Scan all registered nodes
    for slot in &reg.nodes_by_id {
        let Some(def) = slot.as_ref() else {
            continue;
        };
        if def.id == CONTENT_AIR || def.id == CONTENT_IGNORE || def.id == CONTENT_UNKNOWN {
            continue;
        }
        if def.name.is_empty() || def.name == "air" || def.name == "ignore" || def.name == "unknown" {
            continue;
        }
        total_nodes += 1;

        if def.walkable && !is_liquid_node(def) && def.drawtype != DrawType::Airlike {
            let boxes = world_collision_boxes(IVec3::ZERO, def.id, 0, &dummy_world, reg);
            let valid = !boxes.is_empty()
                && boxes.iter().all(|b| {
                    b.min.is_finite()
                        && b.max.is_finite()
                        && b.max.x > b.min.x
                        && b.max.y > b.min.y
                        && b.max.z > b.min.z
                });
            if !valid {
                empty_aabb_blocks.push(def.name.clone());
            }
        }

        if is_safe_terrain_node(&def.name, &def.groups) && def.damage_per_second != 0 {
            damaging_safe_terrain_blocks.push(format!("{} (dps={})", def.name, def.damage_per_second));
        }

        check_node_textures(
            def,
            atlas,
            &mut missing_core_textures,
            &mut missing_optional_texture_count,
            &mut missing_optional_texture_samples,
        );
    }

    let _ = std::fs::remove_dir_all(&tmp_world_dir);

    // 2. Scan items, tools, and foods
    let mut sorted_items: Vec<&ItemDef> = reg.items.values().collect();
    sorted_items.sort_by(|a, b| a.name.cmp(&b.name));
    for item in &sorted_items {
        if item.name.is_empty() || item.name == "air" || item.name == "ignore" {
            continue;
        }
        if item.item_type == "tool" || item.tool_capabilities.is_some() {
            total_tools += 1;
        }
        if is_food_item(item) {
            total_foods += 1;
        }
        if is_core_item(&item.name) && node_def_by_name(reg, &item.name).is_none() {
            check_item_textures(
                item,
                reg,
                atlas,
                &mut missing_core_textures,
                &mut missing_optional_texture_count,
                &mut missing_optional_texture_samples,
            );
        }
    }

    // Also ensure core wood/stone/ore aliases resolve to valid atlas textures
    for &core_name in CORE_AUDIT_ITEMS {
        let resolved = reg.resolve_alias(core_name);
        if let Some(def) = node_def_by_name(reg, resolved) {
            check_node_textures(
                def,
                atlas,
                &mut missing_core_textures,
                &mut missing_optional_texture_count,
                &mut missing_optional_texture_samples,
            );
        } else if let Some(item) = reg.get_item_def(resolved) {
            check_item_textures(
                item,
                reg,
                atlas,
                &mut missing_core_textures,
                &mut missing_optional_texture_count,
                &mut missing_optional_texture_samples,
            );
        }
    }

    missing_core_textures.sort();
    missing_core_textures.dedup();

    // 3. Interactive block handlers check across all 10 required categories
    let interactive_checks = audit_interactive_handlers(reg, lua_rt);
    for chk in &interactive_checks {
        if !chk.registered {
            failures.push(format!("Interactive category '{}' has no registered node", chk.category));
        } else if !chk.has_handler {
            failures.push(format!(
                "Interactive block '{}' ({}) lacks interaction/physics handler",
                chk.node_id, chk.category
            ));
        }
    }

    // 4. Loot tables check
    let mut loot_entries_checked = 0usize;
    let mut missing_loot_items = Vec::new();
    let mut resolved_loot_items = HashSet::new();
    for kind in LootTableKind::ALL {
        for candidates in loot_pool_candidates(kind) {
            loot_entries_checked += 1;
            let resolved = candidates
                .iter()
                .map(|c| reg.resolve_alias(c))
                .find(|r| reg.get_item_def(r).is_some() || node_def_by_name(reg, r).is_some());
            match resolved {
                Some(id) => {
                    resolved_loot_items.insert(id.to_string());
                }
                None => {
                    missing_loot_items.push(format!("{:?}: {:?}", kind, candidates));
                }
            }
        }
    }

    // 5. Survival crafting reachability graph
    let (reachable, craft_graph_iterations, unreachable_milestones) =
        compute_craft_reachability(reg, &resolved_loot_items);

    if !missing_core_textures.is_empty() {
        failures.push(format!(
            "Missing {} core texture(s): {}",
            missing_core_textures.len(),
            missing_core_textures.join(", ")
        ));
    }
    if !empty_aabb_blocks.is_empty() {
        failures.push(format!(
            "{} walkable block(s) had empty/invalid AABB(s): {}",
            empty_aabb_blocks.len(),
            empty_aabb_blocks.iter().take(8).cloned().collect::<Vec<_>>().join(", ")
        ));
    }
    if !damaging_safe_terrain_blocks.is_empty() {
        failures.push(format!(
            "{} safe terrain block(s) deal contact damage: {}",
            damaging_safe_terrain_blocks.len(),
            damaging_safe_terrain_blocks.join(", ")
        ));
    }
    if !missing_loot_items.is_empty() {
        failures.push(format!(
            "{} structure loot table entry(s) failed to resolve: {}",
            missing_loot_items.len(),
            missing_loot_items.join("; ")
        ));
    }
    if !unreachable_milestones.is_empty() {
        failures.push(format!(
            "Crafting reachability dead-end(s): {}",
            unreachable_milestones.join(", ")
        ));
    }

    StaticAuditReport {
        passed: failures.is_empty(),
        total_nodes,
        total_items: reg.items.len(),
        total_tools,
        total_foods,
        total_recipes: reg.crafts.len(),
        loot_entries_checked,
        missing_core_textures,
        missing_optional_texture_count,
        missing_optional_texture_samples,
        empty_aabb_blocks,
        damaging_safe_terrain_blocks,
        interactive_checks,
        missing_loot_items,
        reachable_item_count: reachable.len(),
        craft_graph_iterations,
        unreachable_milestones,
        failures,
    }
}

fn is_food_item(item: &ItemDef) -> bool {
    if item.groups.get("food").copied().unwrap_or(0) > 0
        || item.groups.get("eatable").copied().unwrap_or(0) > 0
    {
        return true;
    }
    let l = item.name.to_ascii_lowercase();
    l.contains("mcl_core:apple")
        || l.contains("mcl_farming:bread")
        || l.contains("mcl_mobitems:cooked_")
}

fn is_core_item(name: &str) -> bool {
    if CORE_AUDIT_ITEMS.contains(&name) {
        return true;
    }
    let l = name.to_ascii_lowercase();
    l.starts_with("mcl_trees:tree_")
        || l.starts_with("mcl_trees:wood_")
        || l.contains("raw_iron")
        || l.contains("stone_with_iron")
        || l.contains("stone_with_coal")
}

fn check_node_textures(
    def: &NodeDef,
    atlas: &TextureAtlas,
    missing_core: &mut Vec<String>,
    opt_count: &mut usize,
    opt_samples: &mut Vec<String>,
) {
    if def.drawtype == DrawType::Airlike {
        return;
    }
    let mut specs: Vec<&str> = def
        .tiles
        .iter()
        .map(|t| t.name.trim())
        .filter(|s| !s.is_empty())
        .collect();
    if specs.is_empty() && !def.inventory_image.trim().is_empty() {
        specs.push(def.inventory_image.trim());
    }
    let is_core = is_core_item(&def.name);
    if specs.is_empty() && !matches!(def.drawtype, DrawType::Mesh) {
        record_missing_tex(
            format!("{} (no tiles)", def.name),
            is_core,
            missing_core,
            opt_count,
            opt_samples,
        );
        return;
    }
    for spec in specs {
        if atlas.resolve_spec_immut(spec).is_none() {
            record_missing_tex(
                format!("{} -> {}", def.name, spec),
                is_core,
                missing_core,
                opt_count,
                opt_samples,
            );
        }
    }
}

fn check_item_textures(
    item: &ItemDef,
    reg: &EngineRegistries,
    atlas: &TextureAtlas,
    missing_core: &mut Vec<String>,
    opt_count: &mut usize,
    opt_samples: &mut Vec<String>,
) {
    let spec = if !item.inventory_image.trim().is_empty() {
        item.inventory_image.trim().to_string()
    } else if !item.wield_image.trim().is_empty() {
        item.wield_image.trim().to_string()
    } else if let Some(def) = node_def_by_name(reg, &item.name) {
        if let Some(t) = def.tiles.first() {
            t.name.trim().to_string()
        } else {
            String::new()
        }
    } else {
        String::new()
    };

    let is_core = is_core_item(&item.name);
    if spec.is_empty() {
        if item.name != "cubex:ram_mic" {
            record_missing_tex(
                format!("{} (no inventory_image)", item.name),
                is_core,
                missing_core,
                opt_count,
                opt_samples,
            );
        }
        return;
    }
    if atlas.resolve_spec_immut(&spec).is_none() {
        record_missing_tex(
            format!("{} -> {}", item.name, spec),
            is_core,
            missing_core,
            opt_count,
            opt_samples,
        );
    }
}

fn record_missing_tex(
    label: String,
    is_core: bool,
    missing_core: &mut Vec<String>,
    opt_count: &mut usize,
    opt_samples: &mut Vec<String>,
) {
    if is_core {
        missing_core.push(label);
    } else {
        *opt_count += 1;
        if opt_samples.len() < 16 {
            opt_samples.push(label);
        }
    }
}

fn lua_node_callbacks(lua_rt: Option<&LuaRuntime>, node_name: &str) -> Vec<&'static str> {
    let mut found = Vec::new();
    let Some(rt) = lua_rt else {
        return found;
    };
    let Ok(core) = rt.lua.globals().get::<mlua::Table>("core") else {
        return found;
    };
    let Ok(rn) = core.get::<mlua::Table>("registered_nodes") else {
        return found;
    };
    let Ok(def) = rn.get::<mlua::Table>(node_name) else {
        return found;
    };
    for key in [
        "on_rightclick",
        "on_construct",
        "on_timer",
        "allow_metadata_inventory_put",
    ] {
        if def.get::<mlua::Function>(key).is_ok() {
            found.push(key);
        }
    }
    found
}

fn find_first_matching_node(
    reg: &EngineRegistries,
    exact: &[&str],
    predicate: impl Fn(&NodeDef) -> bool,
) -> Option<String> {
    for &name in exact {
        let resolved = reg.resolve_alias(name);
        if node_def_by_name(reg, resolved).is_some() {
            return Some(resolved.to_string());
        }
    }
    reg.nodes_by_id
        .iter()
        .flatten()
        .find(|d| predicate(d))
        .map(|d| d.name.clone())
}

fn audit_interactive_handlers(
    reg: &EngineRegistries,
    lua_rt: Option<&LuaRuntime>,
) -> Vec<InteractiveHandlerCheck> {
    struct CategorySpec<'a> {
        category: &'a str,
        exact: &'a [&'a str],
    }
    let specs = [
        CategorySpec {
            category: "chest",
            exact: &["mcl_chests:chest", "mcl_chests:chest_small", "default:chest"],
        },
        CategorySpec {
            category: "furnace",
            exact: &["mcl_furnaces:furnace", "default:furnace"],
        },
        CategorySpec {
            category: "crafting_table",
            exact: &["mcl_crafting_table:crafting_table", "crafting:workbench"],
        },
        CategorySpec {
            category: "bed",
            exact: &[
                "mcl_beds:bed_red_bottom",
                "mcl_beds:bed_white_bottom",
                "beds:bed_bottom",
            ],
        },
        CategorySpec {
            category: "door",
            exact: &[
                "mcl_doors:wooden_door_b_1",
                "mcl_doors:door_oak_b_1",
                "doors:door_wood_a",
            ],
        },
        CategorySpec {
            category: "nether_portal",
            exact: &[NETHER_PORTAL_NODE_ID, "mcl_portals:portal"],
        },
        CategorySpec {
            category: "water",
            exact: &["mcl_core:water_source", "default:water_source"],
        },
        CategorySpec {
            category: "lava",
            exact: &["mcl_core:lava_source", "default:lava_source"],
        },
        CategorySpec {
            category: "sand",
            exact: &["mcl_core:sand", "default:sand"],
        },
        CategorySpec {
            category: "gravel",
            exact: &["mcl_core:gravel", "default:gravel"],
        },
    ];

    let mut checks = Vec::with_capacity(specs.len());
    for sp in specs {
        let matched = find_first_matching_node(reg, sp.exact, |d| {
            let l = d.name.to_ascii_lowercase();
            match sp.category {
                "chest" => is_chest_node_name(&l),
                "furnace" => l.contains("furnace"),
                "crafting_table" => l.contains("crafting_table"),
                "bed" => l.starts_with("mcl_beds:bed_") || l.starts_with("beds:bed"),
                "door" => (l.starts_with("mcl_doors:") || l.starts_with("doors:")) && l.contains("door"),
                "nether_portal" => l.contains("portal"),
                "water" => l.contains("water_source"),
                "lava" => l.contains("lava_source"),
                "sand" => l.ends_with(":sand"),
                "gravel" => l.ends_with(":gravel"),
                _ => false,
            }
        });

        let Some(node_id) = matched else {
            checks.push(InteractiveHandlerCheck {
                category: sp.category.into(),
                node_id: sp.exact.first().copied().unwrap_or("").into(),
                registered: false,
                has_handler: false,
                handler_kind: "unregistered".into(),
            });
            continue;
        };

        let def = node_def_by_name(reg, &node_id);
        let cbs = lua_node_callbacks(lua_rt, &node_id);
        let (has_handler, handler_kind) = match sp.category {
            "chest" => {
                let native = is_chest_node_name(&node_id);
                (native || !cbs.is_empty(), format!("native_chest({native})+lua{cbs:?}"))
            }
            "furnace" => {
                let ok = if lua_rt.is_some() {
                    !cbs.is_empty()
                } else {
                    def.is_some()
                };
                (ok, if cbs.is_empty() { "registered_furnace".into() } else { format!("lua{cbs:?}") })
            }
            "crafting_table" => {
                let ok = if lua_rt.is_some() {
                    !cbs.is_empty()
                } else {
                    def.is_some()
                };
                (ok, if cbs.is_empty() { "registered_crafting_table".into() } else { format!("lua{cbs:?}") })
            }
            "bed" => {
                let ok = if lua_rt.is_some() {
                    !cbs.is_empty() || node_id.contains("mcl_beds:")
                } else {
                    def.is_some()
                };
                (ok, format!("nether_bed_intercept+lua{cbs:?}"))
            }
            "door" => {
                let ok = if lua_rt.is_some() {
                    !cbs.is_empty() || node_id.contains("doors:") || node_id.contains("mcl_doors:")
                } else {
                    def.is_some()
                };
                (ok, format!("door+lua{cbs:?}"))
            }
            "nether_portal" => {
                let native = def.is_some_and(|d| is_nether_portal_node(d.id, reg));
                (native, "native_portal_teleport".into())
            }
            "water" => {
                let liq = def.is_some_and(is_liquid_node);
                let bucket = reg.items.contains_key("mcl_buckets:bucket_empty")
                    || reg.items.contains_key("bucket:bucket_empty");
                (liq && bucket, format!("liquid={liq}+bucket={bucket}"))
            }
            "lava" => {
                let liq = def.is_some_and(is_liquid_node);
                let dmg = def.is_some_and(|d| d.damage_per_second > 0);
                (liq && dmg, format!("liquid={liq}+dps={dmg}"))
            }
            "sand" | "gravel" => {
                let falling = def.is_some_and(|d| d.groups.get("falling_node").copied().unwrap_or(0) > 0);
                (falling, format!("falling_node={falling}"))
            }
            _ => (false, "unknown".into()),
        };

        checks.push(InteractiveHandlerCheck {
            category: sp.category.into(),
            node_id,
            registered: true,
            has_handler,
            handler_kind,
        });
    }
    checks
}

fn clean_item_spec_name(spec: &str) -> String {
    spec.split_whitespace().next().unwrap_or("").trim().to_string()
}

fn seed_node_and_drops(reg: &EngineRegistries, raw_name: &str, reachable: &mut HashSet<String>) {
    let resolved = reg.resolve_alias(raw_name.trim()).to_string();
    if resolved.is_empty() || resolved == "air" || resolved == "ignore" {
        return;
    }
    reachable.insert(raw_name.trim().to_string());
    reachable.insert(resolved.clone());
    if let Some(def) = node_def_by_name(reg, &resolved) {
        match &def.drop {
            NodeDrop::Default => {}
            NodeDrop::Empty => {}
            NodeDrop::Single(s) => {
                let d = clean_item_spec_name(&s);
                if !d.is_empty() {
                    let dr = reg.resolve_alias(&d).to_string();
                    reachable.insert(d);
                    reachable.insert(dr);
                }
            }
            NodeDrop::Table { items, .. } => {
                for entry in items {
                    for s in &entry.items {
                        let d = clean_item_spec_name(s);
                        if !d.is_empty() {
                            let dr = reg.resolve_alias(&d).to_string();
                            reachable.insert(d);
                            reachable.insert(dr);
                        }
                    }
                }
            }
        }
    }
}

fn item_matches_ingredient(
    ing: &str,
    reachable: &HashSet<String>,
    reg: &EngineRegistries,
) -> bool {
    let ing = ing.trim();
    if ing.is_empty() {
        return true;
    }
    if let Some(grp_expr) = ing.strip_prefix("group:") {
        let req_groups: Vec<&str> = grp_expr.split(',').map(|g| g.trim()).collect();
        return reachable.iter().any(|r| {
            let resolved = reg.resolve_alias(r);
            let item_ok = reg.get_item_def(resolved).is_some_and(|it| {
                req_groups
                    .iter()
                    .all(|g| it.groups.get(*g).copied().unwrap_or(0) > 0)
            });
            let node_ok = node_def_by_name(reg, resolved).is_some_and(|nd| {
                req_groups
                    .iter()
                    .all(|g| nd.groups.get(*g).copied().unwrap_or(0) > 0)
            });
            item_ok || node_ok
        });
    }
    let resolved = reg.resolve_alias(ing);
    reachable.contains(ing) || reachable.contains(resolved)
}

fn propagate_aliases(reg: &EngineRegistries, reachable: &mut HashSet<String>) -> bool {
    let mut added = false;
    for (alias, target) in &reg.aliases {
        if reachable.contains(target) && reachable.insert(alias.clone()) {
            added = true;
        }
        if reachable.contains(alias) && reachable.insert(target.clone()) {
            added = true;
        }
    }
    added
}

fn compute_craft_reachability(
    reg: &EngineRegistries,
    loot_items: &HashSet<String>,
) -> (HashSet<String>, usize, Vec<String>) {
    let mut reachable: HashSet<String> = HashSet::new();
    let pal = MapgenPalette::from_registries_with_game_dir(reg, None);
    let npal = NetherPalette::from_registries(reg);
    let spal = StructurePalette::from_registries(reg);

    // Seed natural worldgen terrain nodes, trees, ores, structures, liquids, mobs, and chest loot
    for &id in &[
        pal.stone,
        pal.dirt,
        pal.grass,
        pal.sand,
        pal.gravel,
        pal.snow,
        pal.oak_log,
        pal.birch_log,
        pal.spruce_log,
        pal.water,
        pal.lava,
        pal.clay,
        pal.iron_ore,
        pal.coal_ore,
        npal.netherrack,
        npal.soul_sand,
        npal.glowstone,
        npal.quartz_ore,
        npal.lava_source,
        spal.nether_brick,
        spal.prismarine,
        spal.sea_lantern,
        spal.sponge,
    ] {
        let n = reg.get_name_from_content_id(id);
        seed_node_and_drops(reg, n, &mut reachable);
    }

    for slot in &reg.nodes_by_id {
        if let Some(def) = slot.as_ref() {
            if def.groups.get("tree").copied().unwrap_or(0) > 0
                || def.groups.get("leaves").copied().unwrap_or(0) > 0
                || def.groups.get("stone").copied().unwrap_or(0) > 0
                || def.groups.get("soil").copied().unwrap_or(0) > 0
                || def.groups.get("sand").copied().unwrap_or(0) > 0
            {
                seed_node_and_drops(reg, &def.name, &mut reachable);
            }
        }
    }
    for ore in &reg.ores {
        seed_node_and_drops(reg, &ore.ore, &mut reachable);
    }
    for biome in &reg.biomes {
        seed_node_and_drops(reg, &biome.node_top, &mut reachable);
        seed_node_and_drops(reg, &biome.node_filler, &mut reachable);
        seed_node_and_drops(reg, &biome.node_stone, &mut reachable);
        seed_node_and_drops(reg, &biome.node_water_top, &mut reachable);
    }
    for deco in &reg.decorations {
        for d in &deco.decoration {
            seed_node_and_drops(reg, d, &mut reachable);
        }
    }
    for li in loot_items {
        reachable.insert(li.clone());
    }
    for mob_drop in [
        "mcl_mobitems:blaze_rod",
        "mcl_mobitems:ghast_tear",
        "mcl_mobitems:gunpowder",
        "mcl_mobitems:rotten_flesh",
        "mcl_core:gold_nugget",
        "mcl_mobitems:bone",
        "mcl_mobitems:coal",
    ] {
        let r = reg.resolve_alias(mob_drop).to_string();
        if reg.get_item_def(&r).is_some() {
            reachable.insert(mob_drop.to_string());
            reachable.insert(r);
        }
    }

    let _ = propagate_aliases(reg, &mut reachable);

    // Fixed-point BFS over CraftRecipe list + native bucket casting & portal ignition
    let mut iterations = 0usize;
    loop {
        iterations += 1;
        let mut changed = false;

        if (reachable.contains("mcl_buckets:bucket_empty")
            || reachable.contains("bucket:bucket_empty"))
            && (reachable.contains("mcl_core:water_source")
                || reachable.contains("default:water_source"))
            && (reachable.contains("mcl_core:lava_source")
                || reachable.contains("default:lava_source"))
        {
            if reachable.insert("mcl_core:obsidian".to_string()) {
                changed = true;
            }
        }

        if reachable.contains("mcl_core:obsidian")
            && (reachable.contains("mcl_fire:flint_and_steel")
                || reachable.contains("mcl_fire:fire_charge"))
        {
            if reachable.insert(NETHER_PORTAL_NODE_ID.to_string()) {
                changed = true;
            }
        }

        for craft in &reg.crafts {
            if craft.recipe_type == "fuel" {
                continue;
            }
            let out_item = clean_item_spec_name(&craft.output);
            if out_item.is_empty() {
                continue;
            }
            let out_resolved = reg.resolve_alias(&out_item).to_string();
            if reachable.contains(&out_item) && reachable.contains(&out_resolved) {
                continue;
            }
            let all_inputs_ok = craft.recipe.iter().flatten().all(|cell| {
                let c = cell.trim();
                c.is_empty() || item_matches_ingredient(c, &reachable, reg)
            });
            if all_inputs_ok {
                if reachable.insert(out_item) {
                    changed = true;
                }
                if reachable.insert(out_resolved) {
                    changed = true;
                }
            }
        }

        if propagate_aliases(reg, &mut reachable) {
            changed = true;
        }
        if !changed || iterations >= 32 {
            break;
        }
    }

    let milestones = [
        "mcl_core:wood",
        "mcl_core:stick",
        "mcl_crafting_table:crafting_table",
        "mcl_tools:pick_wood",
        "mcl_tools:pick_stone",
        "mcl_furnaces:furnace",
        "mcl_core:iron_ingot",
        "mcl_tools:pick_iron",
        "mcl_buckets:bucket_empty",
        "mcl_fire:flint_and_steel",
        "mcl_core:obsidian",
        NETHER_PORTAL_NODE_ID,
    ];

    let mut unreachable = Vec::new();
    for &m in &milestones {
        let resolved = reg.resolve_alias(m);
        let wood_fallback = m == "mcl_core:wood"
            && reachable
                .iter()
                .any(|r| r == "mcl_trees:wood_oak" || r.starts_with("mcl_trees:wood_"));
        let flint_and_steel_ok = m == "mcl_fire:flint_and_steel"
            && (reachable.contains(m) || reachable.contains(resolved));
        if !reachable.contains(m)
            && !reachable.contains(resolved)
            && !wood_fallback
            && !flint_and_steel_ok
        {
            unreachable.push(m.to_string());
        }
    }

    (reachable, iterations, unreachable)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{CraftRecipe, ItemDef, NodeDef, TileSpec, ToolCapabilities};

    fn make_minimal_qa_registry_and_atlas() -> (EngineRegistries, TextureAtlas) {
        let mut reg = EngineRegistries::new();
        let mut atlas = TextureAtlas::placeholder();

        let register_node = |reg: &mut EngineRegistries,
                             name: &str,
                             drawtype: DrawType,
                             walkable: bool,
                             dps: i32,
                             groups: &[(&str, i32)],
                             drop: NodeDrop| {
            let mut d = NodeDef {
                name: name.into(),
                drawtype,
                walkable,
                damage_per_second: dps,
                drop,
                tiles: vec![TileSpec {
                    name: "cubex_test_tile.png".into(),
                    color: None,
                }],
                inventory_image: "cubex_test_tile.png".into(),
                ..Default::default()
            };
            if drawtype == DrawType::Liquid {
                d.liquidtype = "source".into();
                d.liquid_viscosity = 1;
            }
            for &(gk, gv) in groups {
                d.groups.insert(gk.into(), gv);
            }
            reg.register_node(d);
        };

        // Ensure fallback layer resolves cleanly when using resolve_spec_immut on a real tile name
        atlas
            .resolved_cache
            .insert("cubex_test_tile.png".into(), atlas.fallback_layer);
        atlas
            .raw_cache
            .insert("cubex_test_tile.png".into(), atlas.layers[0].clone());

        register_node(
            &mut reg,
            "mcl_core:stone",
            DrawType::Normal,
            true,
            0,
            &[("cracky", 3), ("stone", 1)],
            NodeDrop::Single("mcl_core:cobble".into()),
        );
        register_node(
            &mut reg,
            "mcl_core:cobble",
            DrawType::Normal,
            true,
            0,
            &[("cracky", 3), ("cobble", 1)],
            NodeDrop::Default,
        );
        register_node(
            &mut reg,
            "mcl_core:dirt",
            DrawType::Normal,
            true,
            0,
            &[("soil", 1)],
            NodeDrop::Default,
        );
        register_node(
            &mut reg,
            "mcl_core:dirt_with_grass",
            DrawType::Normal,
            true,
            0,
            &[("soil", 1)],
            NodeDrop::Single("mcl_core:dirt".into()),
        );
        register_node(
            &mut reg,
            "mcl_core:sand",
            DrawType::Normal,
            true,
            0,
            &[("sand", 1), ("falling_node", 1)],
            NodeDrop::Default,
        );
        register_node(
            &mut reg,
            "mcl_core:gravel",
            DrawType::Normal,
            true,
            0,
            &[("falling_node", 1)],
            NodeDrop::Single("mcl_core:flint".into()),
        );
        register_node(
            &mut reg,
            "mcl_core:bedrock",
            DrawType::Normal,
            true,
            0,
            &[],
            NodeDrop::Empty,
        );
        register_node(
            &mut reg,
            "mcl_core:obsidian",
            DrawType::Normal,
            true,
            0,
            &[("cracky", 1)],
            NodeDrop::Default,
        );
        register_node(
            &mut reg,
            "mcl_trees:tree_oak",
            DrawType::Normal,
            true,
            0,
            &[("tree", 1)],
            NodeDrop::Default,
        );
        register_node(
            &mut reg,
            "mcl_trees:wood_oak",
            DrawType::Normal,
            true,
            0,
            &[("wood", 1)],
            NodeDrop::Default,
        );
        reg.aliases.insert("mcl_core:wood".into(), "mcl_trees:wood_oak".into());
        register_node(
            &mut reg,
            "mcl_core:stone_with_iron",
            DrawType::Normal,
            true,
            0,
            &[("cracky", 2)],
            NodeDrop::Single("mcl_raw_ores:raw_iron".into()),
        );
        register_node(
            &mut reg,
            "mcl_core:stone_with_coal",
            DrawType::Normal,
            true,
            0,
            &[("cracky", 3)],
            NodeDrop::Single("mcl_core:coal_lump".into()),
        );
        register_node(
            &mut reg,
            "mcl_core:water_source",
            DrawType::Liquid,
            false,
            0,
            &[("water", 3)],
            NodeDrop::Empty,
        );
        register_node(
            &mut reg,
            "mcl_core:lava_source",
            DrawType::Liquid,
            false,
            8,
            &[("lava", 3)],
            NodeDrop::Empty,
        );
        register_node(
            &mut reg,
            "mcl_crafting_table:crafting_table",
            DrawType::Normal,
            true,
            0,
            &[],
            NodeDrop::Default,
        );
        register_node(
            &mut reg,
            "mcl_furnaces:furnace",
            DrawType::Normal,
            true,
            0,
            &[],
            NodeDrop::Default,
        );
        register_node(
            &mut reg,
            "mcl_chests:chest",
            DrawType::Normal,
            true,
            0,
            &[],
            NodeDrop::Default,
        );
        register_node(
            &mut reg,
            "mcl_beds:bed_red_bottom",
            DrawType::NodeBox,
            true,
            0,
            &[],
            NodeDrop::Default,
        );
        register_node(
            &mut reg,
            "mcl_doors:wooden_door_b_1",
            DrawType::NodeBox,
            true,
            0,
            &[],
            NodeDrop::Default,
        );
        register_node(
            &mut reg,
            NETHER_PORTAL_NODE_ID,
            DrawType::Normal,
            false,
            0,
            &[],
            NodeDrop::Empty,
        );
        register_node(
            &mut reg,
            "mcl_nether:netherrack",
            DrawType::Normal,
            true,
            0,
            &[],
            NodeDrop::Default,
        );
        register_node(
            &mut reg,
            "mcl_nether:nether_brick",
            DrawType::Normal,
            true,
            0,
            &[],
            NodeDrop::Default,
        );
        register_node(
            &mut reg,
            "mcl_nether:glowstone",
            DrawType::Normal,
            true,
            0,
            &[],
            NodeDrop::Default,
        );
        register_node(
            &mut reg,
            "mcl_ocean:prismarine",
            DrawType::Normal,
            true,
            0,
            &[],
            NodeDrop::Default,
        );
        register_node(
            &mut reg,
            "mcl_ocean:sea_lantern",
            DrawType::Normal,
            true,
            0,
            &[],
            NodeDrop::Default,
        );
        register_node(
            &mut reg,
            "mcl_nether:nether_wart",
            DrawType::Plantlike,
            false,
            0,
            &[],
            NodeDrop::Default,
        );
        register_node(
            &mut reg,
            "mcl_core:goldblock",
            DrawType::Normal,
            true,
            0,
            &[],
            NodeDrop::Default,
        );
        register_node(
            &mut reg,
            "mcl_core:crying_obsidian",
            DrawType::Normal,
            true,
            0,
            &[],
            NodeDrop::Default,
        );
        register_node(
            &mut reg,
            "mcl_sponges:sponge",
            DrawType::Normal,
            true,
            0,
            &[],
            NodeDrop::Default,
        );

        let reg_item = |reg: &mut EngineRegistries, name: &str, is_tool: bool, is_food: bool| {
            let mut it = ItemDef {
                name: name.into(),
                item_type: if is_tool { "tool".into() } else { "craft".into() },
                inventory_image: "cubex_test_tile.png".into(),
                tool_capabilities: is_tool.then(ToolCapabilities::default),
                ..Default::default()
            };
            if is_food {
                it.groups.insert("food".into(), 2);
            }
            reg.register_item(it);
        };

        for name in [
            "mcl_core:stick",
            "mcl_core:coal_lump",
            "mcl_raw_ores:raw_iron",
            "mcl_core:iron_ingot",
            "mcl_core:gold_ingot",
            "mcl_core:diamond",
            "mcl_core:flint",
            "mcl_buckets:bucket_empty",
            "mcl_buckets:bucket_water",
            "mcl_buckets:bucket_lava",
            "mcl_fire:flint_and_steel",
            "mcl_mobitems:blaze_rod",
            "mcl_throwing:ender_pearl",
            "mcl_ocean:prismarine_shard",
            "mcl_mobitems:gunpowder",
            "mcl_mobitems:bone",
            "cubex:ram_mic",
        ] {
            reg_item(&mut reg, name, false, false);
        }
        for tool in [
            "mcl_tools:pick_wood",
            "mcl_tools:pick_stone",
            "mcl_tools:pick_iron",
            "mcl_tools:pick_diamond",
            "mcl_tools:sword_stone",
            "mcl_tools:sword_gold",
            "mcl_tools:sword_diamond",
            "mcl_armor:chestplate_gold",
            "mcl_armor:helmet_diamond",
            "mcl_armor:boots_gold",
            "mcl_armor:chestplate_diamond",
        ] {
            reg_item(&mut reg, tool, true, false);
        }
        reg_item(&mut reg, "mcl_core:apple_gold", false, true);

        let push_craft = |reg: &mut EngineRegistries, ty: &str, out: &str, rows: Vec<Vec<&str>>| {
            reg.crafts.push(CraftRecipe {
                recipe_type: ty.into(),
                output: out.into(),
                recipe: rows
                    .into_iter()
                    .map(|r| r.into_iter().map(str::to_string).collect())
                    .collect(),
                replacements: Vec::new(),
                cooktime: 1.0,
                burntime: 0.0,
            });
        };

        push_craft(
            &mut reg,
            "shapeless",
            "mcl_trees:wood_oak 4",
            vec![vec!["mcl_trees:tree_oak"]],
        );
        push_craft(
            &mut reg,
            "normal",
            "mcl_core:stick 4",
            vec![vec!["group:wood"], vec!["group:wood"]],
        );
        push_craft(
            &mut reg,
            "normal",
            "mcl_crafting_table:crafting_table",
            vec![
                vec!["group:wood", "group:wood"],
                vec!["group:wood", "group:wood"],
            ],
        );
        push_craft(
            &mut reg,
            "normal",
            "mcl_tools:pick_wood",
            vec![
                vec!["group:wood", "group:wood", "group:wood"],
                vec!["", "mcl_core:stick", ""],
                vec!["", "mcl_core:stick", ""],
            ],
        );
        push_craft(
            &mut reg,
            "normal",
            "mcl_tools:pick_stone",
            vec![
                vec!["mcl_core:cobble", "mcl_core:cobble", "mcl_core:cobble"],
                vec!["", "mcl_core:stick", ""],
                vec!["", "mcl_core:stick", ""],
            ],
        );
        push_craft(
            &mut reg,
            "normal",
            "mcl_furnaces:furnace",
            vec![
                vec!["mcl_core:cobble", "mcl_core:cobble", "mcl_core:cobble"],
                vec!["mcl_core:cobble", "", "mcl_core:cobble"],
                vec!["mcl_core:cobble", "mcl_core:cobble", "mcl_core:cobble"],
            ],
        );
        push_craft(
            &mut reg,
            "cooking",
            "mcl_core:iron_ingot",
            vec![vec!["mcl_raw_ores:raw_iron"]],
        );
        push_craft(
            &mut reg,
            "normal",
            "mcl_tools:pick_iron",
            vec![
                vec!["mcl_core:iron_ingot", "mcl_core:iron_ingot", "mcl_core:iron_ingot"],
                vec!["", "mcl_core:stick", ""],
                vec!["", "mcl_core:stick", ""],
            ],
        );
        push_craft(
            &mut reg,
            "normal",
            "mcl_buckets:bucket_empty",
            vec![
                vec!["mcl_core:iron_ingot", "", "mcl_core:iron_ingot"],
                vec!["", "mcl_core:iron_ingot", ""],
            ],
        );
        push_craft(
            &mut reg,
            "shapeless",
            "mcl_fire:flint_and_steel",
            vec![vec!["mcl_core:iron_ingot", "mcl_core:flint"]],
        );

        (reg, atlas)
    }

    #[test]
    fn static_audit_passes_valid_registry_and_flags_defects() {
        let (mut reg, atlas) = make_minimal_qa_registry_and_atlas();
        let report = run_static_audit(&reg, &atlas, None);
        assert!(
            report.passed,
            "Expected clean static audit, got failures: {:?}",
            report.failures
        );
        assert_eq!(report.interactive_checks.len(), 10);
        assert!(report.unreachable_milestones.is_empty());

        // Inject a damaging safe terrain node and verify audit catches it
        let grass_id = reg.get_content_id("mcl_core:dirt_with_grass");
        if let Some(Some(def)) = reg.nodes_by_id.get_mut(grass_id as usize) {
            def.damage_per_second = 2;
        }
        let bad_report = run_static_audit(&reg, &atlas, None);
        assert!(!bad_report.passed);
        assert!(!bad_report.damaging_safe_terrain_blocks.is_empty());
    }
}
