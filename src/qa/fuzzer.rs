use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use glam::{IVec3, Vec3};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::config::EngineSettings;
use crate::lua_api::drain_player_inventory_for_death;
use crate::lua_api::userdata::{ItemStackData, ObjectState};
use crate::physics::collision::{
    collision_move_simple_ex, Aabb, GameMode, MoveInput, PlayerController,
};
use crate::physics::raycast::{raycast_voxel_ex, try_interact_or_place, DigController};
use crate::registry::{EngineRegistries, CONTENT_AIR, CONTENT_UNKNOWN, NETHER_PORTAL_NODE_ID};
use crate::world::mobs::{MobManager, MobPlayerInput, MAX_ACTIVE_MOBS_PER_DIM};
use crate::world::portal::{
    find_or_build_destination_portal, is_nether_portal_node, try_ignite_portal, PortalAxis,
};
use crate::world::{scale_portal_coords, Chunk, DimensionId, WorldStore};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzScenarioResult {
    pub name: String,
    pub seed: u64,
    pub ticks: u64,
    pub passed: bool,
    pub violation: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzerReport {
    pub passed: bool,
    pub seeds_tested: usize,
    pub total_ticks: u64,
    pub total_portal_cycles: u32,
    pub max_entities_observed: usize,
    pub scenarios_passed: usize,
    pub scenarios_total: usize,
    pub invariant_violations: Vec<String>,
    pub scenarios: Vec<FuzzScenarioResult>,
    pub elapsed_secs: f64,
}

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

struct FuzzRng(u64);

impl FuzzRng {
    fn new(seed: u64) -> Self {
        Self(splitmix64(seed | 1))
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = splitmix64(self.0);
        self.0
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() & 0x00FF_FFFF) as f32 / 16_777_215.0
    }

    fn chance(&mut self, num: u64, den: u64) -> bool {
        (self.next_u64() % den) < num
    }
}

#[inline]
fn player_transforms_finite(p: &PlayerController) -> bool {
    p.pos.is_finite()
        && p.vel.is_finite()
        && p.yaw.is_finite()
        && p.pitch.is_finite()
        && p.eye_height.is_finite()
        && p.horizontal_speed.is_finite()
}

fn fill_test_arena_chunks(world: &WorldStore, min_cx: i32, max_cx: i32, min_cy: i32, max_cy: i32, min_cz: i32, max_cz: i32) {
    for cz in min_cz..=max_cz {
        for cy in min_cy..=max_cy {
            for cx in min_cx..=max_cx {
                world.insert_chunk(Chunk::new(IVec3::new(cx, cy, cz)));
            }
        }
    }
}

fn resolve_first_id(reg: &EngineRegistries, names: &[&str], fallback: u16) -> u16 {
    for &n in names {
        let r = reg.resolve_alias(n);
        let id = reg.get_content_id(r);
        if id != CONTENT_UNKNOWN && id != CONTENT_AIR {
            return id;
        }
    }
    fallback
}

/// Phase 2: Run deterministic multi-seed physics & edge-case chaos fuzzing against `reg`.
pub fn run_physics_fuzzer(seed_count: usize, reg: &EngineRegistries) -> FuzzerReport {
    let start = Instant::now();
    let seeds_tested = seed_count.max(1);
    let tmp_root = std::env::temp_dir().join("cubex_qa_fuzzer_worlds");
    let _ = std::fs::remove_dir_all(&tmp_root);

    let mut total_ticks = 0u64;
    let mut total_portal_cycles = 0u32;
    let mut max_entities_observed = 0usize;
    let mut scenarios = Vec::with_capacity(seeds_tested * 4);
    let mut invariant_violations = Vec::new();

    for idx in 0..seeds_tested {
        let seed = 0xC0BE_F000_0000_0001u64 ^ ((idx as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let seed_dir = tmp_root.join(format!("seed_{idx}"));

        let s1 = fuzz_terrain_steps_slabs_snow_corners(seed, &seed_dir, reg);
        total_ticks += s1.ticks;
        if let Some(ref v) = s1.violation {
            invariant_violations.push(v.clone());
        }
        scenarios.push(s1);

        let (s2, s2_ents) = fuzz_underfoot_dig_place_and_falling(seed ^ 0x1111, &seed_dir, reg);
        total_ticks += s2.ticks;
        max_entities_observed = max_entities_observed.max(s2_ents);
        if let Some(ref v) = s2.violation {
            invariant_violations.push(v.clone());
        }
        scenarios.push(s2);

        let (s3, portal_cycles) = fuzz_portal_cycling_and_death_guard(seed ^ 0x2222, &seed_dir, reg);
        total_ticks += s3.ticks;
        total_portal_cycles += portal_cycles;
        if let Some(ref v) = s3.violation {
            invariant_violations.push(v.clone());
        }
        scenarios.push(s3);

        let (s4, s4_mobs) = fuzz_split_screen_p1_p2_independence(seed ^ 0x3333, &seed_dir, reg);
        total_ticks += s4.ticks;
        max_entities_observed = max_entities_observed.max(s4_mobs);
        if let Some(ref v) = s4.violation {
            invariant_violations.push(v.clone());
        }
        scenarios.push(s4);
    }

    let _ = std::fs::remove_dir_all(&tmp_root);

    let scenarios_total = scenarios.len();
    let scenarios_passed = scenarios.iter().filter(|s| s.passed).count();

    FuzzerReport {
        passed: invariant_violations.is_empty() && scenarios_passed == scenarios_total,
        seeds_tested,
        total_ticks,
        total_portal_cycles,
        max_entities_observed,
        scenarios_passed,
        scenarios_total,
        invariant_violations,
        scenarios,
        elapsed_secs: start.elapsed().as_secs_f64(),
    }
}

fn fuzz_terrain_steps_slabs_snow_corners(
    seed: u64,
    dir: &PathBuf,
    reg: &EngineRegistries,
) -> FuzzScenarioResult {
    let name = "terrain_steps_slabs_snow_corners".to_string();
    let world = Arc::new(WorldStore::new(&dir.join("s1"), seed));
    fill_test_arena_chunks(&world, -2, 2, 0, 2, -2, 2);

    let stone = resolve_first_id(reg, &["mcl_core:stone", "mcl_core:cobble"], 1);
    let grass = resolve_first_id(reg, &["mcl_core:dirt_with_grass", "mcl_core:dirt"], stone);
    let slab = reg
        .nodes_by_id
        .iter()
        .flatten()
        .find(|d| d.walkable && d.name.contains("slab") && !d.node_boxes.is_empty())
        .map(|d| d.id)
        .unwrap_or(stone);
    let snow = reg
        .nodes_by_id
        .iter()
        .flatten()
        .find(|d| d.name == "mcl_core:snow" && d.walkable)
        .map(|d| d.id)
        .unwrap_or(grass);

    // Build sturdy 29x29 floor at y=19..=20 with steps, slabs, snow patches, 2-wall L-corners, 1x2 tunnel & outer barrier
    for z in -14..=14 {
        for x in -14..=14 {
            world.set_node(IVec3::new(x, 19, z), stone, 15, 0);
            world.set_node(IVec3::new(x, 20, z), grass, 15, 0);
            if x.abs() == 14 || z.abs() == 14 {
                for wy in 21..=25 {
                    world.set_node(IVec3::new(x, wy, z), stone, 15, 0);
                }
            }
        }
    }
    // 1-block auto-steps + slabs + snow strips
    for z in -6..=6 {
        world.set_node(IVec3::new(3, 21, z), stone, 15, 0);
        world.set_node(IVec3::new(-4, 21, z), slab, 15, 0);
        world.set_node(IVec3::new(z, 21, 5), snow, 15, 0);
    }
    // L-corner walls (y=21..=23) + 1x2 tunnel at x=7..=10, z=-1
    for dy in 21..=23 {
        world.set_node(IVec3::new(6, dy, 6), stone, 15, 0);
        world.set_node(IVec3::new(7, dy, 6), stone, 15, 0);
        world.set_node(IVec3::new(6, dy, 7), stone, 15, 0);
    }
    for x in 7..=11 {
        world.set_node(IVec3::new(x, 21, -2), stone, 15, 0);
        world.set_node(IVec3::new(x, 22, -2), stone, 15, 0);
        world.set_node(IVec3::new(x, 21, 0), stone, 15, 0);
        world.set_node(IVec3::new(x, 22, 0), stone, 15, 0);
        world.set_node(IVec3::new(x, 23, -1), stone, 15, 0);
    }

    let mut player = PlayerController::default();
    player.gamemode = GameMode::Survival;
    player.pos = Vec3::new(0.5, 21.05, 0.5);
    let mut settings = EngineSettings::default();
    settings.auto_jump = true;
    let mut rng = FuzzRng::new(seed);
    let dt = 0.025f32;
    let ticks = 240u64;

    for tick in 0..ticks {
        player.yaw += (rng.next_f32() - 0.5) * 0.65;
        player.pitch = (player.pitch + (rng.next_f32() - 0.5) * 0.2).clamp(-1.4, 1.4);
        let input = MoveInput {
            forward: rng.chance(4, 5),
            backward: rng.chance(1, 10),
            left: rng.chance(1, 4),
            right: rng.chance(1, 4),
            jump: rng.chance(1, 3),
            sprint: true,
            ..Default::default()
        };
        let ev = player.step(dt, &input, &world, reg, &settings);

        if !player_transforms_finite(&player) {
            return FuzzScenarioResult {
                name,
                seed,
                ticks: tick + 1,
                passed: false,
                violation: Some(format!("seed={seed} tick={tick}: non-finite transform")),
            };
        }
        if player.pos.y < 19.8 {
            return FuzzScenarioResult {
                name,
                seed,
                ticks: tick + 1,
                passed: false,
                violation: Some(format!("seed={seed} tick={tick}: fell below arena floor (y={:.2})", player.pos.y)),
            };
        }
        if player.hp < 20 || ev.fall.is_some() {
            return FuzzScenarioResult {
                name,
                seed,
                ticks: tick + 1,
                passed: false,
                violation: Some(format!(
                    "seed={seed} tick={tick}: phantom damage hp={} fall={:?} pos={:?}",
                    player.hp, ev.fall, player.pos
                )),
            };
        }
    }

    FuzzScenarioResult {
        name,
        seed,
        ticks,
        passed: true,
        violation: None,
    }
}

struct FallingBlockSim {
    node_id: u16,
    pos: Vec3,
    vel: Vec3,
    settled: bool,
}

fn fuzz_underfoot_dig_place_and_falling(
    seed: u64,
    dir: &PathBuf,
    reg: &EngineRegistries,
) -> (FuzzScenarioResult, usize) {
    let name = "underfoot_dig_place_and_falling_nodes".to_string();
    let world = Arc::new(WorldStore::new(&dir.join("s2"), seed));
    fill_test_arena_chunks(&world, -1, 1, 0, 2, -1, 1);

    let bedrock = resolve_first_id(reg, &["mcl_core:bedrock"], 1);
    let cobble = resolve_first_id(reg, &["mcl_core:cobble", "mcl_core:stone"], bedrock);
    let sand = resolve_first_id(reg, &["mcl_core:sand"], cobble);
    let gravel = resolve_first_id(reg, &["mcl_core:gravel"], cobble);

    for z in -10..=10 {
        for x in -10..=10 {
            world.set_node(IVec3::new(x, 0, z), bedrock, 15, 0);
            for y in 1..=6 {
                world.set_node(IVec3::new(x, y, z), cobble, 15, 0);
            }
        }
    }

    let mut player = PlayerController::default();
    player.gamemode = GameMode::Survival;
    player.pos = Vec3::new(0.5, 7.05, 0.5);
    player.pitch = -1.25;
    let ps = Arc::new(RwLock::new(ObjectState::new_player("Fuzzer")));
    {
        let mut psw = ps.write();
        {
            let mut mg = psw.meta.write();
            mg.inventories.insert(
                "main".into(),
                vec![ItemStackData::from_spec(&format!("{} 64", reg.get_name_from_content_id(cobble))); 9],
            );
        }
        psw.wield_index = 1;
    }

    let settings = EngineSettings::default();
    let mut digger = DigController::new();
    let mut rng = FuzzRng::new(seed);
    let mut falling: Vec<FallingBlockSim> = Vec::new();
    let mut max_ents = 0usize;
    let dt = 0.025f32;
    let ticks = 160u64;

    for tick in 0..ticks {
        player.yaw += (rng.next_f32() - 0.5) * 0.45;
        player.pitch = if tick % 2 == 0 { -1.32 } else { -0.85 };
        let input = MoveInput {
            forward: rng.chance(2, 3),
            left: rng.chance(1, 5),
            right: rng.chance(1, 5),
            jump: rng.chance(1, 4),
            ..Default::default()
        };
        let _ = player.step(dt, &input, &world, reg, &settings);

        let hit = raycast_voxel_ex(player.eye_pos(), player.look_dir(), 4.5, false, &world, reg);
        let mine_held = rng.chance(1, 2);
        let _ = digger.update_with_physical_drops(
            dt,
            mine_held,
            hit.as_ref(),
            &world,
            reg,
            &player,
            &ps,
            None,
            false,
        );
        if rng.chance(1, 3) {
            if let Some(ref h) = hit {
                let _ = try_interact_or_place(h, &world, reg, &player, &ps, None);
            }
        }

        // Keep stack inventory topped up for rapid placement fuzzing
        if tick % 20 == 0 {
            let psw = ps.write();
            let mut mg = psw.meta.write();
            if let Some(main) = mg.inventories.get_mut("main") {
                main[0] = ItemStackData::from_spec(&format!("{} 64", reg.get_name_from_content_id(cobble)));
            }
        }

        // Spawn & step falling sand/gravel entities onto bedrock/cobble and assert grid snap
        if tick % 20 == 0 && falling.len() < 8 {
            let nid = if (tick / 20) % 2 == 0 { sand } else { gravel };
            let gx = ((rng.next_u64() % 5) as i32) - 2;
            let gz = ((rng.next_u64() % 5) as i32) - 2;
            falling.push(FallingBlockSim {
                node_id: nid,
                pos: Vec3::new(gx as f32 + 0.5, 13.0, gz as f32 + 0.5),
                vel: Vec3::ZERO,
                settled: false,
            });
        }
        let f_box = Aabb::new(Vec3::new(-0.49, 0.0, -0.49), Vec3::new(0.49, 0.98, 0.49));
        for ent in &mut falling {
            if ent.settled {
                continue;
            }
            let res = collision_move_simple_ex(
                &world,
                reg,
                f_box,
                0.0,
                dt,
                &mut ent.pos,
                &mut ent.vel,
                Vec3::new(0.0, -14.0, 0.0),
                0.0,
            );
            if !ent.pos.is_finite() || !ent.vel.is_finite() {
                return (
                    FuzzScenarioResult {
                        name,
                        seed,
                        ticks: tick + 1,
                        passed: false,
                        violation: Some(format!("seed={seed} tick={tick}: non-finite falling node")),
                    },
                    max_ents,
                );
            }
            if res.touching_ground {
                let snap = IVec3::new(
                    ent.pos.x.floor() as i32,
                    (ent.pos.y + 0.05).floor() as i32,
                    ent.pos.z.floor() as i32,
                );
                if snap.y < 1 {
                    return (
                        FuzzScenarioResult {
                            name,
                            seed,
                            ticks: tick + 1,
                            passed: false,
                            violation: Some(format!("seed={seed} tick={tick}: falling node fell through bedrock to {snap:?}")),
                        },
                        max_ents,
                    );
                }
                world.set_node(snap, ent.node_id, 15, 0);
                ent.settled = true;
            }
        }
        falling.retain(|e| !e.settled);
        max_ents = max_ents.max(falling.len());

        if !player_transforms_finite(&player) {
            return (
                FuzzScenarioResult {
                    name,
                    seed,
                    ticks: tick + 1,
                    passed: false,
                    violation: Some(format!("seed={seed} tick={tick}: non-finite player transform")),
                },
                max_ents,
            );
        }
        if player.pos.y < 0.95 {
            return (
                FuzzScenarioResult {
                    name,
                    seed,
                    ticks: tick + 1,
                    passed: false,
                    violation: Some(format!(
                        "seed={seed} tick={tick}: fell past bedrock y={:.2}",
                        player.pos.y
                    )),
                },
                max_ents,
            );
        }
    }

    (
        FuzzScenarioResult {
            name,
            seed,
            ticks,
            passed: true,
            violation: None,
        },
        max_ents,
    )
}

fn fuzz_portal_cycling_and_death_guard(
    seed: u64,
    dir: &PathBuf,
    reg: &EngineRegistries,
) -> (FuzzScenarioResult, u32) {
    let name = "nether_portal_cycling_and_death_guard".to_string();
    let ow = Arc::new(WorldStore::new(&dir.join("s3_ow"), seed));
    let nw = Arc::new(WorldStore::new_for_dimension(
        &dir.join("s3_nw"),
        seed,
        DimensionId::Nether,
        crate::world::WorldGenType::Default,
        crate::world::Difficulty::Normal,
        crate::world::Gamerules::default(),
        "Survival",
    ));
    fill_test_arena_chunks(&ow, -1, 1, 0, 2, -1, 1);
    fill_test_arena_chunks(&nw, -1, 1, 0, 4, -1, 1);

    let obsidian = resolve_first_id(reg, &["mcl_core:obsidian"], 1);
    let stone = resolve_first_id(reg, &["mcl_core:stone"], obsidian);
    let netherrack = resolve_first_id(reg, &["mcl_nether:netherrack"], stone);
    for z in -6..=6 {
        for x in -6..=6 {
            ow.set_node(IVec3::new(x, 19, z), stone, 15, 0);
            nw.set_node(IVec3::new(x, 47, z), netherrack, 15, 0);
        }
    }

    // Build & ignite a 4x5 Overworld obsidian frame at (0, 20, 0)
    for dx in -1..=2 {
        ow.set_node(IVec3::new(dx, 20, 0), obsidian, 15, 0);
        ow.set_node(IVec3::new(dx, 24, 0), obsidian, 15, 0);
    }
    for dy in 21..=23 {
        ow.set_node(IVec3::new(-1, dy, 0), obsidian, 15, 0);
        ow.set_node(IVec3::new(2, dy, 0), obsidian, 15, 0);
    }
    if reg.get_content_id(NETHER_PORTAL_NODE_ID) != CONTENT_UNKNOWN {
        let ignited = try_ignite_portal(&ow, reg, IVec3::new(0, 21, 0));
        if ignited.is_none() {
            return (
                FuzzScenarioResult {
                    name,
                    seed,
                    ticks: 0,
                    passed: false,
                    violation: Some("Failed to ignite 4x5 portal frame in fuzzer".into()),
                },
                0,
            );
        }
    }

    let mut player = PlayerController::default();
    player.gamemode = GameMode::Survival;
    player.pos = Vec3::new(0.5, 21.0, 0.5);
    let settings = EngineSettings::default();
    let mut cycles = 0u32;
    let ticks = 120u64;
    let dt = 0.05f32;

    for tick in 0..ticks {
        let cur_w = if player.dimension.is_nether() { &nw } else { &ow };
        let _ = player.step(dt, &MoveInput::default(), cur_w, reg, &settings);

        let p_feet = IVec3::new(
            player.pos.x.floor() as i32,
            (player.pos.y + 0.1).floor() as i32,
            player.pos.z.floor() as i32,
        );
        if is_nether_portal_node(cur_w.get_node(p_feet).0, reg) {
            player.portal_charge = (player.portal_charge + dt / 1.5).min(1.0);
            if player.portal_charge >= 1.0 && cycles < 4 {
                let target_dim = if player.dimension.is_nether() {
                    DimensionId::Overworld
                } else {
                    DimensionId::Nether
                };
                let dst_w = if target_dim.is_nether() { &nw } else { &ow };
                let scaled = scale_portal_coords(p_feet, player.dimension, target_dim);
                let stand = find_or_build_destination_portal(dst_w, reg, scaled, PortalAxis::X);
                player.dimension = target_dim;
                player.pos = Vec3::new(
                    stand.x as f32 + 0.5,
                    stand.y as f32 + 0.05,
                    stand.z as f32 + 0.5,
                );
                player.vel = Vec3::ZERO;
                player.portal_charge = 0.0;
                player.reset_fall_state();
                cycles += 1;
            }
        }

        if !player_transforms_finite(&player) {
            return (
                FuzzScenarioResult {
                    name,
                    seed,
                    ticks: tick + 1,
                    passed: false,
                    violation: Some(format!("seed={seed} tick={tick}: non-finite portal pos")),
                },
                cycles,
            );
        }
    }

    // Verify death state lock & no item-duplication across repeated death ticks
    let ps = Arc::new(RwLock::new(ObjectState::new_player("DeathGuard")));
    {
        let psw = ps.write();
        let mut mg = psw.meta.write();
        mg.inventories.insert(
            "main".into(),
            vec![ItemStackData::from_spec(&format!("{} 5", reg.get_name_from_content_id(stone)))],
        );
    }
    player.hp = 0;
    player.is_dead = true;
    {
        ps.write().is_dead = true;
    }
    let first_drop = drain_player_inventory_for_death(&ps, false);
    let second_drop = drain_player_inventory_for_death(&ps, false);
    if first_drop.is_empty() || !second_drop.is_empty() {
        return (
            FuzzScenarioResult {
                name,
                seed,
                ticks,
                passed: false,
                violation: Some("Death inventory drain duplicated or dropped zero items".into()),
            },
            cycles,
        );
    }
    for _ in 0..10 {
        let cur_w = if player.dimension.is_nether() { &nw } else { &ow };
        let _ = player.step(
            dt,
            &MoveInput {
                forward: true,
                jump: true,
                ..Default::default()
            },
            cur_w,
            reg,
            &settings,
        );
        if !player.is_dead || player.hp != 0 {
            return (
                FuzzScenarioResult {
                    name,
                    seed,
                    ticks,
                    passed: false,
                    violation: Some("Dead player left is_dead state without explicit respawn".into()),
                },
                cycles,
            );
        }
    }

    (
        FuzzScenarioResult {
            name,
            seed,
            ticks,
            passed: true,
            violation: None,
        },
        cycles,
    )
}

fn fuzz_split_screen_p1_p2_independence(
    seed: u64,
    dir: &PathBuf,
    reg: &EngineRegistries,
) -> (FuzzScenarioResult, usize) {
    let name = "split_screen_p1_p2_independence".to_string();
    let ow = Arc::new(WorldStore::new(&dir.join("s4_ow"), seed));
    let nw = Arc::new(WorldStore::new_for_dimension(
        &dir.join("s4_nw"),
        seed,
        DimensionId::Nether,
        crate::world::WorldGenType::Default,
        crate::world::Difficulty::Normal,
        crate::world::Gamerules::default(),
        "Survival",
    ));
    fill_test_arena_chunks(&ow, -1, 1, 0, 2, -1, 1);
    fill_test_arena_chunks(&nw, -1, 1, 2, 4, -1, 1);

    let stone = resolve_first_id(reg, &["mcl_core:stone"], 1);
    let netherrack = resolve_first_id(reg, &["mcl_nether:netherrack"], stone);
    for z in -10..=10 {
        for x in -10..=10 {
            ow.set_node(IVec3::new(x, 19, z), stone, 15, 0);
            nw.set_node(IVec3::new(x, 47, z), netherrack, 15, 0);
        }
    }

    let mut p1 = PlayerController::default();
    p1.gamemode = GameMode::Survival;
    p1.dimension = DimensionId::Overworld;
    p1.pos = Vec3::new(0.5, 20.05, 0.5);

    let mut p2 = PlayerController::default();
    p2.gamemode = GameMode::Survival;
    p2.dimension = DimensionId::Nether;
    p2.pos = Vec3::new(-2.5, 48.05, 2.5);

    let settings = EngineSettings::default();
    let mut mob_mgr = MobManager::new(seed);
    let mut rng = FuzzRng::new(seed);
    let mut max_mobs = 0usize;
    let dt = 0.025f32;
    let ticks = 120u64;

    for tick in 0..ticks {
        p1.yaw += (rng.next_f32() - 0.5) * 0.4;
        p2.yaw += (rng.next_f32() - 0.5) * 0.5;
        let in1 = MoveInput {
            forward: true,
            sprint: rng.chance(1, 2),
            jump: rng.chance(1, 6),
            ..Default::default()
        };
        let in2 = MoveInput {
            backward: rng.chance(1, 3),
            forward: rng.chance(2, 3),
            right: rng.chance(1, 4),
            jump: rng.chance(1, 6),
            ..Default::default()
        };
        let _ = p1.step(dt, &in1, &ow, reg, &settings);
        let _ = p2.step(dt, &in2, &nw, reg, &settings);

        let mob_inputs = [
            MobPlayerInput {
                slot: 0,
                dimension: p1.dimension,
                pos: p1.pos,
                eye_pos: p1.eye_pos(),
                alive: !p1.is_dead,
                survival: p1.gamemode == GameMode::Survival,
            },
            MobPlayerInput {
                slot: 1,
                dimension: p2.dimension,
                pos: p2.pos,
                eye_pos: p2.eye_pos(),
                alive: !p2.is_dead,
                survival: p2.gamemode == GameMode::Survival,
            },
        ];
        let _ = mob_mgr.step(dt, &mob_inputs, &ow, &nw, reg);

        let c_ow = mob_mgr.active_mob_count(DimensionId::Overworld);
        let c_nw = mob_mgr.active_mob_count(DimensionId::Nether);
        max_mobs = max_mobs.max(c_ow + c_nw);

        if c_ow > MAX_ACTIVE_MOBS_PER_DIM || c_nw > MAX_ACTIVE_MOBS_PER_DIM {
            return (
                FuzzScenarioResult {
                    name,
                    seed,
                    ticks: tick + 1,
                    passed: false,
                    violation: Some(format!(
                        "Mob cap exceeded: ow={c_ow} nw={c_nw} max={MAX_ACTIVE_MOBS_PER_DIM}"
                    )),
                },
                max_mobs,
            );
        }
        if !player_transforms_finite(&p1) || !player_transforms_finite(&p2) {
            return (
                FuzzScenarioResult {
                    name,
                    seed,
                    ticks: tick + 1,
                    passed: false,
                    violation: Some("Non-finite split-screen player transform".into()),
                },
                max_mobs,
            );
        }
        if p1.dimension == p2.dimension {
            return (
                FuzzScenarioResult {
                    name,
                    seed,
                    ticks: tick + 1,
                    passed: false,
                    violation: Some("Split-screen P1/P2 dimension leaked".into()),
                },
                max_mobs,
            );
        }
    }

    (
        FuzzScenarioResult {
            name,
            seed,
            ticks,
            passed: true,
            violation: None,
        },
        max_mobs,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{ItemDef, NodeDef};

    #[test]
    fn fuzzer_passes_two_seeds_cleanly() {
        let mut reg = EngineRegistries::new();
        for n in [
            "mcl_core:stone",
            "mcl_core:cobble",
            "mcl_core:dirt_with_grass",
            "mcl_core:bedrock",
            "mcl_core:obsidian",
            "mcl_core:sand",
            "mcl_core:gravel",
            "mcl_nether:netherrack",
        ] {
            reg.register_node(NodeDef {
                name: n.into(),
                ..Default::default()
            });
            reg.register_item(ItemDef {
                name: n.into(),
                is_node: true,
                ..Default::default()
            });
        }
        reg.register_node(NodeDef {
            name: NETHER_PORTAL_NODE_ID.into(),
            walkable: false,
            diggable: false,
            buildable_to: true,
            ..Default::default()
        });

        let report = run_physics_fuzzer(2, &reg);
        assert!(
            report.passed,
            "Expected clean 2-seed fuzz report, got violations: {:?}",
            report.invariant_violations
        );
        assert_eq!(report.seeds_tested, 2);
        assert!(report.total_portal_cycles >= 2);
        assert_eq!(report.scenarios_passed, report.scenarios_total);
    }
}
