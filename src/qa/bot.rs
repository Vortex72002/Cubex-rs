use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use glam::{IVec3, Vec3};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::autotest::{
    AutoBotTickOutput, AutoTestConfig, AutoTestMode, SurvivalNetherBot, SURVIVAL_NETHER_SEED,
};
use crate::config::EngineSettings;
use crate::lua_api::userdata::ObjectState;
use crate::lua_api::LuaRuntime;
use crate::physics::collision::{GameMode, MoveInput, PlayerController};
use crate::physics::raycast::{
    add_drops_to_inventory, held_item_is_empty_bucket, raycast_voxel_ex, try_interact_or_place,
    DigController,
};
use crate::registry::{EngineRegistries, CONTENT_AIR};
use crate::world::mapgen::{
    chunk_y_range_for, find_spawn_pos, generate_chunk_for_world, MapgenPalette,
};
use crate::world::nether::{ensure_nether_columns, NetherPalette};
use crate::world::portal::{
    find_or_build_destination_portal, is_nether_portal_node, PortalAxis,
};
use crate::world::structures::{
    locate_nearest_structure, populate_structure_chests_for_chunk, StructureAnchor, StructureKind,
    StructurePalette,
};
use crate::world::{scale_portal_coords, DimensionId, MobManager, MeleeNativeResult, NativeMobKind, WorldGenType, WorldStore};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaythroughReport {
    pub passed: bool,
    pub seed: u64,
    pub stages_completed: u8,
    pub total_stages: u8,
    pub final_stage: String,
    pub nether_entered: bool,
    pub mobs_defeated: u32,
    pub chest_looted_items: u32,
    pub ocean_monument_verified: bool,
    pub ocean_monument_origin: Option<[i32; 3]>,
    pub nether_fortress_origin: Option<[i32; 3]>,
    pub final_hp: i32,
    pub final_hunger: i32,
    pub avg_fps: f32,
    pub elapsed_sim_secs: f32,
    pub wall_secs: f64,
    pub failure_reason: Option<String>,
}

#[derive(Debug, Clone)]
pub struct QaBotHudStatus {
    pub stage_num: u8,
    pub total_stages: u8,
    pub goal: String,
    pub elapsed_secs: f32,
}

#[derive(Debug, Default)]
pub struct QaBotTickOutput {
    pub move_input: MoveInput,
    pub mine_held: bool,
    pub place_just: bool,
    pub melee_attack_just: bool,
    pub force_liquids_pointable: bool,
    pub remesh_positions: Vec<IVec3>,
    pub teleport_request: Option<(DimensionId, Vec3)>,
    pub finished: Option<PlaythroughReport>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QaBotPhase {
    SurvivalStages1To5,
    NetherFortressCombatAndLoot,
    VerifyOceanMonument,
    Done,
}

/// Live & headless QA playthrough bot: completes Stages 1–5 via `SurvivalNetherBot`,
/// then fights a Nether mob, loots a Nether structure chest, and verifies an Ocean Monument.
pub struct QaPlaythroughBot {
    out_dir: PathBuf,
    seed: u64,
    started_wall: Instant,
    base_bot: SurvivalNetherBot,
    phase: QaBotPhase,
    stages_completed: u8,
    elapsed_sim_secs: f32,
    goal_label: String,
    fps_sum: f64,
    frame_count: u64,

    // Stage 6 & 7 state
    fortress_anchor: Option<StructureAnchor>,
    fortress_chunks_ready: bool,
    fortress_teleported: bool,
    target_mob_id: Option<u64>,
    melee_cooldown: f32,
    mobs_defeated: u32,
    chest_looted_items: u32,
    monument_anchor: Option<StructureAnchor>,
    ocean_monument_verified: bool,
    phase_timer: f32,
}

impl QaPlaythroughBot {
    pub fn new(out_dir: PathBuf, player_spawn: Vec3) -> Self {
        let cfg = AutoTestConfig {
            mode: AutoTestMode::SurvivalNether,
            out_dir: out_dir.clone(),
        };
        Self {
            out_dir,
            seed: SURVIVAL_NETHER_SEED,
            started_wall: Instant::now(),
            base_bot: SurvivalNetherBot::new(cfg, player_spawn),
            phase: QaBotPhase::SurvivalStages1To5,
            stages_completed: 0,
            elapsed_sim_secs: 0.0,
            goal_label: "Locating trees".into(),
            fps_sum: 0.0,
            frame_count: 0,
            fortress_anchor: None,
            fortress_chunks_ready: false,
            fortress_teleported: false,
            target_mob_id: None,
            melee_cooldown: 0.0,
            mobs_defeated: 0,
            chest_looted_items: 0,
            monument_anchor: None,
            ocean_monument_verified: false,
            phase_timer: 0.0,
        }
    }

    pub fn out_dir(&self) -> &Path {
        &self.out_dir
    }

    pub fn hud_status(&self) -> QaBotHudStatus {
        let stage_num = match self.phase {
            QaBotPhase::SurvivalStages1To5 => self.base_bot.hud_status().stage_num,
            QaBotPhase::NetherFortressCombatAndLoot => 6,
            QaBotPhase::VerifyOceanMonument | QaBotPhase::Done => 7,
        };
        QaBotHudStatus {
            stage_num,
            total_stages: 7,
            goal: self.goal_label.clone(),
            elapsed_secs: self.elapsed_sim_secs,
        }
    }

    fn avg_fps(&self, fallback: f32) -> f32 {
        if self.frame_count > 0 {
            (self.fps_sum / self.frame_count as f64) as f32
        } else {
            fallback
        }
    }

    fn build_report(
        &self,
        passed: bool,
        player: &PlayerController,
        current_fps: f32,
        failure_reason: Option<String>,
    ) -> PlaythroughReport {
        PlaythroughReport {
            passed,
            seed: self.seed,
            stages_completed: self.stages_completed,
            total_stages: 7,
            final_stage: match self.phase {
                QaBotPhase::SurvivalStages1To5 => self.base_bot.hud_status().stage_label.to_string(),
                QaBotPhase::NetherFortressCombatAndLoot => "Nether Combat & Chest Loot".into(),
                QaBotPhase::VerifyOceanMonument => "Ocean Monument Verification".into(),
                QaBotPhase::Done => "Completed 7/7 QA Stages".into(),
            },
            nether_entered: self.stages_completed >= 5 || player.dimension.is_nether(),
            mobs_defeated: self.mobs_defeated,
            chest_looted_items: self.chest_looted_items,
            ocean_monument_verified: self.ocean_monument_verified,
            ocean_monument_origin: self
                .monument_anchor
                .as_ref()
                .map(|a| [a.origin.x, a.origin.y, a.origin.z]),
            nether_fortress_origin: self
                .fortress_anchor
                .as_ref()
                .map(|a| [a.origin.x, a.origin.y, a.origin.z]),
            final_hp: player.hp,
            final_hunger: player.hunger,
            avg_fps: self.avg_fps(current_fps),
            elapsed_sim_secs: self.elapsed_sim_secs,
            wall_secs: self.started_wall.elapsed().as_secs_f64(),
            failure_reason,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn tick(
        &mut self,
        dt: f32,
        current_fps: f32,
        player: &mut PlayerController,
        player_state: &Arc<RwLock<ObjectState>>,
        ow_world: &Arc<WorldStore>,
        nw_world: &Arc<WorldStore>,
        ow_palette: &MapgenPalette,
        nether_palette: &NetherPalette,
        reg: &EngineRegistries,
        lua_rt: Option<&LuaRuntime>,
        mob_manager: &mut MobManager,
        physical_drops: bool,
    ) -> QaBotTickOutput {
        let mut out = QaBotTickOutput::default();
        if self.phase == QaBotPhase::Done {
            return out;
        }

        let dt = dt.clamp(0.001, 0.05);
        self.elapsed_sim_secs += dt;
        self.frame_count += 1;
        if current_fps.is_finite() && current_fps > 0.5 {
            self.fps_sum += current_fps as f64;
        }
        self.melee_cooldown = (self.melee_cooldown - dt).max(0.0);
        self.phase_timer += dt;

        match self.phase {
            QaBotPhase::SurvivalStages1To5 => {
                let cur_world = if player.dimension.is_nether() {
                    nw_world
                } else {
                    ow_world
                };
                let base_out: AutoBotTickOutput = self.base_bot.tick(
                    dt,
                    current_fps,
                    player,
                    player_state,
                    cur_world,
                    reg,
                    lua_rt,
                    physical_drops,
                );
                let hud = self.base_bot.hud_status();
                self.goal_label = format!("{} ({}/5)", hud.detail, hud.stage_num);
                out.move_input = base_out.move_input;
                out.mine_held = base_out.mine_held;
                out.place_just = base_out.place_just;
                out.force_liquids_pointable = base_out.force_liquids_pointable;
                out.remesh_positions = base_out.remesh_positions;

                if let Some(rep) = base_out.finished {
                    self.stages_completed = self.stages_completed.max(rep.stages_completed);
                    if !rep.passed {
                        let fail_rep = self.build_report(false, player, current_fps, rep.failure_reason);
                        self.phase = QaBotPhase::Done;
                        out.finished = Some(fail_rep);
                        return out;
                    }
                    self.phase = QaBotPhase::NetherFortressCombatAndLoot;
                    self.phase_timer = 0.0;
                    self.goal_label = "Locating Nether Fortress & engaging mobs".into();
                }
            }
            QaBotPhase::NetherFortressCombatAndLoot => {
                if self.phase_timer > 60.0 {
                    let rep = self.build_report(
                        false,
                        player,
                        current_fps,
                        Some("Timed out in Stage 6 (Nether Fortress combat & chest loot)".into()),
                    );
                    self.phase = QaBotPhase::Done;
                    out.finished = Some(rep);
                    return out;
                }
                if self.fortress_anchor.is_none() {
                    self.fortress_anchor = locate_nearest_structure(
                        StructureKind::NetherFortress,
                        self.seed,
                        WorldGenType::Default,
                        (player.pos.x.floor() as i32, player.pos.z.floor() as i32),
                    )
                    .or_else(|| {
                        locate_nearest_structure(
                            StructureKind::BastionRemnant,
                            self.seed,
                            WorldGenType::Default,
                            (0, 0),
                        )
                    });
                }
                let Some(anchor) = self.fortress_anchor.clone() else {
                    let rep = self.build_report(
                        false,
                        player,
                        current_fps,
                        Some("Failed to locate Nether Fortress or Bastion Remnant".into()),
                    );
                    self.phase = QaBotPhase::Done;
                    out.finished = Some(rep);
                    return out;
                };

                if !self.fortress_chunks_ready {
                    let acx = anchor.origin.x.div_euclid(16);
                    let acz = anchor.origin.z.div_euclid(16);
                    let cols = ensure_nether_columns(
                        nw_world,
                        acx,
                        acz,
                        2,
                        self.seed,
                        nether_palette,
                        reg,
                    );
                    for &cp in &cols {
                        populate_structure_chests_for_chunk(
                            nw_world,
                            cp,
                            self.seed,
                            WorldGenType::Default,
                            reg,
                        );
                    }
                    // Also populate existing overlapping fortress columns
                    for cz in (anchor.bbox_min.z.div_euclid(16))..=(anchor.bbox_max.z.div_euclid(16)) {
                        for cx in (anchor.bbox_min.x.div_euclid(16))..=(anchor.bbox_max.x.div_euclid(16)) {
                            for cy in crate::world::nether::NETHER_CHUNK_Y_RANGE {
                                populate_structure_chests_for_chunk(
                                    nw_world,
                                    IVec3::new(cx, cy, cz),
                                    self.seed,
                                    WorldGenType::Default,
                                    reg,
                                );
                            }
                        }
                    }
                    self.fortress_chunks_ready = true;
                }

                if !self.fortress_teleported {
                    self.fortress_teleported = true;
                    let target = anchor.teleport_pos + Vec3::new(0.0, 0.25, 0.0);
                    out.teleport_request = Some((DimensionId::Nether, target));
                    self.goal_label = "Teleporting to Nether Fortress".into();
                    return out;
                }

                // Combat step: engage and defeat at least 1 Nether mob
                if self.mobs_defeated == 0 {
                    equip_best_qa_weapon(player, player_state);
                    let tracked_dead = self
                        .target_mob_id
                        .is_some_and(|tid| !mob_manager.mobs.iter().any(|m| m.id == tid && m.hp > 0));
                    if tracked_dead {
                        self.mobs_defeated += 1;
                        self.target_mob_id = None;
                    }
                }

                if self.mobs_defeated == 0 {
                    self.goal_label = "Engaging Nether Fortress mob".into();
                    let nearest = mob_manager
                        .mobs
                        .iter()
                        .filter(|m| m.dimension == DimensionId::Nether && m.hp > 0)
                        .min_by(|a, b| {
                            a.pos
                                .distance_squared(player.pos)
                                .total_cmp(&b.pos.distance_squared(player.pos))
                        })
                        .map(|m| (m.id, m.pos));

                    let (mid, mpos) = match nearest {
                        Some((id, p)) if p.distance(player.pos) <= 14.0 => (id, p),
                        _ => {
                            let spawn_p = Vec3::new(
                                anchor.teleport_pos.x,
                                anchor.teleport_pos.y + 0.1,
                                anchor.teleport_pos.z + 3.0,
                            );
                            let id = mob_manager.spawn_mob(
                                NativeMobKind::Blaze,
                                DimensionId::Nether,
                                spawn_p,
                                1,
                            );
                            (id, spawn_p)
                        }
                    };
                    self.target_mob_id = Some(mid);
                    let aim_target = mpos + Vec3::new(0.0, 0.9, 0.0);
                    let aimed = aim_towards(player, aim_target, dt);
                    let d = mpos.distance(player.pos);
                    if d > 2.3 {
                        out.move_input.forward = true;
                    }
                    if aimed && d <= 3.6 && self.melee_cooldown <= 0.0 {
                        out.melee_attack_just = true;
                        out.mine_held = true;
                        self.melee_cooldown = 0.25;
                    }
                    return out;
                }

                // Chest loot step: loot fortress structure chest
                if self.chest_looted_items == 0 {
                    self.goal_label = "Looting Nether Fortress chest".into();
                    for &(cpos, _) in &anchor.chests {
                        let chunk_pos = IVec3::new(
                            cpos.x.div_euclid(16),
                            cpos.y.div_euclid(16),
                            cpos.z.div_euclid(16),
                        );
                        populate_structure_chests_for_chunk(
                            nw_world,
                            chunk_pos,
                            self.seed,
                            WorldGenType::Default,
                            reg,
                        );
                        let meta = nw_world.get_node_meta(cpos);
                        let mut taken_specs = Vec::new();
                        {
                            let mut mg = meta.write();
                            if let Some(inv) = mg.inventories.get_mut("main") {
                                for slot in inv.iter_mut() {
                                    if !slot.is_empty() {
                                        taken_specs.push(format!("{} {}", slot.name, slot.count));
                                        self.chest_looted_items += slot.count as u32;
                                        *slot = Default::default();
                                    }
                                }
                            }
                        }
                        if !taken_specs.is_empty() {
                            add_drops_to_inventory(player_state, &taken_specs, reg);
                            break;
                        }
                    }
                    if self.chest_looted_items == 0 {
                        let rep = self.build_report(
                            false,
                            player,
                            current_fps,
                            Some("Nether Fortress chests contained 0 loot items".into()),
                        );
                        self.phase = QaBotPhase::Done;
                        out.finished = Some(rep);
                        return out;
                    }
                }

                self.stages_completed = self.stages_completed.max(6);
                self.phase = QaBotPhase::VerifyOceanMonument;
                self.phase_timer = 0.0;
                self.goal_label = "Verifying Ocean Monument generation".into();
            }
            QaBotPhase::VerifyOceanMonument => {
                let anchor = self.monument_anchor.get_or_insert_with(|| {
                    locate_nearest_structure(
                        StructureKind::OceanMonument,
                        self.seed,
                        WorldGenType::Default,
                        (0, 0),
                    )
                    .expect("Ocean Monument deterministic anchor")
                });

                let min_cx = anchor.bbox_min.x.div_euclid(16);
                let max_cx = anchor.bbox_max.x.div_euclid(16);
                let min_cz = anchor.bbox_min.z.div_euclid(16);
                let max_cz = anchor.bbox_max.z.div_euclid(16);
                for cz in min_cz..=max_cz {
                    for cx in min_cx..=max_cx {
                        for cy in chunk_y_range_for(WorldGenType::Default) {
                            let cp = IVec3::new(cx, cy, cz);
                            if ow_world.get_chunk(cp).is_none() {
                                ow_world.insert_chunk(generate_chunk_for_world(
                                    cp,
                                    self.seed,
                                    WorldGenType::Default,
                                    ow_palette,
                                    reg,
                                ));
                            }
                            populate_structure_chests_for_chunk(
                                ow_world,
                                cp,
                                self.seed,
                                WorldGenType::Default,
                                reg,
                            );
                        }
                    }
                }

                let sp = StructurePalette::from_registries(reg);
                let mut found_monument_blocks = false;
                'scan: for y in anchor.bbox_min.y..=anchor.bbox_max.y {
                    for z in (anchor.origin.z - 8)..=(anchor.origin.z + 8) {
                        for x in (anchor.origin.x - 8)..=(anchor.origin.x + 8) {
                            let id = ow_world.get_node(IVec3::new(x, y, z)).0;
                            if id != CONTENT_AIR
                                && (id == sp.prismarine
                                    || id == sp.prismarine_brick
                                    || id == sp.sea_lantern
                                    || id == sp.chest)
                            {
                                found_monument_blocks = true;
                                break 'scan;
                            }
                        }
                    }
                }
                let vault_chest_has_loot = anchor.chests.iter().any(|&(cpos, _)| {
                    let meta = ow_world.get_node_meta(cpos);
                    let guard = meta.read();
                    guard
                        .inventories
                        .get("main")
                        .is_some_and(|v| v.iter().any(|s| !s.is_empty()))
                });

                if !found_monument_blocks || !vault_chest_has_loot {
                    let rep = self.build_report(
                        false,
                        player,
                        current_fps,
                        Some(format!(
                            "Ocean Monument verification failed (blocks={found_monument_blocks}, vault_loot={vault_chest_has_loot})"
                        )),
                    );
                    self.phase = QaBotPhase::Done;
                    out.finished = Some(rep);
                    return out;
                }

                self.ocean_monument_verified = true;
                self.stages_completed = 7;
                self.phase = QaBotPhase::Done;
                self.goal_label = "Completed all 7 QA stages".into();
                out.finished = Some(self.build_report(true, player, current_fps, None));
            }
            QaBotPhase::Done => {}
        }

        out
    }
}

fn aim_towards(player: &mut PlayerController, target: Vec3, dt: f32) -> bool {
    let eye = player.eye_pos();
    let d = target - eye;
    if d.length_squared() < 1e-6 {
        return true;
    }
    let horiz = (d.x * d.x + d.z * d.z).sqrt().max(1e-5);
    let desired_yaw = (-d.x).atan2(d.z);
    let desired_pitch = d.y.atan2(horiz).clamp(-1.48, 1.48);

    let mut dyaw = desired_yaw - player.yaw;
    while dyaw > std::f32::consts::PI {
        dyaw -= std::f32::consts::TAU;
    }
    while dyaw < -std::f32::consts::PI {
        dyaw += std::f32::consts::TAU;
    }
    let dpitch = desired_pitch - player.pitch;
    let step = 14.0 * dt;
    player.yaw = (player.yaw + dyaw.clamp(-step, step) + std::f32::consts::PI)
        .rem_euclid(std::f32::consts::TAU)
        - std::f32::consts::PI;
    player.pitch = (player.pitch + dpitch.clamp(-step, step)).clamp(-1.48, 1.48);
    dyaw.abs() < 0.14 && dpitch.abs() < 0.14
}

fn equip_best_qa_weapon(player: &mut PlayerController, player_state: &Arc<RwLock<ObjectState>>) {
    let prefs = [
        "mcl_tools:sword_diamond",
        "mcl_tools:sword_iron",
        "mcl_tools:sword_stone",
        "mcl_tools:sword_wood",
        "mcl_tools:pick_iron",
        "mcl_tools:pick_stone",
        "mcl_tools:pick_wood",
    ];
    let mut best_slot: Option<usize> = None;
    let mut best_rank = usize::MAX;
    {
        let ps = player_state.read();
        let mg = ps.meta.read();
        if let Some(main) = mg.inventories.get("main") {
            for (idx, stack) in main.iter().take(9).enumerate() {
                if stack.is_empty() {
                    continue;
                }
                if let Some(rank) = prefs.iter().position(|&p| p == stack.name) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_slot = Some(idx);
                    }
                }
            }
        }
    }
    if let Some(slot) = best_slot {
        player.selected_slot = slot;
        player_state.write().wield_index = slot + 1;
    }
}

/// Phase 3 headless runner used by `--qa-suite` to validate Stages 1–7 end-to-end.
pub fn run_headless_playthrough(
    seed: u64,
    reg: &EngineRegistries,
    _lua_rt: Option<&LuaRuntime>,
) -> PlaythroughReport {
    let dir = std::env::temp_dir().join("cubex_qa_headless_playthrough");
    let _ = std::fs::remove_dir_all(&dir);

    let ow_pal = MapgenPalette::from_registries_with_game_dir(reg, None);
    let nw_pal = NetherPalette::from_registries(reg);
    let ow = Arc::new(WorldStore::new(&dir.join("ow"), seed));
    let nw = Arc::new(WorldStore::new_for_dimension(
        &dir.join("nw"),
        seed,
        DimensionId::Nether,
        WorldGenType::Default,
        crate::world::Difficulty::Normal,
        crate::world::Gamerules::default(),
        "Survival",
    ));

    let mut ow_cols = HashSet::new();
    for cz in -5..=5 {
        for cx in -5..=5 {
            ow_cols.insert((cx, cz));
            for cy in chunk_y_range_for(WorldGenType::Default) {
                ow.insert_chunk(generate_chunk_for_world(
                    IVec3::new(cx, cy, cz),
                    seed,
                    WorldGenType::Default,
                    &ow_pal,
                    reg,
                ));
            }
        }
    }

    let sp = find_spawn_pos(seed, WorldGenType::Default);
    let mut player = PlayerController::default();
    player.gamemode = GameMode::Survival;
    player.pos = Vec3::new(sp.x as f32 + 0.5, sp.y as f32, sp.z as f32 + 0.5);
    let ps = Arc::new(RwLock::new(ObjectState::new_player("QaBot")));
    let settings = EngineSettings::default();
    let mut digger = DigController::new();
    let mut mob_manager = MobManager::new(seed);
    let mut bot = QaPlaythroughBot::new(dir.join("out"), player.pos);
    let dt = 0.05f32;
    let mut final_rep: Option<PlaythroughReport> = None;

    for _ in 0..18000 {
        if !player.dimension.is_nether() {
            let pcx = (player.pos.x / 16.0).floor() as i32;
            let pcz = (player.pos.z / 16.0).floor() as i32;
            for dz in -3..=3 {
                for dx in -3..=3 {
                    if ow_cols.insert((pcx + dx, pcz + dz)) {
                        for cy in chunk_y_range_for(WorldGenType::Default) {
                            ow.insert_chunk(generate_chunk_for_world(
                                IVec3::new(pcx + dx, cy, pcz + dz),
                                seed,
                                WorldGenType::Default,
                                &ow_pal,
                                reg,
                            ));
                        }
                    }
                }
            }
        }

        let out = bot.tick(
            dt,
            60.0,
            &mut player,
            &ps,
            &ow,
            &nw,
            &ow_pal,
            &nw_pal,
            reg,
            None,
            &mut mob_manager,
            false,
        );

        if let Some((dest_dim, dest_pos)) = out.teleport_request {
            player.dimension = dest_dim;
            player.pos = dest_pos;
            player.vel = Vec3::ZERO;
            player.reset_fall_state();
        }
        if let Some(rep) = out.finished {
            final_rep = Some(rep);
            break;
        }

        let cur_world = if player.dimension.is_nether() {
            &nw
        } else {
            &ow
        };
        player.step(dt, &out.move_input, cur_world, reg, &settings);

        // Portal charge & teleportation
        let p_feet = IVec3::new(
            player.pos.x.floor() as i32,
            (player.pos.y + 0.1).floor() as i32,
            player.pos.z.floor() as i32,
        );
        let p_eye = IVec3::new(
            player.pos.x.floor() as i32,
            (player.pos.y + 1.3).floor() as i32,
            player.pos.z.floor() as i32,
        );
        if is_nether_portal_node(cur_world.get_node(p_feet).0, reg)
            || is_nether_portal_node(cur_world.get_node(p_eye).0, reg)
        {
            player.portal_charge = (player.portal_charge + dt / 3.0).min(1.0);
            if player.portal_charge >= 1.0 && !player.dimension.is_nether() {
                let scaled = scale_portal_coords(p_feet, DimensionId::Overworld, DimensionId::Nether);
                let _ = ensure_nether_columns(
                    &nw,
                    scaled.x.div_euclid(16),
                    scaled.z.div_euclid(16),
                    1,
                    seed,
                    &nw_pal,
                    reg,
                );
                let stand = find_or_build_destination_portal(&nw, reg, scaled, PortalAxis::X);
                player.dimension = DimensionId::Nether;
                player.pos = Vec3::new(stand.x as f32 + 0.5, stand.y as f32 + 0.05, stand.z as f32 + 0.5);
                player.vel = Vec3::ZERO;
                player.portal_charge = 0.0;
                player.reset_fall_state();
            }
        }

        if out.melee_attack_just {
            if let Some(res) = mob_manager.try_melee_hit(
                player.eye_pos(),
                player.look_dir(),
                4.2,
                player.dimension,
                0,
                7,
            ) {
                if let MeleeNativeResult::HitMob { drops, .. } = res {
                    if !drops.is_empty() {
                        add_drops_to_inventory(&ps, &drops, reg);
                    }
                }
            }
        }

        let point_liq = out.force_liquids_pointable || held_item_is_empty_bucket(&player, &ps);
        let hit = raycast_voxel_ex(
            player.eye_pos(),
            player.look_dir(),
            5.5,
            point_liq,
            cur_world,
            reg,
        );
        let _ = digger.update_with_physical_drops(
            dt,
            out.mine_held && !out.melee_attack_just,
            hit.as_ref(),
            cur_world,
            reg,
            &player,
            &ps,
            None,
            false,
        );
        if out.place_just {
            if let Some(ref h) = hit {
                let _ = try_interact_or_place(h, cur_world, reg, &player, &ps, None);
            }
        }
    }

    let _ = std::fs::remove_dir_all(&dir);
    final_rep.unwrap_or_else(|| PlaythroughReport {
        passed: false,
        seed,
        stages_completed: bot.stages_completed,
        total_stages: 7,
        final_stage: bot.hud_status().goal,
        nether_entered: player.dimension.is_nether(),
        mobs_defeated: bot.mobs_defeated,
        chest_looted_items: bot.chest_looted_items,
        ocean_monument_verified: bot.ocean_monument_verified,
        ocean_monument_origin: None,
        nether_fortress_origin: None,
        final_hp: player.hp,
        final_hunger: player.hunger,
        avg_fps: 60.0,
        elapsed_sim_secs: bot.elapsed_sim_secs,
        wall_secs: bot.started_wall.elapsed().as_secs_f64(),
        failure_reason: Some("Headless playthrough exceeded max simulation tick budget".into()),
    })
}
