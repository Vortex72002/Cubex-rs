use std::collections::{HashMap, HashSet};
use std::f32::consts::{PI, TAU};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use glam::{IVec3, Vec3};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::lua_api::core_api::resolve_craft_recipe;
use crate::lua_api::userdata::{ItemStackData, ObjectState};
use crate::lua_api::LuaRuntime;
use crate::physics::collision::{is_liquid_node, MoveInput, PlayerController};
use crate::physics::raycast::{add_drops_to_inventory, is_empty_bucket_item, raycast_voxel_ex};
use crate::registry::{EngineRegistries, CONTENT_AIR, CONTENT_IGNORE, CONTENT_UNKNOWN};
use crate::world::biome::SEA_LEVEL;
use crate::world::portal::{detect_portal_frame, is_nether_portal_node, is_obsidian, PortalFrame};
use crate::world::{DimensionId, WorldStore, CHUNK_SIZE};

pub const SURVIVAL_NETHER_SEED: u64 = 133742;
pub const DEFAULT_AUTOTEST_OUT_DIR: &str = "autotest_report";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoTestMode {
    SurvivalNether,
}

impl AutoTestMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SurvivalNether => "survival_nether",
        }
    }
}

#[derive(Debug, Clone)]
pub struct AutoTestConfig {
    pub mode: AutoTestMode,
    pub out_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoTestReport {
    pub passed: bool,
    pub mode: String,
    pub seed: u64,
    pub stages_completed: u8,
    pub final_stage: String,
    pub final_dimension: String,
    pub final_hp: i32,
    pub final_hunger: i32,
    pub avg_fps: f32,
    pub elapsed_secs: f32,
    pub screenshot: String,
    pub failure_reason: Option<String>,
}

#[derive(Debug, Clone)]
pub struct AutoBotHudStatus {
    pub stage_num: u8,
    pub stage_label: &'static str,
    pub detail: String,
}

#[derive(Debug, Default)]
pub struct AutoBotTickOutput {
    pub move_input: MoveInput,
    pub mine_held: bool,
    pub place_just: bool,
    pub force_liquids_pointable: bool,
    pub remesh_positions: Vec<IVec3>,
    pub finished: Option<AutoTestReport>,
}

/// Parses `--autotest <mode>` and optional `--autotest-out <dir>` CLI arguments.
pub fn parse_autotest_args(args: &[String]) -> anyhow::Result<Option<AutoTestConfig>> {
    let mut mode: Option<AutoTestMode> = None;
    let mut out_dir = PathBuf::from(DEFAULT_AUTOTEST_OUT_DIR);
    let mut i = 1usize;
    while i < args.len() {
        let arg = &args[i];
        if let Some(val) = arg.strip_prefix("--autotest=") {
            mode = Some(parse_autotest_mode(val)?);
            i += 1;
        } else if arg == "--autotest" {
            let val = args
                .get(i + 1)
                .ok_or_else(|| anyhow::anyhow!("Missing value after --autotest"))?;
            mode = Some(parse_autotest_mode(val)?);
            i += 2;
        } else if let Some(val) = arg.strip_prefix("--autotest-out=") {
            if val.trim().is_empty() {
                anyhow::bail!("Empty path passed to --autotest-out");
            }
            out_dir = PathBuf::from(val);
            i += 1;
        } else if arg == "--autotest-out" {
            let val = args
                .get(i + 1)
                .ok_or_else(|| anyhow::anyhow!("Missing value after --autotest-out"))?;
            if val.trim().is_empty() {
                anyhow::bail!("Empty path passed to --autotest-out");
            }
            out_dir = PathBuf::from(val);
            i += 2;
        } else {
            i += 1;
        }
    }
    Ok(mode.map(|m| AutoTestConfig { mode: m, out_dir }))
}

fn parse_autotest_mode(raw: &str) -> anyhow::Result<AutoTestMode> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "survival_nether" => Ok(AutoTestMode::SurvivalNether),
        other => anyhow::bail!(
            "Unsupported --autotest mode '{other}' (supported: survival_nether)"
        ),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BotPhase {
    // Stage 1
    S1GatherLogs,
    S1CraftAndPlaceTable,
    // Stage 2
    S2DigStone,
    S2CraftStonePick,
    S2MineOresAndStone,
    S2ReturnCraftFurnace,
    S2GatherFlint,
    // Stage 3
    S3PrepareAndPlaceFurnace,
    S3SmeltIron,
    S3CraftPortalTools,
    // Stage 4
    S4FindWaterAndFillBucket,
    S4FindLavaPool,
    S4PreparePortalSite,
    S4CastObsidianFrame,
    // Stage 5
    S5IgnitePortal,
    S5EnterPortal,
    S5WalkNetherGround,
    S5FinishHold,
    Done,
}

impl BotPhase {
    fn stage_num(self) -> u8 {
        match self {
            Self::S1GatherLogs | Self::S1CraftAndPlaceTable => 1,
            Self::S2DigStone
            | Self::S2CraftStonePick
            | Self::S2MineOresAndStone
            | Self::S2ReturnCraftFurnace
            | Self::S2GatherFlint => 2,
            Self::S3PrepareAndPlaceFurnace | Self::S3SmeltIron | Self::S3CraftPortalTools => 3,
            Self::S4FindWaterAndFillBucket
            | Self::S4FindLavaPool
            | Self::S4PreparePortalSite
            | Self::S4CastObsidianFrame => 4,
            Self::S5IgnitePortal
            | Self::S5EnterPortal
            | Self::S5WalkNetherGround
            | Self::S5FinishHold
            | Self::Done => 5,
        }
    }

    fn stage_label(self) -> &'static str {
        match self.stage_num() {
            1 => "Gather Wood & Craft",
            2 => "Mine Stone, Iron & Flint",
            3 => "Smelt & Craft Portal Tools",
            4 => "Build/Cast Obsidian Portal",
            _ => "Ignite & Enter Nether",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CastStep {
    ScoopLava,
    PourLava,
    ScoopWater,
    PourWaterCool,
    RescoopWater,
    RefillReservoir,
}

#[derive(Debug, Clone, Copy)]
struct FrameCellSpec {
    pos: IVec3,
    support: IVec3,
    face_normal: IVec3,
}

pub struct SurvivalNetherBot {
    config: AutoTestConfig,
    phase: BotPhase,
    stages_completed: u8,
    detail: String,
    elapsed_secs: f32,
    frame_count: u64,
    fps_sum: f64,

    // Target / cooldown / anti-stuck tracking
    spawn_origin: IVec3,
    current_target: Option<IVec3>,
    blacklisted: HashSet<IVec3>,
    pickup_wait: f32,
    place_cooldown: f32,
    explore_idx: usize,
    last_progress_pos: Vec3,
    no_move_timer: f32,
    unstuck_dig_target: Option<IVec3>,
    target_timer: f32,

    // Progression state
    primary_log_item: Option<String>,
    primary_plank_item: Option<String>,
    crafting_table_pos: Option<IVec3>,
    furnace_pos: Option<IVec3>,
    furnace_loaded: bool,
    furnace_fallback_timer: f32,
    gravel_place_pos: Option<IVec3>,
    lava_pool_center: Option<IVec3>,
    portal_origin: Option<IVec3>,
    reservoir_pos: Option<IVec3>,
    scaffold_cells: Vec<(IVec3, IVec3)>,
    frame_cells: Vec<FrameCellSpec>,
    cast_index: usize,
    cast_step: CastStep,
    cool_water_pos: Option<IVec3>,
    detected_frame: Option<PortalFrame>,
    nether_entry_pos: Option<Vec3>,
    nether_walk_dir: Option<IVec3>,
    finish_hold_timer: f32,
    last_smelt_total: u16,
}

impl SurvivalNetherBot {
    pub fn new(config: AutoTestConfig, player_spawn: Vec3) -> Self {
        let spawn_origin = IVec3::new(
            player_spawn.x.floor() as i32,
            player_spawn.y.floor() as i32,
            player_spawn.z.floor() as i32,
        );
        Self {
            config,
            phase: BotPhase::S1GatherLogs,
            stages_completed: 0,
            detail: "Locating trees".into(),
            elapsed_secs: 0.0,
            frame_count: 0,
            fps_sum: 0.0,
            spawn_origin,
            current_target: None,
            blacklisted: HashSet::new(),
            pickup_wait: 0.0,
            place_cooldown: 0.0,
            explore_idx: 0,
            last_progress_pos: player_spawn,
            no_move_timer: 0.0,
            unstuck_dig_target: None,
            target_timer: 0.0,
            primary_log_item: None,
            primary_plank_item: None,
            crafting_table_pos: None,
            furnace_pos: None,
            furnace_loaded: false,
            furnace_fallback_timer: 0.0,
            gravel_place_pos: None,
            lava_pool_center: None,
            portal_origin: None,
            reservoir_pos: None,
            scaffold_cells: Vec::new(),
            frame_cells: Vec::new(),
            cast_index: 0,
            cast_step: CastStep::ScoopLava,
            cool_water_pos: None,
            detected_frame: None,
            nether_entry_pos: None,
            nether_walk_dir: None,
            finish_hold_timer: 0.0,
            last_smelt_total: 0,
        }
    }

    pub fn out_dir(&self) -> &Path {
        &self.config.out_dir
    }

    pub fn hud_status(&self) -> AutoBotHudStatus {
        AutoBotHudStatus {
            stage_num: self.phase.stage_num(),
            stage_label: self.phase.stage_label(),
            detail: self.detail.clone(),
        }
    }

    fn avg_fps(&self, fallback_fps: f32) -> f32 {
        if self.frame_count > 0 {
            (self.fps_sum / self.frame_count as f64) as f32
        } else {
            fallback_fps
        }
    }

    fn build_report(
        &self,
        passed: bool,
        player: &PlayerController,
        current_fps: f32,
        failure_reason: Option<String>,
    ) -> AutoTestReport {
        let screenshot_path = self.config.out_dir.join("survival_nether.png");
        AutoTestReport {
            passed,
            mode: self.config.mode.as_str().to_string(),
            seed: SURVIVAL_NETHER_SEED,
            stages_completed: self.stages_completed,
            final_stage: self.phase.stage_label().to_string(),
            final_dimension: player.dimension.label().to_string(),
            final_hp: player.hp,
            final_hunger: player.hunger,
            avg_fps: self.avg_fps(current_fps),
            elapsed_secs: self.elapsed_secs,
            screenshot: screenshot_path.to_string_lossy().to_string(),
            failure_reason,
        }
    }

    fn transition(&mut self, next: BotPhase, completed_stage: Option<u8>, detail: &str) {
        if let Some(s) = completed_stage {
            self.stages_completed = self.stages_completed.max(s);
        }
        self.phase = next;
        self.detail = detail.to_string();
        self.current_target = None;
        self.target_timer = 0.0;
        self.no_move_timer = 0.0;
        self.unstuck_dig_target = None;
    }

    /// Advance the deterministic state machine by one simulation frame.
    #[allow(clippy::too_many_arguments)]
    pub fn tick(
        &mut self,
        dt: f32,
        current_fps: f32,
        player: &mut PlayerController,
        player_state: &Arc<RwLock<ObjectState>>,
        world: &Arc<WorldStore>,
        reg: &EngineRegistries,
        lua_rt: Option<&LuaRuntime>,
        _physical_drops: bool,
    ) -> AutoBotTickOutput {
        let mut out = AutoBotTickOutput::default();
        if self.phase == BotPhase::Done {
            return out;
        }

        let dt = dt.clamp(0.001, 0.05);
        self.elapsed_secs += dt;
        self.frame_count += 1;
        if current_fps.is_finite() && current_fps > 0.5 {
            self.fps_sum += current_fps as f64;
        }
        self.place_cooldown = (self.place_cooldown - dt).max(0.0);
        self.pickup_wait = (self.pickup_wait - dt).max(0.0);

        if player.is_dead || player.hp <= 0 {
            let reason = if player.death_cause.is_empty() {
                "AutoBot player died".to_string()
            } else {
                format!("AutoBot player died ({})", player.death_cause)
            };
            let rep = self.build_report(false, player, current_fps, Some(reason));
            self.phase = BotPhase::Done;
            out.finished = Some(rep);
            return out;
        }

        if self.elapsed_secs > 720.0 {
            let stage_num = self.phase.stage_num();
            let rep = self.build_report(
                false,
                player,
                current_fps,
                Some(format!(
                    "AutoBot timed out in Stage {stage_num}/5 ({}) at pos=({:.2},{:.2},{:.2}) target={:?} unstuck={:?} bl={} og={} liq={}",
                    self.detail,
                    player.pos.x,
                    player.pos.y,
                    player.pos.z,
                    self.current_target,
                    self.unstuck_dig_target,
                    self.blacklisted.len(),
                    player.on_ground,
                    player.in_liquid,
                )),
            );
            self.phase = BotPhase::Done;
            out.finished = Some(rep);
            return out;
        }

        // Track horizontal & landed elevation progress so swimming/jumping bobbing never resets stuck detection
        let horiz_moved = Vec3::new(
            player.pos.x - self.last_progress_pos.x,
            0.0,
            player.pos.z - self.last_progress_pos.z,
        )
        .length();
        let landed_new_y =
            player.on_ground && (player.pos.y - self.last_progress_pos.y).abs() > 0.75;
        if horiz_moved > 0.45 || landed_new_y {
            self.last_progress_pos = player.pos;
            self.no_move_timer = 0.0;
            self.unstuck_dig_target = None;
        } else {
            self.no_move_timer += dt;
        }
        self.target_timer += dt;

        // If still waiting briefly after breaking a block for physical drop magnet pickup, step in
        if self.pickup_wait > 0.0 {
            if let Some(t) = self.current_target {
                let target_c = t.as_vec3() + Vec3::new(0.5, 0.0, 0.5);
                if (Vec3::new(target_c.x, 0.0, target_c.z)
                    - Vec3::new(player.pos.x, 0.0, player.pos.z))
                .length()
                    > 0.75
                {
                    self.steer_surface(player, player_state, target_c, world, reg, dt, &mut out, false);
                }
            }
            return out;
        }

        match self.phase {
            BotPhase::S1GatherLogs => self.step_s1_gather_logs(dt, player, player_state, world, reg, &mut out),
            BotPhase::S1CraftAndPlaceTable => {
                self.step_s1_craft_and_place(dt, player, player_state, world, reg, &mut out)
            }
            BotPhase::S2DigStone => self.step_s2_dig_stone(dt, player, player_state, world, reg, &mut out),
            BotPhase::S2CraftStonePick => {
                self.step_s2_craft_stone_pick(dt, player, player_state, world, reg, &mut out)
            }
            BotPhase::S2MineOresAndStone => {
                self.step_s2_mine_ores_and_stone(dt, player, player_state, world, reg, &mut out)
            }
            BotPhase::S2ReturnCraftFurnace => {
                self.step_s2_return_craft_furnace(dt, player, player_state, world, reg, &mut out)
            }
            BotPhase::S2GatherFlint => {
                self.step_s2_gather_flint(dt, player, player_state, world, reg, &mut out)
            }
            BotPhase::S3PrepareAndPlaceFurnace => {
                self.step_s3_place_furnace(dt, player, player_state, world, reg, &mut out)
            }
            BotPhase::S3SmeltIron => {
                self.step_s3_smelt_iron(dt, player, player_state, world, reg, lua_rt, &mut out)
            }
            BotPhase::S3CraftPortalTools => {
                self.step_s3_craft_portal_tools(dt, player, player_state, world, reg, &mut out)
            }
            BotPhase::S4FindWaterAndFillBucket => {
                self.step_s4_find_water(dt, player, player_state, world, reg, &mut out)
            }
            BotPhase::S4FindLavaPool => {
                self.step_s4_find_lava_pool(dt, player, player_state, world, reg, &mut out)
            }
            BotPhase::S4PreparePortalSite => {
                self.step_s4_prepare_site(dt, player, player_state, world, reg, &mut out)
            }
            BotPhase::S4CastObsidianFrame => {
                self.step_s4_cast_frame(dt, player, player_state, world, reg, &mut out)
            }
            BotPhase::S5IgnitePortal => {
                self.step_s5_ignite(dt, player, player_state, world, reg, &mut out)
            }
            BotPhase::S5EnterPortal => {
                self.step_s5_enter_portal(dt, player, player_state, world, reg, &mut out)
            }
            BotPhase::S5WalkNetherGround => {
                self.step_s5_walk_nether(dt, player, player_state, world, reg, &mut out)
            }
            BotPhase::S5FinishHold => {
                self.finish_hold_timer += dt;
                smooth_aim_at(player, player.eye_pos() + player_horizontal_fwd(player.yaw) * 4.0 + Vec3::new(0.0, -0.25, 0.0), dt);
                if self.finish_hold_timer >= 0.35 {
                    self.phase = BotPhase::Done;
                    self.detail = "Verified Nether Arrival".into();
                    out.finished = Some(self.build_report(true, player, current_fps, None));
                }
            }
            BotPhase::Done => {}
        }

        out
    }

    // -------------------------------------------------------------------------
    // Stage 1: Gather Wood & Craft
    // -------------------------------------------------------------------------

    fn step_s1_gather_logs(
        &mut self,
        dt: f32,
        player: &mut PlayerController,
        player_state: &Arc<RwLock<ObjectState>>,
        world: &Arc<WorldStore>,
        reg: &EngineRegistries,
        out: &mut AutoBotTickOutput,
    ) {
        let total_logs = count_items_matching(player_state, is_log_item_name);
        let best_species = most_common_item_matching(player_state, is_log_item_name);
        if total_logs >= 6 && best_species.as_ref().is_some_and(|(_, c)| *c >= 4) {
            self.primary_log_item = best_species.map(|(n, _)| n);
            self.transition(BotPhase::S1CraftAndPlaceTable, None, "Crafting wood tools");
            return;
        }
        if let Some((ref name, _)) = best_species {
            if self.primary_log_item.is_none() {
                self.primary_log_item = Some(name.clone());
            }
        }
        self.detail = format!("Logs {total_logs}/6");

        let desired_species = self.primary_log_item.clone();
        if self.current_target.is_none_or(|p| {
            let (id, _, _) = world.get_node(p);
            !is_log_node_id(id, reg) || self.target_timer > 9.0
        }) {
            if let Some(old) = self.current_target.take() {
                let (id, _, _) = world.get_node(old);
                if is_log_node_id(id, reg) && self.target_timer > 9.0 {
                    self.blacklisted.insert(old);
                }
            }
            let preferred = desired_species.as_deref();
            self.current_target = find_nearest_node(
                world,
                reg,
                player.pos,
                6,
                SEA_LEVEL - 1,
                108,
                &self.blacklisted,
                |name, pos| {
                    if let Some(pref) = preferred {
                        name == pref && pos.y >= SEA_LEVEL
                    } else {
                        is_log_item_name(name) && pos.y >= SEA_LEVEL
                    }
                },
                true,
            )
            .or_else(|| {
                find_nearest_node(
                    world,
                    reg,
                    player.pos,
                    6,
                    SEA_LEVEL - 2,
                    112,
                    &self.blacklisted,
                    |name, _| is_log_item_name(name),
                    true,
                )
            });
            self.target_timer = 0.0;
        }

        if let Some(target) = self.current_target {
            if self.mine_block_or_approach(dt, player, player_state, world, reg, target, true, out) {
                self.current_target = None;
                self.pickup_wait = 0.45;
            }
        } else {
            self.detail = "Exploring for trees".into();
            self.explore_outward(dt, player, player_state, world, reg, out);
        }
    }

    fn step_s1_craft_and_place(
        &mut self,
        dt: f32,
        player: &mut PlayerController,
        player_state: &Arc<RwLock<ObjectState>>,
        world: &Arc<WorldStore>,
        reg: &EngineRegistries,
        out: &mut AutoBotTickOutput,
    ) {
        let Some(log_id) = self
            .primary_log_item
            .clone()
            .or_else(|| most_common_item_matching(player_state, is_log_item_name).map(|(n, _)| n))
        else {
            self.transition(BotPhase::S1GatherLogs, None, "Need more logs");
            return;
        };

        // 1. Craft up to 4 logs -> planks (16 planks)
        if self.primary_plank_item.is_none() || count_items_matching(player_state, is_plank_item_name) < 8 {
            if count_item(player_state, &log_id) > 0 {
                if let Some(crafted) = craft_via_registry(reg, player_state, 2, &[Some((&log_id, 1)), None, None, None]) {
                    if self.primary_plank_item.is_none() {
                        self.primary_plank_item = Some(crafted);
                    }
                    return;
                }
            }
        }

        let Some(plank_id) = self
            .primary_plank_item
            .clone()
            .or_else(|| most_common_item_matching(player_state, is_plank_item_name).map(|(n, _)| n))
        else {
            self.transition(BotPhase::S1GatherLogs, None, "Gathering logs for planks");
            return;
        };

        // Craft additional planks if still below 12 and logs remain
        if count_items_matching(player_state, is_plank_item_name) < 11
            && count_item(player_state, &log_id) > 0
        {
            let _ = craft_via_registry(reg, player_state, 2, &[Some((&log_id, 1)), None, None, None]);
            return;
        }

        // 2. Craft sticks (need >= 4)
        if count_item(player_state, "mcl_core:stick") < 4 {
            let _ = craft_via_registry(
                reg,
                player_state,
                2,
                &[Some((&plank_id, 1)), None, Some((&plank_id, 1)), None],
            );
            return;
        }

        // 3. Craft crafting table if not crafted/placed yet
        if self.crafting_table_pos.is_none()
            && count_item(player_state, "mcl_crafting_table:crafting_table") == 0
        {
            let _ = craft_via_registry(
                reg,
                player_state,
                2,
                &[
                    Some((&plank_id, 1)),
                    Some((&plank_id, 1)),
                    Some((&plank_id, 1)),
                    Some((&plank_id, 1)),
                ],
            );
            return;
        }

        // 4. Place crafting table nearby on flat solid ground
        if self.crafting_table_pos.is_none() {
            if self.current_target.is_none() {
                self.current_target =
                    find_nearby_placement_spot(world, reg, player.pos, &self.blacklisted);
            }
            if let Some(place_pos) = self.current_target {
                let placed_id = world.get_node(place_pos).0;
                if reg.get_name_from_content_id(placed_id) == "mcl_crafting_table:crafting_table" {
                    self.crafting_table_pos = Some(place_pos);
                    self.current_target = None;
                } else {
                    self.detail = "Placing Crafting Table".into();
                    if !self.place_item_at(
                        dt,
                        player,
                        player_state,
                        world,
                        reg,
                        "mcl_crafting_table:crafting_table",
                        place_pos,
                        place_pos - IVec3::Y,
                        IVec3::Y,
                        out,
                    ) {
                        if self.target_timer > 6.0 {
                            self.blacklisted.insert(place_pos);
                            self.current_target = None;
                            self.target_timer = 0.0;
                        }
                    }
                    return;
                }
            }
        }

        // 5. Craft Wooden Pickaxe via 3x3 recipe
        if count_item(player_state, "mcl_tools:pick_wood") == 0 {
            let crafted = craft_via_registry(
                reg,
                player_state,
                3,
                &[
                    Some((&plank_id, 1)),
                    Some((&plank_id, 1)),
                    Some((&plank_id, 1)),
                    None,
                    Some(("mcl_core:stick", 1)),
                    None,
                    None,
                    Some(("mcl_core:stick", 1)),
                    None,
                ],
            );
            if crafted.is_none() {
                self.transition(BotPhase::S1GatherLogs, None, "Need wood for pickaxe");
                return;
            }
        }

        self.transition(
            BotPhase::S2DigStone,
            Some(1),
            "Mining stone for Stone Pickaxe",
        );
    }

    // -------------------------------------------------------------------------
    // Stage 2: Mine Stone, Iron & Flint
    // -------------------------------------------------------------------------

    fn step_s2_dig_stone(
        &mut self,
        dt: f32,
        player: &mut PlayerController,
        player_state: &Arc<RwLock<ObjectState>>,
        world: &Arc<WorldStore>,
        reg: &EngineRegistries,
        out: &mut AutoBotTickOutput,
    ) {
        let cobble = count_item(player_state, "mcl_core:cobble");
        if cobble >= 4 {
            self.transition(BotPhase::S2CraftStonePick, None, "Crafting Stone Pickaxe");
            return;
        }
        self.detail = format!("Cobblestone {cobble}/4");

        if self.current_target.is_none_or(|p| !is_stone_or_cobble(world.get_node(p).0, reg) || self.target_timer > 10.0) {
            if let Some(old) = self.current_target.take() {
                if self.target_timer > 10.0 {
                    self.blacklisted.insert(old);
                }
            }
            self.current_target = find_nearest_node(
                world,
                reg,
                player.pos,
                5,
                20,
                84,
                &self.blacklisted,
                |name, pos| {
                    (name == "mcl_core:stone" || name == "mcl_core:cobble")
                        && pos.y >= 20
                        && (pos.y as f32) < player.pos.y + 6.0
                },
                false,
            );
            self.target_timer = 0.0;
        }

        if let Some(target) = self.current_target {
            if self.mine_block_or_approach(dt, player, player_state, world, reg, target, false, out) {
                self.current_target = None;
                self.pickup_wait = 0.45;
            }
        } else {
            self.explore_outward(dt, player, player_state, world, reg, out);
        }
    }

    fn step_s2_craft_stone_pick(
        &mut self,
        dt: f32,
        player: &mut PlayerController,
        player_state: &Arc<RwLock<ObjectState>>,
        world: &Arc<WorldStore>,
        reg: &EngineRegistries,
        out: &mut AutoBotTickOutput,
    ) {
        if count_item(player_state, "mcl_tools:pick_stone") > 0 {
            self.transition(
                BotPhase::S2MineOresAndStone,
                None,
                "Mining iron, coal & cobble",
            );
            return;
        }

        // Return close to crafting table first
        if let Some(ct_pos) = self.crafting_table_pos {
            let stand = ct_pos + IVec3::new(0, 0, -2);
            if player.pos.distance(ct_pos.as_vec3() + 0.5) > 4.2 {
                self.move_or_mine_to_voxel(dt, player, player_state, world, reg, stand, out);
                return;
            }
        }

        // Ensure >= 2 sticks
        if count_item(player_state, "mcl_core:stick") < 2 {
            if let Some(plank) = most_common_item_matching(player_state, is_plank_item_name).map(|(n, _)| n) {
                let _ = craft_via_registry(
                    reg,
                    player_state,
                    2,
                    &[Some((&plank, 1)), None, Some((&plank, 1)), None],
                );
            }
        }

        let crafted = craft_via_registry(
            reg,
            player_state,
            3,
            &[
                Some(("mcl_core:cobble", 1)),
                Some(("mcl_core:cobble", 1)),
                Some(("mcl_core:cobble", 1)),
                None,
                Some(("mcl_core:stick", 1)),
                None,
                None,
                Some(("mcl_core:stick", 1)),
                None,
            ],
        );
        if crafted.is_none() {
            self.transition(BotPhase::S2DigStone, None, "Need cobble for Stone Pickaxe");
            return;
        }
        self.transition(
            BotPhase::S2MineOresAndStone,
            None,
            "Mining iron, coal & cobble",
        );
    }

    fn step_s2_mine_ores_and_stone(
        &mut self,
        dt: f32,
        player: &mut PlayerController,
        player_state: &Arc<RwLock<ObjectState>>,
        world: &Arc<WorldStore>,
        reg: &EngineRegistries,
        out: &mut AutoBotTickOutput,
    ) {
        let iron = count_raw_iron(player_state);
        let coal = count_item(player_state, "mcl_core:coal_lump");
        let cobble = count_item(player_state, "mcl_core:cobble");
        self.detail = format!("Iron {iron}/4 | Coal {coal}/2 | Cobble {cobble}/22");

        if iron >= 4 && coal >= 2 && cobble >= 22 {
            self.transition(
                BotPhase::S2ReturnCraftFurnace,
                None,
                "Returning to craft Furnace",
            );
            return;
        }

        enum Need {
            Iron,
            Coal,
            Cobble,
        }
        let need = if iron < 4 {
            Need::Iron
        } else if coal < 2 {
            Need::Coal
        } else {
            Need::Cobble
        };

        let target_still_valid = self.current_target.is_some_and(|p| {
            let n = reg.get_name_from_content_id(world.get_node(p).0);
            match need {
                Need::Iron => is_iron_ore_name(n),
                Need::Coal => is_coal_ore_name(n),
                Need::Cobble => n == "mcl_core:stone" || n == "mcl_core:cobble",
            }
        });
        if !target_still_valid || self.target_timer > 18.0 {
            if let Some(old) = self.current_target.take() {
                if self.target_timer > 18.0 {
                    self.blacklisted.insert(old);
                }
            }
            self.current_target = match need {
                Need::Iron => find_nearest_node(
                    world,
                    reg,
                    player.pos,
                    6,
                    18,
                    75,
                    &self.blacklisted,
                    |n, _| is_iron_ore_name(n),
                    false,
                ),
                Need::Coal => find_nearest_node(
                    world,
                    reg,
                    player.pos,
                    6,
                    18,
                    78,
                    &self.blacklisted,
                    |n, _| is_coal_ore_name(n),
                    false,
                ),
                Need::Cobble => find_nearest_node(
                    world,
                    reg,
                    player.pos,
                    4,
                    18,
                    78,
                    &self.blacklisted,
                    |n, _| n == "mcl_core:stone" || n == "mcl_core:cobble",
                    false,
                ),
            };
            self.target_timer = 0.0;
        }

        if let Some(target) = self.current_target {
            if self.mine_block_or_approach(dt, player, player_state, world, reg, target, false, out) {
                self.current_target = None;
                self.pickup_wait = 0.45;
            }
        } else {
            self.explore_outward(dt, player, player_state, world, reg, out);
        }
    }

    fn step_s2_return_craft_furnace(
        &mut self,
        dt: f32,
        player: &mut PlayerController,
        player_state: &Arc<RwLock<ObjectState>>,
        world: &Arc<WorldStore>,
        reg: &EngineRegistries,
        out: &mut AutoBotTickOutput,
    ) {
        if count_item(player_state, "mcl_furnaces:furnace") > 0 {
            self.transition(BotPhase::S2GatherFlint, None, "Gathering flint from gravel");
            return;
        }

        if let Some(ct_pos) = self.crafting_table_pos {
            let stand = ct_pos + IVec3::new(0, 0, -1);
            if player.pos.distance(ct_pos.as_vec3() + 0.5) > 4.2 {
                self.detail = "Returning to Crafting Table".into();
                self.move_or_mine_to_voxel(dt, player, player_state, world, reg, stand, out);
                return;
            }
        }

        if count_item(player_state, "mcl_core:cobble") < 8 {
            self.transition(
                BotPhase::S2MineOresAndStone,
                None,
                "Mining cobble for furnace",
            );
            return;
        }

        let _ = craft_via_registry(
            reg,
            player_state,
            3,
            &[
                Some(("mcl_core:cobble", 1)),
                Some(("mcl_core:cobble", 1)),
                Some(("mcl_core:cobble", 1)),
                Some(("mcl_core:cobble", 1)),
                None,
                Some(("mcl_core:cobble", 1)),
                Some(("mcl_core:cobble", 1)),
                Some(("mcl_core:cobble", 1)),
                Some(("mcl_core:cobble", 1)),
            ],
        );
        if count_item(player_state, "mcl_furnaces:furnace") > 0 {
            self.transition(BotPhase::S2GatherFlint, None, "Gathering flint from gravel");
        }
    }

    fn step_s2_gather_flint(
        &mut self,
        dt: f32,
        player: &mut PlayerController,
        player_state: &Arc<RwLock<ObjectState>>,
        world: &Arc<WorldStore>,
        reg: &EngineRegistries,
        out: &mut AutoBotTickOutput,
    ) {
        let flint = count_item(player_state, "mcl_core:flint");
        if flint >= 1 {
            self.gravel_place_pos = None;
            self.transition(
                BotPhase::S3PrepareAndPlaceFurnace,
                Some(2),
                "Placing Furnace",
            );
            return;
        }

        let gravel_inv = count_item(player_state, "mcl_core:gravel");
        self.detail = format!("Flint {flint}/1 (Gravel in bag: {gravel_inv})");

        // If we already have gravel in inventory, cycle placing & re-mining it at a clean spot until flint drops
        if gravel_inv >= 1 {
            if self.gravel_place_pos.is_some() && self.target_timer > 12.0 {
                if let Some(old) = self.gravel_place_pos.take() {
                    self.blacklisted.insert(old);
                }
                self.target_timer = 0.0;
            }
            if self.gravel_place_pos.is_none() {
                self.gravel_place_pos =
                    find_nearby_placement_spot(world, reg, player.pos, &self.blacklisted);
                self.target_timer = 0.0;
            }
            if let Some(gpos) = self.gravel_place_pos {
                let nid = world.get_node(gpos).0;
                if reg.get_name_from_content_id(nid) == "mcl_core:gravel" {
                    if self.mine_block_or_approach(dt, player, player_state, world, reg, gpos, false, out) {
                        self.pickup_wait = 0.45;
                        self.target_timer = 0.0;
                    }
                } else {
                    let _ = self.place_item_at(
                        dt,
                        player,
                        player_state,
                        world,
                        reg,
                        "mcl_core:gravel",
                        gpos,
                        gpos - IVec3::Y,
                        IVec3::Y,
                        out,
                    );
                    if out.place_just {
                        self.target_timer = 0.0;
                    }
                }
                return;
            }
        }

        // Otherwise locate natural gravel in loaded chunks (prefer dry gravel above deep water first)
        if self.current_target.is_none_or(|p| {
            reg.get_name_from_content_id(world.get_node(p).0) != "mcl_core:gravel"
                || self.target_timer > 14.0
        }) {
            if let Some(old) = self.current_target.take() {
                if self.target_timer > 14.0 {
                    self.blacklisted.insert(old);
                }
            }
            self.current_target = find_nearest_node(
                world,
                reg,
                player.pos,
                6,
                22,
                80,
                &self.blacklisted,
                |n, pos| {
                    n == "mcl_core:gravel" && !is_submerged_around(world, reg, pos)
                },
                false,
            )
            .or_else(|| {
                find_nearest_node(
                    world,
                    reg,
                    player.pos,
                    6,
                    22,
                    80,
                    &self.blacklisted,
                    |n, _| n == "mcl_core:gravel",
                    false,
                )
            });
            self.target_timer = 0.0;
        }

        if let Some(target) = self.current_target {
            if self.mine_block_or_approach(dt, player, player_state, world, reg, target, false, out) {
                self.current_target = None;
                self.pickup_wait = 0.45;
            }
        } else {
            self.explore_outward(dt, player, player_state, world, reg, out);
        }
    }

    // -------------------------------------------------------------------------
    // Stage 3: Smelt & Craft Portal Tools
    // -------------------------------------------------------------------------

    fn step_s3_place_furnace(
        &mut self,
        dt: f32,
        player: &mut PlayerController,
        player_state: &Arc<RwLock<ObjectState>>,
        world: &Arc<WorldStore>,
        reg: &EngineRegistries,
        out: &mut AutoBotTickOutput,
    ) {
        if let Some(fp) = self.furnace_pos {
            let n = reg.get_name_from_content_id(world.get_node(fp).0);
            if n == "mcl_furnaces:furnace" || n == "mcl_furnaces:furnace_active" {
                self.transition(BotPhase::S3SmeltIron, None, "Smelting Iron 0/4");
                return;
            }
        }

        if let Some(ct_pos) = self.crafting_table_pos {
            if player.pos.distance(ct_pos.as_vec3() + 0.5) > 4.2
                || (player.pos.y - ct_pos.y as f32).abs() > 1.5
            {
                self.detail = "Returning to surface to place Furnace".into();
                self.move_or_mine_to_voxel(
                    dt,
                    player,
                    player_state,
                    world,
                    reg,
                    ct_pos + IVec3::new(0, 0, -1),
                    out,
                );
                return;
            }
            if self.current_target.is_none() {
                for off in [IVec3::X, IVec3::NEG_X, IVec3::Z, IVec3::NEG_Z] {
                    let cand = ct_pos + off;
                    if !self.blacklisted.contains(&cand) && is_good_place_spot(world, reg, cand) {
                        self.current_target = Some(cand);
                        break;
                    }
                }
            }
        }
        if self.current_target.is_none() {
            self.current_target =
                find_nearby_placement_spot(world, reg, player.pos, &self.blacklisted);
        }

        if let Some(fpos) = self.current_target {
            let n = reg.get_name_from_content_id(world.get_node(fpos).0);
            if n == "mcl_furnaces:furnace" || n == "mcl_furnaces:furnace_active" {
                self.furnace_pos = Some(fpos);
                self.furnace_loaded = false;
                self.transition(BotPhase::S3SmeltIron, None, "Smelting Iron 0/4");
                return;
            }
            if count_item(player_state, "mcl_furnaces:furnace") == 0 {
                self.transition(
                    BotPhase::S2ReturnCraftFurnace,
                    None,
                    "Crafting Furnace",
                );
                return;
            }
            self.detail = "Placing Furnace".into();
            if self.place_item_at(
                dt,
                player,
                player_state,
                world,
                reg,
                "mcl_furnaces:furnace",
                fpos,
                fpos - IVec3::Y,
                IVec3::Y,
                out,
            ) {
                self.furnace_pos = Some(fpos);
                self.furnace_loaded = false;
            } else if self.target_timer > 7.0 {
                self.blacklisted.insert(fpos);
                self.current_target = None;
                self.target_timer = 0.0;
            }
        } else {
            self.explore_outward(dt, player, player_state, world, reg, out);
        }
    }

    fn step_s3_smelt_iron(
        &mut self,
        dt: f32,
        player: &mut PlayerController,
        player_state: &Arc<RwLock<ObjectState>>,
        world: &Arc<WorldStore>,
        _reg: &EngineRegistries,
        lua_rt: Option<&LuaRuntime>,
        out: &mut AutoBotTickOutput,
    ) {
        let Some(fpos) = self.furnace_pos else {
            self.transition(BotPhase::S3PrepareAndPlaceFurnace, None, "Re-placing Furnace");
            return;
        };

        // Load raw iron & coal into the placed furnace
        if !self.furnace_loaded {
            let need_src = 4u16.saturating_sub(count_item(player_state, "mcl_core:iron_ingot"));
            let take_raw = take_raw_iron_from_inventory(player_state, need_src);
            let take_coal = take_item_from_inventory(player_state, "mcl_core:coal_lump", 2);

            let meta = world.get_node_meta(fpos);
            {
                let mut g = meta.write();
                g.inventories
                    .entry("src".into())
                    .or_insert_with(|| vec![ItemStackData::default()]);
                g.inventories
                    .entry("fuel".into())
                    .or_insert_with(|| vec![ItemStackData::default()]);
                g.inventories
                    .entry("dst".into())
                    .or_insert_with(|| vec![ItemStackData::default()]);
                if !take_raw.is_empty() {
                    g.inventories.get_mut("src").unwrap()[0] = take_raw.clone();
                }
                if !take_coal.is_empty() {
                    g.inventories.get_mut("fuel").unwrap()[0] = take_coal.clone();
                }
            }
            if let Some(rt) = lua_rt {
                if !take_raw.is_empty() {
                    let allowed = rt.allow_metadata_inventory_put(
                        fpos,
                        "src",
                        1,
                        &take_raw,
                        player_state,
                    );
                    if allowed > 0 {
                        rt.on_metadata_inventory_put(fpos, "src", 1, &take_raw, player_state);
                    }
                }
                if !take_coal.is_empty() {
                    let allowed = rt.allow_metadata_inventory_put(
                        fpos,
                        "fuel",
                        1,
                        &take_coal,
                        player_state,
                    );
                    if allowed > 0 {
                        rt.on_metadata_inventory_put(fpos, "fuel", 1, &take_coal, player_state);
                    }
                }
            }
            self.furnace_loaded = true;
            self.furnace_fallback_timer = 0.0;
        }

        // Look at furnace cleanly while smelting
        smooth_aim_at(player, fpos.as_vec3() + 0.5, dt);

        // Fallback advancement only when no LuaRuntime is driving node timers (headless non-Lua test)
        if lua_rt.is_none() {
            let meta = world.get_node_meta(fpos);
            let mut g = meta.write();
            let has_src = g
                .inventories
                .get("src")
                .and_then(|v| v.first())
                .is_some_and(|s| !s.is_empty() && s.count > 0);
            let has_fuel = g
                .inventories
                .get("fuel")
                .and_then(|v| v.first())
                .is_some_and(|s| !s.is_empty() && s.count > 0);
            if has_src && has_fuel {
                self.furnace_fallback_timer += dt;
                if self.furnace_fallback_timer >= 1.5 {
                    self.furnace_fallback_timer = 0.0;
                    if let Some(src) = g.inventories.get_mut("src").and_then(|v| v.first_mut()) {
                        src.count -= 1;
                        if src.count == 0 {
                            *src = ItemStackData::default();
                        }
                    }
                    if let Some(dst) = g.inventories.get_mut("dst").and_then(|v| v.first_mut()) {
                        if dst.is_empty() {
                            *dst = ItemStackData::from_spec("mcl_core:iron_ingot 1");
                        } else {
                            dst.count += 1;
                        }
                    }
                }
            }
        }

        let dst_ingots = {
            let meta = world.get_node_meta(fpos);
            let g = meta.read();
            g.inventories
                .get("dst")
                .and_then(|v| v.first())
                .filter(|s| s.name == "mcl_core:iron_ingot")
                .map(|s| s.count)
                .unwrap_or(0)
        };
        let total_ingots = dst_ingots + count_item(player_state, "mcl_core:iron_ingot");
        if total_ingots > self.last_smelt_total {
            self.last_smelt_total = total_ingots;
            self.target_timer = 0.0;
        }
        self.detail = format!("Smelting Iron {}/4", total_ingots.min(4));

        if total_ingots >= 4 {
            let take_stack = {
                let meta = world.get_node_meta(fpos);
                let mut g = meta.write();
                g.inventories
                    .get_mut("dst")
                    .and_then(|v| v.first_mut())
                    .map(std::mem::take)
                    .unwrap_or_default()
            };
            if !take_stack.is_empty() {
                add_drops_to_inventory(
                    player_state,
                    &[format!("{} {}", take_stack.name, take_stack.count)],
                    _reg,
                );
            }
            out.remesh_positions.push(fpos);
            self.transition(
                BotPhase::S3CraftPortalTools,
                None,
                "Crafting Bucket & Flint and Steel",
            );
        }
    }

    fn step_s3_craft_portal_tools(
        &mut self,
        _dt: f32,
        _player: &mut PlayerController,
        player_state: &Arc<RwLock<ObjectState>>,
        _world: &Arc<WorldStore>,
        reg: &EngineRegistries,
        _out: &mut AutoBotTickOutput,
    ) {
        let has_any_bucket = count_item(player_state, "mcl_buckets:bucket_empty") > 0
            || count_item(player_state, "mcl_buckets:bucket_water") > 0
            || count_item(player_state, "mcl_buckets:bucket_lava") > 0;

        if !has_any_bucket {
            let _ = craft_via_registry(
                reg,
                player_state,
                3,
                &[
                    Some(("mcl_core:iron_ingot", 1)),
                    None,
                    Some(("mcl_core:iron_ingot", 1)),
                    None,
                    Some(("mcl_core:iron_ingot", 1)),
                    None,
                    None,
                    None,
                    None,
                ],
            );
        }

        if count_item(player_state, "mcl_fire:flint_and_steel") == 0 {
            let _ = craft_via_registry(
                reg,
                player_state,
                2,
                &[
                    Some(("mcl_core:iron_ingot", 1)),
                    Some(("mcl_core:flint", 1)),
                    None,
                    None,
                ],
            );
        }

        if (count_item(player_state, "mcl_buckets:bucket_empty") > 0
            || count_item(player_state, "mcl_buckets:bucket_water") > 0)
            && count_item(player_state, "mcl_fire:flint_and_steel") > 0
        {
            self.transition(
                BotPhase::S4FindWaterAndFillBucket,
                Some(3),
                "Collecting water bucket",
            );
        } else {
            self.transition(BotPhase::S3SmeltIron, None, "Need more iron ingots");
        }
    }

    // -------------------------------------------------------------------------
    // Stage 4: Build / Cast Obsidian Portal
    // -------------------------------------------------------------------------

    fn step_s4_find_water(
        &mut self,
        dt: f32,
        player: &mut PlayerController,
        player_state: &Arc<RwLock<ObjectState>>,
        world: &Arc<WorldStore>,
        reg: &EngineRegistries,
        out: &mut AutoBotTickOutput,
    ) {
        if count_item(player_state, "mcl_buckets:bucket_water") > 0 {
            self.transition(BotPhase::S4FindLavaPool, None, "Locating lava pool");
            return;
        }
        self.detail = "Finding water source".into();

        if self.current_target.is_none_or(|p| {
            !is_water_node(world.get_node(p).0, reg)
                || (self.target_timer > 14.0
                    && (player.pos.distance(p.as_vec3() + 0.5) < 5.5 || self.target_timer > 36.0))
        }) {
            if let Some(old) = self.current_target.take() {
                if self.target_timer > 14.0 {
                    self.blacklisted.insert(old);
                }
            }
            self.current_target = find_nearest_surface_liquid(
                world,
                reg,
                player.pos,
                8,
                SEA_LEVEL - 4,
                86,
                &self.blacklisted,
                |n, _| is_water_source_item(n),
                true,
            );
            self.target_timer = 0.0;
        }

        if let Some(wpos) = self.current_target {
            if self.scoop_or_pour_bucket(
                dt,
                player,
                player_state,
                world,
                reg,
                "mcl_buckets:bucket_empty",
                wpos,
                true,
                2.0,
                true,
                out,
            ) {
                self.current_target = None;
            }
        } else {
            self.detail = "Exploring for water".into();
            self.explore_outward(dt, player, player_state, world, reg, out);
        }
    }

    fn step_s4_find_lava_pool(
        &mut self,
        dt: f32,
        player: &mut PlayerController,
        player_state: &Arc<RwLock<ObjectState>>,
        world: &Arc<WorldStore>,
        reg: &EngineRegistries,
        out: &mut AutoBotTickOutput,
    ) {
        if self.lava_pool_center.is_none() || self.target_timer > 20.0 {
            self.lava_pool_center = find_surface_lava_pool(world, reg, player.pos, &self.blacklisted);
            if self.lava_pool_center.is_some() {
                self.target_timer = 0.0;
            }
        }

        if let Some(lava_c) = self.lava_pool_center {
            self.detail = format!("Moving to lava pool ({}, {}, {})", lava_c.x, lava_c.y, lava_c.z);
            let dist = Vec3::new(
                lava_c.x as f32 + 0.5 - player.pos.x,
                0.0,
                lava_c.z as f32 + 0.5 - player.pos.z,
            )
            .length();
            if dist <= 6.5 {
                self.transition(
                    BotPhase::S4PreparePortalSite,
                    None,
                    "Preparing portal site",
                );
                return;
            }
            if self.no_move_timer > 2.5 {
                self.move_or_mine_to_voxel(
                    dt,
                    player,
                    player_state,
                    world,
                    reg,
                    lava_c + IVec3::new(0, 1, -5),
                    out,
                );
            } else {
                let walk_t = lava_c.as_vec3() + Vec3::new(0.5, 0.0, -4.5);
                self.steer_surface(player, player_state, walk_t, world, reg, dt, out, true);
            }
        } else {
            self.detail = "Exploring for surface lava pool".into();
            self.explore_outward(dt, player, player_state, world, reg, out);
        }
    }

    fn step_s4_prepare_site(
        &mut self,
        dt: f32,
        player: &mut PlayerController,
        player_state: &Arc<RwLock<ObjectState>>,
        world: &Arc<WorldStore>,
        reg: &EngineRegistries,
        out: &mut AutoBotTickOutput,
    ) {
        let Some(lava_c) = self.lava_pool_center else {
            self.transition(BotPhase::S4FindLavaPool, None, "Re-acquiring lava pool");
            return;
        };

        // Pick a clean flat portal origin just beside the lava pool
        if self.portal_origin.is_none() {
            if let Some((origin, res)) = find_portal_site_near_lava(world, reg, lava_c) {
                self.portal_origin = Some(origin);
                self.reservoir_pos = Some(res);
                self.scaffold_cells = build_cast_staircase_specs(origin);
                self.frame_cells = build_4x5_frame_specs(origin);
                self.cast_index = 0;
                self.cast_step = CastStep::ScoopLava;
            } else {
                self.blacklisted.insert(lava_c);
                self.lava_pool_center = None;
                self.transition(BotPhase::S4FindLavaPool, None, "Finding better lava pool");
                return;
            }
        }

        let origin = self.portal_origin.unwrap();
        let res = self.reservoir_pos.unwrap();
        let stand_y = origin.y;

        if (player.pos.y - stand_y as f32).abs() > 1.8 {
            self.detail = "Climbing to portal site".into();
            self.move_or_mine_to_voxel(
                dt,
                player,
                player_state,
                world,
                reg,
                IVec3::new(origin.x, stand_y, origin.z - 2),
                out,
            );
            return;
        }

        // 1. Clear non-air obstructions in the portal + front casting workspace (`origin.y..=origin.y + 4`)
        let mut clear_target: Option<IVec3> = None;
        'clear_scan: for y in origin.y..=(origin.y + 4) {
            for z in (origin.z - 3)..=origin.z {
                for x in (origin.x - 3)..=(origin.x + 2) {
                    let p = IVec3::new(x, y, z);
                    let id = world.get_node(p).0;
                    if id != CONTENT_AIR && id != CONTENT_IGNORE && !is_obsidian(id, reg) {
                        let name = reg.get_name_from_content_id(id);
                        if !is_liquid_node_id(id, reg) && name != "mcl_core:bedrock" {
                            clear_target = Some(p);
                            break 'clear_scan;
                        }
                    }
                }
            }
        }
        if let Some(ct) = clear_target {
            self.detail = "Clearing portal workspace".into();
            if self.mine_block_or_approach(dt, player, player_state, world, reg, ct, true, out) {
                self.pickup_wait = 0.15;
            }
            return;
        }

        // 2. Ensure reservoir pit (`res` at `origin.y - 1`) is hollowed cleanly & top cell is clear
        let res_id = world.get_node(res).0;
        if res_id != CONTENT_AIR && !is_water_node(res_id, reg) {
            self.detail = "Digging casting reservoir".into();
            if self.mine_block_or_approach(dt, player, player_state, world, reg, res, true, out) {
                self.pickup_wait = 0.15;
            }
            return;
        }
        let res_top = res + IVec3::Y;
        if world.get_node(res_top).0 != CONTENT_AIR && !is_liquid_node_id(world.get_node(res_top).0, reg) {
            let _ = self.mine_block_or_approach(dt, player, player_state, world, reg, res_top, true, out);
            return;
        }

        // 3. Ensure solid floor at `origin.y - 1` under the frame and front casting walkway (and solid reservoir bottom)
        let find_fill_support = |target: IVec3| -> (IVec3, IVec3, IVec3) {
            let below = target - IVec3::Y;
            if is_solid_walkable(world.get_node(below).0, reg) {
                return (target, below, IVec3::Y);
            }
            for dir in [IVec3::X, -IVec3::X, IVec3::Z, -IVec3::Z] {
                let nbr = target + dir;
                if is_solid_walkable(world.get_node(nbr).0, reg) {
                    return (target, nbr, -dir);
                }
            }
            let mut cur = target;
            while cur.y > 40 && !is_solid_walkable(world.get_node(cur - IVec3::Y).0, reg) {
                cur -= IVec3::Y;
            }
            (cur, cur - IVec3::Y, IVec3::Y)
        };

        let mut patch_target: Option<(IVec3, IVec3, IVec3)> = None;
        let res_bottom = res - IVec3::Y;
        if !is_solid_walkable(world.get_node(res_bottom).0, reg) {
            patch_target = Some(find_fill_support(res_bottom));
        }
        if patch_target.is_none() {
            // Prefer gaps touching existing solid neighbors first so deep pits bridge cleanly horizontally
            let mut fallback_gap: Option<IVec3> = None;
            'floor_scan: for z in (origin.z - 3)..=origin.z {
                for x in (origin.x - 3)..=(origin.x + 2) {
                    let fp = IVec3::new(x, origin.y - 1, z);
                    if fp == res {
                        continue;
                    }
                    let id = world.get_node(fp).0;
                    if id == CONTENT_AIR || is_liquid_node_id(id, reg) {
                        let candidate = find_fill_support(fp);
                        if candidate.0 == fp {
                            patch_target = Some(candidate);
                            break 'floor_scan;
                        }
                        if fallback_gap.is_none() {
                            fallback_gap = Some(fp);
                        }
                    }
                }
            }
            if patch_target.is_none() {
                if let Some(fp) = fallback_gap {
                    patch_target = Some(find_fill_support(fp));
                }
            }
        }
        if let Some((pp, sup, normal)) = patch_target {
            if let Some(fill) = find_placeable_filler_block(player_state) {
                self.detail = "Leveling portal floor".into();
                let _ = self.place_item_at(
                    dt,
                    player,
                    player_state,
                    world,
                    reg,
                    &fill,
                    pp,
                    sup,
                    normal,
                    out,
                );
                return;
            }
        }

        // 4. Dump water into reservoir so bucket is empty & ready for ScoopLava
        if count_item(player_state, "mcl_buckets:bucket_water") > 0 && !is_water_node(world.get_node(res).0, reg) {
            self.detail = "Filling local water reservoir".into();
            let res_stand = Vec3::new(res.x as f32 + 0.5, origin.y as f32, res.z as f32 - 0.45);
            let horiz = Vec3::new(res_stand.x - player.pos.x, 0.0, res_stand.z - player.pos.z).length();
            if horiz > 0.35 || (player.pos.y - origin.y as f32).abs() > 0.9 {
                self.steer_surface(player, player_state, res_stand, world, reg, dt, out, true);
                return;
            }
            if equip_item_exact(player, player_state, "mcl_buckets:bucket_water") {
                out.force_liquids_pointable = false;
                let aim_pt = (res - IVec3::Y).as_vec3() + Vec3::new(0.5, 0.96, 0.5);
                if smooth_aim_at(player, aim_pt, dt) {
                    if let Some(hit) = raycast_voxel_ex(player.eye_pos(), player.look_dir(), 5.0, false, world, reg) {
                        if hit.under == res - IVec3::Y && hit.above == res && self.place_cooldown <= 0.0 {
                            out.place_just = true;
                            self.place_cooldown = 0.25;
                        }
                    }
                }
            }
            return;
        }

        // Walk safely to portal front walkway (`stand_y`)
        let stand = Vec3::new(origin.x as f32 + 0.5, stand_y as f32, (origin.z - 2) as f32 + 0.5);
        if player.pos.distance(stand) > 2.5 {
            self.steer_surface(player, player_state, stand, world, reg, dt, out, true);
            return;
        }

        self.transition(
            BotPhase::S4CastObsidianFrame,
            None,
            "Casting obsidian 0/14",
        );
    }

    fn step_s4_cast_frame(
        &mut self,
        dt: f32,
        player: &mut PlayerController,
        player_state: &Arc<RwLock<ObjectState>>,
        world: &Arc<WorldStore>,
        reg: &EngineRegistries,
        out: &mut AutoBotTickOutput,
    ) {
        let Some(origin) = self.portal_origin else {
            self.transition(BotPhase::S4PreparePortalSite, None, "Re-selecting site");
            return;
        };
        let Some(res) = self.reservoir_pos else {
            self.transition(BotPhase::S4PreparePortalSite, None, "Re-selecting reservoir");
            return;
        };
        let stand_y = origin.y as f32;

        let has_empty = count_item(player_state, "mcl_buckets:bucket_empty") > 0;
        let has_lava = count_item(player_state, "mcl_buckets:bucket_lava") > 0;
        let has_water = count_item(player_state, "mcl_buckets:bucket_water") > 0;

        // 1. Recover cooling water first BEFORE advancing cast_index
        if let Some(cw) = self.cool_water_pos {
            if !is_water_node(world.get_node(cw).0, reg) {
                self.cool_water_pos = None;
            } else if has_empty {
                let cw_stand_x = if cw.x < origin.x {
                    origin.x as f32 + 0.35
                } else if cw.x > origin.x + 1 {
                    origin.x as f32 + 1.65
                } else {
                    cw.x as f32 + 0.50
                };
                if cw.y >= origin.y + 2 {
                    if !self.walk_to_cast_platform(
                        origin,
                        cw_stand_x,
                        dt,
                        player,
                        player_state,
                        world,
                        reg,
                        out,
                    ) {
                        return;
                    }
                } else {
                    let stand = Vec3::new(cw.x as f32 + 0.5, stand_y, (origin.z - 2) as f32 + 0.45);
                    let horiz_stand = Vec3::new(stand.x - player.pos.x, 0.0, stand.z - player.pos.z).length();
                    if horiz_stand > 0.65 || (player.pos.y - stand_y).abs() > 0.9 {
                        self.steer_surface(player, player_state, stand, world, reg, dt, out, true);
                        return;
                    }
                }
                if equip_item_exact(player, player_state, "mcl_buckets:bucket_empty") {
                    out.force_liquids_pointable = true;
                    let aim_pt = cw.as_vec3() + Vec3::splat(0.5);
                    if smooth_aim_at(player, aim_pt, dt) {
                        if let Some(hit) =
                            raycast_voxel_ex(player.eye_pos(), player.look_dir(), 5.0, true, world, reg)
                        {
                            if hit.under == cw && self.place_cooldown <= 0.0 {
                                out.place_just = true;
                                self.place_cooldown = 0.25;
                            }
                        }
                    }
                }
                return;
            }
        }

        // 2. Refill reservoir BEFORE advancing cast_index (when holding water and active cell doesn't still need cooling)
        let current_cell_is_lava = self.cast_index < self.frame_cells.len()
            && is_lava_node(world.get_node(self.frame_cells[self.cast_index].pos).0, reg);
        if has_water && !is_water_node(world.get_node(res).0, reg) && !current_cell_is_lava {
            if self.descend_cast_staircase(origin, dt, player, player_state, world, reg, out) {
                return;
            }
            let res_stand = Vec3::new(res.x as f32 + 0.5, stand_y, res.z as f32 - 0.45);
            let horiz = Vec3::new(res_stand.x - player.pos.x, 0.0, res_stand.z - player.pos.z).length();
            if horiz > 0.35 || (player.pos.y - stand_y).abs() > 0.9 {
                self.steer_surface(player, player_state, res_stand, world, reg, dt, out, true);
                return;
            }
            if equip_item_exact(player, player_state, "mcl_buckets:bucket_water") {
                out.force_liquids_pointable = false;
                let aim_pt = (res - IVec3::Y).as_vec3() + Vec3::new(0.5, 0.96, 0.5);
                if smooth_aim_at(player, aim_pt, dt) {
                    if let Some(hit) =
                        raycast_voxel_ex(player.eye_pos(), player.look_dir(), 5.0, false, world, reg)
                    {
                        if hit.under == res - IVec3::Y
                            && hit.above == res
                            && self.place_cooldown <= 0.0
                        {
                            out.place_just = true;
                            self.place_cooldown = 0.25;
                        }
                    }
                }
            }
            return;
        }

        // 3. Advance safely past already-obsidian cells
        while self.cast_index < self.frame_cells.len() {
            let cell = self.frame_cells[self.cast_index];
            if is_obsidian(world.get_node(cell.pos).0, reg) {
                self.cast_index += 1;
                self.cast_step = CastStep::ScoopLava;
            } else {
                break;
            }
        }

        if self.cast_index >= self.frame_cells.len() {
            let interior_seed = IVec3::new(origin.x, origin.y + 1, origin.z);
            if let Some(frame) = detect_portal_frame(world, reg, interior_seed) {
                self.detected_frame = Some(frame);
                self.transition(BotPhase::S5IgnitePortal, Some(4), "Igniting Nether Portal");
                return;
            } else {
                self.cast_index = 0;
                return;
            }
        }

        let done_count = self
            .frame_cells
            .iter()
            .filter(|c| is_obsidian(world.get_node(c.pos).0, reg))
            .count();
        self.detail = format!("Casting obsidian {done_count}/{}", self.frame_cells.len());

        let cell = self.frame_cells[self.cast_index];

        // Recover step defensively from actual inventory / cell state
        if has_lava && self.cast_step != CastStep::PourLava {
            self.cast_step = CastStep::PourLava;
        } else if is_lava_node(world.get_node(cell.pos).0, reg) {
            self.cast_step = if has_water {
                CastStep::PourWaterCool
            } else {
                CastStep::ScoopWater
            };
        } else if !has_lava && self.cast_step == CastStep::PourLava {
            self.cast_step = CastStep::ScoopLava;
        }

        // Build casting staircase once rows reach `origin.y + 2`
        if cell.pos.y >= origin.y + 2
            && matches!(self.cast_step, CastStep::PourLava | CastStep::PourWaterCool)
            && !self.ensure_cast_staircase_built(origin, dt, player, player_state, world, reg, out)
        {
            self.detail = "Building portal scaffold".into();
            return;
        }
        // Build staircase before climbing up on first high cell even while ScoopLava
        if cell.pos.y >= origin.y + 2
            && self
                .scaffold_cells
                .iter()
                .any(|&(pos, _)| !is_solid_walkable(world.get_node(pos).0, reg))
        {
            if !self.ensure_cast_staircase_built(origin, dt, player, player_state, world, reg, out) {
                self.detail = "Building portal scaffold".into();
                return;
            }
        }

        let high_row = cell.pos.y >= origin.y + 2;
        let stand_x = if cell.pos.x < origin.x {
            origin.x as f32 + 0.35
        } else if cell.pos.x > origin.x + 1 {
            origin.x as f32 + 1.65
        } else {
            cell.pos.x as f32 + 0.50
        };

        match self.cast_step {
            CastStep::ScoopLava => {
                if !has_empty {
                    return;
                }
                if self.descend_cast_staircase(origin, dt, player, player_state, world, reg, out) {
                    return;
                }
                if self.current_target.is_none_or(|p| {
                    !is_lava_node(world.get_node(p).0, reg)
                        || self.is_frame_pos(p)
                        || self.target_timer > 8.0
                }) {
                    if let Some(old) = self.current_target.take() {
                        if self.target_timer > 8.0 {
                            self.blacklisted.insert(old);
                        }
                    }
                    self.current_target = find_nearest_surface_liquid(
                        world,
                        reg,
                        self.lava_pool_center.unwrap_or(origin).as_vec3(),
                        4,
                        SEA_LEVEL - 4,
                        90,
                        &self.blacklisted,
                        |n, pos| is_lava_source_item(n) && !self.is_frame_pos(pos),
                        true,
                    );
                    self.target_timer = 0.0;
                }
                if let Some(lpos) = self.current_target {
                    if self.scoop_or_pour_bucket(
                        dt,
                        player,
                        player_state,
                        world,
                        reg,
                        "mcl_buckets:bucket_empty",
                        lpos,
                        true,
                        1.65,
                        true,
                        out,
                    ) {
                        self.current_target = None;
                        if count_item(player_state, "mcl_buckets:bucket_lava") > 0 {
                            self.cast_step = CastStep::PourLava;
                        }
                    }
                } else {
                    self.blacklisted.extend(self.lava_pool_center);
                    self.lava_pool_center = None;
                    self.portal_origin = None;
                    self.transition(BotPhase::S4FindLavaPool, None, "Need more lava sources");
                }
            }
            CastStep::PourLava => {
                if !has_lava {
                    self.cast_step = CastStep::ScoopLava;
                    return;
                }
                if is_lava_node(world.get_node(cell.pos).0, reg) {
                    self.cast_step = CastStep::ScoopWater;
                    return;
                }
                if high_row {
                    if !self.walk_to_cast_platform(
                        origin,
                        stand_x,
                        dt,
                        player,
                        player_state,
                        world,
                        reg,
                        out,
                    ) {
                        return;
                    }
                } else {
                    let stand = Vec3::new(
                        cell.pos.x as f32 + 0.5,
                        stand_y,
                        (origin.z - 2) as f32 + 0.45,
                    );
                    let horiz_stand = Vec3::new(stand.x - player.pos.x, 0.0, stand.z - player.pos.z).length();
                    if horiz_stand > 0.65 || (player.pos.y - stand_y).abs() > 0.9 {
                        self.steer_surface(player, player_state, stand, world, reg, dt, out, true);
                        return;
                    }
                }
                let aim_pt = cell.support.as_vec3()
                    + Vec3::splat(0.5)
                    + cell.face_normal.as_vec3() * 0.49;
                equip_item_exact(player, player_state, "mcl_buckets:bucket_lava");
                out.force_liquids_pointable = false;
                if smooth_aim_at(player, aim_pt, dt) {
                    if let Some(hit) =
                        raycast_voxel_ex(player.eye_pos(), player.look_dir(), 5.0, false, world, reg)
                    {
                        if hit.under == cell.support && hit.above == cell.pos {
                            if self.place_cooldown <= 0.0 {
                                out.place_just = true;
                                self.place_cooldown = 0.24;
                            }
                        } else if hit.distance_sq <= 4.5 * 4.5
                            && !is_protected_node(hit.node_id, reg)
                            && !self.is_frame_pos(hit.under)
                            && !self.is_scaffold_pos(hit.under)
                        {
                            equip_best_tool_for_block(
                                player,
                                player_state,
                                reg.get_name_from_content_id(hit.node_id),
                            );
                            out.mine_held = true;
                        }
                    }
                }
            }
            CastStep::ScoopWater => {
                if !is_lava_node(world.get_node(cell.pos).0, reg)
                    && !is_obsidian(world.get_node(cell.pos).0, reg)
                {
                    self.cast_step = CastStep::ScoopLava;
                    return;
                }
                if self.descend_cast_staircase(origin, dt, player, player_state, world, reg, out) {
                    return;
                }
                if is_water_node(world.get_node(res).0, reg) {
                    let res_stand = Vec3::new(res.x as f32 + 0.5, stand_y, res.z as f32 - 0.45);
                    let horiz = Vec3::new(res_stand.x - player.pos.x, 0.0, res_stand.z - player.pos.z).length();
                    if horiz > 0.35 || (player.pos.y - stand_y).abs() > 0.9 {
                        self.steer_surface(player, player_state, res_stand, world, reg, dt, out, true);
                        return;
                    }
                    if equip_item_exact(player, player_state, "mcl_buckets:bucket_empty") {
                        out.force_liquids_pointable = true;
                        let aim_pt = res.as_vec3() + Vec3::new(0.5, 0.94, 0.5);
                        if smooth_aim_at(player, aim_pt, dt) {
                            if let Some(hit) =
                                raycast_voxel_ex(player.eye_pos(), player.look_dir(), 5.0, true, world, reg)
                            {
                                if hit.under == res && self.place_cooldown <= 0.0 {
                                    out.place_just = true;
                                    self.place_cooldown = 0.25;
                                    self.cast_step = CastStep::PourWaterCool;
                                }
                            }
                        }
                    }
                    return;
                }
                if let Some(wpos) = find_nearest_surface_liquid(
                    world,
                    reg,
                    player.pos,
                    6,
                    SEA_LEVEL - 4,
                    86,
                    &self.blacklisted,
                    |n, _| is_water_source_item(n),
                    true,
                ) {
                    let _ = self.scoop_or_pour_bucket(
                        dt,
                        player,
                        player_state,
                        world,
                        reg,
                        "mcl_buckets:bucket_empty",
                        wpos,
                        true,
                        2.1,
                        true,
                        out,
                    );
                }
            }
            CastStep::PourWaterCool => {
                if !has_water {
                    self.cast_step = CastStep::ScoopWater;
                    return;
                }
                if is_obsidian(world.get_node(cell.pos).0, reg) {
                    self.cast_step = CastStep::RefillReservoir;
                    return;
                }
                let cool_pos = cell.pos - IVec3::Z;
                if high_row {
                    if !self.walk_to_cast_platform(
                        origin,
                        stand_x,
                        dt,
                        player,
                        player_state,
                        world,
                        reg,
                        out,
                    ) {
                        return;
                    }
                } else {
                    let stand = Vec3::new(
                        cell.pos.x as f32 + 0.5,
                        stand_y,
                        (origin.z - 2) as f32 + 0.25,
                    );
                    let horiz_stand = Vec3::new(stand.x - player.pos.x, 0.0, stand.z - player.pos.z).length();
                    if horiz_stand > 0.65 || (player.pos.y - stand_y).abs() > 0.9 {
                        self.steer_surface(player, player_state, stand, world, reg, dt, out, true);
                        return;
                    }
                }
                if is_water_node(world.get_node(cool_pos).0, reg) {
                    self.cool_water_pos = Some(cool_pos);
                    self.cast_step = CastStep::RescoopWater;
                    return;
                }
                let aim_pt = cell.pos.as_vec3() + Vec3::new(0.5, 0.5, 0.04);
                equip_item_exact(player, player_state, "mcl_buckets:bucket_water");
                out.force_liquids_pointable = true;
                if smooth_aim_at(player, aim_pt, dt) {
                    if let Some(hit) =
                        raycast_voxel_ex(player.eye_pos(), player.look_dir(), 5.0, true, world, reg)
                    {
                        if hit.under == cell.pos && self.place_cooldown <= 0.0 {
                            self.cool_water_pos = Some(hit.above);
                            out.place_just = true;
                            self.place_cooldown = 0.24;
                            self.cast_step = CastStep::RescoopWater;
                        }
                    }
                }
            }
            CastStep::RescoopWater | CastStep::RefillReservoir => {
                self.cast_step = CastStep::ScoopLava;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Stage 5: Ignite & Enter Nether
    // -------------------------------------------------------------------------

    fn step_s5_ignite(
        &mut self,
        dt: f32,
        player: &mut PlayerController,
        player_state: &Arc<RwLock<ObjectState>>,
        world: &Arc<WorldStore>,
        reg: &EngineRegistries,
        out: &mut AutoBotTickOutput,
    ) {
        let origin = self.portal_origin.unwrap_or(IVec3::ZERO);
        let interior = IVec3::new(origin.x, origin.y + 1, origin.z);
        if is_nether_portal_node(world.get_node(interior).0, reg) {
            self.transition(BotPhase::S5EnterPortal, None, "Entering Nether Portal");
            return;
        }

        // Clear any stray non-portal non-air block inside 2x3 interior before igniting
        if let Some(frame) = detect_portal_frame(world, reg, interior) {
            self.detected_frame = Some(frame);
        } else if let Some(ref cells) = Some(&self.frame_cells) {
            for dy in 1..=3 {
                for dw in 0..=1 {
                    let ip = IVec3::new(origin.x + dw, origin.y + dy, origin.z);
                    let id = world.get_node(ip).0;
                    if id != CONTENT_AIR && !is_nether_portal_node(id, reg) {
                        if self.mine_block_or_approach(dt, player, player_state, world, reg, ip, true, out) {
                            self.pickup_wait = 0.15;
                        }
                        return;
                    }
                }
            }
            if !cells.iter().all(|c| is_obsidian(world.get_node(c.pos).0, reg)) {
                self.transition(BotPhase::S4CastObsidianFrame, None, "Finishing portal frame");
                return;
            }
        }

        if self.descend_cast_staircase(origin, dt, player, player_state, world, reg, out) {
            return;
        }
        let corridor_z = origin.z as f32 - 0.5;
        let stand = Vec3::new(origin.x as f32 + 0.5, origin.y as f32, corridor_z);
        if (player.pos.z - corridor_z).abs() > 0.55 && player.pos.x > origin.x as f32 - 2.1 {
            let entry = Vec3::new(origin.x as f32 - 2.5, origin.y as f32, corridor_z);
            self.steer_surface(player, player_state, entry, world, reg, dt, out, true);
            return;
        }
        if player.pos.distance(stand) > 0.65 {
            self.steer_surface(player, player_state, stand, world, reg, dt, out, true);
            return;
        }

        equip_item_exact(player, player_state, "mcl_fire:flint_and_steel");
        out.force_liquids_pointable = false;
        let floor_obs = IVec3::new(origin.x, origin.y, origin.z);
        let aim_pt = floor_obs.as_vec3() + Vec3::new(0.5, 0.96, 0.5);
        self.detail = "Igniting portal frame".into();
        if smooth_aim_at(player, aim_pt, dt) {
            if let Some(hit) = raycast_voxel_ex(player.eye_pos(), player.look_dir(), 5.0, false, world, reg) {
                if hit.under == floor_obs && self.place_cooldown <= 0.0 {
                    out.place_just = true;
                    self.place_cooldown = 0.25;
                }
            }
        }
    }

    fn step_s5_enter_portal(
        &mut self,
        dt: f32,
        player: &mut PlayerController,
        player_state: &Arc<RwLock<ObjectState>>,
        world: &Arc<WorldStore>,
        reg: &EngineRegistries,
        out: &mut AutoBotTickOutput,
    ) {
        if player.dimension == DimensionId::Nether {
            self.nether_entry_pos = Some(player.pos);
            self.nether_walk_dir = None;
            self.transition(
                BotPhase::S5WalkNetherGround,
                None,
                "Walking 5 blocks onto Nether ground",
            );
            return;
        }

        let origin = self.portal_origin.unwrap_or(IVec3::ZERO);
        let portal_center = Vec3::new(
            origin.x as f32 + 1.0,
            (origin.y + 1) as f32,
            origin.z as f32 + 0.5,
        );
        self.detail = format!("Teleporting {:.0}%", (player.portal_charge * 100.0).clamp(0.0, 100.0));

        let d_xz = Vec3::new(portal_center.x - player.pos.x, 0.0, portal_center.z - player.pos.z).length();
        if d_xz > 0.38 || player.pos.y < (origin.y + 1) as f32 - 0.15 {
            self.steer_surface(player, player_state, portal_center, world, reg, dt, out, false);
        } else {
            // Look straight into portal shimmer cleanly
            smooth_aim_at(player, portal_center + Vec3::new(0.0, 1.3, 1.5), dt);
        }
    }

    fn step_s5_walk_nether(
        &mut self,
        dt: f32,
        player: &mut PlayerController,
        player_state: &Arc<RwLock<ObjectState>>,
        world: &Arc<WorldStore>,
        reg: &EngineRegistries,
        out: &mut AutoBotTickOutput,
    ) {
        if player.dimension != DimensionId::Nether {
            self.transition(BotPhase::S5EnterPortal, None, "Re-entering Nether Portal");
            return;
        }
        let entry = *self.nether_entry_pos.get_or_insert(player.pos);
        let walked = Vec3::new(player.pos.x - entry.x, 0.0, player.pos.z - entry.z).length();
        self.detail = format!("Nether ground {walked:.1}/5.0m");

        if walked >= 5.0 && player.on_ground && !is_dangerous_floor_below(player.pos, world, reg) {
            self.finish_hold_timer = 0.0;
            self.transition(BotPhase::S5FinishHold, Some(5), "Arrived in Nether");
            return;
        }

        if self.nether_walk_dir.is_none() {
            self.nether_walk_dir = Some(pick_best_nether_exit_dir(player.pos, world, reg));
        }
        if self.no_move_timer > 5.0 {
            let cur_d = self.nether_walk_dir.unwrap_or(IVec3::Z);
            self.nether_walk_dir = Some(if cur_d == IVec3::Z {
                IVec3::X
            } else if cur_d == IVec3::X {
                IVec3::NEG_Z
            } else if cur_d == IVec3::NEG_Z {
                IVec3::NEG_X
            } else {
                IVec3::Z
            });
            self.no_move_timer = 0.0;
        }
        let dir = self.nether_walk_dir.unwrap();
        let goal = IVec3::new(
            (entry.x + dir.x as f32 * 6.5).floor() as i32,
            player.pos.y.floor() as i32,
            (entry.z + dir.z as f32 * 6.5).floor() as i32,
        );
        self.move_or_mine_to_voxel(dt, player, player_state, world, reg, goal, out);
    }

    // -------------------------------------------------------------------------
    // Shared Movement, Raycast Mining, Placement, Bucket & Anti-Stuck Helpers
    // -------------------------------------------------------------------------

    #[inline]
    fn is_frame_pos(&self, p: IVec3) -> bool {
        self.frame_cells.iter().any(|c| c.pos == p)
    }

    #[inline]
    fn is_scaffold_pos(&self, p: IVec3) -> bool {
        self.scaffold_cells.iter().any(|&(pos, _)| pos == p)
    }

    #[inline]
    fn is_active_scaffold_protected(&self, p: IVec3) -> bool {
        matches!(self.phase, BotPhase::S4CastObsidianFrame) && self.is_scaffold_pos(p)
    }

    fn descend_cast_staircase(
        &mut self,
        origin: IVec3,
        dt: f32,
        player: &mut PlayerController,
        _player_state: &Arc<RwLock<ObjectState>>,
        _world: &Arc<WorldStore>,
        _reg: &EngineRegistries,
        out: &mut AutoBotTickOutput,
    ) -> bool {
        if player.pos.y <= origin.y as f32 + 0.75 {
            return false;
        }
        let exit_pt = Vec3::new(
            origin.x as f32 - 2.5,
            origin.y as f32,
            origin.z as f32 - 1.5,
        );
        smooth_aim_at(
            player,
            Vec3::new(exit_pt.x, player.eye_pos().y - 0.1, exit_pt.z),
            dt,
        );
        let d = Vec3::new(exit_pt.x - player.pos.x, 0.0, exit_pt.z - player.pos.z);
        let desired_yaw = (-d.x).atan2(d.z);
        let mut yaw_diff = desired_yaw - player.yaw;
        while yaw_diff > PI {
            yaw_diff -= TAU;
        }
        while yaw_diff < -PI {
            yaw_diff += TAU;
        }
        if yaw_diff.abs() < 0.22 {
            out.move_input.forward = true;
        }
        true
    }

    #[allow(clippy::too_many_arguments)]
    fn walk_to_cast_platform(
        &mut self,
        origin: IVec3,
        target_x: f32,
        dt: f32,
        player: &mut PlayerController,
        player_state: &Arc<RwLock<ObjectState>>,
        world: &Arc<WorldStore>,
        reg: &EngineRegistries,
        out: &mut AutoBotTickOutput,
    ) -> bool {
        let rail_z = origin.z as f32 - 1.5;
        if player.pos.y >= origin.y as f32 + 2.75
            && (player.pos.x - target_x).abs() <= 0.25
            && (player.pos.z - rail_z).abs() <= 0.30
        {
            return true;
        }
        if player.pos.y < origin.y as f32 + 0.5
            && (player.pos.x > origin.x as f32 - 2.1 || (player.pos.z - rail_z).abs() > 0.45)
        {
            let target = if player.pos.x > origin.x as f32 - 2.1 && (player.pos.z - rail_z).abs() > 0.45 {
                Vec3::new(origin.x as f32 - 2.5, origin.y as f32, player.pos.z)
            } else {
                Vec3::new(origin.x as f32 - 2.5, origin.y as f32, rail_z)
            };
            self.steer_surface(player, player_state, target, world, reg, dt, out, true);
            return false;
        }

        let walk_t = Vec3::new(target_x, player.eye_pos().y - 0.1, rail_z);
        smooth_aim_at(player, walk_t, dt);
        let dx = target_x - player.pos.x;
        let dz = rail_z - player.pos.z;
        let desired_yaw = (-dx).atan2(dz);
        let mut yaw_diff = desired_yaw - player.yaw;
        while yaw_diff > PI {
            yaw_diff -= TAU;
        }
        while yaw_diff < -PI {
            yaw_diff += TAU;
        }
        if yaw_diff.abs() < 0.18 {
            out.move_input.forward = true;
        }
        if player.pos.y < origin.y as f32 + 2.75 {
            let hlen = (dx * dx + dz * dz).sqrt().max(1e-4);
            let ax = (player.pos.x + (dx / hlen) * 0.75).floor() as i32;
            let az = rail_z.floor() as i32;
            let fy = player.pos.y.floor() as i32;
            if is_solid_walkable(world.get_node(IVec3::new(ax, fy, az)).0, reg) {
                out.move_input.jump = true;
            }
        } else {
            out.move_input.sneak = true;
        }
        false
    }

    #[allow(clippy::too_many_arguments)]
    fn ensure_cast_staircase_built(
        &mut self,
        origin: IVec3,
        dt: f32,
        player: &mut PlayerController,
        player_state: &Arc<RwLock<ObjectState>>,
        world: &Arc<WorldStore>,
        reg: &EngineRegistries,
        out: &mut AutoBotTickOutput,
    ) -> bool {
        let next = self
            .scaffold_cells
            .iter()
            .copied()
            .find(|&(pos, _)| !is_solid_walkable(world.get_node(pos).0, reg));
        let Some((pos, sup)) = next else {
            return true;
        };
        let Some(fill) = find_placeable_filler_block(player_state) else {
            return true;
        };

        if pos.y <= origin.y + 1 {
            if player.pos.y > origin.y as f32 + 0.8 {
                let _ = self.descend_cast_staircase(origin, dt, player, player_state, world, reg, out);
                return false;
            }
            let stand = Vec3::new(
                pos.x as f32 + 0.5,
                origin.y as f32,
                origin.z as f32 - 2.65,
            );
            if (player.pos.x - stand.x).abs() > 0.25
                || player.pos.z > origin.z as f32 - 2.38
                || (player.pos.z - stand.z).abs() > 0.30
                || (player.pos.y - origin.y as f32).abs() > 0.9
            {
                self.steer_surface(player, player_state, stand, world, reg, dt, out, true);
                return false;
            }
        } else {
            let rail_z = origin.z as f32 - 1.5;
            let step2_x = origin.x as f32 - 0.55;
            if player.pos.y < origin.y as f32 + 1.8 {
                if player.pos.y < origin.y as f32 + 0.5
                    && (player.pos.x > origin.x as f32 - 2.1 || (player.pos.z - rail_z).abs() > 0.45)
                {
                    let target = if player.pos.x > origin.x as f32 - 2.1 && (player.pos.z - rail_z).abs() > 0.45 {
                        Vec3::new(origin.x as f32 - 2.5, origin.y as f32, player.pos.z)
                    } else {
                        Vec3::new(origin.x as f32 - 2.5, origin.y as f32, rail_z)
                    };
                    self.steer_surface(player, player_state, target, world, reg, dt, out, true);
                    return false;
                }
                let walk_t = Vec3::new(step2_x, player.eye_pos().y - 0.1, rail_z);
                smooth_aim_at(player, walk_t, dt);
                let dx = step2_x - player.pos.x;
                let dz = rail_z - player.pos.z;
                let desired_yaw = (-dx).atan2(dz);
                let mut yaw_diff = desired_yaw - player.yaw;
                while yaw_diff > PI {
                    yaw_diff -= TAU;
                }
                while yaw_diff < -PI {
                    yaw_diff += TAU;
                }
                if yaw_diff.abs() < 0.18 {
                    out.move_input.forward = true;
                }
                let hlen = (dx * dx + dz * dz).sqrt().max(1e-4);
                let ax = (player.pos.x + (dx / hlen) * 0.75).floor() as i32;
                let az = rail_z.floor() as i32;
                let fy = player.pos.y.floor() as i32;
                if is_solid_walkable(world.get_node(IVec3::new(ax, fy, az)).0, reg) {
                    out.move_input.jump = true;
                }
                return false;
            }
            if player.pos.x > origin.x as f32 - 0.36
                || (player.pos.x - step2_x).abs() > 0.16
                || (player.pos.z - rail_z).abs() > 0.25
            {
                let walk_t = Vec3::new(step2_x, player.eye_pos().y - 0.1, rail_z);
                smooth_aim_at(player, walk_t, dt);
                let dx = step2_x - player.pos.x;
                let dz = rail_z - player.pos.z;
                let desired_yaw = (-dx).atan2(dz);
                let mut yaw_diff = desired_yaw - player.yaw;
                while yaw_diff > PI {
                    yaw_diff -= TAU;
                }
                while yaw_diff < -PI {
                    yaw_diff += TAU;
                }
                if yaw_diff.abs() < 0.18 {
                    out.move_input.forward = true;
                }
                out.move_input.sneak = true;
                return false;
            }
            out.move_input.sneak = true;
        }

        if equip_item_exact(player, player_state, &fill) {
            out.force_liquids_pointable = false;
            let aim_pt = sup.as_vec3() + Vec3::new(0.5, 0.96, 0.5);
            if smooth_aim_at(player, aim_pt, dt) {
                if let Some(hit) =
                    raycast_voxel_ex(player.eye_pos(), player.look_dir(), 5.0, false, world, reg)
                {
                    if hit.under == sup && hit.above == pos && self.place_cooldown <= 0.0 {
                        out.place_just = true;
                        self.place_cooldown = 0.25;
                    }
                }
            }
        }
        false
    }

    #[allow(clippy::too_many_arguments)]
    fn mine_block_or_approach(
        &mut self,
        dt: f32,
        player: &mut PlayerController,
        player_state: &Arc<RwLock<ObjectState>>,
        world: &Arc<WorldStore>,
        reg: &EngineRegistries,
        target: IVec3,
        prefer_surface: bool,
        out: &mut AutoBotTickOutput,
    ) -> bool {
        let cur_id = world.get_node(target).0;
        if cur_id == CONTENT_AIR || cur_id == CONTENT_IGNORE {
            return true;
        }
        let target_name = reg.get_name_from_content_id(cur_id);
        equip_best_tool_for_block(player, player_state, target_name);

        let target_c = target.as_vec3() + Vec3::splat(0.5);
        let dist_eye = player.eye_pos().distance(target_c);
        let horiz_dist = Vec3::new(target_c.x - player.pos.x, 0.0, target_c.z - player.pos.z).length();

        if dist_eye <= 4.1 {
            smooth_aim_at(player, target_c, dt);
            if let Some(hit) = raycast_voxel_ex(player.eye_pos(), player.look_dir(), 4.9, false, world, reg) {
                let hit_name = reg.get_name_from_content_id(hit.node_id);
                if hit.under == target {
                    out.mine_held = true;
                    return false;
                }
                // Clear obstructing breakable voxel blocking our raycast
                if hit.distance_sq <= 4.5 * 4.5
                    && hit_name != "mcl_core:bedrock"
                    && !is_obsidian(hit.node_id, reg)
                    && !self.is_active_scaffold_protected(hit.under)
                {
                    equip_best_tool_for_block(player, player_state, hit_name);
                    out.mine_held = true;
                    return false;
                }
            }
        }

        // Approach or tunnel toward stand position adjacent to target
        if prefer_surface && horiz_dist > 2.2 {
            self.steer_surface(player, player_state, target_c, world, reg, dt, out, false);
        } else {
            let stand_y = (target.y - 1).max(12);
            let stand = IVec3::new(target.x, stand_y, target.z);
            self.move_or_mine_to_voxel(dt, player, player_state, world, reg, stand, out);
        }
        false
    }

    #[allow(clippy::too_many_arguments)]
    fn place_item_at(
        &mut self,
        dt: f32,
        player: &mut PlayerController,
        player_state: &Arc<RwLock<ObjectState>>,
        world: &Arc<WorldStore>,
        reg: &EngineRegistries,
        item_name: &str,
        place_pos: IVec3,
        support_pos: IVec3,
        face_normal: IVec3,
        out: &mut AutoBotTickOutput,
    ) -> bool {
        self.place_item_at_inner(
            dt,
            player,
            player_state,
            world,
            reg,
            item_name,
            place_pos,
            support_pos,
            face_normal,
            true,
            out,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn place_item_at_inner(
        &mut self,
        dt: f32,
        player: &mut PlayerController,
        player_state: &Arc<RwLock<ObjectState>>,
        world: &Arc<WorldStore>,
        reg: &EngineRegistries,
        item_name: &str,
        place_pos: IVec3,
        support_pos: IVec3,
        face_normal: IVec3,
        allow_move_mine: bool,
        out: &mut AutoBotTickOutput,
    ) -> bool {
        let cur_name = reg.get_name_from_content_id(world.get_node(place_pos).0);
        if cur_name == item_name {
            return true;
        }
        let dist = player.pos.distance(place_pos.as_vec3() + 0.5);
        let horiz = Vec3::new(
            place_pos.x as f32 + 0.5 - player.pos.x,
            0.0,
            place_pos.z as f32 + 0.5 - player.pos.z,
        )
        .length();

        // Climb or step onto target elevation if separated by >1.8 blocks vertically
        if allow_move_mine && (place_pos.y as f32 - player.pos.y).abs() > 1.8 {
            self.move_or_mine_to_voxel(
                dt,
                player,
                player_state,
                world,
                reg,
                place_pos + IVec3::new(0, 0, -2),
                out,
            );
            return false;
        }

        let min_horiz = if place_pos.y <= player.pos.y.floor() as i32 {
            0.70
        } else {
            1.15
        };
        let max_horiz = if place_pos.y < player.pos.y.floor() as i32 && face_normal == IVec3::Y {
            1.12
        } else {
            3.5
        };

        // Don't stand overlapping the target placement block or too far away
        if horiz < min_horiz || horiz > max_horiz || dist > 3.9 {
            let dir = if horiz < 0.1 {
                Vec3::NEG_Z
            } else {
                Vec3::new(
                    player.pos.x - (place_pos.x as f32 + 0.5),
                    0.0,
                    player.pos.z - (place_pos.z as f32 + 0.5),
                )
                .normalize_or_zero()
            };
            let push_dist = if max_horiz < 2.0 {
                0.92
            } else if min_horiz < 1.0 {
                1.15
            } else {
                2.2
            };
            let desired = Vec3::new(
                place_pos.x as f32 + 0.5 + dir.x * push_dist,
                place_pos.y as f32,
                place_pos.z as f32 + 0.5 + dir.z * push_dist,
            );
            self.steer_surface(player, player_state, desired, world, reg, dt, out, false);
            return false;
        }

        if !equip_item_exact(player, player_state, item_name) {
            return false;
        }

        // Clear replaceable vegetation/block in place_pos if needed
        let (pid, _, _) = world.get_node(place_pos);
        if pid != CONTENT_AIR && !reg.get_node_def(pid).is_some_and(|d| d.buildable_to) {
            if allow_move_mine {
                let _ = self.mine_block_or_approach(dt, player, player_state, world, reg, place_pos, true, out);
            }
            return false;
        }

        let aim_pt = support_pos.as_vec3() + Vec3::splat(0.5) + face_normal.as_vec3() * 0.45;
        if smooth_aim_at(player, aim_pt, dt) {
            if let Some(hit) = raycast_voxel_ex(player.eye_pos(), player.look_dir(), 5.0, false, world, reg) {
                let hit_buildable = reg.get_node_def(hit.node_id).is_some_and(|d| d.buildable_to);
                let target_hit = (hit.under == support_pos && hit.above == place_pos)
                    || (hit.under == place_pos && hit_buildable);
                if target_hit && self.place_cooldown <= 0.0 {
                    out.place_just = true;
                    self.place_cooldown = 0.25;
                    return false;
                }
                if !target_hit
                    && hit.under != support_pos
                    && hit.under.y > place_pos.y
                    && hit.distance_sq <= 4.5 * 4.5
                {
                    let hn = reg.get_name_from_content_id(hit.node_id);
                    if hn != "mcl_core:bedrock" && !is_obsidian(hit.node_id, reg) {
                        equip_best_tool_for_block(player, player_state, hn);
                        out.mine_held = true;
                    }
                }
            }
        }
        false
    }

    #[allow(clippy::too_many_arguments)]
    fn scoop_or_pour_bucket(
        &mut self,
        dt: f32,
        player: &mut PlayerController,
        player_state: &Arc<RwLock<ObjectState>>,
        world: &Arc<WorldStore>,
        reg: &EngineRegistries,
        bucket_item: &str,
        target_voxel: IVec3,
        liquids_pointable: bool,
        desired_stand_dist: f32,
        accept_any_matching_liquid: bool,
        out: &mut AutoBotTickOutput,
    ) -> bool {
        if !equip_item_exact(player, player_state, bucket_item) {
            return false;
        }
        out.force_liquids_pointable = liquids_pointable;

        let target_id = world.get_node(target_voxel).0;
        let target_is_water = is_water_node(target_id, reg);
        let target_is_lava = is_lava_node(target_id, reg);

        let t_center = target_voxel.as_vec3() + Vec3::splat(0.5);
        let horiz = Vec3::new(t_center.x - player.pos.x, 0.0, t_center.z - player.pos.z).length();
        let eye_dist = player.eye_pos().distance(t_center);

        let min_horiz = if target_is_water && liquids_pointable {
            0.0
        } else if target_is_lava {
            1.2
        } else {
            0.70
        };
        let max_eye_dist = if target_is_lava { 3.5 } else { 3.9 };

        let away = if horiz < 0.1 {
            Vec3::NEG_Z
        } else {
            Vec3::new(player.pos.x - t_center.x, 0.0, player.pos.z - t_center.z).normalize_or_zero()
        };

        if eye_dist > max_eye_dist || horiz < min_horiz {
            if eye_dist > max_eye_dist
                && horiz <= desired_stand_dist + 0.65
                && player.pos.y > target_voxel.y as f32 + 1.6
            {
                if target_is_water {
                    self.steer_surface(player, player_state, t_center, world, reg, dt, out, false);
                } else {
                    let rim_target = IVec3::new(
                        (t_center.x + away.x * desired_stand_dist).floor() as i32,
                        target_voxel.y + 1,
                        (t_center.z + away.z * desired_stand_dist).floor() as i32,
                    );
                    self.move_or_mine_to_voxel(
                        dt,
                        player,
                        player_state,
                        world,
                        reg,
                        rim_target,
                        out,
                    );
                }
            } else {
                let stand = Vec3::new(
                    t_center.x + away.x * desired_stand_dist,
                    t_center.y + 1.0,
                    t_center.z + away.z * desired_stand_dist,
                );
                self.steer_surface(
                    player,
                    player_state,
                    stand,
                    world,
                    reg,
                    dt,
                    out,
                    !target_is_water,
                );
            }
            return false;
        }

        let aim_pt = if liquids_pointable {
            if player.eye_pos().y > target_voxel.y as f32 + 0.9 {
                target_voxel.as_vec3() + Vec3::new(0.5, 0.94, 0.5)
            } else {
                t_center
            }
        } else {
            target_voxel.as_vec3() + Vec3::new(0.5, 0.96, 0.5)
        };
        if smooth_aim_at(player, aim_pt, dt) {
            if let Some(hit) = raycast_voxel_ex(
                player.eye_pos(),
                player.look_dir(),
                5.0,
                liquids_pointable,
                world,
                reg,
            ) {
                let exact_hit = hit.under == target_voxel;
                let matching_wild_liquid = accept_any_matching_liquid
                    && liquids_pointable
                    && !self.is_frame_pos(hit.under)
                    && ((target_is_water && is_water_source_item(reg.get_name_from_content_id(hit.node_id)))
                        || (target_is_lava && is_lava_source_item(reg.get_name_from_content_id(hit.node_id))));
                if (exact_hit || matching_wild_liquid) && self.place_cooldown <= 0.0 {
                    out.place_just = true;
                    self.place_cooldown = 0.25;
                    return true;
                }
                if !exact_hit
                    && !is_protected_node(hit.node_id, reg)
                    && !is_liquid_node_id(hit.node_id, reg)
                    && !self.is_frame_pos(hit.under)
                    && !self.is_scaffold_pos(hit.under)
                    && hit.distance_sq <= 4.5 * 4.5
                {
                    equip_best_tool_for_block(
                        player,
                        player_state,
                        reg.get_name_from_content_id(hit.node_id),
                    );
                    out.mine_held = true;
                }
            }
        }
        false
    }

    #[allow(clippy::too_many_arguments)]
    fn steer_surface(
        &mut self,
        player: &mut PlayerController,
        player_state: &Arc<RwLock<ObjectState>>,
        target: Vec3,
        world: &Arc<WorldStore>,
        reg: &EngineRegistries,
        dt: f32,
        out: &mut AutoBotTickOutput,
        avoid_lava: bool,
    ) {
        let dx = target.x - player.pos.x;
        let dz = target.z - player.pos.z;
        let horiz_dist = (dx * dx + dz * dz).sqrt();
        if horiz_dist < 0.2 {
            return;
        }
        let dir_x = dx / horiz_dist;
        let dir_z = dz / horiz_dist;

        let aim_target = Vec3::new(target.x, player.eye_pos().y - 0.15, target.z);

        let feet_y = player.pos.y.floor() as i32;
        let cur_x = player.pos.x.floor() as i32;
        let cur_z = player.pos.z.floor() as i32;
        let ahead_x = (player.pos.x + dir_x * 0.85).floor() as i32;
        let ahead_z = (player.pos.z + dir_z * 0.85).floor() as i32;

        // Avoid stepping onto lava
        if avoid_lava && has_lava_in_column(world, reg, ahead_x, feet_y - 2..=feet_y + 1, ahead_z) {
            smooth_aim_at(player, aim_target, dt);
            out.move_input.right = true;
            out.move_input.jump = true;
            return;
        }

        let mut cols: Vec<(i32, i32)> = Vec::with_capacity(3);
        for &(cx, cz) in &[(ahead_x, ahead_z), (ahead_x, cur_z), (cur_x, ahead_z)] {
            if (cx, cz) != (cur_x, cur_z) && !cols.contains(&(cx, cz)) {
                cols.push((cx, cz));
            }
        }
        if cols.is_empty() {
            cols.push((ahead_x, ahead_z));
        }

        let mut f_solid = false;
        let mut h_solid = false;
        let mut head_bp: Option<IVec3> = None;
        let mut foot_bp: Option<IVec3> = None;
        let mut top_bp: Option<IVec3> = None;
        for &(cx, cz) in &cols {
            let fp = IVec3::new(cx, feet_y, cz);
            let hp = IVec3::new(cx, feet_y + 1, cz);
            let tp = IVec3::new(cx, feet_y + 2, cz);
            let fid = world.get_node(fp).0;
            let hid = world.get_node(hp).0;
            let tid = world.get_node(tp).0;
            if is_solid_walkable(fid, reg) {
                f_solid = true;
                if foot_bp.is_none() && !is_protected_node(fid, reg) && !self.is_active_scaffold_protected(fp) {
                    foot_bp = Some(fp);
                }
            }
            if is_solid_walkable(hid, reg) {
                h_solid = true;
                if head_bp.is_none() && !is_protected_node(hid, reg) && !self.is_active_scaffold_protected(hp) {
                    head_bp = Some(hp);
                }
            }
            if top_bp.is_none()
                && is_solid_walkable(tid, reg)
                && !is_protected_node(tid, reg)
                && !self.is_active_scaffold_protected(tp)
            {
                top_bp = Some(tp);
            }
        }

        if player.in_liquid {
            out.move_input.jump = true;
        } else if f_solid && !h_solid {
            out.move_input.jump = true;
        }

        let head_ceil = world.get_node(IVec3::new(cur_x, feet_y + 2, cur_z)).0;
        let overhead_blocked = is_solid_walkable(head_ceil, reg)
            && !is_protected_node(head_ceil, reg)
            && !self.is_active_scaffold_protected(IVec3::new(cur_x, feet_y + 2, cur_z));

        let in_strafe_escape = self.no_move_timer > 6.5 && (self.no_move_timer % 8.5) > 6.5;
        if in_strafe_escape {
            self.unstuck_dig_target = None;
        }

        let valid_unstuck = self.unstuck_dig_target.filter(|&p| {
            p.y >= feet_y
                && p.y <= feet_y + 2
                && !self.is_active_scaffold_protected(p)
                && is_solid_walkable(world.get_node(p).0, reg)
                && player.eye_pos().distance(p.as_vec3() + 0.5) <= 2.6
        });

        let break_pos = if in_strafe_escape {
            None
        } else if let Some(hp) = head_bp {
            if !player.on_ground {
                valid_unstuck.or(Some(hp))
            } else {
                Some(hp)
            }
        } else if self.no_move_timer > 1.2 {
            if let Some(p) = valid_unstuck {
                Some(p)
            } else {
                let cand = if overhead_blocked {
                    Some(IVec3::new(cur_x, feet_y + 2, cur_z)).or(top_bp).or(foot_bp)
                } else {
                    foot_bp.or(top_bp)
                };
                self.unstuck_dig_target = cand;
                cand
            }
        } else if !player.on_ground {
            valid_unstuck.filter(|&p| p.y >= feet_y + 1)
        } else {
            None
        };

        if let Some(bp) = break_pos {
            let b_id = world.get_node(bp).0;
            if b_id != CONTENT_AIR && b_id != CONTENT_IGNORE {
                self.unstuck_dig_target = Some(bp);
                equip_best_tool_for_block(player, player_state, reg.get_name_from_content_id(b_id));
                out.move_input.jump = false;
                let aimed = smooth_aim_at(player, bp.as_vec3() + 0.5, dt);
                if !aimed {
                    return;
                }
                if let Some(hit) = raycast_voxel_ex(player.eye_pos(), player.look_dir(), 4.5, false, world, reg) {
                    if !is_protected_node(hit.node_id, reg) && !self.is_active_scaffold_protected(hit.under) {
                        out.mine_held = true;
                        return;
                    }
                }
            }
        } else if self.no_move_timer > 3.5 {
            smooth_aim_at(player, aim_target, dt);
            let side = ((self.no_move_timer / 2.0) as usize + self.explore_idx) % 2;
            out.move_input.right = side == 0;
            out.move_input.left = side == 1;
            out.move_input.forward = in_strafe_escape;
            out.move_input.jump = true;
            return;
        }

        smooth_aim_at(player, aim_target, dt);
        out.move_input.forward = true;
        out.move_input.sprint = horiz_dist > 4.5 && player.hunger > 8 && !player.in_liquid;
        if self.no_move_timer > 1.0 && !player.in_liquid {
            out.move_input.jump = true;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn move_or_mine_to_voxel(
        &mut self,
        dt: f32,
        player: &mut PlayerController,
        player_state: &Arc<RwLock<ObjectState>>,
        world: &Arc<WorldStore>,
        reg: &EngineRegistries,
        goal: IVec3,
        out: &mut AutoBotTickOutput,
    ) {
        let cur = IVec3::new(
            player.pos.x.floor() as i32,
            player.pos.y.floor() as i32,
            player.pos.z.floor() as i32,
        );
        let dx = goal.x - cur.x;
        let dz = goal.z - cur.z;
        let dy = goal.y - cur.y;

        if dx == 0 && dz == 0 && dy == 0 {
            let center = Vec3::new(goal.x as f32 + 0.5, goal.y as f32, goal.z as f32 + 0.5);
            self.steer_surface(player, player_state, center, world, reg, dt, out, true);
            return;
        }

        // Strict orthogonal X/Z step toward goal
        let step_xz = if dx.abs() >= dz.abs() && dx != 0 {
            IVec3::new(dx.signum(), 0, 0)
        } else if dz != 0 {
            IVec3::new(0, 0, dz.signum())
        } else {
            IVec3::ZERO
        };

        let horiz_rem = dx.abs() + dz.abs();
        let step_y = if dy > 0 && (horiz_rem <= dy.abs() || step_xz == IVec3::ZERO) {
            1
        } else if dy < 0 && horiz_rem > 0 && (horiz_rem <= dy.abs() || step_xz != IVec3::ZERO) {
            -1
        } else {
            0
        };
        let next = cur + step_xz + IVec3::new(0, step_y, 0);
        let floor_below = next - IVec3::Y;
        let fb_id = world.get_node(floor_below).0;
        let fb2_id = world.get_node(floor_below - IVec3::Y).0;

        // Clean jump-pillar escape whenever vertical climb >= 2, mid-pillar jump, or ascending without an adjacent shelf
        let mid_pillar_jump = !player.on_ground
            && dy >= 1
            && world.get_node(cur - IVec3::Y).0 == CONTENT_AIR
            && is_solid_walkable(world.get_node(cur - IVec3::Y * 2).0, reg);
        if dy >= 2 || mid_pillar_jump || (step_y > 0 && !is_solid_walkable(fb_id, reg)) {
            if self.pillar_up_at_current_col(dt, player, player_state, world, reg, out) {
                return;
            }
        }

        // Determine required clearance voxels so player fits cleanly on step up, flat, or step down
        let mut required_clear = Vec::with_capacity(5);
        if step_y > 0 {
            required_clear.push(cur + IVec3::Y * 2);
            required_clear.push(next);
            required_clear.push(next + IVec3::Y);
            required_clear.push(next + IVec3::Y * 2);
        } else if step_y < 0 {
            required_clear.push(IVec3::new(next.x, cur.y + 1, next.z));
            required_clear.push(IVec3::new(next.x, cur.y, next.z));
            required_clear.push(next);
        } else {
            required_clear.push(next);
            required_clear.push(next + IVec3::Y);
        }

        for vp in required_clear {
            let id = world.get_node(vp).0;
            if is_solid_walkable(id, reg) {
                if is_protected_node(id, reg) {
                    out.move_input.right = true;
                    out.move_input.jump = true;
                    return;
                }
                let nname = reg.get_name_from_content_id(id);
                equip_best_tool_for_block(player, player_state, nname);
                smooth_aim_at(player, vp.as_vec3() + 0.5, dt);
                if let Some(hit) = raycast_voxel_ex(player.eye_pos(), player.look_dir(), 4.8, false, world, reg) {
                    if !is_protected_node(hit.node_id, reg) {
                        out.mine_held = true;
                    }
                }
                return;
            }
        }

        // Check floor safety & bridge gaps before moving into `next`
        if is_lava_node(fb_id, reg) || (fb_id == CONTENT_AIR && is_lava_node(fb2_id, reg)) {
            if !(matches!(self.phase, BotPhase::S4CastObsidianFrame) && is_lava_node(fb_id, reg)) {
                if let Some(fill) = find_placeable_filler_block(player_state) {
                    if is_solid_walkable(fb2_id, reg) {
                        let _ = self.place_item_at_inner(
                            dt,
                            player,
                            player_state,
                            world,
                            reg,
                            &fill,
                            floor_below,
                            floor_below - IVec3::Y,
                            IVec3::Y,
                            false,
                            out,
                        );
                        return;
                    }
                }
            }
            out.move_input.right = true;
            return;
        }

        if step_y == 0 && fb_id == CONTENT_AIR && !is_solid_walkable(fb2_id, reg) {
            if let Some(fill) = find_placeable_filler_block(player_state) {
                let fb3_id = world.get_node(floor_below - IVec3::Y * 2).0;
                if is_solid_walkable(fb3_id, reg) {
                    let _ = self.place_item_at_inner(
                        dt,
                        player,
                        player_state,
                        world,
                        reg,
                        &fill,
                        floor_below - IVec3::Y,
                        floor_below - IVec3::Y * 2,
                        IVec3::Y,
                        false,
                        out,
                    );
                    return;
                }
                let edge_pt = cur.as_vec3()
                    + Vec3::new(0.5, 0.0, 0.5)
                    + step_xz.as_vec3() * 0.72;
                let fwd_progress = (player.pos.x - (cur.x as f32 + 0.5)) * step_xz.x as f32
                    + (player.pos.z - (cur.z as f32 + 0.5)) * step_xz.z as f32;
                out.move_input.sneak = true;
                if fwd_progress < 0.62 {
                    smooth_aim_at(
                        player,
                        Vec3::new(edge_pt.x, player.eye_pos().y - 0.1, edge_pt.z),
                        dt,
                    );
                    out.move_input.forward = true;
                } else if equip_item_exact(player, player_state, &fill) {
                    let aim = (cur - IVec3::Y).as_vec3()
                        + Vec3::splat(0.5)
                        + step_xz.as_vec3() * 0.48;
                    if smooth_aim_at(player, aim, dt) && self.place_cooldown <= 0.0 {
                        if let Some(hit) = raycast_voxel_ex(player.eye_pos(), player.look_dir(), 4.5, false, world, reg) {
                            if hit.under == cur - IVec3::Y && hit.above == floor_below {
                                out.place_just = true;
                                self.place_cooldown = 0.24;
                            }
                        }
                    }
                }
                return;
            }
        }

        let next_center = next.as_vec3() + Vec3::new(0.5, 0.0, 0.5);
        smooth_aim_at(player, Vec3::new(next_center.x, player.eye_pos().y - 0.1, next_center.z), dt);
        out.move_input.forward = true;
        if step_y > 0 || player.in_liquid || self.no_move_timer > 1.0 {
            out.move_input.jump = true;
        }
    }

    fn pillar_up_at_current_col(
        &mut self,
        dt: f32,
        player: &mut PlayerController,
        player_state: &Arc<RwLock<ObjectState>>,
        world: &Arc<WorldStore>,
        reg: &EngineRegistries,
        out: &mut AutoBotTickOutput,
    ) -> bool {
        let py_i = (player.pos.y + 0.05).floor() as i32;
        let center_cx = player.pos.x.floor() as i32;
        let center_cz = player.pos.z.floor() as i32;

        let Some(fill) = find_placeable_filler_block(player_state) else {
            // Mine an adjacent solid block to replenish filler if underground with 0 blocks
            for off in [
                IVec3::new(1, 1, 0),
                IVec3::new(-1, 1, 0),
                IVec3::new(0, 1, 1),
                IVec3::new(0, 1, -1),
                IVec3::new(1, 0, 0),
                IVec3::new(-1, 0, 0),
            ] {
                let bp = IVec3::new(center_cx, py_i, center_cz) + off;
                let bid = world.get_node(bp).0;
                if is_solid_walkable(bid, reg) && !is_protected_node(bid, reg) {
                    let bname = reg.get_name_from_content_id(bid);
                    equip_best_tool_for_block(player, player_state, bname);
                    smooth_aim_at(player, bp.as_vec3() + 0.5, dt);
                    if let Some(hit) = raycast_voxel_ex(player.eye_pos(), player.look_dir(), 4.5, false, world, reg) {
                        if !is_protected_node(hit.node_id, reg) {
                            out.mine_held = true;
                        }
                    }
                    return true;
                }
            }
            return false;
        };

        // If standing on a voxel edge (like x=3.297 supported by x=2), pick the actual solid column under player's AABB!
        let cx0 = (player.pos.x - 0.30).floor() as i32;
        let cx1 = (player.pos.x + 0.30).floor() as i32;
        let cz0 = (player.pos.z - 0.30).floor() as i32;
        let cz1 = (player.pos.z + 0.30).floor() as i32;

        let mut found: Option<(i32, i32, i32)> = None;
        if is_solid_walkable(world.get_node(IVec3::new(center_cx, py_i - 1, center_cz)).0, reg)
            && !is_solid_walkable(world.get_node(IVec3::new(center_cx, py_i, center_cz)).0, reg)
        {
            found = Some((center_cx, py_i - 1, center_cz));
        } else if !player.on_ground
            && is_solid_walkable(world.get_node(IVec3::new(center_cx, py_i - 2, center_cz)).0, reg)
            && !is_solid_walkable(world.get_node(IVec3::new(center_cx, py_i - 1, center_cz)).0, reg)
        {
            found = Some((center_cx, py_i - 2, center_cz));
        } else {
            for cz in cz0..=cz1 {
                for cx in cx0..=cx1 {
                    if is_solid_walkable(world.get_node(IVec3::new(cx, py_i - 1, cz)).0, reg)
                        && !is_solid_walkable(world.get_node(IVec3::new(cx, py_i, cz)).0, reg)
                    {
                        found = Some((cx, py_i - 1, cz));
                        break;
                    }
                }
                if found.is_some() {
                    break;
                }
            }
        }
        let Some((cx, floor_y, cz)) = found else {
            return false;
        };

        let off_x = player.pos.x - (cx as f32 + 0.5);
        let off_z = player.pos.z - (cz as f32 + 0.5);
        if off_x.abs() > 0.18 || off_z.abs() > 0.18 {
            let center = Vec3::new(cx as f32 + 0.5, player.eye_pos().y - 0.1, cz as f32 + 0.5);
            smooth_aim_at(player, center, dt);
            let desired_yaw = off_x.atan2(-off_z);
            let mut yaw_diff = desired_yaw - player.yaw;
            while yaw_diff > PI {
                yaw_diff -= TAU;
            }
            while yaw_diff < -PI {
                yaw_diff += TAU;
            }
            if yaw_diff.abs() < 0.35 {
                out.move_input.forward = true;
            }
            out.move_input.sneak = true;
            return true;
        }

        // Clear 3 blocks above feet so a full jump + placed block never bumps ceiling
        for dy in 2..=4 {
            let vp = IVec3::new(cx, floor_y + dy, cz);
            let id = world.get_node(vp).0;
            if is_solid_walkable(id, reg) && !is_protected_node(id, reg) {
                equip_best_tool_for_block(player, player_state, reg.get_name_from_content_id(id));
                smooth_aim_at(player, vp.as_vec3() + 0.5, dt);
                if let Some(hit) = raycast_voxel_ex(player.eye_pos(), player.look_dir(), 4.5, false, world, reg) {
                    if !is_protected_node(hit.node_id, reg) {
                        out.mine_held = true;
                    }
                }
                return true;
            }
        }

        if !equip_item_exact(player, player_state, &fill) {
            return false;
        }
        let support = IVec3::new(cx, floor_y, cz);
        let place_pos = IVec3::new(cx, floor_y + 1, cz);
        let aim_pt = Vec3::new(cx as f32 + 0.5, floor_y as f32 + 0.96, cz as f32 + 0.5);
        if smooth_aim_at(player, aim_pt, dt) {
            out.move_input.jump = true;
            if player.pos.y >= (floor_y + 2) as f32 + 0.03 && self.place_cooldown <= 0.0 {
                if let Some(hit) = raycast_voxel_ex(player.eye_pos(), player.look_dir(), 5.0, false, world, reg) {
                    if hit.under == support && hit.above == place_pos {
                        out.place_just = true;
                        self.place_cooldown = 0.22;
                    }
                }
            }
        }
        true
    }

    fn explore_outward(
        &mut self,
        dt: f32,
        player: &mut PlayerController,
        player_state: &Arc<RwLock<ObjectState>>,
        world: &Arc<WorldStore>,
        reg: &EngineRegistries,
        out: &mut AutoBotTickOutput,
    ) {
        const WAYPOINTS: [(i32, i32); 16] = [
            (28, 0),
            (28, 28),
            (0, 28),
            (-28, 28),
            (-28, 0),
            (-28, -28),
            (0, -28),
            (28, -28),
            (56, 0),
            (56, 56),
            (0, 56),
            (-56, 56),
            (-56, 0),
            (-56, -56),
            (0, -56),
            (56, -56),
        ];
        let (wx, wz) = WAYPOINTS[self.explore_idx % WAYPOINTS.len()];
        let target = Vec3::new(
            (self.spawn_origin.x + wx) as f32 + 0.5,
            player.pos.y,
            (self.spawn_origin.z + wz) as f32 + 0.5,
        );
        let horiz = Vec3::new(target.x - player.pos.x, 0.0, target.z - player.pos.z).length();
        if horiz < 6.0 || self.no_move_timer > 4.5 {
            self.explore_idx = (self.explore_idx + 1) % WAYPOINTS.len();
            self.no_move_timer = 0.0;
        }
        self.steer_surface(player, player_state, target, world, reg, dt, out, true);
    }
}

// =============================================================================
// Registry, Craft, Inventory, Aim & World Query Helpers
// =============================================================================

fn smooth_aim_at(player: &mut PlayerController, target: Vec3, dt: f32) -> bool {
    let eye = player.eye_pos();
    let d = target - eye;
    if d.length_squared() < 1e-6 {
        return true;
    }
    let horiz_raw = (d.x * d.x + d.z * d.z).sqrt();
    let horiz = horiz_raw.max(1e-5);
    let desired_yaw = if horiz_raw < 0.05 {
        player.yaw
    } else {
        (-d.x).atan2(d.z)
    };
    let desired_pitch = d.y.atan2(horiz).clamp(-1.48, 1.48);

    let mut dyaw = desired_yaw - player.yaw;
    while dyaw > PI {
        dyaw -= TAU;
    }
    while dyaw < -PI {
        dyaw += TAU;
    }
    let dpitch = desired_pitch - player.pitch;
    let max_step = 14.0 * dt;
    player.yaw = (player.yaw + dyaw.clamp(-max_step, max_step) + PI).rem_euclid(TAU) - PI;
    player.pitch = (player.pitch + dpitch.clamp(-max_step, max_step)).clamp(-1.48, 1.48);

    dyaw.abs() < 0.06 && dpitch.abs() < 0.06
}

fn player_horizontal_fwd(yaw: f32) -> Vec3 {
    Vec3::new(-yaw.sin(), 0.0, yaw.cos()).normalize_or_zero()
}

fn is_log_item_name(name: &str) -> bool {
    let l = name.to_ascii_lowercase();
    (l.starts_with("mcl_trees:tree_") || l == "mcl_core:tree")
        && !l.contains("stripped")
        && !l.contains("bark")
}

fn is_plank_item_name(name: &str) -> bool {
    let l = name.to_ascii_lowercase();
    l.starts_with("mcl_trees:wood_") || l == "mcl_core:wood"
}

fn is_log_node_id(id: u16, reg: &EngineRegistries) -> bool {
    id != CONTENT_AIR
        && id != CONTENT_IGNORE
        && is_log_item_name(reg.get_name_from_content_id(id))
}

fn is_stone_or_cobble(id: u16, reg: &EngineRegistries) -> bool {
    if id == CONTENT_AIR || id == CONTENT_IGNORE {
        return false;
    }
    let n = reg.get_name_from_content_id(id);
    n == "mcl_core:stone" || n == "mcl_core:cobble"
}

fn is_iron_ore_name(name: &str) -> bool {
    matches!(name, "mcl_core:stone_with_iron" | "mcl_core:iron_ore")
}

fn is_coal_ore_name(name: &str) -> bool {
    matches!(name, "mcl_core:stone_with_coal" | "mcl_core:coal_ore")
}

fn is_water_source_item(name: &str) -> bool {
    matches!(
        name,
        "mcl_core:water_source"
            | "mcl_core:river_water_source"
            | "default:water_source"
    )
}

fn is_lava_source_item(name: &str) -> bool {
    matches!(
        name,
        "mcl_core:lava_source" | "mcl_nether:nether_lava_source" | "default:lava_source"
    )
}

fn is_water_node(id: u16, reg: &EngineRegistries) -> bool {
    id != CONTENT_AIR
        && id != CONTENT_IGNORE
        && is_water_source_item(reg.get_name_from_content_id(id))
}

fn is_lava_node(id: u16, reg: &EngineRegistries) -> bool {
    if id == CONTENT_AIR || id == CONTENT_IGNORE {
        return false;
    }
    let n = reg.get_name_from_content_id(id);
    n.contains("lava")
}

fn is_liquid_node_id(id: u16, reg: &EngineRegistries) -> bool {
    id != CONTENT_AIR
        && id != CONTENT_IGNORE
        && reg.get_node_def(id).is_some_and(is_liquid_node)
}

fn is_solid_walkable(id: u16, reg: &EngineRegistries) -> bool {
    id != CONTENT_AIR
        && id != CONTENT_IGNORE
        && reg
            .get_node_def(id)
            .is_some_and(|d| d.walkable && !is_liquid_node(d))
}

fn is_protected_node(id: u16, reg: &EngineRegistries) -> bool {
    if id == CONTENT_AIR || id == CONTENT_IGNORE {
        return false;
    }
    let n = reg.get_name_from_content_id(id);
    n == "mcl_core:bedrock" || is_obsidian(id, reg) || is_nether_portal_node(id, reg)
}

fn has_lava_in_column(
    world: &WorldStore,
    reg: &EngineRegistries,
    x: i32,
    y_range: std::ops::RangeInclusive<i32>,
    z: i32,
) -> bool {
    y_range.into_iter().any(|y| is_lava_node(world.get_node(IVec3::new(x, y, z)).0, reg))
}

fn is_submerged_around(world: &WorldStore, reg: &EngineRegistries, pos: IVec3) -> bool {
    [
        pos + IVec3::Y,
        pos + IVec3::X,
        pos - IVec3::X,
        pos + IVec3::Z,
        pos - IVec3::Z,
    ]
    .into_iter()
    .any(|p| is_liquid_node_id(world.get_node(p).0, reg))
}

fn count_item(player_state: &Arc<RwLock<ObjectState>>, item_name: &str) -> u16 {
    count_items_matching(player_state, |n| n == item_name)
}

fn count_raw_iron(player_state: &Arc<RwLock<ObjectState>>) -> u16 {
    count_items_matching(player_state, |n| {
        n == "mcl_raw_ores:raw_iron" || n == "mcl_core:iron_lump"
    })
}

fn count_items_matching<F: Fn(&str) -> bool>(
    player_state: &Arc<RwLock<ObjectState>>,
    pred: F,
) -> u16 {
    let ps = player_state.read();
    let mg = ps.meta.read();
    mg.inventories
        .get("main")
        .map(|inv| {
            inv.iter()
                .filter(|s| !s.is_empty() && pred(&s.name))
                .map(|s| s.count)
                .sum()
        })
        .unwrap_or(0)
}

fn most_common_item_matching<F: Fn(&str) -> bool>(
    player_state: &Arc<RwLock<ObjectState>>,
    pred: F,
) -> Option<(String, u16)> {
    let ps = player_state.read();
    let mg = ps.meta.read();
    let inv = mg.inventories.get("main")?;
    let mut counts: std::collections::HashMap<String, u16> = std::collections::HashMap::new();
    for s in inv {
        if !s.is_empty() && pred(&s.name) {
            *counts.entry(s.name.clone()).or_insert(0) += s.count;
        }
    }
    counts.into_iter().max_by_key(|(_, c)| *c)
}

fn equip_item_exact(
    player: &mut PlayerController,
    player_state: &Arc<RwLock<ObjectState>>,
    item_name: &str,
) -> bool {
    equip_item_matching(player, player_state, |n| n == item_name)
}

fn equip_item_matching<F: Fn(&str) -> bool>(
    player: &mut PlayerController,
    player_state: &Arc<RwLock<ObjectState>>,
    pred: F,
) -> bool {
    let mut chosen_hotbar: Option<usize> = None;
    {
        let ps = player_state.read();
        let mut mg = ps.meta.write();
        if let Some(main) = mg.inventories.get_mut("main") {
            if let Some(hb) = (0..9.min(main.len())).find(|&i| !main[i].is_empty() && pred(&main[i].name)) {
                chosen_hotbar = Some(hb);
            } else if let Some(bag_idx) = (9..main.len()).find(|&i| !main[i].is_empty() && pred(&main[i].name)) {
                let target_hb = (0..9.min(main.len()))
                    .find(|&i| main[i].is_empty())
                    .unwrap_or(8);
                main.swap(bag_idx, target_hb);
                chosen_hotbar = Some(target_hb);
            }
        }
    }
    if let Some(slot) = chosen_hotbar {
        player.selected_slot = slot;
        player_state.write().wield_index = slot + 1;
        true
    } else {
        false
    }
}

fn equip_best_tool_for_block(
    player: &mut PlayerController,
    player_state: &Arc<RwLock<ObjectState>>,
    node_name: &str,
) {
    let needs_pick = is_iron_ore_name(node_name)
        || is_coal_ore_name(node_name)
        || node_name == "mcl_core:stone"
        || node_name == "mcl_core:cobble"
        || node_name == "mcl_nether:netherrack";
    if needs_pick {
        if equip_item_exact(player, player_state, "mcl_tools:pick_stone") {
            return;
        }
        let _ = equip_item_exact(player, player_state, "mcl_tools:pick_wood");
        return;
    }
    // Keep buckets/tools away when punching wood/dirt/gravel if an empty hand slot exists
    let ps = player_state.read();
    let mg = ps.meta.read();
    if let Some(main) = mg.inventories.get("main") {
        let cur = main.get(player.selected_slot.min(8));
        if cur.is_some_and(|s| is_empty_bucket_item(&s.name) || s.name.contains("bucket_")) {
            if let Some(safe) = (0..9.min(main.len())).find(|&i| {
                main[i].is_empty() || !main[i].name.contains("bucket")
            }) {
                drop(mg);
                drop(ps);
                player.selected_slot = safe;
                player_state.write().wield_index = safe + 1;
            }
        }
    }
}

fn find_placeable_filler_block(player_state: &Arc<RwLock<ObjectState>>) -> Option<String> {
    let candidates = [
        "mcl_core:cobble",
        "mcl_core:dirt",
        "mcl_nether:netherrack",
        "mcl_core:gravel",
    ];
    for &c in &candidates {
        if count_item(player_state, c) > 0 {
            return Some(c.to_string());
        }
    }
    most_common_item_matching(player_state, is_plank_item_name).map(|(n, _)| n)
}

fn take_raw_iron_from_inventory(player_state: &Arc<RwLock<ObjectState>>, max_take: u16) -> ItemStackData {
    let raw = take_item_from_inventory(player_state, "mcl_raw_ores:raw_iron", max_take);
    if !raw.is_empty() {
        return raw;
    }
    take_item_from_inventory(player_state, "mcl_core:iron_lump", max_take)
}

fn take_item_from_inventory(
    player_state: &Arc<RwLock<ObjectState>>,
    item_name: &str,
    max_take: u16,
) -> ItemStackData {
    if max_take == 0 {
        return ItemStackData::default();
    }
    let ps = player_state.read();
    let mut mg = ps.meta.write();
    let Some(main) = mg.inventories.get_mut("main") else {
        return ItemStackData::default();
    };
    let mut taken = 0u16;
    for slot in main.iter_mut() {
        if slot.is_empty() || slot.name != item_name {
            continue;
        }
        let take = (max_take - taken).min(slot.count);
        slot.count -= take;
        taken += take;
        if slot.count == 0 {
            *slot = ItemStackData::default();
        }
        if taken >= max_take {
            break;
        }
    }
    if taken == 0 {
        ItemStackData::default()
    } else {
        ItemStackData::from_spec(&format!("{item_name} {taken}"))
    }
}

/// Evaluate a 2x2 or 3x3 recipe using `resolve_craft_recipe`, consume exact ingredients, and add output.
fn craft_via_registry(
    reg: &EngineRegistries,
    player_state: &Arc<RwLock<ObjectState>>,
    width: usize,
    pattern: &[Option<(&str, u16)>],
) -> Option<String> {
    let mut need: std::collections::HashMap<&str, u16> = std::collections::HashMap::new();
    let grid: Vec<ItemStackData> = pattern
        .iter()
        .map(|entry| match entry {
            Some((name, count)) => {
                *need.entry(*name).or_insert(0) += *count;
                ItemStackData::from_spec(&format!("{name} {count}"))
            }
            None => ItemStackData::default(),
        })
        .collect();

    for (&name, &req_count) in &need {
        if count_item(player_state, name) < req_count {
            return None;
        }
    }

    let craft_out = resolve_craft_recipe(reg, &grid, width, "normal")?;
    if craft_out.item.is_empty() {
        return None;
    }

    // Consume 1 item per pattern slot minus recipe replacements/decrements
    for (idx, entry) in pattern.iter().enumerate() {
        if let Some((name, count)) = entry {
            let taken = take_item_from_inventory(player_state, name, *count);
            if taken.count < *count {
                return None;
            }
            if let Some(rem) = craft_out.decrements.get(idx) {
                if !rem.is_empty() {
                    add_drops_to_inventory(
                        player_state,
                        &[format!("{} {}", rem.name, rem.count)],
                        reg,
                    );
                }
            }
        }
    }

    let out_name = craft_out.item.name.clone();
    let out_spec = format!("{} {}", out_name, craft_out.item.count.max(1));
    add_drops_to_inventory(player_state, &[out_spec], reg);
    Some(out_name)
}

#[allow(clippy::too_many_arguments)]
fn find_nearest_node<F: Fn(&str, IVec3) -> bool>(
    world: &WorldStore,
    reg: &EngineRegistries,
    origin: Vec3,
    chunk_radius: i32,
    min_y: i32,
    max_y: i32,
    blacklist: &HashSet<IVec3>,
    pred: F,
    prefer_lowest_y: bool,
) -> Option<IVec3> {
    let ocx = (origin.x / 16.0).floor() as i32;
    let ocz = (origin.z / 16.0).floor() as i32;
    let mut best: Option<(f32, IVec3)> = None;
    let chunks_guard = world.chunks.read();

    for (&cpos, chunk_arc) in chunks_guard.iter() {
        if (cpos.x - ocx).abs() > chunk_radius || (cpos.z - ocz).abs() > chunk_radius {
            continue;
        }
        let base_y = cpos.y * 16;
        if base_y + 15 < min_y || base_y > max_y {
            continue;
        }
        let chunk = chunk_arc.read();
        let base_x = cpos.x * 16;
        let base_z = cpos.z * 16;
        for lz in 0..CHUNK_SIZE {
            let wz = base_z + lz as i32;
            for lx in 0..CHUNK_SIZE {
                let wx = base_x + lx as i32;
                for ly in 0..CHUNK_SIZE {
                    let wy = base_y + ly as i32;
                    if wy < min_y || wy > max_y {
                        continue;
                    }
                    let id = chunk.nodes[crate::world::local_index(lx, ly, lz)];
                    if id == CONTENT_AIR || id == CONTENT_IGNORE {
                        continue;
                    }
                    let wpos = IVec3::new(wx, wy, wz);
                    if blacklist.contains(&wpos) {
                        continue;
                    }
                    let name = reg.get_name_from_content_id(id);
                    if !pred(name, wpos) {
                        continue;
                    }
                    let d2 = (wpos.as_vec3() + 0.5).distance_squared(origin)
                        + if prefer_lowest_y {
                            ((wy - SEA_LEVEL) as f32).max(0.0) * 1.5
                        } else {
                            0.0
                        };
                    if best.is_none_or(|(bd, _)| d2 < bd) {
                        best = Some((d2, wpos));
                    }
                }
            }
        }
    }
    best.map(|(_, p)| p)
}

#[allow(clippy::too_many_arguments)]
fn find_nearest_surface_liquid<F: Fn(&str, IVec3) -> bool>(
    world: &WorldStore,
    reg: &EngineRegistries,
    origin: Vec3,
    chunk_radius: i32,
    min_y: i32,
    max_y: i32,
    blacklist: &HashSet<IVec3>,
    pred: F,
    prefer_lowest_y: bool,
) -> Option<IVec3> {
    let ocx = (origin.x / 16.0).floor() as i32;
    let ocz = (origin.z / 16.0).floor() as i32;
    let mut best: Option<(f32, IVec3)> = None;
    let chunks_guard = world.chunks.read();

    for (&cpos, chunk_arc) in chunks_guard.iter() {
        if (cpos.x - ocx).abs() > chunk_radius || (cpos.z - ocz).abs() > chunk_radius {
            continue;
        }
        let base_y = cpos.y * 16;
        if base_y + 15 < min_y || base_y > max_y {
            continue;
        }
        let chunk = chunk_arc.read();
        let above_chunk = chunks_guard
            .get(&IVec3::new(cpos.x, cpos.y + 1, cpos.z))
            .map(|a| a.read());
        let base_x = cpos.x * 16;
        let base_z = cpos.z * 16;
        for lz in 0..CHUNK_SIZE {
            let wz = base_z + lz as i32;
            for lx in 0..CHUNK_SIZE {
                let wx = base_x + lx as i32;
                for ly in 0..CHUNK_SIZE {
                    let wy = base_y + ly as i32;
                    if wy < min_y || wy > max_y {
                        continue;
                    }
                    let id = chunk.nodes[crate::world::local_index(lx, ly, lz)];
                    if id == CONTENT_AIR || id == CONTENT_IGNORE {
                        continue;
                    }
                    let above_air = if ly + 1 < CHUNK_SIZE {
                        chunk.nodes[crate::world::local_index(lx, ly + 1, lz)] == CONTENT_AIR
                    } else {
                        above_chunk.as_ref().is_some_and(|ac| {
                            ac.nodes[crate::world::local_index(lx, 0, lz)] == CONTENT_AIR
                        })
                    };
                    if !above_air {
                        continue;
                    }
                    let wpos = IVec3::new(wx, wy, wz);
                    if blacklist.contains(&wpos) {
                        continue;
                    }
                    let name = reg.get_name_from_content_id(id);
                    if !pred(name, wpos) {
                        continue;
                    }
                    let d2 = (wpos.as_vec3() + 0.5).distance_squared(origin)
                        + if prefer_lowest_y {
                            ((wy - SEA_LEVEL) as f32).max(0.0) * 1.5
                        } else {
                            0.0
                        };
                    if best.is_none_or(|(bd, _)| d2 < bd) {
                        best = Some((d2, wpos));
                    }
                }
            }
        }
    }
    best.map(|(_, p)| p)
}

fn find_surface_lava_pool(
    world: &WorldStore,
    reg: &EngineRegistries,
    origin: Vec3,
    blacklist: &HashSet<IVec3>,
) -> Option<IVec3> {
    let lava_id = reg.get_content_id("mcl_core:lava_source");
    if lava_id == CONTENT_UNKNOWN {
        return None;
    }
    let chunks_guard = world.chunks.read();
    let mut buckets: HashMap<(i32, i32, i32), (f32, IVec3)> = HashMap::new();
    let ocx = (origin.x / 16.0).floor() as i32;
    let ocz = (origin.z / 16.0).floor() as i32;

    for (&cpos, chunk_arc) in chunks_guard.iter() {
        if (cpos.x - ocx).abs() > 12 || (cpos.z - ocz).abs() > 12 {
            continue;
        }
        let base_y = cpos.y * 16;
        if base_y + 15 < SEA_LEVEL || base_y > 88 {
            continue;
        }
        let chunk = chunk_arc.read();
        let base_x = cpos.x * 16;
        let base_z = cpos.z * 16;
        for lz in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                for ly in 0..CHUNK_SIZE {
                    if chunk.nodes[crate::world::local_index(lx, ly, lz)] == lava_id {
                        let p = IVec3::new(base_x + lx as i32, base_y + ly as i32, base_z + lz as i32);
                        if blacklist.contains(&p) {
                            continue;
                        }
                        let d2 = (p.as_vec3() + 0.5).distance_squared(origin);
                        let key = (p.x.div_euclid(4), p.y, p.z.div_euclid(4));
                        let entry = buckets.entry(key).or_insert((d2, p));
                        if d2 < entry.0 {
                            *entry = (d2, p);
                        }
                    }
                }
            }
        }
    }
    drop(chunks_guard);

    let mut candidates: Vec<(f32, IVec3)> = buckets.into_values().collect();
    candidates.sort_by(|a, b| a.0.total_cmp(&b.0));

    for (_, cand) in candidates {
        let mut members = Vec::new();
        for dz in -4..=4 {
            for dx in -4..=4 {
                let p = cand + IVec3::new(dx, 0, dz);
                if world.get_node(p).0 == lava_id {
                    members.push(p);
                }
            }
        }
        if members.len() >= 14 {
            let n = members.len() as f32;
            let mean_x = members.iter().map(|p| p.x as f32 + 0.5).sum::<f32>() / n;
            let mean_z = members.iter().map(|p| p.z as f32 + 0.5).sum::<f32>() / n;
            let center_target = Vec3::new(mean_x, cand.y as f32 + 0.5, mean_z);
            let closest = members
                .into_iter()
                .min_by(|a, b| {
                    let da = (a.as_vec3() + 0.5).distance_squared(center_target);
                    let db = (b.as_vec3() + 0.5).distance_squared(center_target);
                    da.total_cmp(&db)
                })
                .unwrap_or(cand);
            return Some(closest);
        }
    }
    None
}

fn is_good_place_spot(world: &WorldStore, reg: &EngineRegistries, pos: IVec3) -> bool {
    let floor = world.get_node(pos - IVec3::Y).0;
    let here = world.get_node(pos).0;
    let above = world.get_node(pos + IVec3::Y).0;
    is_solid_walkable(floor, reg)
        && (here == CONTENT_AIR || reg.get_node_def(here).is_some_and(|d| d.buildable_to))
        && !is_liquid_node_id(here, reg)
        && above == CONTENT_AIR
}

fn find_nearby_placement_spot(
    world: &WorldStore,
    reg: &EngineRegistries,
    player_pos: Vec3,
    blacklist: &HashSet<IVec3>,
) -> Option<IVec3> {
    let base = IVec3::new(
        player_pos.x.floor() as i32,
        player_pos.y.floor() as i32,
        player_pos.z.floor() as i32,
    );
    let mut best: Option<(f32, IVec3)> = None;
    for dz in -3i32..=3 {
        for dx in -3i32..=3 {
            if dx.abs() + dz.abs() < 2 {
                continue;
            }
            for dy in -1..=1 {
                let p = base + IVec3::new(dx, dy, dz);
                if !blacklist.contains(&p) && is_good_place_spot(world, reg, p) {
                    let d2 = (p.as_vec3() + 0.5).distance_squared(player_pos);
                    if best.is_none_or(|(bd, _)| d2 < bd) {
                        best = Some((d2, p));
                    }
                }
            }
        }
    }
    best.map(|(_, p)| p)
}

fn find_local_ground_top(world: &WorldStore, reg: &EngineRegistries, x: i32, z: i32, min_y: i32, max_y: i32) -> i32 {
    for y in (min_y..=max_y).rev() {
        let id = world.get_node(IVec3::new(x, y, z)).0;
        let above = world.get_node(IVec3::new(x, y + 1, z)).0;
        if is_solid_walkable(id, reg) && !is_solid_walkable(above, reg) && !is_liquid_node_id(above, reg) {
            return y;
        }
    }
    min_y
}

fn find_portal_site_near_lava(
    world: &WorldStore,
    reg: &EngineRegistries,
    lava_c: IVec3,
) -> Option<(IVec3, IVec3)> {
    // Pick the flattest solid-base site outside the 5x4 lava ellipse around rim height
    let rim_y = lava_c.y + 1;
    let offsets = [
        IVec3::new(-1, 0, -5),
        IVec3::new(-1, 0, 5),
        IVec3::new(5, 0, 0),
        IVec3::new(-7, 0, 0),
        IVec3::new(0, 0, -7),
    ];
    let mut best: Option<(usize, IVec3, IVec3)> = None;
    for off in offsets {
        let gx = lava_c.x + off.x;
        let gz = lava_c.z + off.z;
        let mut min_g = i32::MAX;
        for z in (gz - 3)..=gz {
            for x in (gx - 3)..=(gx + 2) {
                let g = find_local_ground_top(world, reg, x, z, rim_y - 3, rim_y + 4);
                min_g = min_g.min(g);
            }
        }
        let ground_y = min_g.max(lava_c.y);
        let origin = IVec3::new(gx, ground_y + 1, gz);
        // Ensure zero overlap with lava in the frame or immediate front walkway
        let mut touches_lava = false;
        'chk: for y in (origin.y - 2)..=(origin.y + 4) {
            for z in (origin.z - 3)..=(origin.z + 1) {
                for x in (origin.x - 3)..=(origin.x + 2) {
                    if is_lava_node(world.get_node(IVec3::new(x, y, z)).0, reg) {
                        touches_lava = true;
                        break 'chk;
                    }
                }
            }
        }
        if touches_lava {
            continue;
        }
        // Sunken 1-block reservoir to the left of the portal at ground level `origin.y - 1`
        let res = IVec3::new(origin.x - 3, origin.y - 1, origin.z);
        if !is_solid_walkable(world.get_node(res - IVec3::Y).0, reg)
            && !is_solid_walkable(world.get_node(res).0, reg)
        {
            continue;
        }
        let mut missing_floor = 0usize;
        let mut above_blocks = 0usize;
        for z in (origin.z - 3)..=origin.z {
            for x in (origin.x - 3)..=(origin.x + 2) {
                let fp = IVec3::new(x, origin.y - 1, z);
                if fp != res && !is_solid_walkable(world.get_node(fp).0, reg) {
                    missing_floor += 1;
                }
                for y in origin.y..=(origin.y + 4) {
                    let id = world.get_node(IVec3::new(x, y, z)).0;
                    if id != CONTENT_AIR && id != CONTENT_IGNORE {
                        above_blocks += 1;
                    }
                }
            }
        }
        let score = missing_floor * 4 + above_blocks;
        if best.is_none_or(|(bs, _, _)| score < bs) {
            best = Some((score, origin, res));
        }
    }
    if let Some((_, origin, res)) = best {
        return Some((origin, res));
    }

    let gx = lava_c.x - 1;
    let gz = lava_c.z - 6;
    let mut min_g = i32::MAX;
    for z in (gz - 3)..=gz {
        for x in (gx - 3)..=(gx + 2) {
            let g = find_local_ground_top(world, reg, x, z, rim_y - 3, rim_y + 4);
            min_g = min_g.min(g);
        }
    }
    let ground_y = min_g.max(lava_c.y);
    let fallback_origin = IVec3::new(gx, ground_y + 1, gz);
    let fallback_res = IVec3::new(fallback_origin.x - 3, fallback_origin.y - 1, fallback_origin.z);
    Some((fallback_origin, fallback_res))
}

fn build_cast_staircase_specs(origin: IVec3) -> Vec<(IVec3, IVec3)> {
    let ox = origin.x;
    let oy = origin.y;
    let oz = origin.z - 2;
    vec![
        (IVec3::new(ox - 2, oy, oz), IVec3::new(ox - 2, oy - 1, oz)),
        (IVec3::new(ox - 1, oy, oz), IVec3::new(ox - 1, oy - 1, oz)),
        (IVec3::new(ox, oy, oz), IVec3::new(ox, oy - 1, oz)),
        (IVec3::new(ox + 1, oy, oz), IVec3::new(ox + 1, oy - 1, oz)),
        (IVec3::new(ox - 1, oy + 1, oz), IVec3::new(ox - 1, oy, oz)),
        (IVec3::new(ox, oy + 1, oz), IVec3::new(ox, oy, oz)),
        (IVec3::new(ox + 1, oy + 1, oz), IVec3::new(ox + 1, oy, oz)),
        // Place ox + 1 BEFORE ox at oy + 2 so ox does not occlude support of ox + 1
        (IVec3::new(ox + 1, oy + 2, oz), IVec3::new(ox + 1, oy + 1, oz)),
        (IVec3::new(ox, oy + 2, oz), IVec3::new(ox, oy + 1, oz)),
    ]
}

fn build_4x5_frame_specs(origin: IVec3) -> Vec<FrameCellSpec> {
    let mut out = Vec::with_capacity(14);
    let ox = origin.x;
    let oy = origin.y;
    let oz = origin.z;

    // Row dy = 0: bottom obsidian bar (4 cells)
    for dx in -1..=2 {
        let p = IVec3::new(ox + dx, oy, oz);
        out.push(FrameCellSpec {
            pos: p,
            support: p - IVec3::Y,
            face_normal: IVec3::Y,
        });
    }
    // Rows dy = 1..=3: left & right pillars sorted by ascending dy (6 cells)
    for dy in 1..=3 {
        for px in [ox - 1, ox + 2] {
            let p = IVec3::new(px, oy + dy, oz);
            out.push(FrameCellSpec {
                pos: p,
                support: p - IVec3::Y,
                face_normal: IVec3::Y,
            });
        }
    }
    // Row dy = 4: top corner pillars first (vertical support), then top interior bridge (4 cells)
    for px in [ox - 1, ox + 2] {
        let p = IVec3::new(px, oy + 4, oz);
        out.push(FrameCellSpec {
            pos: p,
            support: p - IVec3::Y,
            face_normal: IVec3::Y,
        });
    }
    out.push(FrameCellSpec {
        pos: IVec3::new(ox, oy + 4, oz),
        support: IVec3::new(ox - 1, oy + 4, oz),
        face_normal: IVec3::X,
    });
    out.push(FrameCellSpec {
        pos: IVec3::new(ox + 1, oy + 4, oz),
        support: IVec3::new(ox + 2, oy + 4, oz),
        face_normal: IVec3::NEG_X,
    });
    out
}

fn is_dangerous_floor_below(pos: Vec3, world: &WorldStore, reg: &EngineRegistries) -> bool {
    let fy = pos.y.floor() as i32;
    let min_x = (pos.x - 0.28).floor() as i32;
    let max_x = (pos.x + 0.28).floor() as i32;
    let min_z = (pos.z - 0.28).floor() as i32;
    let max_z = (pos.z + 0.28).floor() as i32;
    let mut any_solid = false;
    for vz in min_z..=max_z {
        for vx in min_x..=max_x {
            let bid = world.get_node(IVec3::new(vx, fy - 1, vz)).0;
            if is_lava_node(bid, reg) {
                return true;
            }
            if is_solid_walkable(bid, reg) {
                any_solid = true;
            }
        }
    }
    !any_solid
}

fn pick_best_nether_exit_dir(pos: Vec3, world: &WorldStore, reg: &EngineRegistries) -> IVec3 {
    let base = IVec3::new(pos.x.floor() as i32, pos.y.floor() as i32, pos.z.floor() as i32);
    let dirs = [IVec3::Z, IVec3::NEG_Z, IVec3::X, IVec3::NEG_X];
    let mut best = (i32::MIN, IVec3::Z);
    for d in dirs {
        let mut score = 0i32;
        for step in 1..=6 {
            let p = base + d * step;
            let body = world.get_node(p).0;
            let head = world.get_node(p + IVec3::Y).0;
            if is_nether_portal_node(body, reg) || is_obsidian(body, reg) {
                score -= 20;
            } else if is_lava_node(body, reg) || is_lava_node(head, reg) {
                score -= 200;
            }
            if has_lava_in_column(world, reg, p.x, (p.y - 3)..=p.y, p.z) {
                score -= 50;
            }
            let solid_ground = (-1..=1).any(|oy| is_solid_walkable(world.get_node(p + IVec3::new(0, oy - 1, 0)).0, reg));
            if solid_ground {
                score += 15;
            } else {
                score -= 25;
            }
            if body == CONTENT_AIR && head == CONTENT_AIR {
                score += 8;
            }
        }
        if score > best.0 {
            best = (score, d);
        }
    }
    best.1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EngineSettings;
    use crate::physics::collision::GameMode;
    use crate::physics::raycast::{held_item_is_empty_bucket, try_interact_or_place, DigController};
    use crate::registry::{CraftRecipe, DrawType, ItemDef, NodeDef, ToolGroupCap};
    use crate::world::mapgen::{
        chunk_y_range_for, find_spawn_pos, generate_chunk_for_world, MapgenPalette,
    };
    use crate::world::nether::{ensure_nether_columns, NetherPalette};
    use crate::world::portal::{find_or_build_destination_portal, PortalAxis};
    use crate::world::{scale_portal_coords, WorldGenType};

    #[test]
    fn parse_autotest_args_handles_valid_and_invalid_flags() {
        let none_args = vec!["cubex-rs".into(), "--dedicated".into()];
        assert!(parse_autotest_args(&none_args).unwrap().is_none());

        let default_args = vec![
            "cubex-rs".into(),
            "--autotest".into(),
            "survival_nether".into(),
        ];
        let cfg = parse_autotest_args(&default_args).unwrap().unwrap();
        assert_eq!(cfg.mode, AutoTestMode::SurvivalNether);
        assert_eq!(cfg.out_dir, PathBuf::from(DEFAULT_AUTOTEST_OUT_DIR));

        let custom_args = vec![
            "cubex-rs".into(),
            "--autotest=survival_nether".into(),
            "--autotest-out".into(),
            "custom_report_dir".into(),
        ];
        let cfg2 = parse_autotest_args(&custom_args).unwrap().unwrap();
        assert_eq!(cfg2.mode, AutoTestMode::SurvivalNether);
        assert_eq!(cfg2.out_dir, PathBuf::from("custom_report_dir"));

        let bad_args = vec!["cubex-rs".into(), "--autotest".into(), "unknown_mode".into()];
        assert!(parse_autotest_args(&bad_args).is_err());
    }

    #[test]
    fn report_serializes_required_fields() {
        let rep = AutoTestReport {
            passed: true,
            mode: "survival_nether".into(),
            seed: SURVIVAL_NETHER_SEED,
            stages_completed: 5,
            final_stage: "Ignite & Enter Nether".into(),
            final_dimension: "Nether".into(),
            final_hp: 20,
            final_hunger: 18,
            avg_fps: 112.4,
            elapsed_secs: 42.5,
            screenshot: "autotest_report/survival_nether.png".into(),
            failure_reason: None,
        };
        let json_str = serde_json::to_string_pretty(&rep).unwrap();
        let val: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(val["passed"], true);
        assert_eq!(val["seed"], SURVIVAL_NETHER_SEED);
        assert_eq!(val["final_hp"], 20);
        assert_eq!(val["final_dimension"], "Nether");
        assert!(val["avg_fps"].as_f64().unwrap() > 100.0);
    }

    fn build_full_survival_test_registry() -> EngineRegistries {
        let mut reg = EngineRegistries::new();
        for (name, dt, walkable, pointable, groups) in [
            ("mcl_core:bedrock", DrawType::Normal, true, true, vec![]),
            (
                "mcl_core:stone",
                DrawType::Normal,
                true,
                true,
                vec![("pickaxey", 1), ("cracky", 3)],
            ),
            (
                "mcl_core:cobble",
                DrawType::Normal,
                true,
                true,
                vec![("pickaxey", 1), ("cracky", 3)],
            ),
            (
                "mcl_deepslate:deepslate",
                DrawType::Normal,
                true,
                true,
                vec![("pickaxey", 1), ("cracky", 2)],
            ),
            (
                "mcl_core:dirt",
                DrawType::Normal,
                true,
                true,
                vec![("handy", 1), ("crumbly", 3)],
            ),
            (
                "mcl_core:dirt_with_grass",
                DrawType::Normal,
                true,
                true,
                vec![("handy", 1), ("crumbly", 3)],
            ),
            (
                "mcl_core:sand",
                DrawType::Normal,
                true,
                true,
                vec![("handy", 1), ("crumbly", 3)],
            ),
            (
                "mcl_core:gravel",
                DrawType::Normal,
                true,
                true,
                vec![("handy", 1), ("crumbly", 3)],
            ),
            (
                "mcl_core:stone_with_coal",
                DrawType::Normal,
                true,
                true,
                vec![("pickaxey", 1), ("cracky", 3)],
            ),
            (
                "mcl_core:stone_with_iron",
                DrawType::Normal,
                true,
                true,
                vec![("pickaxey", 3), ("cracky", 2)],
            ),
            (
                "mcl_trees:tree_oak",
                DrawType::Normal,
                true,
                true,
                vec![("handy", 1), ("choppy", 3)],
            ),
            (
                "mcl_trees:leaves_oak",
                DrawType::AllfacesOptional,
                true,
                true,
                vec![("handy", 1), ("snappy", 3)],
            ),
            (
                "mcl_trees:wood_oak",
                DrawType::Normal,
                true,
                true,
                vec![("handy", 1), ("choppy", 3), ("wood", 1)],
            ),
            (
                "mcl_crafting_table:crafting_table",
                DrawType::Normal,
                true,
                true,
                vec![("handy", 1), ("choppy", 3)],
            ),
            (
                "mcl_furnaces:furnace",
                DrawType::Normal,
                true,
                true,
                vec![("pickaxey", 1), ("cracky", 3)],
            ),
            (
                "mcl_core:obsidian",
                DrawType::Normal,
                true,
                true,
                vec![("pickaxey", 5)],
            ),
            (
                "mcl_nether:netherrack",
                DrawType::Normal,
                true,
                true,
                vec![("pickaxey", 1)],
            ),
            (
                "mcl_nether:nether_brick",
                DrawType::Normal,
                true,
                true,
                vec![("pickaxey", 1)],
            ),
            (
                "mcl_nether:bedrock",
                DrawType::Normal,
                true,
                true,
                vec![],
            ),
        ] {
            let mut gm = std::collections::HashMap::new();
            for (k, v) in groups {
                gm.insert(k.to_string(), v);
            }
            let drop_item = match name {
                "mcl_core:stone" => crate::registry::NodeDrop::Single("mcl_core:cobble".into()),
                "mcl_core:stone_with_coal" => {
                    crate::registry::NodeDrop::Single("mcl_core:coal_lump".into())
                }
                "mcl_core:stone_with_iron" => {
                    crate::registry::NodeDrop::Single("mcl_raw_ores:raw_iron".into())
                }
                "mcl_core:gravel" => crate::registry::NodeDrop::Single("mcl_core:flint".into()),
                _ => crate::registry::NodeDrop::Single(name.into()),
            };
            reg.register_node(NodeDef {
                name: name.into(),
                drawtype: dt,
                walkable,
                pointable,
                diggable: name != "mcl_core:bedrock",
                groups: gm,
                drop: drop_item,
                ..Default::default()
            });
        }

        for (liq, ls) in [("mcl_core:water_source", 0), ("mcl_core:lava_source", 15)] {
            reg.register_node(NodeDef {
                name: liq.into(),
                drawtype: DrawType::Liquid,
                walkable: false,
                pointable: false,
                buildable_to: true,
                diggable: false,
                light_source: ls,
                ..Default::default()
            });
        }

        reg.register_node(NodeDef {
            name: crate::registry::NETHER_PORTAL_NODE_ID.into(),
            drawtype: DrawType::NodeBox,
            walkable: false,
            pointable: true,
            diggable: false,
            light_source: 11,
            ..Default::default()
        });

        for item in [
            "mcl_core:stick",
            "mcl_core:coal_lump",
            "mcl_raw_ores:raw_iron",
            "mcl_core:iron_ingot",
            "mcl_core:flint",
        ] {
            reg.register_item(ItemDef {
                name: item.into(),
                stack_max: 64,
                ..Default::default()
            });
        }
        for tool in [
            "mcl_buckets:bucket_empty",
            "mcl_buckets:bucket_water",
            "mcl_buckets:bucket_lava",
            "mcl_fire:flint_and_steel",
        ] {
            reg.register_item(ItemDef {
                name: tool.into(),
                item_type: if tool.contains("flint_and_steel") {
                    "tool".into()
                } else {
                    "craft".into()
                },
                stack_max: 1,
                ..Default::default()
            });
        }
        reg.register_item(ItemDef {
            name: "mcl_meshhand:hand_surv".into(),
            item_type: "none".into(),
            stack_max: 1,
            tool_capabilities: Some(crate::registry::ToolCapabilities {
                groupcaps: [
                    (
                        "handy".into(),
                        ToolGroupCap {
                            maxlevel: 1,
                            uses: 0,
                            times: [(1, 0.40), (2, 0.30), (3, 0.20)].into_iter().collect(),
                        },
                    ),
                    (
                        "choppy".into(),
                        ToolGroupCap {
                            maxlevel: 1,
                            uses: 0,
                            times: [(1, 0.45), (2, 0.35), (3, 0.25)].into_iter().collect(),
                        },
                    ),
                    (
                        "crumbly".into(),
                        ToolGroupCap {
                            maxlevel: 1,
                            uses: 0,
                            times: [(1, 0.35), (2, 0.25), (3, 0.15)].into_iter().collect(),
                        },
                    ),
                    (
                        "snappy".into(),
                        ToolGroupCap {
                            maxlevel: 1,
                            uses: 0,
                            times: [(1, 0.25), (2, 0.20), (3, 0.15)].into_iter().collect(),
                        },
                    ),
                ]
                .into_iter()
                .collect(),
                ..Default::default()
            }),
            ..Default::default()
        });
        reg.register_item(ItemDef {
            name: "mcl_tools:pick_wood".into(),
            item_type: "tool".into(),
            stack_max: 1,
            tool_capabilities: Some(crate::registry::ToolCapabilities {
                groupcaps: [(
                    "pickaxey".into(),
                    ToolGroupCap {
                        maxlevel: 1,
                        uses: 59,
                        times: [(1, 0.25), (2, 0.35), (3, 0.45)].into_iter().collect(),
                    },
                )]
                .into_iter()
                .collect(),
                ..Default::default()
            }),
            ..Default::default()
        });
        reg.register_item(ItemDef {
            name: "mcl_tools:pick_stone".into(),
            item_type: "tool".into(),
            stack_max: 1,
            tool_capabilities: Some(crate::registry::ToolCapabilities {
                groupcaps: [(
                    "pickaxey".into(),
                    ToolGroupCap {
                        maxlevel: 3,
                        uses: 131,
                        times: [(1, 0.15), (2, 0.20), (3, 0.25)].into_iter().collect(),
                    },
                )]
                .into_iter()
                .collect(),
                ..Default::default()
            }),
            ..Default::default()
        });

        reg.crafts.extend([
            CraftRecipe {
                recipe_type: "shapeless".into(),
                output: "mcl_trees:wood_oak 4".into(),
                recipe: vec![vec!["mcl_trees:tree_oak".into()]],
                replacements: Vec::new(),
                cooktime: 0.0,
                burntime: 0.0,
            },
            CraftRecipe {
                recipe_type: "shaped".into(),
                output: "mcl_core:stick 4".into(),
                recipe: vec![
                    vec!["mcl_trees:wood_oak".into()],
                    vec!["mcl_trees:wood_oak".into()],
                ],
                replacements: Vec::new(),
                cooktime: 0.0,
                burntime: 0.0,
            },
            CraftRecipe {
                recipe_type: "shaped".into(),
                output: "mcl_crafting_table:crafting_table 1".into(),
                recipe: vec![
                    vec!["mcl_trees:wood_oak".into(), "mcl_trees:wood_oak".into()],
                    vec!["mcl_trees:wood_oak".into(), "mcl_trees:wood_oak".into()],
                ],
                replacements: Vec::new(),
                cooktime: 0.0,
                burntime: 0.0,
            },
            CraftRecipe {
                recipe_type: "shaped".into(),
                output: "mcl_tools:pick_wood 1".into(),
                recipe: vec![
                    vec![
                        "mcl_trees:wood_oak".into(),
                        "mcl_trees:wood_oak".into(),
                        "mcl_trees:wood_oak".into(),
                    ],
                    vec!["".into(), "mcl_core:stick".into(), "".into()],
                    vec!["".into(), "mcl_core:stick".into(), "".into()],
                ],
                replacements: Vec::new(),
                cooktime: 0.0,
                burntime: 0.0,
            },
            CraftRecipe {
                recipe_type: "shaped".into(),
                output: "mcl_tools:pick_stone 1".into(),
                recipe: vec![
                    vec![
                        "mcl_core:cobble".into(),
                        "mcl_core:cobble".into(),
                        "mcl_core:cobble".into(),
                    ],
                    vec!["".into(), "mcl_core:stick".into(), "".into()],
                    vec!["".into(), "mcl_core:stick".into(), "".into()],
                ],
                replacements: Vec::new(),
                cooktime: 0.0,
                burntime: 0.0,
            },
            CraftRecipe {
                recipe_type: "shaped".into(),
                output: "mcl_furnaces:furnace 1".into(),
                recipe: vec![
                    vec![
                        "mcl_core:cobble".into(),
                        "mcl_core:cobble".into(),
                        "mcl_core:cobble".into(),
                    ],
                    vec!["mcl_core:cobble".into(), "".into(), "mcl_core:cobble".into()],
                    vec![
                        "mcl_core:cobble".into(),
                        "mcl_core:cobble".into(),
                        "mcl_core:cobble".into(),
                    ],
                ],
                replacements: Vec::new(),
                cooktime: 0.0,
                burntime: 0.0,
            },
            CraftRecipe {
                recipe_type: "shaped".into(),
                output: "mcl_buckets:bucket_empty 1".into(),
                recipe: vec![
                    vec![
                        "mcl_core:iron_ingot".into(),
                        "".into(),
                        "mcl_core:iron_ingot".into(),
                    ],
                    vec!["".into(), "mcl_core:iron_ingot".into(), "".into()],
                ],
                replacements: Vec::new(),
                cooktime: 0.0,
                burntime: 0.0,
            },
            CraftRecipe {
                recipe_type: "shapeless".into(),
                output: "mcl_fire:flint_and_steel 1".into(),
                recipe: vec![vec!["mcl_core:iron_ingot".into(), "mcl_core:flint".into()]],
                replacements: Vec::new(),
                cooktime: 0.0,
                burntime: 0.0,
            },
        ]);
        reg
    }

    #[test]
    fn overworld_seed_133742_has_trees_ores_gravel_water_and_lava() {
        let reg = build_full_survival_test_registry();
        let pal = MapgenPalette::from_registries_with_game_dir(&reg, None);
        let dir = std::env::temp_dir().join("cubex_rs_seed_133742_res_test");
        let _ = std::fs::remove_dir_all(&dir);
        let world = Arc::new(WorldStore::new(&dir, SURVIVAL_NETHER_SEED));

        for cz in -5..=5 {
            for cx in -5..=5 {
                for cy in chunk_y_range_for(WorldGenType::Default) {
                    world.insert_chunk(generate_chunk_for_world(
                        IVec3::new(cx, cy, cz),
                        SURVIVAL_NETHER_SEED,
                        WorldGenType::Default,
                        &pal,
                        &reg,
                    ));
                }
            }
        }
        let sp = find_spawn_pos(SURVIVAL_NETHER_SEED, WorldGenType::Default).as_vec3();
        let bl = HashSet::new();

        assert!(
            find_nearest_node(&world, &reg, sp, 5, SEA_LEVEL, 108, &bl, |n, _| is_log_item_name(n), true).is_some(),
            "Seed 133742 must have reachable trees near spawn"
        );
        assert!(
            find_nearest_node(&world, &reg, sp, 5, 20, 80, &bl, |n, _| is_iron_ore_name(n), false).is_some(),
            "Seed 133742 must have iron ore near spawn"
        );
        assert!(
            find_nearest_node(&world, &reg, sp, 5, 20, 80, &bl, |n, _| is_coal_ore_name(n), false).is_some(),
            "Seed 133742 must have coal ore near spawn"
        );
        assert!(
            find_nearest_node(&world, &reg, sp, 5, 20, 80, &bl, |n, _| n == "mcl_core:gravel", false).is_some(),
            "Seed 133742 must have gravel near spawn"
        );
        assert!(
            find_nearest_node(&world, &reg, sp, 5, SEA_LEVEL - 4, 86, &bl, |n, _| is_water_source_item(n), true).is_some(),
            "Seed 133742 must have water near spawn"
        );
        assert!(
            find_surface_lava_pool(&world, &reg, sp, &bl).is_some(),
            "Seed 133742 must have a surface lava pool within loaded area"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bucket_water_on_lava_casts_obsidian() {
        let reg = build_full_survival_test_registry();
        let dir = std::env::temp_dir().join("cubex_rs_bucket_cast_obsidian_test");
        let _ = std::fs::remove_dir_all(&dir);
        let world = WorldStore::new(&dir, 7);

        let stone_id = reg.get_content_id("mcl_core:stone");
        let lava_id = reg.get_content_id("mcl_core:lava_source");
        let obs_id = reg.get_content_id("mcl_core:obsidian");
        world.set_node(IVec3::new(4, 10, 4), stone_id, 15, 0);
        world.set_node(IVec3::new(4, 11, 4), lava_id, 15 << 4 | 15, 0);

        let ps = Arc::new(RwLock::new(ObjectState::new_player("BucketTester")));
        {
            let p = ps.read();
            let mut m = p.meta.write();
            m.inventories.get_mut("main").unwrap()[0] =
                ItemStackData::from_spec("mcl_buckets:bucket_empty 1");
        }
        let mut pc = PlayerController::default();
        pc.gamemode = GameMode::Survival;
        pc.pos = Vec3::new(2.0, 11.0, 4.5);

        // 1. Raycast with liquids_pointable=true onto lava_source -> scoop into bucket_lava
        let hit_lava = raycast_voxel_ex(
            Vec3::new(2.0, 12.6, 4.5),
            Vec3::new(2.5, -1.1, 0.0).normalize(),
            5.0,
            true,
            &world,
            &reg,
        )
        .unwrap();
        assert_eq!(hit_lava.under, IVec3::new(4, 11, 4));
        let pr1 = try_interact_or_place(&hit_lava, &world, &reg, &pc, &ps, None);
        assert!(pr1.placed);
        assert_eq!(
            ps.read().meta.read().inventories["main"][0].name,
            "mcl_buckets:bucket_lava"
        );
        assert_eq!(world.get_node(IVec3::new(4, 11, 4)).0, CONTENT_AIR);

        // 2. Put lava back and pour bucket_water onto its -X face -> cools immediately to obsidian
        world.set_node(IVec3::new(4, 11, 4), lava_id, 15 << 4 | 15, 0);
        ps.read().meta.write().inventories.get_mut("main").unwrap()[0] =
            ItemStackData::from_spec("mcl_buckets:bucket_water 1");

        let hit_lava2 = raycast_voxel_ex(
            Vec3::new(2.0, 11.5, 4.5),
            Vec3::X,
            5.0,
            true,
            &world,
            &reg,
        )
        .unwrap();
        assert_eq!(hit_lava2.under, IVec3::new(4, 11, 4));
        assert_eq!(hit_lava2.above, IVec3::new(3, 11, 4));
        let pr2 = try_interact_or_place(&hit_lava2, &world, &reg, &pc, &ps, None);
        assert!(pr2.placed);
        assert_eq!(world.get_node(IVec3::new(4, 11, 4)).0, obs_id);
        assert_eq!(
            ps.read().meta.read().inventories["main"][0].name,
            "mcl_buckets:bucket_empty"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn survival_nether_bot_completes_all_five_stages_deterministically() {
        let reg = build_full_survival_test_registry();
        let pal = MapgenPalette::from_registries_with_game_dir(&reg, None);
        let n_pal = NetherPalette::from_registries(&reg);
        let dir = std::env::temp_dir().join("cubex_rs_autotest_5stage_sim");
        let _ = std::fs::remove_dir_all(&dir);

        let ow = Arc::new(WorldStore::new(&dir.join("ow"), SURVIVAL_NETHER_SEED));
        let nw = Arc::new(WorldStore::new_for_dimension(
            &dir.join("nw"),
            SURVIVAL_NETHER_SEED,
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
                        SURVIVAL_NETHER_SEED,
                        WorldGenType::Default,
                        &pal,
                        &reg,
                    ));
                }
            }
        }

        let sp = find_spawn_pos(SURVIVAL_NETHER_SEED, WorldGenType::Default);
        let mut player = PlayerController::default();
        player.gamemode = GameMode::Survival;
        player.pos = Vec3::new(sp.x as f32 + 0.5, sp.y as f32, sp.z as f32 + 0.5);
        let ps = Arc::new(RwLock::new(ObjectState::new_player("AutoTester")));
        let settings = EngineSettings::default();
        let mut digger = DigController::new();

        let cfg = AutoTestConfig {
            mode: AutoTestMode::SurvivalNether,
            out_dir: dir.join("out"),
        };
        let mut bot = SurvivalNetherBot::new(cfg, player.pos);

        let dt = 0.05f32;
        let mut final_rep: Option<AutoTestReport> = None;
        for _tick in 0..9700 {
            let cur_world = if player.dimension.is_nether() { &nw } else { &ow };
            // Dynamically generate chunks if bot wanders outward
            if !player.dimension.is_nether() {
                let pcx = (player.pos.x / 16.0).floor() as i32;
                let pcz = (player.pos.z / 16.0).floor() as i32;
                for dz in -3..=3 {
                    for dx in -3..=3 {
                        if ow_cols.insert((pcx + dx, pcz + dz)) {
                            for cy in chunk_y_range_for(WorldGenType::Default) {
                                ow.insert_chunk(generate_chunk_for_world(
                                    IVec3::new(pcx + dx, cy, pcz + dz),
                                    SURVIVAL_NETHER_SEED,
                                    WorldGenType::Default,
                                    &pal,
                                    &reg,
                                ));
                            }
                        }
                    }
                }
            }

            let out = bot.tick(dt, 60.0, &mut player, &ps, cur_world, &reg, None, false);
            if let Some(rep) = out.finished {
                final_rep = Some(rep);
                break;
            }

            player.step(dt, &out.move_input, cur_world, &reg, &settings);

            // Simulate portal charge & teleportation identically to `tick_player_portal`
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
            if is_nether_portal_node(cur_world.get_node(p_feet).0, &reg)
                || is_nether_portal_node(cur_world.get_node(p_eye).0, &reg)
            {
                player.portal_charge = (player.portal_charge + dt / 3.0).min(1.0);
                if player.portal_charge >= 1.0 && !player.dimension.is_nether() {
                    let scaled = scale_portal_coords(p_feet, DimensionId::Overworld, DimensionId::Nether);
                    let _ = ensure_nether_columns(
                        &nw,
                        scaled.x.div_euclid(16),
                        scaled.z.div_euclid(16),
                        1,
                        SURVIVAL_NETHER_SEED,
                        &n_pal,
                        &reg,
                    );
                    let stand = find_or_build_destination_portal(&nw, &reg, scaled, PortalAxis::X);
                    player.dimension = DimensionId::Nether;
                    player.pos = Vec3::new(stand.x as f32 + 0.5, stand.y as f32 + 0.05, stand.z as f32 + 0.5);
                    player.vel = Vec3::ZERO;
                    player.portal_charge = 0.0;
                    player.reset_fall_state();
                }
            }

            let point_liq = out.force_liquids_pointable || held_item_is_empty_bucket(&player, &ps);
            let hit = raycast_voxel_ex(
                player.eye_pos(),
                player.look_dir(),
                5.5,
                point_liq,
                cur_world,
                &reg,
            );
            let _ = digger.update_with_physical_drops(
                dt,
                out.mine_held,
                hit.as_ref(),
                cur_world,
                &reg,
                &player,
                &ps,
                None,
                false,
            );
            if out.place_just {
                if let Some(ref h) = hit {
                    let _ = try_interact_or_place(h, cur_world, &reg, &player, &ps, None);
                }
            }
        }

        let rep = final_rep.unwrap_or_else(|| {
            let hud = bot.hud_status();
            let px = player.pos.x.floor() as i32;
            let py = player.pos.y.floor() as i32;
            let pz = player.pos.z.floor() as i32;
            let mut nearby = Vec::new();
            for dy in 0..=2 {
                for dz in -1..=1 {
                    for dx in -1..=1 {
                        let p = IVec3::new(px + dx, py + dy, pz + dz);
                        let id = ow.get_node(p).0;
                        if id != CONTENT_AIR {
                            nearby.push((p, reg.get_name_from_content_id(id)));
                        }
                    }
                }
            }
            panic!(
                "AutoBot timed out: stage={}/5 detail={} pos={:?} look={:?} on_ground={} in_liq={} target={:?} unstuck={:?} nearby={:?}",
                hud.stage_num,
                hud.detail,
                player.pos,
                player.look_dir(),
                player.on_ground,
                player.in_liquid,
                bot.current_target,
                bot.unstuck_dig_target,
                nearby,
            );
        });
        assert!(
            rep.passed,
            "AutoBot failed at stage {} ({}): {:?}",
            rep.stages_completed,
            rep.final_stage,
            rep.failure_reason
        );
        assert_eq!(rep.stages_completed, 5);
        assert_eq!(rep.final_dimension, DimensionId::Nether.label());
        assert!(rep.final_hp > 0);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
