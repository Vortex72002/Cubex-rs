use glam::{IVec3, Vec3};

use crate::registry::{DrawType, EngineRegistries, CONTENT_AIR, CONTENT_IGNORE};
use crate::render::entity_model::EntityRenderSnapshot;
use crate::world::dimension::DimensionId;
use crate::world::nether::{sample_nether_biome, NetherBiome, NETHER_LAVA_LEVEL};
use crate::world::structures::{structures_overlapping_chunk, StructureKind};
use crate::world::{WorldGenType, WorldStore};

pub const MAX_ACTIVE_MOBS_PER_DIM: usize = 24;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NativeMobKind {
    ZombiePiglin,
    Blaze,
    Ghast,
    WitherSkeleton,
    MagmaCube,
    Guardian,
}

impl NativeMobKind {
    pub fn dimension(self) -> DimensionId {
        match self {
            Self::Guardian => DimensionId::Overworld,
            _ => DimensionId::Nether,
        }
    }

    pub fn entity_name(self) -> &'static str {
        match self {
            Self::ZombiePiglin => "mobs_mc:pigman",
            Self::Blaze => "mobs_mc:blaze",
            Self::Ghast => "mobs_mc:ghast",
            Self::WitherSkeleton => "mobs_mc:witherskeleton",
            Self::MagmaCube => "mobs_mc:magma_cube",
            Self::Guardian => "mobs_mc:guardian",
        }
    }

    pub fn max_hp(self, size: u8) -> i32 {
        match self {
            Self::ZombiePiglin => 20,
            Self::Blaze => 20,
            Self::Ghast => 10,
            Self::WitherSkeleton => 20,
            Self::MagmaCube => match size {
                3 => 16,
                2 => 8,
                _ => 4,
            },
            Self::Guardian => 30,
        }
    }

    pub fn collisionbox(self, size: u8) -> [f32; 6] {
        match self {
            Self::ZombiePiglin => [-0.32, 0.0, -0.32, 0.32, 1.95, 0.32],
            Self::Blaze => [-0.38, 0.0, -0.38, 0.38, 1.80, 0.38],
            Self::Ghast => [-1.9, 0.0, -1.9, 1.9, 4.0, 1.9],
            Self::WitherSkeleton => [-0.38, 0.0, -0.38, 0.38, 2.40, 0.38],
            Self::MagmaCube => {
                let half = match size {
                    3 => 0.85,
                    2 => 0.55,
                    _ => 0.32,
                };
                [-half, 0.0, -half, half, half * 2.0, half]
            }
            Self::Guardian => [-0.45, 0.0, -0.45, 0.45, 0.90, 0.45],
        }
    }

    pub fn texture_name(self, firing: bool) -> &'static str {
        match self {
            Self::ZombiePiglin => "extra_mobs_zombified_piglin.png",
            Self::Blaze => "mobs_mc_blaze.png",
            Self::Ghast => {
                if firing {
                    "mobs_mc_ghast_firing.png"
                } else {
                    "mobs_mc_ghast.png"
                }
            }
            Self::WitherSkeleton => "mobs_mc_wither_skeleton.png",
            Self::MagmaCube => "mobs_mc_magmacube.png",
            Self::Guardian => "mobs_mc_guardian.png",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MobAiState {
    Idle,
    Wander,
    Aggro,
    Attack,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectileKind {
    BlazeFireball,
    GhastFireball,
}

#[derive(Debug, Clone)]
pub struct NativeMob {
    pub id: u64,
    pub kind: NativeMobKind,
    pub dimension: DimensionId,
    pub pos: Vec3,
    pub vel: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub hp: i32,
    pub max_hp: i32,
    pub size: u8,
    pub ai_state: MobAiState,
    pub state_timer: f32,
    pub wander_dir: Vec3,
    pub attack_cooldown: f32,
    pub volley_remaining: u8,
    pub volley_timer: f32,
    pub laser_charge: f32,
    pub laser_target_player: Option<usize>,
    pub aggro_target: Option<usize>,
    pub aggro_timer: f32,
    pub hurt_flash: f32,
    pub walk_phase: f32,
    pub walk_blend: f32,
    pub anim_time: f32,
    pub on_ground: bool,
    pub firing_visual: bool,
    pub fire_windup_timer: f32,
}

impl NativeMob {
    pub fn half_extents(&self) -> (f32, f32) {
        let b = self.kind.collisionbox(self.size);
        ((b[3] - b[0]) * 0.5, b[4] - b[1])
    }

    pub fn eye_pos(&self) -> Vec3 {
        let (_, h) = self.half_extents();
        self.pos + Vec3::new(0.0, h * 0.72, 0.0)
    }

    pub fn world_aabb(&self) -> (Vec3, Vec3) {
        let b = self.kind.collisionbox(self.size);
        (
            self.pos + Vec3::new(b[0], b[1], b[2]),
            self.pos + Vec3::new(b[3], b[4], b[5]),
        )
    }
}

#[derive(Debug, Clone)]
pub struct MobProjectile {
    pub id: u64,
    pub kind: ProjectileKind,
    pub dimension: DimensionId,
    pub pos: Vec3,
    pub vel: Vec3,
    pub life: f32,
    pub owner_mob_id: u64,
    pub deflected_by_player: bool,
    pub anim_time: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct GuardianLaserBeam {
    pub dimension: DimensionId,
    pub start: Vec3,
    pub end: Vec3,
    pub progress: f32,
}

#[derive(Debug, Clone)]
pub struct MobPlayerInput {
    pub slot: usize,
    pub dimension: DimensionId,
    pub pos: Vec3,
    pub eye_pos: Vec3,
    pub alive: bool,
    pub survival: bool,
}

#[derive(Debug, Clone)]
pub struct MobPlayerDamage {
    pub player_slot: usize,
    pub damage: i32,
    pub knockback: Vec3,
    pub apply_wither_secs: f32,
    pub death_cause: &'static str,
}

#[derive(Debug, Clone, Copy)]
pub enum MobParticleKind {
    Flame,
    Smoke,
    Explosion,
    LaserHit,
}

#[derive(Debug, Clone)]
pub struct MobParticleEvent {
    pub dimension: DimensionId,
    pub pos: Vec3,
    pub kind: MobParticleKind,
    pub count: usize,
}

#[derive(Debug, Default)]
pub struct MobTickEvents {
    pub player_damages: Vec<MobPlayerDamage>,
    pub sounds: Vec<(DimensionId, Vec3, &'static str)>,
    pub particles: Vec<MobParticleEvent>,
    pub modified_blocks: Vec<(DimensionId, IVec3)>,
    pub mob_drops: Vec<(DimensionId, Vec3, Vec<String>)>,
}

#[derive(Debug, Clone)]
pub enum MeleeNativeResult {
    DeflectedFireball(Vec3),
    HitMob {
        mob_id: u64,
        kind: NativeMobKind,
        pos: Vec3,
        killed: bool,
        drops: Vec<String>,
        sound: &'static str,
    },
}

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

pub struct MobManager {
    next_id: u64,
    pub mobs: Vec<NativeMob>,
    pub projectiles: Vec<MobProjectile>,
    spawn_timer: f32,
    rng_state: u64,
}

impl Default for MobManager {
    fn default() -> Self {
        Self::new(0xC0B3_1234_5678_9ABC)
    }
}

impl MobManager {
    pub fn new(seed: u64) -> Self {
        Self {
            next_id: 1,
            mobs: Vec::new(),
            projectiles: Vec::new(),
            spawn_timer: 0.4,
            rng_state: splitmix64(seed | 1),
        }
    }

    pub fn clear(&mut self) {
        self.mobs.clear();
        self.projectiles.clear();
        self.spawn_timer = 0.4;
    }

    fn next_rand_u64(&mut self) -> u64 {
        self.rng_state = splitmix64(self.rng_state);
        self.rng_state
    }

    fn rand_f32(&mut self) -> f32 {
        (self.next_rand_u64() & 0x00FF_FFFF) as f32 / 16_777_215.0
    }

    pub fn active_mob_count(&self, dim: DimensionId) -> usize {
        self.mobs.iter().filter(|m| m.dimension == dim).count()
    }

    pub fn spawn_mob(
        &mut self,
        kind: NativeMobKind,
        dim: DimensionId,
        pos: Vec3,
        size: u8,
    ) -> u64 {
        let clamped_size = match kind {
            NativeMobKind::MagmaCube => size.clamp(1, 3),
            _ => 1,
        };
        let hp = kind.max_hp(clamped_size);
        let id = self.next_id;
        self.next_id += 1;
        let yaw = self.rand_f32() * std::f32::consts::TAU;
        let state_timer = 1.5 + self.rand_f32() * 2.0;
        let attack_cooldown = 0.6 + self.rand_f32() * 0.8;
        let walk_phase = self.rand_f32() * 6.0;
        let anim_time = self.rand_f32() * 10.0;
        self.mobs.push(NativeMob {
            id,
            kind,
            dimension: dim,
            pos,
            vel: Vec3::ZERO,
            yaw,
            pitch: 0.0,
            hp,
            max_hp: hp,
            size: clamped_size,
            ai_state: MobAiState::Wander,
            state_timer,
            wander_dir: Vec3::new(yaw.sin(), 0.0, yaw.cos()),
            attack_cooldown,
            volley_remaining: 0,
            volley_timer: 0.0,
            laser_charge: 0.0,
            laser_target_player: None,
            aggro_target: None,
            aggro_timer: 0.0,
            hurt_flash: 0.0,
            walk_phase,
            walk_blend: 0.0,
            anim_time,
            on_ground: false,
            firing_visual: false,
            fire_windup_timer: 0.0,
        });
        id
    }

    pub fn spawn_projectile(
        &mut self,
        kind: ProjectileKind,
        dim: DimensionId,
        pos: Vec3,
        vel: Vec3,
        owner_mob_id: u64,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.projectiles.push(MobProjectile {
            id,
            kind,
            dimension: dim,
            pos,
            vel,
            life: 6.0,
            owner_mob_id,
            deflected_by_player: false,
            anim_time: 0.0,
        });
        id
    }

    #[inline]
    fn is_solid_at(world: &WorldStore, reg: &EngineRegistries, wp: IVec3) -> bool {
        let (id, _, _) = world.get_node(wp);
        if id == CONTENT_AIR || id == CONTENT_IGNORE {
            return false;
        }
        reg.get_node_def(id)
            .is_some_and(|d| d.walkable && d.drawtype != DrawType::Liquid && d.drawtype != DrawType::FlowingLiquid)
    }

    #[inline]
    fn is_water_at(world: &WorldStore, reg: &EngineRegistries, wp: IVec3) -> bool {
        let (id, _, _) = world.get_node(wp);
        if id == CONTENT_AIR || id == CONTENT_IGNORE {
            return false;
        }
        reg.get_node_def(id).is_some_and(|d| {
            matches!(d.drawtype, DrawType::Liquid | DrawType::FlowingLiquid)
                && !d.name.to_ascii_lowercase().contains("lava")
        })
    }

    #[inline]
    fn is_hazard_liquid_at(world: &WorldStore, reg: &EngineRegistries, wp: IVec3) -> bool {
        let (id, _, _) = world.get_node(wp);
        if id == CONTENT_AIR || id == CONTENT_IGNORE {
            return false;
        }
        reg.get_node_def(id).is_some_and(|d| {
            matches!(d.drawtype, DrawType::Liquid | DrawType::FlowingLiquid)
                && d.name.to_ascii_lowercase().contains("lava")
        })
    }

    fn mob_collides_world(
        pos: Vec3,
        kind: NativeMobKind,
        size: u8,
        world: &WorldStore,
        reg: &EngineRegistries,
    ) -> bool {
        let b = kind.collisionbox(size);
        let min_x = (pos.x + b[0] + 0.04).floor() as i32;
        let max_x = (pos.x + b[3] - 0.04).floor() as i32;
        let min_y = (pos.y + b[1] + 0.04).floor() as i32;
        let max_y = (pos.y + b[4] - 0.04).floor() as i32;
        let min_z = (pos.z + b[2] + 0.04).floor() as i32;
        let max_z = (pos.z + b[5] - 0.04).floor() as i32;

        for y in min_y..=max_y {
            for z in min_z..=max_z {
                for x in min_x..=max_x {
                    if Self::is_solid_at(world, reg, IVec3::new(x, y, z)) {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn has_line_of_sight(world: &WorldStore, reg: &EngineRegistries, from: Vec3, to: Vec3) -> bool {
        let diff = to - from;
        let dist = diff.length();
        if dist <= 0.5 {
            return true;
        }
        let dir = diff / dist;
        let steps = (dist * 2.0).ceil() as i32;
        for s in 1..steps {
            let p = from + dir * (s as f32 * 0.5);
            if Self::is_solid_at(world, reg, p.floor().as_ivec3()) {
                return false;
            }
        }
        true
    }

    fn ground_height_below(world: &WorldStore, reg: &EngineRegistries, pos: Vec3, max_drop: i32) -> Option<i32> {
        let bx = pos.x.floor() as i32;
        let bz = pos.z.floor() as i32;
        let top = pos.y.floor() as i32;
        for y in ((top - max_drop).max(1)..=top).rev() {
            if Self::is_solid_at(world, reg, IVec3::new(bx, y, bz)) {
                return Some(y);
            }
        }
        None
    }

    pub fn drops_for_killed(&mut self, kind: NativeMobKind, size: u8, guarantee_rare_head: bool) -> Vec<String> {
        let mut out = Vec::new();
        match kind {
            NativeMobKind::ZombiePiglin => {
                let f_cnt = 1 + (self.next_rand_u64() % 2) as u16;
                let g_cnt = 1 + (self.next_rand_u64() % 2) as u16;
                out.push(format!("mcl_mobitems:rotten_flesh {f_cnt}"));
                out.push(format!("mcl_core:gold_nugget {g_cnt}"));
            }
            NativeMobKind::Blaze => {
                let cnt = 1 + (self.next_rand_u64() % 2) as u16;
                out.push(format!("mcl_mobitems:blaze_rod {cnt}"));
            }
            NativeMobKind::Ghast => {
                out.push("mcl_mobitems:ghast_tear 1".into());
                let gp = 1 + (self.next_rand_u64() % 2) as u16;
                out.push(format!("mcl_mobitems:gunpowder {gp}"));
            }
            NativeMobKind::WitherSkeleton => {
                let coal = 1 + (self.next_rand_u64() % 2) as u16;
                let bone = 1 + (self.next_rand_u64() % 2) as u16;
                out.push(format!("mcl_core:coal_lump {coal}"));
                out.push(format!("mcl_mobitems:bone {bone}"));
                if guarantee_rare_head || (self.next_rand_u64() % 10 == 0) {
                    out.push("mcl_heads:wither_skeleton 1".into());
                }
            }
            NativeMobKind::MagmaCube => {
                if size >= 2 || self.next_rand_u64() % 2 == 0 {
                    out.push("mcl_mobitems:magma_cream 1".into());
                }
            }
            NativeMobKind::Guardian => {
                let cnt = 1 + (self.next_rand_u64() % 2) as u16;
                out.push(format!("mcl_ocean:prismarine_shard {cnt}"));
            }
        }
        out
    }

    pub fn hurt_sound(kind: NativeMobKind) -> &'static str {
        match kind {
            NativeMobKind::ZombiePiglin => "mobs_mc_zombiepig_hurt",
            NativeMobKind::Blaze => "mobs_mc_blaze_hurt",
            NativeMobKind::Ghast => "mobs_mc_ghast_hurt",
            NativeMobKind::WitherSkeleton => "mobs_mc_skeleton_hurt",
            NativeMobKind::MagmaCube => "mobs_mc_magma_cube_small",
            NativeMobKind::Guardian => "mobs_mc_guardian_hurt",
        }
    }

    pub fn death_sound(kind: NativeMobKind) -> &'static str {
        match kind {
            NativeMobKind::ZombiePiglin => "mobs_mc_zombiepig_death",
            NativeMobKind::Blaze => "mobs_mc_blaze_died",
            NativeMobKind::Ghast => "mobs_mc_ghast_dying",
            NativeMobKind::WitherSkeleton => "mobs_mc_skeleton_hurt",
            NativeMobKind::MagmaCube => "mobs_mc_magma_cube_big",
            NativeMobKind::Guardian => "mobs_mc_guardian_death",
        }
    }

    /// Fast non-mutating check to determine whether a melee ray hits any native mob or fireball.
    pub fn has_melee_target(&self, eye: Vec3, dir: Vec3, max_dist: f32, dim: DimensionId) -> bool {
        let dir_n = dir.normalize_or_zero();
        if dir_n.length_squared() < 1e-6 || max_dist <= 0.05 {
            return false;
        }
        let fb_reach = max_dist.max(4.5);
        for proj in &self.projectiles {
            if proj.dimension != dim || proj.kind != ProjectileKind::GhastFireball {
                continue;
            }
            let to_fb = proj.pos - eye;
            let along = to_fb.dot(dir_n);
            if along >= 0.0 && along <= fb_reach {
                let perp_sq = to_fb.length_squared() - along * along;
                if perp_sq <= 0.85 * 0.85 {
                    return true;
                }
            }
        }
        for mob in &self.mobs {
            if mob.dimension != dim {
                continue;
            }
            let (bmin, bmax) = mob.world_aabb();
            if ray_aabb_intersect(eye, dir_n, bmin, bmax, max_dist).is_some() {
                return true;
            }
        }
        false
    }

    /// Melee hit test against active fireballs (deflection) and native mobs in `dim`.
    pub fn try_melee_hit(
        &mut self,
        eye: Vec3,
        dir: Vec3,
        max_dist: f32,
        dim: DimensionId,
        player_slot: usize,
        damage: i32,
    ) -> Option<MeleeNativeResult> {
        let dir_n = dir.normalize_or_zero();
        if dir_n.length_squared() < 1e-6 || max_dist <= 0.05 {
            return None;
        }

        // 1. Check Ghast Fireballs first so melee strikes reliably deflect them along look dir
        let fireball_reach = max_dist.max(4.5);
        let mut best_fb_idx: Option<usize> = None;
        let mut best_fb_t = fireball_reach;
        for (i, proj) in self.projectiles.iter().enumerate() {
            if proj.dimension != dim || proj.kind != ProjectileKind::GhastFireball {
                continue;
            }
            let to_fb = proj.pos - eye;
            let along = to_fb.dot(dir_n);
            if along < 0.0 || along > best_fb_t {
                continue;
            }
            let perp_sq = to_fb.length_squared() - along * along;
            if perp_sq <= 0.85 * 0.85 {
                best_fb_t = along;
                best_fb_idx = Some(i);
            }
        }
        if let Some(fb_idx) = best_fb_idx {
            let proj = &mut self.projectiles[fb_idx];
            proj.vel = dir_n * 20.0;
            proj.deflected_by_player = true;
            proj.life = 5.0;
            return Some(MeleeNativeResult::DeflectedFireball(proj.pos));
        }

        // 2. Raycast active native mob AABBs in `dim`
        let mut best_mob_idx: Option<usize> = None;
        let mut best_t = max_dist;
        for (i, mob) in self.mobs.iter().enumerate() {
            if mob.dimension != dim {
                continue;
            }
            let (bmin, bmax) = mob.world_aabb();
            if let Some(t) = ray_aabb_intersect(eye, dir_n, bmin, bmax, best_t) {
                best_t = t;
                best_mob_idx = Some(i);
            }
        }
        let idx = best_mob_idx?;

        let (kind, pos, mob_id, size, killed) = {
            let mob = &mut self.mobs[idx];
            mob.hp -= damage.max(1);
            mob.hurt_flash = 0.25;
            mob.vel += Vec3::new(dir_n.x * 5.2, 3.6, dir_n.z * 5.2);
            mob.aggro_target = Some(player_slot);
            mob.aggro_timer = 25.0;
            mob.ai_state = MobAiState::Aggro;
            (
                mob.kind,
                mob.pos,
                mob.id,
                mob.size,
                mob.hp <= 0,
            )
        };

        // Zombie Piglin 16-block pack aggro propagation!
        if kind == NativeMobKind::ZombiePiglin {
            for other in &mut self.mobs {
                if other.dimension == dim
                    && other.kind == NativeMobKind::ZombiePiglin
                    && other.pos.distance_squared(pos) <= 16.0 * 16.0
                {
                    other.aggro_target = Some(player_slot);
                    other.aggro_timer = 25.0;
                    other.ai_state = MobAiState::Aggro;
                }
            }
        }

        let drops = if killed {
            let dead = self.mobs.remove(idx);
            if dead.kind == NativeMobKind::MagmaCube && dead.size > 1 {
                let child_size = dead.size - 1;
                for ox in [-0.45f32, 0.45f32] {
                    let cid = self.spawn_mob(
                        NativeMobKind::MagmaCube,
                        dim,
                        dead.pos + Vec3::new(ox, 0.2, 0.0),
                        child_size,
                    );
                    if let Some(child) = self.mobs.iter_mut().find(|m| m.id == cid) {
                        child.vel = Vec3::new(ox * 4.5, 4.8, -ox * 2.0);
                        child.aggro_target = dead.aggro_target;
                        child.aggro_timer = dead.aggro_timer;
                        child.ai_state = MobAiState::Aggro;
                    }
                }
            }
            self.drops_for_killed(dead.kind, size, false)
        } else {
            Vec::new()
        };

        let sound = if killed {
            Self::death_sound(kind)
        } else {
            Self::hurt_sound(kind)
        };

        Some(MeleeNativeResult::HitMob {
            mob_id,
            kind,
            pos,
            killed,
            drops,
            sound,
        })
    }

    pub fn try_spawn_near_players(
        &mut self,
        players: &[MobPlayerInput],
        ow_world: &WorldStore,
        nw_world: &WorldStore,
        reg: &EngineRegistries,
    ) {
        // 1. Nether Mobs
        let nw_players: Vec<&MobPlayerInput> = players
            .iter()
            .filter(|p| p.alive && p.dimension == DimensionId::Nether)
            .collect();
        if !nw_players.is_empty() && self.active_mob_count(DimensionId::Nether) < MAX_ACTIVE_MOBS_PER_DIM {
            let anchor_p = nw_players[(self.next_rand_u64() as usize) % nw_players.len()];
            let ang = self.rand_f32() * std::f32::consts::TAU;
            let dist = 18.0 + self.rand_f32() * 34.0;
            let sx = (anchor_p.pos.x + ang.cos() * dist).floor() as i32;
            let sz = (anchor_p.pos.z + ang.sin() * dist).floor() as i32;
            let cpos = IVec3::new(
                sx.div_euclid(16),
                (anchor_p.pos.y.floor() as i32).div_euclid(16).clamp(1, 6),
                sz.div_euclid(16),
            );
            if nw_world.get_chunk(cpos).is_some() {
                let structs = structures_overlapping_chunk(
                    cpos,
                    DimensionId::Nether,
                    nw_world.seed,
                    WorldGenType::Default,
                );
                let in_fortress = structs
                    .iter()
                    .any(|s| s.kind == StructureKind::NetherFortress);
                let in_bastion = structs
                    .iter()
                    .any(|s| s.kind == StructureKind::BastionRemnant);

                let biome = sample_nether_biome(sx, sz, nw_world.seed);
                let roll = self.next_rand_u64() % 100;
                let (kind, size, want_air_high) = if in_fortress {
                    if roll < 50 {
                        (NativeMobKind::Blaze, 1, false)
                    } else if roll < 90 {
                        (NativeMobKind::WitherSkeleton, 1, false)
                    } else {
                        (NativeMobKind::ZombiePiglin, 1, false)
                    }
                } else if in_bastion {
                    if roll < 45 {
                        (NativeMobKind::MagmaCube, 3, false)
                    } else {
                        (NativeMobKind::ZombiePiglin, 1, false)
                    }
                } else {
                    match biome {
                        NetherBiome::BasaltDeltas => {
                            if roll < 65 {
                                (NativeMobKind::MagmaCube, 2 + (roll as u8 % 2), false)
                            } else {
                                (NativeMobKind::Ghast, 1, true)
                            }
                        }
                        NetherBiome::NetherWastes => {
                            if roll < 55 {
                                (NativeMobKind::ZombiePiglin, 1, false)
                            } else if roll < 78 {
                                (NativeMobKind::MagmaCube, 2, false)
                            } else {
                                (NativeMobKind::Ghast, 1, true)
                            }
                        }
                        NetherBiome::CrimsonForest => {
                            if roll < 82 {
                                (NativeMobKind::ZombiePiglin, 1, false)
                            } else {
                                (NativeMobKind::Ghast, 1, true)
                            }
                        }
                        NetherBiome::WarpedForest | NetherBiome::SoulSandValley => {
                            if roll < 45 {
                                (NativeMobKind::Ghast, 1, true)
                            } else {
                                (NativeMobKind::ZombiePiglin, 1, false)
                            }
                        }
                    }
                };

                if want_air_high {
                    let fy = (anchor_p.pos.y as i32 + 8 + (self.next_rand_u64() % 14) as i32)
                        .clamp(42, 92);
                    let probe = Vec3::new(sx as f32 + 0.5, fy as f32, sz as f32 + 0.5);
                    if !Self::mob_collides_world(probe, kind, size, nw_world, reg) {
                        self.spawn_mob(kind, DimensionId::Nether, probe, size);
                    }
                } else {
                    let y_hi = (anchor_p.pos.y as i32 + 12).min(115);
                    let y_lo = (anchor_p.pos.y as i32 - 14).max(NETHER_LAVA_LEVEL + 1);
                    for wy in (y_lo..=y_hi).rev() {
                        let stand = IVec3::new(sx, wy, sz);
                        if Self::is_solid_at(nw_world, reg, stand)
                            && !Self::is_solid_at(nw_world, reg, stand + IVec3::Y)
                            && !Self::is_solid_at(nw_world, reg, stand + IVec3::Y * 2)
                            && !Self::is_hazard_liquid_at(nw_world, reg, stand + IVec3::Y)
                        {
                            let pos = Vec3::new(sx as f32 + 0.5, (wy + 1) as f32, sz as f32 + 0.5);
                            if !Self::mob_collides_world(pos, kind, size, nw_world, reg) {
                                self.spawn_mob(kind, DimensionId::Nether, pos, size);
                            }
                            break;
                        }
                    }
                }
            }
        }

        // 2. Ocean Monument Guardians (Overworld)
        let ow_players: Vec<&MobPlayerInput> = players
            .iter()
            .filter(|p| p.alive && p.dimension == DimensionId::Overworld)
            .collect();
        if !ow_players.is_empty()
            && self.active_mob_count(DimensionId::Overworld) < MAX_ACTIVE_MOBS_PER_DIM
        {
            let wt = *ow_world.world_type.read();
            for p in &ow_players {
                let cpos = IVec3::new(
                    (p.pos.x.floor() as i32).div_euclid(16),
                    3,
                    (p.pos.z.floor() as i32).div_euclid(16),
                );
                let structs =
                    structures_overlapping_chunk(cpos, DimensionId::Overworld, ow_world.seed, wt);
                for a in structs {
                    if a.kind != StructureKind::OceanMonument {
                        continue;
                    }
                    let ox = a.origin.x + ((self.next_rand_u64() % 36) as i32 - 18);
                    let oz = a.origin.z + ((self.next_rand_u64() % 36) as i32 - 18);
                    let oy = a.origin.y + 4 + (self.next_rand_u64() % 14) as i32;
                    let wp = IVec3::new(ox, oy, oz);
                    if Self::is_water_at(ow_world, reg, wp)
                        && Self::is_water_at(ow_world, reg, wp + IVec3::Y)
                    {
                        let gpos = Vec3::new(ox as f32 + 0.5, oy as f32 + 0.1, oz as f32 + 0.5);
                        if p.pos.distance_squared(gpos) >= 6.0 * 6.0 {
                            self.spawn_mob(
                                NativeMobKind::Guardian,
                                DimensionId::Overworld,
                                gpos,
                                1,
                            );
                            break;
                        }
                    }
                }
            }
        }
    }

    pub fn step(
        &mut self,
        dt: f32,
        players: &[MobPlayerInput],
        ow_world: &WorldStore,
        nw_world: &WorldStore,
        reg: &EngineRegistries,
    ) -> MobTickEvents {
        let dt = dt.clamp(0.0, 0.05);
        let mut events = MobTickEvents::default();

        // Despawn mobs far (> 96 blocks) from every active player in their dimension
        self.mobs.retain(|m| {
            let has_nearby_player = players.iter().any(|p| {
                p.dimension == m.dimension && p.pos.distance_squared(m.pos) <= 96.0 * 96.0
            });
            has_nearby_player
        });

        self.spawn_timer -= dt;
        if self.spawn_timer <= 0.0 {
            self.spawn_timer = 1.1;
            self.try_spawn_near_players(players, ow_world, nw_world, reg);
        }

        // Collect spawnable projectiles from mob AI (avoiding double-mut borrow on self)
        struct PendingShot {
            kind: ProjectileKind,
            dim: DimensionId,
            pos: Vec3,
            vel: Vec3,
            owner: u64,
        }
        let mut pending_shots: Vec<PendingShot> = Vec::new();
        let mut rng_local = self.rng_state;

        for mob in &mut self.mobs {
            let world = if mob.dimension.is_nether() {
                nw_world
            } else {
                ow_world
            };
            mob.anim_time += dt;
            mob.hurt_flash = (mob.hurt_flash - dt).max(0.0);
            mob.attack_cooldown = (mob.attack_cooldown - dt).max(0.0);
            if mob.aggro_timer > 0.0 {
                mob.aggro_timer = (mob.aggro_timer - dt).max(0.0);
                if mob.aggro_timer <= 0.0 {
                    mob.aggro_target = None;
                }
            }

            // Pick closest alive player in same dimension
            let nearest_player = players
                .iter()
                .filter(|p| p.alive && p.dimension == mob.dimension)
                .map(|p| (p.pos.distance_squared(mob.pos), p))
                .min_by(|a, b| a.0.total_cmp(&b.0));

            let active_target: Option<&MobPlayerInput> = match mob.kind {
                NativeMobKind::ZombiePiglin => {
                    mob.aggro_target.and_then(|slot| {
                        players
                            .iter()
                            .find(|p| p.alive && p.dimension == mob.dimension && p.slot == slot)
                    })
                }
                NativeMobKind::Blaze | NativeMobKind::WitherSkeleton => nearest_player
                    .filter(|(d2, _)| *d2 <= 28.0 * 28.0)
                    .map(|(_, p)| p),
                NativeMobKind::MagmaCube => nearest_player
                    .filter(|(d2, _)| *d2 <= 22.0 * 22.0)
                    .map(|(_, p)| p),
                NativeMobKind::Ghast => nearest_player
                    .filter(|(d2, _)| *d2 <= 48.0 * 48.0)
                    .map(|(_, p)| p),
                NativeMobKind::Guardian => nearest_player
                    .filter(|(d2, _)| *d2 <= 18.0 * 18.0)
                    .map(|(_, p)| p),
            };

            let mut desired_horiz = Vec3::ZERO;
            match mob.kind {
                NativeMobKind::ZombiePiglin | NativeMobKind::WitherSkeleton => {
                    if let Some(target) = active_target {
                        let delta = target.pos - mob.pos;
                        let dist = delta.length();
                        mob.yaw = (-delta.x).atan2(delta.z);
                        mob.ai_state = if dist <= 2.35 {
                            MobAiState::Attack
                        } else {
                            MobAiState::Aggro
                        };
                        let speed = if mob.kind == NativeMobKind::ZombiePiglin {
                            4.2
                        } else {
                            3.6
                        };
                        if dist > 1.4 {
                            let flat = Vec3::new(delta.x, 0.0, delta.z).normalize_or_zero();
                            desired_horiz = flat * speed;
                        }
                        if dist <= 2.35 && mob.attack_cooldown <= 0.0 && target.survival {
                            mob.attack_cooldown = 1.05;
                            let flat_kb = Vec3::new(delta.x, 0.0, delta.z).normalize_or_zero() * 5.0;
                            let (dmg, wither, cause) = match mob.kind {
                                NativeMobKind::WitherSkeleton => {
                                    (5, 5.0, "Slain by Wither Skeleton")
                                }
                                _ => (4, 0.0, "Slain by Zombified Piglin"),
                            };
                            events.player_damages.push(MobPlayerDamage {
                                player_slot: target.slot,
                                damage: dmg,
                                knockback: Vec3::new(flat_kb.x, 3.8, flat_kb.z),
                                apply_wither_secs: wither,
                                death_cause: cause,
                            });
                        }
                    } else {
                        mob.state_timer -= dt;
                        if mob.state_timer <= 0.0 {
                            rng_local = splitmix64(rng_local);
                            if matches!(mob.ai_state, MobAiState::Wander) && (rng_local % 3 == 0) {
                                mob.ai_state = MobAiState::Idle;
                                mob.wander_dir = Vec3::ZERO;
                                mob.state_timer = 1.5 + (rng_local % 20) as f32 * 0.1;
                            } else {
                                mob.ai_state = MobAiState::Wander;
                                let ang = (rng_local & 0xFFFF) as f32 / 65535.0 * std::f32::consts::TAU;
                                mob.wander_dir = Vec3::new(-ang.sin(), 0.0, ang.cos());
                                mob.yaw = ang;
                                mob.state_timer = 2.0 + ((rng_local >> 16) % 25) as f32 * 0.1;
                            }
                        }
                        desired_horiz = mob.wander_dir * 1.45;
                    }
                    // Ground gravity & jump over 1-block steps
                    mob.vel.y -= 19.62 * dt;
                    mob.vel.x = desired_horiz.x;
                    mob.vel.z = desired_horiz.z;
                    step_ground_mob(mob, dt, world, reg);
                }
                NativeMobKind::Blaze => {
                    mob.firing_visual = mob.volley_remaining > 0;
                    if let Some(target) = active_target {
                        let to_t = target.eye_pos - mob.eye_pos();
                        let dist = to_t.length();
                        mob.yaw = (-to_t.x).atan2(to_t.z);
                        let los = Self::has_line_of_sight(world, reg, mob.eye_pos(), target.eye_pos);
                        if los && dist <= 24.0 {
                            mob.ai_state = MobAiState::Attack;
                            if dist > 11.0 {
                                let flat = Vec3::new(to_t.x, 0.0, to_t.z).normalize_or_zero();
                                desired_horiz = flat * 2.4;
                            }
                            if mob.volley_remaining > 0 {
                                mob.volley_timer -= dt;
                                if mob.volley_timer <= 0.0 {
                                    mob.volley_timer = 0.35;
                                    mob.volley_remaining -= 1;
                                    let aim = (target.eye_pos - mob.eye_pos()).normalize_or_zero();
                                    pending_shots.push(PendingShot {
                                        kind: ProjectileKind::BlazeFireball,
                                        dim: mob.dimension,
                                        pos: mob.eye_pos() + aim * 0.65,
                                        vel: aim * 14.0,
                                        owner: mob.id,
                                    });
                                    events.sounds.push((mob.dimension, mob.pos, "mobs_fireball"));
                                    events.particles.push(MobParticleEvent {
                                        dimension: mob.dimension,
                                        pos: mob.eye_pos(),
                                        kind: MobParticleKind::Flame,
                                        count: 5,
                                    });
                                }
                            } else if mob.attack_cooldown <= 0.0 {
                                mob.volley_remaining = 3;
                                mob.volley_timer = 0.05;
                                mob.attack_cooldown = 3.2;
                                events.sounds.push((mob.dimension, mob.pos, "mobs_mc_blaze_breath"));
                            }
                        } else {
                            mob.ai_state = MobAiState::Aggro;
                            desired_horiz = Vec3::new(to_t.x, 0.0, to_t.z).normalize_or_zero() * 2.4;
                        }
                    } else {
                        mob.state_timer -= dt;
                        if mob.state_timer <= 0.0 {
                            rng_local = splitmix64(rng_local);
                            let ang = (rng_local & 0xFFFF) as f32 / 65535.0 * std::f32::consts::TAU;
                            mob.wander_dir = Vec3::new(-ang.sin(), 0.0, ang.cos());
                            mob.yaw = ang;
                            mob.ai_state = MobAiState::Wander;
                            mob.state_timer = 2.5;
                        }
                        desired_horiz = mob.wander_dir * 1.4;
                    }
                    // Ambient flame sparks around Blaze
                    if ((mob.anim_time * 8.0) as i32) != (((mob.anim_time - dt) * 8.0) as i32) {
                        events.particles.push(MobParticleEvent {
                            dimension: mob.dimension,
                            pos: mob.pos + Vec3::new(0.0, 0.9, 0.0),
                            kind: MobParticleKind::Flame,
                            count: 2,
                        });
                    }
                    // Hovering controller above solid surface
                    let ground_y = Self::ground_height_below(world, reg, mob.pos, 8)
                        .map(|y| y as f32 + 1.0)
                        .unwrap_or(mob.pos.y - 2.0);
                    let target_y = ground_y + 2.2 + (mob.anim_time * 2.1 + mob.id as f32).sin() * 0.45;
                    mob.vel.y = ((target_y - mob.pos.y) * 2.6).clamp(-2.8, 3.2);
                    mob.vel.x = desired_horiz.x;
                    mob.vel.z = desired_horiz.z;
                    let nx = mob.pos + Vec3::new(mob.vel.x * dt, 0.0, 0.0);
                    if !Self::mob_collides_world(nx, mob.kind, mob.size, world, reg) {
                        mob.pos.x = nx.x;
                    }
                    let nz = mob.pos + Vec3::new(0.0, 0.0, mob.vel.z * dt);
                    if !Self::mob_collides_world(nz, mob.kind, mob.size, world, reg) {
                        mob.pos.z = nz.z;
                    }
                    let ny = mob.pos + Vec3::new(0.0, mob.vel.y * dt, 0.0);
                    if !Self::mob_collides_world(ny, mob.kind, mob.size, world, reg) {
                        mob.pos.y = ny.y.clamp(NETHER_LAVA_LEVEL as f32 + 1.5, 118.0);
                    }
                }
                NativeMobKind::Ghast => {
                    mob.firing_visual = mob.fire_windup_timer > 0.0;
                    if let Some(target) = active_target {
                        let to_t = target.eye_pos - mob.eye_pos();
                        let dist = to_t.length();
                        mob.yaw = (-to_t.x).atan2(to_t.z);
                        let los = Self::has_line_of_sight(world, reg, mob.eye_pos(), target.eye_pos);
                        if los && dist <= 46.0 {
                            if mob.fire_windup_timer > 0.0 {
                                mob.ai_state = MobAiState::Attack;
                                mob.fire_windup_timer -= dt;
                                if mob.fire_windup_timer <= 0.0 {
                                    let aim = (target.eye_pos - mob.eye_pos()).normalize_or_zero();
                                    pending_shots.push(PendingShot {
                                        kind: ProjectileKind::GhastFireball,
                                        dim: mob.dimension,
                                        pos: mob.eye_pos() + aim * 2.4,
                                        vel: aim * 11.5,
                                        owner: mob.id,
                                    });
                                    events.sounds.push((mob.dimension, mob.pos, "mobs_fireball"));
                                    mob.attack_cooldown = 3.8;
                                }
                            } else {
                                mob.ai_state = MobAiState::Aggro;
                                if dist > 18.0 {
                                    desired_horiz = Vec3::new(to_t.x, 0.0, to_t.z).normalize_or_zero() * 2.0;
                                }
                                if mob.attack_cooldown <= 0.0 {
                                    mob.fire_windup_timer = 0.65;
                                    events.sounds.push((mob.dimension, mob.pos, "mobs_mc_ghast_shot"));
                                }
                            }
                        } else {
                            mob.ai_state = MobAiState::Wander;
                        }
                    } else {
                        mob.fire_windup_timer = 0.0;
                        mob.state_timer -= dt;
                        if mob.state_timer <= 0.0 {
                            rng_local = splitmix64(rng_local);
                            let ang = (rng_local & 0xFFFF) as f32 / 65535.0 * std::f32::consts::TAU;
                            mob.wander_dir = Vec3::new(-ang.sin(), 0.0, ang.cos());
                            mob.yaw = ang;
                            mob.ai_state = MobAiState::Wander;
                            mob.state_timer = 4.0;
                        }
                        desired_horiz = mob.wander_dir * 1.8;
                    }
                    let bob_y = (mob.anim_time * 1.1 + mob.id as f32).sin() * 0.7;
                    mob.vel = Vec3::new(desired_horiz.x, bob_y, desired_horiz.z);
                    let np = mob.pos + mob.vel * dt;
                    if np.y > NETHER_LAVA_LEVEL as f32 + 6.0
                        && np.y < 108.0
                        && !Self::mob_collides_world(np, mob.kind, mob.size, world, reg)
                    {
                        mob.pos = np;
                    } else {
                        mob.wander_dir = -mob.wander_dir;
                    }
                }
                NativeMobKind::MagmaCube => {
                    mob.state_timer -= dt;
                    if mob.on_ground && mob.state_timer <= 0.0 {
                        if let Some(target) = active_target {
                            let d = target.pos - mob.pos;
                            mob.yaw = (-d.x).atan2(d.z);
                            let flat = Vec3::new(d.x, 0.0, d.z).normalize_or_zero();
                            mob.wander_dir = flat;
                            mob.ai_state = if d.length_squared() <= 4.0 {
                                MobAiState::Attack
                            } else {
                                MobAiState::Aggro
                            };
                        } else {
                            rng_local = splitmix64(rng_local);
                            let ang = (rng_local & 0xFFFF) as f32 / 65535.0 * std::f32::consts::TAU;
                            mob.wander_dir = Vec3::new(-ang.sin(), 0.0, ang.cos());
                            mob.yaw = ang;
                            mob.ai_state = MobAiState::Wander;
                        }
                        mob.vel.y = 4.8 + (mob.size as f32) * 1.05;
                        mob.vel.x = mob.wander_dir.x * 3.4;
                        mob.vel.z = mob.wander_dir.z * 3.4;
                        mob.on_ground = false;
                        mob.state_timer = 0.85 + ((mob.size as f32) * 0.15);
                        events.particles.push(MobParticleEvent {
                            dimension: mob.dimension,
                            pos: mob.pos,
                            kind: MobParticleKind::Flame,
                            count: 4,
                        });
                    } else if mob.on_ground {
                        mob.vel.x *= 0.85;
                        mob.vel.z *= 0.85;
                    }
                    mob.vel.y -= 18.0 * dt;
                    step_ground_mob(mob, dt, world, reg);
                    if let Some(target) = active_target {
                        let reach = 1.0 + mob.size as f32 * 0.45;
                        if target.pos.distance_squared(mob.pos) <= reach * reach
                            && mob.attack_cooldown <= 0.0
                            && target.survival
                        {
                            mob.attack_cooldown = 0.95;
                            let dmg = (mob.size as i32 * 2).max(2);
                            let kb = Vec3::new(target.pos.x - mob.pos.x, 0.0, target.pos.z - mob.pos.z)
                                .normalize_or_zero()
                                * 4.2;
                            events.player_damages.push(MobPlayerDamage {
                                player_slot: target.slot,
                                damage: dmg,
                                knockback: Vec3::new(kb.x, 3.2, kb.z),
                                apply_wither_secs: 0.0,
                                death_cause: "Burned by Magma Cube",
                            });
                        }
                    }
                }
                NativeMobKind::Guardian => {
                    if let Some(target) = active_target {
                        let to_t = target.eye_pos - mob.eye_pos();
                        let dist = to_t.length();
                        mob.yaw = (-to_t.x).atan2(to_t.z);
                        let los = Self::has_line_of_sight(world, reg, mob.eye_pos(), target.eye_pos);
                        if los && dist <= 18.0 && mob.attack_cooldown <= 0.0 {
                            mob.ai_state = MobAiState::Attack;
                            mob.laser_target_player = Some(target.slot);
                            mob.laser_charge = (mob.laser_charge + dt / 2.4).min(1.0);
                            if mob.laser_charge >= 1.0 {
                                mob.laser_charge = 0.0;
                                mob.attack_cooldown = 2.0;
                                if target.survival {
                                    events.player_damages.push(MobPlayerDamage {
                                        player_slot: target.slot,
                                        damage: 6,
                                        knockback: Vec3::new(to_t.x, 0.2, to_t.z).normalize_or_zero() * 3.5,
                                        apply_wither_secs: 0.0,
                                        death_cause: "Zapped by Guardian",
                                    });
                                }
                                events.particles.push(MobParticleEvent {
                                    dimension: mob.dimension,
                                    pos: target.eye_pos,
                                    kind: MobParticleKind::LaserHit,
                                    count: 8,
                                });
                            }
                        } else {
                            mob.laser_charge = 0.0;
                            mob.laser_target_player = None;
                            mob.ai_state = MobAiState::Aggro;
                        }
                        if dist > 6.0 {
                            mob.vel = to_t.normalize_or_zero() * 2.5;
                        } else {
                            mob.vel *= 0.85;
                        }
                    } else {
                        mob.laser_charge = 0.0;
                        mob.laser_target_player = None;
                        mob.state_timer -= dt;
                        if mob.state_timer <= 0.0 {
                            rng_local = splitmix64(rng_local);
                            let ang = (rng_local & 0xFFFF) as f32 / 65535.0 * std::f32::consts::TAU;
                            mob.wander_dir = Vec3::new(-ang.sin(), 0.0, ang.cos());
                            mob.yaw = ang;
                            mob.ai_state = MobAiState::Wander;
                            mob.state_timer = 2.5;
                        }
                        mob.vel = mob.wander_dir * 1.8;
                    }
                    // Swim strictly inside submerged water volumes
                    let next_p = mob.pos + mob.vel * dt;
                    let check = IVec3::new(
                        next_p.x.floor() as i32,
                        (next_p.y + 0.4).floor() as i32,
                        next_p.z.floor() as i32,
                    );
                    if Self::is_water_at(world, reg, check)
                        && !Self::mob_collides_world(next_p, mob.kind, mob.size, world, reg)
                    {
                        mob.pos = next_p;
                    } else {
                        mob.wander_dir = -mob.wander_dir;
                        mob.vel = -mob.vel * 0.5;
                    }
                }
            }

            let hspeed = Vec3::new(mob.vel.x, 0.0, mob.vel.z).length();
            mob.walk_blend = (hspeed / 3.8).clamp(0.0, 1.0);
            mob.walk_phase += dt * (2.5 + hspeed * 2.2) * mob.walk_blend;
        }
        self.rng_state = rng_local;

        for s in pending_shots {
            self.spawn_projectile(s.kind, s.dim, s.pos, s.vel, s.owner);
        }

        // Step projectiles (fireballs, impacts, and deflected hits!)
        let mut i = 0;
        while i < self.projectiles.len() {
            let mut hit_impact: Option<Vec3> = None;
            {
                let p = &mut self.projectiles[i];
                p.life -= dt;
                p.anim_time += dt;
                p.pos += p.vel * dt;
                let world = if p.dimension.is_nether() {
                    nw_world
                } else {
                    ow_world
                };
                if ((p.anim_time * 16.0) as i32) != (((p.anim_time - dt) * 16.0) as i32) {
                    events.particles.push(MobParticleEvent {
                        dimension: p.dimension,
                        pos: p.pos,
                        kind: if p.kind == ProjectileKind::GhastFireball {
                            MobParticleKind::Smoke
                        } else {
                            MobParticleKind::Flame
                        },
                        count: 1,
                    });
                }

                // 1. Deflected fireball hitting a mob (especially Ghast!)
                if p.deflected_by_player {
                    for m_idx in 0..self.mobs.len() {
                        if self.mobs[m_idx].dimension != p.dimension {
                            continue;
                        }
                        let (bmin, bmax) = self.mobs[m_idx].world_aabb();
                        if p.pos.x >= bmin.x - 0.5
                            && p.pos.x <= bmax.x + 0.5
                            && p.pos.y >= bmin.y - 0.5
                            && p.pos.y <= bmax.y + 0.5
                            && p.pos.z >= bmin.z - 0.5
                            && p.pos.z <= bmax.z + 0.5
                        {
                            self.mobs[m_idx].hp -= 100;
                            self.mobs[m_idx].hurt_flash = 0.35;
                            hit_impact = Some(p.pos);
                            break;
                        }
                    }
                } else {
                    // 2. Hostile fireball hitting player
                    for pl in players {
                        if !pl.alive || pl.dimension != p.dimension {
                            continue;
                        }
                        let center = pl.pos + Vec3::new(0.0, 0.9, 0.0);
                        if p.pos.distance_squared(center) <= 0.95 * 0.95 {
                            if pl.survival {
                                let dmg = match p.kind {
                                    ProjectileKind::GhastFireball => 8,
                                    ProjectileKind::BlazeFireball => 5,
                                };
                                events.player_damages.push(MobPlayerDamage {
                                    player_slot: pl.slot,
                                    damage: dmg,
                                    knockback: p.vel.normalize_or_zero() * 5.0 + Vec3::Y * 3.0,
                                    apply_wither_secs: 0.0,
                                    death_cause: match p.kind {
                                        ProjectileKind::GhastFireball => "Fireballed by Ghast",
                                        ProjectileKind::BlazeFireball => "Burned by Blaze",
                                    },
                                });
                            }
                            hit_impact = Some(p.pos);
                            break;
                        }
                    }
                }

                // 3. Terrain collision
                if hit_impact.is_none()
                    && Self::is_solid_at(world, reg, p.pos.floor().as_ivec3())
                {
                    hit_impact = Some(p.pos);
                }
            }

            if let Some(impact) = hit_impact {
                let proj = self.projectiles.remove(i);
                if proj.kind == ProjectileKind::GhastFireball {
                    let world = if proj.dimension.is_nether() {
                        nw_world
                    } else {
                        ow_world
                    };
                    let center = impact.floor().as_ivec3();
                    for dz in -1..=1 {
                        for dy in -1..=1 {
                            for dx in -1..=1 {
                                let bp = center + IVec3::new(dx, dy, dz);
                                if bp.y <= 0 || bp.y >= 126 {
                                    continue;
                                }
                                let (nid, _, _) = world.get_node(bp);
                                if nid == CONTENT_AIR || nid == CONTENT_IGNORE {
                                    continue;
                                }
                                let nm = reg.get_name_from_content_id(nid).to_ascii_lowercase();
                                if nm.contains("bedrock")
                                    || nm.contains("obsidian")
                                    || nm.contains("chest")
                                {
                                    continue;
                                }
                                world.set_node(bp, CONTENT_AIR, 0, 0);
                                world.mark_priority_dirty_with_neighbors(bp);
                                events.modified_blocks.push((proj.dimension, bp));
                            }
                        }
                    }
                    events.particles.push(MobParticleEvent {
                        dimension: proj.dimension,
                        pos: impact,
                        kind: MobParticleKind::Explosion,
                        count: 20,
                    });
                    events
                        .sounds
                        .push((proj.dimension, impact, "tnt_explode"));
                } else {
                    events.particles.push(MobParticleEvent {
                        dimension: proj.dimension,
                        pos: impact,
                        kind: MobParticleKind::Flame,
                        count: 6,
                    });
                }
                continue;
            }
            if self.projectiles[i].life <= 0.0 {
                self.projectiles.remove(i);
            } else {
                i += 1;
            }
        }

        // Clean up mobs killed by deflected fireballs or hazard falls
        let mut m_idx = 0;
        while m_idx < self.mobs.len() {
            if self.mobs[m_idx].hp <= 0 {
                let dead = self.mobs.remove(m_idx);
                let drops = self.drops_for_killed(dead.kind, dead.size, false);
                if !drops.is_empty() {
                    events.mob_drops.push((dead.dimension, dead.pos, drops));
                }
                events.sounds.push((dead.dimension, dead.pos, Self::death_sound(dead.kind)));
                events.particles.push(MobParticleEvent {
                    dimension: dead.dimension,
                    pos: dead.pos + Vec3::Y,
                    kind: MobParticleKind::Smoke,
                    count: 10,
                });
            } else {
                m_idx += 1;
            }
        }

        events
    }

    pub fn active_laser_beams(&self, dim: DimensionId, players: &[MobPlayerInput]) -> Vec<GuardianLaserBeam> {
        let mut beams = Vec::new();
        for m in &self.mobs {
            if m.dimension == dim && m.kind == NativeMobKind::Guardian && m.laser_charge > 0.01 {
                if let Some(slot) = m.laser_target_player {
                    if let Some(p) = players.iter().find(|pl| pl.slot == slot && pl.alive && pl.dimension == dim) {
                        beams.push(GuardianLaserBeam {
                            dimension: dim,
                            start: m.eye_pos(),
                            end: p.eye_pos - Vec3::new(0.0, 0.25, 0.0),
                            progress: m.laser_charge,
                        });
                    }
                }
            }
        }
        beams
    }

    pub fn render_snapshots_for(
        &self,
        dim: DimensionId,
        world: &WorldStore,
    ) -> Vec<EntityRenderSnapshot> {
        let mut snaps = Vec::with_capacity(self.mobs.len() + self.projectiles.len());
        for m in &self.mobs {
            if m.dimension != dim {
                continue;
            }
            let sp = IVec3::new(
                m.pos.x.floor() as i32,
                (m.pos.y + 0.5).floor() as i32,
                m.pos.z.floor() as i32,
            );
            let (_, p1, _) = world.get_node(sp);
            let sky = if dim.is_nether() {
                0
            } else {
                (p1 & 0x0F).max(5)
            };
            let blk = ((p1 >> 4) & 0x0F).max(match m.kind {
                NativeMobKind::Blaze => 15,
                NativeMobKind::MagmaCube => 10,
                NativeMobKind::Ghast => 8,
                _ => 4,
            });
            snaps.push(EntityRenderSnapshot {
                id: m.id,
                entity_name: m.kind.entity_name().into(),
                pos: m.pos,
                yaw: m.yaw,
                collisionbox: m.kind.collisionbox(m.size),
                visual: "mesh".into(),
                visual_size: Vec3::ONE,
                mesh: String::new(),
                texture_name: m.kind.texture_name(m.firing_visual).into(),
                wield_item: String::new(),
                walk_phase: m.walk_phase,
                walk_blend: m.walk_blend,
                anim_time: m.anim_time,
                vel_y: m.vel.y,
                hurt_flash: m.hurt_flash,
                sky_light: sky,
                block_light: blk,
            });
        }
        for p in &self.projectiles {
            if p.dimension != dim {
                continue;
            }
            let half = if p.kind == ProjectileKind::GhastFireball {
                0.42
            } else {
                0.22
            };
            snaps.push(EntityRenderSnapshot {
                id: p.id,
                entity_name: "cubex:fireball".into(),
                pos: p.pos,
                yaw: 0.0,
                collisionbox: [-half, -half, -half, half, half, half],
                visual: "mesh".into(),
                visual_size: Vec3::ONE,
                mesh: String::new(),
                texture_name: String::new(),
                wield_item: String::new(),
                walk_phase: 0.0,
                walk_blend: 0.0,
                anim_time: p.anim_time,
                vel_y: p.vel.y,
                hurt_flash: 0.0,
                sky_light: 15,
                block_light: 15,
            });
        }
        snaps
    }
}

fn step_ground_mob(
    mob: &mut NativeMob,
    dt: f32,
    world: &WorldStore,
    reg: &EngineRegistries,
) {
    // Horizontal X + step-up attempt (1.05 block climbable ledge)
    let dx = mob.vel.x * dt;
    if dx.abs() > 1e-6 {
        let cand_x = mob.pos + Vec3::new(dx, 0.0, 0.0);
        if !MobManager::mob_collides_world(cand_x, mob.kind, mob.size, world, reg) {
            mob.pos.x = cand_x.x;
        } else if mob.on_ground {
            let step_up = mob.pos + Vec3::new(dx, 1.05, 0.0);
            if !MobManager::mob_collides_world(step_up, mob.kind, mob.size, world, reg) {
                mob.pos = step_up;
            }
        }
    }
    // Horizontal Z + step-up attempt
    let dz = mob.vel.z * dt;
    if dz.abs() > 1e-6 {
        let cand_z = mob.pos + Vec3::new(0.0, 0.0, dz);
        if !MobManager::mob_collides_world(cand_z, mob.kind, mob.size, world, reg) {
            mob.pos.z = cand_z.z;
        } else if mob.on_ground {
            let step_up = mob.pos + Vec3::new(0.0, 1.05, dz);
            if !MobManager::mob_collides_world(step_up, mob.kind, mob.size, world, reg) {
                mob.pos = step_up;
            }
        }
    }
    // Vertical Y
    let dy = mob.vel.y * dt;
    let cand_y = mob.pos + Vec3::new(0.0, dy, 0.0);
    if !MobManager::mob_collides_world(cand_y, mob.kind, mob.size, world, reg) {
        mob.pos.y = cand_y.y;
        mob.on_ground = false;
    } else {
        if mob.vel.y < 0.0 {
            let snap_y = (mob.pos.y + dy).ceil();
            let snapped = Vec3::new(mob.pos.x, snap_y, mob.pos.z);
            if !MobManager::mob_collides_world(snapped, mob.kind, mob.size, world, reg) {
                mob.pos.y = snap_y;
            }
            mob.on_ground = true;
        }
        mob.vel.y = 0.0;
    }
}

fn ray_aabb_intersect(
    origin: Vec3,
    dir: Vec3,
    min: Vec3,
    max: Vec3,
    max_dist: f32,
) -> Option<f32> {
    let mut tmin = 0.0f32;
    let mut tmax = max_dist;
    for a in 0..3 {
        let d = dir[a];
        let o = origin[a];
        if d.abs() < 1e-6 {
            if o < min[a] || o > max[a] {
                return None;
            }
        } else {
            let mut t1 = (min[a] - o) / d;
            let mut t2 = (max[a] - o) / d;
            if t1 > t2 {
                std::mem::swap(&mut t1, &mut t2);
            }
            tmin = tmin.max(t1);
            tmax = tmax.min(t2);
            if tmin > tmax {
                return None;
            }
        }
    }
    Some(tmin)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{DrawType, NodeDef};
    use crate::world::nether::{ensure_nether_columns, NetherPalette};

    fn make_mob_test_worlds(tag: &str) -> (EngineRegistries, WorldStore, WorldStore) {
        let mut reg = EngineRegistries::new();
        for (name, dt) in [
            ("mcl_core:stone", DrawType::Normal),
            ("mcl_core:bedrock", DrawType::Normal),
            ("mcl_core:water_source", DrawType::Liquid),
            ("mcl_nether:netherrack", DrawType::Normal),
            ("mcl_nether:nether_brick", DrawType::Normal),
            ("mcl_nether:chiseled_nether_brick", DrawType::Normal),
            ("mcl_nether:nether_lava_source", DrawType::Liquid),
            ("mcl_nether:glowstone", DrawType::Normal),
            ("mcl_nether:soul_sand", DrawType::Normal),
            ("mcl_blackstone:blackstone", DrawType::Normal),
        ] {
            reg.register_node(NodeDef {
                name: name.into(),
                drawtype: dt,
                walkable: dt == DrawType::Normal,
                sunlight_propagates: dt != DrawType::Normal,
                ..Default::default()
            });
        }
        let dir = std::env::temp_dir().join(format!("cubex_rs_mob_test_{tag}"));
        let _ = std::fs::remove_dir_all(&dir);
        let ow = WorldStore::new(&dir, 4242);
        let nw = WorldStore::new_for_dimension(
            &dir,
            4242,
            DimensionId::Nether,
            WorldGenType::Default,
            crate::world::Difficulty::Normal,
            crate::world::Gamerules::default(),
            "Survival",
        );
        (reg, ow, nw)
    }

    #[test]
    fn nether_mob_spawning_respects_24_cap_and_ai_states() {
        let (reg, ow, nw) = make_mob_test_worlds("cap");
        let npal = NetherPalette::from_registries(&reg);
        let _ = ensure_nether_columns(&nw, 0, 0, 2, 4242, &npal, &reg);

        let mut mgr = MobManager::new(12345);
        let players = vec![MobPlayerInput {
            slot: 0,
            dimension: DimensionId::Nether,
            pos: Vec3::new(8.0, 48.0, 8.0),
            eye_pos: Vec3::new(8.0, 49.5, 8.0),
            alive: true,
            survival: true,
        }];

        for _ in 0..80 {
            mgr.try_spawn_near_players(&players, &ow, &nw, &reg);
            mgr.step(0.05, &players, &ow, &nw, &reg);
        }
        let count = mgr.active_mob_count(DimensionId::Nether);
        assert!(count > 0, "Expected Nether mobs to spawn around player");
        assert!(
            count <= MAX_ACTIVE_MOBS_PER_DIM,
            "Active Nether mobs ({count}) must never exceed cap ({MAX_ACTIVE_MOBS_PER_DIM})"
        );
        let _ = std::fs::remove_dir_all(nw.world_path);
    }

    #[test]
    fn zombie_piglin_16_block_pack_aggro() {
        let (reg, ow, nw) = make_mob_test_worlds("piglin_aggro");
        let mut mgr = MobManager::new(99);
        let p0 = mgr.spawn_mob(
            NativeMobKind::ZombiePiglin,
            DimensionId::Nether,
            Vec3::new(0.0, 50.0, 0.0),
            1,
        );
        let p_near = mgr.spawn_mob(
            NativeMobKind::ZombiePiglin,
            DimensionId::Nether,
            Vec3::new(10.0, 50.0, 0.0),
            1,
        );
        let p_far = mgr.spawn_mob(
            NativeMobKind::ZombiePiglin,
            DimensionId::Nether,
            Vec3::new(28.0, 50.0, 0.0),
            1,
        );

        let players = vec![MobPlayerInput {
            slot: 0,
            dimension: DimensionId::Nether,
            pos: Vec3::new(0.0, 50.0, -3.0),
            eye_pos: Vec3::new(0.0, 51.5, -3.0),
            alive: true,
            survival: true,
        }];

        // Step passively: all Zombie Piglins remain neutral (Wander/Idle)
        let ev = mgr.step(0.05, &players, &ow, &nw, &reg);
        assert!(ev.player_damages.is_empty());
        assert!(mgr.mobs.iter().all(|m| m.aggro_target.is_none()));

        // Player punches p0 -> p0 AND p_near (10 blocks <= 16) aggro, p_far (28 blocks > 16) stays neutral!
        let hit = mgr
            .try_melee_hit(
                Vec3::new(0.0, 51.0, -3.0),
                Vec3::Z,
                5.0,
                DimensionId::Nether,
                0,
                4,
            )
            .expect("Expected melee hit on piglin p0");
        assert!(matches!(hit, MeleeNativeResult::HitMob { mob_id, .. } if mob_id == p0));

        let m_near = mgr.mobs.iter().find(|m| m.id == p_near).unwrap();
        let m_far = mgr.mobs.iter().find(|m| m.id == p_far).unwrap();
        assert_eq!(m_near.aggro_target, Some(0));
        assert_eq!(m_near.ai_state, MobAiState::Aggro);
        assert_eq!(m_far.aggro_target, None);
        let _ = std::fs::remove_dir_all(nw.world_path);
    }

    #[test]
    fn ghast_fireball_deflection_kills_ghast_and_drops_tear() {
        let (reg, ow, nw) = make_mob_test_worlds("ghast_deflect");
        let mut mgr = MobManager::new(77);
        let gid = mgr.spawn_mob(
            NativeMobKind::Ghast,
            DimensionId::Nether,
            Vec3::new(0.0, 60.0, 10.0),
            1,
        );
        mgr.spawn_projectile(
            ProjectileKind::GhastFireball,
            DimensionId::Nether,
            Vec3::new(0.0, 61.5, 2.5),
            Vec3::new(0.0, 0.0, -10.0),
            gid,
        );
        let hit = mgr
            .try_melee_hit(
                Vec3::new(0.0, 61.5, 0.0),
                Vec3::Z,
                4.0,
                DimensionId::Nether,
                0,
                1,
            )
            .expect("Player swing must deflect approaching fireball");
        assert!(matches!(hit, MeleeNativeResult::DeflectedFireball(_)));

        let players = vec![MobPlayerInput {
            slot: 0,
            dimension: DimensionId::Nether,
            pos: Vec3::new(0.0, 60.0, 0.0),
            eye_pos: Vec3::new(0.0, 61.5, 0.0),
            alive: true,
            survival: true,
        }];
        let mut killed = false;
        for _ in 0..40 {
            mgr.step(0.03, &players, &ow, &nw, &reg);
            if !mgr.mobs.iter().any(|m| m.id == gid) {
                killed = true;
                break;
            }
        }
        assert!(killed, "Deflected Ghast fireball must fly back and slay the Ghast");
        let _ = std::fs::remove_dir_all(nw.world_path);
    }

    #[test]
    fn wither_skeleton_inflicts_5s_wither_and_magma_cube_splits_and_guardian_lasers() {
        let (reg, ow, nw) = make_mob_test_worlds("wither_magma_guardian");
        let stone = reg.get_content_id("mcl_core:stone");
        let water = reg.get_content_id("mcl_core:water_source");
        let mut chunk = crate::world::Chunk::new(IVec3::ZERO);
        for z in 0..16 {
            for x in 0..16 {
                chunk.set(x, 0, z, stone, 15, 0);
                for y in 2..=12 {
                    chunk.set(x, y, z, water, 15, 0);
                }
            }
        }
        ow.insert_chunk(chunk);

        let mut mgr = MobManager::new(55);
        // 1. Wither Skeleton -> 5s wither
        mgr.spawn_mob(
            NativeMobKind::WitherSkeleton,
            DimensionId::Nether,
            Vec3::new(4.0, 50.0, 4.0),
            1,
        );
        let nw_players = vec![MobPlayerInput {
            slot: 0,
            dimension: DimensionId::Nether,
            pos: Vec3::new(4.0, 50.0, 5.2),
            eye_pos: Vec3::new(4.0, 51.5, 5.2),
            alive: true,
            survival: true,
        }];
        mgr.mobs[0].attack_cooldown = 0.0;
        let ev = mgr.step(0.02, &nw_players, &ow, &nw, &reg);
        assert!(
            ev.player_damages
                .iter()
                .any(|d| (d.apply_wither_secs - 5.0).abs() < 0.01),
            "Wither Skeleton hit must inflict 5-second Wither decay"
        );

        // 2. Magma Cube size=3 split on death
        mgr.mobs.clear();
        mgr.spawn_mob(
            NativeMobKind::MagmaCube,
            DimensionId::Nether,
            Vec3::new(0.0, 50.0, 3.0),
            3,
        );
        let kill_res = mgr
            .try_melee_hit(
                Vec3::new(0.0, 51.0, 0.0),
                Vec3::Z,
                5.0,
                DimensionId::Nether,
                0,
                20,
            )
            .expect("Expected lethal hit on Magma Cube");
        assert!(matches!(kill_res, MeleeNativeResult::HitMob { killed: true, .. }));
        assert_eq!(
            mgr.mobs.iter().filter(|m| m.kind == NativeMobKind::MagmaCube && m.size == 2).count(),
            2,
            "Killing size-3 Magma Cube must split into 2 size-2 cubes"
        );

        // 3. Guardian charges laser beam in water & hits player
        mgr.mobs.clear();
        mgr.spawn_mob(
            NativeMobKind::Guardian,
            DimensionId::Overworld,
            Vec3::new(8.0, 6.0, 8.0),
            1,
        );
        mgr.mobs[0].attack_cooldown = 0.0;
        let ow_players = vec![MobPlayerInput {
            slot: 0,
            dimension: DimensionId::Overworld,
            pos: Vec3::new(8.0, 6.0, 12.0),
            eye_pos: Vec3::new(8.0, 7.5, 12.0),
            alive: true,
            survival: true,
        }];
        let mut saw_laser = false;
        let mut laser_hit = false;
        for _ in 0..80 {
            let e = mgr.step(0.05, &ow_players, &ow, &nw, &reg);
            if !mgr.active_laser_beams(DimensionId::Overworld, &ow_players).is_empty() {
                saw_laser = true;
            }
            if e.player_damages.iter().any(|d| d.death_cause == "Zapped by Guardian") {
                laser_hit = true;
                break;
            }
        }
        assert!(saw_laser, "Expected Guardian charging laser beam");
        assert!(laser_hit, "Expected Guardian laser blast to damage player");

        let _ = std::fs::remove_dir_all(nw.world_path);
    }
}
