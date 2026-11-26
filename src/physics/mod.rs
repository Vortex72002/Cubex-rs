pub mod collision;
pub mod raycast;

pub use collision::{
    collision_move_simple, collision_move_simple_ex, log_player_damage, Aabb, CollisionAxis,
    CollisionInfo, CollisionMoveResult, DamageSource, FallEvent, GameMode, MoveInput,
    PlayerController, StepEvents,
};
pub use raycast::{
    add_drops_to_inventory, box_line_collision, dig_targeted, dir_to_facedir, dir_to_wallmounted,
    place_targeted, raycast_entities, raycast_player, raycast_voxel, raycast_voxel_ex,
    resolve_food_stats, selected_world_boxes, targeted_node_has_interaction, try_eat_held_food,
    try_interact_or_place, CompletedDig, DigController, DigTickResult, EntityHit, FoodStats,
    PlaceResult, RaycastHit, RaycastState,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EngineSettings;
    use crate::lua_api::userdata::ObjectState;
    use crate::registry::{
        DrawType, EngineRegistries, ItemDef, NodeDef, NodeSounds, SoundSpec, ToolCapabilities,
        ToolGroupCap, CONTENT_AIR,
    };
    use crate::world::{Gamerules, WorldStore};
    use glam::{IVec3, Vec3};
    use parking_lot::RwLock;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn make_world(tag: &str) -> WorldStore {
        let dir = std::env::temp_dir().join(format!("cubex_phys_unit_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        WorldStore::new(dir, 12345)
    }

    fn register_basic_stone(reg: &mut EngineRegistries) -> u16 {
        reg.register_node(NodeDef {
            name: "test:stone".into(),
            drawtype: DrawType::Normal,
            walkable: true,
            pointable: true,
            diggable: true,
            groups: HashMap::from([("cracky".into(), 3)]),
            sounds: NodeSounds {
                footstep: Some(SoundSpec {
                    name: "default_hard_footstep".into(),
                    gain: 0.3,
                    pitch: 1.0,
                }),
                dig: Some(SoundSpec {
                    name: "default_dig_cracky".into(),
                    gain: 0.5,
                    pitch: 1.0,
                }),
                dug: Some(SoundSpec {
                    name: "default_dug_node".into(),
                    gain: 0.6,
                    pitch: 1.0,
                }),
                place: Some(SoundSpec {
                    name: "default_place_node_hard".into(),
                    gain: 1.0,
                    pitch: 1.0,
                }),
                fall: None,
            },
            ..Default::default()
        })
    }

    #[test]
    fn player_aabb_and_smooth_eye_height() {
        let b = PlayerController::collision_box();
        assert!((b.min - Vec3::new(-0.3, 0.0, -0.3)).length() < 1e-6);
        assert!((b.max - Vec3::new(0.3, 1.75, 0.3)).length() < 1e-6);

        let mut reg = EngineRegistries::new();
        let stone = register_basic_stone(&mut reg);
        let world = make_world("eye_height");
        for z in -2..=2 {
            for x in -2..=2 {
                world.set_node(IVec3::new(x, 0, z), stone, 15, 0);
            }
        }
        let mut p = PlayerController::default();
        p.pos = Vec3::new(0.5, 1.0, 0.5);
        p.on_ground = true;
        assert!((p.eye_pos().y - 2.55).abs() < 1e-4);

        let settings = EngineSettings::default();
        let mut sneak_in = MoveInput::default();
        sneak_in.sneak = true;
        for _ in 0..30 {
            p.step(0.025, &sneak_in, &world, &reg, &settings);
        }
        assert!(p.is_sneaking);
        assert!((p.eye_pos().y - 2.38).abs() < 0.01);
    }

    #[test]
    fn stepheight_auto_step_onto_slab_and_rotated_stair() {
        let mut reg = EngineRegistries::new();
        let stone = register_basic_stone(&mut reg);
        let slab = reg.register_node(NodeDef {
            name: "test:slab".into(),
            drawtype: DrawType::NodeBox,
            node_box_type: "fixed".into(),
            node_boxes: vec![[-0.5, -0.5, -0.5, 0.5, 0.0, 0.5]],
            walkable: true,
            pointable: true,
            ..Default::default()
        });
        let stair = reg.register_node(NodeDef {
            name: "test:stair".into(),
            drawtype: DrawType::NodeBox,
            node_box_type: "fixed".into(),
            paramtype2: "facedir".into(),
            node_boxes: vec![
                [-0.5, -0.5, -0.5, 0.5, 0.0, 0.5],
                [-0.5, 0.0, 0.0, 0.5, 0.5, 0.5],
            ],
            walkable: true,
            pointable: true,
            ..Default::default()
        });

        let world = make_world("stepheight");
        for z in -1..=2 {
            for x in -1..=4 {
                world.set_node(IVec3::new(x, 0, z), stone, 15, 0);
            }
        }
        // Slab at x=1, y=1 (top at 1.5); facedir=1 stair at x=2, y=1 (lower=1.5 at -X, upper=2.0 at +X)
        world.set_node(IVec3::new(1, 1, 0), slab, 15, 0);
        world.set_node(IVec3::new(2, 1, 0), stair, 15, 1);

        let mut p = PlayerController::default();
        p.pos = Vec3::new(0.2, 1.0, 0.5);
        p.on_ground = true;
        // Face +X: yaw = -PI/2 => look_dir = (-sin(-pi/2), 0, cos(-pi/2)) = (+1, 0, 0)
        p.yaw = -std::f32::consts::FRAC_PI_2;
        let settings = EngineSettings::default();
        assert!(!settings.auto_jump);

        let mut inp = MoveInput::default();
        inp.forward = true;
        for _ in 0..27 {
            p.step(0.02, &inp, &world, &reg, &settings);
        }
        assert!(p.on_ground);
        assert!(
            p.pos.x > 2.3 && p.pos.x < 3.1 && p.pos.y >= 1.98 && p.pos.y <= 2.02,
            "Expected smooth stepheight climb up slab + rotated stair to y=2.0, got pos={:?}",
            p.pos
        );
    }

    #[test]
    fn sneak_edge_catch_prevents_fall_and_allows_sideways_sliding() {
        let mut reg = EngineRegistries::new();
        let stone = register_basic_stone(&mut reg);
        let world = make_world("sneak_edge");
        // 1x1x3 bridge along Z: x=0, y=0, z in 0..=2
        for z in 0..=2 {
            world.set_node(IVec3::new(0, 0, z), stone, 15, 0);
        }

        let mut p = PlayerController::default();
        p.pos = Vec3::new(0.5, 1.0, 0.5);
        p.on_ground = true;
        // Face (+1, 0, +1) diagonally: yaw = -PI/4 -> look_dir = (+0.707, 0, +0.707)
        p.yaw = -std::f32::consts::FRAC_PI_4;
        let settings = EngineSettings::default();
        let mut inp = MoveInput::default();
        inp.sneak = true;
        inp.forward = true;

        for _ in 0..80 {
            p.step(0.025, &inp, &world, &reg, &settings);
        }

        assert!(p.on_ground, "Sneaking player must not fall off ledge");
        assert!((p.pos.y - 1.0).abs() < 0.02, "pos.y={}", p.pos.y);
        assert!(p.pos.x <= 1.31, "X should stay clamped near ledge x<=1.30, got {}", p.pos.x);
        assert!(
            p.pos.z > 1.5,
            "Player should slide freely sideways/along Z ledge while clamped on X, got z={}",
            p.pos.z
        );
    }

    #[test]
    fn sprint_ctrl_and_double_tap_w_and_fov_widen() {
        let mut reg = EngineRegistries::new();
        let stone = register_basic_stone(&mut reg);
        let world = make_world("sprint");
        for x in -2..=12 {
            world.set_node(IVec3::new(x, 0, 0), stone, 15, 0);
        }
        let settings = EngineSettings::default();
        let mut p = PlayerController::default();
        p.pos = Vec3::new(0.5, 1.0, 0.5);
        p.on_ground = true;

        // Double-tap W: press -> release -> press within <280ms
        let mut inp = MoveInput::default();
        inp.forward = true;
        p.step(0.02, &inp, &world, &reg, &settings);
        assert!(!p.is_sprinting);
        inp.forward = false;
        p.step(0.02, &inp, &world, &reg, &settings);
        inp.forward = true;
        for _ in 0..20 {
            p.step(0.02, &inp, &world, &reg, &settings);
        }
        assert!(p.is_sprinting, "Double-tap W within 280ms should trigger sprint");
        assert!(
            p.current_fov > settings.fov + 4.0,
            "Sprint should widen FOV toward +8 deg, got {}",
            p.current_fov
        );
    }

    #[test]
    fn liquid_swimming_viscosity_and_climbable_ladders() {
        let mut reg = EngineRegistries::new();
        let stone = register_basic_stone(&mut reg);
        let water = reg.register_node(NodeDef {
            name: "test:water".into(),
            drawtype: DrawType::Liquid,
            walkable: false,
            pointable: false,
            liquidtype: "source".into(),
            liquid_viscosity: 1,
            ..Default::default()
        });
        let lava = reg.register_node(NodeDef {
            name: "test:lava".into(),
            drawtype: DrawType::Liquid,
            walkable: false,
            liquidtype: "source".into(),
            liquid_viscosity: 7,
            ..Default::default()
        });
        let ladder = reg.register_node(NodeDef {
            name: "test:ladder".into(),
            drawtype: DrawType::Signlike,
            walkable: false,
            climbable: true,
            ..Default::default()
        });

        let world = make_world("liquid_climb");
        for y in 0..=4 {
            world.set_node(IVec3::new(0, y, 0), water, 15, 0);
            world.set_node(IVec3::new(5, y, 0), lava, 15, 0);
            world.set_node(IVec3::new(10, y, 0), ladder, 15, 0);
        }
        world.set_node(IVec3::new(0, -1, 0), stone, 15, 0);
        world.set_node(IVec3::new(5, -1, 0), stone, 15, 0);

        let settings = EngineSettings::default();

        // 1. Swim up in water
        let mut p_water = PlayerController::default();
        p_water.pos = Vec3::new(0.5, 1.0, 0.5);
        let mut swim_up = MoveInput::default();
        swim_up.jump = true;
        for _ in 0..15 {
            p_water.step(0.025, &swim_up, &world, &reg, &settings);
        }
        assert!(p_water.in_liquid);
        assert!(p_water.pos.y > 1.3, "Expected swimming up in water, got y={}", p_water.pos.y);

        // 2. Lava viscosity slows horizontal movement more than water
        let mut move_water = PlayerController::default();
        move_water.pos = Vec3::new(0.5, 1.0, 0.5);
        let mut move_lava = PlayerController::default();
        move_lava.pos = Vec3::new(5.5, 1.0, 0.5);
        let mut fwd = MoveInput::default();
        fwd.forward = true;
        for _ in 0..5 {
            move_water.step(0.025, &fwd, &world, &reg, &settings);
            move_lava.step(0.025, &fwd, &world, &reg, &settings);
        }
        assert!(
            move_water.horizontal_speed > move_lava.horizontal_speed,
            "Water speed ({}) must exceed high-viscosity lava speed ({})",
            move_water.horizontal_speed,
            move_lava.horizontal_speed
        );

        // 3. Climbable ladder: zero gravity when idle, climb up with Space, down with Shift
        let mut p_climb = PlayerController::default();
        p_climb.pos = Vec3::new(10.5, 2.0, 0.5);
        let idle = MoveInput::default();
        for _ in 0..15 {
            p_climb.step(0.025, &idle, &world, &reg, &settings);
        }
        assert!(p_climb.is_climbing);
        assert!((p_climb.pos.y - 2.0).abs() < 1e-4, "Zero gravity on ladder failed: y={}", p_climb.pos.y);

        let mut up = MoveInput::default();
        up.jump = true;
        for _ in 0..10 {
            p_climb.step(0.025, &up, &world, &reg, &settings);
        }
        assert!(p_climb.pos.y > 2.5);

        let y_before_down = p_climb.pos.y;
        let mut down = MoveInput::default();
        down.sneak = true;
        for _ in 0..10 {
            p_climb.step(0.025, &down, &world, &reg, &settings);
        }
        assert!(p_climb.pos.y < y_before_down);
    }

    #[test]
    fn fall_damage_respects_groups_and_gamerules() {
        let mut reg = EngineRegistries::new();
        let stone = register_basic_stone(&mut reg);
        let cushion = reg.register_node(NodeDef {
            name: "test:hay".into(),
            drawtype: DrawType::Normal,
            walkable: true,
            groups: HashMap::from([("fall_damage_add_percent".into(), -100)]),
            ..Default::default()
        });
        let world = make_world("fall_dmg");
        world.set_node(IVec3::new(0, 0, 0), stone, 15, 0);
        world.set_node(IVec3::new(4, 0, 0), cushion, 15, 0);

        let settings = EngineSettings::default();
        let inp = MoveInput::default();

        // High fall onto stone
        let mut p = PlayerController::default();
        p.pos = Vec3::new(0.5, 14.0, 0.5);
        let mut landed_fall = None;
        for _ in 0..100 {
            let ev = p.step(0.025, &inp, &world, &reg, &settings);
            if ev.fall.is_some() {
                landed_fall = ev.fall;
                break;
            }
        }
        let fe = landed_fall.expect("Expected FallEvent on high landing");
        assert!(fe.damage >= 4, "Expected >=4 fall damage, got {}", fe.damage);
        assert_eq!(fe.node_id, stone);
        assert!(p.hp < 20);

        // High fall onto hay (-100%)
        let mut p2 = PlayerController::default();
        p2.pos = Vec3::new(4.5, 14.0, 0.5);
        let mut hay_fall = None;
        for _ in 0..100 {
            let ev = p2.step(0.025, &inp, &world, &reg, &settings);
            if ev.fall.is_some() {
                hay_fall = ev.fall;
            }
        }
        assert!(hay_fall.is_none());
        assert_eq!(p2.hp, 20);

        // Gamerules fall_damage = false disables fall damage on stone
        *world.gamerules.write() = Gamerules {
            fall_damage: false,
            ..Default::default()
        };
        let mut p3 = PlayerController::default();
        p3.pos = Vec3::new(0.5, 14.0, 0.5);
        for _ in 0..100 {
            let ev = p3.step(0.025, &inp, &world, &reg, &settings);
            assert!(ev.fall.is_none());
        }
        assert_eq!(p3.hp, 20);
    }

    #[test]
    fn dda_raycast_subvoxel_selection_and_liquids() {
        let mut reg = EngineRegistries::new();
        let stone = register_basic_stone(&mut reg);
        let slab = reg.register_node(NodeDef {
            name: "test:slab".into(),
            drawtype: DrawType::NodeBox,
            node_box_type: "fixed".into(),
            paramtype2: "facedir".into(),
            node_boxes: vec![[-0.5, -0.5, -0.5, 0.5, 0.0, 0.5]],
            selection_boxes: vec![[-0.5, -0.5, -0.5, 0.5, 0.0, 0.5]],
            walkable: true,
            pointable: true,
            ..Default::default()
        });
        let water = reg.register_node(NodeDef {
            name: "test:water".into(),
            drawtype: DrawType::Liquid,
            walkable: false,
            pointable: false,
            liquidtype: "source".into(),
            liquid_viscosity: 1,
            ..Default::default()
        });

        let world = make_world("raycast_subvoxel");
        world.set_node(IVec3::new(0, 0, 2), water, 15, 0);
        world.set_node(IVec3::new(0, 0, 3), slab, 15, 0);
        world.set_node(IVec3::new(0, 0, 4), stone, 15, 0);

        // Shoot at y=0.75 (upper half of z=3 slab): misses slab, misses water, hits stone at z=4
        let hit_stone = raycast_voxel(Vec3::new(0.5, 0.75, 0.5), Vec3::Z, 6.0, &world, &reg)
            .expect("should hit stone behind slab");
        assert_eq!(hit_stone.under, IVec3::new(0, 0, 4));
        assert_eq!(hit_stone.above, IVec3::new(0, 0, 3));
        assert_eq!(hit_stone.node_id, stone);

        // Shoot at y=0.25 (lower half of z=3 slab): misses water, hits slab at z=3
        let hit_slab = raycast_voxel(Vec3::new(0.5, 0.25, 0.5), Vec3::Z, 6.0, &world, &reg)
            .expect("should hit slab");
        assert_eq!(hit_slab.under, IVec3::new(0, 0, 3));
        assert_eq!(hit_slab.above, IVec3::new(0, 0, 2));
        assert!((hit_slab.intersection_point.z - 3.0).abs() < 1e-4);

        let sel = selected_world_boxes(&hit_slab, &reg);
        assert_eq!(sel.len(), 1);
        assert!((sel[0].0[1] - 0.0).abs() < 1e-4 && (sel[0].1[1] - 0.5).abs() < 1e-4);

        // With liquids_pointable = true: hits water at z=2!
        let hit_water = raycast_voxel_ex(Vec3::new(0.5, 0.25, 0.5), Vec3::Z, 6.0, true, &world, &reg)
            .expect("should hit water when liquids_pointable=true");
        assert_eq!(hit_water.under, IVec3::new(0, 0, 2));
        assert_eq!(hit_water.node_id, water);
    }

    #[test]
    fn dig_controller_progress_reset_completion_and_place() {
        let mut reg = EngineRegistries::new();
        let stone = register_basic_stone(&mut reg);
        let torch = reg.register_node(NodeDef {
            name: "test:torch".into(),
            drawtype: DrawType::Torchlike,
            paramtype2: "wallmounted".into(),
            walkable: false,
            pointable: true,
            ..Default::default()
        });
        let mut groupcaps = HashMap::new();
        groupcaps.insert(
            "cracky".into(),
            ToolGroupCap {
                times: HashMap::from([(3, 0.5)]),
                uses: 20,
                maxlevel: 1,
            },
        );
        reg.register_item(ItemDef {
            name: "test:pick".into(),
            item_type: "tool".into(),
            tool_capabilities: Some(ToolCapabilities {
                full_punch_interval: 0.9,
                max_drop_level: 1,
                groupcaps,
                damage_groups: HashMap::new(),
                punch_attack_uses: 0,
            }),
            ..Default::default()
        });

        let world = make_world("dig_place");
        world.set_node(IVec3::new(0, 0, 2), stone, 15, 0);
        world.set_node(IVec3::new(0, 0, 3), stone, 15, 0);

        let pstate = Arc::new(RwLock::new(ObjectState::new_player("Tester")));
        {
            let ps = pstate.read();
            let mut mg = ps.meta.write();
            let inv = mg.inventories.get_mut("main").unwrap();
            inv[0] = crate::world::ItemStackData::from_spec("test:pick 1");
            inv[1] = crate::world::ItemStackData::from_spec("test:torch 4");
            inv[2] = crate::world::ItemStackData::from_spec("test:stone 4");
        }

        let mut player = PlayerController::default();
        player.pos = Vec3::new(0.5, 0.0, 0.5);
        player.selected_slot = 0;

        let hit = raycast_voxel(Vec3::new(0.5, 0.5, 0.5), Vec3::Z, 5.0, &world, &reg).unwrap();
        let mut digger = DigController::new();

        // Advance halfway (0.25s of 0.5s)
        let r1 = digger.update(0.25, true, Some(&hit), &world, &reg, &player, &pstate, None);
        assert!(r1.swing_triggered);
        assert!(r1.is_digging);
        assert_eq!(r1.crack_stage, Some(5));
        assert!(r1.spawn_dig_particles);
        assert!(r1.sound_dig.is_some());

        // Switching hotbar slot resets dig progress
        player.selected_slot = 1;
        let r_reset = digger.update(0.05, true, Some(&hit), &world, &reg, &player, &pstate, None);
        assert_eq!(digger.dig_progress, 0.0);
        assert!(r_reset.completed_dig.is_none());

        // Switch back to pickaxe and complete digging
        player.selected_slot = 0;
        let _ = digger.update(0.30, true, Some(&hit), &world, &reg, &player, &pstate, None);
        let r_done = digger.update(0.25, true, Some(&hit), &world, &reg, &player, &pstate, None);
        let comp = r_done.completed_dig.expect("expected completed dig");
        assert_eq!(comp.pos, IVec3::new(0, 0, 2));
        assert_eq!(comp.node_name, "test:stone");
        assert!(comp.burst_particles);
        assert!(comp.sound_dug.is_some());
        assert_eq!(world.get_node(IVec3::new(0, 0, 2)).0, CONTENT_AIR);
        assert!(!world.take_priority_dirty_chunks().is_empty());

        // Verify tool wear and drop added to inventory
        {
            let ps = pstate.read();
            let mg = ps.meta.read();
            let inv = mg.inventories.get("main").unwrap();
            assert!(inv[0].wear > 0, "Pickaxe should receive wear");
            assert_eq!(inv[2].count, 5, "Stone drop should stack into existing slot");
        }

        // Place wallmounted torch onto front face of z=3 stone (at z=2)
        let hit2 = raycast_voxel(Vec3::new(0.5, 0.5, 0.5), Vec3::Z, 5.0, &world, &reg).unwrap();
        assert_eq!(hit2.under, IVec3::new(0, 0, 3));
        assert_eq!(hit2.above, IVec3::new(0, 0, 2));
        player.selected_slot = 1;
        let pr = try_interact_or_place(&hit2, &world, &reg, &player, &pstate, None);
        assert!(pr.placed);
        assert_eq!(pr.placed_pos, Some(IVec3::new(0, 0, 2)));
        let (placed_id, _, p2) = world.get_node(IVec3::new(0, 0, 2));
        assert_eq!(placed_id, torch);
        // Normal is -Z (0,0,-1), attachment dir = +Z (0,0,1) -> wallmounted = 4
        assert_eq!(p2, 4);

        // Player-AABB overlap protection: cannot place solid stone inside player's own body
        player.selected_slot = 2;
        player.pos = Vec3::new(0.5, 0.0, 2.5);
        let pr_blocked = try_interact_or_place(&hit2, &world, &reg, &player, &pstate, None);
        assert!(!pr_blocked.placed);
    }

    #[test]
    fn creative_flight_spectator_auto_jump_and_telemetry() {
        let mut reg = EngineRegistries::new();
        let stone = register_basic_stone(&mut reg);
        let sticky = reg.register_node(NodeDef {
            name: "test:honey".into(),
            drawtype: DrawType::Normal,
            walkable: true,
            groups: HashMap::from([("disable_jump".into(), 1)]),
            ..Default::default()
        });
        let world = make_world("flight_autojump");
        for x in 0..=10 {
            world.set_node(IVec3::new(x, 0, 0), stone, 15, 0);
        }
        // Raised 1-block plateau at x in 3..=10, y=1
        for x in 3..=10 {
            world.set_node(IVec3::new(x, 1, 0), stone, 15, 0);
        }
        world.set_node(IVec3::new(6, 0, 4), sticky, 15, 0);

        let mut settings = EngineSettings::default();
        let mut p = PlayerController::default();
        p.pos = Vec3::new(0.5, 1.0, 0.5);
        p.on_ground = true;

        // Double-tap Space in Creative toggles flying
        p.gamemode = GameMode::Creative;
        let mut jump_in = MoveInput::default();
        jump_in.jump = true;
        p.step(0.02, &jump_in, &world, &reg, &settings);
        assert!(!p.flying);
        jump_in.jump = false;
        p.step(0.02, &jump_in, &world, &reg, &settings);
        jump_in.jump = true;
        p.step(0.02, &jump_in, &world, &reg, &settings);
        assert!(p.flying, "Double-tap Space in Creative must toggle flying");

        // Switching to Survival clears flying
        p.gamemode = GameMode::Survival;
        p.step(0.02, &MoveInput::default(), &world, &reg, &settings);
        assert!(!p.flying);

        // Settle back on ground at x=1.5, enable auto_jump and walk +X into 1-block wall at x=3
        p.pos = Vec3::new(1.5, 1.0, 0.5);
        p.vel = Vec3::ZERO;
        p.on_ground = true;
        p.yaw = -std::f32::consts::FRAC_PI_2; // +X
        settings.auto_jump = true;
        let mut fwd = MoveInput::default();
        fwd.forward = true;
        let mut saw_footstep = false;
        for _ in 0..55 {
            let ev = p.step(0.02, &fwd, &world, &reg, &settings);
            if ev.footstep_node.is_some() {
                saw_footstep = true;
            }
        }
        assert!(saw_footstep, "Expected landing/stride footstep telemetry");
        assert!(p.on_ground && p.pos.x > 3.1 && p.pos.y >= 1.98, "auto_jump should climb 1-block plateau, pos={:?}", p.pos);

        // Flat ground stride telemetry check
        let mut p_flat = PlayerController::default();
        p_flat.pos = Vec3::new(0.5, 1.0, 0.5);
        p_flat.on_ground = true;
        p_flat.yaw = -std::f32::consts::FRAC_PI_2;
        settings.auto_jump = false;
        let mut stride_hits = 0usize;
        for _ in 0..25 {
            let ev = p_flat.step(0.02, &fwd, &world, &reg, &settings);
            if ev.footstep_node == Some(stone) {
                stride_hits += 1;
            }
        }
        assert!(stride_hits >= 1, "Expected stride footstep on stone after walking >1.85 blocks");

        // disable_jump blocks jumping
        p.pos = Vec3::new(6.5, 1.0, 4.5);
        p.vel = Vec3::ZERO;
        p.on_ground = true;
        let mut jump_only = MoveInput::default();
        jump_only.jump = true;
        p.step(0.02, &jump_only, &world, &reg, &settings);
        assert!(p.vel.y <= 0.0, "disable_jump should prevent jump velocity, vel.y={}", p.vel.y);
    }

    #[test]
    fn connected_fence_and_buildable_to_and_creative_dig() {
        let mut reg = EngineRegistries::new();
        let stone = register_basic_stone(&mut reg);
        let fence = reg.register_node(NodeDef {
            name: "test:fence".into(),
            drawtype: DrawType::NodeBox,
            node_box_type: "connected".into(),
            connects_to: vec!["test:stone".into(), "test:fence".into()],
            node_boxes: vec![[-0.125, -0.5, -0.125, 0.125, 0.5, 0.125]],
            connect_right: vec![[0.125, -0.1, -0.06, 0.5, 0.4, 0.06]],
            walkable: true,
            pointable: true,
            ..Default::default()
        });
        let tallgrass = reg.register_node(NodeDef {
            name: "test:tallgrass".into(),
            drawtype: DrawType::Plantlike,
            walkable: false,
            pointable: true,
            buildable_to: true,
            ..Default::default()
        });

        let world = make_world("connected_buildable");
        world.set_node(IVec3::new(0, 0, 2), fence, 15, 0);
        world.set_node(IVec3::new(1, 0, 2), stone, 15, 0); // +X neighbor triggers connect_right!
        world.set_node(IVec3::new(0, 0, 4), tallgrass, 15, 0);

        // Aim at fence right arm at x=0.8 (x=0.5+0.3): only hits because connect_right is active!
        let hit_fence = raycast_voxel(Vec3::new(0.8, 0.6, 0.5), Vec3::Z, 5.0, &world, &reg)
            .expect("should hit connected right fence rail");
        assert_eq!(hit_fence.under, IVec3::new(0, 0, 2));
        let sel = selected_world_boxes(&hit_fence, &reg);
        assert_eq!(sel.len(), 2, "Expected center post + connect_right arm");

        // buildable_to replacement on tallgrass + Creative instant dig
        let pstate = Arc::new(RwLock::new(ObjectState::new_player("Builder")));
        {
            let ps = pstate.read();
            let mut mg = ps.meta.write();
            let inv = mg.inventories.get_mut("main").unwrap();
            inv[0] = crate::world::ItemStackData::from_spec("test:stone 2");
        }
        let mut player = PlayerController::default();
        player.gamemode = GameMode::Creative;
        player.pos = Vec3::new(0.5, 0.0, 5.5);
        player.yaw = std::f32::consts::PI; // -Z

        let hit_grass = raycast_voxel(Vec3::new(0.5, 0.5, 5.5), Vec3::NEG_Z, 5.0, &world, &reg).unwrap();
        assert_eq!(hit_grass.under, IVec3::new(0, 0, 4));
        let pr = try_interact_or_place(&hit_grass, &world, &reg, &player, &pstate, None);
        assert!(pr.placed);
        assert_eq!(pr.placed_pos, Some(IVec3::new(0, 0, 4)));
        assert_eq!(world.get_node(IVec3::new(0, 0, 4)).0, stone);
        // Creative mode must not consume held item
        assert_eq!(pstate.read().meta.read().inventories["main"][0].count, 2);

        // Creative instant dig removes block on first tick
        let mut digger = DigController::new();
        let hit_placed = raycast_voxel(Vec3::new(0.5, 0.5, 5.5), Vec3::NEG_Z, 5.0, &world, &reg).unwrap();
        let d_res = digger.update(0.016, true, Some(&hit_placed), &world, &reg, &player, &pstate, None);
        assert!(d_res.completed_dig.is_some());
        assert_eq!(world.get_node(IVec3::new(0, 0, 4)).0, CONTENT_AIR);
    }

    #[test]
    fn standing_in_1x2_tunnel_and_below_surface_deals_zero_suffocation() {
        let mut reg = EngineRegistries::new();
        let stone = register_basic_stone(&mut reg);
        let slab = reg.register_node(NodeDef {
            name: "test:slab_stone".into(),
            drawtype: DrawType::NodeBox,
            node_box_type: "fixed".into(),
            node_boxes: vec![[-0.5, -0.5, -0.5, 0.5, 0.0, 0.5]],
            walkable: true,
            ..Default::default()
        });
        let leaves = reg.register_node(NodeDef {
            name: "test:leaves".into(),
            drawtype: DrawType::AllfacesOptional,
            walkable: true,
            sunlight_propagates: true,
            ..Default::default()
        });
        let water = reg.register_node(NodeDef {
            name: "test:water".into(),
            drawtype: DrawType::Liquid,
            walkable: false,
            liquid_viscosity: 1,
            ..Default::default()
        });

        let world = make_world("tunnel_no_suffocation");
        // Solid stone shell around (x=0, y=62..64, z=-2..4) with 1x2 tunnel at x=0, y=62..63
        for z in -2..=4 {
            for y in 61..=65 {
                for x in -1..=1 {
                    world.set_node(IVec3::new(x, y, z), stone, 15, 0);
                }
            }
        }
        // Carve 1x2 tunnel at x=0, y=62..63
        for z in 0..=3 {
            world.set_node(IVec3::new(0, 62, z), CONTENT_AIR, 15, 0);
            world.set_node(IVec3::new(0, 63, z), CONTENT_AIR, 15, 0);
        }
        // Carve a 1-block-deep surface hole next to surface ground at (4, 63, 4)
        for z in 2..=6 {
            for x in 2..=6 {
                world.set_node(IVec3::new(x, 62, z), stone, 15, 0);
                world.set_node(IVec3::new(x, 63, z), stone, 15, 0);
            }
        }
        world.set_node(IVec3::new(4, 63, 4), CONTENT_AIR, 15, 0);

        let settings = EngineSettings::default();
        let mut p = PlayerController::default();
        p.pos = Vec3::new(0.5, 62.0, 1.5);
        p.on_ground = true;

        // Walk forward/strafe/jump inside 1x2 tunnel for 2.0s -> zero suffocation damage
        let mut tunnel_in = MoveInput::default();
        tunnel_in.forward = true;
        tunnel_in.right = true;
        tunnel_in.jump = true;
        for _ in 0..80 {
            p.step(0.025, &tunnel_in, &world, &reg, &settings);
        }
        assert_eq!(p.hp, 20, "1x2 tunnel must deal 0 suffocation damage");
        assert_eq!(p.suffocation_timer, 0.0);

        // Stand inside 1-block-deep hole below surface (`pos.y = 63.0`, eye at 64.55 air)
        p.pos = Vec3::new(4.5, 63.0, 4.5);
        p.vel = Vec3::ZERO;
        for _ in 0..60 {
            p.step(0.025, &MoveInput::default(), &world, &reg, &settings);
        }
        assert_eq!(p.hp, 20, "1 block below surface must deal 0 suffocation damage");

        // Non-full blocks at eye level (slab, leaves, water) must never suffocate
        for &non_solid in &[slab, leaves, water] {
            world.set_node(IVec3::new(0, 63, 1), non_solid, 15, 0);
            p.pos = Vec3::new(0.5, 62.0, 1.5);
            p.vel = Vec3::ZERO;
            for _ in 0..40 {
                p.step(0.025, &MoveInput::default(), &world, &reg, &settings);
            }
            assert_eq!(p.hp, 20, "Non-opaque/partial block at eye pos must not suffocate");
        }

        // Solid opaque block directly covering exact eye position `(pos.x, pos.y + 1.55, pos.z)`:
        // grace period of 0.5s prevents instant damage, then deals damage after >0.5s
        world.set_node(IVec3::new(0, 63, 1), stone, 15, 0);
        p.pos = Vec3::new(0.5, 62.0, 1.5);
        p.vel = Vec3::ZERO;
        for _ in 0..16 {
            p.step(0.025, &MoveInput::default(), &world, &reg, &settings); // 0.40s
        }
        assert_eq!(p.hp, 20, "0.5s suffocation grace period must protect brief block placement");
        for _ in 0..12 {
            p.step(0.025, &MoveInput::default(), &world, &reg, &settings); // +0.30s -> 0.70s total
        }
        assert!(p.hp < 20, "Solid opaque block at eye pos must suffocate after 0.5s grace period");
    }

    #[test]
    fn corner_two_wall_collision_never_suffocates() {
        let mut reg = EngineRegistries::new();
        let stone = register_basic_stone(&mut reg);
        let world = make_world("two_wall_corner");
        for z in -2..=4 {
            for x in -2..=4 {
                world.set_node(IVec3::new(x, 61, z), stone, 15, 0);
            }
        }
        // Two perpendicular solid walls forming an inside corner at +X (x=1) and +Z (z=1), plus step slab on -X
        for y in 62..=65 {
            for z in -1..=2 {
                world.set_node(IVec3::new(1, y, z), stone, 15, 0);
            }
            for x in -1..=2 {
                world.set_node(IVec3::new(x, y, 1), stone, 15, 0);
            }
        }
        world.set_node(IVec3::new(0, 62, -1), stone, 15, 0);

        let settings = EngineSettings::default();
        let mut p = PlayerController::default();
        p.pos = Vec3::new(0.7, 62.0, 0.7);
        p.on_ground = true;

        let mut push_corner = MoveInput::default();
        push_corner.forward = true;
        push_corner.right = true;
        for _ in 0..80 {
            p.step(0.025, &push_corner, &world, &reg, &settings);
        }
        assert_eq!(p.hp, 20, "Touching two adjacent corner walls must never suffocate");
        assert!(!PlayerController::is_eye_position_suffocating(p.pos, &world, &reg));
    }

    #[test]
    fn hunger_exhaustion_food_eating_regen_and_starvation_work() {
        let mut reg = EngineRegistries::new();
        let stone = register_basic_stone(&mut reg);
        reg.register_item(ItemDef {
            name: "mcl_core:apple".into(),
            stack_max: 64,
            groups: HashMap::from([("eatable".into(), 4)]),
            ..Default::default()
        });
        let world = make_world("hunger_food");
        for z in -4..=32 {
            for x in -4..=4 {
                world.set_node(IVec3::new(x, 61, z), stone, 15, 0);
            }
        }

        let mut settings = EngineSettings::default();
        settings.difficulty = crate::world::Difficulty::Normal;
        let ps = Arc::new(RwLock::new(ObjectState::new_player("Hungry")));
        ps.read().meta.write().inventories.get_mut("main").unwrap()[0] =
            crate::lua_api::ItemStackData::from_spec("mcl_core:apple 3");

        let mut p = PlayerController::default();
        p.pos = Vec3::new(0.5, 62.0, 0.5);
        p.on_ground = true;

        // 1. Sprinting + jumping depletes saturation first, then hunger
        let mut sprint_move = MoveInput::default();
        sprint_move.forward = true;
        sprint_move.sprint = true;
        sprint_move.jump = true;
        for _ in 0..200 {
            p.step(0.05, &sprint_move, &world, &reg, &settings);
            if p.pos.z > 24.0 {
                p.pos.z -= 20.0;
            }
        }
        assert!(
            p.saturation < 5.0,
            "sprinting/jumping must deplete saturation from 5.0, got sat={}",
            p.saturation
        );
        for _ in 0..500 {
            p.step(0.05, &sprint_move, &world, &reg, &settings);
            if p.pos.z > 24.0 {
                p.pos.z -= 20.0;
            }
        }
        assert!(p.hunger < 20, "sprinting/jumping must deplete hunger from 20, got {}", p.hunger);

        // 2. Right-click eating consumes 1 apple and restores hunger + saturation
        p.hunger = 14;
        p.saturation = 0.0;
        p.eat_cooldown = 0.0;
        let ate = try_eat_held_food(&mut p, &ps, &reg, None);
        assert!(ate, "held apple must be eaten when hunger < 20");
        assert_eq!(p.hunger, 18);
        assert!(p.saturation > 2.0);
        assert_eq!(ps.read().meta.read().inventories["main"][0].count, 2);

        // 3. Natural health regeneration at hunger >= 18 every 2.5s
        p.hp = 16;
        p.pos = Vec3::new(0.5, 62.0, 0.5);
        p.vel = Vec3::ZERO;
        for _ in 0..60 {
            p.step(0.05, &MoveInput::default(), &world, &reg, &settings); // 3.0s
        }
        assert!(p.hp > 16, "hunger >= 18 must regenerate HP over 2.5s, got hp={}", p.hp);

        // 4. Starvation at hunger == 0 every 4.0s stops at 1 HP on Normal, reaches 0 on Hard
        p.hunger = 0;
        p.saturation = 0.0;
        p.hp = 2;
        for _ in 0..180 {
            p.step(0.05, &MoveInput::default(), &world, &reg, &settings); // 9.0s
        }
        assert_eq!(p.hp, 1, "Normal difficulty starvation must stop at 1 HP");

        settings.difficulty = crate::world::Difficulty::Hard;
        for _ in 0..90 {
            p.step(0.05, &MoveInput::default(), &world, &reg, &settings); // 4.5s
        }
        assert_eq!(p.hp, 0, "Hard difficulty starvation must be lethal at 0 HP");
        assert_eq!(p.death_cause, "Starved to death");
    }

    #[test]
    fn walking_50_blocks_mixed_terrain_snow_grass_slabs_steps_deals_zero_damage() {
        let mut reg = EngineRegistries::new();
        let dirt = reg.register_node(NodeDef {
            name: "mcl_core:dirt".into(),
            drawtype: DrawType::Normal,
            walkable: true,
            pointable: true,
            damage_per_second: 5,
            ..Default::default()
        });
        let grass = reg.register_node(NodeDef {
            name: "mcl_core:dirt_with_grass".into(),
            drawtype: DrawType::Normal,
            walkable: true,
            pointable: true,
            damage_per_second: 4,
            ..Default::default()
        });
        let snow_block = reg.register_node(NodeDef {
            name: "mcl_core:snowblock".into(),
            drawtype: DrawType::Normal,
            walkable: true,
            pointable: true,
            damage_per_second: 4,
            ..Default::default()
        });
        let snow_layer = reg.register_node(NodeDef {
            name: "mcl_core:snow".into(),
            drawtype: DrawType::NodeBox,
            node_box_type: "fixed".into(),
            node_boxes: vec![[-0.5, -0.5, -0.5, 0.5, -0.375, 0.5]],
            walkable: true,
            pointable: true,
            damage_per_second: 3,
            ..Default::default()
        });
        let tall_grass = reg.register_node(NodeDef {
            name: "mcl_flowers:tallgrass".into(),
            drawtype: DrawType::Plantlike,
            walkable: false,
            pointable: true,
            damage_per_second: 2,
            ..Default::default()
        });
        let slab = reg.register_node(NodeDef {
            name: "mcl_stairs:slab_stone".into(),
            drawtype: DrawType::NodeBox,
            node_box_type: "fixed".into(),
            node_boxes: vec![[-0.5, -0.5, -0.5, 0.5, 0.0, 0.5]],
            walkable: true,
            pointable: true,
            damage_per_second: 6,
            ..Default::default()
        });
        let water = reg.register_node(NodeDef {
            name: "mcl_core:water_source".into(),
            drawtype: DrawType::Liquid,
            walkable: false,
            groups: HashMap::from([("water".into(), 3)]),
            damage_per_second: 4,
            ..Default::default()
        });

        for id in [dirt, grass, snow_block, snow_layer, tall_grass, slab, water] {
            let def = reg.get_node_def(id).expect("node registered");
            assert_eq!(
                def.damage_per_second, 0,
                "Safe terrain node {} must force damage_per_second=0",
                def.name
            );
        }

        let world = make_world("walk_50_mixed");
        for x in -2..=72 {
            for z in -1..=1 {
                let base = if x <= 15 {
                    grass
                } else if x <= 30 {
                    snow_block
                } else {
                    dirt
                };
                world.set_node(IVec3::new(x, 0, z), base, 15, 0);
            }
            if (4..=12).contains(&x) {
                world.set_node(IVec3::new(x, 1, 0), tall_grass, 15, 0);
                world.set_node(IVec3::new(x, 2, 0), tall_grass, 15, 0);
                // Side walls at head height to ensure brushing walls does not suffocate
                world.set_node(IVec3::new(x, 1, -1), dirt, 15, 0);
                world.set_node(IVec3::new(x, 2, -1), dirt, 15, 0);
                world.set_node(IVec3::new(x, 1, 1), dirt, 15, 0);
                world.set_node(IVec3::new(x, 2, 1), dirt, 15, 0);
            }
            if (18..=28).contains(&x) && x % 2 == 0 {
                world.set_node(IVec3::new(x, 1, 0), snow_layer, 15, 0);
            }
            if (32..=36).contains(&x) || (39..=44).contains(&x) {
                world.set_node(IVec3::new(x, 1, 0), slab, 15, 0);
            }
            if x == 37 {
                world.set_node(IVec3::new(x, 1, 0), water, 15, 0);
            }
            if (46..=51).contains(&x) {
                for z in -1..=1 {
                    world.set_node(IVec3::new(x, 1, z), dirt, 15, 0);
                }
            }
            if (52..=57).contains(&x) {
                for z in -1..=1 {
                    world.set_node(IVec3::new(x, 1, z), dirt, 15, 0);
                    world.set_node(IVec3::new(x, 2, z), dirt, 15, 0);
                }
            }
            if (58..=63).contains(&x) {
                for z in -1..=1 {
                    world.set_node(IVec3::new(x, 1, z), dirt, 15, 0);
                }
            }
        }

        let mut p = PlayerController::default();
        p.pos = Vec3::new(0.5, 1.0, 0.5);
        p.on_ground = true;
        p.yaw = -std::f32::consts::FRAC_PI_2; // +X forward
        let settings = EngineSettings {
            auto_jump: true,
            ..Default::default()
        };
        let mut inp = MoveInput::default();
        inp.forward = true;
        let dts = [0.016f32, 0.033, 0.022, 0.041, 0.018];
        for i in 0..900 {
            if i == 100 {
                // Nudge right up against wall z=1 side during wall-adjacent segment
                p.pos.z = 0.69;
            } else if i == 160 {
                p.pos.z = 0.5;
            }
            p.step(dts[i % dts.len()], &inp, &world, &reg, &settings);
            assert_eq!(
                p.hp, 20,
                "Player took unexpected damage at tick {i}, pos={:?}, cause={:?}",
                p.pos, p.death_cause
            );
            if p.pos.x >= 66.0 {
                break;
            }
        }
        assert!(
            p.pos.x >= 55.0,
            "Expected player to walk at least 55 blocks across mixed terrain, reached x={}",
            p.pos.x
        );
        // Settle cleanly on ground
        inp.forward = false;
        for _ in 0..20 {
            p.step(0.02, &inp, &world, &reg, &settings);
            assert_eq!(p.hp, 20);
        }
        assert_eq!(p.fall_distance, 0.0, "fall_distance must reset to 0 on ground");

        // Drops <= 3.5 blocks deal 0 fall damage
        p.pos = Vec3::new(1.5, 4.2, 0.5);
        p.vel = Vec3::ZERO;
        p.on_ground = false;
        p.reset_fall_state();
        for _ in 0..120 {
            p.step(0.02, &MoveInput::default(), &world, &reg, &settings);
        }
        assert!(p.on_ground);
        assert_eq!(p.hp, 20, "3.2-block drop should deal 0 fall damage");

        // Drops > 3.5 blocks still deal real fall damage
        p.pos = Vec3::new(1.5, 7.5, 0.5);
        p.vel = Vec3::ZERO;
        p.on_ground = false;
        p.reset_fall_state();
        for _ in 0..160 {
            p.step(0.02, &MoveInput::default(), &world, &reg, &settings);
        }
        assert!(p.on_ground);
        assert!(p.hp < 20, "6.5-block drop should deal fall damage, got hp={}", p.hp);
    }
}
