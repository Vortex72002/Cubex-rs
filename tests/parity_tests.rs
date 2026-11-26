use cubex_rs::audio::{AudioEngine, SoundCategory};
use cubex_rs::config::{AudioSettings, EngineSettings, MinetestConfig};
use cubex_rs::lua_api::userdata::{ItemStackData, ObjectState};
use cubex_rs::lua_api::{compute_dig_params, resolve_craft_recipe, LuaRuntime};
use cubex_rs::paths::EnginePaths;
use cubex_rs::physics::{
    raycast_entities, raycast_player, raycast_voxel, selected_world_boxes, try_interact_or_place,
    DigController, GameMode, MoveInput, PlayerController, RaycastHit,
};
use cubex_rs::registry::{
    CraftRecipe, DrawType, EngineRegistries, ItemDef, NodeDef, NodeSounds, SoundSpec, TileSpec,
    ToolCapabilities, ToolGroupCap, CONTENT_AIR,
};
use cubex_rs::render::camera::{compute_camera_rig, CameraMode, IDEAL_THIRD_PERSON_DIST};
use cubex_rs::render::mesher::{facedir_face_tile, mesh_chunk_greedy, TileRot};
use cubex_rs::render::viewmodel::{create_extruded_mesh, ViewmodelState};
use cubex_rs::render::WgpuRenderer;
use cubex_rs::textures::TextureAtlas;
use cubex_rs::ui::formspec::{
    consume_craft_ingredients, ensure_craft_grid_size, formspec_requires_3x3_craft,
    handle_slot_click, parse_formspec, refresh_craft_preview, resolve_inv_meta,
    shift_transfer_slot, FormElement, InventoryCursor, SlotClickButton,
};
use cubex_rs::world::light::{
    get_block_light, get_sky_light, propagate_chunk_light, propagate_world_light, relight_around_pos,
};
use cubex_rs::world::mapgen::{
    biome_name_at, chunk_y_range_for, find_spawn_pos, generate_chunk_for_world, sample_biome,
    terrain_height_for, GenBiome, MapgenPalette, SEA_LEVEL,
};
use cubex_rs::world::persistence::{load_world, save_world};
use cubex_rs::world::schematic::{
    PlaceFlags, Schematic, SchematicNode, SchematicRotation, MTSCHEM_PROB_ALWAYS,
};
use cubex_rs::world::{Chunk, Difficulty, Gamerules, WorldGenType, WorldStore};

use glam::{IVec3, Vec3};
use image::{Rgba, RgbaImage};
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

fn tmp_dir(tag: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("cubex_parity_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    p
}

#[test]
fn mts_schematic_decoding_and_real_mineclonia_trees() {
    let mut reg = EngineRegistries::new();
    let trunk_id = reg.register_node(NodeDef {
        name: "mcl_trees:tree_oak".into(),
        paramtype2: "facedir".into(),
        ..Default::default()
    });
    let birch_trunk_id = reg.register_node(NodeDef {
        name: "mcl_trees:tree_birch".into(),
        paramtype2: "facedir".into(),
        ..Default::default()
    });
    let leaves_id = reg.register_node(NodeDef {
        name: "mcl_trees:leaves_oak".into(),
        drawtype: DrawType::AllfacesOptional,
        ..Default::default()
    });
    let gold_id = reg.register_node(NodeDef {
        name: "mcl_core:goldblock".into(),
        ..Default::default()
    });

    // 1. Synthetic .mts encoding/decoding + rotations 0/90/180/270 + replacements + flags
    let (sx, sy, sz) = (3usize, 2usize, 2usize);
    let mut data = vec![SchematicNode::default(); sx * sy * sz];
    let idx = |x: usize, y: usize, z: usize| z * (sx * sy) + y * sx + x;
    data[idx(0, 0, 0)] = SchematicNode {
        name: "mcl_trees:tree_oak".into(),
        prob: MTSCHEM_PROB_ALWAYS,
        force_place: true,
        param2: 0,
    };
    data[idx(2, 1, 1)] = SchematicNode {
        name: "mcl_trees:leaves_oak".into(),
        prob: MTSCHEM_PROB_ALWAYS,
        force_place: false,
        param2: 0,
    };

    let schem = Schematic {
        size: IVec3::new(sx as i32, sy as i32, sz as i32),
        yslice_prob: vec![MTSCHEM_PROB_ALWAYS; sy],
        node_names: vec![
            "air".into(),
            "mcl_trees:tree_oak".into(),
            "mcl_trees:leaves_oak".into(),
        ],
        data,
    };
    let bytes = schem.to_mts_bytes().expect("to_mts_bytes");
    let dec = Schematic::from_mts_bytes(&bytes).expect("from_mts_bytes");
    assert_eq!(dec.size, IVec3::new(3, 2, 2));

    let dir = tmp_dir("mts");
    let mut replacements = HashMap::new();
    replacements.insert("mcl_trees:leaves_oak".to_string(), "mcl_core:goldblock".to_string());

    for (rot, exp_x, exp_z, exp_p2) in [
        (SchematicRotation::Rot0, 0, 0, 0),
        (SchematicRotation::Rot90, 0, 2, 1),
        (SchematicRotation::Rot180, 2, 1, 2),
        (SchematicRotation::Rot270, 1, 0, 3),
    ] {
        let w = WorldStore::new(dir.join(format!("r_{exp_p2}")), 1);
        dec.place_on_world(
            &w,
            &reg,
            IVec3::ZERO,
            rot,
            PlaceFlags::default(),
            Some(&replacements),
            42,
        );
        let (tid, _, tp2) = w.get_node(IVec3::new(exp_x, 0, exp_z));
        assert_eq!(tid, trunk_id, "rot={exp_p2} trunk mismatch");
        assert_eq!(tp2, exp_p2, "rot={exp_p2} param2 mismatch");
    }

    // Check replacement + force_placement
    let wf = WorldStore::new(dir.join("force"), 1);
    wf.set_node(IVec3::ZERO, gold_id, 15, 0);
    wf.set_node(IVec3::new(2, 1, 1), trunk_id, 15, 0);
    dec.place_on_world(
        &wf,
        &reg,
        IVec3::ZERO,
        SchematicRotation::Rot0,
        PlaceFlags {
            force_placement: false,
            ..Default::default()
        },
        Some(&replacements),
        1,
    );
    // Node (0,0,0) has per-node force_place=true -> overwrites gold with trunk
    assert_eq!(wf.get_node(IVec3::ZERO).0, trunk_id);
    // Node (2,1,1) has force_place=false -> keeps existing trunk rather than replacing with gold
    assert_eq!(wf.get_node(IVec3::new(2, 1, 1)).0, trunk_id);

    // 2. Real Mineclonia tree .mts files under mcl_core/schematics/
    let paths = EnginePaths::resolve();
    let schem_dir = paths.game_dir.join("mods/ITEMS/mcl_core/schematics");
    assert!(
        schem_dir.is_dir(),
        "Expected Mineclonia schematics dir at {}",
        schem_dir.display()
    );

    let oak_file = schem_dir.join("mcl_core_oak_classic.mts");
    let birch_file = schem_dir.join("mcl_core_birch.mts");
    let spruce_file = schem_dir.join("mcl_core_spruce_1.mts");
    for f in [&oak_file, &birch_file, &spruce_file] {
        let s = Schematic::load_from_file(f).expect("load real .mts tree");
        assert!(s.size.x > 0 && s.size.y > 0 && s.size.z > 0);
    }

    let oak = Schematic::load_from_file(&oak_file).unwrap();
    let w_oak = WorldStore::new(dir.join("real_oak"), 7);
    oak.place_on_world(
        &w_oak,
        &reg,
        IVec3::new(0, 1, 0),
        SchematicRotation::Rot0,
        PlaceFlags {
            place_center_x: true,
            place_center_y: false,
            place_center_z: true,
            force_placement: false,
        },
        None,
        7,
    );
    for y in 1..=4 {
        let (nid, _, p2) = w_oak.get_node(IVec3::new(0, y, 0));
        assert_eq!(nid, trunk_id, "oak trunk missing at y={y}");
        assert_eq!(p2, 0, "oak trunk must be vertical param2=0 at y={y}");
    }
    let mut oak_leaves = 0usize;
    for y in 3..=9 {
        for z in -3..=3 {
            for x in -3..=3 {
                if w_oak.get_node(IVec3::new(x, y, z)).0 == leaves_id {
                    oak_leaves += 1;
                }
            }
        }
    }
    assert!(oak_leaves >= 15, "expected oak leaf canopy, got {oak_leaves}");

    let birch = Schematic::load_from_file(&birch_file).unwrap();
    let w_birch = WorldStore::new(dir.join("real_birch"), 9);
    birch.place_on_world(
        &w_birch,
        &reg,
        IVec3::new(0, 1, 0),
        SchematicRotation::Rot180,
        PlaceFlags {
            place_center_x: true,
            place_center_y: false,
            place_center_z: true,
            force_placement: false,
        },
        None,
        9,
    );
    assert_eq!(w_birch.get_node(IVec3::new(0, 1, 0)).0, birch_trunk_id);

    let _ = std::fs::remove_dir_all(&dir);
}


#[test]
fn mesher_leaves_nodebox_facedir_and_dual_light() {
    let mut reg = EngineRegistries::new();
    let stone = reg.register_node(NodeDef {
        name: "mcl_core:stone".into(),
        drawtype: DrawType::Normal,
        ..Default::default()
    });
    let leaves = reg.register_node(NodeDef {
        name: "mcl_trees:leaves_oak".into(),
        drawtype: DrawType::AllfacesOptional,
        sunlight_propagates: true,
        paramtype: "light".into(),
        ..Default::default()
    });
    let log = reg.register_node(NodeDef {
        name: "mcl_trees:tree_oak".into(),
        drawtype: DrawType::Normal,
        paramtype2: "facedir".into(),
        tiles: vec![
            TileSpec { name: "top.png".into(), color: None },
            TileSpec { name: "top.png".into(), color: None },
            TileSpec { name: "side.png".into(), color: None },
        ],
        ..Default::default()
    });
    let slab = reg.register_node(NodeDef {
        name: "mcl_stairs:slab_stone".into(),
        drawtype: DrawType::NodeBox,
        node_box_type: "fixed".into(),
        paramtype2: "facedir".into(),
        node_boxes: vec![[-0.5, -0.5, -0.5, 0.5, 0.0, 0.5]],
        ..Default::default()
    });
    let stair = reg.register_node(NodeDef {
        name: "mcl_stairs:stair_stone".into(),
        drawtype: DrawType::NodeBox,
        node_box_type: "fixed".into(),
        paramtype2: "facedir".into(),
        node_boxes: vec![
            [-0.5, -0.5, -0.5, 0.5, 0.0, 0.5],
            [-0.5, 0.0, 0.0, 0.5, 0.5, 0.5],
        ],
        ..Default::default()
    });
    let fence = reg.register_node(NodeDef {
        name: "mcl_fences:oak_fence".into(),
        drawtype: DrawType::NodeBox,
        node_box_type: "connected".into(),
        connects_to: vec!["mcl_fences:oak_fence".into()],
        node_boxes: vec![[-0.125, -0.5, -0.125, 0.125, 0.5, 0.125]],
        connect_right: vec![[0.125, 0.2, -0.06, 0.5, 0.4, 0.06]],
        ..Default::default()
    });
    let torch = reg.register_node(NodeDef {
        name: "mcl_torches:torch".into(),
        drawtype: DrawType::Torchlike,
        walkable: false,
        sunlight_propagates: true,
        light_source: 14,
        ..Default::default()
    });

    // Verify facedir rotates faces & UVs
    let (slot, rot) = facedir_face_tile(12, 0);
    assert_eq!(slot, 2); // +Y top tile rotates onto +X face when facedir=12
    assert_eq!(rot, TileRot::R270);

    let dir = tmp_dir("mesh_light");
    let world = WorldStore::new(&dir, 5);
    let mut ch = Chunk::new(IVec3::ZERO);
    ch.set(4, 4, 4, stone, 15, 0);
    ch.set(4, 5, 4, leaves, 15, 0);
    ch.set(8, 4, 4, log, 15, 12);
    ch.set(10, 4, 4, slab, 15, 0);
    ch.set(11, 4, 4, stair, 15, 1);
    ch.set(12, 4, 4, fence, 15, 0);
    ch.set(13, 4, 4, fence, 15, 0);
    ch.set(2, 10, 2, torch, 0, 0);
    propagate_chunk_light(&mut ch, &reg, true);
    world.insert_chunk(ch);

    // Check dual-channel sky (0..15) + torch blocklight (0..15)
    let (_, adj_p1, _) = world.get_node(IVec3::new(3, 10, 2));
    assert!(get_block_light(adj_p1) >= 13);
    assert_eq!(get_sky_light(adj_p1), 15);

    // Incremental relight when placing an opaque ceiling block above (3, 10, 2)
    world.set_node(IVec3::new(3, 12, 2), stone, 0, 0);
    let touched = relight_around_pos(&world, IVec3::new(3, 12, 2), &reg);
    assert!(!touched.is_empty());

    let snap = world.snapshot_chunk(IVec3::ZERO).unwrap();
    let mut layers = HashMap::new();
    layers.insert(stone, [1; 6]);
    layers.insert(leaves, [2; 6]);
    layers.insert(log, [20, 20, 10, 10, 20, 20]);
    layers.insert(slab, [30; 6]);
    layers.insert(stair, [31; 6]);
    layers.insert(fence, [40; 6]);
    layers.insert(torch, [50; 6]);
    let mesh = mesh_chunk_greedy(&snap, &reg, &layers, true);

    assert!(!mesh.opaque_verts.is_empty());
    assert!(!mesh.opaque_indices.is_empty());
    // Leaves at (4,5,4) must NOT cull stone +Y top face at (4,4,4)
    let stone_top_present = mesh
        .opaque_verts
        .iter()
        .any(|v| (v.packed & 0x7FF) == 1 && ((v.packed >> 11) & 7) == 2 && (v.pos[1] - 5.0).abs() < 0.01);
    assert!(stone_top_present, "leaves above stone must not cull stone top face");

    // Horizontal log +X face must use top ring layer 10
    assert!(mesh
        .opaque_verts
        .iter()
        .any(|v| (v.packed & 0x7FF) == 10 && ((v.packed >> 11) & 7) == 0));

    let _ = std::fs::remove_dir_all(&dir);
}


#[test]
fn gameplay_dig_params_priority_remesh_drops_crafting_and_audio() {
    let mut stone_groups = HashMap::new();
    stone_groups.insert("cracky".into(), 3);
    let mut times = HashMap::new();
    times.insert(3, 0.4);
    let mut groupcaps = HashMap::new();
    groupcaps.insert(
        "cracky".into(),
        ToolGroupCap {
            times,
            uses: 25,
            maxlevel: 1,
        },
    );
    let pick_tc = ToolCapabilities {
        full_punch_interval: 0.9,
        max_drop_level: 1,
        groupcaps,
        damage_groups: HashMap::new(),
        punch_attack_uses: 0,
    };

    let hand_dp = compute_dig_params(&stone_groups, None, false);
    assert!(!hand_dp.diggable);
    let pick_dp = compute_dig_params(&stone_groups, Some(&pick_tc), false);
    assert!(pick_dp.diggable);
    assert!((pick_dp.time - 0.4).abs() < 1e-5);
    assert!(pick_dp.wear > 0);

    // Full break -> priority remesh queue -> drop pickup -> crafting 2x2 / 3x3 + NodeMetaRef + Audio
    let mut reg = EngineRegistries::new();
    let stone = reg.register_node(NodeDef {
        name: "mcl_core:stone".into(),
        drawtype: DrawType::Normal,
        groups: stone_groups,
        sounds: NodeSounds {
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
            ..Default::default()
        },
        ..Default::default()
    });
    reg.register_item(ItemDef {
        name: "mcl_tools:pick_stone".into(),
        item_type: "tool".into(),
        tool_capabilities: Some(pick_tc),
        ..Default::default()
    });
    reg.register_item(ItemDef {
        name: "mcl_core:stick".into(),
        item_type: "craft".into(),
        ..Default::default()
    });
    reg.crafts.push(CraftRecipe {
        recipe_type: "shaped".into(),
        output: "mcl_tools:pick_stone 1".into(),
        recipe: vec![
            vec!["mcl_core:stone".into(), "mcl_core:stone".into(), "mcl_core:stone".into()],
            vec!["".into(), "mcl_core:stick".into(), "".into()],
            vec!["".into(), "mcl_core:stick".into(), "".into()],
        ],
        replacements: Vec::new(),
        cooktime: 0.0,
        burntime: 0.0,
    });
    reg.crafts.push(CraftRecipe {
        recipe_type: "shapeless".into(),
        output: "mcl_core:stick 2".into(),
        recipe: vec![vec!["mcl_core:stone".into()]],
        replacements: Vec::new(),
        cooktime: 0.0,
        burntime: 0.0,
    });

    let dir = tmp_dir("gameplay");
    let world = WorldStore::new(&dir, 11);
    world.set_node(IVec3::new(0, 0, 2), stone, 15, 0);

    let atlas = TextureAtlas::build(&HashMap::new(), Path::new(""), None, &reg);
    let mut renderer = WgpuRenderer::new_headless(128, 72, &atlas).expect("headless renderer");

    let pstate = Arc::new(RwLock::new(ObjectState::new_player("Miner")));
    {
        let ps = pstate.read();
        let mut mg = ps.meta.write();
        let inv = mg.inventories.get_mut("main").unwrap();
        inv.fill(ItemStackData::default());
        inv[0] = ItemStackData::from_spec("mcl_tools:pick_stone 1");
    }

    let mut player = PlayerController::default();
    player.pos = Vec3::new(0.5, 0.0, 0.5);
    player.selected_slot = 0;

    let hit = raycast_voxel(Vec3::new(0.5, 0.5, 0.5), Vec3::Z, 5.0, &world, &reg).unwrap();
    let mut digger = DigController::new();
    let _ = digger.update(0.25, true, Some(&hit), &world, &reg, &player, &pstate, None);
    let res = digger.update(0.25, true, Some(&hit), &world, &reg, &player, &pstate, None);
    let comp = res.completed_dig.expect("stone broken");
    assert_eq!(world.get_node(IVec3::new(0, 0, 2)).0, CONTENT_AIR);
    assert_eq!(comp.drops, vec!["mcl_core:stone".to_string()]);

    // Immediate <=0ms priority remesh drains queue
    let remeshed = renderer.remesh_priority_chunks_now(&world, &reg, &atlas, true);
    assert!(remeshed >= 1);
    assert!(world.take_priority_dirty_chunks().is_empty());

    // Drop pickup landed in main inv + wear applied to pick
    {
        let ps = pstate.read();
        let mg = ps.meta.read();
        let inv = &mg.inventories["main"];
        assert!(inv[0].wear > 0);
        assert_eq!(inv[1].name, "mcl_core:stone");
        assert_eq!(inv[1].count, 1);
    }

    // 2x2 craft
    let grid2 = [
        ItemStackData::from_spec("mcl_core:stone 1"),
        ItemStackData::default(),
        ItemStackData::default(),
        ItemStackData::default(),
    ];
    let out2 = resolve_craft_recipe(&reg, &grid2, 2, "normal").unwrap();
    assert_eq!(out2.item.name, "mcl_core:stick");
    assert_eq!(out2.item.count, 2);

    // 3x3 craft
    let grid3 = [
        ItemStackData::from_spec("mcl_core:stone 1"),
        ItemStackData::from_spec("mcl_core:stone 1"),
        ItemStackData::from_spec("mcl_core:stone 1"),
        ItemStackData::default(),
        ItemStackData::from_spec("mcl_core:stick 1"),
        ItemStackData::default(),
        ItemStackData::default(),
        ItemStackData::from_spec("mcl_core:stick 1"),
        ItemStackData::default(),
    ];
    let out3 = resolve_craft_recipe(&reg, &grid3, 3, "normal").unwrap();
    assert_eq!(out3.item.name, "mcl_tools:pick_stone");

    // AudioEngine silent fallback never panics
    let audio = AudioEngine::new_silent();
    audio.play_node_sound(&comp.sound_dug.unwrap(), SoundCategory::Blocks, &AudioSettings::default());
    audio.play_ui_click(&AudioSettings::default());
    audio.play_pickup(&AudioSettings::default());

    let _ = std::fs::remove_dir_all(&dir);
}


#[test]
fn viewmodel_animations_and_extruded_tool_mesh() {
    let mut vm = ViewmodelState::new();
    assert!(!vm.is_swinging());
    vm.trigger_swing();
    assert!(vm.is_swinging());
    vm.update(0.05, 6.2, true, true, false);
    assert!(vm.swing_progress() > 0.0);
    assert!(vm.bob_offset().length() > 0.0);

    vm.on_slot_changed();
    vm.update(0.06, 0.0, true, false, false);
    assert!(vm.slot_dip_offset() < -0.1);
    for _ in 0..4 {
        vm.update(0.05, 0.0, true, false, false);
    }
    assert!(vm.slot_dip_offset().abs() < 1e-5);

    let mut img = RgbaImage::from_pixel(16, 16, Rgba([0, 0, 0, 0]));
    for i in 2..14 {
        img.put_pixel(i, i, Rgba([90, 230, 250, 255]));
    }
    let mesh = create_extruded_mesh(&img);
    assert!(mesh.verts.len() > 48);
    assert_eq!(mesh.indices.len() / 6 * 4, mesh.verts.len());
}

#[test]
fn physics_persistence_and_interactive_formspec_roundtrip() {
    let mut reg = EngineRegistries::new();
    let stone = reg.register_node(NodeDef {
        name: "mcl_core:stone".into(),
        drawtype: DrawType::Normal,
        walkable: true,
        pointable: true,
        ..Default::default()
    });
    let slab = reg.register_node(NodeDef {
        name: "mcl_stairs:slab_stone".into(),
        drawtype: DrawType::NodeBox,
        node_box_type: "fixed".into(),
        node_boxes: vec![[-0.5, -0.5, -0.5, 0.5, 0.0, 0.5]],
        selection_boxes: vec![[-0.5, -0.5, -0.5, 0.5, 0.0, 0.5]],
        walkable: true,
        pointable: true,
        ..Default::default()
    });
    reg.register_item(ItemDef {
        name: "mcl_core:stone".into(),
        item_type: "node".into(),
        is_node: true,
        ..Default::default()
    });
    reg.crafts.push(CraftRecipe {
        recipe_type: "shapeless".into(),
        output: "mcl_core:slab 2".into(),
        recipe: vec![vec!["mcl_core:stone".into()]],
        replacements: Vec::new(),
        cooktime: 0.0,
        burntime: 0.0,
    });

    let dir = tmp_dir("phys_persist_ui");
    let world = WorldStore::new_with_options(
        &dir,
        5555,
        WorldGenType::Default,
        Difficulty::Hard,
        Gamerules {
            keep_inventory: true,
            do_daylight_cycle: false,
            pvp: true,
            fall_damage: true,
        },
        "Survival",
    );
    for x in -1..=3 {
        world.set_node(IVec3::new(x, 0, 0), stone, 15, 0);
    }
    world.set_node(IVec3::new(1, 1, 0), slab, 15, 0);

    // 0.6-block stepheight onto slab (auto_jump = false!)
    let mut settings = EngineSettings::default();
    settings.auto_jump = false;
    let mut player = PlayerController::default();
    player.pos = Vec3::new(0.2, 1.0, 0.5);
    player.on_ground = true;
    player.yaw = -std::f32::consts::FRAC_PI_2; // +X
    let mut fwd = MoveInput::default();
    fwd.forward = true;
    for _ in 0..18 {
        player.step(0.02, &fwd, &world, &reg, &settings);
    }
    assert!(player.on_ground);
    assert!(player.pos.x > 0.9 && (player.pos.y - 1.5).abs() < 0.05);

    // Sneak edge-catch on ledge at x=0 (move -X towards x=-1.5 air drop)
    let mut p_edge = PlayerController::default();
    p_edge.pos = Vec3::new(-0.5, 1.0, 0.5);
    p_edge.on_ground = true;
    p_edge.yaw = std::f32::consts::FRAC_PI_2; // -X
    let mut sneak_fwd = MoveInput::default();
    sneak_fwd.sneak = true;
    sneak_fwd.forward = true;
    for _ in 0..40 {
        p_edge.step(0.025, &sneak_fwd, &world, &reg, &settings);
    }
    assert!(p_edge.on_ground);
    assert!((p_edge.pos.y - 1.0).abs() < 0.02);

    // DDA raycast onto slab
    let hit = raycast_voxel(Vec3::new(1.5, 2.0, 0.5), Vec3::NEG_Y, 4.0, &world, &reg).unwrap();
    assert_eq!(hit.under, IVec3::new(1, 1, 0));
    assert_eq!(hit.above, IVec3::new(1, 2, 0));
    assert_eq!(hit.normal, IVec3::Y);
    let boxes = selected_world_boxes(&hit, &reg);
    assert_eq!(boxes.len(), 1);

    // Interactive formspec slot click + Shift-click transfer + craft preview + NodeMetaRef
    let pstate = Arc::new(RwLock::new(ObjectState::new_player("Builder")));
    let chest_meta = world.get_node_meta(IVec3::new(2, 1, 0));
    chest_meta
        .write()
        .inventories
        .insert("main".into(), vec![ItemStackData::default(); 27]);
    let resolved_chest =
        resolve_inv_meta("nodemeta:2,1,0", None, &pstate, Some(&world)).unwrap();
    {
        let pmeta = pstate.read().meta.clone();
        let mut pm = pmeta.write();
        let main = pm.inventories.get_mut("main").unwrap();
        main.fill(ItemStackData::default());
        main[0] = ItemStackData::from_spec("mcl_core:stone 12");
        let mut cursor = InventoryCursor::default();
        assert!(handle_slot_click(
            &mut cursor,
            main,
            0,
            SlotClickButton::Right,
            false,
            64
        ));
        assert_eq!(cursor.held.count, 6);
        assert_eq!(main[0].count, 6);
        assert!(handle_slot_click(
            &mut cursor,
            main,
            1,
            SlotClickButton::Left,
            false,
            64
        ));
        assert!(cursor.is_empty());

        let mut cm = resolved_chest.write();
        let chest_slots = cm.inventories.get_mut("main").unwrap();
        assert!(shift_transfer_slot(main, 1, chest_slots, 64));
        assert_eq!(chest_slots[0].count, 6);
    }
    {
        let pmeta = pstate.read().meta.clone();
        pmeta.write().inventories.get_mut("craft").unwrap()[0] =
            ItemStackData::from_spec("mcl_core:stone 2");
        refresh_craft_preview(&pmeta, &reg);
        assert_eq!(
            pmeta.read().inventories["craftpreview"][0].name,
            "mcl_core:slab"
        );
        consume_craft_ingredients(&pmeta, &reg);
        assert_eq!(pmeta.read().inventories["craft"][0].count, 1);
    }

    // Binary world save/load roundtrip
    {
        let mut ps = pstate.write();
        ps.pos = player.pos;
        ps.hp = 18;
    }
    save_world(&world, &reg, Some(&pstate.read())).expect("save_world");

    let reloaded = WorldStore::new(&dir, 5555);
    let mut loaded_player = ObjectState::new_player("Builder");
    let saved = load_world(&reloaded, &reg, Some(&mut loaded_player)).expect("load_world");
    assert_eq!(saved.seed, 5555);
    assert_eq!(saved.difficulty, Difficulty::Hard);
    assert!(saved.gamerules.keep_inventory);
    assert_eq!(reloaded.get_node(IVec3::new(1, 1, 0)).0, slab);
    assert_eq!(
        reloaded
            .get_node_meta(IVec3::new(2, 1, 0))
            .read()
            .inventories["main"][0]
            .count,
        6
    );
    assert_eq!(loaded_player.hp, 18);
    assert_eq!(
        loaded_player.meta.read().inventories["main"][0].count,
        6
    );

    let _ = std::fs::remove_dir_all(&dir);
}


#[test]
fn survival_empty_spawn_vs_creative_starter_and_drops() {
    // 1. New Survival player starts with 36 completely empty main inventory slots
    let surv = ObjectState::new_player("SurvivalTester");
    {
        let meta = surv.meta.read();
        let main = &meta.inventories["main"];
        assert_eq!(main.len(), 36);
        assert!(
            main.iter().all(|s| s.is_empty()),
            "Survival main inventory must initialize empty"
        );
    }

    // 2. Creative starter populates hotbar explicitly
    let mut crea = ObjectState::new_player("CreativeTester");
    crea.populate_creative_starter();
    assert!(
        !crea.meta.read().inventories["main"][0].is_empty(),
        "Creative starter must populate hotbar"
    );

    // 3. Survival node drops via real Mineclonia LuaRuntime + DigController
    let paths = EnginePaths::resolve();
    if paths.game_dir.join("mods").is_dir() {
        let dir = tmp_dir("surv_drops");
        let world = Arc::new(WorldStore::new(&dir, 777));
        let registries = Arc::new(RwLock::new(EngineRegistries::new()));
        let config = Arc::new(RwLock::new(MinetestConfig::load(&paths.conf_file)));
        let runtime = LuaRuntime::boot(paths, registries.clone(), world.clone(), config)
            .expect("LuaRuntime::boot");

        assert_eq!(
            runtime.get_node_drops("mcl_core:stone", Some("mcl_tools:pick_stone"), None),
            vec!["mcl_core:cobble".to_string()]
        );
        assert_eq!(
            runtime.get_node_drops("mcl_trees:tree_oak", None, None),
            vec!["mcl_trees:tree_oak".to_string()]
        );
        assert_eq!(
            runtime.get_node_drops("mcl_core:dirt_with_grass", None, None),
            vec!["mcl_core:dirt".to_string()]
        );
        assert!(runtime.get_node_drops("mcl_core:glass", None, None).is_empty());

        let surv_arc = Arc::new(RwLock::new(ObjectState::new_player("SurvivalMiner")));
        runtime
            .join_player(surv_arc.clone())
            .expect("join survival player");
        assert!(
            surv_arc.read().meta.read().inventories["main"]
                .iter()
                .all(|s| s.is_empty()),
            "join_player must keep Survival main inventory empty"
        );

        let reg = registries.read();
        let stone_id = reg.get_content_id("mcl_core:stone");
        let oak_id = reg.get_content_id("mcl_trees:tree_oak");
        let grass_id = reg.get_content_id("mcl_core:dirt_with_grass");
        let glass_id = reg.get_content_id("mcl_core:glass");

        let mut pctrl = PlayerController::default();
        pctrl.gamemode = GameMode::Survival;
        pctrl.pos = Vec3::new(0.5, 10.0, 0.5);
        pctrl.selected_slot = 0;

        let mut digger = DigController::new();

        // Break oak trunk -> slot 0 receives mcl_trees:tree_oak
        world.set_node(IVec3::new(0, 10, 2), oak_id, 15, 0);
        let hit_oak = raycast_voxel(Vec3::new(0.5, 10.5, 0.5), Vec3::Z, 5.0, &world, &reg).unwrap();
        let mut comp_oak = None;
        for _ in 0..40 {
            let res = digger.update(
                0.25,
                true,
                Some(&hit_oak),
                &world,
                &reg,
                &pctrl,
                &surv_arc,
                Some(&runtime),
            );
            if let Some(c) = res.completed_dig {
                comp_oak = Some(c);
                break;
            }
        }
        let comp_oak = comp_oak.expect("oak breaks");
        assert!(comp_oak.picked_up_any);
        assert_eq!(
            surv_arc.read().meta.read().inventories["main"][0].name,
            "mcl_trees:tree_oak"
        );
        let _ = digger.update(0.5, false, None, &world, &reg, &pctrl, &surv_arc, Some(&runtime));

        // Break grass block -> slot 1 receives mcl_core:dirt
        world.set_node(IVec3::new(0, 10, 2), grass_id, 15, 0);
        let hit_grass = raycast_voxel(Vec3::new(0.5, 10.5, 0.5), Vec3::Z, 5.0, &world, &reg).unwrap();
        let mut comp_grass = None;
        for _ in 0..30 {
            let res = digger.update(
                0.25,
                true,
                Some(&hit_grass),
                &world,
                &reg,
                &pctrl,
                &surv_arc,
                Some(&runtime),
            );
            if let Some(c) = res.completed_dig {
                comp_grass = Some(c);
                break;
            }
        }
        let comp_grass = comp_grass.expect("grass breaks");
        assert!(comp_grass.picked_up_any);
        assert_eq!(
            surv_arc.read().meta.read().inventories["main"][1].name,
            "mcl_core:dirt"
        );
        let _ = digger.update(0.5, false, None, &world, &reg, &pctrl, &surv_arc, Some(&runtime));

        // Break glass (drop = "") -> picked_up_any == false, no slot added
        world.set_node(IVec3::new(0, 10, 2), glass_id, 15, 0);
        let hit_glass = raycast_voxel(Vec3::new(0.5, 10.5, 0.5), Vec3::Z, 5.0, &world, &reg).unwrap();
        let mut comp_glass = None;
        for _ in 0..20 {
            let res = digger.update(
                0.25,
                true,
                Some(&hit_glass),
                &world,
                &reg,
                &pctrl,
                &surv_arc,
                Some(&runtime),
            );
            if let Some(c) = res.completed_dig {
                comp_glass = Some(c);
                break;
            }
        }
        let comp_glass = comp_glass.expect("glass breaks");
        assert!(!comp_glass.picked_up_any);
        assert!(surv_arc.read().meta.read().inventories["main"][2].is_empty());
        let _ = digger.update(0.5, false, None, &world, &reg, &pctrl, &surv_arc, Some(&runtime));

        // Break stone with stone pickaxe in slot 2 -> slot 3 receives mcl_core:cobble
        surv_arc.write().meta.write().inventories.get_mut("main").unwrap()[2] =
            ItemStackData::from_spec("mcl_tools:pick_stone 1");
        pctrl.selected_slot = 2;
        world.set_node(IVec3::new(0, 10, 2), stone_id, 15, 0);
        let hit_stone = raycast_voxel(Vec3::new(0.5, 10.5, 0.5), Vec3::Z, 5.0, &world, &reg).unwrap();
        let mut comp_stone = None;
        for _ in 0..30 {
            let res = digger.update(
                0.25,
                true,
                Some(&hit_stone),
                &world,
                &reg,
                &pctrl,
                &surv_arc,
                Some(&runtime),
            );
            if let Some(c) = res.completed_dig {
                comp_stone = Some(c);
                break;
            }
        }
        let comp_stone = comp_stone.expect("stone breaks");
        assert!(comp_stone.picked_up_any);
        assert_eq!(
            surv_arc.read().meta.read().inventories["main"][3].name,
            "mcl_core:cobble"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[test]
fn crafting_2x2_and_3x3_formspec_consumption() {
    let mut reg = EngineRegistries::new();
    for item_name in [
        "mcl_trees:tree_oak",
        "mcl_trees:wood_oak",
        "mcl_core:stick",
        "mcl_crafting_table:crafting_table",
        "mcl_tools:pick_wood",
    ] {
        reg.register_item(ItemDef {
            name: item_name.into(),
            ..Default::default()
        });
    }

    reg.crafts.push(CraftRecipe {
        recipe_type: "shapeless".into(),
        output: "mcl_trees:wood_oak 4".into(),
        recipe: vec![vec!["mcl_trees:tree_oak".into()]],
        replacements: vec![],
        cooktime: 0.0,
        burntime: 0.0,
    });
    reg.crafts.push(CraftRecipe {
        recipe_type: "shaped".into(),
        output: "mcl_core:stick 4".into(),
        recipe: vec![
            vec!["mcl_trees:wood_oak".into()],
            vec!["mcl_trees:wood_oak".into()],
        ],
        replacements: vec![],
        cooktime: 0.0,
        burntime: 0.0,
    });
    reg.crafts.push(CraftRecipe {
        recipe_type: "shaped".into(),
        output: "mcl_crafting_table:crafting_table".into(),
        recipe: vec![
            vec!["mcl_trees:wood_oak".into(), "mcl_trees:wood_oak".into()],
            vec!["mcl_trees:wood_oak".into(), "mcl_trees:wood_oak".into()],
        ],
        replacements: vec![],
        cooktime: 0.0,
        burntime: 0.0,
    });
    reg.crafts.push(CraftRecipe {
        recipe_type: "shaped".into(),
        output: "mcl_tools:pick_wood".into(),
        recipe: vec![
            vec![
                "mcl_trees:wood_oak".into(),
                "mcl_trees:wood_oak".into(),
                "mcl_trees:wood_oak".into(),
            ],
            vec!["".into(), "mcl_core:stick".into(), "".into()],
            vec!["".into(), "mcl_core:stick".into(), "".into()],
        ],
        replacements: vec![],
        cooktime: 0.0,
        burntime: 0.0,
    });

    let player = ObjectState::new_player("Crafter");
    let pmeta = player.meta.clone();

    // Survival 2x2 formspec uses 4 slots [0, 1, 2, 3]:
    // Step A: oak log -> 4 oak planks
    ensure_craft_grid_size(&pmeta, 4);
    {
        let mut g = pmeta.write();
        let craft = g.inventories.get_mut("craft").unwrap();
        assert_eq!(craft.len(), 4);
        craft.fill(ItemStackData::default());
        craft[0] = ItemStackData::from_spec("mcl_trees:tree_oak 1");
    }
    refresh_craft_preview(&pmeta, &reg);
    assert_eq!(pmeta.read().inventories["craftpreview"][0].name, "mcl_trees:wood_oak");
    assert_eq!(pmeta.read().inventories["craftpreview"][0].count, 4);
    consume_craft_ingredients(&pmeta, &reg);
    assert!(pmeta.read().inventories["craft"][0].is_empty());

    // Step B: 2x2 row1[0] + row2[2] planks -> 4 sticks
    {
        let mut g = pmeta.write();
        let craft = g.inventories.get_mut("craft").unwrap();
        craft[0] = ItemStackData::from_spec("mcl_trees:wood_oak 2");
        craft[2] = ItemStackData::from_spec("mcl_trees:wood_oak 1");
    }
    refresh_craft_preview(&pmeta, &reg);
    assert_eq!(pmeta.read().inventories["craftpreview"][0].name, "mcl_core:stick");
    assert_eq!(pmeta.read().inventories["craftpreview"][0].count, 4);
    consume_craft_ingredients(&pmeta, &reg);
    assert_eq!(pmeta.read().inventories["craft"][0].count, 1);
    assert!(pmeta.read().inventories["craft"][2].is_empty());

    // Step C: 2x2 planks ([0, 1, 2, 3]) -> crafting_table
    {
        let mut g = pmeta.write();
        let craft = g.inventories.get_mut("craft").unwrap();
        craft.fill(ItemStackData::default());
        for idx in [0usize, 1, 2, 3] {
            craft[idx] = ItemStackData::from_spec("mcl_trees:wood_oak 1");
        }
    }
    refresh_craft_preview(&pmeta, &reg);
    assert_eq!(
        pmeta.read().inventories["craftpreview"][0].name,
        "mcl_crafting_table:crafting_table"
    );
    consume_craft_ingredients(&pmeta, &reg);
    assert!(pmeta.read().inventories["craft"].iter().all(|s| s.is_empty()));

    // Step D: expand dynamically to 3x3 crafting table -> pick_wood + full ingredient consumption
    ensure_craft_grid_size(&pmeta, 9);
    {
        let mut g = pmeta.write();
        let craft = g.inventories.get_mut("craft").unwrap();
        assert_eq!(craft.len(), 9);
        craft[0] = ItemStackData::from_spec("mcl_trees:wood_oak 1");
        craft[1] = ItemStackData::from_spec("mcl_trees:wood_oak 1");
        craft[2] = ItemStackData::from_spec("mcl_trees:wood_oak 1");
        craft[4] = ItemStackData::from_spec("mcl_core:stick 1");
        craft[7] = ItemStackData::from_spec("mcl_core:stick 1");
    }
    refresh_craft_preview(&pmeta, &reg);
    assert_eq!(pmeta.read().inventories["craftpreview"][0].name, "mcl_tools:pick_wood");
    consume_craft_ingredients(&pmeta, &reg);
    assert!(pmeta.read().inventories["craft"].iter().all(|s| s.is_empty()));
    ensure_craft_grid_size(&pmeta, 4);
    assert_eq!(pmeta.read().inventories["craft"].len(), 4);
}

#[test]
fn sea_level_62_all_seven_biomes_and_unclipped_cross_chunk_trees() {
    assert_eq!(SEA_LEVEL, 62);
    assert_eq!(chunk_y_range_for(WorldGenType::Default), 0..=8);

    let mut seen_biomes = HashSet::new();
    for z in (-1400..=1400).step_by(32) {
        for x in (-1400..=1400).step_by(32) {
            let b = sample_biome(x, z, 4242);
            assert_eq!(b.name(), biome_name_at(x, z, 4242));
            seen_biomes.insert(b);
        }
    }
    for b in [
        GenBiome::Plains,
        GenBiome::Forest,
        GenBiome::BirchForest,
        GenBiome::Desert,
        GenBiome::Taiga,
        GenBiome::Swamp,
        GenBiome::SnowyTundra,
    ] {
        assert!(seen_biomes.contains(&b), "Expected biome {b:?} to generate");
    }

    let mut reg = EngineRegistries::new();
    for name in [
        "mcl_core:bedrock",
        "mcl_core:stone",
        "mcl_core:deepslate",
        "mcl_core:dirt",
        "mcl_core:dirt_with_grass",
        "mcl_core:dirt_with_grass_snow",
        "mcl_core:podzol",
        "mcl_core:clay",
        "mcl_core:sand",
        "mcl_core:sandstone",
        "mcl_core:gravel",
        "mcl_core:water_source",
        "mcl_trees:tree_oak",
        "mcl_trees:tree_birch",
        "mcl_trees:tree_spruce",
        "mcl_trees:tree_jungle",
        "mcl_trees:tree_acacia",
        "mcl_trees:tree_dark_oak",
        "mcl_crimson:crimson_hyphae",
        "mcl_crimson:warped_hyphae",
    ] {
        reg.register_node(NodeDef {
            name: name.into(),
            ..Default::default()
        });
    }
    for leaves in [
        "mcl_trees:leaves_oak",
        "mcl_trees:leaves_birch",
        "mcl_trees:leaves_spruce",
        "mcl_trees:leaves_jungle",
        "mcl_trees:leaves_acacia",
        "mcl_trees:leaves_dark_oak",
    ] {
        reg.register_node(NodeDef {
            name: leaves.into(),
            drawtype: DrawType::AllfacesOptional,
            paramtype: "light".into(),
            sunlight_propagates: true,
            ..Default::default()
        });
    }

    let pal = MapgenPalette::from_registries_with_game_dir(&reg, None);
    let spawn = find_spawn_pos(4242, WorldGenType::Default);
    assert!(spawn.y > SEA_LEVEL, "Dry spawn {spawn:?} must be above SEA_LEVEL=62");

    let dir = tmp_dir("cross_chunk_trees");
    let world = WorldStore::new(&dir, 4242);
    let cx0 = spawn.x.div_euclid(16);
    let cz0 = spawn.z.div_euclid(16);
    let mut keys = Vec::new();
    for cz in (cz0 - 2)..=(cz0 + 2) {
        for cx in (cx0 - 2)..=(cx0 + 2) {
            for cy in chunk_y_range_for(WorldGenType::Default) {
                let cpos = IVec3::new(cx, cy, cz);
                let ch = generate_chunk_for_world(cpos, 4242, WorldGenType::Default, &pal, &reg);
                world.insert_chunk(ch);
                keys.push(cpos);
            }
        }
    }
    propagate_world_light(&world, &keys, &reg);

    // Every generated trunk in this region must have leaf canopy extending around it smoothly across chunk borders
    let oak_log = pal.oak_log;
    let birch_log = pal.birch_log;
    let spruce_log = pal.spruce_log;
    let mut verified_canopy_across_border = false;
    let mut trunk_bases = Vec::new();
    for cz in (cz0 - 1)..=(cz0 + 1) {
        for cx in (cx0 - 1)..=(cx0 + 1) {
            for lz in 0..16 {
                for lx in 0..16 {
                    let wx = cx * 16 + lx;
                    let wz = cz * 16 + lz;
                    let h = terrain_height_for(wx, wz, 4242, WorldGenType::Default);
                    let id = world.get_node(IVec3::new(wx, h + 1, wz)).0;
                    if id == oak_log || id == birch_log || id == spruce_log {
                        trunk_bases.push((wx, h + 1, wz));
                    }
                }
            }
        }
    }
    for &(tx, ty, tz) in &trunk_bases {
        // Count canopy leaf nodes in a 5x5 top region around trunk (must cross chunk border if near edge)
        let mut canopy_leaves = 0usize;
        for dy in 2..=7 {
            for dz in -2..=2 {
                for dx in -2..=2 {
                    let pos = IVec3::new(tx + dx, ty + dy, tz + dz);
                    let (nid, p1, _) = world.get_node(pos);
                    if nid == pal.oak_leaves || nid == pal.birch_leaves || nid == pal.spruce_leaves {
                        canopy_leaves += 1;
                        assert!(get_sky_light(p1) >= 8, "Leaf sky light too low at {pos:?}");
                        if pos.x.div_euclid(16) != tx.div_euclid(16)
                            || pos.z.div_euclid(16) != tz.div_euclid(16)
                        {
                            verified_canopy_across_border = true;
                        }
                    }
                }
            }
        }
        assert!(canopy_leaves >= 8, "Tree at ({tx},{ty},{tz}) looks clipped: only {canopy_leaves} leaves");
    }
    // Check an explicit tree placed directly straddling a chunk border stays unclipped across chunks
    if !verified_canopy_across_border {
        let paths = EnginePaths::resolve();
        let oak_path = paths
            .game_dir
            .join("mods/ITEMS/mcl_core/schematics/mcl_core_oak_classic.mts");
        if oak_path.is_file() {
            let oak = Schematic::load_from_file(&oak_path).unwrap();
            oak.place_on_world(
                &world,
                &reg,
                IVec3::new(cx0 * 16 + 15, spawn.y, cz0 * 16 + 15),
                SchematicRotation::Rot0,
                PlaceFlags {
                    place_center_x: true,
                    place_center_y: false,
                    place_center_z: true,
                    force_placement: true,
                },
                None,
                99,
            );
            let mut left_leaves = 0;
            let mut right_leaves = 0;
            for y in spawn.y..=(spawn.y + 8) {
                for dz in -2..=2 {
                    if world.get_node(IVec3::new(cx0 * 16 + 14, y, cz0 * 16 + 15 + dz)).0 == oak_log
                        || world.get_node(IVec3::new(cx0 * 16 + 14, y, cz0 * 16 + 15 + dz)).0 == pal.oak_leaves
                    {
                        left_leaves += 1;
                    }
                    if world.get_node(IVec3::new(cx0 * 16 + 16, y, cz0 * 16 + 15 + dz)).0 == pal.oak_leaves {
                        right_leaves += 1;
                    }
                }
            }
            assert!(left_leaves > 0 && right_leaves > 0);
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}


#[test]
fn third_person_camera_modes_and_wall_pull_in() {
    assert_eq!(CameraMode::FirstPerson.cycle(), CameraMode::ThirdPersonBack);
    assert_eq!(CameraMode::ThirdPersonBack.cycle(), CameraMode::ThirdPersonFront);
    assert_eq!(CameraMode::ThirdPersonFront.cycle(), CameraMode::FirstPerson);

    let mut reg = EngineRegistries::new();
    let stone = reg.register_node(NodeDef {
        name: "mcl_core:stone".into(),
        drawtype: DrawType::Normal,
        walkable: true,
        pointable: true,
        ..Default::default()
    });

    let dir = tmp_dir("camera_rig");
    let world = WorldStore::new(&dir, 1);
    let mut player = PlayerController::default();
    player.pos = Vec3::new(0.5, 64.0, 0.5);
    player.yaw = 0.0; // look_dir = +Z
    player.pitch = 0.0;
    let settings = EngineSettings::default();

    let fp = compute_camera_rig(CameraMode::FirstPerson, &player, 16.0 / 9.0, &settings, &world, &reg);
    assert!(fp.show_viewmodel);
    assert!(!fp.show_player_model);

    let tp_open = compute_camera_rig(CameraMode::ThirdPersonBack, &player, 16.0 / 9.0, &settings, &world, &reg);
    assert!(!tp_open.show_viewmodel);
    assert!(tp_open.show_player_model);
    assert!((tp_open.effective_distance - IDEAL_THIRD_PERSON_DIST).abs() < 1e-3);

    // Place a solid stone wall behind player at z = -2 so ThirdPersonBack pulls forward cleanly
    world.set_node(IVec3::new(0, 65, -2), stone, 15, 0);
    let tp_blocked = compute_camera_rig(CameraMode::ThirdPersonBack, &player, 16.0 / 9.0, &settings, &world, &reg);
    assert!(tp_blocked.effective_distance < 1.5);
    assert!(tp_blocked.eye_pos.z > -1.0);

    let _ = std::fs::remove_dir_all(&dir);
}


#[test]
fn furnace_meta_formspec_smelting_and_3x3_stone_pickaxe() {
    let paths = EnginePaths::resolve();
    assert!(paths.game_dir.is_dir(), "Mineclonia game dir required");

    let dir = tmp_dir("furnace_craft3x3");
    let world = Arc::new(WorldStore::new(&dir, 42));
    let registries = Arc::new(RwLock::new(EngineRegistries::new()));
    let mut cfg = MinetestConfig::load(&paths.conf_file);
    cfg.set("creative_mode", false);
    cfg.set("enable_damage", true);
    cfg.set("only_peaceful_mobs", false);
    let config = Arc::new(RwLock::new(cfg));
    let runtime = LuaRuntime::boot(paths, registries.clone(), world.clone(), config)
        .expect("LuaRuntime::boot");

    let player_arc = Arc::new(RwLock::new(ObjectState::new_player("SmeltTester")));
    runtime.join_player(player_arc.clone()).expect("join_player");

    let mut player = PlayerController::default();
    player.gamemode = GameMode::Survival;
    player.pos = Vec3::new(0.5, 11.0, -1.5);
    player.selected_slot = 0;

    let (stone_id, furn_id, ct_id) = {
        let reg = registries.read();
        (
            reg.get_content_id("mcl_core:stone"),
            reg.get_content_id("mcl_furnaces:furnace"),
            reg.get_content_id("mcl_crafting_table:crafting_table"),
        )
    };
    let floor_pos = IVec3::new(2, 10, 2);
    let furn_pos = IVec3::new(2, 11, 2);
    let chest_floor = IVec3::new(3, 10, 2);
    let chest_pos = IVec3::new(3, 11, 2);
    let ct_pos = IVec3::new(4, 11, 2);
    world.set_node(floor_pos, stone_id, 15, 0);
    world.set_node(chest_floor, stone_id, 15, 0);
    world.set_node(IVec3::new(4, 10, 2), stone_id, 15, 0);

    {
        let p = player_arc.read();
        let mut mg = p.meta.write();
        let main = mg.inventories.get_mut("main").unwrap();
        main[0] = ItemStackData::from_spec("mcl_furnaces:furnace 1");
        main[1] = ItemStackData::from_spec("mcl_chests:chest 1");
    }

    // 1. Place furnace via try_interact_or_place -> on_construct initializes meta["formspec"] + src/fuel/dst
    let place_hit = RaycastHit {
        pos: floor_pos,
        prev_pos: furn_pos,
        normal: IVec3::Y,
        node_id: stone_id,
        under: floor_pos,
        above: furn_pos,
        intersection_point: Vec3::new(2.5, 11.0, 2.5),
        intersection_normal: Vec3::Y,
        box_id: 0,
        distance_sq: 4.0,
        param2: 0,
        neighbors: 0,
    };
    {
        let reg = registries.read();
        let pr = try_interact_or_place(
            &place_hit,
            &world,
            &reg,
            &player,
            &player_arc,
            Some(&runtime),
        );
        assert!(pr.placed, "placing furnace should succeed");
    }
    assert_eq!(world.get_node(furn_pos).0, furn_id);
    let fmeta = world.get_node_meta(furn_pos);
    {
        let g = fmeta.read();
        assert!(
            !g.fields.get("formspec").map(String::as_str).unwrap_or("").is_empty(),
            "furnace on_construct must set meta['formspec']"
        );
        assert_eq!(g.inventories.get("src").map(Vec::len), Some(1));
        assert_eq!(g.inventories.get("fuel").map(Vec::len), Some(1));
        assert_eq!(g.inventories.get("dst").map(Vec::len), Some(1));
    }

    // Right-clicking the furnace while not sneaking opens its nodemeta formspec (not block placement!)
    let _ = runtime.pop_formspecs();
    let furn_hit = RaycastHit {
        pos: furn_pos,
        prev_pos: IVec3::new(2, 12, 2),
        normal: IVec3::Y,
        node_id: furn_id,
        under: furn_pos,
        above: IVec3::new(2, 12, 2),
        intersection_point: Vec3::new(2.5, 12.0, 2.5),
        intersection_normal: Vec3::Y,
        box_id: 0,
        distance_sq: 4.0,
        param2: 0,
        neighbors: 0,
    };
    {
        let reg = registries.read();
        let res = try_interact_or_place(
            &furn_hit,
            &world,
            &reg,
            &player,
            &player_arc,
            Some(&runtime),
        );
        assert!(res.interacted && !res.placed, "right-clicking furnace must open UI, not place block");
    }
    let f_forms = runtime.pop_formspecs();
    assert_eq!(f_forms.len(), 1, "expected 1 furnace nodemeta formspec request");
    assert_eq!(f_forms[0].formname, "nodemeta:2,11,2");
    let parsed_furn = parse_formspec(&f_forms[0].formspec);
    let furn_labels: Vec<&str> = parsed_furn
        .elements
        .iter()
        .filter_map(|e| match e {
            FormElement::Label { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        furn_labels.iter().any(|l| l.contains("Furnace")) && furn_labels.contains(&"Inventory"),
        "furnace labels should be clean and unescaped, got {furn_labels:?}"
    );
    assert!(
        furn_labels.iter().all(|l| !l.contains('\x1b')),
        "formspec labels must not contain escape chars: {furn_labels:?}"
    );

    // Place and right-click mcl_chests:chest -> opens chest formspec with clean labels
    player.selected_slot = 1;
    player_arc.write().wield_index = 2;
    let chest_place_hit = RaycastHit {
        pos: chest_floor,
        prev_pos: chest_pos,
        normal: IVec3::Y,
        node_id: stone_id,
        under: chest_floor,
        above: chest_pos,
        intersection_point: Vec3::new(3.5, 11.0, 2.5),
        intersection_normal: Vec3::Y,
        box_id: 0,
        distance_sq: 4.0,
        param2: 0,
        neighbors: 0,
    };
    {
        let reg = registries.read();
        let pr = try_interact_or_place(&chest_place_hit, &world, &reg, &player, &player_arc, Some(&runtime));
        assert!(pr.placed, "placing chest should succeed");
    }
    let chest_id = world.get_node(chest_pos).0;
    let _ = runtime.pop_formspecs();
    let chest_hit = RaycastHit {
        pos: chest_pos,
        prev_pos: IVec3::new(3, 12, 2),
        normal: IVec3::Y,
        node_id: chest_id,
        under: chest_pos,
        above: IVec3::new(3, 12, 2),
        intersection_point: Vec3::new(3.5, 12.0, 2.5),
        intersection_normal: Vec3::Y,
        box_id: 0,
        distance_sq: 4.0,
        param2: 0,
        neighbors: 0,
    };
    {
        let reg = registries.read();
        let res = try_interact_or_place(&chest_hit, &world, &reg, &player, &player_arc, Some(&runtime));
        assert!(res.interacted && !res.placed, "right-clicking chest must open chest UI");
    }
    let chest_forms = runtime.pop_formspecs();
    assert_eq!(chest_forms.len(), 1, "chest right-click should open chest formspec");
    let parsed_chest = parse_formspec(&chest_forms[0].formspec);
    assert!(parsed_chest.elements.iter().any(|e| matches!(e, FormElement::Label { text, .. } if text == "Chest")));

    // 2. Put mcl_core:cobble in src + mcl_trees:wood_oak in fuel -> triggers timer & smelts mcl_core:stone
    let cobble_stack = ItemStackData::from_spec("mcl_core:cobble 2");
    let fuel_stack = ItemStackData::from_spec("mcl_trees:wood_oak 2");
    assert_eq!(
        runtime.allow_metadata_inventory_put(furn_pos, "src", 1, &cobble_stack, &player_arc),
        2
    );
    assert_eq!(
        runtime.allow_metadata_inventory_put(furn_pos, "fuel", 1, &fuel_stack, &player_arc),
        2
    );
    {
        let mut g = fmeta.write();
        g.inventories.get_mut("src").unwrap()[0] = cobble_stack.clone();
        g.inventories.get_mut("fuel").unwrap()[0] = fuel_stack.clone();
    }
    runtime.on_metadata_inventory_put(furn_pos, "src", 1, &cobble_stack, &player_arc);
    runtime.on_metadata_inventory_put(furn_pos, "fuel", 1, &fuel_stack, &player_arc);
    assert!(
        world.node_timers.read().contains_key(&furn_pos),
        "furnace node timer must start on inventory put"
    );

    for _ in 0..15 {
        runtime.step(1.0);
    }

    let smelted_dst = fmeta.read().inventories["dst"][0].clone();
    assert_eq!(
        smelted_dst.name, "mcl_core:stone",
        "1x cobble should smelt into 1x mcl_core:stone after 15s"
    );
    assert!(smelted_dst.count >= 1);
    assert!(
        fmeta.read().inventories["src"][0].count < 2,
        "source cobble stack must decrement after smelting"
    );
    let active_fs = fmeta.read().fields.get("formspec").cloned().unwrap_or_default();
    assert!(
        active_fs.contains("lowpart"),
        "running furnace should update meta['formspec'] with live lowpart progress bar"
    );

    // 3. 3x3 Crafting Table -> mcl_tools:pick_stone + shrink back to 2x2 safely returning items
    world.set_node(ct_pos, ct_id, 15, 0);
    player_arc.write().pos = Vec3::new(4.5, 11.0, 1.0);
    let ct_hit = RaycastHit {
        pos: ct_pos,
        prev_pos: IVec3::new(4, 12, 2),
        normal: IVec3::Y,
        node_id: ct_id,
        under: ct_pos,
        above: IVec3::new(4, 12, 2),
        intersection_point: Vec3::new(4.5, 12.0, 2.5),
        intersection_normal: Vec3::Y,
        box_id: 0,
        distance_sq: 2.0,
        param2: 0,
        neighbors: 0,
    };
    let _ = runtime.pop_formspecs();
    {
        let reg = registries.read();
        let res = try_interact_or_place(&ct_hit, &world, &reg, &player, &player_arc, Some(&runtime));
        assert!(res.interacted && !res.placed);
    }
    let ct_forms = runtime.pop_formspecs();
    assert_eq!(ct_forms.len(), 1);
    let parsed_ct = parse_formspec(&ct_forms[0].formspec);
    assert!(formspec_requires_3x3_craft(&parsed_ct));

    let pmeta = player_arc.read().meta.clone();
    ensure_craft_grid_size(&pmeta, 9);
    {
        let mut g = pmeta.write();
        let craft = g.inventories.get_mut("craft").unwrap();
        craft.fill(ItemStackData::default());
        craft[0] = ItemStackData::from_spec("mcl_core:cobble 2");
        craft[1] = ItemStackData::from_spec("mcl_core:cobble 1");
        craft[2] = ItemStackData::from_spec("mcl_core:cobble 1");
        craft[4] = ItemStackData::from_spec("mcl_core:stick 1");
        craft[7] = ItemStackData::from_spec("mcl_core:stick 1");
    }
    {
        let reg = registries.read();
        refresh_craft_preview(&pmeta, &reg);
        let out = pmeta.read().inventories["craftpreview"][0].clone();
        assert_eq!(out.name, "mcl_tools:pick_stone");
        assert_eq!(out.count, 1);
        consume_craft_ingredients(&pmeta, &reg);
    }
    assert_eq!(pmeta.read().inventories["craft"][0].count, 1);
    assert!(pmeta.read().inventories["craft"][1].is_empty());
    assert!(pmeta.read().inventories["craft"][2].is_empty());
    assert!(pmeta.read().inventories["craft"][4].is_empty());
    assert!(pmeta.read().inventories["craft"][7].is_empty());

    // Put cobblestone into 3x3-only slot 6 and close (shrink back to 4) -> moved back into main
    pmeta.write().inventories.get_mut("craft").unwrap()[6] =
        ItemStackData::from_spec("mcl_core:cobble 5");
    ensure_craft_grid_size(&pmeta, 4);
    assert_eq!(pmeta.read().inventories["craft"].len(), 4);
    assert!(pmeta.read().inventories["main"]
        .iter()
        .any(|s| s.name == "mcl_core:cobble" && s.count == 5));

    let _ = std::fs::remove_dir_all(&dir);
}


#[test]
fn large_u64_seed_boots_mcl_core_without_honeycomb_fallback() {
    let paths = EnginePaths::resolve();
    if !paths.game_dir.join("mods").is_dir() {
        return;
    }
    let dir = std::env::temp_dir().join("cubex_parity_large_seed");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let registries = Arc::new(RwLock::new(EngineRegistries::new()));
    // Exact large u64 seed (> i64::MAX) that previously overflowed PcgRandom and aborted mcl_core
    let world = Arc::new(WorldStore::new(&dir, 15905940020608058721u64));
    let config = Arc::new(RwLock::new(MinetestConfig::load(&dir.join("minetest.conf"))));

    let rt = LuaRuntime::boot(paths, registries.clone(), world, config)
        .expect("LuaRuntime boot with large u64 seed");
    assert!(
        rt.load_report.errors.is_empty(),
        "Expected zero Lua load errors with large u64 seed, got: {:?}",
        rt.load_report.errors
    );

    let reg = registries.read();
    let palette = MapgenPalette::from_registries(&reg);
    let grass_id = reg.get_content_id("mcl_core:dirt_with_grass");
    let oak_id = reg.get_content_id("mcl_trees:tree_oak");
    let honey_id = reg.get_content_id("mcl_honey:honeycomb_block");

    assert_ne!(grass_id, cubex_rs::registry::CONTENT_UNKNOWN);
    assert_ne!(oak_id, cubex_rs::registry::CONTENT_UNKNOWN);
    assert_eq!(palette.grass, grass_id);
    assert_eq!(palette.oak_log, oak_id);
    assert_ne!(palette.grass, honey_id);

    let _ = std::fs::remove_dir_all(&dir);
}


#[test]
fn entity_register_spawn_objectref_step_physics_and_combat() {
    let paths = EnginePaths::resolve();
    if !paths.game_dir.join("mods").is_dir() {
        return;
    }

    let dir = tmp_dir("entity_combat");
    let world = Arc::new(WorldStore::new(&dir, 777));
    let registries = Arc::new(RwLock::new(EngineRegistries::new()));
    let mut cfg = MinetestConfig::load(&paths.conf_file);
    cfg.set("creative_mode", false);
    cfg.set("enable_damage", true);
    cfg.set("only_peaceful_mobs", false);
    let config = Arc::new(RwLock::new(cfg));
    let runtime = LuaRuntime::boot(paths, registries.clone(), world.clone(), config)
        .expect("LuaRuntime::boot");

    let stone_id = registries.read().get_content_id("mcl_core:stone");
    for z in -4..=8 {
        for x in -4..=8 {
            world.set_node(IVec3::new(x, 10, z), stone_id, 15, 0);
        }
    }
    // 1-block-tall step ledge starting at x = 3
    for z in 0..=4 {
        for x in 3..=6 {
            world.set_node(IVec3::new(x, 11, z), stone_id, 15, 0);
        }
    }

    let player_arc = Arc::new(RwLock::new(ObjectState::new_player("CombatTester")));
    {
        let mut p = player_arc.write();
        p.pos = Vec3::new(1.5, 11.0, 2.5);
        p.hp = 20;
        p.wield_index = 1;
        let mut mg = p.meta.write();
        mg.inventories.get_mut("main").unwrap()[0] =
            ItemStackData::from_spec("mcl_tools:sword_diamond 1");
    }
    runtime.join_player(player_arc.clone()).expect("join_player");

    let walker_id: u64 = runtime
        .lua
        .load(
            r#"
            _G.cubex_test_walker_died = false
            core.register_entity(":cubex_test:walker", {
                initial_properties = {
                    hp_max = 20,
                    physical = true,
                    collisionbox = {-0.3, 0.0, -0.3, 0.3, 1.8, 0.3},
                    stepheight = 1.05,
                    visual = "mesh",
                    textures = {"mobs_mc_zombie.png"},
                    visual_size = {x = 1, y = 1, z = 1},
                    armor_groups = {fleshy = 100},
                },
                on_activate = function(self, staticdata)
                    self.activated_data = staticdata
                    self.step_count = 0
                end,
                on_step = function(self, dtime, moveresult)
                    self.step_count = (self.step_count or 0) + 1
                    self.last_ground = moveresult and moveresult.touching_ground or false
                end,
                on_punch = function(self, puncher, tflp, tool_caps, dir, damage)
                    self.punched_damage = damage
                    self.punched_count = (self.punched_count or 0) + 1
                end,
                on_death = function(self, killer)
                    _G.cubex_test_walker_died = true
                end,
                get_staticdata = function(self)
                    return "walker_saved"
                end,
            })

            local obj = core.add_entity({x = 1.5, y = 12.2, z = 2.5}, "cubex_test:walker", "hello_mob")
            assert(obj ~= nil, "core.add_entity returned nil")
            assert(obj:is_player() == false, "entity is_player must be false")
            local ent = obj:get_luaentity()
            assert(ent ~= nil and ent.activated_data == "hello_mob", "on_activate staticdata missing")
            assert(ent:get_staticdata() == "walker_saved", "get_staticdata mismatch")

            local objs = core.get_objects_inside_radius({x = 1.5, y = 12.2, z = 2.5}, 3.0)
            local saw_ent, saw_player = false, false
            for _, o in ipairs(objs) do
                if o:is_player() then saw_player = true else saw_ent = true end
            end
            assert(saw_ent and saw_player, "get_objects_inside_radius must find both entity and player")

            obj:set_pos({x = 1.9, y = 10.6, z = 2.0})
            assert(math.abs(obj:get_pos().x - 1.9) < 0.01)
            obj:move_to({x = 1.9, y = 10.6, z = 2.0})
            obj:set_velocity({x = 4.0, y = 0.0, z = 0.0})
            obj:add_velocity({x = 0.0, y = 0.0, z = 0.0})
            assert(math.abs(obj:get_velocity().x - 4.0) < 0.01)
            obj:set_acceleration({x = 0.0, y = -9.81, z = 0.0})
            assert(math.abs(obj:get_acceleration().y + 9.81) < 0.01)
            obj:set_yaw(1.25)
            assert(math.abs(obj:get_yaw() - 1.25) < 0.01)
            obj:set_armor_groups({fleshy = 100})
            assert(obj:get_armor_groups().fleshy == 100)
            obj:set_properties({stepheight = 1.05})
            assert(math.abs(obj:get_properties().stepheight - 1.05) < 0.01)
            obj:set_animation({x = 0, y = 40}, 30, 0, true)
            obj:set_hp(20)
            assert(obj:get_hp() == 20)

            _G.cubex_test_walker_obj = obj
            return ent._id
        "#,
        )
        .eval()
        .expect("Lua entity registration & ObjectRef test");

    for _ in 0..20 {
        runtime.step(0.05);
    }

    let (stepped_ok, stepped_x, stepped_y): (bool, f32, f32) = runtime
        .lua
        .load(
            r#"
            local obj = _G.cubex_test_walker_obj
            local ent = obj:get_luaentity()
            local p = obj:get_pos()
            return (ent.step_count == 20 and ent.last_ground == true), p.x, p.y
        "#,
        )
        .eval()
        .expect("check step physics");
    assert!(stepped_ok, "expected 20 steps with touching_ground");
    assert!(
        stepped_x > 3.0 && stepped_y >= 11.45,
        "expected stepheight auto-step up 1-block ledge onto x>=3, y>=11.45 (luanti); got ({stepped_x}, {stepped_y})"
    );

    // Reposition directly ahead of player at (3.5, 12.0, 2.5) and verify raycast + melee combat
    let walker_arc = runtime
        .active_entities()
        .into_iter()
        .find(|e| e.read().id == walker_id)
        .expect("walker active");
    {
        let mut g = walker_arc.write();
        g.pos = Vec3::new(3.5, 12.0, 2.5);
        g.vel = Vec3::ZERO;
    }

    let hit = raycast_entities(
        Vec3::new(1.5, 12.6, 2.5),
        Vec3::X,
        4.0,
        &runtime.active_entities(),
    )
    .expect("raycast_entities must hit walker");
    assert_eq!(hit.entity_id, walker_id);

    let weak = runtime.punch_entity(walker_id, &player_arc, 0.25, Vec3::X);
    let strong = runtime.punch_entity(walker_id, &player_arc, 1.0, Vec3::X);
    assert!(weak.hit && strong.hit);
    assert!(
        strong.damage > weak.damage,
        "full cooldown swing ({}) must out-damage spam swing ({})",
        strong.damage,
        weak.damage
    );
    assert!(
        walker_arc.read().hurt_flash > 0.2,
        "damaged entity should activate 0.25s red hurt flash"
    );
    assert!(
        walker_arc.read().vel.x > 1.0,
        "damaged entity should receive directional knockback"
    );

    // Finish killing walker -> triggers on_death and despawn
    let mut died = strong.killed;
    for _ in 0..4 {
        if died {
            break;
        }
        died = runtime.punch_entity(walker_id, &player_arc, 1.0, Vec3::X).killed;
    }
    assert!(died, "repeated diamond sword hits must kill 20-HP walker");
    let died_flag: bool = runtime.lua.load("return _G.cubex_test_walker_died").eval().unwrap();
    assert!(died_flag, "on_death callback must run when killed");
    assert!(
        !runtime.active_entities().iter().any(|e| e.read().id == walker_id),
        "dead entity must be removed from active_entities"
    );

    // Test mob attacking player (damage + hurt_flash + knockback) & ObjectRef:remove()
    let attacker_id: u64 = runtime
        .lua
        .load(
            r#"
            local attacker = core.add_entity({x = 1.5, y = 11.0, z = 1.0}, "cubex_test:walker")
            local p = core.get_player_by_name("CombatTester")
            assert(p ~= nil and p:is_player() == true)
            p:punch(attacker, 1.0, {full_punch_interval = 1.0, damage_groups = {fleshy = 5}}, {x = 0, y = 0, z = 1})
            local removable = core.add_entity({x = 2.0, y = 11.0, z = 1.0}, "cubex_test:walker")
            local rid = removable:get_luaentity()._id
            removable:remove()
            return rid
        "#,
        )
        .eval()
        .expect("player punch & remove");

    runtime.step(0.05);
    {
        let p = player_arc.read();
        assert!(p.hp < 20, "player HP must drop on mob punch (got {})", p.hp);
        assert!(p.hp_dirty_from_lua, "player HP change must mark dirty for PlayerController sync");
        assert!(p.hurt_flash > 0.2, "player hurt_flash must activate");
        assert!(p.pending_knockback.z > 0.5, "player must receive knockback");
    }
    assert!(
        !runtime.active_entities().iter().any(|e| e.read().id == attacker_id),
        "ObjectRef:remove() entity must be cleaned up"
    );

    let _ = std::fs::remove_dir_all(&dir);
}


#[test]
fn mineclonia_mob_spawn_step_and_punch() {
    let paths = EnginePaths::resolve();
    if !paths.game_dir.join("mods").is_dir() {
        return;
    }

    let dir = tmp_dir("mcl_mob_spawn");
    let world = Arc::new(WorldStore::new(&dir, 9090));
    let registries = Arc::new(RwLock::new(EngineRegistries::new()));
    let mut cfg = MinetestConfig::load(&paths.conf_file);
    cfg.set("creative_mode", false);
    cfg.set("enable_damage", true);
    cfg.set("only_peaceful_mobs", false);
    let config = Arc::new(RwLock::new(cfg));
    let runtime = LuaRuntime::boot(paths, registries.clone(), world.clone(), config)
        .expect("LuaRuntime::boot");

    let stone_id = registries.read().get_content_id("mcl_core:stone");
    for y in 60..=63 {
        for z in -8..=8 {
            for x in -8..=8 {
                world.set_node(IVec3::new(x, y, z), stone_id, 15, 0);
            }
        }
    }

    let player_arc = Arc::new(RwLock::new(ObjectState::new_player("MobFighter")));
    {
        let mut p = player_arc.write();
        p.pos = Vec3::new(0.5, 64.0, 0.5);
        p.hp = 20;
        p.wield_index = 1;
        let mut mg = p.meta.write();
        mg.inventories.get_mut("main").unwrap()[0] =
            ItemStackData::from_spec("mcl_tools:sword_diamond 1");
    }
    runtime.join_player(player_arc.clone()).expect("join_player");

    let (mobs_mc_count, chosen_name, mob_id): (usize, String, u64) = runtime
        .lua
        .load(
            r#"
            local cnt = 0
            local picked = nil
            for name, _ in pairs(core.registered_entities) do
                if type(name) == "string" and name:sub(1, 8) == "mobs_mc:" then
                    cnt = cnt + 1
                    if name == "mobs_mc:zombie" then
                        picked = name
                    elseif not picked then
                        picked = name
                    end
                end
            end
            assert(picked ~= nil, "expected Mineclonia mobs_mc entities registered")
            local obj = core.add_entity({x = 2.0, y = 63.7, z = 0.0}, picked)
            assert(obj ~= nil, "core.add_entity failed for " .. picked)
            assert(obj:is_player() == false)
            local ent = obj:get_luaentity()
            assert(type(ent) == "table", "get_luaentity must return a table")
            return cnt, picked, ent._id
        "#,
        )
        .eval()
        .expect("spawn Mineclonia mob");

    assert!(mobs_mc_count >= 10, "expected >= 10 mobs_mc entities, got {mobs_mc_count}");

    for _ in 0..10 {
        runtime.step(0.05);
    }

    let active = runtime.active_entities();
    let mob_arc = active
        .iter()
        .find(|e| e.read().id == mob_id)
        .cloned()
        .unwrap_or_else(|| panic!("spawned Mineclonia mob {chosen_name} must remain active after stepping"));
    {
        let mg = mob_arc.read();
        assert!(!mg.textures.is_empty(), "Mineclonia mob {chosen_name} should expose textures");
    }

    let mob_snapshot = mob_arc.read().clone();
    let mob_center = mob_snapshot.pos
        + Vec3::new(0.0, (mob_snapshot.collisionbox[1] + mob_snapshot.collisionbox[4]) * 0.5, 0.0);
    let eye = Vec3::new(0.5, 65.2, 0.5);
    let ray_dir = (mob_center - eye).normalize();
    let hit = raycast_entities(eye, ray_dir, 6.0, &active).unwrap_or_else(|| {
        panic!(
            "raycast_entities must hit Mineclonia mob {chosen_name}: pos={:?} vel={:?} cbox={:?} sbox={:?} pointable={} visible={} dist={:.3}",
            mob_snapshot.pos,
            mob_snapshot.vel,
            mob_snapshot.collisionbox,
            mob_snapshot.selectionbox,
            mob_snapshot.pointable,
            mob_snapshot.is_visible,
            eye.distance(mob_center),
        )
    });
    assert_eq!(hit.entity_id, mob_id);

    let punch_res = runtime.punch_entity(mob_id, &player_arc, 0.9, ray_dir);
    assert!(punch_res.hit, "punch_entity must hit Mineclonia mob {chosen_name}");
    assert!(punch_res.damage > 0, "diamond sword should damage Mineclonia mob {chosen_name}");
    assert!(
        punch_res.killed || mob_arc.read().hurt_flash > 0.0,
        "Mineclonia mob must flash red or die on punch"
    );

    let _ = std::fs::remove_dir_all(&dir);
}


#[test]
fn natural_mob_spawning_populates_day_animals_and_night_hostiles() {
    let paths = EnginePaths::resolve();
    if !paths.game_dir.join("mods").is_dir() {
        return;
    }

    let seed = 4242u64;
    let dir = tmp_dir("natural_mobs");
    let world = Arc::new(WorldStore::new(&dir, seed));
    let registries = Arc::new(RwLock::new(EngineRegistries::new()));
    let mut cfg = MinetestConfig::load(&paths.conf_file);
    cfg.set("creative_mode", false);
    cfg.set("enable_damage", true);
    cfg.set("only_peaceful_mobs", false);
    let config = Arc::new(RwLock::new(cfg));
    let runtime = LuaRuntime::boot(paths, registries.clone(), world.clone(), config)
        .expect("LuaRuntime::boot");

    let spawn = find_spawn_pos(seed, WorldGenType::Default);
    let pal = {
        let reg = registries.read();
        MapgenPalette::from_registries(&reg)
    };
    let cx0 = spawn.x.div_euclid(16);
    let cz0 = spawn.z.div_euclid(16);
    let mut keys = Vec::new();
    {
        let reg = registries.read();
        for cz in (cz0 - 2)..=(cz0 + 2) {
            for cx in (cx0 - 2)..=(cx0 + 2) {
                for cy in chunk_y_range_for(WorldGenType::Default) {
                    let cpos = IVec3::new(cx, cy, cz);
                    let ch = generate_chunk_for_world(cpos, seed, WorldGenType::Default, &pal, &reg);
                    world.insert_chunk(ch);
                    keys.push(cpos);
                }
            }
        }
        propagate_world_light(&world, &keys, &reg);
    }

    let player_arc = Arc::new(RwLock::new(ObjectState::new_player("NaturalExplorer")));
    {
        let mut p = player_arc.write();
        p.pos = Vec3::new(spawn.x as f32 + 0.5, spawn.y as f32, spawn.z as f32 + 0.5);
        p.hp = 20;
    }
    runtime.join_player(player_arc).expect("join_player");

    let (box_cnt, luaents_ok): (usize, bool) = runtime
        .lua
        .load(&format!(
            r#"
            local boxes = core.get_node_boxes("collision_box", {{x = {}, y = {}, z = {}}})
            return #boxes, (type(core.luaentities) == "table")
        "#,
            spawn.x,
            spawn.y - 1,
            spawn.z
        ))
        .eval()
        .expect("get_node_boxes & core.luaentities");
    assert!(box_cnt >= 1, "solid ground at spawn should report >= 1 collision box");
    assert!(luaents_ok, "core.luaentities must be exposed as a live table");

    *runtime.shared.time_of_day.write() = 0.35;
    for _ in 0..20 {
        runtime.step(0.05);
    }

    let day_mobs: Vec<_> = runtime
        .active_entities()
        .into_iter()
        .filter(|e| {
            let g = e.read();
            g.entity_name.starts_with("mobs_mc:")
                && g.is_visible
                && g.pointable
                && g.pos.y >= SEA_LEVEL as f32
        })
        .collect();
    assert!(
        day_mobs.len() >= 3,
        "expected >= 3 natural daytime Mineclonia mobs around spawn, got {}",
        day_mobs.len()
    );

    let lua_ent_count: usize = runtime
        .lua
        .load(
            r#"
            local n = 0
            for _, v in pairs(core.luaentities) do
                if v and type(v.name) == "string" and v.name:sub(1, 8) == "mobs_mc:" then
                    n = n + 1
                end
            end
            return n
        "#,
        )
        .eval()
        .unwrap();
    assert!(
        lua_ent_count >= 3,
        "core.luaentities should mirror spawned Mineclonia mobs (got {lua_ent_count})"
    );

    // Nighttime spawning produces hostile Mineclonia monsters
    *runtime.shared.time_of_day.write() = 0.0;
    for _ in 0..10 {
        *runtime.shared.time_of_day.write() = 0.0;
        runtime.step(0.5);
    }

    let hostile_cnt = runtime
        .active_entities()
        .into_iter()
        .filter(|e| {
            let n = e.read().entity_name.to_ascii_lowercase();
            n.contains("zombie")
                || n.contains("skeleton")
                || n.contains("creeper")
                || n.contains("spider")
                || n.contains("husk")
        })
        .count();
    assert!(
        hostile_cnt >= 1,
        "expected >= 1 natural night hostile Mineclonia mob at midnight, got {hostile_cnt}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}


#[test]
fn coop_p1_p2_pvp_combat_and_respawn() {
    let paths = EnginePaths::resolve();
    if !paths.game_dir.join("mods").is_dir() {
        return;
    }

    let dir = tmp_dir("coop_pvp");
    let world = Arc::new(WorldStore::new(&dir, 31415));
    let registries = Arc::new(RwLock::new(EngineRegistries::new()));
    let mut cfg = MinetestConfig::load(&paths.conf_file);
    cfg.set("creative_mode", false);
    cfg.set("enable_damage", true);
    cfg.set("only_peaceful_mobs", false);
    let config = Arc::new(RwLock::new(cfg));
    let runtime = LuaRuntime::boot(paths, registries.clone(), world.clone(), config)
        .expect("LuaRuntime::boot");

    let p1_arc = Arc::new(RwLock::new(ObjectState::new_player("Player1")));
    {
        let mut p1 = p1_arc.write();
        p1.pos = Vec3::new(0.5, 64.0, 0.5);
        p1.hp = 20;
        p1.wield_index = 1;
        let mut mg = p1.meta.write();
        mg.inventories.get_mut("main").unwrap()[0] =
            ItemStackData::from_spec("mcl_tools:sword_diamond 1");
    }
    runtime.join_player(p1_arc.clone()).expect("join P1");

    let p2_arc = Arc::new(RwLock::new(ObjectState::new_player("Player2")));
    {
        let mut p2 = p2_arc.write();
        p2.pos = Vec3::new(2.5, 64.0, 0.5);
        p2.hp = 20;
        p2.wield_index = 1;
        let mut mg = p2.meta.write();
        mg.inventories.get_mut("main").unwrap()[0] =
            ItemStackData::from_spec("mcl_tools:sword_iron 1");
    }
    runtime.join_player(p2_arc.clone()).expect("join P2");

    let hit_p2 = raycast_player(Vec3::new(0.5, 65.62, 0.5), Vec3::X, 4.0, &p2_arc)
        .expect("P1 crosshair should hit P2");
    assert_eq!(hit_p2.entity_name, "Player2");

    let r1 = runtime.punch_player(&p2_arc, &p1_arc, 0.9, Vec3::X);
    assert!(r1.hit && r1.damage > 0, "P1 diamond sword punch must damage P2: {r1:?}");
    {
        let p2 = p2_arc.read();
        assert!(p2.hp < 20, "P2 HP must drop after punch, got {}", p2.hp);
        assert!(p2.hp_dirty_from_lua, "P2 hp_dirty_from_lua must trigger");
        assert!(p2.hurt_flash > 0.2, "P2 hurt_flash must activate");
        assert!(p2.pending_knockback.x > 1.0, "P2 must receive +X knockback");
    }

    let hit_p1 = raycast_player(Vec3::new(2.5, 65.62, 0.5), -Vec3::X, 4.0, &p1_arc)
        .expect("P2 crosshair should hit P1");
    assert_eq!(hit_p1.entity_name, "Player1");

    let r2 = runtime.punch_player(&p1_arc, &p2_arc, 0.9, -Vec3::X);
    assert!(r2.hit && r2.damage > 0, "P2 iron sword punch must damage P1: {r2:?}");
    {
        let p1 = p1_arc.read();
        assert!(p1.hp < 20, "P1 HP must drop after punch, got {}", p1.hp);
        assert!(p1.hurt_flash > 0.2, "P1 hurt_flash must activate");
        assert!(p1.pending_knockback.x < -1.0, "P1 must receive -X knockback");
    }

    // Disabling PvP gamerule blocks player-vs-player damage cleanly
    *world.gamerules.write() = Gamerules {
        pvp: false,
        ..Default::default()
    };
    let hp_before = p2_arc.read().hp;
    let r_no_pvp = runtime.punch_player(&p2_arc, &p1_arc, 1.0, Vec3::X);
    assert!(!r_no_pvp.hit && r_no_pvp.damage == 0);
    assert_eq!(p2_arc.read().hp, hp_before);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn mineclonia_mobs_step_without_lua_errors_and_move() {
    let paths = EnginePaths::resolve();
    if !paths.game_dir.join("mods").is_dir() {
        return;
    }

    let dir = tmp_dir("mcl_mobs_move");
    let world = Arc::new(WorldStore::new(&dir, 777));
    let registries = Arc::new(RwLock::new(EngineRegistries::new()));
    let mut cfg = MinetestConfig::load(&paths.conf_file);
    cfg.set("creative_mode", false);
    cfg.set("enable_damage", true);
    cfg.set("only_peaceful_mobs", false);
    let config = Arc::new(RwLock::new(cfg));
    let runtime = LuaRuntime::boot(paths, registries.clone(), world.clone(), config)
        .expect("LuaRuntime::boot");

    let grass_id = registries.read().get_content_id("mcl_core:dirt_with_grass");
    let stone_id = registries.read().get_content_id("mcl_core:stone");
    for z in -20..=20 {
        for x in -20..=20 {
            for y in 60..=62 {
                world.set_node(IVec3::new(x, y, z), stone_id, 15, 0);
            }
            world.set_node(IVec3::new(x, 63, z), grass_id, 15, 0);
            for y in 64..=72 {
                world.set_node(IVec3::new(x, y, z), cubex_rs::registry::CONTENT_AIR, 15, 0);
            }
        }
    }

    // Verify find_nodes_in_area_under_air ONLY returns nodes with air above
    let under_air_ok: bool = runtime
        .lua
        .load(
            r#"
            local nodes = core.find_nodes_in_area_under_air(
                {x = -5, y = 60, z = -5},
                {x = 5, y = 65, z = 5},
                {"group:solid"}
            )
            if #nodes == 0 then return false end
            for _, p in ipairs(nodes) do
                if p.y ~= 63 then return false end
            end
            return true
        "#,
        )
        .eval()
        .expect("find_nodes_in_area_under_air check");
    assert!(under_air_ok, "find_nodes_in_area_under_air must only return surface nodes under air");

    let player_arc = Arc::new(RwLock::new(ObjectState::new_player("Runner")));
    player_arc.write().pos = Vec3::new(0.5, 64.0, 0.5);
    runtime.join_player(player_arc.clone()).expect("join_player");
    runtime.set_gamemode(player_arc.clone(), "survival");

    let species = [
        ("mobs_mc:cow", Vec3::new(4.0, 63.55, 4.0)),
        ("mobs_mc:pig", Vec3::new(-4.0, 63.55, 4.0)),
        ("mobs_mc:sheep", Vec3::new(5.0, 63.55, -3.0)),
        ("mobs_mc:chicken", Vec3::new(-5.0, 63.55, -4.0)),
        ("mobs_mc:wolf", Vec3::new(7.0, 63.55, 1.0)),
        ("mobs_mc:rabbit", Vec3::new(-6.0, 63.55, 1.0)),
        ("mobs_mc:zombie", Vec3::new(9.0, 63.55, 0.0)),
    ];
    let mut spawned: Vec<(&str, u64, Vec3)> = Vec::new();
    for &(name, pos) in &species {
        let eid: u64 = runtime
            .lua
            .load(
                r#"
                local name, x, y, z = ...
                local obj = core.add_entity({x = x, y = y, z = z}, name)
                if not obj then return 0 end
                local ent = obj:get_luaentity()
                if ent and name == "mobs_mc:cow" then
                    ent._pace_asap = true
                end
                return ent and ent._id or 0
            "#,
            )
            .call((name, pos.x, pos.y, pos.z))
            .expect("spawn mob");
        assert!(eid > 0, "failed to spawn {name}");
        spawned.push((name, eid, pos + Vec3::splat(0.5)));
    }

    // Set night time so zombie does not burn/flee sun, then step 160 ticks = 8.0s
    *runtime.shared.time_of_day.write() = 0.0;
    for _ in 0..160 {
        runtime.step(0.05);
    }

    let step_errs = runtime
        .entity_step_error_count
        .load(std::sync::atomic::Ordering::Relaxed);
    let err_msgs = runtime.entity_step_errors.lock().clone();
    assert_eq!(
        step_errs, 0,
        "expected ZERO Lua on_step errors across Mineclonia mobs, got {step_errs}: {err_msgs:#?}"
    );

    let active = runtime.active_entities();
    let mut max_passive_move = 0.0f32;
    for &(name, eid, init_pos) in &spawned {
        if let Some(earc) = active.iter().find(|e| e.read().id == eid) {
            let g = earc.read();
            let disp = Vec3::new(g.pos.x - init_pos.x, 0.0, g.pos.z - init_pos.z).length();
            if name == "mobs_mc:zombie" {
                let dbg: String = runtime
                    .lua
                    .load(
                        r#"
                        local eid = ...
                        local e = core._cubex_luaentities[eid]
                        if not e then return "missing" end
                        local v = e.object:get_velocity()
                        local n = #e._targeting_rules
                        return string.format("act=%s target=%s attack=%s goal=%s wp=%s pctx=%s proto=%s on=%s in=%s acc=%.2f vel=(%.2f,%.2f,%.2f) nrules=%d",
                            tostring(e._active_activity), tostring(e._active_target), tostring(e.attack),
                            tostring(e.movement_goal), tostring(e.waypoints and #e.waypoints or nil),
                            tostring(e.pathfinding_context ~= nil), tostring(e._in_proto_chunk),
                            tostring(e.standing_on), tostring(e.standing_in),
                            tonumber(e.acc_speed or 0), v.x, v.y, v.z, n)
                    "#,
                    )
                    .call(eid)
                    .unwrap_or_default();
                let d0 = init_pos.distance(Vec3::new(0.5, 64.0, 0.5));
                let d1 = g.pos.distance(Vec3::new(0.5, 64.0, 0.5));
                assert!(
                    d1 < d0 - 0.5,
                    "zombie must chase nearby player (d0={d0:.2} -> d1={d1:.2}, pos={:?}, dbg={dbg})",
                    g.pos
                );
            } else {
                max_passive_move = max_passive_move.max(disp);
            }
        }
    }
    assert!(
        max_passive_move > 0.35,
        "at least some passive mobs should pace/wander cleanly within 8s (max_disp={max_passive_move:.3})"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn physical_drops_rotate_and_magnetically_pull_to_player() {
    let dir = tmp_dir("phys_drops");
    let world = Arc::new(WorldStore::new(&dir, 5150));
    let registries = Arc::new(RwLock::new(EngineRegistries::new()));
    let stone_id = {
        let mut reg = registries.write();
        let mut stone_groups = HashMap::new();
        stone_groups.insert("cracky".into(), 3);
        let st_id = reg.register_node(NodeDef {
            name: "mcl_core:stone".into(),
            drawtype: DrawType::Normal,
            walkable: true,
            diggable: true,
            groups: stone_groups,
            ..Default::default()
        });
        let mut times = HashMap::new();
        times.insert(3, 0.2);
        let mut groupcaps = HashMap::new();
        groupcaps.insert(
            "cracky".into(),
            ToolGroupCap {
                times,
                uses: 100,
                maxlevel: 2,
            },
        );
        reg.register_item(ItemDef {
            name: "mcl_tools:pick_stone".into(),
            tool_capabilities: Some(ToolCapabilities {
                full_punch_interval: 1.0,
                max_drop_level: 1,
                groupcaps,
                damage_groups: HashMap::new(),
                punch_attack_uses: 0,
            }),
            ..Default::default()
        });
        st_id
    };

    // Build a solid 7x7 stone floor at y = 9 (top = 10.0) and a target block at (2, 10, 2)
    for x in -2..=6 {
        for z in -2..=6 {
            world.set_node(IVec3::new(x, 9, z), stone_id, 15, 0);
        }
    }
    world.set_node(IVec3::new(2, 10, 2), stone_id, 15, 0);

    let mut paths = EnginePaths::resolve();
    paths.game_dir = dir.join("no_game");
    let cfg = Arc::new(RwLock::new(MinetestConfig::load(&paths.conf_file)));
    let runtime = LuaRuntime::boot(paths, registries.clone(), world.clone(), cfg).expect("boot");

    let pstate = Arc::new(RwLock::new(ObjectState::new_player("MagnetTester")));
    {
        let ps = pstate.write();
        ps.meta.write().inventories.get_mut("main").unwrap()[0] =
            ItemStackData::from_spec("mcl_tools:pick_stone");
    }
    runtime.join_player(pstate.clone()).expect("join player");

    let mut player = PlayerController::default();
    player.gamemode = GameMode::Survival;
    player.pos = Vec3::new(0.5, 10.0, 0.5);
    player.selected_slot = 0;

    let hit_dir = (Vec3::new(2.5, 10.5, 2.5) - player.eye_pos()).normalize();
    let hit = raycast_voxel(player.eye_pos(), hit_dir, 6.0, &world, &registries.read()).unwrap();
    let mut digger = DigController::new();
    let mut comp = None;
    for _ in 0..10 {
        let res = digger.update_with_physical_drops(
            0.05,
            true,
            Some(&hit),
            &world,
            &registries.read(),
            &player,
            &pstate,
            Some(&runtime),
            true,
        );
        if let Some(c) = res.completed_dig {
            comp = Some(c);
            break;
        }
    }
    let comp = comp.expect("target stone should break");
    assert!(!comp.picked_up_any, "physical drop should spawn in-world first");

    let items_before: Vec<_> = runtime
        .active_entities()
        .into_iter()
        .filter(|e| e.read().entity_name == "__builtin:item")
        .collect();
    assert_eq!(items_before.len(), 1, "expected 1 physical dropped item entity");
    let drop_arc = items_before[0].clone();
    let yaw0 = drop_arc.read().yaw;

    // Place player far enough away (> 3 blocks) while item pops, lands on floor, and spins
    pstate.write().pos = Vec3::new(-1.5, 10.0, -1.5);
    for _ in 0..15 {
        runtime.step(0.05);
    }
    assert!(!drop_arc.read().removed, "drop outside magnet radius stays in world");
    let yaw1 = drop_arc.read().yaw;
    assert!((yaw1 - yaw0).abs() > 0.05, "dropped item should rotate");
    assert!(
        drop_arc.read().pos.y >= 10.0 - 0.05,
        "dropped item should rest on solid floor y=10, got {}",
        drop_arc.read().pos.y
    );

    // Move player within magnetic range (2.1 blocks) and step until pulled in & collected
    let drop_pos = drop_arc.read().pos;
    pstate.write().pos = drop_pos + Vec3::new(2.05, 0.0, 0.0);
    let mut moved_closer = false;
    for _ in 0..30 {
        runtime.step(0.05);
        if drop_arc.read().removed {
            moved_closer = true;
            break;
        }
        let d_now = drop_arc
            .read()
            .pos
            .distance(pstate.read().pos + Vec3::new(0.0, 0.85, 0.0));
        if d_now < 1.9 {
            moved_closer = true;
        }
    }
    assert!(moved_closer, "dropped item should magnetically accelerate toward nearby player");
    assert!(drop_arc.read().removed, "magnetic drop should be collected on contact");

    let has_stone = pstate
        .read()
        .meta
        .read()
        .inventories["main"]
        .iter()
        .any(|s| s.name == "mcl_core:stone" && s.count >= 1);
    assert!(has_stone, "collected physical item drop must appear in player inventory");

    let _ = std::fs::remove_dir_all(&dir);
}
