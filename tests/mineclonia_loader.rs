use cubex_rs::config::MinetestConfig;
use cubex_rs::lua_api::LuaRuntime;
use cubex_rs::paths::EnginePaths;
use cubex_rs::registry::{EngineRegistries, CONTENT_UNKNOWN};
use cubex_rs::textures::TextureAtlas;
use cubex_rs::world::WorldStore;
use parking_lot::RwLock;
use std::sync::Arc;

#[test]
fn milestone1_loads_mineclonia_and_craft_ruin_textures() {
    let paths = EnginePaths::resolve();
    assert!(
        paths.game_dir.join("mods").is_dir(),
        "Mineclonia game dir not found at {}",
        paths.game_dir.display()
    );
    assert!(
        paths.builtin_dir.join("init.lua").is_file(),
        "Luanti builtin/init.lua not found at {}",
        paths.builtin_dir.display()
    );

    let registries = Arc::new(RwLock::new(EngineRegistries::new()));
    let tmp_world = std::env::temp_dir().join(format!("cubex_rs_m1_world_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp_world);
    let world = Arc::new(WorldStore::new(&tmp_world, 1337));
    let config = Arc::new(RwLock::new(MinetestConfig::load(&paths.conf_file)));

    let runtime = LuaRuntime::boot(paths.clone(), registries.clone(), world, config)
        .expect("LuaRuntime boot failed");

    println!(
        "Mods discovered: {}, loaded: {}, errors: {}",
        runtime.load_report.mods_discovered,
        runtime.load_report.mods_loaded,
        runtime.load_report.errors.len()
    );
    for (m, err) in &runtime.load_report.errors {
        println!("MOD ERROR [{}]: {}", m, err);
    }

    assert_eq!(
        runtime.load_report.mods_discovered, 224,
        "Expected 224 Mineclonia mods discovered, got {}",
        runtime.load_report.mods_discovered
    );
    assert_eq!(
        runtime.load_report.mods_loaded, 224,
        "Expected 224/224 Mineclonia mods loaded, got {}",
        runtime.load_report.mods_loaded
    );
    assert_eq!(
        runtime.load_report.errors.len(),
        0,
        "Expected 0 Lua mod load errors, got {:?}",
        runtime.load_report.errors
    );

    let reg = registries.read();
    println!(
        "Registered nodes: {}, items: {}, crafts: {}, biomes: {}",
        reg.node_count(),
        reg.items.len(),
        reg.crafts.len(),
        reg.biomes.len()
    );

    assert!(reg.node_count() > 1000, "Expected >1000 nodes, got {}", reg.node_count());
    assert!(reg.items.len() > 1500, "Expected >1500 items, got {}", reg.items.len());
    assert!(reg.crafts.len() > 500, "Expected >500 crafts, got {}", reg.crafts.len());
    assert!(reg.biomes.len() > 10, "Expected >10 biomes, got {}", reg.biomes.len());

    let stone_id = reg.get_content_id("mcl_core:stone");
    let grass_id = reg.get_content_id("mcl_core:dirt_with_grass");
    let water_id = reg.get_content_id("mcl_core:water_source");
    let torch_id = reg.get_content_id("mcl_torches:torch");
    let fence_id = reg.get_content_id("mcl_fences:oak_fence");
    let leaves_id = reg.get_content_id("mcl_trees:leaves_oak");
    assert_ne!(stone_id, CONTENT_UNKNOWN);
    assert_ne!(grass_id, CONTENT_UNKNOWN);
    assert_ne!(water_id, CONTENT_UNKNOWN);
    assert_ne!(torch_id, CONTENT_UNKNOWN);
    assert_ne!(fence_id, CONTENT_UNKNOWN);
    assert_ne!(leaves_id, CONTENT_UNKNOWN);

    let fence = reg.get_node_def(fence_id).expect("oak fence def");
    assert_eq!(fence.node_box_type, "connected");
    assert!(!fence.connects_to.is_empty());
    assert!(!fence.selection_boxes.is_empty());
    assert!(!fence.collision_boxes.is_empty());

    let leaves = reg.get_node_def(leaves_id).expect("oak leaves def");
    assert!(leaves.palette.is_some());

    let pick = reg
        .get_item_def("mcl_tools:pick_diamond")
        .expect("diamond pick item def");
    assert!(
        pick.tool_capabilities
            .as_ref()
            .map(|tc| !tc.groupcaps.is_empty())
            .unwrap_or(false),
        "diamond pick must have groupcaps"
    );

    let pack_dir = paths.textures_dir.join("craft_and_ruin");
    let atlas = TextureAtlas::build(
        &runtime.shared.mod_paths.read(),
        &paths.game_dir,
        Some(pack_dir.as_path()),
        &reg,
    );
    println!("Packed texture array layers: {}", atlas.layers.len());
    assert!(atlas.layers.len() > 100 && atlas.layers.len() <= 2048);
    let stone_faces = atlas.node_face_layers.get(&stone_id).expect("stone textures missing");
    assert!(stone_faces[0] > 0);
    assert!(atlas.get_item_rgba("mcl_tools:pick_diamond", &reg).is_some());
    assert!(atlas.get_item_rgba("mcl_core:stone", &reg).is_some());
    let cracks = atlas.get_crack_layers();
    assert!(cracks.iter().all(|&l| l > 0));
    let skin = atlas.get_player_skin_rgba();
    assert!(skin.width() >= 64 && skin.height() >= 32);

    let _ = std::fs::remove_dir_all(&tmp_world);
}
