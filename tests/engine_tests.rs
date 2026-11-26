use cubex_rs::config::{EngineSettings, GraphicsQuality, MinetestConfig};
use cubex_rs::network::invite::{decode_invite, encode_invite};
use cubex_rs::network::motd::{parse_motd, strip_motd_codes};
use cubex_rs::network::protocol::{connect_client, start_server, NetPacket};
use cubex_rs::network::servers::ping_server;
use cubex_rs::network::tunnel::PlayitTunnelManager;
use cubex_rs::paths::EnginePaths;
use cubex_rs::physics::{raycast_voxel, MoveInput, PlayerController};
use cubex_rs::registry::{DrawType, EngineRegistries, NodeDef, CONTENT_AIR};
use cubex_rs::render::frustum::Frustum;
use cubex_rs::render::mesher::mesh_chunk_greedy;
use cubex_rs::ui::formspec::{parse_formspec, FormElement};
use cubex_rs::world::light::propagate_chunk_light;
use cubex_rs::world::{Chunk, WorldStore};

use glam::{IVec3, Mat4, Vec3};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::os::unix::fs::PermissionsExt;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::UdpSocket;

#[test]
fn chunk_voxelmanip_and_lighting_roundtrip() {
    let mut reg = EngineRegistries::new();
    let stone_id = reg.register_node(NodeDef {
        name: "mcl_core:stone".into(),
        drawtype: DrawType::Normal,
        ..Default::default()
    });
    let torch_id = reg.register_node(NodeDef {
        name: "mcl_torches:torch".into(),
        drawtype: DrawType::Torchlike,
        walkable: false,
        sunlight_propagates: true,
        light_source: 14,
        ..Default::default()
    });

    let mut chunk = Chunk::new(IVec3::ZERO);
    chunk.set(8, 2, 8, stone_id, 0, 0);
    chunk.set(8, 3, 8, torch_id, 0, 0);
    propagate_chunk_light(&mut chunk, &reg, true);

    assert_eq!(chunk.get(8, 2, 8).0, stone_id);
    // Neighboring air block next to torch receives propagated artificial light (> 10)
    let (_, p1, _) = chunk.get(9, 3, 8);
    let block_light = p1 >> 4;
    assert!(block_light >= 12, "Expected propagated torchlight >= 12, got {block_light}");
}

#[test]
fn greedy_meshing_reduces_quad_count() {
    let mut reg = EngineRegistries::new();
    let stone_id = reg.register_node(NodeDef {
        name: "mcl_core:stone".into(),
        drawtype: DrawType::Normal,
        ..Default::default()
    });
    let tmp = std::env::temp_dir().join("cubex_rs_mesh_test");
    let world = WorldStore::new(tmp, 42);
    let mut chunk = Chunk::new(IVec3::ZERO);
    for z in 0..16 {
        for x in 0..16 {
            chunk.set(x, 0, z, stone_id, 15, 0);
        }
    }
    world.insert_chunk(chunk);

    let snap = world.snapshot_chunk(IVec3::ZERO).unwrap();
    let mut layers = HashMap::new();
    layers.insert(stone_id, [1; 6]);
    let mesh = mesh_chunk_greedy(&snap, &reg, &layers, false);

    // Naive meshing of 16x16 flat slab = 256*2 (top+bottom) + 64 side quads = 576 quads = 2304 verts
    // Greedy meshing merges the entire 16x16 top & bottom faces into single quads -> vastly fewer vertices!
    let quad_count = mesh.opaque_indices.len() / 6;
    assert!(quad_count < 32, "Expected greedy-merged quad count < 32, got {quad_count}");
}

#[test]
fn frustum_culling_accepts_visible_and_rejects_behind() {
    let proj = Mat4::perspective_rh(70.0f32.to_radians(), 16.0 / 9.0, 0.1, 200.0);
    let view = Mat4::look_at_rh(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0), Vec3::Y);
    let frustum = Frustum::from_view_proj(proj * view);

    assert!(frustum.intersects_aabb(Vec3::new(-4.0, -4.0, -32.0), Vec3::new(4.0, 4.0, -16.0)));
    assert!(!frustum.intersects_aabb(Vec3::new(-4.0, -4.0, 16.0), Vec3::new(4.0, 4.0, 32.0)));
}

#[test]
fn physics_collision_and_dda_raycast() {
    let mut reg = EngineRegistries::new();
    let stone_id = reg.register_node(NodeDef {
        name: "mcl_core:stone".into(),
        drawtype: DrawType::Normal,
        walkable: true,
        pointable: true,
        ..Default::default()
    });
    let tmp = std::env::temp_dir().join("cubex_rs_phys_test");
    let world = WorldStore::new(tmp, 1);
    for z in -2..=2 {
        for x in -2..=2 {
            world.set_node(IVec3::new(x, 0, z), stone_id, 15, 0);
        }
    }

    let mut player = PlayerController::default();
    player.pos = Vec3::new(0.5, 2.0, 0.5);
    let settings = EngineSettings::default();
    let input = MoveInput::default();
    for _ in 0..40 {
        player.step(0.025, &input, &world, &reg, &settings);
    }
    assert!(player.on_ground);
    assert!(player.pos.y >= 0.99 && player.pos.y <= 1.05);

    // Raycast straight down from player eye hits block at y=0
    let hit = raycast_voxel(player.eye_pos(), Vec3::NEG_Y, 5.0, &world, &reg).expect("raycast hit");
    assert_eq!(hit.pos.y, 0);
    assert_eq!(hit.node_id, stone_id);
}

#[test]
fn formspec_parser_parses_elements() {
    let fs = "size[9,8.5]box[0,0;9,8.5;#C6C6C6]label[0.4,0.3;Chest]field[1,1;3,0.8;q;Search;iron]list[current_player;main;0,4;9,4;]button_exit[3,7;3,0.9;ok;Done]";
    let parsed = parse_formspec(fs);
    assert_eq!(parsed.size, [9.0, 8.5]);
    assert!(parsed.elements.iter().any(|e| matches!(e, FormElement::InvList { w: 9, h: 4, .. })));
    assert!(parsed.elements.iter().any(|e| matches!(e, FormElement::Button { exit: true, .. })));
    assert!(parsed.elements.iter().any(|e| matches!(e, FormElement::Field { name, .. } if name == "q")));
}

#[test]
fn motd_and_invite_codec_work() {
    let spans = parse_motd("§a§lMineclonia §r§7- §eSurvival");
    assert_eq!(spans.len(), 3);
    assert!(spans[0].bold);
    assert_eq!(strip_motd_codes("§a§lMineclonia §r§7- §eSurvival"), "Mineclonia - Survival");

    let code = encode_invite("emerald-dragon.ply.gg", 44123).unwrap();
    assert!(code.starts_with("CBX-"));
    let (host, port) = decode_invite(&code).unwrap();
    assert_eq!(host, "emerald-dragon.ply.gg");
    assert_eq!(port, 44123);
    assert_eq!(
        decode_invite("127.0.0.1").unwrap(),
        ("127.0.0.1".into(), cubex_rs::network::DEFAULT_LAN_PORT)
    );
    assert_eq!(cubex_rs::network::DEFAULT_LAN_PORT, 8090);
}

#[test]
fn tokio_pinger_and_dedicated_protocol_handshake() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        // 1. UDP pinger test
        let sock = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let udp_port = sock.local_addr().unwrap().port();
        tokio::spawn(async move {
            let mut buf = [0u8; 128];
            if let Ok((_n, addr)) = sock.recv_from(&mut buf).await {
                let _ = sock.send_to(b"\x4f\x45\x74\x03\x00\x01\x00\x00\x00", addr).await;
            }
        });
        let (on, ms) = ping_server("127.0.0.1", udp_port, Duration::from_secs(1)).await;
        assert!(on);
        assert!(ms.is_some());

        // 2. TCP Dedicated Server handshake test
        let tmp = std::env::temp_dir().join("cubex_rs_net_world");
        let world = Arc::new(WorldStore::new(tmp, 777));
        let reg = Arc::new(RwLock::new(EngineRegistries::new()));
        let (tcp_port, _) = start_server(0, world, reg).await.unwrap();
        let reply = connect_client("127.0.0.1", tcp_port, "Tester").await.unwrap();
        match reply {
            NetPacket::Welcome { seed, .. } => assert_eq!(seed, 777),
            other => panic!("Unexpected packet: {other:?}"),
        }
    });
}

#[test]
fn playit_tunnel_lifecycle_and_zero_orphans() {
    let dir = std::env::temp_dir().join("cubex_rs_playit_mock");
    let _ = std::fs::create_dir_all(&dir);
    let mock_bin = dir.join("mock_playit.sh");
    std::fs::write(
        &mock_bin,
        "#!/bin/sh\necho 'Visit https://playit.gg/claim/testclaim99 to link'\necho 'tunnel active: voxel-test.gl.at.ply.gg:45123 -> 127.0.0.1:30000'\nsleep 30\n",
    )
    .unwrap();
    std::fs::set_permissions(&mock_bin, std::fs::Permissions::from_mode(0o755)).unwrap();

    let mgr = PlayitTunnelManager::new(
        &mock_bin,
        dir.join("playit.toml"),
        dir.join("playit.log"),
    );
    assert_eq!(mgr.build_args()[0], "--secret-path");
    mgr.start().unwrap();
    let pid = mgr.pid().unwrap();
    assert!(mgr.is_running());

    for _ in 0..30 {
        if !mgr.status.lock().public_address.is_empty() {
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }

    let st = mgr.status.lock().clone();
    assert_eq!(st.claim_url, "https://playit.gg/claim/testclaim99");
    assert_eq!(st.public_address, "voxel-test.gl.at.ply.gg:45123");
    assert_eq!(decode_invite(&st.invite_code).unwrap(), ("voxel-test.gl.at.ply.gg".into(), 45123));

    mgr.stop();
    assert!(!mgr.is_running());
    // Confirm child process no longer exists
    let proc_dir = std::path::PathBuf::from(format!("/proc/{pid}"));
    assert!(!proc_dir.exists(), "Orphan playit process left behind: {pid}");
}

#[test]
fn settings_and_minetest_conf_sync() {
    let dir = std::env::temp_dir().join("cubex_rs_conf_test");
    let _ = std::fs::create_dir_all(&dir);
    let conf_path = dir.join("minetest.conf");
    std::fs::write(&conf_path, "fov = 70\nfov = 85\nviewing_range = 160\n").unwrap();

    let mut cfg = MinetestConfig::load(&conf_path);
    assert_eq!(cfg.get("fov"), Some("85"));
    let mut es = cfg.to_engine_settings("");
    es.fov = 100.0;
    es.render_distance_chunks = 16;
    es.graphics = GraphicsQuality::Fabulous;
    cfg.apply_engine_settings(&es);
    cfg.save().unwrap();

    let saved = std::fs::read_to_string(&conf_path).unwrap();
    assert_eq!(saved.lines().filter(|l| l.starts_with("fov =")).count(), 1);
    assert_eq!(cfg.get("viewing_range"), Some("256"));
    assert_eq!(cfg.get("enable_bloom"), Some("true"));

    let _ = CONTENT_AIR;
    let _ = EnginePaths::resolve();
}
