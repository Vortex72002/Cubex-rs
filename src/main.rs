use cubex_rs::audio::{AudioEngine, SoundCategory};
use cubex_rs::autotest::{
    parse_autotest_args, AutoTestConfig, AutoTestReport, SurvivalNetherBot, SURVIVAL_NETHER_SEED,
};
use cubex_rs::qa::bot::{run_headless_playthrough, PlaythroughReport, QaPlaythroughBot};
use cubex_rs::qa::fuzzer::run_physics_fuzzer;
use cubex_rs::qa::static_audit::run_static_audit;
use cubex_rs::qa::visual::run_visual_regression;
use cubex_rs::qa::{
    parse_qa_args, unix_timestamp_now, write_master_report, QaCliCommand, QaMasterReport,
};
use cubex_rs::config::{
    AudioSettings, CoopMode, EngineSettings, GraphicsQuality, MinetestConfig, PlayerInputDevice,
};
use cubex_rs::input::{GamepadManager, TouchControlsState, TouchEventPhase};
use cubex_rs::lua_api::userdata::{ItemStackData, ObjectState};
use cubex_rs::lua_api::{
    discover_and_sort_mods, drain_player_inventory_for_death, format_death_cause,
    LuaIncrementalBoot, LuaRuntime, ModSpec,
};
use cubex_rs::network::lan::{local_lan_ipv4, LanBeaconInfo, LanBroadcaster, DEFAULT_LAN_PORT};
use cubex_rs::network::protocol::{
    decode_chunk_data, run_dedicated_server, start_server_with_handle, LocalServerHandle,
    NetClientSession, NetPacket, RemotePlayerState,
};
use cubex_rs::network::tunnel::PlayitTunnelManager;
use cubex_rs::paths::EnginePaths;
use cubex_rs::physics::raycast::held_item_is_empty_bucket;
use cubex_rs::physics::{
    add_drops_to_inventory, log_player_damage, raycast_entities, raycast_player, raycast_voxel_ex,
    selected_world_boxes, targeted_node_has_interaction, try_eat_held_food,
    try_interact_or_place, DamageSource, DigController, GameMode, MoveInput, PlayerController,
    RaycastHit,
};
use cubex_rs::registry::{
    DrawType, EngineRegistries, CONTENT_AIR, CONTENT_IGNORE, CONTENT_UNKNOWN, RAM_MIC_ITEM_ID,
};
use cubex_rs::render::camera::{compute_camera_rig, CameraMode};
use cubex_rs::render::mesher::{
    mesh_chunk_with_atlas, mesh_chunks_parallel_with_atlas, ChunkMeshData,
};
use cubex_rs::render::{
    EntityRenderSnapshot, RemotePlayerRenderSnapshot, SecondaryRenderTarget, ViewportCamera,
    WgpuRenderer,
};
use cubex_rs::textures::{
    is_carterpcs_skin, list_available_skins, list_resource_packs, resolve_skin_rgba,
    PreloadedTextures, ResourcePackInfo, SkinInfo, TextureAtlas,
};
use cubex_rs::ui::creative_inventory::{
    render_creative_inventory, CreativeInventoryState, CREATIVE_INVENTORY_FORMNAME,
};
use cubex_rs::ui::death_screen::{render_death_screen, DeathScreenAction};
use cubex_rs::ui::formspec::{
    build_loot_chest_formspec, ensure_craft_grid_size, formspec_requires_3x3_craft, parse_formspec,
    refresh_craft_preview, render_formspec_interactive, resolve_inv_node_pos, shift_transfer_slot,
    ControllerFormInput, FormspecContext, InventoryCursor, ParsedFormspec,
};
use cubex_rs::ui::hud::{render_hud_full, render_hud_in_rect, HudTelemetry};
use cubex_rs::ui::loading_screen::{render_loading_screen, LoadingScreenInfo};
use cubex_rs::ui::pause_menu::{render_pause_menu, PauseAction, PauseMenuState, PausePage};
use cubex_rs::ui::splash_screen::{render_splash_screen, SplashScreenInfo};
use cubex_rs::ui::theme::{ItemIconCache, McTheme};
use cubex_rs::ui::title_screen::{render_title_screen_with_atlas, TitleAction, TitleScreenState};
use cubex_rs::ui::touch_hud::render_touch_hud;
use cubex_rs::world::light::{propagate_world_light, relight_around_pos};
use cubex_rs::world::mapgen::{
    biome_name_at, chunk_y_range_for, find_spawn_pos, generate_chunk_for_world,
    terrain_height_for, MapgenPalette, SEA_LEVEL,
};
use cubex_rs::world::mobs::{
    MeleeNativeResult, MobManager, MobParticleKind, MobPlayerInput,
};
use cubex_rs::world::nether::{
    ensure_nether_columns, generate_nether_chunk, nether_biome_name_at, NetherPalette,
    NETHER_CHUNK_Y_RANGE,
};
use cubex_rs::world::persistence::{
    load_dimension_chunks, load_player_by_name, load_world, save_dimension_chunks, save_world,
    save_world_players,
};
use cubex_rs::world::portal::{
    extinguish_portal_around, find_or_build_destination_portal, is_nether_portal_node, PortalAxis,
};
use cubex_rs::world::schematic::{PlaceFlags, Schematic, SchematicRotation};
use cubex_rs::world::structures::{
    locate_nearest_structure, populate_structure_chests_for_chunk, StructureKind,
};
use cubex_rs::world::{
    scale_portal_coords, Chunk, Difficulty, DimensionId, Gamerules, WorldGenType, WorldInfo,
    WorldStore,
};

use glam::{IVec3, Vec3};
use mlua::ObjectLike;
use parking_lot::RwLock;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};

use winit::application::ApplicationHandler;
use winit::dpi::{LogicalPosition, LogicalSize};
use winit::event::{
    DeviceEvent, ElementState, KeyEvent, MouseButton, MouseScrollDelta, TouchPhase, WindowEvent,
};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Fullscreen, Window, WindowId};

const PANORAMA_RADIUS: i32 = 7;
const PANORAMA_Y_MIN: i32 = 2;
const PANORAMA_Y_MAX: i32 = 7;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AppMode {
    Splash,
    Title,
    Loading,
    Playing,
    Paused,
}

struct PanoramaReadyBundle {
    atlas: Arc<TextureAtlas>,
    palette: MapgenPalette,
    generated_columns: HashSet<(i32, i32)>,
    meshes: Vec<ChunkMeshData>,
    panorama_center: Vec3,
}

enum StartupBootState {
    WaitingForFirstPaint {
        started_at: Instant,
        mods_rx: Receiver<Vec<ModSpec>>,
        assets_rx: Receiver<(PreloadedTextures, Arc<AudioEngine>)>,
    },
    LoadingMods {
        started_at: Instant,
        boot: Option<LuaIncrementalBoot>,
        mods_rx: Receiver<Vec<ModSpec>>,
        assets_rx: Receiver<(PreloadedTextures, Arc<AudioEngine>)>,
        preloaded_assets: Option<(PreloadedTextures, Arc<AudioEngine>)>,
        loaded: usize,
        total: usize,
        current_mod: String,
    },
    BuildingAtlasAndPanorama {
        started_at: Instant,
        rx: Receiver<PanoramaReadyBundle>,
        stage_atomic: Arc<AtomicU8>,
        done_atomic: Arc<AtomicUsize>,
        total_atomic: Arc<AtomicUsize>,
    },
    UploadingPanorama {
        started_at: Instant,
        pending: VecDeque<ChunkMeshData>,
        total: usize,
        uploaded: usize,
    },
    Ready,
}

fn panorama_disk_coords(center_cx: i32, center_cz: i32, radius: i32) -> Vec<(i32, i32)> {
    let mut cols = Vec::new();
    for dz in -radius..=radius {
        for dx in -radius..=radius {
            let d2 = dx * dx + dz * dz;
            if d2 <= radius * radius + 2 {
                cols.push((center_cx + dx, center_cz + dz, d2));
            }
        }
    }
    cols.sort_by_key(|&(_, _, d2)| d2);
    cols.into_iter().map(|(cx, cz, _)| (cx, cz)).collect()
}

fn spawn_asset_preloader(
    paths: &EnginePaths,
    rp_path: Option<PathBuf>,
) -> (
    Receiver<Vec<ModSpec>>,
    Receiver<(PreloadedTextures, Arc<AudioEngine>)>,
) {
    let (mods_tx, mods_rx) = mpsc::channel();
    let (assets_tx, assets_rx) = mpsc::channel();
    let paths_bg = paths.clone();
    std::thread::Builder::new()
        .name("cubex-preloader".into())
        .spawn(move || {
            let ov = if paths_bg.lua_override_dir.is_dir() {
                Some(paths_bg.lua_override_dir.as_path())
            } else {
                None
            };
            let mods = discover_and_sort_mods(&paths_bg.game_dir, ov);
            let mut mod_paths = HashMap::with_capacity(mods.len());
            for m in &mods {
                mod_paths.insert(m.name.clone(), m.path.clone());
            }
            let _ = mods_tx.send(mods);

            let (preloaded, audio) = rayon::join(
                || PreloadedTextures::load(&mod_paths, &paths_bg.game_dir, rp_path.as_deref()),
                || Arc::new(AudioEngine::new(&mod_paths, &paths_bg.game_dir)),
            );
            let _ = assets_tx.send((preloaded, audio));
        })
        .expect("spawn cubex-preloader thread");
    (mods_rx, assets_rx)
}

fn build_panorama_bundle(
    preloaded: PreloadedTextures,
    registries: &Arc<RwLock<EngineRegistries>>,
    world: &Arc<WorldStore>,
    smooth_lighting: bool,
    stage_atomic: &AtomicU8,
    done_atomic: &AtomicUsize,
    total_atomic: &AtomicUsize,
) -> PanoramaReadyBundle {
    let reg_guard = registries.read();
    let atlas = Arc::new(TextureAtlas::from_preloaded(preloaded, &reg_guard));
    let palette = MapgenPalette::from_registries(&reg_guard);
    let wt = *world.world_type.read();
    let seed = world.seed;
    let spawn_p = find_spawn_pos(seed, wt);
    let surf_y = spawn_p.y.max(SEA_LEVEL) as f32;
    let panorama_center = Vec3::new(spawn_p.x as f32, surf_y + 4.0, spawn_p.z as f32);
    let center_cx = (spawn_p.x as f32 / 16.0).floor() as i32;
    let center_cz = (spawn_p.z as f32 / 16.0).floor() as i32;

    let cols = panorama_disk_coords(center_cx, center_cz, PANORAMA_RADIUS);
    let generated_columns: HashSet<(i32, i32)> = cols.iter().copied().collect();
    total_atomic.store(cols.len(), Ordering::Relaxed);
    done_atomic.store(0, Ordering::Relaxed);
    stage_atomic.store(1, Ordering::Relaxed);

    let existing: HashSet<IVec3> = world.chunks.read().keys().copied().collect();
    let chunks: Vec<Chunk> = cols
        .par_iter()
        .flat_map_iter(|&(cx, cz)| {
            let col_chunks: Vec<Chunk> = (PANORAMA_Y_MIN..=PANORAMA_Y_MAX)
                .filter_map(|cy| {
                    let cpos = IVec3::new(cx, cy, cz);
                    if existing.contains(&cpos) {
                        None
                    } else {
                        Some(generate_chunk_for_world(
                            cpos, seed, wt, &palette, &reg_guard,
                        ))
                    }
                })
                .collect();
            done_atomic.fetch_add(1, Ordering::Relaxed);
            col_chunks
        })
        .collect();
    let inserted_coords: Vec<IVec3> = chunks.iter().map(|c| c.pos).collect();
    for ch in chunks {
        world.insert_chunk(ch);
    }
    if !inserted_coords.is_empty() {
        propagate_world_light(world, &inserted_coords, &reg_guard);
        let mut dq = world.dirty_queue.write();
        for cpos in &inserted_coords {
            dq.remove(cpos);
        }
    }

    let all_coords: Vec<IVec3> = cols
        .iter()
        .flat_map(|&(cx, cz)| (PANORAMA_Y_MIN..=PANORAMA_Y_MAX).map(move |cy| IVec3::new(cx, cy, cz)))
        .collect();
    stage_atomic.store(2, Ordering::Relaxed);
    total_atomic.store(all_coords.len(), Ordering::Relaxed);
    done_atomic.store(0, Ordering::Relaxed);
    let snaps: Vec<_> = all_coords
        .iter()
        .filter_map(|&cpos| world.snapshot_chunk(cpos))
        .collect();
    let meshes: Vec<ChunkMeshData> = snaps
        .into_par_iter()
        .filter_map(|snap| {
            let mesh = mesh_chunk_with_atlas(&snap, &reg_guard, &atlas, smooth_lighting);
            done_atomic.fetch_add(1, Ordering::Relaxed);
            if mesh.opaque_indices.is_empty() && mesh.trans_indices.is_empty() {
                None
            } else {
                Some(mesh)
            }
        })
        .collect();

    PanoramaReadyBundle {
        atlas,
        palette,
        generated_columns,
        meshes,
        panorama_center,
    }
}

struct GeneratedColumnBatch {
    epoch: u64,
    dimension: DimensionId,
    columns: Vec<(i32, i32)>,
}

struct MeshedChunkPacket {
    epoch: u64,
    dimension: DimensionId,
    mesh: ChunkMeshData,
}

enum WorldLoadPhase {
    Preparing {
        info: WorldInfo,
        radius: i32,
        painted: bool,
    },
    GeneratingTerrain {
        world_name: String,
        radius: i32,
        total_cols: usize,
        done_cols: usize,
        ready_columns: Vec<bool>,
        target_chunks: Vec<IVec3>,
        col_index_map: HashMap<(i32, i32), usize>,
        rx: Receiver<((i32, i32), Vec<Chunk>)>,
        had_saved_player: bool,
    },
    BuildingMeshes {
        world_name: String,
        radius: i32,
        total_chunks: usize,
        uploaded_chunks: usize,
        pending_set: HashSet<IVec3>,
        ready_columns: Vec<bool>,
        chunk_col_index: HashMap<IVec3, usize>,
        col_remaining_chunks: Vec<usize>,
    },
}

struct ActiveFormspec {
    formname: String,
    raw_formspec: String,
    parsed: ParsedFormspec,
    context_pos: Option<IVec3>,
}

struct LocalPlayerSlot {
    joined: bool,
    player: PlayerController,
    player_state: Arc<RwLock<ObjectState>>,
    move_input: MoveInput,
    digger: DigController,
    camera_mode: CameraMode,
    last_interact_pos: Option<IVec3>,
    last_inv_total_count: u32,
    active_formspec: Option<ActiveFormspec>,
    formspec_fields: HashMap<String, String>,
    creative_state: CreativeInventoryState,
    inv_cursor: InventoryCursor,
    form_slot_idx: usize,
    place_repeat_cd: f32,
    attack_tflp: f32,
    mine_was_held: bool,
    hud_icons: ItemIconCache,
    form_icons: ItemIconCache,
    sneak_toggled: bool,
    sprint_toggled: bool,
    targeted: Option<RaycastHit>,
    cam_pos: Vec3,
    view_proj: glam::Mat4,
    show_viewmodel: bool,
}

impl LocalPlayerSlot {
    fn new(name: &str) -> Self {
        let ps = Arc::new(RwLock::new(ObjectState::new_player(name)));
        Self {
            joined: false,
            player: PlayerController::default(),
            player_state: ps,
            move_input: MoveInput::default(),
            digger: DigController::new(),
            camera_mode: CameraMode::FirstPerson,
            last_interact_pos: None,
            last_inv_total_count: 0,
            active_formspec: None,
            formspec_fields: HashMap::new(),
            creative_state: CreativeInventoryState::default(),
            inv_cursor: InventoryCursor::default(),
            form_slot_idx: 0,
            place_repeat_cd: 0.0,
            attack_tflp: 0.5,
            mine_was_held: false,
            hud_icons: ItemIconCache::new(),
            form_icons: ItemIconCache::new(),
            sneak_toggled: false,
            sprint_toggled: false,
            targeted: None,
            cam_pos: Vec3::ZERO,
            view_proj: glam::Mat4::IDENTITY,
            show_viewmodel: true,
        }
    }
}

struct StreamPlayerRuntime {
    name: String,
    player: PlayerController,
    player_state: Arc<RwLock<ObjectState>>,
    digger: DigController,
    attack_tflp: f32,
    last_interact_pos: Option<IVec3>,
    last_inv_total_count: u32,
    swing_timer: f32,
    sneaking: bool,
    horizontal_speed: f32,
}

impl StreamPlayerRuntime {
    fn new(_id: u64, name: String, spawn_pos: Vec3) -> Self {
        let mut player = PlayerController::default();
        player.pos = spawn_pos;
        let player_state = Arc::new(RwLock::new(ObjectState::new_player(&name)));
        let last_inv_total_count = total_inventory_items(&player_state);
        Self {
            name,
            player,
            player_state,
            digger: DigController::new(),
            attack_tflp: 1.0,
            last_interact_pos: None,
            last_inv_total_count,
            swing_timer: 0.0,
            sneaking: false,
            horizontal_speed: 0.0,
        }
    }
}

struct RemoteLanPlayer {
    name: String,
    pos: Vec3,
    prev_pos: Vec3,
    yaw: f32,
    pitch: f32,
    horizontal_speed: f32,
    swinging: bool,
    swing_timer: f32,
    sneaking: bool,
    held_item: String,
    hurt_flash: f32,
    last_seen: Instant,
}

#[derive(Default)]
struct Kb2State {
    forward: bool,
    backward: bool,
    left: bool,
    right: bool,
    jump: bool,
    sneak: bool,
    yaw_left: bool,
    yaw_right: bool,
    pitch_up: bool,
    pitch_down: bool,
    mine_held: bool,
    place_just: bool,
    inv_just: bool,
    hotbar_prev: bool,
    hotbar_next: bool,
}

impl Kb2State {
    fn handle_key(&mut self, code: KeyCode, pressed: bool) {
        match code {
            KeyCode::ArrowUp => self.forward = pressed,
            KeyCode::ArrowDown => self.backward = pressed,
            KeyCode::ArrowLeft => self.left = pressed,
            KeyCode::ArrowRight => self.right = pressed,
            KeyCode::ShiftRight => self.jump = pressed,
            KeyCode::ControlRight => self.sneak = pressed,
            KeyCode::KeyJ => self.yaw_left = pressed,
            KeyCode::KeyL => self.yaw_right = pressed,
            KeyCode::KeyI => self.pitch_up = pressed,
            KeyCode::KeyK => self.pitch_down = pressed,
            KeyCode::KeyU => self.mine_held = pressed,
            KeyCode::KeyO if pressed => self.place_just = true,
            KeyCode::KeyP if pressed => self.inv_just = true,
            KeyCode::BracketLeft if pressed => self.hotbar_prev = true,
            KeyCode::BracketRight if pressed => self.hotbar_next = true,
            _ => {}
        }
    }
}

struct CubexEngine {
    paths: EnginePaths,
    config: Arc<RwLock<MinetestConfig>>,
    settings: EngineSettings,
    audio: Arc<AudioEngine>,
    audio_slot: Arc<RwLock<Arc<AudioEngine>>>,
    audio_settings_shared: Arc<RwLock<AudioSettings>>,
    registries: Arc<RwLock<EngineRegistries>>,
    world: Arc<WorldStore>,
    nether_world: Arc<WorldStore>,
    lua_rt: Option<LuaRuntime>,
    palette: MapgenPalette,
    nether_palette: NetherPalette,
    atlas: Arc<TextureAtlas>,
    packs: Vec<ResourcePackInfo>,
    available_skins: Vec<SkinInfo>,
    last_p1_skin: String,
    last_p2_skin: String,
    tunnel: PlayitTunnelManager,
    tokio_rt: tokio::runtime::Runtime,
    net_session: Option<NetClientSession>,
    local_server: Option<LocalServerHandle>,
    local_server_rx: Option<tokio::sync::broadcast::Receiver<NetPacket>>,
    host_spawn_shared: Arc<RwLock<[f32; 3]>>,
    lan_host_addr: Option<String>,
    remote_players: HashMap<String, RemoteLanPlayer>,
    stream_players: HashMap<u64, StreamPlayerRuntime>,
    last_stream_frame: Instant,
    last_stream_hud: Instant,
    pending_remote_spawn: Option<Vec3>,
    lan_broadcaster: Option<LanBroadcaster>,
    last_net_sync: Instant,

    mode: AppMode,
    player: PlayerController,
    player_state: Arc<RwLock<ObjectState>>,
    move_input: MoveInput,
    digger: DigController,
    last_interact_pos: Option<IVec3>,
    last_inv_total_count: u32,
    title_state: TitleScreenState,
    pause_state: PauseMenuState,
    active_formspec: Option<ActiveFormspec>,
    formspec_fields: HashMap<String, String>,
    chat_messages: Vec<String>,
    chat_typing: Option<String>,

    world_epoch: u64,
    world_epoch_shared: Arc<AtomicU64>,
    generated_columns: HashSet<(i32, i32)>,
    gen_in_flight: HashSet<(i32, i32)>,
    nether_generated_columns: HashSet<(i32, i32)>,
    nether_gen_in_flight: HashSet<(i32, i32)>,
    gen_tx: Sender<GeneratedColumnBatch>,
    gen_rx: Receiver<GeneratedColumnBatch>,
    meshing_in_flight: HashSet<IVec3>,
    pending_meshes: VecDeque<ChunkMeshData>,
    nether_meshing_in_flight: HashSet<IVec3>,
    nether_pending_meshes: VecDeque<ChunkMeshData>,
    mesh_tx: Sender<MeshedChunkPacket>,
    mesh_rx: Receiver<MeshedChunkPacket>,
    loaded_atlas_pack: Option<Option<PathBuf>>,
    last_prune_chunks: Option<Vec<(i32, i32)>>,
    last_nether_prune_chunks: Option<Vec<(i32, i32)>>,
    nether_ambient_timer: f32,
    mob_manager: MobManager,
    pending_load: Option<WorldLoadPhase>,
    panorama_center: Vec3,
    panorama_fog_dist: f32,
    peers: [LocalPlayerSlot; 3],

    last_frame: Instant,
    fps: f32,
    frame_ms: f32,
}

impl CubexEngine {
    /// Minimal shell boot (<10ms) for instant Vortex Studios splash screen display on Frame 1.
    fn boot_shell(paths: EnginePaths) -> anyhow::Result<Self> {
        paths.ensure_dirs();
        let config = Arc::new(RwLock::new(MinetestConfig::load(&paths.conf_file)));
        let default_pack = paths.textures_dir.join("craft_and_ruin");
        let default_pack_s = if default_pack.is_dir() {
            default_pack.to_string_lossy().to_string()
        } else {
            String::new()
        };
        let mut settings = config.read().to_engine_settings(&default_pack_s);
        settings.overlay_graphics_from_toml(&paths.settings_toml_file);
        let audio_settings_shared = Arc::new(RwLock::new(settings.audio.clone()));
        let packs = list_resource_packs(&paths.textures_dir);

        let registries = Arc::new(RwLock::new(EngineRegistries::new()));
        let default_world = std::env::temp_dir().join("cubex_rs_panorama");
        let world = Arc::new(WorldStore::new(&default_world, 1337));
        let nether_world = Arc::new(WorldStore::new_for_dimension(
            &default_world,
            1337,
            DimensionId::Nether,
            WorldGenType::Default,
            settings.difficulty,
            settings.gamerules.clone(),
            "Survival",
        ));

        let palette = MapgenPalette::default();
        let nether_palette = NetherPalette::from_registries(&registries.read());
        let audio = Arc::new(AudioEngine::empty_silent());
        let audio_slot = Arc::new(RwLock::new(audio.clone()));
        let atlas = Arc::new(TextureAtlas::placeholder());
        let available_skins = list_available_skins(&paths.skins_dir, &atlas.get_player_skin_rgba());
        let last_p1_skin = settings.player_skin.clone();
        let last_p2_skin = settings.p2_skin.clone();

        let player_state = Arc::new(RwLock::new(ObjectState::new_player(&settings.player_name)));
        let last_inv_total_count = total_inventory_items(&player_state);

        let tunnel = PlayitTunnelManager::new(
            &paths.playit_bin,
            paths.playit_config_dir.join("playit.toml"),
            paths.playit_config_dir.join("playit.log"),
        );

        let tokio_rt = tokio::runtime::Runtime::new()?;
        let servers_file = paths.cubex_data_dir.join("servers.json");
        let mut title_state = TitleScreenState::new(&paths.worlds_dir, &servers_file);
        title_state.available_skins = available_skins.clone();
        let mut pause_state = PauseMenuState::default();
        pause_state.available_skins = available_skins.clone();
        let peers = [
            LocalPlayerSlot::new(&settings.p2_name),
            LocalPlayerSlot::new(&settings.p3_name),
            LocalPlayerSlot::new(&settings.p4_name),
        ];

        let spawn_p = find_spawn_pos(world.seed, *world.world_type.read());
        let surf_y = spawn_p.y.max(SEA_LEVEL) as f32;
        let panorama_center = Vec3::new(spawn_p.x as f32, surf_y + 4.0, spawn_p.z as f32);
        let host_spawn_shared = Arc::new(RwLock::new([
            spawn_p.x as f32 + 0.5,
            surf_y + 1.0,
            spawn_p.z as f32 + 0.5,
        ]));

        let (gen_tx, gen_rx) = mpsc::channel();
        let (mesh_tx, mesh_rx) = mpsc::channel();
        let world_epoch_shared = Arc::new(AtomicU64::new(0));

        Ok(Self {
            paths,
            config,
            settings,
            audio,
            audio_slot,
            audio_settings_shared,
            registries,
            world,
            nether_world,
            lua_rt: None,
            palette,
            nether_palette,
            atlas,
            packs,
            available_skins,
            last_p1_skin,
            last_p2_skin,
            tunnel,
            tokio_rt,
            net_session: None,
            local_server: None,
            local_server_rx: None,
            host_spawn_shared,
            lan_host_addr: None,
            remote_players: HashMap::new(),
            stream_players: HashMap::new(),
            last_stream_frame: Instant::now(),
            last_stream_hud: Instant::now(),
            pending_remote_spawn: None,
            lan_broadcaster: None,
            last_net_sync: Instant::now(),
            mode: AppMode::Splash,
            player: PlayerController::default(),
            player_state,
            move_input: MoveInput::default(),
            digger: DigController::new(),
            last_interact_pos: None,
            last_inv_total_count,
            title_state,
            pause_state,
            active_formspec: None,
            formspec_fields: HashMap::new(),
            chat_messages: vec![
                "Welcome to Cubex-RS (Mineclonia + LuaJIT + Vulkan)!".into(),
            ],
            chat_typing: None,
            world_epoch: 0,
            world_epoch_shared,
            generated_columns: HashSet::new(),
            gen_in_flight: HashSet::new(),
            nether_generated_columns: HashSet::new(),
            nether_gen_in_flight: HashSet::new(),
            gen_tx,
            gen_rx,
            meshing_in_flight: HashSet::new(),
            pending_meshes: VecDeque::new(),
            nether_meshing_in_flight: HashSet::new(),
            nether_pending_meshes: VecDeque::new(),
            mesh_tx,
            mesh_rx,
            loaded_atlas_pack: None,
            last_prune_chunks: None,
            last_nether_prune_chunks: None,
            nether_ambient_timer: 0.0,
            mob_manager: MobManager::new(1337),
            pending_load: None,
            panorama_center,
            panorama_fog_dist: 92.0,
            peers,
            last_frame: Instant::now(),
            fps: 120.0,
            frame_ms: 4.0,
        })
    }

    fn set_audio_loaded(&mut self, loaded: Arc<AudioEngine>) {
        self.audio = loaded.clone();
        *self.audio_slot.write() = loaded;
    }

    fn refresh_available_skins(&mut self) {
        let def_skin = self.atlas.get_player_skin_rgba();
        let skins = list_available_skins(&self.paths.skins_dir, &def_skin);
        self.available_skins = skins.clone();
        self.title_state.available_skins = skins.clone();
        self.pause_state.available_skins = skins;
        self.title_state.skin_preview_handles.clear();
        self.pause_state.skin_preview_handles.clear();
    }

    fn sync_skins_and_props(
        &mut self,
        renderer: Option<&mut WgpuRenderer>,
        egui_ctx: Option<&egui::Context>,
        force_equip: bool,
    ) {
        if self.available_skins.is_empty() {
            self.refresh_available_skins();
        }
        let fallback = self.atlas.get_player_skin_rgba();
        let p1_skin = self.settings.player_skin.clone();
        let p2_skin = self.settings.p2_skin.clone();
        let p1_rgba = resolve_skin_rgba(&self.available_skins, &p1_skin, &fallback);
        let p2_rgba = resolve_skin_rgba(&self.available_skins, &p2_skin, &fallback);

        if let Some(ctx) = egui_ctx {
            self.title_state.icons.set_player_skin_rgba(
                ctx,
                p1_rgba.as_raw(),
                p1_rgba.width(),
                p1_rgba.height(),
            );
        }
        if let Some(r) = renderer {
            r.set_player_skin(0, &p1_skin, &p1_rgba);
            r.set_player_skin(1, &p2_skin, &p2_rgba);
        }

        let p1_changed = self.last_p1_skin != p1_skin;
        let p2_changed = self.last_p2_skin != p2_skin;
        self.last_p1_skin = p1_skin.clone();
        self.last_p2_skin = p2_skin.clone();

        if is_carterpcs_skin(&p1_skin) && (force_equip || p1_changed) {
            equip_ram_mic_for_carterpcs(&self.player_state, &mut self.player.selected_slot);
            self.last_inv_total_count = total_inventory_items(&self.player_state);
        }
        if is_carterpcs_skin(&p2_skin) && (force_equip || p2_changed) && self.peers[0].joined {
            equip_ram_mic_for_carterpcs(
                &self.peers[0].player_state,
                &mut self.peers[0].player.selected_slot,
            );
            self.peers[0].last_inv_total_count =
                total_inventory_items(&self.peers[0].player_state);
        }
    }

    fn boot_base(paths: EnginePaths) -> anyhow::Result<Self> {
        let mut engine = Self::boot_shell(paths)?;
        let lua_rt = LuaRuntime::boot(
            engine.paths.clone(),
            engine.registries.clone(),
            engine.world.clone(),
            engine.config.clone(),
        )?;
        engine.palette = MapgenPalette::from_registries(&engine.registries.read());
        engine.nether_palette = NetherPalette::from_registries(&engine.registries.read());
        let loaded_audio = Arc::new(AudioEngine::new(
            &lua_rt.shared.mod_paths.read(),
            &engine.paths.game_dir,
        ));
        engine.set_audio_loaded(loaded_audio);

        let rp_path = engine.resolve_rp_path();
        engine.atlas = Arc::new(TextureAtlas::build(
            &lua_rt.shared.mod_paths.read(),
            &engine.paths.game_dir,
            rp_path.as_deref(),
            &engine.registries.read(),
        ));
        engine.loaded_atlas_pack = Some(rp_path);
        engine.refresh_available_skins();

        let _ = lua_rt.join_player(engine.player_state.clone());
        engine.last_inv_total_count = total_inventory_items(&engine.player_state);
        engine.lua_rt = Some(lua_rt);
        engine.mode = AppMode::Title;

        // Pre-generate immediate spawn area (5x5 chunk columns) for tests/playtest/smoke-test
        engine.generate_spawn_area(2, true);
        Ok(engine)
    }

    fn resolve_rp_path(&self) -> Option<PathBuf> {
        if !self.settings.active_resource_pack.is_empty()
            && PathBuf::from(&self.settings.active_resource_pack).is_dir()
        {
            Some(PathBuf::from(&self.settings.active_resource_pack))
        } else {
            let def = self.paths.textures_dir.join("craft_and_ruin");
            if def.is_dir() {
                Some(def)
            } else {
                None
            }
        }
    }

    fn generate_spawn_area(&mut self, radius_chunks: i32, update_player_pos: bool) {
        let (center_cx, center_cz) = if update_player_pos {
            (0, 0)
        } else {
            (
                (self.player.pos.x / 16.0).floor() as i32,
                (self.player.pos.z / 16.0).floor() as i32,
            )
        };
        let mut cols = Vec::new();
        for cz in (center_cz - radius_chunks)..=(center_cz + radius_chunks) {
            for cx in (center_cx - radius_chunks)..=(center_cx + radius_chunks) {
                if !self.generated_columns.contains(&(cx, cz)) {
                    cols.push((cx, cz));
                    self.generated_columns.insert((cx, cz));
                }
            }
        }
        if !cols.is_empty() {
            let seed = self.world.seed;
            let wt = *self.world.world_type.read();
            let y_range = chunk_y_range_for(wt);
            let palette = self.palette.clone();
            let reg = self.registries.read();
            let existing: HashSet<IVec3> = self.world.chunks.read().keys().copied().collect();
            let missing_coords: Vec<IVec3> = cols
                .into_iter()
                .flat_map(|(cx, cz)| y_range.clone().map(move |cy| IVec3::new(cx, cy, cz)))
                .filter(|cpos| !existing.contains(cpos))
                .collect();
            let chunks: Vec<_> = missing_coords
                .into_par_iter()
                .map(|cpos| generate_chunk_for_world(cpos, seed, wt, &palette, &reg))
                .collect();
            let inserted_coords: Vec<IVec3> = chunks.iter().map(|c| c.pos).collect();
            for ch in chunks {
                self.world.insert_chunk(ch);
            }
            if !inserted_coords.is_empty() {
                for &cpos in &inserted_coords {
                    populate_structure_chests_for_chunk(&self.world, cpos, seed, wt, &reg);
                }
                propagate_world_light(&self.world, &inserted_coords, &reg);
            }
        }

        if update_player_pos {
            let wt = *self.world.world_type.read();
            let sp = find_spawn_pos(self.world.seed, wt);
            self.player.pos = Vec3::new(sp.x as f32 + 0.5, sp.y as f32, sp.z as f32 + 0.5);
            self.player.vel = Vec3::ZERO;
        }
    }

    fn stream_chunks_around_player(&mut self) {
        let mut ow_centers: Vec<(i32, i32)> = Vec::new();
        let mut nw_centers: Vec<(i32, i32)> = Vec::new();
        let push_unique = |list: &mut Vec<(i32, i32)>, pos: Vec3| {
            let pc = (
                (pos.x / 16.0).floor() as i32,
                (pos.z / 16.0).floor() as i32,
            );
            if !list.contains(&pc) {
                list.push(pc);
            }
        };
        if self.player.dimension.is_nether() {
            push_unique(&mut nw_centers, self.player.pos);
        } else {
            push_unique(&mut ow_centers, self.player.pos);
        }
        if self.mode == AppMode::Playing && self.settings.coop_mode.is_active() {
            for peer in &self.peers {
                if peer.joined {
                    if peer.player.dimension.is_nether() {
                        push_unique(&mut nw_centers, peer.player.pos);
                    } else {
                        push_unique(&mut ow_centers, peer.player.pos);
                    }
                }
            }
        }
        if self.mode == AppMode::Playing {
            for sp in self.stream_players.values() {
                if sp.player.dimension.is_nether() {
                    push_unique(&mut nw_centers, sp.player.pos);
                } else {
                    push_unique(&mut ow_centers, sp.player.pos);
                }
            }
        }
        self.stream_chunks_for_dimensions(&ow_centers, &nw_centers);
    }

    fn stream_chunks_for_dimensions(
        &mut self,
        ow_centers: &[(i32, i32)],
        nw_centers: &[(i32, i32)],
    ) {
        while let Ok(batch) = self.gen_rx.try_recv() {
            if batch.epoch == self.world_epoch {
                let target = if batch.dimension.is_nether() {
                    &mut self.nether_gen_in_flight
                } else {
                    &mut self.gen_in_flight
                };
                for col in batch.columns {
                    target.remove(&col);
                }
            }
        }

        let r = (self.settings.render_distance_chunks as i32).clamp(2, 32);

        if !ow_centers.is_empty() && self.gen_in_flight.len() < 12 {
            let mut candidates: HashMap<(i32, i32), i32> = HashMap::new();
            for &(pcx, pcz) in ow_centers {
                for dz in -r..=r {
                    for dx in -r..=r {
                        let d2 = dx * dx + dz * dz;
                        if d2 > r * r + 2 {
                            continue;
                        }
                        let col = (pcx + dx, pcz + dz);
                        if !self.generated_columns.contains(&col) {
                            candidates
                                .entry(col)
                                .and_modify(|prev| *prev = (*prev).min(d2))
                                .or_insert(d2);
                        }
                    }
                }
            }
            if !candidates.is_empty() {
                let mut sorted: Vec<(i32, (i32, i32))> =
                    candidates.into_iter().map(|(c, d2)| (d2, c)).collect();
                sorted.sort_by_key(|(d2, _)| *d2);
                let budget = (12 - self.gen_in_flight.len()).min(4);
                let mut dispatch_cols = Vec::with_capacity(budget);
                for (_, col) in sorted.into_iter().take(budget) {
                    self.generated_columns.insert(col);
                    self.gen_in_flight.insert(col);
                    dispatch_cols.push(col);
                }
                if !dispatch_cols.is_empty() {
                    let seed = self.world.seed;
                    let wt = *self.world.world_type.read();
                    let y_range = chunk_y_range_for(wt);
                    let palette = self.palette.clone();
                    let reg_arc = self.registries.clone();
                    let world_arc = self.world.clone();
                    let epoch = self.world_epoch;
                    let epoch_shared = self.world_epoch_shared.clone();
                    let gen_tx = self.gen_tx.clone();
                    rayon::spawn(move || {
                        if epoch_shared.load(Ordering::Acquire) != epoch {
                            return;
                        }
                        let existing: HashSet<IVec3> =
                            world_arc.chunks.read().keys().copied().collect();
                        let missing_coords: Vec<IVec3> = dispatch_cols
                            .iter()
                            .copied()
                            .flat_map(|(cx, cz)| {
                                y_range.clone().map(move |cy| IVec3::new(cx, cy, cz))
                            })
                            .filter(|cpos| !existing.contains(cpos))
                            .collect();
                        let chunks: Vec<Chunk> = {
                            let reg = reg_arc.read();
                            missing_coords
                                .into_par_iter()
                                .map(|cpos| generate_chunk_for_world(cpos, seed, wt, &palette, &reg))
                                .collect()
                        };
                        if epoch_shared.load(Ordering::Acquire) != epoch {
                            return;
                        }
                        let inserted_coords: Vec<IVec3> = chunks.iter().map(|c| c.pos).collect();
                        for ch in chunks {
                            world_arc.insert_chunk(ch);
                        }
                        if !inserted_coords.is_empty() {
                            let reg = reg_arc.read();
                            for &cpos in &inserted_coords {
                                populate_structure_chests_for_chunk(
                                    &world_arc,
                                    cpos,
                                    seed,
                                    wt,
                                    &reg,
                                );
                            }
                            propagate_world_light(&world_arc, &inserted_coords, &reg);
                        }
                        let _ = gen_tx.send(GeneratedColumnBatch {
                            epoch,
                            dimension: DimensionId::Overworld,
                            columns: dispatch_cols,
                        });
                    });
                }
            }
        }

        if !nw_centers.is_empty() && self.nether_gen_in_flight.len() < 12 {
            let mut candidates: HashMap<(i32, i32), i32> = HashMap::new();
            let nr = r.min(12);
            for &(pcx, pcz) in nw_centers {
                for dz in -nr..=nr {
                    for dx in -nr..=nr {
                        let d2 = dx * dx + dz * dz;
                        if d2 > nr * nr + 2 {
                            continue;
                        }
                        let col = (pcx + dx, pcz + dz);
                        if !self.nether_generated_columns.contains(&col) {
                            candidates
                                .entry(col)
                                .and_modify(|prev| *prev = (*prev).min(d2))
                                .or_insert(d2);
                        }
                    }
                }
            }
            if !candidates.is_empty() {
                let mut sorted: Vec<(i32, (i32, i32))> =
                    candidates.into_iter().map(|(c, d2)| (d2, c)).collect();
                sorted.sort_by_key(|(d2, _)| *d2);
                let budget = (12 - self.nether_gen_in_flight.len()).min(4);
                let mut dispatch_cols = Vec::with_capacity(budget);
                for (_, col) in sorted.into_iter().take(budget) {
                    self.nether_generated_columns.insert(col);
                    self.nether_gen_in_flight.insert(col);
                    dispatch_cols.push(col);
                }
                if !dispatch_cols.is_empty() {
                    let seed = self.nether_world.seed;
                    let pal = self.nether_palette.clone();
                    let reg_arc = self.registries.clone();
                    let world_arc = self.nether_world.clone();
                    let epoch = self.world_epoch;
                    let epoch_shared = self.world_epoch_shared.clone();
                    let gen_tx = self.gen_tx.clone();
                    rayon::spawn(move || {
                        if epoch_shared.load(Ordering::Acquire) != epoch {
                            return;
                        }
                        let existing: HashSet<IVec3> =
                            world_arc.chunks.read().keys().copied().collect();
                        let missing_coords: Vec<IVec3> = dispatch_cols
                            .iter()
                            .copied()
                            .flat_map(|(cx, cz)| {
                                NETHER_CHUNK_Y_RANGE.map(move |cy| IVec3::new(cx, cy, cz))
                            })
                            .filter(|cpos| !existing.contains(cpos))
                            .collect();
                        let chunks: Vec<Chunk> = {
                            let reg = reg_arc.read();
                            missing_coords
                                .into_par_iter()
                                .map(|cpos| generate_nether_chunk(cpos, seed, &pal, &reg))
                                .collect()
                        };
                        if epoch_shared.load(Ordering::Acquire) != epoch {
                            return;
                        }
                        let inserted_coords: Vec<IVec3> = chunks.iter().map(|c| c.pos).collect();
                        for ch in chunks {
                            world_arc.insert_chunk(ch);
                        }
                        if !inserted_coords.is_empty() {
                            let reg = reg_arc.read();
                            for &cpos in &inserted_coords {
                                populate_structure_chests_for_chunk(
                                    &world_arc,
                                    cpos,
                                    seed,
                                    WorldGenType::Default,
                                    &reg,
                                );
                            }
                            propagate_world_light(&world_arc, &inserted_coords, &reg);
                        }
                        let _ = gen_tx.send(GeneratedColumnBatch {
                            epoch,
                            dimension: DimensionId::Nether,
                            columns: dispatch_cols,
                        });
                    });
                }
            }
        }

        // Dispatch background dirty chunks for parallel rayon meshing (Overworld + Nether)
        if !ow_centers.is_empty() && self.meshing_in_flight.len() < 48 {
            let max_take = (48 - self.meshing_in_flight.len()).min(16);
            let dirty = self.world.take_dirty_chunks(max_take);
            let mut snaps = Vec::new();
            for cpos in dirty {
                if self.meshing_in_flight.insert(cpos) {
                    if let Some(snap) = self.world.snapshot_chunk(cpos) {
                        snaps.push(snap);
                    } else {
                        self.meshing_in_flight.remove(&cpos);
                    }
                }
            }
            if !snaps.is_empty() {
                let reg_arc = self.registries.clone();
                let atlas_arc = self.atlas.clone();
                let smooth = self.settings.smooth_lighting;
                let tx = self.mesh_tx.clone();
                let epoch = self.world_epoch;
                let epoch_shared = self.world_epoch_shared.clone();
                rayon::spawn(move || {
                    if epoch_shared.load(Ordering::Acquire) != epoch {
                        return;
                    }
                    let reg = reg_arc.read();
                    snaps.into_par_iter().for_each_with(tx, |tx, snap| {
                        if epoch_shared.load(Ordering::Acquire) != epoch {
                            return;
                        }
                        let mesh = mesh_chunk_with_atlas(&snap, &reg, &atlas_arc, smooth);
                        let _ = tx.send(MeshedChunkPacket {
                            epoch,
                            dimension: DimensionId::Overworld,
                            mesh,
                        });
                    });
                });
            }
        }

        if !nw_centers.is_empty() && self.nether_meshing_in_flight.len() < 48 {
            let max_take = (48 - self.nether_meshing_in_flight.len()).min(16);
            let dirty = self.nether_world.take_dirty_chunks(max_take);
            let mut snaps = Vec::new();
            for cpos in dirty {
                if self.nether_meshing_in_flight.insert(cpos) {
                    if let Some(snap) = self.nether_world.snapshot_chunk(cpos) {
                        snaps.push(snap);
                    } else {
                        self.nether_meshing_in_flight.remove(&cpos);
                    }
                }
            }
            if !snaps.is_empty() {
                let reg_arc = self.registries.clone();
                let atlas_arc = self.atlas.clone();
                let smooth = self.settings.smooth_lighting;
                let tx = self.mesh_tx.clone();
                let epoch = self.world_epoch;
                let epoch_shared = self.world_epoch_shared.clone();
                rayon::spawn(move || {
                    if epoch_shared.load(Ordering::Acquire) != epoch {
                        return;
                    }
                    let reg = reg_arc.read();
                    snaps.into_par_iter().for_each_with(tx, |tx, snap| {
                        if epoch_shared.load(Ordering::Acquire) != epoch {
                            return;
                        }
                        let mesh = mesh_chunk_with_atlas(&snap, &reg, &atlas_arc, smooth);
                        let _ = tx.send(MeshedChunkPacket {
                            epoch,
                            dimension: DimensionId::Nether,
                            mesh,
                        });
                    });
                });
            }
        }

        while let Ok(pkt) = self.mesh_rx.try_recv() {
            if pkt.epoch != self.world_epoch {
                continue;
            }
            if pkt.dimension.is_nether() {
                self.nether_meshing_in_flight.remove(&pkt.mesh.chunk_pos);
                self.nether_pending_meshes.push_back(pkt.mesh);
            } else {
                self.meshing_in_flight.remove(&pkt.mesh.chunk_pos);
                self.pending_meshes.push_back(pkt.mesh);
            }
        }
    }

    fn apply_settings_to_lua_and_disk(&mut self) {
        *self.audio_settings_shared.write() = self.settings.audio.clone();
        *self.world.difficulty.write() = self.settings.difficulty;
        *self.world.gamerules.write() = self.settings.gamerules.clone();

        {
            let mut cfg = self.config.write();
            cfg.apply_engine_settings(&self.settings);
            let _ = cfg.save();
        }
        let _ = self.settings.save_toml(&self.paths.settings_toml_file);
        if let Some(ref rt) = self.lua_rt {
            if let Ok(core) = rt.lua.globals().get::<mlua::Table>("core") {
                if let Ok(s_ud) = core.get::<mlua::AnyUserData>("settings") {
                    if let Ok(set_fn) = s_ud.get::<mlua::Function>("set") {
                        let _ = set_fn.call::<()>((
                            s_ud.clone(),
                            "fov",
                            self.settings.fov.round() as i32,
                        ));
                        let _ = set_fn.call::<()>((
                            s_ud.clone(),
                            "viewing_range",
                            self.settings.render_distance_chunks * 16,
                        ));
                    }
                    if let Ok(set_b) = s_ud.get::<mlua::Function>("set_bool") {
                        let _ = set_b.call::<()>((
                            s_ud.clone(),
                            "smooth_lighting",
                            self.settings.smooth_lighting,
                        ));
                        let _ = set_b.call::<()>((
                            s_ud.clone(),
                            "enable_dynamic_shadows",
                            self.settings.dynamic_shadows,
                        ));
                        let _ = set_b.call::<()>((
                            s_ud.clone(),
                            "cubex_mod_dynamic_lights",
                            self.settings.handheld_torch_light,
                        ));
                        let _ = set_b.call::<()>((
                            s_ud.clone(),
                            "enable_bloom",
                            self.settings.bloom_glow,
                        ));
                        let _ = set_b.call::<()>((
                            s_ud.clone(),
                            "liquids_pointable",
                            self.settings.liquids_pointable,
                        ));
                        let _ = set_b.call::<()>((
                            s_ud.clone(),
                            "mcl_keepInventory",
                            self.settings.gamerules.keep_inventory,
                        ));
                    }
                }
            }
        }
    }

    fn reload_resource_pack(&mut self, pack_path: &str, renderer: Option<&mut WgpuRenderer>) {
        self.settings.active_resource_pack = pack_path.to_string();
        self.apply_settings_to_lua_and_disk();
        let rp = if pack_path.is_empty() {
            None
        } else {
            Some(PathBuf::from(pack_path))
        };
        if let Some(ref rt) = self.lua_rt {
            self.atlas = Arc::new(TextureAtlas::build(
                &rt.shared.mod_paths.read(),
                &self.paths.game_dir,
                rp.as_deref(),
                &self.registries.read(),
            ));
            self.loaded_atlas_pack = Some(rp);
            self.refresh_available_skins();
            if let Some(r) = renderer {
                r.reload_atlas(&self.atlas);
                self.sync_skins_and_props(Some(r), None, false);
            } else {
                self.sync_skins_and_props(None, None, false);
            }
            let keys: Vec<IVec3> = self.world.chunks.read().keys().copied().collect();
            self.world.dirty_queue.write().extend(keys);
        }
    }

    fn open_lan_server(&mut self, port: u16) {
        if self.local_server.is_some() {
            return;
        }
        *self.host_spawn_shared.write() = [
            self.player.pos.x,
            self.player.pos.y,
            self.player.pos.z,
        ];
        let w = self.world.clone();
        let r = self.registries.clone();
        let sp = self.host_spawn_shared.clone();
        let res = self
            .tokio_rt
            .block_on(async move { start_server_with_handle(port, w, r, sp).await });
        match res {
            Ok(handle) => {
                let bound_port = handle.port;
                let rx = handle.subscribe();
                let lan_ip = local_lan_ipv4();
                let addr_str = format!("{lan_ip}:{bound_port}");
                let world_title = format!("{}'s LAN World", self.settings.player_name);
                let host_name = self.settings.player_name.clone();
                let pos = self.player.pos;
                let yaw = self.player.yaw;
                let pitch = self.player.pitch;
                let sneak = self.player.is_sneaking;
                let sprint = self.player.is_sprinting;
                handle.update_host_player(RemotePlayerState {
                    player_name: host_name,
                    x: pos.x,
                    y: pos.y,
                    z: pos.z,
                    yaw,
                    pitch,
                    swinging: false,
                    sneaking: sneak,
                    sprinting: sprint,
                    held_item: String::new(),
                });
                let motd_str = format!(
                    "§a{}'s Cubex-RS LAN World\n§7Mineclonia Port {bound_port}",
                    self.settings.player_name
                );
                self.lan_broadcaster = Some(self.tokio_rt.block_on(async move {
                    LanBroadcaster::spawn(LanBeaconInfo {
                        name: world_title,
                        port: bound_port,
                        motd: motd_str,
                        players_online: 1,
                        players_max: 16,
                    })
                }));
                self.local_server = Some(handle);
                self.local_server_rx = Some(rx);
                self.lan_host_addr = Some(addr_str.clone());
                self.pause_state.lan_active_addr = Some(addr_str.clone());
                self.chat_messages.push(format!(
                    "§aLocal LAN game opened at §e{addr_str}§a (Port {bound_port})."
                ));
            }
            Err(e) => {
                self.chat_messages
                    .push(format!("§cCould not open LAN server on port {port}: {e}"));
            }
        }
    }

    fn stop_lan_server(&mut self) {
        if let Some(b) = self.lan_broadcaster.take() {
            b.stop();
        }
        if let Some(srv) = self.local_server.take() {
            srv.stop();
            self.chat_messages.push("§eClosed local LAN server.".into());
        }
        self.local_server_rx = None;
        self.lan_host_addr = None;
        self.pause_state.lan_active_addr = None;
        if self.net_session.is_none() {
            self.remote_players.clear();
        }
        self.stream_players.clear();
    }

    /// Initialize `WorldStore`, Lua mods, player spawn/persistence, and atlas for a world session.
    /// Returns `true` if a saved player was restored from disk.
    fn prepare_world_session(&mut self, info: &WorldInfo, renderer: &mut WgpuRenderer) -> bool {
        self.world_epoch = self.world_epoch.wrapping_add(1);
        self.world_epoch_shared.store(self.world_epoch, Ordering::Release);
        self.generated_columns.clear();
        self.gen_in_flight.clear();
        self.meshing_in_flight.clear();
        self.pending_meshes.clear();
        self.last_prune_chunks = None;
        self.nether_generated_columns.clear();
        self.nether_gen_in_flight.clear();
        self.nether_meshing_in_flight.clear();
        self.nether_pending_meshes.clear();
        self.last_nether_prune_chunks = None;
        while self.gen_rx.try_recv().is_ok() {}
        while self.mesh_rx.try_recv().is_ok() {}

        renderer.chunk_meshes.clear();
        renderer.nether_chunk_meshes.clear();
        renderer.update_entities(&[], &self.registries.read(), &self.atlas);
        renderer.update_nether_entities(&[], &self.registries.read(), &self.atlas);
        renderer.update_guardian_lasers(&[]);
        renderer.particles.clear();
        renderer.nether_particles.clear();
        renderer.remote_players.clear();
        renderer.fog_dist_override = None;
        self.mob_manager = MobManager::new(info.seed);
        self.remote_players.clear();
        self.stream_players.clear();
        self.digger.reset();
        self.active_formspec = None;
        self.chat_typing = None;
        self.last_interact_pos = None;
        for peer in &mut self.peers {
            peer.joined = false;
            peer.digger.reset();
            peer.active_formspec = None;
            peer.inv_cursor.clear();
            peer.last_interact_pos = None;
        }

        // Create fresh WorldStore + fresh LuaRuntime so Lua shared.world matches self.world 1:1
        let new_world = Arc::new(WorldStore::new_with_options(
            &info.path,
            info.seed,
            info.world_type,
            info.difficulty,
            info.gamerules.clone(),
            &info.gamemode,
        ));
        let new_nether_world = Arc::new(WorldStore::new_for_dimension(
            &info.path,
            info.seed,
            DimensionId::Nether,
            info.world_type,
            info.difficulty,
            info.gamerules.clone(),
            &info.gamemode,
        ));
        {
            let mut cfg = self.config.write();
            cfg.set(
                "creative_mode",
                info.gamemode.eq_ignore_ascii_case("creative"),
            );
            cfg.set("enable_damage", true);
        }
        let new_reg = Arc::new(RwLock::new(EngineRegistries::new()));
        let new_rt = LuaRuntime::boot(
            self.paths.clone(),
            new_reg.clone(),
            new_world.clone(),
            self.config.clone(),
        )
        .expect("LuaRuntime boot for entered world");

        self.palette = MapgenPalette::from_registries(&new_reg.read());
        self.nether_palette = NetherPalette::from_registries(&new_reg.read());
        let rp = self.resolve_rp_path();
        let can_reuse_atlas = self.loaded_atlas_pack.as_ref() == Some(&rp) && {
            let old_reg = self.registries.read();
            let new_reg_g = new_reg.read();
            old_reg.nodes_by_id.len() == new_reg_g.nodes_by_id.len()
                && old_reg
                    .nodes_by_id
                    .iter()
                    .zip(new_reg_g.nodes_by_id.iter())
                    .all(|(a, b)| {
                        a.as_ref().map(|d| (d.id, d.name.as_str()))
                            == b.as_ref().map(|d| (d.id, d.name.as_str()))
                    })
        };
        if !can_reuse_atlas {
            self.atlas = Arc::new(TextureAtlas::build(
                &new_rt.shared.mod_paths.read(),
                &self.paths.game_dir,
                rp.as_deref(),
                &new_reg.read(),
            ));
            self.loaded_atlas_pack = Some(rp);
            renderer.reload_atlas(&self.atlas);
        }

        self.world = new_world;
        self.nether_world = new_nether_world;
        self.registries = new_reg;
        self.lua_rt = Some(new_rt);

        self.player_state = Arc::new(RwLock::new(ObjectState::new_player(&self.settings.player_name)));
        if let Some(ref rt) = self.lua_rt {
            let _ = rt.join_player(self.player_state.clone());
        }

        // Load persisted chunks, node metadata, and player state if saved on disk
        let had_saved_player = {
            let mut ps_guard = self.player_state.write();
            let reg_guard = self.registries.read();
            let _ = load_dimension_chunks(&self.nether_world, &reg_guard);
            match load_world(&self.world, &reg_guard, Some(&mut ps_guard)) {
                Ok(saved) => {
                    self.settings.difficulty = saved.difficulty;
                    self.settings.gamerules = saved.gamerules.clone();
                    if let Some(sp) = saved.player {
                        self.player.pos = Vec3::new(sp.pos[0], sp.pos[1], sp.pos[2]);
                        self.player.yaw = sp.yaw;
                        self.player.pitch = sp.pitch;
                        self.player.hp = if sp.hp <= 0 { 20 } else { sp.hp.min(20) };
                        self.player.hunger = if sp.hunger <= 0 { 20 } else { sp.hunger.min(20) };
                        self.player.saturation = (self.player.hunger as f32).min(5.0);
                        self.player.exhaustion = 0.0;
                        self.player.regen_timer = 0.0;
                        self.player.starve_timer = 0.0;
                        self.player.eat_cooldown = 0.0;
                        self.player.vel = Vec3::ZERO;
                        self.player.selected_slot = sp.wield_index.saturating_sub(1).min(8);
                        self.player.gamemode = parse_gamemode(&sp.gamemode);
                        self.player.dimension = sp.dimension;
                        self.player.portal_charge = 0.0;
                        self.player.portal_cooldown = 0.0;
                        self.player.portal_warp_flash = 0.0;
                        ps_guard.hp = self.player.hp;
                        ps_guard.hunger = self.player.hunger;
                        ps_guard.dimension = sp.dimension;
                        true
                    } else {
                        self.player.gamemode = parse_gamemode(&saved.gamemode);
                        self.player.hp = 20;
                        self.player.hunger = 20;
                        self.player.saturation = 5.0;
                        self.player.exhaustion = 0.0;
                        self.player.regen_timer = 0.0;
                        self.player.starve_timer = 0.0;
                        self.player.eat_cooldown = 0.0;
                        self.player.vel = Vec3::ZERO;
                        self.player.selected_slot = 0;
                        self.player.dimension = DimensionId::Overworld;
                        self.player.portal_charge = 0.0;
                        self.player.portal_cooldown = 0.0;
                        self.player.portal_warp_flash = 0.0;
                        false
                    }
                }
                Err(_) => {
                    self.player.gamemode = parse_gamemode(&info.gamemode);
                    self.player.hp = 20;
                    self.player.hunger = 20;
                    self.player.saturation = 5.0;
                    self.player.exhaustion = 0.0;
                    self.player.regen_timer = 0.0;
                    self.player.starve_timer = 0.0;
                    self.player.eat_cooldown = 0.0;
                    self.player.vel = Vec3::ZERO;
                    self.player.selected_slot = 0;
                    self.player.dimension = DimensionId::Overworld;
                    self.player.portal_charge = 0.0;
                    self.player.portal_cooldown = 0.0;
                    self.player.portal_warp_flash = 0.0;
                    false
                }
            }
        };
        if self.player.dimension == DimensionId::Nether {
            let pcx = (self.player.pos.x / 16.0).floor() as i32;
            let pcz = (self.player.pos.z / 16.0).floor() as i32;
            let reg_g = self.registries.read();
            let _ = ensure_nether_columns(
                &self.nether_world,
                pcx,
                pcz,
                2,
                self.nether_world.seed,
                &self.nether_palette,
                &reg_g,
            );
            for cz in (pcz - 2)..=(pcz + 2) {
                for cx in (pcx - 2)..=(pcx + 2) {
                    self.nether_generated_columns.insert((cx, cz));
                }
            }
        }

        if !had_saved_player && self.player.gamemode == GameMode::Creative {
            self.player_state.write().populate_creative_starter();
        }
        self.player.flying = self.player.gamemode == GameMode::Spectator;

        if !had_saved_player {
            let wt = *self.world.world_type.read();
            let sp = find_spawn_pos(self.world.seed, wt);
            self.player.pos = Vec3::new(sp.x as f32 + 0.5, sp.y as f32, sp.z as f32 + 0.5);
            self.player.vel = Vec3::ZERO;
        }
        self.player.wither_timer = 0.0;
        self.player.wither_tick_accum = 0.0;
        self.player.reset_fall_state();
        had_saved_player
    }

    /// Finalize player spawn safety check after initial chunks exist, sync state, and apply Lua/disk config.
    fn finalize_loaded_player_spawn(&mut self, check_saved_surface: bool) {
        if check_saved_surface && self.player.dimension == DimensionId::Overworld {
            let wt = *self.world.world_type.read();
            if matches!(wt, WorldGenType::Default | WorldGenType::Amplified) {
                let px = self.player.pos.x.floor() as i32;
                let py = self.player.pos.y.floor() as i32;
                let pz = self.player.pos.z.floor() as i32;
                let reg = self.registries.read();
                let is_solid = |p: IVec3| {
                    let (id, _, _) = self.world.get_node(p);
                    id != CONTENT_AIR && reg.get_node_def(id).map_or(false, |d| d.walkable)
                };
                if self.player.pos.y <= 16.0
                    || (is_solid(IVec3::new(px, py, pz)) && is_solid(IVec3::new(px, py + 1, pz)))
                {
                    let surface_y =
                        terrain_height_for(px, pz, self.world.seed, wt).max(SEA_LEVEL);
                    self.player.pos.y = (surface_y + 1) as f32;
                    self.player.vel = Vec3::ZERO;
                }
            }
        }
        if let Some(rp) = self.pending_remote_spawn.take() {
            self.player.pos = rp;
            self.player.vel = Vec3::ZERO;
        }
        *self.host_spawn_shared.write() = [
            self.player.pos.x,
            self.player.pos.y,
            self.player.pos.z,
        ];
        {
            let mut ps = self.player_state.write();
            ps.pos = self.player.pos;
            ps.hp = self.player.hp;
            ps.hunger = self.player.hunger;
            ps.wield_index = self.player.selected_slot + 1;
            let mut mg = ps.meta.write();
            mg.fields.insert("mcl_health".into(), self.player.hp.to_string());
            mg.fields.insert("mcl_hunger:hunger".into(), self.player.hunger.to_string());
            mg.fields.insert(
                "mcl_hunger:saturation".into(),
                format!("{:.2}", self.player.saturation),
            );
        }
        self.last_inv_total_count = total_inventory_items(&self.player_state);

        if let Some(ref rt) = self.lua_rt {
            rt.set_gamemode(self.player_state.clone(), self.player.gamemode.as_str());
        }
        self.apply_settings_to_lua_and_disk();
        if self.settings.coop_mode.is_active() {
            self.ensure_coop_players_joined();
        }
        if is_carterpcs_skin(&self.settings.player_skin) {
            equip_ram_mic_for_carterpcs(&self.player_state, &mut self.player.selected_slot);
            self.last_inv_total_count = total_inventory_items(&self.player_state);
        }
    }

    fn enter_world(&mut self, info: WorldInfo, renderer: &mut WgpuRenderer) {
        let had_saved_player = self.prepare_world_session(&info, renderer);
        self.generate_spawn_area(3, !had_saved_player);
        self.finalize_loaded_player_spawn(had_saved_player);
        self.sync_skins_and_props(Some(renderer), None, false);
        self.pending_load = None;
        self.mode = AppMode::Playing;
    }

    fn ensure_coop_players_joined(&mut self) {
        let want_count = self
            .settings
            .coop_mode
            .active_player_count(self.settings.coop_player_count)
            .clamp(1, 4);
        let offsets = [
            Vec3::new(1.5, 0.2, 0.0),
            Vec3::new(-1.5, 0.2, 0.0),
            Vec3::new(0.0, 0.2, 1.5),
        ];

        for slot in 1..4usize {
            let idx = slot - 1;
            if slot >= want_count {
                self.peers[idx].joined = false;
                continue;
            }
            if self.peers[idx].joined {
                continue;
            }
            let raw_name = self.settings.player_name_for_slot(slot).trim();
            let pname = if raw_name.is_empty() {
                format!("Player{}", slot + 1)
            } else {
                raw_name.to_string()
            };
            self.peers[idx].player_state = Arc::new(RwLock::new(ObjectState::new_player(&pname)));
            if let Some(ref rt) = self.lua_rt {
                let _ = rt.join_player(self.peers[idx].player_state.clone());
            }

            let loaded = {
                let mut ps_peer = self.peers[idx].player_state.write();
                load_player_by_name(&self.world, &mut ps_peer)
            };
            if let Some(sp) = loaded {
                self.peers[idx].player.pos = Vec3::new(sp.pos[0], sp.pos[1], sp.pos[2]);
                self.peers[idx].player.yaw = sp.yaw;
                self.peers[idx].player.pitch = sp.pitch;
                self.peers[idx].player.hp = if sp.hp <= 0 { 20 } else { sp.hp.min(20) };
                self.peers[idx].player.hunger = if sp.hunger <= 0 { 20 } else { sp.hunger.min(20) };
                self.peers[idx].player.saturation = (self.peers[idx].player.hunger as f32).min(5.0);
                self.peers[idx].player.exhaustion = 0.0;
                self.peers[idx].player.regen_timer = 0.0;
                self.peers[idx].player.starve_timer = 0.0;
                self.peers[idx].player.eat_cooldown = 0.0;
                self.peers[idx].player.vel = Vec3::ZERO;
                self.peers[idx].player.selected_slot = sp.wield_index.saturating_sub(1).min(8);
                self.peers[idx].player.gamemode = parse_gamemode(&sp.gamemode);
                self.peers[idx].player.dimension = sp.dimension;
                if sp.dimension == DimensionId::Nether {
                    let pcx = (self.peers[idx].player.pos.x / 16.0).floor() as i32;
                    let pcz = (self.peers[idx].player.pos.z / 16.0).floor() as i32;
                    let reg_g = self.registries.read();
                    let _ = ensure_nether_columns(
                        &self.nether_world,
                        pcx,
                        pcz,
                        2,
                        self.nether_world.seed,
                        &self.nether_palette,
                        &reg_g,
                    );
                }
            } else {
                self.peers[idx].player.pos = self.player.pos + offsets[idx];
                self.peers[idx].player.dimension = self.player.dimension;
                self.peers[idx].player.yaw = self.player.yaw;
                self.peers[idx].player.pitch = 0.0;
                self.peers[idx].player.vel = Vec3::ZERO;
                self.peers[idx].player.hp = 20;
                self.peers[idx].player.hunger = 20;
                self.peers[idx].player.saturation = 5.0;
                self.peers[idx].player.exhaustion = 0.0;
                self.peers[idx].player.regen_timer = 0.0;
                self.peers[idx].player.starve_timer = 0.0;
                self.peers[idx].player.eat_cooldown = 0.0;
                self.peers[idx].player.selected_slot = 0;
                self.peers[idx].player.gamemode = self.player.gamemode;
                if self.peers[idx].player.gamemode == GameMode::Creative {
                    self.peers[idx].player_state.write().populate_creative_starter();
                }
            }
            self.peers[idx].player.flying = self.peers[idx].player.gamemode == GameMode::Spectator;
            self.peers[idx].player.reset_fall_state();
            {
                let mut ps_peer = self.peers[idx].player_state.write();
                ps_peer.pos = self.peers[idx].player.pos;
                ps_peer.dimension = self.peers[idx].player.dimension;
                ps_peer.hp = self.peers[idx].player.hp;
                ps_peer.hunger = self.peers[idx].player.hunger;
                ps_peer.wield_index = self.peers[idx].player.selected_slot + 1;
                let mut mg = ps_peer.meta.write();
                mg.fields
                    .insert("mcl_health".into(), self.peers[idx].player.hp.to_string());
                mg.fields.insert(
                    "mcl_hunger:hunger".into(),
                    self.peers[idx].player.hunger.to_string(),
                );
                mg.fields.insert(
                    "mcl_hunger:saturation".into(),
                    format!("{:.2}", self.peers[idx].player.saturation),
                );
            }
            if let Some(ref rt) = self.lua_rt {
                rt.set_gamemode(
                    self.peers[idx].player_state.clone(),
                    self.peers[idx].player.gamemode.as_str(),
                );
            }
            if slot == 1 && is_carterpcs_skin(&self.settings.p2_skin) {
                equip_ram_mic_for_carterpcs(
                    &self.peers[idx].player_state,
                    &mut self.peers[idx].player.selected_slot,
                );
            }
            self.peers[idx].last_inv_total_count =
                total_inventory_items(&self.peers[idx].player_state);
            self.peers[idx].joined = true;
        }
    }

    fn save_active_world(&self) {
        if !matches!(self.mode, AppMode::Splash | AppMode::Title) && self.net_session.is_none() {
            let reg = self.registries.read();
            let ps1 = self.player_state.read();
            let peer_guards: Vec<_> = self
                .peers
                .iter()
                .filter(|p| p.joined)
                .map(|p| p.player_state.read())
                .collect();
            if !peer_guards.is_empty() {
                let mut refs: Vec<&ObjectState> = Vec::with_capacity(1 + peer_guards.len());
                refs.push(&*ps1);
                for g in &peer_guards {
                    refs.push(&**g);
                }
                if let Err(e) = save_world_players(&self.world, &reg, &refs) {
                    tracing::warn!("Failed to save co-op world: {e}");
                }
            } else if let Err(e) = save_world(&self.world, &reg, Some(&ps1)) {
                tracing::warn!("Failed to save world: {e}");
            }
            if let Err(e) = save_dimension_chunks(&self.nether_world, &reg) {
                tracing::warn!("Failed to save Nether chunks: {e}");
            }
        }
    }

    fn handle_chat_submission(&mut self, line: String) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return;
        }
        if let Some(ref session) = self.net_session {
            session.send(NetPacket::Chat {
                sender: self.settings.player_name.clone(),
                message: trimmed.to_string(),
            });
            return;
        }
        if !trimmed.starts_with('/') {
            if let Some(ref srv) = self.local_server {
                let _ = srv.tx.send(NetPacket::Chat {
                    sender: self.settings.player_name.clone(),
                    message: trimmed.to_string(),
                });
                return;
            }
        }

        if let Some(cmd_body) = trimmed.strip_prefix('/') {
            let mut parts = cmd_body.split_whitespace();
            let cmd = parts.next().unwrap_or("");
            match cmd {
                "help" => {
                    self.chat_messages.push(
                        "§eCommands: /help, /time <day|night|sunrise|sunset|0..1>, /gamemode <survival|creative|spectator>, /weather <clear|rain|thunder>, /seed, /tp <x> <y> <z> [overworld|nether], /locate <fortress|bastion|monument|temple> [info]".into(),
                    );
                }
                "seed" => {
                    self.chat_messages
                        .push(format!("§bWorld Seed: {}", self.world.seed));
                }
                "time" => {
                    let arg = parts.next().unwrap_or("day");
                    let tod = match arg {
                        "day" | "noon" => 0.5,
                        "night" | "midnight" => 0.0,
                        "sunrise" => 0.25,
                        "sunset" => 0.75,
                        other => other.parse::<f32>().unwrap_or(0.5).clamp(0.0, 1.0),
                    };
                    if let Some(ref rt) = self.lua_rt {
                        *rt.shared.time_of_day.write() = tod;
                    }
                    self.chat_messages
                        .push(format!("§eTime of day set to {tod:.2}"));
                }
                "weather" => {
                    let w = parts.next().unwrap_or("clear");
                    let label = match w.to_ascii_lowercase().as_str() {
                        "rain" => "Rain",
                        "thunder" | "storm" => "Thunder",
                        _ => "Clear",
                    };
                    self.pause_state.weather = label.to_string();
                    self.chat_messages
                        .push(format!("§bWeather set to {label}"));
                }
                "gamemode" | "gm" => {
                    let arg = parts.next().unwrap_or("survival");
                    let gm = parse_gamemode(arg);
                    self.player.gamemode = gm;
                    self.player.flying = gm == GameMode::Spectator;
                    *self.world.gamemode.write() = gm.label().to_string();
                    if let Some(ref rt) = self.lua_rt {
                        rt.set_gamemode(self.player_state.clone(), gm.as_str());
                    }
                    self.chat_messages
                        .push(format!("§aGame mode set to {}", gm.label()));
                }
                "tp" => {
                    if let (Some(xs), Some(ys), Some(zs)) = (parts.next(), parts.next(), parts.next())
                    {
                        if let (Ok(x), Ok(y), Ok(z)) =
                            (xs.parse::<f32>(), ys.parse::<f32>(), zs.parse::<f32>())
                        {
                            let target_dim = match parts.next().map(|s| s.to_ascii_lowercase()).as_deref() {
                                Some("nether") | Some("the_nether") => DimensionId::Nether,
                                Some("overworld") | Some("ow") => DimensionId::Overworld,
                                _ => self.player.dimension,
                            };
                            self.teleport_player_to(target_dim, Vec3::new(x, y, z));
                            self.chat_messages.push(format!(
                                "§aTeleported to ({x:.1}, {y:.1}, {z:.1}) in {}",
                                target_dim.label()
                            ));
                        }
                    }
                }
                "locate" => {
                    let arg = parts.next().unwrap_or("");
                    let info_only = parts
                        .next()
                        .is_some_and(|s| s.eq_ignore_ascii_case("info"));
                    if let Some(kind) = StructureKind::from_command_arg(arg) {
                        let target_dim = kind.dimension();
                        let cur_x = self.player.pos.x.floor() as i32;
                        let cur_z = self.player.pos.z.floor() as i32;
                        let near_xz = match (self.player.dimension, target_dim) {
                            (DimensionId::Overworld, DimensionId::Nether) => {
                                (cur_x.div_euclid(8), cur_z.div_euclid(8))
                            }
                            (DimensionId::Nether, DimensionId::Overworld) => (cur_x * 8, cur_z * 8),
                            _ => (cur_x, cur_z),
                        };
                        let wt = *self.world.world_type.read();
                        let seed = if target_dim.is_nether() {
                            self.nether_world.seed
                        } else {
                            self.world.seed
                        };
                        if let Some(anchor) = locate_nearest_structure(kind, seed, wt, near_xz) {
                            self.chat_messages.push(format!(
                                "§aLocated {} at ({}, {}, {}) [{}]",
                                kind.name(),
                                anchor.origin.x,
                                anchor.origin.y,
                                anchor.origin.z,
                                target_dim.label()
                            ));
                            if !info_only {
                                self.teleport_player_to(target_dim, anchor.teleport_pos);
                                self.chat_messages.push(format!(
                                    "§bTeleported to {}!",
                                    kind.name()
                                ));
                            }
                        } else {
                            self.chat_messages.push(format!(
                                "§cCould not find {} nearby (WorldType={}).",
                                kind.name(),
                                wt.label()
                            ));
                        }
                    } else {
                        self.chat_messages.push(
                            "§eUsage: /locate <fortress|bastion|monument|temple> [info]".into(),
                        );
                    }
                }
                other => {
                    self.chat_messages
                        .push(format!("§cUnknown command: /{other}. Try /help"));
                }
            }
        } else {
            self.chat_messages
                .push(format!("<{}> {trimmed}", self.settings.player_name));
        }
    }

    fn teleport_player_to(&mut self, target_dim: DimensionId, target_pos: Vec3) {
        let cx = (target_pos.x / 16.0).floor() as i32;
        let cz = (target_pos.z / 16.0).floor() as i32;
        {
            let reg = self.registries.read();
            if target_dim.is_nether() {
                let _ = ensure_nether_columns(
                    &self.nether_world,
                    cx,
                    cz,
                    2,
                    self.nether_world.seed,
                    &self.nether_palette,
                    &reg,
                );
                for dz in -2..=2 {
                    for dx in -2..=2 {
                        self.nether_generated_columns.insert((cx + dx, cz + dz));
                    }
                }
                let mut pq = self.nether_world.priority_dirty_queue.write();
                for dz in -1..=1 {
                    for dx in -1..=1 {
                        for cy in NETHER_CHUNK_Y_RANGE {
                            pq.insert(IVec3::new(cx + dx, cy, cz + dz));
                        }
                    }
                }
            } else {
                ensure_overworld_columns(&self.world, cx, cz, 2, &self.palette, &reg);
                for dz in -2..=2 {
                    for dx in -2..=2 {
                        self.generated_columns.insert((cx + dx, cz + dz));
                    }
                }
                let wt = *self.world.world_type.read();
                let mut pq = self.world.priority_dirty_queue.write();
                for dz in -1..=1 {
                    for dx in -1..=1 {
                        for cy in chunk_y_range_for(wt) {
                            pq.insert(IVec3::new(cx + dx, cy, cz + dz));
                        }
                    }
                }
            }
        }
        self.player.dimension = target_dim;
        self.player.pos = target_pos;
        self.player.vel = Vec3::ZERO;
        self.player.portal_charge = 0.0;
        self.player.portal_cooldown = 2.0;
        self.player.spawn_invuln_timer = self.player.spawn_invuln_timer.max(2.0);
        self.player.wither_timer = 0.0;
        self.player.wither_tick_accum = 0.0;
        self.player.reset_fall_state();
        let mut ps = self.player_state.write();
        ps.dimension = target_dim;
        ps.pos = target_pos;
        ps.spawn_invuln_timer = ps.spawn_invuln_timer.max(2.0);
    }
}

fn parse_gamemode(s: &str) -> GameMode {
    let lower = s.trim().to_ascii_lowercase();
    if lower == "creative" || lower == "1" || lower == "c" {
        GameMode::Creative
    } else if lower == "spectator" || lower == "3" || lower == "sp" {
        GameMode::Spectator
    } else {
        GameMode::Survival
    }
}

fn total_inventory_items(player_state: &Arc<RwLock<ObjectState>>) -> u32 {
    let ps = player_state.read();
    let mg = ps.meta.read();
    mg.inventories
        .get("main")
        .map(|v| v.iter().map(|s| s.count as u32).sum())
        .unwrap_or(0)
}

fn keycode_matches(code: KeyCode, configured: Option<&str>, default_code: KeyCode) -> bool {
    let Some(lbl) = configured else {
        return code == default_code;
    };
    let u = lbl.trim().to_ascii_uppercase();
    let expected = match u.as_str() {
        "W" | "KEYW" => Some(KeyCode::KeyW),
        "S" | "KEYS" => Some(KeyCode::KeyS),
        "A" | "KEYA" => Some(KeyCode::KeyA),
        "D" | "KEYD" => Some(KeyCode::KeyD),
        "E" | "KEYE" => Some(KeyCode::KeyE),
        "C" | "KEYC" => Some(KeyCode::KeyC),
        "Q" | "KEYQ" => Some(KeyCode::KeyQ),
        "T" | "KEYT" => Some(KeyCode::KeyT),
        "K" | "KEYK" => Some(KeyCode::KeyK),
        "H" | "KEYH" => Some(KeyCode::KeyH),
        "F" | "KEYF" => Some(KeyCode::KeyF),
        "R" | "KEYR" => Some(KeyCode::KeyR),
        "G" | "KEYG" => Some(KeyCode::KeyG),
        "I" | "KEYI" => Some(KeyCode::KeyI),
        "B" | "KEYB" => Some(KeyCode::KeyB),
        "Z" | "KEYZ" => Some(KeyCode::KeyZ),
        "X" | "KEYX" => Some(KeyCode::KeyX),
        "SPACE" => Some(KeyCode::Space),
        "LSHIFT" | "SHIFTLEFT" => Some(KeyCode::ShiftLeft),
        "RSHIFT" | "SHIFTRIGHT" => Some(KeyCode::ShiftRight),
        "LCONTROL" | "CONTROLLEFT" => Some(KeyCode::ControlLeft),
        "RCONTROL" | "CONTROLRIGHT" => Some(KeyCode::ControlRight),
        "ARROWUP" => Some(KeyCode::ArrowUp),
        "ARROWDOWN" => Some(KeyCode::ArrowDown),
        "ARROWLEFT" => Some(KeyCode::ArrowLeft),
        "ARROWRIGHT" => Some(KeyCode::ArrowRight),
        "TAB" => Some(KeyCode::Tab),
        _ => None,
    };
    expected.map(|k| k == code).unwrap_or(code == default_code)
}

struct SecondaryWindowState {
    // Drop GPU swapchain target BEFORE Arc<Window> so Surface never outlives Window
    target: Option<SecondaryRenderTarget>,
    egui_state: egui_winit::State,
    theme: McTheme,
    egui_ctx: egui::Context,
    window: Arc<Window>,
    fullscreen: bool,
    player_slots: Vec<usize>,
}

struct WindowedApp {
    engine: CubexEngine,
    // Strict field declaration order = reverse Rust drop order safety:
    // secondary_windows (dropped first) -> renderer (Surface/Device) -> egui/theme -> window (dropped last)
    secondary_windows: Vec<SecondaryWindowState>,
    renderer: Option<WgpuRenderer>,
    egui_ctx: egui::Context,
    egui_state: Option<egui_winit::State>,
    theme: Option<McTheme>,
    window: Option<Arc<Window>>,
    hud_icons: ItemIconCache,
    form_icons: ItemIconCache,
    inv_cursor: InventoryCursor,
    p1_creative_state: CreativeInventoryState,
    window_focused: bool,
    cursor_locked: bool,
    fullscreen: bool,
    lmb_down: bool,
    lmb_just_pressed: bool,
    rmb_just_pressed: bool,
    mmb_just_pressed: bool,
    last_smooth_lighting: bool,
    camera_mode: CameraMode,
    gamepads: GamepadManager,
    p1_ctrl_sneak: bool,
    p1_ctrl_sprint: bool,
    p1_place_cd: f32,
    p1_attack_tflp: f32,
    p1_mine_was_held: bool,
    p1_form_slot: usize,
    kb2: Kb2State,
    touch_controls: TouchControlsState,
    last_touch_viewport: Option<egui::Rect>,
    mouse_touch_down: bool,
    pending_touch_egui_events: Vec<egui::Event>,
    boot_state: StartupBootState,
    splash_fade: f32,
    torch_ember_timer: f32,
    pending_autotest_config: Option<AutoTestConfig>,
    autotest_bot: Option<SurvivalNetherBot>,
    pending_autotest_finish: Option<(PathBuf, AutoTestReport)>,
    pending_qa_playthrough: Option<PathBuf>,
    qa_bot: Option<QaPlaythroughBot>,
    pending_qa_finish: Option<(PathBuf, PlaythroughReport)>,
}

impl Drop for WindowedApp {
    fn drop(&mut self) {
        self.shutdown_gpu_resources();
        self.engine.stop_lan_server();
        self.engine.tunnel.stop();
    }
}

impl WindowedApp {
    #[cfg(test)]
    fn new(engine: CubexEngine) -> Self {
        let last_smooth_lighting = engine.settings.smooth_lighting;
        Self {
            engine,
            secondary_windows: Vec::new(),
            renderer: None,
            egui_ctx: egui::Context::default(),
            egui_state: None,
            theme: None,
            window: None,
            hud_icons: ItemIconCache::new(),
            form_icons: ItemIconCache::new(),
            inv_cursor: InventoryCursor::default(),
            p1_creative_state: CreativeInventoryState::default(),
            window_focused: true,
            cursor_locked: false,
            fullscreen: false,
            lmb_down: false,
            lmb_just_pressed: false,
            rmb_just_pressed: false,
            mmb_just_pressed: false,
            last_smooth_lighting,
            camera_mode: CameraMode::FirstPerson,
            gamepads: GamepadManager::new(),
            p1_ctrl_sneak: false,
            p1_ctrl_sprint: false,
            p1_place_cd: 0.0,
            p1_attack_tflp: 0.5,
            p1_mine_was_held: false,
            p1_form_slot: 0,
            kb2: Kb2State::default(),
            touch_controls: TouchControlsState::default(),
            last_touch_viewport: None,
            mouse_touch_down: false,
            pending_touch_egui_events: Vec::new(),
            boot_state: StartupBootState::Ready,
            splash_fade: 0.0,
            torch_ember_timer: 0.0,
            pending_autotest_config: None,
            autotest_bot: None,
            pending_autotest_finish: None,
            pending_qa_playthrough: None,
            qa_bot: None,
            pending_qa_finish: None,
        }
    }

    /// Instant shell boot (<15ms) so Frame 1 of the Vortex Studios splash renders in <= 0.4s.
    fn new_booting(paths: EnginePaths) -> anyhow::Result<Self> {
        let started_at = Instant::now();
        let engine = CubexEngine::boot_shell(paths)?;
        let rp_path = engine.resolve_rp_path();
        let (mods_rx, assets_rx) = spawn_asset_preloader(&engine.paths, rp_path);
        let last_smooth_lighting = engine.settings.smooth_lighting;
        Ok(Self {
            engine,
            secondary_windows: Vec::new(),
            renderer: None,
            egui_ctx: egui::Context::default(),
            egui_state: None,
            theme: None,
            window: None,
            hud_icons: ItemIconCache::new(),
            form_icons: ItemIconCache::new(),
            inv_cursor: InventoryCursor::default(),
            p1_creative_state: CreativeInventoryState::default(),
            window_focused: true,
            cursor_locked: false,
            fullscreen: false,
            lmb_down: false,
            lmb_just_pressed: false,
            rmb_just_pressed: false,
            mmb_just_pressed: false,
            last_smooth_lighting,
            camera_mode: CameraMode::FirstPerson,
            gamepads: GamepadManager::empty(),
            p1_ctrl_sneak: false,
            p1_ctrl_sprint: false,
            p1_place_cd: 0.0,
            p1_attack_tflp: 0.5,
            p1_mine_was_held: false,
            p1_form_slot: 0,
            kb2: Kb2State::default(),
            touch_controls: TouchControlsState::default(),
            last_touch_viewport: None,
            mouse_touch_down: false,
            pending_touch_egui_events: Vec::new(),
            boot_state: StartupBootState::WaitingForFirstPaint {
                started_at,
                mods_rx,
                assets_rx,
            },
            splash_fade: 1.0,
            torch_ember_timer: 0.0,
            pending_autotest_config: None,
            autotest_bot: None,
            pending_autotest_finish: None,
            pending_qa_playthrough: None,
            qa_bot: None,
            pending_qa_finish: None,
        })
    }

    fn new_booting_with_cli(
        paths: EnginePaths,
        autotest_cfg: Option<AutoTestConfig>,
        qa_playthrough_out: Option<PathBuf>,
    ) -> anyhow::Result<Self> {
        let mut app = Self::new_booting(paths)?;
        app.pending_autotest_config = autotest_cfg;
        app.pending_qa_playthrough = qa_playthrough_out;
        Ok(app)
    }

    fn boot_survival_bot_world(&mut self, world_dir_name: &str, world_label: &str) {
        self.engine.settings.coop_mode = CoopMode::Off;
        self.engine.settings.touch_controls = false;
        let world_dir = std::env::temp_dir().join(world_dir_name);
        let _ = std::fs::remove_dir_all(&world_dir);
        let _ = std::fs::create_dir_all(&world_dir);
        let info = WorldInfo {
            name: world_label.into(),
            path: world_dir,
            gamemode: "Survival".into(),
            seed: SURVIVAL_NETHER_SEED,
            world_type: WorldGenType::Default,
            difficulty: Difficulty::Normal,
            gamerules: Gamerules::default(),
        };
        if let Some(ref mut r) = self.renderer {
            self.engine.enter_world(info, r);
        }
        self.engine.player.gamemode = GameMode::Survival;
        self.engine.player.dimension = DimensionId::Overworld;
        self.engine.player.hp = 20;
        self.engine.player.hunger = 20;
        self.engine.player.saturation = 5.0;
        self.engine.player.is_dead = false;
        self.engine.player.selected_slot = 0;
        self.engine.player.reset_fall_state();
        {
            let mut ps = self.engine.player_state.write();
            ps.hp = 20;
            ps.hunger = 20;
            ps.wield_index = 1;
            let mut mg = ps.meta.write();
            mg.inventories.insert("main".into(), vec![ItemStackData::default(); 32]);
            mg.inventories.insert("craft".into(), vec![ItemStackData::default(); 9]);
            mg.inventories.insert("craftpreview".into(), vec![ItemStackData::default(); 1]);
            mg.inventories.insert("armor".into(), vec![ItemStackData::default(); 4]);
            mg.inventories.insert("offhand".into(), vec![ItemStackData::default(); 1]);
        }
        self.engine.last_inv_total_count = 0;
        self.engine.active_formspec = None;
        self.engine.chat_typing = None;
        if let Some(ref rt) = self.engine.lua_rt {
            *rt.shared.time_of_day.write() = 0.28;
        }
        self.set_cursor_locked(false);
    }

    fn drop_secondary_windows(&mut self) {
        if self.secondary_windows.is_empty() {
            return;
        }
        if let Some(ref r) = self.renderer {
            r.wait_idle();
        }
        for mut sw in self.secondary_windows.drain(..) {
            drop(sw.target.take());
            drop(sw);
        }
        for peer in &mut self.engine.peers {
            peer.hud_icons = ItemIconCache::new();
            peer.form_icons = ItemIconCache::new();
        }
    }

    fn shutdown_gpu_resources(&mut self) {
        self.drop_secondary_windows();
        if let Some(ref r) = self.renderer {
            r.wait_idle();
        }
        self.renderer = None;
        self.egui_state = None;
        self.theme = None;
        self.window = None;
    }

    fn splash_screen_info(&self) -> SplashScreenInfo {
        match &self.boot_state {
            StartupBootState::WaitingForFirstPaint { .. } => SplashScreenInfo {
                stage: "Initializing Vortex Core...".into(),
                detail: "Preparing Vulkan renderer".into(),
                progress: 0.06,
            },
            StartupBootState::LoadingMods {
                loaded,
                total,
                current_mod,
                ..
            } => {
                let frac = if *total > 0 {
                    (*loaded as f32 / *total as f32).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let detail = if *total == 0 {
                    "Scanning Mineclonia mods...".into()
                } else if current_mod.is_empty() {
                    format!("{loaded} / {total} mods")
                } else {
                    format!("{loaded} / {total} ({current_mod})")
                };
                SplashScreenInfo {
                    stage: "Loading Mineclonia mods...".into(),
                    detail,
                    progress: 0.10 + 0.48 * frac,
                }
            }
            StartupBootState::BuildingAtlasAndPanorama {
                stage_atomic,
                done_atomic,
                total_atomic,
                ..
            } => {
                let st = stage_atomic.load(Ordering::Relaxed);
                let done = done_atomic.load(Ordering::Relaxed);
                let total = total_atomic.load(Ordering::Relaxed).max(1);
                let frac = (done as f32 / total as f32).clamp(0.0, 1.0);
                match st {
                    0 => SplashScreenInfo {
                        stage: "Packing texture atlas...".into(),
                        detail: "Resolving tiles & palettes".into(),
                        progress: 0.60,
                    },
                    1 => SplashScreenInfo {
                        stage: "Generating 3D world panorama...".into(),
                        detail: format!("{done} / {total} columns"),
                        progress: 0.64 + 0.16 * frac,
                    },
                    _ => SplashScreenInfo {
                        stage: "Meshing panorama terrain...".into(),
                        detail: format!("{done} / {total} chunks"),
                        progress: 0.80 + 0.12 * frac,
                    },
                }
            }
            StartupBootState::UploadingPanorama {
                total, uploaded, ..
            } => {
                let frac = if *total > 0 {
                    (*uploaded as f32 / *total as f32).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                SplashScreenInfo {
                    stage: "Uploading panorama to GPU...".into(),
                    detail: format!("{uploaded} / {total} chunks"),
                    progress: 0.92 + 0.08 * frac,
                }
            }
            StartupBootState::Ready => SplashScreenInfo {
                stage: "Ready!".into(),
                detail: "Welcome to Cubex-RS".into(),
                progress: 1.0,
            },
        }
    }

    fn step_startup_boot(&mut self) {
        let cur = std::mem::replace(&mut self.boot_state, StartupBootState::Ready);
        match cur {
            StartupBootState::WaitingForFirstPaint {
                started_at,
                mods_rx,
                assets_rx,
            } => {
                tracing::info!(
                    "[STARTUP] First Vortex Studios splash frame presented in {:.0} ms",
                    started_at.elapsed().as_secs_f64() * 1000.0
                );
                println!(
                    "[STARTUP] First Vortex Studios splash frame presented in {:.0} ms",
                    started_at.elapsed().as_secs_f64() * 1000.0
                );
                let (boot, loaded, total) = match mods_rx.try_recv() {
                    Ok(mods) => {
                        let total = mods.len();
                        let b = LuaIncrementalBoot::begin(
                            self.engine.paths.clone(),
                            self.engine.registries.clone(),
                            self.engine.world.clone(),
                            self.engine.config.clone(),
                            mods,
                        )
                        .expect("LuaIncrementalBoot::begin");
                        (Some(b), 0, total)
                    }
                    Err(_) => (None, 0, 0),
                };
                self.boot_state = StartupBootState::LoadingMods {
                    started_at,
                    boot,
                    mods_rx,
                    assets_rx,
                    preloaded_assets: None,
                    loaded,
                    total,
                    current_mod: String::new(),
                };
            }
            StartupBootState::LoadingMods {
                started_at,
                mut boot,
                mods_rx,
                assets_rx,
                mut preloaded_assets,
                mut loaded,
                mut total,
                mut current_mod,
            } => {
                if preloaded_assets.is_none() {
                    if let Ok(pair) = assets_rx.try_recv() {
                        preloaded_assets = Some(pair);
                    }
                }
                if boot.is_none() {
                    if let Ok(mods) = mods_rx.try_recv() {
                        total = mods.len();
                        boot = Some(
                            LuaIncrementalBoot::begin(
                                self.engine.paths.clone(),
                                self.engine.registries.clone(),
                                self.engine.world.clone(),
                                self.engine.config.clone(),
                                mods,
                            )
                            .expect("LuaIncrementalBoot::begin"),
                        );
                    }
                }
                if let Some(ref mut b) = boot {
                    if !b.is_done() {
                        let (l, t, last_name) = b.step_budget(Duration::from_millis(12));
                        loaded = l;
                        total = t;
                        if let Some(n) = last_name {
                            current_mod = n;
                        }
                    }
                    if b.is_done() {
                        if preloaded_assets.is_none() {
                            if let Ok(pair) = assets_rx.try_recv() {
                                preloaded_assets = Some(pair);
                            }
                        }
                        if let Some((preloaded, audio_arc)) = preloaded_assets.take() {
                            let b_done = boot.take().unwrap();
                            let lua_rt = b_done.finish();
                            self.engine.set_audio_loaded(audio_arc);
                            let _ = lua_rt.join_player(self.engine.player_state.clone());
                            self.engine.last_inv_total_count =
                                total_inventory_items(&self.engine.player_state);
                            self.engine.lua_rt = Some(lua_rt);

                            let (tx, rx) = mpsc::channel();
                            let stage_atomic = Arc::new(AtomicU8::new(0));
                            let done_atomic = Arc::new(AtomicUsize::new(0));
                            let total_atomic = Arc::new(AtomicUsize::new(1));
                            let st_c = stage_atomic.clone();
                            let dn_c = done_atomic.clone();
                            let tot_c = total_atomic.clone();
                            let reg_arc = self.engine.registries.clone();
                            let world_arc = self.engine.world.clone();
                            let smooth = self.engine.settings.smooth_lighting;

                            rayon::spawn(move || {
                                let bundle = build_panorama_bundle(
                                    preloaded, &reg_arc, &world_arc, smooth, &st_c, &dn_c, &tot_c,
                                );
                                let _ = tx.send(bundle);
                            });

                            self.boot_state = StartupBootState::BuildingAtlasAndPanorama {
                                started_at,
                                rx,
                                stage_atomic,
                                done_atomic,
                                total_atomic,
                            };
                            return;
                        }
                    }
                }
                self.boot_state = StartupBootState::LoadingMods {
                    started_at,
                    boot,
                    mods_rx,
                    assets_rx,
                    preloaded_assets,
                    loaded,
                    total,
                    current_mod,
                };
            }
            StartupBootState::BuildingAtlasAndPanorama {
                started_at,
                rx,
                stage_atomic,
                done_atomic,
                total_atomic,
            } => {
                if let Ok(bundle) = rx.try_recv() {
                    let rp_path = self.engine.resolve_rp_path();
                    self.engine.atlas = bundle.atlas;
                    self.engine.loaded_atlas_pack = Some(rp_path);
                    self.engine.palette = bundle.palette;
                    self.engine.generated_columns = bundle.generated_columns;
                    self.engine.panorama_center = bundle.panorama_center;
                    self.engine.panorama_fog_dist = 92.0;
                    self.engine.refresh_available_skins();

                    if let Some(ref mut r) = self.renderer {
                        r.reload_atlas(&self.engine.atlas);
                        r.fog_dist_override = Some(self.engine.panorama_fog_dist);
                    }
                    self.engine
                        .sync_skins_and_props(self.renderer.as_mut(), Some(&self.egui_ctx), false);

                    let total = bundle.meshes.len();
                    let pending = VecDeque::from(bundle.meshes);
                    self.boot_state = StartupBootState::UploadingPanorama {
                        started_at,
                        pending,
                        total,
                        uploaded: 0,
                    };
                } else {
                    self.boot_state = StartupBootState::BuildingAtlasAndPanorama {
                        started_at,
                        rx,
                        stage_atomic,
                        done_atomic,
                        total_atomic,
                    };
                }
            }
            StartupBootState::UploadingPanorama {
                started_at,
                mut pending,
                total,
                mut uploaded,
            } => {
                if let Some(ref mut r) = self.renderer {
                    let t0 = Instant::now();
                    let mut n = 0usize;
                    while !pending.is_empty() {
                        if n > 0 && t0.elapsed().as_secs_f32() > 0.010 {
                            break;
                        }
                        if let Some(mesh) = pending.pop_front() {
                            r.upload_chunk_mesh_now(mesh);
                            uploaded += 1;
                            n += 1;
                        }
                    }
                    r.fog_dist_override = Some(self.engine.panorama_fog_dist);
                } else {
                    pending.clear();
                }

                if pending.is_empty() {
                    self.gamepads = GamepadManager::new();
                    self.splash_fade = 1.0;
                    self.engine.mode = AppMode::Title;
                    self.boot_state = StartupBootState::Ready;
                    println!(
                        "[STARTUP] Title Screen 3D Panorama ready ({uploaded} chunks uploaded) in {:.0} ms",
                        started_at.elapsed().as_secs_f64() * 1000.0
                    );
                    if let Some(out_dir) = self.pending_qa_playthrough.take() {
                        self.pending_autotest_config = None;
                        self.boot_survival_bot_world("cubex_rs_qa_playthrough", "QA_Survival_Nether");
                        self.qa_bot = Some(QaPlaythroughBot::new(out_dir, self.engine.player.pos));
                        println!(
                            "[QA BOT] Started live 7-stage QA Playthrough Bot on seed {}",
                            SURVIVAL_NETHER_SEED
                        );
                    } else if let Some(cfg) = self.pending_autotest_config.take() {
                        self.boot_survival_bot_world(
                            "cubex_rs_autotest_survival_nether",
                            "AutoTest_Survival_Nether",
                        );
                        self.autotest_bot = Some(SurvivalNetherBot::new(cfg, self.engine.player.pos));
                        println!(
                            "[AUTOBOT] Started Survival-to-Nether AutoBot on seed {}",
                            SURVIVAL_NETHER_SEED
                        );
                    }
                } else {
                    self.boot_state = StartupBootState::UploadingPanorama {
                        started_at,
                        pending,
                        total,
                        uploaded,
                    };
                }
            }
            StartupBootState::Ready => {
                self.boot_state = StartupBootState::Ready;
            }
        }
    }
    fn should_capture_cursor(&self) -> bool {
        self.engine.mode == AppMode::Playing
            && self.window_focused
            && !self.engine.player.is_dead
            && self.engine.active_formspec.is_none()
            && self.engine.chat_typing.is_none()
            && !self.engine.settings.touch_controls
            && self.autotest_bot.is_none()
            && self.pending_autotest_config.is_none()
            && self.qa_bot.is_none()
            && self.pending_qa_playthrough.is_none()
    }

    fn is_gameplay_captured(&self) -> bool {
        self.should_capture_cursor() && self.cursor_locked
    }

    fn set_cursor_locked(&mut self, locked: bool) {
        let effective_lock = locked && self.should_capture_cursor();
        if !effective_lock {
            self.cursor_locked = false;
            self.lmb_down = false;
            self.rmb_just_pressed = false;
            self.mmb_just_pressed = false;
            if let Some(ref win) = self.window {
                let _ = win.set_cursor_grab(CursorGrabMode::None);
                win.set_cursor_visible(true);
            }
            return;
        }
        if let Some(ref win) = self.window {
            let res = win
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| win.set_cursor_grab(CursorGrabMode::Confined));
            if res.is_ok() {
                win.set_cursor_visible(false);
                self.cursor_locked = true;
            } else {
                let _ = win.set_cursor_grab(CursorGrabMode::None);
                win.set_cursor_visible(true);
                self.cursor_locked = false;
            }
        } else {
            self.cursor_locked = true;
        }
    }

    fn on_focus_changed(&mut self, focused: bool) {
        self.window_focused = focused;
        if !focused {
            self.set_cursor_locked(false);
            self.engine.move_input = MoveInput::default();
            self.engine.digger.reset();
        }
    }

    fn return_cursor_to_inventory(&mut self) {
        let held = std::mem::take(&mut self.inv_cursor.held);
        if !held.is_empty() {
            let ps = self.engine.player_state.read();
            let mut mg = ps.meta.write();
            let main_slots = mg
                .inventories
                .entry("main".into())
                .or_insert_with(|| vec![ItemStackData::default(); 36]);
            let mut src = vec![held];
            shift_transfer_slot(&mut src, 0, main_slots, 64);
        }
    }

    fn close_active_formspec(&mut self) {
        let was_creative = self
            .engine
            .active_formspec
            .as_ref()
            .is_some_and(|f| f.formname == CREATIVE_INVENTORY_FORMNAME);
        self.engine.active_formspec = None;
        if was_creative {
            self.inv_cursor.clear();
        } else {
            self.return_cursor_to_inventory();
            {
                let ps = self.engine.player_state.read();
                let reg = self.engine.registries.read();
                ensure_craft_grid_size(&ps.meta, 4);
                refresh_craft_preview(&ps.meta, &reg);
            }
        }
        if self.engine.mode == AppMode::Playing && self.engine.chat_typing.is_none() {
            self.set_cursor_locked(true);
        }
    }

    fn close_peer_formspec(&mut self, slot: usize) {
        let Some(peer) = self.engine.peers.get_mut(slot.saturating_sub(1)) else {
            return;
        };
        let was_creative = peer
            .active_formspec
            .as_ref()
            .is_some_and(|f| f.formname == CREATIVE_INVENTORY_FORMNAME);
        peer.active_formspec = None;
        let held = std::mem::take(&mut peer.inv_cursor.held);
        if !was_creative {
            if !held.is_empty() {
                let ps = peer.player_state.read();
                let mut mg = ps.meta.write();
                let main_slots = mg
                    .inventories
                    .entry("main".into())
                    .or_insert_with(|| vec![ItemStackData::default(); 36]);
                let mut src = vec![held];
                shift_transfer_slot(&mut src, 0, main_slots, 64);
            }
            {
                let ps = peer.player_state.read();
                let reg = self.engine.registries.read();
                ensure_craft_grid_size(&ps.meta, 4);
                refresh_craft_preview(&ps.meta, &reg);
            }
        }
    }

    fn open_loot_chest_formspec(&mut self, slot: usize, pos: IVec3) {
        let dim = if slot == 0 {
            self.engine.player.dimension
        } else if let Some(p) = self.engine.peers.get(slot - 1) {
            p.player.dimension
        } else {
            return;
        };
        let w = world_for(&self.engine.world, &self.engine.nether_world, dim);
        let title = w
            .peek_node_meta(pos)
            .and_then(|m| m.read().fields.get("infotext").cloned())
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| "Loot Chest".into());
        let raw = build_loot_chest_formspec(pos, &title);
        let parsed = parse_formspec(&raw);
        self.engine.audio.play_named(
            "default_chest_open",
            0.7,
            1.0,
            SoundCategory::Blocks,
            &self.engine.settings.audio,
        );
        let formname = format!("cubex:chest:{},{},{}", pos.x, pos.y, pos.z);
        if slot == 0 {
            self.inv_cursor.clear();
            self.engine.active_formspec = Some(ActiveFormspec {
                formname,
                raw_formspec: raw,
                parsed,
                context_pos: Some(pos),
            });
            self.p1_form_slot = 0;
            self.engine.move_input = MoveInput::default();
            self.engine.digger.reset();
            self.set_cursor_locked(false);
        } else if let Some(peer) = self.engine.peers.get_mut(slot - 1) {
            peer.inv_cursor.clear();
            peer.active_formspec = Some(ActiveFormspec {
                formname,
                raw_formspec: raw,
                parsed,
                context_pos: Some(pos),
            });
            peer.form_slot_idx = 0;
            peer.move_input = MoveInput::default();
            peer.digger.reset();
        }
    }

    fn apply_lua_formspec_request(
        &mut self,
        req: cubex_rs::lua_api::core_api::FormspecRequest,
        fallback_pos: Option<IVec3>,
    ) {
        let peer_idx = if !req.player_name.is_empty() {
            self.engine
                .peers
                .iter()
                .position(|p| p.joined && req.player_name == p.player_state.read().name)
        } else {
            None
        };
        if let Some(idx) = peer_idx {
            let slot = idx + 1;
            if self.engine.peers[idx].player.is_dead {
                return;
            }
            if req.formspec.is_empty() {
                self.close_peer_formspec(slot);
                return;
            }
            let parsed = parse_formspec(&req.formspec);
            let peer = &mut self.engine.peers[idx];
            if formspec_requires_3x3_craft(&parsed) {
                let ps = peer.player_state.read();
                ensure_craft_grid_size(&ps.meta, 9);
            }
            let ctx_pos = resolve_inv_node_pos(&req.formname, None)
                .or(peer.last_interact_pos)
                .or(fallback_pos);
            peer.active_formspec = Some(ActiveFormspec {
                formname: req.formname,
                raw_formspec: req.formspec,
                parsed,
                context_pos: ctx_pos,
            });
            peer.form_slot_idx = 0;
            peer.move_input = MoveInput::default();
            peer.digger.reset();
            return;
        }
        if self.engine.player.is_dead {
            return;
        }
        if req.formspec.is_empty() {
            self.close_active_formspec();
            return;
        }
        let parsed = parse_formspec(&req.formspec);
        if formspec_requires_3x3_craft(&parsed) {
            let ps = self.engine.player_state.read();
            ensure_craft_grid_size(&ps.meta, 9);
        }
        let ctx_pos = resolve_inv_node_pos(&req.formname, None).or(fallback_pos);
        self.engine.active_formspec = Some(ActiveFormspec {
            formname: req.formname,
            raw_formspec: req.formspec,
            parsed,
            context_pos: ctx_pos,
        });
        self.p1_form_slot = 0;
        self.engine.move_input = MoveInput::default();
        self.engine.digger.reset();
        self.set_cursor_locked(false);
    }

    fn open_peer_inventory_formspec(&mut self, slot: usize) {
        let Some(peer) = self.engine.peers.get_mut(slot.saturating_sub(1)) else {
            return;
        };
        if peer.player.gamemode == GameMode::Creative {
            peer.inv_cursor.clear();
            {
                let reg = self.engine.registries.read();
                peer.creative_state.ensure_synced(&reg);
            }
            peer.active_formspec = Some(ActiveFormspec {
                formname: CREATIVE_INVENTORY_FORMNAME.into(),
                raw_formspec: String::new(),
                parsed: ParsedFormspec::default(),
                context_pos: None,
            });
            peer.form_slot_idx = 0;
            peer.move_input = MoveInput::default();
            peer.digger.reset();
            return;
        }
        let held = std::mem::take(&mut peer.inv_cursor.held);
        if !held.is_empty() {
            let ps = peer.player_state.read();
            let mut mg = ps.meta.write();
            let main_slots = mg
                .inventories
                .entry("main".into())
                .or_insert_with(|| vec![ItemStackData::default(); 36]);
            let mut src = vec![held];
            shift_transfer_slot(&mut src, 0, main_slots, 64);
        }
        {
            let ps = peer.player_state.read();
            let reg = self.engine.registries.read();
            ensure_craft_grid_size(&ps.meta, 4);
            refresh_craft_preview(&ps.meta, &reg);
        }
        let raw_inv_fs = "formspec_version[4]size[11.75,9.6]label[2.25,0.35;Crafting]list[current_player;craft;2.25,0.75;2,2;0]image[5.0,1.35;1.25,1.0;gui_crafting_arrow.png]list[current_player;craftpreview;6.6,1.35;1,1;]label[0.375,3.4;Inventory]list[current_player;main;0.375,3.8;9,3;9]list[current_player;main;0.375,7.85;9,1;0]listring[current_player;main]listring[current_player;craft]button_exit[9.0,0.75;2.2,0.8;close;Done]".to_string();
        let parsed = parse_formspec(&raw_inv_fs);
        peer.active_formspec = Some(ActiveFormspec {
            formname: "cubex:inventory".into(),
            raw_formspec: raw_inv_fs,
            parsed,
            context_pos: None,
        });
        peer.form_slot_idx = 0;
        peer.move_input = MoveInput::default();
        peer.digger.reset();
    }

    fn desired_secondary_window_slots(mode: CoopMode, player_count: u8) -> Vec<Vec<usize>> {
        let n = mode.active_player_count(player_count);
        match mode {
            CoopMode::DualMonitor if n >= 2 => vec![vec![1]],
            CoopMode::DualScreenTwoEach if n >= 3 => {
                if n == 3 {
                    vec![vec![2]]
                } else {
                    vec![vec![2, 3]]
                }
            }
            CoopMode::DualScreenTwoEach if n == 2 => vec![vec![1]],
            CoopMode::QuadMonitor => (1..n).map(|s| vec![s]).collect(),
            _ => Vec::new(),
        }
    }

    fn sync_coop_window(&mut self, event_loop: &ActiveEventLoop) {
        let desired = if matches!(self.engine.mode, AppMode::Splash | AppMode::Title) {
            Vec::new()
        } else {
            Self::desired_secondary_window_slots(
                self.engine.settings.coop_mode,
                self.engine.settings.coop_player_count,
            )
        };

        if desired.is_empty() {
            self.drop_secondary_windows();
            return;
        }

        let current_slots: Vec<Vec<usize>> = self
            .secondary_windows
            .iter()
            .map(|sw| sw.player_slots.clone())
            .collect();
        if current_slots == desired {
            return;
        }

        self.drop_secondary_windows();
        let Some(ref r) = self.renderer else {
            return;
        };

        let p1_mon_name = self
            .window
            .as_ref()
            .and_then(|w| w.current_monitor())
            .and_then(|m| m.name());
        let p1_mon_pos = self
            .window
            .as_ref()
            .and_then(|w| w.current_monitor())
            .map(|m| m.position());

        let other_monitors: Vec<_> = event_loop
            .available_monitors()
            .filter(|m| m.name() != p1_mon_name || Some(m.position()) != p1_mon_pos)
            .collect();

        for (win_idx, slots) in desired.iter().enumerate() {
            let names: Vec<String> = slots
                .iter()
                .map(|&s| {
                    format!(
                        "P{} ({})",
                        s + 1,
                        self.engine.settings.player_name_for_slot(s)
                    )
                })
                .collect();
            let title = format!("Cubex-RS — {}  [F11 Fullscreen]", names.join(" & "));
            let init_size = if other_monitors.get(win_idx).is_some() {
                LogicalSize::new(1280, 720)
            } else {
                LogicalSize::new(720, 480)
            };
            let mut attrs = Window::default_attributes()
                .with_title(title)
                .with_inner_size(init_size)
                .with_resizable(true)
                .with_fullscreen(None);
            if let Some(mon) = other_monitors.get(win_idx) {
                attrs = attrs.with_position(mon.position());
            } else {
                attrs = attrs.with_position(LogicalPosition::new(
                    60 + (win_idx as i32) * 90,
                    60 + (win_idx as i32) * 70,
                ));
            }

            match event_loop.create_window(attrs) {
                Ok(win_raw) => {
                    let win_arc = Arc::new(win_raw);
                    match r.create_secondary_target(win_arc.clone(), self.engine.settings.vsync) {
                        Ok(sec) => {
                            let ctx = egui::Context::default();
                            let es = egui_winit::State::new(
                                ctx.clone(),
                                egui::ViewportId::from_hash_of(format!("cubex_sec_win_{win_idx}")),
                                &*win_arc,
                                Some(win_arc.scale_factor() as f32),
                                None,
                                None,
                            );
                            let audio_slot_ref = self.engine.audio_slot.clone();
                            let audio_cfg_ref = self.engine.audio_settings_shared.clone();
                            let theme = McTheme::load(&ctx).with_click_callback(Arc::new(move || {
                                let s = audio_cfg_ref.read().clone();
                                audio_slot_ref.read().play_ui_click(&s);
                            }));
                            self.secondary_windows.push(SecondaryWindowState {
                                target: Some(sec),
                                egui_state: es,
                                theme,
                                egui_ctx: ctx,
                                window: win_arc,
                                fullscreen: false,
                                player_slots: slots.clone(),
                            });
                        }
                        Err(e) => {
                            tracing::warn!(
                                "Could not initialize secondary monitor surface ({e}); falling back to split-screen"
                            );
                            self.drop_secondary_windows();
                            self.engine.settings.coop_mode =
                                if self.engine.settings.coop_mode.max_players() > 2 {
                                    CoopMode::QuadSplit
                                } else {
                                    CoopMode::SplitVertical
                                };
                            self.engine.chat_messages.push(format!(
                                "§eMulti-Monitor unavailable ({e}), switched to Split-Screen."
                            ));
                            break;
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        "Could not open secondary player window ({e}); falling back to split-screen"
                    );
                    self.drop_secondary_windows();
                    self.engine.settings.coop_mode =
                        if self.engine.settings.coop_mode.max_players() > 2 {
                            CoopMode::QuadSplit
                        } else {
                            CoopMode::SplitVertical
                        };
                    break;
                }
            }
        }
        for peer in &mut self.engine.peers {
            peer.hud_icons = ItemIconCache::new();
            peer.form_icons = ItemIconCache::new();
        }
    }

    fn primary_touch_viewport(&self) -> egui::Rect {
        if let Some(vp) = self.last_touch_viewport {
            return vp;
        }
        let (w, h) = self
            .renderer
            .as_ref()
            .map(|r| (r.surface_config.width as f32, r.surface_config.height as f32))
            .unwrap_or((1280.0, 720.0));
        let scale = self
            .window
            .as_ref()
            .map(|w| w.scale_factor() as f32)
            .unwrap_or(1.0)
            .max(0.1);
        egui::Rect::from_min_size(egui::Pos2::ZERO, egui::Vec2::new(w / scale, h / scale))
    }

    fn open_inventory_formspec(&mut self) {
        self.set_cursor_locked(false);
        self.engine.move_input = MoveInput::default();
        self.engine.digger.reset();
        if self.engine.player.gamemode == GameMode::Creative {
            self.inv_cursor.clear();
            {
                let reg = self.engine.registries.read();
                self.p1_creative_state.ensure_synced(&reg);
            }
            self.engine.active_formspec = Some(ActiveFormspec {
                formname: CREATIVE_INVENTORY_FORMNAME.into(),
                raw_formspec: String::new(),
                parsed: ParsedFormspec::default(),
                context_pos: None,
            });
            return;
        }
        self.return_cursor_to_inventory();
        {
            let ps = self.engine.player_state.read();
            let reg = self.engine.registries.read();
            ensure_craft_grid_size(&ps.meta, 4);
            refresh_craft_preview(&ps.meta, &reg);
        }
        let raw_inv_fs = "formspec_version[4]size[11.75,9.6]label[2.25,0.35;Crafting]list[current_player;craft;2.25,0.75;2,2;0]image[5.0,1.35;1.25,1.0;gui_crafting_arrow.png]list[current_player;craftpreview;6.6,1.35;1,1;]label[0.375,3.4;Inventory]list[current_player;main;0.375,3.8;9,3;9]list[current_player;main;0.375,7.85;9,1;0]listring[current_player;main]listring[current_player;craft]button_exit[9.0,0.75;2.2,0.8;close;Done]".to_string();
        let parsed = parse_formspec(&raw_inv_fs);
        self.engine.active_formspec = Some(ActiveFormspec {
            formname: "cubex:inventory".into(),
            raw_formspec: raw_inv_fs,
            parsed,
            context_pos: None,
        });
    }
}

impl ApplicationHandler for WindowedApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        // Always start in resizable windowed mode (never exclusive fullscreen) so dual-monitor Wayland/Hyprland pointer focus works cleanly.
        let attrs = Window::default_attributes()
            .with_title("Cubex-RS — Vortex Studios")
            .with_inner_size(LogicalSize::new(1280, 720))
            .with_resizable(true)
            .with_fullscreen(None);
        let win = Arc::new(event_loop.create_window(attrs).expect("create window"));
        let mut renderer = WgpuRenderer::new_windowed(win.clone(), &self.engine.atlas)
            .expect("initialize wgpu renderer");
        renderer.set_vsync(self.engine.settings.vsync);
        renderer.set_anisotropy_quality(self.engine.settings.anisotropy_quality);
        let egui_state = egui_winit::State::new(
            self.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &*win,
            Some(win.scale_factor() as f32),
            None,
            None,
        );
        let audio_slot_ref = self.engine.audio_slot.clone();
        let audio_cfg_ref = self.engine.audio_settings_shared.clone();
        let theme = McTheme::load(&self.egui_ctx).with_click_callback(Arc::new(move || {
            let s = audio_cfg_ref.read().clone();
            audio_slot_ref.read().play_ui_click(&s);
        }));
        self.theme = Some(theme);
        self.egui_state = Some(egui_state);
        self.renderer = Some(renderer);
        self.engine
            .sync_skins_and_props(self.renderer.as_mut(), Some(&self.egui_ctx), false);
        win.request_redraw();
        self.window = Some(win);
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if self.is_gameplay_captured() {
            if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
                self.engine
                    .player
                    .apply_mouse_look(dx, dy, &self.engine.settings);
                if let Some(ref mut r) = self.renderer {
                    r.viewmodel.add_look_inertia(dx as f32, dy as f32);
                }
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        id: WindowId,
        event: WindowEvent,
    ) {
        if let Some(sec_idx) = self
            .secondary_windows
            .iter()
            .position(|sw| sw.window.id() == id)
        {
            {
                let sw = &mut self.secondary_windows[sec_idx];
                let _ = sw.egui_state.on_window_event(&sw.window, &event);
            }
            match event {
                WindowEvent::CloseRequested => {
                    self.drop_secondary_windows();
                    self.engine.settings.coop_mode = CoopMode::Off;
                    self.engine.ensure_coop_players_joined();
                    self.engine.apply_settings_to_lua_and_disk();
                }
                WindowEvent::Resized(new_sz) => {
                    if new_sz.width > 0 && new_sz.height > 0 {
                        if let Some(ref r) = self.renderer {
                            r.wait_idle();
                            if let Some(ref mut sec) = self.secondary_windows[sec_idx].target {
                                r.resize_secondary(
                                    sec,
                                    new_sz.width,
                                    new_sz.height,
                                    self.engine.settings.vsync,
                                );
                            }
                        }
                    }
                }
                WindowEvent::KeyboardInput { event, .. } => {
                    let pressed = event.state == ElementState::Pressed;
                    if let PhysicalKey::Code(code) = event.physical_key {
                        if code == KeyCode::F11 && pressed && !event.repeat {
                            let sw = &mut self.secondary_windows[sec_idx];
                            sw.fullscreen = !sw.fullscreen;
                            sw.window.set_fullscreen(if sw.fullscreen {
                                Some(Fullscreen::Borderless(sw.window.current_monitor()))
                            } else {
                                None
                            });
                        } else if self.engine.settings.coop_mode.is_active()
                            && (1..self.engine.settings.coop_mode.active_player_count(self.engine.settings.coop_player_count))
                                .any(|s| self.engine.settings.player_input_for_slot(s) == PlayerInputDevice::Keyboard2)
                        {
                            self.kb2.handle_key(code, pressed);
                        }
                    }
                }
                WindowEvent::RedrawRequested => {
                    self.render_secondary_window(sec_idx);
                }
                _ => {}
            }
            return;
        }

        // Touch controls on primary window (auto-enables touch HUD & synthetic egui pointer input)
        if let WindowEvent::Touch(touch) = event {
            if !self.engine.settings.touch_controls {
                self.engine.settings.touch_controls = true;
                self.set_cursor_locked(false);
            }
            let scale = self
                .window
                .as_ref()
                .map(|w| w.scale_factor())
                .unwrap_or(1.0)
                .max(0.1);
            let lpos = egui::Pos2::new(
                (touch.location.x / scale) as f32,
                (touch.location.y / scale) as f32,
            );
            let phase = match touch.phase {
                TouchPhase::Started => TouchEventPhase::Started,
                TouchPhase::Moved => TouchEventPhase::Moved,
                TouchPhase::Ended => TouchEventPhase::Ended,
                TouchPhase::Cancelled => TouchEventPhase::Cancelled,
            };
            if self.engine.mode == AppMode::Playing
                && !self.engine.player.is_dead
                && self.engine.active_formspec.is_none()
                && self.engine.chat_typing.is_none()
            {
                let vp = self.primary_touch_viewport();
                self.touch_controls.on_touch_event(touch.id, phase, lpos, vp);
                return;
            }
            self.pending_touch_egui_events
                .push(egui::Event::PointerMoved(lpos));
            match touch.phase {
                TouchPhase::Started => {
                    self.pending_touch_egui_events.push(egui::Event::PointerButton {
                        pos: lpos,
                        button: egui::PointerButton::Primary,
                        pressed: true,
                        modifiers: egui::Modifiers::NONE,
                    });
                }
                TouchPhase::Ended | TouchPhase::Cancelled => {
                    self.pending_touch_egui_events.push(egui::Event::PointerButton {
                        pos: lpos,
                        button: egui::PointerButton::Primary,
                        pressed: false,
                        modifiers: egui::Modifiers::NONE,
                    });
                    self.pending_touch_egui_events.push(egui::Event::PointerGone);
                }
                TouchPhase::Moved => {}
            }
            return;
        }

        // Single-touch / pointer fallback when Touch Controls toggle is ON during active gameplay
        if self.engine.settings.touch_controls
            && self.engine.mode == AppMode::Playing
            && !self.engine.player.is_dead
            && self.engine.active_formspec.is_none()
            && self.engine.chat_typing.is_none()
        {
            let scale = self
                .window
                .as_ref()
                .map(|w| w.scale_factor())
                .unwrap_or(1.0)
                .max(0.1);
            match event {
                WindowEvent::MouseInput {
                    button: MouseButton::Left,
                    state,
                    ..
                } => {
                    let down = state == ElementState::Pressed;
                    self.mouse_touch_down = down;
                    let vp = self.primary_touch_viewport();
                    let lpos = self
                        .egui_ctx
                        .input(|i| i.pointer.latest_pos())
                        .unwrap_or_else(|| vp.center());
                    let phase = if down {
                        TouchEventPhase::Started
                    } else {
                        TouchEventPhase::Ended
                    };
                    self.touch_controls
                        .on_touch_event(u64::MAX, phase, lpos, vp);
                    return;
                }
                WindowEvent::CursorMoved { position, .. } => {
                    let lpos = egui::Pos2::new(
                        (position.x / scale) as f32,
                        (position.y / scale) as f32,
                    );
                    if self.mouse_touch_down {
                        let vp = self.primary_touch_viewport();
                        self.touch_controls
                            .on_touch_event(u64::MAX, TouchEventPhase::Moved, lpos, vp);
                        return;
                    }
                }
                _ => {}
            }
        }

        // Strict Cursor Capture: route gameplay mouse clicks/motion directly BEFORE egui
        // so egui NEVER swallows world clicks while captured!
        let gameplay_captured = self.is_gameplay_captured();
        if gameplay_captured {
            match event {
                WindowEvent::MouseInput { button, state, .. } => {
                    let down = state == ElementState::Pressed;
                    if button == MouseButton::Left {
                        self.lmb_down = down;
                        if down {
                            self.lmb_just_pressed = true;
                        }
                    } else if button == MouseButton::Right && down {
                        self.rmb_just_pressed = true;
                    } else if button == MouseButton::Middle && down {
                        self.mmb_just_pressed = true;
                    }
                    return;
                }
                WindowEvent::CursorMoved { .. } => {
                    return;
                }
                _ => {}
            }
        }

        // If playing without active overlay and user clicks window, acquire pointer lock immediately
        if self.should_capture_cursor() && !self.cursor_locked {
            if let WindowEvent::MouseInput {
                state: ElementState::Pressed,
                ..
            } = event
            {
                self.set_cursor_locked(true);
                return;
            }
        }

        if let (Some(ref win), Some(ref mut es)) = (&self.window, &mut self.egui_state) {
            let _ = es.on_window_event(win, &event);
        }

        match event {
            WindowEvent::CloseRequested => {
                self.engine.save_active_world();
                if let Some(ref s) = self.engine.net_session {
                    s.send(NetPacket::PlayerLeave {
                        player_name: self.engine.settings.player_name.clone(),
                    });
                }
                self.engine.net_session = None;
                self.engine.stop_lan_server();
                self.engine.tunnel.stop();
                self.shutdown_gpu_resources();
                event_loop.exit();
            }
            WindowEvent::Focused(focused) => {
                self.on_focus_changed(focused);
            }
            WindowEvent::Resized(new_sz) => {
                if new_sz.width > 0 && new_sz.height > 0 {
                    if let Some(ref mut r) = self.renderer {
                        r.wait_idle();
                        r.resize(new_sz.width, new_sz.height, self.engine.settings.vsync);
                    }
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state,
                        text,
                        ..
                    },
                ..
            } => {
                let pressed = state == ElementState::Pressed;
                if pressed && code == KeyCode::F11 {
                    self.fullscreen = !self.fullscreen;
                    if let Some(ref win) = self.window {
                        win.set_fullscreen(if self.fullscreen {
                            Some(Fullscreen::Borderless(None))
                        } else {
                            None
                        });
                    }
                    return;
                }
                if pressed
                    && code == KeyCode::F3
                    && matches!(self.engine.mode, AppMode::Playing | AppMode::Paused)
                {
                    self.engine.settings.f3_enabled = !self.engine.settings.f3_enabled;
                    self.engine.apply_settings_to_lua_and_disk();
                    return;
                }
                if pressed
                    && code == KeyCode::F5
                    && matches!(self.engine.mode, AppMode::Playing | AppMode::Paused)
                {
                    self.camera_mode = self.camera_mode.cycle();
                    return;
                }
                if pressed
                    && code == KeyCode::F8
                    && matches!(self.engine.mode, AppMode::Playing | AppMode::Paused)
                {
                    self.engine.settings.coop_mode = self.engine.settings.coop_mode.cycle();
                    if self.engine.settings.coop_mode.is_active() {
                        self.engine.ensure_coop_players_joined();
                    }
                    self.engine.apply_settings_to_lua_and_disk();
                    self.engine.chat_messages.push(format!(
                        "§aLocal Co-op: {} ({}P)",
                        self.engine.settings.coop_mode.label(),
                        self.engine
                            .settings
                            .coop_mode
                            .active_player_count(self.engine.settings.coop_player_count)
                    ));
                    return;
                }
                if self.engine.mode == AppMode::Playing && self.engine.player.is_dead {
                    self.engine.move_input = MoveInput::default();
                    return;
                }
                if pressed && code == KeyCode::Escape {
                    if self.autotest_bot.is_some()
                        || self.pending_autotest_config.is_some()
                        || self.qa_bot.is_some()
                        || self.pending_qa_playthrough.is_some()
                    {
                        let was_qa = self.qa_bot.is_some() || self.pending_qa_playthrough.is_some();
                        self.autotest_bot = None;
                        self.pending_autotest_config = None;
                        self.pending_autotest_finish = None;
                        self.qa_bot = None;
                        self.pending_qa_playthrough = None;
                        self.pending_qa_finish = None;
                        self.engine.move_input = MoveInput::default();
                        self.engine.digger.reset();
                        self.lmb_down = false;
                        self.lmb_just_pressed = false;
                        self.rmb_just_pressed = false;
                        self.engine.chat_messages.push(if was_qa {
                            "§e[QA BOT] Cancelled by user (manual control restored).".into()
                        } else {
                            "§e[AUTOBOT] Cancelled by user (manual control restored).".into()
                        });
                        if self.engine.mode == AppMode::Playing {
                            self.set_cursor_locked(true);
                        }
                        return;
                    }
                    if self.engine.active_formspec.is_some() {
                        self.close_active_formspec();
                    } else if self.engine.chat_typing.is_some() {
                        self.engine.chat_typing = None;
                        self.set_cursor_locked(true);
                    } else if self.engine.mode == AppMode::Playing {
                        self.engine.mode = AppMode::Paused;
                        self.engine.pause_state.page = PausePage::Main;
                        self.engine.move_input = MoveInput::default();
                        self.engine.digger.reset();
                        self.set_cursor_locked(false);
                    } else if self.engine.mode == AppMode::Paused
                        && self.engine.pause_state.listening_keybind.is_none()
                    {
                        self.engine.mode = AppMode::Playing;
                        self.set_cursor_locked(true);
                    }
                    return;
                }

                // Active chat typing handler
                if self.engine.mode == AppMode::Playing {
                    if let Some(ref mut chat_buf) = self.engine.chat_typing {
                        if pressed {
                            if code == KeyCode::Enter || code == KeyCode::NumpadEnter {
                                let msg = self.engine.chat_typing.take().unwrap_or_default();
                                self.engine.handle_chat_submission(msg);
                                self.set_cursor_locked(true);
                            } else if code == KeyCode::Backspace {
                                chat_buf.pop();
                            } else if let Some(txt) = text {
                                for ch in txt.chars() {
                                    if !ch.is_control() {
                                        chat_buf.push(ch);
                                    }
                                }
                            }
                        }
                        return;
                    }
                }

                // Formspec overlay open: allow E to close player inventory cleanly (unless typing in search)
                if self.engine.mode == AppMode::Playing && self.engine.active_formspec.is_some() {
                    let inv_key = self
                        .engine
                        .settings
                        .keybinds
                        .get("keymap_inventory")
                        .map(String::as_str);
                    if pressed
                        && keycode_matches(code, inv_key, KeyCode::KeyE)
                        && !self.egui_ctx.wants_keyboard_input()
                        && self
                            .engine
                            .active_formspec
                            .as_ref()
                            .map(|f| is_player_inventory_formname(&f.formname))
                            .unwrap_or(false)
                    {
                        self.close_active_formspec();
                    }
                    return;
                }

                if self.engine.mode == AppMode::Playing
                    && self.engine.active_formspec.is_none()
                    && self.engine.chat_typing.is_none()
                {
                    let kb = &self.engine.settings.keybinds;
                    if pressed && keycode_matches(code, kb.get("keymap_inventory").map(String::as_str), KeyCode::KeyE) {
                        self.open_inventory_formspec();
                        return;
                    }
                    if pressed && keycode_matches(code, kb.get("keymap_chat").map(String::as_str), KeyCode::KeyT) {
                        self.engine.chat_typing = Some(String::new());
                        self.engine.move_input = MoveInput::default();
                        self.engine.digger.reset();
                        self.set_cursor_locked(false);
                        return;
                    }
                    if pressed && code == KeyCode::Slash {
                        self.engine.chat_typing = Some("/".into());
                        self.engine.move_input = MoveInput::default();
                        self.engine.digger.reset();
                        self.set_cursor_locked(false);
                        return;
                    }
                    if pressed && keycode_matches(code, kb.get("keymap_freemove").map(String::as_str), KeyCode::KeyK) {
                        if matches!(self.engine.player.gamemode, GameMode::Creative | GameMode::Spectator) {
                            self.engine.player.flying = !self.engine.player.flying;
                            self.engine.player.vel.y = 0.0;
                        }
                    }
                    if pressed && keycode_matches(code, kb.get("keymap_drop").map(String::as_str), KeyCode::KeyQ) {
                        drop_held_slot_item(
                            &self.engine.player,
                            &self.engine.player_state,
                            self.engine.lua_rt.as_ref(),
                        );
                    }

                    if pressed {
                        let slot = match code {
                            KeyCode::Digit1 => Some(0),
                            KeyCode::Digit2 => Some(1),
                            KeyCode::Digit3 => Some(2),
                            KeyCode::Digit4 => Some(3),
                            KeyCode::Digit5 => Some(4),
                            KeyCode::Digit6 => Some(5),
                            KeyCode::Digit7 => Some(6),
                            KeyCode::Digit8 => Some(7),
                            KeyCode::Digit9 => Some(8),
                            _ => None,
                        };
                        if let Some(s) = slot {
                            self.engine.player.selected_slot = s;
                            self.engine.player_state.write().wield_index = s + 1;
                        }
                    }

                    if keycode_matches(code, kb.get("keymap_forward").map(String::as_str), KeyCode::KeyW) {
                        self.engine.move_input.forward = pressed;
                    } else if keycode_matches(code, kb.get("keymap_backward").map(String::as_str), KeyCode::KeyS) {
                        self.engine.move_input.backward = pressed;
                    } else if keycode_matches(code, kb.get("keymap_left").map(String::as_str), KeyCode::KeyA) {
                        self.engine.move_input.left = pressed;
                    } else if keycode_matches(code, kb.get("keymap_right").map(String::as_str), KeyCode::KeyD) {
                        self.engine.move_input.right = pressed;
                    } else if keycode_matches(code, kb.get("keymap_jump").map(String::as_str), KeyCode::Space) {
                        self.engine.move_input.jump = pressed;
                    } else if keycode_matches(code, kb.get("keymap_sneak").map(String::as_str), KeyCode::ShiftLeft) {
                        self.engine.move_input.sneak = pressed;
                    } else if keycode_matches(code, kb.get("keymap_aux1").map(String::as_str), KeyCode::ControlLeft) {
                        self.engine.move_input.sprint = pressed;
                    } else if keycode_matches(code, kb.get("keymap_zoom").map(String::as_str), KeyCode::KeyC) {
                        self.engine.move_input.zoom = pressed;
                    }

                    // Optional Keyboard 2 bindings when any Co-op peer uses Keyboard2
                    if self.engine.settings.coop_mode.is_active()
                        && (1..self
                            .engine
                            .settings
                            .coop_mode
                            .active_player_count(self.engine.settings.coop_player_count))
                            .any(|s| {
                                self.engine.settings.player_input_for_slot(s)
                                    == PlayerInputDevice::Keyboard2
                            })
                    {
                        self.kb2.handle_key(code, pressed);
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if self.engine.mode == AppMode::Playing
                    && self.engine.active_formspec.is_none()
                    && self.engine.chat_typing.is_none()
                {
                    let dy = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y,
                        MouseScrollDelta::PixelDelta(p) => p.y as f32 / 24.0,
                    };
                    if dy > 0.0 {
                        self.engine.player.selected_slot =
                            (self.engine.player.selected_slot + 8) % 9;
                    } else if dy < 0.0 {
                        self.engine.player.selected_slot =
                            (self.engine.player.selected_slot + 1) % 9;
                    }
                    self.engine.player_state.write().wield_index =
                        self.engine.player.selected_slot + 1;
                }
            }
            WindowEvent::RedrawRequested => {
                self.render_tick(event_loop);
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(ref win) = self.window {
            win.request_redraw();
        }
        for sw in &self.secondary_windows {
            sw.window.request_redraw();
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        self.engine.save_active_world();
        self.engine.stop_lan_server();
        self.engine.tunnel.stop();
        self.shutdown_gpu_resources();
    }
}

impl WindowedApp {
    fn begin_world_load(&mut self, info: WorldInfo) {
        self.set_cursor_locked(false);
        self.engine.player.is_dead = false;
        self.engine.player.has_dropped_death_items = false;
        self.engine.player.spawn_invuln_timer = 0.0;
        self.engine.player.suffocation_timer = 0.0;
        self.engine.player.death_cause.clear();
        {
            let mut ps = self.engine.player_state.write();
            ps.is_dead = false;
            ps.has_dropped_death_items = false;
            ps.spawn_invuln_timer = 0.0;
            ps.last_death_cause.clear();
        }
        for peer in &mut self.engine.peers {
            peer.player.is_dead = false;
            peer.player.has_dropped_death_items = false;
            peer.player.spawn_invuln_timer = 0.0;
            peer.player.suffocation_timer = 0.0;
            peer.player.death_cause.clear();
            let mut pps = peer.player_state.write();
            pps.is_dead = false;
            pps.has_dropped_death_items = false;
            pps.spawn_invuln_timer = 0.0;
            pps.last_death_cause.clear();
        }
        self.engine.active_formspec = None;
        self.engine.chat_typing = None;
        self.engine.move_input = MoveInput::default();
        self.engine.digger.reset();
        self.inv_cursor.clear();
        self.engine.pending_load = Some(WorldLoadPhase::Preparing {
            info,
            radius: 4,
            painted: false,
        });
        self.engine.mode = AppMode::Loading;
    }

    fn loading_screen_info(&self) -> LoadingScreenInfo {
        match self.engine.pending_load.as_ref() {
            Some(WorldLoadPhase::Preparing { info, radius, .. }) => {
                let side = (*radius * 2 + 1).max(1) as usize;
                LoadingScreenInfo {
                    world_name: info.name.clone(),
                    stage: "Preparing world...".to_string(),
                    detail: "Loading scripts & world data".to_string(),
                    progress: 0.05,
                    grid_radius: *radius,
                    ready_columns: vec![false; side * side],
                }
            }
            Some(WorldLoadPhase::GeneratingTerrain {
                world_name,
                radius,
                total_cols,
                done_cols,
                ready_columns,
                ..
            }) => LoadingScreenInfo {
                world_name: world_name.clone(),
                stage: "Generating terrain chunks...".to_string(),
                detail: format!("{done_cols} / {total_cols} columns"),
                progress: 0.10 + 0.45 * (*done_cols as f32 / (*total_cols).max(1) as f32),
                grid_radius: *radius,
                ready_columns: ready_columns.clone(),
            },
            Some(WorldLoadPhase::BuildingMeshes {
                world_name,
                radius,
                total_chunks,
                uploaded_chunks,
                ready_columns,
                ..
            }) => LoadingScreenInfo {
                world_name: world_name.clone(),
                stage: "Building terrain meshes...".to_string(),
                detail: format!("{uploaded_chunks} / {total_chunks} chunks"),
                progress: 0.55 + 0.45 * (*uploaded_chunks as f32 / (*total_chunks).max(1) as f32),
                grid_radius: *radius,
                ready_columns: ready_columns.clone(),
            },
            None => LoadingScreenInfo::default(),
        }
    }

    fn step_world_load(&mut self) {
        let Some(phase) = self.engine.pending_load.take() else {
            return;
        };

        match phase {
            WorldLoadPhase::Preparing {
                info,
                radius,
                painted,
            } => {
                if !painted {
                    self.engine.pending_load = Some(WorldLoadPhase::Preparing {
                        info,
                        radius,
                        painted: true,
                    });
                    return;
                }
                let Some(ref mut r) = self.renderer else {
                    return;
                };
                let had_saved_player = self.engine.prepare_world_session(&info, r);
                let center_cx = (self.engine.player.pos.x / 16.0).floor() as i32;
                let center_cz = (self.engine.player.pos.z / 16.0).floor() as i32;
                let wt = *self.engine.world.world_type.read();
                let y_range = chunk_y_range_for(wt);
                let side = (radius * 2 + 1).max(1) as usize;
                let total_cols = side * side;
                let mut ready_columns = vec![false; total_cols];
                let mut col_index_map = HashMap::with_capacity(total_cols);
                let mut ordered_cols = Vec::with_capacity(total_cols);

                for dz in -radius..=radius {
                    for dx in -radius..=radius {
                        let col = (center_cx + dx, center_cz + dz);
                        let col_idx =
                            ((dz + radius) as usize) * side + ((dx + radius) as usize);
                        col_index_map.insert(col, col_idx);
                        ordered_cols.push((dx * dx + dz * dz, col, col_idx));
                    }
                }
                ordered_cols.sort_by_key(|(d2, _, _)| *d2);

                let existing_chunks: HashSet<IVec3> =
                    self.engine.world.chunks.read().keys().copied().collect();
                let mut missing_cols: Vec<(i32, i32)> = Vec::new();
                let mut done_cols = 0usize;
                let mut target_chunks = Vec::with_capacity(total_cols * y_range.clone().count());

                for (_, col @ (cx, cz), col_idx) in ordered_cols {
                    self.engine.generated_columns.insert(col);
                    let mut all_present = true;
                    for cy in y_range.clone() {
                        let cpos = IVec3::new(cx, cy, cz);
                        target_chunks.push(cpos);
                        if !existing_chunks.contains(&cpos) {
                            all_present = false;
                        }
                    }
                    if all_present {
                        ready_columns[col_idx] = true;
                        done_cols += 1;
                    } else {
                        missing_cols.push(col);
                    }
                }

                let (tx, rx) = mpsc::channel();
                if !missing_cols.is_empty() {
                    let seed = self.engine.world.seed;
                    let palette = self.engine.palette.clone();
                    let reg_arc = self.engine.registries.clone();
                    let epoch = self.engine.world_epoch;
                    let epoch_shared = self.engine.world_epoch_shared.clone();
                    rayon::spawn(move || {
                        let reg = reg_arc.read();
                        missing_cols.into_par_iter().for_each_with(tx, |tx, col| {
                            if epoch_shared.load(Ordering::Acquire) != epoch {
                                return;
                            }
                            let mut col_chunks = Vec::new();
                            for cy in y_range.clone() {
                                let cpos = IVec3::new(col.0, cy, col.1);
                                if !existing_chunks.contains(&cpos) {
                                    col_chunks.push(generate_chunk_for_world(
                                        cpos, seed, wt, &palette, &reg,
                                    ));
                                }
                            }
                            let _ = tx.send((col, col_chunks));
                        });
                    });
                }

                self.engine.pending_load = Some(WorldLoadPhase::GeneratingTerrain {
                    world_name: info.name,
                    radius,
                    total_cols,
                    done_cols,
                    ready_columns,
                    target_chunks,
                    col_index_map,
                    rx,
                    had_saved_player,
                });
            }

            WorldLoadPhase::GeneratingTerrain {
                world_name,
                radius,
                total_cols,
                mut done_cols,
                mut ready_columns,
                target_chunks,
                col_index_map,
                rx,
                had_saved_player,
            } => {
                let mut disconnected = false;
                let mut drained = 0usize;
                while drained < 64 {
                    match rx.try_recv() {
                        Ok((col, chunks)) => {
                            drained += 1;
                            let seed = self.engine.world.seed;
                            let wt = *self.engine.world.world_type.read();
                            let reg = self.engine.registries.read();
                            for ch in chunks {
                                let cpos = ch.pos;
                                self.engine.world.insert_chunk(ch);
                                populate_structure_chests_for_chunk(
                                    &self.engine.world,
                                    cpos,
                                    seed,
                                    wt,
                                    &reg,
                                );
                            }
                            if let Some(&col_idx) = col_index_map.get(&col) {
                                if !ready_columns[col_idx] {
                                    ready_columns[col_idx] = true;
                                    done_cols += 1;
                                }
                            }
                        }
                        Err(mpsc::TryRecvError::Empty) => break,
                        Err(mpsc::TryRecvError::Disconnected) => {
                            disconnected = true;
                            break;
                        }
                    }
                }

                if done_cols < total_cols && !disconnected {
                    self.engine.pending_load = Some(WorldLoadPhase::GeneratingTerrain {
                        world_name,
                        radius,
                        total_cols,
                        done_cols,
                        ready_columns,
                        target_chunks,
                        col_index_map,
                        rx,
                        had_saved_player,
                    });
                    return;
                }

                // All initial terrain columns are inserted; finalize player position on safe surface
                self.engine.finalize_loaded_player_spawn(had_saved_player);
                self.engine
                    .sync_skins_and_props(self.renderer.as_mut(), Some(&self.egui_ctx), false);

                let side = (radius * 2 + 1).max(1) as usize;
                let present_coords: Vec<IVec3> = {
                    let chunks_g = self.engine.world.chunks.read();
                    target_chunks
                        .into_iter()
                        .filter(|cpos| chunks_g.contains_key(cpos))
                        .collect()
                };
                {
                    let mut dq = self.engine.world.dirty_queue.write();
                    for cpos in &present_coords {
                        dq.remove(cpos);
                    }
                }

                let mut pending_set = HashSet::with_capacity(present_coords.len());
                let mut chunk_col_index = HashMap::with_capacity(present_coords.len());
                let mut col_remaining_chunks = vec![0usize; side * side];
                let mut mesh_ready_cols = vec![false; side * side];

                for &cpos in &present_coords {
                    pending_set.insert(cpos);
                    if let Some(&col_idx) = col_index_map.get(&(cpos.x, cpos.z)) {
                        chunk_col_index.insert(cpos, col_idx);
                        col_remaining_chunks[col_idx] += 1;
                    }
                }
                for (i, rem) in col_remaining_chunks.iter().enumerate() {
                    if *rem == 0 {
                        mesh_ready_cols[i] = true;
                    }
                }

                let total_chunks = pending_set.len();
                if total_chunks == 0 {
                    self.engine.pending_load = None;
                    self.engine.mode = AppMode::Playing;
                    self.set_cursor_locked(true);
                    return;
                }

                let world_arc = self.engine.world.clone();
                let reg_arc = self.engine.registries.clone();
                let atlas_arc = self.engine.atlas.clone();
                let smooth = self.engine.settings.smooth_lighting;
                let epoch = self.engine.world_epoch;
                let epoch_shared = self.engine.world_epoch_shared.clone();
                let mesh_tx = self.engine.mesh_tx.clone();

                rayon::spawn(move || {
                    if epoch_shared.load(Ordering::Acquire) != epoch {
                        return;
                    }
                    let reg = reg_arc.read();
                    propagate_world_light(&world_arc, &present_coords, &reg);
                    {
                        let mut dq = world_arc.dirty_queue.write();
                        for cpos in &present_coords {
                            dq.remove(cpos);
                        }
                    }
                    let snaps: Vec<_> = present_coords
                        .iter()
                        .filter_map(|&cpos| world_arc.snapshot_chunk(cpos))
                        .collect();
                    snaps.into_par_iter().for_each_with(mesh_tx, |tx, snap| {
                        if epoch_shared.load(Ordering::Acquire) != epoch {
                            return;
                        }
                        let mesh = mesh_chunk_with_atlas(&snap, &reg, &atlas_arc, smooth);
                        let _ = tx.send(MeshedChunkPacket {
                            epoch,
                            dimension: DimensionId::Overworld,
                            mesh,
                        });
                    });
                });

                self.engine.pending_load = Some(WorldLoadPhase::BuildingMeshes {
                    world_name,
                    radius,
                    total_chunks,
                    uploaded_chunks: 0,
                    pending_set,
                    ready_columns: mesh_ready_cols,
                    chunk_col_index,
                    col_remaining_chunks,
                });
            }

            WorldLoadPhase::BuildingMeshes {
                world_name,
                radius,
                total_chunks,
                mut uploaded_chunks,
                mut pending_set,
                mut ready_columns,
                chunk_col_index,
                mut col_remaining_chunks,
            } => {
                while let Ok(pkt) = self.engine.mesh_rx.try_recv() {
                    if pkt.epoch == self.engine.world_epoch {
                        self.engine.pending_meshes.push_back(pkt.mesh);
                    }
                }

                if let Some(ref mut r) = self.renderer {
                    let t0 = Instant::now();
                    let mut count_this_tick = 0usize;
                    while !self.engine.pending_meshes.is_empty() {
                        if count_this_tick > 0 && t0.elapsed().as_secs_f32() > 0.008 {
                            break;
                        }
                        let Some(mesh) = self.engine.pending_meshes.pop_front() else {
                            break;
                        };
                        let cpos = mesh.chunk_pos;
                        r.upload_chunk_mesh_now(mesh);
                        count_this_tick += 1;
                        if pending_set.remove(&cpos) {
                            uploaded_chunks += 1;
                            if let Some(&col_idx) = chunk_col_index.get(&cpos) {
                                col_remaining_chunks[col_idx] =
                                    col_remaining_chunks[col_idx].saturating_sub(1);
                                if col_remaining_chunks[col_idx] == 0 {
                                    ready_columns[col_idx] = true;
                                }
                            }
                        }
                    }
                }

                if pending_set.is_empty() {
                    self.engine.pending_load = None;
                    self.engine.mode = AppMode::Playing;
                    self.set_cursor_locked(true);
                } else {
                    self.engine.pending_load = Some(WorldLoadPhase::BuildingMeshes {
                        world_name,
                        radius,
                        total_chunks,
                        uploaded_chunks,
                        pending_set,
                        ready_columns,
                        chunk_col_index,
                        col_remaining_chunks,
                    });
                }
            }
        }
    }

    fn render_secondary_window(&mut self, sec_idx: usize) {
        if self.engine.mode == AppMode::Title || sec_idx >= self.secondary_windows.len() {
            return;
        }
        let tod = self
            .engine
            .lua_rt
            .as_ref()
            .map(|rt| *rt.shared.time_of_day.read())
            .unwrap_or(0.35);

        let ppp = self.secondary_windows[sec_idx].window.scale_factor() as f32;
        let (surf_w, surf_h) = match self.secondary_windows[sec_idx].target.as_ref() {
            Some(sec) => (
                sec.surface_config.width.max(1),
                sec.surface_config.height.max(1),
            ),
            None => return,
        };
        let assigned_slots = self.secondary_windows[sec_idx].player_slots.clone();
        let active_slots: Vec<usize> = assigned_slots
            .into_iter()
            .filter(|&s| s >= 1 && s <= 3 && self.engine.peers[s - 1].joined)
            .collect();
        if active_slots.is_empty() {
            return;
        }

        let ctrl_inputs: Vec<Option<ControllerFormInput>> = active_slots
            .iter()
            .map(|&s| self.controller_form_input_for_player(s))
            .collect();
        let raw_input = {
            let sw = &mut self.secondary_windows[sec_idx];
            sw.egui_state.take_egui_input(&sw.window)
        };
        let (visible_chunks, draw_calls, adapter_name) = match self.renderer.as_ref() {
            Some(r) => (
                r.last_visible_chunks,
                r.last_draw_calls,
                r.adapter_name.clone(),
            ),
            None => return,
        };

        let mut peer_form_submits: Vec<(usize, String, Vec<(String, String)>, bool)> = Vec::new();
        let mut peer_death_respawns: Vec<usize> = Vec::new();
        let mut built_vp_px: Vec<(usize, [u32; 4])> = Vec::new();

        let (output_sec, nametag_targets): (egui::FullOutput, Vec<(String, Vec3, DimensionId)>) = {
            let mut tags: Vec<(String, Vec3, DimensionId)> = Vec::new();
            tags.push((
                self.engine.settings.player_name.clone(),
                self.engine.player.pos,
                self.engine.player.dimension,
            ));
            for (i, p) in self.engine.peers.iter().enumerate() {
                if p.joined {
                    tags.push((
                        self.engine.settings.player_name_for_slot(i + 1).to_string(),
                        p.player.pos,
                        p.player.dimension,
                    ));
                }
            }
            for rp in self.engine.remote_players.values() {
                tags.push((rp.name.clone(), rp.pos, DimensionId::Overworld));
            }
            (
                {
                    let sw = &self.secondary_windows[sec_idx];
                    let ctx_clone = sw.egui_ctx.clone();
                    let theme_ref = &sw.theme;
                    let win_mode = self.engine.mode;
                    let vp_layouts = Self::compute_window_viewports(
                        self.engine.settings.coop_mode,
                        self.engine.settings.coop_player_count,
                        surf_w,
                        surf_h,
                        ctx_clone.screen_rect(),
                        sec_idx + 1,
                    );
                    let reg = self.engine.registries.read();
                    ctx_clone.run(raw_input, |ctx| {
                        let screen = ctx.screen_rect();
                        for (pos_idx, &slot) in active_slots.iter().enumerate() {
                            let peer_idx = slot - 1;
                            let (px, log_rect) = vp_layouts
                                .iter()
                                .find(|(s, _, _)| *s == slot)
                                .map(|&(_, px, r)| (px, r))
                                .unwrap_or_else(|| {
                                    if active_slots.len() == 1 {
                                        ([0, 0, surf_w, surf_h], screen)
                                    } else if pos_idx == 0 {
                                        let lw = (surf_w / 2).max(1);
                                        let mx = screen.center().x;
                                        (
                                            [0, 0, lw, surf_h],
                                            egui::Rect::from_min_max(
                                                screen.min,
                                                egui::Pos2::new(mx, screen.bottom()),
                                            ),
                                        )
                                    } else {
                                        let lw = (surf_w / 2).max(1);
                                        let rw = (surf_w - lw).max(1);
                                        let mx = screen.center().x;
                                        (
                                            [lw, 0, rw, surf_h],
                                            egui::Rect::from_min_max(
                                                egui::Pos2::new(mx, screen.top()),
                                                screen.max,
                                            ),
                                        )
                                    }
                                });
                            built_vp_px.push((slot, px));

                            let (xp_level, xp_progress, hurt_flash, my_name) = {
                                let ps = self.engine.peers[peer_idx].player_state.read();
                                (ps.xp_level, ps.xp_progress, ps.hurt_flash, ps.name.clone())
                            };
                            let peer_cam_pos = self.engine.peers[peer_idx].cam_pos;
                            let peer_vp = self.engine.peers[peer_idx].view_proj;
                            let peer_dim = self.engine.peers[peer_idx].player.dimension;
                            let peer_tags: Vec<(String, Vec3)> = tags
                                .iter()
                                .filter(|(n, _, d)| *n != my_name && *d == peer_dim)
                                .map(|(n, p, _)| (n.clone(), *p))
                                .collect();
                            render_nametags_in_viewport(
                                ctx,
                                log_rect,
                                peer_cam_pos,
                                peer_vp,
                                &peer_tags,
                            );

                            let peer = &mut self.engine.peers[peer_idx];
                            let peer_world = world_for(
                                &self.engine.world,
                                &self.engine.nether_world,
                                peer.player.dimension,
                            );
                            let peer_chunks_loaded = peer_world.chunks.read().len();
                            let peer_warp_alpha = (peer.player.portal_charge * 0.72)
                                .max(peer.player.portal_warp_flash * 0.85);
                            let badge = match slot {
                                1 => "P2",
                                2 => "P3",
                                _ => "P4",
                            };
                            let ctrl_in = ctrl_inputs.get(pos_idx).copied().flatten();
                            if peer.player.is_dead {
                                let act = render_death_screen(
                                    ctx,
                                    theme_ref,
                                    log_rect,
                                    (slot as u64 + 1) * 100 + 77,
                                    &my_name,
                                    &peer.player.death_cause,
                                    xp_level,
                                    false,
                                    ctrl_in.is_some_and(|c| c.confirm),
                                );
                                if act == DeathScreenAction::Respawn {
                                    peer_death_respawns.push(slot);
                                }
                            } else {
                                render_hud_in_rect(
                                    ctx,
                                    log_rect,
                                    &peer.player,
                                    &peer.player_state,
                                    &self.engine.settings,
                                    &reg,
                                    peer.targeted.as_ref(),
                                    self.engine.fps,
                                    self.engine.frame_ms,
                                    peer_chunks_loaded,
                                    visible_chunks,
                                    draw_calls,
                                    &adapter_name,
                                    &self.engine.chat_messages,
                                    None,
                                    HudTelemetry {
                                        atlas: Some(&self.engine.atlas),
                                        icon_cache: Some(&mut peer.hud_icons),
                                        world: Some(peer_world),
                                        dig_progress: peer.digger.dig_progress,
                                        biome_name: Some(biome_label_for(
                                            peer.player.dimension,
                                            peer.player.pos,
                                            peer_world.seed,
                                        )),
                                        targeted_param2: peer.targeted.as_ref().map(|h| h.param2),
                                        armor_points: 0,
                                        xp_level,
                                        xp_progress,
                                        player_badge: Some(badge),
                                        hurt_flash,
                                        dimension: peer.player.dimension,
                                        portal_warp_alpha: peer_warp_alpha,
                                    },
                                );
                                if let Some(ref af) = peer.active_formspec {
                                    if af.formname == CREATIVE_INVENTORY_FORMNAME {
                                        let close = render_creative_inventory(
                                            ctx,
                                            theme_ref,
                                            &mut peer.creative_state,
                                            &peer.player_state,
                                            peer.player.selected_slot,
                                            &reg,
                                            &self.engine.atlas,
                                            &mut peer.form_icons,
                                            &mut peer.inv_cursor,
                                            log_rect,
                                            (slot as u64 + 1) * 100 + 1,
                                            ctrl_in,
                                        );
                                        if close {
                                            peer_form_submits.push((
                                                slot,
                                                af.formname.clone(),
                                                Vec::new(),
                                                true,
                                            ));
                                        }
                                    } else {
                                        let (fields, close) = render_formspec_interactive(
                                            ctx,
                                            theme_ref,
                                            &af.formname,
                                            &af.parsed,
                                            &peer.player_state,
                                            &mut peer.formspec_fields,
                                            FormspecContext {
                                                world: Some(&self.engine.world),
                                                context_pos: af.context_pos,
                                                registries: Some(&reg),
                                                atlas: Some(&self.engine.atlas),
                                                icon_cache: Some(&mut peer.form_icons),
                                                cursor: Some(&mut peer.inv_cursor),
                                                lua_rt: self.engine.lua_rt.as_ref(),
                                                viewport: Some(log_rect),
                                                id_salt: (slot as u64 + 1) * 100 + 1,
                                                controller_input: ctrl_in,
                                                controller_slot: ctrl_in
                                                    .map(|_| &mut peer.form_slot_idx),
                                            },
                                        );
                                        if !fields.is_empty() || close {
                                            peer_form_submits.push((
                                                slot,
                                                af.formname.clone(),
                                                fields,
                                                close,
                                            ));
                                        }
                                    }
                                }
                            }
                        }

                        if built_vp_px.len() == 2 {
                            let div_p = ctx.layer_painter(egui::LayerId::new(
                                egui::Order::Foreground,
                                egui::Id::new(format!("cubex_sec_div_{sec_idx}")),
                            ));
                            let mx = screen.center().x;
                            div_p.line_segment(
                                [
                                    egui::Pos2::new(mx, screen.top()),
                                    egui::Pos2::new(mx, screen.bottom()),
                                ],
                                egui::Stroke::new(2.0, egui::Color32::from_rgb(20, 20, 24)),
                            );
                        }

                        if win_mode == AppMode::Paused {
                            let p = ctx.layer_painter(egui::LayerId::new(
                                egui::Order::Foreground,
                                egui::Id::new(format!("cubex_sec_pause_{sec_idx}")),
                            ));
                            p.rect_filled(
                                screen,
                                0.0,
                                egui::Color32::from_rgba_unmultiplied(0, 0, 0, 120),
                            );
                            cubex_rs::ui::theme::mc_shadow_text(
                                &p,
                                screen.center(),
                                egui::Align2::CENTER_CENTER,
                                "Game Paused (Window 1)",
                                22.0,
                                egui::Color32::WHITE,
                            );
                        }
                    })
                },
                tags,
            )
        };
        let _ = nametag_targets;

        {
            let sw = &mut self.secondary_windows[sec_idx];
            sw.egui_state
                .handle_platform_output(&sw.window, output_sec.platform_output.clone());
        }

        let ow_mask = self.active_local_body_mask_in_dimension(DimensionId::Overworld);
        let nw_mask = self.active_local_body_mask_in_dimension(DimensionId::Nether);
        let Some(ref mut r) = self.renderer else {
            return;
        };
        let cams: Vec<ViewportCamera> = built_vp_px
            .iter()
            .map(|&(slot, px)| {
                let peer_idx = slot - 1;
                let peer = &self.engine.peers[peer_idx];
                let peer_dim = peer.player.dimension;
                let peer_show_self = r
                    .peer_slot(slot)
                    .map(|p| p.show_player_model)
                    .unwrap_or(false);
                let dim_mask = if peer_dim.is_nether() { nw_mask } else { ow_mask };
                let mut mask = dim_mask & !(1u8 << slot);
                if peer_show_self {
                    mask |= 1u8 << slot;
                }
                ViewportCamera {
                    cam_pos: peer.cam_pos,
                    view_proj: peer.view_proj,
                    viewport_px: px,
                    player_slot: slot,
                    draw_local_player_mask: mask,
                    draw_viewmodel: peer.show_viewmodel,
                    dimension: peer_dim,
                }
            })
            .collect();

        let sw = &mut self.secondary_windows[sec_idx];
        if let Some(ref mut sec_target) = sw.target {
            let _ = r.render_secondary_viewports_frame(
                sec_target,
                &cams,
                tod,
                &self.engine.settings,
                Some(&sw.egui_ctx),
                Some(output_sec),
                ppp,
            );
        }

        for (slot, fname, fields, close) in peer_form_submits {
            let idx = slot - 1;
            let ctx_pos = self.engine.peers[idx]
                .active_formspec
                .as_ref()
                .and_then(|f| f.context_pos);
            if !fields.is_empty() {
                if let Some(ref rt) = self.engine.lua_rt {
                    rt.send_formspec_fields_at(
                        self.engine.peers[idx].player_state.clone(),
                        &fname,
                        &fields,
                        ctx_pos,
                    );
                }
            }
            if close {
                self.close_peer_formspec(slot);
            }
        }
        for slot in peer_death_respawns {
            let lbl = self.engine.settings.player_name_for_slot(slot).to_string();
            if let Some(peer) = self.engine.peers.get_mut(slot - 1) {
                let reg = self.engine.registries.read();
                respawn_player(
                    &mut peer.player,
                    &peer.player_state,
                    &self.engine.world,
                    &reg,
                    self.engine.lua_rt.as_ref(),
                    &mut self.engine.chat_messages,
                    &lbl,
                );
            }
        }
    }

    fn active_local_body_mask_in_dimension(&self, dim: DimensionId) -> u8 {
        let mut mask = if self.engine.player.dimension == dim {
            1u8
        } else {
            0u8
        };
        for (i, peer) in self.engine.peers.iter().enumerate() {
            if peer.joined && peer.player.dimension == dim {
                mask |= 1u8 << (i + 1);
            }
        }
        mask
    }

    fn gamepad_slot_for_player(&self, slot: usize) -> Option<usize> {
        if slot == 0 {
            match self.engine.settings.p1_input {
                PlayerInputDevice::Gamepad(s) => Some(s),
                PlayerInputDevice::KeyboardMouse if !self.engine.settings.coop_mode.is_active() => {
                    Some(0)
                }
                _ => None,
            }
        } else if !self.engine.settings.coop_mode.is_active() {
            None
        } else {
            let dev = self.engine.settings.player_input_for_slot(slot);
            match dev {
                PlayerInputDevice::Gamepad(s) => {
                    let already_claimed = (0..slot).any(|prev| {
                        self.engine.settings.player_input_for_slot(prev)
                            == PlayerInputDevice::Gamepad(s)
                    });
                    if already_claimed {
                        None
                    } else {
                        Some(s)
                    }
                }
                _ => None,
            }
        }
    }

    fn controller_form_input_for_slot(&self, slot: Option<usize>) -> Option<ControllerFormInput> {
        let gf = slot.and_then(|s| self.gamepads.frame_for_slot(s))?;
        let move_x = if gf.ui_right {
            1
        } else if gf.ui_left {
            -1
        } else {
            0
        };
        let move_y = if gf.ui_down {
            1
        } else if gf.ui_up {
            -1
        } else {
            0
        };
        Some(ControllerFormInput {
            move_x,
            move_y,
            confirm: gf.ui_confirm,
            secondary: gf.ui_secondary,
            shift: gf.ui_shift,
            cancel: gf.ui_cancel,
        })
    }

    fn controller_form_input_for_player(&self, slot: usize) -> Option<ControllerFormInput> {
        self.controller_form_input_for_slot(self.gamepad_slot_for_player(slot))
    }

    /// Return `(player_slot, [px, py, pw, ph], logical_rect)` for every viewport rendered in `window_idx`
    /// (`0` = primary window, `1..=3` = secondary windows).
    fn compute_window_viewports(
        mode: CoopMode,
        player_count: u8,
        surf_w: u32,
        surf_h: u32,
        log_rect: egui::Rect,
        window_idx: usize,
    ) -> Vec<(usize, [u32; 4], egui::Rect)> {
        let w = surf_w.max(2);
        let h = surf_h.max(2);
        let n = mode.active_player_count(player_count);
        let full_px = [0, 0, w, h];
        let lw = (w / 2).max(1);
        let rw = (w - lw).max(1);
        let th = (h / 2).max(1);
        let bh = (h - th).max(1);
        let mx = log_rect.center().x;
        let my = log_rect.center().y;

        match mode {
            CoopMode::Off => {
                if window_idx == 0 {
                    vec![(0, full_px, log_rect)]
                } else {
                    Vec::new()
                }
            }
            CoopMode::SplitVertical => {
                if window_idx != 0 {
                    return Vec::new();
                }
                vec![
                    (
                        0,
                        [0, 0, lw, h],
                        egui::Rect::from_min_max(log_rect.min, egui::Pos2::new(mx, log_rect.bottom())),
                    ),
                    (
                        1,
                        [lw, 0, rw, h],
                        egui::Rect::from_min_max(egui::Pos2::new(mx, log_rect.top()), log_rect.max),
                    ),
                ]
            }
            CoopMode::SplitHorizontal => {
                if window_idx != 0 {
                    return Vec::new();
                }
                vec![
                    (
                        0,
                        [0, 0, w, th],
                        egui::Rect::from_min_max(log_rect.min, egui::Pos2::new(log_rect.right(), my)),
                    ),
                    (
                        1,
                        [0, th, w, bh],
                        egui::Rect::from_min_max(egui::Pos2::new(log_rect.left(), my), log_rect.max),
                    ),
                ]
            }
            CoopMode::DualMonitor => {
                if window_idx == 0 {
                    vec![(0, full_px, log_rect)]
                } else if window_idx == 1 && n >= 2 {
                    vec![(1, full_px, log_rect)]
                } else {
                    Vec::new()
                }
            }
            CoopMode::QuadSplit => {
                if window_idx != 0 {
                    return Vec::new();
                }
                if n <= 2 {
                    return vec![
                        (
                            0,
                            [0, 0, lw, h],
                            egui::Rect::from_min_max(log_rect.min, egui::Pos2::new(mx, log_rect.bottom())),
                        ),
                        (
                            1,
                            [lw, 0, rw, h],
                            egui::Rect::from_min_max(egui::Pos2::new(mx, log_rect.top()), log_rect.max),
                        ),
                    ];
                }
                let mut out = vec![
                    (
                        0,
                        [0, 0, lw, th],
                        egui::Rect::from_min_max(log_rect.min, egui::Pos2::new(mx, my)),
                    ),
                    (
                        1,
                        [lw, 0, rw, th],
                        egui::Rect::from_min_max(
                            egui::Pos2::new(mx, log_rect.top()),
                            egui::Pos2::new(log_rect.right(), my),
                        ),
                    ),
                    (
                        2,
                        [0, th, if n == 3 { w } else { lw }, bh],
                        egui::Rect::from_min_max(
                            egui::Pos2::new(log_rect.left(), my),
                            egui::Pos2::new(if n == 3 { log_rect.right() } else { mx }, log_rect.bottom()),
                        ),
                    ),
                ];
                if n >= 4 {
                    out.push((
                        3,
                        [lw, th, rw, bh],
                        egui::Rect::from_min_max(egui::Pos2::new(mx, my), log_rect.max),
                    ));
                }
                out
            }
            CoopMode::DualScreenTwoEach => {
                if n <= 2 {
                    if window_idx == 0 {
                        vec![(0, full_px, log_rect)]
                    } else if window_idx == 1 {
                        vec![(1, full_px, log_rect)]
                    } else {
                        Vec::new()
                    }
                } else if window_idx == 0 {
                    vec![
                        (
                            0,
                            [0, 0, lw, h],
                            egui::Rect::from_min_max(log_rect.min, egui::Pos2::new(mx, log_rect.bottom())),
                        ),
                        (
                            1,
                            [lw, 0, rw, h],
                            egui::Rect::from_min_max(egui::Pos2::new(mx, log_rect.top()), log_rect.max),
                        ),
                    ]
                } else if window_idx == 1 {
                    if n == 3 {
                        vec![(2, full_px, log_rect)]
                    } else {
                        vec![
                            (
                                2,
                                [0, 0, lw, h],
                                egui::Rect::from_min_max(log_rect.min, egui::Pos2::new(mx, log_rect.bottom())),
                            ),
                            (
                                3,
                                [lw, 0, rw, h],
                                egui::Rect::from_min_max(egui::Pos2::new(mx, log_rect.top()), log_rect.max),
                            ),
                        ]
                    }
                } else {
                    Vec::new()
                }
            }
            CoopMode::QuadMonitor => {
                if window_idx < n {
                    vec![(window_idx, full_px, log_rect)]
                } else {
                    Vec::new()
                }
            }
        }
    }

    fn apply_single_net_packet(&mut self, pkt: NetPacket) {
        match pkt {
            NetPacket::BlockEdit {
                x,
                y,
                z,
                node_name,
                param1,
                param2,
            } => {
                let id = if node_name == "air" {
                    CONTENT_AIR
                } else {
                    self.engine.registries.read().get_content_id(&node_name)
                };
                let wp = IVec3::new(x, y, z);
                if self.engine.world.get_node(wp) != (id, param1, param2) {
                    self.engine.world.set_node(wp, id, param1, param2);
                    self.engine.world.mark_priority_dirty_with_neighbors(wp);
                }
            }
            NetPacket::BlockSet {
                x,
                y,
                z,
                node_name,
            } => {
                let id = if node_name == "air" {
                    CONTENT_AIR
                } else {
                    self.engine.registries.read().get_content_id(&node_name)
                };
                let wp = IVec3::new(x, y, z);
                if self.engine.world.get_node(wp) != (id, 15, 0) {
                    self.engine.world.set_node(wp, id, 15, 0);
                    self.engine.world.mark_priority_dirty_with_neighbors(wp);
                }
            }
            NetPacket::ChunkDelta {
                cx,
                cy,
                cz,
                edits,
            } => {
                let reg = self.engine.registries.read();
                let mut touched = false;
                for d in edits {
                    let id = if d.node_name == "air" {
                        CONTENT_AIR
                    } else {
                        reg.get_content_id(&d.node_name)
                    };
                    let wp = IVec3::new(
                        cx * 16 + d.lx as i32,
                        cy * 16 + d.ly as i32,
                        cz * 16 + d.lz as i32,
                    );
                    if self.engine.world.get_node(wp) != (id, d.param1, d.param2) {
                        self.engine.world.set_node(wp, id, d.param1, d.param2);
                        self.engine.world.mark_priority_dirty_with_neighbors(wp);
                        touched = true;
                    }
                }
                if touched {
                    self.engine
                        .world
                        .dirty_queue
                        .write()
                        .insert(IVec3::new(cx, cy, cz));
                }
            }
            NetPacket::Chat { sender, message } => {
                self.engine
                    .chat_messages
                    .push(format!("<{sender}> {message}"));
            }
            NetPacket::ChunkData {
                cx,
                cy,
                cz,
                compressed_b64,
            } => {
                let reg = self.engine.registries.read();
                if let Some(ch) = decode_chunk_data(cx, cy, cz, &compressed_b64, &reg) {
                    self.engine.world.insert_chunk(ch);
                }
            }
            NetPacket::TimeWeather {
                time_of_day,
                weather,
            } => {
                if time_of_day >= 0.0 {
                    if let Some(ref rt) = self.engine.lua_rt {
                        *rt.shared.time_of_day.write() = time_of_day;
                    }
                }
                if weather != "unchanged" {
                    self.engine.pause_state.weather = weather;
                }
            }
            NetPacket::PlayerMove {
                player_name,
                x,
                y,
                z,
                yaw,
                pitch,
                swinging,
                sneaking,
                held_item,
                ..
            } => {
                let is_local = player_name == self.engine.settings.player_name
                    || self
                        .engine
                        .peers
                        .iter()
                        .any(|p| p.joined && p.player_state.read().name == player_name);
                if is_local {
                    return;
                }
                let new_pos = Vec3::new(x, y, z);
                let now = Instant::now();
                let entry = self
                    .engine
                    .remote_players
                    .entry(player_name.clone())
                    .or_insert_with(|| RemoteLanPlayer {
                        name: player_name,
                        pos: new_pos,
                        prev_pos: new_pos,
                        yaw,
                        pitch,
                        horizontal_speed: 0.0,
                        swinging,
                        swing_timer: if swinging { 0.25 } else { 0.0 },
                        sneaking,
                        held_item: held_item.clone(),
                        hurt_flash: 0.0,
                        last_seen: now,
                    });
                let dt_move = now.duration_since(entry.last_seen).as_secs_f32().clamp(0.016, 0.5);
                let dist_xz = ((new_pos.x - entry.pos.x).powi(2)
                    + (new_pos.z - entry.pos.z).powi(2))
                .sqrt();
                let inst_speed = dist_xz / dt_move;
                entry.prev_pos = entry.pos;
                entry.pos = new_pos;
                entry.horizontal_speed = entry.horizontal_speed * 0.4 + inst_speed * 0.6;
                entry.yaw = yaw;
                entry.pitch = pitch;
                entry.sneaking = sneaking;
                entry.held_item = held_item;
                entry.swinging = swinging;
                if swinging {
                    entry.swing_timer = 0.28;
                }
                entry.last_seen = now;
            }
            NetPacket::PlayerPos {
                player_name,
                x,
                y,
                z,
                yaw,
            } => {
                let is_local = player_name == self.engine.settings.player_name
                    || self
                        .engine
                        .peers
                        .iter()
                        .any(|p| p.joined && p.player_state.read().name == player_name);
                if !is_local {
                    let new_pos = Vec3::new(x, y, z);
                    let now = Instant::now();
                    let entry = self
                        .engine
                        .remote_players
                        .entry(player_name.clone())
                        .or_insert_with(|| RemoteLanPlayer {
                            name: player_name,
                            pos: new_pos,
                            prev_pos: new_pos,
                            yaw,
                            pitch: 0.0,
                            horizontal_speed: 0.0,
                            swinging: false,
                            swing_timer: 0.0,
                            sneaking: false,
                            held_item: String::new(),
                            hurt_flash: 0.0,
                            last_seen: now,
                        });
                    entry.pos = new_pos;
                    entry.yaw = yaw;
                    entry.last_seen = now;
                }
            }
            NetPacket::PlayerLeave { player_name } => {
                self.engine.remote_players.remove(&player_name);
            }
            NetPacket::PlayerPunch {
                target,
                damage,
                kb_x,
                kb_y,
                kb_z,
                ..
            } => {
                let dmg = damage.max(1);
                let kb = Vec3::new(kb_x, kb_y, kb_z);
                if target == self.engine.settings.player_name {
                    if self.engine.player.gamemode == GameMode::Survival
                        && !self.engine.player.is_dead
                        && self.engine.player.spawn_invuln_timer <= 0.0
                    {
                        let old_hp = self.engine.player.hp;
                        self.engine.player.hp = (self.engine.player.hp - dmg).max(0);
                        log_player_damage(
                            DamageSource::Mob,
                            dmg,
                            old_hp,
                            self.engine.player.hp,
                            self.engine.player.pos,
                            "net_punch",
                        );
                        if self.engine.player.hp == 0 {
                            self.engine.player.death_cause = "Slain in combat".into();
                        }
                        self.engine.player.vel += kb;
                        let mut ps = self.engine.player_state.write();
                        ps.hp = self.engine.player.hp;
                        ps.hurt_flash = 0.25;
                    }
                    self.engine.audio.play_named(
                        "player_damage",
                        0.8,
                        1.0,
                        SoundCategory::Blocks,
                        &self.engine.settings.audio,
                    );
                } else {
                    for peer in &mut self.engine.peers {
                        if peer.joined && peer.player_state.read().name == target {
                            if peer.player.gamemode == GameMode::Survival
                                && !peer.player.is_dead
                                && peer.player.spawn_invuln_timer <= 0.0
                            {
                                let old_hp = peer.player.hp;
                                peer.player.hp = (peer.player.hp - dmg).max(0);
                                log_player_damage(
                                    DamageSource::Mob,
                                    dmg,
                                    old_hp,
                                    peer.player.hp,
                                    peer.player.pos,
                                    "net_punch_peer",
                                );
                                if peer.player.hp == 0 {
                                    peer.player.death_cause = "Slain in combat".into();
                                }
                                peer.player.vel += kb;
                                let mut ps = peer.player_state.write();
                                ps.hp = peer.player.hp;
                                ps.hurt_flash = 0.25;
                            }
                            break;
                        }
                    }
                }
                if let Some(rp) = self.engine.remote_players.get_mut(&target) {
                    rp.hurt_flash = 0.25;
                }
            }
            _ => {}
        }
    }

    fn apply_all_incoming_net_packets(&mut self, dt: f32) {
        for rp in self.engine.remote_players.values_mut() {
            rp.hurt_flash = (rp.hurt_flash - dt).max(0.0);
            rp.swing_timer = (rp.swing_timer - dt).max(0.0);
            if rp.last_seen.elapsed() > Duration::from_millis(250) {
                rp.horizontal_speed *= (1.0 - (dt * 6.0).min(1.0)).max(0.0);
            }
        }
        self.engine
            .remote_players
            .retain(|_, rp| rp.last_seen.elapsed() < Duration::from_secs(20));

        if let Some(ref session) = self.engine.net_session {
            let pkts = session.drain_packets();
            for pkt in pkts {
                self.apply_single_net_packet(pkt);
            }
        }
        let mut srv_pkts = Vec::new();
        if let Some(ref mut rx) = self.engine.local_server_rx {
            while let Ok(pkt) = rx.try_recv() {
                srv_pkts.push(pkt);
            }
        }
        for pkt in srv_pkts {
            self.apply_single_net_packet(pkt);
        }
    }

    fn render_tick(&mut self, event_loop: &ActiveEventLoop) {
        let now = Instant::now();
        let dt = now
            .duration_since(self.engine.last_frame)
            .as_secs_f32()
            .clamp(0.0005, 0.05);
        self.engine.last_frame = now;
        let cur_fps = 1.0 / dt;
        self.engine.fps = self.engine.fps * 0.9 + cur_fps * 0.1;
        self.engine.frame_ms = dt * 1000.0;

        // Poll PS5 DualSense / standard gamepads once per tick
        self.gamepads
            .poll(self.engine.settings.controller_deadzone, dt);
        let controller_labels: Vec<String> = self
            .gamepads
            .connected_gamepads()
            .iter()
            .map(|gp| {
                let tag = if gp.is_dualsense { "PS5 DualSense" } else { "Gamepad" };
                format!("#{} {} [{}]", gp.slot + 1, gp.name, tag)
            })
            .collect();
        self.engine.title_state.connected_controllers = controller_labels.clone();
        self.engine.pause_state.connected_controllers = controller_labels;

        // Keep multi-monitor windows synchronized with active CoopMode
        self.sync_coop_window(event_loop);

        // Instant drop-in up to 4 players: unassigned controller pressing Options joins next slot cleanly
        if self.engine.mode == AppMode::Playing {
            let active_count = self
                .engine
                .settings
                .coop_mode
                .active_player_count(self.engine.settings.coop_player_count);
            let mut assigned = Vec::new();
            if !self.engine.settings.coop_mode.is_active() {
                if let PlayerInputDevice::Gamepad(s) = self.engine.settings.p1_input {
                    assigned.push(s);
                } else if self.engine.settings.p1_input == PlayerInputDevice::KeyboardMouse
                    && self.gamepads.connected_gamepads().len() == 1
                {
                    if let Some(gp) = self.gamepads.connected_gamepads().first() {
                        assigned.push(gp.slot);
                    }
                }
            } else {
                for slot in 0..active_count {
                    if let PlayerInputDevice::Gamepad(s) =
                        self.engine.settings.player_input_for_slot(slot)
                    {
                        assigned.push(s);
                    }
                }
            }
            if active_count < 4 {
                if let Some(new_slot) = self.gamepads.any_unassigned_start_pressed(&assigned) {
                    let mon_count = event_loop.available_monitors().count();
                    if !self.engine.settings.coop_mode.is_active() {
                        self.engine.settings.coop_mode = if mon_count >= 2 {
                            CoopMode::DualMonitor
                        } else {
                            CoopMode::SplitVertical
                        };
                        self.engine.settings.coop_player_count = 2;
                        self.engine.settings.p2_input = PlayerInputDevice::Gamepad(new_slot);
                    } else {
                        let next_n = (active_count + 1).clamp(2, 4);
                        self.engine.settings.coop_player_count = next_n as u8;
                        if next_n > self.engine.settings.coop_mode.max_players() {
                            self.engine.settings.coop_mode = if mon_count >= next_n {
                                CoopMode::QuadMonitor
                            } else if mon_count >= 2 {
                                CoopMode::DualScreenTwoEach
                            } else {
                                CoopMode::QuadSplit
                            };
                        }
                        self.engine
                            .settings
                            .set_player_input_for_slot(next_n - 1, PlayerInputDevice::Gamepad(new_slot));
                    }
                    self.engine.ensure_coop_players_joined();
                    self.engine.apply_settings_to_lua_and_disk();
                    self.sync_coop_window(event_loop);
                    let new_n = self
                        .engine
                        .settings
                        .coop_mode
                        .active_player_count(self.engine.settings.coop_player_count);
                    self.engine.chat_messages.push(format!(
                        "§aPlayer {new_n} joined with Controller #{} ({})!",
                        new_slot + 1,
                        self.engine.settings.coop_mode.label()
                    ));
                }
            }
        }

        if self.engine.mode == AppMode::Playing && self.engine.settings.coop_mode.is_active() {
            self.engine.ensure_coop_players_joined();
        }

        // Poll incoming network packets (LAN host + client sync)
        self.apply_all_incoming_net_packets(dt);

        // Progress multi-phase world loading screen without blocking event loop
        if self.engine.mode == AppMode::Loading {
            self.step_world_load();
        }

        // Stream terrain and queue parallel background meshing during active world play
        if self.engine.mode == AppMode::Playing {
            self.engine.stream_chunks_around_player();
        } else {
            self.lmb_just_pressed = false;
            self.p1_mine_was_held = false;
            for peer in &mut self.engine.peers {
                peer.mine_was_held = false;
            }
            if !matches!(self.engine.mode, AppMode::Paused) {
                if let Some(ref mut r) = self.renderer {
                    let reg = self.engine.registries.read();
                    r.update_entities(&[], &reg, &self.engine.atlas);
                    r.update_nether_entities(&[], &reg, &self.engine.atlas);
                    r.update_guardian_lasers(&[]);
                    r.update_remote_players(&[], dt, &reg, &self.engine.atlas);
                }
            }
        }

        let (cam_pos, view_proj, targeted_opt) = if self.engine.mode == AppMode::Splash {
            if let Some(ref mut r) = self.renderer {
                r.show_viewmodel = false;
                r.show_player_model = false;
                r.set_selection_boxes(&[]);
                r.set_crack_overlay(&[], None, &self.engine.atlas);
                r.fog_dist_override = Some(1.0);
            }
            let aspect = self
                .renderer
                .as_ref()
                .map(|r| {
                    r.surface_config.width as f32 / r.surface_config.height.max(1) as f32
                })
                .unwrap_or(16.0 / 9.0);
            let proj = glam::Mat4::perspective_rh(70.0f32.to_radians(), aspect.max(0.1), 0.1, 32.0)
                * glam::Mat4::look_at_rh(Vec3::ZERO, Vec3::Z, Vec3::Y);
            (Vec3::ZERO, proj, None)
        } else if self.engine.mode == AppMode::Title {
            let t = self
                .renderer
                .as_ref()
                .map(|r| r.start_time.elapsed().as_secs_f32())
                .unwrap_or(0.0);
            let aspect = self
                .renderer
                .as_ref()
                .map(|r| {
                    r.surface_config.width as f32 / r.surface_config.height.max(1) as f32
                })
                .unwrap_or(16.0 / 9.0);
            let (eye, vp) = WgpuRenderer::compute_panorama_camera(
                t,
                aspect,
                self.engine.panorama_center,
            );
            if let Some(ref mut r) = self.renderer {
                r.show_viewmodel = false;
                r.show_player_model = false;
                r.set_selection_boxes(&[]);
                r.set_crack_overlay(&[], None, &self.engine.atlas);
                r.fog_dist_override = Some(self.engine.panorama_fog_dist);
            }
            (eye, vp, None)
        } else if self.engine.mode == AppMode::Loading {
            if let Some(ref mut r) = self.renderer {
                r.show_viewmodel = false;
                r.show_player_model = false;
                r.set_selection_boxes(&[]);
                r.set_crack_overlay(&[], None, &self.engine.atlas);
                r.fog_dist_override = None;
            }
            let aspect = self
                .renderer
                .as_ref()
                .map(|r| {
                    r.surface_config.width as f32 / r.surface_config.height.max(1) as f32
                })
                .unwrap_or(16.0 / 9.0);
            let proj = glam::Mat4::perspective_rh(70.0f32.to_radians(), aspect.max(0.1), 0.1, 64.0)
                * glam::Mat4::look_at_rh(Vec3::ZERO, Vec3::Z, Vec3::Y);
            (Vec3::ZERO, proj, None)
        } else {
            if let Some(ref mut r) = self.renderer {
                r.fog_dist_override = None;
            }
            let p1_gp = self
                .gamepad_slot_for_player(0)
                .and_then(|s| self.gamepads.frame_for_slot(s).cloned());
            let peer_gps: [Option<cubex_rs::input::GamepadFrameInput>; 3] = [
                self.gamepad_slot_for_player(1)
                    .and_then(|s| self.gamepads.frame_for_slot(s).cloned()),
                self.gamepad_slot_for_player(2)
                    .and_then(|s| self.gamepads.frame_for_slot(s).cloned()),
                self.gamepad_slot_for_player(3)
                    .and_then(|s| self.gamepads.frame_for_slot(s).cloned()),
            ];

            if self.engine.mode == AppMode::Paused {
                if p1_gp.as_ref().is_some_and(|g| g.pause_pressed)
                    || peer_gps.iter().any(|g| g.as_ref().is_some_and(|f| f.pause_pressed))
                {
                    self.engine.mode = AppMode::Playing;
                    self.set_cursor_locked(true);
                }
            }

            let mut p1_step_input = self.engine.move_input.clone();
            let mut p1_mine_held = self.lmb_down && self.cursor_locked;
            let mut p1_place_just = self.rmb_just_pressed && self.cursor_locked;
            let mut p1_touch_mine_just = false;
            let mut bot_force_liquids_pointable = false;
            let mut bot_melee_just = false;
            self.rmb_just_pressed = false;

            if self.engine.mode == AppMode::Playing && self.engine.settings.touch_controls {
                let tvp = self.primary_touch_viewport();
                let tf = self.touch_controls.take_frame_input(tvp);
                if self.engine.active_formspec.is_none() && self.engine.chat_typing.is_none() {
                    if tf.pause_just_pressed {
                        self.engine.mode = AppMode::Paused;
                        self.engine.pause_state.page = PausePage::Main;
                        self.engine.move_input = MoveInput::default();
                        self.engine.digger.reset();
                        self.touch_controls.reset_all();
                        self.set_cursor_locked(false);
                    } else if tf.open_inv_just_pressed {
                        self.open_inventory_formspec();
                    } else {
                        let touch_sens = self.engine.settings.touch_sensitivity * 0.006;
                        if tf.look_dx != 0.0 || tf.look_dy != 0.0 {
                            self.engine.player.yaw += tf.look_dx * touch_sens;
                            let sign = if self.engine.settings.invert_mouse { 1.0 } else { -1.0 };
                            self.engine.player.pitch = (self.engine.player.pitch
                                + tf.look_dy * touch_sens * sign)
                                .clamp(-1.55, 1.55);
                            if let Some(ref mut r) = self.renderer {
                                r.viewmodel.add_look_inertia(tf.look_dx, tf.look_dy);
                            }
                        }
                        if tf.move_x != 0.0 || tf.move_y != 0.0 {
                            p1_step_input.analog_x = tf.move_x;
                            p1_step_input.analog_y = tf.move_y;
                            p1_step_input.sprint |= tf.sprint;
                        }
                        p1_step_input.jump |= tf.jump_held;
                        p1_step_input.sneak |= tf.sneak_toggled;
                        p1_mine_held |= tf.attack_held;
                        p1_touch_mine_just = tf.attack_just_pressed;
                        if tf.place_just_pressed {
                            p1_place_just = true;
                        }
                        if tf.cycle_cam_just_pressed {
                            self.camera_mode = self.camera_mode.cycle();
                        }
                        if let Some(slot) = tf.select_slot {
                            self.engine.player.selected_slot = slot.min(8);
                            self.engine.player_state.write().wield_index = slot.min(8) + 1;
                        }
                    }
                }
            }

            if self.engine.mode == AppMode::Playing {
                if let Some(ref gf) = p1_gp {
                    if gf.f3_pressed {
                        self.engine.settings.f3_enabled = !self.engine.settings.f3_enabled;
                        self.engine.apply_settings_to_lua_and_disk();
                    }
                    if gf.pause_pressed {
                        self.engine.mode = AppMode::Paused;
                        self.engine.pause_state.page = PausePage::Main;
                        self.engine.move_input = MoveInput::default();
                        self.engine.digger.reset();
                        self.set_cursor_locked(false);
                    } else if self.engine.active_formspec.is_some() {
                        if gf.inventory_pressed
                            && self
                                .engine
                                .active_formspec
                                .as_ref()
                                .is_some_and(|f| is_player_inventory_formname(&f.formname))
                        {
                            self.close_active_formspec();
                        }
                    } else if self.engine.chat_typing.is_none() {
                        self.engine
                            .player
                            .apply_gamepad_look(gf.look_x, gf.look_y, dt, &self.engine.settings);
                        if gf.look_x.abs() > 0.01 || gf.look_y.abs() > 0.01 {
                            if let Some(ref mut r) = self.renderer {
                                r.viewmodel
                                    .add_look_inertia(gf.look_x * 14.0, -gf.look_y * 14.0);
                            }
                        }
                        if gf.sneak_toggled {
                            self.p1_ctrl_sneak = !self.p1_ctrl_sneak;
                        }
                        if gf.sprint_toggled {
                            self.p1_ctrl_sprint = !self.p1_ctrl_sprint;
                        }
                        if (gf.move_x * gf.move_x + gf.move_y * gf.move_y) < 0.04 {
                            self.p1_ctrl_sprint = false;
                        }
                        if p1_step_input.analog_x == 0.0 && p1_step_input.analog_y == 0.0 {
                            p1_step_input.analog_x = gf.move_x;
                            p1_step_input.analog_y = gf.move_y;
                        }
                        p1_step_input.jump |= gf.jump;
                        p1_step_input.sneak |= self.p1_ctrl_sneak;
                        p1_step_input.sprint |= self.p1_ctrl_sprint;
                        p1_step_input.zoom |= gf.zoom;
                        if gf.camera_toggle_pressed {
                            self.camera_mode = self.camera_mode.cycle();
                        }
                        if gf.hotbar_prev {
                            self.engine.player.selected_slot =
                                (self.engine.player.selected_slot + 8) % 9;
                            self.engine.player_state.write().wield_index =
                                self.engine.player.selected_slot + 1;
                        }
                        if gf.hotbar_next {
                            self.engine.player.selected_slot =
                                (self.engine.player.selected_slot + 1) % 9;
                            self.engine.player_state.write().wield_index =
                                self.engine.player.selected_slot + 1;
                        }
                        if gf.drop_pressed {
                            drop_held_slot_item(
                                &self.engine.player,
                                &self.engine.player_state,
                                self.engine.lua_rt.as_ref(),
                            );
                        }
                        if gf.inventory_pressed {
                            self.open_inventory_formspec();
                        } else {
                            p1_mine_held |= gf.mine_held;
                            if gf.place_pressed {
                                p1_place_just = true;
                                self.p1_place_cd = 0.24;
                            } else if gf.place_held {
                                self.p1_place_cd -= dt;
                                if self.p1_place_cd <= 0.0 {
                                    p1_place_just = true;
                                    self.p1_place_cd = 0.20;
                                }
                            } else {
                                self.p1_place_cd = 0.0;
                            }
                        }
                    }
                }
            }

            // Collect per-peer (P2..P4) inputs when Co-op active
            let mut peer_inputs: [MoveInput; 3] = Default::default();
            let mut peer_mine_held = [false; 3];
            let mut peer_place_just = [false; 3];
            let mut want_open_peer_inv = [false; 3];
            let mut want_close_peer_inv = [false; 3];
            let mut want_pause_from_peer = false;

            if self.engine.mode == AppMode::Playing && self.engine.settings.coop_mode.is_active() {
                for idx in 0..3 {
                    let slot = idx + 1;
                    if !self.engine.peers[idx].joined {
                        continue;
                    }
                    let dev = self.engine.settings.player_input_for_slot(slot);
                    let peer = &mut self.engine.peers[idx];
                    if let Some(ref gf) = peer_gps[idx] {
                        if gf.pause_pressed {
                            want_pause_from_peer = true;
                        } else if peer.active_formspec.is_some() {
                            if gf.inventory_pressed
                                && peer
                                    .active_formspec
                                    .as_ref()
                                    .is_some_and(|f| is_player_inventory_formname(&f.formname))
                            {
                                want_close_peer_inv[idx] = true;
                            }
                        } else {
                            peer.player.apply_gamepad_look(
                                gf.look_x,
                                gf.look_y,
                                dt,
                                &self.engine.settings,
                            );
                            if gf.look_x.abs() > 0.01 || gf.look_y.abs() > 0.01 {
                                if let Some(ref mut r) = self.renderer {
                                    if let Some(pslot) = r.peer_slot_mut(slot) {
                                        pslot
                                            .viewmodel
                                            .add_look_inertia(gf.look_x * 14.0, -gf.look_y * 14.0);
                                    }
                                }
                            }
                            if gf.sneak_toggled {
                                peer.sneak_toggled = !peer.sneak_toggled;
                            }
                            if gf.sprint_toggled {
                                peer.sprint_toggled = !peer.sprint_toggled;
                            }
                            if (gf.move_x * gf.move_x + gf.move_y * gf.move_y) < 0.04 {
                                peer.sprint_toggled = false;
                            }
                            peer_inputs[idx].analog_x = gf.move_x;
                            peer_inputs[idx].analog_y = gf.move_y;
                            peer_inputs[idx].jump = gf.jump;
                            peer_inputs[idx].sneak = peer.sneak_toggled;
                            peer_inputs[idx].sprint = peer.sprint_toggled;
                            peer_inputs[idx].zoom = gf.zoom;
                            if gf.camera_toggle_pressed {
                                peer.camera_mode = peer.camera_mode.cycle();
                            }
                            if gf.hotbar_prev {
                                peer.player.selected_slot = (peer.player.selected_slot + 8) % 9;
                                peer.player_state.write().wield_index =
                                    peer.player.selected_slot + 1;
                            }
                            if gf.hotbar_next {
                                peer.player.selected_slot = (peer.player.selected_slot + 1) % 9;
                                peer.player_state.write().wield_index =
                                    peer.player.selected_slot + 1;
                            }
                            if gf.drop_pressed {
                                drop_held_slot_item(
                                    &peer.player,
                                    &peer.player_state,
                                    self.engine.lua_rt.as_ref(),
                                );
                            }
                            if gf.inventory_pressed {
                                want_open_peer_inv[idx] = true;
                            } else {
                                peer_mine_held[idx] = gf.mine_held;
                                if gf.place_pressed {
                                    peer_place_just[idx] = true;
                                    peer.place_repeat_cd = 0.24;
                                } else if gf.place_held {
                                    peer.place_repeat_cd -= dt;
                                    if peer.place_repeat_cd <= 0.0 {
                                        peer_place_just[idx] = true;
                                        peer.place_repeat_cd = 0.20;
                                    }
                                } else {
                                    peer.place_repeat_cd = 0.0;
                                }
                            }
                        }
                    } else if dev == PlayerInputDevice::Keyboard2 {
                        let rx = (if self.kb2.yaw_right { 1.0 } else { 0.0 })
                            - (if self.kb2.yaw_left { 1.0 } else { 0.0 });
                        let ry = (if self.kb2.pitch_up { 1.0 } else { 0.0 })
                            - (if self.kb2.pitch_down { 1.0 } else { 0.0 });
                        peer.player
                            .apply_gamepad_look(rx, ry, dt, &self.engine.settings);
                        peer_inputs[idx].forward = self.kb2.forward;
                        peer_inputs[idx].backward = self.kb2.backward;
                        peer_inputs[idx].left = self.kb2.left;
                        peer_inputs[idx].right = self.kb2.right;
                        peer_inputs[idx].jump = self.kb2.jump;
                        peer_inputs[idx].sneak = self.kb2.sneak;
                        if self.kb2.hotbar_prev {
                            self.kb2.hotbar_prev = false;
                            peer.player.selected_slot = (peer.player.selected_slot + 8) % 9;
                            peer.player_state.write().wield_index = peer.player.selected_slot + 1;
                        }
                        if self.kb2.hotbar_next {
                            self.kb2.hotbar_next = false;
                            peer.player.selected_slot = (peer.player.selected_slot + 1) % 9;
                            peer.player_state.write().wield_index = peer.player.selected_slot + 1;
                        }
                        if self.kb2.inv_just {
                            self.kb2.inv_just = false;
                            if peer.active_formspec.is_some() {
                                want_close_peer_inv[idx] = true;
                            } else {
                                want_open_peer_inv[idx] = true;
                            }
                        }
                        peer_mine_held[idx] = self.kb2.mine_held;
                        if self.kb2.place_just {
                            self.kb2.place_just = false;
                            peer_place_just[idx] = true;
                        }
                    }
                }
                if want_pause_from_peer {
                    self.engine.mode = AppMode::Paused;
                    self.engine.pause_state.page = PausePage::Main;
                    self.engine.digger.reset();
                    for peer in &mut self.engine.peers {
                        peer.digger.reset();
                    }
                    self.set_cursor_locked(false);
                } else {
                    for idx in 0..3 {
                        if want_close_peer_inv[idx] {
                            self.close_peer_formspec(idx + 1);
                        } else if want_open_peer_inv[idx] {
                            self.open_peer_inventory_formspec(idx + 1);
                        }
                    }
                }
            }

            if self.engine.mode == AppMode::Playing && self.autotest_bot.is_some() {
                self.engine.active_formspec = None;
                self.engine.chat_typing = None;
                if let Some(mut bot) = self.autotest_bot.take() {
                    let p1_world_for_bot = world_for(
                        &self.engine.world,
                        &self.engine.nether_world,
                        self.engine.player.dimension,
                    );
                    let p1_lua_for_bot = if self.engine.player.dimension.is_nether() {
                        None
                    } else {
                        self.engine.lua_rt.as_ref()
                    };
                    let bot_out = {
                        let reg = self.engine.registries.read();
                        bot.tick(
                            dt,
                            self.engine.fps,
                            &mut self.engine.player,
                            &self.engine.player_state,
                            p1_world_for_bot,
                            &reg,
                            p1_lua_for_bot,
                            self.engine.settings.block_shatter_physics,
                        )
                    };
                    p1_step_input = bot_out.move_input;
                    p1_mine_held = bot_out.mine_held;
                    p1_place_just = bot_out.place_just;
                    bot_force_liquids_pointable = bot_out.force_liquids_pointable;
                    if !bot_out.remesh_positions.is_empty() {
                        let reg = self.engine.registries.read();
                        if let Some(ref mut r) = self.renderer {
                            for bpos in &bot_out.remesh_positions {
                                r.relight_and_remesh_at(
                                    p1_world_for_bot,
                                    *bpos,
                                    &reg,
                                    &self.engine.atlas,
                                    self.engine.settings.smooth_lighting,
                                );
                            }
                        }
                    }
                    if let Some(report) = bot_out.finished {
                        self.pending_autotest_finish =
                            Some((bot.out_dir().to_path_buf(), report));
                    }
                    self.autotest_bot = Some(bot);
                }
            } else if self.engine.mode == AppMode::Playing && self.qa_bot.is_some() {
                self.engine.active_formspec = None;
                self.engine.chat_typing = None;
                if let Some(mut bot) = self.qa_bot.take() {
                    let bot_out = {
                        let reg = self.engine.registries.read();
                        let p1_lua_for_bot = if self.engine.player.dimension.is_nether() {
                            None
                        } else {
                            self.engine.lua_rt.as_ref()
                        };
                        bot.tick(
                            dt,
                            self.engine.fps,
                            &mut self.engine.player,
                            &self.engine.player_state,
                            &self.engine.world,
                            &self.engine.nether_world,
                            &self.engine.palette,
                            &self.engine.nether_palette,
                            &reg,
                            p1_lua_for_bot,
                            &mut self.engine.mob_manager,
                            self.engine.settings.block_shatter_physics,
                        )
                    };
                    p1_step_input = bot_out.move_input;
                    p1_mine_held = bot_out.mine_held;
                    p1_place_just = bot_out.place_just;
                    bot_melee_just = bot_out.melee_attack_just;
                    bot_force_liquids_pointable = bot_out.force_liquids_pointable;

                    if let Some((dest_dim, dest_pos)) = bot_out.teleport_request {
                        self.engine.player.dimension = dest_dim;
                        self.engine.player.pos = dest_pos;
                        self.engine.player.vel = Vec3::ZERO;
                        self.engine.player.reset_fall_state();
                        self.engine.digger.reset();
                        let dest_world = world_for(
                            &self.engine.world,
                            &self.engine.nether_world,
                            dest_dim,
                        );
                        let cx = (dest_pos.x / 16.0).floor() as i32;
                        let cz = (dest_pos.z / 16.0).floor() as i32;
                        {
                            let gen_set = if dest_dim.is_nether() {
                                &mut self.engine.nether_generated_columns
                            } else {
                                &mut self.engine.generated_columns
                            };
                            for dz in -2..=2 {
                                for dx in -2..=2 {
                                    gen_set.insert((cx + dx, cz + dz));
                                }
                            }
                        }
                        if let Some(ref mut r) = self.renderer {
                            let reg = self.engine.registries.read();
                            let y_range: Vec<i32> = if dest_dim.is_nether() {
                                NETHER_CHUNK_Y_RANGE.collect()
                            } else {
                                chunk_y_range_for(*dest_world.world_type.read()).collect()
                            };
                            for dz in -2..=2 {
                                for dx in -2..=2 {
                                    for &cy in &y_range {
                                        if let Some(snap) =
                                            dest_world.snapshot_chunk(IVec3::new(cx + dx, cy, cz + dz))
                                        {
                                            let mesh = mesh_chunk_with_atlas(
                                                &snap,
                                                &reg,
                                                &self.engine.atlas,
                                                self.engine.settings.smooth_lighting,
                                            );
                                            r.upload_chunk_mesh_for_dimension(mesh, dest_dim);
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if !bot_out.remesh_positions.is_empty() {
                        let p1_world_for_bot = world_for(
                            &self.engine.world,
                            &self.engine.nether_world,
                            self.engine.player.dimension,
                        );
                        let reg = self.engine.registries.read();
                        if let Some(ref mut r) = self.renderer {
                            for bpos in &bot_out.remesh_positions {
                                r.relight_and_remesh_at(
                                    p1_world_for_bot,
                                    *bpos,
                                    &reg,
                                    &self.engine.atlas,
                                    self.engine.settings.smooth_lighting,
                                );
                            }
                        }
                    }

                    if let Some(report) = bot_out.finished {
                        self.pending_qa_finish = Some((bot.out_dir().to_path_buf(), report));
                    }
                    self.qa_bot = Some(bot);
                }
            }

            // Step P1 & P2..P4 movement, physics, and Survival respawn
            let p1_active_input = self.engine.mode == AppMode::Playing
                && self.engine.active_formspec.is_none()
                && self.engine.chat_typing.is_none();
            let p1_mine_just = p1_active_input
                && ((self.lmb_just_pressed && self.cursor_locked)
                    || p1_touch_mine_just
                    || bot_melee_just
                    || (p1_mine_held && !self.p1_mine_was_held));
            self.lmb_just_pressed = false;
            self.p1_mine_was_held = p1_mine_held && p1_active_input;

            let mut peer_mine_just = [false; 3];
            for idx in 0..3 {
                let peer = &mut self.engine.peers[idx];
                let active = self.engine.mode == AppMode::Playing
                    && self.engine.settings.coop_mode.is_active()
                    && peer.joined
                    && peer.active_formspec.is_none();
                peer_mine_just[idx] = active && peer_mine_held[idx] && !peer.mine_was_held;
                peer.mine_was_held = peer_mine_held[idx] && active;
            }

            if self.engine.mode == AppMode::Playing {
                let keep_inv = self.engine.settings.gamerules.keep_inventory;
                if self.engine.active_formspec.is_none() && self.engine.chat_typing.is_none() {
                    let reg = self.engine.registries.read();
                    let p1_world = world_for(
                        &self.engine.world,
                        &self.engine.nether_world,
                        self.engine.player.dimension,
                    );
                    let p1_lua = if self.engine.player.dimension.is_nether() {
                        None
                    } else {
                        self.engine.lua_rt.as_ref()
                    };
                    step_player_physics_and_respawn(
                        &mut self.engine.player,
                        &self.engine.player_state,
                        &p1_step_input,
                        p1_world,
                        &reg,
                        p1_lua,
                        &self.engine.audio,
                        &self.engine.settings,
                        keep_inv,
                        &mut self.engine.chat_messages,
                        "You",
                        dt,
                    );
                    if let Some(new_dim) = tick_player_portal(
                        &mut self.engine.player,
                        &self.engine.player_state,
                        &self.engine.world,
                        &self.engine.nether_world,
                        &self.engine.palette,
                        &self.engine.nether_palette,
                        &reg,
                        &mut self.engine.chat_messages,
                        "You",
                        dt,
                    ) {
                        let dest_world = world_for(&self.engine.world, &self.engine.nether_world, new_dim);
                        let (cx, cz) = (
                            (self.engine.player.pos.x / 16.0).floor() as i32,
                            (self.engine.player.pos.z / 16.0).floor() as i32,
                        );
                        let gen_set = if new_dim.is_nether() {
                            &mut self.engine.nether_generated_columns
                        } else {
                            &mut self.engine.generated_columns
                        };
                        for dz in -1..=1 {
                            for dx in -1..=1 {
                                gen_set.insert((cx + dx, cz + dz));
                            }
                        }
                        if let Some(ref mut r) = self.renderer {
                            r.remesh_priority_chunks_now(
                                dest_world,
                                &reg,
                                &self.engine.atlas,
                                self.engine.settings.smooth_lighting,
                            );
                        }
                    }
                } else {
                    sync_player_lua_combat_state(
                        &mut self.engine.player,
                        &self.engine.player_state,
                        dt,
                    );
                    let p1_lua = if self.engine.player.dimension.is_nether() {
                        None
                    } else {
                        self.engine.lua_rt.as_ref()
                    };
                    handle_player_death_if_needed(
                        &mut self.engine.player,
                        &self.engine.player_state,
                        p1_lua,
                        keep_inv,
                        &mut self.engine.chat_messages,
                        "You",
                    );
                }
                if self.engine.player.is_dead {
                    self.engine.active_formspec = None;
                    self.engine.chat_typing = None;
                    self.engine.move_input = MoveInput::default();
                    self.engine.digger.reset();
                    if self.cursor_locked {
                        self.set_cursor_locked(false);
                    }
                }

                if self.engine.settings.coop_mode.is_active() {
                    let reg = self.engine.registries.read();
                    for idx in 0..3 {
                        let slot = idx + 1;
                        if !self.engine.peers[idx].joined {
                            continue;
                        }
                        let lbl = self.engine.settings.player_name_for_slot(slot).to_string();
                        let peer = &mut self.engine.peers[idx];
                        let pw = world_for(
                            &self.engine.world,
                            &self.engine.nether_world,
                            peer.player.dimension,
                        );
                        let plua = if peer.player.dimension.is_nether() {
                            None
                        } else {
                            self.engine.lua_rt.as_ref()
                        };
                        if peer.active_formspec.is_none() {
                            step_player_physics_and_respawn(
                                &mut peer.player,
                                &peer.player_state,
                                &peer_inputs[idx],
                                pw,
                                &reg,
                                plua,
                                &self.engine.audio,
                                &self.engine.settings,
                                keep_inv,
                                &mut self.engine.chat_messages,
                                &lbl,
                                dt,
                            );
                            if let Some(new_dim) = tick_player_portal(
                                &mut peer.player,
                                &peer.player_state,
                                &self.engine.world,
                                &self.engine.nether_world,
                                &self.engine.palette,
                                &self.engine.nether_palette,
                                &reg,
                                &mut self.engine.chat_messages,
                                &lbl,
                                dt,
                            ) {
                                let dest_world = world_for(&self.engine.world, &self.engine.nether_world, new_dim);
                                let (cx, cz) = (
                                    (peer.player.pos.x / 16.0).floor() as i32,
                                    (peer.player.pos.z / 16.0).floor() as i32,
                                );
                                let gen_set = if new_dim.is_nether() {
                                    &mut self.engine.nether_generated_columns
                                } else {
                                    &mut self.engine.generated_columns
                                };
                                for dz in -1..=1 {
                                    for dx in -1..=1 {
                                        gen_set.insert((cx + dx, cz + dz));
                                    }
                                }
                                if let Some(ref mut r) = self.renderer {
                                    r.remesh_priority_chunks_now(
                                        dest_world,
                                        &reg,
                                        &self.engine.atlas,
                                        self.engine.settings.smooth_lighting,
                                    );
                                }
                            }
                        } else {
                            sync_player_lua_combat_state(&mut peer.player, &peer.player_state, dt);
                            handle_player_death_if_needed(
                                &mut peer.player,
                                &peer.player_state,
                                plua,
                                keep_inv,
                                &mut self.engine.chat_messages,
                                &lbl,
                            );
                        }
                        if peer.player.is_dead {
                            peer.active_formspec = None;
                            peer.move_input = MoveInput::default();
                            peer.digger.reset();
                        }
                    }
                }
            }

            // Step Lua runtime (globalstep, core.after, ABMs, entities, node timers)
            if self.engine.mode == AppMode::Playing {
                if let Some(ref rt) = self.engine.lua_rt {
                    rt.step(dt as f64);
                    for sev in rt.pop_sounds() {
                        self.engine.audio.play_node_sound(
                            &sev.spec,
                            SoundCategory::Blocks,
                            &self.engine.settings.audio,
                        );
                    }
                    let fallback_pos = self.engine.last_interact_pos;
                    for req in rt.pop_formspecs() {
                        self.apply_lua_formspec_request(req, fallback_pos);
                    }
                }
                let keep_inv = self.engine.settings.gamerules.keep_inventory;
                sync_player_lua_combat_state(&mut self.engine.player, &self.engine.player_state, 0.0);
                if handle_player_death_if_needed(
                    &mut self.engine.player,
                    &self.engine.player_state,
                    self.engine.lua_rt.as_ref(),
                    keep_inv,
                    &mut self.engine.chat_messages,
                    "You",
                ) {
                    self.engine.active_formspec = None;
                    self.engine.chat_typing = None;
                    self.engine.move_input = MoveInput::default();
                    self.engine.digger.reset();
                    self.set_cursor_locked(false);
                }
                for idx in 0..3 {
                    if !self.engine.peers[idx].joined {
                        continue;
                    }
                    let lbl = self.engine.settings.player_name_for_slot(idx + 1).to_string();
                    let peer = &mut self.engine.peers[idx];
                    sync_player_lua_combat_state(&mut peer.player, &peer.player_state, 0.0);
                    if handle_player_death_if_needed(
                        &mut peer.player,
                        &peer.player_state,
                        self.engine.lua_rt.as_ref(),
                        keep_inv,
                        &mut self.engine.chat_messages,
                        &lbl,
                    ) {
                        peer.active_formspec = None;
                        peer.move_input = MoveInput::default();
                        peer.digger.reset();
                    }
                }
                refresh_live_nodemeta_formspec(
                    self.engine.active_formspec.as_mut(),
                    &self.engine.world,
                );
                for peer in &mut self.engine.peers {
                    if peer.joined {
                        refresh_live_nodemeta_formspec(
                            peer.active_formspec.as_mut(),
                            &self.engine.world,
                        );
                    }
                }
                let cur_total = total_inventory_items(&self.engine.player_state);
                if cur_total > self.engine.last_inv_total_count {
                    self.engine.audio.play_pickup(&self.engine.settings.audio);
                }
                self.engine.last_inv_total_count = cur_total;
                for peer in &mut self.engine.peers {
                    if peer.joined {
                        let cur_p = total_inventory_items(&peer.player_state);
                        if cur_p > peer.last_inv_total_count {
                            self.engine.audio.play_pickup(&self.engine.settings.audio);
                        }
                        peer.last_inv_total_count = cur_p;
                    }
                }
            }

            // Raycast, DigController, Right-Click Place/Interact, Combat (Mobs + 4-way PvP + LAN PvP)
            let p1_world_now = world_for(
                &self.engine.world,
                &self.engine.nether_world,
                self.engine.player.dimension,
            );
            let targeted = if self.engine.mode == AppMode::Playing
                && self.engine.active_formspec.is_none()
                && self.engine.chat_typing.is_none()
            {
                let reg = self.engine.registries.read();
                let point_liq = self.engine.settings.liquids_pointable
                    || bot_force_liquids_pointable
                    || held_item_is_empty_bucket(&self.engine.player, &self.engine.player_state);
                raycast_voxel_ex(
                    self.engine.player.eye_pos(),
                    self.engine.player.look_dir(),
                    5.5,
                    point_liq,
                    p1_world_now,
                    &reg,
                )
            } else {
                None
            };

            if self.engine.mode == AppMode::Playing
                && !self.engine.player.is_dead
                && self.engine.active_formspec.is_none()
                && self.engine.chat_typing.is_none()
            {
                let peer_targets_for_p1: Vec<Arc<RwLock<ObjectState>>> = if self
                    .engine
                    .settings
                    .coop_mode
                    .is_active()
                {
                    self.engine
                        .peers
                        .iter()
                        .filter(|p| p.joined && p.player.dimension == self.engine.player.dimension)
                        .map(|p| p.player_state.clone())
                        .collect()
                } else {
                    Vec::new()
                };
                let p1_lua_act = if self.engine.player.dimension.is_nether() {
                    None
                } else {
                    self.engine.lua_rt.as_ref()
                };
                let p1_targeted_entity = try_melee_entity_attack(
                    0,
                    p1_mine_just,
                    p1_mine_held,
                    &mut self.p1_attack_tflp,
                    dt,
                    targeted.as_ref(),
                    &self.engine.player,
                    &self.engine.player_state,
                    &peer_targets_for_p1,
                    &mut self.engine.remote_players,
                    &mut self.engine.mob_manager,
                    &self.engine.registries,
                    p1_lua_act,
                    &self.engine.audio,
                    &self.engine.settings,
                    self.engine.net_session.as_ref(),
                    self.engine.local_server.as_ref(),
                    self.renderer.as_mut(),
                );
                if p1_targeted_entity {
                    self.engine.digger.reset();
                    self.engine.player.add_exhaustion(0.10);
                }
                if self.mmb_just_pressed && self.cursor_locked {
                    self.mmb_just_pressed = false;
                    if let Some(hit) = targeted.as_ref() {
                        let reg = self.engine.registries.read();
                        if pick_block_into_hotbar(
                            &mut self.engine.player,
                            &self.engine.player_state,
                            hit.node_id,
                            &reg,
                        ) {
                            self.engine.last_inv_total_count =
                                total_inventory_items(&self.engine.player_state);
                        }
                    }
                } else {
                    self.mmb_just_pressed = false;
                }
                let p1_opened_chest = run_player_block_actions(
                    0,
                    p1_mine_held && !p1_targeted_entity,
                    p1_place_just,
                    dt,
                    if p1_targeted_entity { None } else { targeted.as_ref() },
                    &mut self.engine.player,
                    &self.engine.player_state,
                    &mut self.engine.digger,
                    &mut self.engine.last_interact_pos,
                    &mut self.engine.last_inv_total_count,
                    p1_world_now,
                    &self.engine.registries,
                    p1_lua_act,
                    &self.engine.audio,
                    &self.engine.atlas,
                    &self.engine.settings,
                    self.engine.net_session.as_ref(),
                    self.engine.local_server.as_ref(),
                    self.renderer.as_mut(),
                );
                if let Some(chest_pos) = p1_opened_chest {
                    self.open_loot_chest_formspec(0, chest_pos);
                }
                let pending_fs = self
                    .engine
                    .lua_rt
                    .as_ref()
                    .map(|rt| rt.pop_formspecs())
                    .unwrap_or_default();
                let fb = self.engine.last_interact_pos;
                for req in pending_fs {
                    self.apply_lua_formspec_request(req, fb);
                }
            } else if let Some(ref mut r) = self.renderer {
                r.set_selection_boxes(&[]);
                r.set_crack_overlay(&[], None, &self.engine.atlas);
                r.viewmodel.set_digging(false);
            }

            // Peer (P2..P4) block interactions, raycast, and combat
            for idx in 0..3 {
                let slot = idx + 1;
                if self.engine.mode == AppMode::Playing
                    && self.engine.settings.coop_mode.is_active()
                    && self.engine.peers[idx].joined
                    && !self.engine.peers[idx].player.is_dead
                    && self.engine.peers[idx].active_formspec.is_none()
                {
                    let peer_dim = self.engine.peers[idx].player.dimension;
                    let peer_world = world_for(&self.engine.world, &self.engine.nether_world, peer_dim);
                    let targeted_peer = {
                        let reg = self.engine.registries.read();
                        let point_liq = self.engine.settings.liquids_pointable
                            || held_item_is_empty_bucket(
                                &self.engine.peers[idx].player,
                                &self.engine.peers[idx].player_state,
                            );
                        raycast_voxel_ex(
                            self.engine.peers[idx].player.eye_pos(),
                            self.engine.peers[idx].player.look_dir(),
                            5.5,
                            point_liq,
                            peer_world,
                            &reg,
                        )
                    };
                    self.engine.peers[idx].targeted = targeted_peer;

                    let mut other_locals = Vec::new();
                    if self.engine.player.dimension == peer_dim {
                        other_locals.push(self.engine.player_state.clone());
                    }
                    for (j, op) in self.engine.peers.iter().enumerate() {
                        if j != idx && op.joined && op.player.dimension == peer_dim {
                            other_locals.push(op.player_state.clone());
                        }
                    }

                    let peer_lua = if peer_dim.is_nether() {
                        None
                    } else {
                        self.engine.lua_rt.as_ref()
                    };
                    let peer = &mut self.engine.peers[idx];
                    let peer_hit_ent = try_melee_entity_attack(
                        slot,
                        peer_mine_just[idx],
                        peer_mine_held[idx],
                        &mut peer.attack_tflp,
                        dt,
                        peer.targeted.as_ref(),
                        &peer.player,
                        &peer.player_state,
                        &other_locals,
                        &mut self.engine.remote_players,
                        &mut self.engine.mob_manager,
                        &self.engine.registries,
                        peer_lua,
                        &self.engine.audio,
                        &self.engine.settings,
                        self.engine.net_session.as_ref(),
                        self.engine.local_server.as_ref(),
                        self.renderer.as_mut(),
                    );
                    if peer_hit_ent {
                        peer.digger.reset();
                        peer.player.add_exhaustion(0.10);
                    }
                    let peer_opened_chest = run_player_block_actions(
                        slot,
                        peer_mine_held[idx] && !peer_hit_ent,
                        peer_place_just[idx],
                        dt,
                        if peer_hit_ent { None } else { peer.targeted.as_ref() },
                        &mut peer.player,
                        &peer.player_state,
                        &mut peer.digger,
                        &mut peer.last_interact_pos,
                        &mut peer.last_inv_total_count,
                        peer_world,
                        &self.engine.registries,
                        peer_lua,
                        &self.engine.audio,
                        &self.engine.atlas,
                        &self.engine.settings,
                        self.engine.net_session.as_ref(),
                        self.engine.local_server.as_ref(),
                        self.renderer.as_mut(),
                    );
                    if let Some(chest_pos) = peer_opened_chest {
                        self.open_loot_chest_formspec(slot, chest_pos);
                    }
                    let pending_fsp = self
                        .engine
                        .lua_rt
                        .as_ref()
                        .map(|rt| rt.pop_formspecs())
                        .unwrap_or_default();
                    let fb_p = self.engine.peers[idx].last_interact_pos;
                    for req in pending_fsp {
                        self.apply_lua_formspec_request(req, fb_p);
                    }
                } else {
                    self.engine.peers[idx].targeted = None;
                    if let Some(ref mut r) = self.renderer {
                        r.set_peer_selection_boxes(slot, &[]);
                        r.set_peer_crack_overlay(slot, &[], None, &self.engine.atlas);
                        if let Some(pslot) = r.peer_slot_mut(slot) {
                            pslot.viewmodel.set_digging(false);
                        }
                    }
                }
            }

            // Step PC-to-Phone Cloud Streaming players (server-side physics, combat, block edits)
            if self.engine.mode == AppMode::Playing {
                if let Some(ref srv) = self.engine.local_server {
                    let hub = srv.stream_hub.clone();
                    let active_clients = hub.list_clients();
                    let active_ids: std::collections::HashSet<u64> =
                        active_clients.iter().map(|(id, _)| *id).collect();
                    self.engine
                        .stream_players
                        .retain(|id, _| active_ids.contains(id));

                    for (id, cname) in active_clients {
                        if !self.engine.stream_players.contains_key(&id) {
                            let sp = self.engine.player.pos + Vec3::new(1.5, 0.2, 1.5);
                            self.engine
                                .stream_players
                                .insert(id, StreamPlayerRuntime::new(id, cname.clone(), sp));
                        } else if let Some(sp) = self.engine.stream_players.get_mut(&id) {
                            sp.name = cname;
                        }
                    }

                    let stream_ids: Vec<u64> =
                        self.engine.stream_players.keys().copied().collect();
                    for sid in stream_ids {
                        let Some(inp) = hub.take_input(sid) else {
                            continue;
                        };
                        let mut other_locals = vec![self.engine.player_state.clone()];
                        for op in &self.engine.peers {
                            if op.joined {
                                other_locals.push(op.player_state.clone());
                            }
                        }
                        for (oid, osp) in &self.engine.stream_players {
                            if *oid != sid {
                                other_locals.push(osp.player_state.clone());
                            }
                        }

                        let Some(sp) = self.engine.stream_players.get_mut(&sid) else {
                            continue;
                        };
                        sp.player.yaw += inp.look_dx;
                        sp.player.pitch = (sp.player.pitch + inp.look_dy).clamp(-1.54, 1.54);
                        if let Some(slot_sel) = inp.selected_slot {
                            let s = slot_sel.min(8);
                            sp.player.selected_slot = s;
                            sp.player_state.write().wield_index = s + 1;
                        }
                        sp.sneaking = inp.sneak;
                        if sp.swing_timer > 0.0 {
                            sp.swing_timer = (sp.swing_timer - dt).max(0.0);
                        }
                        if inp.attack_just {
                            sp.swing_timer = 0.28;
                        }

                        let prev_pos = sp.player.pos;
                        let move_in = MoveInput {
                            forward: inp.move_y > 0.2,
                            backward: inp.move_y < -0.2,
                            left: inp.move_x < -0.2,
                            right: inp.move_x > 0.2,
                            jump: inp.jump,
                            sneak: inp.sneak,
                            sprint: inp.sprint,
                            zoom: false,
                            analog_x: inp.move_x,
                            analog_y: inp.move_y,
                        };
                        let sp_dim = sp.player.dimension;
                        let sp_world = world_for(&self.engine.world, &self.engine.nether_world, sp_dim);
                        let sp_lua = if sp_dim.is_nether() {
                            None
                        } else {
                            self.engine.lua_rt.as_ref()
                        };
                        {
                            let reg = self.engine.registries.read();
                            let keep_inv = self.engine.settings.gamerules.keep_inventory;
                            step_player_physics_and_respawn(
                                &mut sp.player,
                                &sp.player_state,
                                &move_in,
                                sp_world,
                                &reg,
                                sp_lua,
                                &self.engine.audio,
                                &self.engine.settings,
                                keep_inv,
                                &mut self.engine.chat_messages,
                                &sp.name,
                                dt,
                            );
                            let _ = tick_player_portal(
                                &mut sp.player,
                                &sp.player_state,
                                &self.engine.world,
                                &self.engine.nether_world,
                                &self.engine.palette,
                                &self.engine.nether_palette,
                                &reg,
                                &mut self.engine.chat_messages,
                                &sp.name,
                                dt,
                            );
                            if sp.player.is_dead {
                                respawn_player(
                                    &mut sp.player,
                                    &sp.player_state,
                                    &self.engine.world,
                                    &reg,
                                    self.engine.lua_rt.as_ref(),
                                    &mut self.engine.chat_messages,
                                    &sp.name,
                                );
                            }
                        }
                        let disp = Vec3::new(
                            sp.player.pos.x - prev_pos.x,
                            0.0,
                            sp.player.pos.z - prev_pos.z,
                        );
                        sp.horizontal_speed = if dt > 1e-4 { disp.length() / dt } else { 0.0 };

                        let targeted_s = {
                            let reg = self.engine.registries.read();
                            raycast_voxel_ex(
                                sp.player.eye_pos(),
                                sp.player.look_dir(),
                                5.5,
                                self.engine.settings.liquids_pointable,
                                world_for(&self.engine.world, &self.engine.nether_world, sp.player.dimension),
                                &reg,
                            )
                        };
                        let sp_cur_world =
                            world_for(&self.engine.world, &self.engine.nether_world, sp.player.dimension);
                        let sp_cur_lua = if sp.player.dimension.is_nether() {
                            None
                        } else {
                            self.engine.lua_rt.as_ref()
                        };
                        let slot_id = 10 + (sid as usize);
                        let hit_ent = try_melee_entity_attack(
                            slot_id,
                            inp.attack_just,
                            inp.attack_held,
                            &mut sp.attack_tflp,
                            dt,
                            targeted_s.as_ref(),
                            &sp.player,
                            &sp.player_state,
                            &other_locals,
                            &mut self.engine.remote_players,
                            &mut self.engine.mob_manager,
                            &self.engine.registries,
                            sp_cur_lua,
                            &self.engine.audio,
                            &self.engine.settings,
                            self.engine.net_session.as_ref(),
                            self.engine.local_server.as_ref(),
                            self.renderer.as_mut(),
                        );
                        if hit_ent {
                            sp.digger.reset();
                            sp.player.add_exhaustion(0.10);
                        }
                        let _ = run_player_block_actions(
                            slot_id,
                            inp.attack_held && !hit_ent,
                            inp.place_just,
                            dt,
                            if hit_ent { None } else { targeted_s.as_ref() },
                            &mut sp.player,
                            &sp.player_state,
                            &mut sp.digger,
                            &mut sp.last_interact_pos,
                            &mut sp.last_inv_total_count,
                            sp_cur_world,
                            &self.engine.registries,
                            sp_cur_lua,
                            &self.engine.audio,
                            &self.engine.atlas,
                            &self.engine.settings,
                            self.engine.net_session.as_ref(),
                            self.engine.local_server.as_ref(),
                            self.renderer.as_mut(),
                        );
                    }
                }
            }

            // Step Native Nether Mobs & Ocean Monument Guardians + AI + Projectiles + Drops
            if self.engine.mode == AppMode::Playing {
                let mob_players = collect_mob_player_inputs(&self.engine);
                let mob_ev = {
                    let reg = self.engine.registries.read();
                    self.engine.mob_manager.step(
                        dt,
                        &mob_players,
                        &self.engine.world,
                        &self.engine.nether_world,
                        &reg,
                    )
                };
                for dmg in mob_ev.player_damages {
                    let keep_inv = self.engine.settings.gamerules.keep_inventory;
                    if dmg.player_slot == 0 {
                        let p = &mut self.engine.player;
                        if !p.is_dead && p.gamemode == GameMode::Survival && p.spawn_invuln_timer <= 0.0 {
                            p.hp = (p.hp - dmg.damage).max(0);
                            p.vel += dmg.knockback;
                            if dmg.apply_wither_secs > 0.0 {
                                p.apply_wither(dmg.apply_wither_secs);
                            }
                            if p.hp == 0 {
                                p.death_cause = dmg.death_cause.into();
                            }
                            {
                                let mut ps = self.engine.player_state.write();
                                ps.hp = p.hp;
                                ps.hurt_flash = 0.25;
                                if p.hp == 0 {
                                    ps.last_death_cause = dmg.death_cause.into();
                                }
                            }
                            let _ = self.engine.audio.play_named(
                                "player_damage",
                                0.85,
                                1.0,
                                SoundCategory::Hostile,
                                &self.engine.settings.audio,
                            );
                            let lua = if p.dimension.is_nether() {
                                None
                            } else {
                                self.engine.lua_rt.as_ref()
                            };
                            handle_player_death_if_needed(
                                p,
                                &self.engine.player_state,
                                lua,
                                keep_inv,
                                &mut self.engine.chat_messages,
                                "You",
                            );
                        }
                    } else if dmg.player_slot >= 1 && dmg.player_slot <= 3 {
                        let peer = &mut self.engine.peers[dmg.player_slot - 1];
                        if peer.joined
                            && !peer.player.is_dead
                            && peer.player.gamemode == GameMode::Survival
                            && peer.player.spawn_invuln_timer <= 0.0
                        {
                            peer.player.hp = (peer.player.hp - dmg.damage).max(0);
                            peer.player.vel += dmg.knockback;
                            if dmg.apply_wither_secs > 0.0 {
                                peer.player.apply_wither(dmg.apply_wither_secs);
                            }
                            if peer.player.hp == 0 {
                                peer.player.death_cause = dmg.death_cause.into();
                            }
                            {
                                let mut ps = peer.player_state.write();
                                ps.hp = peer.player.hp;
                                ps.hurt_flash = 0.25;
                                if peer.player.hp == 0 {
                                    ps.last_death_cause = dmg.death_cause.into();
                                }
                            }
                            let _ = self.engine.audio.play_named(
                                "player_damage",
                                0.85,
                                1.0,
                                SoundCategory::Hostile,
                                &self.engine.settings.audio,
                            );
                            let lua = if peer.player.dimension.is_nether() {
                                None
                            } else {
                                self.engine.lua_rt.as_ref()
                            };
                            let lbl = self
                                .engine
                                .settings
                                .player_name_for_slot(dmg.player_slot)
                                .to_string();
                            handle_player_death_if_needed(
                                &mut peer.player,
                                &peer.player_state,
                                lua,
                                keep_inv,
                                &mut self.engine.chat_messages,
                                &lbl,
                            );
                        }
                    }
                }
                if !mob_ev.mob_drops.is_empty() {
                    let reg = self.engine.registries.read();
                    for (dim, pos, drops) in mob_ev.mob_drops {
                        if self.engine.player.dimension == dim
                            && !self.engine.player.is_dead
                            && self.engine.player.pos.distance_squared(pos) <= 48.0 * 48.0
                        {
                            if add_drops_to_inventory(&self.engine.player_state, &drops, &reg) {
                                self.engine.audio.play_pickup(&self.engine.settings.audio);
                            }
                        } else if !dim.is_nether() {
                            if let Some(ref rt) = self.engine.lua_rt {
                                let bpos = pos.floor().as_ivec3();
                                for d in drops {
                                    rt.spawn_physical_item_drop(bpos, &d);
                                }
                            }
                        }
                    }
                }
                for (dim, pos, snd) in mob_ev.sounds.into_iter().take(4) {
                    if dim == self.engine.player.dimension {
                        let d = pos.distance(self.engine.player.pos);
                        if d < 42.0 {
                            let gain = (1.0 - d / 42.0).clamp(0.12, 0.85);
                            let _ = self.engine.audio.play_named(
                                snd,
                                gain,
                                1.0,
                                SoundCategory::Hostile,
                                &self.engine.settings.audio,
                            );
                        }
                    }
                }
                if let Some(ref mut r) = self.renderer {
                    for pev in mob_ev.particles {
                        let psys = r.particles_for_mut(pev.dimension);
                        for _ in 0..pev.count {
                            match pev.kind {
                                MobParticleKind::Flame => psys.spawn_torch_ember(pev.pos),
                                MobParticleKind::Smoke => psys.spawn_nether_ash(pev.pos),
                                MobParticleKind::LaserHit => psys.spawn_portal_spark(pev.pos),
                                MobParticleKind::Explosion => {
                                    psys.spawn_explosion_burst(pev.pos, 1);
                                }
                            }
                        }
                    }
                    if !mob_ev.modified_blocks.is_empty() {
                        let reg = self.engine.registries.read();
                        for (dim, bpos) in mob_ev.modified_blocks {
                            let w = world_for(&self.engine.world, &self.engine.nether_world, dim);
                            r.relight_and_remesh_at(
                                w,
                                bpos,
                                &reg,
                                &self.engine.atlas,
                                self.engine.settings.smooth_lighting,
                            );
                        }
                    }
                }
            }

            // Throttled multiplayer PlayerMove sync (~20 Hz) over client session OR host LocalServerHandle
            let held_name = {
                let ps = self.engine.player_state.read();
                let mg = ps.meta.read();
                mg.inventories
                    .get("main")
                    .and_then(|v| v.get(self.engine.player.selected_slot))
                    .filter(|s| !s.is_empty())
                    .map(|s| s.name.clone())
            };

            *self.engine.host_spawn_shared.write() = [
                self.engine.player.pos.x,
                self.engine.player.pos.y,
                self.engine.player.pos.z,
            ];

            if self.engine.last_net_sync.elapsed() >= Duration::from_millis(50)
                && (self.engine.net_session.is_some() || self.engine.local_server.is_some())
            {
                self.engine.last_net_sync = Instant::now();
                let swinging = self
                    .renderer
                    .as_ref()
                    .map(|r| r.viewmodel.is_swinging())
                    .unwrap_or(false);
                let host_state = RemotePlayerState {
                    player_name: self.engine.settings.player_name.clone(),
                    x: self.engine.player.pos.x,
                    y: self.engine.player.pos.y,
                    z: self.engine.player.pos.z,
                    yaw: self.engine.player.yaw,
                    pitch: self.engine.player.pitch,
                    swinging,
                    sneaking: self.engine.player.is_sneaking,
                    sprinting: self.engine.player.is_sprinting,
                    held_item: held_name.clone().unwrap_or_default(),
                };
                if let Some(ref session) = self.engine.net_session {
                    session.send(NetPacket::PlayerMove {
                        player_name: host_state.player_name.clone(),
                        x: host_state.x,
                        y: host_state.y,
                        z: host_state.z,
                        yaw: host_state.yaw,
                        pitch: host_state.pitch,
                        swinging: host_state.swinging,
                        sneaking: host_state.sneaking,
                        sprinting: host_state.sprinting,
                        held_item: host_state.held_item.clone(),
                    });
                }
                if let Some(ref srv) = self.engine.local_server {
                    srv.update_host_player(host_state);
                    for idx in 0..3 {
                        if !self.engine.peers[idx].joined {
                            continue;
                        }
                        let slot = idx + 1;
                        let peer = &self.engine.peers[idx];
                        let h_item = {
                            let ps = peer.player_state.read();
                            let mg = ps.meta.read();
                            mg.inventories
                                .get("main")
                                .and_then(|v| v.get(peer.player.selected_slot))
                                .filter(|s| !s.is_empty())
                                .map(|s| s.name.clone())
                                .unwrap_or_default()
                        };
                        let sw_p = self
                            .renderer
                            .as_ref()
                            .and_then(|r| r.peer_slot(slot))
                            .map(|ps| ps.viewmodel.is_swinging())
                            .unwrap_or(false);
                        srv.update_host_player(RemotePlayerState {
                            player_name: self.engine.settings.player_name_for_slot(slot).to_string(),
                            x: peer.player.pos.x,
                            y: peer.player.pos.y,
                            z: peer.player.pos.z,
                            yaw: peer.player.yaw,
                            pitch: peer.player.pitch,
                            swinging: sw_p,
                            sneaking: peer.player.is_sneaking,
                            sprinting: peer.player.is_sprinting,
                            held_item: h_item,
                        });
                    }
                }
            }

            // Compute per-viewport aspect ratios, camera rigs, peer models & remote LAN player models
            let (surf_w, surf_h) = self
                .renderer
                .as_ref()
                .map(|r| (r.surface_config.width.max(1), r.surface_config.height.max(1)))
                .unwrap_or((1280, 720));
            let coop = self.engine.settings.coop_mode;
            let coop_pcount = self.engine.settings.coop_player_count;
            let mut player_aspects = [surf_w as f32 / surf_h as f32; 4];
            if coop.is_active() {
                let dummy_rect = egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::Vec2::new(surf_w as f32, surf_h as f32),
                );
                for &(slot, px, _) in &Self::compute_window_viewports(
                    coop,
                    coop_pcount,
                    surf_w,
                    surf_h,
                    dummy_rect,
                    0,
                ) {
                    if slot < 4 {
                        player_aspects[slot] = px[2] as f32 / px[3].max(1) as f32;
                    }
                }
                for (sec_i, sw) in self.secondary_windows.iter().enumerate() {
                    if let Some(ref sec_t) = sw.target {
                        let sw_w = sec_t.surface_config.width.max(1);
                        let sw_h = sec_t.surface_config.height.max(1);
                        let sec_dummy = egui::Rect::from_min_size(
                            egui::Pos2::ZERO,
                            egui::Vec2::new(sw_w as f32, sw_h as f32),
                        );
                        for &(slot, px, _) in &Self::compute_window_viewports(
                            coop,
                            coop_pcount,
                            sw_w,
                            sw_h,
                            sec_dummy,
                            sec_i + 1,
                        ) {
                            if slot < 4 {
                                player_aspects[slot] = px[2] as f32 / px[3].max(1) as f32;
                            }
                        }
                    }
                }
            }

            let rig = {
                let reg = self.engine.registries.read();
                let p1_cam_world = world_for(
                    &self.engine.world,
                    &self.engine.nether_world,
                    self.engine.player.dimension,
                );
                let rig = compute_camera_rig(
                    self.camera_mode,
                    &self.engine.player,
                    player_aspects[0],
                    &self.engine.settings,
                    p1_cam_world,
                    &reg,
                );
                let any_peer_joined = coop.is_active() && self.engine.peers.iter().any(|p| p.joined);
                let mut peer_rigs = [None; 3];
                let mut peer_helds: [Option<String>; 3] = [None, None, None];
                if coop.is_active() {
                    for idx in 0..3 {
                        let slot = idx + 1;
                        if !self.engine.peers[idx].joined {
                            continue;
                        }
                        let hp = {
                            let ps = self.engine.peers[idx].player_state.read();
                            let mg = ps.meta.read();
                            mg.inventories
                                .get("main")
                                .and_then(|v| v.get(self.engine.peers[idx].player.selected_slot))
                                .filter(|s| !s.is_empty())
                                .map(|s| s.name.clone())
                        };
                        let peer_cam_world = world_for(
                            &self.engine.world,
                            &self.engine.nether_world,
                            self.engine.peers[idx].player.dimension,
                        );
                        let rp = compute_camera_rig(
                            self.engine.peers[idx].camera_mode,
                            &self.engine.peers[idx].player,
                            player_aspects[slot],
                            &self.engine.settings,
                            peer_cam_world,
                            &reg,
                        );
                        self.engine.peers[idx].cam_pos = rp.eye_pos;
                        self.engine.peers[idx].view_proj = rp.view_proj;
                        self.engine.peers[idx].show_viewmodel = self.engine.mode == AppMode::Playing
                            && self.engine.peers[idx].active_formspec.is_none()
                            && rp.show_viewmodel;
                        peer_rigs[idx] = Some(rp);
                        peer_helds[idx] = hp;
                    }
                }

                let is_playing = self.engine.mode == AppMode::Playing;
                let emit_embers = is_playing && self.engine.settings.extruded_held_items;
                let mut ember_trigger = false;
                if emit_embers {
                    self.torch_ember_timer += dt;
                    if self.torch_ember_timer >= 0.14 {
                        self.torch_ember_timer = 0.0;
                        ember_trigger = true;
                    }
                } else {
                    self.torch_ember_timer = 0.0;
                }
                let mut nether_ambient_trigger = false;
                if is_playing
                    && (self.engine.player.dimension.is_nether()
                        || (any_peer_joined
                            && self
                                .engine
                                .peers
                                .iter()
                                .any(|p| p.joined && p.player.dimension.is_nether())))
                {
                    self.engine.nether_ambient_timer += dt;
                    if self.engine.nether_ambient_timer >= 0.09 {
                        self.engine.nether_ambient_timer = 0.0;
                        nether_ambient_trigger = true;
                    }
                } else {
                    self.engine.nether_ambient_timer = 0.0;
                }

                if let Some(ref mut r) = self.renderer {
                    r.set_extruded_held_items(self.engine.settings.extruded_held_items);
                    r.particles
                        .update_with_world(dt, Some(&self.engine.world), Some(&reg));
                    r.nether_particles
                        .update_with_world(dt, Some(&self.engine.nether_world), Some(&reg));
                    if ember_trigger {
                        let is_lit_hand = |name_opt: Option<&str>| {
                            let s = name_opt.unwrap_or("").to_ascii_lowercase();
                            s.contains("torch") || s.contains("lantern")
                        };
                        if is_lit_hand(held_name.as_deref()) {
                            let fwd = self.engine.player.look_dir();
                            let right = fwd.cross(Vec3::Y).normalize_or_zero();
                            let epos = self.engine.player.eye_pos()
                                + fwd * 0.62
                                + right * 0.28
                                - Vec3::Y * 0.05;
                            r.particles_for_mut(self.engine.player.dimension)
                                .spawn_torch_ember(epos);
                        }
                        if any_peer_joined {
                            for idx in 0..3 {
                                if self.engine.peers[idx].joined && is_lit_hand(peer_helds[idx].as_deref()) {
                                    let p = &self.engine.peers[idx].player;
                                    let fwd = p.look_dir();
                                    let right = fwd.cross(Vec3::Y).normalize_or_zero();
                                    let epos = p.eye_pos()
                                        + fwd * 0.62
                                        + right * 0.28
                                        - Vec3::Y * 0.05;
                                    r.particles_for_mut(p.dimension).spawn_torch_ember(epos);
                                }
                            }
                        }
                    }
                    if nether_ambient_trigger {
                        use rand::Rng;
                        let mut rng = rand::thread_rng();
                        let mut nether_eyes = Vec::new();
                        if self.engine.player.dimension.is_nether() {
                            nether_eyes.push(self.engine.player.eye_pos());
                        }
                        if any_peer_joined {
                            for p in &self.engine.peers {
                                if p.joined && p.player.dimension.is_nether() {
                                    nether_eyes.push(p.player.eye_pos());
                                }
                            }
                        }
                        for center in nether_eyes {
                            for _ in 0..2 {
                                let ppos = center
                                    + Vec3::new(
                                        rng.gen_range(-10.0..10.0),
                                        rng.gen_range(-4.0..7.0),
                                        rng.gen_range(-10.0..10.0),
                                    );
                                let gp = IVec3::new(
                                    ppos.x.floor() as i32,
                                    ppos.y.floor() as i32,
                                    ppos.z.floor() as i32,
                                );
                                if self.engine.nether_world.get_node(gp).0 == CONTENT_AIR {
                                    r.nether_particles.spawn_nether_ash(ppos);
                                }
                            }
                            let epos = center
                                + Vec3::new(
                                    rng.gen_range(-10.0..10.0),
                                    rng.gen_range(-6.0..3.0),
                                    rng.gen_range(-10.0..10.0),
                                );
                            let eg = IVec3::new(
                                epos.x.floor() as i32,
                                epos.y.floor() as i32,
                                epos.z.floor() as i32,
                            );
                            if self.engine.nether_world.get_node(eg).0 == CONTENT_AIR {
                                r.nether_particles.spawn_nether_ember(epos);
                            }
                        }
                    }
                    r.show_viewmodel = self.engine.mode == AppMode::Playing
                        && self.engine.active_formspec.is_none()
                        && rig.show_viewmodel;
                    r.show_player_model =
                        matches!(self.engine.mode, AppMode::Playing | AppMode::Paused)
                            && rig.show_player_model;
                    r.coop_render_peer = any_peer_joined;
                    r.update_viewmodel(
                        held_name.as_deref(),
                        self.engine.player.selected_slot,
                        dt,
                        self.engine.player.horizontal_speed,
                        self.engine.player.on_ground,
                        self.engine.player.is_sprinting,
                        self.engine.player.is_sneaking,
                        &reg,
                        &self.engine.atlas,
                    );
                    if r.show_player_model || any_peer_joined {
                        r.player_hurt_flash = self.engine.player_state.read().hurt_flash;
                        r.update_player_model(
                            self.engine.player.pos,
                            self.engine.player.yaw,
                            self.engine.player.pitch,
                            self.engine.player.horizontal_speed,
                            self.engine.player.is_sneaking,
                            held_name.as_deref(),
                            dt,
                            &reg,
                            &self.engine.atlas,
                        );
                    }
                    for idx in 0..3 {
                        let slot = idx + 1;
                        if let Some(ref peer_rig) = peer_rigs[idx] {
                            let peer = &self.engine.peers[idx];
                            let peer_hurt = peer.player_state.read().hurt_flash;
                            if let Some(pslot) = r.peer_slot_mut(slot) {
                                pslot.show_viewmodel = peer.show_viewmodel;
                                pslot.show_player_model =
                                    matches!(self.engine.mode, AppMode::Playing | AppMode::Paused)
                                        && peer_rig.show_player_model;
                                pslot.hurt_flash = peer_hurt;
                            }
                            r.update_peer_viewmodel(
                                slot,
                                peer_helds[idx].as_deref(),
                                peer.player.selected_slot,
                                dt,
                                peer.player.horizontal_speed,
                                peer.player.on_ground,
                                peer.player.is_sprinting,
                                peer.player.is_sneaking,
                                &reg,
                                &self.engine.atlas,
                            );
                            r.update_peer_player_model(
                                slot,
                                peer.player.pos,
                                peer.player.yaw,
                                peer.player.pitch,
                                peer.player.horizontal_speed,
                                peer.player.is_sneaking,
                                peer_helds[idx].as_deref(),
                                dt,
                                &reg,
                                &self.engine.atlas,
                            );
                        } else if let Some(pslot) = r.peer_slot_mut(slot) {
                            pslot.show_player_model = false;
                            pslot.show_viewmodel = false;
                        }
                    }

                    if matches!(self.engine.mode, AppMode::Playing | AppMode::Paused) {
                        let mut snapshots: Vec<EntityRenderSnapshot> = self
                            .engine
                            .lua_rt
                            .as_ref()
                            .map(|rt| {
                                rt.active_entities()
                                    .into_iter()
                                    .map(|e| {
                                        EntityRenderSnapshot::from_object_state(
                                            &e.read(),
                                            &self.engine.world,
                                        )
                                    })
                                    .collect()
                            })
                            .unwrap_or_default();
                        snapshots.extend(self.engine.mob_manager.render_snapshots_for(
                            DimensionId::Overworld,
                            &self.engine.world,
                        ));
                        r.update_entities(&snapshots, &reg, &self.engine.atlas);
                        let nw_snaps = self.engine.mob_manager.render_snapshots_for(
                            DimensionId::Nether,
                            &self.engine.nether_world,
                        );
                        r.update_nether_entities(&nw_snaps, &reg, &self.engine.atlas);
                        let mob_pls = collect_mob_player_inputs(&self.engine);
                        let beams = self
                            .engine
                            .mob_manager
                            .active_laser_beams(DimensionId::Overworld, &mob_pls);
                        r.update_guardian_lasers(&beams);

                        let mut remote_snaps: Vec<RemotePlayerRenderSnapshot> = self
                            .engine
                            .remote_players
                            .values()
                            .map(|rp| RemotePlayerRenderSnapshot {
                                name: rp.name.clone(),
                                pos: rp.pos,
                                yaw: rp.yaw,
                                pitch: rp.pitch,
                                horizontal_speed: rp.horizontal_speed,
                                is_sneaking: rp.sneaking,
                                is_swinging: rp.swing_timer > 0.0,
                                swing_progress: if rp.swing_timer > 0.0 {
                                    1.0 - (rp.swing_timer / 0.28).clamp(0.0, 1.0)
                                } else {
                                    0.0
                                },
                                held_item: if rp.held_item.is_empty() {
                                    None
                                } else {
                                    Some(rp.held_item.clone())
                                },
                                hurt_flash: rp.hurt_flash,
                            })
                            .collect();
                        for sp in self.engine.stream_players.values() {
                            let st = sp.player_state.read();
                            let idx = st.wield_index.saturating_sub(1).min(8);
                            let held = st
                                .meta
                                .read()
                                .inventories
                                .get("main")
                                .and_then(|list| list.get(idx))
                                .filter(|it| !it.is_empty())
                                .map(|it| it.name.clone());
                            remote_snaps.push(RemotePlayerRenderSnapshot {
                                name: sp.name.clone(),
                                pos: sp.player.pos,
                                yaw: sp.player.yaw,
                                pitch: sp.player.pitch,
                                horizontal_speed: sp.horizontal_speed,
                                is_sneaking: sp.sneaking,
                                is_swinging: sp.swing_timer > 0.0,
                                swing_progress: if sp.swing_timer > 0.0 {
                                    1.0 - (sp.swing_timer / 0.28).clamp(0.0, 1.0)
                                } else {
                                    0.0
                                },
                                held_item: held,
                                hurt_flash: st.hurt_flash,
                            });
                        }
                        r.update_remote_players(&remote_snaps, dt, &reg, &self.engine.atlas);
                    } else {
                        r.update_entities(&[], &reg, &self.engine.atlas);
                        r.update_nether_entities(&[], &reg, &self.engine.atlas);
                        r.update_guardian_lasers(&[]);
                        r.update_remote_players(&[], dt, &reg, &self.engine.atlas);
                    }
                }
                rig
            };
            (rig.eye_pos, rig.view_proj, targeted)
        };

        // Prune distant chunk GPU buffers around all active local players & upload adaptively
        if !matches!(self.engine.mode, AppMode::Splash | AppMode::Loading) {
            if let Some(ref mut r) = self.renderer {
                if self.engine.mode == AppMode::Playing {
                    let mut ow_centers = Vec::new();
                    let mut nw_centers = Vec::new();
                    let push_center = |list: &mut Vec<(i32, i32)>, pos: Vec3| {
                        let pc = (
                            (pos.x / 16.0).floor() as i32,
                            (pos.z / 16.0).floor() as i32,
                        );
                        if !list.contains(&pc) {
                            list.push(pc);
                        }
                    };
                    if self.engine.player.dimension.is_nether() {
                        push_center(&mut nw_centers, self.engine.player.pos);
                    } else {
                        push_center(&mut ow_centers, self.engine.player.pos);
                    }
                    if self.engine.settings.coop_mode.is_active() {
                        for peer in &self.engine.peers {
                            if peer.joined {
                                if peer.player.dimension.is_nether() {
                                    push_center(&mut nw_centers, peer.player.pos);
                                } else {
                                    push_center(&mut ow_centers, peer.player.pos);
                                }
                            }
                        }
                    }
                    for sp in self.engine.stream_players.values() {
                        if sp.player.dimension.is_nether() {
                            push_center(&mut nw_centers, sp.player.pos);
                        } else {
                            push_center(&mut ow_centers, sp.player.pos);
                        }
                    }
                    let r_vis =
                        (self.engine.settings.render_distance_chunks as i32).clamp(2, 32);
                    if self.engine.last_prune_chunks.as_deref() != Some(ow_centers.as_slice()) {
                        self.engine.last_prune_chunks = Some(ow_centers.clone());
                        r.prune_distant_chunks_for_dimension(
                            DimensionId::Overworld,
                            &ow_centers,
                            r_vis + 4,
                        );
                        let chunks_g = self.engine.world.chunks.read();
                        let mut dq = self.engine.world.dirty_queue.write();
                        for &cpos in chunks_g.keys() {
                            if ow_centers.iter().any(|&(cx, cz)| {
                                (cpos.x - cx).abs() <= r_vis && (cpos.z - cz).abs() <= r_vis
                            }) && !r.chunk_meshes.contains_key(&cpos)
                                && !self.engine.meshing_in_flight.contains(&cpos)
                            {
                                dq.insert(cpos);
                            }
                        }
                    }
                    if self.engine.last_nether_prune_chunks.as_deref() != Some(nw_centers.as_slice())
                    {
                        self.engine.last_nether_prune_chunks = Some(nw_centers.clone());
                        let nr_vis = r_vis.min(12);
                        r.prune_distant_chunks_for_dimension(
                            DimensionId::Nether,
                            &nw_centers,
                            nr_vis + 4,
                        );
                        let chunks_g = self.engine.nether_world.chunks.read();
                        let mut dq = self.engine.nether_world.dirty_queue.write();
                        for &cpos in chunks_g.keys() {
                            if nw_centers.iter().any(|&(cx, cz)| {
                                (cpos.x - cx).abs() <= nr_vis && (cpos.z - cz).abs() <= nr_vis
                            }) && !r.nether_chunk_meshes.contains_key(&cpos)
                                && !self.engine.nether_meshing_in_flight.contains(&cpos)
                            {
                                dq.insert(cpos);
                            }
                        }
                    }
                }

                let reg = self.engine.registries.read();
                r.remesh_priority_chunks_now(
                    &self.engine.world,
                    &reg,
                    &self.engine.atlas,
                    self.engine.settings.smooth_lighting,
                );
                r.remesh_priority_chunks_now(
                    &self.engine.nether_world,
                    &reg,
                    &self.engine.atlas,
                    self.engine.settings.smooth_lighting,
                );
                drop(reg);
                let (up_count, up_ms) = if self.engine.pending_meshes.len() > 24 {
                    (24, 2.0)
                } else {
                    (16, 1.2)
                };
                r.upload_meshes_budgeted_for_dimension(
                    &mut self.engine.pending_meshes,
                    up_count,
                    up_ms,
                    DimensionId::Overworld,
                );
                let (nup_count, nup_ms) = if self.engine.nether_pending_meshes.len() > 24 {
                    (24, 2.0)
                } else {
                    (16, 1.2)
                };
                r.upload_meshes_budgeted_for_dimension(
                    &mut self.engine.nether_pending_meshes,
                    nup_count,
                    nup_ms,
                    DimensionId::Nether,
                );
            }
        }

        // Build egui UI frame (Splash, Title, Loading, P1/multi-split P1..P4, Touch HUD, Nametags)
        let splash_info = if self.engine.mode == AppMode::Splash || self.splash_fade > 0.0 {
            Some(self.splash_screen_info())
        } else {
            None
        };
        let cur_splash_fade = self.splash_fade;
        let loading_info = if self.engine.mode == AppMode::Loading {
            Some(self.loading_screen_info())
        } else {
            None
        };
        let slot_ctrl_inputs: [Option<ControllerFormInput>; 4] = [
            self.controller_form_input_for_player(0),
            self.controller_form_input_for_player(1),
            self.controller_form_input_for_player(2),
            self.controller_form_input_for_player(3),
        ];
        let coop_mode_now = self.engine.settings.coop_mode;
        let coop_count_now = self.engine.settings.coop_player_count;
        let p1_badge = if coop_mode_now.is_active() {
            Some("P1")
        } else {
            None
        };

        let (Some(win), Some(es), Some(theme)) =
            (&self.window, &mut self.egui_state, &self.theme)
        else {
            return;
        };
        let mut raw_input = es.take_egui_input(win);
        if !self.pending_touch_egui_events.is_empty() {
            raw_input
                .events
                .extend(self.pending_touch_egui_events.drain(..));
        }

        let mut title_act = TitleAction::None;
        let mut pause_act = PauseAction::None;
        let mut p1_death_act = DeathScreenAction::None;
        let mut peer_death_respawns_primary: Vec<usize> = Vec::new();
        let mut form_submit: Option<(String, Vec<(String, String)>, bool)> = None;
        let mut peer_form_submits: Vec<(usize, String, Vec<(String, String)>, bool)> = Vec::new();

        let chunks_loaded = world_for(
            &self.engine.world,
            &self.engine.nether_world,
            self.engine.player.dimension,
        )
        .chunks
        .read()
        .len();
        let (visible_chunks, draw_calls, adapter_name, elapsed_sec, surf_w, surf_h) = self
            .renderer
            .as_ref()
            .map(|r| {
                (
                    r.last_visible_chunks,
                    r.last_draw_calls,
                    r.adapter_name.clone(),
                    r.start_time.elapsed().as_secs_f32(),
                    r.surface_config.width.max(1),
                    r.surface_config.height.max(1),
                )
            })
            .unwrap_or((0, 0, "GPU".into(), 0.0, 1280, 720));

        let (xp_level, xp_progress, p1_hurt_flash, p1_name) = {
            let ps = self.engine.player_state.read();
            (ps.xp_level, ps.xp_progress, ps.hurt_flash, ps.name.clone())
        };
        let mut all_world_tags: Vec<(String, Vec3, DimensionId)> =
            Vec::with_capacity(4 + self.engine.remote_players.len());
        all_world_tags.push((
            p1_name.clone(),
            self.engine.player.pos,
            self.engine.player.dimension,
        ));
        for (idx, peer) in self.engine.peers.iter().enumerate() {
            if peer.joined {
                all_world_tags.push((
                    self.engine.settings.player_name_for_slot(idx + 1).to_string(),
                    peer.player.pos,
                    peer.player.dimension,
                ));
            }
        }
        for rp in self.engine.remote_players.values() {
            all_world_tags.push((rp.name.clone(), rp.pos, DimensionId::Overworld));
        }

        let mut primary_vp_specs: Vec<(usize, [u32; 4])> = Vec::new();
        let mut new_last_touch_vp = self.last_touch_viewport;
        let output = self.egui_ctx.run(raw_input, |ctx| {
            match self.engine.mode {
                AppMode::Splash => {
                    if let Some(ref info) = splash_info {
                        render_splash_screen(ctx, info, elapsed_sec, 1.0);
                    }
                }
                AppMode::Title => {
                    let servers_file = self.engine.paths.cubex_data_dir.join("servers.json");
                    title_act = render_title_screen_with_atlas(
                        ctx,
                        theme,
                        &mut self.engine.title_state,
                        &mut self.engine.settings,
                        &self.engine.paths.worlds_dir,
                        &servers_file,
                        &self.engine.packs,
                        &self.engine.tunnel,
                        &self.engine.tokio_rt,
                        Some(&self.engine.atlas),
                        elapsed_sec,
                    );
                    if cur_splash_fade > 0.0 {
                        if let Some(ref info) = splash_info {
                            render_splash_screen(ctx, info, elapsed_sec, cur_splash_fade);
                        }
                    }
                }
                AppMode::Loading => {
                    if let Some(ref info) = loading_info {
                        render_loading_screen(ctx, theme, info, elapsed_sec);
                    }
                }
                AppMode::Playing => {
                let reg = self.engine.registries.read();
                let full_screen = ctx.screen_rect();
                let mut vps = Self::compute_window_viewports(
                    coop_mode_now,
                    coop_count_now,
                    surf_w,
                    surf_h,
                    full_screen,
                    0,
                );
                vps.retain(|&(slot, _, _)| {
                    slot == 0 || self.engine.peers.get(slot - 1).is_some_and(|p| p.joined)
                });
                if vps.is_empty() {
                    vps.push((0, [0, 0, surf_w, surf_h], full_screen));
                }

                for &(slot, px, vp_rect) in &vps {
                    primary_vp_specs.push((slot, px));
                    if slot == 0 {
                        new_last_touch_vp = Some(vp_rect);
                        if self.engine.player.is_dead {
                            let ctrl_confirm = slot_ctrl_inputs[0].is_some_and(|c| c.confirm);
                            p1_death_act = render_death_screen(
                                ctx,
                                theme,
                                vp_rect,
                                0,
                                "You",
                                &self.engine.player.death_cause,
                                xp_level,
                                true,
                                ctrl_confirm,
                            );
                        } else {
                            let p1_dim = self.engine.player.dimension;
                            let p1_world =
                                world_for(&self.engine.world, &self.engine.nether_world, p1_dim);
                            let p1_warp_alpha = (self.engine.player.portal_charge * 0.72)
                                .max(self.engine.player.portal_warp_flash * 0.85);
                            let p1_tags: Vec<(String, Vec3)> = all_world_tags
                                .iter()
                                .filter(|(n, _, d)| *n != p1_name && *d == p1_dim)
                                .map(|(n, p, _)| (n.clone(), *p))
                                .collect();
                            render_nametags_in_viewport(ctx, vp_rect, cam_pos, view_proj, &p1_tags);
                            render_hud_in_rect(
                                ctx,
                                vp_rect,
                                &self.engine.player,
                                &self.engine.player_state,
                                &self.engine.settings,
                                &reg,
                                targeted_opt.as_ref(),
                                self.engine.fps,
                                self.engine.frame_ms,
                                chunks_loaded,
                                visible_chunks,
                                draw_calls,
                                &adapter_name,
                                &self.engine.chat_messages,
                                self.engine.chat_typing.as_deref(),
                                HudTelemetry {
                                    atlas: Some(&self.engine.atlas),
                                    icon_cache: Some(&mut self.hud_icons),
                                    world: Some(p1_world),
                                    dig_progress: self.engine.digger.dig_progress,
                                    biome_name: Some(biome_label_for(
                                        p1_dim,
                                        self.engine.player.pos,
                                        p1_world.seed,
                                    )),
                                    targeted_param2: targeted_opt.as_ref().map(|h| h.param2),
                                    armor_points: 0,
                                    xp_level,
                                    xp_progress,
                                    player_badge: p1_badge,
                                    hurt_flash: p1_hurt_flash,
                                    dimension: p1_dim,
                                    portal_warp_alpha: p1_warp_alpha,
                                },
                            );
                            if self.engine.settings.touch_controls
                                && self.engine.active_formspec.is_none()
                                && self.engine.chat_typing.is_none()
                            {
                                render_touch_hud(ctx, vp_rect, &self.touch_controls);
                            }
                            if let Some(ref af) = self.engine.active_formspec {
                                let ctrl_in = slot_ctrl_inputs[0];
                                if af.formname == CREATIVE_INVENTORY_FORMNAME {
                                    let close = render_creative_inventory(
                                        ctx,
                                        theme,
                                        &mut self.p1_creative_state,
                                        &self.engine.player_state,
                                        self.engine.player.selected_slot,
                                        &reg,
                                        &self.engine.atlas,
                                        &mut self.form_icons,
                                        &mut self.inv_cursor,
                                        vp_rect,
                                        101,
                                        ctrl_in,
                                    );
                                    if close {
                                        form_submit = Some((af.formname.clone(), Vec::new(), true));
                                    }
                                } else {
                                    let p1_fs_dim = self.engine.player.dimension;
                                    let p1_fs_world = world_for(
                                        &self.engine.world,
                                        &self.engine.nether_world,
                                        p1_fs_dim,
                                    );
                                    let p1_fs_lua = if p1_fs_dim.is_nether() {
                                        None
                                    } else {
                                        self.engine.lua_rt.as_ref()
                                    };
                                    let (fields, close) = render_formspec_interactive(
                                        ctx,
                                        theme,
                                        &af.formname,
                                        &af.parsed,
                                        &self.engine.player_state,
                                        &mut self.engine.formspec_fields,
                                        FormspecContext {
                                            world: Some(p1_fs_world),
                                            context_pos: af.context_pos,
                                            registries: Some(&reg),
                                            atlas: Some(&self.engine.atlas),
                                            icon_cache: Some(&mut self.form_icons),
                                            cursor: Some(&mut self.inv_cursor),
                                            lua_rt: p1_fs_lua,
                                            viewport: Some(vp_rect),
                                            id_salt: 101,
                                            controller_input: ctrl_in,
                                            controller_slot: ctrl_in.map(|_| &mut self.p1_form_slot),
                                        },
                                    );
                                    if !fields.is_empty() || close {
                                        form_submit = Some((af.formname.clone(), fields, close));
                                    }
                                }
                            }
                        }
                    } else {
                        let peer_idx = slot - 1;
                        let (pxp, pxprog, phurt, peer_name) = {
                            let ps = self.engine.peers[peer_idx].player_state.read();
                            (ps.xp_level, ps.xp_progress, ps.hurt_flash, ps.name.clone())
                        };
                        let peer = &mut self.engine.peers[peer_idx];
                        let badge = match slot {
                            1 => "P2",
                            2 => "P3",
                            _ => "P4",
                        };
                        let ctrl_in = slot_ctrl_inputs[slot];
                        if peer.player.is_dead {
                            let act = render_death_screen(
                                ctx,
                                theme,
                                vp_rect,
                                (slot as u64 + 1) * 100 + 77,
                                &peer_name,
                                &peer.player.death_cause,
                                pxp,
                                false,
                                ctrl_in.is_some_and(|c| c.confirm),
                            );
                            if act == DeathScreenAction::Respawn {
                                peer_death_respawns_primary.push(slot);
                            }
                        } else {
                            let peer_dim = peer.player.dimension;
                            let peer_world =
                                world_for(&self.engine.world, &self.engine.nether_world, peer_dim);
                            let peer_chunks_loaded = peer_world.chunks.read().len();
                            let peer_warp_alpha = (peer.player.portal_charge * 0.72)
                                .max(peer.player.portal_warp_flash * 0.85);
                            let peer_tags: Vec<(String, Vec3)> = all_world_tags
                                .iter()
                                .filter(|(n, _, d)| *n != peer_name && *d == peer_dim)
                                .map(|(n, p, _)| (n.clone(), *p))
                                .collect();
                            render_nametags_in_viewport(
                                ctx,
                                vp_rect,
                                peer.cam_pos,
                                peer.view_proj,
                                &peer_tags,
                            );
                            render_hud_in_rect(
                                ctx,
                                vp_rect,
                                &peer.player,
                                &peer.player_state,
                                &self.engine.settings,
                                &reg,
                                peer.targeted.as_ref(),
                                self.engine.fps,
                                self.engine.frame_ms,
                                peer_chunks_loaded,
                                visible_chunks,
                                draw_calls,
                                &adapter_name,
                                &self.engine.chat_messages,
                                None,
                                HudTelemetry {
                                    atlas: Some(&self.engine.atlas),
                                    icon_cache: Some(&mut self.hud_icons),
                                    world: Some(peer_world),
                                    dig_progress: peer.digger.dig_progress,
                                    biome_name: Some(biome_label_for(
                                        peer_dim,
                                        peer.player.pos,
                                        peer_world.seed,
                                    )),
                                    targeted_param2: peer.targeted.as_ref().map(|h| h.param2),
                                    armor_points: 0,
                                    xp_level: pxp,
                                    xp_progress: pxprog,
                                    player_badge: Some(badge),
                                    hurt_flash: phurt,
                                    dimension: peer_dim,
                                    portal_warp_alpha: peer_warp_alpha,
                                },
                            );
                            if let Some(ref af) = peer.active_formspec {
                                if af.formname == CREATIVE_INVENTORY_FORMNAME {
                                    let close = render_creative_inventory(
                                        ctx,
                                        theme,
                                        &mut peer.creative_state,
                                        &peer.player_state,
                                        peer.player.selected_slot,
                                        &reg,
                                        &self.engine.atlas,
                                        &mut self.form_icons,
                                        &mut peer.inv_cursor,
                                        vp_rect,
                                        (slot as u64 + 1) * 100 + 2,
                                        ctrl_in,
                                    );
                                    if close {
                                        peer_form_submits.push((
                                            slot,
                                            af.formname.clone(),
                                            Vec::new(),
                                            true,
                                        ));
                                    }
                                } else {
                                    let p_fs_dim = peer.player.dimension;
                                    let p_fs_world = world_for(
                                        &self.engine.world,
                                        &self.engine.nether_world,
                                        p_fs_dim,
                                    );
                                    let p_fs_lua = if p_fs_dim.is_nether() {
                                        None
                                    } else {
                                        self.engine.lua_rt.as_ref()
                                    };
                                    let (fields, close) = render_formspec_interactive(
                                        ctx,
                                        theme,
                                        &af.formname,
                                        &af.parsed,
                                        &peer.player_state,
                                        &mut peer.formspec_fields,
                                        FormspecContext {
                                            world: Some(p_fs_world),
                                            context_pos: af.context_pos,
                                            registries: Some(&reg),
                                            atlas: Some(&self.engine.atlas),
                                            icon_cache: Some(&mut self.form_icons),
                                            cursor: Some(&mut peer.inv_cursor),
                                            lua_rt: p_fs_lua,
                                            viewport: Some(vp_rect),
                                            id_salt: (slot as u64 + 1) * 100 + 2,
                                            controller_input: ctrl_in,
                                            controller_slot: ctrl_in
                                                .map(|_| &mut peer.form_slot_idx),
                                        },
                                    );
                                    if !fields.is_empty() || close {
                                        peer_form_submits.push((
                                            slot,
                                            af.formname.clone(),
                                            fields,
                                            close,
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }

                // Draw crisp split divider lines between viewports on Window 1
                if vps.len() > 1 {
                    let div_painter = ctx.layer_painter(egui::LayerId::new(
                        egui::Order::Foreground,
                        egui::Id::new("cubex_split_divider"),
                    ));
                    let stroke = egui::Stroke::new(2.0, egui::Color32::from_rgb(20, 20, 24));
                    if vps.len() == 2 {
                        let r0 = vps[0].2;
                        if (r0.right() - full_screen.right()).abs() < 2.0 {
                            div_painter.line_segment(
                                [
                                    egui::Pos2::new(full_screen.left(), r0.bottom()),
                                    egui::Pos2::new(full_screen.right(), r0.bottom()),
                                ],
                                stroke,
                            );
                        } else {
                            div_painter.line_segment(
                                [
                                    egui::Pos2::new(r0.right(), full_screen.top()),
                                    egui::Pos2::new(r0.right(), full_screen.bottom()),
                                ],
                                stroke,
                            );
                        }
                    } else {
                        let mx = full_screen.center().x;
                        let my = full_screen.center().y;
                        div_painter.line_segment(
                            [
                                egui::Pos2::new(full_screen.left(), my),
                                egui::Pos2::new(full_screen.right(), my),
                            ],
                            stroke,
                        );
                        div_painter.line_segment(
                            [
                                egui::Pos2::new(mx, full_screen.top()),
                                egui::Pos2::new(
                                    mx,
                                    if vps.len() == 3 {
                                        my
                                    } else {
                                        full_screen.bottom()
                                    },
                                ),
                            ],
                            stroke,
                        );
                    }
                }
            }
            AppMode::Paused => {
                let mut cfg_guard = self.engine.config.write();
                pause_act = render_pause_menu(
                    ctx,
                    theme,
                    &mut self.engine.pause_state,
                    &mut self.engine.settings,
                    &mut cfg_guard,
                    self.engine.player.gamemode,
                    &self.engine.tunnel,
                    &self.engine.packs,
                    chunks_loaded,
                );
            }
            }
            if let Some(ref bot) = self.qa_bot {
                let st = bot.hud_status();
                let dim_str = if self.engine.player.dimension.is_nether() {
                    "Nether"
                } else {
                    "Overworld"
                };
                let secs = st.elapsed_secs.max(0.0) as u32;
                let pill_text = format!(
                    "[QA BOT] Stage {}/{} | {} | {:02}:{:02} | HP {}/20 | Food {}/20 | {} | {:.0} FPS  (Esc: Manual)",
                    st.stage_num,
                    st.total_stages,
                    st.goal,
                    secs / 60,
                    secs % 60,
                    self.engine.player.hp,
                    self.engine.player.hunger,
                    dim_str,
                    self.engine.fps
                );
                egui::Area::new(egui::Id::new("cubex_qa_bot_pill"))
                    .fixed_pos(egui::pos2(10.0, 10.0))
                    .order(egui::Order::Foreground)
                    .show(ctx, |ui| {
                        egui::Frame::NONE
                            .fill(egui::Color32::from_rgba_unmultiplied(12, 16, 22, 215))
                            .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(72, 214, 132)))
                            .inner_margin(egui::Margin::symmetric(10, 6))
                            .show(ui, |ui| {
                                ui.label(
                                    egui::RichText::new(pill_text)
                                        .monospace()
                                        .size(13.0)
                                        .strong()
                                        .color(egui::Color32::from_rgb(235, 245, 240)),
                                );
                            });
                    });
            } else if self.pending_qa_playthrough.is_some() {
                let pill_text = format!(
                    "[QA BOT] Booting Survival world (seed {})...  (Esc: Manual)",
                    SURVIVAL_NETHER_SEED
                );
                egui::Area::new(egui::Id::new("cubex_qa_bot_pill"))
                    .fixed_pos(egui::pos2(10.0, 10.0))
                    .order(egui::Order::Foreground)
                    .show(ctx, |ui| {
                        egui::Frame::NONE
                            .fill(egui::Color32::from_rgba_unmultiplied(12, 16, 22, 215))
                            .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(240, 185, 65)))
                            .inner_margin(egui::Margin::symmetric(10, 6))
                            .show(ui, |ui| {
                                ui.label(
                                    egui::RichText::new(pill_text)
                                        .monospace()
                                        .size(13.0)
                                        .strong()
                                        .color(egui::Color32::from_rgb(245, 240, 225)),
                                );
                            });
                    });
            } else if let Some(ref bot) = self.autotest_bot {
                let st = bot.hud_status();
                let dim_str = if self.engine.player.dimension.is_nether() {
                    "Nether"
                } else {
                    "Overworld"
                };
                let pill_text = format!(
                    "[AUTOBOT] Stage {}/5 | {} | HP {}/20 | {} | {:.0} FPS  (Esc: Cancel)",
                    st.stage_num, st.detail, self.engine.player.hp, dim_str, self.engine.fps
                );
                egui::Area::new(egui::Id::new("cubex_autobot_pill"))
                    .fixed_pos(egui::pos2(10.0, 10.0))
                    .order(egui::Order::Foreground)
                    .show(ctx, |ui| {
                        egui::Frame::NONE
                            .fill(egui::Color32::from_rgba_unmultiplied(12, 16, 22, 215))
                            .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(72, 214, 132)))
                            .inner_margin(egui::Margin::symmetric(10, 6))
                            .show(ui, |ui| {
                                ui.label(
                                    egui::RichText::new(pill_text)
                                        .monospace()
                                        .size(13.0)
                                        .strong()
                                        .color(egui::Color32::from_rgb(235, 245, 240)),
                                );
                            });
                    });
            } else if self.pending_autotest_config.is_some() {
                let pill_text = format!(
                    "[AUTOBOT] Booting Survival world (seed {})...  (Esc: Cancel)",
                    SURVIVAL_NETHER_SEED
                );
                egui::Area::new(egui::Id::new("cubex_autobot_pill"))
                    .fixed_pos(egui::pos2(10.0, 10.0))
                    .order(egui::Order::Foreground)
                    .show(ctx, |ui| {
                        egui::Frame::NONE
                            .fill(egui::Color32::from_rgba_unmultiplied(12, 16, 22, 215))
                            .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(240, 185, 65)))
                            .inner_margin(egui::Margin::symmetric(10, 6))
                            .show(ui, |ui| {
                                ui.label(
                                    egui::RichText::new(pill_text)
                                        .monospace()
                                        .size(13.0)
                                        .strong()
                                        .color(egui::Color32::from_rgb(245, 240, 225)),
                                );
                            });
                    });
            }
        });
        self.last_touch_viewport = new_last_touch_vp;

        if self.engine.mode == AppMode::Title && self.splash_fade > 0.0 {
            self.splash_fade = (self.splash_fade - dt / 0.35).max(0.0);
        }

        es.handle_platform_output(win, output.platform_output.clone());

        let tod = self
            .engine
            .lua_rt
            .as_ref()
            .map(|rt| *rt.shared.time_of_day.read())
            .unwrap_or(0.35);
        let ppp = win.scale_factor() as f32;

        let ow_mask = self.active_local_body_mask_in_dimension(DimensionId::Overworld);
        let nw_mask = self.active_local_body_mask_in_dimension(DimensionId::Nether);
        let p1_mask_base = if self.engine.player.dimension.is_nether() {
            nw_mask
        } else {
            ow_mask
        };
        let autotest_capture_shapes =
            if self.pending_autotest_finish.is_some() || self.pending_qa_finish.is_some() {
                Some(output.shapes.clone())
            } else {
                None
            };
        if let Some(ref mut r) = self.renderer {
            if primary_vp_specs.len() > 1 {
                let cams: Vec<ViewportCamera> = primary_vp_specs
                    .iter()
                    .map(|&(slot, px)| {
                        if slot == 0 {
                            let mut mask = p1_mask_base & !(1u8 << 0);
                            if r.show_player_model {
                                mask |= 1u8 << 0;
                            }
                            ViewportCamera {
                                cam_pos,
                                view_proj,
                                viewport_px: px,
                                player_slot: 0,
                                draw_local_player_mask: mask,
                                draw_viewmodel: r.show_viewmodel,
                                dimension: self.engine.player.dimension,
                            }
                        } else {
                            let peer = &self.engine.peers[slot - 1];
                            let peer_dim = peer.player.dimension;
                            let peer_show_self = r
                                .peer_slot(slot)
                                .map(|ps| ps.show_player_model)
                                .unwrap_or(false);
                            let base_m = if peer_dim.is_nether() { nw_mask } else { ow_mask };
                            let mut mask = base_m & !(1u8 << slot);
                            if peer_show_self {
                                mask |= 1u8 << slot;
                            }
                            ViewportCamera {
                                cam_pos: peer.cam_pos,
                                view_proj: peer.view_proj,
                                viewport_px: px,
                                player_slot: slot,
                                draw_local_player_mask: mask,
                                draw_viewmodel: peer.show_viewmodel,
                                dimension: peer_dim,
                            }
                        }
                    })
                    .collect();
                let _ = r.render_viewports_frame(
                    &cams,
                    tod,
                    &self.engine.settings,
                    Some(&self.egui_ctx),
                    Some(output),
                    ppp,
                );
            } else {
                let _ = r.render_frame_in_dimension(
                    cam_pos,
                    view_proj,
                    self.engine.player.dimension,
                    tod,
                    &self.engine.settings,
                    Some(&self.egui_ctx),
                    Some(output),
                    ppp,
                );
            }

            if let Some((out_dir, report)) = self.pending_autotest_finish.take() {
                let _ = std::fs::create_dir_all(&out_dir);
                let mut cap_mask = p1_mask_base & !(1u8 << 0);
                if r.show_player_model {
                    cap_mask |= 1u8 << 0;
                }
                let cap_w = surf_w.clamp(320, 1920);
                let cap_h = surf_h.clamp(240, 1080);
                let cam = ViewportCamera {
                    cam_pos,
                    view_proj,
                    viewport_px: [0, 0, cap_w, cap_h],
                    player_slot: 0,
                    draw_local_player_mask: cap_mask,
                    draw_viewmodel: r.show_viewmodel,
                    dimension: self.engine.player.dimension,
                };
                let cap_egui = autotest_capture_shapes.as_ref().map(|shapes| egui::FullOutput {
                    shapes: shapes.clone(),
                    pixels_per_point: ppp,
                    ..Default::default()
                });
                match r.capture_viewport_with_egui_rgba(
                    cam,
                    cap_w,
                    cap_h,
                    tod,
                    &self.engine.settings,
                    Some(&self.egui_ctx),
                    cap_egui,
                    ppp,
                ) {
                    Ok(img) => {
                        if let Err(e) = img.save(out_dir.join("survival_nether.png")) {
                            eprintln!("[AUTOBOT] Failed to save screenshot: {e}");
                        }
                    }
                    Err(e) => {
                        eprintln!("[AUTOBOT] Failed to capture viewport: {e}");
                    }
                }
                if let Ok(json) = serde_json::to_string_pretty(&report) {
                    if let Err(e) = std::fs::write(out_dir.join("report.json"), json) {
                        eprintln!("[AUTOBOT] Failed to write report.json: {e}");
                    }
                }
                println!(
                    "[AUTOBOT] Complete: passed={} stages={}/5 dim={} hp={} avg_fps={:.1} out={}",
                    report.passed,
                    report.stages_completed,
                    report.final_dimension,
                    report.final_hp,
                    report.avg_fps,
                    out_dir.display()
                );
                self.autotest_bot = None;
                event_loop.exit();
            } else if let Some((out_dir, report)) = self.pending_qa_finish.take() {
                let _ = std::fs::create_dir_all(&out_dir);
                let mut cap_mask = p1_mask_base & !(1u8 << 0);
                if r.show_player_model {
                    cap_mask |= 1u8 << 0;
                }
                let cap_w = surf_w.clamp(320, 1920);
                let cap_h = surf_h.clamp(240, 1080);
                let cam = ViewportCamera {
                    cam_pos,
                    view_proj,
                    viewport_px: [0, 0, cap_w, cap_h],
                    player_slot: 0,
                    draw_local_player_mask: cap_mask,
                    draw_viewmodel: r.show_viewmodel,
                    dimension: self.engine.player.dimension,
                };
                let cap_egui = autotest_capture_shapes.as_ref().map(|shapes| egui::FullOutput {
                    shapes: shapes.clone(),
                    pixels_per_point: ppp,
                    ..Default::default()
                });
                match r.capture_viewport_with_egui_rgba(
                    cam,
                    cap_w,
                    cap_h,
                    tod,
                    &self.engine.settings,
                    Some(&self.egui_ctx),
                    cap_egui,
                    ppp,
                ) {
                    Ok(img) => {
                        if let Err(e) = img.save(out_dir.join("qa_playthrough.png")) {
                            eprintln!("[QA BOT] Failed to save screenshot: {e}");
                        }
                    }
                    Err(e) => {
                        eprintln!("[QA BOT] Failed to capture viewport: {e}");
                    }
                }
                if let Ok(json) = serde_json::to_string_pretty(&report) {
                    if let Err(e) = std::fs::write(out_dir.join("qa_playthrough_report.json"), json) {
                        eprintln!("[QA BOT] Failed to write qa_playthrough_report.json: {e}");
                    }
                }
                println!(
                    "[QA BOT] Complete: passed={} stages={}/{} nether={} mobs={} loot={} monument={} hp={} avg_fps={:.1} out={}",
                    report.passed,
                    report.stages_completed,
                    report.total_stages,
                    report.nether_entered,
                    report.mobs_defeated,
                    report.chest_looted_items,
                    report.ocean_monument_verified,
                    report.final_hp,
                    report.avg_fps,
                    out_dir.display()
                );
                self.qa_bot = None;
                event_loop.exit();
            }

            // Render & stream PC GPU frames (768x432 JPEG @ ~30 FPS) to any connected phone clients
            if self.engine.mode == AppMode::Playing && !self.engine.stream_players.is_empty() {
                if let Some(ref srv) = self.engine.local_server {
                    let hub = srv.stream_hub.clone();
                    let send_hud = self.engine.last_stream_hud.elapsed().as_secs_f32() >= 0.10;
                    if send_hud {
                        self.engine.last_stream_hud = Instant::now();
                        for (&sid, sp) in &self.engine.stream_players {
                            let st = sp.player_state.read();
                            let idx = st.wield_index.saturating_sub(1).min(8);
                            let meta_g = st.meta.read();
                            let main_inv = meta_g.inventories.get("main");
                            let hotbar: Vec<String> = (0..9)
                                .map(|i| {
                                    main_inv
                                        .and_then(|l| l.get(i))
                                        .filter(|it| !it.is_empty())
                                        .map(|it| it.name.clone())
                                        .unwrap_or_default()
                                })
                                .collect();
                            let held_item = hotbar.get(idx).cloned().unwrap_or_default();
                            hub.publish_hud(
                                sid,
                                &cubex_rs::network::web_client::StreamHudTelemetry {
                                    hp: st.hp,
                                    hunger: st.hunger as i32,
                                    x: sp.player.pos.x,
                                    y: sp.player.pos.y,
                                    z: sp.player.pos.z,
                                    selected_slot: idx,
                                    held_item,
                                    hotbar,
                                },
                            );
                        }
                    }

                    if self.engine.last_stream_frame.elapsed().as_secs_f32() >= (1.0 / 30.0) {
                        self.engine.last_stream_frame = Instant::now();
                        let far = (self.engine.settings.render_distance_chunks as f32 * 16.0).max(64.0);
                        let proj = glam::Mat4::perspective_rh(
                            72f32.to_radians(),
                            768.0 / 432.0,
                            0.1,
                            far,
                        );
                        for (&sid, sp) in &self.engine.stream_players {
                            let sp_dim = sp.player.dimension;
                            let sp_mask = if sp_dim.is_nether() { nw_mask } else { ow_mask };
                            let dir = sp.player.look_dir();
                            let eye = sp.player.eye_pos() + dir * 0.28;
                            let vp = proj * glam::Mat4::look_at_rh(eye, eye + dir, glam::Vec3::Y);
                            let cam = ViewportCamera {
                                cam_pos: eye,
                                view_proj: vp,
                                viewport_px: [0, 0, 768, 432],
                                player_slot: 0,
                                draw_local_player_mask: sp_mask,
                                draw_viewmodel: false,
                                dimension: sp_dim,
                            };
                            if let Ok(img) =
                                r.render_stream_rgba(cam, 768, 432, tod, &self.engine.settings)
                            {
                                let hub_bg = hub.clone();
                                self.engine.tokio_rt.spawn_blocking(move || {
                                    let rgb: Vec<u8> = img
                                        .as_raw()
                                        .chunks_exact(4)
                                        .flat_map(|c| [c[0], c[1], c[2]])
                                        .collect();
                                    let mut jpeg = Vec::with_capacity(64 * 1024);
                                    let mut enc =
                                        image::codecs::jpeg::JpegEncoder::new_with_quality(
                                            &mut jpeg, 72,
                                        );
                                    if enc
                                        .encode(
                                            &rgb,
                                            img.width(),
                                            img.height(),
                                            image::ExtendedColorType::Rgb8,
                                        )
                                        .is_ok()
                                    {
                                        hub_bg.publish_frame(sid, jpeg);
                                    }
                                });
                            }
                        }
                    }
                }
            }
        }

        if !matches!(self.boot_state, StartupBootState::Ready) {
            self.step_startup_boot();
        }

        if coop_mode_now.uses_secondary_windows()
            && !matches!(self.engine.mode, AppMode::Splash | AppMode::Title)
        {
            for sec_idx in 0..self.secondary_windows.len() {
                self.render_secondary_window(sec_idx);
            }
        }

        // Handle dispatched formspec actions for P1 & P2..P4
        if let Some((fname, fields, close)) = form_submit {
            let ctx_pos = self.engine.active_formspec.as_ref().and_then(|f| f.context_pos);
            if !fields.is_empty() {
                if let Some(ref rt) = self.engine.lua_rt {
                    rt.send_formspec_fields_at(self.engine.player_state.clone(), &fname, &fields, ctx_pos);
                }
            }
            if close {
                self.close_active_formspec();
            }
        }
        for (slot, fname, fields, close) in peer_form_submits {
            let idx = slot - 1;
            let ctx_pos = self.engine.peers[idx]
                .active_formspec
                .as_ref()
                .and_then(|f| f.context_pos);
            if !fields.is_empty() {
                if let Some(ref rt) = self.engine.lua_rt {
                    rt.send_formspec_fields_at(
                        self.engine.peers[idx].player_state.clone(),
                        &fname,
                        &fields,
                        ctx_pos,
                    );
                }
            }
            if close {
                self.close_peer_formspec(slot);
            }
        }
        match p1_death_act {
            DeathScreenAction::Respawn => {
                {
                    let reg = self.engine.registries.read();
                    respawn_player(
                        &mut self.engine.player,
                        &self.engine.player_state,
                        &self.engine.world,
                        &reg,
                        self.engine.lua_rt.as_ref(),
                        &mut self.engine.chat_messages,
                        "You",
                    );
                }
                self.set_cursor_locked(true);
            }
            DeathScreenAction::MainMenu => {
                pause_act = PauseAction::SaveAndQuit;
            }
            DeathScreenAction::None => {}
        }
        for slot in peer_death_respawns_primary {
            let lbl = self.engine.settings.player_name_for_slot(slot).to_string();
            if let Some(peer) = self.engine.peers.get_mut(slot - 1) {
                let reg = self.engine.registries.read();
                respawn_player(
                    &mut peer.player,
                    &peer.player_state,
                    &self.engine.world,
                    &reg,
                    self.engine.lua_rt.as_ref(),
                    &mut self.engine.chat_messages,
                    &lbl,
                );
            }
        }

        match title_act {
            TitleAction::LaunchSingleplayer(w) => {
                self.engine.pending_remote_spawn = None;
                if self.renderer.is_some() {
                    self.begin_world_load(w);
                }
            }
            TitleAction::JoinServer(host, port) => {
                let pname = self.engine.settings.player_name.clone();
                let res = self
                    .engine
                    .tokio_rt
                    .block_on(NetClientSession::connect(&host, port, &pname));
                match res {
                    Ok((welcome, session)) => {
                        let (seed, wt, sp) = match &welcome {
                            NetPacket::Welcome {
                                seed,
                                motd,
                                world_type,
                                spawn_pos,
                            } => {
                                self.engine
                                    .chat_messages
                                    .push(format!("§aConnected to {host}:{port} — {motd}"));
                                let parsed_wt = match world_type.trim().to_ascii_lowercase().as_str() {
                                    "flat" | "superflat" => WorldGenType::Superflat,
                                    "amplified" => WorldGenType::Amplified,
                                    "skyblock" | "singlenode" => WorldGenType::Skyblock,
                                    _ => WorldGenType::Default,
                                };
                                (*seed, parsed_wt, spawn_pos.map(Vec3::from_array))
                            }
                            other => {
                                self.engine
                                    .chat_messages
                                    .push(format!("Connected to {host}:{port}: {other:?}"));
                                (1337, WorldGenType::Default, None)
                            }
                        };
                        self.engine.stop_lan_server();
                        self.engine.pending_remote_spawn = sp;
                        self.engine.net_session = Some(session);
                        if self.renderer.is_some() {
                            let info = WorldInfo {
                                name: format!("{host}_{port}"),
                                path: self.engine.paths.worlds_dir.join("Cubex_Remote_Cache"),
                                gamemode: "Survival".into(),
                                seed,
                                world_type: wt,
                                ..Default::default()
                            };
                            self.begin_world_load(info);
                        }
                    }
                    Err(e) => {
                        self.engine.title_state.status_msg = format!("Connect failed: {e}");
                    }
                }
            }
            TitleAction::ApplySettings => {
                self.engine.apply_settings_to_lua_and_disk();
                self.engine.sync_skins_and_props(
                    self.renderer.as_mut(),
                    Some(&self.egui_ctx),
                    false,
                );
                if let Some(ref mut r) = self.renderer {
                    r.set_vsync(self.engine.settings.vsync);
                    r.set_anisotropy_quality(self.engine.settings.anisotropy_quality);
                }
            }
            TitleAction::ReloadPack(p) => {
                self.engine.reload_resource_pack(&p, self.renderer.as_mut());
            }
            TitleAction::StartTunnel => {
                self.engine.open_lan_server(DEFAULT_LAN_PORT);
                let _ = self.engine.tunnel.start();
            }
            TitleAction::StopTunnel => self.engine.tunnel.stop(),
            TitleAction::QuitGame => {
                self.engine.save_active_world();
                self.engine.stop_lan_server();
                self.engine.tunnel.stop();
                self.shutdown_gpu_resources();
                event_loop.exit();
            }
            TitleAction::None => {}
        }

        match pause_act {
            PauseAction::Resume => {
                self.engine.mode = AppMode::Playing;
                self.set_cursor_locked(true);
            }
            PauseAction::SaveAndQuit => {
                self.engine.save_active_world();
                if let Some(ref s) = self.engine.net_session {
                    s.send(NetPacket::PlayerLeave {
                        player_name: self.engine.settings.player_name.clone(),
                    });
                }
                self.engine.net_session = None;
                self.engine.stop_lan_server();
                self.engine.remote_players.clear();
                self.engine.pending_remote_spawn = None;
                self.engine.pending_load = None;
                self.engine.panorama_center = Vec3::new(
                    self.engine.player.pos.x,
                    self.engine.player.pos.y.max(SEA_LEVEL as f32 + 4.0),
                    self.engine.player.pos.z,
                );
                self.engine.panorama_fog_dist =
                    (self.engine.settings.render_distance_chunks as f32 * 16.0).min(112.0);
                self.engine.mode = AppMode::Title;
                self.engine.player.is_dead = false;
                self.engine.player.has_dropped_death_items = false;
                self.engine.player.spawn_invuln_timer = 0.0;
                self.engine.player.death_cause.clear();
                self.engine.active_formspec = None;
                self.engine.chat_typing = None;
                self.engine.move_input = MoveInput::default();
                self.engine.digger.reset();
                self.inv_cursor.clear();
                self.touch_controls.reset_all();
                for peer in &mut self.engine.peers {
                    peer.active_formspec = None;
                    peer.joined = false;
                    peer.player.is_dead = false;
                    peer.player.has_dropped_death_items = false;
                    peer.player.spawn_invuln_timer = 0.0;
                    peer.player.death_cause.clear();
                    peer.move_input = MoveInput::default();
                    peer.digger.reset();
                    peer.inv_cursor.clear();
                }
                self.sync_coop_window(event_loop);
                self.engine
                    .title_state
                    .refresh_worlds(&self.engine.paths.worlds_dir);
                self.set_cursor_locked(false);
            }
            PauseAction::TeleportP2ToP1 => {
                if self.engine.settings.coop_mode.is_active() {
                    self.engine.ensure_coop_players_joined();
                    let offsets = [
                        Vec3::new(1.5, 0.2, 0.0),
                        Vec3::new(-1.5, 0.2, 0.0),
                        Vec3::new(0.0, 0.2, 1.5),
                    ];
                    for (idx, peer) in self.engine.peers.iter_mut().enumerate() {
                        if peer.joined {
                            peer.player.pos = self.engine.player.pos + offsets[idx];
                            peer.player.vel = Vec3::ZERO;
                            peer.player_state.write().pos = peer.player.pos;
                        }
                    }
                    self.engine
                        .chat_messages
                        .push("§aTeleported co-op players to Player 1.".into());
                }
            }
            PauseAction::SetGameMode(gm) => {
                self.engine.player.gamemode = gm;
                self.engine.player.flying = gm == GameMode::Spectator;
                if gm != GameMode::Creative
                    && self
                        .engine
                        .active_formspec
                        .as_ref()
                        .is_some_and(|f| f.formname == CREATIVE_INVENTORY_FORMNAME)
                {
                    self.engine.active_formspec = None;
                    self.inv_cursor.clear();
                }
                for peer in &mut self.engine.peers {
                    if peer.joined {
                        peer.player.gamemode = gm;
                        peer.player.flying = gm == GameMode::Spectator;
                        if gm != GameMode::Creative
                            && peer
                                .active_formspec
                                .as_ref()
                                .is_some_and(|f| f.formname == CREATIVE_INVENTORY_FORMNAME)
                        {
                            peer.active_formspec = None;
                            peer.inv_cursor.clear();
                        }
                    }
                }
                *self.engine.world.gamemode.write() = gm.label().to_string();
                if let Some(ref rt) = self.engine.lua_rt {
                    rt.set_gamemode(self.engine.player_state.clone(), gm.as_str());
                    for peer in &self.engine.peers {
                        if peer.joined {
                            rt.set_gamemode(peer.player_state.clone(), gm.as_str());
                        }
                    }
                }
            }
            PauseAction::SetTime(t) => {
                if let Some(ref rt) = self.engine.lua_rt {
                    *rt.shared.time_of_day.write() = t;
                }
            }
            PauseAction::ReloadResourcePack(p) => {
                self.engine.reload_resource_pack(&p, self.renderer.as_mut());
            }
            PauseAction::ApplySettings => {
                if let Some(weather) = self.engine.pause_state.take_weather_change() {
                    self.engine
                        .chat_messages
                        .push(format!("§bWeather changed to {weather}"));
                }
                self.engine.apply_settings_to_lua_and_disk();
                self.engine.ensure_coop_players_joined();
                self.engine.sync_skins_and_props(
                    self.renderer.as_mut(),
                    Some(&self.egui_ctx),
                    false,
                );
                self.sync_coop_window(event_loop);
                if let Some(ref mut r) = self.renderer {
                    r.set_vsync(self.engine.settings.vsync);
                    r.set_anisotropy_quality(self.engine.settings.anisotropy_quality);
                    for sw in &mut self.secondary_windows {
                        if let Some(ref mut sec) = sw.target {
                            r.set_secondary_vsync(sec, self.engine.settings.vsync);
                        }
                    }
                }
                if self.last_smooth_lighting != self.engine.settings.smooth_lighting {
                    self.last_smooth_lighting = self.engine.settings.smooth_lighting;
                    let keys: Vec<IVec3> = self.engine.world.chunks.read().keys().copied().collect();
                    self.engine.world.dirty_queue.write().extend(keys);
                    let nkeys: Vec<IVec3> =
                        self.engine.nether_world.chunks.read().keys().copied().collect();
                    self.engine.nether_world.dirty_queue.write().extend(nkeys);
                }
            }
            PauseAction::OpenToLan => {
                self.engine.open_lan_server(DEFAULT_LAN_PORT);
            }
            PauseAction::StopLan => {
                self.engine.stop_lan_server();
            }
            PauseAction::StartTunnel => {
                self.engine.open_lan_server(DEFAULT_LAN_PORT);
                let _ = self.engine.tunnel.start();
            }
            PauseAction::StopTunnel => self.engine.tunnel.stop(),
            PauseAction::None => {}
        }

        // Hybrid sleep + spin-wait frame cap if user set max_fps > 0
        if self.engine.settings.max_fps > 0 {
            let target = Duration::from_secs_f64(1.0 / self.engine.settings.max_fps as f64);
            let spent = now.elapsed();
            if spent < target {
                let remaining = target - spent;
                if remaining > Duration::from_millis(2) {
                    std::thread::sleep(remaining - Duration::from_millis(2));
                }
                while now.elapsed() < target {
                    std::hint::spin_loop();
                }
            }
        }
    }
}

fn run_smoke_test() -> anyhow::Result<()> {
    let paths = EnginePaths::resolve();
    let mut engine = CubexEngine::boot_base(paths)?;
    let lua = engine.lua_rt.as_ref().unwrap();
    println!(
        "[SMOKE] Mineclonia loaded: {} mods, {} errors | {} nodes | {} crafts | {} atlas layers | {} sounds",
        lua.load_report.mods_loaded,
        lua.load_report.errors.len(),
        engine.registries.read().node_count(),
        engine.registries.read().crafts.len(),
        engine.atlas.layers.len(),
        engine.audio.sound_count(),
    );
    assert_eq!(lua.load_report.errors.len(), 0);

    // Exercise ESC menu live settings synchronization
    engine.settings.fov = 95.0;
    engine.settings.render_distance_chunks = 14;
    engine.settings.smooth_lighting = true;
    engine.settings.graphics = GraphicsQuality::Fancy;
    engine.settings.handheld_torch_light = true;
    engine.apply_settings_to_lua_and_disk();
    assert_eq!(engine.config.read().get("fov"), Some("95"));
    assert_eq!(engine.config.read().get("viewing_range"), Some("224"));

    // Initialize headless wgpu Vulkan renderer on GTX 745 & upload meshed chunks
    let mut renderer = WgpuRenderer::new_headless(1280, 720, &engine.atlas)?;
    println!("[SMOKE] wgpu adapter: {}", renderer.adapter_name);

    let keys: Vec<IVec3> = engine.world.chunks.read().keys().copied().collect();
    let snaps: Vec<_> = keys
        .into_iter()
        .filter_map(|c| engine.world.snapshot_chunk(c))
        .collect();
    let reg = engine.registries.read();
    let mut meshes: VecDeque<_> =
        mesh_chunks_parallel_with_atlas(snaps, &reg, &engine.atlas, true).into();
    let total_meshes = meshes.len();
    renderer.upload_meshes_budgeted(&mut meshes, total_meshes, 1000.0);
    renderer.show_viewmodel = true;
    renderer.update_viewmodel(
        Some("mcl_torches:torch"),
        0,
        0.016,
        3.5,
        true,
        false,
        false,
        &reg,
        &engine.atlas,
    );
    drop(reg);

    let proj = glam::Mat4::perspective_rh(95.0f32.to_radians(), 1280.0 / 720.0, 0.1, 250.0);
    let spawn_p = find_spawn_pos(engine.world.seed, *engine.world.world_type.read());
    let eye = Vec3::new(
        spawn_p.x as f32,
        spawn_p.y.max(SEA_LEVEL) as f32 + 8.0,
        spawn_p.z as f32,
    );

    // Warmup 5 frames
    for i in 0..5 {
        let ang = i as f32 * 0.05;
        let view = glam::Mat4::look_at_rh(
            eye,
            eye + Vec3::new(ang.cos(), -0.25, ang.sin()),
            Vec3::Y,
        );
        renderer.render_frame(eye, proj * view, 0.35, &engine.settings, None, None, 1.0)?;
    }

    // Benchmark 120 GPU frames on offscreen 1280x720 Vulkan target
    let t0 = Instant::now();
    for i in 0..120 {
        let ang = i as f32 * 0.03;
        let view = glam::Mat4::look_at_rh(
            eye,
            eye + Vec3::new(ang.cos(), -0.25, ang.sin()),
            Vec3::Y,
        );
        renderer.render_frame(eye, proj * view, 0.35, &engine.settings, None, None, 1.0)?;
    }
    renderer.device.poll(wgpu::Maintain::Wait);
    let elapsed = t0.elapsed().as_secs_f64();
    let fps = 120.0 / elapsed;
    let avg_ms = (elapsed * 1000.0) / 120.0;
    println!(
        "[SMOKE] Rendered 120 frames across {} chunks ({} visible, {} draw calls) in {:.3}s => {:.1} FPS ({:.2} ms/frame)",
        total_meshes,
        renderer.last_visible_chunks,
        renderer.last_draw_calls,
        elapsed,
        fps,
        avg_ms
    );
    assert!(
        fps > 120.0,
        "Expected >120 FPS on GTX 745 Vulkan, got {fps:.1} FPS"
    );
    Ok(())
}

fn resolve_first_registered_item(reg: &EngineRegistries, candidates: &[&str]) -> Option<String> {
    for &c in candidates {
        let resolved = reg.resolve_alias(c);
        if reg.get_item_def(resolved).is_some()
            || reg.get_node_def(reg.get_content_id(resolved)).is_some_and(|d| d.id != CONTENT_UNKNOWN)
        {
            return Some(resolved.to_string());
        }
    }
    None
}

fn run_playtest() -> anyhow::Result<()> {
    let out_dir = std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join("playtest_out");
    std::fs::create_dir_all(&out_dir)?;

    let paths = EnginePaths::resolve();
    let mut engine = CubexEngine::boot_base(paths.clone())?;
    let mut renderer = WgpuRenderer::new_headless(1280, 720, &engine.atlas)?;
    let egui_ctx = egui::Context::default();
    let theme = McTheme::load(&egui_ctx);
    let mut hud_icons = ItemIconCache::new();

    let playtest_world_dir = paths.worlds_dir.join(".playtest_bot_world");
    let _ = std::fs::remove_dir_all(&playtest_world_dir);
    let info = WorldInfo {
        name: ".playtest_bot_world".into(),
        path: playtest_world_dir.clone(),
        gamemode: "Survival".into(),
        seed: 4242,
        world_type: WorldGenType::Default,
        ..Default::default()
    };
    engine.enter_world(info, &mut renderer);

    // Ensure a flat grass clearing around spawn + place real Mineclonia oak trees directly ahead
    let spawn_p = find_spawn_pos(engine.world.seed, WorldGenType::Default);
    let cx_base = spawn_p.x;
    let cz_base = spawn_p.z;
    let ground_y = terrain_height_for(cx_base, cz_base + 5, engine.world.seed, WorldGenType::Default)
        .max(SEA_LEVEL + 2);
    {
        let grass_id = engine.palette.grass;
        let dirt_id = engine.palette.dirt;
        for z in (cz_base - 4)..=(cz_base + 14) {
            for x in (cx_base - 7)..=(cx_base + 7) {
                for dy in 1..=3 {
                    engine.world.set_node(IVec3::new(x, ground_y - dy, z), dirt_id, 0, 0);
                }
                engine.world.set_node(IVec3::new(x, ground_y, z), grass_id, 15, 0);
                for y in (ground_y + 1)..=(ground_y + 16) {
                    engine.world.set_node(IVec3::new(x, y, z), CONTENT_AIR, 15, 0);
                }
            }
        }
        let oak_path = paths
            .game_dir
            .join("mods/ITEMS/mcl_core/schematics/mcl_core_oak_classic.mts");
        let oak = Schematic::load_from_file(&oak_path)?;
        let reg = engine.registries.read();
        let flags = PlaceFlags {
            place_center_x: true,
            place_center_y: false,
            place_center_z: true,
            force_placement: true,
        };
        oak.place_on_world(
            &engine.world,
            &reg,
            IVec3::new(cx_base, ground_y + 1, cz_base + 5),
            SchematicRotation::Rot0,
            flags,
            None,
            101,
        );
        oak.place_on_world(
            &engine.world,
            &reg,
            IVec3::new(cx_base - 4, ground_y + 1, cz_base + 9),
            SchematicRotation::Rot90,
            flags,
            None,
            202,
        );
        oak.place_on_world(
            &engine.world,
            &reg,
            IVec3::new(cx_base + 5, ground_y + 1, cz_base + 10),
            SchematicRotation::Rot180,
            flags,
            None,
            303,
        );
    }

    // Relight across all active chunks and synchronously upload every initial chunk mesh
    {
        let reg = engine.registries.read();
        let keys: Vec<IVec3> = engine.world.chunks.read().keys().copied().collect();
        propagate_world_light(&engine.world, &keys, &reg);
        let _ = relight_around_pos(&engine.world, IVec3::new(cx_base, ground_y + 3, cz_base + 5), &reg);
        let snaps: Vec<_> = keys
            .into_iter()
            .filter_map(|c| engine.world.snapshot_chunk(c))
            .collect();
        let meshes =
            mesh_chunks_parallel_with_atlas(snaps, &reg, &engine.atlas, engine.settings.smooth_lighting);
        for mesh in meshes {
            renderer.upload_chunk_mesh_now(mesh);
        }
    }

    // Verify Survival starts with a completely empty inventory before equipping playtest kit
    {
        let ps = engine.player_state.read();
        let mg = ps.meta.read();
        let inv = mg
            .inventories
            .get("main")
            .expect("Survival player must have main inventory");
        assert!(
            inv.iter().all(|s| s.is_empty()),
            "Survival starter inventory must start empty"
        );
    }

    // Equip hotbar with axe (slot 0), placeable block (slot 1), and torch (slot 2)
    let (axe_name, place_block_name) = {
        let reg = engine.registries.read();
        let axe = resolve_first_registered_item(
            &reg,
            &[
                "mcl_tools:axe_stone",
                "mcl_tools:axe_wood",
                "mcl_tools:axe_iron",
                "mcl_tools:pick_stone",
            ],
        )
        .unwrap_or_else(|| "mcl_tools:axe_stone".into());
        let blk = resolve_first_registered_item(
            &reg,
            &[
                "mcl_core:cobble",
                "mcl_trees:wood_oak",
                "mcl_core:stone",
                "mcl_core:dirt",
            ],
        )
        .unwrap_or_else(|| "mcl_core:cobble".into());
        (axe, blk)
    };
    {
        let ps = engine.player_state.read();
        let mut mg = ps.meta.write();
        if let Some(inv) = mg.inventories.get_mut("main") {
            inv.fill(ItemStackData::default());
            inv[0] = ItemStackData::from_spec(&format!("{axe_name} 1"));
            inv[1] = ItemStackData::from_spec(&format!("{place_block_name} 16"));
            inv[2] = ItemStackData::from_spec("mcl_torches:torch 8");
        }
    }

    // Position player facing the primary oak tree, starting slightly offset left so raw mouse look rotates toward it
    engine.player.pos = Vec3::new(cx_base as f32 + 0.5, (ground_y + 1) as f32, cz_base as f32 + 1.4);
    engine.player.vel = Vec3::ZERO;
    engine.player.on_ground = true;
    engine.player.selected_slot = 0;
    engine.player_state.write().wield_index = 1;
    engine.player.yaw = -0.30; // Rotate smoothly right toward yaw=0.0 (+Z)
    engine.player.pitch = 0.08; // Slight upward look toward trunk + canopy
    if let Some(ref rt) = engine.lua_rt {
        *rt.shared.time_of_day.write() = 0.42; // Bright midday sun
    }

    // Simulated cursor/focus state matching WindowedApp rules
    let mut window_focused = true;
    let mut cursor_locked = true;
    let mut camera_mode = CameraMode::FirstPerson;
    let mut saved_01 = false;
    let mut saved_02 = false;
    let mut saved_03 = false;
    let mut saved_04 = false;
    let mut third_person_ok = false;
    let mut picked_up_drop = false;
    let mut mined_pos: Option<IVec3> = None;
    let mut mined_node_name = String::new();
    let mut placed_pos: Option<IVec3> = None;
    let mut placed_ok = false;
    let mut esc_cursor_released = false;
    let mut focus_loss_cursor_released = false;

    let dt = 1.0f32 / 60.0;
    let target_yaw = 0.0f32;
    let target_pitch = 0.04f32;
    let sens = engine.settings.mouse_sensitivity * 0.0045;
    let sign = if engine.settings.invert_mouse { 1.0 } else { -1.0 };

    let total_frames = 240usize; // 4.0 seconds @ 60 Hz sim
    let t_start = Instant::now();

    for frame in 0..total_frames {
        let elapsed_sim = frame as f32 * dt;

        // Stage transitions across the 4-second scripted sequence:
        //   0..60   (1.0s): Smooth raw-delta mouse rotation onto oak trunk + capture 01_world_trees.png
        //  60..170  (1.8s): Hold LMB to mine trunk, capture 02_mining_swing.png mid-swing w/ cracks + chips
        // 170..180  (0.17s): Switch slot & place block cleanly with player-AABB overlap protection
        // 180..195  (0.25s): Toggle F5 ThirdPersonBack camera & capture 04_third_person.png
        // 195..240  (0.75s): Open ESC Pause Menu (verify cursor release + capture 03_esc_menu.png) & test Focused(false)
        if frame < 50 && engine.mode == AppMode::Playing && cursor_locked && window_focused {
            let rem_yaw = target_yaw - engine.player.yaw;
            let rem_pitch = target_pitch - engine.player.pitch;
            let dx = (rem_yaw * 0.14 / sens) as f64;
            let dy = (rem_pitch * 0.14 / (sens * sign)) as f64;
            engine.player.apply_mouse_look(dx, dy, &engine.settings);
            renderer.viewmodel.add_look_inertia(dx as f32, dy as f32);
        }

        if frame == 170 {
            engine.player.selected_slot = 1;
            engine.player_state.write().wield_index = 2;
        }
        if frame == 180 {
            camera_mode = CameraMode::ThirdPersonBack;
            engine.player.selected_slot = 0;
            engine.player_state.write().wield_index = 1;
        }
        if frame == 195 {
            camera_mode = CameraMode::FirstPerson;
            engine.mode = AppMode::Paused;
            engine.pause_state.page = PausePage::Main;
            cursor_locked = false;
            esc_cursor_released = !cursor_locked;
        }
        if frame == 235 {
            // Re-enter Playing with lock, then fire Focused(false) to verify dual-monitor cursor release
            engine.mode = AppMode::Playing;
            cursor_locked = window_focused;
            assert!(cursor_locked);
            window_focused = false;
            let still_allowed =
                engine.mode == AppMode::Playing && window_focused && engine.active_formspec.is_none();
            if !still_allowed {
                cursor_locked = false;
            }
            focus_loss_cursor_released = !window_focused && !cursor_locked && !still_allowed;
            // Restore Paused for clean idle finish
            engine.mode = AppMode::Paused;
        }

        if engine.mode == AppMode::Playing {
            let reg = engine.registries.read();
            let bot_input = if (180..=186).contains(&frame) {
                MoveInput {
                    backward: true,
                    ..Default::default()
                }
            } else {
                MoveInput::default()
            };
            let _ = engine
                .player
                .step(dt, &bot_input, &engine.world, &reg, &engine.settings);
            {
                let mut ps = engine.player_state.write();
                ps.pos = engine.player.pos;
                ps.yaw = engine.player.yaw;
                ps.pitch = engine.player.pitch;
            }

            let targeted = raycast_voxel_ex(
                engine.player.eye_pos(),
                engine.player.look_dir(),
                5.5,
                engine.settings.liquids_pointable,
                &engine.world,
                &reg,
            );
            let sel_boxes = targeted
                .as_ref()
                .map(|h| selected_world_boxes(h, &reg))
                .unwrap_or_default();
            renderer.set_selection_boxes(&sel_boxes);

            let holding_lmb = (60..170).contains(&frame) && mined_pos.is_none();
            let dig_res = engine.digger.update(
                dt,
                holding_lmb,
                targeted.as_ref(),
                &engine.world,
                &reg,
                &engine.player,
                &engine.player_state,
                engine.lua_rt.as_ref(),
            );
            renderer.set_crack_overlay(&sel_boxes, dig_res.crack_stage, &engine.atlas);
            if dig_res.swing_triggered {
                renderer.viewmodel.trigger_swing();
            }
            renderer.viewmodel.set_digging(dig_res.is_digging);
            if dig_res.spawn_dig_particles {
                if let Some(ref hit) = targeted {
                    let (_, p1, _) = engine.world.get_node(hit.under);
                    renderer.particles.spawn_dig_chips(
                        hit.under,
                        hit.node_id,
                        8,
                        &engine.atlas,
                        &reg,
                        p1 & 0x0F,
                        p1 >> 4,
                    );
                }
            }
            if let Some(ref comp) = dig_res.completed_dig {
                mined_pos = Some(comp.pos);
                mined_node_name = comp.node_name.clone();
                if comp.picked_up_any {
                    picked_up_drop = true;
                }
                renderer.particles.spawn_break_burst(
                    comp.pos,
                    comp.node_id,
                    &engine.atlas,
                    &reg,
                );
                renderer.relight_and_remesh_at(
                    &engine.world,
                    comp.pos,
                    &reg,
                    &engine.atlas,
                    engine.settings.smooth_lighting,
                );
            }

            if frame == 175 {
                // Tilt down slightly onto the stump base so target is nearby & outside player AABB
                engine.player.pitch = -0.18;
                let place_hit = raycast_voxel_ex(
                    engine.player.eye_pos(),
                    engine.player.look_dir(),
                    5.5,
                    engine.settings.liquids_pointable,
                    &engine.world,
                    &reg,
                );
                if let Some(ref ph) = place_hit {
                    let pr = try_interact_or_place(
                        ph,
                        &engine.world,
                        &reg,
                        &engine.player,
                        &engine.player_state,
                        engine.lua_rt.as_ref(),
                    );
                    if let Some(pp) = pr.placed_pos {
                        placed_pos = Some(pp);
                        placed_ok = engine.world.get_node(pp).0 != CONTENT_AIR;
                        renderer.viewmodel.trigger_swing();
                        renderer.relight_and_remesh_at(
                            &engine.world,
                            pp,
                            &reg,
                            &engine.atlas,
                            engine.settings.smooth_lighting,
                        );
                    }
                }
            }

            let held_name = {
                let ps = engine.player_state.read();
                let mg = ps.meta.read();
                mg.inventories
                    .get("main")
                    .and_then(|v| v.get(engine.player.selected_slot))
                    .filter(|s| !s.is_empty())
                    .map(|s| s.name.clone())
            };
            renderer.particles.update(dt);
            let rig = compute_camera_rig(
                camera_mode,
                &engine.player,
                1280.0 / 720.0,
                &engine.settings,
                &engine.world,
                &reg,
            );
            renderer.show_viewmodel = rig.show_viewmodel;
            renderer.show_player_model = rig.show_player_model;
            renderer.update_viewmodel(
                held_name.as_deref(),
                engine.player.selected_slot,
                dt,
                engine.player.horizontal_speed,
                engine.player.on_ground,
                engine.player.is_sprinting,
                engine.player.is_sneaking,
                &reg,
                &engine.atlas,
            );
            renderer.update_player_model(
                engine.player.pos,
                engine.player.yaw,
                engine.player.pitch,
                engine.player.horizontal_speed,
                engine.player.is_sneaking,
                held_name.as_deref(),
                dt,
                &reg,
                &engine.atlas,
            );
            if frame == 188 {
                third_person_ok = rig.show_player_model && !rig.show_viewmodel && rig.effective_distance > 1.0;
            }
            drop(reg);
            if let Some(ref rt) = engine.lua_rt {
                rt.step(dt as f64);
            }
        } else {
            renderer.set_selection_boxes(&[]);
            renderer.set_crack_overlay(&[], None, &engine.atlas);
            renderer.show_viewmodel = true;
            renderer.show_player_model = false;
        }

        // Synchronous priority remesh check each frame
        {
            let reg = engine.registries.read();
            renderer.remesh_priority_chunks_now(
                &engine.world,
                &reg,
                &engine.atlas,
                engine.settings.smooth_lighting,
            );
        }

        let targeted_hud = if engine.mode == AppMode::Playing {
            let reg = engine.registries.read();
            raycast_voxel_ex(
                engine.player.eye_pos(),
                engine.player.look_dir(),
                5.5,
                engine.settings.liquids_pointable,
                &engine.world,
                &reg,
            )
        } else {
            None
        };

        let mut raw_input = egui::RawInput::default();
        raw_input.screen_rect = Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 720.0),
        ));
        raw_input.time = Some(elapsed_sim as f64);

        let chunks_loaded = engine.world.chunks.read().len();
        let vis_chunks = renderer.last_visible_chunks;
        let draws = renderer.last_draw_calls;
        let adapter_name = renderer.adapter_name.clone();
        let sim_fps = 1.0 / dt;

        let output = egui_ctx.run(raw_input, |ctx| match engine.mode {
            AppMode::Playing | AppMode::Title => {
                let reg = engine.registries.read();
                render_hud_full(
                    ctx,
                    &engine.player,
                    &engine.player_state,
                    &engine.settings,
                    &reg,
                    targeted_hud.as_ref(),
                    sim_fps,
                    dt * 1000.0,
                    chunks_loaded,
                    vis_chunks,
                    draws,
                    &adapter_name,
                    &engine.chat_messages,
                    None,
                    HudTelemetry {
                        atlas: Some(&engine.atlas),
                        icon_cache: Some(&mut hud_icons),
                        world: Some(&engine.world),
                        dig_progress: engine.digger.dig_progress,
                        biome_name: Some(biome_name_at(
                            engine.player.pos.x.floor() as i32,
                            engine.player.pos.z.floor() as i32,
                            engine.world.seed,
                        )),
                        targeted_param2: targeted_hud.as_ref().map(|h| h.param2),
                        armor_points: 0,
                        xp_level: 1,
                        xp_progress: 0.35,
                        player_badge: None,
                        hurt_flash: 0.0,
                        dimension: engine.player.dimension,
                        portal_warp_alpha: 0.0,
                    },
                );
            }
            AppMode::Paused => {
                let mut cfg_guard = engine.config.write();
                let _ = render_pause_menu(
                    ctx,
                    &theme,
                    &mut engine.pause_state,
                    &mut engine.settings,
                    &mut cfg_guard,
                    engine.player.gamemode,
                    &engine.tunnel,
                    &engine.packs,
                    chunks_loaded,
                );
            }
            AppMode::Splash | AppMode::Loading => {}
        });

        let (cam_pos, vp) = {
            let reg = engine.registries.read();
            let rig = compute_camera_rig(
                camera_mode,
                &engine.player,
                1280.0 / 720.0,
                &engine.settings,
                &engine.world,
                &reg,
            );
            (rig.eye_pos, rig.view_proj)
        };
        let tod = engine
            .lua_rt
            .as_ref()
            .map(|rt| *rt.shared.time_of_day.read())
            .unwrap_or(0.42);

        let want_shot_01 = !saved_01 && frame == 58;
        let want_shot_02 = !saved_02
            && engine.digger.dig_progress >= 0.35
            && renderer.viewmodel.is_swinging()
            && renderer.viewmodel.swing_progress() >= 0.15;
        let want_shot_04 = !saved_04 && frame == 188 && engine.mode == AppMode::Playing;
        let want_shot_03 = !saved_03 && frame == 205 && engine.mode == AppMode::Paused;

        if want_shot_01 || want_shot_02 || want_shot_03 || want_shot_04 {
            let img = renderer.render_to_headless_target_with_egui(
                cam_pos,
                vp,
                tod,
                &engine.settings,
                Some(&egui_ctx),
                Some(output),
                1.0,
            )?;
            let fname = if want_shot_01 {
                saved_01 = true;
                "01_world_trees.png"
            } else if want_shot_02 {
                saved_02 = true;
                "02_mining_swing.png"
            } else if want_shot_04 {
                saved_04 = true;
                "04_third_person.png"
            } else {
                saved_03 = true;
                "03_esc_menu.png"
            };
            img.save(out_dir.join(fname))?;
        } else {
            renderer.render_frame(
                cam_pos,
                vp,
                tod,
                &engine.settings,
                Some(&egui_ctx),
                Some(output),
                1.0,
            )?;
        }
    }

    renderer.device.poll(wgpu::Maintain::Wait);
    let elapsed = t_start.elapsed().as_secs_f64();
    let avg_fps = (total_frames as f64) / elapsed;

    // Fallback check if axe mined faster than 1 frame without catching swing phase
    anyhow::ensure!(
        saved_01 && saved_02 && saved_03 && saved_04,
        "Missing playtest screenshots: 01={saved_01} 02={saved_02} 03={saved_03} 04={saved_04}"
    );
    let mp = mined_pos.expect("Expected oak trunk block to be mined during playtest");
    assert_eq!(
        engine.world.get_node(mp).0,
        CONTENT_AIR,
        "Mined block at {mp:?} should be CONTENT_AIR"
    );
    anyhow::ensure!(
        picked_up_drop,
        "Expected mining oak trunk in Survival to drop & auto-pickup item into inventory"
    );
    let dropped_item = {
        let ps = engine.player_state.read();
        let mg = ps.meta.read();
        mg.inventories
            .get("main")
            .and_then(|inv| inv[3..].iter().find(|s| !s.is_empty()))
            .map(|s| format!("{} x{}", s.name, s.count))
    };
    anyhow::ensure!(
        dropped_item.is_some(),
        "Expected Survival inventory slots 3.. to contain mined drop"
    );
    anyhow::ensure!(placed_ok, "Expected block placement to succeed in playtest");
    anyhow::ensure!(
        third_person_ok,
        "Expected 3rd-person camera mode to enable player_model and disable viewmodel"
    );
    anyhow::ensure!(
        esc_cursor_released && focus_loss_cursor_released,
        "Expected ESC and Focused(false) to release cursor lock cleanly"
    );

    println!(
        "[PLAYTEST] Adapter: {} | 240 frames in {:.3}s => {:.1} avg FPS ({:.2} ms/frame)",
        renderer.adapter_name,
        elapsed,
        avg_fps,
        (elapsed * 1000.0) / total_frames as f64
    );
    println!(
        "[PLAYTEST] Mined {} at ({}, {}, {}) -> verified AIR | Drop picked up: {:?} | Placed block at {:?} -> verified={} | 3P={}",
        mined_node_name,
        mp.x,
        mp.y,
        mp.z,
        dropped_item,
        placed_pos,
        placed_ok,
        third_person_ok
    );
    println!(
        "[PLAYTEST] Cursor release verified: ESC={}, Focused(false)={}",
        esc_cursor_released, focus_loss_cursor_released
    );
    println!(
        "[PLAYTEST] Saved screenshots: {}, {}, {}, {}",
        out_dir.join("01_world_trees.png").display(),
        out_dir.join("02_mining_swing.png").display(),
        out_dir.join("03_esc_menu.png").display(),
        out_dir.join("04_third_person.png").display()
    );

    let _ = std::fs::remove_dir_all(&playtest_world_dir);
    Ok(())
}

fn run_bench_startup() -> anyhow::Result<()> {
    let out_dir = std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join("playtest_out");
    std::fs::create_dir_all(&out_dir)?;

    let t0 = Instant::now();
    let paths = EnginePaths::resolve();
    let mut engine = CubexEngine::boot_shell(paths)?;
    let rp_path = engine.resolve_rp_path();
    let (mods_rx, assets_rx) = spawn_asset_preloader(&engine.paths, rp_path);

    let mut renderer = WgpuRenderer::new_headless(1280, 720, &engine.atlas)?;
    let egui_ctx = egui::Context::default();
    let theme = McTheme::load(&egui_ctx);

    let splash_info = SplashScreenInfo::default();
    let splash_raw = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 720.0),
        )),
        time: Some(0.05),
        ..Default::default()
    };
    let splash_out = egui_ctx.run(splash_raw, |c| {
        render_splash_screen(c, &splash_info, 0.05, 1.0);
    });
    let proj = glam::Mat4::perspective_rh(70.0f32.to_radians(), 1280.0 / 720.0, 0.1, 32.0)
        * glam::Mat4::look_at_rh(Vec3::ZERO, Vec3::Z, Vec3::Y);
    renderer.fog_dist_override = Some(1.0);
    let splash_img = renderer.render_to_headless_target_with_egui(
        Vec3::ZERO,
        proj,
        0.4,
        &engine.settings,
        Some(&egui_ctx),
        Some(splash_out),
        1.0,
    )?;
    let splash_ms = t0.elapsed().as_secs_f64() * 1000.0;
    splash_img.save(out_dir.join("00_vortex_splash.png"))?;

    // Drive incremental Lua mod boot + receive background preloaded textures & audio
    let mods = mods_rx.recv()?;
    let mut lua_inc = LuaIncrementalBoot::begin(
        engine.paths.clone(),
        engine.registries.clone(),
        engine.world.clone(),
        engine.config.clone(),
        mods,
    )?;
    while !lua_inc.is_done() {
        let _ = lua_inc.step_budget(Duration::from_millis(12));
    }
    let lua_rt = lua_inc.finish();
    let (preloaded, audio_arc) = assets_rx.recv()?;
    engine.set_audio_loaded(audio_arc);
    let _ = lua_rt.join_player(engine.player_state.clone());
    engine.last_inv_total_count = total_inventory_items(&engine.player_state);
    engine.lua_rt = Some(lua_rt);

    // Build complete pre-lit + pre-meshed 3D Title Panorama
    let stage_a = AtomicU8::new(0);
    let done_a = AtomicUsize::new(0);
    let total_a = AtomicUsize::new(1);
    let bundle = build_panorama_bundle(
        preloaded,
        &engine.registries,
        &engine.world,
        engine.settings.smooth_lighting,
        &stage_a,
        &done_a,
        &total_a,
    );
    engine.atlas = bundle.atlas;
    engine.loaded_atlas_pack = Some(engine.resolve_rp_path());
    engine.palette = bundle.palette;
    engine.generated_columns = bundle.generated_columns;
    engine.panorama_center = bundle.panorama_center;
    engine.panorama_fog_dist = 92.0;

    renderer.reload_atlas(&engine.atlas);
    let uploaded_chunks = bundle.meshes.len();
    for mesh in bundle.meshes {
        renderer.upload_chunk_mesh_now(mesh);
    }

    renderer.fog_dist_override = Some(engine.panorama_fog_dist);
    renderer.show_viewmodel = false;
    renderer.show_player_model = false;
    let (eye, vp) =
        WgpuRenderer::compute_panorama_camera(0.4, 1280.0 / 720.0, engine.panorama_center);
    let title_raw = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 720.0),
        )),
        time: Some(0.4),
        ..Default::default()
    };
    let servers_file = engine.paths.cubex_data_dir.join("servers.json");
    let title_out = egui_ctx.run(title_raw, |c| {
        let _ = render_title_screen_with_atlas(
            c,
            &theme,
            &mut engine.title_state,
            &mut engine.settings,
            &engine.paths.worlds_dir,
            &servers_file,
            &engine.packs,
            &engine.tunnel,
            &engine.tokio_rt,
            Some(&engine.atlas),
            0.4,
        );
    });
    let has_title_menu = title_out.shapes.iter().any(|s| {
        matches!(&s.shape, egui::epaint::Shape::Text(t) if t.galley.text() == "Singleplayer")
    });
    anyhow::ensure!(
        has_title_menu,
        "Expected Title Screen UI ('Singleplayer' button) to render on Frame 1"
    );
    let title_img = renderer.render_to_headless_target_with_egui(
        eye,
        vp,
        0.4,
        &engine.settings,
        Some(&egui_ctx),
        Some(title_out),
        1.0,
    )?;
    let title_ms = t0.elapsed().as_secs_f64() * 1000.0;
    title_img.save(out_dir.join("00_title_panorama_frame1.png"))?;

    println!(
        "[BENCH-STARTUP] Adapter: {} | Vortex Studios Splash Frame 1: {:.1} ms",
        renderer.adapter_name, splash_ms
    );
    println!(
        "[BENCH-STARTUP] Title Screen 3D Panorama Frame 1: {:.1} ms total | uploaded_chunks={} (visible={}, draw_calls={}) | cols={}",
        title_ms,
        uploaded_chunks,
        renderer.last_visible_chunks,
        renderer.last_draw_calls,
        engine.generated_columns.len(),
    );
    println!(
        "[BENCH-STARTUP] Saved: {}, {}",
        out_dir.join("00_vortex_splash.png").display(),
        out_dir.join("00_title_panorama_frame1.png").display()
    );

    anyhow::ensure!(
        splash_ms <= 400.0,
        "Vortex Studios Splash Frame 1 took {splash_ms:.1} ms (> 400 ms budget)"
    );
    anyhow::ensure!(
        uploaded_chunks >= 100,
        "Expected >= 100 uploaded chunks on Title Frame 1, got {uploaded_chunks}"
    );
    Ok(())
}

fn run_qa_fuzz(seeds: usize, out_dir: PathBuf) -> anyhow::Result<()> {
    let t0 = Instant::now();
    let paths = EnginePaths::resolve();
    let engine = CubexEngine::boot_base(paths)?;
    let reg = engine.registries.read();
    let fuzzer = run_physics_fuzzer(seeds, &reg);
    let passed = fuzzer.passed;
    let mut failures = Vec::new();
    if !passed {
        failures.extend(fuzzer.invariant_violations.clone());
        if failures.is_empty() {
            failures.push("Physics fuzzer failed".into());
        }
    }
    let report = QaMasterReport {
        passed,
        mode: format!("fuzz({seeds})"),
        timestamp_unix: unix_timestamp_now(),
        elapsed_secs: t0.elapsed().as_secs_f64(),
        static_audit: None,
        fuzzer: Some(fuzzer.clone()),
        playthrough: None,
        visual: None,
        failures,
    };
    let report_path = write_master_report(&out_dir, &report)?;
    println!(
        "[QA FUZZ] passed={} seeds={} scenarios={}/{} ticks={} portals={} report={}",
        report.passed,
        fuzzer.seeds_tested,
        fuzzer.scenarios_passed,
        fuzzer.scenarios_total,
        fuzzer.total_ticks,
        fuzzer.total_portal_cycles,
        report_path.display()
    );
    anyhow::ensure!(
        report.passed,
        "QA physics fuzzer failed (see {})",
        report_path.display()
    );
    Ok(())
}

fn run_qa_update_baselines(baselines_dir: PathBuf, out_dir: PathBuf) -> anyhow::Result<()> {
    let t0 = Instant::now();
    let paths = EnginePaths::resolve();
    let engine = CubexEngine::boot_base(paths)?;
    let reg = engine.registries.read();
    let visual = run_visual_regression(
        &engine.paths,
        &reg,
        &engine.atlas,
        &engine.settings,
        &out_dir,
        &baselines_dir,
        true,
    )?;
    let passed = visual.passed;
    let failures = visual.failures.clone();
    let report = QaMasterReport {
        passed,
        mode: "update_baselines".into(),
        timestamp_unix: unix_timestamp_now(),
        elapsed_secs: t0.elapsed().as_secs_f64(),
        static_audit: None,
        fuzzer: None,
        playthrough: None,
        visual: Some(visual.clone()),
        failures,
    };
    let report_path = write_master_report(&out_dir, &report)?;
    for fr in &visual.frames {
        println!(
            "[QA BASELINES] {} | luma={:.1} std={:.1} status={} out={}",
            fr.name, fr.mean_luma, fr.rgb_stddev, fr.baseline_status, fr.output_path
        );
    }
    println!(
        "[QA BASELINES] passed={} avg_fps={:.1} 1pct_low={:.1} peak_mesh_ms={:.2} report={}",
        visual.passed,
        visual.telemetry.avg_fps,
        visual.telemetry.one_percent_low_fps,
        visual.telemetry.peak_chunk_mesh_ms,
        report_path.display()
    );
    anyhow::ensure!(
        report.passed,
        "QA baseline generation failed (see {})",
        report_path.display()
    );
    Ok(())
}

fn run_qa_suite(
    out_dir: PathBuf,
    baselines_dir: PathBuf,
    fuzz_seeds: usize,
) -> anyhow::Result<()> {
    let t0 = Instant::now();
    let paths = EnginePaths::resolve();
    let engine = CubexEngine::boot_base(paths)?;
    let reg = engine.registries.read();

    println!("[QA SUITE] Phase 1: Static registry & crafting reachability audit...");
    let static_audit = run_static_audit(&reg, &engine.atlas, engine.lua_rt.as_ref());
    println!(
        "[QA SUITE] Phase 1 done: passed={} nodes={} tools={} foods={} recipes={} failures={}",
        static_audit.passed,
        static_audit.total_nodes,
        static_audit.total_tools,
        static_audit.total_foods,
        static_audit.total_recipes,
        static_audit.failures.len()
    );

    println!("[QA SUITE] Phase 2: Physics & edge-case chaos fuzzer ({fuzz_seeds} seeds)...");
    let fuzzer = run_physics_fuzzer(fuzz_seeds, &reg);
    println!(
        "[QA SUITE] Phase 2 done: passed={} scenarios={}/{} ticks={} portals={}",
        fuzzer.passed,
        fuzzer.scenarios_passed,
        fuzzer.scenarios_total,
        fuzzer.total_ticks,
        fuzzer.total_portal_cycles
    );

    println!("[QA SUITE] Phase 3: Headless 7-stage Survival-to-Nether & Structure playthrough...");
    let playthrough =
        run_headless_playthrough(SURVIVAL_NETHER_SEED, &reg, engine.lua_rt.as_ref());
    println!(
        "[QA SUITE] Phase 3 done: passed={} stages={}/{} mobs={} loot={} monument={}",
        playthrough.passed,
        playthrough.stages_completed,
        playthrough.total_stages,
        playthrough.mobs_defeated,
        playthrough.chest_looted_items,
        playthrough.ocean_monument_verified
    );

    println!("[QA SUITE] Phase 4: Offscreen wgpu visual regression & perf telemetry...");
    let visual = run_visual_regression(
        &engine.paths,
        &reg,
        &engine.atlas,
        &engine.settings,
        &out_dir,
        &baselines_dir,
        false,
    )?;
    println!(
        "[QA SUITE] Phase 4 done: passed={} frames={} avg_fps={:.1} 1pct_low={:.1} peak_mesh_ms={:.2}",
        visual.passed,
        visual.frames.len(),
        visual.telemetry.avg_fps,
        visual.telemetry.one_percent_low_fps,
        visual.telemetry.peak_chunk_mesh_ms
    );

    let mut failures = Vec::new();
    if !static_audit.passed {
        failures.extend(static_audit.failures.iter().map(|f| format!("Phase1: {f}")));
    }
    if !fuzzer.passed {
        failures.extend(
            fuzzer
                .invariant_violations
                .iter()
                .map(|f| format!("Phase2: {f}")),
        );
    }
    if !playthrough.passed {
        failures.push(format!(
            "Phase3: {}",
            playthrough
                .failure_reason
                .clone()
                .unwrap_or_else(|| "Playthrough failed".into())
        ));
    }
    if !visual.passed {
        failures.extend(visual.failures.iter().map(|f| format!("Phase4: {f}")));
    }

    let passed = static_audit.passed
        && fuzzer.passed
        && playthrough.passed
        && visual.passed
        && failures.is_empty();
    let report = QaMasterReport {
        passed,
        mode: "suite".into(),
        timestamp_unix: unix_timestamp_now(),
        elapsed_secs: t0.elapsed().as_secs_f64(),
        static_audit: Some(static_audit),
        fuzzer: Some(fuzzer),
        playthrough: Some(playthrough),
        visual: Some(visual),
        failures,
    };
    let report_path = write_master_report(&out_dir, &report)?;
    println!(
        "[QA SUITE] Complete: passed={} elapsed={:.2}s report={}",
        report.passed,
        report.elapsed_secs,
        report_path.display()
    );
    anyhow::ensure!(
        report.passed,
        "QA suite reported {} failure(s) (see {})",
        report.failures.len(),
        report_path.display()
    );
    Ok(())
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--smoke-test") {
        return run_smoke_test();
    }
    if args.iter().any(|a| a == "--playtest") {
        return run_playtest();
    }
    if args.iter().any(|a| a == "--bench-startup") {
        return run_bench_startup();
    }

    if args.iter().any(|a| a == "--dedicated") {
        let paths = EnginePaths::resolve();
        let mut world_dir = paths.worlds_dir.join("Cubex_Dedicated");
        let mut port = DEFAULT_LAN_PORT;
        let mut i = 1;
        while i < args.len() {
            if args[i] == "--world" && i + 1 < args.len() {
                world_dir = PathBuf::from(&args[i + 1]);
                i += 2;
            } else if args[i] == "--port" && i + 1 < args.len() {
                port = args[i + 1].parse().unwrap_or(DEFAULT_LAN_PORT);
                i += 2;
            } else {
                i += 1;
            }
        }
        return run_dedicated_server(&world_dir, port);
    }

    let qa_playthrough_out = match parse_qa_args(&args)? {
        Some(QaCliCommand::Suite {
            out_dir,
            baselines_dir,
            fuzz_seeds,
        }) => return run_qa_suite(out_dir, baselines_dir, fuzz_seeds),
        Some(QaCliCommand::Fuzz { seeds, out_dir }) => return run_qa_fuzz(seeds, out_dir),
        Some(QaCliCommand::UpdateBaselines {
            baselines_dir,
            out_dir,
        }) => return run_qa_update_baselines(baselines_dir, out_dir),
        Some(QaCliCommand::Playthrough { out_dir }) => Some(out_dir),
        None => None,
    };

    let autotest_cfg = parse_autotest_args(&args)?;
    let paths = EnginePaths::resolve();
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = WindowedApp::new_booting_with_cli(paths, autotest_cfg, qa_playthrough_out)?;
    event_loop.run_app(&mut app)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dual_monitor_cursor_lock_respects_focus_and_overlays() {
        let paths = EnginePaths::resolve();
        let engine = CubexEngine::boot_base(paths).expect("boot_base");
        let mut app = WindowedApp::new(engine);

        // Splash & Title menus never capture cursor
        app.engine.mode = AppMode::Splash;
        assert!(!app.should_capture_cursor());
        app.set_cursor_locked(true);
        assert!(!app.cursor_locked);
        app.engine.mode = AppMode::Title;
        assert!(!app.should_capture_cursor());
        app.set_cursor_locked(true);
        assert!(!app.cursor_locked);

        // Playing + focused + no overlay allows capture
        app.engine.mode = AppMode::Playing;
        assert!(app.should_capture_cursor());
        app.set_cursor_locked(true);
        assert!(app.is_gameplay_captured());

        // Window loses focus (Alt+Tab or cursor moved to secondary monitor) -> immediate unlock
        app.on_focus_changed(false);
        assert!(!app.window_focused);
        assert!(!app.cursor_locked);
        assert!(!app.should_capture_cursor());
        app.set_cursor_locked(true);
        assert!(!app.cursor_locked);

        // Focus returns + click relocks, then opening chat or inventory releases lock
        app.on_focus_changed(true);
        app.set_cursor_locked(true);
        assert!(app.is_gameplay_captured());
        app.open_inventory_formspec();
        assert!(!app.cursor_locked);
        assert!(!app.should_capture_cursor());
    }

    #[test]
    fn compute_window_viewports_covers_2p_3p_4p_and_multi_monitor() {
        let log = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::Vec2::new(1280.0, 720.0));
        let v2 = WindowedApp::compute_window_viewports(CoopMode::SplitVertical, 2, 1280, 720, log, 0);
        assert_eq!(v2.len(), 2);
        assert_eq!(v2[0].1, [0, 0, 640, 720]);
        assert_eq!(v2[1].1, [640, 0, 640, 720]);

        let h2 = WindowedApp::compute_window_viewports(CoopMode::SplitHorizontal, 2, 1280, 720, log, 0);
        assert_eq!(h2.len(), 2);
        assert_eq!(h2[0].1, [0, 0, 1280, 360]);
        assert_eq!(h2[1].1, [0, 360, 1280, 360]);

        let q3 = WindowedApp::compute_window_viewports(CoopMode::QuadSplit, 3, 1280, 720, log, 0);
        assert_eq!(q3.len(), 3);
        assert_eq!(q3[2].1, [0, 360, 1280, 360]);

        let q4 = WindowedApp::compute_window_viewports(CoopMode::QuadSplit, 4, 1280, 720, log, 0);
        assert_eq!(q4.len(), 4);
        assert_eq!(q4[0].1, [0, 0, 640, 360]);
        assert_eq!(q4[1].1, [640, 0, 640, 360]);
        assert_eq!(q4[2].1, [0, 360, 640, 360]);
        assert_eq!(q4[3].1, [640, 360, 640, 360]);

        let d0 = WindowedApp::compute_window_viewports(CoopMode::DualScreenTwoEach, 4, 1280, 720, log, 0);
        let d1 = WindowedApp::compute_window_viewports(CoopMode::DualScreenTwoEach, 4, 1280, 720, log, 1);
        assert_eq!((d0[0].0, d0[1].0, d1[0].0, d1[1].0), (0, 1, 2, 3));

        for widx in 0..4 {
            let m = WindowedApp::compute_window_viewports(CoopMode::QuadMonitor, 4, 1280, 720, log, widx);
            assert_eq!(m.len(), 1);
            assert_eq!(m[0].0, widx);
            assert_eq!(m[0].1, [0, 0, 1280, 720]);
        }
    }

    #[test]
    fn carterpcs_skin_auto_equips_ram_mic_in_hotbar_slot_1() {
        let ps = Arc::new(RwLock::new(ObjectState::new_player("singleplayer")));
        {
            let guard = ps.read();
            let mut mg = guard.meta.write();
            let main = mg.inventories.get_mut("main").unwrap();
            main[0] = ItemStackData::from_spec("mcl_core:dirt");
            main[0].count = 5;
        }

        let mut selected_slot = 3usize;
        equip_ram_mic_for_carterpcs(&ps, &mut selected_slot);
        assert_eq!(selected_slot, 0);
        {
            let guard = ps.read();
            assert_eq!(guard.wield_index, 1);
            let mg = guard.meta.read();
            let main = mg.inventories.get("main").unwrap();
            assert_eq!(main[0].name, RAM_MIC_ITEM_ID);
            assert_eq!(main[0].count, 1);
            assert_eq!(main[1].name, "mcl_core:dirt");
            assert_eq!(main[1].count, 5);
        }

        // Calling again must not create duplicate RAM mics in the inventory
        selected_slot = 4;
        equip_ram_mic_for_carterpcs(&ps, &mut selected_slot);
        assert_eq!(selected_slot, 0);
        let guard = ps.read();
        let mg = guard.meta.read();
        let main = mg.inventories.get("main").unwrap();
        let mic_slots: Vec<usize> = main
            .iter()
            .enumerate()
            .filter_map(|(i, st)| (!st.is_empty() && st.name == RAM_MIC_ITEM_ID).then_some(i))
            .collect();
        assert_eq!(mic_slots, vec![0]);
    }

    #[test]
    fn creative_pick_block_and_creative_inventory_mode_selection() {
        let paths = EnginePaths::resolve();
        let mut engine = CubexEngine::boot_base(paths).expect("boot_base");
        let stone = {
            let mut reg = engine.registries.write();
            reg.register_node(cubex_rs::registry::NodeDef {
                name: "mcl_core:stone".into(),
                description: "Stone".into(),
                drawtype: DrawType::Normal,
                ..Default::default()
            })
        };

        engine.player.gamemode = GameMode::Creative;
        engine.player.selected_slot = 2;
        {
            let reg = engine.registries.read();
            assert!(pick_block_into_hotbar(
                &mut engine.player,
                &engine.player_state,
                stone,
                &reg
            ));
        }
        {
            let ps = engine.player_state.read();
            let mg = ps.meta.read();
            let main = mg.inventories.get("main").unwrap();
            assert_eq!(main[2].name, "mcl_core:stone");
            assert_eq!(main[2].count, 64);
        }

        // Switching slots then pick-blocking an existing hotbar block jumps back to slot 2
        engine.player.selected_slot = 0;
        {
            let reg = engine.registries.read();
            assert!(pick_block_into_hotbar(
                &mut engine.player,
                &engine.player_state,
                stone,
                &reg
            ));
        }
        assert_eq!(engine.player.selected_slot, 2);

        // open_inventory_formspec opens Creative Catalog in Creative and 2x2 Crafting in Survival
        let mut app = WindowedApp::new(engine);
        app.engine.mode = AppMode::Playing;
        app.engine.player.gamemode = GameMode::Creative;
        app.open_inventory_formspec();
        assert_eq!(
            app.engine
                .active_formspec
                .as_ref()
                .map(|f| f.formname.as_str()),
            Some(CREATIVE_INVENTORY_FORMNAME)
        );
        app.close_active_formspec();

        app.engine.player.gamemode = GameMode::Survival;
        app.open_inventory_formspec();
        assert_eq!(
            app.engine
                .active_formspec
                .as_ref()
                .map(|f| f.formname.as_str()),
            Some("cubex:inventory")
        );
    }
}

pub fn is_player_inventory_formname(formname: &str) -> bool {
    formname == "cubex:inventory" || formname == CREATIVE_INVENTORY_FORMNAME
}

pub fn pick_block_into_hotbar(
    player: &mut PlayerController,
    player_state: &Arc<RwLock<ObjectState>>,
    node_id: u16,
    reg: &EngineRegistries,
) -> bool {
    if matches!(node_id, CONTENT_AIR | CONTENT_IGNORE | CONTENT_UNKNOWN) {
        return false;
    }
    let node_name = reg.get_name_from_content_id(node_id).trim().to_string();
    if node_name.is_empty()
        || node_name == "air"
        || node_name == "ignore"
        || node_name == "unknown"
    {
        return false;
    }
    let stack_max = reg
        .get_item_def(&node_name)
        .map(|d| d.stack_max)
        .or_else(|| reg.get_node_def(node_id).map(|d| d.stack_max))
        .unwrap_or(64)
        .clamp(1, 64);

    let mut target_slot = player.selected_slot.min(8);
    let mut changed = false;
    {
        let ps = player_state.read();
        let mut mg = ps.meta.write();
        let main = mg
            .inventories
            .entry("main".into())
            .or_insert_with(|| vec![ItemStackData::default(); 36]);
        if main.len() < 36 {
            main.resize_with(36, ItemStackData::default);
        }

        if let Some(hot_idx) = main
            .iter()
            .take(9)
            .position(|st| !st.is_empty() && st.name == node_name)
        {
            target_slot = hot_idx;
            if player.gamemode == GameMode::Creative && main[hot_idx].count < stack_max {
                main[hot_idx].count = stack_max;
            }
            changed = true;
        } else if player.gamemode == GameMode::Creative {
            if !main[target_slot].is_empty() {
                if let Some(empty_hot) = main.iter().take(9).position(|st| st.is_empty()) {
                    target_slot = empty_hot;
                } else {
                    let displaced = main[target_slot].clone();
                    if let Some(empty_idx) = main.iter().skip(9).position(|st| st.is_empty()) {
                        main[9 + empty_idx] = displaced;
                    }
                }
            }
            let mut st = ItemStackData::from_spec(&node_name);
            st.count = stack_max;
            main[target_slot] = st;
            changed = true;
        } else if let Some(inv_idx) = main
            .iter()
            .enumerate()
            .skip(9)
            .find_map(|(i, st)| (!st.is_empty() && st.name == node_name).then_some(i))
        {
            main.swap(target_slot, inv_idx);
            changed = true;
        }
    }

    if changed {
        player.selected_slot = target_slot;
        player_state.write().wield_index = target_slot + 1;
    }
    changed
}

fn drop_held_slot_item(
    player: &PlayerController,
    player_state: &Arc<RwLock<ObjectState>>,
    lua_rt: Option<&LuaRuntime>,
) {
    let slot = player.selected_slot.min(8);
    let dropped_name = {
        let ps = player_state.read();
        let mut mg = ps.meta.write();
        mg.inventories.get_mut("main").and_then(|inv| {
            inv.get_mut(slot).and_then(|st| {
                if st.is_empty() {
                    None
                } else {
                    let name = st.name.clone();
                    st.count = st.count.saturating_sub(1);
                    if st.count == 0 {
                        *st = Default::default();
                    }
                    Some(name)
                }
            })
        })
    };
    if let (Some(item_name), Some(rt)) = (dropped_name, lua_rt) {
        let dir = player.look_dir().normalize_or_zero();
        let spawn_pos = player.eye_pos() + dir * 0.75 - Vec3::Y * 0.15;
        let vel = dir * 4.2 + Vec3::Y * 1.4;
        let _ = rt.spawn_thrown_item_drop(spawn_pos, vel, &item_name);
    }
}

fn refresh_live_nodemeta_formspec(
    active: Option<&mut ActiveFormspec>,
    world: &Arc<WorldStore>,
) {
    if let Some(af) = active {
        if af.formname.starts_with("nodemeta:") {
            if let Some(pos) = af.context_pos {
                let live_fs = world
                    .node_meta
                    .read()
                    .get(&pos)
                    .and_then(|m| m.read().fields.get("formspec").cloned());
                if let Some(fs_str) = live_fs {
                    if !fs_str.trim().is_empty() && fs_str != af.raw_formspec {
                        af.parsed = parse_formspec(&fs_str);
                        af.raw_formspec = fs_str;
                    }
                }
            }
        }
    }
}

fn sync_player_lua_combat_state(
    player: &mut PlayerController,
    player_state: &Arc<RwLock<ObjectState>>,
    dt: f32,
) -> i32 {
    let mut ps = player_state.write();
    ps.spawn_invuln_timer = player.spawn_invuln_timer;
    ps.hurt_flash = (ps.hurt_flash - dt).max(0.0);
    if player.is_dead {
        ps.is_dead = true;
        ps.hp = 0;
        ps.hp_dirty_from_lua = false;
        ps.pending_knockback = Vec3::ZERO;
        return 0;
    }
    if ps.hp_dirty_from_lua {
        if player.gamemode == GameMode::Survival {
            player.hp = ps.hp.clamp(0, 20);
        }
        ps.hp_dirty_from_lua = false;
    }
    if ps.pending_knockback.length_squared() > 1e-6 {
        player.vel += ps.pending_knockback;
        ps.pending_knockback = Vec3::ZERO;
    }
    player.hp
}

fn handle_player_death_if_needed(
    player: &mut PlayerController,
    player_state: &Arc<RwLock<ObjectState>>,
    lua_rt: Option<&LuaRuntime>,
    keep_inventory: bool,
    chat_messages: &mut Vec<String>,
    who_label: &str,
) -> bool {
    if player.gamemode != GameMode::Survival || player.hp > 0 || player.is_dead {
        return false;
    }
    player.is_dead = true;
    player.vel = Vec3::ZERO;
    player.hp = 0;

    let already_dropped = {
        let g = player_state.read();
        if player.death_cause.is_empty() && !g.last_death_cause.is_empty() {
            player.death_cause = g.last_death_cause.clone();
        }
        g.has_dropped_death_items
    };
    if player.death_cause.is_empty() {
        player.death_cause = format_death_cause("").to_string();
    }

    if !already_dropped {
        let dropped = drain_player_inventory_for_death(player_state, keep_inventory);
        player.has_dropped_death_items = true;
        if let Some(rt) = lua_rt {
            if !dropped.is_empty() {
                let drop_block = player.pos.floor().as_ivec3();
                for st in &dropped {
                    let spec = if st.wear > 0 {
                        format!("{} {} {}", st.name, st.count.max(1), st.wear)
                    } else if st.count > 1 {
                        format!("{} {}", st.name, st.count)
                    } else {
                        st.name.clone()
                    };
                    rt.spawn_physical_item_drop(drop_block, &spec);
                }
            }
            let reason = if player.death_cause.contains("Suffocated") {
                "in_wall"
            } else if player.death_cause.contains("Fell") {
                "fall"
            } else {
                "punch"
            };
            rt.notify_player_died(player_state, reason);
        }
    } else {
        player.has_dropped_death_items = true;
        player_state.write().is_dead = true;
    }

    chat_messages.push(format!("§c{who_label} died ({}).", player.death_cause));
    true
}

fn find_safe_respawn_pos(world: &WorldStore, reg: &EngineRegistries) -> Vec3 {
    let wt = *world.world_type.read();
    let sp = find_spawn_pos(world.seed, wt);
    let max_y = (sp.y + 48).min(240);
    for y in (0..=max_y).rev() {
        let (nid, _, _) = world.get_node(IVec3::new(sp.x, y, sp.z));
        if nid == CONTENT_AIR || nid == CONTENT_IGNORE {
            continue;
        }
        let is_solid = reg
            .get_node_def(nid)
            .is_some_and(|d| d.walkable && d.drawtype != DrawType::Liquid && d.drawtype != DrawType::FlowingLiquid);
        if !is_solid {
            continue;
        }
        let above1 = world.get_node(IVec3::new(sp.x, y + 1, sp.z)).0;
        let above2 = world.get_node(IVec3::new(sp.x, y + 2, sp.z)).0;
        let clear1 = above1 == CONTENT_AIR || !reg.get_node_def(above1).is_some_and(|d| d.walkable);
        let clear2 = above2 == CONTENT_AIR || !reg.get_node_def(above2).is_some_and(|d| d.walkable);
        if clear1 && clear2 {
            return Vec3::new(sp.x as f32 + 0.5, (y + 1) as f32 + 1.0, sp.z as f32 + 0.5);
        }
    }
    Vec3::new(sp.x as f32 + 0.5, sp.y as f32 + 1.0, sp.z as f32 + 0.5)
}

pub fn equip_ram_mic_for_carterpcs(
    player_state: &Arc<RwLock<ObjectState>>,
    selected_slot: &mut usize,
) {
    let mut ps = player_state.write();
    {
        let mut mg = ps.meta.write();
        let main = mg
            .inventories
            .entry("main".into())
            .or_insert_with(|| vec![ItemStackData::default(); 36]);
        if main.len() < 36 {
            main.resize(36, ItemStackData::default());
        }
        if main[0].name == RAM_MIC_ITEM_ID && !main[0].is_empty() {
            main[0].count = main[0].count.max(1);
        } else if let Some(existing_idx) = main
            .iter()
            .position(|st| !st.is_empty() && st.name == RAM_MIC_ITEM_ID)
        {
            main.swap(0, existing_idx);
        } else {
            if !main[0].is_empty() {
                let displaced = main[0].clone();
                if let Some(empty_idx) = main.iter().skip(1).position(|st| st.is_empty()) {
                    main[1 + empty_idx] = displaced;
                }
            }
            main[0] = ItemStackData::from_spec(RAM_MIC_ITEM_ID);
        }
    }
    *selected_slot = 0;
    ps.wield_index = 1;
}

#[inline]
fn world_for<'a>(
    ow: &'a Arc<WorldStore>,
    nw: &'a Arc<WorldStore>,
    dim: DimensionId,
) -> &'a Arc<WorldStore> {
    if dim.is_nether() {
        nw
    } else {
        ow
    }
}

#[inline]
fn biome_label_for(dim: DimensionId, pos: Vec3, seed: u64) -> &'static str {
    let wx = pos.x.floor() as i32;
    let wz = pos.z.floor() as i32;
    if dim.is_nether() {
        nether_biome_name_at(wx, wz, seed)
    } else {
        biome_name_at(wx, wz, seed)
    }
}

fn ensure_overworld_columns(
    world: &WorldStore,
    center_cx: i32,
    center_cz: i32,
    radius: i32,
    palette: &MapgenPalette,
    reg: &EngineRegistries,
) {
    let wt = *world.world_type.read();
    let y_range = chunk_y_range_for(wt);
    let existing: HashSet<IVec3> = world.chunks.read().keys().copied().collect();
    let mut missing = Vec::new();
    for cz in (center_cz - radius)..=(center_cz + radius) {
        for cx in (center_cx - radius)..=(center_cx + radius) {
            for cy in y_range.clone() {
                let cpos = IVec3::new(cx, cy, cz);
                if !existing.contains(&cpos) {
                    missing.push(cpos);
                }
            }
        }
    }
    if missing.is_empty() {
        return;
    }
    let seed = world.seed;
    let chunks: Vec<Chunk> = missing
        .into_par_iter()
        .map(|cpos| generate_chunk_for_world(cpos, seed, wt, palette, reg))
        .collect();
    let coords: Vec<IVec3> = chunks.iter().map(|c| c.pos).collect();
    for ch in chunks {
        world.insert_chunk(ch);
    }
    for &cpos in &coords {
        populate_structure_chests_for_chunk(world, cpos, seed, wt, reg);
    }
    propagate_world_light(world, &coords, reg);
}

#[allow(clippy::too_many_arguments)]
fn tick_player_portal(
    player: &mut PlayerController,
    player_state: &Arc<RwLock<ObjectState>>,
    ow_world: &Arc<WorldStore>,
    nw_world: &Arc<WorldStore>,
    ow_palette: &MapgenPalette,
    nw_palette: &NetherPalette,
    reg: &EngineRegistries,
    chat_messages: &mut Vec<String>,
    who_label: &str,
    dt: f32,
) -> Option<DimensionId> {
    player.portal_cooldown = (player.portal_cooldown - dt).max(0.0);
    player.portal_warp_flash = (player.portal_warp_flash - dt * 1.4).max(0.0);
    if player.is_dead {
        player.portal_charge = 0.0;
        return None;
    }

    let cur_world = world_for(ow_world, nw_world, player.dimension);
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
    let (nid_f, _, p2_f) = cur_world.get_node(p_feet);
    let (nid_e, _, p2_e) = cur_world.get_node(p_eye);
    let feet_portal = is_nether_portal_node(nid_f, reg);
    let eye_portal = is_nether_portal_node(nid_e, reg);

    if !feet_portal && !eye_portal {
        player.portal_charge = (player.portal_charge - dt * 2.5).max(0.0);
        return None;
    }
    if player.portal_cooldown > 0.0 {
        return None;
    }

    let rate = if player.gamemode == GameMode::Creative {
        1.0 / 0.15
    } else {
        1.0 / 3.0
    };
    player.portal_charge = (player.portal_charge + dt * rate).min(1.0);
    player.portal_warp_flash = player
        .portal_warp_flash
        .max(0.35 + player.portal_charge * 0.65);

    if player.portal_charge < 1.0 {
        return None;
    }

    let axis = if feet_portal {
        PortalAxis::from_param2(p2_f)
    } else {
        PortalAxis::from_param2(p2_e)
    };
    let from_dim = player.dimension;
    let to_dim = from_dim.opposite();
    let scaled = scale_portal_coords(p_feet, from_dim, to_dim);
    let dest_cx = scaled.x.div_euclid(16);
    let dest_cz = scaled.z.div_euclid(16);
    let dest_world = world_for(ow_world, nw_world, to_dim);

    if to_dim.is_nether() {
        let _ = ensure_nether_columns(
            nw_world,
            dest_cx,
            dest_cz,
            1,
            nw_world.seed,
            nw_palette,
            reg,
        );
    } else {
        ensure_overworld_columns(ow_world, dest_cx, dest_cz, 1, ow_palette, reg);
    }

    let stand = find_or_build_destination_portal(dest_world, reg, scaled, axis);
    player.pos = Vec3::new(
        stand.x as f32 + 0.5,
        stand.y as f32 + 0.05,
        stand.z as f32 + 0.5,
    );
    player.vel = Vec3::ZERO;
    player.dimension = to_dim;
    player.portal_charge = 0.0;
    player.portal_cooldown = 4.0;
    player.portal_warp_flash = 1.0;
    player.spawn_invuln_timer = player.spawn_invuln_timer.max(2.0);
    player.reset_fall_state();

    {
        let mut ps = player_state.write();
        ps.pos = player.pos;
        ps.dimension = to_dim;
        ps.spawn_invuln_timer = ps.spawn_invuln_timer.max(2.0);
    }
    chat_messages.push(format!("§d{who_label} traveled to {}.", to_dim.label()));
    Some(to_dim)
}

fn respawn_player(
    player: &mut PlayerController,
    player_state: &Arc<RwLock<ObjectState>>,
    world: &Arc<WorldStore>,
    reg: &EngineRegistries,
    lua_rt: Option<&LuaRuntime>,
    chat_messages: &mut Vec<String>,
    who_label: &str,
) {
    player.dimension = DimensionId::Overworld;
    player.portal_charge = 0.0;
    player.portal_cooldown = 0.0;
    player.portal_warp_flash = 0.0;
    player.pos = find_safe_respawn_pos(world, reg);
    player.vel = Vec3::ZERO;
    player.reset_fall_state();
    player.hp = 20;
    player.hunger = 20;
    player.saturation = 5.0;
    player.exhaustion = 0.0;
    player.regen_timer = 0.0;
    player.starve_timer = 0.0;
    player.eat_cooldown = 0.0;
    player.is_dead = false;
    player.has_dropped_death_items = false;
    player.spawn_invuln_timer = 2.0;
    player.suffocation_timer = 0.0;
    player.wither_timer = 0.0;
    player.wither_tick_accum = 0.0;
    player.death_cause.clear();
    {
        let mut ps = player_state.write();
        ps.dimension = DimensionId::Overworld;
        ps.pos = player.pos;
        ps.hp = ps.hp_max.max(20);
        ps.hunger = 20;
        ps.hurt_flash = 0.0;
        ps.pending_knockback = Vec3::ZERO;
        ps.hp_dirty_from_lua = false;
        ps.is_dead = false;
        ps.has_dropped_death_items = false;
        ps.spawn_invuln_timer = 2.0;
        ps.last_death_cause.clear();
        let mut mg = ps.meta.write();
        mg.fields.insert("mcl_health".into(), "20".into());
        mg.fields.insert("mcl_hunger:hunger".into(), "20".into());
        mg.fields.insert("mcl_hunger:saturation".into(), "5.00".into());
    }
    if let Some(rt) = lua_rt {
        rt.notify_player_respawned(player_state);
    }
    chat_messages.push(format!("§a{who_label} respawned at world spawn."));
}

#[allow(clippy::too_many_arguments)]
fn step_player_physics_and_respawn(
    player: &mut PlayerController,
    player_state: &Arc<RwLock<ObjectState>>,
    move_input: &MoveInput,
    world: &Arc<WorldStore>,
    reg: &EngineRegistries,
    lua_rt: Option<&LuaRuntime>,
    audio: &AudioEngine,
    settings: &EngineSettings,
    keep_inventory: bool,
    chat_messages: &mut Vec<String>,
    who_label: &str,
    dt: f32,
) {
    let prev_hp = sync_player_lua_combat_state(player, player_state, dt);
    let step_ev = player.step(dt, move_input, world, reg, settings);
    {
        let mut ps = player_state.write();
        ps.pos = player.pos;
        ps.yaw = player.yaw;
        ps.pitch = player.pitch;
        ps.hp = player.hp;
        ps.hunger = player.hunger;
        ps.wield_index = player.selected_slot + 1;
        ps.sneak = player.is_sneaking;
        {
            let mut mg = ps.meta.write();
            mg.fields.insert("mcl_hunger:hunger".into(), player.hunger.to_string());
            mg.fields.insert(
                "mcl_hunger:saturation".into(),
                format!("{:.2}", player.saturation),
            );
        }
        if player.hp < prev_hp {
            ps.hurt_flash = 0.25;
        }
    }
    if let Some(nid) = step_ev.footstep_node {
        if let Some(spec) = reg.get_node_def(nid).and_then(|d| d.sounds.footstep.as_ref()) {
            audio.play_footstep(&spec.name, spec.gain, &settings.audio);
        }
    }
    if let Some(fe) = step_ev.fall {
        if let Some(spec) = reg.get_node_def(fe.node_id).and_then(|d| d.sounds.fall.as_ref()) {
            audio.play_node_sound(spec, SoundCategory::Blocks, &settings.audio);
        }
    }
    if step_ev.wither_damaged {
        let _ = audio.play_named("player_damage", 0.75, 0.9, SoundCategory::Hostile, &settings.audio);
    }
    handle_player_death_if_needed(
        player,
        player_state,
        lua_rt,
        keep_inventory,
        chat_messages,
        who_label,
    );
}

fn raycast_player_aabb(origin: Vec3, dir: Vec3, max_dist: f32, pos: Vec3) -> Option<f32> {
    let dir_n = dir.normalize_or_zero();
    if dir_n.length_squared() < 1e-6 || max_dist <= 0.0 {
        return None;
    }
    let min = pos + Vec3::new(-0.30, 0.0, -0.30);
    let max = pos + Vec3::new(0.30, 1.75, 0.30);
    let mut tmin = 0.0f32;
    let mut tmax = max_dist;
    for axis in 0..3 {
        let d = dir_n[axis];
        let o = origin[axis];
        if d.abs() < 1e-6 {
            if o < min[axis] || o > max[axis] {
                return None;
            }
        } else {
            let mut t1 = (min[axis] - o) / d;
            let mut t2 = (max[axis] - o) / d;
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
    Some(tmin * tmin)
}

fn render_nametags_in_viewport(
    ctx: &egui::Context,
    vp_rect: egui::Rect,
    cam_pos: Vec3,
    view_proj: glam::Mat4,
    players: &[(String, Vec3)],
) {
    if players.is_empty() {
        return;
    }
    let painter = ctx
        .layer_painter(egui::LayerId::new(
            egui::Order::Background,
            egui::Id::new(format!(
                "cubex_nametags_{}_{}",
                vp_rect.left() as i32,
                vp_rect.top() as i32
            )),
        ))
        .with_clip_rect(vp_rect);
    for (name, pos) in players {
        if pos.distance_squared(cam_pos) > 48.0 * 48.0 {
            continue;
        }
        let world_pt = Vec3::new(pos.x, pos.y + 2.15, pos.z);
        let clip = view_proj * world_pt.extend(1.0);
        if clip.w <= 0.05 {
            continue;
        }
        let ndc = clip.truncate() / clip.w;
        if ndc.x.abs() > 1.05 || ndc.y.abs() > 1.05 || !(0.0..=1.0).contains(&ndc.z) {
            continue;
        }
        let sx = vp_rect.left() + (ndc.x * 0.5 + 0.5) * vp_rect.width();
        let sy = vp_rect.top() + (-ndc.y * 0.5 + 0.5) * vp_rect.height();
        let galley = painter.layout_no_wrap(
            name.clone(),
            egui::FontId::monospace(12.5),
            egui::Color32::WHITE,
        );
        let size = galley.size() + egui::Vec2::new(8.0, 4.0);
        let box_r = egui::Rect::from_center_size(egui::Pos2::new(sx, sy), size);
        if vp_rect.intersects(box_r) {
            painter.rect_filled(
                box_r,
                2.0,
                egui::Color32::from_rgba_unmultiplied(10, 10, 14, 150),
            );
            painter.galley(
                box_r.min + egui::Vec2::new(4.0, 2.0),
                galley,
                egui::Color32::WHITE,
            );
        }
    }
}

fn collect_mob_player_inputs(engine: &CubexEngine) -> Vec<MobPlayerInput> {
    let mut out = Vec::with_capacity(4);
    out.push(MobPlayerInput {
        slot: 0,
        dimension: engine.player.dimension,
        pos: engine.player.pos,
        eye_pos: engine.player.eye_pos(),
        alive: !engine.player.is_dead,
        survival: engine.player.gamemode == GameMode::Survival,
    });
    if engine.settings.coop_mode.is_active() {
        for (idx, p) in engine.peers.iter().enumerate() {
            if p.joined {
                out.push(MobPlayerInput {
                    slot: idx + 1,
                    dimension: p.player.dimension,
                    pos: p.player.pos,
                    eye_pos: p.player.eye_pos(),
                    alive: !p.player.is_dead,
                    survival: p.player.gamemode == GameMode::Survival,
                });
            }
        }
    }
    out
}

/// Raycast up to 4 blocks against active entities, native mobs, co-op peers, and remote LAN players before block digging.
#[allow(clippy::too_many_arguments)]
fn try_melee_entity_attack(
    player_slot: usize,
    mine_just: bool,
    mine_held: bool,
    attack_tflp: &mut f32,
    dt: f32,
    targeted_block: Option<&RaycastHit>,
    player: &PlayerController,
    player_state: &Arc<RwLock<ObjectState>>,
    peer_players: &[Arc<RwLock<ObjectState>>],
    remote_players: &mut HashMap<String, RemoteLanPlayer>,
    mob_manager: &mut MobManager,
    registries: &Arc<RwLock<EngineRegistries>>,
    lua_rt: Option<&LuaRuntime>,
    audio: &AudioEngine,
    settings: &EngineSettings,
    net_session: Option<&NetClientSession>,
    local_server: Option<&LocalServerHandle>,
    mut renderer: Option<&mut WgpuRenderer>,
) -> bool {
    *attack_tflp = (*attack_tflp + dt).min(10.0);
    let eye = player.eye_pos();
    let dir = player.look_dir();
    let block_dist = targeted_block
        .map(|b| b.distance_sq.sqrt())
        .unwrap_or(f32::INFINITY);
    let max_dist = 4.0f32.min(block_dist);
    if max_dist <= 0.05 {
        return false;
    }

    let held_item_name = {
        let ps = player_state.read();
        let widx = ps.wield_index.saturating_sub(1);
        let mg = ps.meta.read();
        mg.inventories
            .get("main")
            .and_then(|v| v.get(widx))
            .map(|s| s.name.clone())
            .unwrap_or_default()
    };
    let held_dmg = {
        let reg = registries.read();
        reg.get_item_def(&held_item_name)
            .and_then(|d| d.tool_capabilities.as_ref())
            .and_then(|tc| tc.damage_groups.get("fleshy").copied())
            .unwrap_or(2)
            .max(2)
    };

    if mob_manager.has_melee_target(eye, dir, max_dist, player.dimension) {
        if mine_just || (mine_held && *attack_tflp >= 0.45) {
            *attack_tflp = 0.0;
            if let Some(ref mut r) = renderer {
                if player_slot == 0 {
                    r.viewmodel.trigger_swing();
                } else if let Some(pslot) = r.peer_slot_mut(player_slot) {
                    pslot.viewmodel.trigger_swing();
                }
            }
            if let Some(res) = mob_manager.try_melee_hit(
                eye,
                dir,
                max_dist,
                player.dimension,
                player_slot,
                held_dmg,
            ) {
                match res {
                    MeleeNativeResult::DeflectedFireball(fpos) => {
                        let _ = audio.play_named(
                            "mobs_fireball",
                            0.85,
                            1.15,
                            SoundCategory::Hostile,
                            &settings.audio,
                        );
                        if let Some(ref mut r) = renderer {
                            let psys = r.particles_for_mut(player.dimension);
                            for _ in 0..8 {
                                psys.spawn_torch_ember(fpos);
                            }
                        }
                    }
                    MeleeNativeResult::HitMob {
                        pos,
                        killed,
                        drops,
                        sound,
                        ..
                    } => {
                        let _ = audio.play_named(
                            sound,
                            0.85,
                            1.0,
                            SoundCategory::Hostile,
                            &settings.audio,
                        );
                        if let Some(ref mut r) = renderer {
                            let psys = r.particles_for_mut(player.dimension);
                            for _ in 0..(if killed { 12 } else { 5 }) {
                                psys.spawn_nether_ash(pos + Vec3::Y);
                            }
                        }
                        if !drops.is_empty() {
                            let reg = registries.read();
                            if add_drops_to_inventory(player_state, &drops, &reg) {
                                audio.play_pickup(&settings.audio);
                            }
                        }
                    }
                }
            }
        }
        return true;
    }

    enum MeleeTarget {
        Entity(u64),
        Peer(Arc<RwLock<ObjectState>>),
        Remote(String),
    }

    let mut best_dsq = f32::INFINITY;
    let mut chosen = None;

    if let Some(rt) = lua_rt {
        if !player.dimension.is_nether() {
            let entities = rt.active_entities();
            if let Some(eh) = raycast_entities(eye, dir, max_dist, &entities) {
                best_dsq = eh.distance_sq;
                chosen = Some(MeleeTarget::Entity(eh.entity_id));
            }
        }
    }
    for peer in peer_players {
        if peer.read().dimension != player.dimension {
            continue;
        }
        if let Some(ph) = raycast_player(eye, dir, max_dist, peer) {
            if ph.distance_sq < best_dsq {
                best_dsq = ph.distance_sq;
                chosen = Some(MeleeTarget::Peer(peer.clone()));
            }
        }
    }
    if !player.dimension.is_nether() {
        for (rname, rp) in remote_players.iter() {
            if let Some(dsq) = raycast_player_aabb(eye, dir, max_dist, rp.pos) {
                if dsq < best_dsq {
                    best_dsq = dsq;
                    chosen = Some(MeleeTarget::Remote(rname.clone()));
                }
            }
        }
    }
    let Some(target) = chosen else {
        return false;
    };

    if mine_just || (mine_held && *attack_tflp >= 0.55) {
        let tflp = *attack_tflp;
        *attack_tflp = 0.0;
        if let Some(ref mut r) = renderer {
            if player_slot == 0 {
                r.viewmodel.trigger_swing();
            } else if let Some(pslot) = r.peer_slot_mut(player_slot) {
                pslot.viewmodel.trigger_swing();
            }
        }
        match target {
            MeleeTarget::Peer(peer) => {
                if let Some(rt) = lua_rt {
                    let res = rt.punch_player(&peer, player_state, tflp, dir);
                    if res.hit {
                        for cand in ["player_damage", "mobs_mc_generic_hurt", "default_dig_choppy"] {
                            if audio.play_named(cand, 0.75, 1.0, SoundCategory::Blocks, &settings.audio) {
                                break;
                            }
                        }
                    }
                }
            }
            MeleeTarget::Entity(eid) => {
                if let Some(rt) = lua_rt {
                    let res = rt.punch_entity(eid, player_state, tflp, dir);
                    if res.hit {
                        for cand in ["player_damage", "mobs_mc_generic_hurt", "default_dig_choppy"] {
                            if audio.play_named(cand, 0.75, 1.0, SoundCategory::Blocks, &settings.audio) {
                                break;
                            }
                        }
                    }
                    if !res.drops.is_empty() {
                        let reg = registries.read();
                        if add_drops_to_inventory(player_state, &res.drops, &reg) {
                            audio.play_pickup(&settings.audio);
                        }
                    }
                }
            }
            MeleeTarget::Remote(rname) => {
                let held_item = {
                    let ps = player_state.read();
                    let widx = ps.wield_index.saturating_sub(1);
                    let mg = ps.meta.read();
                    mg.inventories
                        .get("main")
                        .and_then(|v| v.get(widx))
                        .map(|s| s.name.clone())
                        .unwrap_or_default()
                };
                let dmg = {
                    let reg = registries.read();
                    reg.get_item_def(&held_item)
                        .and_then(|d| d.tool_capabilities.as_ref())
                        .and_then(|tc| tc.damage_groups.get("fleshy").copied())
                        .unwrap_or(1)
                        .max(1)
                };
                if let Some(rp) = remote_players.get_mut(&rname) {
                    rp.hurt_flash = 0.25;
                }
                let pkt = NetPacket::PlayerPunch {
                    attacker: player_state.read().name.clone(),
                    target: rname,
                    damage: dmg,
                    kb_x: dir.x * 5.0,
                    kb_y: 4.0,
                    kb_z: dir.z * 5.0,
                };
                if let Some(s) = net_session {
                    s.send(pkt.clone());
                }
                if let Some(srv) = local_server {
                    let _ = srv.tx.send(pkt);
                }
                for cand in ["player_damage", "mobs_mc_generic_hurt", "default_dig_choppy"] {
                    if audio.play_named(cand, 0.75, 1.0, SoundCategory::Blocks, &settings.audio) {
                        break;
                    }
                }
            }
        }
    }
    true
}

#[allow(clippy::too_many_arguments)]
fn run_player_block_actions(
    player_slot: usize,
    mine_active: bool,
    place_just: bool,
    dt: f32,
    targeted: Option<&RaycastHit>,
    player: &mut PlayerController,
    player_state: &Arc<RwLock<ObjectState>>,
    digger: &mut DigController,
    last_interact_pos: &mut Option<IVec3>,
    last_inv_total_count: &mut u32,
    world: &Arc<WorldStore>,
    registries: &Arc<RwLock<EngineRegistries>>,
    lua_rt: Option<&LuaRuntime>,
    audio: &AudioEngine,
    atlas: &TextureAtlas,
    settings: &EngineSettings,
    net_session: Option<&NetClientSession>,
    local_server: Option<&LocalServerHandle>,
    mut renderer: Option<&mut WgpuRenderer>,
) -> Option<IVec3> {
    if player.is_dead {
        digger.reset();
        return None;
    }
    let mut opened_chest_pos: Option<IVec3> = None;
    let reg = registries.read();
    let sel_boxes = targeted
        .map(|h| selected_world_boxes(h, &reg))
        .unwrap_or_default();

    let dig_res = digger.update_with_physical_drops(
        dt,
        mine_active,
        targeted,
        world,
        &reg,
        player,
        player_state,
        lua_rt,
        settings.block_shatter_physics,
    );

    if let Some(ref mut r) = renderer {
        if player_slot == 0 {
            r.set_selection_boxes(&sel_boxes);
            r.set_crack_overlay(&sel_boxes, dig_res.crack_stage, atlas);
            if dig_res.swing_triggered {
                r.viewmodel.trigger_swing();
            }
            r.viewmodel.set_digging(dig_res.is_digging);
        } else {
            r.set_peer_selection_boxes(player_slot, &sel_boxes);
            r.set_peer_crack_overlay(player_slot, &sel_boxes, dig_res.crack_stage, atlas);
            if let Some(pslot) = r.peer_slot_mut(player_slot) {
                if dig_res.swing_triggered {
                    pslot.viewmodel.trigger_swing();
                }
                pslot.viewmodel.set_digging(dig_res.is_digging);
            }
        }
        if dig_res.spawn_dig_particles && settings.block_shatter_physics {
            if let Some(hit) = targeted {
                let (_, p1, _) = world.get_node(hit.under);
                r.particles_for_mut(world.dimension).spawn_dig_chips(
                    hit.under,
                    hit.node_id,
                    4,
                    atlas,
                    &reg,
                    p1 & 0x0F,
                    p1 >> 4,
                );
            }
        }
    }
    if let Some(ref s) = dig_res.sound_dig {
        audio.play_node_sound(s, SoundCategory::Blocks, &settings.audio);
    }
    if let Some(ref comp) = dig_res.completed_dig {
        player.add_exhaustion(0.05);
        let _ = extinguish_portal_around(world, &reg, comp.pos);
        if let Some(ref s) = comp.sound_dug {
            audio.play_node_sound(s, SoundCategory::Blocks, &settings.audio);
        }
        if player.gamemode != GameMode::Creative && comp.picked_up_any {
            audio.play_pickup(&settings.audio);
        }
        if let Some(ref mut r) = renderer {
            if settings.block_shatter_physics {
                let (_, p1, _) = world.get_node(comp.pos);
                r.particles_for_mut(world.dimension).spawn_shatter_debris(
                    comp.pos,
                    comp.node_id,
                    atlas,
                    &reg,
                    p1 & 0x0F,
                    p1 >> 4,
                );
            }
            r.relight_and_remesh_at(world, comp.pos, &reg, atlas, settings.smooth_lighting);
        }
        if !world.dimension.is_nether() {
            if let Some(session) = net_session {
                session.send(NetPacket::BlockEdit {
                    x: comp.pos.x,
                    y: comp.pos.y,
                    z: comp.pos.z,
                    node_name: "air".into(),
                    param1: 15,
                    param2: 0,
                });
            }
            if let Some(srv) = local_server {
                srv.record_host_block_edit(comp.pos, "air".into(), 15, 0);
            }
        }
    }

    if place_just {
        let should_interact_first = !player.is_sneaking
            && targeted.is_some_and(|h| targeted_node_has_interaction(h, world, &reg, lua_rt));
        if !should_interact_first && try_eat_held_food(player, player_state, &reg, lua_rt) {
            audio.play_eat(&settings.audio);
            if let Some(ref mut r) = renderer {
                if player_slot == 0 {
                    r.viewmodel.trigger_eat();
                } else if let Some(pslot) = r.peer_slot_mut(player_slot) {
                    pslot.viewmodel.trigger_eat();
                }
            }
            *last_inv_total_count = total_inventory_items(player_state);
            return None;
        }
        if let Some(hit) = targeted {
            *last_interact_pos = Some(hit.under);
            let pr = try_interact_or_place(hit, world, &reg, player, player_state, lua_rt);
            if pr.open_chest_pos.is_some() {
                opened_chest_pos = pr.open_chest_pos;
            }
            if pr.placed || pr.interacted {
                if let Some(ref mut r) = renderer {
                    if player_slot == 0 {
                        r.viewmodel.trigger_swing();
                    } else if let Some(pslot) = r.peer_slot_mut(player_slot) {
                        pslot.viewmodel.trigger_swing();
                    }
                }
            }
            if let Some(ref s) = pr.sound_place {
                audio.play_node_sound(s, SoundCategory::Blocks, &settings.audio);
            }
            if let Some(origin) = pr.portal_ignited {
                if let Some(ref mut r) = renderer {
                    for off in 0..10 {
                        let c = origin.as_vec3() + Vec3::new(0.5 + (off % 2) as f32 * 0.4, 0.5 + (off % 3) as f32 * 0.6, 0.5);
                        r.particles_for_mut(world.dimension).spawn_portal_spark(c);
                    }
                    r.relight_and_remesh_at(world, origin, &reg, atlas, settings.smooth_lighting);
                }
            }
            if let Some(epos) = pr.water_evaporated_pos {
                if let Some(ref mut r) = renderer {
                    r.particles_for_mut(world.dimension).spawn_steam_burst(epos, 16);
                }
            }
            if let Some(bpos) = pr.bed_exploded_pos {
                if let Some(ref mut r) = renderer {
                    r.particles_for_mut(world.dimension)
                        .spawn_explosion_burst(bpos.as_vec3() + Vec3::splat(0.5), 32);
                    r.relight_and_remesh_at(world, bpos, &reg, atlas, settings.smooth_lighting);
                }
                if player.gamemode != GameMode::Creative && player.spawn_invuln_timer <= 0.0 {
                    player.hp = (player.hp - 14).max(0);
                    player.vel.y = 7.5;
                    let mut ps = player_state.write();
                    ps.hp = player.hp;
                    ps.hurt_flash = 0.25;
                    if player.hp == 0 && player.death_cause.is_empty() {
                        player.death_cause = "Killed by [Intentional Game Design]".into();
                        ps.last_death_cause = player.death_cause.clone();
                    }
                }
            }
            if let Some(ppos) = pr.placed_pos {
                if let Some(ref mut r) = renderer {
                    r.relight_and_remesh_at(world, ppos, &reg, atlas, settings.smooth_lighting);
                }
                if !world.dimension.is_nether() {
                    let (nid, np1, np2) = world.get_node(ppos);
                    let nname = reg.get_name_from_content_id(nid).to_string();
                    if let Some(session) = net_session {
                        session.send(NetPacket::BlockEdit {
                            x: ppos.x,
                            y: ppos.y,
                            z: ppos.z,
                            node_name: nname.clone(),
                            param1: np1,
                            param2: np2,
                        });
                    }
                    if let Some(srv) = local_server {
                        srv.record_host_block_edit(ppos, nname, np1, np2);
                    }
                }
            }
        }
    }
    *last_inv_total_count = total_inventory_items(player_state);
    opened_chest_pos
}
